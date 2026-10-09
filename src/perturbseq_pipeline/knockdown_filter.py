"""Knockdown-efficiency mask.

Marks, but never removes, targeting cells whose own target gene is not knocked
down. Stage 5 still estimates perturbation strength on every cell, and both the
mask and those continuous estimates are written to the outputs, so the final
filtering decision stays with the user.

For every targeting cell ``i`` with target ``g`` in context ``c``::

    kd_ratio_i = x_ig / mean(x_g over non-targeting cells in c)

on library-size-normalized, *linear* expression (``expm1`` of the lognorm
layer). The control baseline is always per context, so in ``pooled`` mode a
context with a naturally lower baseline cannot pass for a knockdown.

Per group — (target, context) in ``per_context`` / ``any_context``, target in
``pooled`` — the filter then runs:

1. the group passes when the mean ``kd_ratio`` of its cells is below
   ``max_mean_ratio``. The baseline is constant within a context, so this is
   the ratio of mean target-cell to mean control expression. A median would
   be 0 for any target detected in under half of the cells, knockdown or not;
   the mean counts dropout zeros the same way in both groups;
2. in a passing group, cells whose own ratio is at or above ``max_cell_ratio``
   are marked as escapers;
3. a group with fewer than ``min_cells`` cells is ``non_testable`` and left
   unmarked.

``any_context`` runs step 1 per context; a target passing in at least one
context has step 2 applied in its passing contexts only and keeps every cell in
the rest. A target passing nowhere is marked everywhere it was testable.

``method: count_model`` replaces steps 1 and 2 with a two-component count
model on the raw counts of the target gene (``layers['counts']``). Cell ``i``
with library-size factor ``s_i`` in context ``c`` is

    x_i ~ pi * NB(mu_c * s_i, phi_c) + (1 - pi) * NB(rho * mu_c * s_i, phi_c)

where ``mu_c`` and ``phi_c`` (mean per unit library size and overdispersion)
come from the non-targeting cells of ``c``, with ``phi_c`` shared across genes
(see ``_control_fit``); ``pi`` (escaper fraction) and
``rho`` (expression left in knocked-down cells) are fitted per group by maximum likelihood. The
group passes when ``rho < max_rho``, so escapers do not dilute the test, and
the escaper fraction is below ``max_escaper_fraction`` (a group with no
knockdown can otherwise be fitted as a near-zero majority plus escapers); and a
cell is an escaper when its posterior ``P(escaper | x_i)`` reaches
``min_escaper_prob``. For a weakly expressed target one cell's count carries
little information, the posterior stays near ``pi`` and no cell is marked.

Non-targeting, ambiguous and unassigned cells are never marked.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import least_squares
from scipy.special import expit, gammaln, logit

from .cluster import LOGNORM_LAYER
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET
from .perturbation import PerturbationResults

logger = logging.getLogger(__name__)

OBS_KD_RATIO = "kd_ratio"
OBS_KD_STATUS = "kd_status"
OBS_KD_KEEP = "kd_keep"
OBS_KD_ESCAPER_PROB = "kd_escaper_prob"  # count_model only

#: Values of ``rho`` searched by ``_fit_mixture``: expression left in
#: knocked-down cells, from 1% of control to no knockdown at all.
RHO_GRID = np.linspace(0.01, 1.0, 100)

#: Context label when ``context_key`` is null.
ALL_CONTEXTS = "all"

# Per-cell statuses. The first five keep the cell, the rest mark it.
STATUS_CONTROL = "control"
STATUS_UNTOUCHED = "untouched"  # ambiguous / unassigned
STATUS_KNOCKDOWN = "knockdown"
STATUS_NON_TESTABLE = "non_testable"
STATUS_UNFILTERED_CONTEXT = "unfiltered_context"  # any_context, failing context
STATUS_ESCAPER = "escaper"
STATUS_FAILED_GROUP = "failed_group"
STATUS_LOW_EXPRESSION = "low_control_expression"
STATUS_NOT_MEASURED = "not_measured"

KEEP_STATUSES = (STATUS_CONTROL, STATUS_UNTOUCHED, STATUS_KNOCKDOWN, STATUS_NON_TESTABLE, STATUS_UNFILTERED_CONTEXT)
ALL_STATUSES = KEEP_STATUSES + (STATUS_ESCAPER, STATUS_FAILED_GROUP, STATUS_LOW_EXPRESSION, STATUS_NOT_MEASURED)

#: Stage-5 columns copied into the table and ``obs`` (suffixed with the control).
_STRENGTH_COLUMNS = ("log2fc", "pct_knockdown", "ks_fdr", "is_hit")


def _linear_target_columns(expr: ad.AnnData, genes: List[str]) -> sparse.csc_matrix:
    """Normalized linear expression of ``genes``, one column each.

    All targets are sliced at once: per-gene column access on a CSR matrix
    rescans every non-zero, which does not scale to a genome-wide screen.
    """
    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    sub = layer[:, [expr.var_names.get_loc(g) for g in genes]]
    sub = sparse.csc_matrix(sub, dtype=np.float64)
    sub.data = np.expm1(sub.data)
    return sub


def _contexts(expr: ad.AnnData, cfg: Config) -> np.ndarray:
    key = cfg.knockdown_filter.context_key
    if key is None:
        return np.full(expr.n_obs, ALL_CONTEXTS, dtype=object)
    if key not in expr.obs.columns:
        raise ValueError(
            f"knockdown_filter.context_key {key!r} is not an obs column; available: {sorted(expr.obs.columns)}"
        )
    values = expr.obs[key]
    if values.isna().any():
        raise ValueError(
            f"obs[{key!r}] has {int(values.isna().sum())} missing values; every cell "
            "needs a context for the knockdown baseline."
        )
    return values.astype(str).to_numpy()


def _mark_group(status: np.ndarray, ratio: np.ndarray, cells: np.ndarray, max_cell_ratio: float) -> None:
    """Step 2: split a passing group into knockdowns and escapers."""
    status[cells] = np.where(ratio[cells] < max_cell_ratio, STATUS_KNOCKDOWN, STATUS_ESCAPER)


# count_model


def _raw_target_columns(expr: ad.AnnData, genes: List[str]) -> sparse.csc_matrix:
    """Raw counts of ``genes``, one column each (same slicing as above)."""
    sub = expr.layers["counts"][:, [expr.var_names.get_loc(g) for g in genes]]
    return sparse.csc_matrix(sub, dtype=np.float64)


def _size_factors(expr: ad.AnnData) -> np.ndarray:
    """Library size of each cell over the median library size."""
    library = np.asarray(expr.layers["counts"].sum(axis=1), dtype=np.float64).ravel()
    return library / np.median(library)


def _log_gene_wise_phi(counts: sparse.csr_matrix, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean per unit library size, and log of the method-of-moments ``phi``, per gene.

    ``phi`` is estimated around each cell's expected count m = mu * s, with
    Var(x) = m + phi * m^2. Underdispersed (Poisson-like) genes sit at a floor
    of 1e-4 rather than at log(0); genes with no counts are NaN.
    """
    mu = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel() / s.sum()
    sum_x2 = np.asarray(counts.multiply(counts).sum(axis=0), dtype=np.float64).ravel()
    sum_xs = np.asarray(counts.T @ s, dtype=np.float64).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        resid2 = sum_x2 - 2 * mu * sum_xs + mu**2 * np.sum(s**2)
        phi = (resid2 - mu * s.sum()) / (mu**2 * np.sum(s**2))
        return mu, np.log(np.maximum(phi, 1e-4))


def _control_fit(counts: sparse.csr_matrix, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-gene mean and overdispersion in one context's controls, shared across genes.

    ``counts`` holds every gene for the context's non-targeting cells. Gene-wise
    ``phi`` is noisy when controls are few or counts low, so, as in DESeq2, its
    log is shrunk toward a trend over genes, phi(mu) = a + b / mu, fitted
    robustly on log scale, with a normal prior around the trend.

    How far each gene is shrunk depends on how noisy its own estimate is. That
    sampling variance is measured, not assumed: the controls are split in two
    halves, and the squared difference between the halves' log phi, divided by
    4, estimates the variance of the full-sample log phi. It is averaged within
    10 bins of mean expression, because low-count genes are much noisier. The
    prior variance is the genes' spread around the trend minus the average
    sampling variance, at least 0.25 (DESeq2's floor), so genes that really
    differ from the trend are not forced onto it.
    """
    mu, log_gene = _log_gene_wise_phi(counts, s)
    expressed = np.isfinite(log_gene)
    phi = np.full(mu.shape, np.nan)
    halves = np.random.default_rng(0).permutation(counts.shape[0]) % 2 == 0
    _, log_a = _log_gene_wise_phi(counts[halves], s[halves])
    _, log_b = _log_gene_wise_phi(counts[~halves], s[~halves])
    both = expressed & np.isfinite(log_a) & np.isfinite(log_b)
    # Ten equal-count bins of mean expression (by rank, so ties in the mean, common
    # with few controls, cannot leave a bin empty), interpolated between bin centres.
    log_mu = np.log(mu[both])
    half_diff2 = (log_a[both] - log_b[both]) ** 2
    bins = np.array_split(np.argsort(log_mu), 10)
    centres = np.array([np.median(log_mu[b]) for b in bins])
    var_per_bin = np.array([np.mean(half_diff2[b]) / 4 for b in bins])
    sampling_var = np.interp(np.log(mu[expressed]), centres, var_per_bin)
    m, log_g = mu[expressed], log_gene[expressed]
    # Fit the trend on genes with at least 0.1 counts per cell on average.
    use = m * np.mean(s) >= 0.1
    fit = least_squares(
        lambda ab: np.log(ab[0] + ab[1] / m[use]) - log_g[use],
        x0=[0.1, 0.1],
        bounds=([1e-8, 0.0], [np.inf, np.inf]),
        loss="soft_l1",
    )
    log_trend = np.log(fit.x[0] + fit.x[1] / m)
    spread = float(np.var(log_g[use] - log_trend[use]))
    prior_var = max(spread - float(np.mean(sampling_var[use])), 0.25)
    # Precision-weighted average of gene-wise and trend; written with the weight on
    # the gene so that a sampling variance of 0 (identical halves) gives weight 1.
    gene_weight = prior_var / (prior_var + sampling_var)
    log_phi = gene_weight * log_g + (1 - gene_weight) * log_trend
    phi[expressed] = np.exp(log_phi)
    return mu, phi


def _nb_logpmf(x: np.ndarray, mean: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Negative-binomial log pmf with mean ``mean`` and Var = mean + phi * mean^2.

    ``phi`` of 0 is the Poisson limit; it is evaluated as phi = 1e-8, where the
    two agree to far below the precision that matters here.
    """
    r = 1.0 / np.maximum(phi, 1e-8)
    return gammaln(x + r) - gammaln(r) - gammaln(x + 1) - r * np.log1p(mean / r) + x * np.log(mean / (r + mean))


def _fit_mixture(x: np.ndarray, mean: np.ndarray, phi: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Maximum-likelihood ``rho`` and ``pi`` for one group; returns them and P(escaper) per cell.

    ``mean`` and ``phi`` are each cell's unperturbed expectation, from its own
    context's controls. For each ``rho`` on RHO_GRID the log-likelihood is concave
    in ``pi`` and peaks where ``pi`` equals the mean P(escaper) over cells, found
    by bisection; the (rho, pi) pair with the highest log-likelihood wins.
    """
    log_escaper = _nb_logpmf(x, mean, phi)[None, :]
    log_kd = _nb_logpmf(x[None, :], RHO_GRID[:, None] * mean[None, :], phi[None, :])
    # A NaN here would make the rho search silently return the grid's first value
    # (rho = 0.01) and pass a group with no knockdown.
    if not (np.isfinite(log_escaper).all() and np.isfinite(log_kd).all()):
        raise ValueError("count_model: non-finite likelihood; check control mean and dispersion")
    # P(escaper) of each cell for each rho is expit(logit(pi) + log f_E/f_K).
    log_ratio = log_escaper - log_kd
    # pi stays inside (0, 1) so both components stay in the model.
    lo, hi = np.full(RHO_GRID.size, 1e-6), np.full(RHO_GRID.size, 1 - 1e-6)
    for _ in range(30):
        pi = (lo + hi) / 2
        rising = expit(logit(pi)[:, None] + log_ratio).mean(axis=1) > pi
        lo, hi = np.where(rising, pi, lo), np.where(rising, hi, pi)
    pi = (lo + hi) / 2
    log_lik = np.logaddexp(np.log(pi)[:, None] + log_escaper, np.log1p(-pi)[:, None] + log_kd).sum(axis=1)
    best = int(np.argmax(log_lik))
    p_escaper = expit(logit(pi[best]) + log_ratio[best])
    return float(RHO_GRID[best]), float(pi[best]), p_escaper


def compute_knockdown_mask(expr: ad.AnnData, cfg: Config) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Write ``kd_ratio`` / ``kd_status`` / ``kd_keep`` into ``obs``.

    Returns the AnnData (same cells, nothing removed) and a table with one row
    per (target, context).
    """
    from .high_moi import membership_index

    kcfg = cfg.knockdown_filter
    obs = expr.obs
    targets = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    # High-MOI membership: every (cell, carried target) pair is evaluated. The per-target machinery below runs
    # unchanged on the target's member cells; its per-cell scratch arrays are harvested into sparse per-(cell, target)
    # matrices and reset before the next target, and obs keeps each cell's values for its PRIMARY target.
    membership = membership_index(expr, cfg)
    if membership is not None:
        logger.info("knockdown_filter: high-MOI membership — evaluating every (cell, carried target) pair")
    contexts = _contexts(expr, cfg)
    context_values = sorted(set(contexts))
    count_model = kcfg.method == "count_model"
    if kcfg.mode == "any_context" and kcfg.max_mean_ratio_any is not None:
        mean_cut = kcfg.max_mean_ratio_any
    else:
        mean_cut = kcfg.max_mean_ratio
    ntc = klass == CLASS_NTC
    targeting = klass == CLASS_TARGETING
    status = np.full(expr.n_obs, STATUS_UNTOUCHED, dtype=object)
    status[ntc] = STATUS_CONTROL
    ratio = np.full(expr.n_obs, np.nan)
    all_targets = list(membership.targets) if membership is not None else sorted(set(targets[targeting]))
    measured = [g for g in all_targets if g in expr.var_names]
    # membership-mode outputs
    primary_status = status.copy()
    primary_ratio = ratio.copy()
    primary_escaper = np.full(expr.n_obs, np.nan)
    status_codes = {label: code for code, label in enumerate(ALL_STATUSES, start=1)}
    mem_rows: List[np.ndarray] = []
    mem_cols: List[np.ndarray] = []
    mem_ratio: List[np.ndarray] = []
    mem_status: List[np.ndarray] = []
    columns = _linear_target_columns(expr, measured) if measured else None
    col_of = {g: j for j, g in enumerate(measured)}
    if count_model:
        counts = sparse.csr_matrix(expr.layers["counts"])
        # The NB likelihood is only meaningful on raw counts. Normalized values
        # run without error but give meaningless rho and escaper calls. The
        # first 10,000 cells are enough to tell.
        sample = counts[:10_000].data
        if (sample < 0).any() or (sample != np.round(sample)).any():
            raise ValueError(
                "knockdown_filter.method=count_model needs raw counts (non-negative integers) in layers['counts']"
            )
        raw_columns = _raw_target_columns(expr, measured) if measured else None
        size_factor = _size_factors(expr)
        # Mean and shared overdispersion of every gene in each context's controls.
        control_fit = {}
        for ctx in context_values:
            ctrl_rows = np.flatnonzero(ntc & (contexts == ctx))
            if ctrl_rows.size >= kcfg.min_control_cells:
                control_fit[ctx] = _control_fit(counts[ctrl_rows], size_factor[ctrl_rows])
        # Each targeting cell's unperturbed expectation, from its context's controls.
        model_mean = np.full(expr.n_obs, np.nan)
        model_phi = np.full(expr.n_obs, np.nan)
        escaper_prob = np.full(expr.n_obs, np.nan)
    rows: List[Dict[str, object]] = []
    for gene_number, gene in enumerate(all_targets):
        gene_cells = membership.mask(gene) if membership is not None else targeting & (targets == gene)
        x = None
        if gene in col_of:
            x = columns[:, col_of[gene]].toarray().ravel()
            if count_model:
                x_raw = raw_columns[:, col_of[gene]].toarray().ravel()
        # Per-context baseline and ratio; `ok` groups go on to steps 1-3.
        gene_rows: List[Dict[str, object]] = []
        for ctx in context_values:
            in_ctx = contexts == ctx
            cells = gene_cells & in_ctx
            n_cells = int(cells.sum())
            if n_cells == 0:
                continue
            ctrl = ntc & in_ctx
            row: Dict[str, object] = {
                "target_gene": gene,
                "context": ctx,
                "n_cells": n_cells,
                "n_control": int(ctrl.sum()),
                "control_mean": np.nan,
                "pct_control_expressing": np.nan,
                "mean_ratio": np.nan,
                "_cells": cells,
            }
            if x is None:
                row["group_status"] = STATUS_NOT_MEASURED
            elif row["n_control"] < kcfg.min_control_cells:
                row["group_status"] = STATUS_NON_TESTABLE
                row["reason"] = f"fewer than {kcfg.min_control_cells} control cells"
            else:
                mean = float(x[ctrl].mean())
                pct = float(100 * np.mean(x[ctrl] > 0))
                row["control_mean"] = mean
                row["pct_control_expressing"] = pct
                if pct < kcfg.min_pct_expressing_control:
                    row["group_status"] = STATUS_LOW_EXPRESSION
                else:
                    ratio[cells] = x[cells] / mean
                    row["mean_ratio"] = float(np.mean(ratio[cells]))
                    row["group_status"] = "ok"
                    if count_model:
                        gene_index = expr.var_names.get_loc(gene)
                        mu = control_fit[ctx][0][gene_index]
                        phi = control_fit[ctx][1][gene_index]
                        row["control_mean_counts"] = mu
                        row["control_dispersion"] = phi
                        model_mean[cells] = mu * size_factor[cells]
                        model_phi[cells] = phi
            gene_rows.append(row)
        if count_model:

            def group_test(cells):
                rho, pi, p_escaper = _fit_mixture(x_raw[cells], model_mean[cells], model_phi[cells])
                escaper_prob[cells] = p_escaper
                fields = {"group_mean_ratio": float(np.mean(ratio[cells])), "rho": rho, "escaper_fraction": pi}
                # A low rho alone is not enough: a group without knockdown can be fitted
                # as "most cells knocked down to ~0 plus many escapers". Most cells must
                # be knockdowns.
                passed = rho < kcfg.max_rho and pi < kcfg.max_escaper_fraction
                return passed, fields

            def mark(cells):
                is_escaper = escaper_prob[cells] >= kcfg.min_escaper_prob
                status[cells] = np.where(is_escaper, STATUS_ESCAPER, STATUS_KNOCKDOWN)
        else:

            def group_test(cells):
                value = float(np.mean(ratio[cells]))
                return value < mean_cut, {"group_mean_ratio": value}

            def mark(cells):
                _mark_group(status, ratio, cells, kcfg.max_cell_ratio)

        if kcfg.mode == "pooled":
            _decide_pooled(gene_rows, status, kcfg, group_test, mark)
        else:
            _decide_per_context(gene_rows, status, kcfg, group_test, mark)
        if membership is not None:
            # finalise this target's rows now, harvest its member cells, then reset the scratch arrays
            for row in gene_rows:
                cells = row.pop("_cells")
                if row["group_status"] in (STATUS_NOT_MEASURED, STATUS_LOW_EXPRESSION, STATUS_NON_TESTABLE):
                    status[cells] = row["group_status"]
                row["n_kept"] = int(np.isin(status[cells], KEEP_STATUSES).sum())
                row["n_escaper"] = int((status[cells] == STATUS_ESCAPER).sum())
            idx = np.flatnonzero(gene_cells)
            mem_rows.append(idx.astype(np.int64))
            mem_cols.append(np.full(idx.size, gene_number, dtype=np.int64))
            mem_ratio.append(ratio[idx].astype(np.float32))
            mem_status.append(np.array([status_codes.get(v, 0) for v in status[idx]], dtype=np.int8))
            prim = idx[targets[idx] == gene]
            primary_status[prim] = status[prim]
            primary_ratio[prim] = ratio[prim]
            if count_model:
                primary_escaper[prim] = escaper_prob[prim]
                model_mean[idx] = np.nan
                model_phi[idx] = np.nan
                escaper_prob[idx] = np.nan
            status[idx] = STATUS_UNTOUCHED
            ratio[idx] = np.nan
        rows.extend(gene_rows)
    for row in rows:
        if "_cells" not in row:
            continue  # membership mode: already finalised per target
        cells = row.pop("_cells")
        if row["group_status"] in (STATUS_NOT_MEASURED, STATUS_LOW_EXPRESSION, STATUS_NON_TESTABLE):
            status[cells] = row["group_status"]
        row["n_kept"] = int(np.isin(status[cells], KEEP_STATUSES).sum())
        row["n_escaper"] = int((status[cells] == STATUS_ESCAPER).sum())
    if membership is not None:
        status, ratio, escaper_prob = primary_status, primary_ratio, primary_escaper
        if mem_rows:
            r = np.concatenate(mem_rows)
            c = np.concatenate(mem_cols)
            shape = (expr.n_obs, len(all_targets))
            expr.obsm["kd_ratio_membership"] = sparse.csr_matrix((np.concatenate(mem_ratio), (r, c)), shape=shape, dtype=np.float32)
            expr.obsm["kd_status_membership"] = sparse.csr_matrix((np.concatenate(mem_status), (r, c)), shape=shape, dtype=np.int8)
            expr.uns["kd_membership_targets"] = list(all_targets)
            expr.uns["kd_status_codes"] = {str(k): int(v) for k, v in status_codes.items()}
    expr.obs[OBS_KD_RATIO] = ratio
    expr.obs[OBS_KD_STATUS] = pd.Categorical(status, categories=list(ALL_STATUSES))
    expr.obs[OBS_KD_KEEP] = np.isin(status, KEEP_STATUSES)
    if count_model:
        expr.obs[OBS_KD_ESCAPER_PROB] = escaper_prob
    table = pd.DataFrame(rows)
    if not table.empty:
        table.insert(2, "mode", kcfg.mode)
        table.insert(3, "method", kcfg.method)
    n_marked = int((targeting & ~expr.obs[OBS_KD_KEEP].to_numpy()).sum())
    logger.info(
        "Knockdown mask (method=%s, mode=%s, context=%s): %d/%d targeting cells "
        "marked for removal across %d targets; no cells removed",
        kcfg.method,
        kcfg.mode,
        kcfg.context_key or "none",
        n_marked,
        int(targeting.sum()),
        len(all_targets),
    )
    return expr, table


def _decide_per_context(gene_rows: List[Dict[str, object]], status: np.ndarray, kcfg, group_test, mark) -> None:
    """``per_context`` and ``any_context``: steps 1-3 on each (target, context).

    ``group_test(cells)`` returns whether the group passes step 1 and the fields
    to record; ``mark(cells)`` splits a passing group into knockdowns and
    escapers.
    """
    testable = []
    for row in gene_rows:
        if row["group_status"] != "ok":
            continue
        if row["n_cells"] < kcfg.min_cells:
            row["group_status"] = STATUS_NON_TESTABLE
            row["reason"] = f"fewer than {kcfg.min_cells} cells"
            continue
        passed, fields = group_test(row["_cells"])
        row.update(fields)
        row["passed_group"] = bool(passed)
        testable.append(row)
    any_passed = any(row["passed_group"] for row in testable)
    for row in testable:
        cells = row["_cells"]
        if row["passed_group"]:
            row["group_status"] = "pass"
            mark(cells)
        elif kcfg.mode == "any_context" and any_passed:
            row["group_status"] = STATUS_UNFILTERED_CONTEXT
            status[cells] = STATUS_UNFILTERED_CONTEXT
        else:
            row["group_status"] = STATUS_FAILED_GROUP
            status[cells] = STATUS_FAILED_GROUP


def _decide_pooled(gene_rows: List[Dict[str, object]], status: np.ndarray, kcfg, group_test, mark) -> None:
    """``pooled``: one group per target over every context with a baseline.

    The group test runs on the pooled cells, so contexts are weighted by their
    cell count; each cell keeps its own context's baseline, and the
    per-context mean ratios stay in the table.
    """
    ok = [row for row in gene_rows if row["group_status"] == "ok"]
    if not ok:
        return
    cells = np.logical_or.reduce([row["_cells"] for row in ok])
    n_cells = int(cells.sum())
    if n_cells < kcfg.min_cells:
        for row in ok:
            row["group_status"] = STATUS_NON_TESTABLE
            row["reason"] = f"fewer than {kcfg.min_cells} cells pooled"
        return
    passed, fields = group_test(cells)
    passed = bool(passed)
    for row in ok:
        row.update(fields)
        row["passed_group"] = passed
        row["group_status"] = "pass" if passed else STATUS_FAILED_GROUP
    if passed:
        mark(cells)
    else:
        status[cells] = STATUS_FAILED_GROUP


def attach_perturbation_strength(
    expr: ad.AnnData, table: pd.DataFrame, results: PerturbationResults
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Join stage-5 per-target estimates onto the mask table and ``obs``.

    Stage 5 runs on every cell (the mask removes none), so these are an
    independent, continuous view of the same knockdown next to the mask.
    """
    if results.table.empty or table.empty:
        return expr, table
    control = results.primary_control
    cols = [f"{c}_{control}" for c in _STRENGTH_COLUMNS if f"{c}_{control}" in results.table]
    strength = results.table.set_index("target_gene")[cols]
    table = table.merge(strength, left_on="target_gene", right_index=True, how="left")
    targets = expr.obs[OBS_TARGET].astype(str)
    for col in cols:
        values = targets.map(strength[col])
        if col.startswith("is_hit"):
            values = values.astype("boolean")
        expr.obs[f"pert_{col}"] = values.to_numpy()
    return expr, table
