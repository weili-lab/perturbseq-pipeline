"""Membership regression for high-MOI screens (``regression.enabled``, PR D).

In a high-MOI screen a cell carries several targets, so a pseudobulk contrast
"cells carrying *t*" vs control also picks up the effects of every target that
is co-carried with *t*. This stage fits all targets jointly, one linear model
per gene over the assigned cells (targeting + NTC-only)::

    lognorm_g ~ 1 + membership (all targets) + n_guides + log(total_counts) + lane

The membership coefficients are ridge-penalised (``ridge_alpha``; intercept and
covariates are not), which keeps rare / near-collinear targets stable. The
design is targets x targets small, so it is factorised once per design and the
cost is ``Xᵀ Y`` over genes, done in ``scaling.effect_gene_chunk`` chunks from
the sparse layer (no dense cells x genes matrix).

Inference uses permutations: the membership rows (with ``n_guides``) are
shuffled across cells within each lane (``batch_key``), which keeps the lane
composition, the MOI distribution and the depth covariate fixed while breaking
the cell <-> guide link. They are used twice:

* calibration (genomic control): per target, ``lambda = max(1, median(null t²) /
  median(t² under the t distribution))``; the observed t is divided by
  ``sqrt(lambda)`` before its two-sided t-test p-value (residual df), and BH runs
  across genes within the target (the modules convention; ``fdr_scope: global``
  runs it over all (target, gene) pairs, which controls the FDR of the whole
  call set rather than per target);
* check: the same calls are made on every permuted data set; the mean number of
  permuted (target, gene) pairs passing the FDR cut over the observed number is
  reported as the empirical FDR of the call set (``regression_design``).

A purely empirical p-value is not used: it cannot go below
``1 / (1 + n_permutations * genes)``, which caps the BH q of a target with a
single true hit at about ``1 / n_permutations``.

Effects are reported on the log2 scale: ``log2fc = coef / ln 2`` where ``coef``
is the lognorm-space coefficient. This is a log2 ratio of geometric means of
(normalised counts + 1) adjusted for the other targets and covariates, not the
pseudobulk mean-difference log2FC of ``modules``.

Notes
-----
* ``n_guides`` is close to the row sum of the membership matrix, so it is only
  identified through the cells whose guide count differs from their target
  count (NTC guides, several guides of one target) and through the ridge
  penalty. Its role is to absorb a generic guide-burden effect; a response that
  every target shares is attributed to it rather than to the targets.
* Only available in ``guides.assignment_mode: high_moi``.
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.stats import t as _tdist

from .cluster import LOGNORM_LAYER
from .config import Config
from .high_moi import OBS_N_GUIDES, OBS_NTC_ONLY, membership_index
from .perturbation import benjamini_hochberg

logger = logging.getLogger(__name__)

_LN2 = math.log(2.0)


@dataclass
class RegressionResults:
    """Joint membership regression: per (target, gene) effects and permutation FDR."""

    #: Reported targets x genes, log2-scale coefficients.
    log2fc: pd.DataFrame
    #: Reported targets x genes, t statistic (coef / sandwich SE).
    tstat: pd.DataFrame
    #: Reported targets x genes, t-test p-value after the per-target genomic-control deflation.
    pval: pd.DataFrame
    #: Reported targets x genes, BH FDR (within each target, or over all pairs: ``regression.fdr_scope``).
    fdr: pd.DataFrame
    #: One row per reported target.
    summary: pd.DataFrame
    #: Long table of the significant (target, gene) pairs.
    de: pd.DataFrame
    #: metric / value table describing the fit.
    design: pd.DataFrame
    #: Member cells per reported target (among the fitted cells).
    n_cells: pd.Series
    info: Dict[str, object] = field(default_factory=dict)

    def de_mask(self, fdr_alpha: float, min_abs_log2fc: float) -> pd.DataFrame:
        return (self.fdr < fdr_alpha) & (self.log2fc.abs() > min_abs_log2fc)


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------


def _select_genes(expr: ad.AnnData, cfg: Config, n_targets: int) -> List[str]:
    rcfg = cfg.regression
    if rcfg.genes == "all":
        return [str(g) for g in expr.var_names]
    if rcfg.genes == "hvg":
        if "highly_variable" not in expr.var.columns:
            raise ValueError("regression.genes is 'hvg' but var['highly_variable'] is missing")
        return [str(g) for g in expr.var_names[expr.var["highly_variable"].to_numpy(dtype=bool)]]
    from .modules import select_genes

    return select_genes(expr, cfg, large_mode=cfg.use_large_mode(expr.n_obs, n_perturbations=n_targets))


def _bh(pval: np.ndarray, scope: str) -> np.ndarray:
    """BH across genes within each target (``target``) or across all pairs (``global``)."""
    if scope == "global":
        return benjamini_hochberg(pval.ravel()).reshape(pval.shape)
    return np.vstack([benjamini_hochberg(row) for row in pval])


def _within_group_permutation(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row order that shuffles positions within each group."""
    perm = np.arange(groups.size)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = idx[rng.permutation(idx.size)]
    return perm


class _Design:
    """One design (observed or permuted): Xᵀ, Cholesky factor of XᵀX + Λ, SE factors."""

    def __init__(
        self, M: sparse.csr_matrix, n_guides: Optional[np.ndarray], fixed: np.ndarray, alpha: float, n_report: int
    ):
        cols = [M]
        if n_guides is not None:
            cols.append(sparse.csr_matrix(n_guides[:, None]))
        cols.append(sparse.csr_matrix(fixed))
        X = sparse.hstack(cols, format="csr", dtype=np.float64)
        self.Xt = sparse.csr_matrix(X.T)
        m = M.shape[1]
        penalty = np.zeros(X.shape[1])
        penalty[:m] = alpha
        A = (self.Xt @ X).toarray()
        A[np.diag_indices_from(A)] += penalty
        try:
            self.factor = linalg.cho_factor(A, lower=False, check_finite=False)
        except linalg.LinAlgError as exc:
            raise RuntimeError(
                "regression: the design is singular (a covariate is constant or collinear with the intercept); "
                "set regression.ridge_alpha > 0 or drop the covariate"
            ) from exc
        Ainv = linalg.cho_solve(self.factor, np.eye(A.shape[0]), check_finite=False)
        # Var(beta) = s2 * A^-1 (XᵀX) A^-1 = s2 * (A^-1 - A^-1 Λ A^-1); only the reported rows are needed.
        se_fac = np.diag(Ainv)[:n_report] - alpha * np.sum(Ainv[:n_report, :m] ** 2, axis=1)
        self.se_fac = np.maximum(se_fac, 0.0)
        self.p = X.shape[1]
        self.m = m
        # Effective residual df n - tr(H), with tr(H) = tr(A^-1 XᵀX) = p - α tr(A^-1 on the penalised block):
        # a ridge-penalised column uses less than one df (equal to p - α·0 = p when α = 0).
        n = X.shape[0]
        self.df_model = float(self.p - alpha * np.trace(Ainv[:m, :m]))
        self.resid_df = float(n - self.df_model)
        if self.resid_df < 1.0:
            raise ValueError(
                f"regression: the design has {self.df_model:.1f} effective parameters for {n} fitted cells "
                "(residual df < 1); increase regression.ridge_alpha or fit more cells"
            )


def _fit_chunk(design: _Design, Y: sparse.csr_matrix, yty: np.ndarray, alpha: float, n_report: int):
    """coef and t (n_report x genes) for one gene chunk."""
    B = design.Xt @ Y
    B = B.toarray() if sparse.issparse(B) else np.asarray(B)
    beta = linalg.cho_solve(design.factor, B, check_finite=False)
    # RSS = yᵀy - βᵀXᵀy - α‖β_M‖²  (from (XᵀX + Λ)β = Xᵀy)
    rss = yty - np.einsum("ij,ij->j", beta, B) - alpha * np.einsum("ij,ij->j", beta[: design.m], beta[: design.m])
    s2 = np.maximum(rss, 0.0) / design.resid_df
    coef = beta[:n_report]
    se = np.sqrt(np.outer(design.se_fac, s2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, coef / se, 0.0)
    return coef, t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_regression(expr: ad.AnnData, cfg: Config) -> Optional[RegressionResults]:
    """Fit the joint membership model; ``None`` when there is nothing to fit."""
    rcfg = cfg.regression
    index = membership_index(expr, cfg)
    if index is None:
        raise ValueError("regression needs guides.assignment_mode: high_moi")
    obs = expr.obs
    ntc_only = obs[OBS_NTC_ONLY].to_numpy(dtype=bool) if OBS_NTC_ONLY in obs.columns else np.zeros(expr.n_obs, bool)
    fit = index.targeting_mask | ntc_only
    fit_idx = np.flatnonzero(fit)
    n = int(fit_idx.size)
    T = sparse.csr_matrix(index.indicator(index.targets).T)[fit_idx]
    counts = np.asarray(T.sum(axis=0)).ravel()
    in_design = np.flatnonzero(counts > 0)
    # reported targets first, so their coefficients are the leading rows
    reported = [j for j in in_design if counts[j] >= rcfg.min_cells]
    others = [j for j in in_design if counts[j] < rcfg.min_cells]
    order = reported + others
    targets = [index.targets[j] for j in reported]
    n_report = len(targets)
    if n_report == 0 or n < 3:
        logger.info("regression: no target with >= %d member cells — skipping", rcfg.min_cells)
        return None
    M = sparse.csr_matrix(T[:, order])
    del T
    # --- covariates (centred), lanes, intercept --------------------------------
    covariate_names: List[str] = []
    n_guides = None
    if rcfg.n_guides_covariate:
        if OBS_N_GUIDES not in obs.columns:
            raise ValueError(f"regression.n_guides_covariate needs obs['{OBS_N_GUIDES}']")
        v = obs[OBS_N_GUIDES].to_numpy(dtype=np.float64)[fit_idx]
        if np.ptp(v) > 0:
            n_guides = v - v.mean()
            covariate_names.append(OBS_N_GUIDES)
        else:
            logger.info("regression: n_guides is constant over the fitted cells — covariate dropped")
    fixed_cols = [np.ones(n)]
    fixed_names = ["intercept"]
    if rcfg.depth_covariate:
        if "total_counts" not in obs.columns:
            raise ValueError("regression.depth_covariate needs obs['total_counts']")
        v = np.log(np.maximum(obs["total_counts"].to_numpy(dtype=np.float64)[fit_idx], 1.0))
        if np.ptp(v) > 0:
            fixed_cols.append(v - v.mean())
            fixed_names.append("log_total_counts")
    groups = np.zeros(n, dtype=np.int64)
    batch_used = None
    if rcfg.batch_key:
        if rcfg.batch_key in obs.columns:
            lanes = obs[rcfg.batch_key].astype(str).to_numpy()[fit_idx]
            levels = sorted(set(lanes))
            groups = np.searchsorted(levels, lanes)
            for k, lev in enumerate(levels[1:], start=1):
                fixed_cols.append((groups == k).astype(np.float64))
                fixed_names.append(f"{rcfg.batch_key}={lev}")
            batch_used = rcfg.batch_key
        else:
            logger.warning(
                "regression: obs[%r] not found — no lane term, permutations are not stratified", rcfg.batch_key
            )
    covariate_names += fixed_names[1:]
    fixed = np.column_stack(fixed_cols)
    # --- genes ---------------------------------------------------------------------
    genes = _select_genes(expr, cfg, n_report)
    if not genes:
        logger.info("regression: no genes selected — skipping")
        return None
    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    gene_pos = np.asarray([expr.var_names.get_loc(g) for g in genes], dtype=np.int64)
    G = len(genes)
    n_perm = int(rcfg.n_permutations)
    alpha = float(rcfg.ridge_alpha)
    logger.info(
        "regression: %d cells x (%d targets [%d reported] + %d covariates) x %d genes; ridge %.3g; %d permutations%s",
        n,
        M.shape[1],
        n_report,
        len(covariate_names),
        G,
        alpha,
        n_perm,
        f" within {batch_used}" if batch_used else "",
    )
    # --- designs: observed + permutations, one at a time --------------------------------
    # Each design owns a dense (targets x targets) Cholesky factor, so only one is alive at any time; the gene
    # chunks are re-read per design. Peak memory: one factor + the stored null (4 bytes x targets x genes x perms).
    rng = np.random.default_rng(cfg.run.seed)
    coef_all = np.empty((n_report, G), dtype=np.float64)
    t_all = np.empty((n_report, G), dtype=np.float64)
    null = np.empty((n_report, n_perm * G), dtype=np.float32)
    # permuted |log2fc|, only needed when the calls also have an effect-size cut
    null_lfc = np.empty((n_report, n_perm * G), dtype=np.float32) if rcfg.min_abs_log2fc > 0 else None
    chunk = cfg.scaling.effect_gene_chunk
    residual_df = 1.0
    for d in range(n_perm + 1):
        if d == 0:
            design = _Design(M, n_guides, fixed, alpha, n_report)
            residual_df = design.resid_df
        else:
            perm = _within_group_permutation(groups, rng)
            design = _Design(M[perm], None if n_guides is None else n_guides[perm], fixed, alpha, n_report)
        for start in range(0, G, chunk):
            stop = min(start + chunk, G)
            Y = layer[:, gene_pos[start:stop]]
            Y = (
                sparse.csr_matrix(Y, dtype=np.float64)[fit_idx]
                if sparse.issparse(Y)
                else sparse.csr_matrix(np.asarray(Y, dtype=np.float64)[fit_idx])
            )
            yty = np.asarray(Y.multiply(Y).sum(axis=0)).ravel()
            coef, t = _fit_chunk(design, Y, yty, alpha, n_report)
            if d == 0:
                coef_all[:, start:stop] = coef
                t_all[:, start:stop] = t
            else:
                cols = slice((d - 1) * G + start, (d - 1) * G + stop)
                null[:, cols] = np.abs(t)
                if null_lfc is not None:
                    null_lfc[:, cols] = np.abs(coef) / _LN2
            del Y
        del design
        gc.collect()
        logger.info(
            "regression: %s design complete (%d genes)", "observed" if d == 0 else f"permutation {d}/{n_perm}", G
        )
    # --- p-values: t test deflated by the per-target permutation null (genomic control), BH within target ---
    # A pooled empirical p-value cannot go below 1 / (1 + n_perm * G), which caps the BH q of a lone hit at
    # ~1 / n_perm; so the permutations calibrate the t statistic instead of replacing its distribution, and
    # the same calls made on every permuted data set give an empirical FDR of the whole call set.
    t_med2 = float(_tdist.ppf(0.75, residual_df) ** 2)  # median of t^2 under the model null
    lam = np.array([max(1.0, float(np.median(np.square(row, dtype=np.float64))) / t_med2) for row in null])
    scale = np.sqrt(lam)[:, None]
    pval = 2.0 * _tdist.sf(np.abs(t_all) / scale, residual_df)
    fdr = _bh(pval, rcfg.fdr_scope)
    # the permuted calls use the same criteria as the reported calls (FDR and, if set, |log2fc|)
    n_sig_null = np.zeros(n_perm, dtype=np.int64)
    for d in range(n_perm):
        cols = slice(d * G, (d + 1) * G)
        p0 = 2.0 * _tdist.sf(null[:, cols].astype(np.float64) / scale, residual_df)
        call0 = _bh(p0, rcfg.fdr_scope) < rcfg.fdr_alpha
        if null_lfc is not None:
            call0 &= null_lfc[:, cols] > rcfg.min_abs_log2fc
        n_sig_null[d] = int(call0.sum())
    del null, null_lfc
    perm_mean_sig = float(n_sig_null.mean())
    log2fc = coef_all / _LN2
    lfc = pd.DataFrame(log2fc, index=targets, columns=genes)
    tdf = pd.DataFrame(t_all, index=targets, columns=genes)
    pdf = pd.DataFrame(pval, index=targets, columns=genes)
    fdf = pd.DataFrame(fdr, index=targets, columns=genes)
    n_cells = pd.Series(counts[reported].astype(int), index=targets, name="n_cells")
    sig = (fdr < rcfg.fdr_alpha) & (np.abs(log2fc) > rcfg.min_abs_log2fc)
    gene_set = {g: i for i, g in enumerate(genes)}
    rows = []
    for r, tgt in enumerate(targets):
        own = gene_set.get(tgt)
        rows.append(
            {
                "target_gene": tgt,
                "n_cells": int(n_cells.iloc[r]),
                "n_de_genes": int(sig[r].sum()),
                "n_up": int((sig[r] & (log2fc[r] > 0)).sum()),
                "n_down": int((sig[r] & (log2fc[r] < 0)).sum()),
                "min_fdr": float(fdr[r].min()),
                "own_gene_log2fc": float(log2fc[r, own]) if own is not None else np.nan,
                "own_gene_fdr": float(fdr[r, own]) if own is not None else np.nan,
                "gc_lambda": float(lam[r]),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["n_de_genes", "min_fdr"], ascending=[False, True], ignore_index=True)
    rr, cc = np.nonzero(sig)
    de = pd.DataFrame(
        {
            "target_gene": np.asarray(targets, dtype=object)[rr],
            "gene": np.asarray(genes, dtype=object)[cc],
            "log2fc": log2fc[rr, cc],
            "t": t_all[rr, cc],
            "pval": pval[rr, cc],
            "fdr": fdr[rr, cc],
        }
    )
    de = de.iloc[np.lexsort((-np.abs(de["t"].to_numpy()), de["fdr"].to_numpy(), de["target_gene"].to_numpy()))]
    de = de.reset_index(drop=True)
    info = {
        "n_cells_fit": n,
        "n_ntc_only_cells": int(ntc_only[fit_idx].sum()),
        "n_targets_in_design": int(M.shape[1]),
        "n_targets_reported": n_report,
        "min_cells": int(rcfg.min_cells),
        "covariates": ", ".join(covariate_names) if covariate_names else "none",
        "batch_key": batch_used or "none",
        "n_genes": G,
        "gene_selection": rcfg.genes,
        "ridge_alpha": alpha,
        "n_permutations": n_perm,
        "null_values_per_target": n_perm * G,
        "residual_df": round(residual_df, 2),
        "fdr_alpha": float(rcfg.fdr_alpha),
        "fdr_scope": rcfg.fdr_scope,
        "min_abs_log2fc": float(rcfg.min_abs_log2fc),
        "n_significant_pairs": int(sig.sum()),
        "n_targets_with_de": int((sig.sum(axis=1) > 0).sum()),
        "gc_lambda_median": float(np.median(lam)),
        "gc_lambda_max": float(lam.max()),
        "perm_mean_significant_pairs": perm_mean_sig,
        "empirical_fdr": (perm_mean_sig / int(sig.sum())) if sig.any() else float("nan"),
    }
    design_tbl = pd.DataFrame([(k, v) for k, v in info.items()], columns=["metric", "value"])
    design_tbl["value"] = design_tbl["value"].astype(str)
    logger.info(
        "regression: %d significant (target, gene) pairs; %d/%d targets with >= 1 DE gene; "
        "permuted data pass the same FDR cut %.1f times on average (empirical FDR %.3g); GC lambda median %.2f",
        info["n_significant_pairs"],
        info["n_targets_with_de"],
        n_report,
        perm_mean_sig,
        info["empirical_fdr"],
        info["gc_lambda_median"],
    )
    return RegressionResults(
        log2fc=lfc, tstat=tdf, pval=pdf, fdr=fdf, summary=summary, de=de, design=design_tbl, n_cells=n_cells, info=info
    )
