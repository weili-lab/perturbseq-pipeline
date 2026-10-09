"""Perturbation-strength analysis.

For every target gene that is also measured in the expression matrix, the gene's
*own* expression is compared between cells carrying guides against it and
control cells. A CRISPRi/KO perturbation is expected to push that expression
*down*, so the analysis is directional: significance alone is not enough, the
fold change must be negative.

Two control definitions are reported side by side:

``ntc``
    Cells carrying non-targeting guides. Preferred — these cells experienced the
    same transduction and selection but no on-target effect.

``other``
    Cells assigned to a *different* target gene (what the prototype notebooks
    used). Larger n, but every control cell is itself perturbed.

Tests are run on ``layers['lognorm']`` (log1p of library-size-normalized
counts), never on scaled values.

Large-dataset execution
-----------------------
The original implementation extracts the expression of each target gene over
every cell and compares the target cells against the complete control
population. This is appropriate for ordinary Perturb-seq experiments and is
preserved as the STANDARD execution path.

For million-cell experiments, repeatedly materialising a dense expression
vector over millions of cells and repeatedly running distributional tests
against million-cell control populations becomes unnecessarily expensive.
For example, a 2.6-million-cell dataset with >10,000 targets would require
billions of expression values to be materialised and repeatedly scanned.

Large-dataset mode therefore:

* keeps **all perturbed cells** for every target;
* uses a large, reproducible control reference sample for KS and Mann-Whitney
  tests;
* extracts expression only for the cells needed for a given comparison rather
  than converting the full gene column to a dense vector;
* reuses the same sampled control reference across targets where possible;
* stores intermediate arrays as float32 where appropriate.

The underlying biological test is unchanged: target expression must be lower in
perturbed cells and statistically supported. The large-data path changes the
computational reference size, not the definition of perturbation strength.

Execution mode is selected automatically. Replogle-sized datasets retain the
original implementation. Million-cell datasets such as KOLF use the scalable
path and report this explicitly in the log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp, mannwhitneyu

from .cluster import LOGNORM_LAYER
from .compute import derive_seed, log_compute_decision, resolve_stage_backend, run_parallel
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET

logger = logging.getLogger(__name__)


CONTROL_NTC = "ntc"
CONTROL_OTHER = "other"

CONTROL_LABELS = {CONTROL_NTC: "non-targeting control cells", CONTROL_OTHER: "cells assigned to other target genes"}
#: Labels when the sets come from the high-MOI membership matrix.
CONTROL_LABELS_MEMBERSHIP = {
    CONTROL_NTC: "non-targeting-only control cells",
    CONTROL_OTHER: "targeting cells not carrying the target (membership)",
}


# Numerical settings

#: Small pseudocount in normalized-expression units.
_PSEUDOCOUNT = 0.01


# Maximum control population used for KS/MWU in large-data mode.
#
# 100k controls already gives vastly more statistical power than normally
# required while preventing repeated million-element tests.
LARGE_DATASET_MAX_TEST_CONTROLS = 100_000


@dataclass
class PerturbationResults:
    """Everything the report needs about perturbation strength."""

    table: pd.DataFrame
    #: Controls actually usable on this dataset, in priority order.
    controls_used: List[str]
    #: Control driving ranking and hit calling.
    primary_control: str
    #: Targets that could not be tested.
    skipped: pd.DataFrame
    n_control_cells: Dict[str, int]

    @property
    def hits(self) -> pd.DataFrame:
        """Targets called effectively perturbed under the primary control."""
        col = f"is_hit_{self.primary_control}"
        if col not in self.table.columns:
            return self.table.iloc[0:0]
        return self.table[self.table[col]]

    def top_effects(self, n: int) -> pd.DataFrame:
        """The n strongest knockdowns."""
        return self.table.head(n)


# Generic helpers


def control_masks(expr: ad.AnnData, cfg: Config) -> Dict[str, np.ndarray]:
    """Boolean masks for each control definition."""
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    return {CONTROL_NTC: (klass == CLASS_NTC), CONTROL_OTHER: (klass == CLASS_TARGETING)}


def _control_mask_for_target(control: str, base: Dict[str, np.ndarray], targets: np.ndarray, gene: str) -> np.ndarray:
    """Original per-target control-mask implementation."""
    if control == CONTROL_NTC:
        return base[CONTROL_NTC]
    return base[CONTROL_OTHER] & (targets != gene)


# Expression extraction


def _expression_layer(expr: ad.AnnData):
    """Return the expression matrix used for perturbation testing."""
    if LOGNORM_LAYER in expr.layers:
        return expr.layers[LOGNORM_LAYER]
    return expr.X


def _gene_vector(expr: ad.AnnData, gene: str) -> np.ndarray:
    """Original full dense gene vector used by STANDARD mode."""
    layer = _expression_layer(expr)
    idx = expr.var_names.get_loc(gene)
    col = layer[:, idx]
    if sparse.issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel().astype(np.float64)


def _gene_values_at_indices(expr: ad.AnnData, gene_index: int, cell_indices: np.ndarray) -> np.ndarray:
    """Extract one gene only for requested cells.

    This is the key large-dataset primitive. It never constructs a dense
    n_cells-long vector when only a small subset is required.
    """
    if cell_indices.size == 0:
        return np.empty(0, dtype=np.float32)
    layer = _expression_layer(expr)
    values = layer[cell_indices, gene_index]
    if sparse.issparse(values):
        values = values.toarray().ravel()
    else:
        values = np.asarray(values).ravel()
    return values.astype(np.float32, copy=False)


# Statistics


def compare_groups(perturbed: np.ndarray, control: np.ndarray) -> Dict[str, float]:
    """Effect size and significance for one target/control pair.

    ``log2fc`` is calculated on de-logged normalized expression.

    Mann-Whitney is one-sided:

        perturbed < control
    """
    mean_p_log = float(np.mean(perturbed)) if perturbed.size else np.nan
    mean_c_log = float(np.mean(control)) if control.size else np.nan
    mean_p = float(np.mean(np.expm1(perturbed))) if perturbed.size else np.nan
    mean_c = float(np.mean(np.expm1(control))) if control.size else np.nan
    log2fc = float(np.log2((mean_p + _PSEUDOCOUNT) / (mean_c + _PSEUDOCOUNT)))
    pct_kd = float(100.0 * (1.0 - (mean_p + _PSEUDOCOUNT) / (mean_c + _PSEUDOCOUNT)))
    ks_stat, ks_p = ks_2samp(perturbed, control)
    try:
        mwu_p = float(mannwhitneyu(perturbed, control, alternative="less").pvalue)
    except ValueError:
        mwu_p = np.nan
    return {
        "mean_lognorm_perturbed": (mean_p_log),
        "mean_lognorm_control": (mean_c_log),
        "log2fc": (log2fc),
        "pct_knockdown": (pct_kd),
        "pct_cells_expressing_perturbed": (float(100 * np.mean(perturbed > 0)) if perturbed.size else np.nan),
        "pct_cells_expressing_control": (float(100 * np.mean(control > 0)) if control.size else np.nan),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_p),
        "mwu_pval_less": (mwu_p),
    }


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """BH-FDR correction that preserves NaNs."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    vals = p[ok]
    n = vals.size
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    res = np.empty(n)
    res[order] = q
    out[ok] = res
    return out


# ===========================================================================
# Shared setup
# ===========================================================================


def _prepare_analysis(expr: ad.AnnData, cfg: Config) -> Tuple:
    """Prepare target/control metadata shared by both execution modes.

    In ``high_moi`` mode the last element is a :class:`~.high_moi.MembershipIndex`
    (perturbed = cells carrying the target, ``other`` = targeting cells not
    carrying it); otherwise it is ``None`` and the label-based sets are used.
    """
    from .high_moi import membership_index

    pcfg = cfg.perturbation
    membership = membership_index(expr, cfg)
    obs = expr.obs
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    base = control_masks(expr, cfg)
    n_control_cells = {CONTROL_NTC: int(base[CONTROL_NTC].sum()), CONTROL_OTHER: int(base[CONTROL_OTHER].sum())}
    controls_used = []
    for control in pcfg.controls:
        if control == CONTROL_NTC and n_control_cells[CONTROL_NTC] < pcfg.min_control_cells:
            logger.warning(
                "Only %d non-targeting control cells (need %d); skipping NTC control arm.",
                n_control_cells[CONTROL_NTC],
                pcfg.min_control_cells,
            )
            continue
        controls_used.append(control)
    if not controls_used:
        raise ValueError(
            "No usable control group. There are "
            f"{n_control_cells[CONTROL_NTC]} non-targeting and "
            f"{n_control_cells[CONTROL_OTHER]} targeting cells, but "
            f"perturbation.min_control_cells="
            f"{pcfg.min_control_cells}."
        )
    primary = pcfg.primary_control if pcfg.primary_control in controls_used else controls_used[0]
    if primary != pcfg.primary_control:
        logger.warning("Requested primary control %r unavailable; using %r.", pcfg.primary_control, primary)
    if membership is not None:
        all_targets = list(membership.targets)
    else:
        all_targets = sorted(set(targets_col[klass == CLASS_TARGETING]))
    return (targets_col, klass, base, n_control_cells, controls_used, primary, all_targets, membership)


# ===========================================================================
# ORIGINAL / STANDARD IMPLEMENTATION
# ===========================================================================


def _test_all_targets_standard(expr: ad.AnnData, cfg: Config) -> PerturbationResults:
    """Original full-vector implementation.

    This path preserves previous behaviour for ordinary-sized datasets such as
    Replogle.
    """
    logger.info("Perturbation-strength execution mode: STANDARD (full expression vectors)")
    pcfg = cfg.perturbation
    (targets_col, klass, base, n_control_cells, controls_used, primary, all_targets, membership) = _prepare_analysis(
        expr, cfg
    )
    decision = resolve_stage_backend("perturbation", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)
    measured = set(expr.var_names)
    layer = _expression_layer(expr)
    var_dict = {g: i for i, g in enumerate(expr.var_names)}

    def _perturbed_mask(gene: str) -> np.ndarray:
        if membership is not None:
            return membership.mask(gene)
        return (targets_col == gene) & (klass == CLASS_TARGETING)

    def _ctrl_mask(control: str, gene: str) -> np.ndarray:
        if membership is not None and control == CONTROL_OTHER:
            return membership.other_mask(gene)
        return _control_mask_for_target(control, base, targets_col, gene)

    def _eval_target_std(gene: str) -> Tuple[Optional[dict], Optional[dict]]:
        pert_mask = _perturbed_mask(gene)
        n_pert = int(pert_mask.sum())
        if gene not in measured:
            return None, {
                "target_gene": gene,
                "n_perturbed": n_pert,
                "reason": "target gene not present in the expression matrix",
            }
        if n_pert < pcfg.min_cells_per_target:
            return None, {
                "target_gene": gene,
                "n_perturbed": n_pert,
                "reason": f"fewer than {pcfg.min_cells_per_target} perturbed cells",
            }
        gene_idx = var_dict[gene]
        col = layer[:, gene_idx]
        if sparse.issparse(col):
            col = col.toarray()
        values = np.asarray(col).ravel().astype(np.float64)
        primary_mask = _ctrl_mask(primary, gene)
        pct_expressing = float(100 * np.mean(values[primary_mask] > 0)) if primary_mask.any() else 0.0
        if pct_expressing < pcfg.min_pct_expressing_control:
            return None, {
                "target_gene": gene,
                "n_perturbed": n_pert,
                "reason": (
                    "not detectably expressed in control cells "
                    f"({pct_expressing:.2f}% of controls, threshold "
                    f"{pcfg.min_pct_expressing_control}%)"
                ),
            }
        row: Dict[str, object] = {"target_gene": gene, "n_perturbed": n_pert}
        for control in controls_used:
            cmask = _ctrl_mask(control, gene)
            n_ctrl = int(cmask.sum())
            row[f"n_control_{control}"] = n_ctrl
            if n_ctrl < pcfg.min_control_cells:
                _fill_missing_stats(row, control)
                continue
            stats = compare_groups(values[pert_mask], values[cmask])
            for key, val in stats.items():
                row[f"{key}_{control}"] = val
        return row, None

    results = run_parallel(
        _eval_target_std,
        all_targets,
        n_jobs=decision.n_jobs,
        blas_threads=cfg.compute.blas_threads_per_worker,
        backend=cfg.compute.cpu_parallel_backend,
    )
    rows = [r for r, s in results if r is not None]
    skipped = [s for r, s in results if s is not None]
    return _finalize_results(
        rows=rows,
        skipped=skipped,
        controls_used=controls_used,
        primary=primary,
        n_control_cells=n_control_cells,
        all_targets=all_targets,
        cfg=cfg,
        membership_aware=membership is not None,
    )


# ===========================================================================
# LARGE-DATASET IMPLEMENTATION
# ===========================================================================


def _sample_reference_indices(indices: np.ndarray, max_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Create a reproducible large reference subset."""
    if indices.size <= max_cells:
        return indices.astype(np.int64, copy=False)
    sampled = rng.choice(indices, size=max_cells, replace=False)
    sampled.sort()
    return sampled.astype(np.int64, copy=False)


def _test_all_targets_large(expr: ad.AnnData, cfg: Config) -> PerturbationResults:
    """Scalable perturbation-strength analysis for million-cell datasets."""
    logger.info(
        "Perturbation-strength execution mode: LARGE-DATASET "
        "(indexed expression extraction + sampled distributional controls)"
    )
    pcfg = cfg.perturbation
    (targets_col, klass, base, n_control_cells, controls_used, primary, all_targets, membership) = _prepare_analysis(
        expr, cfg
    )
    # Target metadata
    measured_index = {gene: idx for idx, gene in enumerate(expr.var_names)}
    targeting_indices = np.flatnonzero(klass == CLASS_TARGETING).astype(np.int64, copy=False)
    ntc_indices = np.flatnonzero(klass == CLASS_NTC).astype(np.int64, copy=False)
    # Build target -> cell indices ONCE.
    #
    # This prevents allocating a new 2.6-million-element Boolean mask for every
    # target.
    if membership is not None:
        target_indices: Dict[str, np.ndarray] = {target: membership.indices(target) for target in all_targets}
    else:
        targeting_frame = pd.DataFrame(
            {"target": (targets_col[targeting_indices]), "cell_index": (targeting_indices)}
        )
        target_indices = {
            target: group["cell_index"].to_numpy(dtype=np.int64, copy=True)
            for target, group in targeting_frame.groupby("target", observed=True, sort=False)
        }
        del targeting_frame
    rng = np.random.default_rng(cfg.run.seed)
    # Reusable reference populations
    ntc_test_indices = _sample_reference_indices(ntc_indices, LARGE_DATASET_MAX_TEST_CONTROLS, rng)
    # One reproducible sample of targeting cells.
    #
    # For gene G, cells carrying G are removed from this sample, yielding the
    # "other-target" reference. This avoids constructing an N-cell mask for
    # every gene.
    other_reference_indices = _sample_reference_indices(targeting_indices, LARGE_DATASET_MAX_TEST_CONTROLS, rng)
    other_reference_targets = targets_col[other_reference_indices]

    def _other_keep(gene: str) -> np.ndarray:
        """Reference cells not carrying ``gene`` (membership) / not labelled ``gene`` (legacy)."""
        if membership is not None:
            return ~membership.mask(gene)[other_reference_indices]
        return other_reference_targets != gene
    logger.info(
        "Large-data statistical references: %d/%d NTC cells and %d/%d targeting cells",
        len(ntc_test_indices),
        len(ntc_indices),
        len(other_reference_indices),
        len(targeting_indices),
    )
    decision = resolve_stage_backend("perturbation", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)

    def _eval_target_large(gene: str) -> Tuple[Optional[dict], Optional[dict]]:
        pert_indices = target_indices.get(gene, np.empty(0, dtype=np.int64))
        n_pert = int(pert_indices.size)
        if gene not in measured_index:
            return None, {
                "target_gene": gene,
                "n_perturbed": n_pert,
                "reason": "target gene not present in the expression matrix",
            }
        if n_pert < pcfg.min_cells_per_target:
            return None, {
                "target_gene": gene,
                "n_perturbed": n_pert,
                "reason": f"fewer than {pcfg.min_cells_per_target} perturbed cells",
            }
        gene_index = measured_index[gene]
        perturbed_values = _gene_values_at_indices(expr, gene_index, pert_indices)
        if primary == CONTROL_NTC:
            primary_indices = ntc_test_indices
        else:
            primary_indices = other_reference_indices[_other_keep(gene)]
        if primary_indices.size < pcfg.min_control_cells:
            return None, {
                "target_gene": gene,
                "n_perturbed": n_pert,
                "reason": "insufficient primary-control cells after large-data reference sampling",
            }
        primary_values = _gene_values_at_indices(expr, gene_index, primary_indices)
        pct_expressing = float(100 * np.mean(primary_values > 0))
        if pct_expressing < pcfg.min_pct_expressing_control:
            return None, {
                "target_gene": gene,
                "n_perturbed": n_pert,
                "reason": (
                    "not detectably expressed in sampled control cells "
                    f"({pct_expressing:.2f}% of controls, threshold "
                    f"{pcfg.min_pct_expressing_control}%)"
                ),
            }
        row: Dict[str, object] = {"target_gene": gene, "n_perturbed": n_pert}
        for control in controls_used:
            if control == CONTROL_NTC:
                ctrl_indices = ntc_test_indices
                n_ctrl_full = int(ntc_indices.size)
            else:
                keep = _other_keep(gene)
                ctrl_indices = other_reference_indices[keep]
                n_ctrl_full = int(targeting_indices.size - n_pert)
            n_ctrl_test = int(ctrl_indices.size)
            row[f"n_control_{control}"] = n_ctrl_full
            row[f"n_control_tested_{control}"] = n_ctrl_test
            if n_ctrl_test < pcfg.min_control_cells:
                _fill_missing_stats(row, control)
                continue
            if control == primary and np.array_equal(ctrl_indices, primary_indices):
                control_values = primary_values
            else:
                control_values = _gene_values_at_indices(expr, gene_index, ctrl_indices)
            stats = compare_groups(perturbed_values, control_values)
            for key, val in stats.items():
                row[f"{key}_{control}"] = val
        return row, None

    results = run_parallel(
        _eval_target_large,
        all_targets,
        n_jobs=decision.n_jobs,
        blas_threads=cfg.compute.blas_threads_per_worker,
        backend=cfg.compute.cpu_parallel_backend,
    )
    rows = [r for r, s in results if r is not None]
    skipped = [s for r, s in results if s is not None]
    return _finalize_results(
        rows=rows,
        skipped=skipped,
        controls_used=controls_used,
        primary=primary,
        n_control_cells=n_control_cells,
        all_targets=all_targets,
        cfg=cfg,
        membership_aware=membership is not None,
    )


# ===========================================================================
# Shared result finalization
# ===========================================================================


def _fill_missing_stats(row: Dict[str, object], control: str) -> None:
    """Populate unavailable statistics with NaN."""
    for key in (
        "log2fc",
        "pct_knockdown",
        "ks_stat",
        "ks_pval",
        "mwu_pval_less",
        "mean_lognorm_perturbed",
        "mean_lognorm_control",
        "pct_cells_expressing_perturbed",
        "pct_cells_expressing_control",
    ):
        row[f"{key}_{control}"] = np.nan


def _finalize_results(
    rows: List[dict],
    skipped: List[dict],
    controls_used: List[str],
    primary: str,
    n_control_cells: Dict[str, int],
    all_targets: List[str],
    cfg: Config,
    membership_aware: bool = False,
) -> PerturbationResults:
    """Shared BH-FDR, hit calling and ranking."""
    pcfg = cfg.perturbation
    labels = CONTROL_LABELS_MEMBERSHIP if membership_aware else CONTROL_LABELS
    if not rows:
        logger.warning("No target gene was testable; returning an empty result table.")
        return PerturbationResults(
            table=pd.DataFrame(),
            controls_used=controls_used,
            primary_control=primary,
            skipped=pd.DataFrame(skipped),
            n_control_cells=n_control_cells,
        )
    table = pd.DataFrame(rows)
    for control in controls_used:
        ks_col = f"ks_pval_{control}"
        mwu_col = f"mwu_pval_less_{control}"
        if ks_col not in table.columns:
            table[ks_col] = np.nan
        if mwu_col not in table.columns:
            table[mwu_col] = np.nan
        table[f"ks_fdr_{control}"] = benjamini_hochberg(table[ks_col].to_numpy())
        table[f"mwu_fdr_{control}"] = benjamini_hochberg(table[mwu_col].to_numpy())
        table[f"is_hit_{control}"] = (
            (table[f"ks_fdr_{control}"] < pcfg.fdr_alpha) & (table[f"log2fc_{control}"] < pcfg.max_log2fc_for_hit)
        ).fillna(False)
    table = table.sort_values([f"is_hit_{primary}", f"log2fc_{primary}"], ascending=[False, True]).reset_index(
        drop=True
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    n_hits = int(table[f"is_hit_{primary}"].sum())
    logger.info(
        "Perturbation strength: %d/%d targets tested, %d effective at FDR < %.2f (control: %s)",
        len(table),
        len(all_targets),
        n_hits,
        pcfg.fdr_alpha,
        labels[primary],
    )
    if skipped:
        logger.info("%d target(s) skipped; see the skipped table.", len(skipped))
    return PerturbationResults(
        table=table,
        controls_used=controls_used,
        primary_control=primary,
        skipped=pd.DataFrame(skipped),
        n_control_cells=n_control_cells,
    )


# ===========================================================================
# Public driver
# ===========================================================================


def test_all_targets(expr: ad.AnnData, cfg: Config) -> PerturbationResults:
    """Run perturbation-strength analysis with automatic scaling.

    Replogle and similar datasets use the original full-vector algorithm.

    Million-cell/high-target-count datasets automatically use indexed
    extraction and sampled statistical control populations.
    """
    from .high_moi import membership_index

    pcfg = cfg.perturbation
    membership = membership_index(expr, cfg)
    if membership is not None:
        target_counts = membership.counts
    else:
        klass = expr.obs[OBS_CLASS].astype(str)
        targets = expr.obs.loc[klass == CLASS_TARGETING, OBS_TARGET].astype(str)
        target_counts = targets.value_counts()
    n_testable_targets = int((target_counts >= pcfg.min_cells_per_target).sum())
    large_mode = cfg.use_large_mode(expr.n_obs, n_perturbations=n_testable_targets)
    logger.info(
        "Perturbation-strength input: %d cells, %d testable targets%s",
        expr.n_obs,
        n_testable_targets,
        " (high-MOI membership: perturbed = cells carrying the target)" if membership is not None else "",
    )
    if large_mode:
        logger.info(
            "Large-dataset perturbation-strength mode selected (%d cells, %d targets)", expr.n_obs, n_testable_targets
        )
        return _test_all_targets_large(expr, cfg)
    logger.info("Standard perturbation-strength mode selected")
    return _test_all_targets_standard(expr, cfg)


# ===========================================================================
# Reporting
# ===========================================================================


def format_results_table(results: PerturbationResults, cfg: Config) -> pd.DataFrame:
    """Reader-friendly view for the HTML report."""
    if results.table.empty:
        return results.table
    primary = results.primary_control
    cols = {
        "rank": "Rank",
        "target_gene": "Target",
        "n_perturbed": "Perturbed cells",
        f"n_control_{primary}": ("Control cells"),
        f"n_control_tested_{primary}": ("Control cells tested"),
        f"log2fc_{primary}": ("log2FC"),
        f"pct_knockdown_{primary}": ("% knockdown"),
        f"ks_stat_{primary}": ("KS stat"),
        f"ks_fdr_{primary}": ("KS FDR"),
        f"is_hit_{primary}": ("Effective"),
    }
    other = CONTROL_OTHER if primary == CONTROL_NTC else CONTROL_NTC
    if other in results.controls_used:
        cols[f"log2fc_{other}"] = f"log2FC ({other})"
        cols[f"ks_fdr_{other}"] = f"KS FDR ({other})"
    present = {key: value for key, value in cols.items() if key in results.table.columns}
    out = results.table[list(present)].rename(columns=present).copy()
    for col in out.columns:
        if out[col].dtype.kind == "f":
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.3g}")
    return out
