"""Per-cell perturbation-response scores via the lab's PS_python package.

The perturbation-strength stage (:mod:`perturbseq_pipeline.perturbation`) asks a
group-level question: across cells carrying guides against a gene, did that
gene's own expression decrease?

This module adds the per-cell view by delegating to ``pertps``
(https://github.com/weili-lab/PS_python), the lab's Python implementation of the
scMAGeCK-style perturbation score.

For each target, PS_python compares target cells with non-targeting controls,
learns a transcriptional perturbation signature, and projects cells onto that
signature to obtain a perturbation-response score in [0, 1].

Combining PS with expression of the targeted gene separates cells into four
practically useful classes:

``successful knockdown``
    High PS and low target expression.

``escaper``
    High PS but high target expression.

``non-responder``
    Low PS and high target expression.

``low signal``
    Low PS and low target expression.

``pertps`` is an optional dependency. When unavailable, this stage is skipped
unless ``ps_score.require`` is enabled.

Large-dataset execution
-----------------------
The standard implementation creates a complete working AnnData object and
passes it to a single ``PerturbAnalyzer``. This preserves the original behavior
for ordinary datasets such as Replogle.

For million-cell experiments this is unnecessarily expensive because copying a
multi-million-cell expression matrix can require hundreds of GB of memory.

PS_python itself already performs a target-specific analysis: for target g it
uses only:

    cells carrying g + non-targeting control cells

Large-dataset mode therefore performs this subset *before* constructing the
PerturbAnalyzer.

For each target:

    all cells carrying target g
            +
    reproducibly sampled NTC cells
            ↓
      temporary AnnData
            ↓
       PerturbAnalyzer
            ↓
     calculate_ps_score(g)

The underlying PS_python scoring algorithm is unchanged. Only the number of
control cells supplied to each target-specific analysis is bounded.

Large-dataset mode also avoids materialising thousands of ``ps_<target>``
columns. It stores each targeting cell's own PS in:

    adata.obs["ps_score"]

and its classification in:

    adata.obs["ps_quadrant"]

Target-level summaries are retained separately.

The optional supervised LDA/UMAP visualization is not run on the entire
million-cell object because PS_python scales and densifies its working matrix.
For large datasets it is either skipped or built on a bounded representative
subset.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import LOGNORM_LAYER
from .compute import log_compute_decision, resolve_stage_backend, run_parallel
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET

logger = logging.getLogger(__name__)


# PS_python vocabulary

PERTPS_NEG_CTRL = "Non-Targeting"
PERTPS_GENE_COL = "gene"

PS_PREFIX = "ps_"


# Quadrants

QUADRANT_KD = "successful knockdown"
QUADRANT_ESCAPER = "escaper"
QUADRANT_NONRESPONDER = "non-responder"
QUADRANT_LOW = "low signal"

QUADRANT_COLORS = {
    QUADRANT_KD: "#2f855a",
    QUADRANT_ESCAPER: "#c53030",
    QUADRANT_NONRESPONDER: "#2b6cb0",
    QUADRANT_LOW: "#a0aec0",
}


# Maximum NTC cells passed to one PS_python target-specific analysis.
LARGE_PS_MAX_CONTROLS = 50_000


class PertpsUnavailable(RuntimeError):
    """Raised when pertps is required but unavailable."""


@dataclass
class PSResults:
    """Per-cell perturbation scores and target-level summaries."""

    #: One row per target.
    summary: pd.DataFrame
    #: Standard mode:
    #:     {target: Series containing target + control PS values}.
    #:
    #: Large mode:
    #:     may contain only target-cell scores, avoiding huge control duplication.
    scores: Dict[str, pd.Series] = field(default_factory=dict)
    #: Per-target quadrant assignments.
    quadrants: Dict[str, pd.Series] = field(default_factory=dict)
    #: Horizontal target-expression threshold per target.
    expression_cut: Dict[str, float] = field(default_factory=dict)
    #: Optional LDA/UMAP coordinates aligned to original cells.
    lda_umap: Optional[np.ndarray] = None
    #: Optional labels used for the LDA representation.
    lda_label: Optional[pd.Series] = None
    lda_note: str = ""
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    ps_threshold: float = 0.5
    note: str = ""
    #: True when scalable target-wise execution was used.
    large_mode: bool = False
    #: Direct own-target PS vector for scalable output.
    own_score: Optional[np.ndarray] = None
    #: Direct own-target quadrant vector.
    own_quadrant: Optional[np.ndarray] = None
    #: High-MOI mode: sparse cells x targets PS for member cells (``obsm['ps_score_membership']``).
    membership_scores: Optional[sparse.csr_matrix] = None
    membership_targets: List[str] = field(default_factory=list)
    membership_aware: bool = False

    @property
    def targets(self) -> List[str]:
        if self.summary.empty:
            return []
        return list(self.summary["target_gene"])

    def top_targets(self, n: int) -> List[str]:
        """Targets with the largest fraction of successful KD cells."""
        if self.summary.empty:
            return []
        return list(self.summary.head(n)["target_gene"])


# Availability


def pertps_available() -> bool:
    """Return True if the optional pertps package can be imported."""
    try:
        import pertps  # noqa: F401
    except Exception:
        return False
    return True


def _pertps_version() -> str:
    """Best-effort package version."""
    try:
        import pertps

        return getattr(pertps, "__version__", "unknown")
    except Exception:
        return "not installed"


# Generic helpers


def _expression_layer(expr: ad.AnnData):
    """Expression matrix that PS should consume."""
    if LOGNORM_LAYER in expr.layers:
        return expr.layers[LOGNORM_LAYER]
    return expr.X


def _sample_indices(indices: np.ndarray, max_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministically sample at most max_cells indices."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= max_cells:
        return indices
    selected = rng.choice(indices, size=max_cells, replace=False)
    selected.sort()
    return selected.astype(np.int64, copy=False)


def _extract_gene_values(expr: ad.AnnData, gene_index: int, cell_indices: np.ndarray) -> np.ndarray:
    """Extract one gene for selected cells without densifying the full column."""
    if cell_indices.size == 0:
        return np.empty(0, dtype=np.float32)
    layer = _expression_layer(expr)
    values = layer[cell_indices, gene_index]
    if sparse.issparse(values):
        values = values.toarray().ravel()
    else:
        values = np.asarray(values).ravel()
    return values.astype(np.float32, copy=False)


# Standard full-object adapter


def _prepare_for_pertps(expr: ad.AnnData) -> ad.AnnData:
    """Build the complete working AnnData expected by pertps.

    This is deliberately retained for STANDARD mode to preserve the previous
    Replogle behavior.
    """
    layer = LOGNORM_LAYER if LOGNORM_LAYER in expr.layers else None
    work = ad.AnnData(
        X=(expr.layers[layer].copy() if layer else expr.X.copy()),
        obs=expr.obs[[OBS_TARGET, OBS_CLASS]].copy(),
        var=expr.var[[]].copy(),
    )
    work.obs_names = expr.obs_names.copy()
    work.var_names = expr.var_names.copy()
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    gene = expr.obs[OBS_TARGET].astype(str).to_numpy().copy()
    gene[klass == CLASS_NTC] = PERTPS_NEG_CTRL
    gene[(klass != CLASS_NTC) & (klass != CLASS_TARGETING)] = "Other"
    work.obs[PERTPS_GENE_COL] = pd.Categorical(gene)
    return work


# Large-data target adapter


def _prepare_target_for_pertps(
    expr: ad.AnnData, target_gene: str, target_indices: np.ndarray, control_indices: np.ndarray
) -> ad.AnnData:
    """Construct a small target + NTC AnnData for PS_python.

    The expression matrix is sliced before copying, so large datasets never
    create another full-matrix copy.
    """
    selected = np.concatenate([target_indices, control_indices]).astype(np.int64, copy=False)
    # Sorting generally improves CSR slicing locality.
    selected.sort()
    layer = _expression_layer(expr)
    X = layer[selected, :].copy()
    obs = expr.obs.iloc[selected][[OBS_TARGET, OBS_CLASS]].copy()
    work = ad.AnnData(X=X, obs=obs, var=expr.var[[]].copy())
    work.obs_names = expr.obs_names[selected].copy()
    work.var_names = expr.var_names.copy()
    klass = work.obs[OBS_CLASS].astype(str).to_numpy()
    pertps_gene = np.full(work.n_obs, target_gene, dtype=object)
    pertps_gene[klass == CLASS_NTC] = PERTPS_NEG_CTRL
    work.obs[PERTPS_GENE_COL] = pd.Categorical(pertps_gene)
    return work


# Expression validation


def _pct_control_expressing_standard(expr: ad.AnnData, gene: str) -> float:
    """Original control-expression check."""
    ctrl = (expr.obs[OBS_CLASS].astype(str) == CLASS_NTC).to_numpy()
    if not ctrl.any():
        return 100.0
    layer = _expression_layer(expr)
    col = layer[:, expr.var_names.get_loc(gene)]
    if sparse.issparse(col):
        col = col.toarray()
    values = np.asarray(col).ravel()
    return float(100 * np.mean(values[ctrl] > 0))


def _pct_control_expressing_large(expr: ad.AnnData, gene_index: int, control_indices: np.ndarray) -> float:
    """Memory-aware control-expression check."""
    if control_indices.size == 0:
        return 100.0
    values = _extract_gene_values(expr, gene_index, control_indices)
    return float(100 * np.mean(values > 0))


# Expression cuts / quadrants


def _expression_cut(reference: pd.Series | np.ndarray, pcfg) -> float:
    """Place the target-expression cut within the control distribution."""
    values = np.asarray(reference, dtype=float)
    if values.size == 0:
        return 0.0
    if pcfg.expression_cut == "median":
        return float(np.median(values))
    if pcfg.expression_cut == "quantile":
        return float(np.quantile(values, pcfg.expression_cut_quantile))
    return float(np.mean(values))


def _classify_quadrants(scores: np.ndarray, expression: np.ndarray, cut: float, ps_threshold: float) -> np.ndarray:
    """Vectorized PS/expression quadrant classification."""
    scores = np.asarray(scores, dtype=float)
    expression = np.asarray(expression, dtype=float)
    high_ps = scores >= ps_threshold
    high_expr = expression > cut
    quadrant = np.full(scores.shape, QUADRANT_LOW, dtype=object)
    quadrant[high_ps & ~high_expr] = QUADRANT_KD
    quadrant[high_ps & high_expr] = QUADRANT_ESCAPER
    quadrant[~high_ps & high_expr] = QUADRANT_NONRESPONDER
    return quadrant


# Original target summarizer


def _summarize_target_standard(expr: ad.AnnData, gene: str, series: pd.Series, cfg: Config):
    """Original per-target summary used in standard mode."""
    pcfg = cfg.ps_score
    klass = expr.obs[OBS_CLASS].astype(str)
    targets = expr.obs[OBS_TARGET].astype(str)
    layer = _expression_layer(expr)
    col = layer[:, expr.var_names.get_loc(gene)]
    if sparse.issparse(col):
        col = col.toarray()
    expression = pd.Series(np.asarray(col).ravel(), index=expr.obs_names)
    cells = series.index.intersection(expr.obs_names)
    if len(cells) == 0:
        return (None, None, float("nan"))
    is_target = (targets.loc[cells] == gene) & (klass.loc[cells] == CLASS_TARGETING)
    is_ctrl = klass.loc[cells] == CLASS_NTC
    if int(is_target.sum()) == 0:
        return (None, None, float("nan"))
    ctrl_expr = expression.loc[cells][is_ctrl]
    reference = ctrl_expr if len(ctrl_expr) else expression
    cut = _expression_cut(reference, pcfg)
    ps = series.loc[cells]
    ex = expression.loc[cells]
    quadrant = pd.Series(
        _classify_quadrants(ps.to_numpy(), ex.to_numpy(), cut, pcfg.ps_threshold), index=cells, dtype=object
    )
    tq = quadrant[is_target]
    cq = quadrant[is_ctrl]
    n_target = len(tq)
    ctrl_kd = float(100 * (cq == QUADRANT_KD).mean()) if len(cq) else float("nan")
    row = {
        "target_gene": gene,
        "n_perturbed_cells": int(n_target),
        "n_control_cells": int(is_ctrl.sum()),
        "mean_ps": float(ps[is_target].mean()),
        "median_ps": float(ps[is_target].median()),
        "pct_high_ps": float(100 * (ps[is_target] >= pcfg.ps_threshold).mean()),
        "pct_successful_kd": float(100 * (tq == QUADRANT_KD).mean()),
        "pct_escaper": float(100 * (tq == QUADRANT_ESCAPER).mean()),
        "pct_non_responder": float(100 * (tq == QUADRANT_NONRESPONDER).mean()),
        "pct_low_signal": float(100 * (tq == QUADRANT_LOW).mean()),
        "pct_controls_called_kd": (ctrl_kd),
        "net_pct_kd": (float(100 * (tq == QUADRANT_KD).mean() - ctrl_kd) if len(cq) else float("nan")),
        "expression_cut": cut,
        "expression_cut_method": (pcfg.expression_cut),
    }
    return (row, quadrant, cut)


# Large-mode target summarizer


def _summarize_target_large(
    expr: ad.AnnData, gene: str, series: pd.Series, target_indices: np.ndarray, control_indices: np.ndarray, cfg: Config
) -> Tuple[Optional[dict], Optional[pd.Series], float]:
    """Summarize one target without extracting a full n_cells expression vector."""
    pcfg = cfg.ps_score
    gene_index = expr.var_names.get_loc(gene)
    target_names = expr.obs_names[target_indices]
    control_names = expr.obs_names[control_indices]
    # Series returned by pertps includes target + NTC cells.
    target_common = series.index.intersection(target_names)
    control_common = series.index.intersection(control_names)
    if len(target_common) == 0:
        return (None, None, float("nan"))
    # Map cell names back to integer indices.
    target_pos = expr.obs_names.get_indexer(target_common)
    control_pos = expr.obs_names.get_indexer(control_common)
    target_expr = _extract_gene_values(expr, gene_index, target_pos.astype(np.int64))
    control_expr = _extract_gene_values(expr, gene_index, control_pos.astype(np.int64))
    reference = control_expr if control_expr.size else target_expr
    cut = _expression_cut(reference, pcfg)
    target_ps = series.loc[target_common].to_numpy(dtype=np.float32)
    target_quad_values = _classify_quadrants(target_ps, target_expr, cut, pcfg.ps_threshold)
    target_quad = pd.Series(target_quad_values, index=target_common, dtype=object)
    # Apply the same quadrant definition to sampled controls for auditability.
    if len(control_common):
        control_ps = series.loc[control_common].to_numpy(dtype=np.float32)
        control_quad = _classify_quadrants(control_ps, control_expr, cut, pcfg.ps_threshold)
        ctrl_kd = float(100 * np.mean(control_quad == QUADRANT_KD))
    else:
        ctrl_kd = float("nan")
    pct_success = float(100 * np.mean(target_quad_values == QUADRANT_KD))
    row = {
        "target_gene": gene,
        "n_perturbed_cells": int(len(target_common)),
        "n_control_cells": int(len(control_common)),
        "mean_ps": float(np.mean(target_ps)),
        "median_ps": float(np.median(target_ps)),
        "pct_high_ps": float(100 * np.mean(target_ps >= pcfg.ps_threshold)),
        "pct_successful_kd": (pct_success),
        "pct_escaper": float(100 * np.mean(target_quad_values == QUADRANT_ESCAPER)),
        "pct_non_responder": float(100 * np.mean(target_quad_values == QUADRANT_NONRESPONDER)),
        "pct_low_signal": float(100 * np.mean(target_quad_values == QUADRANT_LOW)),
        "pct_controls_called_kd": (ctrl_kd),
        "net_pct_kd": (pct_success - ctrl_kd if np.isfinite(ctrl_kd) else float("nan")),
        "expression_cut": cut,
        "expression_cut_method": (pcfg.expression_cut),
    }
    return (row, target_quad, cut)


# Standard PS execution


def _compute_ps_scores_standard(expr: ad.AnnData, cfg: Config) -> PSResults:
    """Original full-object PS execution path."""
    from pertps import PerturbAnalyzer

    pcfg = cfg.ps_score
    logger.info("PS execution mode: STANDARD (full PerturbAnalyzer working object)")
    work = _prepare_for_pertps(expr)
    analyzer = PerturbAnalyzer(work, neg_ctrl=PERTPS_NEG_CTRL, scale_factor=pcfg.scale_factor)
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    targets_col = expr.obs[OBS_TARGET].astype(str).to_numpy()
    n_ctrl = int((klass == CLASS_NTC).sum())
    if n_ctrl < pcfg.min_control_cells:
        msg = (
            f"Only {n_ctrl} non-targeting control cells "
            f"(need {pcfg.min_control_cells}); "
            "perturbation scores were skipped."
        )
        logger.warning(msg)
        return PSResults(summary=pd.DataFrame(), note=msg, large_mode=False)
    counts = pd.Series(targets_col[klass == CLASS_TARGETING]).value_counts()
    candidates = [target for target in sorted(counts.index) if counts[target] >= pcfg.min_cells_per_target]
    rows: List[dict] = []
    scores: Dict[str, pd.Series] = {}
    quadrants: Dict[str, pd.Series] = {}
    expression_cut: Dict[str, float] = {}
    skipped: List[dict] = []
    for gene in candidates:
        if gene not in expr.var_names:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": ("target gene not in the expression matrix"),
                }
            )
            continue
        pct_expressing = _pct_control_expressing_standard(expr, gene)
        if pct_expressing < cfg.perturbation.min_pct_expressing_control:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": (f"not detectably expressed in control cells ({pct_expressing:.2f}% of control cells)"),
                }
            )
            continue
        try:
            series = analyzer.calculate_ps_score(gene, top_n=pcfg.top_n_biomarkers)
        except Exception as exc:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": (f"pertps failed: {type(exc).__name__}: {exc}"),
                }
            )
            continue
        if series is None or len(series) == 0:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": ("pertps returned no score (too few cells or no signature)"),
                }
            )
            continue
        scores[gene] = series
        (summary_row, quadrant, cut) = _summarize_target_standard(expr, gene, series, cfg)
        if summary_row is None:
            skipped.append(
                {"target_gene": gene, "n_cells": int(counts[gene]), "reason": ("no perturbed cells carried a score")}
            )
            continue
        rows.append(summary_row)
        quadrants[gene] = quadrant
        expression_cut[gene] = cut
    if not rows:
        msg = "No target produced a usable perturbation score."
        logger.warning(msg)
        return PSResults(
            summary=pd.DataFrame(),
            skipped=pd.DataFrame(skipped),
            note=msg,
            ps_threshold=pcfg.ps_threshold,
            large_mode=False,
        )
    summary = pd.DataFrame(rows).sort_values("pct_successful_kd", ascending=False).reset_index(drop=True)
    (lda_umap, lda_label, lda_note) = _compute_lda_embedding_standard(analyzer, expr, list(summary["target_gene"]), cfg)
    logger.info(
        "Perturbation scores: %d target(s) scored, median %.0f%% successful knockdown",
        len(summary),
        float(summary["pct_successful_kd"].median()),
    )
    if skipped:
        logger.info("%d target(s) skipped in the PS stage", len(skipped))
    return PSResults(
        summary=summary,
        scores=scores,
        quadrants=quadrants,
        expression_cut=expression_cut,
        skipped=pd.DataFrame(skipped),
        ps_threshold=pcfg.ps_threshold,
        lda_umap=lda_umap,
        lda_label=lda_label,
        lda_note=lda_note,
        large_mode=False,
    )


# Large PS execution


def _compute_ps_scores_large(expr: ad.AnnData, cfg: Config) -> PSResults:
    """Scalable target-wise PS execution."""
    from pertps import PerturbAnalyzer

    from .high_moi import membership_index

    pcfg = cfg.ps_score
    membership = membership_index(expr, cfg)
    logger.info(
        "PS execution mode: LARGE-DATASET (target-wise PerturbAnalyzer objects)%s",
        " on high-MOI membership (perturbed = cells carrying the target)" if membership is not None else "",
    )
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    targets_col = expr.obs[OBS_TARGET].astype(str).to_numpy()
    targeting_indices = np.flatnonzero(klass == CLASS_TARGETING).astype(np.int64, copy=False)
    ntc_indices = np.flatnonzero(klass == CLASS_NTC).astype(np.int64, copy=False)
    n_ctrl = int(ntc_indices.size)
    if n_ctrl < pcfg.min_control_cells:
        msg = (
            f"Only {n_ctrl} non-targeting control cells "
            f"(need {pcfg.min_control_cells}); "
            "perturbation scores were skipped."
        )
        logger.warning(msg)
        return PSResults(
            summary=pd.DataFrame(),
            note=msg,
            ps_threshold=pcfg.ps_threshold,
            large_mode=True,
            membership_aware=membership is not None,
        )
    rng = np.random.default_rng(cfg.run.seed)
    sampled_ctrl = _sample_indices(ntc_indices, LARGE_PS_MAX_CONTROLS, rng)
    logger.info("Large PS mode: using %d/%d NTC cells per target", sampled_ctrl.size, ntc_indices.size)
    # Build target -> integer cell indices once.
    if membership is not None:
        target_indices = {target: membership.indices(target) for target in membership.targets}
    else:
        target_frame = pd.DataFrame({"target": (targets_col[targeting_indices]), "cell_index": (targeting_indices)})
        target_indices = {
            target: group["cell_index"].to_numpy(dtype=np.int64, copy=True)
            for target, group in target_frame.groupby("target", observed=True, sort=False)
        }
        del target_frame
    gc.collect()
    # Membership mode: every (cell, carried target) score goes into a sparse matrix; obs keeps the primary target's.
    mem_rows: List[np.ndarray] = []
    mem_cols: List[np.ndarray] = []
    mem_vals: List[np.ndarray] = []
    mem_targets: List[str] = []
    counts = {target: len(indices) for target, indices in target_indices.items()}
    candidates = sorted([target for target, n in counts.items() if n >= pcfg.min_cells_per_target])
    measured_index = {gene: idx for idx, gene in enumerate(expr.var_names)}
    rows: List[dict] = []
    skipped: List[dict] = []
    expression_cut: Dict[str, float] = {}
    # Large mode does NOT need one full per-target array over all cells.
    #
    # These vectors are the final scalable outputs.
    own_score = np.full(expr.n_obs, np.nan, dtype=np.float32)
    own_quadrant = np.full(expr.n_obs, "not applicable", dtype=object)
    # Keep target-cell Series only. This is useful for report plots without
    # duplicating 50k controls thousands of times.
    scores: Dict[str, pd.Series] = {}
    quadrants: Dict[str, pd.Series] = {}
    for target_number, gene in enumerate(candidates, start=1):
        indices = target_indices[gene]
        n_target = int(len(indices))
        if gene not in measured_index:
            skipped.append(
                {"target_gene": gene, "n_cells": n_target, "reason": ("target gene not in the expression matrix")}
            )
            continue
        gene_index = measured_index[gene]
        pct_expressing = _pct_control_expressing_large(expr, gene_index, sampled_ctrl)
        if pct_expressing < cfg.perturbation.min_pct_expressing_control:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": n_target,
                    "reason": (f"not detectably expressed in sampled control cells ({pct_expressing:.2f}% controls)"),
                }
            )
            continue
        work = None
        analyzer = None
        try:
            work = _prepare_target_for_pertps(expr, gene, indices, sampled_ctrl)
            analyzer = PerturbAnalyzer(work, neg_ctrl=PERTPS_NEG_CTRL, scale_factor=pcfg.scale_factor)
            series = analyzer.calculate_ps_score(gene, top_n=pcfg.top_n_biomarkers)
        except Exception as exc:
            skipped.append(
                {"target_gene": gene, "n_cells": n_target, "reason": (f"pertps failed: {type(exc).__name__}: {exc}")}
            )
            del analyzer
            del work
            gc.collect()
            continue
        if series is None or len(series) == 0:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": n_target,
                    "reason": ("pertps returned no score (too few cells or no signature)"),
                }
            )
            del analyzer
            del work
            gc.collect()
            continue
        (summary_row, target_quadrant, cut) = _summarize_target_large(expr, gene, series, indices, sampled_ctrl, cfg)
        if summary_row is None:
            skipped.append({"target_gene": gene, "n_cells": n_target, "reason": ("no perturbed cells carried a score")})
            del analyzer
            del work
            gc.collect()
            continue
        rows.append(summary_row)
        expression_cut[gene] = cut
        # Keep only target-cell PS values in persistent result structures.
        target_names = expr.obs_names[indices]
        target_series = series.reindex(target_names).dropna().astype(np.float32)
        scores[gene] = target_series
        quadrants[gene] = target_quadrant
        target_positions = expr.obs_names.get_indexer(target_series.index)
        valid_pos = target_positions >= 0
        pos = target_positions[valid_pos]
        values = target_series.to_numpy(dtype=np.float32)[valid_pos]
        quadrant_aligned = target_quadrant.reindex(target_series.index).fillna("not applicable").astype(str).to_numpy()
        if membership is not None:
            mem_rows.append(pos.astype(np.int64))
            mem_cols.append(np.full(pos.size, len(mem_targets), dtype=np.int64))
            mem_vals.append(values)
            mem_targets.append(gene)
            is_primary = targets_col[pos] == gene
            own_score[pos[is_primary]] = values[is_primary]
            own_quadrant[pos[is_primary]] = quadrant_aligned[valid_pos][is_primary]
        else:
            own_score[pos] = values
            own_quadrant[pos] = quadrant_aligned[valid_pos]
        del analyzer
        del work
        del series
        gc.collect()
        if target_number % 100 == 0:
            logger.info("Large PS progress: %d / %d targets", target_number, len(candidates))
    if not rows:
        msg = "No target produced a usable perturbation score."
        logger.warning(msg)
        return PSResults(
            summary=pd.DataFrame(),
            skipped=pd.DataFrame(skipped),
            note=msg,
            ps_threshold=pcfg.ps_threshold,
            large_mode=True,
            own_score=own_score,
            own_quadrant=own_quadrant,
            membership_aware=membership is not None,
        )
    summary = pd.DataFrame(rows).sort_values("pct_successful_kd", ascending=False).reset_index(drop=True)
    membership_scores = None
    if membership is not None and mem_targets:
        membership_scores = sparse.csr_matrix(
            (np.concatenate(mem_vals), (np.concatenate(mem_rows), np.concatenate(mem_cols))),
            shape=(expr.n_obs, len(mem_targets)),
            dtype=np.float32,
        )
    # Full PS_python LDA is intentionally not attempted over millions of cells.
    lda_umap = None
    lda_label = None
    if pcfg.compute_lda_umap:
        lda_note = (
            "Large-dataset mode: full PS_python LDA/UMAP was skipped because "
            "the upstream implementation scales and densifies its working "
            "matrix. Build the visualization on a bounded representative "
            f"subset (recommended <= {pcfg.lda_large_max_cells:,} cells)."
        )
        logger.warning("%s", lda_note)
    else:
        lda_note = "ps_score.compute_lda_umap is false"
    logger.info(
        "Large-data perturbation scores: %d target(s) scored, median %.0f%% successful knockdown",
        len(summary),
        float(summary["pct_successful_kd"].median()),
    )
    if skipped:
        logger.info("%d target(s) skipped in the PS stage", len(skipped))
    return PSResults(
        summary=summary,
        scores=scores,
        quadrants=quadrants,
        expression_cut=expression_cut,
        skipped=pd.DataFrame(skipped),
        ps_threshold=pcfg.ps_threshold,
        lda_umap=lda_umap,
        lda_label=lda_label,
        lda_note=lda_note,
        large_mode=True,
        own_score=own_score,
        own_quadrant=own_quadrant,
        membership_scores=membership_scores,
        membership_targets=mem_targets,
        membership_aware=membership is not None,
        note=(
            f"Large-dataset PS mode used up to {LARGE_PS_MAX_CONTROLS:,} NTC controls per target."
            + (
                " High-MOI membership: perturbed = cells carrying the target; obs['ps_score'] is the primary target's "
                "score, obsm['ps_score_membership'] every (cell, carried target) score."
                if membership is not None
                else ""
            )
        ),
    )


# LDA -- STANDARD mode only


def _compute_lda_embedding_standard(analyzer, expr: ad.AnnData, targets: List[str], cfg: Config):
    """Build the original PS_python supervised LDA/UMAP representation."""
    pcfg = cfg.ps_score
    if not pcfg.compute_lda_umap:
        return (None, None, "ps_score.compute_lda_umap is false")
    if not targets:
        return (None, None, "no scored targets to train on")
    logger.info("Building supervised LDA embedding over %d target(s)", len(targets))
    original = analyzer.adata
    capped = _cap_genes_for_lda(original, expr, pcfg.lda_max_genes)
    analyzer.adata = capped
    try:
        analyzer.compute_lda_umap(targets, n_pcs=pcfg.lda_n_pcs)
    except Exception as exc:
        note = f"the LDA embedding could not be built ({type(exc).__name__}: {exc})"
        logger.warning("Skipping the LDA embedding: %s", exc)
        analyzer.adata = original
        return (None, None, note)
    work = analyzer.adata
    analyzer.adata = original
    if "X_lda_umap" not in work.obsm:
        return (None, None, ("pertps did not return an X_lda_umap embedding"))
    coords = np.asarray(work.obsm["X_lda_umap"], dtype=np.float32)
    if "lda_label" in work.obs:
        labels = pd.Series(work.obs["lda_label"].astype(str).to_numpy(), index=work.obs_names)
    else:
        labels = None
    coords = pd.DataFrame(coords, index=work.obs_names).reindex(expr.obs_names).to_numpy(dtype=np.float32)
    if labels is not None:
        labels = labels.reindex(expr.obs_names)
    n_placed = int(np.isfinite(coords).all(axis=1).sum())
    logger.info("LDA embedding: %d/%d cells placed", n_placed, expr.n_obs)
    return (coords, labels, "")


def _cap_genes_for_lda(work: ad.AnnData, expr: ad.AnnData, max_genes: Optional[int]) -> ad.AnnData:
    """Restrict the standard-mode LDA input to a bounded gene set."""
    if not max_genes or work.n_vars <= max_genes:
        return work
    if "highly_variable" in expr.var.columns:
        keep = expr.var_names[expr.var["highly_variable"].to_numpy()]
        if len(keep) > max_genes:
            keep = keep[:max_genes]
    else:
        import scanpy as sc

        tmp = work.copy()
        sc.pp.highly_variable_genes(tmp, n_top_genes=max_genes)
        keep = tmp.var_names[tmp.var["highly_variable"].to_numpy()]
        del tmp
    keep_set = set(work.var_names)
    keep = [gene for gene in keep if gene in keep_set]
    logger.info("Capping LDA input at %d genes (from %d)", len(keep), work.n_vars)
    return work[:, keep].copy()


# Public API


def compute_ps_scores(expr: ad.AnnData, cfg: Config) -> Optional[PSResults]:
    """Compute per-cell perturbation-response scores.

    Execution mode is selected automatically.

    Standard mode
        Preserves the previous full-object PS_python workflow.

    Large mode
        Performs target-wise PS_python analysis using all target cells and a
        reproducibly sampled NTC reference.
    """
    pcfg = cfg.ps_score
    if not pcfg.enabled:
        logger.info("Perturbation scores disabled (ps_score.enabled: false)")
        return None
    if not pertps_available():
        msg = (
            "ps_score is enabled but the 'pertps' package is not installed. "
            'Install it with: pip install -e ".[ps]" '
            "(https://github.com/weili-lab/PS_python)"
        )
        if pcfg.require:
            raise PertpsUnavailable(msg)
        logger.warning("%s — skipping perturbation-score stage.", msg)
        return PSResults(summary=pd.DataFrame(), note=msg)
    from .high_moi import membership_index

    logger.info("Using pertps %s for per-cell perturbation scores", _pertps_version())
    membership = membership_index(expr, cfg)
    if membership is not None:
        target_counts = membership.counts
    else:
        klass = expr.obs[OBS_CLASS].astype(str)
        target_counts = expr.obs.loc[klass == CLASS_TARGETING, OBS_TARGET].astype(str).value_counts()
    n_testable_targets = int((target_counts >= pcfg.min_cells_per_target).sum())
    large_mode = cfg.use_large_mode(expr.n_obs, n_perturbations=n_testable_targets)
    if membership is not None and not large_mode:
        # pertps takes one label per cell; the target-wise path is the only one that fits multi-target cells
        logger.info("High-MOI membership: using the target-wise PS path (PS_python LDA/UMAP is not computed)")
        large_mode = True
    decision = resolve_stage_backend("ps_score", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)
    logger.info("PS input: %d cells, %d testable targets", expr.n_obs, n_testable_targets)
    if large_mode:
        logger.info("Large-dataset PS mode selected (%d cells, %d targets)", expr.n_obs, n_testable_targets)
        return _compute_ps_scores_large(expr, cfg)
    logger.info("Standard PS mode selected")
    return _compute_ps_scores_standard(expr, cfg)


# Attach results


def attach_scores(expr: ad.AnnData, results: Optional[PSResults]) -> ad.AnnData:
    """Attach PS outputs to ``AnnData.obs``.

    STANDARD mode preserves the previous detailed per-target columns.

    LARGE mode writes only:

        ps_score
        ps_quadrant

    because creating thousands of mostly-empty target-specific columns for
    millions of cells is unnecessary and can make ``obs`` enormous.
    """
    if results is None:
        return expr
    if results.membership_scores is not None:
        expr.obsm["ps_score_membership"] = results.membership_scores
        expr.uns["ps_score_membership_targets"] = list(results.membership_targets)
    # Large mode
    if results.large_mode:
        if results.own_score is not None:
            expr.obs["ps_score"] = results.own_score.astype(np.float32, copy=False)
        if results.own_quadrant is not None:
            expr.obs["ps_quadrant"] = pd.Categorical(
                pd.Series(results.own_quadrant, index=expr.obs_names).fillna("not applicable").astype(str)
            )
        return expr
    # Original standard mode
    if not results.scores:
        return expr
    own = pd.Series(np.nan, index=expr.obs_names, dtype=float)
    own_quad = pd.Series("not applicable", index=expr.obs_names, dtype=object)
    targets = expr.obs[OBS_TARGET].astype(str)
    for gene, series in results.scores.items():
        aligned = series.reindex(expr.obs_names)
        expr.obs[f"{PS_PREFIX}{gene}"] = aligned.to_numpy(dtype=float)
        quadrant = results.quadrants.get(gene)
        if quadrant is not None:
            expr.obs[f"{PS_PREFIX}quadrant_{gene}"] = pd.Categorical(
                quadrant.reindex(expr.obs_names).fillna("not applicable").astype(str)
            )
            mine = targets == gene
            own[mine] = aligned[mine]
            aligned_quadrant = quadrant.reindex(expr.obs_names)
            own_quad[mine] = aligned_quadrant[mine].fillna("not applicable")
    expr.obs["ps_score"] = own.to_numpy(dtype=float)
    expr.obs["ps_quadrant"] = pd.Categorical(own_quad.astype(str))
    return expr


# Comparison with group-level perturbation strength


def compare_with_perturbation_strength(
    results: Optional[PSResults], perturbation_table: pd.DataFrame, primary_control: str
) -> pd.DataFrame:
    """Join PS summaries with direct target-knockdown statistics."""
    if results is None or results.summary.empty or perturbation_table.empty:
        return pd.DataFrame()
    lfc_col = f"log2fc_{primary_control}"
    hit_col = f"is_hit_{primary_control}"
    keep = [col for col in ("target_gene", lfc_col, hit_col) if col in perturbation_table.columns]
    if len(keep) < 2:
        return pd.DataFrame()
    return results.summary.merge(perturbation_table[keep], on="target_gene", how="inner")
