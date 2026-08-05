"""Per-cell perturbation-response scores, via the lab's PS_python package.

The perturbation-strength stage (:mod:`perturbseq_pipeline.perturbation`) asks a
*group-level* question: across all cells carrying guides against a gene, did that
gene's expression drop? That answer is a single number per target, and it hides
the fact that a perturbed population is rarely uniform — some cells respond
strongly, others escape the knockdown entirely.

This stage adds the *per-cell* view by delegating to ``pertps``
(https://github.com/weili-lab/PS_python), the lab's Python implementation of the
scMAGeCK perturbation score. For each target it learns the expression signature
of the perturbation from target-vs-control cells, then projects every cell onto
that signature, giving a score in [0, 1] per cell.

Combining the score with the target's own expression classifies each perturbed
cell into one of four quadrants, which is the practically useful output:

``successful knockdown``
    high score, low target expression — the perturbation worked.
``escaper``
    high score, high target expression — the cell carries the guide and shows
    the signature but still expresses the gene.
``non-responder``
    low score, high expression — indistinguishable from an unperturbed cell.
``low signal``
    low score, low expression — uninformative, often low-quality cells.

``pertps`` is an optional dependency (``pip install -e ".[ps]"``). When it is
absent the stage is skipped and the report says so explicitly rather than
quietly omitting a section.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from .cluster import LOGNORM_LAYER
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET

logger = logging.getLogger(__name__)

#: Label ``pertps`` expects for the control population.
PERTPS_NEG_CTRL = "Non-Targeting"
#: ``obs`` column ``pertps`` reads the perturbation identity from.
PERTPS_GENE_COL = "gene"

#: Prefix for the per-cell score columns written into ``obs``.
PS_PREFIX = "ps_"

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


class PertpsUnavailable(RuntimeError):
    """Raised when ``pertps`` is required but not installed."""


@dataclass
class PSResults:
    """Per-cell perturbation scores and their per-target summary."""

    #: One row per target: cell counts and quadrant fractions.
    summary: pd.DataFrame
    #: ``{target: Series of per-cell scores}`` for the cells of that target.
    scores: Dict[str, pd.Series] = field(default_factory=dict)
    #: Per-target quadrant assignment, indexed by cell.
    quadrants: Dict[str, pd.Series] = field(default_factory=dict)
    #: Control median expression used as the horizontal cut, per target.
    expression_cut: Dict[str, float] = field(default_factory=dict)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    ps_threshold: float = 0.5
    #: Set when the stage ran but produced nothing usable.
    note: str = ""

    @property
    def targets(self) -> List[str]:
        return list(self.summary["target_gene"]) if not self.summary.empty else []

    def top_targets(self, n: int) -> List[str]:
        """Targets with the largest fraction of successfully knocked-down cells."""
        if self.summary.empty:
            return []
        return list(self.summary.head(n)["target_gene"])


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def pertps_available() -> bool:
    """True when the optional ``pertps`` package can be imported."""
    try:
        import pertps  # noqa: F401
    except Exception:
        return False
    return True


def _pertps_version() -> str:
    try:
        import pertps

        return getattr(pertps, "__version__", "unknown")
    except Exception:
        return "not installed"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _prepare_for_pertps(expr: ad.AnnData) -> ad.AnnData:
    """Build the AnnData shape ``pertps`` expects.

    ``PerturbAnalyzer`` reads the perturbation identity from ``obs['gene']`` and
    treats a fixed label as the control, so this maps the pipeline's own
    vocabulary onto that. The log-normalized layer is put in ``X`` explicitly:
    the scores are a projection of expression, so feeding scaled values would
    silently change them.
    """
    layer = LOGNORM_LAYER if LOGNORM_LAYER in expr.layers else None
    work = ad.AnnData(
        X=expr.layers[layer].copy() if layer else expr.X.copy(),
        obs=expr.obs[[OBS_TARGET, OBS_CLASS]].copy(),
        var=expr.var[[]].copy(),
    )
    work.obs_names = expr.obs_names
    work.var_names = expr.var_names

    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    gene = expr.obs[OBS_TARGET].astype(str).to_numpy().copy()
    gene[klass == CLASS_NTC] = PERTPS_NEG_CTRL
    # Anything that is neither a targeting cell nor a control is invisible to
    # the score; naming it keeps pertps from treating it as a target.
    gene[(klass != CLASS_NTC) & (klass != CLASS_TARGETING)] = "Other"
    work.obs[PERTPS_GENE_COL] = pd.Categorical(gene)
    return work


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def compute_ps_scores(expr: ad.AnnData, cfg: Config) -> Optional[PSResults]:
    """Compute per-cell perturbation scores for every testable target.

    Returns ``None`` when the stage is disabled or ``pertps`` is unavailable and
    not required, so the caller can record why the section is missing.
    """
    pcfg = cfg.ps_score
    if not pcfg.enabled:
        logger.info("Perturbation scores disabled (ps_score.enabled: false)")
        return None

    if not pertps_available():
        msg = (
            "ps_score is enabled but the 'pertps' package is not installed. "
            "Install it with:  pip install -e \".[ps]\"  "
            "(it comes from https://github.com/weili-lab/PS_python)"
        )
        if pcfg.require:
            raise PertpsUnavailable(msg)
        logger.warning("%s — skipping the perturbation-score stage.", msg)
        return PSResults(summary=pd.DataFrame(), note=msg)

    from pertps import PerturbAnalyzer

    logger.info("Using pertps %s for per-cell perturbation scores", _pertps_version())

    work = _prepare_for_pertps(expr)
    analyzer = PerturbAnalyzer(
        work, neg_ctrl=PERTPS_NEG_CTRL, scale_factor=pcfg.scale_factor
    )

    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    targets_col = expr.obs[OBS_TARGET].astype(str).to_numpy()
    n_ctrl = int((klass == CLASS_NTC).sum())
    if n_ctrl < pcfg.min_control_cells:
        msg = (
            f"Only {n_ctrl} non-targeting control cells (need "
            f"{pcfg.min_control_cells}); perturbation scores need a control "
            "population and were skipped."
        )
        logger.warning(msg)
        return PSResults(summary=pd.DataFrame(), note=msg)

    counts = pd.Series(targets_col[klass == CLASS_TARGETING]).value_counts()
    candidates = [t for t in sorted(counts.index) if counts[t] >= pcfg.min_cells_per_target]

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
                    "reason": "target gene not in the expression matrix",
                }
            )
            continue

        # A gene that controls do not express cannot be shown to be knocked
        # down. Its control median is 0, so every cell falls on the "low
        # expression" side and any high score is misread as a successful
        # knockdown — the olfactory-receptor controls in a TF screen would
        # otherwise top the efficiency ranking.
        pct_expressing = _pct_control_expressing(expr, gene)
        if pct_expressing < cfg.perturbation.min_pct_expressing_control:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": (
                        f"not detectably expressed in control cells "
                        f"({pct_expressing:.2f}% of control cells)"
                    ),
                }
            )
            continue
        try:
            series = analyzer.calculate_ps_score(gene, top_n=pcfg.top_n_biomarkers)
        except Exception as exc:  # upstream raises bare errors on odd inputs
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": f"pertps failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if series is None or len(series) == 0:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": "pertps returned no score (too few cells or no signature)",
                }
            )
            continue

        scores[gene] = series
        summary_row, quad, cut = _summarize_target(expr, gene, series, cfg)
        if summary_row is None:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_cells": int(counts[gene]),
                    "reason": "no perturbed cells carried a score",
                }
            )
            continue
        rows.append(summary_row)
        quadrants[gene] = quad
        expression_cut[gene] = cut

    if not rows:
        msg = "No target produced a usable perturbation score."
        logger.warning(msg)
        return PSResults(
            summary=pd.DataFrame(),
            skipped=pd.DataFrame(skipped),
            note=msg,
            ps_threshold=pcfg.ps_threshold,
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values("pct_successful_kd", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "Perturbation scores: %d target(s) scored, median %.0f%% of perturbed "
        "cells classed as successful knockdown",
        len(summary),
        float(summary["pct_successful_kd"].median()),
    )
    if skipped:
        logger.info("%d target(s) skipped in the score stage", len(skipped))

    return PSResults(
        summary=summary,
        scores=scores,
        quadrants=quadrants,
        expression_cut=expression_cut,
        skipped=pd.DataFrame(skipped),
        ps_threshold=pcfg.ps_threshold,
    )


def _expression_cut(reference: pd.Series, pcfg) -> float:
    """Place the horizontal quadrant cut inside the control distribution."""
    values = np.asarray(reference, dtype=float)
    if values.size == 0:
        return 0.0
    if pcfg.expression_cut == "median":
        return float(np.median(values))
    if pcfg.expression_cut == "quantile":
        return float(np.quantile(values, pcfg.expression_cut_quantile))
    return float(np.mean(values))


def _pct_control_expressing(expr: ad.AnnData, gene: str) -> float:
    """Percent of non-targeting control cells with non-zero expression."""
    from scipy import sparse

    ctrl = (expr.obs[OBS_CLASS].astype(str) == CLASS_NTC).to_numpy()
    if not ctrl.any():
        return 100.0
    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    col = layer[:, expr.var_names.get_loc(gene)]
    if sparse.issparse(col):
        col = col.toarray()
    values = np.asarray(col).ravel()
    return float(100 * np.mean(values[ctrl] > 0))


def _summarize_target(expr: ad.AnnData, gene: str, series: pd.Series, cfg: Config):
    """Quadrant-classify the cells of one target and summarize them."""
    from scipy import sparse

    pcfg = cfg.ps_score
    klass = expr.obs[OBS_CLASS].astype(str)
    targets = expr.obs[OBS_TARGET].astype(str)

    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    col = layer[:, expr.var_names.get_loc(gene)]
    if sparse.issparse(col):
        col = col.toarray()
    expression = pd.Series(np.asarray(col).ravel(), index=expr.obs_names)

    cells = series.index.intersection(expr.obs_names)
    if len(cells) == 0:
        return None, None, float("nan")

    is_target = (targets.loc[cells] == gene) & (klass.loc[cells] == CLASS_TARGETING)
    is_ctrl = klass.loc[cells] == CLASS_NTC
    if int(is_target.sum()) == 0:
        return None, None, float("nan")

    # The horizontal cut sits inside the control population: "low" means below
    # what an unperturbed cell typically shows. The mean is the default rather
    # than the median because single-cell counts are zero-inflated and the
    # median collapses to exactly 0 for most targets.
    ctrl_expr = expression.loc[cells][is_ctrl]
    reference = ctrl_expr if len(ctrl_expr) else expression
    cut = _expression_cut(reference, pcfg)

    ps = series.loc[cells]
    ex = expression.loc[cells]
    high_ps = ps >= pcfg.ps_threshold
    high_expr = ex > cut

    quad = pd.Series(QUADRANT_LOW, index=cells, dtype=object)
    quad[high_ps & ~high_expr] = QUADRANT_KD
    quad[high_ps & high_expr] = QUADRANT_ESCAPER
    quad[~high_ps & high_expr] = QUADRANT_NONRESPONDER

    tq = quad[is_target]
    cq = quad[is_ctrl]
    n = len(tq)
    # The same classification applied to control cells. Any cut can be argued
    # over; reporting what fraction of *controls* it would call knocked down
    # makes the number auditable and turns the headline into a net effect.
    ctrl_kd = float(100 * (cq == QUADRANT_KD).mean()) if len(cq) else float("nan")
    row = {
        "target_gene": gene,
        "n_perturbed_cells": int(n),
        "n_control_cells": int(is_ctrl.sum()),
        "mean_ps": float(ps[is_target].mean()),
        "median_ps": float(ps[is_target].median()),
        "pct_high_ps": float(100 * high_ps[is_target].mean()),
        "pct_successful_kd": float(100 * (tq == QUADRANT_KD).mean()),
        "pct_escaper": float(100 * (tq == QUADRANT_ESCAPER).mean()),
        "pct_non_responder": float(100 * (tq == QUADRANT_NONRESPONDER).mean()),
        "pct_low_signal": float(100 * (tq == QUADRANT_LOW).mean()),
        "pct_controls_called_kd": ctrl_kd,
        "net_pct_kd": float(100 * (tq == QUADRANT_KD).mean() - ctrl_kd)
        if len(cq)
        else float("nan"),
        "expression_cut": cut,
        "expression_cut_method": cfg.ps_score.expression_cut,
    }
    return row, quad, cut


def attach_scores(expr: ad.AnnData, results: Optional[PSResults]) -> ad.AnnData:
    """Write per-cell scores and quadrant labels into ``obs``.

    Each target gets ``ps_<GENE>`` (score, NaN for cells outside that target's
    comparison) and ``ps_quadrant_<GENE>``. A single ``ps_score`` column carries
    each cell's score for its *own* target, which is the column most analyses
    actually want.
    """
    if results is None or not results.scores:
        return expr
    own = pd.Series(np.nan, index=expr.obs_names, dtype=float)
    own_quad = pd.Series("not applicable", index=expr.obs_names, dtype=object)
    targets = expr.obs[OBS_TARGET].astype(str)

    for gene, series in results.scores.items():
        aligned = series.reindex(expr.obs_names)
        expr.obs[f"{PS_PREFIX}{gene}"] = aligned.to_numpy(dtype=float)
        quad = results.quadrants.get(gene)
        if quad is not None:
            expr.obs[f"{PS_PREFIX}quadrant_{gene}"] = pd.Categorical(
                quad.reindex(expr.obs_names).fillna("not applicable").astype(str)
            )
            mine = targets == gene
            own[mine] = aligned[mine]
            own_quad[mine] = quad.reindex(expr.obs_names)[mine].fillna("not applicable")

    expr.obs["ps_score"] = own.to_numpy(dtype=float)
    expr.obs["ps_quadrant"] = pd.Categorical(own_quad.astype(str))
    return expr


def compare_with_perturbation_strength(
    results: Optional[PSResults], perturbation_table: pd.DataFrame, primary_control: str
) -> pd.DataFrame:
    """Join the per-cell scores to the group-level knockdown results.

    The two are independent measurements of the same thing — one a projection
    onto an expression signature, the other a direct test of the target's own
    expression — so their agreement is a useful check on both.
    """
    if results is None or results.summary.empty or perturbation_table.empty:
        return pd.DataFrame()
    lfc_col = f"log2fc_{primary_control}"
    hit_col = f"is_hit_{primary_control}"
    keep = [c for c in ("target_gene", lfc_col, hit_col) if c in perturbation_table.columns]
    if len(keep) < 2:
        return pd.DataFrame()
    merged = results.summary.merge(perturbation_table[keep], on="target_gene", how="inner")
    return merged
