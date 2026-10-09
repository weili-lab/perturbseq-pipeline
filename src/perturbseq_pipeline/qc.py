"""Quality control: standard single-cell metrics plus Perturb-seq guide QC.

Metrics are normally computed *before* the strict filters run, so the diagnostic
figures show the raw distribution and the reader can judge whether configured
thresholds were sensible. Every filtering step is recorded in a table that is
included in the report.

Large-dataset execution
-----------------------
For ordinary datasets, QC preserves the original behaviour:

* annotate mitochondrial/ribosomal/hemoglobin genes;
* calculate Scanpy QC metrics from the expression/count matrix;
* apply configured cell and gene filters;
* copy a subset only when filtering is actually required.

For million-cell H5AD datasets, several additional safeguards are used.

1. **Reuse existing QC metrics when possible.**
   Large processed datasets frequently already contain columns such as
   ``total_counts``, ``n_genes_by_counts``, ``pct_counts_mt`` and
   ``pct_counts_ribo``. Recalculating them across millions of cells and tens of
   thousands of genes is unnecessary when the existing metrics are complete.

2. **Avoid no-op AnnData copies.**
   A statement such as::

       expr = expr[mask].copy()

   can transiently duplicate an enormous expression object. Large-data mode
   first checks whether any cells actually fail the mask and leaves ``expr``
   untouched when nothing would be removed.

3. **Filter once where possible.**
   Multiple cell thresholds are combined into one Boolean mask before slicing,
   so a million-cell dataset is copied at most once during strict cell QC rather
   than once for every threshold.

4. **Preserve scientific QC semantics.**
   Threshold definitions are unchanged. The scalable path changes only how
   filtering and metric reuse are implemented.

Execution mode is selected automatically. Replogle-sized datasets retain the
standard behaviour, while datasets with >=1 million cells use the memory-aware
path.
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .config import Config
from .guides import (
    CLASS_AMBIGUOUS,
    CLASS_NTC,
    CLASS_TARGETING,
    CLASS_UNASSIGNED,
    OBS_CLASS,
    OBS_NDETECTED,
    OBS_SECOND,
    OBS_TOP,
    OBS_TOTAL,
)
from .io import LANE_KEY

logger = logging.getLogger(__name__)


# Public QC metrics

CELL_QC_METRICS = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"]


# QC columns sufficient for most downstream reporting/filtering.
_REQUIRED_CELL_QC = {"n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"}


# Helpers


def _has_existing_qc_metrics(expr: ad.AnnData) -> bool:
    """Whether the input already contains the core per-cell QC metrics."""
    return _REQUIRED_CELL_QC.issubset(set(expr.obs.columns))


def _record_step(
    steps: List[dict], step: str, detail: str, cells_before: int, cells_after: int, genes_before: int, genes_after: int
) -> None:
    """Append one QC filtering step."""
    steps.append(
        {
            "step": step,
            "threshold": detail,
            "cells_before": cells_before,
            "cells_after": cells_after,
            "cells_removed": (cells_before - cells_after),
            "genes_before": genes_before,
            "genes_after": genes_after,
            "genes_removed": (genes_before - genes_after),
        }
    )


# Gene annotation


def annotate_gene_classes(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Flag mitochondrial, ribosomal and hemoglobin genes in ``var``.

    Existing annotation columns are intentionally overwritten so they are
    guaranteed to reflect the current QC configuration.
    """
    q = cfg.qc
    names = expr.var_names.astype(str).str.upper()
    expr.var["mt"] = names.str.startswith(q.mito_prefix.upper())
    expr.var["ribo"] = names.str.startswith(tuple(prefix.upper() for prefix in q.ribo_prefix))
    expr.var["hb"] = names.str.contains(q.hb_pattern, regex=True)
    logger.info(
        "Gene classes: %d mitochondrial, %d ribosomal, %d hemoglobin",
        int(expr.var["mt"].sum()),
        int(expr.var["ribo"].sum()),
        int(expr.var["hb"].sum()),
    )
    if expr.var["mt"].sum() == 0:
        logger.warning(
            "No mitochondrial genes matched prefix %r — check qc.mito_prefix (human 'MT-', mouse 'mt-').", q.mito_prefix
        )
    return expr


# QC metric computation


def compute_qc_metrics(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Compute or reuse standard QC metrics.

    For large H5AD inputs that already contain complete QC columns, expensive
    matrix-wide recalculation is skipped.

    Gene classes are still annotated so downstream code sees consistent
    ``var['mt']``, ``var['ribo']`` and ``var['hb']`` flags.
    """
    expr = annotate_gene_classes(expr, cfg)
    large_mode = cfg.use_large_mode(expr.n_obs)
    if large_mode and _has_existing_qc_metrics(expr):
        logger.info(
            "Large-dataset QC mode: reusing existing per-cell QC metrics "
            "for %d cells instead of recalculating them from the matrix",
            expr.n_obs,
        )
        return expr
    logger.info("Computing Scanpy QC metrics over %d cells x %d genes", expr.n_obs, expr.n_vars)
    sc.pp.calculate_qc_metrics(expr, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True, percent_top=None)
    return expr


# Prefilter


def prefilter(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Apply the permissive initial cell/gene filters.

    For large datasets, no operation is performed when both thresholds are
    disabled. This matters for already-QC-filtered H5AD inputs such as KOLF,
    where a gratuitous slicing/copying pass would provide no benefit.
    """
    q = cfg.qc
    n0 = expr.n_obs
    g0 = expr.n_vars
    if not q.min_genes_per_cell and not q.min_cells_per_gene:
        logger.info(
            "Pre-filter (min_genes=None, min_cells=None): %d -> %d cells, %d -> %d genes",
            n0,
            expr.n_obs,
            g0,
            expr.n_vars,
        )
        return expr
    if q.min_genes_per_cell:
        sc.pp.filter_cells(expr, min_genes=q.min_genes_per_cell)
    if q.min_cells_per_gene:
        sc.pp.filter_genes(expr, min_cells=q.min_cells_per_gene)
    logger.info(
        "Pre-filter (min_genes=%s, min_cells=%s): %d -> %d cells, %d -> %d genes",
        q.min_genes_per_cell,
        q.min_cells_per_gene,
        n0,
        expr.n_obs,
        g0,
        expr.n_vars,
    )
    return expr


# Standard filtering path


def _filter_standard(expr: ad.AnnData, cfg: Config) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Original sequential filtering path for ordinary datasets."""
    q = cfg.qc
    steps: List[dict] = []
    _record_step(steps, "input", "-", expr.n_obs, expr.n_obs, expr.n_vars, expr.n_vars)
    if q.min_genes_final and q.min_genes_final > 0:
        c0 = expr.n_obs
        g0 = expr.n_vars
        sc.pp.filter_cells(expr, min_genes=q.min_genes_final)
        _record_step(steps, "min genes per cell", f">= {q.min_genes_final}", c0, expr.n_obs, g0, expr.n_vars)
    if q.min_counts_per_cell:
        c0 = expr.n_obs
        g0 = expr.n_vars
        sc.pp.filter_cells(expr, min_counts=q.min_counts_per_cell)
        _record_step(steps, "min counts per cell", f">= {q.min_counts_per_cell}", c0, expr.n_obs, g0, expr.n_vars)
    if q.max_pct_mt is not None:
        c0 = expr.n_obs
        g0 = expr.n_vars
        mask = expr.obs["pct_counts_mt"].to_numpy() < q.max_pct_mt
        if not mask.all():
            expr = expr[mask].copy()
        _record_step(steps, "max % mitochondrial", f"< {q.max_pct_mt}%", c0, expr.n_obs, g0, expr.n_vars)
    if q.max_pct_hb is not None:
        c0 = expr.n_obs
        g0 = expr.n_vars
        mask = expr.obs["pct_counts_hb"].to_numpy() < q.max_pct_hb
        if not mask.all():
            expr = expr[mask].copy()
        _record_step(steps, "max % hemoglobin", f"< {q.max_pct_hb}%", c0, expr.n_obs, g0, expr.n_vars)
    if q.min_cells_per_gene and q.min_cells_per_gene > 0:
        c0 = expr.n_obs
        g0 = expr.n_vars
        sc.pp.filter_genes(expr, min_cells=q.min_cells_per_gene)
        _record_step(steps, "min cells per gene", f">= {q.min_cells_per_gene}", c0, expr.n_obs, g0, expr.n_vars)
    if expr.n_obs == 0:
        raise ValueError(
            "All cells were removed by QC filters. "
            "Loosen qc.min_genes_final / qc.max_pct_mt, "
            "or verify that the input matrix is appropriate."
        )
    table = pd.DataFrame(steps)
    return (expr, table)


# Large filtering path


def _filter_large(expr: ad.AnnData, cfg: Config) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Memory-aware filtering for million-cell datasets.

    Cell thresholds are combined into one Boolean mask and the AnnData object
    is sliced at most once.
    """
    q = cfg.qc
    steps: List[dict] = []
    original_cells = expr.n_obs
    original_genes = expr.n_vars
    _record_step(steps, "input", "-", original_cells, original_cells, original_genes, original_genes)
    # Build a single cumulative cell mask
    keep = np.ones(expr.n_obs, dtype=bool)
    # Keep a running number only for reporting.
    running_cells = expr.n_obs
    if q.min_genes_final and q.min_genes_final > 0:
        if "n_genes_by_counts" not in expr.obs.columns:
            raise ValueError("qc.min_genes_final is configured but 'n_genes_by_counts' is missing from obs.")
        before = running_cells
        step_mask = expr.obs["n_genes_by_counts"].to_numpy() >= q.min_genes_final
        keep &= step_mask
        running_cells = int(keep.sum())
        _record_step(
            steps,
            "min genes per cell",
            f">= {q.min_genes_final}",
            before,
            running_cells,
            original_genes,
            original_genes,
        )
    if q.min_counts_per_cell:
        if "total_counts" not in expr.obs.columns:
            raise ValueError("qc.min_counts_per_cell is configured but 'total_counts' is missing from obs.")
        before = running_cells
        step_mask = expr.obs["total_counts"].to_numpy() >= q.min_counts_per_cell
        keep &= step_mask
        running_cells = int(keep.sum())
        _record_step(
            steps,
            "min counts per cell",
            f">= {q.min_counts_per_cell}",
            before,
            running_cells,
            original_genes,
            original_genes,
        )
    if q.max_pct_mt is not None:
        if "pct_counts_mt" not in expr.obs.columns:
            raise ValueError("qc.max_pct_mt is configured but 'pct_counts_mt' is missing from obs.")
        before = running_cells
        step_mask = expr.obs["pct_counts_mt"].to_numpy() < q.max_pct_mt
        keep &= step_mask
        running_cells = int(keep.sum())
        _record_step(
            steps, "max % mitochondrial", f"< {q.max_pct_mt}%", before, running_cells, original_genes, original_genes
        )
    if q.max_pct_hb is not None:
        if "pct_counts_hb" not in expr.obs.columns:
            raise ValueError("qc.max_pct_hb is configured but 'pct_counts_hb' is missing from obs.")
        before = running_cells
        step_mask = expr.obs["pct_counts_hb"].to_numpy() < q.max_pct_hb
        keep &= step_mask
        running_cells = int(keep.sum())
        _record_step(
            steps, "max % hemoglobin", f"< {q.max_pct_hb}%", before, running_cells, original_genes, original_genes
        )
    # Slice ONCE, and only if required
    if not keep.all():
        n_remove = int((~keep).sum())
        logger.info("Large-data QC: applying one combined cell filter (removing %d/%d cells)", n_remove, expr.n_obs)
        expr = expr[keep].copy()
        gc.collect()
    else:
        logger.info(
            "Large-data QC: all %d cells passed configured cell thresholds; avoiding an unnecessary AnnData copy",
            expr.n_obs,
        )
    if expr.n_obs == 0:
        raise ValueError("All cells were removed by QC filters.")
    # Gene filtering
    if q.min_cells_per_gene and q.min_cells_per_gene > 0:
        c0 = expr.n_obs
        g0 = expr.n_vars
        # If preprocessing already supplied n_cells_by_counts we can filter from
        # metadata instead of rescanning the entire expression matrix.
        if "n_cells_by_counts" in expr.var.columns:
            gene_keep = expr.var["n_cells_by_counts"].to_numpy() >= q.min_cells_per_gene
            if not gene_keep.all():
                expr = expr[:, gene_keep].copy()
                gc.collect()
        elif "n_cells" in expr.var.columns:
            gene_keep = expr.var["n_cells"].to_numpy() >= q.min_cells_per_gene
            if not gene_keep.all():
                expr = expr[:, gene_keep].copy()
                gc.collect()
        else:
            logger.info(
                "Large-data QC: no precomputed per-gene detection counts found; falling back to Scanpy filter_genes"
            )
            sc.pp.filter_genes(expr, min_cells=q.min_cells_per_gene)
        _record_step(steps, "min cells per gene", f">= {q.min_cells_per_gene}", c0, expr.n_obs, g0, expr.n_vars)
    table = pd.DataFrame(steps)
    return (expr, table)


# Public filtering API


def filter_cells_and_genes(expr: ad.AnnData, cfg: Config) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Apply configured QC filters and record every step.

    Execution switches automatically between the original sequential
    implementation and the large-data memory-aware implementation.
    """
    if cfg.use_large_mode(expr.n_obs):
        logger.info("QC execution mode: LARGE-DATASET (combined masks / no-op copy avoidance)")
        expr, table = _filter_large(expr, cfg)
    else:
        logger.info("QC execution mode: STANDARD")
        expr, table = _filter_standard(expr, cfg)
    logger.info(
        "QC filtering: %d -> %d cells, %d -> %d genes",
        int(table["cells_before"].iloc[0]),
        expr.n_obs,
        int(table["genes_before"].iloc[0]),
        expr.n_vars,
    )
    return (expr, table)


# Standard QC summary


def qc_summary_table(expr: ad.AnnData, lane_key: str = LANE_KEY) -> pd.DataFrame:
    """Per-lane summary of standard QC metrics.

    This operates entirely on ``obs`` and therefore remains inexpensive even
    for million-cell datasets.
    """
    obs = expr.obs
    if lane_key in obs.columns:
        group = obs[lane_key].astype(str)
    else:
        group = pd.Series(["all"] * expr.n_obs, index=obs.index)
    rows = []
    metrics = [
        ("n_genes_by_counts", "median_genes"),
        ("total_counts", "median_umis"),
        ("pct_counts_mt", "median_pct_mt"),
        ("pct_counts_ribo", "median_pct_ribo"),
    ]
    for lane, sub in obs.groupby(group, observed=True):
        row = {"lane": lane, "n_cells": len(sub)}
        for metric, label in metrics:
            if metric in sub.columns:
                row[label] = float(np.median(sub[metric].to_numpy()))
        rows.append(row)
    total = {"lane": "ALL", "n_cells": expr.n_obs}
    for metric, label in metrics:
        if metric in obs.columns:
            total[label] = float(np.median(obs[metric].to_numpy()))
    rows.append(total)
    return pd.DataFrame(rows).round(2)


# Perturb-seq-specific QC


def guide_qc_summary(expr: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Headline Perturb-seq guide-QC values."""
    obs = expr.obs
    n = expr.n_obs
    rows: List[Tuple[str, object]] = [("Cells after QC", f"{n:,}")]
    if OBS_CLASS in obs.columns:
        counts = obs[OBS_CLASS].value_counts()
        for label, key in [
            ("Cells with a targeting guide", CLASS_TARGETING),
            ("Cells with a non-targeting guide", CLASS_NTC),
            ("Ambiguous (no dominant guide)", CLASS_AMBIGUOUS),
            ("Unassigned (no guide counts)", CLASS_UNASSIGNED),
        ]:
            count = int(counts.get(key, 0))
            rows.append((label, (f"{count:,} ({100 * count / max(n, 1):.1f}%)")))
    if OBS_TOTAL in obs.columns:
        rows.append(("Median guide UMIs per cell", f"{np.median(obs[OBS_TOTAL].to_numpy()):,.0f}"))
    if OBS_NDETECTED in obs.columns:
        detected = obs[OBS_NDETECTED].to_numpy()
        rows.append(
            (
                (f"Median guides detected per cell (> {cfg.guides.detection_threshold} UMI)"),
                f"{np.median(detected):.0f}",
            )
        )
        rows.append(("Cells with exactly 1 guide detected", f"{int((detected == 1).sum()):,}"))
        rows.append(("Cells with >1 guide detected (multiplet-like)", f"{int((detected > 1).sum()):,}"))
        rows.append(("Estimated MOI (mean guides/cell)", f"{detected.mean():.2f}"))
    if OBS_TOP in obs.columns and OBS_SECOND in obs.columns:
        top = obs[OBS_TOP].to_numpy()
        second = obs[OBS_SECOND].to_numpy()
        ratio = top / np.maximum(second, 1.0)
        rows.append(("Median top:second guide ratio", f"{np.median(ratio):.1f}"))
    if cfg.guides.assignment_mode == "high_moi":
        rows.extend(_guide_qc_rows_high_moi(expr, cfg))
    return pd.DataFrame(rows, columns=["metric", "value"])


def _guide_qc_rows_high_moi(expr: ad.AnnData, cfg: Config) -> List[Tuple[str, object]]:
    """Membership rows appended to the guide-QC summary in high-MOI mode."""
    from .high_moi import OBS_N_GUIDES, OBS_N_TARGETS, OBS_NTC_ONLY

    obs = expr.obs
    if OBS_N_GUIDES not in obs.columns:
        return []
    ng = obs[OBS_N_GUIDES].to_numpy()
    nt = obs[OBS_N_TARGETS].to_numpy()
    assigned = ng > 0
    med = lambda v: f"{np.median(v):.0f}" if v.size else "n/a"
    hcfg = cfg.guides.high_moi
    return [
        (f"Cells with >= 1 called guide (high-MOI, >= {hcfg.min_umi} UMI)", f"{int(assigned.sum()):,} ({100 * assigned.mean():.1f}%)"),
        ("Median called guides per assigned cell", med(ng[assigned])),
        ("Median targets per assigned cell", med(nt[assigned])),
        ("NTC-only cells", f"{int(obs[OBS_NTC_ONLY].sum()):,}"),
        (f"Cells above max_guides_per_cell={hcfg.max_guides_per_cell} (ambiguous)", f"{int((obs[OBS_CLASS].astype(str) == CLASS_AMBIGUOUS).sum()):,}"),
    ]


def check_guide_qc(expr: ad.AnnData, cfg: Config) -> List[str]:
    """Return human-readable warnings about guide assignment quality."""
    if cfg.guides.assignment_mode == "high_moi":
        return _check_guide_qc_high_moi(expr, cfg)
    warnings: List[str] = []
    obs = expr.obs
    n = expr.n_obs
    if OBS_CLASS not in obs.columns:
        return warnings
    counts = obs[OBS_CLASS].value_counts()
    assigned = int(counts.get(CLASS_TARGETING, 0)) + int(counts.get(CLASS_NTC, 0))
    frac = assigned / max(n, 1)
    if frac < 0.5:
        warnings.append(
            f"Only {100 * frac:.1f}% of cells received a confident guide call. "
            f"Consider lowering guides.min_umi "
            f"(currently {cfg.guides.min_umi}) or guides.dominance_ratio "
            f"(currently {cfg.guides.dominance_ratio})."
        )
    if int(counts.get(CLASS_NTC, 0)) == 0:
        warnings.append(
            "No non-targeting control cells were detected. "
            "The 'ntc' perturbation-control arm will be unavailable; "
            "check guides.ntc_patterns against the library naming."
        )
    ambiguous = int(counts.get(CLASS_AMBIGUOUS, 0))
    if ambiguous / max(n, 1) > 0.3:
        warnings.append(
            f"{100 * ambiguous / max(n, 1):.1f}% of cells are ambiguous "
            "(no dominant guide). This can indicate high MOI or "
            "guide-index swapping."
        )
    if OBS_NDETECTED in obs.columns:
        detected = obs[OBS_NDETECTED].to_numpy()
        multi = float((detected > 1).mean())
        if multi > 0.3:
            warnings.append(
                f"{100 * multi:.1f}% of cells carry more than one detected "
                "guide; single-guide analysis assumptions may not hold."
            )
    return warnings


def _check_guide_qc_high_moi(expr: ad.AnnData, cfg: Config) -> List[str]:
    """High-MOI mode: multi-guide cells are the design, so the single-guide warnings do not apply."""
    from .high_moi import OBS_N_GUIDES, OBS_NTC_ONLY

    warnings: List[str] = []
    obs = expr.obs
    n = expr.n_obs
    if OBS_CLASS not in obs.columns or OBS_N_GUIDES not in obs.columns:
        return warnings
    hcfg = cfg.guides.high_moi
    counts = obs[OBS_CLASS].value_counts()
    assigned = int(counts.get(CLASS_TARGETING, 0)) + int(counts.get(CLASS_NTC, 0))
    frac = assigned / max(n, 1)
    if frac < 0.5:
        warnings.append(
            f"Only {100 * frac:.1f}% of cells received at least one called guide. "
            f"Consider lowering guides.high_moi.min_umi (currently {hcfg.min_umi}) or "
            f"guides.high_moi.min_frac_of_top (currently {hcfg.min_frac_of_top})."
        )
    n_ntc_only = int(obs[OBS_NTC_ONLY].sum())
    if n_ntc_only < cfg.perturbation.min_control_cells:
        warnings.append(
            f"Only {n_ntc_only} NTC-only cells (cells carrying non-targeting guides and no targeting guide); "
            f"perturbation.min_control_cells is {cfg.perturbation.min_control_cells}, so the 'ntc' control arm "
            "is skipped. Use the 'other' arm (perturbation.primary_control: other): targeting cells that do not "
            "carry the tested target."
        )
    over_cap = int(counts.get(CLASS_AMBIGUOUS, 0))
    if over_cap / max(n, 1) > 0.05:
        warnings.append(
            f"{100 * over_cap / max(n, 1):.1f}% of cells carry more than guides.high_moi.max_guides_per_cell="
            f"{hcfg.max_guides_per_cell} called guides and are classed ambiguous (doublet-like). "
            "Check the rank-ordered guide UMI profile; raise the cap only if the knee supports it."
        )
    warnings.append(
        "High-MOI mode: perturbation strength, cluster enrichment, co-functional modules and perturbation distance use "
        "the membership matrix (perturbed = cells carrying the target; 'other' = targeting cells not carrying it; "
        "non-targeting guides are tested as negative-control pseudo-targets). Per-cell scores (PS, lochNESS) and the "
        "knockdown filter still evaluate each cell under its PRIMARY (highest-UMI) target. Membership: obsm['"
        f"{hcfg.membership_obsm_key}'], targets in uns['membership_targets']."
    )
    return warnings


# ===========================================================================
# Basic QC stage: sample-aware expression thresholds and flags (no removal)
# ===========================================================================

#: Independent expression-QC flags written by :func:`flag_expression_qc`.
QC_FLAG_COLUMNS = ["qc_low_counts", "qc_high_counts", "qc_low_genes", "qc_high_genes", "qc_high_mt", "qc_high_hb"]

#: Composite expression-quality verdict. Depends ONLY on the flags above —
#: never on doublet or guide annotations.
GEX_QC_PASS = "gex_qc_pass"


def annotate_gene_classes_case_insensitive(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Like :func:`annotate_gene_classes` but never rewrites ``var_names``.

    Matching is case-insensitive so mouse ``mt-`` and human ``MT-`` both work
    with the default prefix while the stored gene symbols stay untouched.
    """
    q = cfg.qc
    names = pd.Series(expr.var_names.astype(str), index=expr.var_names).str.upper()
    expr.var["mt"] = names.str.startswith(q.mito_prefix.upper()).to_numpy()
    expr.var["ribo"] = names.str.startswith(tuple(p.upper() for p in q.ribo_prefix)).to_numpy()
    expr.var["hb"] = names.str.contains(q.hb_pattern.upper(), regex=True).fillna(False).to_numpy()
    logger.info(
        "Gene classes: %d mitochondrial, %d ribosomal, %d hemoglobin",
        int(expr.var["mt"].sum()),
        int(expr.var["ribo"].sum()),
        int(expr.var["hb"].sum()),
    )
    if expr.var["mt"].sum() == 0:
        logger.warning("No mitochondrial genes matched prefix %r — check qc.mito_prefix", q.mito_prefix)
    return expr


def compute_basic_qc_metrics(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Scanpy QC metrics for one sample without touching ``X``.

    Adds ``total_counts``, ``n_genes_by_counts``, ``pct_counts_mt``,
    ``pct_counts_ribo``, ``pct_counts_hb``, ``pct_counts_in_top_20_genes``
    and their log1p companions to ``obs``; gene-level metrics to ``var``.
    """
    annotate_gene_classes_case_insensitive(expr, cfg)
    sc.pp.calculate_qc_metrics(expr, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True, percent_top=[20])
    return expr


def _mad_bounds(values: np.ndarray, n_mads: float, log_transform: bool) -> Tuple[float, float, float, float]:
    """Return ``(lower, upper, median, mad)`` on the original scale."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 4
    if log_transform:
        x = np.log1p(x)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    lo, hi = med - n_mads * mad, med + n_mads * mad
    if log_transform:
        lo, hi, med_out = np.expm1(lo), np.expm1(hi), np.expm1(med)
    else:
        med_out = med
    return float(lo), float(hi), float(med_out), float(mad)


def resolve_sample_thresholds(
    obs: pd.DataFrame, cfg: Config, sample_id: str, condition: object = None
) -> Dict[str, object]:
    """Resolve expression-QC thresholds for one sample.

    Returns a flat, JSON-friendly dict (``min_genes``, ``max_genes``,
    ``min_counts``, ``max_counts``, ``max_pct_mt``, ``max_pct_hb`` plus the
    medians/MADs and the rule used). ``None`` means "not applied".
    """
    thr = cfg.qc.thresholds
    out: Dict[str, object] = {
        "sample_id": sample_id,
        "method": thr.method,
        "n_mads": thr.n_mads if thr.method == "mad" else None,
        "log_transform": thr.log_transform if thr.method == "mad" else None,
        "n_cells": int(obs.shape[0]),
        "min_genes": thr.min_genes_floor,
        "max_genes": thr.max_genes_ceiling,
        "min_counts": thr.min_counts_floor,
        "max_counts": thr.max_counts_ceiling,
        "max_pct_mt": thr.max_pct_mt,
        "max_pct_hb": thr.max_pct_hb,
    }
    metric_keys = {
        "n_genes_by_counts": ("genes", "min_genes", "max_genes"),
        "total_counts": ("counts", "min_counts", "max_counts"),
    }
    for metric, (short, lo_key, hi_key) in metric_keys.items():
        if metric not in obs.columns:
            continue
        lo, hi, med, mad = _mad_bounds(obs[metric].to_numpy(), thr.n_mads, thr.log_transform)
        out[f"median_{short}"] = med
        out[f"mad_{short}"] = mad
        out[f"mad_lower_{short}"] = lo
        out[f"mad_upper_{short}"] = hi
        if thr.method == "mad" and metric in thr.mad_metrics:
            if thr.flag_low:
                floor = out[lo_key]
                out[lo_key] = float(max(lo, floor)) if floor is not None else float(lo)
            if thr.flag_high:
                ceiling = out[hi_key]
                out[hi_key] = float(min(hi, ceiling)) if ceiling is not None else float(hi)
    # Mitochondrial cap: absolute, optionally condition-specific, optionally MAD-tightened.
    mt_cap = thr.max_pct_mt
    if condition is not None and str(condition) in thr.max_pct_mt_by_condition:
        mt_cap = thr.max_pct_mt_by_condition[str(condition)]
        out["max_pct_mt_source"] = f"condition:{condition}"
    else:
        out["max_pct_mt_source"] = "global"
    if "pct_counts_mt" in obs.columns:
        lo, hi, med, mad = _mad_bounds(obs["pct_counts_mt"].to_numpy(), thr.n_mads, False)
        out["median_pct_mt"] = med
        out["mad_upper_pct_mt"] = hi
        if thr.max_pct_mt_mad and mt_cap is not None:
            mt_cap = float(min(hi, mt_cap))
            out["max_pct_mt_source"] += "+mad"
        elif thr.max_pct_mt_mad:
            mt_cap = float(hi)
            out["max_pct_mt_source"] = "mad"
    out["max_pct_mt"] = mt_cap
    # Explicit per-sample overrides win.
    for key, val in thr.per_sample.get(sample_id, {}).items():
        out[key] = val
        out[f"{key}_source"] = "per_sample_override"
    return out


def flag_expression_qc(expr: ad.AnnData, thresholds: Dict[str, object]) -> ad.AnnData:
    """Write independent QC flags and ``gex_qc_pass`` to ``obs``. No subsetting.

    A flag is False everywhere when its threshold is ``None``.
    """
    n_before = expr.n_obs
    obs = expr.obs
    counts = obs["total_counts"].to_numpy(dtype=float)
    genes = obs["n_genes_by_counts"].to_numpy(dtype=float)

    def _flag(values, thr, op):
        if thr is None or (isinstance(thr, float) and np.isnan(thr)):
            return np.zeros(values.shape[0], dtype=bool)
        return op(values, float(thr))

    obs["qc_low_counts"] = _flag(counts, thresholds.get("min_counts"), np.less)
    obs["qc_high_counts"] = _flag(counts, thresholds.get("max_counts"), np.greater)
    obs["qc_low_genes"] = _flag(genes, thresholds.get("min_genes"), np.less)
    obs["qc_high_genes"] = _flag(genes, thresholds.get("max_genes"), np.greater)
    mt = obs["pct_counts_mt"].to_numpy(dtype=float) if "pct_counts_mt" in obs else np.zeros(n_before)
    obs["qc_high_mt"] = _flag(mt, thresholds.get("max_pct_mt"), np.greater)
    hb = obs["pct_counts_hb"].to_numpy(dtype=float) if "pct_counts_hb" in obs else np.zeros(n_before)
    obs["qc_high_hb"] = _flag(hb, thresholds.get("max_pct_hb"), np.greater)
    any_fail = np.zeros(n_before, dtype=bool)
    for col in QC_FLAG_COLUMNS:
        any_fail |= obs[col].to_numpy(dtype=bool)
    obs[GEX_QC_PASS] = ~any_fail
    assert expr.n_obs == n_before
    logger.info(
        "%s: expression QC flags — low_counts %d, high_counts %d, low_genes %d, "
        "high_genes %d, high_mt %d, high_hb %d -> gex_qc_pass %d/%d (no cells removed)",
        thresholds.get("sample_id", "?"),
        int(obs["qc_low_counts"].sum()),
        int(obs["qc_high_counts"].sum()),
        int(obs["qc_low_genes"].sum()),
        int(obs["qc_high_genes"].sum()),
        int(obs["qc_high_mt"].sum()),
        int(obs["qc_high_hb"].sum()),
        int(obs[GEX_QC_PASS].sum()),
        n_before,
    )
    return expr


def expression_qc_step_table(expr: ad.AnnData, sample_id: str) -> pd.DataFrame:
    """Per-flag summary rows (cells flagged) for one sample — nothing removed."""
    n = expr.n_obs
    rows = []
    for col in QC_FLAG_COLUMNS + [GEX_QC_PASS]:
        k = int(expr.obs[col].sum()) if col in expr.obs else 0
        rows.append(
            {
                "sample_id": sample_id,
                "flag": col,
                "n_cells": n,
                "n_flagged" if col != GEX_QC_PASS else "n_pass": k,
                "fraction": (k / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)
