"""Quality control: standard single-cell metrics plus Perturb-seq guide QC.

Metrics are computed *before* the strict filters run, so the diagnostic figures
show the raw picture and the reader can judge whether the thresholds were
sensible. Every filtering step is recorded in a table that goes into the report.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

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

#: QC metrics carried into the report, in display order.
CELL_QC_METRICS = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"]


def annotate_gene_classes(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Flag mitochondrial, ribosomal and hemoglobin genes in ``var``."""
    q = cfg.qc
    names = expr.var_names.str.upper()
    expr.var["mt"] = names.str.startswith(q.mito_prefix.upper())
    expr.var["ribo"] = names.str.startswith(tuple(p.upper() for p in q.ribo_prefix))
    expr.var["hb"] = names.str.contains(q.hb_pattern, regex=True)
    logger.info(
        "Gene classes: %d mitochondrial, %d ribosomal, %d hemoglobin",
        int(expr.var["mt"].sum()),
        int(expr.var["ribo"].sum()),
        int(expr.var["hb"].sum()),
    )
    if expr.var["mt"].sum() == 0:
        logger.warning(
            "No mitochondrial genes matched prefix %r — check qc.mito_prefix "
            "(human 'MT-', mouse 'mt-'); pct_counts_mt will be all zero.",
            q.mito_prefix,
        )
    return expr


def compute_qc_metrics(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Compute per-cell and per-gene QC metrics on the raw counts."""
    expr = annotate_gene_classes(expr, cfg)
    sc.pp.calculate_qc_metrics(
        expr, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True, percent_top=None
    )
    return expr


def filter_cells_and_genes(expr: ad.AnnData, cfg: Config) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Apply the configured QC filters, recording what each step removed."""
    q = cfg.qc
    steps: List[dict] = []

    def record(step: str, detail: str, before: Tuple[int, int]) -> None:
        steps.append(
            {
                "step": step,
                "threshold": detail,
                "cells_before": before[0],
                "cells_after": expr.n_obs,
                "cells_removed": before[0] - expr.n_obs,
                "genes_before": before[1],
                "genes_after": expr.n_vars,
                "genes_removed": before[1] - expr.n_vars,
            }
        )

    steps.append(
        {
            "step": "input",
            "threshold": "-",
            "cells_before": expr.n_obs,
            "cells_after": expr.n_obs,
            "cells_removed": 0,
            "genes_before": expr.n_vars,
            "genes_after": expr.n_vars,
            "genes_removed": 0,
        }
    )

    if q.min_genes_final and q.min_genes_final > 0:
        before = (expr.n_obs, expr.n_vars)
        sc.pp.filter_cells(expr, min_genes=q.min_genes_final)
        record("min genes per cell", f">= {q.min_genes_final}", before)

    if q.min_counts_per_cell:
        before = (expr.n_obs, expr.n_vars)
        sc.pp.filter_cells(expr, min_counts=q.min_counts_per_cell)
        record("min counts per cell", f">= {q.min_counts_per_cell}", before)

    if q.max_pct_mt is not None:
        before = (expr.n_obs, expr.n_vars)
        expr = expr[expr.obs["pct_counts_mt"] < q.max_pct_mt].copy()
        record("max % mitochondrial", f"< {q.max_pct_mt}%", before)

    if q.max_pct_hb is not None:
        before = (expr.n_obs, expr.n_vars)
        expr = expr[expr.obs["pct_counts_hb"] < q.max_pct_hb].copy()
        record("max % hemoglobin", f"< {q.max_pct_hb}%", before)

    if q.min_cells_per_gene and q.min_cells_per_gene > 0:
        before = (expr.n_obs, expr.n_vars)
        sc.pp.filter_genes(expr, min_cells=q.min_cells_per_gene)
        record("min cells per gene", f">= {q.min_cells_per_gene}", before)

    if expr.n_obs == 0:
        raise ValueError(
            "All cells were removed by QC filters. Loosen qc.min_genes_final / "
            "qc.max_pct_mt, or check that the input is a filtered (not raw) matrix."
        )

    table = pd.DataFrame(steps)
    logger.info(
        "QC filtering: %d -> %d cells, %d -> %d genes",
        table["cells_before"].iloc[0],
        expr.n_obs,
        table["genes_before"].iloc[0],
        expr.n_vars,
    )
    return expr, table


def prefilter(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Drop empty droplets and never-detected genes before computing metrics.

    This is the permissive first pass from the prototype notebooks
    (``min_genes=200``, ``min_cells=3``); the strict thresholds are applied later
    by :func:`filter_cells_and_genes` so the QC figures still show the tail.
    """
    q = cfg.qc
    n0, g0 = expr.n_obs, expr.n_vars
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


def qc_summary_table(expr: ad.AnnData, lane_key: str = LANE_KEY) -> pd.DataFrame:
    """Per-lane summary of the standard QC metrics."""
    obs = expr.obs
    group = obs[lane_key].astype(str) if lane_key in obs.columns else pd.Series(
        ["all"] * expr.n_obs, index=obs.index
    )
    rows = []
    for lane, sub in obs.groupby(group, observed=True):
        row = {"lane": lane, "n_cells": len(sub)}
        for metric, label in [
            ("n_genes_by_counts", "median_genes"),
            ("total_counts", "median_umis"),
            ("pct_counts_mt", "median_pct_mt"),
            ("pct_counts_ribo", "median_pct_ribo"),
        ]:
            if metric in sub.columns:
                row[label] = float(np.median(sub[metric]))
        rows.append(row)
    total = {"lane": "ALL", "n_cells": expr.n_obs}
    for metric, label in [
        ("n_genes_by_counts", "median_genes"),
        ("total_counts", "median_umis"),
        ("pct_counts_mt", "median_pct_mt"),
        ("pct_counts_ribo", "median_pct_ribo"),
    ]:
        if metric in obs.columns:
            total[label] = float(np.median(obs[metric]))
    rows.append(total)
    return pd.DataFrame(rows).round(2)


# ---------------------------------------------------------------------------
# Perturb-seq specific QC
# ---------------------------------------------------------------------------


def guide_qc_summary(expr: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Headline Perturb-seq QC numbers as a two-column table for the report."""
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
            c = int(counts.get(key, 0))
            rows.append((label, f"{c:,} ({100 * c / max(n, 1):.1f}%)"))

    if OBS_TOTAL in obs.columns:
        rows.append(("Median guide UMIs per cell", f"{np.median(obs[OBS_TOTAL]):,.0f}"))
    if OBS_NDETECTED in obs.columns:
        det = obs[OBS_NDETECTED].to_numpy()
        rows.append(
            (
                f"Median guides detected per cell (> {cfg.guides.detection_threshold} UMI)",
                f"{np.median(det):.0f}",
            )
        )
        rows.append(("Cells with exactly 1 guide detected", f"{int((det == 1).sum()):,}"))
        rows.append(("Cells with >1 guide detected (multiplet-like)", f"{int((det > 1).sum()):,}"))
        rows.append(("Estimated MOI (mean guides/cell)", f"{det.mean():.2f}"))
    if OBS_TOP in obs.columns and OBS_SECOND in obs.columns:
        ratio = obs[OBS_TOP].to_numpy() / np.maximum(obs[OBS_SECOND].to_numpy(), 1.0)
        rows.append(("Median top:second guide ratio", f"{np.median(ratio):.1f}"))

    return pd.DataFrame(rows, columns=["metric", "value"])


def check_guide_qc(expr: ad.AnnData, cfg: Config) -> List[str]:
    """Return human-readable warnings about the guide data.

    These are surfaced in the report so a bad run is obvious rather than buried
    in the figures.
    """
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
            f"Consider lowering guides.min_umi (currently {cfg.guides.min_umi}) or "
            f"guides.dominance_ratio (currently {cfg.guides.dominance_ratio})."
        )
    if int(counts.get(CLASS_NTC, 0)) == 0:
        warnings.append(
            "No non-targeting control cells were detected. The 'ntc' control "
            "arm of the perturbation test will be unavailable; check "
            "guides.ntc_patterns against your library naming."
        )
    amb = int(counts.get(CLASS_AMBIGUOUS, 0))
    if amb / max(n, 1) > 0.3:
        warnings.append(
            f"{100 * amb / max(n, 1):.1f}% of cells are ambiguous (no dominant "
            "guide). This usually means a high MOI or guide-index swapping."
        )
    if OBS_NDETECTED in obs.columns:
        multi = float((obs[OBS_NDETECTED].to_numpy() > 1).mean())
        if multi > 0.3:
            warnings.append(
                f"{100 * multi:.1f}% of cells carry more than one detected guide; "
                "single-guide analysis assumptions may not hold."
            )
    return warnings
