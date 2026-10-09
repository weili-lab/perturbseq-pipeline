"""Guide QC for the basic QC stage — attach counts, summarise, flag. Never subset.

Per cell we store guide UMI totals, detected-guide counts (overall and per
scaffold class), top-guide identities per scaffold class and three
independent flags:

``guide_detected``
    at least one guide reaches the detection threshold.
``guide_structure_pass``
    the detected guides match the expected vector structure (default: exactly
    one guide per scaffold class).
``guide_multiplet_flag``
    more guides than allowed in at least one scaffold class — the guide-derived
    multiplet indicator. Flag only; nothing is removed.

``perturbation_assignable`` is always ``False`` here: guide identification is
not perturbation assignment, and the intended vector pairing is a separate
input that this stage does not consume.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import Config
from .guide_counting import GuideCountResult
from .io import BARCODE_KEY, strip_barcode_suffix

logger = logging.getLogger(__name__)

GUIDE_OBSM_KEY_DEFAULT = "guide_counts"
UNKNOWN_SCAFFOLD = "unknown"

#: Boolean guide flags (kept separate from expression flags and doublet calls).
GUIDE_FLAG_COLUMNS = ["guide_detected", "guide_structure_pass", "guide_multiplet_flag", "perturbation_assignable"]


def detection_threshold(cfg: Config) -> int:
    thr = cfg.guides.multiplet.detection_threshold
    return int(thr if thr is not None else cfg.guides.detection_threshold)


def detected_mask(counts: sp.csr_matrix, threshold: int, min_fraction_of_top: Optional[float] = None) -> sp.csr_matrix:
    """Boolean (0/1 int) CSR of guides counted as detected per cell.

    A guide is detected when its UMIs >= ``threshold`` and, if
    ``min_fraction_of_top`` is set, >= that fraction of the cell's top guide.
    """
    counts = sp.csr_matrix(counts)
    row_thr = np.full(counts.shape[0], float(threshold))
    if min_fraction_of_top:
        top = np.asarray(counts.max(axis=1).todense()).ravel().astype(float)
        row_thr = np.maximum(row_thr, min_fraction_of_top * top)
    per_entry = np.repeat(row_thr, np.diff(counts.indptr))
    keep = counts.data >= per_entry
    out = sp.csr_matrix((keep.astype(np.int32), counts.indices.copy(), counts.indptr.copy()), shape=counts.shape)
    out.eliminate_zeros()  # operates on the copies; ``counts`` is left untouched
    return out


def _per_cell_multiplet_rule(n_guides: np.ndarray, cfg: Config, thr) -> tuple:
    """No-scaffold multiplet / structure flags and their description.

    ``guides.multiplet.expected_guides_per_cell: null`` (high-MOI designs, where
    several guides per cell are expected) disables both flags: no cell is a
    multiplet and every cell with >= 1 guide passes the structure check.
    """
    mcfg = cfg.guides.multiplet
    expected = mcfg.expected_guides_per_cell
    if expected is None:
        multiplet = np.zeros(n_guides.shape[0], dtype=bool)
        structure = np.ones(n_guides.shape[0], dtype=bool)
        rule = (
            f"no scaffold classes; guides.multiplet.expected_guides_per_cell is null (high-MOI design): "
            f"no multiplet flag; structure_pass if n_guides >= 1 (>= {thr} UMIs)"
        )
    else:
        multiplet = n_guides > expected
        structure = n_guides == expected
        rule = (
            f"no scaffold classes: multiplet if n_guides > {expected} "
            f"(>= {thr} UMIs); structure_pass if n_guides == {expected}"
        )
    return multiplet, structure, rule


def _structure_flags(detected: sp.csr_matrix, design: pd.DataFrame, cfg: Config):
    """Return (n_guides, n_by_class, n_unknown, multiplet, structure) for a detection mask."""
    mcfg = cfg.guides.multiplet
    classes = scaffold_classes(design)
    scaf = design["scaffold"].astype(str).to_numpy() if "scaffold" in design else np.full(len(design), UNKNOWN_SCAFFOLD)
    n_guides = np.asarray(detected.sum(axis=1)).ravel().astype(np.int64)
    n_by_class = {
        c: np.asarray(detected[:, np.where(scaf == c)[0]].sum(axis=1)).ravel().astype(np.int64) for c in classes
    }
    unknown_cols = np.where(scaf == UNKNOWN_SCAFFOLD)[0]
    n_unknown = (
        np.asarray(detected[:, unknown_cols].sum(axis=1)).ravel().astype(np.int64)
        if len(unknown_cols)
        else np.zeros(detected.shape[0], dtype=np.int64)
    )
    if classes:
        multiplet = np.zeros(detected.shape[0], dtype=bool)
        structure = np.ones(detected.shape[0], dtype=bool)
        for c in classes:
            multiplet |= n_by_class[c] > mcfg.max_guides_per_scaffold
            structure &= n_by_class[c] == mcfg.expected_guides_per_scaffold
        multiplet |= n_unknown > mcfg.max_guides_per_scaffold
        structure &= n_unknown == 0
    else:
        multiplet, structure, _ = _per_cell_multiplet_rule(n_guides, cfg, cfg.guides.multiplet.detection_threshold)
    structure &= n_guides >= 1
    return n_guides, n_by_class, n_unknown, multiplet, structure


def guide_detection_sensitivity(
    counts: sp.csr_matrix, design: pd.DataFrame, cfg: Config, sample_id: str
) -> pd.DataFrame:
    """Flag fractions over a grid of detection rules (assessment only, nothing stored in obs)."""
    mcfg = cfg.guides.multiplet
    rows = []
    n = counts.shape[0]
    for thr in mcfg.sensitivity_thresholds:
        for frac in mcfg.sensitivity_fractions:
            det = detected_mask(counts, int(thr), float(frac) or None)
            n_guides, _, _, multiplet, structure = _structure_flags(det, design, cfg)
            rows.append(
                {
                    "sample_id": sample_id,
                    "detection_threshold_umi": int(thr),
                    "min_fraction_of_top": float(frac),
                    "n_cells": n,
                    "frac_guide_detected": float((n_guides >= 1).mean()) if n else float("nan"),
                    "frac_guide_structure_pass": float(structure.mean()) if n else float("nan"),
                    "frac_guide_multiplet_flag": float(multiplet.mean()) if n else float("nan"),
                    "median_n_guides": float(np.median(n_guides)) if n else float("nan"),
                    "is_current_rule": int(thr) == detection_threshold(cfg)
                    and (float(frac) or None) == mcfg.detection_min_fraction_of_top,
                }
            )
    return pd.DataFrame(rows)


def scaffold_classes(design: pd.DataFrame) -> List[str]:
    """Known scaffold classes present in the design (excluding ``unknown``)."""
    if "scaffold" not in design.columns:
        return []
    vals = [str(v) for v in design["scaffold"].astype(str).unique() if str(v) != UNKNOWN_SCAFFOLD]
    return sorted(vals)


def infer_scaffold_classes(design: pd.DataFrame, results: Dict[str, GuideCountResult], cfg: Config) -> pd.DataFrame:
    """Assign an empirical scaffold class to design guides lacking one.

    Pools matched reads per (guide, scaffold anchor) across all counted
    samples; a guide gets the majority class when it has at least
    ``scaffold_min_reads`` reads and purity >= ``scaffold_purity_min``.
    """
    design = design.copy()
    if "scaffold" not in design.columns:
        design["scaffold"] = UNKNOWN_SCAFFOLD
        design["scaffold_source"] = "unspecified"
    fastq_results = [r for r in results.values() if r.source == "fastq"]
    if not fastq_results:
        return design
    names = fastq_results[0].scaffold_names
    pooled = np.zeros((len(design), len(names)), dtype=np.int64)
    for r in fastq_results:
        if r.guide_scaffold_reads.shape != pooled.shape:
            raise ValueError("scaffold read matrices differ in shape between samples")
        pooled += r.guide_scaffold_reads
    total = pooled.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(total > 0, pooled.max(axis=1) / np.maximum(total, 1), np.nan)
    majority = np.array(names, dtype=object)[pooled.argmax(axis=1)]
    design["scaffold_reads_total"] = total
    design["scaffold_purity"] = purity
    for j, name in enumerate(names):
        design[f"scaffold_reads_{name}"] = pooled[:, j]
    mcfg = cfg.guides.multiplet
    unknown = design["scaffold"].astype(str) == UNKNOWN_SCAFFOLD
    eligible = unknown & (total >= mcfg.scaffold_min_reads) & (purity >= mcfg.scaffold_purity_min)
    design.loc[eligible, "scaffold"] = majority[eligible.to_numpy()]
    design.loc[eligible, "scaffold_source"] = "empirical"
    low_purity = unknown & (total >= mcfg.scaffold_min_reads) & (purity < mcfg.scaffold_purity_min)
    if low_purity.any():
        logger.warning(
            "%d guide(s) have mixed scaffold reads (purity < %.2f) and stay 'unknown': %s",
            int(low_purity.sum()),
            mcfg.scaffold_purity_min,
            design.loc[low_purity, "guide_id"].head(5).tolist(),
        )
    logger.info(
        "Scaffold classes: %s; %d guides assigned empirically, %d remain unknown (%d with no reads)",
        dict(design["scaffold"].value_counts()),
        int(eligible.sum()),
        int((design["scaffold"].astype(str) == UNKNOWN_SCAFFOLD).sum()),
        int((total == 0).sum()),
    )
    return design


def _top_two(X: sp.csr_matrix):
    """Per-row top value, its column, and second-highest value of a CSR matrix."""
    n = X.shape[0]
    top = np.zeros(n, dtype=np.int64)
    top_idx = np.full(n, -1, dtype=np.int64)
    second = np.zeros(n, dtype=np.int64)
    X = X.tocsr()
    for i in range(n):
        s, e = X.indptr[i], X.indptr[i + 1]
        if e == s:
            continue
        data = X.data[s:e]
        idx = X.indices[s:e]
        order = np.argsort(-data, kind="stable")
        top[i] = data[order[0]]
        top_idx[i] = idx[order[0]]
        if len(order) > 1:
            second[i] = data[order[1]]
    return top, top_idx, second


def attach_guide_counts(expr: ad.AnnData, result: GuideCountResult, design: pd.DataFrame, cfg: Config) -> ad.AnnData:
    """Attach guide UMI counts to ``expr`` and annotate ``obs``. No subsetting."""
    n_before = expr.n_obs
    bare = strip_barcode_suffix(expr.obs[BARCODE_KEY].astype(str), cfg.guides.fastq.barcode_suffix_regex)
    if list(bare) != list(result.cell_barcodes):
        pos = {b: i for i, b in enumerate(result.cell_barcodes)}
        rows = np.array([pos.get(b, -1) for b in bare])
        if (rows < 0).any():
            raise ValueError(f"{result.sample_id}: {(rows < 0).sum()} GEX barcodes missing from the guide count rows")
        counts = result.counts[rows]
    else:
        counts = result.counts
    counts = sp.csr_matrix(counts, dtype=np.int32)
    if counts.shape != (expr.n_obs, len(design)):
        raise ValueError(f"{result.sample_id}: guide matrix {counts.shape} does not match cells x designed guides")
    key = cfg.output.guide_obsm_key or GUIDE_OBSM_KEY_DEFAULT
    expr.obsm[key] = counts
    expr.uns["guide_names"] = list(design["guide_id"].astype(str))
    expr.uns["guide_target_genes"] = list(design["target"].astype(str))
    expr.uns["guide_source"] = result.source
    thr = detection_threshold(cfg)
    min_frac = cfg.guides.multiplet.detection_min_fraction_of_top
    detected = detected_mask(counts, thr, min_frac)
    obs = expr.obs
    obs["guide_umi_total"] = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    obs["n_guides"] = np.asarray(detected.sum(axis=1)).ravel().astype(np.int64)
    top, top_idx, second = _top_two(counts)
    guide_ids = np.array(design["guide_id"].astype(str), dtype=object)
    obs["top_guide"] = np.where(top_idx >= 0, guide_ids[np.clip(top_idx, 0, None)], "none")
    obs["top_guide_umi"] = top
    obs["second_guide_umi"] = second
    classes = scaffold_classes(design)
    scaf = design["scaffold"].astype(str).to_numpy() if "scaffold" in design else np.full(len(design), UNKNOWN_SCAFFOLD)
    class_cols: Dict[str, np.ndarray] = {c: np.where(scaf == c)[0] for c in classes}
    unknown_cols = np.where(scaf == UNKNOWN_SCAFFOLD)[0]
    n_by_class: Dict[str, np.ndarray] = {}
    for c, cols in class_cols.items():
        sub = counts[:, cols]
        subd = detected[:, cols]
        n_by_class[c] = np.asarray(subd.sum(axis=1)).ravel().astype(np.int64)
        obs[f"n_guides_{c}"] = n_by_class[c]
        obs[f"guide_umi_{c}"] = np.asarray(sub.sum(axis=1)).ravel().astype(np.int64)
        t, ti, s2 = _top_two(sub)
        ids_c = guide_ids[cols]
        obs[f"top_guide_{c}"] = np.where(ti >= 0, ids_c[np.clip(ti, 0, None)], "none")
        obs[f"top_guide_{c}_umi"] = t
        obs[f"second_guide_{c}_umi"] = s2
    n_unknown = (
        np.asarray(detected[:, unknown_cols].sum(axis=1)).ravel().astype(np.int64)
        if len(unknown_cols)
        else np.zeros(expr.n_obs, dtype=np.int64)
    )
    if len(unknown_cols) and classes:
        obs["n_guides_unknown_scaffold"] = n_unknown
    mcfg = cfg.guides.multiplet
    n_guides = obs["n_guides"].to_numpy()
    obs["guide_detected"] = n_guides >= 1
    if classes:
        multiplet = np.zeros(expr.n_obs, dtype=bool)
        structure = np.ones(expr.n_obs, dtype=bool)
        for c in classes:
            multiplet |= n_by_class[c] > mcfg.max_guides_per_scaffold
            structure &= n_by_class[c] == mcfg.expected_guides_per_scaffold
        multiplet |= n_unknown > mcfg.max_guides_per_scaffold
        structure &= n_unknown == 0
        rule = (
            f"scaffold classes {classes}: multiplet if any class has > {mcfg.max_guides_per_scaffold} "
            f"detected guides (>= {thr} UMIs{' and >= ' + str(min_frac) + ' x top guide' if min_frac else ''}); "
            f"structure_pass if every class has exactly {mcfg.expected_guides_per_scaffold}"
        )
    else:
        multiplet, structure, rule = _per_cell_multiplet_rule(n_guides, cfg, thr)
    obs["guide_multiplet_flag"] = multiplet
    obs["guide_structure_pass"] = structure & obs["guide_detected"].to_numpy()
    obs["perturbation_assignable"] = False
    expr.uns["guide_qc"] = {
        "detection_threshold_umi": thr,
        "detection_min_fraction_of_top": float(min_frac) if min_frac else "None",
        "scaffold_classes": list(classes),
        "rule": rule,
        "perturbation_assignment": "not performed in basic QC (vector pairing not supplied)",
    }
    assert expr.n_obs == n_before, "guide QC must never change the number of cells"
    logger.info(
        "%s: guide QC — detected %d/%d cells, structure_pass %d, multiplet_flag %d (flag only; %d cells retained)",
        result.sample_id,
        int(obs["guide_detected"].sum()),
        n_before,
        int(obs["guide_structure_pass"].sum()),
        int(obs["guide_multiplet_flag"].sum()),
        expr.n_obs,
    )
    return expr


def empty_guide_annotations(expr: ad.AnnData, cfg: Config, classes: Sequence[str] = ()) -> ad.AnnData:
    """Add neutral guide columns when a sample has no guide data (GEX only)."""
    n = expr.n_obs
    obs = expr.obs
    obs["guide_umi_total"] = np.zeros(n, dtype=np.int64)
    obs["n_guides"] = np.zeros(n, dtype=np.int64)
    obs["top_guide"] = "none"
    obs["top_guide_umi"] = np.zeros(n, dtype=np.int64)
    obs["second_guide_umi"] = np.zeros(n, dtype=np.int64)
    for c in classes:
        obs[f"n_guides_{c}"] = np.zeros(n, dtype=np.int64)
        obs[f"guide_umi_{c}"] = np.zeros(n, dtype=np.int64)
        obs[f"top_guide_{c}"] = "none"
        obs[f"top_guide_{c}_umi"] = np.zeros(n, dtype=np.int64)
        obs[f"second_guide_{c}_umi"] = np.zeros(n, dtype=np.int64)
    obs["guide_detected"] = False
    obs["guide_structure_pass"] = False
    obs["guide_multiplet_flag"] = False
    obs["perturbation_assignable"] = False
    expr.uns["guide_source"] = "none"
    return expr


def guide_sample_summary(
    expr: ad.AnnData, sample_id: str, result: Optional[GuideCountResult], design: Optional[pd.DataFrame], cfg: Config
) -> Dict[str, object]:
    """One row of the guide QC summary table for a sample."""
    obs = expr.obs
    row: Dict[str, object] = {
        "sample_id": sample_id,
        "guide_source": expr.uns.get("guide_source", "none"),
        "n_cells": int(expr.n_obs),
        "detection_threshold_umi": detection_threshold(cfg),
    }
    if result is not None:
        st = result.stats
        for k in (
            "reads_total",
            "reads_with_tso",
            "reads_with_scaffold_anchor",
            "reads_spacer_matched",
            "reads_spacer_matched_via_shift",
            "reads_spacer_unmatched",
            "reads_matched_barcode_in_gex",
            "reads_matched_barcode_not_in_gex",
            "frac_reads_spacer_matched",
            "frac_matched_reads_in_gex_barcodes",
            "unique_cell_guide_umis",
            "total_guide_umis_in_matrix",
            "guides_designed",
            "guides_detected_any_umi",
        ):
            if k in st:
                row[k] = st[k]
        umis = result.guide_umis()
        row["guides_with_any_cell_umi"] = int((umis > 0).sum())
        row["guides_with_positive_cells_at_threshold"] = int(
            (result.guide_positive_cells(detection_threshold(cfg)) > 0).sum()
        )
    if "guide_detected" in obs:
        row["cells_guide_detected"] = int(obs["guide_detected"].sum())
        row["cells_guide_structure_pass"] = int(obs["guide_structure_pass"].sum())
        row["cells_guide_multiplet_flag"] = int(obs["guide_multiplet_flag"].sum())
        row["median_guide_umi_total"] = float(np.median(obs["guide_umi_total"])) if expr.n_obs else float("nan")
        row["median_n_guides"] = float(np.median(obs["n_guides"])) if expr.n_obs else float("nan")
        vc = obs["n_guides"].value_counts()
        for k in range(0, 5):
            row[f"cells_n_guides_{k}"] = int(vc.get(k, 0))
        row["cells_n_guides_ge5"] = int(vc[vc.index >= 5].sum())
        for c in scaffold_classes(design) if design is not None else []:
            col = f"n_guides_{c}"
            if col in obs:
                vcc = obs[col].value_counts()
                row[f"cells_{col}_0"] = int(vcc.get(0, 0))
                row[f"cells_{col}_1"] = int(vcc.get(1, 0))
                row[f"cells_{col}_ge2"] = int(vcc[vcc.index >= 2].sum())
    return row


def scrublet_vs_guide_multiplet_table(obs: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    """Cross-tabulate Scrublet calls against guide multiplet flags (assessment only)."""
    if "predicted_doublet" not in obs or "guide_multiplet_flag" not in obs:
        return pd.DataFrame()
    frames = []
    groups = [(None, obs)] if by is None else list(obs.groupby(by, observed=True))
    for name, sub in groups:
        pred = sub["predicted_doublet"].astype(bool)
        gm = sub["guide_multiplet_flag"].astype(bool)
        det = sub["guide_detected"].astype(bool) if "guide_detected" in sub else pd.Series(True, index=sub.index)
        rows = {
            "group": "ALL" if name is None else str(name),
            "n_cells": int(len(sub)),
            "scrublet_singlet_guide_clean": int((~pred & ~gm & det).sum()),
            "scrublet_singlet_guide_multiplet": int((~pred & gm).sum()),
            "scrublet_doublet_guide_clean": int((pred & ~gm & det).sum()),
            "scrublet_doublet_guide_multiplet": int((pred & gm).sum()),
            "scrublet_singlet_no_guide": int((~pred & ~det).sum()),
            "scrublet_doublet_no_guide": int((pred & ~det).sum()),
            "frac_scrublet_doublet": float(pred.mean()) if len(sub) else float("nan"),
            "frac_guide_multiplet": float(gm.mean()) if len(sub) else float("nan"),
        }
        both = int((pred & gm).sum())
        rows["frac_guide_multiplets_called_by_scrublet"] = both / max(int(gm.sum()), 1)
        rows["frac_scrublet_doublets_with_guide_multiplet"] = both / max(int(pred.sum()), 1)
        if "doublet_score" in sub:
            sc_ = pd.to_numeric(sub["doublet_score"], errors="coerce")
            rows["median_doublet_score_guide_clean"] = float(sc_[~gm].median())
            rows["median_doublet_score_guide_multiplet"] = float(sc_[gm].median()) if gm.any() else float("nan")
        frames.append(rows)
    return pd.DataFrame(frames)
