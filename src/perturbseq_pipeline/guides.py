"""Guide calling: assign each cell to a guide, then to a target gene.

The prototype notebooks looped over every cell in Python to find the top two
guide counts. That is replaced here by a chunked, vectorized top-2 search, and
the assignment rule is stated once with named thresholds:

* no guide counts at all              -> ``unassigned``
* top >= ``min_umi`` and top > ``dominance_ratio`` x second  -> assigned
* anything else                       -> ``ambiguous``

Both failure categories are kept in the object and reported; they are never
silently dropped.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config, GuideConfig
from .io import RAW_GUIDE_LABEL

logger = logging.getLogger(__name__)

#: ``obs`` columns written by :func:`assign_guides`.
OBS_TARGET = "target_gene"
OBS_GUIDE = "guide_id"
OBS_CLASS = "perturbation_class"
OBS_TOP = "top_guide_count"
OBS_SECOND = "second_guide_count"
OBS_TOTAL = "total_guide_counts"
OBS_NDETECTED = "n_guides_detected"

#: Values of :data:`OBS_CLASS`.
CLASS_TARGETING = "targeting"
CLASS_NTC = "non-targeting"
CLASS_AMBIGUOUS = "ambiguous"
CLASS_UNASSIGNED = "unassigned"


# ---------------------------------------------------------------------------
# Target-gene parsing
# ---------------------------------------------------------------------------


def parse_target_genes(guide_ids: Sequence[str], gcfg: GuideConfig) -> np.ndarray:
    """Map guide identifiers to target gene symbols.

    With the default settings the identifier is split on ``_``, ``-`` and ``.``
    and the first field is taken, which handles both library conventions seen so
    far: ``AFF4_P1P2_1`` (10x features file) and ``AFF4-P1P2.2`` (Seurat
    ``genotype`` column) both give ``AFF4``. Set ``guides.target_regex`` when a
    library uses target names that themselves contain a delimiter.
    """
    ids = [str(g) for g in guide_ids]
    if gcfg.target_regex:
        pattern = re.compile(gcfg.target_regex)
        out = []
        for gid in ids:
            m = pattern.match(gid)
            if m is None or not m.groups():
                logger.warning(
                    "guides.target_regex did not match guide %r; using it verbatim",
                    gid,
                )
                out.append(gid)
            else:
                out.append(m.group(1))
        return np.asarray(out, dtype=object)

    if not gcfg.target_split_delims:
        return np.asarray(ids, dtype=object)
    splitter = re.compile("[" + re.escape("".join(gcfg.target_split_delims)) + "]")
    return np.asarray([splitter.split(g)[0] for g in ids], dtype=object)


def is_non_targeting(labels: Sequence[str], gcfg: GuideConfig) -> np.ndarray:
    """Boolean mask of labels matching any non-targeting control pattern."""
    if not gcfg.ntc_patterns:
        return np.zeros(len(labels), dtype=bool)
    patterns = [re.compile(p, re.IGNORECASE) for p in gcfg.ntc_patterns]
    return np.asarray(
        [any(p.search(str(x)) for p in patterns) for x in labels], dtype=bool
    )


# ---------------------------------------------------------------------------
# Top-2 guide search
# ---------------------------------------------------------------------------


def top_two_guides(
    X, chunk_size: int = 20_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(top_index, top_value, second_value)`` per cell.

    Works on sparse or dense matrices and densifies at most ``chunk_size`` rows
    at a time, so memory stays bounded for large guide libraries.
    """
    n_obs, n_vars = X.shape
    top_idx = np.zeros(n_obs, dtype=np.int64)
    top_val = np.zeros(n_obs, dtype=np.float64)
    second_val = np.zeros(n_obs, dtype=np.float64)

    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        block = X[start:stop]
        dense = np.asarray(block.todense()) if sparse.issparse(block) else np.asarray(block)
        dense = dense.astype(np.float64, copy=False)

        if n_vars == 1:
            top_idx[start:stop] = 0
            top_val[start:stop] = dense[:, 0]
            second_val[start:stop] = 0.0
            continue

        # argpartition puts the two largest values in the last two slots.
        part = np.argpartition(dense, -2, axis=1)[:, -2:]
        rows = np.arange(dense.shape[0])[:, None]
        vals = dense[rows, part]
        order = np.argsort(vals, axis=1)  # ascending: [second, top]
        top_idx[start:stop] = part[rows[:, 0], order[:, 1]]
        top_val[start:stop] = vals[rows[:, 0], order[:, 1]]
        second_val[start:stop] = vals[rows[:, 0], order[:, 0]]

    return top_idx, top_val, second_val


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def assign_guides(
    expr: ad.AnnData, guides: Optional[ad.AnnData], cfg: Config
) -> ad.AnnData:
    """Write guide/target assignments into ``expr.obs``.

    When ``guides`` is ``None`` the input only carried a per-cell label
    (``obs['guide_id_raw']``); the label is then parsed directly and the
    count-based diagnostics are unavailable.
    """
    if guides is None:
        return _assign_from_labels(expr, cfg)
    return _assign_from_matrix(expr, guides, cfg)


def _assign_from_matrix(expr: ad.AnnData, guides: ad.AnnData, cfg: Config) -> ad.AnnData:
    gcfg = cfg.guides
    guides = guides[expr.obs_names].copy()

    X = guides.layers["counts"] if "counts" in guides.layers else guides.X
    top_idx, top_val, second_val = top_two_guides(X)

    total = np.asarray(X.sum(axis=1)).ravel()
    detected = np.asarray((X > gcfg.detection_threshold).sum(axis=1)).ravel()

    guide_ids = guides.var_names.to_numpy().astype(str)
    guide_targets = parse_target_genes(guide_ids, gcfg)

    assigned = (top_val >= max(gcfg.min_umi, 1)) & (
        top_val > gcfg.dominance_ratio * second_val
    )
    has_counts = top_val > 0

    guide_call = np.full(expr.n_obs, gcfg.unassigned_label, dtype=object)
    target_call = np.full(expr.n_obs, gcfg.unassigned_label, dtype=object)
    guide_call[has_counts & ~assigned] = gcfg.ambiguous_label
    target_call[has_counts & ~assigned] = gcfg.ambiguous_label
    guide_call[assigned] = guide_ids[top_idx[assigned]]
    target_call[assigned] = guide_targets[top_idx[assigned]]

    # Store guide-level diagnostics before collapsing NTCs into one label.
    expr.obs[OBS_TOP] = top_val
    expr.obs[OBS_SECOND] = second_val
    expr.obs[OBS_TOTAL] = total
    expr.obs[OBS_NDETECTED] = detected
    expr.obs[OBS_GUIDE] = pd.Categorical(guide_call.astype(str))

    _finalize_labels(expr, target_call, gcfg)

    guides.var["target_gene"] = guide_targets
    guides.var["is_non_targeting"] = is_non_targeting(guide_targets, gcfg)
    guides.obs[OBS_GUIDE] = expr.obs[OBS_GUIDE].to_numpy()
    guides.obs[OBS_TARGET] = expr.obs[OBS_TARGET].to_numpy()

    _log_assignment(expr, cfg)
    return expr


def _assign_from_labels(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Assignment path for inputs that already carry a per-cell guide label."""
    gcfg = cfg.guides
    if RAW_GUIDE_LABEL not in expr.obs.columns:
        raise ValueError(
            f"No guide matrix and no obs['{RAW_GUIDE_LABEL}'] column; "
            "cannot determine perturbations."
        )
    raw = expr.obs[RAW_GUIDE_LABEL].astype(str).to_numpy()
    target_call = parse_target_genes(raw, gcfg).astype(object)

    # Honour labels that already encode the two failure modes.
    for label in (gcfg.unassigned_label, gcfg.ambiguous_label, "NA", "nan", "None", ""):
        target_call[np.char.lower(raw.astype(str)) == label.lower()] = (
            gcfg.ambiguous_label if label == gcfg.ambiguous_label else gcfg.unassigned_label
        )

    expr.obs[OBS_GUIDE] = pd.Categorical(raw)
    _finalize_labels(expr, target_call, gcfg)
    _log_assignment(expr, cfg)
    return expr


def _finalize_labels(expr: ad.AnnData, target_call: np.ndarray, gcfg: GuideConfig) -> None:
    """Collapse non-targeting guides to one label and set the class column."""
    ntc_mask = is_non_targeting(target_call, gcfg)
    special = {gcfg.unassigned_label, gcfg.ambiguous_label}
    ntc_mask &= ~np.isin(target_call.astype(str), list(special))
    target_call = target_call.copy()
    target_call[ntc_mask] = gcfg.ntc_label

    klass = np.full(len(target_call), CLASS_TARGETING, dtype=object)
    klass[ntc_mask] = CLASS_NTC
    klass[target_call == gcfg.ambiguous_label] = CLASS_AMBIGUOUS
    klass[target_call == gcfg.unassigned_label] = CLASS_UNASSIGNED

    expr.obs[OBS_TARGET] = pd.Categorical(target_call.astype(str))
    expr.obs[OBS_CLASS] = pd.Categorical(
        klass.astype(str),
        categories=[CLASS_TARGETING, CLASS_NTC, CLASS_AMBIGUOUS, CLASS_UNASSIGNED],
    )


def _log_assignment(expr: ad.AnnData, cfg: Config) -> None:
    counts = expr.obs[OBS_CLASS].value_counts()
    n = expr.n_obs
    logger.info(
        "Guide assignment: %d targeting (%.1f%%), %d non-targeting, "
        "%d ambiguous, %d unassigned",
        counts.get(CLASS_TARGETING, 0),
        100 * counts.get(CLASS_TARGETING, 0) / max(n, 1),
        counts.get(CLASS_NTC, 0),
        counts.get(CLASS_AMBIGUOUS, 0),
        counts.get(CLASS_UNASSIGNED, 0),
    )
    n_targets = target_genes(expr, cfg).size
    logger.info("%d distinct target genes assigned", n_targets)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def target_genes(expr: ad.AnnData, cfg: Config) -> np.ndarray:
    """Sorted array of real target genes (excludes NTC/ambiguous/unassigned)."""
    obs = expr.obs
    mask = obs[OBS_CLASS] == CLASS_TARGETING
    return np.array(sorted(obs.loc[mask, OBS_TARGET].astype(str).unique()))


def assignment_summary(expr: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Per-target cell counts, flagged for downstream testability."""
    obs = expr.obs
    tab = (
        obs[OBS_TARGET]
        .astype(str)
        .value_counts()
        .rename_axis(OBS_TARGET)
        .reset_index(name="n_cells")
    )
    klass = (
        obs.groupby(obs[OBS_TARGET].astype(str), observed=True)[OBS_CLASS]
        .agg(lambda s: s.iloc[0])
        .astype(str)
    )
    tab["class"] = tab[OBS_TARGET].map(klass)
    tab["detected_in_expression"] = tab[OBS_TARGET].isin(set(expr.var_names))
    tab["testable"] = (
        (tab["class"] == CLASS_TARGETING)
        & tab["detected_in_expression"]
        & (tab["n_cells"] >= cfg.perturbation.min_cells_per_target)
    )
    return tab.sort_values("n_cells", ascending=False).reset_index(drop=True)


def per_lane_assignment(expr: ad.AnnData, lane_key: str = "lane_id") -> pd.DataFrame:
    """Assignment-class breakdown per lane — a common batch-failure readout."""
    if lane_key not in expr.obs.columns:
        return pd.DataFrame()
    tab = (
        expr.obs.groupby([lane_key, OBS_CLASS], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    tab["n_cells"] = tab.sum(axis=1)
    for c in (CLASS_TARGETING, CLASS_NTC, CLASS_AMBIGUOUS, CLASS_UNASSIGNED):
        if c in tab.columns:
            tab[f"pct_{c}"] = 100 * tab[c] / tab["n_cells"]
    return tab.reset_index()


def guide_representation(guides: Optional[ad.AnnData], expr: ad.AnnData) -> pd.DataFrame:
    """Cells per guide — flags library skew and dropped guides."""
    if OBS_GUIDE not in expr.obs.columns:
        return pd.DataFrame()
    counts = expr.obs[OBS_GUIDE].astype(str).value_counts()
    df = counts.rename_axis("guide_id").reset_index(name="n_cells")
    if guides is not None and "target_gene" in guides.var.columns:
        mapping = guides.var["target_gene"].astype(str).to_dict()
        df["target_gene"] = df["guide_id"].map(mapping)
        missing = sorted(set(guides.var_names.astype(str)) - set(df["guide_id"]))
        if missing:
            extra = pd.DataFrame(
                {
                    "guide_id": missing,
                    "n_cells": 0,
                    "target_gene": [mapping.get(g) for g in missing],
                }
            )
            df = pd.concat([df, extra], ignore_index=True)
    return df.sort_values("n_cells", ascending=False).reset_index(drop=True)
