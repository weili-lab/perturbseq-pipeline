"""Guide calling: assign each cell to a guide, then to a target gene.

The prototype notebooks looped over every cell in Python to find the top two
guide counts. The standard implementation replaces that with a chunked,
vectorized top-2 search.

Assignment rule
---------------
* no guide counts at all
      -> ``unassigned``

* top >= ``min_umi``, top > ``dominance_ratio`` x second, and optionally
  second <= ``max_second_umi``
      -> assigned

* anything else
      -> ``ambiguous``

Both failure categories are retained and reported.

Adaptive large-data execution
-----------------------------
For ordinary datasets, the original behaviour is preserved.

For million-cell datasets such as KOLF, several operations become unnecessarily
expensive if performed once per cell:

* parsing guide/target strings;
* regular-expression NTC detection;
* converting sparse guide matrices into dense 20k-row blocks;
* copying an already aligned guide matrix.

Large-data mode therefore changes implementation, not biological semantics:

LABEL INPUT
    Unique guide labels are parsed exactly once and then mapped back to cells.
    For example, 2.6 million cells carrying ~12,000 distinct labels require
    ~12,000 parsing operations rather than 2.6 million.

GUIDE-MATRIX INPUT
    Sparse CSR guide matrices use a sparse row-wise top-two implementation,
    optionally accelerated with Numba. No dense cells x guides block is created.

The assignment thresholds and output columns are identical in standard and
large modes.
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


# Output columns

OBS_TARGET = "target_gene"
#: Per-cell label in the guide object's obs for cells absent from the analysed
#: expression object (e.g. removed by QC before assignment).
NOT_EVALUATED_LABEL = "not_evaluated"
OBS_GUIDE = "guide_id"
OBS_CLASS = "perturbation_class"

OBS_TOP = "top_guide_count"
OBS_SECOND = "second_guide_count"
OBS_TOTAL = "total_guide_counts"
OBS_NDETECTED = "n_guides_detected"


# Perturbation classes

CLASS_TARGETING = "targeting"
CLASS_NTC = "non-targeting"
CLASS_AMBIGUOUS = "ambiguous"
CLASS_UNASSIGNED = "unassigned"


# Target-gene parsing & resolution


def resolve_guide_targets(guides: ad.AnnData, gcfg: GuideConfig) -> np.ndarray:
    """Resolve guide features to biological target names.

    When ``gcfg.target_feature_column`` is configured, that column of
    ``guides.var`` is treated as the authoritative source of target gene names.
    Values matching ``gcfg.ignored_target_values`` (case-insensitively) or
    empty/missing entries are mapped to ``gcfg.unassigned_label``.

    When ``gcfg.target_feature_column`` is None, falls back to legacy
    guide-ID parsing via :func:`parse_target_genes`.
    """
    guide_ids = guides.var_names.to_numpy().astype(str)
    if gcfg.target_feature_column is None:
        logger.info("Guide target mapping: parsing target names from guide IDs")
        return parse_target_genes(guide_ids, gcfg)
    col = gcfg.target_feature_column
    if col not in guides.var.columns:
        raise ValueError(
            f"Configured guide target column {col!r} not found in guides.var. "
            f"Available var columns: {list(guides.var.columns)}"
        )
    raw_series = guides.var[col]
    n_features = guides.n_vars
    out = np.empty(n_features, dtype=object)
    ignored_set = {str(v).strip().lower() for v in (gcfg.ignored_target_values or []) if str(v).strip()}
    n_ignored = 0
    n_missing = 0
    for idx, val in enumerate(raw_series):
        if pd.isna(val) or val is None:
            out[idx] = gcfg.unassigned_label
            n_missing += 1
            continue
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            out[idx] = gcfg.unassigned_label
            n_missing += 1
        elif s.lower() in ignored_set:
            out[idx] = gcfg.unassigned_label
            n_ignored += 1
        else:
            out[idx] = s
    logger.info(
        "Guide target mapping: using var[%r] for %d guide features (%d ignored, %d missing)",
        col,
        n_features,
        n_ignored,
        n_missing,
    )
    return out


def parse_target_genes(guide_ids: Sequence[str], gcfg: GuideConfig) -> np.ndarray:
    """Map guide identifiers to target gene symbols.

    With default delimiter parsing:

        AFF4_P1P2_1
        -> AFF4

        AFF4-P1P2.2
        -> AFF4

    Set ``guides.target_regex`` when target names themselves contain the
    configured delimiters.
    """
    ids = [str(guide) for guide in guide_ids]
    if gcfg.target_regex:
        pattern = re.compile(gcfg.target_regex)
        out = []
        for guide_id in ids:
            match = pattern.match(guide_id)
            if match is None or not match.groups():
                logger.warning("guides.target_regex did not match guide %r; using it verbatim", guide_id)
                out.append(guide_id)
            else:
                out.append(match.group(1))
        return np.asarray(out, dtype=object)
    if not gcfg.target_split_delims:
        return np.asarray(ids, dtype=object)
    splitter = re.compile("[" + re.escape("".join(gcfg.target_split_delims)) + "]")
    return np.asarray([splitter.split(guide)[0] for guide in ids], dtype=object)


def _parse_target_genes_unique(labels: Sequence[str], gcfg: GuideConfig) -> np.ndarray:
    """Parse millions of labels by operating only on unique values.

    The result remains one target label per cell and is therefore semantically
    identical to calling :func:`parse_target_genes` on every row.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.asarray([], dtype=object)
    # np.unique also returns the inverse mapping from each cell to its unique
    # label, so parsing is performed once per distinct guide rather than once
    # per cell.
    unique_labels, inverse = np.unique(labels.astype(str), return_inverse=True)
    parsed_unique = parse_target_genes(unique_labels, gcfg)
    return parsed_unique[inverse]


# NTC detection


def is_non_targeting(labels: Sequence[str], gcfg: GuideConfig) -> np.ndarray:
    """Boolean mask of labels matching any non-targeting pattern."""
    if not gcfg.ntc_patterns:
        return np.zeros(len(labels), dtype=bool)
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in gcfg.ntc_patterns]
    return np.asarray([any(pattern.search(str(value)) for pattern in patterns) for value in labels], dtype=bool)


def _is_non_targeting_unique(labels: Sequence[str], gcfg: GuideConfig) -> np.ndarray:
    """NTC detection using one regex evaluation per unique label."""
    labels = np.asarray(labels)
    if not gcfg.ntc_patterns:
        return np.zeros(len(labels), dtype=bool)
    if labels.size == 0:
        return np.asarray([], dtype=bool)
    unique_labels, inverse = np.unique(labels.astype(str), return_inverse=True)
    unique_mask = is_non_targeting(unique_labels, gcfg)
    return unique_mask[inverse]


# STANDARD top-two guide search


def _top_two_guides_dense_chunked(X, chunk_size: int = 20_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original chunked/vectorized top-two implementation."""
    n_obs, n_vars = X.shape
    top_idx = np.zeros(n_obs, dtype=np.int64)
    top_val = np.zeros(n_obs, dtype=np.float64)
    second_val = np.zeros(n_obs, dtype=np.float64)
    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        block = X[start:stop]
        if sparse.issparse(block):
            dense = block.toarray()
        else:
            dense = np.asarray(block)
        dense = dense.astype(np.float64, copy=False)
        if n_vars == 0:
            continue
        if n_vars == 1:
            top_idx[start:stop] = 0
            top_val[start:stop] = dense[:, 0]
            second_val[start:stop] = 0.0
            continue
        # argpartition places the two largest values in the final two slots.
        part = np.argpartition(dense, -2, axis=1)[:, -2:]
        rows = np.arange(dense.shape[0])[:, None]
        values = dense[rows, part]
        order = np.argsort(values, axis=1)
        row_flat = rows[:, 0]
        top_idx[start:stop] = part[row_flat, order[:, 1]]
        top_val[start:stop] = values[row_flat, order[:, 1]]
        second_val[start:stop] = values[row_flat, order[:, 0]]
    return (top_idx, top_val, second_val)


# LARGE sparse top-two implementation


def _csr_top_two_numba(
    X: sparse.csr_matrix,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Numba sparse top-two implementation.

    Returns None when Numba is unavailable so the caller can fall back to the
    pure-Python sparse implementation.

    Returns
    -------
    top_idx
        Column index of highest-count guide.
    top_val
        Highest guide count.
    second_val
        Second-highest guide count.
    total
        Total guide counts per cell.
    detected
        Number of guides above the detection threshold is added later because
        that threshold comes from configuration.
    """
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def _scan(indptr, indices, data, n_rows):
        top_idx = np.zeros(n_rows, dtype=np.int64)
        top_val = np.zeros(n_rows, dtype=np.float64)
        second_val = np.zeros(n_rows, dtype=np.float64)
        total = np.zeros(n_rows, dtype=np.float64)
        for row in range(n_rows):
            start = indptr[row]
            stop = indptr[row + 1]
            best_value = 0.0
            second_value = 0.0
            best_index = 0
            row_total = 0.0
            for pos in range(start, stop):
                value = float(data[pos])
                column = int(indices[pos])
                row_total += value
                if value > best_value:
                    second_value = best_value
                    best_value = value
                    best_index = column
                elif value > second_value:
                    second_value = value
            top_idx[row] = best_index
            top_val[row] = best_value
            second_val[row] = second_value
            total[row] = row_total
        return (top_idx, top_val, second_val, total)

    result = _scan(X.indptr, X.indices, X.data, X.shape[0])
    return result


def _csr_top_two_python(X: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse fallback when Numba is unavailable.

    Memory use remains bounded, although it is slower than the Numba path.
    """
    n_obs = X.shape[0]
    top_idx = np.zeros(n_obs, dtype=np.int64)
    top_val = np.zeros(n_obs, dtype=np.float64)
    second_val = np.zeros(n_obs, dtype=np.float64)
    total = np.zeros(n_obs, dtype=np.float64)
    indptr = X.indptr
    indices = X.indices
    data = X.data
    for row in range(n_obs):
        start = indptr[row]
        stop = indptr[row + 1]
        if start == stop:
            continue
        row_values = data[start:stop]
        row_indices = indices[start:stop]
        total[row] = float(row_values.sum())
        if len(row_values) == 1:
            top_idx[row] = int(row_indices[0])
            top_val[row] = float(row_values[0])
            continue
        # Guide matrices are normally extremely sparse, so sorting the few
        # non-zero entries in a row is inexpensive.
        largest = np.argpartition(row_values, -2)[-2:]
        values = row_values[largest]
        order = np.argsort(values)
        second_pos = largest[order[0]]
        top_pos = largest[order[1]]
        top_idx[row] = int(row_indices[top_pos])
        top_val[row] = float(row_values[top_pos])
        second_val[row] = float(row_values[second_pos])
    return (top_idx, top_val, second_val, total)


def _detected_guides_csr(X: sparse.csr_matrix, threshold: float) -> np.ndarray:
    """Count guides above threshold without constructing ``X > threshold``."""
    n_obs = X.shape[0]
    detected = np.zeros(n_obs, dtype=np.int32)
    try:
        from numba import njit

        @njit(cache=True)
        def _count(indptr, data, threshold_value, n_rows):
            out = np.zeros(n_rows, dtype=np.int32)
            for row in range(n_rows):
                count = 0
                for pos in range(indptr[row], indptr[row + 1]):
                    if data[pos] > threshold_value:
                        count += 1
                out[row] = count
            return out

        return _count(X.indptr, X.data, float(threshold), n_obs)
    except ImportError:
        for row in range(n_obs):
            start = X.indptr[row]
            stop = X.indptr[row + 1]
            detected[row] = int(np.count_nonzero(X.data[start:stop] > threshold))
        return detected


# Public top-two API


def top_two_guides(
    X, chunk_size: int = 20_000, max_dense_elements: int = 20_000_000, *, force_sparse: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(top_index, top_value, second_value)`` per cell.

    STANDARD
        Chunked dense/vectorized algorithm.

    LARGE sparse matrix
        CSR algorithm, optionally Numba accelerated.
    """
    n_obs, n_vars = X.shape
    use_sparse = sparse.issparse(X) and (force_sparse or (chunk_size * max(n_vars, 1) > max_dense_elements))
    if use_sparse:
        csr = sparse.csr_matrix(X)
        result = _csr_top_two_numba(csr)
        if result is not None:
            top_idx, top_val, second_val, _ = result
            return (top_idx, top_val, second_val)
        top_idx, top_val, second_val, _ = _csr_top_two_python(csr)
        return (top_idx, top_val, second_val)
    return _top_two_guides_dense_chunked(X, chunk_size)


# Assignment entry point


def assign_guides(expr: ad.AnnData, guides: Optional[ad.AnnData], cfg: Config) -> ad.AnnData:
    """Write guide/target assignments into ``expr.obs``."""
    if guides is None:
        if cfg.guides.assignment_mode in ("dual_guide_pair", "pair"):
            raise ValueError(
                "guides.assignment_mode='dual_guide_pair' needs a guide count matrix; "
                "label-based guide input cannot be pair-resolved"
            )
        return _assign_from_labels(expr, cfg)
    if cfg.guides.assignment_mode in ("dual_guide_pair", "pair"):
        from .dual_guides import assign_guide_pairs

        return assign_guide_pairs(expr, guides, cfg)
    expr = _assign_from_matrix(expr, guides, cfg)
    expr.obs["guide_assignment_mode"] = pd.Categorical(["single_guide"] * expr.n_obs)
    return expr


# Guide-matrix assignment


def _assign_from_matrix(expr: ad.AnnData, guides: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Assign cells from a guide count matrix."""
    gcfg = cfg.guides
    large_mode = cfg.use_large_mode(expr.n_obs)
    # Avoid copying an already cell-aligned guide matrix.
    if len(guides.obs_names) == len(expr.obs_names) and guides.obs_names.equals(expr.obs_names):
        aligned_guides = guides
    else:
        logger.info("Aligning guide matrix to expression cells")
        aligned_guides = guides[expr.obs_names].copy()
    X = aligned_guides.layers["counts"] if "counts" in aligned_guides.layers else aligned_guides.X
    # Large sparse execution
    if large_mode and sparse.issparse(X):
        logger.info("Guide assignment mode: LARGE sparse (%d cells x %d guides)", X.shape[0], X.shape[1])
        X_csr = sparse.csr_matrix(X)
        result = _csr_top_two_numba(X_csr)
        if result is not None:
            (top_idx, top_val, second_val, total) = result
            logger.info("Guide top-two search: sparse Numba implementation")
        else:
            (top_idx, top_val, second_val, total) = _csr_top_two_python(X_csr)
            logger.info(
                "Guide top-two search: sparse Python implementation (install numba for faster million-cell execution)"
            )
        detected = _detected_guides_csr(X_csr, gcfg.detection_threshold)
    # Standard execution
    else:
        logger.info("Guide assignment mode: STANDARD (%d cells x %d guides)", X.shape[0], X.shape[1])
        (top_idx, top_val, second_val) = top_two_guides(
            X, chunk_size=cfg.scaling.guide_chunk_size, max_dense_elements=cfg.scaling.guide_max_dense_elements
        )
        total = np.asarray(X.sum(axis=1)).ravel()
        detected = np.asarray((X > gcfg.detection_threshold).sum(axis=1)).ravel()
    guide_ids = aligned_guides.var_names.to_numpy().astype(str)
    guide_targets = resolve_guide_targets(aligned_guides, gcfg)
    # Assignment gate
    assigned = (top_val >= max(gcfg.min_umi, 1)) & (top_val > gcfg.dominance_ratio * second_val)
    if gcfg.max_second_umi is not None and gcfg.max_second_umi >= 0:
        n_before = int(assigned.sum())
        assigned &= second_val <= gcfg.max_second_umi
        logger.info(
            "Multiplet gate (second guide <= %d UMIs): %d -> %d assigned cells",
            gcfg.max_second_umi,
            n_before,
            int(assigned.sum()),
        )
    has_counts = top_val > 0
    guide_call = np.full(expr.n_obs, gcfg.unassigned_label, dtype=object)
    target_call = np.full(expr.n_obs, gcfg.unassigned_label, dtype=object)
    ambiguous = has_counts & ~assigned
    guide_call[ambiguous] = gcfg.ambiguous_label
    target_call[ambiguous] = gcfg.ambiguous_label
    guide_call[assigned] = guide_ids[top_idx[assigned]]
    target_call[assigned] = guide_targets[top_idx[assigned]]
    # Diagnostics
    expr.obs[OBS_TOP] = top_val
    expr.obs[OBS_SECOND] = second_val
    expr.obs[OBS_TOTAL] = total
    expr.obs[OBS_NDETECTED] = detected
    expr.obs[OBS_GUIDE] = pd.Categorical(guide_call.astype(str))
    _finalize_labels(expr, target_call, gcfg, large_mode=large_mode)
    # Guide metadata — written to the ORIGINAL guides object as well as the aligned
    # view: when QC removed cells the aligned object is a throwaway copy, and
    # guide_representation(), write_guide_table() and uns['guide_target_genes'] all
    # read the targets from the object the caller keeps.
    ntc_flags = is_non_targeting(guide_targets, gcfg)
    guide_calls = expr.obs[OBS_GUIDE].astype(str)
    target_calls = expr.obs[OBS_TARGET].astype(str)
    for obj in {id(aligned_guides): aligned_guides, id(guides): guides}.values():
        obj.var["target_gene"] = guide_targets
        obj.var["is_non_targeting"] = ntc_flags
        # Cells that are not in the analysed expression object (QC-filtered) were
        # never evaluated; say so rather than leaving them blank or 'unassigned'.
        obj.obs[OBS_GUIDE] = guide_calls.reindex(obj.obs_names).fillna(NOT_EVALUATED_LABEL).to_numpy()
        obj.obs[OBS_TARGET] = target_calls.reindex(obj.obs_names).fillna(NOT_EVALUATED_LABEL).to_numpy()
    _log_assignment(expr, cfg)
    return expr


# Label-based assignment


def _assign_from_labels(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Assignment path for pre-computed per-cell guide labels.

    This is the path used by datasets such as Replogle and KOLF when the H5AD
    already provides one perturbation/guide identity per cell.
    """
    gcfg = cfg.guides
    if RAW_GUIDE_LABEL not in expr.obs.columns:
        raise ValueError(f"No guide matrix and no obs['{RAW_GUIDE_LABEL}'] column; cannot determine perturbations.")
    large_mode = cfg.use_large_mode(expr.n_obs)
    raw_series = expr.obs[RAW_GUIDE_LABEL]
    # LARGE: parse each unique label only once
    if large_mode:
        # Categorical representation is especially efficient for KOLF, where
        # millions of cells share only ~12k perturbation labels.
        categorical = pd.Categorical(raw_series.astype(str))
        categories = np.asarray(categorical.categories.astype(str), dtype=str)
        codes = categorical.codes
        parsed_categories = parse_target_genes(categories, gcfg).astype(object)
        target_call = np.empty(expr.n_obs, dtype=object)
        valid = codes >= 0
        target_call[valid] = parsed_categories[codes[valid]]
        target_call[~valid] = gcfg.unassigned_label
        raw = np.empty(expr.n_obs, dtype=object)
        raw[valid] = categories[codes[valid]]
        raw[~valid] = ""
        logger.info(
            "Guide label parsing mode: LARGE — %d cells mapped through %d unique labels", expr.n_obs, len(categories)
        )
    # STANDARD: original direct parsing
    else:
        raw = raw_series.astype(str).to_numpy()
        target_call = parse_target_genes(raw, gcfg).astype(object)
    # Explicit failure labels
    raw_str = np.asarray(raw).astype(str)
    raw_lower = np.char.lower(raw_str)
    failure_labels = (
        (gcfg.unassigned_label, gcfg.unassigned_label),
        (gcfg.ambiguous_label, gcfg.ambiguous_label),
        ("NA", gcfg.unassigned_label),
        ("nan", gcfg.unassigned_label),
        ("None", gcfg.unassigned_label),
        ("", gcfg.unassigned_label),
    )
    for source_label, replacement in failure_labels:
        target_call[raw_lower == source_label.lower()] = replacement
    expr.obs[OBS_GUIDE] = pd.Categorical(raw_str)
    _finalize_labels(expr, target_call, gcfg, large_mode=large_mode)
    _log_assignment(expr, cfg)
    return expr


# Label finalization


def _finalize_labels(expr: ad.AnnData, target_call: np.ndarray, gcfg: GuideConfig, *, large_mode: bool = False) -> None:
    """Collapse NTC guides and assign perturbation class."""
    if large_mode:
        ntc_mask = _is_non_targeting_unique(target_call, gcfg)
    else:
        ntc_mask = is_non_targeting(target_call, gcfg)
    special = {gcfg.unassigned_label, gcfg.ambiguous_label}
    ntc_mask &= ~np.isin(target_call.astype(str), list(special))
    target_call = target_call.copy()
    target_call[ntc_mask] = gcfg.ntc_label
    # An object array of 2.6m entries is still manageable, but keep it only for
    # this short construction step and immediately convert the final result to
    # pandas categoricals.
    klass = np.full(len(target_call), CLASS_TARGETING, dtype=object)
    klass[ntc_mask] = CLASS_NTC
    klass[target_call == gcfg.ambiguous_label] = CLASS_AMBIGUOUS
    klass[target_call == gcfg.unassigned_label] = CLASS_UNASSIGNED
    expr.obs[OBS_TARGET] = pd.Categorical(target_call.astype(str))
    expr.obs[OBS_CLASS] = pd.Categorical(
        klass.astype(str), categories=[CLASS_TARGETING, CLASS_NTC, CLASS_AMBIGUOUS, CLASS_UNASSIGNED]
    )


# Logging


def _log_assignment(expr: ad.AnnData, cfg: Config) -> None:
    """Log assignment statistics."""
    counts = expr.obs[OBS_CLASS].value_counts()
    n = expr.n_obs
    n_targeting = int(counts.get(CLASS_TARGETING, 0))
    n_ntc = int(counts.get(CLASS_NTC, 0))
    n_ambiguous = int(counts.get(CLASS_AMBIGUOUS, 0))
    n_unassigned = int(counts.get(CLASS_UNASSIGNED, 0))
    logger.info(
        "Guide assignment: %d targeting (%.1f%%), %d non-targeting, %d ambiguous, %d unassigned",
        n_targeting,
        100 * n_targeting / max(n, 1),
        n_ntc,
        n_ambiguous,
        n_unassigned,
    )
    n_targets = target_genes(expr, cfg).size
    logger.info("%d distinct target genes assigned", n_targets)


# Summaries


def target_genes(expr: ad.AnnData, cfg: Config) -> np.ndarray:
    """Sorted real target genes excluding controls/failure classes."""
    obs = expr.obs
    mask = obs[OBS_CLASS] == CLASS_TARGETING
    return np.array(sorted(obs.loc[mask, OBS_TARGET].astype(str).unique()))


def assignment_summary(expr: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Per-target counts and downstream testability."""
    obs = expr.obs
    tab = obs[OBS_TARGET].astype(str).value_counts().rename_axis(OBS_TARGET).reset_index(name="n_cells")
    # Each target should have one class. Instead of an expensive Python lambda
    # over every row, drop duplicate target/class pairs first.
    class_map = (
        obs[[OBS_TARGET, OBS_CLASS]]
        .astype({OBS_TARGET: str, OBS_CLASS: str})
        .drop_duplicates(subset=[OBS_TARGET])
        .set_index(OBS_TARGET)[OBS_CLASS]
    )
    tab["class"] = tab[OBS_TARGET].map(class_map)
    measured = set(expr.var_names)
    tab["detected_in_expression"] = tab[OBS_TARGET].isin(measured)
    tab["testable"] = (
        (tab["class"] == CLASS_TARGETING)
        & (tab["detected_in_expression"])
        & (tab["n_cells"] >= cfg.perturbation.min_cells_per_target)
    )
    return tab.sort_values("n_cells", ascending=False).reset_index(drop=True)


def per_lane_assignment(expr: ad.AnnData, lane_key: str = "lane_id") -> pd.DataFrame:
    """Assignment-class breakdown per lane."""
    if lane_key not in expr.obs.columns:
        return pd.DataFrame()
    tab = expr.obs.groupby([lane_key, OBS_CLASS], observed=True).size().unstack(fill_value=0)
    tab["n_cells"] = tab.sum(axis=1)
    for klass in (CLASS_TARGETING, CLASS_NTC, CLASS_AMBIGUOUS, CLASS_UNASSIGNED):
        if klass in tab.columns:
            tab[f"pct_{klass}"] = 100 * tab[klass] / tab["n_cells"]
    return tab.reset_index()


def guide_representation(guides: Optional[ad.AnnData], expr: ad.AnnData) -> pd.DataFrame:
    """Cells per guide."""
    if OBS_GUIDE not in expr.obs.columns:
        return pd.DataFrame()
    counts = expr.obs[OBS_GUIDE].astype(str).value_counts()
    df = counts.rename_axis("guide_id").reset_index(name="n_cells")
    if guides is not None and "target_gene" in guides.var.columns:
        mapping = guides.var["target_gene"].astype(str).to_dict()
        df["target_gene"] = df["guide_id"].map(mapping)
        present = set(df["guide_id"])
        missing = [guide for guide in guides.var_names.astype(str) if guide not in present]
        if missing:
            extra = pd.DataFrame(
                {
                    "guide_id": (missing),
                    "n_cells": (np.zeros(len(missing), dtype=int)),
                    "target_gene": [mapping.get(guide) for guide in missing],
                }
            )
            df = pd.concat([df, extra], ignore_index=True)
    return df.sort_values("n_cells", ascending=False).reset_index(drop=True)
