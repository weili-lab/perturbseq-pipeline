"""High-MOI guide assignment: multi-guide membership per cell.

Selected with ``guides.assignment_mode: high_moi``. In a high-MOI Perturb-seq
design every cell carries several guides, so the single-guide dominance rule
(``single_guide``) calls almost every cell *ambiguous*. Here a cell becomes a
**member** of every guide whose UMI count passes the call, and of every target
those guides map to.

Outputs (the contract every later stage can build on)
------------------------------------------------------
``obsm[guides.high_moi.guide_membership_obsm_key]``
    cells x guides, sparse CSR int8; 1 where the guide is called in the cell.
``obsm[guides.high_moi.membership_obsm_key]``
    cells x targets, sparse CSR int8; 1 where the cell carries at least one
    guide of the target. Non-targeting guides collapse into one final column
    named ``guides.ntc_label``; ``uns['membership_targets']`` holds the column
    names, ``uns['membership_guides']`` the guide ids and
    ``uns['membership_ntc_guides']`` the ids of the non-targeting guides (the
    negative-control pseudo-targets of the membership-aware statistics).
``obs['n_guides_assigned']``, ``obs['n_targets_assigned']``
    row sums (targets exclude the NTC column).
``obs['n_guides_called']``
    guides passing the call **before** the ``max_guides_per_cell`` /
    ``min_guides_per_cell`` gates (so over-cap cells keep their real count for
    diagnostics while their membership rows are cleared).
``obs['is_ntc_only']``
    at least one NTC guide and no targeting guide.
``obs['perturbation_class']``
    ``targeting`` (>= 1 targeting membership), ``non-targeting`` (NTC-only),
    ``ambiguous`` (more than ``max_guides_per_cell`` guides: doublet-like, no
    membership kept) or ``unassigned`` (no guide called).

Legacy columns (``obs['target_gene']``, ``obs['guide_id']``, ``top_guide_count``,
``second_guide_count``, ``total_guide_counts``, ``n_guides_detected``) are
written exactly as the single-guide path writes them, with the **primary**
label = the cell's highest-UMI targeting guide (or NTC guide for NTC-only
cells). Every downstream stage therefore runs unchanged on the primary label;
membership-aware statistics read the ``obsm`` matrices.

The ``single_guide`` and ``dual_guide_pair`` paths are untouched by this module.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config
from .guides import (
    CLASS_AMBIGUOUS,
    CLASS_NTC,
    CLASS_TARGETING,
    CLASS_UNASSIGNED,
    NOT_EVALUATED_LABEL,
    OBS_CLASS,
    OBS_GUIDE,
    OBS_NDETECTED,
    OBS_SECOND,
    OBS_TARGET,
    OBS_TOP,
    OBS_TOTAL,
    _csr_top_two_numba,
    _csr_top_two_python,
    _detected_guides_csr,
    _log_assignment,
    is_non_targeting,
    resolve_guide_targets,
)

logger = logging.getLogger(__name__)

MODE_HIGH_MOI = "high_moi"
OBS_MODE = "guide_assignment_mode"
OBS_N_GUIDES = "n_guides_assigned"
OBS_N_CALLED = "n_guides_called"
OBS_N_TARGETS = "n_targets_assigned"
OBS_NTC_ONLY = "is_ntc_only"
UNS_TARGETS = "membership_targets"
UNS_GUIDES = "membership_guides"
UNS_NTC_GUIDES = "membership_ntc_guides"
UNS_GUIDE_TARGETS = "membership_guide_targets"
UNS_CALLING = "high_moi_calling"
UNS_RANK_PROFILE = "high_moi_rank_umi_profile"

TABLE_CALLING = "high_moi_calling"
TABLE_RANK_PROFILE = "high_moi_rank_umi_profile"
TABLE_CELLS_PER_TARGET = "high_moi_cells_per_target"


def is_high_moi_mode(cfg: Config) -> bool:
    return cfg.guides.assignment_mode == MODE_HIGH_MOI


# Membership calling


def _row_index_of_entries(X: sparse.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))


def call_membership_threshold(X: sparse.csr_matrix, top_val: np.ndarray, min_umi: int, min_frac_of_top: float):
    """Guide is a member when ``umi >= min_umi`` and ``umi >= min_frac_of_top * top``."""
    rows = _row_index_of_entries(X)
    thr = np.maximum(float(min_umi), min_frac_of_top * top_val[rows])
    keep = X.data >= thr
    out = sparse.csr_matrix((keep.astype(np.int8), X.indices.copy(), X.indptr.copy()), shape=X.shape)
    out.eliminate_zeros()
    return out


def call_membership_knee(X: sparse.csr_matrix, min_umi: int):
    """Per cell, the largest ``log1p`` drop between consecutive ranked guides.

    Only guides with ``>= min_umi`` UMIs are candidates; the drop from the last
    candidate to the first sub-threshold guide (or to zero) competes too, so a
    cell whose candidates are all real keeps all of them.
    """
    n_cells = X.shape[0]
    keep = np.zeros(X.nnz, dtype=np.int8)
    for i in range(n_cells):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        if hi == lo:
            continue
        vals = X.data[lo:hi]
        order = np.argsort(-vals, kind="mergesort")
        sorted_vals = vals[order]
        n_cand = int((sorted_vals >= min_umi).sum())
        if n_cand == 0:
            continue
        tail = sorted_vals[n_cand] if n_cand < sorted_vals.size else 0.0
        seq = np.log1p(np.concatenate([sorted_vals[:n_cand], [tail]]))
        gaps = seq[:-1] - seq[1:]
        k = int(np.argmax(gaps)) + 1
        keep[lo + order[:k]] = 1
    out = sparse.csr_matrix((keep, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    out.eliminate_zeros()
    return out


def _zero_rows(M: sparse.csr_matrix, drop: np.ndarray) -> sparse.csr_matrix:
    if not drop.any():
        return M
    keep = sparse.diags((~drop).astype(np.int8), dtype=np.int8)
    out = sparse.csr_matrix(keep @ M, dtype=np.int8)
    out.eliminate_zeros()
    return out


def _zero_cols(M: sparse.csr_matrix, drop: np.ndarray) -> sparse.csr_matrix:
    if not drop.any():
        return M
    keep = sparse.diags((~drop).astype(np.int8), dtype=np.int8)
    out = sparse.csr_matrix(M @ keep, dtype=np.int8)
    out.eliminate_zeros()
    return out


def _rank_umi_profile(X: sparse.csr_matrix, max_rank: int) -> pd.DataFrame:
    """Median / p10 / p90 UMI of the k-th ranked guide per cell, k = 1..max_rank."""
    if X.nnz == 0:
        return pd.DataFrame(columns=["rank", "n_cells", "median_umi", "p10_umi", "p90_umi"])
    rows = _row_index_of_entries(X)
    order = np.lexsort((-X.data, rows))
    rows_sorted = rows[order]
    vals_sorted = X.data[order]
    starts = X.indptr[rows_sorted]
    rank = np.arange(X.nnz, dtype=np.int64) - starts + 1
    out = []
    for k in range(1, max_rank + 1):
        v = vals_sorted[rank == k]
        if v.size == 0:
            break
        out.append(
            {
                "rank": k,
                "n_cells": int(v.size),
                "median_umi": float(np.median(v)),
                "p10_umi": float(np.percentile(v, 10)),
                "p90_umi": float(np.percentile(v, 90)),
            }
        )
    return pd.DataFrame(out)


# Assignment entry point


def assign_high_moi(expr: ad.AnnData, guides: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Write membership matrices and the primary-label contract into ``expr``."""
    gcfg = cfg.guides
    hcfg = gcfg.high_moi
    if len(guides.obs_names) == len(expr.obs_names) and guides.obs_names.equals(expr.obs_names):
        aligned = guides
    else:
        logger.info("Aligning guide matrix to expression cells")
        aligned = guides[expr.obs_names].copy()
    X = aligned.layers["counts"] if "counts" in aligned.layers else aligned.X
    X = sparse.csr_matrix(X, dtype=np.float64)
    X.sum_duplicates()
    n_cells, n_guides = X.shape
    logger.info("Guide assignment mode: HIGH-MOI membership (%d cells x %d guides, method=%s)", n_cells, n_guides, hcfg.method)
    # Legacy per-cell diagnostics, identical in meaning to the single-guide path.
    result = _csr_top_two_numba(X)
    if result is None:
        result = _csr_top_two_python(X)
    top_idx, top_val, second_val, total = result[:4]
    detected = _detected_guides_csr(X, gcfg.detection_threshold)
    # Guide -> target
    guide_ids = aligned.var_names.to_numpy().astype(str)
    guide_targets = np.asarray(resolve_guide_targets(aligned, gcfg), dtype=object)
    ntc_guide = is_non_targeting(guide_targets, gcfg)
    unusable = np.isin(guide_targets.astype(str), [gcfg.unassigned_label, gcfg.ambiguous_label])
    targeting_guide = ~ntc_guide & ~unusable
    target_names: List[str] = sorted({str(t) for t in guide_targets[targeting_guide]})
    columns = target_names + [gcfg.ntc_label]
    col_of_guide = np.full(n_guides, -1, dtype=np.int64)
    pos = {t: i for i, t in enumerate(target_names)}
    for j in np.flatnonzero(targeting_guide):
        col_of_guide[j] = pos[str(guide_targets[j])]
    col_of_guide[ntc_guide & ~unusable] = len(target_names)
    usable = col_of_guide >= 0
    if (~usable).any():
        logger.warning("%d guide(s) have no target label and are excluded from membership", int((~usable).sum()))
    # Membership call
    if hcfg.method == "knee":
        M = call_membership_knee(X, hcfg.min_umi)
    else:
        M = call_membership_threshold(X, top_val, hcfg.min_umi, hcfg.min_frac_of_top)
    M = _zero_cols(M, ~usable)
    n_called = np.asarray(M.sum(axis=1)).ravel().astype(np.int64)
    over_cap = n_called > hcfg.max_guides_per_cell
    too_few = (n_called > 0) & (n_called < hcfg.min_guides_per_cell)
    M = _zero_rows(M, over_cap | too_few)
    M = sparse.csr_matrix(M, dtype=np.int8)
    n_guides_assigned = np.asarray(M.sum(axis=1)).ravel().astype(np.int64)
    # Guide membership -> target membership (NTC guides collapse into the last column)
    g2t = sparse.csr_matrix(
        (np.ones(int(usable.sum()), dtype=np.int8), (np.flatnonzero(usable), col_of_guide[usable])),
        shape=(n_guides, len(columns)),
    )
    T = sparse.csr_matrix(M @ g2t)
    T.data[:] = 1
    T = sparse.csr_matrix(T, dtype=np.int8)
    ntc_member = np.asarray(T[:, -1].todense()).ravel() > 0
    n_targets_assigned = np.asarray(T.sum(axis=1)).ravel().astype(np.int64) - ntc_member.astype(np.int64)
    is_ntc_only = ntc_member & (n_targets_assigned == 0)
    # Class
    klass = np.full(n_cells, CLASS_UNASSIGNED, dtype=object)
    klass[n_targets_assigned > 0] = CLASS_TARGETING
    klass[is_ntc_only] = CLASS_NTC
    klass[over_cap] = CLASS_AMBIGUOUS
    # Primary label: highest-UMI targeting member (NTC member for NTC-only cells)
    V = sparse.csr_matrix(X.multiply(M)).tocoo()
    guide_call = np.full(n_cells, gcfg.unassigned_label, dtype=object)
    target_call = np.full(n_cells, gcfg.unassigned_label, dtype=object)
    if V.nnz:
        is_t = targeting_guide[V.col]
        order = np.lexsort((-V.data, ~is_t, V.row))
        rows_sorted = V.row[order]
        first = np.unique(rows_sorted, return_index=True)[1]
        prim_rows = rows_sorted[first]
        prim_cols = V.col[order][first]
        guide_call[prim_rows] = guide_ids[prim_cols]
        target_call[prim_rows] = guide_targets[prim_cols]
    target_call[is_ntc_only] = gcfg.ntc_label
    guide_call[over_cap] = gcfg.ambiguous_label
    target_call[over_cap] = gcfg.ambiguous_label
    # Legacy contract
    expr.obs[OBS_TOP] = top_val
    expr.obs[OBS_SECOND] = second_val
    expr.obs[OBS_TOTAL] = total
    expr.obs[OBS_NDETECTED] = detected
    expr.obs[OBS_GUIDE] = pd.Categorical(guide_call.astype(str))
    expr.obs[OBS_TARGET] = pd.Categorical(target_call.astype(str))
    expr.obs[OBS_CLASS] = pd.Categorical(
        klass.astype(str), categories=[CLASS_TARGETING, CLASS_NTC, CLASS_AMBIGUOUS, CLASS_UNASSIGNED]
    )
    expr.obs[OBS_MODE] = pd.Categorical([MODE_HIGH_MOI] * n_cells)
    # Membership contract
    expr.obs[OBS_N_CALLED] = n_called
    expr.obs[OBS_N_GUIDES] = n_guides_assigned
    expr.obs[OBS_N_TARGETS] = n_targets_assigned
    expr.obs[OBS_NTC_ONLY] = is_ntc_only
    expr.obsm[hcfg.guide_membership_obsm_key] = M
    expr.obsm[hcfg.membership_obsm_key] = T
    expr.uns[UNS_TARGETS] = list(columns)
    expr.uns[UNS_GUIDES] = list(guide_ids)
    expr.uns[UNS_NTC_GUIDES] = [str(g) for g in guide_ids[ntc_guide & ~unusable]]
    expr.uns[UNS_GUIDE_TARGETS] = [str(t) for t in guide_targets]
    expr.uns[UNS_RANK_PROFILE] = _rank_umi_profile(X, hcfg.rank_profile_max_rank)
    expr.uns[UNS_CALLING] = {
        "method": hcfg.method,
        "min_umi": int(hcfg.min_umi),
        "min_frac_of_top": float(hcfg.min_frac_of_top),
        "max_guides_per_cell": int(hcfg.max_guides_per_cell),
        "min_guides_per_cell": int(hcfg.min_guides_per_cell),
        "n_targets": int(len(target_names)),
        "n_guides_usable": int(usable.sum()),
        "n_cells_over_cap": int(over_cap.sum()),
        "n_cells_too_few": int(too_few.sum()),
        "n_cells_ntc_only": int(is_ntc_only.sum()),
    }
    # Guide metadata on the caller's objects (same as the single-guide path)
    guide_calls = expr.obs[OBS_GUIDE].astype(str)
    target_calls = expr.obs[OBS_TARGET].astype(str)
    for obj in {id(aligned): aligned, id(guides): guides}.values():
        obj.var["target_gene"] = guide_targets
        obj.var["is_non_targeting"] = ntc_guide
        obj.obs[OBS_GUIDE] = guide_calls.reindex(obj.obs_names).fillna(NOT_EVALUATED_LABEL).to_numpy()
        obj.obs[OBS_TARGET] = target_calls.reindex(obj.obs_names).fillna(NOT_EVALUATED_LABEL).to_numpy()
    _log_assignment(expr, cfg)
    assigned = n_guides_assigned > 0
    logger.info(
        "High-MOI membership: %d/%d cells with >= 1 called guide; median %.0f guides and %.0f targets per assigned "
        "cell; %d NTC-only cells; %d cells above max_guides_per_cell=%d; %d targets with >= 1 cell",
        int(assigned.sum()),
        n_cells,
        float(np.median(n_guides_assigned[assigned])) if assigned.any() else 0.0,
        float(np.median(n_targets_assigned[assigned])) if assigned.any() else 0.0,
        int(is_ntc_only.sum()),
        int(over_cap.sum()),
        hcfg.max_guides_per_cell,
        int((np.asarray(T[:, :-1].sum(axis=0)).ravel() > 0).sum()),
    )
    return expr


# Accessors


def membership_matrix(expr: ad.AnnData, cfg: Config) -> sparse.csr_matrix:
    key = cfg.guides.high_moi.membership_obsm_key
    if key not in expr.obsm:
        raise KeyError(f"obsm[{key!r}] not found; run guide assignment in high_moi mode first")
    return sparse.csr_matrix(expr.obsm[key])


def membership_targets(expr: ad.AnnData) -> List[str]:
    return [str(t) for t in expr.uns[UNS_TARGETS]]


def cells_per_target(expr: ad.AnnData, cfg: Config) -> pd.Series:
    """Membership column sums, indexed by target (NTC column included)."""
    T = membership_matrix(expr, cfg)
    return pd.Series(np.asarray(T.sum(axis=0)).ravel().astype(int), index=membership_targets(expr))


# Summaries (membership-aware siblings of guides.assignment_summary & co.)


def membership_assignment_summary(expr: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Per-target membership counts and testability (legacy columns plus membership ones).

    ``n_cells`` counts every cell carrying the target; ``n_cells_primary`` the
    cells whose primary label is the target (what the legacy table would show).
    ``testable`` keeps its legacy meaning — enough **primary-label** cells for
    the downstream stages, which test each cell under its primary target in
    this version — and ``testable_membership`` says whether the target has
    enough member cells for membership-aware statistics.
    """
    gcfg = cfg.guides
    counts = cells_per_target(expr, cfg)
    primary = expr.obs[OBS_TARGET].astype(str).value_counts()
    klass = expr.obs[OBS_CLASS].astype(str).value_counts()
    rows = []
    for target, n in counts.items():
        is_ntc = target == gcfg.ntc_label
        rows.append(
            {
                OBS_TARGET: target,
                "n_cells": int(n),
                "n_cells_primary": int(primary.get(target, 0)),
                "class": CLASS_NTC if is_ntc else CLASS_TARGETING,
            }
        )
    for label, cls in ((gcfg.ambiguous_label, CLASS_AMBIGUOUS), (gcfg.unassigned_label, CLASS_UNASSIGNED)):
        n = int(klass.get(cls, 0))
        if n:
            rows.append({OBS_TARGET: label, "n_cells": n, "n_cells_primary": n, "class": cls})
    tab = pd.DataFrame(rows, columns=[OBS_TARGET, "n_cells", "n_cells_primary", "class"])
    measured = set(expr.var_names)
    tab["detected_in_expression"] = tab[OBS_TARGET].isin(measured)
    eligible = (tab["class"] == CLASS_TARGETING) & tab["detected_in_expression"]
    min_cells = cfg.perturbation.min_cells_per_target
    tab["testable"] = eligible & (tab["n_cells_primary"] >= min_cells)
    tab["testable_membership"] = eligible & (tab["n_cells"] >= min_cells)
    return tab.sort_values(["n_cells", OBS_TARGET], ascending=[False, True]).reset_index(drop=True)


def membership_guide_representation(guides: Optional[ad.AnnData], expr: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Cells per guide under membership, plus the primary-label count."""
    key = cfg.guides.high_moi.guide_membership_obsm_key
    if key not in expr.obsm or UNS_GUIDES not in expr.uns:
        return pd.DataFrame()
    M = sparse.csr_matrix(expr.obsm[key])
    guide_ids = [str(g) for g in expr.uns[UNS_GUIDES]]
    df = pd.DataFrame({"guide_id": guide_ids, "n_cells": np.asarray(M.sum(axis=0)).ravel().astype(int)})
    primary = expr.obs[OBS_GUIDE].astype(str).value_counts()
    df["n_cells_primary"] = df["guide_id"].map(primary).fillna(0).astype(int)
    if guides is not None and "target_gene" in guides.var.columns:
        mapping = guides.var["target_gene"].astype(str).to_dict()
        df["target_gene"] = df["guide_id"].map(mapping)
    return df.sort_values(["n_cells", "guide_id"], ascending=[False, True]).reset_index(drop=True)


def high_moi_tables(expr: ad.AnnData, cfg: Config, lane_key: str = "lane_id") -> Dict[str, pd.DataFrame]:
    """Diagnostics: calling summary (overall + per lane), rank-UMI profile, cells per target."""
    obs = expr.obs
    hcfg = cfg.guides.high_moi
    info = dict(expr.uns.get(UNS_CALLING, {}))
    n = expr.n_obs
    ng = obs[OBS_N_GUIDES].to_numpy()
    nt = obs[OBS_N_TARGETS].to_numpy()
    assigned = ng > 0
    counts = cells_per_target(expr, cfg)
    targets_only = counts.drop(index=cfg.guides.ntc_label, errors="ignore")

    def _q(v, q):
        return float(np.percentile(v, q)) if v.size else float("nan")

    def _f(x):
        return "n/a" if not np.isfinite(x) else f"{x:.0f}"

    rows = [
        ("Calling method", f"{info.get('method', hcfg.method)} (min_umi={info.get('min_umi', hcfg.min_umi)}, min_frac_of_top={info.get('min_frac_of_top', hcfg.min_frac_of_top)})"),
        ("Cells after QC", f"{n:,}"),
        ("Cells with >= 1 called guide", f"{int(assigned.sum()):,} ({100 * assigned.mean():.1f}%)"),
        ("Cells with >= 1 targeting membership", f"{int((nt > 0).sum()):,} ({100 * (nt > 0).mean():.1f}%)"),
        ("NTC-only cells", f"{int(obs[OBS_NTC_ONLY].sum()):,}"),
        (f"Cells above max_guides_per_cell={hcfg.max_guides_per_cell} (ambiguous)", f"{int(info.get('n_cells_over_cap', 0)):,}"),
        ("Guides per assigned cell (median; p10-p90)", f"{_f(_q(ng[assigned], 50))} ({_f(_q(ng[assigned], 10))}-{_f(_q(ng[assigned], 90))})"),
        ("Targets per assigned cell (median; p10-p90)", f"{_f(_q(nt[assigned], 50))} ({_f(_q(nt[assigned], 10))}-{_f(_q(nt[assigned], 90))})"),
        ("Targets with >= 1 cell", f"{int((targets_only > 0).sum()):,} of {len(targets_only):,}"),
        (f"Targets with >= {cfg.perturbation.min_cells_per_target} member cells", f"{int((targets_only >= cfg.perturbation.min_cells_per_target).sum()):,}"),
        (
            f"Targets with >= {cfg.perturbation.min_cells_per_target} primary-label cells (tested downstream in this version)",
            f"{int((obs.loc[obs[OBS_CLASS].astype(str) == CLASS_TARGETING, OBS_TARGET].astype(str).value_counts() >= cfg.perturbation.min_cells_per_target).sum()):,}",
        ),
        ("Cells per target (median; p10-p90)", f"{_f(_q(targets_only.to_numpy(), 50))} ({_f(_q(targets_only.to_numpy(), 10))}-{_f(_q(targets_only.to_numpy(), 90))})"),
    ]
    if lane_key in obs.columns:
        for lane, sub in obs.groupby(obs[lane_key].astype(str), observed=True):
            g = sub[OBS_N_GUIDES].to_numpy()
            a = g > 0
            rows.append(
                (
                    f"Lane {lane}: assigned cells; median guides/cell",
                    f"{int(a.sum()):,}/{len(sub):,} ({100 * a.mean():.1f}%); {_f(_q(g[a], 50))}",
                )
            )
    calling = pd.DataFrame(rows, columns=["metric", "value"])
    profile = expr.uns.get(UNS_RANK_PROFILE)
    profile = pd.DataFrame(profile) if profile is not None else pd.DataFrame()
    per_target = (
        counts.rename_axis(OBS_TARGET)
        .reset_index(name="n_cells")
        .sort_values(["n_cells", OBS_TARGET], ascending=[False, True])
        .reset_index(drop=True)
    )
    return {TABLE_CALLING: calling, TABLE_RANK_PROFILE: profile, TABLE_CELLS_PER_TARGET: per_target}


# ===========================================================================
# Membership-aware statistics: per-target cell sets
# ===========================================================================

PSEUDO_PREFIX = "NTC:"


class MembershipIndex:
    """Per-target cell sets derived from the membership matrices.

    ``targets`` are the targeting columns (the NTC column is excluded); every
    member cell of a target is a ``targeting`` cell by construction. The
    membership-aware control for target *t* is ``targeting_mask & ~mask(t)``:
    cells that carry at least one targeting guide but none for *t*.

    ``pseudo_targets`` are the non-targeting guides, each treated like a target
    (cells carrying that NTC guide), so the false-positive rate of any
    per-target statistic can be measured on constructs with no biological
    effect. Their names carry the ``NTC:`` prefix.
    """

    def __init__(self, expr: ad.AnnData, cfg: Config):
        T = membership_matrix(expr, cfg)
        names = membership_targets(expr)
        self.ntc_label = cfg.guides.ntc_label
        cols = [i for i, t in enumerate(names) if t != self.ntc_label]
        self.targets: List[str] = [names[i] for i in cols]
        self._pos = {t: j for j, t in enumerate(self.targets)}
        self._csc = sparse.csc_matrix(T[:, cols], dtype=np.int8)
        self.n_cells = int(T.shape[0])
        self.counts = pd.Series(np.asarray(self._csc.sum(axis=0)).ravel().astype(int), index=self.targets)
        self.targeting_mask = np.asarray(self._csc.sum(axis=1)).ravel() > 0
        self.targeting_indices = np.flatnonzero(self.targeting_mask).astype(np.int64)
        # Pseudo-targets: NTC guides -> member cell indices (any class)
        gkey = cfg.guides.high_moi.guide_membership_obsm_key
        self._pseudo: Dict[str, np.ndarray] = {}
        if gkey in expr.obsm and UNS_GUIDES in expr.uns:
            guide_ids = [str(g) for g in expr.uns[UNS_GUIDES]]
            if UNS_NTC_GUIDES in expr.uns:
                ntc_ids = {str(g) for g in expr.uns[UNS_NTC_GUIDES]}
            else:
                ntc_ids = {g for g, f in zip(guide_ids, is_non_targeting(guide_ids, cfg.guides)) if f}
            G = sparse.csc_matrix(expr.obsm[gkey])
            for j, g in enumerate(guide_ids):
                if g in ntc_ids:
                    idx = G.indices[G.indptr[j] : G.indptr[j + 1]].astype(np.int64)
                    if idx.size:
                        self._pseudo[PSEUDO_PREFIX + g] = np.sort(idx)

    # -- real targets
    def has(self, target: str) -> bool:
        return target in self._pos or target in self._pseudo

    def indices(self, target: str) -> np.ndarray:
        """Sorted cell indices carrying ``target`` (or a pseudo-target)."""
        if target in self._pseudo:
            return self._pseudo[target]
        j = self._pos.get(target)
        if j is None:
            return np.empty(0, dtype=np.int64)
        return self._csc.indices[self._csc.indptr[j] : self._csc.indptr[j + 1]].astype(np.int64)

    def mask(self, target: str) -> np.ndarray:
        out = np.zeros(self.n_cells, dtype=bool)
        out[self.indices(target)] = True
        return out

    def other_mask(self, target: str) -> np.ndarray:
        """Targeting cells not carrying ``target`` (the membership-aware 'other' control)."""
        return self.targeting_mask & ~self.mask(target)

    def other_indices(self, target: str) -> np.ndarray:
        return np.flatnonzero(self.other_mask(target)).astype(np.int64)

    # -- pseudo-targets
    @property
    def pseudo_targets(self) -> List[str]:
        return sorted(self._pseudo)

    def pseudo_counts(self) -> pd.Series:
        return pd.Series({k: int(v.size) for k, v in self._pseudo.items()}, dtype=int)

    @staticmethod
    def is_pseudo(target: str) -> bool:
        return str(target).startswith(PSEUDO_PREFIX)

    # -- aggregated counts
    def counts_by(self, values: np.ndarray, categories: Sequence[str], cell_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
        """targets x categories: number of member cells per category (``membership.T @ onehot``).

        ``cell_mask`` restricts the cells counted (e.g. to one stratum).
        """
        values = np.asarray(values).astype(str)
        cat_pos = {c: i for i, c in enumerate(categories)}
        codes = np.array([cat_pos.get(v, -1) for v in values], dtype=np.int64)
        keep = codes >= 0
        if cell_mask is not None:
            keep &= np.asarray(cell_mask, dtype=bool)
        rows = np.flatnonzero(keep)
        onehot = sparse.csr_matrix(
            (np.ones(rows.size, dtype=np.int64), (rows, codes[rows])), shape=(self.n_cells, len(categories))
        )
        counts = (self._csc.T.astype(np.int64) @ onehot).toarray()
        return pd.DataFrame(counts, index=self.targets, columns=list(categories))

    def pseudo_counts_by(
        self, values: np.ndarray, categories: Sequence[str], cell_mask: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """pseudo-targets x categories counts (same convention as :meth:`counts_by`)."""
        values = np.asarray(values).astype(str)
        rows = []
        for name in self.pseudo_targets:
            idx = self._pseudo[name]
            if cell_mask is not None:
                idx = idx[np.asarray(cell_mask, dtype=bool)[idx]]
            vc = pd.Series(values[idx]).value_counts()
            rows.append([int(vc.get(c, 0)) for c in categories])
        return pd.DataFrame(rows, index=self.pseudo_targets, columns=list(categories), dtype=np.int64)

    def guide_members(self, expr: ad.AnnData, cfg: Config, target: str) -> Dict[str, np.ndarray]:
        """{guide_id: member cell indices} for the guides of ``target`` (guide concordance)."""
        gkey = cfg.guides.high_moi.guide_membership_obsm_key
        if gkey not in expr.obsm or UNS_GUIDES not in expr.uns or UNS_GUIDE_TARGETS not in expr.uns:
            return {}
        guide_ids = [str(g) for g in expr.uns[UNS_GUIDES]]
        guide_targets = [str(t) for t in expr.uns[UNS_GUIDE_TARGETS]]
        G = sparse.csc_matrix(expr.obsm[gkey])
        out = {}
        for j, (g, t) in enumerate(zip(guide_ids, guide_targets)):
            if t == target:
                out[g] = G.indices[G.indptr[j] : G.indptr[j + 1]].astype(np.int64)
        return out

    def indicator(self, targets: Sequence[str]) -> sparse.csr_matrix:
        """targets x cells 0/1 indicator (rows in the order given; unknown targets are empty rows)."""
        blocks = []
        for t in targets:
            idx = self.indices(t)
            blocks.append(
                sparse.csr_matrix((np.ones(idx.size, dtype=np.float64), (np.zeros(idx.size, dtype=np.int64), idx)), shape=(1, self.n_cells))
            )
        if not blocks:
            return sparse.csr_matrix((0, self.n_cells), dtype=np.float64)
        return sparse.csr_matrix(sparse.vstack(blocks))


def membership_index(expr: ad.AnnData, cfg: Config) -> Optional["MembershipIndex"]:
    """The index in ``high_moi`` mode, ``None`` otherwise (the legacy stages then run unchanged)."""
    if not is_high_moi_mode(cfg):
        return None
    return MembershipIndex(expr, cfg)
