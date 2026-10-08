"""Pair-aware guide assignment for dual-guide (scaffold A + scaffold C) libraries.

Selected with ``guides.assignment_mode: pair`` (alias ``dual_guide_pair``). The
single-guide dominance rule (``assignment_mode: single_guide``) is untouched and
remains the default; this module adds a second matrix-based assignment path in
which the *pair* is the primary label.

Rule
----
For each of the two scaffold classes (default ``A`` and ``C``) the strongest
guide of that class in the cell must pass the same gate the single-guide rule
uses (``min_umi``; ``dominance_ratio`` against the runner-up *of the same
class*; optional ``max_second_umi``). Each class slot is therefore
``resolved``, ``multiple`` (several strong guides in the class) or ``none``.
The (A, C) pair is then interpreted:

* explicit pair/construct ids in the pair reference: the pair is valid only if
  both guides share a designed construct id (a guide may list several ids,
  separated by ``guides.pair_id_delimiter``) -> ``pair_targeting`` /
  ``pair_non_targeting`` / ``pair_targeting_plus_ntc`` (designed target+NTC
  construct, assigned to the target); two different targets ->
  ``dual_target_ambiguous``; every other non-designed combination ->
  ``unresolved_pair`` with the reason in ``pair_resolution_detail``
  (``same_target_not_designed`` / ``targeting_plus_ntc_not_designed`` /
  ``ntc_pair_not_designed``);
* no explicit ids (provisional same-target rule): same target ->
  ``pair_targeting``; both NTC -> ``pair_non_targeting``; targeting + NTC ->
  ``pair_targeting_plus_ntc`` (ambiguous, excluded from primary testing, unless
  ``ntc_partner_policy: provisional_target``); two different targets ->
  ``dual_target_ambiguous`` (never collapsed onto one target);
* one slot resolved, the other empty -> ``incomplete_pair``;
* a slot with several strong guides -> ``ambiguous_scaffold_A`` /
  ``ambiguous_scaffold_C`` / ``ambiguous_scaffold_A_and_C``;
* strong guides only among guides without a scaffold class -> ``unknown_guide``;
* counts present but nothing at ``min_umi`` -> ``below_min_umi``; no guide UMIs
  at all -> ``no_guide``.

The downstream contract of :mod:`perturbseq_pipeline.guides` is preserved:
``perturbation_class`` keeps its four values, ``target_gene`` carries the pair
target, ``guide_id`` carries ``"<A guide>|<C guide>"`` for assigned pairs. The
pair-level detail lives in additional ``obs`` columns (``OBS_*``). When
``guides.single_guide_diagnostic`` is on, the historical rule is evaluated on the
same matrix and stored as ``single_guide_diagnostic_*`` columns for comparison
only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config, GuideConfig
from .guides import (
    CLASS_AMBIGUOUS,
    CLASS_NTC,
    CLASS_TARGETING,
    CLASS_UNASSIGNED,
    OBS_CLASS,
    OBS_GUIDE,
    OBS_NDETECTED,
    OBS_SECOND,
    OBS_TARGET,
    OBS_TOP,
    OBS_TOTAL,
    _csr_top_two_numba,
    _csr_top_two_python,
    _log_assignment,
    is_non_targeting,
    resolve_guide_targets,
)

logger = logging.getLogger(__name__)

MODE_SINGLE = "single_guide"
MODE_DUAL = "dual_guide_pair"
MODE_PAIR = "pair"
PAIR_MODES = (MODE_DUAL, MODE_PAIR)

NTC_PARTNER_POLICIES = ("ambiguous", "provisional_target")

OBS_MODE = "guide_assignment_mode"
OBS_PAIR_ID = "pair_id"
OBS_PAIR = "pair_assignment"
OBS_PAIR_STATUS = "pair_assignment_status"
OBS_PAIR_PROVISIONAL = "pair_assignment_provisional"
OBS_PAIR_PRIMARY = "pair_assigned_primary"  # bool: cell carries a primary pair label (targeting or NTC pair)
OBS_PAIR_DETAIL = (
    "pair_resolution_detail"  # why a cell got its status (designed construct / not designed / slot problem)
)
OBS_CONSTRUCT_TYPE = "construct_type"  # dual_targeting / targeting_plus_ntc / ntc_pair / dual_target_construct / none
OBS_TARGET_SYMBOL = (
    "target_symbol"  # HGNC-style symbol of the measured transcript for the assigned target (from the pair reference)
)

CONSTRUCT_DUAL = "dual_targeting"
CONSTRUCT_SINGLE_NTC = "targeting_plus_ntc"
CONSTRUCT_NTC = "ntc_pair"
CONSTRUCT_TWO_TARGETS = "dual_target_construct"
CONSTRUCT_NONE = "none"
#: ``require_complete_pair: false`` only — one resolved scaffold slot carried the assignment.
CONSTRUCT_SINGLE_SLOT = "single_slot"

DETAIL_DESIGNED_DUAL = "designed_dual_targeting_construct"
DETAIL_DESIGNED_SINGLE_NTC = "designed_targeting_plus_ntc_construct"
DETAIL_DESIGNED_NTC = "designed_ntc_pair_construct"
DETAIL_DESIGNED_TWO_TARGETS = "designed_two_target_construct"
DETAIL_SAME_TARGET_NOT_DESIGNED = "same_target_not_designed"
DETAIL_TARGET_NTC_NOT_DESIGNED = "targeting_plus_ntc_not_designed"
DETAIL_NTC_NOT_DESIGNED = "ntc_pair_not_designed"
DETAIL_TWO_TARGETS_NOT_DESIGNED = "two_different_targets_not_designed"
DETAIL_PROVISIONAL_SAME = "provisional_same_target_rule"
DETAIL_PROVISIONAL_NTC = "provisional_ntc_pair_rule"
DETAIL_PROVISIONAL_TARGET_NTC = "provisional_targeting_plus_ntc"
DETAIL_PROVISIONAL_TWO = "two_different_targets"
DETAIL_INCOMPLETE_SINGLE_SLOT = "incomplete_pair_single_slot_accepted"

OBS_SG_CLASS = "single_guide_diagnostic_class"
OBS_SG_TARGET = "single_guide_diagnostic_target"
OBS_SG_GUIDE = "single_guide_diagnostic_guide"

OBS_SLOT_ID = "guide_{c}_id"
OBS_SLOT_TARGET = "guide_{c}_target"
OBS_SLOT_COUNT = "guide_{c}_count"
OBS_SLOT_SECOND = "guide_{c}_second_count"
OBS_SLOT_NSTRONG = "n_strong_guides_{c}"
OBS_SLOT_STATUS = "guide_{c}_slot_status"
OBS_SLOT_RATIO = "guide_{c}_dominance_ratio"  # (top + pseudocount) / (second + pseudocount)

SLOT_RESOLVED = "resolved"
SLOT_MULTIPLE = "multiple"
SLOT_NONE = "none"

STATUS_PAIR_TARGETING = "pair_targeting"
STATUS_PAIR_NTC = "pair_non_targeting"
STATUS_PAIR_TARGET_NTC = "pair_targeting_plus_ntc"
STATUS_PAIR_TARGET_NTC_PROVISIONAL = "pair_targeting_plus_ntc_provisional"
STATUS_DUAL_TARGET = "dual_target_ambiguous"
STATUS_INCOMPLETE = "incomplete_pair"
STATUS_AMBIGUOUS_SLOT = "ambiguous_scaffold_{c}"
STATUS_AMBIGUOUS_BOTH = "ambiguous_scaffold_{a}_and_{c}"
STATUS_UNKNOWN_GUIDE = "unknown_guide"
STATUS_BELOW_MIN_UMI = "below_min_umi"
STATUS_NO_GUIDE = "no_guide"
STATUS_UNRESOLVED = "unresolved_pair"

PAIR_STATUS_ORDER = [
    STATUS_PAIR_TARGETING,
    STATUS_PAIR_NTC,
    STATUS_PAIR_TARGET_NTC,
    STATUS_PAIR_TARGET_NTC_PROVISIONAL,
    STATUS_DUAL_TARGET,
    STATUS_UNRESOLVED,
    STATUS_INCOMPLETE,
    "ambiguous_scaffold_A",
    "ambiguous_scaffold_C",
    "ambiguous_scaffold_A_and_C",
    STATUS_UNKNOWN_GUIDE,
    STATUS_BELOW_MIN_UMI,
    STATUS_NO_GUIDE,
]

PAIR_SEP = "|"

_SCAFFOLD_CANDIDATES = ("scaffold", "scaffold_class", "scaffold_id", "vector_position")
_PAIR_ID_CANDIDATES = ("pair_id", "construct_id", "vector_id", "pair", "construct")
_SEQ_CANDIDATES = ("protospacer", "designed_sequence", "seq", "sequence", "spacer", "guide_sequence")
_TARGET_CANDIDATES = ("target_gene_name", "target", "target_gene", "gene", "target_symbol")
_NTC_CANDIDATES = ("is_non_targeting", "is_control", "non_targeting", "ntc")


def is_pair_mode(gcfg: GuideConfig) -> bool:
    return gcfg.assignment_mode in PAIR_MODES


# Pair reference


def _pick(columns, explicit: str, candidates, what: str, required: bool) -> Optional[str]:
    if explicit and explicit != "auto":
        if explicit in columns:
            return explicit
        raise ValueError(f"pair reference has no column {explicit!r} for {what}; columns: {list(columns)}")
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    if required:
        raise ValueError(f"could not detect the {what} column in the pair reference; columns: {list(columns)}")
    return None


def load_pair_map(path: str | Path, gcfg: GuideConfig) -> pd.DataFrame:
    """Read the pair reference (CSV/TSV) indexed by ``guide_id``.

    Returns a frame with normalised columns ``pair_id`` (``""`` = no designed
    pair), ``scaffold`` (``unknown`` if absent), ``target_gene_name`` (optional),
    ``is_non_targeting`` (optional bool) and ``designed_sequence`` (optional);
    the original columns are kept as well.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"guides.pair_map_file / pair_reference not found: {path}")
    df = pd.read_csv(
        path, sep="\t" if path.suffix.lower() in (".tsv", ".txt") else ",", dtype=str, keep_default_na=False
    )
    if "guide_id" not in df.columns:
        raise ValueError(f"pair reference {path} has no 'guide_id' column; columns: {list(df.columns)}")
    df["guide_id"] = df["guide_id"].astype(str).str.strip()
    if df["guide_id"].duplicated().any():
        raise ValueError(f"pair reference {path} lists guide ids more than once")
    df = df.set_index("guide_id")
    cols = list(df.columns)
    pcol = _pick(cols, gcfg.pair_id_column, _PAIR_ID_CANDIDATES, "pair id", required=False)
    df["pair_id"] = df[pcol].astype(str).str.strip().replace({"nan": "", "NaN": "", "None": ""}) if pcol else ""
    scol = _pick(cols, gcfg.scaffold_column, _SCAFFOLD_CANDIDATES, "scaffold", required=False)
    df["scaffold"] = df[scol].astype(str).str.strip().replace({"": "unknown", "nan": "unknown"}) if scol else "unknown"
    tcol = _pick(cols, "auto", _TARGET_CANDIDATES, "target", required=False)
    if tcol:
        df["target_gene_name"] = df[tcol].astype(str).str.strip()
    ncol = _pick(cols, "auto", _NTC_CANDIDATES, "non-targeting flag", required=False)
    if ncol:
        df["is_non_targeting"] = df[ncol].astype(str).str.lower().isin(["true", "1", "yes"])
    qcol = _pick(cols, gcfg.sequence_column, _SEQ_CANDIDATES, "designed sequence", required=False)
    if qcol:
        df["designed_sequence"] = df[qcol].astype(str).str.upper().str.strip()
    return df


def pair_map_is_explicit(pair_map: Optional[pd.DataFrame]) -> bool:
    return pair_map is not None and bool((pair_map["pair_id"] != "").any())


# Helpers


def _top_two(X: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = sparse.csr_matrix(X)
    if X.shape[1] == 0:
        n = X.shape[0]
        return np.zeros(n, dtype=np.int64), np.zeros(n), np.zeros(n), np.zeros(n)
    res = _csr_top_two_numba(X)
    if res is None:
        res = _csr_top_two_python(X)
    top_idx, top_val, second_val, total = res
    return (
        np.asarray(top_idx, dtype=np.int64),
        np.asarray(top_val, dtype=float),
        np.asarray(second_val, dtype=float),
        np.asarray(total, dtype=float),
    )


def slot_dominance_ratio(top_val, second_val, gcfg: GuideConfig) -> np.ndarray:
    """Per-slot dominance ratio ``(top + pseudocount) / (second + pseudocount)``."""
    pc = float(getattr(gcfg, "dominance_pseudocount", 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(top_val, dtype=float) + pc) / (np.asarray(second_val, dtype=float) + pc)


def _slot_gate(top_val, second_val, gcfg: GuideConfig):
    """A scaffold slot is ``resolved`` when its top guide reaches ``min_umi`` and the
    dominance ratio is at least ``dominance_ratio``; ``multiple`` when the top guide is
    strong but not dominant. Returns ``(resolved, multiple, ratio)``."""
    min_umi = max(int(gcfg.min_umi), 1)
    strong = np.asarray(top_val) >= min_umi
    ratio = slot_dominance_ratio(top_val, second_val, gcfg)
    resolved = strong & (ratio >= float(gcfg.dominance_ratio))
    if gcfg.max_second_umi is not None and gcfg.max_second_umi >= 0:
        resolved &= np.asarray(second_val) <= gcfg.max_second_umi
    return resolved, strong & ~resolved, ratio


def resolve_guide_metadata(guides: ad.AnnData, gcfg: GuideConfig, pair_map: Optional[pd.DataFrame]):
    """(targets, is_ntc, scaffold, pair_id) per guide; the pair reference is authoritative when present."""
    ids = pd.Index(guides.var_names.astype(str))
    # targets
    if pair_map is not None and "target_gene_name" in pair_map.columns and ids.isin(pair_map.index).all():
        targets = pair_map.loc[ids, "target_gene_name"].to_numpy().astype(str)
        src = "pair reference"
    else:
        targets = resolve_guide_targets(guides, gcfg)
        src = f"guides.var[{gcfg.target_feature_column!r}]" if gcfg.target_feature_column else "guide-id parsing"
    ntc = is_non_targeting(targets, gcfg)
    if pair_map is not None and "is_non_targeting" in pair_map.columns and ids.isin(pair_map.index).all():
        ntc = ntc | pair_map.loc[ids, "is_non_targeting"].to_numpy().astype(bool)
    elif "is_non_targeting" in guides.var.columns:
        ntc = ntc | guides.var["is_non_targeting"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    # scaffold
    scol = gcfg.scaffold_column if gcfg.scaffold_column != "auto" else None
    var_scol = (
        scol
        if (scol and scol in guides.var.columns)
        else next((c for c in _SCAFFOLD_CANDIDATES if c in guides.var.columns), None)
    )
    if var_scol is not None:
        scaf = (
            guides.var[var_scol]
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
            .to_numpy()
            .astype(str)
        )
    elif pair_map is not None:
        scaf = pd.Series(ids, index=ids).map(pair_map["scaffold"]).fillna("unknown").to_numpy().astype(str)
    else:
        raise ValueError(
            "pair assignment needs a scaffold class per guide: neither a scaffold column in guides.var nor a pair "
            f"reference with one is available (var columns: {list(guides.var.columns)})"
        )
    pair_id = (
        pd.Series(ids, index=ids).map(pair_map["pair_id"]).fillna("").to_numpy().astype(object)
        if pair_map is not None
        else np.array([""] * len(ids), dtype=object)
    )
    missing = ~ids.isin(pair_map.index) if pair_map is not None else np.zeros(len(ids), bool)
    if missing.any():
        logger.warning(
            "%d guide(s) in the count matrix are absent from the pair reference (treated as unknown scaffold): %s",
            int(missing.sum()),
            ids[missing][:5].tolist(),
        )
        scaf = np.where(missing, "unknown", scaf)
    logger.info("Guide targets from %s; scaffold classes: %s", src, dict(pd.Series(scaf).value_counts()))
    return targets, ntc, scaf, pair_id


_EXTRA_REF_COLUMNS = (
    "design_guide_id",
    "designed_slot",
    "feature_role",
    "construct_types",
    "construct_position",
    "target_symbol",
    "control_status",
)


def _shared_construct_ids(pair_ids_a: np.ndarray, pair_ids_c: np.ndarray, delim: str) -> np.ndarray:
    """First construct id shared by each (A guide, C guide) pair, or ``""``.

    Pair ids may list several constructs (``ACYP1_1F;ACYP1_S1``); a pair is
    designed when the two lists intersect. Combinations are cached so the loop
    runs once per distinct (A, C) feature pair rather than per cell.
    """
    out = np.full(len(pair_ids_a), "", dtype=object)
    cache: Dict[Tuple[str, str], str] = {}
    for k, (pa, pc) in enumerate(zip(pair_ids_a, pair_ids_c)):
        key = (pa, pc)
        hit = cache.get(key)
        if hit is None:
            sa = {x.strip() for x in str(pa).split(delim) if x.strip()}
            sc = {x.strip() for x in str(pc).split(delim) if x.strip()}
            inter = sorted(sa & sc)
            hit = cache[key] = inter[0] if inter else ""
        out[k] = hit
    return out


# Main entry point


def assign_guide_pairs(expr: ad.AnnData, guides: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Pair-aware assignment; writes into ``expr.obs`` and returns ``expr``."""
    gcfg = cfg.guides
    if gcfg.ntc_partner_policy not in NTC_PARTNER_POLICIES:
        raise ValueError(f"guides.ntc_partner_policy must be one of {NTC_PARTNER_POLICIES}")
    classes = [str(c) for c in gcfg.scaffold_classes]
    if len(classes) != 2:
        raise ValueError("guides.scaffold_classes must name exactly two scaffold classes")
    cA, cC = classes
    aligned = (
        guides
        if (guides.n_obs == expr.n_obs and guides.obs_names.equals(expr.obs_names))
        else guides[expr.obs_names].copy()
    )
    X = sparse.csr_matrix(aligned.layers["counts"] if "counts" in aligned.layers else aligned.X)
    n = X.shape[0]
    ref_path = gcfg.pair_map_file or gcfg.pair_reference
    pair_map = load_pair_map(ref_path, gcfg) if ref_path else None
    explicit = pair_map_is_explicit(pair_map)
    guide_ids = aligned.var_names.to_numpy().astype(str)
    guide_targets, guide_ntc, guide_scaf, guide_pair_id = resolve_guide_metadata(aligned, gcfg, pair_map)
    known = np.isin(guide_scaf, classes)
    if not known.any():
        raise ValueError(f"no guide carries a scaffold class in {classes}; values: {np.unique(guide_scaf).tolist()}")
    logger.info(
        "Pair assignment: %d guides (%s), %d without a usable scaffold class; pair reference: %s; ntc partner policy: %s; require_complete_pair=%s",
        len(guide_ids),
        ", ".join(f"{c}={int((guide_scaf == c).sum())}" for c in classes),
        int((~known).sum()),
        ("explicit pair ids (%s)" % ref_path)
        if explicit
        else ("no explicit pair ids -> provisional same-target rule" + (f" ({ref_path})" if ref_path else "")),
        gcfg.ntc_partner_policy,
        gcfg.require_complete_pair,
    )
    g_top_idx, g_top, g_second, total = _top_two(X)
    detected = np.asarray((X > gcfg.detection_threshold).sum(axis=1)).ravel().astype(np.int32)
    min_umi = max(int(gcfg.min_umi), 1)
    slot: Dict[str, Dict[str, np.ndarray]] = {}
    for c in classes:
        cols = np.flatnonzero(guide_scaf == c)
        sub = X[:, cols] if len(cols) else sparse.csr_matrix((n, 0))
        t_idx, t_val, s_val, _ = _top_two(sub)
        resolved, multiple, ratio = _slot_gate(t_val, s_val, gcfg)
        n_strong = (
            np.asarray((sub >= min_umi).sum(axis=1)).ravel().astype(np.int32)
            if len(cols)
            else np.zeros(n, dtype=np.int32)
        )
        slot[c] = dict(
            idx=cols[t_idx] if len(cols) else np.zeros(n, dtype=np.int64),
            val=t_val,
            second=s_val,
            ratio=ratio,
            resolved=resolved,
            multiple=multiple,
            none=~(resolved | multiple),
            n_strong=n_strong,
        )
    unk_cols = np.flatnonzero(~known)
    unk_strong = np.asarray((X[:, unk_cols] >= min_umi).sum(axis=1)).ravel() if len(unk_cols) else np.zeros(n)
    A, C = slot[cA], slot[cC]
    unl, amb = gcfg.unassigned_label, gcfg.ambiguous_label
    status = np.full(n, "", dtype=object)
    klass = np.full(n, CLASS_UNASSIGNED, dtype=object)
    target_call = np.full(n, unl, dtype=object)
    guide_call = np.full(n, unl, dtype=object)
    pair_call = np.full(n, unl, dtype=object)
    pair_id_call = np.full(n, "", dtype=object)
    provisional = np.zeros(n, dtype=bool)
    detail = np.full(n, "", dtype=object)
    construct_type = np.full(n, CONSTRUCT_NONE, dtype=object)

    def set_amb(mask, st):
        status[mask] = st
        klass[mask] = CLASS_AMBIGUOUS
        target_call[mask] = amb
        guide_call[mask] = amb
        pair_call[mask] = amb
        detail[mask] = st

    has_counts = total > 0
    no_strong = A["none"] & C["none"]
    status[no_strong & ~has_counts] = STATUS_NO_GUIDE
    detail[no_strong & ~has_counts] = STATUS_NO_GUIDE
    set_amb(no_strong & has_counts & (unk_strong == 0), STATUS_BELOW_MIN_UMI)
    set_amb(no_strong & has_counts & (unk_strong > 0), STATUS_UNKNOWN_GUIDE)
    both_mul = A["multiple"] & C["multiple"]
    set_amb(both_mul, STATUS_AMBIGUOUS_BOTH.format(a=cA, c=cC))
    set_amb(A["multiple"] & ~C["multiple"], STATUS_AMBIGUOUS_SLOT.format(c=cA))
    set_amb(C["multiple"] & ~A["multiple"], STATUS_AMBIGUOUS_SLOT.format(c=cC))
    incomplete = (A["resolved"] & C["none"]) | (C["resolved"] & A["none"])
    if gcfg.require_complete_pair:
        set_amb(incomplete, STATUS_INCOMPLETE)
    else:
        # require_complete_pair: false — the single resolved slot carries the
        # assignment (its target, or NTC). The status stays ``incomplete_pair`` so
        # the QC tables keep counting these cells; construct_type / detail say the
        # cell was accepted on one slot.
        for S in (A, C):
            m = incomplete & S["resolved"]
            if not m.any():
                continue
            gi = S["idx"][m]
            is_ntc = np.asarray(guide_ntc[gi], dtype=bool)
            status[m] = STATUS_INCOMPLETE
            detail[m] = DETAIL_INCOMPLETE_SINGLE_SLOT
            construct_type[m] = CONSTRUCT_SINGLE_SLOT
            guide_call[m] = guide_ids[gi]
            pair_call[m] = guide_ids[gi]
            klass[m] = np.where(is_ntc, CLASS_NTC, CLASS_TARGETING)
            target_call[m] = np.where(is_ntc, gcfg.ntc_label, guide_targets[gi])
    both = A["resolved"] & C["resolved"]
    idx = np.flatnonzero(both)
    if idx.size:
        ai, ci = A["idx"][idx], C["idx"][idx]
        tA, tC = guide_targets[ai].astype(object), guide_targets[ci].astype(object)
        nA, nC = guide_ntc[ai], guide_ntc[ci]
        gA, gC = guide_ids[ai], guide_ids[ci]
        pair_label = np.char.add(np.char.add(gA.astype(str), PAIR_SEP), gC.astype(str)).astype(object)
        target_of = np.where(nA, tC, tA).astype(object)  # the targeting partner
        two_label = np.char.add(np.char.add(tA.astype(str), PAIR_SEP), tC.astype(str)).astype(object)
        st = np.full(idx.size, "", dtype=object)
        kl = np.full(idx.size, CLASS_AMBIGUOUS, dtype=object)
        tg = np.full(idx.size, amb, dtype=object)
        pc = np.full(idx.size, amb, dtype=object)
        pid = pair_label.copy()
        prov = np.zeros(idx.size, dtype=bool)
        dt = np.full(idx.size, "", dtype=object)
        ct = np.full(idx.size, CONSTRUCT_NONE, dtype=object)
        same = tA == tC
        ntc_pair = nA & nC
        one_ntc = nA ^ nC
        two_targets = ~same & ~nA & ~nC
        if explicit:
            shared = _shared_construct_ids(guide_pair_id[ai], guide_pair_id[ci], gcfg.pair_id_delimiter or ";")
            designed = shared != ""
            pid[designed] = shared[designed]
            # designed constructs
            m = designed & ntc_pair
            st[m], kl[m], tg[m], pc[m], dt[m], ct[m] = (
                STATUS_PAIR_NTC,
                CLASS_NTC,
                gcfg.ntc_label,
                gcfg.ntc_label,
                DETAIL_DESIGNED_NTC,
                CONSTRUCT_NTC,
            )
            m = designed & one_ntc
            st[m], dt[m], ct[m], pc[m] = (
                STATUS_PAIR_TARGET_NTC,
                DETAIL_DESIGNED_SINGLE_NTC,
                CONSTRUCT_SINGLE_NTC,
                target_of[m],
            )
            if gcfg.designed_targeting_plus_ntc_primary:
                kl[m], tg[m] = (
                    CLASS_TARGETING,
                    target_of[m],
                )  # designed single-guide + NTC constructs are primary targeting labels
            # else: kept as a designed construct but class stays ambiguous -> sensitivity stratum only
            m = designed & same & ~nA
            st[m], kl[m], tg[m], pc[m], dt[m], ct[m] = (
                STATUS_PAIR_TARGETING,
                CLASS_TARGETING,
                tA[m],
                tA[m],
                DETAIL_DESIGNED_DUAL,
                CONSTRUCT_DUAL,
            )
            m = designed & two_targets
            st[m], pc[m], dt[m], ct[m] = (
                STATUS_DUAL_TARGET,
                two_label[m],
                DETAIL_DESIGNED_TWO_TARGETS,
                CONSTRUCT_TWO_TARGETS,
            )
            # combinations that are not a designed construct: never assigned
            m = ~designed & two_targets
            st[m], pc[m], dt[m] = STATUS_DUAL_TARGET, two_label[m], DETAIL_TWO_TARGETS_NOT_DESIGNED
            m = ~designed & same & ~nA
            st[m], pc[m], dt[m] = STATUS_UNRESOLVED, tA[m], DETAIL_SAME_TARGET_NOT_DESIGNED
            m = ~designed & one_ntc
            st[m], pc[m], dt[m] = STATUS_UNRESOLVED, target_of[m], DETAIL_TARGET_NTC_NOT_DESIGNED
            m = ~designed & ntc_pair
            st[m], pc[m], dt[m] = STATUS_UNRESOLVED, gcfg.ntc_label, DETAIL_NTC_NOT_DESIGNED
        else:
            prov[:] = True
            m = ntc_pair
            st[m], kl[m], tg[m], pc[m], dt[m], ct[m] = (
                STATUS_PAIR_NTC,
                CLASS_NTC,
                gcfg.ntc_label,
                gcfg.ntc_label,
                DETAIL_PROVISIONAL_NTC,
                CONSTRUCT_NTC,
            )
            m = same & ~nA
            st[m], kl[m], tg[m], pc[m], dt[m], ct[m] = (
                STATUS_PAIR_TARGETING,
                CLASS_TARGETING,
                tA[m],
                tA[m],
                DETAIL_PROVISIONAL_SAME,
                CONSTRUCT_DUAL,
            )
            m = one_ntc
            if gcfg.ntc_partner_policy == "provisional_target":
                st[m], kl[m], ct[m] = STATUS_PAIR_TARGET_NTC_PROVISIONAL, CLASS_TARGETING, CONSTRUCT_SINGLE_NTC
                tg[m] = target_of[m]
            else:
                st[m] = STATUS_PAIR_TARGET_NTC  # ambiguous: excluded from primary testing
            pc[m], dt[m] = target_of[m], DETAIL_PROVISIONAL_TARGET_NTC
            m = two_targets
            st[m], pc[m], dt[m] = STATUS_DUAL_TARGET, two_label[m], DETAIL_PROVISIONAL_TWO
        status[idx], klass[idx], target_call[idx], pair_call[idx], provisional[idx], pair_id_call[idx] = (
            st,
            kl,
            tg,
            pc,
            prov,
            pid,
        )
        detail[idx], construct_type[idx] = dt, ct
        assigned_pair = np.isin(kl, [CLASS_TARGETING, CLASS_NTC])
        guide_call[idx[assigned_pair]] = pair_label[assigned_pair]
        guide_call[idx[~assigned_pair]] = amb
    assert (status != "").all(), "every cell must receive a pair assignment status"
    assert (detail != "").all(), "every cell must receive a pair resolution detail"
    # ---- obs -------------------------------------------------------------------------
    expr.obs[OBS_TOP] = g_top
    expr.obs[OBS_SECOND] = g_second
    expr.obs[OBS_TOTAL] = total
    expr.obs[OBS_NDETECTED] = detected
    expr.obs[OBS_GUIDE] = pd.Categorical(guide_call.astype(str))
    expr.obs[OBS_TARGET] = pd.Categorical(target_call.astype(str))
    expr.obs[OBS_CLASS] = pd.Categorical(
        klass.astype(str), categories=[CLASS_TARGETING, CLASS_NTC, CLASS_AMBIGUOUS, CLASS_UNASSIGNED]
    )
    expr.obs[OBS_MODE] = pd.Categorical([MODE_PAIR] * n)
    for c in classes:
        s = slot[c]
        sid = np.where(s["resolved"], guide_ids[s["idx"]], np.where(s["multiple"], amb, unl)).astype(str)
        stg = np.where(s["resolved"], guide_targets[s["idx"]], np.where(s["multiple"], amb, unl)).astype(str)
        stg = np.where(s["resolved"] & guide_ntc[s["idx"]], gcfg.ntc_label, stg)
        expr.obs[OBS_SLOT_ID.format(c=c)] = pd.Categorical(sid)
        expr.obs[OBS_SLOT_TARGET.format(c=c)] = pd.Categorical(stg)
        expr.obs[OBS_SLOT_COUNT.format(c=c)] = s["val"]
        expr.obs[OBS_SLOT_SECOND.format(c=c)] = s["second"]
        expr.obs[OBS_SLOT_RATIO.format(c=c)] = s["ratio"]
        expr.obs[OBS_SLOT_NSTRONG.format(c=c)] = s["n_strong"]
        expr.obs[OBS_SLOT_STATUS.format(c=c)] = pd.Categorical(
            np.where(s["resolved"], SLOT_RESOLVED, np.where(s["multiple"], SLOT_MULTIPLE, SLOT_NONE)).astype(str),
            categories=[SLOT_RESOLVED, SLOT_MULTIPLE, SLOT_NONE],
        )
    expr.obs[OBS_PAIR_ID] = pd.Categorical(pair_id_call.astype(str))
    expr.obs[OBS_PAIR] = pd.Categorical(pair_call.astype(str))
    expr.obs[OBS_PAIR_STATUS] = pd.Categorical(status.astype(str))
    expr.obs[OBS_PAIR_PROVISIONAL] = provisional
    expr.obs[OBS_PAIR_PRIMARY] = np.isin(klass, [CLASS_TARGETING, CLASS_NTC])
    expr.obs[OBS_PAIR_DETAIL] = pd.Categorical(detail.astype(str))
    if pair_map is not None and "target_symbol" in pair_map.columns:
        sym_of_guide = (
            pd.Series(pd.Index(guide_ids), index=pd.Index(guide_ids))
            .map(pair_map["target_symbol"].astype(str))
            .fillna("")
            .to_numpy()
            .astype(object)
        )
        sym = np.full(n, "", dtype=object)
        tmask = (klass == CLASS_TARGETING) | (
            status == STATUS_PAIR_TARGET_NTC
        )  # incl. designed targeting+NTC constructs (sensitivity)
        if tmask.any():
            a_i, c_i = A["idx"][tmask], C["idx"][tmask]
            sym[tmask] = np.where(guide_ntc[a_i], sym_of_guide[c_i], sym_of_guide[a_i])
        sym[klass == CLASS_NTC] = gcfg.ntc_label
        expr.obs[OBS_TARGET_SYMBOL] = pd.Categorical(sym.astype(str))
    expr.obs[OBS_CONSTRUCT_TYPE] = pd.Categorical(
        construct_type.astype(str),
        categories=[
            CONSTRUCT_DUAL,
            CONSTRUCT_SINGLE_NTC,
            CONSTRUCT_NTC,
            CONSTRUCT_TWO_TARGETS,
            CONSTRUCT_SINGLE_SLOT,
            CONSTRUCT_NONE,
        ],
    )
    if gcfg.single_guide_diagnostic:
        sg_assigned = (g_top >= min_umi) & (g_top > float(gcfg.dominance_ratio) * g_second)
        if gcfg.max_second_umi is not None and gcfg.max_second_umi >= 0:
            sg_assigned &= g_second <= gcfg.max_second_umi
        sg_class = np.where(
            sg_assigned,
            np.where(guide_ntc[g_top_idx], CLASS_NTC, CLASS_TARGETING),
            np.where(has_counts, CLASS_AMBIGUOUS, CLASS_UNASSIGNED),
        ).astype(object)
        sg_target = np.where(
            sg_assigned,
            np.where(guide_ntc[g_top_idx], gcfg.ntc_label, guide_targets[g_top_idx]),
            np.where(has_counts, amb, unl),
        ).astype(object)
        sg_guide = np.where(sg_assigned, guide_ids[g_top_idx], np.where(has_counts, amb, unl)).astype(object)
        expr.obs[OBS_SG_CLASS] = pd.Categorical(
            sg_class.astype(str), categories=[CLASS_TARGETING, CLASS_NTC, CLASS_AMBIGUOUS, CLASS_UNASSIGNED]
        )
        expr.obs[OBS_SG_TARGET] = pd.Categorical(sg_target.astype(str))
        expr.obs[OBS_SG_GUIDE] = pd.Categorical(sg_guide.astype(str))
        logger.info("Single-guide diagnostic (not used downstream): %s", dict(pd.Series(sg_class).value_counts()))
    expr.uns["guide_assignment"] = {
        "mode": MODE_PAIR,
        "primary_labels": "pair",
        "scaffold_classes": classes,
        "pair_reference": str(ref_path or ""),
        "pair_reference_explicit_ids": bool(explicit),
        "ntc_partner_policy": gcfg.ntc_partner_policy,
        "require_complete_pair": bool(gcfg.require_complete_pair),
        "unresolved_pair_policy": gcfg.unresolved_pair_policy,
        "single_guide_diagnostic": bool(gcfg.single_guide_diagnostic),
        "min_umi": int(gcfg.min_umi),
        "dominance_ratio": float(gcfg.dominance_ratio),
        "max_second_umi": int(gcfg.max_second_umi),
        "pair_id_delimiter": gcfg.pair_id_delimiter,
        "dominance_pseudocount": float(getattr(gcfg, "dominance_pseudocount", 1.0)),
        "designed_targeting_plus_ntc_primary": bool(gcfg.designed_targeting_plus_ntc_primary),
        "slot_rule": f"slot passes when top_umi >= {int(gcfg.min_umi)} and (top_umi + {float(getattr(gcfg, 'dominance_pseudocount', 1.0)):g}) / (second_umi + {float(getattr(gcfg, 'dominance_pseudocount', 1.0)):g}) >= {float(gcfg.dominance_ratio):g}, evaluated independently for scaffold A and scaffold C",
        "rule": (
            "strongest guide per scaffold class must reach min_umi and exceed dominance_ratio x the class runner-up; "
            "explicit reference: valid pair = the two slot features share a designed construct id (designed dual-targeting, designed targeting+NTC and designed NTC-NTC constructs are assigned; "
            "two different targets -> dual_target_ambiguous; any other non-designed combination -> unresolved_pair, see pair_resolution_detail); "
            "provisional rule (no ids): same target -> pair_targeting, both NTC -> pair_non_targeting, targeting+NTC per ntc_partner_policy; "
            "scaffold-ambiguous, dual-target and unresolved cells are excluded from primary testing; incomplete pairs "
            "(one resolved slot) are excluded when require_complete_pair is true and assigned from the resolved slot "
            "(construct_type single_slot) when it is false"
        ),
    }
    for obj in {id(aligned): aligned, id(guides): guides}.values():
        obj.var["target_gene"] = guide_targets
        obj.var["is_non_targeting"] = guide_ntc
        obj.var["scaffold"] = guide_scaf
        obj.var["pair_id"] = guide_pair_id.astype(str)
        if pair_map is not None:
            ids_all = pd.Index(obj.var_names.astype(str))
            for col in _EXTRA_REF_COLUMNS:
                if col in pair_map.columns:
                    obj.var[col] = (
                        pd.Series(ids_all, index=ids_all)
                        .map(pair_map[col].astype(str))
                        .fillna("")
                        .to_numpy()
                        .astype(str)
                    )
    aligned.obs[OBS_GUIDE] = expr.obs[OBS_GUIDE].to_numpy()
    aligned.obs[OBS_TARGET] = expr.obs[OBS_TARGET].to_numpy()
    counts = pd.Series(status).value_counts()
    logger.info("Pair assignment status: %s", ", ".join(f"{k}={int(v)}" for k, v in counts.items()))
    _log_assignment(expr, cfg)
    return expr


# Summaries


def pair_assignment_summary(expr: ad.AnnData) -> Optional[pd.DataFrame]:
    if OBS_PAIR_STATUS not in expr.obs.columns:
        return None
    df = (
        expr.obs.groupby([OBS_PAIR_STATUS, OBS_CLASS, OBS_PAIR_PROVISIONAL], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    df["pct_of_cells"] = 100.0 * df["n_cells"] / max(expr.n_obs, 1)
    order = {s: i for i, s in enumerate(PAIR_STATUS_ORDER)}
    df["_o"] = df[OBS_PAIR_STATUS].astype(str).map(lambda s: order.get(s, len(order)))
    return df.sort_values(["_o", OBS_CLASS]).drop(columns="_o").reset_index(drop=True)


def pair_assignment_per_lane(expr: ad.AnnData, lane_key: str = "lane_id") -> Optional[pd.DataFrame]:
    if OBS_PAIR_STATUS not in expr.obs.columns or lane_key not in expr.obs.columns:
        return None
    tab = pd.crosstab(expr.obs[lane_key].astype(str), expr.obs[OBS_PAIR_STATUS].astype(str))
    cols = [c for c in PAIR_STATUS_ORDER if c in tab.columns] + [c for c in tab.columns if c not in PAIR_STATUS_ORDER]
    tab = tab[cols]
    if len(tab) > 1:
        tab.loc["ALL"] = tab.sum(axis=0)
    tab.insert(0, "n_cells", tab.sum(axis=1))
    if OBS_PAIR_PRIMARY in expr.obs.columns:
        prim = expr.obs.groupby(expr.obs[lane_key].astype(str), observed=True)[OBS_PAIR_PRIMARY].sum().astype(int)
        if len(tab) > 1:
            prim.loc["ALL"] = int(prim.sum())
        tab.insert(1, "n_pair_assigned_primary", prim.reindex(tab.index).fillna(0).astype(int).to_numpy())
    else:
        tab.insert(
            1,
            "n_pair_assigned_primary",
            tab.get(STATUS_PAIR_TARGETING, 0)
            + tab.get(STATUS_PAIR_NTC, 0)
            + tab.get(STATUS_PAIR_TARGET_NTC_PROVISIONAL, 0),
        )
    tab.insert(2, "frac_strict_primary_pair", tab["n_pair_assigned_primary"] / tab["n_cells"])
    tab.insert(
        2,
        "frac_complete_pair",
        (
            tab.get(STATUS_PAIR_TARGETING, 0)
            + tab.get(STATUS_PAIR_NTC, 0)
            + tab.get(STATUS_PAIR_TARGET_NTC, 0)
            + tab.get(STATUS_PAIR_TARGET_NTC_PROVISIONAL, 0)
            + tab.get(STATUS_DUAL_TARGET, 0)
            + tab.get(STATUS_UNRESOLVED, 0)
        )
        / tab["n_cells"],
    )
    tab.insert(3, "frac_incomplete_pair", tab.get(STATUS_INCOMPLETE, 0) / tab["n_cells"])
    return tab.reset_index().rename(columns={lane_key: "lane_id"})


def pair_resolution_detail_table(expr: ad.AnnData, lane_key: str = "lane_id") -> Optional[pd.DataFrame]:
    """Cells per (pair status, resolution detail, construct type, class), per lane and overall."""
    if OBS_PAIR_DETAIL not in expr.obs.columns:
        return None
    keys = [OBS_PAIR_STATUS, OBS_PAIR_DETAIL, OBS_CONSTRUCT_TYPE, OBS_CLASS]
    df = expr.obs.groupby(keys, observed=True).size().reset_index(name="n_cells_all")
    if lane_key in expr.obs.columns:
        per = pd.crosstab([expr.obs[k].astype(str) for k in keys], expr.obs[lane_key].astype(str)).reset_index()
        per.columns = keys + [f"n_cells_{c}" for c in per.columns[len(keys) :]]
        for k in keys:
            df[k] = df[k].astype(str)
        df = df.merge(per, on=keys, how="left")
    df["pct_of_cells"] = 100.0 * df["n_cells_all"] / max(expr.n_obs, 1)
    order = {s: i for i, s in enumerate(PAIR_STATUS_ORDER)}
    df["_o"] = df[OBS_PAIR_STATUS].astype(str).map(lambda s: order.get(s, len(order)))
    df["enters_primary_testing"] = df[OBS_CLASS].astype(str).isin([CLASS_TARGETING, CLASS_NTC])
    return df.sort_values(["_o", "n_cells_all"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def single_guide_diagnostic_table(expr: ad.AnnData) -> Optional[pd.DataFrame]:
    """Cross-tabulate the diagnostic single-guide class against the pair status (diagnostic only)."""
    if OBS_SG_CLASS not in expr.obs.columns:
        return None
    ct = pd.crosstab(expr.obs[OBS_SG_CLASS].astype(str), expr.obs[OBS_PAIR_STATUS].astype(str))
    cols = [c for c in PAIR_STATUS_ORDER if c in ct.columns]
    ct = ct[cols]
    ct.index.name = "single_guide_diagnostic_class"
    out = ct.reset_index()
    out.insert(1, "n_cells", ct.sum(axis=1).to_numpy())
    same_t = (expr.obs[OBS_SG_CLASS].astype(str) == CLASS_TARGETING) & (
        expr.obs[OBS_CLASS].astype(str) == CLASS_TARGETING
    )
    agree = int((same_t & (expr.obs[OBS_SG_TARGET].astype(str) == expr.obs[OBS_TARGET].astype(str))).sum())
    out.attrs["targeting_in_both_same_target"] = agree
    out.attrs["targeting_in_both_different_target"] = int(same_t.sum()) - agree
    return out
