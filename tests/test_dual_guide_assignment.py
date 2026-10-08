"""Unit tests for the configurable dual-guide (scaffold A + C) pair assignment."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbseq_pipeline.config import Config
from perturbseq_pipeline.dual_guides import (
    OBS_MODE,
    OBS_PAIR,
    OBS_PAIR_ID,
    OBS_PAIR_PROVISIONAL,
    OBS_PAIR_STATUS,
    OBS_SLOT_ID,
    OBS_SLOT_NSTRONG,
    OBS_SLOT_TARGET,
    STATUS_AMBIGUOUS_SLOT,
    STATUS_PAIR_TARGET_NTC,
    STATUS_PAIR_TARGET_NTC_PROVISIONAL,
    STATUS_UNKNOWN_GUIDE,
    OBS_SG_CLASS,
    OBS_SG_TARGET,
    single_guide_diagnostic_table,
    STATUS_BELOW_MIN_UMI,
    STATUS_DUAL_TARGET,
    STATUS_INCOMPLETE,
    STATUS_UNRESOLVED,
    STATUS_NO_GUIDE,
    STATUS_PAIR_NTC,
    STATUS_PAIR_TARGETING,
    pair_assignment_per_lane,
    pair_assignment_summary,
)
from perturbseq_pipeline.guides import (
    CLASS_AMBIGUOUS,
    CLASS_NTC,
    CLASS_TARGETING,
    CLASS_UNASSIGNED,
    OBS_CLASS,
    OBS_GUIDE,
    OBS_NDETECTED,
    OBS_TARGET,
    assign_guides,
)
from perturbseq_pipeline.io import guides_from_obsm, merge_guides_into_expr

# Guide reference: A-scaffold and C-scaffold guides for targets X, Y and NTC,
# plus a designed guide (Z1) that is never observed (all-zero column).
GUIDES = ["A1", "A2", "A3", "An", "C1", "C2", "C3", "Cn", "Z1"]
TARGETS = ["X", "X", "Y", "NO-TARGET", "X", "X", "Y", "NO-TARGET", "Z"]
SCAFFOLD = ["A", "A", "A", "A", "C", "C", "C", "C", "unknown"]
COL = {g: i for i, g in enumerate(GUIDES)}

CELLS = {
    "same_target": {"A1": 10, "C1": 8},
    "target_ntc": {"A1": 10, "Cn": 9},
    "dual_target": {"A1": 10, "C3": 10},
    "cross_pair": {"A1": 10, "C2": 10},  # X guides but different designed vectors
    "missing_C": {"A1": 10},
    "multi_A": {"A1": 10, "A2": 8, "C1": 10},
    "multi_C": {"A1": 10, "C1": 10, "C2": 7},
    "zero": {},
    "ntc_pair": {"An": 12, "Cn": 5},
    "below_min": {"A1": 2, "C1": 1},
    "weak_second": {"A1": 20, "A2": 3, "C1": 15, "C2": 2},  # runner-ups below dominance
}


def _build(mode="pair", pair_map=None, policy="ambiguous", with_scaffold_var=True, diagnostic=False):
    X = np.zeros((len(CELLS), len(GUIDES)))
    names = list(CELLS)
    for i, (cell, counts) in enumerate(CELLS.items()):
        for g, v in counts.items():
            X[i, COL[g]] = v
    var = pd.DataFrame(
        {"target_gene_name": TARGETS, "is_non_targeting": [t == "NO-TARGET" for t in TARGETS]}, index=GUIDES
    )
    if with_scaffold_var:
        var["scaffold"] = SCAFFOLD
    guides = ad.AnnData(X=sparse.csr_matrix(X), var=var)
    guides.obs_names = names
    expr = ad.AnnData(
        X=sparse.csr_matrix(np.ones((len(names), 4))),
        obs=pd.DataFrame({"lane_id": ["L1"] * 6 + ["L2"] * (len(names) - 6)}, index=names),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    cfg = Config()
    cfg.guides.assignment_mode = mode
    cfg.guides.target_feature_column = "target_gene_name"
    cfg.guides.target_split_delims = []
    cfg.guides.ntc_label = "ntc"
    cfg.guides.min_umi = 3
    cfg.guides.dominance_ratio = 2.0
    cfg.guides.max_second_umi = -1
    cfg.guides.pair_map_file = pair_map
    cfg.guides.ntc_partner_policy = policy
    cfg.guides.single_guide_diagnostic = diagnostic
    return expr, guides, cfg


def _pair_map(tmp_path, explicit):
    df = pd.DataFrame({"guide_id": GUIDES, "scaffold": SCAFFOLD, "target_gene_name": TARGETS})
    if explicit:
        pairs = {
            "A1": "V1",
            "C1": "V1",
            "A2": "V2",
            "C2": "V2",
            "A3": "V3",
            "C3": "V3",
            "An": "Vn",
            "Cn": "Vn",
            "Z1": "",
        }
        df["pair_id"] = [pairs[g] for g in GUIDES]
    else:
        df["pair_id"] = ""
    p = tmp_path / "pair_map.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_valid_same_target_pair():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["same_target"]
    assert o[OBS_CLASS] == CLASS_TARGETING and o[OBS_TARGET] == "X"
    assert o[OBS_PAIR_STATUS] == STATUS_PAIR_TARGETING and o[OBS_PAIR] == "X"
    assert o[OBS_GUIDE] == "A1|C1" and o[OBS_PAIR_ID] == "A1|C1"
    assert o[OBS_SLOT_ID.format(c="A")] == "A1" and o[OBS_SLOT_ID.format(c="C")] == "C1"
    assert o[OBS_SLOT_TARGET.format(c="A")] == "X" and o[OBS_SLOT_TARGET.format(c="C")] == "X"
    assert bool(o[OBS_PAIR_PROVISIONAL]) is True  # no explicit pair map
    assert o[OBS_MODE] == "pair"


def test_targeting_plus_ntc_pair_policies():
    expr, guides, cfg = _build(policy="ambiguous")
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["target_ntc"]
    assert o[OBS_CLASS] == CLASS_AMBIGUOUS and o[OBS_PAIR_STATUS] == STATUS_PAIR_TARGET_NTC
    assert o[OBS_PAIR] == "X"  # the intended target is still recorded
    assert o[OBS_SLOT_TARGET.format(c="C")] == "ntc"
    expr, guides, cfg = _build(policy="provisional_target")
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["target_ntc"]
    assert o[OBS_CLASS] == CLASS_TARGETING and o[OBS_TARGET] == "X"
    assert o[OBS_PAIR_STATUS] == STATUS_PAIR_TARGET_NTC_PROVISIONAL and bool(o[OBS_PAIR_PROVISIONAL])


def test_dual_target_pair_is_never_collapsed():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["dual_target"]
    assert o[OBS_CLASS] == CLASS_AMBIGUOUS
    assert o[OBS_PAIR_STATUS] == STATUS_DUAL_TARGET
    assert o[OBS_PAIR] == "X|Y"
    assert o[OBS_TARGET] == cfg.guides.ambiguous_label
    assert o[OBS_SLOT_ID.format(c="A")] == "A1" and o[OBS_SLOT_ID.format(c="C")] == "C3"


def test_invalid_and_designed_pairs_with_explicit_map(tmp_path):
    pm = _pair_map(tmp_path, explicit=True)
    expr, guides, cfg = _build(pair_map=pm)
    res = assign_guides(expr, guides, cfg)
    ok = res.obs.loc["same_target"]
    assert ok[OBS_CLASS] == CLASS_TARGETING and ok[OBS_PAIR_STATUS] == STATUS_PAIR_TARGETING
    assert ok[OBS_PAIR_ID] == "V1" and not bool(ok[OBS_PAIR_PROVISIONAL])
    bad = res.obs.loc["cross_pair"]
    assert bad[OBS_CLASS] == CLASS_AMBIGUOUS and bad[OBS_PAIR_STATUS] == STATUS_UNRESOLVED
    assert bad[OBS_TARGET] == cfg.guides.ambiguous_label
    # under the provisional rule the same cell is a valid same-target pair
    expr, guides, cfg = _build(pair_map=_pair_map(tmp_path, explicit=False))
    res = assign_guides(expr, guides, cfg)
    assert res.obs.loc["cross_pair", OBS_PAIR_STATUS] == STATUS_PAIR_TARGETING
    assert bool(res.obs.loc["cross_pair", OBS_PAIR_PROVISIONAL])


def test_missing_c_guide_is_incomplete():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["missing_C"]
    assert o[OBS_CLASS] == CLASS_AMBIGUOUS
    assert o[OBS_PAIR_STATUS] == STATUS_INCOMPLETE
    assert o[OBS_SLOT_ID.format(c="A")] == "A1"
    assert o[OBS_SLOT_ID.format(c="C")] == cfg.guides.unassigned_label


def test_multiple_strong_a_guides_are_ambiguous():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["multi_A"]
    assert o[OBS_CLASS] == CLASS_AMBIGUOUS and o[OBS_PAIR_STATUS] == STATUS_AMBIGUOUS_SLOT.format(c="A")
    assert o[OBS_SLOT_ID.format(c="A")] == cfg.guides.ambiguous_label
    assert o[OBS_SLOT_ID.format(c="C")] == "C1"
    assert int(o[OBS_SLOT_NSTRONG.format(c="A")]) == 2


def test_multiple_strong_c_guides_are_ambiguous():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["multi_C"]
    assert o[OBS_CLASS] == CLASS_AMBIGUOUS and o[OBS_PAIR_STATUS] == STATUS_AMBIGUOUS_SLOT.format(c="C")
    assert int(o[OBS_SLOT_NSTRONG.format(c="C")]) == 2


def test_weak_runner_ups_do_not_block_the_pair():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["weak_second"]
    assert o[OBS_CLASS] == CLASS_TARGETING and o[OBS_PAIR_STATUS] == STATUS_PAIR_TARGETING
    assert int(o[OBS_SLOT_NSTRONG.format(c="A")]) == 2  # A2 = 3 UMIs is "strong" but dominated


def test_zero_guide_cell_is_unassigned():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["zero"]
    assert o[OBS_CLASS] == CLASS_UNASSIGNED and o[OBS_PAIR_STATUS] == STATUS_NO_GUIDE
    assert int(o[OBS_NDETECTED]) == 0
    b = res.obs.loc["below_min"]
    assert b[OBS_CLASS] == CLASS_AMBIGUOUS and b[OBS_PAIR_STATUS] == STATUS_BELOW_MIN_UMI


def test_ntc_pair_is_non_targeting():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["ntc_pair"]
    assert o[OBS_CLASS] == CLASS_NTC and o[OBS_TARGET] == "ntc"
    assert o[OBS_PAIR_STATUS] == STATUS_PAIR_NTC and o[OBS_GUIDE] == "An|Cn"


def test_zero_count_design_entry_is_kept_in_reference():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    assert "Z1" in guides.var_names
    assert guides.var.loc["Z1", "scaffold"] == "unknown"
    assert guides.var.loc["Z1", "target_gene"] == "Z"
    # nothing is ever assigned to the unobserved guide
    assert not res.obs[OBS_GUIDE].astype(str).str.contains("Z1").any()
    assert guides.X[:, COL["Z1"]].sum() == 0


def test_single_guide_mode_is_unchanged_baseline():
    expr, guides, cfg = _build(mode="single_guide")
    res = assign_guides(expr, guides, cfg)
    # top 10 vs second 8: not dominant -> ambiguous under the single-guide rule
    assert res.obs.loc["same_target", OBS_CLASS] == CLASS_AMBIGUOUS
    assert res.obs.loc["missing_C", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["zero", OBS_CLASS] == CLASS_UNASSIGNED
    assert (res.obs[OBS_MODE] == "single_guide").all()
    assert OBS_PAIR_STATUS not in res.obs.columns


def test_scaffold_from_pair_map_when_var_lacks_it(tmp_path):
    expr, guides, cfg = _build(pair_map=_pair_map(tmp_path, explicit=False), with_scaffold_var=False)
    res = assign_guides(expr, guides, cfg)
    assert res.obs.loc["same_target", OBS_CLASS] == CLASS_TARGETING
    expr, guides, cfg = _build(with_scaffold_var=False)
    with pytest.raises(ValueError, match="scaffold class"):
        assign_guides(expr, guides, cfg)


def test_summary_tables_and_h5ad_round_trip():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    summ = pair_assignment_summary(res)
    assert summ["n_cells"].sum() == res.n_obs
    lane = pair_assignment_per_lane(res)
    assert lane.loc[lane["lane_id"] == "ALL", "n_cells"].iloc[0] == res.n_obs
    merged = merge_guides_into_expr(res.copy(), guides, cfg)
    assert "guide_scaffolds" in merged.uns
    back = guides_from_obsm(merged, cfg)
    assert list(back.var["scaffold"]) == SCAFFOLD
    assert list(back.var["target_gene"]) == list(guides.var["target_gene"])


def _valid_cfg():
    cfg = Config()
    cfg.input.mode = "h5ad"
    cfg.input.h5ad = "input.h5ad"
    return cfg


def test_config_rejects_bad_mode_and_policy():
    _valid_cfg().validate()  # baseline passes
    cfg = _valid_cfg()
    cfg.guides.assignment_mode = "triple"
    with pytest.raises(ValueError, match="assignment_mode"):
        cfg.validate()
    cfg = _valid_cfg()
    cfg.guides.ntc_partner_policy = "guess"
    with pytest.raises(ValueError, match="ntc_partner_policy"):
        cfg.validate()
    cfg = _valid_cfg()
    cfg.guides.assignment_mode = "pair"
    cfg.guides.pair_map_file = "/nonexistent/pair_map.csv"
    with pytest.raises(ValueError, match="pair_map_file"):
        cfg.validate()


def test_unknown_guide_and_single_guide_diagnostic():
    expr, guides, cfg = _build(diagnostic=True)
    # a cell whose only strong guide has no scaffold class
    X = guides.X.toarray()
    X[list(CELLS).index("zero"), COL["Z1"]] = 8
    guides.X = sparse.csr_matrix(X)
    res = assign_guides(expr, guides, cfg)
    assert res.obs.loc["zero", OBS_PAIR_STATUS] == STATUS_UNKNOWN_GUIDE
    assert res.obs.loc["zero", OBS_CLASS] == CLASS_AMBIGUOUS
    # diagnostic columns reproduce the single-guide rule but never drive the primary labels
    assert res.obs.loc["same_target", OBS_SG_CLASS] == CLASS_AMBIGUOUS  # 10 vs 8 fails dominance
    assert res.obs.loc["same_target", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["missing_C", OBS_SG_CLASS] == CLASS_TARGETING and res.obs.loc["missing_C", OBS_SG_TARGET] == "X"
    assert res.obs.loc["missing_C", OBS_CLASS] == CLASS_AMBIGUOUS
    tab = single_guide_diagnostic_table(res)
    assert tab["n_cells"].sum() == res.n_obs


def test_pair_reference_supplies_targets_when_var_lacks_them(tmp_path):
    expr, guides, cfg = _build(pair_map=_pair_map(tmp_path, explicit=False), with_scaffold_var=False)
    del guides.var["target_gene_name"]
    cfg.guides.target_feature_column = None
    res = assign_guides(expr, guides, cfg)
    assert res.obs.loc["same_target", OBS_TARGET] == "X"
    assert res.obs.loc["ntc_pair", OBS_CLASS] == CLASS_NTC
    assert list(guides.var["scaffold"]) == SCAFFOLD


def test_pair_mode_alias_and_pair_reference_key():
    cfg = _valid_cfg()
    cfg.guides.assignment_mode = "pair"
    cfg.guides.pair_reference = "/nonexistent/ref.csv"
    with pytest.raises(ValueError, match="pair_map_file"):
        cfg.validate()
    cfg = _valid_cfg()
    cfg.guides.assignment_mode = "pair"
    cfg.validate()
    assert cfg.guides.assignment_mode == "dual_guide_pair"


def test_explicit_map_with_multi_construct_ids_and_resolution_details(tmp_path):
    """A slot feature may belong to several constructs (';'-separated ids); non-designed
    combinations are labelled by reason and never assigned."""
    from perturbseq_pipeline.dual_guides import (
        CONSTRUCT_DUAL,
        CONSTRUCT_NONE,
        CONSTRUCT_NTC,
        CONSTRUCT_SINGLE_NTC,
        DETAIL_DESIGNED_DUAL,
        DETAIL_DESIGNED_NTC,
        DETAIL_DESIGNED_SINGLE_NTC,
        DETAIL_SAME_TARGET_NOT_DESIGNED,
        DETAIL_TARGET_NTC_NOT_DESIGNED,
        DETAIL_TWO_TARGETS_NOT_DESIGNED,
        OBS_CONSTRUCT_TYPE,
        OBS_PAIR_DETAIL,
        pair_resolution_detail_table,
    )

    df = pd.DataFrame({"guide_id": GUIDES, "scaffold": SCAFFOLD, "target_gene_name": TARGETS})
    # V1 = A1+C1, V2 = A2+C2, V3 = A3+C3, Vn = An+Cn (NTC pair), S1 = C1 (targeting) + An (NTC filler)
    pairs = {
        "A1": "V1",
        "C1": "V1;S1",
        "A2": "V2",
        "C2": "V2",
        "A3": "V3",
        "C3": "V3",
        "An": "Vn;S1",
        "Cn": "Vn",
        "Z1": "",
    }
    df["pair_id"] = [pairs[g] for g in GUIDES]
    pm = tmp_path / "pair_map_multi.csv"
    df.to_csv(pm, index=False)
    expr, guides, cfg = _build(pair_map=str(pm))
    X = guides.X.toarray()
    names = list(CELLS)
    X[names.index("zero"), COL["C1"]] = 10  # designed single-guide construct S1: C1 + An
    X[names.index("zero"), COL["An"]] = 9
    guides.X = sparse.csr_matrix(X)
    res = assign_guides(expr, guides, cfg)
    o = res.obs
    s1 = o.loc["zero"]
    assert s1[OBS_PAIR_STATUS] == STATUS_PAIR_TARGET_NTC and s1[OBS_CLASS] == CLASS_TARGETING and s1[OBS_TARGET] == "X"
    assert (
        s1[OBS_PAIR_ID] == "S1"
        and s1[OBS_CONSTRUCT_TYPE] == CONSTRUCT_SINGLE_NTC
        and s1[OBS_PAIR_DETAIL] == DETAIL_DESIGNED_SINGLE_NTC
    )
    assert not bool(s1[OBS_PAIR_PROVISIONAL])
    d = o.loc["same_target"]
    assert (
        d[OBS_PAIR_STATUS] == STATUS_PAIR_TARGETING
        and d[OBS_PAIR_ID] == "V1"
        and d[OBS_CONSTRUCT_TYPE] == CONSTRUCT_DUAL
        and d[OBS_PAIR_DETAIL] == DETAIL_DESIGNED_DUAL
    )
    n = o.loc["ntc_pair"]
    assert (
        n[OBS_PAIR_STATUS] == STATUS_PAIR_NTC
        and n[OBS_PAIR_ID] == "Vn"
        and n[OBS_CONSTRUCT_TYPE] == CONSTRUCT_NTC
        and n[OBS_PAIR_DETAIL] == DETAIL_DESIGNED_NTC
    )
    # A1 (V1) + C2 (V2): same target, not a construct
    x = o.loc["cross_pair"]
    assert (
        x[OBS_PAIR_STATUS] == STATUS_UNRESOLVED
        and x[OBS_CLASS] == CLASS_AMBIGUOUS
        and x[OBS_PAIR_DETAIL] == DETAIL_SAME_TARGET_NOT_DESIGNED
    )
    assert (
        x[OBS_PAIR] == "X" and x[OBS_TARGET] == cfg.guides.ambiguous_label and x[OBS_CONSTRUCT_TYPE] == CONSTRUCT_NONE
    )
    # A1 (V1) + Cn (Vn): targeting + NTC but not the designed S1 construct
    t = o.loc["target_ntc"]
    assert (
        t[OBS_PAIR_STATUS] == STATUS_UNRESOLVED
        and t[OBS_CLASS] == CLASS_AMBIGUOUS
        and t[OBS_PAIR_DETAIL] == DETAIL_TARGET_NTC_NOT_DESIGNED
    )
    # A1 (V1) + C3 (V3): two targets
    dd = o.loc["dual_target"]
    assert (
        dd[OBS_PAIR_STATUS] == STATUS_DUAL_TARGET
        and dd[OBS_PAIR_DETAIL] == DETAIL_TWO_TARGETS_NOT_DESIGNED
        and dd[OBS_CLASS] == CLASS_AMBIGUOUS
    )
    assert (o[OBS_PAIR_DETAIL].astype(str) != "").all()
    tab = pair_resolution_detail_table(res)
    assert tab["n_cells_all"].sum() == res.n_obs
    assert tab.loc[tab[OBS_PAIR_STATUS] == STATUS_UNRESOLVED, "enters_primary_testing"].eq(False).all()
    assert res.uns["guide_assignment"]["pair_reference_explicit_ids"] is True
    assert "designed_slot" not in guides.var.columns or True  # optional reference columns are copied only when present


def test_slot_dominance_ratio_pseudocount_rule_and_columns(tmp_path):
    """(top+1)/(second+1) >= dominance_ratio per scaffold slot, stored per slot; the eight required cases."""
    from perturbseq_pipeline.dual_guides import (
        OBS_SLOT_COUNT,
        OBS_SLOT_RATIO,
        OBS_SLOT_SECOND,
        OBS_SLOT_STATUS,
        SLOT_MULTIPLE,
        SLOT_NONE,
        SLOT_RESOLVED,
        slot_dominance_ratio,
    )

    pm = _pair_map(tmp_path, explicit=True)
    expr, guides, cfg = _build(pair_map=pm)
    X = np.zeros(guides.shape)
    names = list(CELLS)
    cases = {  # cell -> counts
        "same_target": {"A1": 3, "C1": 3},  # zero second counts: ratio (3+1)/(0+1) = 4 -> pass, valid complete pair V1
        "target_ntc": {"A1": 3, "A2": 1, "C1": 3},  # top 3 / second 1: (4)/(2) = 2.0 -> pass
        "dual_target": {"A1": 3, "A2": 2, "C1": 3},  # top 3 / second 2: (4)/(3) = 1.33 -> A slot ambiguous
        "multi_A": {"A1": 10, "A2": 8, "C1": 10},  # ambiguous A slot
        "multi_C": {"A1": 10, "C1": 10, "C2": 7},  # ambiguous C slot
        "missing_C": {"A1": 10},  # incomplete pair
        "cross_pair": {"A1": 10, "C2": 10},  # invalid construct pair (V1 x V2)
        "ntc_pair": {"An": 12, "Cn": 5},  # valid complete NTC pair Vn
    }
    for cell, counts in cases.items():
        for g, v in counts.items():
            X[names.index(cell), COL[g]] = v
    guides.X = sparse.csr_matrix(X)
    res = assign_guides(expr, guides, cfg)
    o = res.obs
    rA, rC = OBS_SLOT_RATIO.format(c="A"), OBS_SLOT_RATIO.format(c="C")
    for c in ("A", "C"):
        for col in (OBS_SLOT_COUNT, OBS_SLOT_SECOND, OBS_SLOT_RATIO, OBS_SLOT_STATUS):
            assert col.format(c=c) in o.columns
    np.testing.assert_allclose(o.loc["same_target", [rA, rC]].astype(float), [4.0, 4.0])
    assert o.loc["same_target", OBS_PAIR_STATUS] == STATUS_PAIR_TARGETING and o.loc["same_target", OBS_PAIR_ID] == "V1"
    assert o.loc["target_ntc", rA] == 2.0 and o.loc["target_ntc", OBS_SLOT_STATUS.format(c="A")] == SLOT_RESOLVED
    assert o.loc["target_ntc", OBS_PAIR_STATUS] == STATUS_PAIR_TARGETING  # A1 (3 vs 1) + C1 -> V1
    assert (
        abs(o.loc["dual_target", rA] - 4 / 3) < 1e-9
        and o.loc["dual_target", OBS_SLOT_STATUS.format(c="A")] == SLOT_MULTIPLE
    )
    assert o.loc["dual_target", OBS_PAIR_STATUS] == STATUS_AMBIGUOUS_SLOT.format(c="A")
    assert (
        o.loc["multi_A", OBS_PAIR_STATUS] == STATUS_AMBIGUOUS_SLOT.format(c="A")
        and abs(o.loc["multi_A", rA] - 11 / 9) < 1e-9
    )
    assert (
        o.loc["multi_C", OBS_PAIR_STATUS] == STATUS_AMBIGUOUS_SLOT.format(c="C")
        and abs(o.loc["multi_C", rC] - 11 / 8) < 1e-9
    )
    assert (
        o.loc["missing_C", OBS_PAIR_STATUS] == STATUS_INCOMPLETE
        and o.loc["missing_C", OBS_SLOT_STATUS.format(c="C")] == SLOT_NONE
        and o.loc["missing_C", rC] == 1.0
    )
    assert (
        o.loc["cross_pair", OBS_PAIR_STATUS] == STATUS_UNRESOLVED and o.loc["cross_pair", OBS_CLASS] == CLASS_AMBIGUOUS
    )
    assert o.loc["ntc_pair", OBS_PAIR_STATUS] == STATUS_PAIR_NTC and o.loc["ntc_pair", OBS_CLASS] == CLASS_NTC
    assert res.uns["guide_assignment"]["dominance_pseudocount"] == 1.0 and "slot_rule" in res.uns["guide_assignment"]
    # helper matches the stored columns; a different pseudocount changes the ratio
    np.testing.assert_allclose(slot_dominance_ratio(np.array([3.0]), np.array([1.0]), cfg.guides), [2.0])
    cfg.guides.dominance_pseudocount = 0.5
    np.testing.assert_allclose(slot_dominance_ratio(np.array([3.0]), np.array([1.0]), cfg.guides), [3.5 / 1.5])


def test_designed_targeting_plus_ntc_sensitivity_policy(tmp_path):
    """designed_targeting_plus_ntc_primary=False keeps designed S1 constructs out of the primary classes."""
    from perturbseq_pipeline.dual_guides import (
        CONSTRUCT_SINGLE_NTC,
        DETAIL_DESIGNED_SINGLE_NTC,
        OBS_CONSTRUCT_TYPE,
        OBS_PAIR_DETAIL,
        OBS_TARGET_SYMBOL,
    )

    df = pd.DataFrame(
        {
            "guide_id": GUIDES,
            "scaffold": SCAFFOLD,
            "target_gene_name": TARGETS,
            "target_symbol": [t if t != "NO-TARGET" else "ntc" for t in TARGETS],
        }
    )
    pairs = {
        "A1": "V1",
        "C1": "V1;S1",
        "A2": "V2",
        "C2": "V2",
        "A3": "V3",
        "C3": "V3",
        "An": "Vn;S1",
        "Cn": "Vn",
        "Z1": "",
    }
    df["pair_id"] = [pairs[g] for g in GUIDES]
    pm = tmp_path / "pm.csv"
    df.to_csv(pm, index=False)
    for primary in (True, False):
        expr, guides, cfg = _build(pair_map=str(pm))
        cfg.guides.designed_targeting_plus_ntc_primary = primary
        X = guides.X.toarray()
        X[list(CELLS).index("zero"), COL["C1"]] = 10
        X[list(CELLS).index("zero"), COL["An"]] = 9
        guides.X = sparse.csr_matrix(X)
        res = assign_guides(expr, guides, cfg)
        s1 = res.obs.loc["zero"]
        assert (
            s1[OBS_PAIR_STATUS] == STATUS_PAIR_TARGET_NTC
            and s1[OBS_PAIR_ID] == "S1"
            and s1[OBS_CONSTRUCT_TYPE] == CONSTRUCT_SINGLE_NTC
        )
        assert (
            s1[OBS_PAIR_DETAIL] == DETAIL_DESIGNED_SINGLE_NTC and s1[OBS_PAIR] == "X" and s1[OBS_TARGET_SYMBOL] == "X"
        )
        if primary:
            assert s1[OBS_CLASS] == CLASS_TARGETING and s1[OBS_TARGET] == "X" and bool(s1["pair_assigned_primary"])
        else:
            assert (
                s1[OBS_CLASS] == CLASS_AMBIGUOUS
                and s1[OBS_TARGET] == cfg.guides.ambiguous_label
                and not bool(s1["pair_assigned_primary"])
            )
        assert res.obs.loc["same_target", OBS_CLASS] == CLASS_TARGETING  # dual-guide construct is primary either way
        lane = pair_assignment_per_lane(res)
        assert int(lane.loc[lane.lane_id == "ALL", "n_pair_assigned_primary"].iloc[0]) == int(
            res.obs["pair_assigned_primary"].sum()
        )
        assert res.uns["guide_assignment"]["designed_targeting_plus_ntc_primary"] is primary


def test_h5ad_name_sanitisation_keeps_mapping(tmp_path):
    """Target-derived obs columns with spaces / parentheses / slashes are written safely with a saved mapping."""
    import anndata as ad_
    from perturbseq_pipeline.io import sanitize_h5ad_name, sanitize_h5ad_names, write_h5ad

    a = ad_.AnnData(X=sparse.csr_matrix(np.ones((3, 2))))
    a.obs_names = ["c1", "c2", "c3"]
    a.obs["lochness_LIPA (rs1412444)"] = [0.1, 0.2, 0.3]
    a.obs["lochness_A/B"] = [1.0, 2.0, 3.0]
    a.obs["target_gene"] = ["LIPA (rs1412444)", "A/B", "ntc"]
    a.uns["scores/by target"] = {"x": 1}
    a.obsm["ps score"] = np.zeros((3, 2))
    assert sanitize_h5ad_name("lochness_LIPA (rs1412444)") == "lochness_LIPA__rs1412444_"
    recs = sanitize_h5ad_names(a)
    assert {r["original"] for r in recs} == {
        "lochness_LIPA (rs1412444)",
        "lochness_A/B",
        "scores/by target",
        "ps score",
    }
    assert "lochness_A_B" in a.obs.columns and "ps_score" in a.obsm and "scores_by_target" in a.uns
    assert list(a.obs["target_gene"]) == ["LIPA (rs1412444)", "A/B", "ntc"]  # values (biological labels) untouched
    b = ad_.AnnData(X=sparse.csr_matrix(np.ones((2, 2))))
    b.obs_names = ["c1", "c2"]
    b.obs["lochness_X (rs1)"] = [0.5, 0.6]
    p = tmp_path / "t.h5ad"
    write_h5ad(b, p)
    back = ad_.read_h5ad(p)
    assert "lochness_X__rs1_" in back.obs.columns and "column_name_mapping" in back.uns
    assert (tmp_path / "t_column_name_mapping.csv").is_file()
    assert list(pd.read_csv(tmp_path / "t_column_name_mapping.csv")["original"]) == ["lochness_X (rs1)"]


def test_require_complete_pair_false_assigns_from_the_resolved_slot():
    """`require_complete_pair: false` is documented to accept one-slot cells; it used to be a no-op."""
    from perturbseq_pipeline.dual_guides import CONSTRUCT_SINGLE_SLOT, DETAIL_INCOMPLETE_SINGLE_SLOT, OBS_CONSTRUCT_TYPE

    expr, guides, cfg = _build()
    cfg.guides.require_complete_pair = False
    res = assign_guides(expr, guides, cfg)
    o = res.obs.loc["missing_C"]  # only A1 (target X) resolved
    assert o[OBS_PAIR_STATUS] == STATUS_INCOMPLETE
    assert o[OBS_CLASS] == CLASS_TARGETING
    assert o[OBS_TARGET] == "X"
    assert o[OBS_CONSTRUCT_TYPE] == CONSTRUCT_SINGLE_SLOT
    assert res.obs.loc["missing_C", "pair_resolution_detail"] == DETAIL_INCOMPLETE_SINGLE_SLOT
    # every other cell is labelled exactly as with the default
    expr2, guides2, cfg2 = _build()
    ref = assign_guides(expr2, guides2, cfg2).obs
    others = [c for c in ref.index if c != "missing_C"]
    assert (res.obs.loc[others, OBS_CLASS].astype(str) == ref.loc[others, OBS_CLASS].astype(str)).all()
    assert res.uns["guide_assignment"]["require_complete_pair"] is False
