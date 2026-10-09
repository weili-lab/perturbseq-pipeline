"""High-MOI membership assignment (``guides.assignment_mode: high_moi``).

Unit tests on a hand-built guide matrix, an h5ad round trip, config validation,
and one end-to-end run on the ``moi=4`` synthetic fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import KD_TARGETS, NULL_TARGETS, make_dataset, make_lane  # noqa: E402

from perturbseq_pipeline.config import Config  # noqa: E402
from perturbseq_pipeline.guides import (  # noqa: E402
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
    assign_guides,
)
from perturbseq_pipeline.high_moi import (  # noqa: E402
    MODE_HIGH_MOI,
    OBS_MODE,
    OBS_N_CALLED,
    OBS_N_GUIDES,
    OBS_N_TARGETS,
    OBS_NTC_ONLY,
    UNS_RANK_PROFILE,
    UNS_TARGETS,
    cells_per_target,
    high_moi_tables,
    membership_assignment_summary,
    membership_guide_representation,
)

GUIDES = ["X_1", "X_2", "Y_1", "Y_2", "non_targeting_1", "non_targeting_2", "Z_1"]
CELLS = {
    "two_targets": {"X_1": 50, "Y_1": 40, "non_targeting_1": 1},
    "x_and_ntc": {"X_2": 30, "non_targeting_1": 25},
    "ntc_first": {"non_targeting_1": 80, "Y_1": 30},
    "ntc_only": {"non_targeting_1": 60, "non_targeting_2": 12},
    "deep": {"X_1": 2000, "Y_2": 30, "Z_1": 2},
    "below": {"X_1": 4},
    "zero": {},
    "over_cap": {g: 20 for g in GUIDES},
}


def _build(**overrides):
    X = np.zeros((len(CELLS), len(GUIDES)))
    for i, (_, counts) in enumerate(CELLS.items()):
        for g, v in counts.items():
            X[i, GUIDES.index(g)] = v
    names = list(CELLS)
    guides = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame(index=names), var=pd.DataFrame(index=GUIDES))
    expr = ad.AnnData(
        X=sparse.csr_matrix(np.ones((len(names), 3))),
        obs=pd.DataFrame({"lane_id": ["L1"] * 4 + ["L2"] * (len(names) - 4)}, index=names),
        var=pd.DataFrame(index=["X", "Y", "g3"]),
    )
    cfg = Config()
    cfg.guides.assignment_mode = MODE_HIGH_MOI
    cfg.guides.target_split_delims = ["_"]
    cfg.guides.high_moi.max_guides_per_cell = 5
    cfg.perturbation.min_cells_per_target = 2
    for key, value in overrides.items():
        setattr(cfg.guides.high_moi, key, value)
    return expr, guides, cfg


def _member(res, cfg, cell, target):
    T = res.obsm[cfg.guides.high_moi.membership_obsm_key]
    col = list(res.uns[UNS_TARGETS]).index(target)
    return int(T[res.obs_names.get_loc(cell), col])


# Membership calling


def test_threshold_membership_and_classes():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    obs = res.obs
    assert list(res.uns[UNS_TARGETS]) == ["X", "Y", "Z", cfg.guides.ntc_label]
    assert _member(res, cfg, "two_targets", "X") == 1 and _member(res, cfg, "two_targets", "Y") == 1
    assert _member(res, cfg, "two_targets", cfg.guides.ntc_label) == 0  # 1 UMI < min_umi
    assert obs.loc["two_targets", OBS_N_TARGETS] == 2 and obs.loc["two_targets", OBS_N_GUIDES] == 2
    assert obs.loc["two_targets", OBS_CLASS] == CLASS_TARGETING
    assert obs.loc["x_and_ntc", OBS_CLASS] == CLASS_TARGETING and _member(res, cfg, "x_and_ntc", cfg.guides.ntc_label) == 1
    assert obs.loc["below", OBS_CLASS] == CLASS_UNASSIGNED and obs.loc["below", OBS_N_GUIDES] == 0
    assert obs.loc["zero", OBS_CLASS] == CLASS_UNASSIGNED
    assert (obs[OBS_MODE] == MODE_HIGH_MOI).all()
    assert res.obsm[cfg.guides.high_moi.membership_obsm_key].dtype == np.int8


def test_min_frac_of_top_filters_deep_background():
    expr, guides, cfg = _build()  # default min_frac_of_top 0.02 -> 40 UMIs for the 2000-UMI cell
    res = assign_guides(expr, guides, cfg)
    assert _member(res, cfg, "deep", "Y") == 0 and _member(res, cfg, "deep", "X") == 1
    expr, guides, cfg = _build(min_frac_of_top=0.0)
    res = assign_guides(expr, guides, cfg)
    assert _member(res, cfg, "deep", "Y") == 1
    assert _member(res, cfg, "deep", "Z") == 0  # 2 UMIs < min_umi


def test_knee_method():
    expr, guides, cfg = _build(method="knee", min_umi=3)
    res = assign_guides(expr, guides, cfg)
    # 2000 vs 30 vs 2: largest log drop is 2000 -> 30, so only X_1 is a member
    assert _member(res, cfg, "deep", "X") == 1 and _member(res, cfg, "deep", "Y") == 0
    # 50 vs 40 vs 1 (below min_umi -> tail): the big drop is after the two real guides
    assert res.obs.loc["two_targets", OBS_N_GUIDES] == 2
    # a single candidate above min_umi is kept
    assert res.obs.loc["below", OBS_CLASS] == CLASS_TARGETING


def test_ntc_collapse_and_ntc_only():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    obs = res.obs
    assert obs.loc["ntc_only", OBS_CLASS] == CLASS_NTC and bool(obs.loc["ntc_only", OBS_NTC_ONLY])
    assert obs.loc["ntc_only", OBS_TARGET] == cfg.guides.ntc_label
    assert obs.loc["ntc_only", OBS_N_GUIDES] == 2 and obs.loc["ntc_only", OBS_N_TARGETS] == 0
    assert _member(res, cfg, "ntc_only", cfg.guides.ntc_label) == 1
    assert not bool(obs.loc["x_and_ntc", OBS_NTC_ONLY])
    assert cells_per_target(res, cfg)[cfg.guides.ntc_label] == 3  # x_and_ntc, ntc_first, ntc_only


def test_cap_marks_ambiguous_and_clears_membership():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    obs = res.obs
    assert obs.loc["over_cap", OBS_CLASS] == CLASS_AMBIGUOUS
    assert obs.loc["over_cap", OBS_TARGET] == cfg.guides.ambiguous_label
    assert obs.loc["over_cap", OBS_N_GUIDES] == 0
    assert obs.loc["over_cap", OBS_N_CALLED] == 7  # pre-cap count kept for diagnostics
    assert obs.loc["two_targets", OBS_N_CALLED] == obs.loc["two_targets", OBS_N_GUIDES] == 2
    i = res.obs_names.get_loc("over_cap")
    assert res.obsm[cfg.guides.high_moi.guide_membership_obsm_key][i].nnz == 0
    assert res.uns["high_moi_calling"]["n_cells_over_cap"] == 1
    expr, guides, cfg = _build(max_guides_per_cell=7)
    res = assign_guides(expr, guides, cfg)
    assert res.obs.loc["over_cap", OBS_CLASS] == CLASS_TARGETING and res.obs.loc["over_cap", OBS_N_TARGETS] == 3


def test_primary_label_is_the_top_targeting_guide():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    obs = res.obs
    assert obs.loc["two_targets", OBS_TARGET] == "X" and obs.loc["two_targets", OBS_GUIDE] == "X_1"
    # an NTC guide with more UMIs than the targeting guide does not become the primary label
    assert obs.loc["ntc_first", OBS_TARGET] == "Y" and obs.loc["ntc_first", OBS_GUIDE] == "Y_1"
    assert obs.loc["ntc_first", OBS_CLASS] == CLASS_TARGETING
    assert obs.loc["ntc_only", OBS_GUIDE] == "non_targeting_1"


def test_legacy_diagnostic_columns_match_single_guide_path():
    expr, guides, cfg = _build()
    hm = assign_guides(expr.copy(), guides.copy(), cfg)
    cfg_sg = Config()
    cfg_sg.guides.target_split_delims = ["_"]
    sg = assign_guides(expr.copy(), guides.copy(), cfg_sg)
    for col in (OBS_TOP, OBS_SECOND, OBS_TOTAL, OBS_NDETECTED):
        np.testing.assert_array_equal(hm.obs[col].to_numpy(), sg.obs[col].to_numpy(), err_msg=col)
    # the dominance rule calls the two-target cell ambiguous; membership keeps it
    assert sg.obs.loc["two_targets", OBS_CLASS] == CLASS_AMBIGUOUS
    assert hm.obs.loc["two_targets", OBS_CLASS] == CLASS_TARGETING


# Summaries


def test_summary_tables_count_membership_not_primary_labels():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    summ = membership_assignment_summary(res, cfg).set_index(OBS_TARGET)
    assert summ.loc["X", "n_cells"] == 3 and summ.loc["X", "n_cells_primary"] == 3
    assert summ.loc["Y", "n_cells"] == 2 and summ.loc["Y", "n_cells_primary"] == 1  # ntc_first is primary Y
    assert summ.loc["Y", "class"] == CLASS_TARGETING
    # legacy meaning kept: Y has 2 member cells but only 1 primary-label cell (< min_cells_per_target = 2)
    assert not bool(summ.loc["Y", "testable"]) and bool(summ.loc["Y", "testable_membership"])
    assert bool(summ.loc["X", "testable"]) and bool(summ.loc["X", "testable_membership"])
    assert summ.loc["Z", "n_cells"] == 0 and not bool(summ.loc["Z", "testable"])
    assert summ.loc[cfg.guides.ntc_label, "class"] == CLASS_NTC
    assert summ.loc[cfg.guides.ambiguous_label, "n_cells"] == 1 and summ.loc[cfg.guides.unassigned_label, "n_cells"] == 2
    rep = membership_guide_representation(guides, res, cfg).set_index("guide_id")
    assert rep.loc["Y_1", "n_cells"] == 2 and rep.loc["Y_1", "n_cells_primary"] == 1
    assert rep.loc["Z_1", "n_cells"] == 0 and rep.loc["Z_1", "target_gene"] == "Z"
    tabs = high_moi_tables(res, cfg)
    assert set(tabs) == {"high_moi_calling", "high_moi_rank_umi_profile", "high_moi_cells_per_target"}
    assert tabs["high_moi_rank_umi_profile"]["rank"].tolist()[:3] == [1, 2, 3]
    assert tabs["high_moi_cells_per_target"].set_index(OBS_TARGET).loc["X", "n_cells"] == 3
    assert any(m.startswith("Lane L1") for m in tabs["high_moi_calling"]["metric"])


def test_h5ad_round_trip(tmp_path):
    from perturbseq_pipeline.io import write_h5ad

    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    path = write_h5ad(res, tmp_path / "hm.h5ad")
    back = ad.read_h5ad(path)
    for key in (cfg.guides.high_moi.membership_obsm_key, cfg.guides.high_moi.guide_membership_obsm_key):
        assert key in back.obsm
        assert (sparse.csr_matrix(back.obsm[key]) != res.obsm[key]).nnz == 0
    assert [str(t) for t in back.uns[UNS_TARGETS]] == list(res.uns[UNS_TARGETS])
    assert pd.DataFrame(back.uns[UNS_RANK_PROFILE])["rank"].tolist() == res.uns[UNS_RANK_PROFILE]["rank"].tolist()
    assert cells_per_target(back, cfg).equals(cells_per_target(res, cfg))


# Config / dispatch


def test_config_validation():
    base = {"input": {"h5ad": "x.h5ad"}, "guides": {"assignment_mode": "high_moi"}}
    Config.from_dict(base).validate()
    for bad in (
        {"method": "mixture"},
        {"min_umi": 0},
        {"min_frac_of_top": 1.0},
        {"max_guides_per_cell": 0},
        {"min_guides_per_cell": 40},
        {"membership_obsm_key": "guide_membership"},
    ):
        cfg = Config.from_dict({**base, "guides": {"assignment_mode": "high_moi", "high_moi": bad}})
        with pytest.raises(ValueError, match="guides.high_moi"):
            cfg.validate()
    for key in ("membership_obsm_key", "guide_membership_obsm_key"):
        cfg = Config.from_dict({**base, "guides": {"assignment_mode": "high_moi", "high_moi": {key: "guide_counts"}}})
        with pytest.raises(ValueError, match="output.guide_obsm_key"):
            cfg.validate()  # would be overwritten by the merged raw guide counts in stage 13
    cfg = Config.from_dict({"input": {"h5ad": "x.h5ad"}, "guides": {"multiplet": {"expected_guides_per_cell": None}}})
    cfg.validate()
    assert cfg.guides.multiplet.expected_guides_per_cell is None
    assert Config().guides.multiplet.expected_guides_per_cell == 1  # legacy default unchanged


def test_high_moi_needs_a_guide_matrix():
    expr, _, cfg = _build()
    expr.obs["guide_label"] = "X_1"
    with pytest.raises(ValueError, match="needs a guide count matrix"):
        assign_guides(expr, None, cfg)


def test_multiplet_flag_disabled_when_expected_guides_is_null():
    from perturbseq_pipeline.guide_qc import _per_cell_multiplet_rule, _structure_flags

    detected = sparse.csr_matrix(np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]]))
    design = pd.DataFrame({"guide_id": ["a", "b", "c"], "target": ["A", "B", "C"]})
    cfg = Config()
    _, _, _, multiplet, structure = _structure_flags(detected, design, cfg)
    assert multiplet.tolist() == [True, False, False]
    cfg.guides.multiplet.expected_guides_per_cell = None
    _, _, _, multiplet, structure = _structure_flags(detected, design, cfg)
    assert not multiplet.any() and structure.tolist() == [True, True, False]
    # the same rule serves the basic-QC annotation path (attach_guide_counts); None must not raise there either
    n_guides = np.array([3, 1, 0])
    multiplet, structure, rule = _per_cell_multiplet_rule(n_guides, cfg, 3)
    assert not multiplet.any() and structure.all() and "null" in rule
    cfg.guides.multiplet.expected_guides_per_cell = 1
    multiplet, structure, rule = _per_cell_multiplet_rule(n_guides, cfg, 3)
    assert multiplet.tolist() == [True, False, False] and structure.tolist() == [False, True, False]


# Synthetic fixture


def test_synthetic_moi_option(tmp_path):
    import gzip

    import scipy.io

    default = make_lane(tmp_path / "a", "L1", n_cells=50, seed=3)
    explicit = make_lane(tmp_path / "b", "L1", n_cells=50, seed=3, moi=None)
    assert gzip.open(default / "matrix.mtx.gz").read() == gzip.open(explicit / "matrix.mtx.gz").read()
    high = make_lane(tmp_path / "c", "L1", n_cells=200, seed=3, moi=4)
    with gzip.open(high / "matrix.mtx.gz", "rb") as fh:
        m = scipy.io.mmread(fh).tocsr().T  # cells x features
    guide_block = m[:, -21:].toarray()
    per_cell = (guide_block >= 10).sum(axis=1)
    assert per_cell.mean() > 2.5 and (per_cell == 0).mean() < 0.25


# End to end


def test_high_moi_end_to_end(tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300, moi=4)
    cfg = Config.from_dict(
        {
            "run": {"name": "hm", "outdir": str(tmp_path / "run")},
            "input": {"mtx_dirs": data["lanes"]},
            "metadata": {"file": data["metadata"]},
            "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
            "guides": {"assignment_mode": "high_moi", "high_moi": {"min_umi": 5}},
            "cluster": {"n_top_genes": 80, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 5, "top_n_report": 2, "primary_control": "other"},
            "modules": {"enabled": False},
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
            "distance": {"enabled": False},
            "distance_space": {"enabled": False},
        }
    )
    cfg.validate()
    result = run_pipeline(cfg)
    obs = result.adata.obs
    hm = cfg.guides.high_moi
    klass = obs[OBS_CLASS].astype(str)
    frac_targeting = (klass == CLASS_TARGETING).mean()
    frac_assigned = klass.isin([CLASS_TARGETING, CLASS_NTC]).mean()
    # 10 % of fixture cells carry no guide at all; the rest carry Poisson(4) guides.
    assert frac_targeting > 0.75 and frac_assigned > 0.85, (frac_targeting, frac_assigned)
    assert (klass == CLASS_AMBIGUOUS).sum() == 0
    assert np.median(obs.loc[obs[OBS_N_GUIDES] > 0, OBS_N_GUIDES]) >= 3
    assert hm.membership_obsm_key in result.adata.obsm and hm.guide_membership_obsm_key in result.adata.obsm
    assert list(result.adata.uns[UNS_TARGETS]) == sorted(KD_TARGETS + NULL_TARGETS) + [cfg.guides.ntc_label]
    outdir = Path(cfg.run.outdir)
    ga = pd.read_csv(outdir / "tables" / "guide_assignment.csv").set_index("target_gene")
    assert "n_cells_primary" in ga.columns and "testable_membership" in ga.columns
    assert ga.loc[KD_TARGETS + NULL_TARGETS, "testable"].all()
    assert ga.loc[KD_TARGETS + NULL_TARGETS, "testable_membership"].all()
    # `testable` keeps the legacy meaning: it predicts exactly what the perturbation stage tests
    pert_tested = set(pd.read_csv(outdir / "tables" / "perturbation_full.csv")["target_gene"])
    assert set(ga.index[ga["testable"]]) == pert_tested
    assert (ga.loc[KD_TARGETS + NULL_TARGETS, "n_cells"] > ga.loc[KD_TARGETS + NULL_TARGETS, "n_cells_primary"]).all()
    for name in ("high_moi_calling", "high_moi_rank_umi_profile", "high_moi_cells_per_target"):
        assert (outdir / "tables" / f"{name}.csv").is_file(), name
    # Primary-label analysis still recovers the planted knockdowns against the 'other' control
    pert = pd.read_csv(outdir / "tables" / "perturbation_full.csv").set_index("target_gene")
    assert pert.loc[KD_TARGETS, "is_hit_other"].all(), pert.loc[KD_TARGETS]
    assert not pert.loc[NULL_TARGETS, "is_hit_other"].any(), pert.loc[NULL_TARGETS]
    report = (outdir / "report.md").read_text()
    assert "High-MOI guide calling" in report and "primary (highest-UMI) target" in report
    html = (outdir / cfg.output.report_name).read_text()
    assert "Membership calling summary" in html and "Cells with &gt;= 1 called guide" in html
    assert "Rank-ordered guide UMI profile" in html and "median_umi" in html
    assert "PRIMARY (highest-UMI) target" in report  # the QC warning is listed in the report
    back = ad.read_h5ad(outdir / cfg.output.h5ad_name)
    assert hm.membership_obsm_key in back.obsm and UNS_TARGETS in back.uns
    guide_table = pd.read_csv(outdir / f"{cfg.run.name}_guide_barcodes.txt", sep="\t")
    assert "is_member" in guide_table.columns and guide_table["is_member"].sum() > 0
    manifest = pd.read_csv(outdir / "tables" / "figure_manifest.csv")
    assert {"high_moi_rank_umi_profile", "high_moi_guides_per_cell", "high_moi_cells_per_target"} <= set(manifest["name"])
