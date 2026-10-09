"""Membership-aware statistics for ``guides.assignment_mode: high_moi`` (PR B).

1. ``MembershipIndex`` unit tests on a hand-built matrix.
2. Consistency: on cells that carry exactly one target, the membership paths of
   perturbation strength, enrichment and modules reproduce the ``single_guide``
   results (same cells, same clustering).
3. STANDARD vs LARGE equality of the membership enrichment and perturbation paths.
4. Power on the ``moi=4`` fixture: planted knockdowns are hits, null targets are
   not, and the NTC pseudo-targets give a near-zero false-positive rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import KD_TARGETS, NULL_TARGETS, make_dataset  # noqa: E402
from test_high_moi_assignment import _build  # noqa: E402

from perturbseq_pipeline import cluster as cluster_mod  # noqa: E402
from perturbseq_pipeline import enrichment as enrich_mod  # noqa: E402
from perturbseq_pipeline import io as io_mod  # noqa: E402
from perturbseq_pipeline import modules as modules_mod  # noqa: E402
from perturbseq_pipeline import perturbation as pert_mod  # noqa: E402
from perturbseq_pipeline import qc as qc_mod  # noqa: E402
from perturbseq_pipeline.config import Config  # noqa: E402
from perturbseq_pipeline.guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, assign_guides  # noqa: E402
from perturbseq_pipeline.high_moi import OBS_N_TARGETS, MembershipIndex, membership_index  # noqa: E402


# 1. MembershipIndex


def test_membership_index_sets_and_counts():
    expr, guides, cfg = _build()
    res = assign_guides(expr, guides, cfg)
    mi = MembershipIndex(res, cfg)
    assert mi.targets == ["X", "Y", "Z"]  # NTC column excluded
    assert mi.counts.to_dict() == {"X": 3, "Y": 2, "Z": 0}
    names = res.obs_names
    assert list(names[mi.indices("X")]) == ["two_targets", "x_and_ntc", "deep"]
    # 'other' for X = targeting cells not carrying X (ntc_first carries only Y)
    assert list(names[mi.other_indices("X")]) == ["ntc_first"]
    assert mi.other_mask("Y").sum() == 2 and not mi.other_mask("Y")[names.get_loc("two_targets")]
    assert mi.targeting_mask.sum() == 4  # two_targets, x_and_ntc, ntc_first, deep
    # pseudo-targets = NTC guides, cells of any class
    assert mi.pseudo_targets == ["NTC:non_targeting_1", "NTC:non_targeting_2"]
    assert list(names[mi.indices("NTC:non_targeting_1")]) == ["x_and_ntc", "ntc_first", "ntc_only"]
    assert list(names[mi.other_indices("NTC:non_targeting_1")]) == ["two_targets", "deep"]
    lanes = res.obs["lane_id"].to_numpy()
    cb = mi.counts_by(lanes, ["L1", "L2"])
    assert cb.loc["X"].tolist() == [2, 1] and cb.loc["Y"].tolist() == [2, 0]
    pcb = mi.pseudo_counts_by(lanes, ["L1", "L2"], cell_mask=mi.targeting_mask)
    assert pcb.loc["NTC:non_targeting_1"].tolist() == [2, 0]  # ntc_only is not a targeting cell
    ind = mi.indicator(["X", "missing"])
    assert ind.shape == (2, res.n_obs) and ind[0].sum() == 3 and ind[1].sum() == 0
    gm = mi.guide_members(res, cfg, "X")
    assert set(gm) == {"X_1", "X_2"} and gm["X_1"].size == 2 and gm["X_2"].size == 1
    assert mi._guide_matrix(res, cfg)[0] is mi._guide_matrix(res, cfg)[0]  # CSC guide matrix is built once
    assert mi.guide_members(res, cfg, "nope") == {}
    assert membership_index(res, Config()) is None  # legacy modes never build an index


# 2./3. Consistency with single_guide on one-target cells, and STANDARD vs LARGE


def _base_cfg(data, outdir, **guides):
    return Config.from_dict(
        {
            "run": {"name": "c", "outdir": str(outdir), "seed": 0},
            "input": {"mtx_dirs": data["lanes"]},
            "metadata": {"file": data["metadata"]},
            "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
            "guides": guides,
            "cluster": {"n_top_genes": 80, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 5, "primary_control": "other"},
            "enrichment": {"min_cells_per_target": 5, "permutations": 50, "stratify_by": "lane_id"},
            "modules": {"enabled": True, "min_cells_per_perturbation": 5},
            "compute": {"n_jobs": 1},
        }
    )


@pytest.fixture(scope="module")
def paired_objects(tmp_path_factory):
    """Single-guide and high-MOI assignments of the SAME one-target cells with the SAME clustering."""
    root = tmp_path_factory.mktemp("consistency")
    data = make_dataset(root / "synthetic", n_lanes=2, n_cells=300)  # moi=None: one dominant guide per cell
    cfg_sg = _base_cfg(data, root / "sg")
    cfg_hm = _base_cfg(data, root / "hm", assignment_mode="high_moi", high_moi={"min_umi": 5})
    for c in (cfg_sg, cfg_hm):
        c.validate()
    loaded = io_mod.load_data(cfg_sg)
    expr = qc_mod.prefilter(loaded.expr, cfg_sg)
    expr = qc_mod.compute_qc_metrics(expr, cfg_sg)
    expr, _ = qc_mod.filter_cells_and_genes(expr, cfg_sg)
    sg = assign_guides(expr.copy(), loaded.guides.copy(), cfg_sg)
    hm = assign_guides(expr.copy(), loaded.guides.copy(), cfg_hm)
    # cells with exactly one target in BOTH modes
    one_sg = sg.obs[OBS_CLASS].astype(str).isin([CLASS_TARGETING, CLASS_NTC]).to_numpy()
    one_hm = ((hm.obs[OBS_N_TARGETS] == 1) | hm.obs["is_ntc_only"]).to_numpy()
    keep = one_sg & one_hm
    sg, hm = sg[keep].copy(), hm[keep].copy()
    assert (sg.obs["target_gene"].astype(str).to_numpy() == hm.obs["target_gene"].astype(str).to_numpy()).all()
    sg = cluster_mod.embed_and_cluster(sg, cfg_sg)
    for col in ("leiden",):
        hm.obs[col] = sg.obs[col].to_numpy()
    for key in sg.layers:
        if key not in hm.layers:
            hm.layers[key] = sg.layers[key]
    hm.obsm["X_pca"] = sg.obsm["X_pca"]
    hm.var["highly_variable"] = sg.var["highly_variable"].to_numpy()
    return sg, hm, cfg_sg, cfg_hm


def _num(df, key_cols):
    out = df.sort_values(key_cols).reset_index(drop=True)
    return out.set_index(key_cols).select_dtypes(include=[np.number, bool]).astype(float)


def test_perturbation_strength_matches_single_guide_on_one_target_cells(paired_objects):
    sg, hm, cfg_sg, cfg_hm = paired_objects
    a = pert_mod.test_all_targets(sg, cfg_sg).table
    b = pert_mod.test_all_targets(hm, cfg_hm).table
    assert len(a) == len(b) >= 4
    pd.testing.assert_frame_equal(_num(a, ["target_gene"]), _num(b, ["target_gene"]), check_exact=False, rtol=1e-9)


def test_enrichment_matches_single_guide_on_one_target_cells(paired_objects):
    sg, hm, cfg_sg, cfg_hm = paired_objects
    a = enrich_mod.test_cluster_enrichment(sg, cfg_sg)
    b = enrich_mod.test_cluster_enrichment(hm, cfg_hm)
    assert a.stratified and b.stratified and b.membership_aware and not a.membership_aware
    keys = ["target_gene", "cluster", "control"]
    pd.testing.assert_frame_equal(_num(a.table, keys), _num(b.table, keys), check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(a.composition.sort_index(), b.composition.sort_index(), check_exact=False, rtol=1e-9)
    # pseudo-targets exist only in membership mode and never leak into the real table
    assert not b.pseudo_table.empty and b.pseudo_table["target_gene"].str.startswith("NTC:").all()
    assert not b.table["target_gene"].str.startswith("NTC:").any()
    assert set(b.pseudo_summary) >= {"n_pseudo_targets", "pseudo_fpr_pct", "real_hit_rate_pct"}


def test_modules_effect_matrix_matches_single_guide_on_one_target_cells(paired_objects):
    sg, hm, cfg_sg, cfg_hm = paired_objects
    targets = modules_mod.select_perturbations(sg, cfg_sg)
    assert targets == modules_mod.select_perturbations(hm, cfg_hm)
    genes = modules_mod.select_genes(sg, cfg_sg)
    ea, ca, _ = modules_mod.build_effect_matrix(sg, genes, targets, cfg_sg)
    eb, cb, _ = modules_mod.build_effect_matrix(hm, genes, targets, cfg_hm)
    assert ca == cb
    pd.testing.assert_frame_equal(ea, eb, check_exact=False, rtol=1e-9)


def test_membership_paths_standard_equals_large(paired_objects):
    _, hm, _, cfg_hm = paired_objects
    cfg_large = Config.from_dict(cfg_hm.to_dict())
    cfg_large.scaling.mode = "large"
    std = enrich_mod.test_cluster_enrichment(hm, cfg_hm)
    large = enrich_mod.test_cluster_enrichment(hm, cfg_large)
    keys = ["target_gene", "cluster", "control"]
    cols = ["n_target_cells", "n_in_cluster", "n_reference_cells", "pval", "odds_ratio"]
    for a, b in ((std.table, large.table), (std.pseudo_table, large.pseudo_table)):
        pd.testing.assert_frame_equal(
            _num(a, keys)[cols], _num(b, keys)[cols], check_exact=False, rtol=1e-9
        )
    pd.testing.assert_frame_equal(std.composition.sort_index(), large.composition.sort_index(), check_exact=False, rtol=1e-9)
    p_std = pert_mod.test_all_targets(hm, cfg_hm).table.set_index("target_gene").sort_index()
    p_large = pert_mod.test_all_targets(hm, cfg_large).table.set_index("target_gene").sort_index()
    # LARGE samples controls only above 100k cells, so the two paths test the same cells; the LARGE path extracts
    # expression in float32, hence the loose tolerance (same difference as between the legacy STANDARD/LARGE paths)
    for col in ("n_perturbed", "n_control_other"):
        np.testing.assert_array_equal(p_std[col].to_numpy(dtype=float), p_large[col].to_numpy(dtype=float))
    for col in ("log2fc_other", "ks_stat_other"):
        np.testing.assert_allclose(p_std[col].to_numpy(dtype=float), p_large[col].to_numpy(dtype=float), rtol=1e-5)


# 4. Power and false positives on a high-MOI fixture


def test_high_moi_power_and_pseudo_target_fpr(tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300, moi=4)
    cfg = _base_cfg(data, tmp_path / "run", assignment_mode="high_moi", high_moi={"min_umi": 5})
    cfg.compute.n_jobs = 2
    cfg.distance.enabled = True
    cfg.ps_score.enabled = False
    cfg.lochness.enabled = False
    cfg.validate()
    result = run_pipeline(cfg)
    outdir = Path(cfg.run.outdir)
    pert = pd.read_csv(outdir / "tables" / "perturbation_full.csv").set_index("target_gene")
    # membership: perturbed cells per target ~ 45 % of cells, far more than the primary label gives
    assert (pert.loc[KD_TARGETS + NULL_TARGETS, "n_perturbed"] > 150).all(), pert["n_perturbed"]
    assert pert.loc[KD_TARGETS, "is_hit_other"].all(), pert.loc[KD_TARGETS, ["log2fc_other", "ks_fdr_other"]]
    assert not pert.loc[NULL_TARGETS, "is_hit_other"].any()
    assert (pert.loc[KD_TARGETS, "log2fc_other"] < -1.5).all()
    pseudo = pd.read_csv(outdir / "tables" / "enrichment_pseudo_targets.csv")
    summ = pd.read_csv(outdir / "tables" / "enrichment_pseudo_summary.csv").set_index("metric")["value"]
    assert pseudo["target_gene"].str.startswith("NTC:").all() and int(summ["n_pseudo_targets"]) >= 4
    assert float(summ["pseudo_fpr_pct"]) <= 10.0, summ.to_dict()
    enr = pd.read_csv(outdir / "tables" / "enrichment_full.csv")
    assert not enr["target_gene"].str.startswith("NTC:").any()
    dist = pd.read_csv(outdir / "tables" / "perturbation_distance.csv")
    assert set(KD_TARGETS) <= set(dist["target_gene"]) and not dist["target_gene"].str.startswith("NTC:").any()
    dist_pseudo = pd.read_csv(outdir / "tables" / "perturbation_distance_pseudo_targets.csv")
    assert dist_pseudo["target_gene"].str.startswith("NTC:").all() and len(dist_pseudo) >= 4
    # own BH family: the real-target FDRs equal BH over the real p-values alone
    from perturbseq_pipeline.perturbation import benjamini_hochberg

    np.testing.assert_allclose(dist["fdr"].to_numpy(), benjamini_hochberg(dist["pvalue"].to_numpy()), rtol=1e-9)
    assert not dist_pseudo["significant"].any() or dist_pseudo["significant"].mean() <= 0.25
    meta = pd.read_csv(outdir / "tables" / "perturbation_meta.csv")
    assert not meta["target_gene"].astype(str).str.startswith("NTC:").any()
    report = (outdir / "report.md").read_text()
    assert "negative-control pseudo-target" in report and "empirical" in report.lower()
    assert "PRIMARY (highest-UMI) target" in report  # the QC notice still names the primary-label stages
    assert result.adata.obsm[cfg.guides.high_moi.membership_obsm_key].shape[1] == len(KD_TARGETS + NULL_TARGETS) + 1
