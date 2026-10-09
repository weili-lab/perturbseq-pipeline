"""Per-cell scores on membership (``high_moi`` mode, PR C): lochNESS and PS.

On cells carrying exactly one target the membership paths reproduce the legacy
results; on the ``moi=4`` fixture every (cell, carried target) pair gets a
score and ``obs`` keeps the primary target's value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import KD_TARGETS, NULL_TARGETS, make_dataset  # noqa: E402
from test_high_moi_membership_stats import _base_cfg, paired_objects  # noqa: E402,F401

from perturbseq_pipeline import lochness as loch_mod  # noqa: E402
from perturbseq_pipeline import ps_score as ps_mod  # noqa: E402
from perturbseq_pipeline.config import Config  # noqa: E402
from perturbseq_pipeline.guides import CLASS_TARGETING, OBS_CLASS, OBS_TARGET  # noqa: E402
from perturbseq_pipeline.high_moi import OBS_N_TARGETS  # noqa: E402


def test_lochness_matches_single_guide_on_one_target_cells(paired_objects):
    sg, hm, cfg_sg, cfg_hm = paired_objects
    for c in (cfg_sg, cfg_hm):
        c.lochness.enabled = True
        c.lochness.min_cells_per_target = 5
    a = loch_mod.compute_lochness(sg, cfg_sg)
    b = loch_mod.compute_lochness(hm, cfg_hm)
    assert b.membership_aware and not a.membership_aware
    cols = ["n_cells", "mean_lochness_all_cells", "mean_lochness_in_own_cells", "max_lochness", "pct_cells_enriched"]
    sa = a.summary.set_index("target_gene").sort_index()[cols]
    sb = b.summary.set_index("target_gene").sort_index()[cols]
    pd.testing.assert_frame_equal(sa.astype(float), sb.astype(float), check_exact=False, rtol=1e-5)
    np.testing.assert_allclose(a.self_score, b.self_score, rtol=1e-5, equal_nan=True)
    for gene, vec in a.scores.items():
        np.testing.assert_allclose(vec, b.scores[gene], rtol=1e-5)
    # the sparse membership scores hold exactly the member cells' values
    M = b.membership_scores.tocsc()
    j = b.membership_targets.index(a.summary["target_gene"].iloc[0])
    members = M.indices[M.indptr[j] : M.indptr[j + 1]]
    np.testing.assert_allclose(M.data[M.indptr[j] : M.indptr[j + 1]], a.scores[b.membership_targets[j]][members], rtol=1e-5)
    # LARGE execution gives the same member scores and summaries
    cfg_large = Config.from_dict(cfg_hm.to_dict())
    cfg_large.scaling.mode = "large"
    c = loch_mod.compute_lochness(hm, cfg_large)
    assert c.self_only and not c.scores
    pd.testing.assert_frame_equal(sb.astype(float), c.summary.set_index("target_gene").sort_index()[cols].astype(float), check_exact=False, rtol=1e-5)
    np.testing.assert_allclose(b.self_score, c.self_score, rtol=1e-5, equal_nan=True)
    # attach: primary value in obs, membership matrix in obsm
    out = loch_mod.attach_scores(hm.copy(), b)
    assert "lochness_membership" in out.obsm and list(out.uns["lochness_membership_targets"]) == b.membership_targets
    assert np.isfinite(out.obs["lochness_self"].to_numpy()[(out.obs[OBS_CLASS].astype(str) == CLASS_TARGETING).to_numpy()]).all()


@pytest.mark.skipif(not ps_mod.pertps_available(), reason="pertps not installed")
def test_ps_matches_target_wise_legacy_on_one_target_cells(paired_objects):
    sg, hm, cfg_sg, cfg_hm = paired_objects
    cfg_sg_large = Config.from_dict(cfg_sg.to_dict())
    cfg_sg_large.scaling.mode = "large"  # the legacy target-wise path: same per-target index sets as membership
    for c in (cfg_sg_large, cfg_hm):
        c.ps_score.enabled = True
        c.ps_score.min_cells_per_target = 5
        c.ps_score.min_control_cells = 5
        c.ps_score.compute_lda_umap = False
    a = ps_mod.compute_ps_scores(sg, cfg_sg_large)
    b = ps_mod.compute_ps_scores(hm, cfg_hm)
    assert a is not None and b is not None and b.membership_aware and b.large_mode
    cols = ["n_perturbed_cells", "n_control_cells", "mean_ps", "median_ps", "pct_successful_kd"]
    sa = a.summary.set_index("target_gene").sort_index()[cols]
    sb = b.summary.set_index("target_gene").sort_index()[cols]
    pd.testing.assert_frame_equal(sa.astype(float), sb.astype(float), check_exact=False, rtol=1e-5)
    np.testing.assert_allclose(a.own_score, b.own_score, rtol=1e-5, equal_nan=True)
    assert b.membership_scores is not None and b.membership_scores.shape == (hm.n_obs, len(b.membership_targets))
    out = ps_mod.attach_scores(hm.copy(), b)
    assert "ps_score_membership" in out.obsm and "ps_score" in out.obs


@pytest.mark.skipif(not ps_mod.pertps_available(), reason="pertps not installed")
def test_high_moi_cell_scores_end_to_end(tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300, moi=4)
    cfg = _base_cfg(data, tmp_path / "run", assignment_mode="high_moi", high_moi={"min_umi": 5})
    cfg.compute.n_jobs = 2
    cfg.modules.enabled = False
    cfg.ps_score.enabled = True
    cfg.ps_score.min_control_cells = 5
    cfg.lochness.enabled = True
    cfg.lochness.min_cells_per_target = 5
    cfg.validate()
    result = run_pipeline(cfg)
    ad_out = result.adata
    obs = ad_out.obs
    n_targets = len(KD_TARGETS + NULL_TARGETS)
    assert "lochness_membership" in ad_out.obsm and ad_out.obsm["lochness_membership"].shape == (ad_out.n_obs, n_targets)
    multi = (obs[OBS_N_TARGETS] >= 2).to_numpy()
    assert multi.sum() > 100
    # a multi-target cell has one lochNESS entry per carried target, and obs holds its primary target's value
    L = ad_out.obsm["lochness_membership"].tocsr()
    i = int(np.flatnonzero(multi)[0])
    row = L[i]
    assert row.nnz == obs[OBS_N_TARGETS].iloc[i]
    j = list(ad_out.uns["lochness_membership_targets"]).index(str(obs[OBS_TARGET].iloc[i]))
    assert np.isclose(row[0, j], obs["lochness_self"].iloc[i], rtol=1e-5)
    if "ps_score_membership" in ad_out.obsm:
        P = ad_out.obsm["ps_score_membership"].tocsr()
        assert P.shape[0] == ad_out.n_obs and P.nnz > 0
        k = list(ad_out.uns["ps_score_membership_targets"]).index(str(obs[OBS_TARGET].iloc[i]))
        if P[i, k] != 0:
            assert np.isclose(P[i, k], obs["ps_score"].iloc[i], rtol=1e-5)
    outdir = Path(cfg.run.outdir)
    loch = pd.read_csv(outdir / "tables" / "lochness.csv")
    assert set(loch["target_gene"]) == set(KD_TARGETS + NULL_TARGETS)
    ps = pd.read_csv(outdir / "tables" / "ps_score.csv")
    assert len(ps) >= 3 and (ps["n_perturbed_cells"] > 150).all()
    report = (outdir / "report.md").read_text()
    assert "ps_score_membership" in report and "kd_status_membership" in report


def test_knockdown_filter_matches_single_guide_on_one_target_cells(paired_objects):
    from perturbseq_pipeline import knockdown_filter as kd_mod

    sg, hm, cfg_sg, cfg_hm = paired_objects
    for c in (cfg_sg, cfg_hm):
        c.knockdown_filter.enabled = True
        c.knockdown_filter.min_cells = 5
        c.knockdown_filter.min_control_cells = 5
    a, ta = kd_mod.compute_knockdown_mask(sg.copy(), cfg_sg)
    b, tb = kd_mod.compute_knockdown_mask(hm.copy(), cfg_hm)
    keys = ["target_gene", "context"]
    cols = ["n_cells", "n_control", "control_mean", "mean_ratio", "n_kept", "n_escaper"]
    pd.testing.assert_frame_equal(
        ta.set_index(keys).sort_index()[cols].astype(float), tb.set_index(keys).sort_index()[cols].astype(float), check_exact=False, rtol=1e-6
    )
    assert (ta.set_index(keys).sort_index()["group_status"] == tb.set_index(keys).sort_index()["group_status"]).all()
    assert (a.obs["kd_status"].astype(str).to_numpy() == b.obs["kd_status"].astype(str).to_numpy()).all()
    np.testing.assert_allclose(a.obs["kd_ratio"].to_numpy(), b.obs["kd_ratio"].to_numpy(), rtol=1e-6, equal_nan=True)
    assert "kd_status_membership" in b.obsm and "kd_ratio_membership" in b.obsm and "kd_status_membership" not in a.obsm
    S = b.obsm["kd_status_membership"].tocsr()
    codes = b.uns["kd_status_codes"]
    # on one-target cells the membership matrix has one entry per targeting cell, equal to the obs status
    targeting = (b.obs[OBS_CLASS].astype(str) == CLASS_TARGETING).to_numpy()
    assert (np.diff(S.indptr)[targeting] == 1).all()
    inv = {v: k for k, v in codes.items()}
    i = int(np.flatnonzero(targeting)[0])
    assert inv[int(S[i].data[0])] == str(b.obs["kd_status"].iloc[i])
