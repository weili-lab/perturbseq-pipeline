"""Membership regression (``regression.enabled``, high-MOI PR D).

1. The chunked sufficient-statistic solve equals a dense least-squares / ridge
   fit (coefficients and t statistics), with and without covariates.
2. Co-carriage: a target with no effect that is mostly co-carried with an
   effective target looks like a hit in the pseudobulk contrast but not in the
   regression.
3. Global null: the permutation FDR calls (almost) nothing.
4. End to end on the ``moi=4`` fixture, with the modules built from the
   regression effect matrix.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import KD_TARGETS, NULL_TARGETS, make_dataset  # noqa: E402
from test_high_moi_membership_stats import _base_cfg  # noqa: E402

from perturbseq_pipeline import regression as reg_mod  # noqa: E402
from perturbseq_pipeline.config import Config  # noqa: E402


def _membership_adata(
    T: np.ndarray, Y: np.ndarray, targets, *, lanes=None, n_guides=None, total_counts=None, ntc_label="non"
):
    """Minimal high-MOI AnnData: ``T`` = cells x (targets + NTC column), ``Y`` = lognorm cells x genes."""
    n = T.shape[0]
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["is_ntc_only"] = (T[:, -1] > 0) & (T[:, :-1].sum(axis=1) == 0)
    obs["n_guides_assigned"] = T.sum(axis=1) if n_guides is None else n_guides
    obs["total_counts"] = np.full(n, 1000.0) if total_counts is None else total_counts
    if lanes is not None:
        obs["lane_id"] = lanes
    a = ad.AnnData(X=sparse.csr_matrix(Y), obs=obs, var=pd.DataFrame(index=[f"G{j}" for j in range(Y.shape[1])]))
    a.layers["lognorm"] = sparse.csr_matrix(Y)
    a.obsm["perturbation_membership"] = sparse.csr_matrix(T.astype(np.int8))
    a.uns["membership_targets"] = list(targets) + [ntc_label]
    return a


def _cfg(**reg):
    base = {"enabled": True, "genes": "all", "min_cells": 1, "n_permutations": 5}
    base.update(reg)
    cfg = Config.from_dict({"guides": {"assignment_mode": "high_moi", "ntc_label": "non"}, "regression": base})
    cfg.scaling.effect_gene_chunk = 7  # several chunks
    return cfg


def _random_design(rng, n=400, k=6, moi=2.0):
    T = np.zeros((n, k + 1), dtype=np.int8)
    for i in range(n):
        m = rng.poisson(moi)
        if m == 0:
            T[i, -1] = 1
        else:
            T[i, rng.choice(k, size=min(m, k), replace=False)] = 1
            T[i, -1] = rng.random() < 0.3
    return T


@pytest.mark.parametrize("alpha", [0.0, 2.5])
@pytest.mark.parametrize("covariates", [False, True])
def test_solve_matches_dense_fit(alpha, covariates):
    rng = np.random.default_rng(0)
    T = _random_design(rng)
    n, k = T.shape[0], T.shape[1] - 1
    Y = np.abs(rng.normal(1.0, 0.7, size=(n, 20))) * (rng.random((n, 20)) < 0.7)
    lanes = np.where(np.arange(n) < n // 2, "L1", "L2")
    n_guides = T.sum(axis=1) + rng.integers(0, 2, size=n)
    depth = rng.uniform(500, 5000, size=n)
    a = _membership_adata(T, Y, [f"T{j}" for j in range(k)], lanes=lanes, n_guides=n_guides, total_counts=depth)
    cfg = _cfg(
        ridge_alpha=alpha,
        n_guides_covariate=covariates,
        depth_covariate=covariates,
        batch_key="lane_id" if covariates else None,
    )
    res = reg_mod.run_regression(a, cfg)
    cols = [T[:, :k].astype(float)]
    if covariates:
        cols.append((n_guides - n_guides.mean())[:, None])
    cols.append(np.ones((n, 1)))
    if covariates:
        ld = np.log(depth)
        cols += [(ld - ld.mean())[:, None], (lanes == "L2").astype(float)[:, None]]
    X = np.hstack(cols)  # every cell is assigned (targeting or NTC-only) in this design
    pen = np.zeros(X.shape[1])
    pen[:k] = alpha
    A = X.T @ X + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ Y)
    resid = Y - X @ beta
    s2 = (resid**2).sum(axis=0) / (n - X.shape[1])
    Ainv = np.linalg.inv(A)
    V = Ainv @ (X.T @ X) @ Ainv
    t = beta[:k] / np.sqrt(np.outer(np.diag(V)[:k], s2))
    order = [int(name[1:]) for name in res.log2fc.index]
    np.testing.assert_allclose(res.log2fc.to_numpy() * math.log(2), beta[order], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(res.tstat.to_numpy(), t[order], rtol=1e-6, atol=1e-8)
    assert ((res.pval.to_numpy() > 0) & (res.pval.to_numpy() <= 1)).all()


def test_one_target_cells_reduce_to_mean_difference_vs_ntc():
    rng = np.random.default_rng(1)
    n, k = 300, 3
    T = np.zeros((n, k + 1), dtype=np.int8)
    which = rng.integers(0, k + 1, size=n)  # k == NTC-only
    T[np.arange(n), which] = 1
    Y = rng.normal(2.0, 0.5, size=(n, 5))
    a = _membership_adata(T, Y, ["A", "B", "C"])
    res = reg_mod.run_regression(
        a, _cfg(ridge_alpha=0.0, n_guides_covariate=False, depth_covariate=False, batch_key=None)
    )
    for j, name in enumerate(["A", "B", "C"]):
        expected = Y[which == j].mean(axis=0) - Y[which == k].mean(axis=0)
        np.testing.assert_allclose(res.log2fc.loc[name].to_numpy() * math.log(2), expected, rtol=1e-9)
        assert res.n_cells[name] == int((which == j).sum())


def test_co_carried_null_target_is_not_a_hit():
    """B has no effect but rides with A in most of its cells: pseudobulk sees B, the regression does not."""
    rng = np.random.default_rng(2)
    n, G = 1200, 30
    T = np.zeros((n, 4), dtype=np.int8)  # A, B, C, NTC
    u = rng.random(n)
    T[u < 0.25, 0] = 1  # A
    T[u < 0.20, 1] = 1  # B almost always with A
    T[(u >= 0.25) & (u < 0.30), 1] = 1  # a few B-only cells
    T[(u >= 0.30) & (u < 0.60), 2] = 1  # C, unrelated
    T[u >= 0.60, 3] = 1  # NTC-only
    Y = rng.normal(3.0, 0.4, size=(n, G))
    Y[T[:, 0] == 1, :5] -= 1.0  # A represses G0..G4
    lanes = np.where(rng.random(n) < 0.5, "L1", "L2")
    a = _membership_adata(T, Y, ["A", "B", "C"], lanes=lanes)
    res = reg_mod.run_regression(a, _cfg(n_permutations=20, n_guides_covariate=False))
    ntc = T[:, 3] == 1
    naive_B = Y[T[:, 1] == 1, :5].mean(axis=0) - Y[ntc, :5].mean(axis=0)
    assert (naive_B < -0.6).all()  # the pseudobulk contrast attributes A's effect to B
    assert (res.fdr.loc["A"].iloc[:5] < 0.05).all()
    assert (res.log2fc.loc["A"].iloc[:5] * math.log(2) < -0.8).all()
    assert (res.fdr.loc["B"] >= 0.05).all() and (res.fdr.loc["C"] >= 0.05).all()
    np.testing.assert_allclose(res.log2fc.loc["B"].iloc[:5] * math.log(2), 0.0, atol=0.2)
    summ = res.summary.set_index("target_gene")
    assert summ.loc["A", "n_de_genes"] == 5 and summ.loc["B", "n_de_genes"] == 0
    assert set(res.de["target_gene"]) == {"A"}


def test_global_null_calls_nothing():
    rng = np.random.default_rng(3)
    T = _random_design(rng, n=600, k=8)
    Y = rng.normal(2.0, 0.5, size=(600, 40))
    lanes = np.where(rng.random(600) < 0.5, "L1", "L2")
    a = _membership_adata(T, Y, [f"T{j}" for j in range(8)], lanes=lanes)
    res = reg_mod.run_regression(a, _cfg(n_permutations=10))
    p = res.pval.to_numpy()
    assert 0.02 < (p < 0.05).mean() < 0.09  # empirical p-values are roughly uniform under the null
    assert res.info["n_significant_pairs"] <= 2


def test_global_fdr_scope_controls_the_call_set():
    """Many null targets: per-target BH lets spurious calls through, global BH does not."""
    rng = np.random.default_rng(5)
    k = 60
    T = _random_design(rng, n=1500, k=k, moi=4.0)
    Y = rng.normal(2.0, 0.5, size=(1500, 50))
    Y[T[:, 0] == 1, :10] -= 0.6  # one real target
    a = _membership_adata(T, Y, [f"T{j}" for j in range(k)])
    per_target = reg_mod.run_regression(a, _cfg(batch_key=None))
    glob = reg_mod.run_regression(a, _cfg(batch_key=None, fdr_scope="global"))
    for res in (per_target, glob):
        assert (res.fdr.loc["T0"].iloc[:10] < 0.05).all()
    false_pt = per_target.info["n_significant_pairs"] - 10
    false_gl = glob.info["n_significant_pairs"] - 10
    assert false_gl <= 2 and false_gl <= false_pt
    assert glob.info["empirical_fdr"] < 0.2
    np.testing.assert_allclose(per_target.pval.to_numpy(), glob.pval.to_numpy())  # same p-values, other family


def test_empirical_fdr_uses_the_reported_call_criteria():
    """With an effect-size cut the permuted calls get the same cut as the reported calls."""
    rng = np.random.default_rng(6)
    T = _random_design(rng, n=800, k=20, moi=3.0)
    Y = rng.normal(2.0, 0.5, size=(800, 40))
    Y[T[:, 0] == 1, :8] -= 0.5
    a = _membership_adata(T, Y, [f"T{j}" for j in range(20)])
    plain = reg_mod.run_regression(a, _cfg(batch_key=None))
    cut = reg_mod.run_regression(a, _cfg(batch_key=None, min_abs_log2fc=0.3))
    np.testing.assert_allclose(plain.fdr.to_numpy(), cut.fdr.to_numpy())  # same inference, different call rule
    assert cut.info["n_significant_pairs"] == int(((cut.fdr < 0.05) & (cut.log2fc.abs() > 0.3)).to_numpy().sum())
    assert cut.info["n_significant_pairs"] >= 8 and cut.info["n_significant_pairs"] <= plain.info["n_significant_pairs"]
    # null calls under the cut are a subset of the null calls without it
    assert cut.info["perm_mean_significant_pairs"] <= plain.info["perm_mean_significant_pairs"]
    huge = reg_mod.run_regression(a, _cfg(batch_key=None, min_abs_log2fc=10.0))
    assert huge.info["n_significant_pairs"] == 0 and huge.info["perm_mean_significant_pairs"] == 0
    assert np.isnan(huge.info["empirical_fdr"])


def test_modules_never_fall_back_to_pseudobulk():
    from perturbseq_pipeline import modules as modules_mod

    rng = np.random.default_rng(7)
    T = _random_design(rng, n=200, k=6)
    a = _membership_adata(T, rng.normal(1.0, 0.3, size=(200, 12)), [f"T{j}" for j in range(6)])
    cfg = _cfg()
    cfg.modules.effect_source = "regression"
    with pytest.raises(ValueError, match="no regression result"):
        modules_mod.compute_modules(a, cfg, regression=None)


def test_min_cells_reports_only_supported_targets_but_keeps_them_in_the_design():
    rng = np.random.default_rng(4)
    T = _random_design(rng, n=300, k=5)
    T[:, 4] = 0
    T[:3, 4] = 1  # target T4: 3 cells
    Y = rng.normal(1.0, 0.3, size=(300, 6))
    a = _membership_adata(T, Y, [f"T{j}" for j in range(5)])
    res = reg_mod.run_regression(a, _cfg(min_cells=10, batch_key=None))
    assert "T4" not in res.log2fc.index and res.info["n_targets_in_design"] == 5
    assert res.info["n_targets_reported"] == 4


def test_config_validation():
    inp = {"h5ad": "in.h5ad"}
    cfg = Config.from_dict({"input": inp, "regression": {"enabled": True}})
    with pytest.raises(ValueError, match="high_moi"):
        cfg.validate()
    cfg = Config.from_dict(
        {
            "input": inp,
            "guides": {"assignment_mode": "high_moi"},
            "modules": {"enabled": True, "effect_source": "regression"},
        }
    )
    with pytest.raises(ValueError, match="regression.enabled"):
        cfg.validate()
    cfg = Config.from_dict(
        {"input": inp, "guides": {"assignment_mode": "high_moi"}, "regression": {"enabled": True, "n_permutations": 0}}
    )
    with pytest.raises(ValueError, match="n_permutations"):
        cfg.validate()
    Config.from_dict(
        {"input": inp, "guides": {"assignment_mode": "high_moi"}, "regression": {"enabled": True}}
    ).validate()


def test_regression_end_to_end_moi4_with_modules_from_regression(tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300, moi=4)
    cfg = _base_cfg(data, tmp_path / "run", assignment_mode="high_moi", high_moi={"min_umi": 5})
    cfg.regression.enabled = True
    cfg.regression.genes = "all"
    cfg.regression.n_permutations = 5
    cfg.modules.effect_source = "regression"
    cfg.modules.program_enrichment.enabled = False
    cfg.validate()
    result = run_pipeline(cfg)
    tables = Path(cfg.run.outdir) / "tables"
    summ = pd.read_csv(tables / "regression_summary.csv").set_index("target_gene")
    assert set(summ.index) == set(KD_TARGETS + NULL_TARGETS)
    for t in KD_TARGETS:  # planted knockdown of the target's own gene
        assert summ.loc[t, "own_gene_fdr"] < 0.05 and summ.loc[t, "own_gene_log2fc"] < -1.0
    for t in NULL_TARGETS:
        assert summ.loc[t, "own_gene_fdr"] >= 0.05
    eff = pd.read_csv(tables / "regression_effect_matrix.csv", index_col=0)
    fdr = pd.read_csv(tables / "regression_fdr.csv", index_col=0)
    assert eff.shape == fdr.shape and set(eff.index) == set(summ.index)
    de = pd.read_csv(tables / "regression_de.csv")
    assert set(de["target_gene"]) >= set(KD_TARGETS)
    # the modules were built from the regression matrix
    mod_eff = pd.read_csv(tables / "effect_matrix.csv", index_col=0)
    pd.testing.assert_frame_equal(mod_eff, eff.loc[mod_eff.index, mod_eff.columns], check_exact=False, rtol=1e-9)
    status = result.module_status.set_index("module")["status"]
    assert status["regression"] == "completed" and status["modules"] == "completed"
    report = (Path(cfg.run.outdir) / "report.md").read_text()
    assert "Membership regression (high-MOI)" in report


def test_modules_skipped_when_regression_has_no_result(tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300, moi=4)
    cfg = _base_cfg(data, tmp_path / "run", assignment_mode="high_moi", high_moi={"min_umi": 5})
    cfg.regression.enabled = True
    cfg.regression.min_cells = 100_000  # no target qualifies -> the stage is skipped
    cfg.modules.effect_source = "regression"
    cfg.validate()
    result = run_pipeline(cfg)
    status = result.module_status.set_index("module")
    assert status.loc["regression", "status"] == "skipped"
    assert status.loc["modules", "status"] == "skipped" and "regression" in status.loc["modules", "note"]
    tables = Path(cfg.run.outdir) / "tables"
    assert not (tables / "effect_matrix.csv").exists() and not (tables / "regression_summary.csv").exists()
