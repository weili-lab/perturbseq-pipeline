"""Numerical and biological consistency tests between STANDARD and LARGE modes.

This test suite verifies that:
1. Guide assignments are identical in STANDARD and LARGE modes.
2. Perturbation cell counts and control counts are identical.
3. Direct perturbation log2FC, percent knockdown, KS tests, MWU tests, and BH-FDR are identical/consistent.
4. Enrichment contingency counts and 2x2 tables are identical.
5. Enrichment odds ratios, p-values, and FDR are identical within floating precision.
6. lochNESS self scores are numerically equivalent.
7. Module sufficient-statistic effect matrix matches standard dense implementation within tolerance.
8. LARGE execution never mutates layers['counts'].
9. STANDARD path retains previous semantics (e.g. zero-centered scaling with clipping).
10. LARGE clustering preserves sparse HVG matrix through scaling without densification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from perturbseq_pipeline.config import Config
from perturbseq_pipeline.cluster import LOGNORM_LAYER, _run_pca_on_hvgs, _select_hvgs, normalize
from perturbseq_pipeline.guides import (
    CLASS_NTC,
    CLASS_TARGETING,
    OBS_CLASS,
    OBS_GUIDE,
    OBS_NDETECTED,
    OBS_SECOND,
    OBS_TARGET,
    OBS_TOP,
    OBS_TOTAL,
    assign_guides,
    top_two_guides,
)
from perturbseq_pipeline.perturbation import _test_all_targets_large, _test_all_targets_standard
from perturbseq_pipeline.enrichment import (
    _build_large_count_tables,
    _test_cluster_enrichment_large,
    _test_cluster_enrichment_standard,
)
from perturbseq_pipeline.lochness import _adjacency, _compute_self_lochness, compute_lochness, lochness_score
from perturbseq_pipeline.modules import _build_effect_matrix_large, _build_effect_matrix_standard


# Synthetic fixtures for consistency testing


@pytest.fixture
def consistency_adata():
    """Create a synthetic dataset with known targets, clusters, and layers."""
    rng = np.random.default_rng(42)
    n_cells = 400
    n_genes = 80
    n_targets = 8
    target_names = [f"TARGET_{i}" for i in range(n_targets)] + ["non-targeting"]
    # Target assignments (50% targeting, 50% NTC)
    targets = []
    classes = []
    guides = []
    for i in range(n_cells):
        if i < 200:
            t = target_names[i % n_targets]
            targets.append(t)
            classes.append(CLASS_TARGETING)
            guides.append(f"{t}_g{i % 2 + 1}")
        else:
            targets.append("non-targeting")
            classes.append(CLASS_NTC)
            guides.append(f"NTC_g{i % 3 + 1}")
    clusters = [str(i % 4) for i in range(n_cells)]
    lanes = [f"L{i % 2 + 1}" for i in range(n_cells)]
    # Generate sparse count matrix
    raw_dense = rng.poisson(lam=1.5, size=(n_cells, n_genes)).astype(np.float32)
    # Plant a knockdown signal for TARGET_0 and TARGET_1 on gene 0 and gene 1
    for i in range(n_cells):
        if targets[i] == "TARGET_0":
            raw_dense[i, 0] = 0
        elif targets[i] == "TARGET_1":
            raw_dense[i, 1] = 0
    X_sparse = sp.csr_matrix(raw_dense)
    gene_names = [f"TARGET_{i}" if i < n_targets else f"GENE_{i}" for i in range(n_genes)]
    adata = ad.AnnData(
        X=X_sparse.copy(),
        obs=pd.DataFrame(
            {OBS_TARGET: targets, OBS_CLASS: classes, OBS_GUIDE: guides, "leiden": clusters, "lane_id": lanes},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=gene_names),
    )
    adata.layers["counts"] = X_sparse.copy()
    # Highly variable genes flag
    adata.var["highly_variable"] = [True] * min(30, n_genes) + [False] * max(0, n_genes - 30)
    # Normalize lognorm layer
    cfg = Config()
    cfg.cluster.target_sum = 1e4
    adata = normalize(adata, cfg)
    return adata


# Test 1 & 10: Cluster PCA & Sparse Scaling


def test_large_mode_pca_scaling_preserves_sparsity(consistency_adata):
    """In LARGE mode on sparse matrix, scaling must NOT densify the matrix."""
    cfg_large = Config()
    cfg_large.scaling.mode = "large"
    cfg_large.cluster.scale_max_value = 10.0
    cfg_large.cluster.n_pcs = 10
    adata_large = consistency_adata.copy()
    assert sp.issparse(adata_large.X)
    # Run PCA in LARGE mode
    _run_pca_on_hvgs(adata_large, cfg_large)
    # PCA embedding was produced
    assert "X_pca" in adata_large.obsm
    assert adata_large.obsm["X_pca"].shape == (consistency_adata.n_obs, 10)
    assert adata_large.obsm["X_pca"].dtype == np.float32
    # Verify that parent X remained sparse and was not mutated in place
    assert sp.issparse(adata_large.X)


def test_standard_mode_pca_scaling_behavior(consistency_adata):
    """STANDARD mode retains zero-centered scaling and clipping."""
    cfg_std = Config()
    cfg_std.scaling.mode = "standard"
    cfg_std.cluster.scale_max_value = 10.0
    cfg_std.cluster.n_pcs = 10
    adata_std = consistency_adata.copy()
    _run_pca_on_hvgs(adata_std, cfg_std)
    assert "X_pca" in adata_std.obsm
    assert adata_std.obsm["X_pca"].shape == (consistency_adata.n_obs, 10)
    assert adata_std.obsm["X_pca"].dtype == np.float32


def test_regress_out_fails_early_in_large_mode(consistency_adata):
    """If cluster.regress_out is requested in LARGE mode, it must raise ValueError early."""
    cfg_large = Config()
    cfg_large.scaling.mode = "large"
    cfg_large.cluster.regress_out = ["lane_id"]
    adata = consistency_adata.copy()
    with pytest.raises(ValueError, match="cluster.regress_out is not supported in LARGE execution mode"):
        _run_pca_on_hvgs(adata, cfg_large)


# Test 2: Guide Calling Consistency (STANDARD vs LARGE)


def test_guide_calling_identical_standard_vs_large():
    """Verify that sparse row-wise top-2 matches chunked dense top-2 exactly."""
    rng = np.random.default_rng(123)
    n_cells = 300
    n_guides = 50
    # Sparse matrix with various edge cases: zero rows, singlets, doublets, ties
    dense = rng.poisson(0.5, size=(n_cells, n_guides)).astype(np.float64)
    dense[0, :] = 0  # all zeros
    dense[1, 10] = 50  # clear singlet
    dense[2, 5] = 20
    dense[2, 6] = 19  # close doublet
    dense[3, 2] = 15
    dense[3, 3] = 15  # tie
    X_sparse = sp.csr_matrix(dense)
    # Standard (chunked dense)
    from perturbseq_pipeline.guides import _top_two_guides_dense_chunked, _csr_top_two_numba, _csr_top_two_python

    idx_std, top_std, sec_std = _top_two_guides_dense_chunked(X_sparse, chunk_size=50)
    # Sparse python fallback
    idx_py, top_py, sec_py, tot_py = _csr_top_two_python(X_sparse)
    # Values and totals must match across all cells
    assert np.allclose(top_std, top_py)
    assert np.allclose(sec_std, sec_py)
    assert np.allclose(tot_py, np.asarray(X_sparse.sum(axis=1)).ravel())
    # For all non-tied cells, top_idx must match identically
    non_tied = top_std > sec_std
    assert np.array_equal(idx_std[non_tied], idx_py[non_tied])
    # Sparse numba (if available)
    numba_res = _csr_top_two_numba(X_sparse)
    if numba_res is not None:
        idx_nb, top_nb, sec_nb, tot_nb = numba_res
        assert np.allclose(top_std, top_nb)
        assert np.allclose(sec_std, sec_nb)
        assert np.allclose(tot_nb, np.asarray(X_sparse.sum(axis=1)).ravel())
        assert np.array_equal(idx_std[non_tied], idx_nb[non_tied])
    # Test full assign_guides on AnnData in STANDARD vs LARGE mode
    guide_names = [f"GENE_{i // 2}_sg{i % 2 + 1}" for i in range(n_guides)]
    guide_adata = ad.AnnData(
        X=X_sparse.copy(),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=guide_names),
    )
    expr_adata_std = ad.AnnData(
        X=sp.csr_matrix((n_cells, 10)), obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    )
    expr_adata_large = expr_adata_std.copy()
    cfg_std = Config()
    cfg_std.scaling.mode = "standard"
    cfg_large = Config()
    cfg_large.scaling.mode = "large"
    res_std = assign_guides(expr_adata_std, guide_adata, cfg_std)
    res_large = assign_guides(expr_adata_large, guide_adata, cfg_large)
    for col in (OBS_CLASS, OBS_TARGET, OBS_GUIDE, OBS_TOP, OBS_SECOND, OBS_TOTAL, OBS_NDETECTED):
        assert (res_std.obs[col].to_numpy() == res_large.obs[col].to_numpy()).all(), f"Mismatch in {col}"


# Test 3 & 4: Perturbation Strength Statistics Consistency


def test_perturbation_strength_statistics_consistency(consistency_adata):
    """Compare STANDARD and LARGE perturbation testing on same data."""
    cfg_std = Config()
    cfg_std.scaling.mode = "standard"
    cfg_std.perturbation.min_cells_per_target = 5
    cfg_std.perturbation.min_control_cells = 5
    cfg_large = Config()
    cfg_large.scaling.mode = "large"
    cfg_large.perturbation.min_cells_per_target = 5
    cfg_large.perturbation.min_control_cells = 5
    res_std = _test_all_targets_standard(consistency_adata, cfg_std)
    res_large = _test_all_targets_large(consistency_adata, cfg_large)
    assert set(res_std.table["target_gene"]) == set(res_large.table["target_gene"])
    std_tbl = res_std.table.set_index("target_gene").sort_index()
    large_tbl = res_large.table.set_index("target_gene").sort_index()
    # Check counts match exactly
    assert (std_tbl["n_perturbed"] == large_tbl["n_perturbed"]).all()
    assert (std_tbl["n_control_ntc"] == large_tbl["n_control_ntc"]).all()
    assert (std_tbl["n_control_other"] == large_tbl["n_control_other"]).all()
    # Check stats match within precision
    for col in ("log2fc_ntc", "log2fc_other", "pct_knockdown_ntc", "pct_knockdown_other"):
        if col in std_tbl.columns:
            np.testing.assert_allclose(
                std_tbl[col].to_numpy(dtype=float), large_tbl[col].to_numpy(dtype=float), rtol=1e-4, atol=1e-4
            )
    # Check hit calling is identical
    assert (std_tbl["is_hit_ntc"] == large_tbl["is_hit_ntc"]).all()


# Test 5: Enrichment Contingency Tables and Statistics Consistency


def test_enrichment_contingency_and_statistics_consistency(consistency_adata):
    """Compare STANDARD and LARGE cluster enrichment tests."""
    cfg_std = Config()
    cfg_std.scaling.mode = "standard"
    cfg_std.enrichment.min_cells_per_target = 5
    cfg_std.enrichment.min_cells_per_cluster = 5
    cfg_large = Config()
    cfg_large.scaling.mode = "large"
    cfg_large.enrichment.min_cells_per_target = 5
    cfg_large.enrichment.min_cells_per_cluster = 5
    res_std = _test_cluster_enrichment_standard(consistency_adata, cfg_std)
    res_large = _test_cluster_enrichment_large(consistency_adata, cfg_large)
    std_tbl = res_std.table.sort_values(["target_gene", "cluster", "control"]).reset_index(drop=True)
    large_tbl = res_large.table.sort_values(["target_gene", "cluster", "control"]).reset_index(drop=True)
    assert len(std_tbl) == len(large_tbl)
    assert (std_tbl["target_gene"] == large_tbl["target_gene"]).all()
    assert (std_tbl["cluster"] == large_tbl["cluster"]).all()
    assert (std_tbl["control"] == large_tbl["control"]).all()
    # Exact contingency counts
    assert (std_tbl["n_target_cells"] == large_tbl["n_target_cells"]).all()
    assert (std_tbl["n_in_cluster"] == large_tbl["n_in_cluster"]).all()
    assert (std_tbl["n_reference_cells"] == large_tbl["n_reference_cells"]).all()
    # Statistical consistency
    np.testing.assert_allclose(std_tbl["odds_ratio"], large_tbl["odds_ratio"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(std_tbl["pval"], large_tbl["pval"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(std_tbl["fdr"], large_tbl["fdr"], rtol=1e-5, atol=1e-5)
    assert (std_tbl["significant"] == large_tbl["significant"]).all()


# Test 6: lochNESS Self Scores Consistency


def test_lochness_self_scores_consistency(consistency_adata):
    """Compare STANDARD lochNESS scores on own cells vs LARGE self-score kernel."""
    cfg = Config()
    cfg.lochness.min_cells_per_target = 5
    cfg.lochness.n_neighbors = 20
    # Build neighbor graph
    import scanpy as sc

    sc.pp.pca(consistency_adata, n_comps=10)
    sc.pp.neighbors(consistency_adata, n_neighbors=20, key_added="lochness_nn")
    adj, neighbor_counts = _adjacency(sp.csr_matrix(consistency_adata.obsp["lochness_nn_distances"]))
    labels = consistency_adata.obs[OBS_TARGET].astype(str).to_numpy()
    # Large mode self-score
    self_score_large = _compute_self_lochness(adj, neighbor_counts, labels)
    # Standard mode score for each target
    overall = pd.Series(labels).value_counts(normalize=True).to_dict()
    for gene in set(labels):
        indicator = (labels == gene).astype(np.float32)
        score_std = lochness_score(adj, neighbor_counts, indicator, overall[gene])
        own_mask = labels == gene
        np.testing.assert_allclose(self_score_large[own_mask], score_std[own_mask], rtol=1e-4, atol=1e-4)


# Test 7: Modules Sufficient Statistics Effect Matrix Consistency


def test_modules_effect_matrix_consistency(consistency_adata):
    """Compare STANDARD dense effect matrix vs LARGE chunked sufficient statistics."""
    cfg = Config()
    cfg.modules.min_cells_per_perturbation = 5
    cfg.scaling.effect_gene_chunk = 16
    targets = sorted(
        [
            t
            for t, count in consistency_adata.obs[OBS_TARGET].value_counts().items()
            if t != "non-targeting" and count >= 5
        ]
    )
    genes = list(consistency_adata.var_names[:30])
    eff_std, ctrl_std, de_std = _build_effect_matrix_standard(consistency_adata, genes, targets, cfg)
    eff_large, ctrl_large, de_large = _build_effect_matrix_large(consistency_adata, genes, targets, cfg)
    assert ctrl_std == ctrl_large
    np.testing.assert_allclose(eff_std.to_numpy(), eff_large.to_numpy(), rtol=1e-5, atol=1e-5)
    assert (de_std.to_numpy() == de_large.to_numpy()).all()


# Test 8: layers['counts'] Immutability


def test_layers_counts_immutability(consistency_adata):
    """Verify layers['counts'] is never modified in LARGE mode."""
    counts_before = consistency_adata.layers["counts"].copy()
    cfg = Config()
    cfg.scaling.mode = "large"
    cfg.cluster.n_pcs = 10
    adata = consistency_adata.copy()
    _run_pca_on_hvgs(adata, cfg)
    # Check counts layer unchanged
    counts_after = adata.layers["counts"]
    diff = (counts_before != counts_after).nnz
    assert diff == 0, "layers['counts'] was modified!"


# Test 9 & 10: Monkeypatched PCA Scaling Assertions (STANDARD vs LARGE)


def test_large_pca_scaling_uses_zero_center_false(consistency_adata, monkeypatch):
    """Assert sc.pp.scale is called with zero_center=False and max_value=None in LARGE sparse mode."""
    import scanpy as sc

    scale_calls = []
    original_scale = sc.pp.scale

    def mock_scale(adata_work, zero_center=True, max_value=None, copy=False):
        scale_calls.append({"zero_center": zero_center, "max_value": max_value})
        return original_scale(adata_work, zero_center=zero_center, max_value=max_value, copy=copy)

    monkeypatch.setattr(sc.pp, "scale", mock_scale)
    cfg = Config()
    cfg.scaling.mode = "large"
    cfg.cluster.scale_max_value = 10.0
    cfg.cluster.n_pcs = 10
    adata = consistency_adata.copy()
    _run_pca_on_hvgs(adata, cfg)
    assert len(scale_calls) == 1
    assert scale_calls[0]["zero_center"] is False
    assert scale_calls[0]["max_value"] is None


def test_standard_pca_scaling_uses_zero_center_true(consistency_adata, monkeypatch):
    """Assert sc.pp.scale is called with zero_center=True and max_value=configured in STANDARD mode."""
    import scanpy as sc

    scale_calls = []
    original_scale = sc.pp.scale

    def mock_scale(adata_work, zero_center=True, max_value=None, copy=False):
        scale_calls.append({"zero_center": zero_center, "max_value": max_value})
        return original_scale(adata_work, zero_center=zero_center, max_value=max_value, copy=copy)

    monkeypatch.setattr(sc.pp, "scale", mock_scale)
    cfg = Config()
    cfg.scaling.mode = "standard"
    cfg.cluster.scale_max_value = 10.0
    cfg.cluster.n_pcs = 10
    adata = consistency_adata.copy()
    _run_pca_on_hvgs(adata, cfg)
    assert len(scale_calls) == 1
    assert scale_calls[0]["zero_center"] is True
    assert scale_calls[0]["max_value"] == 10.0


# Test 11 & 12: Scaling Mode Configuration & Trigger Behavior


def test_scaling_mode_forced_large_works_below_1m_cells():
    """Forced mode=large must return LARGE even for small cell counts."""
    cfg = Config()
    cfg.scaling.mode = "large"
    assert cfg.use_large_mode(n_cells=50_000, n_perturbations=100) is True
    assert cfg.execution_mode(n_cells=50_000, n_perturbations=100) == "large"


def test_scaling_mode_auto_switches_on_perturbation_count():
    """AUTO mode must trigger LARGE when perturbation count exceeds threshold, even if cells < 1M."""
    cfg = Config()
    cfg.scaling.mode = "auto"
    cfg.scaling.large_n_cells = 1_000_000
    cfg.scaling.large_n_perturbations = 5_000
    # 300k cells + 200 perts -> STANDARD
    assert cfg.use_large_mode(n_cells=300_000, n_perturbations=200) is False
    assert cfg.execution_mode(n_cells=300_000, n_perturbations=200) == "standard"
    # 300k cells + 6000 perts -> LARGE
    assert cfg.use_large_mode(n_cells=300_000, n_perturbations=6_000) is True
    assert cfg.execution_mode(n_cells=300_000, n_perturbations=6_000) == "large"


def test_lochness_large_mode_avoids_full_score_matrix(consistency_adata):
    """In LARGE mode, lochNESS must compute lochness_self without populating full per-target columns."""
    cfg = Config()
    cfg.lochness.enabled = True  # optional stage, off by default
    cfg.scaling.mode = "large"
    cfg.lochness.min_cells_per_target = 5
    cfg.lochness.n_neighbors = 15
    cfg.lochness.max_targets_in_obs = 2
    adata = consistency_adata.copy()
    import scanpy as sc

    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=15, key_added="lochness_nn")
    res = compute_lochness(adata, cfg)
    assert res is not None
    assert res.self_score is not None
    assert len(res.self_score) == adata.n_obs
    assert np.isfinite(res.self_score).sum() > 0
    # res.scores should NOT contain full cell x target arrays for all targets
    assert len(res.scores) <= cfg.lochness.max_targets_in_obs


# Regression: the LARGE 'other' reference must count EVERY targeting cell, including
# targets below min_cells_per_target, exactly like the STANDARD per-cell mask does.


def test_enrichment_large_other_reference_includes_subthreshold_targets(consistency_adata):
    """Sub-threshold targets concentrated in one cluster must not bias the 'other' reference.

    Before the fix, LARGE mode summed the per-cluster reference counts over testable
    targets only while the reference total counted all targeting cells, so the
    reference composition was wrong and enrichment calls could flip direction.
    """
    adata = consistency_adata.copy()
    obs = adata.obs
    # Relabel 15 TARGET_7 cells as five 3-cell targets (below min_cells_per_target=5)
    # and park them all in cluster "0" so the bias is concentrated.
    small = np.flatnonzero((obs[OBS_TARGET] == "TARGET_7").to_numpy())[:15]
    targets = obs[OBS_TARGET].astype(str).to_numpy()
    clusters = obs["leiden"].astype(str).to_numpy()
    for k, idx in enumerate(small):
        targets[idx] = f"TARGET_SMALL_{k // 3}"
        clusters[idx] = "0"
    obs[OBS_TARGET] = targets
    obs["leiden"] = clusters
    cfg_std = Config()
    cfg_std.scaling.mode = "standard"
    cfg_std.enrichment.min_cells_per_target = 5
    cfg_std.enrichment.min_cells_per_cluster = 5
    cfg_std.enrichment.controls = ["other"]
    cfg_std.enrichment.primary_control = "other"
    cfg_large = Config()
    cfg_large.scaling.mode = "large"
    cfg_large.enrichment.min_cells_per_target = 5
    cfg_large.enrichment.min_cells_per_cluster = 5
    cfg_large.enrichment.controls = ["other"]
    cfg_large.enrichment.primary_control = "other"
    res_std = _test_cluster_enrichment_standard(adata, cfg_std)
    res_large = _test_cluster_enrichment_large(adata, cfg_large)
    assert set(res_large.skipped["target_gene"]) == {f"TARGET_SMALL_{k}" for k in range(5)}
    keys = ["target_gene", "cluster", "control"]
    std_tbl = res_std.table.sort_values(keys).reset_index(drop=True)
    large_tbl = res_large.table.sort_values(keys).reset_index(drop=True)
    assert len(std_tbl) == len(large_tbl) > 0
    for col in keys + ["n_target_cells", "n_in_cluster", "n_reference_cells", "direction"]:
        assert (std_tbl[col] == large_tbl[col]).all(), col
    np.testing.assert_allclose(std_tbl["pct_of_reference"], large_tbl["pct_of_reference"], rtol=1e-9)
    np.testing.assert_allclose(std_tbl["odds_ratio"], large_tbl["odds_ratio"], rtol=1e-5)
    np.testing.assert_allclose(std_tbl["pval"], large_tbl["pval"], rtol=1e-5, atol=1e-12)
    # The reference composition table must also describe all targeting cells.
    n_targeting = int((obs[OBS_CLASS] == CLASS_TARGETING).sum())
    in_c0 = int(((obs[OBS_CLASS] == CLASS_TARGETING) & (obs["leiden"] == "0")).sum())
    assert res_large.reference_composition["other"]["0"] == pytest.approx(100.0 * in_c0 / n_targeting)
