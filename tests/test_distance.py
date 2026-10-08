"""Tests for perturbation distance vs control, permutation DistanceTest, and Perturbation Distance Space."""

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from perturbseq_pipeline.config import Config
from perturbseq_pipeline.distance import (
    compute_energy_distance,
    compute_mmd,
    energy_distance_from_cdist,
    distance_test_permutation,
    compute_perturbation_distance,
    compute_pcoa_coordinates,
    compute_distance_space,
    _sample_cell_indices,
)
from perturbseq_pipeline.perturbation import benjamini_hochberg


# Test 1 & 2: Mathematical Distance Properties


def test_energy_distance_near_zero_for_identical_distributions():
    """Energy distance should be near zero for identical synthetic distributions."""
    rng = np.random.default_rng(42)
    # Two independent samples from the same standard normal distribution
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 20))
    Y = rng.normal(loc=0.0, scale=1.0, size=(250, 20))
    edist = compute_energy_distance(X, Y)
    # For identical distributions in 20D with ~200 cells, empirical sample energy distance is small
    assert edist < 0.15, f"Expected small energy distance for null distribution, got {edist}"


def test_energy_distance_increases_for_separated_distributions():
    """Energy distance should increase as two distributions separate."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 10))
    Y_near = rng.normal(loc=0.5, scale=1.0, size=(100, 10))
    Y_far = rng.normal(loc=3.0, scale=1.0, size=(100, 10))
    d_near = compute_energy_distance(X, Y_near)
    d_far = compute_energy_distance(X, Y_far)
    assert d_far > d_near > 0.0, f"Expected d_far > d_near, got {d_far} vs {d_near}"


def test_mmd_distance_properties():
    """MMD should be near zero for identical distributions and increase with separation."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 10))
    Y_null = rng.normal(loc=0.0, scale=1.0, size=(100, 10))
    Y_shift = rng.normal(loc=2.0, scale=1.0, size=(100, 10))
    mmd_null = compute_mmd(X, Y_null)
    mmd_shift = compute_mmd(X, Y_shift)
    assert mmd_shift > mmd_null >= 0.0


# Test 3 & 4: Permutation DistanceTest


def test_distance_test_detects_strong_difference():
    """DistanceTest permutation test should yield a highly significant p-value for distinct distributions."""
    rng = np.random.default_rng(123)
    X = rng.normal(loc=2.0, scale=1.0, size=(60, 10))
    Y = rng.normal(loc=-2.0, scale=1.0, size=(100, 10))
    obs_dist, pval = distance_test_permutation(X, Y, n_permutations=200, seed=123)
    assert obs_dist > 0.5
    assert pval < 0.05, f"Expected p < 0.05 for strongly shifted groups, got {pval}"


def test_distance_test_null_distribution():
    """DistanceTest should yield a non-significant p-value when sampling from identical distributions."""
    rng = np.random.default_rng(999)
    X = rng.normal(loc=0.0, scale=1.0, size=(50, 10))
    Y = rng.normal(loc=0.0, scale=1.0, size=(100, 10))
    obs_dist, pval = distance_test_permutation(X, Y, n_permutations=200, seed=999)
    assert pval > 0.05, f"Expected non-significant p-value under null, got {pval}"


# Test 5: Benjamini-Hochberg Correction


def test_benjamini_hochberg_correction():
    """BH-FDR should preserve monotonicity and bound false discovery rates."""
    pvals = np.array([0.001, 0.01, 0.04, 0.5, 0.8])
    fdrs = benjamini_hochberg(pvals)
    assert len(fdrs) == len(pvals)
    assert np.all(fdrs >= pvals), "FDR values must be >= uncorrected p-values"
    assert np.all(np.diff(fdrs) >= -1e-12), "FDR values must be monotonically non-decreasing with sorted p-values"
    assert np.all((fdrs >= 0) & (fdrs <= 1.0))


# Test 6 & 7: Deterministic Bounded Sampling


def test_sampling_never_exceeds_max_cells():
    """Sampling should never return more cells than max_cells_per_target."""
    all_indices = np.arange(5000)
    rng = np.random.default_rng(123)
    sampled = _sample_cell_indices(all_indices, max_cells=1000, rng=rng)
    assert len(sampled) == 1000
    assert len(np.unique(sampled)) == 1000
    assert np.all(np.isin(sampled, all_indices))


def test_sampling_reproducibility():
    """Sampling must be exactly reproducible with the same random seed."""
    all_indices = np.arange(3000)
    sampled_1 = _sample_cell_indices(all_indices, max_cells=500, rng=np.random.default_rng(42))
    sampled_2 = _sample_cell_indices(all_indices, max_cells=500, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(sampled_1, sampled_2)


def test_sampling_stratification_by_lane():
    """Stratified sampling should proportionally represent batches/lanes."""
    all_indices = np.arange(1000)
    # 800 cells from lane1, 200 from lane2
    strata = np.array(["lane1"] * 800 + ["lane2"] * 200)
    sampled = _sample_cell_indices(all_indices, max_cells=200, rng=np.random.default_rng(123), strata=strata)
    assert len(sampled) == 200
    sampled_strata = strata[sampled]
    n_lane1 = (sampled_strata == "lane1").sum()
    n_lane2 = (sampled_strata == "lane2").sum()
    # ~80% (160) lane1, ~20% (40) lane2
    assert 140 <= n_lane1 <= 180
    assert 20 <= n_lane2 <= 60


# Test 8: Small Perturbation Groups Below min_cells Skipped


def _create_synthetic_anndata():
    """Helper to build a small synthetic AnnData with PCA and guide assignments."""
    rng = np.random.default_rng(123)
    n_cells = 300
    n_pcs = 20
    # 100 NTC cells, 80 TargetA (shifted), 80 TargetB (null), 40 TargetSmall (< min_cells if min_cells=50)
    classes = ["non-targeting"] * 100 + ["targeting"] * 200
    targets = ["ntc"] * 100 + ["TargetA"] * 80 + ["TargetB"] * 80 + ["TargetSmall"] * 40
    pca = rng.normal(0, 1, size=(n_cells, n_pcs))
    # Shift TargetA cells
    pca[100:180, 0] += 3.0
    obs = pd.DataFrame(
        {"target_gene": targets, "perturbation_class": classes, "lane_id": ["lane1"] * 150 + ["lane2"] * 150},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    expr = ad.AnnData(X=sparse.csr_matrix(np.zeros((n_cells, 50))), obs=obs, obsm={"X_pca": pca})
    return expr


def test_small_groups_skipped_and_reported():
    """Targets below min_cells should be skipped cleanly and recorded in skipped table."""
    expr = _create_synthetic_anndata()
    cfg = Config()
    cfg.distance.enabled = True  # optional stage, off by default
    cfg.distance.min_cells = 50
    cfg.distance.n_permutations = 100
    results = compute_perturbation_distance(expr, cfg)
    assert results is not None
    tested_targets = results.table["target_gene"].tolist()
    assert "TargetA" in tested_targets
    assert "TargetB" in tested_targets
    assert "TargetSmall" not in tested_targets
    assert not results.skipped.empty
    skipped_targets = results.skipped["target_gene"].tolist()
    assert "TargetSmall" in skipped_targets


# Test 9, 10, 11: DistanceSpace, PCoA, Neighbors, Phenotype Modules


def test_distance_space_symmetry_and_zero_diagonal():
    """DistanceSpace must produce a symmetric matrix with 0 diagonal."""
    expr = _create_synthetic_anndata()
    cfg = Config()
    cfg.distance_space.enabled = True  # optional stage, off by default
    cfg.distance_space.min_cells = 30
    res = compute_distance_space(expr, cfg)
    assert res is not None
    mat = res.distance_matrix
    assert not mat.empty
    assert (mat.index == mat.columns).all()
    # Check zero diagonal
    np.testing.assert_allclose(np.diag(mat.values), 0.0, atol=1e-10)
    # Check symmetry
    np.testing.assert_allclose(mat.values, mat.values.T, atol=1e-10)


def test_pcoa_coordinates_finite_and_positive_eigenvalues():
    """PCoA coordinates must be finite and properly handle non-Euclidean artifacts."""
    # Synthetic distance matrix
    D = np.array([[0.0, 1.2, 2.5, 3.0], [1.2, 0.0, 2.1, 2.8], [2.5, 2.1, 0.0, 1.0], [3.0, 2.8, 1.0, 0.0]])
    coords, evals = compute_pcoa_coordinates(D, n_components=3)
    assert coords.shape[0] == 4
    assert coords.shape[1] <= 3
    assert np.all(np.isfinite(coords))
    assert np.all(evals > 0)


def test_nearest_neighbors_ranking():
    """Nearest neighbors should return closest phenotypic targets excluding self."""
    expr = _create_synthetic_anndata()
    cfg = Config()
    cfg.distance_space.enabled = True  # optional stage, off by default
    cfg.distance_space.min_cells = 30
    cfg.distance_space.nearest_neighbors = 2
    res = compute_distance_space(expr, cfg)
    assert res is not None
    nn_df = res.neighbors
    assert not nn_df.empty
    assert set(nn_df.columns) == {"target", "neighbor", "distance", "rank"}
    # No self neighbors
    assert (nn_df["target"] != nn_df["neighbor"]).all()
    # Check ranks are 1 and 2
    assert set(nn_df["rank"].unique()) == {1, 2}


# Test 15 & 16: Pipeline Integration with Toggle Switches


def _create_pipeline_synthetic_h5ad(tmp_path):
    """Create a minimal synthetic h5ad file for full CLI pipeline runs."""
    rng = np.random.default_rng(42)
    n_cells = 400
    n_genes = 60
    # Counts
    counts = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)
    var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])
    var["feature_types"] = "Gene Expression"
    targets = ["non-targeting"] * 100 + ["Gene0"] * 100 + ["Gene1"] * 100 + ["Gene2"] * 100
    # Knock down Gene0 in Gene0-targeted cells
    counts[100:200, 0] = 0.0
    obs = pd.DataFrame(
        {
            "target_gene": targets,
            "perturbation_class": ["non-targeting"] * 100 + ["targeting"] * 300,
            "lane_id": ["L1"] * 200 + ["L2"] * 200,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    h5ad_path = tmp_path / "synthetic_input.h5ad"
    adata.write_h5ad(h5ad_path)
    return h5ad_path


def test_pipeline_run_with_distance_modules_disabled(tmp_path):
    """Pipeline should execute cleanly when distance, distance_space, and meta_analysis are disabled."""
    from perturbseq_pipeline.cli import run_pipeline

    h5ad_path = _create_pipeline_synthetic_h5ad(tmp_path)
    outdir = tmp_path / "out_disabled"
    cfg = Config.from_dict(
        {
            "run": {"name": "test_disabled", "outdir": str(outdir)},
            "input": {"mode": "h5ad", "h5ad": str(h5ad_path), "guide_obs_column": "target_gene"},
            "metadata": {"require_for_multilane": False},
            "qc": {"min_genes_per_cell": 5, "min_genes_final": 10, "max_pct_mt": 100},
            "cluster": {"n_top_genes": 40, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 10, "controls": ["ntc"], "primary_control": "ntc"},
            "enrichment": {"enabled": False},
            "modules": {"enabled": False},
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
            "distance": {"enabled": False},
            "distance_space": {"enabled": False},
            "meta_analysis": {"enabled": False},
        }
    )
    result = run_pipeline(cfg)
    assert result.h5ad.exists()
    assert result.report.exists()
    assert result.distance_table is None
    assert result.distance_space_results is None


def test_pipeline_run_with_distance_only(tmp_path):
    """Pipeline should execute with distance enabled and distance_space disabled."""
    from perturbseq_pipeline.cli import run_pipeline

    h5ad_path = _create_pipeline_synthetic_h5ad(tmp_path)
    outdir = tmp_path / "out_dist_only"
    cfg = Config.from_dict(
        {
            "run": {"name": "test_dist_only", "outdir": str(outdir)},
            "input": {"mode": "h5ad", "h5ad": str(h5ad_path), "guide_obs_column": "target_gene"},
            "metadata": {"require_for_multilane": False},
            "qc": {"min_genes_per_cell": 5, "min_genes_final": 10, "max_pct_mt": 100},
            "cluster": {"n_top_genes": 40, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 10, "controls": ["ntc"], "primary_control": "ntc"},
            "enrichment": {"enabled": False},
            "modules": {"enabled": False},
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
            "distance": {"enabled": True, "min_cells": 10, "n_permutations": 50},
            "distance_space": {"enabled": False},
            "meta_analysis": {"enabled": True},
        }
    )
    result = run_pipeline(cfg)
    assert result.h5ad.exists()
    assert result.report.exists()
    assert result.distance_table is not None
    assert (outdir / "tables" / "perturbation_distance.csv").exists()
    assert not (outdir / "tables" / "perturbation_distance_matrix.tsv").exists()


def test_pipeline_run_with_distance_and_distance_space(tmp_path):
    """Full pipeline run with distance, distance_space, and meta_analysis enabled."""
    from perturbseq_pipeline.cli import run_pipeline

    h5ad_path = _create_pipeline_synthetic_h5ad(tmp_path)
    outdir = tmp_path / "out_full"
    cfg = Config.from_dict(
        {
            "run": {"name": "test_full", "outdir": str(outdir)},
            "input": {"mode": "h5ad", "h5ad": str(h5ad_path), "guide_obs_column": "target_gene"},
            "metadata": {"require_for_multilane": False},
            "qc": {"min_genes_per_cell": 5, "min_genes_final": 10, "max_pct_mt": 100},
            "cluster": {"n_top_genes": 40, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 10, "controls": ["ntc"], "primary_control": "ntc"},
            "enrichment": {"enabled": False},
            "modules": {"enabled": False},
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
            "distance": {"enabled": True, "min_cells": 10, "n_permutations": 50},
            "distance_space": {"enabled": True, "min_cells": 10},
            "meta_analysis": {"enabled": True},
        }
    )
    result = run_pipeline(cfg)
    assert result.h5ad.exists()
    assert result.report.exists()
    tables_dir = outdir / "tables"
    assert (tables_dir / "perturbation_distance.csv").exists()
    assert (tables_dir / "perturbation_distance_matrix.tsv").exists()
    assert (tables_dir / "perturbation_space_coordinates.csv").exists()
    assert (tables_dir / "perturbation_neighbors.csv").exists()
    assert (tables_dir / "phenotype_modules.csv").exists()
    assert (tables_dir / "perturbation_meta.csv").exists()
    # Verify H5AD is lean (does NOT contain full distance matrix in uns)
    assert "perturbation_distance_matrix" not in result.adata.uns


# Regression: the 'other' control (all targeting cells) must exclude the focal target's
# own cells, otherwise every target is partly compared against itself.


def _other_control_adata(n_ntc: int = 100):
    expr = _create_synthetic_anndata()
    if n_ntc < 100:
        expr = expr[np.r_[np.arange(n_ntc), np.arange(100, expr.n_obs)]].copy()
    return expr


@pytest.mark.parametrize("how", ["primary_control", "ntc_fallback"])
def test_other_control_excludes_focal_target_cells(how):
    cfg = Config()
    cfg.distance.enabled = True
    cfg.distance.min_cells = 30
    cfg.distance.n_permutations = 200
    cfg.distance.max_control_cells = 10_000
    if how == "primary_control":
        expr = _other_control_adata()
        cfg.perturbation.primary_control = "other"
    else:
        # Too few NTC cells (< distance.min_cells) -> automatic fallback to 'other'.
        expr = _other_control_adata(n_ntc=10)
    results = compute_perturbation_distance(expr, cfg)
    assert results is not None
    assert results.control_used == "other"
    tbl = results.table.set_index("target_gene")
    n_targeting = int((expr.obs["perturbation_class"] == "targeting").sum())
    # Each target's control = all OTHER targeting cells (pool minus its own cells).
    assert tbl.loc["TargetA", "n_control"] == n_targeting - 80
    assert tbl.loc["TargetB", "n_control"] == n_targeting - 80
    assert tbl.loc["TargetSmall", "n_control"] == n_targeting - 40
    # Exact expectation: each target against the OTHER targeting cells only
    # (no subsampling here: every group is below max_cells_per_target / max_control_cells).
    pca = expr.obsm["X_pca"]
    tgt = expr.obs["target_gene"].to_numpy()
    targeting = (expr.obs["perturbation_class"] == "targeting").to_numpy()
    for name in ("TargetA", "TargetB"):
        expected = compute_energy_distance(pca[tgt == name], pca[targeting & (tgt != name)])
        assert tbl.loc[name, "energy_distance"] == pytest.approx(expected, rel=1e-6)
    # Shifted TargetA is far from the other targeting cells and significant.
    assert tbl.loc["TargetA", "energy_distance"] > 2 * tbl.loc["TargetB", "energy_distance"]
    assert tbl.loc["TargetA", "pvalue"] < 0.01


def test_other_control_worker_skips_target_when_too_few_controls_remain():
    """A target whose own cells are almost the whole pool is skipped, not tested against itself."""
    expr = _create_synthetic_anndata()
    keep = np.r_[np.arange(0, 10), np.arange(100, 180), np.arange(180, 200)]  # 10 NTC, 80 A, 20 B
    expr = expr[keep].copy()
    cfg = Config()
    cfg.distance.enabled = True
    cfg.distance.min_cells = 30
    cfg.distance.n_permutations = 50
    results = compute_perturbation_distance(expr, cfg)
    assert results.control_used == "other"
    assert "TargetA" not in results.table["target_gene"].tolist()
    reasons = dict(zip(results.skipped["target_gene"], results.skipped["reason"]))
    assert "control cells" in reasons["TargetA"]
