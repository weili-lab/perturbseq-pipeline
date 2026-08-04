"""End-to-end and unit tests.

The synthetic dataset carries a known ground truth (``KDGENE*`` are really
knocked down, ``NULLGENE*`` are not), so these tests check that the pipeline
recovers the right biology rather than merely running without error.

Run with::

    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))

from make_synthetic import KD_TARGETS, NULL_TARGETS, make_dataset  # noqa: E402

from perturbseq_pipeline.config import Config, GuideConfig  # noqa: E402
from perturbseq_pipeline.guides import (  # noqa: E402
    is_non_targeting,
    parse_target_genes,
    top_two_guides,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """Two synthetic lanes plus a sample metadata CSV."""
    out = tmp_path_factory.mktemp("synthetic")
    return {"dir": out, **make_dataset(out, n_lanes=2, n_cells=300)}


def _base_config(synthetic, outdir: Path, **overrides) -> Config:
    data = {
        "run": {"name": "test", "outdir": str(outdir)},
        "input": {"mtx_dirs": synthetic["lanes"]},
        "metadata": {"file": synthetic["metadata"]},
        "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
        "cluster": {"n_top_genes": 80, "n_pcs": 10},
        "perturbation": {"min_cells_per_target": 5, "top_n_report": 2},
    }
    for key, value in overrides.items():
        data.setdefault(key, {}).update(value)
    return Config.from_dict(data)


@pytest.fixture(scope="session")
def mtx_run(synthetic, tmp_path_factory):
    """One full 10x-mode run, reused by several assertions."""
    from perturbseq_pipeline.cli import run_pipeline

    outdir = tmp_path_factory.mktemp("run_mtx")
    return run_pipeline(_base_config(synthetic, outdir))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown config key"):
        Config.from_dict({"guides": {"not_a_key": 1}})


def test_config_requires_an_input():
    with pytest.raises(ValueError, match="No input given"):
        Config.from_dict({}).validate()


def test_config_rejects_primary_control_outside_controls():
    cfg = Config.from_dict(
        {
            "input": {"h5ad": "x.h5ad"},
            "perturbation": {"controls": ["other"], "primary_control": "ntc"},
        }
    )
    with pytest.raises(ValueError, match="primary_control"):
        cfg.validate()


def test_config_yaml_roundtrip(tmp_path):
    cfg = Config.from_dict({"input": {"h5ad": "a.h5ad"}, "guides": {"min_umi": 7}})
    path = tmp_path / "c.yaml"
    cfg.dump_yaml(path)
    assert Config.from_yaml(path).guides.min_umi == 7


# ---------------------------------------------------------------------------
# Guide calling
# ---------------------------------------------------------------------------


def test_target_parsing_handles_both_library_conventions():
    g = GuideConfig()
    ids = ["AFF4_P1P2_1", "AFF4-P1P2.2", "non_targeting_3", "non-targeting.7", "OR6T1_P1_2"]
    assert list(parse_target_genes(ids, g)) == ["AFF4", "AFF4", "non", "non", "OR6T1"]


def test_ntc_detection():
    g = GuideConfig()
    assert list(is_non_targeting(["non", "NTC", "AFF4", "scramble"], g)) == [
        True,
        True,
        False,
        True,
    ]


def test_target_regex_override():
    g = GuideConfig(target_regex=r"^(.+)_sg\d+$")
    assert list(parse_target_genes(["NKX2-5_sg1"], g)) == ["NKX2-5"]


def test_top_two_matches_brute_force():
    rng = np.random.default_rng(0)
    dense = rng.poisson(2, size=(400, 40)).astype(float)
    X = sp.csr_matrix(dense)
    idx, top, second = top_two_guides(X, chunk_size=57)  # chunk boundary != row count
    ordered = np.sort(dense, axis=1)
    assert np.allclose(top, ordered[:, -1])
    assert np.allclose(second, ordered[:, -2])
    assert np.allclose(dense[np.arange(400), idx], ordered[:, -1])


def test_top_two_handles_all_zero_rows():
    X = sp.csr_matrix(np.zeros((5, 3)))
    _, top, second = top_two_guides(X)
    assert np.all(top == 0) and np.all(second == 0)


# ---------------------------------------------------------------------------
# End-to-end: 10x MTX mode
# ---------------------------------------------------------------------------


def test_mtx_run_produces_all_deliverables(mtx_run):
    assert mtx_run.report.is_file(), "report.html missing"
    assert mtx_run.h5ad.is_file(), "processed h5ad missing"
    assert mtx_run.guide_h5ad is not None and mtx_run.guide_h5ad.is_file()
    assert (mtx_run.outdir / "logs" / "resolved_config.yaml").is_file()
    figures = list(mtx_run.figures_dir.rglob("*.png"))
    assert len(figures) > 15, f"expected many figures, got {len(figures)}"


def test_every_target_gets_a_figure_even_when_not_in_the_report(mtx_run):
    """All diagnostic figures land on disk; the report only embeds the top-N."""
    per_gene = list((mtx_run.figures_dir / "perturbation" / "per_gene").glob("*.png"))
    assert len(per_gene) == mtx_run.n_targets_tested
    assert mtx_run.n_targets_tested > 2, "test needs more targets than top_n_report"


def test_recovers_the_planted_knockdowns(mtx_run):
    tbl = mtx_run.perturbation_table.set_index("target_gene")
    for gene in KD_TARGETS:
        assert tbl.loc[gene, "is_hit_ntc"], f"{gene} should be called effective"
        assert tbl.loc[gene, "log2fc_ntc"] < -1, f"{gene} should show strong knockdown"
    for gene in NULL_TARGETS:
        assert not tbl.loc[gene, "is_hit_ntc"], f"{gene} is a null and must not be a hit"


def test_both_control_arms_are_reported(mtx_run):
    cols = mtx_run.perturbation_table.columns
    for control in ("ntc", "other"):
        assert f"log2fc_{control}" in cols
        assert f"ks_fdr_{control}" in cols
        assert f"is_hit_{control}" in cols


def test_metadata_is_merged_into_obs(mtx_run):
    obs = mtx_run.adata.obs
    for col in ("lane_id", "sample", "condition", "replicate"):
        assert col in obs.columns, f"metadata column {col} missing from obs"
    assert obs["lane_id"].nunique() == 2


def test_ambiguous_and_unassigned_cells_are_kept(mtx_run):
    classes = set(mtx_run.adata.obs["perturbation_class"].astype(str))
    assert {"ambiguous", "unassigned", "targeting", "non-targeting"} <= classes


def test_layers_follow_the_documented_contract(mtx_run):
    adata = mtx_run.adata
    assert "counts" in adata.layers and "lognorm" in adata.layers
    counts = adata.layers["counts"]
    dense = counts[:50].toarray() if sp.issparse(counts) else np.asarray(counts[:50])
    assert np.allclose(dense, np.round(dense)), "counts layer must hold raw integers"
    lognorm = adata.layers["lognorm"]
    ldense = lognorm[:50].toarray() if sp.issparse(lognorm) else np.asarray(lognorm[:50])
    assert ldense.max() < 50, "lognorm layer should hold log-scale values"


def test_report_is_self_contained(mtx_run):
    html = mtx_run.report.read_text()
    assert "{{" not in html and "{%" not in html, "unrendered template syntax"
    assert "data:image/png;base64" in html, "figures should be embedded"


# ---------------------------------------------------------------------------
# Sample metadata rules
# ---------------------------------------------------------------------------


def test_multilane_without_metadata_is_rejected(synthetic, tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "no_meta")
    cfg.metadata.file = None
    with pytest.raises(ValueError, match="sample metadata"):
        run_pipeline(cfg)


def test_metadata_missing_a_lane_is_rejected(synthetic, tmp_path):
    import pandas as pd

    from perturbseq_pipeline.cli import run_pipeline

    meta = pd.read_csv(synthetic["metadata"]).iloc[:1]
    bad = tmp_path / "partial_metadata.csv"
    meta.to_csv(bad, index=False)
    cfg = _base_config(synthetic, tmp_path / "bad_meta")
    cfg.metadata.file = str(bad)
    with pytest.raises(ValueError, match="missing row"):
        run_pipeline(cfg)


# ---------------------------------------------------------------------------
# End-to-end: the three h5ad layouts
# ---------------------------------------------------------------------------


def test_h5ad_mode_with_guide_features_in_var(synthetic, tmp_path):
    """Layout 1: one h5ad holding both gene-expression and guide features."""
    import scanpy as sc

    from perturbseq_pipeline.cli import run_pipeline

    combined = sc.read_10x_mtx(synthetic["lanes"]["L1"], gex_only=False, cache=False)
    combined.var_names_make_unique()
    path = tmp_path / "combined.h5ad"
    combined.write_h5ad(path)

    cfg = _base_config(synthetic, tmp_path / "run_var")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(path)
    cfg.metadata.file = None
    result = run_pipeline(cfg)
    assert result.n_targets_tested >= len(KD_TARGETS)
    tbl = result.perturbation_table.set_index("target_gene")
    assert tbl.loc[KD_TARGETS[0], "is_hit_ntc"]


def test_h5ad_mode_with_companion_guide_file(mtx_run, synthetic, tmp_path):
    """Layout 2: expression h5ad plus a separate guide count h5ad."""
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_companion")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(mtx_run.h5ad)
    cfg.input.guide_h5ad = str(mtx_run.guide_h5ad)
    cfg.input.counts_layer = "counts"
    cfg.metadata.file = None
    result = run_pipeline(cfg)
    assert result.n_effective >= 1


def test_h5ad_mode_with_precomputed_obs_labels(mtx_run, synthetic, tmp_path):
    """Layout 3: no guide matrix, only a per-cell genotype column."""
    import scanpy as sc

    from perturbseq_pipeline.cli import run_pipeline

    adata = sc.read_h5ad(mtx_run.h5ad)
    # Mimic the Seurat export convention: TARGET-SUFFIX.N
    adata.obs["genotype"] = [
        f"{t}-P1P2.1" for t in adata.obs["target_gene"].astype(str)
    ]
    path = tmp_path / "with_genotype.h5ad"
    adata.write_h5ad(path)

    cfg = _base_config(synthetic, tmp_path / "run_obs")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(path)
    cfg.input.guide_obs_column = "genotype"
    cfg.input.counts_layer = "counts"
    cfg.metadata.file = None
    result = run_pipeline(cfg)
    assert result.n_targets_tested >= len(KD_TARGETS)
    assert result.guide_h5ad is None, "no guide matrix exists in this layout"


def test_h5ad_without_any_guide_information_gives_a_clear_error(synthetic, tmp_path):
    import scanpy as sc

    from perturbseq_pipeline.cli import run_pipeline

    adata = sc.read_10x_mtx(synthetic["lanes"]["L1"], gex_only=True, cache=False)
    adata.var_names_make_unique()
    path = tmp_path / "no_guides.h5ad"
    adata.write_h5ad(path)

    cfg = _base_config(synthetic, tmp_path / "run_noguide")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(path)
    cfg.metadata.file = None
    with pytest.raises(ValueError, match="Could not find guide information"):
        run_pipeline(cfg)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_matches_statsmodels():
    from perturbseq_pipeline.perturbation import benjamini_hochberg

    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 50)
    assert np.allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def test_unexpressed_targets_are_reported_untestable_not_scored(synthetic, tmp_path):
    """A gene absent from control cells cannot show knockdown.

    Screens routinely include targets that are not expressed in the cell type
    (olfactory receptors in an ESC TF screen). Scoring them produces a huge
    meaningless fold change driven only by the pseudocount, so they must be
    reported as untestable instead.
    """
    import scanpy as sc

    from perturbseq_pipeline.cli import run_pipeline

    combined = sc.read_10x_mtx(synthetic["lanes"]["L1"], gex_only=False, cache=False)
    combined.var_names_make_unique()
    # Silence one target gene everywhere, mimicking an unexpressed control target.
    silent = NULL_TARGETS[0]
    col = combined.var_names.get_loc(silent)
    X = combined.X.tolil()
    X[:, col] = 0
    combined.X = X.tocsr()
    path = tmp_path / "silent.h5ad"
    combined.write_h5ad(path)

    cfg = _base_config(synthetic, tmp_path / "run_silent")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(path)
    cfg.metadata.file = None
    # Keep the silenced gene in the matrix so the expression guard is what
    # catches it, rather than the earlier min-cells-per-gene filter.
    cfg.qc.min_cells_per_gene = 0
    result = run_pipeline(cfg)

    tested = set(result.perturbation_table["target_gene"])
    assert silent not in tested, f"{silent} is unexpressed and must not be scored"
    skipped = pd.read_csv(result.outdir / "tables" / "skipped.csv")
    reason = skipped.loc[skipped["target_gene"] == silent, "reason"].iloc[0]
    assert "not detectably expressed" in reason


def test_fold_change_stays_bounded_for_complete_knockdown():
    """The pseudocount keeps a total knockdown finite and interpretable."""
    from perturbseq_pipeline.perturbation import compare_groups

    control = np.full(200, np.log1p(5.0))
    perturbed = np.zeros(200)
    stats = compare_groups(perturbed, control)
    assert np.isfinite(stats["log2fc"])
    assert -12 < stats["log2fc"] < -5, stats["log2fc"]
    assert stats["pct_knockdown"] > 99


def test_fold_change_is_accurate_for_expressed_genes():
    """The pseudocount must not distort real, well-expressed genes."""
    from perturbseq_pipeline.perturbation import compare_groups

    control = np.full(200, np.log1p(8.0))
    perturbed = np.full(200, np.log1p(2.0))
    stats = compare_groups(perturbed, control)
    assert abs(stats["log2fc"] - np.log2(2.0 / 8.0)) < 0.01


def test_benjamini_hochberg_tolerates_nan():
    from perturbseq_pipeline.perturbation import benjamini_hochberg

    q = benjamini_hochberg([0.01, np.nan, 0.5])
    assert np.isnan(q[1]) and not np.isnan(q[0])
