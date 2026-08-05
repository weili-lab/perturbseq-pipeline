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


def test_results_archive_bundles_everything_except_matrices(mtx_run):
    """The archive must carry the readable outputs but not the .h5ad files."""
    import tarfile

    assert mtx_run.archive is not None and mtx_run.archive.is_file()
    with tarfile.open(mtx_run.archive, "r:gz") as tar:
        names = tar.getnames()

    assert not any(n.endswith(".h5ad") for n in names), "h5ad must be excluded"
    assert not any(n.endswith(".tar.gz") for n in names), "archive must not nest itself"

    tails = {n.split("/", 1)[-1] for n in names}
    assert "report.html" in tails
    assert any(t.startswith("figures/") and t.endswith(".png") for t in tails)
    assert any(t.startswith("tables/") and t.endswith(".csv") for t in tails)
    assert "logs/run.log" in tails

    # Everything unpacks under a single directory named for the run.
    roots = {n.split("/", 1)[0] for n in names}
    assert roots == {"test"}, roots

    # Completeness: every file in the run directory that is not excluded must
    # be in the archive — "all files except the matrices", with nothing lost.
    on_disk = {
        str(p.relative_to(mtx_run.outdir))
        for p in mtx_run.outdir.rglob("*")
        if p.is_file()
        and not p.name.endswith((".h5ad", ".h5", ".loom", ".tar.gz"))
    }
    assert on_disk - tails == set(), f"missing from archive: {sorted(on_disk - tails)}"


def test_archive_can_be_disabled(synthetic, tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_noarchive")
    cfg.output.archive = False
    result = run_pipeline(cfg)
    assert result.archive is None
    assert not list(result.outdir.glob("*.tar.gz"))


def test_archive_exclude_patterns_are_honoured(synthetic, tmp_path):
    import tarfile

    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_excl")
    cfg.output.archive_name = "custom_bundle.tar.gz"
    cfg.output.archive_exclude = ["*.h5ad", "*.tar.gz", "*.png"]
    result = run_pipeline(cfg)

    assert result.archive.name == "custom_bundle.tar.gz"
    with tarfile.open(result.archive, "r:gz") as tar:
        names = tar.getnames()
    assert not any(n.endswith(".png") for n in names), "figures should be excluded here"
    assert any(n.endswith("report.html") for n in names)


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


def test_harmony_returns_a_correctly_shaped_embedding():
    """Guards a silent orientation flip in the Harmony integration.

    harmonypy 2.0 changed ``Z_corr`` from (components x cells) to
    (cells x components), while scanpy's wrapper still transposes it — which
    yields an embedding of the wrong shape and fails only once anndata tries to
    store it, part-way through a long run.
    """
    pytest.importorskip("harmonypy")
    import anndata as ad

    from perturbseq_pipeline.cluster import _run_harmony

    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.normal(size=(300, 20)).astype("float32"))
    adata.obs["lane_id"] = pd.Categorical(["L1"] * 150 + ["L2"] * 150)
    adata.obsm["X_pca"] = rng.normal(size=(300, 15))

    cfg = Config()
    cfg.cluster.batch_key = "lane_id"
    key = _run_harmony(adata, cfg)
    assert key == "X_pca_harmony"
    assert adata.obsm[key].shape == (300, 15)


def test_harmony_is_skipped_for_a_single_batch():
    pytest.importorskip("harmonypy")
    import anndata as ad

    from perturbseq_pipeline.cluster import _run_harmony

    adata = ad.AnnData(X=np.zeros((10, 5), dtype="float32"))
    adata.obs["lane_id"] = pd.Categorical(["L1"] * 10)
    cfg = Config()
    cfg.cluster.batch_key = "lane_id"
    assert _run_harmony(adata, cfg) is None


# ---------------------------------------------------------------------------
# Cluster enrichment
# ---------------------------------------------------------------------------


def test_enrichment_outputs_are_produced(mtx_run):
    per_target = list((mtx_run.figures_dir / "enrichment" / "per_target").glob("*.png"))
    overview = list((mtx_run.figures_dir / "enrichment").glob("*.png"))
    assert per_target, "expected one enrichment figure per target"
    assert {p.stem for p in overview} >= {
        "enrichment_heatmap",
        "enrichment_volcano",
        "enrichment_composition",
        "enrichment_effect_magnitude",
    }
    tbl = pd.read_csv(mtx_run.outdir / "tables" / "enrichment_full.csv")
    for col in (
        "target_gene",
        "cluster",
        "control",
        "odds_ratio",
        "log2_odds_ratio",
        "pval",
        "fdr",
        "significant",
        "low_power",
        "direction",
    ):
        assert col in tbl.columns, f"missing column {col}"


def test_enrichment_reports_both_control_arms(mtx_run):
    tbl = pd.read_csv(mtx_run.outdir / "tables" / "enrichment_full.csv")
    assert set(tbl["control"]) == {"ntc", "other"}
    # Hit calling is confined to the primary arm, so FDR families stay separate.
    assert set(tbl.loc[tbl["significant"], "control"]) <= {"other"}


def test_enrichment_recovers_a_planted_cluster_association(synthetic, tmp_path):
    """A perturbation that drives cells into their own cluster must be found.

    The synthetic knockdown targets shift expression enough that clustering
    separates them, so each KD target should be significantly enriched in at
    least one cluster.
    """
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_enrich")
    result = run_pipeline(cfg)
    tbl = pd.read_csv(result.outdir / "tables" / "enrichment_full.csv")
    sig = tbl[tbl["significant"]]
    for gene in KD_TARGETS:
        assert gene in set(sig["target_gene"]), f"{gene} should be enriched somewhere"


def test_enrichment_odds_ratios_are_finite(mtx_run):
    """The Haldane-Anscombe correction must keep zero-count cells finite."""
    tbl = pd.read_csv(mtx_run.outdir / "tables" / "enrichment_full.csv")
    assert np.isfinite(tbl["odds_ratio"]).all(), "odds ratios must never be inf/NaN"
    assert np.isfinite(tbl["log2_odds_ratio"]).all()
    assert (tbl["odds_ratio"] > 0).all()


def test_enrichment_can_be_disabled(synthetic, tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_noenrich")
    cfg.enrichment.enabled = False
    result = run_pipeline(cfg)
    assert not (result.outdir / "figures" / "enrichment").exists()
    assert not (result.outdir / "tables" / "enrichment_full.csv").exists()
    assert result.report.is_file()


def test_enrichment_stratified_runs_across_lanes(synthetic, tmp_path):
    """Cochran-Mantel-Haenszel path must work on a genuinely multi-lane run."""
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_cmh")
    cfg.enrichment.stratify_by = "lane_id"
    result = run_pipeline(cfg)
    tbl = pd.read_csv(result.outdir / "tables" / "enrichment_full.csv")
    assert len(tbl) > 0
    assert np.isfinite(tbl["odds_ratio"]).all()


def test_fisher_direction_matches_the_counts():
    """Direction must follow the observed percentages, not the p-value."""
    from perturbseq_pipeline.enrichment import _odds_ratio

    # Target over-represented: 50/100 vs 10/100 -> odds ratio well above 1.
    assert _odds_ratio(50, 50, 10, 90, 0.5) > 1
    # Target under-represented -> below 1.
    assert _odds_ratio(5, 95, 40, 60, 0.5) < 1
    # Zero counts stay finite thanks to the pseudocount.
    assert np.isfinite(_odds_ratio(0, 100, 0, 100, 0.5))


def test_guide_concordance_follows_the_observed_direction():
    """A depletion supported by every guide must count as full agreement.

    Judging agreement only in the enrichment direction would report real
    depletions as 0 guides agreeing — the opposite of the truth.
    """
    from perturbseq_pipeline.enrichment import _guide_concordance
    from perturbseq_pipeline.guides import (
        CLASS_TARGETING,
        OBS_CLASS,
        OBS_GUIDE,
        OBS_TARGET,
    )

    # Three guides for GENE, none of whose cells land in cluster "9",
    # against a reference that puts 20% of its cells there.
    obs = pd.DataFrame(
        {
            OBS_GUIDE: ["g1"] * 10 + ["g2"] * 10 + ["g3"] * 10,
            OBS_TARGET: ["GENE"] * 30,
            OBS_CLASS: [CLASS_TARGETING] * 30,
            "leiden": ["1"] * 30,
        }
    )
    conc, tested = _guide_concordance(
        obs, "GENE", "9", "leiden", ref_fraction=0.20, min_cells=5, direction="depleted"
    )
    assert (conc, tested) == (3, 3), "all three guides support the depletion"

    conc, tested = _guide_concordance(
        obs, "GENE", "9", "leiden", ref_fraction=0.20, min_cells=5, direction="enriched"
    )
    assert (conc, tested) == (0, 3), "none support an enrichment"


def test_omnibus_tolerates_empty_rows_and_columns():
    """A cluster with no targeting cells must not kill the whole stage.

    scipy's chi2_contingency raises on any zero margin, and both kinds occur in
    real data: a small cluster can be entirely ambiguous cells, and a target's
    cells can all fall in clusters too small to test. This surfaced on the
    THP-1 run, where 63 clusters made an empty column certain.
    """
    from perturbseq_pipeline.enrichment import omnibus_test

    tbl = pd.DataFrame(
        [[50, 0, 10], [5, 0, 40], [0, 0, 0]],
        index=["A", "B", "C"],
        columns=["c1", "c2", "c3"],
    )
    res = omnibus_test(tbl, n_permutations=100, seed=0)
    assert res, "should return a result rather than raising"
    assert np.isfinite(res["chi2"])
    assert res["p_permutation"] < 0.5


def test_omnibus_returns_empty_when_nothing_is_left():
    from perturbseq_pipeline.enrichment import omnibus_test

    tbl = pd.DataFrame([[5, 0], [0, 0]], index=["A", "B"], columns=["c1", "c2"])
    assert omnibus_test(tbl, n_permutations=10, seed=0) == {}


def test_omnibus_detects_association_and_null():
    from perturbseq_pipeline.enrichment import omnibus_test

    # Strong association: each target lives in its own cluster.
    assoc = pd.DataFrame([[100, 1], [1, 100]], index=["A", "B"], columns=["c1", "c2"])
    res = omnibus_test(assoc, n_permutations=200, seed=0)
    assert res["p_permutation"] < 0.05

    # No association: identical profiles.
    null = pd.DataFrame([[50, 50], [50, 50]], index=["A", "B"], columns=["c1", "c2"])
    res_null = omnibus_test(null, n_permutations=200, seed=0)
    assert res_null["p_permutation"] > 0.05


# ---------------------------------------------------------------------------
# Per-cell perturbation scores (pertps / PS_python)
# ---------------------------------------------------------------------------

pertps_required = pytest.mark.skipif(
    not __import__("perturbseq_pipeline.ps_score", fromlist=["x"]).pertps_available(),
    reason="optional pertps package not installed",
)


@pertps_required
def test_ps_scores_are_produced(mtx_run):
    summary = pd.read_csv(mtx_run.outdir / "tables" / "ps_score.csv")
    assert len(summary) > 0
    for col in (
        "target_gene",
        "n_perturbed_cells",
        "mean_ps",
        "pct_successful_kd",
        "pct_escaper",
    ):
        assert col in summary.columns
    # Quadrant fractions describe a partition, so they must sum to 100%.
    total = (
        summary["pct_successful_kd"]
        + summary["pct_escaper"]
        + summary["pct_non_responder"]
        + summary["pct_low_signal"]
    )
    assert np.allclose(total, 100.0, atol=0.01), total.tolist()


@pertps_required
def test_ps_scores_land_in_obs(mtx_run):
    obs = mtx_run.adata.obs
    assert "ps_score" in obs.columns and "ps_quadrant" in obs.columns
    scored = obs["ps_score"].notna()
    assert scored.sum() > 0
    vals = obs.loc[scored, "ps_score"].to_numpy(dtype=float)
    assert vals.min() >= 0 and vals.max() <= 1, "scores must be in [0, 1]"


@pertps_required
def test_ps_reports_a_control_baseline(mtx_run):
    """Every knockdown percentage needs the control's own rate beside it."""
    summary = pd.read_csv(mtx_run.outdir / "tables" / "ps_score.csv")
    for col in ("pct_controls_called_kd", "net_pct_kd", "expression_cut"):
        assert col in summary.columns
    net = summary["pct_successful_kd"] - summary["pct_controls_called_kd"]
    assert np.allclose(net, summary["net_pct_kd"], atol=0.01)


@pertps_required
def test_default_expression_cut_is_not_degenerate(mtx_run):
    """The mean cut must not collapse to zero the way the median does.

    Single-cell counts are zero-inflated, so a control median of 0 turns "low
    expression" into "exactly zero" and makes the quadrant split meaningless.
    """
    summary = pd.read_csv(mtx_run.outdir / "tables" / "ps_score.csv")
    assert (summary["expression_cut_method"] == "mean").all()
    assert (summary["expression_cut"] > 0).any(), "mean cut should not be all zeros"


def test_expression_cut_methods_are_selectable():
    from perturbseq_pipeline.config import Config
    from perturbseq_pipeline.ps_score import _expression_cut

    ref = pd.Series([0.0, 0.0, 0.0, 1.0, 3.0])  # zero-inflated, median 0
    cfg = Config()
    cfg.ps_score.expression_cut = "median"
    assert _expression_cut(ref, cfg.ps_score) == 0.0
    cfg.ps_score.expression_cut = "mean"
    assert _expression_cut(ref, cfg.ps_score) == pytest.approx(0.8)
    cfg.ps_score.expression_cut = "quantile"
    cfg.ps_score.expression_cut_quantile = 0.75
    assert _expression_cut(ref, cfg.ps_score) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="expression_cut"):
        bad = Config.from_dict(
            {"input": {"h5ad": "x"}, "ps_score": {"expression_cut": "nonsense"}}
        )
        bad.validate()


@pertps_required
def test_lda_embedding_is_built_and_plotted(mtx_run):
    """PS_python's supervised LDA map must be produced, one figure per target."""
    adata = mtx_run.adata
    assert "X_lda_umap" in adata.obsm, "LDA embedding missing from obsm"
    coords = np.asarray(adata.obsm["X_lda_umap"], dtype=float)
    assert coords.shape == (adata.n_obs, 2)
    placed = np.isfinite(coords).all(axis=1)
    assert placed.sum() > 0, "no cells placed in the LDA embedding"
    # Cells outside the trained classes (ambiguous/unassigned) get no
    # coordinates by design, so this must not be all cells.
    assert "lda_label" in adata.obs.columns

    summary = pd.read_csv(mtx_run.outdir / "tables" / "ps_score.csv")
    lda_figs = list((mtx_run.figures_dir / "ps_score" / "lda").glob("*.png"))
    assert len(lda_figs) == len(summary), "one LDA figure per scored target"
    overview = {p.stem for p in (mtx_run.figures_dir / "ps_score").glob("*.png")}
    assert {"ps_lda_overview", "ps_lda_high_confidence"} <= overview


def test_lda_embedding_can_be_disabled(synthetic, tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_nolda")
    cfg.ps_score.compute_lda_umap = False
    result = run_pipeline(cfg)
    assert "X_lda_umap" not in result.adata.obsm
    assert not (result.outdir / "figures" / "ps_score" / "lda").exists()
    assert result.report.is_file()


@pertps_required
def test_ps_figures_cover_every_scored_target(mtx_run):
    summary = pd.read_csv(mtx_run.outdir / "tables" / "ps_score.csv")
    figs = list((mtx_run.figures_dir / "ps_score" / "per_target").glob("*.png"))
    assert len(figs) == len(summary)


@pertps_required
def test_ps_skips_targets_not_expressed_in_controls(synthetic, tmp_path):
    """An unexpressed gene must not top the knockdown-efficiency ranking.

    With zero expression the control median is 0, so every cell falls on the
    "low expression" side of the quadrant split and any high score is misread
    as a successful knockdown.
    """
    import scanpy as sc

    from perturbseq_pipeline.cli import run_pipeline

    combined = sc.read_10x_mtx(synthetic["lanes"]["L1"], gex_only=False, cache=False)
    combined.var_names_make_unique()
    silent = NULL_TARGETS[0]
    col = combined.var_names.get_loc(silent)
    X = combined.X.tolil()
    X[:, col] = 0
    combined.X = X.tocsr()
    path = tmp_path / "silent_ps.h5ad"
    combined.write_h5ad(path)

    cfg = _base_config(synthetic, tmp_path / "run_ps_silent")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(path)
    cfg.metadata.file = None
    cfg.qc.min_cells_per_gene = 0
    result = run_pipeline(cfg)

    summary_path = result.outdir / "tables" / "ps_score.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        assert silent not in set(summary["target_gene"])


def test_pipeline_completes_without_pertps(synthetic, tmp_path, monkeypatch):
    """A missing optional dependency must skip the stage, not break the run."""
    from perturbseq_pipeline import ps_score as ps_mod
    from perturbseq_pipeline.cli import run_pipeline

    monkeypatch.setattr(ps_mod, "pertps_available", lambda: False)
    cfg = _base_config(synthetic, tmp_path / "run_no_pertps")
    result = run_pipeline(cfg)
    assert result.report.is_file(), "run must still finish"
    # And the report must say the section was skipped rather than omit it silently.
    html = result.report.read_text()
    assert "This section was skipped" in html
    assert "pertps" in html


def test_missing_pertps_can_be_made_fatal(synthetic, tmp_path, monkeypatch):
    from perturbseq_pipeline import ps_score as ps_mod
    from perturbseq_pipeline.cli import run_pipeline

    monkeypatch.setattr(ps_mod, "pertps_available", lambda: False)
    cfg = _base_config(synthetic, tmp_path / "run_require_pertps")
    cfg.ps_score.require = True
    with pytest.raises(ps_mod.PertpsUnavailable):
        run_pipeline(cfg)


# ---------------------------------------------------------------------------
# Separate GEX / guide quantifications (STARsolo layout)
# ---------------------------------------------------------------------------


def test_separate_gex_and_guide_directories(tmp_path):
    """Expression and guides quantified into two MTX dirs, as STARsolo emits.

    The guide matrix also covers a far larger barcode whitelist than the called
    cells, so the loader must subset and align rather than fail or mis-order.
    """
    from make_synthetic import make_split_lane

    from perturbseq_pipeline.cli import run_pipeline

    lanes = {}
    guide_dirs = {}
    for i, lane in enumerate(["L1", "L2"]):
        paths = make_split_lane(tmp_path / "split", lane, n_cells=300, seed=i)
        lanes[lane] = paths["gex"]
        guide_dirs[lane] = paths["guides"]

    meta = tmp_path / "meta.csv"
    pd.DataFrame({"lane_id": list(lanes), "sample": ["S1", "S2"]}).to_csv(meta, index=False)

    cfg = Config.from_dict(
        {
            "run": {"name": "split", "outdir": str(tmp_path / "run_split")},
            "input": {"mtx_dirs": lanes, "guide_mtx_dirs": guide_dirs},
            "metadata": {"file": str(meta)},
            "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
            "cluster": {"n_top_genes": 80, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 5, "top_n_report": 2},
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
        }
    )
    cfg.validate()
    result = run_pipeline(cfg)

    # The whitelist-only barcodes must not leak into the analysis.
    assert result.n_cells <= 600
    assert not any(str(n).startswith("WHITELIST") for n in result.adata.obs_names)

    # Guides aligned correctly, so the planted knockdowns are still recovered.
    tbl = result.perturbation_table.set_index("target_gene")
    for gene in KD_TARGETS:
        assert tbl.loc[gene, "is_hit_ntc"], f"{gene} lost after the split-dir load"
    assert result.guide_h5ad is not None and result.guide_h5ad.is_file()


def test_guide_mtx_dirs_must_cover_every_lane():
    cfg = Config.from_dict(
        {"input": {"mtx_dirs": {"L1": "/a", "L2": "/b"}, "guide_mtx_dirs": {"L1": "/g"}}}
    )
    with pytest.raises(ValueError, match="must cover every lane"):
        cfg.validate()


def test_target_regex_handles_gene_desert_controls():
    """A guide library with multi-token names needs the regex override.

    The THP-1 library uses `gene_desert_1` cutting controls alongside
    `non-targeting_1`; splitting on the first delimiter would turn those into
    a target called "gene".
    """
    g = GuideConfig(target_regex=r"^(.+)_\d+$")
    got = list(
        parse_target_genes(
            ["ADGRV1_1", "gene_desert_3", "non-targeting_20", "AK9_2"], g
        )
    )
    assert got == ["ADGRV1", "gene_desert", "non-targeting", "AK9"]


# ---------------------------------------------------------------------------
# Barcode -> guide table input (the PS_python demo layout)
# ---------------------------------------------------------------------------


def test_guide_table_input_resolves_multiplets_by_count(synthetic, tmp_path):
    """A cell listed twice must follow the dominance rule, not row order."""
    import scanpy as sc

    from perturbseq_pipeline.cli import run_pipeline

    adata = sc.read_10x_mtx(synthetic["lanes"]["L1"], gex_only=True, cache=False)
    adata.var_names_make_unique()
    path = tmp_path / "expr_only.h5ad"
    adata.write_h5ad(path)

    # Every cell gets a dominant guide, plus a decoy row listed afterwards with
    # a much lower count. Taking "the last row wins" would pick the decoy.
    barcodes = list(adata.obs_names)
    rows = []
    for i, bc in enumerate(barcodes):
        winner = "Non-Targeting" if i % 4 == 0 else KD_TARGETS[i % len(KD_TARGETS)]
        rows.append({"cell": bc, "gene": winner, "umi_count": 50})
        rows.append({"cell": bc, "gene": "DECOY", "umi_count": 1})
    table = tmp_path / "barcodes.txt"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)

    cfg = _base_config(synthetic, tmp_path / "run_guide_table")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(path)
    cfg.input.guide_table = str(table)
    # The synthetic barcodes contain underscores themselves, so the library
    # prefix heuristic would mangle them; it is covered by its own test.
    cfg.input.guide_table_strip_prefix = False
    cfg.metadata.file = None
    result = run_pipeline(cfg)

    targets = set(result.adata.obs["target_gene"].astype(str))
    assert "DECOY" not in targets, "the low-count decoy must never win"
    assert set(KD_TARGETS) <= targets
    # 'Non-Targeting' must be recognised as the control population.
    assert (result.adata.obs["perturbation_class"] == "non-targeting").sum() > 0


def test_guide_table_is_written_and_matches_the_matrix(mtx_run):
    """The exported table must be exactly the matrix above the UMI threshold."""
    import scanpy as sc

    table_path = next(mtx_run.outdir.glob("*_guide_barcodes.txt"))
    table = pd.read_csv(table_path, sep="\t")
    for col in ("cell", "barcode", "sgrna", "gene", "umi_count", "assignment"):
        assert col in table.columns, f"missing column {col}"

    assert table["umi_count"].min() >= 3, "entries below the threshold must be dropped"

    guides = sc.read_h5ad(mtx_run.guide_h5ad)
    X = guides.layers["counts"] if "counts" in guides.layers else guides.X
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    expected_rows = int((dense >= 3).sum())
    assert len(table) == expected_rows, "one row per matrix entry above threshold"
    assert int(table["umi_count"].sum()) == int(dense[dense >= 3].sum())


def test_guide_table_puts_the_dominant_guide_last(mtx_run):
    """So a naive 'last row wins' reader still lands on the top guide."""
    table = pd.read_csv(next(mtx_run.outdir.glob("*_guide_barcodes.txt")), sep="\t")
    multi = table.groupby("cell").filter(lambda g: len(g) > 1)
    assert len(multi) > 0, "test needs cells with several guides"
    last = multi.groupby("cell").tail(1).set_index("cell")["umi_count"]
    top = multi.groupby("cell")["umi_count"].max()
    assert (last == top.reindex(last.index)).all()


def test_guide_table_round_trips_through_the_reader(mtx_run, synthetic, tmp_path):
    """Writing then re-reading the table must reproduce the assignments.

    This is what keeps the pipeline and any consumer of the file — PS_python
    included — on the same per-cell calls.
    """
    import scanpy as sc

    from perturbseq_pipeline.cli import run_pipeline

    table_path = next(mtx_run.outdir.glob("*_guide_barcodes.txt"))
    expr = sc.read_h5ad(mtx_run.h5ad)
    original = expr.obs["target_gene"].astype(str)

    # Feed the written table back in as the sole source of guide identity.
    plain = expr.copy()
    for key in list(plain.obs.columns):
        if key.startswith(("ps_", "target_gene", "guide_id", "perturbation_class")):
            del plain.obs[key]
    path = tmp_path / "expr_for_roundtrip.h5ad"
    plain.write_h5ad(path)

    cfg = _base_config(synthetic, tmp_path / "run_roundtrip")
    cfg.input.mtx_dirs = None
    cfg.input.h5ad = str(path)
    cfg.input.guide_table = str(table_path)
    cfg.input.counts_layer = "counts"
    cfg.metadata.file = None
    cfg.ps_score.enabled = False
    result = run_pipeline(cfg)

    reloaded = result.adata.obs["target_gene"].astype(str)
    shared = original.index.intersection(reloaded.index)
    agree = (original.loc[shared] == reloaded.loc[shared]).mean()
    assert agree > 0.99, f"round-trip changed {100 * (1 - agree):.1f}% of assignments"


def test_guide_table_strips_library_prefixes(tmp_path):
    from perturbseq_pipeline.config import Config
    from perturbseq_pipeline.io import read_guide_table

    table = tmp_path / "bc.tsv"
    pd.DataFrame(
        {
            "cell": ["S1L1_AAAC-1", "S2L2_CCCC-1"],
            "gene": ["GENEA", "Non-Targeting"],
            "umi_count": [30, 30],
        }
    ).to_csv(table, sep="\t", index=False)

    cfg = Config.from_dict({"input": {"h5ad": "x", "guide_table": str(table)}})
    labels = read_guide_table(cfg, pd.Index(["AAAC-1", "CCCC-1", "TTTT-1"]))
    assert list(labels) == ["GENEA", "Non-Targeting", "unassigned"]


def test_guide_table_unmatched_barcodes_raise_a_clear_error(tmp_path):
    from perturbseq_pipeline.config import Config
    from perturbseq_pipeline.io import read_guide_table

    table = tmp_path / "bc.tsv"
    pd.DataFrame(
        {"cell": ["WRONG-1"], "gene": ["GENEA"], "umi_count": [30]}
    ).to_csv(table, sep="\t", index=False)
    cfg = Config.from_dict({"input": {"h5ad": "x", "guide_table": str(table)}})
    with pytest.raises(ValueError, match="No barcode"):
        read_guide_table(cfg, pd.Index(["AAAC-1"]))


# ---------------------------------------------------------------------------
# lochNESS
# ---------------------------------------------------------------------------


def _reference_lochness(adata, target_genotype, n_neighbors, nn_name="nn"):
    """Literal port of pertTF's calculate_lonESS_score, for cross-checking.

    Kept deliberately close to the original — per-cell loop and all — so the
    vectorized implementation is validated against the published behaviour
    rather than against a reimplementation that shares its assumptions.
    """
    overall = adata.obs["genotype"].value_counts(normalize=True).to_dict()
    nn_key = f"{nn_name}_distances"
    out = []
    for cell_id in adata.obs_names:
        i = adata.obs_names.get_loc(cell_id)
        g = target_genotype if target_genotype is not None else adata.obs.loc[cell_id, "genotype"]
        idx = adata.obsp[nn_key][i, :].nonzero()[1]
        cnt = sum(adata.obs.loc[adata.obs.index[idx], "genotype"] == g)
        denom = overall[g] if overall[g] else 1e-4
        out.append(cnt / n_neighbors / denom - 1)
    return np.array(out)


@pytest.fixture(scope="module")
def lochness_toy():
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(0)
    n, k = 400, 30
    a = ad.AnnData(X=rng.normal(size=(n, 30)).astype("float32"))
    a.obs["genotype"] = pd.Categorical(
        rng.choice(["G1", "G2", "G3", "NT"], size=n, p=[0.3, 0.25, 0.2, 0.25])
    )
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=k, n_pcs=10, key_added="nn")
    return a, k


def test_lochness_matches_the_perttf_reference(lochness_toy):
    """The vectorized score must reproduce pertTF's per-cell loop exactly."""
    from perturbseq_pipeline.lochness import _adjacency, lochness_score

    a, k = lochness_toy
    adj, counts = _adjacency(sp.csr_matrix(a.obsp["nn_distances"]))
    overall = a.obs["genotype"].value_counts(normalize=True).to_dict()
    labels = a.obs["genotype"].astype(str).to_numpy()

    for gene in ("G1", "G2", "G3", "NT"):
        ref = _reference_lochness(a, gene, n_neighbors=k)
        ours = lochness_score(adj, counts, (labels == gene).astype(float), overall[gene])
        # We divide by the actual neighbour count (k-1, self excluded); the
        # reference divides by the requested k. Undo that to compare directly.
        rescaled = (ours + 1) * (counts[0] / k) - 1
        assert np.allclose(rescaled, ref, atol=1e-10), gene
        assert np.corrcoef(ref, ours)[0, 1] > 0.9999


def test_lochness_is_zero_at_background_frequency():
    """A perturbation spread uniformly must score ~0 everywhere."""
    from perturbseq_pipeline.lochness import _adjacency, lochness_score

    # Every cell neighbours every other; the local fraction then equals the
    # overall fraction by construction.
    n = 50
    dense = np.ones((n, n)) - np.eye(n)
    adj, counts = _adjacency(sp.csr_matrix(dense))
    indicator = np.zeros(n)
    indicator[:10] = 1  # 20% of cells
    score = lochness_score(adj, counts, indicator, 10 / n)
    # A cell carrying the perturbation sees the other 9 among 49 neighbours.
    assert np.allclose(score[10:], (10 / 49) / 0.2 - 1, atol=1e-9)
    assert np.abs(np.mean(score)) < 0.15


def test_lochness_detects_a_planted_neighbourhood():
    """A perturbation confined to one region must score strongly positive."""
    from perturbseq_pipeline.lochness import _adjacency, lochness_score

    # Two disconnected blocks; the perturbation fills the first.
    n = 60
    dense = np.zeros((n, n))
    dense[:30, :30] = 1
    dense[30:, 30:] = 1
    np.fill_diagonal(dense, 0)
    adj, counts = _adjacency(sp.csr_matrix(dense))
    indicator = np.zeros(n)
    indicator[:30] = 1
    score = lochness_score(adj, counts, indicator, 0.5)
    assert score[:30].mean() > 0.9, "inside the block it should be ~+1"
    assert score[30:].mean() < -0.9, "outside it should be ~-1"


def test_lochness_colour_scale_does_not_clip_the_tail():
    """The lochNESS colour scale must span the real range, not a percentile.

    The score is bounded below by -1 but unbounded above, and the signal lives
    in a thin upper tail: on the demo lane SALL4 reaches +47 while the 99th
    percentile is 4.5. Capping at a percentile painted half of SALL4's own
    cells the same saturated colour — flattening exactly what the figure is
    for — so the scale is symlog over the full range instead.
    """
    from perturbseq_pipeline.plots import _lochness_norm

    norm = _lochness_norm(46.6)
    assert norm.vmax >= 46.6, "the largest score must be inside the scale"
    assert norm.vmin <= -1.0, "the -1 floor must be inside the scale"
    # Distinct extreme values must map to distinct colours rather than both
    # saturating at the top.
    assert norm(46.6) > norm(10.0) > norm(4.5), "tail values must stay separable"
    # Near zero the mapping stays linear and symmetric.
    assert norm(0.0) == pytest.approx(0.5, abs=1e-6)


def test_lochness_end_to_end_outputs(mtx_run):
    summary = pd.read_csv(mtx_run.outdir / "tables" / "lochness.csv")
    assert len(summary) > 0
    for col in (
        "target_gene",
        "mean_lochness_in_own_cells",
        "pct_cells_enriched",
        "max_lochness",
    ):
        assert col in summary.columns

    maps = list((mtx_run.figures_dir / "lochness" / "per_target").glob("*.png"))
    assert len(maps) == len(summary), "one map per scored perturbation"
    overview = {p.stem for p in (mtx_run.figures_dir / "lochness").glob("*.png")}
    assert {"lochness_self_enrichment", "lochness_distributions"} <= overview

    obs = mtx_run.adata.obs
    assert "lochness_self" in obs.columns
    assert any(c.startswith("lochness_") and c != "lochness_self" for c in obs.columns)


def test_lochness_can_be_disabled(synthetic, tmp_path):
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _base_config(synthetic, tmp_path / "run_noloch")
    cfg.lochness.enabled = False
    result = run_pipeline(cfg)
    assert not (result.outdir / "figures" / "lochness").exists()
    assert not (result.outdir / "tables" / "lochness.csv").exists()
    assert result.report.is_file()


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
