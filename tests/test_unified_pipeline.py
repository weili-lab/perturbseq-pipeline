"""Unified-branch behaviour: run manifest and module status, optional-module
switches, HDF5-safe names on write, label standardisation across stages, and
compatibility of the shipped configs.

Small synthetic fixtures only (two lanes x 250 cells).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))

from make_synthetic import make_dataset  # noqa: E402

from perturbseq_pipeline.config import Config  # noqa: E402
from perturbseq_pipeline.run_manifest import (  # noqa: E402
    OPTIONAL_MODULES,
    STAGES,
    STATUS_COMPLETED,
    STATUS_DISABLED,
    STATUS_SKIPPED,
    ModuleStatusTracker,
)

REPO = Path(__file__).resolve().parent.parent

ALL_OFF = {name: {"enabled": False} for name in OPTIONAL_MODULES}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("unified_synthetic")
    return {"dir": out, **make_dataset(out, n_lanes=2, n_cells=250)}


def _config(synthetic, outdir: Path, **overrides) -> Config:
    data = {
        "run": {"name": "unified", "outdir": str(outdir), "seed": 7},
        "input": {"mtx_dirs": synthetic["lanes"]},
        "metadata": {"file": synthetic["metadata"]},
        "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
        "cluster": {"n_top_genes": 80, "n_pcs": 10},
        "perturbation": {"min_cells_per_target": 5, "top_n_report": 2},
        "compute": {"n_jobs": 2},
        "output": {"archive": False},
    }
    for key, value in overrides.items():
        data.setdefault(key, {}).update(value)
    return Config.from_dict(data)


@pytest.fixture(scope="module")
def minimal_run(synthetic, tmp_path_factory):
    """Every optional module switched off through YAML-style config."""
    from perturbseq_pipeline.cli import run_pipeline

    outdir = tmp_path_factory.mktemp("run_minimal")
    return run_pipeline(_config(synthetic, outdir, **ALL_OFF))


@pytest.fixture(scope="module")
def distance_run(synthetic, tmp_path_factory):
    """Distance stages on, the heavier optional modules off."""
    from perturbseq_pipeline.cli import run_pipeline

    outdir = tmp_path_factory.mktemp("run_distance")
    overrides = dict(ALL_OFF)
    overrides.update(
        {
            "distance": {"enabled": True, "n_permutations": 30, "min_cells": 5, "max_cells_per_target": 200},
            "distance_space": {"enabled": True, "min_cells": 5, "n_components": 2, "nearest_neighbors": 2},
            "lochness": {"enabled": True, "n_neighbors": 30, "n_pcs": 10, "min_cells_per_target": 5},
            "meta_analysis": {"enabled": True},
        }
    )
    return run_pipeline(_config(synthetic, outdir, **overrides))


# Run manifest and module status


def test_run_manifest_is_written_with_required_fields(minimal_run):
    path = Path(minimal_run.run_manifest)
    assert path.is_file() and path.name == "run_manifest.json"
    rec = json.loads(path.read_text())
    for key in (
        "pipeline_version",
        "git",
        "execution",
        "resolved_config",
        "random_seed",
        "input",
        "guides",
        "enabled_modules",
        "module_status",
        "execution_mode",
        "outputs",
    ):
        assert key in rec, key
    assert rec["random_seed"] == 7
    assert rec["guides"]["assignment_mode"] == "single_guide"
    assert rec["execution"]["command"] and rec["execution"]["python_executable"]
    assert Path(rec["resolved_config"]).is_file()
    assert rec["input"]["mode"] == "mtx"
    assert set(rec["input"]["lanes"]) == set(rec["input"]["loaded_lanes"])
    for lane in rec["input"]["lanes"].values():
        assert lane["expression"]["exists"]
    assert rec["enabled_modules"] == {name: False for name in OPTIONAL_MODULES}
    assert "commit" in rec["git"] and "branch" in rec["git"]


def test_legacy_report_has_no_high_moi_regression_section(minimal_run):
    """The optional high-MOI regression stage must leave legacy HTML / Markdown reports without its section."""
    out = Path(minimal_run.outdir)
    assert 'id="regression"' not in (out / "report.html").read_text()  # the stage still has a module-status row
    headings = [line for line in (out / "report.md").read_text().splitlines() if line.startswith("#")]
    assert not any("Membership regression" in line for line in headings)


def test_module_status_covers_every_stage(minimal_run):
    df = minimal_run.module_status
    assert list(df["module"]) == [k for k, _ in STAGES if k != "basic_qc"]
    status = dict(zip(df["module"], df["status"]))
    for core in ("load", "qc", "guide_assignment", "clustering", "perturbation", "outputs", "report"):
        assert status[core] == STATUS_COMPLETED, core
    for name in OPTIONAL_MODULES:
        assert status[name] == STATUS_DISABLED, name
    csv = Path(minimal_run.outdir) / "tables" / "module_status.csv"
    assert csv.is_file()
    on_disk = pd.read_csv(csv)
    assert list(on_disk["module"]) == list(df["module"])
    incremental = Path(minimal_run.outdir) / "logs" / "module_status.json"
    assert incremental.is_file() and len(json.loads(incremental.read_text())) == len(df)


def test_report_shows_module_status_and_provenance(minimal_run):
    html = Path(minimal_run.report).read_text(encoding="utf-8")
    assert "Module completion status" in html
    assert "Run provenance" in html
    assert "Git branch / commit" in html
    assert "disabled" in html
    # optional sections are absent when their module is off
    for anchor in (
        '<h2 id="distance">',
        '<h2 id="distance_space">',
        '<h2 id="lochness">',
        '<h2 id="modules">',
        '<h2 id="enrichment">',
    ):
        assert anchor not in html, anchor
    md = Path(minimal_run.outdir) / "report.md"
    assert md.is_file() and "Module completion status" in md.read_text()


def test_disabled_modules_write_no_tables(minimal_run):
    tabledir = Path(minimal_run.outdir) / "tables"
    for name in (
        "perturbation_distance",
        "phenotype_modules",
        "lochness",
        "ps_score",
        "cofunctional_modules",
        "enrichment",
        "perturbation_meta",
    ):
        assert not (tabledir / f"{name}.csv").exists(), name
    for name in ("qc_steps", "guide_assignment", "clusters", "perturbation", "module_status"):
        assert (tabledir / f"{name}.csv").is_file(), name


def test_tracker_marks_not_run_stages():
    t = ModuleStatusTracker()
    t.mark("load", STATUS_COMPLETED)
    t.mark("lochness", STATUS_DISABLED, enabled=False)
    df = t.table(include_not_run=True)
    status = dict(zip(df["module"], df["status"]))
    assert status["load"] == STATUS_COMPLETED
    assert status["lochness"] == STATUS_DISABLED
    assert status["report"] == "not_run"


# Optional modules consume the standardized labels


def test_distance_stages_run_when_enabled(distance_run):
    status = dict(zip(distance_run.module_status["module"], distance_run.module_status["status"]))
    assert status["distance"] in (STATUS_COMPLETED, STATUS_SKIPPED)
    assert status["distance_space"] in (STATUS_COMPLETED, STATUS_SKIPPED)
    assert status["lochness"] in (STATUS_COMPLETED, STATUS_SKIPPED)
    assert status["ps_score"] == STATUS_DISABLED
    assert status["modules"] == STATUS_DISABLED
    rec = json.loads(Path(distance_run.run_manifest).read_text())
    assert rec["enabled_modules"]["distance"] is True and rec["enabled_modules"]["modules"] is False
    html = Path(distance_run.report).read_text(encoding="utf-8")
    if status["distance"] == STATUS_COMPLETED:
        assert '<h2 id="distance">' in html
        assert (Path(distance_run.outdir) / "tables" / "perturbation_distance.csv").is_file()


def test_distance_and_lochness_use_guide_assignment_labels(distance_run):
    """Targets scored by the optional stages are exactly the assigned targets;
    no stage re-derives labels from the raw guide counts."""
    obs = distance_run.adata.obs
    assigned = set(obs.loc[obs["perturbation_class"].astype(str) == "targeting", "target_gene"].astype(str))
    if distance_run.distance_table is not None and not distance_run.distance_table.empty:
        assert set(distance_run.distance_table["target_gene"].astype(str)) <= assigned
    loch = Path(distance_run.outdir) / "tables" / "lochness.csv"
    if loch.is_file():
        assert set(pd.read_csv(loch)["target_gene"].astype(str)) <= assigned
    meta = Path(distance_run.outdir) / "tables" / "perturbation_meta.csv"
    if meta.is_file():
        assert set(pd.read_csv(meta)["target_gene"].astype(str)) <= assigned


def test_pair_mode_requires_scaffold_classes(synthetic, tmp_path):
    """Pair mode refuses to guess: guides without a scaffold class (and no pair
    reference supplying one) stop the run with a clear message instead of a
    silent single-guide fallback."""
    from perturbseq_pipeline.cli import run_pipeline

    cfg = _config(synthetic, tmp_path / "pair_no_scaffold", guides={"assignment_mode": "pair"}, **ALL_OFF)
    cfg.validate()
    assert cfg.guides.assignment_mode == "dual_guide_pair"
    with pytest.raises(ValueError, match="scaffold class"):
        run_pipeline(cfg)


def test_pair_mode_rejects_label_only_input(minimal_run, tmp_path):
    """Pair mode needs a guide count matrix; a pre-computed label column cannot be pair-resolved."""
    from perturbseq_pipeline.cli import run_pipeline
    from perturbseq_pipeline.io import write_h5ad

    src = minimal_run.adata
    labelled = ad.AnnData(
        X=src.layers["counts"].copy(), obs=src.obs[["lane_id", "target_gene"]].copy(), var=src.var[[]].copy()
    )
    h5 = write_h5ad(labelled, tmp_path / "labelled.h5ad")
    cfg = Config.from_dict(
        {
            "run": {"name": "labels", "outdir": str(tmp_path / "run_labels")},
            "input": {"mode": "h5ad", "h5ad": str(h5), "guide_obs_column": "target_gene"},
            "metadata": {"require_for_multilane": False},
            "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
            "guides": {"assignment_mode": "dual_guide_pair"},
            "output": {"archive": False, "write_unfiltered_h5ad": False},
            **ALL_OFF,
        }
    )
    with pytest.raises(ValueError, match="guide count matrix"):
        run_pipeline(cfg)


# HDF5-safe names


def test_write_h5ad_sanitises_unsafe_names(tmp_path):
    from perturbseq_pipeline.io import sanitize_h5ad_name, write_h5ad

    assert sanitize_h5ad_name("LIPA (rs1412444)") == "LIPA__rs1412444_"
    assert sanitize_h5ad_name("A/B") == "A_B"
    assert sanitize_h5ad_name("plain_name.1") == "plain_name.1"
    X = sp.random(20, 5, density=0.5, format="csr", random_state=0)
    adata = ad.AnnData(X=X)
    adata.obs["lochness_LIPA (rs1412444)"] = np.arange(20, dtype=float)
    adata.obs["ps_score_FHL3/alt"] = np.arange(20, dtype=float)
    adata.obs["fine"] = np.arange(20)
    adata.uns["per target/summary"] = {"n": 1}
    adata.obsm["X_bad name"] = np.zeros((20, 2))
    out = write_h5ad(adata, tmp_path / "safe.h5ad")
    back = ad.read_h5ad(out)
    assert "lochness_LIPA__rs1412444_" in back.obs.columns
    assert "ps_score_FHL3_alt" in back.obs.columns
    assert "fine" in back.obs.columns
    assert "per_target_summary" in back.uns
    assert "X_bad_name" in back.obsm
    mapping = back.uns["column_name_mapping"]
    assert set(map(str, mapping["original"])) >= {
        "lochness_LIPA (rs1412444)",
        "ps_score_FHL3/alt",
        "per target/summary",
        "X_bad name",
    }
    csv = tmp_path / "safe_column_name_mapping.csv"
    assert csv.is_file()
    df = pd.read_csv(csv)
    assert {"location", "original", "sanitized"} <= set(df.columns)


def test_write_h5ad_without_unsafe_names_writes_no_mapping(tmp_path):
    from perturbseq_pipeline.io import write_h5ad

    adata = ad.AnnData(X=sp.random(5, 3, density=0.5, format="csr", random_state=1))
    adata.obs["ok"] = np.arange(5)
    write_h5ad(adata, tmp_path / "plain.h5ad")
    assert not (tmp_path / "plain_column_name_mapping.csv").exists()
    assert "column_name_mapping" not in ad.read_h5ad(tmp_path / "plain.h5ad").uns


# Shipped configuration compatibility


def test_default_yaml_matches_schema():
    """config/default.yaml is the documented snapshot of Config(); it carries no input."""
    import yaml

    on_disk = yaml.safe_load((REPO / "config" / "default.yaml").read_text())
    assert on_disk == Config().to_dict()
    cfg = Config.from_dict(on_disk)
    for name in OPTIONAL_MODULES:
        assert isinstance(getattr(cfg, name).enabled, bool)


def test_demo_config_still_loads_and_validates(tmp_path):
    """The main-branch demo config keeps working; only the data path changes."""
    cfg = Config.from_yaml(REPO / "config" / "demo.yaml")
    assert cfg.input.mode == "mtx" and cfg.guides.assignment_mode == "single_guide"
    assert cfg.run.name == "demo_ESC_TF_perturbseq"
    cfg.validate()


@pytest.mark.parametrize(
    "name",
    sorted(p.name for p in (REPO / "config").glob("*.yaml"))
    + sorted(str(p.relative_to(REPO / "config")) for p in (REPO / "config" / "examples").glob("*.yaml")),
)
def test_every_shipped_config_parses(name):
    import yaml

    data = yaml.safe_load((REPO / "config" / name).read_text())
    cfg = Config.from_dict(data)  # schema check without touching the filesystem
    assert cfg.guides.assignment_mode in ("single_guide", "pair", "dual_guide_pair")
    if name != "default.yaml":
        cfg.validate()


def test_input_mode_switches_are_accepted():
    for mode in ("auto", "mtx", "h5ad"):
        cfg = Config.from_dict({"input": {"mode": mode, "mtx_dirs": {"L1": "/x"}, "h5ad": "/y.h5ad"}})
        assert cfg.input.mode == mode


def test_paired_guide_config_keys_are_accepted():
    cfg = Config.from_dict(
        {
            "guides": {"assignment_mode": "dual_guide_pair", "pair_assignment_primary": True},
            "input": {"mtx_dirs": {"L1": "/x"}, "guide_mtx_dirs": {"L1": "/g"}},
        }
    )
    cfg.validate()
    assert cfg.guides.assignment_mode == "dual_guide_pair"
    with pytest.raises(ValueError, match="pair_assignment_primary"):
        Config.from_dict(
            {
                "guides": {"assignment_mode": "pair", "pair_assignment_primary": False},
                "input": {"mtx_dirs": {"L1": "/x"}},
            }
        ).validate()
