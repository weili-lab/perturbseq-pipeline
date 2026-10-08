"""Regressions from the 2026-10-08 audit (P1 group 1): guide metadata, --lane, list-form
lanes, the MTX cache, and write_guide_table's in-place mutation."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from perturbseq_pipeline import cli, io as io_mod
from perturbseq_pipeline.config import Config
from perturbseq_pipeline.guides import NOT_EVALUATED_LABEL, OBS_GUIDE, OBS_TARGET, assign_guides, guide_representation


def _expr_and_guides():
    """Guide matrix over 8 cells; expression object over the 5 that survived QC."""
    guide_ids = ["TSS_1", "TSS_2", "TSS_3", "CTRL_1"]
    targets = ["GENE_A", "GENE_A", "GENE_B", "NTC"]
    X = np.zeros((8, 4))
    X[0, 0] = 10  # GENE_A
    X[1, 1] = 12  # GENE_A
    X[2, 2] = 9  # GENE_B
    X[3, 3] = 15  # NTC
    X[4, 0] = 5  # GENE_A
    X[5, 2] = 20  # QC-filtered cell
    X[6, 3] = 7  # QC-filtered cell
    guides = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame({"target_gene_name": targets}, index=guide_ids),
    )
    guides.layers["counts"] = guides.X.copy()
    expr = ad.AnnData(
        X=sparse.csr_matrix(np.ones((5, 3))),
        obs=pd.DataFrame({"lane_id": ["L1"] * 5}, index=[f"c{i}" for i in range(5)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    cfg = Config()
    cfg.guides.target_feature_column = "target_gene_name"
    cfg.guides.target_split_delims = []
    cfg.guides.ntc_patterns = ["^NTC$"]
    cfg.guides.min_umi = 3
    return expr, guides, cfg


def test_single_guide_metadata_lands_on_the_original_guides_object():
    expr, guides, cfg = _expr_and_guides()
    expr = assign_guides(expr, guides, cfg)
    assert expr.obs.loc["c0", OBS_TARGET] == "GENE_A"
    # var: targets resolved through target_feature_column, on the object the caller keeps
    assert guides.var["target_gene"].tolist() == ["GENE_A", "GENE_A", "GENE_B", "NTC"]
    assert guides.var["is_non_targeting"].tolist() == [False, False, False, True]
    # obs: evaluated cells carry their call, QC-filtered cells are marked as not evaluated
    assert guides.obs.loc["c0", OBS_GUIDE] == "TSS_1"
    assert guides.obs.loc["c5", OBS_GUIDE] == NOT_EVALUATED_LABEL
    assert guides.obs.loc["c5", OBS_TARGET] == NOT_EVALUATED_LABEL
    # guide_representation now knows the targets and lists designed guides with zero cells
    rep = guide_representation(guides, expr)
    assert "target_gene" in rep.columns
    assert rep.set_index("guide_id").loc["TSS_3", "target_gene"] == "GENE_B"
    assert rep.set_index("guide_id").loc["TSS_3", "n_cells"] == 1


def test_guide_table_uses_resolved_targets_and_processed_h5ad_keeps_them(tmp_path):
    expr, guides, cfg = _expr_and_guides()
    expr = assign_guides(expr, guides, cfg)
    cfg.output.write_guide_table = True
    cfg.output.guide_table_min_umi = 3
    path = io_mod.write_guide_table(guides, expr, cfg, tmp_path / "guides.txt")
    tab = pd.read_csv(path, sep="\t")
    gene_of = dict(zip(tab["sgrna"], tab["gene"]))
    assert gene_of["TSS_1"] == "GENE_A" and gene_of["TSS_3"] == "GENE_B" and gene_of["CTRL_1"] == "NTC"
    merged = io_mod.merge_guides_into_expr(expr.copy(), guides, cfg)
    assert list(merged.uns["guide_target_genes"]) == ["GENE_A", "GENE_A", "GENE_B", "NTC"]


def test_write_guide_table_does_not_mutate_the_guide_matrix(tmp_path):
    expr, guides, cfg = _expr_and_guides()
    before = guides.layers["counts"].toarray().copy()
    cfg.output.write_guide_table = True
    cfg.output.guide_table_min_umi = 8
    io_mod.write_guide_table(guides, expr, cfg, tmp_path / "guides.txt")
    np.testing.assert_array_equal(guides.layers["counts"].toarray(), before)
    np.testing.assert_array_equal(guides.X.toarray(), before)


def _two_lane_config(tmp_path, with_guide_dirs: bool) -> Path:
    data = {
        "run": {"name": "x", "outdir": str(tmp_path / "out")},
        "input": {"mode": "mtx", "mtx_dirs": {"L1": str(tmp_path / "L1"), "L2": str(tmp_path / "L2")}},
    }
    if with_guide_dirs:
        data["input"]["guide_mtx_dirs"] = {"L1": str(tmp_path / "gL1"), "L2": str(tmp_path / "gL2")}
    cfg = Config.from_dict(data)
    path = tmp_path / "run.yaml"
    cfg.dump_yaml(path)
    return path


@pytest.mark.parametrize("with_guide_dirs", [False, True])
def test_lane_flag_restricts_to_one_lane(tmp_path, monkeypatch, with_guide_dirs):
    captured = {}

    class _Result:
        def summary(self):
            return "ok"

    def fake_run(cfg, verbose=False, config_path=None):
        captured["cfg"] = cfg
        return _Result()

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    path = _two_lane_config(tmp_path, with_guide_dirs)
    assert cli.main(["run", "-c", str(path), "--lane", "L1"]) == 0
    cfg = captured["cfg"]
    assert cfg.input.mtx_dirs == {"L1": str(tmp_path / "L1")}
    if with_guide_dirs:
        assert cfg.input.guide_mtx_dirs == {"L1": str(tmp_path / "gL1")}
    else:
        assert not cfg.input.guide_mtx_dirs
    assert Path(cfg.run.outdir) == tmp_path / "out" / "samples" / "L1"
    assert cfg.run.name == "x_L1"
    with pytest.raises(SystemExit, match="not a lane"):
        cli.main(["run", "-c", str(path), "--lane", "L9"])


def test_list_form_mtx_dirs_rejects_colliding_lane_ids():
    cfg = Config()
    cfg.input.mode = "mtx"
    cfg.input.mtx_dirs = [
        "/data/sampleA/outs/filtered_feature_bc_matrix",
        "/data/sampleB/outs/filtered_feature_bc_matrix",
    ]
    with pytest.raises(ValueError, match="both resolve to lane id"):
        cfg.validate()
    cfg.input.mtx_dirs = ["/data/filtered_feature_bc_matrix_S1", "/data/filtered_feature_bc_matrix_S2"]
    assert set(cfg.input.resolved_mtx_dirs()) == {"S1", "S2"}


def test_mtx_cache_is_off_by_default_and_scoped_to_the_run_directory(tmp_path):
    cfg = Config()
    assert cfg.input.cache_mtx is False
    assert io_mod._configure_mtx_cache(cfg) is None
    cfg.input.cache_mtx = True
    cfg.run.outdir = str(tmp_path / "run")
    assert io_mod._configure_mtx_cache(cfg) == tmp_path / "run" / "cache"
    assert Path(sc.settings.cachedir) == tmp_path / "run" / "cache"
