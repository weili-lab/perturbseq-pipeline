"""Regressions for output relocation and the shipped demo config.

* ``config/demo.yaml`` must not point ``output.large_file_dir`` at a machine-specific
  path: the demo used to move its 600 MB h5ads into a hard-coded personal Drive folder.
* ``demo/fetch_demo_data.py`` must never inherit such a path from its template.
* ``io.relocate_if_large`` must keep the file in the run directory, not crash, when the
  destination cannot be created (missing mount, read-only or non-existent location).
* ``io.write_h5ad`` must cope with anndata >= 0.13 exposing ``X`` as the layer keyed
  ``None`` (the name sanitiser used to rename it and record a non-string key).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbseq_pipeline import io as io_mod
from perturbseq_pipeline.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_fetch_demo_module():
    spec = importlib.util.spec_from_file_location("fetch_demo_data", REPO_ROOT / "demo" / "fetch_demo_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_config_keeps_large_outputs_local():
    cfg = Config.from_yaml(REPO_ROOT / "config" / "demo.yaml")
    assert cfg.output.large_file_dir is None


def test_fetch_demo_write_config_does_not_inherit_large_file_dir(tmp_path):
    template = tmp_path / "template.yaml"
    template.write_text(
        "input:\n  mode: mtx\n  mtx_dirs:\n    placeholder: /path/to/lane\n"
        "output:\n  large_file_dir: /content/drive/MyDrive/somebody_elses_folder\n  large_file_threshold_mb: 50.0\n"
    )
    lane_dir = tmp_path / "filtered_feature_bc_matrix_S1lane1"
    lane_dir.mkdir()
    fetch = _load_fetch_demo_module()
    out = tmp_path / "demo.local.yaml"
    fetch.write_config(
        {"S1lane1": lane_dir},
        out,
        REPO_ROOT / "demo" / "sample_metadata.csv",
        str(tmp_path / "results"),
        template=template,
    )
    cfg = Config.from_yaml(out)
    assert cfg.output.large_file_dir is None
    assert cfg.input.mtx_dirs == {"S1lane1": str(lane_dir)}
    # An explicit destination is honoured.
    fetch.write_config(
        {"S1lane1": lane_dir},
        out,
        REPO_ROOT / "demo" / "sample_metadata.csv",
        str(tmp_path / "results"),
        large_file_dir=str(tmp_path / "big"),
        template=template,
    )
    assert Config.from_yaml(out).output.large_file_dir == str(tmp_path / "big")


def test_relocate_if_large_keeps_file_when_destination_unusable(tmp_path, caplog):
    src = tmp_path / "run" / "big.h5ad"
    src.parent.mkdir()
    src.write_bytes(b"x" * 1024)
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("regular file, so mkdir(parents=True) below it fails")
    cfg = Config()
    cfg.output.large_file_dir = str(blocker / "drive" / "MyDrive")
    cfg.output.large_file_threshold_mb = 0.0
    with caplog.at_level("WARNING"):
        result = io_mod.relocate_if_large(src, cfg)
    assert result == src
    assert src.is_file()
    assert "not usable" in caplog.text


def test_relocate_if_large_moves_when_destination_is_writable(tmp_path):
    src = tmp_path / "run" / "big.h5ad"
    src.parent.mkdir()
    src.write_bytes(b"x" * 1024)
    cfg = Config()
    cfg.output.large_file_dir = str(tmp_path / "elsewhere" / "nested")
    cfg.output.large_file_threshold_mb = 0.0
    result = io_mod.relocate_if_large(src, cfg)
    assert result == tmp_path / "elsewhere" / "nested" / "big.h5ad"
    assert result.is_file() and not src.exists()


def test_write_h5ad_sanitiser_ignores_anndata_x_layer_key(tmp_path):
    """anndata >= 0.13 lists X as layers[None]; the sanitiser must leave it alone."""
    adata = ad.AnnData(
        X=sp.csr_matrix(np.eye(4, dtype=np.float32)),
        obs=pd.DataFrame({"weird col/name": ["a", "b", "c", "d"]}, index=[f"c{i}" for i in range(4)]),
    )
    adata.layers["counts"] = adata.X.copy()
    mapping = io_mod.sanitize_h5ad_names(adata)
    assert all(isinstance(r["original"], str) for r in mapping)
    assert "None" not in adata.layers
    assert adata.X is not None and adata.X.shape == (4, 4)
    out = io_mod.write_h5ad(adata, tmp_path / "x.h5ad")
    back = ad.read_h5ad(out)
    assert back.shape == (4, 4)
    assert [k for k in back.layers.keys() if k is not None] == ["counts"]
    assert "weird_col_name" in back.obs.columns
    mapping_uns = back.uns["column_name_mapping"]
    assert list(np.asarray(mapping_uns["original"])) == ["weird col/name"]
