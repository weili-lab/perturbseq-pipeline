"""Golden-table regression guard for the established assignment modes.

Runs the synthetic fixture end to end in ``single_guide`` mode (every optional
stage on) and in ``dual_guide_pair`` mode, canonicalises every output table and
the per-cell ``obs`` frame, and compares SHA-256 digests against
``tests/golden/low_moi_golden.json``.

Purpose: new assignment modes (high-MOI membership calling) must leave the
existing modes byte-identical. Per-stage unit tests check semantics; this file
catches the glue edits (config defaults, cli dispatch, io round-trip, QC text)
that those tests do not see.

Canonicalisation: columns sorted by name, floats formatted with 5 significant
digits, rows sorted lexicographically; so the digests are insensitive to row
order from parallel workers and to last-digit noise, but sensitive to any
change of a value, a column or a cell label. The Markdown report (warnings,
QC text, section text and tables) is hashed too, after neutralising the
run-specific parts: timestamp, absolute paths, git commit, command line,
package versions and stage timings.

Regenerate the golden file (only when a change to the legacy output is
intended, or after a dependency upgrade changes clustering)::

    PSP_UPDATE_GOLDEN=1 pytest tests/test_low_moi_golden.py

and commit the new JSON together with the change that motivated it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from make_synthetic import make_dataset, make_split_lane  # noqa: E402
from test_pair_mode_pipeline import _pair_up  # noqa: E402

from perturbseq_pipeline.cli import run_pipeline  # noqa: E402
from perturbseq_pipeline.config import Config  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "golden" / "low_moi_golden.json"
UPDATE = bool(os.environ.get("PSP_UPDATE_GOLDEN"))

#: Tables whose content is intrinsically run-specific (timings, absolute paths).
VOLATILE_TABLES = {"compute_profile", "figure_manifest", "module_status"}


# Canonicalisation


def _canon_frame(df: pd.DataFrame) -> str:
    out = pd.DataFrame(index=df.index)
    for col in sorted(df.columns.astype(str)):
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            vals = s.to_numpy(dtype=float)
            txt = np.array([f"{v:.5g}" if np.isfinite(v) else repr(float(v)) for v in vals], dtype=object)
            out[col] = txt
        elif pd.api.types.is_bool_dtype(s):
            out[col] = s.astype(str).to_numpy()
        else:
            out[col] = s.astype(str).to_numpy()
    out = out.sort_values(list(out.columns), kind="mergesort").reset_index(drop=True)
    return out.to_csv(index=False, lineterminator="\n")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_tables(outdir: Path) -> Dict[str, str]:
    digests: Dict[str, str] = {}
    for path in sorted((outdir / "tables").glob("*.csv")):
        if path.stem in VOLATILE_TABLES:
            continue
        df = pd.read_csv(path, low_memory=False)
        digests[f"tables/{path.stem}"] = _sha(_canon_frame(df))
    return digests


_REPORT_DROP_PREFIXES = ("| Git branch / commit |", "| Command |", "| Compute |", "| SLURM job |", "| Pipeline version |")


def _canon_report(text: str, root: Path) -> str:
    """Neutralise the run-specific parts of report.md, keep every other line verbatim."""
    out = []
    skipping_versions = False
    for line in text.splitlines():
        if skipping_versions:
            if not line.strip():
                skipping_versions = False
            continue
        if line.startswith("- package versions:"):
            skipping_versions = True
            continue
        if line.startswith(_REPORT_DROP_PREFIXES):
            continue
        line = line.replace(str(root), "ROOT")
        line = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?", "DATE", line)
        # module completion status: "| ... | <seconds> |" -> "| ... | T |"
        line = re.sub(r"\| ?[0-9]+(\.[0-9]+)? ?\|$", "| T |", line) if line.startswith("| ") and line.count("|") == 7 else line
        out.append(line)
    return "\n".join(out) + "\n"


def _digest_report(outdir: Path, root: Path) -> Dict[str, str]:
    path = outdir / "report.md"
    if not path.is_file():
        return {}
    return {"report_md": _sha(_canon_report(path.read_text(), root))}


def _digest_adata(adata) -> Dict[str, str]:
    obs = adata.obs.copy()
    obs.insert(0, "cell_id", obs.index.astype(str))
    structure = json.dumps(
        {
            "shape": list(adata.shape),
            "obsm": sorted(str(k) for k in adata.obsm.keys()),
            "layers": sorted(str(k) for k in adata.layers.keys()),
            "uns": sorted(str(k) for k in adata.uns.keys()),
            "var_columns": sorted(adata.var.columns.astype(str)),
            "obs_columns": sorted(obs.columns.astype(str)),
        },
        sort_keys=True,
    )
    return {"obs": _sha(_canon_frame(obs)), "structure": _sha(structure)}


# Runs


def _single_guide_run(tmp_path: Path):
    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300)
    cfg = Config.from_dict(
        {
            "run": {"name": "golden_single", "outdir": str(tmp_path / "run_single")},
            "input": {"mtx_dirs": data["lanes"]},
            "metadata": {"file": data["metadata"]},
            "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
            "cluster": {"n_top_genes": 80, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 5, "top_n_report": 2},
            "knockdown_filter": {"enabled": True},
            "modules": {"enabled": True},
            "ps_score": {"enabled": True},
            "lochness": {"enabled": True},
            "distance": {"enabled": True},
            "distance_space": {"enabled": True},
        }
    )
    cfg.validate()
    assert cfg.guides.assignment_mode == "single_guide"
    return run_pipeline(cfg), Path(cfg.run.outdir)


def _pair_run(tmp_path: Path):
    lanes, guide_dirs, ref = {}, {}, None
    for i, lane in enumerate(["L1", "L2"]):
        paths = make_split_lane(tmp_path / "split", lane, n_cells=300, seed=i)
        lanes[lane] = paths["gex"]
        guide_dirs[lane] = paths["guides"]
        ref = _pair_up(Path(paths["guides"]), seed=i)
    ref_path = tmp_path / "pair_reference.csv"
    ref.to_csv(ref_path, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"lane_id": list(lanes), "sample": ["S1", "S2"], "condition": ["c1", "c1"]}).to_csv(meta, index=False)
    cfg = Config.from_dict(
        {
            "run": {"name": "golden_pair", "outdir": str(tmp_path / "run_pair")},
            "input": {"mtx_dirs": lanes, "guide_mtx_dirs": guide_dirs, "cell_id_format": "prefix"},
            "metadata": {"file": str(meta)},
            "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
            "guides": {
                "assignment_mode": "pair",
                "pair_reference": str(ref_path),
                "single_guide_diagnostic": True,
                "ntc_label": "ntc",
                "min_umi": 3,
                "dominance_ratio": 2.0,
            },
            "cluster": {"n_top_genes": 80, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 5, "top_n_report": 2},
            "enrichment": {"enabled": False},
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
            "modules": {"enabled": False},
            "distance": {"enabled": False},
            "distance_space": {"enabled": False},
            "meta_analysis": {"enabled": False},
        }
    )
    cfg.validate()
    assert cfg.guides.assignment_mode == "dual_guide_pair"
    return run_pipeline(cfg), Path(cfg.run.outdir)


RUNS = {"single_guide": _single_guide_run, "dual_guide_pair": _pair_run}


@pytest.fixture(scope="module")
def digests(tmp_path_factory) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for name, fn in RUNS.items():
        root = tmp_path_factory.mktemp(f"golden_{name}")
        result, outdir = fn(root)
        d = _digest_tables(outdir)
        d.update(_digest_adata(result.adata))
        d.update(_digest_report(outdir, root))
        out[name] = d
    return out


def _load_golden() -> dict:
    if not GOLDEN_PATH.is_file():
        pytest.fail(f"{GOLDEN_PATH} is missing; run PSP_UPDATE_GOLDEN=1 pytest {Path(__file__).name} to create it")
    return json.loads(GOLDEN_PATH.read_text())


def test_golden_file_is_current(digests):
    if UPDATE:
        import anndata, scanpy  # noqa: PLC0415

        payload = {
            "meta": {
                "python": platform.python_version(),
                "anndata": anndata.__version__,
                "scanpy": scanpy.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "runs": digests,
        }
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden file regenerated at {GOLDEN_PATH}")
    golden = _load_golden()["runs"]
    problems = []
    for run, expected in golden.items():
        actual = digests.get(run)
        if actual is None:
            problems.append(f"{run}: run missing")
            continue
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                problems.append(f"{run}/{key}: no longer produced")
            elif key not in expected:
                problems.append(f"{run}/{key}: new output not in golden file")
            elif expected[key] != actual[key]:
                problems.append(f"{run}/{key}: digest changed")
    assert not problems, (
        "Legacy-mode outputs changed:\n  "
        + "\n  ".join(problems)
        + "\n(If the change is intended, regenerate with PSP_UPDATE_GOLDEN=1 and commit the JSON.)"
    )


def test_golden_covers_the_core_tables():
    """The guard is only useful while the key tables are hashed."""
    golden = _load_golden()["runs"]
    single = golden["single_guide"]
    for key in (
        "tables/guide_assignment",
        "tables/perturbation_full",
        "tables/enrichment_full",
        "tables/effect_matrix",
        "tables/lochness",
        "tables/ps_score",
        "tables/knockdown_filter",
        "obs",
        "structure",
        "report_md",
    ):
        assert key in single, key
    pair = golden["dual_guide_pair"]
    for key in ("tables/guide_assignment", "tables/pair_assignment_summary", "tables/perturbation_full", "obs", "report_md"):
        assert key in pair, key


def test_report_canonicalisation_neutralises_run_specific_lines(tmp_path):
    text = (
        "Run `x` — generated 2026-10-09 01:04. HTML report: `report.html`.\n"
        f"- sample metadata: `{tmp_path}/meta.csv`\n"
        "- package versions: python 3.12.3\nscanpy 1.12.4\n\n"
        "| Git branch / commit | main abc |\n| Command | pytest -q |\n| Random seed | 0 |\n"
        "| load | Input loading | True | completed | 600 cells | 18.6 |\n"
        "- Only 3% of cells received a confident guide call.\n"
    )
    canon = _canon_report(text, tmp_path)
    assert "2026-10-09" not in canon and str(tmp_path) not in canon and "scanpy" not in canon
    assert "Git branch" not in canon and "Command" not in canon
    assert "| Random seed | 0 |" in canon and "| load | Input loading | True | completed | 600 cells | T |" in canon
    assert "Only 3% of cells received a confident guide call." in canon
