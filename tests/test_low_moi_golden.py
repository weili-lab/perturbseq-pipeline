"""Golden-table regression guard for the established assignment modes.

Runs the synthetic fixture end to end in ``single_guide`` mode (every optional
stage on) and in ``dual_guide_pair`` mode, canonicalises every output table and
the per-cell ``obs`` frame, and compares SHA-256 digests against
``tests/golden/low_moi_golden.json``.

Purpose: new assignment modes (high-MOI membership calling) must leave the
existing modes byte-identical. Per-stage unit tests check semantics; this file
catches the glue edits (config defaults, cli dispatch, io round-trip, QC text)
that those tests do not see.

Output tables are stored in ``tests/golden/low_moi_tables.json`` as canonical
rows (columns sorted by name, rows sorted lexicographically, so row order from
parallel workers does not matter) and compared cell by cell: labels, integers
and booleans exactly, floats with a relative tolerance (``FLOAT_RTOL``). A
digest of floats rounded to N significant digits is not used for tables: a
value that sits on a rounding boundary flips its last digit under harmless
last-ulp noise from a different SciPy/NumPy build (e.g. a Mann-Whitney p-value
moving by 2e-6 relative between SciPy 1.16 and 1.18), which failed the guard
on a stock Colab image although nothing changed. A failure names the table,
column and row that moved.

The per-cell ``obs`` frame, the AnnData structure and the Markdown report
(warnings, QC text, section text and tables) are still compared by SHA-256
digest in ``tests/golden/low_moi_golden.json``, the report after neutralising
the run-specific parts: timestamp, absolute paths, git commit, command line,
package versions and stage timings.

Regenerate the golden files (only when a change to the legacy output is
intended, or after a dependency upgrade changes clustering)::

    PSP_UPDATE_GOLDEN=1 pytest tests/test_low_moi_golden.py

and commit both JSON files together with the change that motivated it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from make_synthetic import make_dataset, make_split_lane  # noqa: E402
from test_pair_mode_pipeline import _pair_up  # noqa: E402

from perturbseq_pipeline.cli import run_pipeline  # noqa: E402
from perturbseq_pipeline.config import Config  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "golden" / "low_moi_golden.json"
GOLDEN_TABLES_PATH = Path(__file__).parent / "golden" / "low_moi_tables.json"
UPDATE = bool(os.environ.get("PSP_UPDATE_GOLDEN"))

#: Tables whose content is intrinsically run-specific (timings, absolute paths).
VOLATILE_TABLES = {"compute_profile", "figure_manifest", "module_status"}

#: Float tolerance for table cells. Cross-build numerical noise is ~1e-6
#: relative; a genuine change to a statistic is orders of magnitude larger.
FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-10
#: Differing cells listed per table in a failure message.
MAX_REPORTED_CELLS = 3


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


def _canon_table(df: pd.DataFrame) -> Dict[str, Any]:
    """Canonical, JSON-serialisable form of a table for tolerant comparison.

    Columns are sorted by name and rows lexicographically (floats rounded to 5
    significant digits *for ordering only*); float columns keep full precision
    so they can be compared with a tolerance, everything else is a string.
    """
    cols = sorted(df.columns.astype(str))
    df = df.copy()
    df.columns = df.columns.astype(str)
    float_cols = [c for c in cols if pd.api.types.is_float_dtype(df[c])]
    key = pd.DataFrame(index=df.index)
    for c in cols:
        if c in float_cols:
            vals = df[c].to_numpy(dtype=float)
            key[c] = np.array([f"{v:.5g}" if np.isfinite(v) else repr(float(v)) for v in vals], dtype=object)
        else:
            key[c] = df[c].astype(object).astype(str).to_numpy()
    order = key.sort_values(cols, kind="mergesort").index
    rows = []
    for idx in order:
        row = []
        for c in cols:
            if c in float_cols:
                v = float(df.at[idx, c])
                row.append(v if np.isfinite(v) else repr(v))
            else:
                row.append(key.at[idx, c])
        rows.append(row)
    return {"columns": cols, "float_columns": float_cols, "rows": rows}


def _as_float(v: Any) -> float:
    return float(v)  # "nan" / "inf" / "-inf" strings and numbers alike


def _compare_table(name: str, expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    """Return human-readable problems; empty when the tables agree."""
    if expected["columns"] != actual["columns"]:
        exp, act = set(expected["columns"]), set(actual["columns"])
        return [f"{name}: columns changed (removed {sorted(exp - act)}, added {sorted(act - exp)})"]
    if len(expected["rows"]) != len(actual["rows"]):
        return [f"{name}: row count {len(expected['rows'])} -> {len(actual['rows'])}"]
    floats = set(expected["float_columns"]) | set(actual["float_columns"])
    cells = []
    for r, (erow, arow) in enumerate(zip(expected["rows"], actual["rows"])):
        for c, ev, av in zip(expected["columns"], erow, arow):
            if c in floats:
                e, a = _as_float(ev), _as_float(av)
                same = (np.isnan(e) and np.isnan(a)) or bool(np.isclose(a, e, rtol=FLOAT_RTOL, atol=FLOAT_ATOL))
            else:
                same = str(ev) == str(av)
            if not same:
                cells.append(f"row {r} {c}: {ev!r} -> {av!r}")
    if not cells:
        return []
    shown = "; ".join(cells[:MAX_REPORTED_CELLS])
    more = f" (+{len(cells) - MAX_REPORTED_CELLS} more)" if len(cells) > MAX_REPORTED_CELLS else ""
    return [f"{name}: {len(cells)} cell(s) changed — {shown}{more}"]


def _collect_tables(outdir: Path) -> Dict[str, Dict[str, Any]]:
    tables: Dict[str, Dict[str, Any]] = {}
    for path in sorted((outdir / "tables").glob("*.csv")):
        if path.stem in VOLATILE_TABLES:
            continue
        df = pd.read_csv(path, low_memory=False)
        tables[f"tables/{path.stem}"] = _canon_table(df)
    return tables


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
def outputs(tmp_path_factory) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per run: ``digests`` (obs, structure, report_md) and canonical ``tables``."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {"digests": {}, "tables": {}}
    for name, fn in RUNS.items():
        root = tmp_path_factory.mktemp(f"golden_{name}")
        result, outdir = fn(root)
        d = dict(_digest_adata(result.adata))
        d.update(_digest_report(outdir, root))
        out["digests"][name] = d
        out["tables"][name] = _collect_tables(outdir)
    return out


def _load_json(path: Path) -> dict:
    if not path.is_file():
        pytest.fail(f"{path} is missing; run PSP_UPDATE_GOLDEN=1 pytest {Path(__file__).name} to create it")
    return json.loads(path.read_text())


def _load_golden() -> dict:
    return _load_json(GOLDEN_PATH)


def _load_golden_tables() -> dict:
    return _load_json(GOLDEN_TABLES_PATH)


def _environment() -> Dict[str, str]:
    import importlib.metadata as md  # noqa: PLC0415

    env = {"python": platform.python_version()}
    for pkg in ("anndata", "scanpy", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels"):
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = "not installed"
    return env


def _key_problems(run: str, expected: dict, actual: dict) -> List[str]:
    problems = []
    for key in sorted(set(expected) | set(actual)):
        if key not in actual:
            problems.append(f"{run}/{key}: no longer produced")
        elif key not in expected:
            problems.append(f"{run}/{key}: new output not in golden file")
    return problems


def test_golden_file_is_current(outputs):
    if UPDATE:
        meta = _environment()
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps({"meta": meta, "runs": outputs["digests"]}, indent=2, sort_keys=True) + "\n")
        GOLDEN_TABLES_PATH.write_text(json.dumps({"meta": meta, "runs": outputs["tables"]}, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"golden files regenerated at {GOLDEN_PATH} and {GOLDEN_TABLES_PATH}")
    golden = _load_golden()["runs"]
    golden_tables = _load_golden_tables()["runs"]
    problems: List[str] = []
    for run in sorted(set(golden) | set(golden_tables) | set(RUNS)):
        if run not in outputs["digests"]:
            problems.append(f"{run}: run missing")
            continue
        # obs / structure / report: exact digests.
        expected, actual = golden.get(run, {}), outputs["digests"][run]
        problems += _key_problems(run, expected, actual)
        for key in sorted(set(expected) & set(actual)):
            if expected[key] != actual[key]:
                problems.append(f"{run}/{key}: digest changed")
        # tables: exact labels, tolerant floats.
        expected_t, actual_t = golden_tables.get(run, {}), outputs["tables"][run]
        problems += _key_problems(run, expected_t, actual_t)
        for key in sorted(set(expected_t) & set(actual_t)):
            problems += _compare_table(f"{run}/{key}", expected_t[key], actual_t[key])
    assert not problems, (
        "Legacy-mode outputs changed:\n  "
        + "\n  ".join(problems)
        + "\n(If the change is intended, regenerate with PSP_UPDATE_GOLDEN=1 and commit both JSON files.)"
    )


def test_golden_covers_the_core_tables():
    """The guard is only useful while the key outputs are recorded."""
    golden = _load_golden()["runs"]
    tables = _load_golden_tables()["runs"]
    for key in ("obs", "structure", "report_md"):
        assert key in golden["single_guide"], key
    for key in ("obs", "report_md"):
        assert key in golden["dual_guide_pair"], key
    for key in (
        "tables/guide_assignment",
        "tables/perturbation_full",
        "tables/enrichment_full",
        "tables/effect_matrix",
        "tables/lochness",
        "tables/ps_score",
        "tables/knockdown_filter",
    ):
        assert key in tables["single_guide"], key
    for key in ("tables/guide_assignment", "tables/pair_assignment_summary", "tables/perturbation_full"):
        assert key in tables["dual_guide_pair"], key


def test_table_comparison_tolerates_build_noise_but_not_real_changes():
    """Cross-build last-digit noise passes; a changed statistic or label fails."""
    base = pd.DataFrame(
        {"target_gene": ["A", "B"], "fdr_mwu": [5.004655e-07, 0.2], "n_cells": [14, 20], "ks_pval": [np.nan, 1e-20]}
    )
    golden = _canon_table(base)
    # The value that broke the digest guard (SciPy 1.16 vs 1.18): 5.0047e-07 vs 5.0046e-07 at 5 s.f.
    noisy = base.assign(fdr_mwu=[5.004647e-07, 0.2], ks_pval=[np.nan, 1.0000001e-20])
    assert _compare_table("t", golden, _canon_table(noisy)) == []
    assert _canon_frame(base) != _canon_frame(noisy)  # the old 5-s.f. digest did flag it
    # Row order does not matter.
    assert _compare_table("t", golden, _canon_table(base.iloc[::-1])) == []
    # Real changes are caught and named.
    assert _compare_table("t", golden, _canon_table(base.assign(fdr_mwu=[5.1e-07, 0.2])))
    assert _compare_table("t", golden, _canon_table(base.assign(target_gene=["A", "C"])))
    assert _compare_table("t", golden, _canon_table(base.assign(n_cells=[14, 21])))
    assert _compare_table("t", golden, _canon_table(base.assign(ks_pval=[0.5, 1e-20])))  # NaN -> value
    assert "columns changed" in _compare_table("t", golden, _canon_table(base.drop(columns="n_cells")))[0]


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
