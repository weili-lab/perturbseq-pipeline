"""Run manifest and module completion status.

Every run writes ``logs/run_manifest.json`` next to ``resolved_config.yaml``.
The manifest records what is needed to reproduce and audit the run without
opening the log: pipeline version, git commit, execution command, input
manifest, assignment mode, random seed, execution and compute mode, which
optional modules were enabled, and the completion status of every stage.

Stage status is tracked with :class:`ModuleStatusTracker`. The tracker also
writes ``logs/module_status.json`` after every update, so a run that fails
half-way still leaves a record of which stages completed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import __version__
from .config import Config
from .provenance import git_info, package_versions, slurm_info

STATUS_COMPLETED = "completed"
#: Enabled, ran, but produced no result (too few cells, missing extra, ...).
STATUS_SKIPPED = "skipped"
#: ``<section>.enabled: false`` in the configuration.
STATUS_DISABLED = "disabled"
STATUS_FAILED = "failed"
#: The run stopped before reaching this stage.
STATUS_NOT_RUN = "not_run"

#: Stage keys in execution order with their report labels.
STAGES: List[tuple] = [
    ("basic_qc", "Basic QC stage (run.stop_after: qc)"),
    ("load", "Input loading"),
    ("qc", "Quality control"),
    ("guide_assignment", "Guide assignment"),
    ("clustering", "Normalization, embedding, clustering"),
    ("perturbation", "Perturbation strength"),
    ("regression", "Membership regression (high-MOI)"),
    ("enrichment", "Cluster enrichment"),
    ("modules", "Co-functional modules and gene programs"),
    ("ps_score", "Per-cell perturbation scores (PS)"),
    ("lochness", "lochNESS neighbourhood enrichment"),
    ("distance", "Perturbation distance vs control"),
    ("distance_space", "Perturbation distance space"),
    ("meta_analysis", "Master perturbation meta table"),
    ("outputs", "Output writing"),
    ("report", "Report"),
]
_LABELS = dict(STAGES)

#: Config sections with an ``enabled`` switch, i.e. the optional modules.
OPTIONAL_MODULES = ("regression", "enrichment", "modules", "ps_score", "lochness", "distance", "distance_space", "meta_analysis")


class ModuleStatusTracker:
    """Record the outcome of each pipeline stage."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._rows: Dict[str, Dict[str, Any]] = {}
        self._started: Dict[str, float] = {}
        self.path = Path(path) if path is not None else None

    def start(self, key: str) -> None:
        self._started[key] = time.time()

    def mark(self, key: str, status: str, note: str = "", *, enabled: bool = True) -> None:
        t0 = self._started.pop(key, None)
        seconds = round(time.time() - t0, 1) if t0 is not None else None
        self._rows[key] = {
            "module": key,
            "label": _LABELS.get(key, key),
            "enabled": bool(enabled),
            "status": status,
            "note": note or "",
            "seconds": seconds,
        }
        self.flush()

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.as_list(), indent=2, default=str))

    def as_list(self) -> List[Dict[str, Any]]:
        return [dict(self._rows[k]) for k, _ in STAGES if k in self._rows]

    def table(self, *, include_not_run: bool = False) -> pd.DataFrame:
        """One row per stage in execution order."""
        rows = []
        for key, label in STAGES:
            if key in self._rows:
                rows.append(self._rows[key])
            elif include_not_run and key != "basic_qc":
                rows.append(
                    {
                        "module": key,
                        "label": label,
                        "enabled": None,
                        "status": STATUS_NOT_RUN,
                        "note": "",
                        "seconds": None,
                    }
                )
        return pd.DataFrame(rows, columns=["module", "label", "enabled", "status", "note", "seconds"])

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self._rows.values():
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out


def execution_command() -> Dict[str, Any]:
    """The command that started this process, plus interpreter and cwd."""
    argv = list(sys.argv) if sys.argv else []
    return {
        "argv": argv,
        "command": " ".join(shlex.quote(a) for a in argv),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
    }


def _file_record(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    rec: Dict[str, Any] = {"path": str(path), "exists": p.exists()}
    if p.is_file():
        rec["size_bytes"] = p.stat().st_size
        rec["modified"] = _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
    elif p.is_dir():
        rec["n_files"] = sum(1 for _ in p.iterdir())
    return rec


def input_manifest(cfg: Config, lanes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Describe every input the configuration points at."""
    mode = cfg.resolved_mode()
    rec: Dict[str, Any] = {"mode": mode}
    inp = cfg.input
    if mode == "mtx":
        mtx = inp.resolved_mtx_dirs()
        guide_dirs = inp.guide_mtx_dirs or {}
        rec["lanes"] = {
            lane: {
                "expression": _file_record(path),
                "guides": _file_record(guide_dirs.get(lane)) if guide_dirs else None,
            }
            for lane, path in mtx.items()
        }
    elif mode == "h5ad":
        rec["h5ad"] = _file_record(inp.h5ad)
        rec["guide_h5ad"] = _file_record(inp.guide_h5ad)
        rec["guide_table"] = _file_record(inp.guide_table)
        rec["guide_obs_column"] = inp.guide_obs_column
        rec["counts_layer"] = inp.counts_layer
        rec["normalized_layer"] = inp.normalized_layer
    elif mode == "samples":
        rec["samples"] = {
            sid: {
                "expression": _file_record(s.gex_h5 or s.gex_mtx_dir),
                "guide_fastq_dir": _file_record(s.guide_fastq_dir),
                "guide_matrix": _file_record(s.guide_matrix),
                "condition_code": s.condition_code,
                "gem_well": s.gem_well,
            }
            for sid, s in cfg.resolved_samples().items()
        }
    if lanes:
        rec["loaded_lanes"] = dict(lanes)
    rec["sample_metadata"] = _file_record(cfg.metadata.file)
    pair_ref = cfg.guides.pair_map_file or cfg.guides.pair_reference
    rec["pair_reference"] = _file_record(pair_ref)
    design = getattr(cfg.guides, "design", None)
    rec["guide_design"] = _file_record(getattr(design, "path", None))
    return rec


def enabled_modules(cfg: Config) -> Dict[str, bool]:
    return {name: bool(getattr(cfg, name).enabled) for name in OPTIONAL_MODULES}


def build_run_manifest(
    cfg: Config,
    *,
    outdir: Path,
    status: ModuleStatusTracker,
    config_path: Optional[str] = None,
    lanes: Optional[Dict[str, str]] = None,
    execution_mode: str = "",
    outputs: Optional[Dict[str, str]] = None,
    counts: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the JSON-serialisable run manifest."""
    from .compute import detect_available_cpus, detect_slurm_cpus, is_gpu_available

    outdir = Path(outdir)
    try:
        gpu = bool(is_gpu_available())
    except Exception:  # pragma: no cover - optional GPU stack
        gpu = False
    rec: Dict[str, Any] = {
        "manifest_version": 1,
        "pipeline_version": __version__,
        "run_name": cfg.run.name,
        "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_seconds": round(runtime_seconds, 1) if runtime_seconds is not None else None,
        "git": git_info(),
        "execution": execution_command(),
        "config_path": str(config_path) if config_path else None,
        "resolved_config": str(outdir / "logs" / "resolved_config.yaml"),
        "random_seed": cfg.run.seed,
        "input": input_manifest(cfg, lanes),
        "guides": {
            "assignment_mode": cfg.guides.assignment_mode,
            "pair_assignment_primary": cfg.guides.pair_assignment_primary,
            "scaffold_classes": list(cfg.guides.scaffold_classes),
            "target_feature_column": cfg.guides.target_feature_column,
            "min_umi": cfg.guides.min_umi,
            "dominance_ratio": cfg.guides.dominance_ratio,
            "max_second_umi": cfg.guides.max_second_umi,
        },
        "execution_mode": execution_mode,
        "scaling_mode": cfg.scaling.mode,
        "storage_mode": cfg.storage.mode,
        "compute": {
            "backend": cfg.compute.backend,
            "n_jobs_configured": cfg.compute.n_jobs,
            "available_cpus": detect_available_cpus(),
            "slurm_cpus": detect_slurm_cpus(),
            "gpu_available": gpu,
        },
        "enabled_modules": enabled_modules(cfg),
        "module_status": status.as_list(),
        "module_status_summary": status.summary(),
        "counts": counts or {},
        "outputs": dict(outputs or {}),
        "warnings": list(warnings or []),
        "environment": {
            "hostname": os.uname().nodename,
            "user": os.environ.get("USER"),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "slurm": slurm_info(),
            "packages": package_versions(),
        },
    }
    return rec


def write_run_manifest(rec: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str))
    return path


def manifest_summary_rows(rec: Dict[str, Any]) -> List[tuple]:
    """Key facts for the report's provenance table."""
    git = rec.get("git") or {}
    comp = rec.get("compute") or {}
    enabled = rec.get("enabled_modules") or {}
    on = ", ".join(k for k, v in enabled.items() if v) or "none"
    off = ", ".join(k for k, v in enabled.items() if not v) or "none"
    commit = git.get("commit") or "not a git checkout"
    if git.get("dirty"):
        commit += " (uncommitted changes)"
    rows = [
        ("Pipeline version", rec.get("pipeline_version", "")),
        ("Git branch / commit", f"{git.get('branch') or ''} {commit}".strip()),
        ("Command", (rec.get("execution") or {}).get("command", "")),
        ("Configuration file", rec.get("config_path") or "(in-memory Config)"),
        ("Resolved configuration", rec.get("resolved_config", "")),
        ("Random seed", rec.get("random_seed", "")),
        ("Input mode", (rec.get("input") or {}).get("mode", "")),
        ("Guide assignment mode", (rec.get("guides") or {}).get("assignment_mode", "")),
        ("Execution mode", str(rec.get("execution_mode", "")).upper()),
        (
            "Compute",
            f"backend={comp.get('backend')} · n_jobs={comp.get('n_jobs_configured')} · "
            f"available CPUs={comp.get('available_cpus')} · GPU={'yes' if comp.get('gpu_available') else 'no'}",
        ),
        ("Modules enabled", on),
        ("Modules disabled", off),
        ("SLURM job", ((rec.get("environment") or {}).get("slurm") or {}).get("slurm_job_id") or "none"),
        (
            "Run manifest",
            str(Path(rec.get("resolved_config", "")).with_name("run_manifest.json"))
            if rec.get("resolved_config")
            else "",
        ),
    ]
    return rows
