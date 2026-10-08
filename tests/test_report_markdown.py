"""The Markdown report must describe the statistics of the assignment mode that ran."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from perturbseq_pipeline.config import Config
from perturbseq_pipeline.plots import FigureRegistry
from perturbseq_pipeline.report import ReportInputs
from perturbseq_pipeline.report_markdown import write_markdown_report


def _inputs(tmp_path, tables):
    cfg = Config()
    cfg.run.outdir = str(tmp_path)
    cfg.report.title = "t"
    reg = FigureRegistry(tmp_path / "figures", cfg)
    return ReportInputs(cfg=cfg, registry=reg, perturbation=None, tables=tables)


def test_single_guide_markdown_describes_the_single_guide_test(tmp_path):
    tables = {
        "perturbation": pd.DataFrame({"target_gene": ["A"], "log2fc_ntc": [-1.0]}),
        "skipped": pd.DataFrame(),
        "enrichment": pd.DataFrame({"target_gene": ["A"], "cluster": ["0"]}),
        "lochness": pd.DataFrame({"target_gene": ["A"]}),
    }
    md = write_markdown_report(_inputs(tmp_path, tables), tmp_path / "report.md").read_text()
    assert "targeting pairs vs NTC pairs" not in md
    assert "ECDF analysis" not in md
    assert "Perturbation strength (target expression vs control)" in md
    assert "BH FDR across all tested targets within each control arm" in md
    # sequential numbering, optional sections only when their table exists
    heads = [line for line in md.splitlines() if line.startswith("## ") and line[3].isdigit()]
    nums = [int(h.split(".")[0][3:]) for h in heads]
    assert nums == list(range(1, len(nums) + 1))
    assert any("Cluster enrichment" in h for h in heads)
    assert any("lochNESS" in h for h in heads)
    assert not any("Per-cell perturbation scores" in h for h in heads)


def test_pair_mode_markdown_keeps_the_pair_methods(tmp_path):
    tables = {"pair_assignment_per_lane": pd.DataFrame({"lane": ["L1"]}), "perturbation": pd.DataFrame()}
    md = write_markdown_report(_inputs(tmp_path, tables), tmp_path / "report.md").read_text()
    assert "targeting pairs vs NTC pairs" in md
    assert "Pair-guide QC" in md
    heads = [line for line in md.splitlines() if line.startswith("## ") and line[3].isdigit()]
    nums = [int(h.split(".")[0][3:]) for h in heads]
    assert nums == list(range(1, len(nums) + 1))
