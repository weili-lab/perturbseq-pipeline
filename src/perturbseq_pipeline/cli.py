"""Pipeline driver and command-line interface.

    perturbseq-pipeline run --config config/demo.yaml
    perturbseq-pipeline init-config my_run.yaml

:func:`run_pipeline` is the same entry point the demo notebook uses, so a
notebook run and a CLI run execute identical code.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import __version__
from .config import Config

logger = logging.getLogger("perturbseq_pipeline")


@dataclass
class PipelineResult:
    """Paths and objects produced by a run."""

    outdir: Path
    report: Path
    h5ad: Path
    guide_h5ad: Optional[Path]
    #: ``.tar.gz`` bundle of the run outputs (None when archiving is disabled).
    archive: Optional[Path] = None
    tables: Dict[str, Path] = field(default_factory=dict)
    figures_dir: Optional[Path] = None
    n_cells: int = 0
    n_genes: int = 0
    n_targets_tested: int = 0
    n_effective: int = 0
    runtime_seconds: float = 0.0
    #: The processed AnnData, for interactive follow-up in a notebook.
    adata: object = None
    perturbation_table: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        return (
            f"{self.n_cells:,} cells x {self.n_genes:,} genes | "
            f"{self.n_effective}/{self.n_targets_tested} targets effectively perturbed | "
            f"report: {self.report}"
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(outdir: Path, verbose: bool = False) -> Path:
    """Log to both the console and ``<outdir>/logs/run.log``."""
    logdir = Path(outdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / "run.log"

    root = logging.getLogger("perturbseq_pipeline")
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    fileh = logging.FileHandler(logfile, mode="w")
    fileh.setFormatter(fmt)
    root.addHandler(fileh)
    return logfile


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_pipeline(cfg: Config, verbose: bool = False) -> PipelineResult:
    """Run every stage and produce the three deliverables."""
    start = time.time()
    cfg.validate()

    outdir = Path(cfg.run.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir, verbose)
    logger.info("perturbseq-pipeline v%s — run %r", __version__, cfg.run.name)
    cfg.dump_yaml(outdir / "logs" / "resolved_config.yaml")

    # Imports are deferred so ``--help`` does not pay the scanpy import cost.
    import numpy as np
    import scanpy as sc

    from . import cluster as cluster_mod
    from . import enrichment as enrich_mod
    from . import guides as guides_mod
    from . import io as io_mod
    from . import perturbation as pert_mod
    from . import plots as plots_mod
    from . import qc as qc_mod
    from .report import ReportInputs, build_report

    sc.settings.verbosity = 1
    np.random.seed(cfg.run.seed)

    registry = plots_mod.FigureRegistry(outdir=outdir, cfg=cfg)
    tables: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []

    # --- 1. load ----------------------------------------------------------
    logger.info("=== Stage 1/8: loading input ===")
    data = io_mod.load_data(cfg)
    expr, guides = data.expr, data.guides
    n_cells_input = expr.n_obs

    # --- 2. QC ------------------------------------------------------------
    logger.info("=== Stage 2/8: quality control ===")
    expr = qc_mod.prefilter(expr, cfg)
    expr = qc_mod.compute_qc_metrics(expr, cfg)
    plots_mod.plot_qc(expr, registry, stage="before filtering")
    expr, qc_steps = qc_mod.filter_cells_and_genes(expr, cfg)
    plots_mod.plot_qc(expr, registry, stage="after filtering")
    tables["qc_steps"] = qc_steps
    tables["qc_summary"] = qc_mod.qc_summary_table(expr)

    # --- 3. guide assignment ---------------------------------------------
    logger.info("=== Stage 3/8: guide assignment ===")
    expr = guides_mod.assign_guides(expr, guides, cfg)
    tables["guide_qc"] = qc_mod.guide_qc_summary(expr, cfg)
    tables["guide_assignment"] = guides_mod.assignment_summary(expr, cfg)
    per_lane = guides_mod.per_lane_assignment(expr)
    if not per_lane.empty:
        tables["assignment_per_lane"] = per_lane
    if guides is not None:
        tables["guide_representation"] = guides_mod.guide_representation(guides, expr)
    warnings.extend(qc_mod.check_guide_qc(expr, cfg))
    plots_mod.plot_guide_qc(expr, guides, registry, cfg)

    # --- 4. clustering ----------------------------------------------------
    logger.info("=== Stage 4/8: normalization, embedding, clustering ===")
    expr = cluster_mod.normalize(expr, cfg)
    expr = cluster_mod.embed_and_cluster(expr, cfg)
    tables["clusters"] = cluster_mod.cluster_summary(expr)
    plots_mod.plot_clustering(expr, registry, cfg)

    # --- 5. perturbation strength ----------------------------------------
    logger.info("=== Stage 5/8: perturbation strength ===")
    results = pert_mod.test_all_targets(expr, cfg)
    tables["perturbation"] = pert_mod.format_results_table(results, cfg)
    tables["perturbation_full"] = results.table
    tables["skipped"] = results.skipped
    plots_mod.plot_perturbation_overview(results, registry, cfg)
    plots_mod.plot_per_target(expr, results, registry, cfg)

    # --- 6. cluster enrichment -------------------------------------------
    enrichment = None
    if cfg.enrichment.enabled:
        logger.info("=== Stage 6/8: perturbation enrichment across clusters ===")
        enrichment = enrich_mod.test_cluster_enrichment(expr, cfg)
        tables["enrichment"] = enrich_mod.format_enrichment_table(enrichment)
        tables["enrichment_full"] = enrichment.table
        tables["enrichment_composition"] = enrichment.composition.reset_index(
            names="target_gene"
        )
        tables["enrichment_effect_magnitude"] = enrichment.effect_magnitude
        plots_mod.plot_enrichment(expr, enrichment, registry, cfg)
        plots_mod.plot_enrichment_per_target(expr, enrichment, registry, cfg)
    else:
        logger.info("Cluster enrichment disabled (enrichment.enabled: false)")

    # --- 7. write deliverables -------------------------------------------
    logger.info("=== Stage 7/8: writing outputs ===")
    tabledir = outdir / "tables"
    tabledir.mkdir(parents=True, exist_ok=True)
    table_paths: Dict[str, Path] = {}
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        p = tabledir / f"{name}.csv"
        df.to_csv(p, index=False)
        table_paths[name] = p
    tables["manifest"] = registry.manifest()
    registry.manifest().to_csv(tabledir / "figure_manifest.csv", index=False)

    h5ad_path = io_mod.write_h5ad(expr, outdir / cfg.output.h5ad_name)
    h5ad_path = io_mod.relocate_if_large(h5ad_path, cfg)

    guide_h5ad_path: Optional[Path] = None
    if guides is not None and cfg.output.write_guide_h5ad:
        gname = Path(cfg.output.h5ad_name).stem + "_guides.h5ad"
        guide_h5ad_path = io_mod.write_h5ad(guides[expr.obs_names].copy(), outdir / gname)
        guide_h5ad_path = io_mod.relocate_if_large(guide_h5ad_path, cfg)

    # --- 8. report --------------------------------------------------------
    logger.info("=== Stage 8/8: building report ===")
    n_hits = len(results.hits) if not results.table.empty else 0
    outputs = {
        "Processed h5ad": str(h5ad_path),
        "Report": str(outdir / cfg.output.report_name),
        "Figures": str(registry.figdir),
        "Per-target figures": str(registry.figdir / plots_mod.SECTION_PER_GENE),
        "Tables": str(tabledir),
        "Log": str(outdir / "logs" / "run.log"),
    }
    if guide_h5ad_path:
        outputs["Guide count h5ad"] = str(guide_h5ad_path)
    archive_name = cfg.output.archive_name or f"{cfg.run.name}_results.tar.gz"
    if cfg.output.archive:
        outputs["Results archive"] = str(outdir / archive_name)

    report_inputs = ReportInputs(
        cfg=cfg,
        registry=registry,
        perturbation=results,
        enrichment=enrichment,
        tables=tables,
        warnings=warnings,
        summary_cards=[
            ("Cells analysed", f"{expr.n_obs:,}"),
            ("Genes", f"{expr.n_vars:,}"),
            ("Lanes", f"{data.n_lanes}"),
            ("Target genes", f"{guides_mod.target_genes(expr, cfg).size}"),
            ("Clusters", f"{expr.obs[cluster_mod.CLUSTER_KEY].nunique()}"),
            ("Effective knockdowns", f"{n_hits}"),
        ],
        input_mode=cfg.resolved_mode(),
        input_source="; ".join(f"{k}: {v}" for k, v in data.lanes.items()),
        lanes=f"{data.n_lanes} ({', '.join(data.lanes)})",
        metadata_source=cfg.metadata.file or "",
        guide_source_text=(
            "guide count matrix"
            if data.guide_source == "matrix"
            else f"pre-computed labels in obs[{cfg.input.guide_obs_column!r}]"
        ),
        outputs=outputs,
    )
    report_path = build_report(report_inputs, outdir / cfg.output.report_name)

    # Archive last, so the bundle contains the finished report.
    archive_path = io_mod.archive_results(outdir, cfg)

    runtime = time.time() - start
    logger.info(
        "Done in %.1f s — %d/%d input cells retained", runtime, expr.n_obs, n_cells_input
    )

    result = PipelineResult(
        outdir=outdir,
        report=report_path,
        h5ad=h5ad_path,
        guide_h5ad=guide_h5ad_path,
        archive=archive_path,
        tables=table_paths,
        figures_dir=registry.figdir,
        n_cells=expr.n_obs,
        n_genes=expr.n_vars,
        n_targets_tested=len(results.table),
        n_effective=n_hits,
        runtime_seconds=runtime,
        adata=expr,
        perturbation_table=results.table,
    )
    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="perturbseq-pipeline",
        description="Perturb-seq QC, clustering and perturbation-strength pipeline.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline from a config file")
    run.add_argument("-c", "--config", required=True, help="path to the run config YAML")
    run.add_argument("-o", "--outdir", default=None, help="override run.outdir")
    run.add_argument("-n", "--name", default=None, help="override run.name")
    run.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")

    init = sub.add_parser("init-config", help="write a fully-commented default config")
    init.add_argument("path", help="where to write the config YAML")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "init-config":
        cfg = Config()
        cfg.dump_yaml(Path(args.path))
        print(f"Wrote default configuration to {args.path}")
        print("Edit input.mtx_dirs (10x mode) or input.h5ad (h5ad mode), then run:")
        print(f"  perturbseq-pipeline run --config {args.path}")
        return 0

    cfg = Config.from_yaml(args.config)
    if args.outdir:
        cfg.run.outdir = args.outdir
    if args.name:
        cfg.run.name = args.name

    try:
        result = run_pipeline(cfg, verbose=args.verbose)
    except Exception as exc:  # surfaced as a clean message, full trace in the log
        logger.exception("Pipeline failed: %s", exc)
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
