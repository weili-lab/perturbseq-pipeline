"""Pipeline driver and command-line interface.

    perturbseq-pipeline run --config config/demo.yaml
    perturbseq-pipeline init-config my_run.yaml

:func:`run_pipeline` is the same entry point used by notebooks and the CLI, so
both execute the same analytical workflow.

Adaptive execution
------------------
The analytical stages themselves decide whether to use STANDARD or LARGE
implementations. This driver adds a corresponding orchestration layer.

STANDARD mode
    Used for ordinary datasets such as Replogle (~310k cells). Existing
    behaviour is preserved as closely as possible.

LARGE mode
    Used automatically for million-cell datasets such as KOLF. The biological
    analysis is unchanged, but object lifetime and output handling are made
    memory-aware:

    * unnecessary AnnData copies are avoided;
    * large tables can be written immediately rather than retained twice;
    * large DataFrames are not duplicated solely for ``reset_index``;
    * intermediate references are released between stages;
    * Python garbage collection is requested after expensive stages;
    * guide matrices are only subset/copied when genuinely required;
    * optional all-cell H5AD output never blindly copies the full object.

The goal is not to alter results between STANDARD and LARGE runs, but to prevent
the driver itself from becoming the memory bottleneck after individual
analytical modules have been made scalable.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import __version__
from .config import Config
from .run_manifest import (
    STATUS_COMPLETED,
    STATUS_DISABLED,
    STATUS_SKIPPED,
    ModuleStatusTracker,
    build_run_manifest,
    manifest_summary_rows,
    write_run_manifest,
)


logger = logging.getLogger("perturbseq_pipeline")


# Result object


@dataclass
class PipelineResult:
    """Paths and objects produced by a run."""

    outdir: Path
    report: Path
    h5ad: Path
    guide_h5ad: Optional[Path]
    #: Pre-filter/all-cell object when requested.
    unfiltered_h5ad: Optional[Path] = None
    #: Optional .tar.gz result bundle.
    archive: Optional[Path] = None
    tables: Dict[str, Path] = field(default_factory=dict)
    figures_dir: Optional[Path] = None
    n_cells: int = 0
    n_genes: int = 0
    n_targets_tested: int = 0
    n_effective: int = 0
    runtime_seconds: float = 0.0
    #: The processed AnnData for notebook follow-up.
    adata: object = None
    perturbation_table: Optional[pd.DataFrame] = None
    distance_table: Optional[pd.DataFrame] = None
    distance_space_results: Optional[object] = None
    meta_table: Optional[pd.DataFrame] = None
    compute_profile: Optional[pd.DataFrame] = None
    #: STANDARD / LARGE, useful for provenance.
    execution_mode: str = "standard"
    #: :class:`perturbseq_pipeline.basic_qc.BasicQCResult` when the run
    #: stopped after the basic QC stage.
    basic_qc: object = None
    #: ``logs/run_manifest.json``: git commit, command, inputs, assignment
    #: mode, seed, enabled modules and per-stage completion status.
    run_manifest: Optional[Path] = None
    #: One row per stage with its completion status.
    module_status: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        if self.basic_qc is not None:
            b = self.basic_qc
            return (
                f"BASIC QC | {b.n_cells_all:,} cells loaded, "
                f"{b.n_cells_pass:,} pass expression QC x {b.n_genes:,} genes | "
                f"predicted doublets retained {b.n_predicted_doublets_all:,} (all) / "
                f"{b.n_predicted_doublets_pass:,} (QC object) | "
                f"guide multiplets retained {b.n_guide_multiplets_all:,} / "
                f"{b.n_guide_multiplets_pass:,} | report: {self.report}"
            )
        return (
            f"{self.n_cells:,} cells x "
            f"{self.n_genes:,} genes | "
            f"{self.n_effective}/"
            f"{self.n_targets_tested} targets effectively perturbed | "
            f"mode: {self.execution_mode.upper()} | "
            f"report: {self.report}"
        )


# Logging


def setup_logging(outdir: Path, verbose: bool = False) -> Path:
    """Log to console and ``<outdir>/logs/run.log``."""
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
    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return logfile


# Memory logging


def _log_memory(label: str, cfg: Optional[Config] = None) -> None:
    """Log resident memory when psutil is available.

    The dependency is optional; absence never affects the run.
    """
    if cfg is not None and not cfg.scaling.log_memory:
        return
    try:
        import os
        import psutil

        process = psutil.Process(os.getpid())
        rss_gb = process.memory_info().rss / 1024**3
        logger.info("Memory after %s: %.1f GB RSS", label, rss_gb)
    except Exception:
        return


def _collect(label: Optional[str] = None, cfg: Optional[Config] = None, large_mode: bool = False) -> None:
    """Release unreachable Python objects between expensive stages."""
    should_collect = True
    if cfg is not None and large_mode:
        should_collect = cfg.scaling.collect_between_stages
    if should_collect:
        gc.collect()
    if label:
        _log_memory(label, cfg=cfg)


# Small helpers


def _assigned_singlet_mask(expr):
    """Boolean mask of targeting / non-targeting assigned cells."""
    from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS

    if OBS_CLASS not in expr.obs.columns:
        return None
    return expr.obs[OBS_CLASS].astype(str).isin([CLASS_TARGETING, CLASS_NTC]).to_numpy()


# Table writing helpers


def _write_table(name: str, df: Optional[pd.DataFrame], tabledir: Path, table_paths: Dict[str, Path]) -> Optional[Path]:
    """Write one table without making another DataFrame copy."""
    if df is None or len(df) == 0:
        return None
    path = tabledir / f"{name}.csv"
    df.to_csv(path, index=False)
    table_paths[name] = path
    return path


def _write_indexed_matrix(
    name: str, df: Optional[pd.DataFrame], index_name: str, tabledir: Path, table_paths: Dict[str, Path]
) -> Optional[Path]:
    """Write an indexed matrix directly, avoiding ``reset_index()``.

    ``reset_index`` duplicates the entire matrix. For a KOLF-scale
    perturbation x gene effect matrix that can mean hundreds of MB of
    unnecessary extra memory.
    """
    if df is None or df.empty:
        return None
    path = tabledir / f"{name}.csv"
    old_name = df.index.name
    try:
        df.index.name = index_name
        df.to_csv(path, index=True)
    finally:
        df.index.name = old_name
    table_paths[name] = path
    return path


def _table_for_report(
    tables: Dict[str, pd.DataFrame],
    name: str,
    df: Optional[pd.DataFrame],
    *,
    large_mode: bool,
    max_rows_large: int = 500,
) -> None:
    """Keep only report-sized tables in memory during LARGE runs.

    Full tables are still written to disk separately. The HTML report does not
    benefit from receiving hundreds of thousands of rows.
    """
    if df is None or len(df) == 0:
        return
    if large_mode and len(df) > max_rows_large:
        tables[name] = df.head(max_rows_large).copy()
    else:
        tables[name] = df


# All-cell H5AD writing


def _write_unfiltered_object(expr, guides, cfg, outdir: Path, io_mod) -> Path:
    """Write the all-cell object without blindly copying AnnData.

    If no guide matrix exists, ``merge_guides_into_expr`` would do nothing, so
    copying ``expr`` first would be pure memory waste.

    If a guide matrix does exist, STANDARD mode retains the safe copy behaviour.
    For LARGE mode we temporarily merge the guide matrix into the existing
    object, write it, then remove the merged guide keys again.
    """
    uname = cfg.output.unfiltered_h5ad_name or (Path(cfg.output.h5ad_name).stem + "_all_cells.h5ad")
    dest = outdir / uname
    if guides is None:
        path = io_mod.write_h5ad(expr, dest)
        return io_mod.relocate_if_large(path, cfg)
    large_mode = cfg.use_large_mode(expr.n_obs)
    if not large_mode:
        work = expr.copy()
        work = io_mod.merge_guides_into_expr(work, guides, cfg)
        path = io_mod.write_h5ad(work, dest)
        del work
        _collect(cfg=cfg, large_mode=large_mode)
        return io_mod.relocate_if_large(path, cfg)
    # LARGE path:
    # merge temporarily into expr itself, write, then remove merged guide keys.
    key = cfg.output.guide_obsm_key
    had_obsm = key in expr.obsm
    old_obsm = expr.obsm[key] if had_obsm else None
    had_names = "guide_names" in expr.uns
    old_names = expr.uns["guide_names"] if had_names else None
    had_targets = "guide_target_genes" in expr.uns
    old_targets = expr.uns["guide_target_genes"] if had_targets else None
    expr = io_mod.merge_guides_into_expr(expr, guides, cfg)
    path = io_mod.write_h5ad(expr, dest)
    # Restore pre-write state.
    if not had_obsm:
        expr.obsm.pop(key, None)
    else:
        expr.obsm[key] = old_obsm
    if old_names is None:
        expr.uns.pop("guide_names", None)
    else:
        expr.uns["guide_names"] = old_names
    if old_targets is None:
        expr.uns.pop("guide_target_genes", None)
    else:
        expr.uns["guide_target_genes"] = old_targets
    _collect("large all-cell h5ad write", cfg=cfg, large_mode=large_mode)
    return io_mod.relocate_if_large(path, cfg)


# Guide-output helper


def _aligned_guides(guides, expr):
    """Return guide object aligned to expr without copying when already aligned."""
    if guides is None:
        return None
    if len(guides.obs_names) == len(expr.obs_names) and guides.obs_names.equals(expr.obs_names):
        return guides
    return guides[expr.obs_names].copy()


# Driver


def run_pipeline(cfg: Config, verbose: bool = False, config_path: Optional[str] = None) -> PipelineResult:
    """Run the full perturb-seq workflow.

    When ``run.stop_after == "qc"`` only the basic QC stage runs
    (:mod:`perturbseq_pipeline.basic_qc`) and the function returns after the
    QC-level objects, tables, figures and report have been written.
    """
    start = time.time()
    cfg.validate()
    outdir = Path(cfg.run.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir, verbose)
    logger.info("perturbseq-pipeline v%s — run %r", __version__, cfg.run.name)
    cfg.dump_yaml(outdir / "logs" / "resolved_config.yaml")
    # Deferred imports keep --help fast.
    import numpy as np
    import scanpy as sc
    from . import cluster as cluster_mod
    from . import distance as dist_mod
    from . import enrichment as enrich_mod
    from . import guides as guides_mod
    from . import io as io_mod
    from . import knockdown_filter as kd_mod
    from . import lochness as loch_mod
    from . import meta as meta_mod
    from . import modules as modules_mod
    from . import perturbation as pert_mod
    from . import plots as plots_mod
    from . import ps_score as ps_mod
    from . import qc as qc_mod
    from .compute import ComputeProfiler, detect_available_cpus, detect_slurm_cpus, is_gpu_available, stage_profile
    from .report import ReportInputs, build_report

    profiler = ComputeProfiler()
    slurm_cpus = detect_slurm_cpus()
    logger.info(
        "Compute environment: %d available CPUs%s, GPU available: %s, backend=%s",
        detect_available_cpus(),
        f" (SLURM: {slurm_cpus})" if slurm_cpus else "",
        is_gpu_available(),
        cfg.compute.backend,
    )
    sc.settings.verbosity = 1
    np.random.seed(cfg.run.seed)
    registry = plots_mod.FigureRegistry(outdir=outdir, cfg=cfg)
    # ``tables`` is for report consumption, not necessarily every full table.
    tables: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []
    tabledir = outdir / "tables"
    tabledir.mkdir(parents=True, exist_ok=True)
    table_paths: Dict[str, Path] = {}
    status = ModuleStatusTracker(outdir / "logs" / "module_status.json")

    def _finish_manifest(*, lanes=None, execution_mode="", outputs=None, counts=None, runtime=None):
        rec = build_run_manifest(
            cfg,
            outdir=outdir,
            status=status,
            config_path=config_path,
            lanes=lanes,
            execution_mode=execution_mode,
            outputs=outputs,
            counts=counts,
            warnings=warnings,
            runtime_seconds=runtime,
        )
        path = write_run_manifest(rec, outdir / "logs" / "run_manifest.json")
        logger.info("Wrote run manifest %s", path)
        return rec, path

    # =====================================================================
    # Basic QC stage (stop_after: qc) — annotate, do not remove; then stop
    # =====================================================================
    if cfg.run.stop_after == "qc":
        from . import basic_qc as basic_qc_mod

        logger.info("=== Basic QC stage (run.stop_after = qc) ===")
        status.start("basic_qc")
        qc_result = basic_qc_mod.run_basic_qc(cfg, outdir, registry, config_path=config_path)
        status.mark(
            "basic_qc",
            STATUS_COMPLETED,
            f"{qc_result.n_cells_all:,} cells loaded, {qc_result.n_cells_pass:,} pass expression QC",
        )
        profiler.save_csv(tabledir / "compute_profile.csv")
        _write_table("module_status", status.table(), tabledir, table_paths)
        _, manifest_path = _finish_manifest(
            execution_mode="basic_qc",
            outputs={"Report": str(qc_result.report), "QC h5ad": str(qc_result.pass_h5ad)},
            counts={
                "cells_loaded": qc_result.n_cells_all,
                "cells_expression_qc_pass": qc_result.n_cells_pass,
                "genes": qc_result.n_genes,
            },
            runtime=time.time() - start,
        )
        result = PipelineResult(
            outdir=outdir,
            report=qc_result.report,
            h5ad=qc_result.pass_h5ad,
            guide_h5ad=None,
            unfiltered_h5ad=qc_result.allcells_h5ad,
            tables=dict(qc_result.tables),
            figures_dir=registry.figdir,
            n_cells=qc_result.n_cells_pass,
            n_genes=qc_result.n_genes,
            runtime_seconds=time.time() - start,
            execution_mode="basic_qc",
            run_manifest=manifest_path,
            module_status=status.table(),
        )
        result.basic_qc = qc_result
        logger.info(
            "Basic QC complete: %d cells (all) / %d cells (expression-QC pass); "
            "predicted doublets retained: %d / %d; guide multiplets retained: %d / %d",
            qc_result.n_cells_all,
            qc_result.n_cells_pass,
            qc_result.n_predicted_doublets_all,
            qc_result.n_predicted_doublets_pass,
            qc_result.n_guide_multiplets_all,
            qc_result.n_guide_multiplets_pass,
        )
        return result
    # =====================================================================
    # Stage 1: load
    # =====================================================================
    logger.info("=== Stage 1/14: loading input ===")
    status.start("load")
    data = io_mod.load_data(cfg)
    expr = data.expr
    guides = data.guides
    n_cells_input = expr.n_obs
    large_mode = cfg.use_large_mode(expr.n_obs)
    execution_mode = cfg.execution_mode(expr.n_obs)
    logger.info("Pipeline execution mode: %s (%d cells x %d genes)", execution_mode.upper(), expr.n_obs, expr.n_vars)
    _log_memory("input loading", cfg=cfg)
    status.mark(
        "load",
        STATUS_COMPLETED,
        f"{expr.n_obs:,} cells x {expr.n_vars:,} genes from {data.n_lanes} lane(s); guide source: {data.guide_source}",
    )
    status.start("qc")
    # =====================================================================
    # Stage 2: QC
    # =====================================================================
    logger.info("=== Stage 2/14: quality control ===")
    unfiltered_h5ad_path: Optional[Path] = None
    qc_before = None  # pre-filter QC metrics of every loaded cell (pair-guide accounting)
    pair_mode = cfg.guides.assignment_mode in ("dual_guide_pair", "pair")
    high_moi_mode = cfg.guides.assignment_mode == "high_moi"
    # All-cells checkpoint: every loaded cell with QC metrics, written BEFORE
    # any filtering so QC-failed cells are never lost. Previously this file
    # was only written in the assigned_only branch (after QC), which made
    # output.write_unfiltered_h5ad a silent no-op for default runs.
    if cfg.output.write_unfiltered_h5ad and not cfg.cluster.assigned_only:
        all_cells = expr.copy()
        qc_mod.compute_qc_metrics(all_cells, cfg)
        qc_before = all_cells.obs[
            [
                c
                for c in all_cells.obs.columns
                if c
                in ("lane_id", "total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb")
            ]
        ].copy()
        all_cells.uns["qc_stage"] = "all loaded cells before any QC filtering (basic QC metrics only)"
        unfiltered_h5ad_path = _write_unfiltered_object(all_cells, guides, cfg, outdir, io_mod)
        logger.info("Wrote all-cells checkpoint (%d cells, pre-QC) to %s", all_cells.n_obs, unfiltered_h5ad_path)
        del all_cells
        _collect("all-cells checkpoint", cfg=cfg, large_mode=large_mode)
    if pair_mode and qc_before is None:
        _tmp = expr.copy()
        qc_mod.compute_qc_metrics(_tmp, cfg)
        qc_before = _tmp.obs[
            [
                c
                for c in _tmp.obs.columns
                if c
                in ("lane_id", "total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb")
            ]
        ].copy()
        del _tmp
    expr = qc_mod.prefilter(expr, cfg)
    expr = qc_mod.compute_qc_metrics(expr, cfg)
    plots_mod.plot_qc(expr, registry, stage="before filtering")
    expr, qc_steps = qc_mod.filter_cells_and_genes(expr, cfg)
    plots_mod.plot_qc(expr, registry, stage="after filtering")
    qc_summary = qc_mod.qc_summary_table(expr)
    tables["qc_steps"] = qc_steps
    tables["qc_summary"] = qc_summary
    _write_table("qc_steps", qc_steps, tabledir, table_paths)
    _write_table("qc_summary", qc_summary, tabledir, table_paths)
    _collect("QC", cfg=cfg, large_mode=large_mode)
    status.mark("qc", STATUS_COMPLETED, f"{expr.n_obs:,} of {n_cells_input:,} cells retained; {expr.n_vars:,} genes")
    status.start("guide_assignment")
    # =====================================================================
    # Stage 3: guide assignment
    # =====================================================================
    logger.info("=== Stage 3/14: guide assignment ===")
    expr = guides_mod.assign_guides(expr, guides, cfg)
    guide_qc = qc_mod.guide_qc_summary(expr, cfg)
    if high_moi_mode:
        from . import high_moi as hm_mod

        guide_assignment = hm_mod.membership_assignment_summary(expr, cfg)
    else:
        guide_assignment = guides_mod.assignment_summary(expr, cfg)
    tables["guide_qc"] = guide_qc
    tables["guide_assignment"] = guide_assignment
    _write_table("guide_qc", guide_qc, tabledir, table_paths)
    _write_table("guide_assignment", guide_assignment, tabledir, table_paths)
    per_lane = guides_mod.per_lane_assignment(expr)
    if not per_lane.empty:
        tables["assignment_per_lane"] = per_lane
        _write_table("assignment_per_lane", per_lane, tabledir, table_paths)
    if guides is not None:
        if high_moi_mode:
            guide_representation = hm_mod.membership_guide_representation(guides, expr, cfg)
        else:
            guide_representation = guides_mod.guide_representation(guides, expr)
        tables["guide_representation"] = guide_representation
        _write_table("guide_representation", guide_representation, tabledir, table_paths)
    if high_moi_mode:
        for name, df in hm_mod.high_moi_tables(expr, cfg).items():
            tables[name] = df
            _write_table(name, df, tabledir, table_paths)
        plots_mod.plot_high_moi_calling(expr, registry, cfg)
    if pair_mode:
        from . import dual_guides as dual_mod
        from . import pair_guide_report as pair_mod

        pair_summary = dual_mod.pair_assignment_summary(expr)
        tables["pair_assignment_summary"] = pair_summary
        _write_table("pair_assignment_summary", pair_summary, tabledir, table_paths)
        pair_per_lane = dual_mod.pair_assignment_per_lane(expr)
        if pair_per_lane is not None:
            tables["pair_assignment_per_lane"] = pair_per_lane
            _write_table("pair_assignment_per_lane", pair_per_lane, tabledir, table_paths)
        for name, df in pair_mod.qc_before_after(expr, qc_before, cfg, registry).items():
            tables[name] = df
            _write_table(name, df, tabledir, table_paths)
        for name, df in pair_mod.pair_guide_qc(expr, guides, cfg, registry).items():
            tables[name] = df
            _write_table(name, df, tabledir, table_paths)
        for name, df in pair_mod.single_guide_diagnostic(expr, registry).items():
            tables[name] = df
            _write_table(name, df, tabledir, table_paths)
    warnings.extend(qc_mod.check_guide_qc(expr, cfg))
    plots_mod.plot_guide_qc(expr, guides, registry, cfg)
    _collect("guide assignment", cfg=cfg, large_mode=large_mode)
    status.mark(
        "guide_assignment",
        STATUS_COMPLETED,
        f"assignment_mode={cfg.guides.assignment_mode}; "
        + "; ".join(f"{k}={v:,}" for k, v in expr.obs[guides_mod.OBS_CLASS].value_counts().items()),
    )
    status.start("clustering")
    # =====================================================================
    # Stage 4: normalization / embedding / clustering
    # =====================================================================
    logger.info("=== Stage 4/14: normalization, embedding, clustering ===")
    expr = cluster_mod.normalize(expr, cfg)
    # Knockdown mask: marks cells in obs, removes none, so stage 5 still sees
    # every cell. Computed before the assigned_only split so both objects carry it.
    kd_table = None
    if cfg.knockdown_filter.enabled:
        expr, kd_table = kd_mod.compute_knockdown_mask(expr, cfg)
    singlets = _assigned_singlet_mask(expr) if cfg.cluster.assigned_only else None
    if singlets is not None and singlets.all():
        # All cells already satisfy the assignment condition.
        singlets = None
    if singlets is not None and not singlets.any():
        raise ValueError(
            "cluster.assigned_only left no guide-assigned singlets to cluster; loosen guide assignment thresholds."
        )
    if singlets is not None:
        n_all = expr.n_obs
        n_keep = int(singlets.sum())
        logger.info(
            "cluster.assigned_only: embedding all %d QC-passing cells, "
            "then re-embedding %d assigned singlets "
            "(%d ambiguous/unassigned excluded from analysis).",
            n_all,
            n_keep,
            n_all - n_keep,
        )
        expr = cluster_mod.embed_and_cluster(expr, cfg)
        clusters_all = cluster_mod.cluster_summary(expr)
        tables["clusters_all_cells"] = clusters_all
        _write_table("clusters_all_cells", clusters_all, tabledir, table_paths)
        plots_mod.plot_clustering(
            expr,
            registry,
            cfg,
            name_prefix="all_cells_",
            section=plots_mod.SECTION_GUIDES,
            label=" — all cells, before guide filtering",
        )
        if cfg.output.write_unfiltered_h5ad:
            unfiltered_h5ad_path = _write_unfiltered_object(expr, guides, cfg, outdir, io_mod)
        warnings.append(
            f"cluster.assigned_only: downstream analysis covers "
            f"{n_keep:,} guide-assigned singlets. "
            f"{n_all - n_keep:,} ambiguous/unassigned cells remain only in "
            "the all-cell embedding/output."
        )
        # This is intrinsically a real subset copy because downstream analysis
        # now requires a different set of cells. No additional copy is made.
        expr = expr[singlets].copy()
        expr = cluster_mod.reset_embedding(expr)
        del singlets
        _collect("assigned-only subset", cfg=cfg, large_mode=large_mode)
    expr = cluster_mod.embed_and_cluster(expr, cfg)
    clusters = cluster_mod.cluster_summary(expr)
    tables["clusters"] = clusters
    _write_table("clusters", clusters, tabledir, table_paths)
    plots_mod.plot_clustering(expr, registry, cfg)
    if pair_mode:
        from . import pair_guide_report as pair_mod

        for name, df in pair_mod.clustering_pair_figures(expr, cfg, registry).items():
            tables[name] = df
            _write_table(name, df, tabledir, table_paths)
    _collect("clustering", cfg=cfg, large_mode=large_mode)
    status.mark("clustering", STATUS_COMPLETED, f"{expr.obs[cluster_mod.CLUSTER_KEY].nunique()} Leiden clusters")
    status.start("perturbation")
    # =====================================================================
    # Stage 5: perturbation strength
    # =====================================================================
    logger.info("=== Stage 5/14: perturbation strength ===")
    results = pert_mod.test_all_targets(expr, cfg)
    perturbation_formatted = pert_mod.format_results_table(results, cfg)
    _write_table("perturbation", perturbation_formatted, tabledir, table_paths)
    _write_table("perturbation_full", results.table, tabledir, table_paths)
    _write_table("skipped", results.skipped, tabledir, table_paths)
    tables["perturbation"] = perturbation_formatted
    _table_for_report(
        tables,
        "perturbation_full",
        results.table,
        large_mode=large_mode,
        max_rows_large=cfg.scaling.report_preview_rows,
    )
    if results.skipped is not None and not results.skipped.empty:
        _table_for_report(
            tables, "skipped", results.skipped, large_mode=large_mode, max_rows_large=cfg.scaling.report_preview_rows
        )
    if kd_table is not None:
        expr, kd_table = kd_mod.attach_perturbation_strength(expr, kd_table, results)
        tables["knockdown_filter"] = kd_table
        _write_table("knockdown_filter", kd_table, tabledir, table_paths)
        targeting = expr.obs[guides_mod.OBS_CLASS].astype(str) == guides_mod.CLASS_TARGETING
        n_marked = int((targeting & ~expr.obs[kd_mod.OBS_KD_KEEP]).sum())
        warnings.append(
            f"Knockdown mask (knockdown_filter.mode: {cfg.knockdown_filter.mode}): "
            f"{n_marked:,} of {int(targeting.sum()):,} targeting cells are marked "
            f"obs['{kd_mod.OBS_KD_KEEP}'] == False. No cells were removed; every "
            f"analysis in this report uses all cells. See tables/knockdown_filter.csv."
        )
    plots_mod.plot_perturbation_overview(results, registry, cfg)
    plots_mod.plot_per_target(expr, results, registry, cfg)
    _collect("perturbation strength", cfg=cfg, large_mode=large_mode)
    status.mark(
        "perturbation",
        STATUS_COMPLETED,
        f"{len(results.table)} targets tested, {len(results.hits) if not results.table.empty else 0} effective",
    )
    if pair_mode:
        from . import pair_guide_report as pair_mod

        for name, df in pair_mod.pair_perturbation(expr, cfg, registry).items():
            tables[name] = df
            _write_table(name, df, tabledir, table_paths)
        _collect("pair perturbation", cfg=cfg, large_mode=large_mode)
    # =====================================================================
    # Stage 6: enrichment
    # =====================================================================
    enrichment = None
    if cfg.enrichment.enabled:
        status.start("enrichment")
        logger.info("=== Stage 6/14: perturbation enrichment across clusters ===")
        enrichment = enrich_mod.test_cluster_enrichment(expr, cfg)
        enrichment_fmt = enrich_mod.format_enrichment_table(enrichment)
        _write_table("enrichment", enrichment_fmt, tabledir, table_paths)
        _write_table("enrichment_full", enrichment.table, tabledir, table_paths)
        # composition is indexed by target; write directly without reset_index.
        _write_indexed_matrix("enrichment_composition", enrichment.composition, "target_gene", tabledir, table_paths)
        _write_table("enrichment_effect_magnitude", enrichment.effect_magnitude, tabledir, table_paths)
        if enrichment.membership_aware and not enrichment.pseudo_table.empty:
            tables["enrichment_pseudo_targets"] = enrichment.pseudo_table
            _write_table("enrichment_pseudo_targets", enrichment.pseudo_table, tabledir, table_paths)
            pseudo_summary = pd.DataFrame(
                [(k, v) for k, v in enrichment.pseudo_summary.items()], columns=["metric", "value"]
            )
            tables["enrichment_pseudo_summary"] = pseudo_summary
            _write_table("enrichment_pseudo_summary", pseudo_summary, tabledir, table_paths)
        tables["enrichment"] = enrichment_fmt
        _table_for_report(
            tables,
            "enrichment_full",
            enrichment.table,
            large_mode=large_mode,
            max_rows_large=cfg.scaling.report_preview_rows,
        )
        _table_for_report(
            tables,
            "enrichment_effect_magnitude",
            enrichment.effect_magnitude,
            large_mode=large_mode,
            max_rows_large=cfg.scaling.report_preview_rows,
        )
        plots_mod.plot_enrichment(expr, enrichment, registry, cfg)
        plots_mod.plot_enrichment_per_target(expr, enrichment, registry, cfg)
        _collect("cluster enrichment", cfg=cfg, large_mode=large_mode)
        status.mark(
            "enrichment",
            STATUS_COMPLETED if not enrichment.table.empty else STATUS_SKIPPED,
            f"{int(enrichment.table['significant'].sum()) if not enrichment.table.empty else 0} significant target x cluster pairs",
        )
    else:
        logger.info("Cluster enrichment disabled (enrichment.enabled: false)")
        status.mark("enrichment", STATUS_DISABLED, "enrichment.enabled: false", enabled=False)
    # =====================================================================
    # Stage 7: modules/programs
    # =====================================================================
    modules_result = None
    if cfg.modules.enabled:
        status.start("modules")
        logger.info("=== Stage 7/14: co-functional modules & gene programs ===")
        modules_result = modules_mod.compute_modules(expr, cfg)
        if modules_result is not None:
            # Do NOT call reset_index() on a potentially 10k x 2k matrix.
            _write_indexed_matrix("effect_matrix", modules_result.effect_matrix, "target_gene", tabledir, table_paths)
            _write_table("gene_programs", modules_result.gene_programs, tabledir, table_paths)
            _write_table("cofunctional_modules", modules_result.modules, tabledir, table_paths)
            _write_indexed_matrix(
                "module_program_strength", modules_result.module_program, "module", tabledir, table_paths
            )
            if not modules_result.program_activity.empty:
                _write_indexed_matrix(
                    "program_activity_by_cluster", modules_result.program_activity, "program", tabledir, table_paths
                )
            if not modules_result.hubs.empty:
                _write_table("tf_hubs", modules_result.hubs, tabledir, table_paths)
            if not modules_result.tf_edges.empty:
                _write_table("tf_edges", modules_result.tf_edges, tabledir, table_paths)
            if not modules_result.module_connectivity.empty:
                _write_indexed_matrix(
                    "module_connectivity", modules_result.module_connectivity, "module", tabledir, table_paths
                )
            if not modules_result.program_enrichment.empty:
                _write_table("program_enrichment", modules_result.program_enrichment, tabledir, table_paths)
            if not modules_result.program_summary.empty:
                _write_table("program_summary", modules_result.program_summary, tabledir, table_paths)
            # Keep only report-friendly summaries in RAM.
            tables["gene_programs"] = modules_result.gene_programs
            tables["cofunctional_modules"] = modules_result.modules
            if not modules_result.program_summary.empty:
                tables["program_summary"] = modules_result.program_summary
            if not modules_result.program_enrichment.empty:
                _table_for_report(
                    tables,
                    "program_enrichment",
                    modules_result.program_enrichment,
                    large_mode=large_mode,
                    max_rows_large=cfg.scaling.report_preview_rows,
                )
            _table_for_report(
                tables,
                "tf_hubs",
                modules_result.hubs,
                large_mode=large_mode,
                max_rows_large=cfg.scaling.report_preview_rows,
            )
            if modules_result.note:
                warnings.append("Modules: " + modules_result.note)
            if modules_result.program_enrichment_error:
                warnings.append(
                    "Modules: program pathway enrichment was NOT run ("
                    + modules_result.program_enrichment_error
                    + "); every program is 'unannotated', which is not a negative result. "
                    "Check network access to the MSigDB download or supply program_enrichment.custom_gmt_files."
                )
            plots_mod.plot_modules(expr, modules_result, registry, cfg)
            _collect("modules/programs", cfg=cfg, large_mode=large_mode)
        if modules_result is not None and not modules_result.effect_matrix.empty:
            detail = f"{modules_result.n_modules} modules, {modules_result.n_programs} programs"
            if modules_result.program_enrichment_error:
                detail += "; program enrichment NOT run: " + modules_result.program_enrichment_error
            status.mark("modules", STATUS_COMPLETED, detail)
        else:
            status.mark(
                "modules", STATUS_SKIPPED, getattr(modules_result, "note", "") or "too few perturbations or genes"
            )
    else:
        logger.info("Modules/programs disabled (modules.enabled: false)")
        status.mark("modules", STATUS_DISABLED, "modules.enabled: false", enabled=False)
    # =====================================================================
    # Stage 8: PS score
    # =====================================================================
    logger.info("=== Stage 8/14: per-cell perturbation scores ===")
    status.start("ps_score")
    ps_results = ps_mod.compute_ps_scores(expr, cfg)
    if ps_results is not None and not ps_results.summary.empty:
        expr = ps_mod.attach_scores(expr, ps_results)
        if ps_results.lda_umap is not None:
            expr.obsm["X_lda_umap"] = ps_results.lda_umap
            if ps_results.lda_label is not None:
                expr.obs["lda_label"] = pd.Categorical(ps_results.lda_label.astype(str))
        _write_table("ps_score", ps_results.summary, tabledir, table_paths)
        tables["ps_score"] = ps_results.summary
        if not ps_results.skipped.empty:
            _write_table("ps_skipped", ps_results.skipped, tabledir, table_paths)
            _table_for_report(
                tables,
                "ps_skipped",
                ps_results.skipped,
                large_mode=large_mode,
                max_rows_large=cfg.scaling.report_preview_rows,
            )
        comparison = ps_mod.compare_with_perturbation_strength(ps_results, results.table, results.primary_control)
        if not comparison.empty:
            _write_table("ps_vs_perturbation", comparison, tabledir, table_paths)
            tables["ps_vs_perturbation"] = comparison
        plots_mod.plot_ps_scores(expr, ps_results, results, registry, cfg)
        plots_mod.plot_ps_lda(expr, ps_results, registry, cfg)
        _collect("PS score", cfg=cfg, large_mode=large_mode)
    elif ps_results is not None and ps_results.note:
        warnings.append(ps_results.note)
    if not cfg.ps_score.enabled:
        status.mark("ps_score", STATUS_DISABLED, "ps_score.enabled: false", enabled=False)
    elif ps_results is not None and not ps_results.summary.empty:
        status.mark("ps_score", STATUS_COMPLETED, f"{len(ps_results.summary)} targets scored")
    else:
        status.mark("ps_score", STATUS_SKIPPED, getattr(ps_results, "note", "") or "no target scored")
    # =====================================================================
    # Stage 9: lochNESS
    # =====================================================================
    lochness = None
    if cfg.lochness.enabled:
        status.start("lochness")
        logger.info("=== Stage 9/14: lochNESS neighbourhood enrichment ===")
        lochness = loch_mod.compute_lochness(expr, cfg)
        if lochness is not None and not lochness.summary.empty:
            expr = loch_mod.attach_scores(expr, lochness)
            _write_table("lochness", lochness.summary, tabledir, table_paths)
            tables["lochness"] = lochness.summary
            if not lochness.by_cluster.empty:
                _write_indexed_matrix("lochness_by_cluster", lochness.by_cluster, "target_gene", tabledir, table_paths)
            if not lochness.skipped.empty:
                _write_table("lochness_skipped", lochness.skipped, tabledir, table_paths)
                _table_for_report(
                    tables,
                    "lochness_skipped",
                    lochness.skipped,
                    large_mode=large_mode,
                    max_rows_large=cfg.scaling.report_preview_rows,
                )
            plots_mod.plot_lochness(expr, lochness, registry, cfg)
            _collect("lochNESS", cfg=cfg, large_mode=large_mode)
        elif lochness is not None and lochness.note:
            warnings.append(lochness.note)
        if lochness is not None and not lochness.summary.empty:
            status.mark("lochness", STATUS_COMPLETED, f"{len(lochness.summary)} targets scored")
        else:
            status.mark("lochness", STATUS_SKIPPED, getattr(lochness, "note", "") or "no target scored")
    else:
        logger.info("lochNESS disabled (lochness.enabled: false)")
        status.mark("lochness", STATUS_DISABLED, "lochness.enabled: false", enabled=False)
    # =====================================================================
    # Stage 10: perturbation distance vs control
    # =====================================================================
    distance_results = None
    if cfg.distance.enabled:
        status.start("distance")
        logger.info("=== Stage 10/14: perturbation distance vs control ===")
        distance_results = dist_mod.compute_perturbation_distance(expr, cfg)
        if distance_results is not None and not distance_results.table.empty:
            _write_table("perturbation_distance", distance_results.table, tabledir, table_paths)
            tables["perturbation_distance"] = distance_results.table
            if not distance_results.skipped.empty:
                _write_table("distance_skipped", distance_results.skipped, tabledir, table_paths)
                _table_for_report(
                    tables,
                    "distance_skipped",
                    distance_results.skipped,
                    large_mode=large_mode,
                    max_rows_large=cfg.scaling.report_preview_rows,
                )
            _collect("perturbation distance", cfg=cfg, large_mode=large_mode)
        elif distance_results is not None and distance_results.note:
            warnings.append(distance_results.note)
        if distance_results is not None and not distance_results.table.empty:
            status.mark("distance", STATUS_COMPLETED, f"{len(distance_results.table)} targets tested")
        else:
            status.mark("distance", STATUS_SKIPPED, getattr(distance_results, "note", "") or "no target tested")
    else:
        logger.info("Perturbation distance disabled (distance.enabled: false)")
        status.mark("distance", STATUS_DISABLED, "distance.enabled: false", enabled=False)
    # =====================================================================
    # Stage 11: perturbation distance space
    # =====================================================================
    dist_space_results = None
    if cfg.distance_space.enabled:
        status.start("distance_space")
        logger.info("=== Stage 11/14: perturbation distance space ===")
        dist_space_results = dist_mod.compute_distance_space(expr, cfg)
        if dist_space_results is not None and not dist_space_results.distance_matrix.empty:
            # Distance matrix saved outside H5AD as TSV
            mat_path = tabledir / "perturbation_distance_matrix.tsv"
            dist_space_results.distance_matrix.to_csv(mat_path, sep="\t")
            table_paths["perturbation_distance_matrix"] = mat_path
            _write_table("perturbation_space_coordinates", dist_space_results.coordinates, tabledir, table_paths)
            tables["perturbation_space_coordinates"] = dist_space_results.coordinates
            _write_table("perturbation_neighbors", dist_space_results.neighbors, tabledir, table_paths)
            tables["perturbation_neighbors"] = dist_space_results.neighbors
            _write_table("phenotype_modules", dist_space_results.phenotype_modules, tabledir, table_paths)
            tables["phenotype_modules"] = dist_space_results.phenotype_modules
            if not dist_space_results.skipped.empty:
                _write_table("distance_space_skipped", dist_space_results.skipped, tabledir, table_paths)
                _table_for_report(
                    tables,
                    "distance_space_skipped",
                    dist_space_results.skipped,
                    large_mode=large_mode,
                    max_rows_large=cfg.scaling.report_preview_rows,
                )
            _collect("distance space", cfg=cfg, large_mode=large_mode)
        elif dist_space_results is not None and dist_space_results.note:
            warnings.append(dist_space_results.note)
        if dist_space_results is not None and not dist_space_results.distance_matrix.empty:
            status.mark(
                "distance_space",
                STATUS_COMPLETED,
                f"{len(dist_space_results.distance_matrix)} perturbations in the distance space",
            )
        else:
            status.mark(
                "distance_space", STATUS_SKIPPED, getattr(dist_space_results, "note", "") or "too few perturbations"
            )
    else:
        logger.info("Perturbation distance space disabled (distance_space.enabled: false)")
        status.mark("distance_space", STATUS_DISABLED, "distance_space.enabled: false", enabled=False)
    # =====================================================================
    # Stage 12: master perturbation meta table & plots
    # =====================================================================
    meta_table = None
    if cfg.meta_analysis.enabled:
        status.start("meta_analysis")
        logger.info("=== Stage 12/14: master perturbation meta table ===")
        meta_table = meta_mod.build_perturbation_meta(
            cfg=cfg,
            perturbation_table=results.table if results is not None else None,
            ps_summary=ps_results.summary if ps_results is not None else None,
            lochness_summary=lochness.summary if lochness is not None else None,
            distance_table=distance_results.table if distance_results is not None else None,
            cofunctional_modules=modules_result.modules if modules_result is not None else None,
            phenotype_modules=dist_space_results.phenotype_modules if dist_space_results is not None else None,
            primary_control=results.primary_control if results is not None else "ntc",
        )
        if not meta_table.empty:
            _write_table("perturbation_meta", meta_table, tabledir, table_paths)
            tables["perturbation_meta"] = meta_table
        status.mark(
            "meta_analysis",
            STATUS_COMPLETED if meta_table is not None and not meta_table.empty else STATUS_SKIPPED,
            f"{0 if meta_table is None else len(meta_table)} targets in tables/perturbation_meta.csv",
        )
    else:
        logger.info("Master perturbation meta table disabled (meta_analysis.enabled: false)")
        status.mark("meta_analysis", STATUS_DISABLED, "meta_analysis.enabled: false", enabled=False)
    # Generate distance and distance space plots
    plots_mod.plot_distance_figures(distance_results, meta_table, registry, cfg)
    plots_mod.plot_distance_space_figures(dist_space_results, meta_table, registry, cfg)
    _collect("distance plots", cfg=cfg, large_mode=large_mode)
    # =====================================================================
    # Stage 13: write outputs
    # =====================================================================
    logger.info("=== Stage 13/14: writing outputs ===")
    status.start("outputs")
    manifest = registry.manifest()
    manifest_path = tabledir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    table_paths["figure_manifest"] = manifest_path
    tables["manifest"] = manifest
    # Merge guides into final object
    expr = io_mod.merge_guides_into_expr(expr, guides, cfg)
    # Final expression H5AD
    h5ad_path = io_mod.write_h5ad(expr, outdir / cfg.output.h5ad_name)
    h5ad_path = io_mod.relocate_if_large(h5ad_path, cfg)
    _collect("processed h5ad write", cfg=cfg, large_mode=large_mode)
    # Guide H5AD
    guide_h5ad_path: Optional[Path] = None
    aligned_guides = _aligned_guides(guides, expr) if guides is not None else None
    if aligned_guides is not None and cfg.output.write_guide_h5ad:
        gname = Path(cfg.output.h5ad_name).stem + "_guides.h5ad"
        guide_h5ad_path = io_mod.write_h5ad(aligned_guides, outdir / gname)
        guide_h5ad_path = io_mod.relocate_if_large(guide_h5ad_path, cfg)
    # Guide barcode table
    guide_table_path: Optional[Path] = None
    if aligned_guides is not None and cfg.output.write_guide_table:
        tname = cfg.output.guide_table_name or (f"{cfg.run.name}_guide_barcodes.txt")
        guide_table_path = io_mod.write_guide_table(aligned_guides, expr, cfg, outdir / tname)
    _collect("output matrices", cfg=cfg, large_mode=large_mode)
    status.mark("outputs", STATUS_COMPLETED, str(h5ad_path))
    status.start("report")
    # =====================================================================
    # Stage 14: report
    # =====================================================================
    logger.info("=== Stage 14/14: building report ===")
    n_hits = len(results.hits) if not results.table.empty else 0
    outputs = {
        "Processed h5ad": str(h5ad_path),
        "Report": str(outdir / cfg.output.report_name),
        "Figures": str(registry.figdir),
        "Per-target figures": str(registry.figdir / plots_mod.SECTION_PER_GENE),
        "Tables": str(tabledir),
        "Log": str(outdir / "logs" / "run.log"),
    }
    if unfiltered_h5ad_path:
        outputs["All-cells h5ad (before guide filtering)"] = str(unfiltered_h5ad_path)
    if guide_h5ad_path:
        outputs["Guide count h5ad"] = str(guide_h5ad_path)
    if guide_table_path:
        outputs["Guide barcode table"] = str(guide_table_path)
    archive_name = cfg.output.archive_name or (f"{cfg.run.name}_results.tar.gz")
    if cfg.output.archive:
        outputs["Results archive"] = str(outdir / archive_name)
    outputs["Run manifest"] = str(outdir / "logs" / "run_manifest.json")
    # The report is the last stage; it is recorded as completed here so the
    # status table inside the report is final. The manifest is rewritten
    # once the report file exists.
    status.mark("report", STATUS_COMPLETED, str(outdir / cfg.output.report_name))
    module_status = status.table()
    _write_table("module_status", module_status, tabledir, table_paths)
    tables["module_status"] = module_status
    run_counts = {
        "cells_input": int(n_cells_input),
        "cells_analysed": int(expr.n_obs),
        "genes": int(expr.n_vars),
        "lanes": int(data.n_lanes),
        "targets_tested": int(len(results.table)),
        "effective_knockdowns": int(n_hits),
        "clusters": int(expr.obs[cluster_mod.CLUSTER_KEY].nunique()),
    }
    manifest_rec, manifest_path = _finish_manifest(
        lanes=data.lanes, execution_mode=execution_mode, outputs=outputs, counts=run_counts, runtime=time.time() - start
    )
    summary_cards = [
        ("Cells analysed", f"{expr.n_obs:,}"),
        ("Genes", f"{expr.n_vars:,}"),
        ("Lanes", f"{data.n_lanes}"),
        ("Target genes", (f"{guides_mod.target_genes(expr, cfg).size}")),
        ("Clusters", (f"{expr.obs[cluster_mod.CLUSTER_KEY].nunique()}")),
        ("Effective knockdowns", f"{n_hits}"),
        ("Execution mode", execution_mode.upper()),
    ]
    report_inputs = ReportInputs(
        cfg=cfg,
        registry=registry,
        perturbation=results,
        enrichment=enrichment,
        modules=modules_result,
        ps=ps_results,
        lochness=lochness,
        distance=distance_results,
        distance_space=dist_space_results,
        meta_table=meta_table,
        tables=tables,
        warnings=warnings,
        summary_cards=summary_cards,
        input_mode=cfg.resolved_mode(),
        input_source="; ".join(f"{key}: {value}" for key, value in data.lanes.items()),
        lanes=(f"{data.n_lanes} ({', '.join(data.lanes)})"),
        metadata_source=(cfg.metadata.file or ""),
        guide_source_text=(
            "guide count matrix"
            if data.guide_source == "matrix"
            else (f"pre-computed labels in obs[{cfg.input.guide_obs_column!r}]")
        ),
        outputs=outputs,
        module_status=module_status,
        provenance_rows=manifest_summary_rows(manifest_rec),
    )
    report_path = build_report(report_inputs, outdir / cfg.output.report_name)
    if getattr(cfg.output, "report_markdown_name", None):
        from .report_markdown import write_markdown_report

        write_markdown_report(report_inputs, outdir / cfg.output.report_markdown_name)
    # Save compute performance profile table before archiving
    profile_df = profiler.to_dataframe()
    profile_path = tabledir / "compute_profile.csv"
    profiler.save_csv(profile_path)
    table_paths["compute_profile"] = profile_path
    tables["compute_profile"] = profile_df
    # Archive only after report generation and all tables are written.
    archive_path = io_mod.archive_results(outdir, cfg)
    runtime = time.time() - start
    logger.info("Done in %.1f s — %d/%d input cells retained", runtime, expr.n_obs, n_cells_input)
    if archive_path:
        outputs["Results archive"] = str(archive_path)
    manifest_rec, manifest_path = _finish_manifest(
        lanes=data.lanes, execution_mode=execution_mode, outputs=outputs, counts=run_counts, runtime=runtime
    )
    result = PipelineResult(
        outdir=outdir,
        report=report_path,
        h5ad=h5ad_path,
        guide_h5ad=guide_h5ad_path,
        unfiltered_h5ad=unfiltered_h5ad_path,
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
        distance_table=distance_results.table if distance_results is not None else None,
        distance_space_results=dist_space_results,
        meta_table=meta_table,
        compute_profile=profile_df if not profile_df.empty else None,
        execution_mode=execution_mode,
        run_manifest=manifest_path,
        module_status=module_status,
    )
    logger.info(result.summary())
    return result


# CLI


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        prog="perturbseq-pipeline", description=("Perturb-seq QC, clustering and perturbation analysis pipeline.")
    )
    parser.add_argument("--version", action="version", version=(f"%(prog)s {__version__}"))
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pipeline from a config file")
    run.add_argument("-c", "--config", required=True, help="path to run config YAML")
    run.add_argument("-o", "--outdir", default=None, help="override run.outdir")
    run.add_argument("-n", "--name", default=None, help="override run.name")
    run.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    run.add_argument(
        "--lane",
        default=None,
        help="restrict input.mtx_dirs / input.guide_mtx_dirs to one lane key and write to <outdir>/samples/<lane> (unless -o is given)",
    )
    run.add_argument(
        "--combined-subdir",
        default=None,
        help="write a multi-lane run to <outdir>/<subdir> (e.g. 'combined') unless -o is given",
    )
    init = sub.add_parser("init-config", help="write a default config")
    init.add_argument("path", help="where to write the config YAML")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    if args.command == "init-config":
        cfg = Config()
        cfg.dump_yaml(Path(args.path))
        print(f"Wrote default configuration to {args.path}")
        print("Edit input.mtx_dirs (10x mode) or input.h5ad (h5ad mode), then run:")
        print(f"  perturbseq-pipeline run --config {args.path}")
        return 0
    cfg = Config.from_yaml(args.config)
    if getattr(args, "lane", None):
        lane = args.lane
        mtx = cfg.input.resolved_mtx_dirs()
        if not mtx:
            raise SystemExit("--lane needs input.mtx_dirs (10x mode); it does not apply to h5ad input")
        if lane not in mtx:
            raise SystemExit(f"--lane {lane!r} is not a lane of input.mtx_dirs: {sorted(mtx)}")
        cfg.input.mtx_dirs = {lane: mtx[lane]}
        # Separate guide matrices are optional (the combined CellRanger layout has none).
        guide_dirs = cfg.input.guide_mtx_dirs
        if guide_dirs:
            if lane not in guide_dirs:
                raise SystemExit(f"--lane {lane!r} is not a key of input.guide_mtx_dirs: {sorted(guide_dirs)}")
            cfg.input.guide_mtx_dirs = {lane: guide_dirs[lane]}
        if not args.outdir:
            cfg.run.outdir = str(Path(cfg.run.outdir) / "samples" / lane)
        cfg.run.name = f"{cfg.run.name}_{lane}"
    elif getattr(args, "combined_subdir", None) and not args.outdir:
        cfg.run.outdir = str(Path(cfg.run.outdir) / args.combined_subdir)
    if args.outdir:
        cfg.run.outdir = args.outdir
    if args.name:
        cfg.run.name = args.name
    try:
        result = run_pipeline(cfg, verbose=args.verbose, config_path=args.config)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    print("\n" + result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
