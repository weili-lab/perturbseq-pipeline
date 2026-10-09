"""All figures produced by the pipeline.

Every figure goes through :class:`FigureRegistry`, which writes the file and
records its title, caption and section. The report builder then renders whatever
is registered.

Library code never calls ``plt.show()``; the Agg backend is forced on import.

Large-dataset plotting
----------------------
The statistical pipeline may contain millions of cells and thousands of
perturbations, but plotting every cell and creating one image for every target
does not add analytical information.

For ordinary datasets the original plotting behaviour is preserved.

For large datasets, plotting automatically becomes bounded:

* scatter/UMAP panels use a reproducible subset of cells;
* QC distributions use a representative cell subset;
* overview target rankings show only the strongest targets;
* large heatmaps are limited to the most informative targets;
* per-target figures are generated only for a bounded top set;
* PS large mode uses ``ps_score`` / ``ps_quadrant`` rather than thousands of
  target-specific ``obs`` columns;
* lochNESS large mode uses ``lochness_self`` and target summaries rather than
  the intentionally omitted full cell × perturbation score matrix.

These restrictions affect visualization only. All analytical stages operate on
the full cell and perturbation populations.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gseapy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure as MplFigure

from .cluster import CLUSTER_KEY, LOGNORM_LAYER
from .config import Config
from .guides import (
    CLASS_AMBIGUOUS,
    CLASS_NTC,
    CLASS_TARGETING,
    CLASS_UNASSIGNED,
    OBS_CLASS,
    OBS_NDETECTED,
    OBS_SECOND,
    OBS_TARGET,
    OBS_TOP,
    OBS_TOTAL,
)
from .io import LANE_KEY
from .perturbation import CONTROL_LABELS, CONTROL_NTC, CONTROL_OTHER, PerturbationResults

logger = logging.getLogger(__name__)

sns.set_theme(style="ticks", context="notebook")


# Report sections

SECTION_QC = "qc"
SECTION_GUIDES = "guides"
SECTION_CLUSTERING = "clustering"
SECTION_PERTURBATION = "perturbation"
SECTION_PER_GENE = "perturbation/per_gene"
SECTION_ENRICHMENT = "enrichment"
SECTION_ENRICH_PER_TARGET = "enrichment/per_target"
SECTION_PS = "ps_score"
SECTION_PS_PER_TARGET = "ps_score/per_target"
SECTION_PS_LDA = "ps_score/lda"
SECTION_LOCHNESS = "lochness"
SECTION_LOCHNESS_PER_TARGET = "lochness/per_target"
SECTION_MODULES = "modules"
SECTION_DISTANCE = "distance"
SECTION_DISTANCE_SPACE = "distance_space"


_CLASS_COLORS = {
    CLASS_TARGETING: "#2b6cb0",
    CLASS_NTC: "#38a169",
    CLASS_AMBIGUOUS: "#dd6b20",
    CLASS_UNASSIGNED: "#a0aec0",
}


# Large-dataset plotting safeguards

# Replogle (~310k cells) keeps standard plotting.
# KOLF (~2.66M cells) automatically enters bounded plotting.
LARGE_DATASET_N_CELLS = 1_000_000

# A huge perturbation collection independently activates bounded plotting.
LARGE_DATASET_N_TARGETS = 5_000

# Maximum number of cells rendered in ordinary UMAP/scatter panels.
LARGE_PLOT_MAX_CELLS = 150_000

# Background cells in per-target panels.
LARGE_PLOT_BACKGROUND_CELLS = 75_000

# Number of control cells drawn in expression distribution panels.
LARGE_PLOT_MAX_CONTROL_CELLS = 20_000

# Maximum number of target-specific figures.
LARGE_PLOT_MAX_PER_TARGET = 50

# Maximum target count in wide overview bar charts.
LARGE_PLOT_MAX_OVERVIEW_TARGETS = 200

# Maximum number of perturbation rows rendered in heatmaps.
LARGE_PLOT_MAX_HEATMAP_TARGETS = 500


# Registry


def _slugify(name: str) -> str:
    """File-system-safe figure name."""
    import re

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return slug or "figure"


@dataclass
class FigureRecord:
    """A figure on disk plus report metadata."""

    path: Path
    name: str
    section: str
    title: str
    caption: str = ""
    #: False for supplementary / non-inline figures.
    in_report: bool = True

    def data_uri(self) -> str:
        """Return the figure as a base64 data URI."""
        mime = "image/png" if self.path.suffix == ".png" else "image/svg+xml"
        payload = base64.b64encode(self.path.read_bytes()).decode()
        return f"data:{mime};base64,{payload}"


@dataclass
class FigureRegistry:
    """Save figures and maintain a report index."""

    outdir: Path
    cfg: Config
    records: List[FigureRecord] = field(default_factory=list)

    @property
    def figdir(self) -> Path:
        return Path(self.outdir) / "figures"

    def save(
        self, fig: MplFigure, name: str, section: str, title: str, caption: str = "", in_report: bool = True
    ) -> FigureRecord:
        """Write and register a figure."""
        ext = self.cfg.report.figure_format
        name = _slugify(name)
        path = self.figdir / section / f"{name}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.cfg.report.figure_dpi, bbox_inches="tight")
        if self.cfg.output.save_figures_pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        rec = FigureRecord(path=path, name=name, section=section, title=title, caption=caption, in_report=in_report)
        self.records.append(rec)
        return rec

    def by_section(self, section: str, only_in_report: bool = True) -> List[FigureRecord]:
        return [
            record
            for record in self.records
            if (record.section == section and (record.in_report or not only_in_report))
        ]

    def extras(self, section: str) -> List[FigureRecord]:
        """Figures written but not embedded inline."""
        return [record for record in self.records if (record.section == section and not record.in_report)]

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "section": record.section,
                    "name": record.name,
                    "title": record.title,
                    "in_report": record.in_report,
                    "caption": record.caption,
                    "path": str(record.path),
                }
                for record in self.records
            ]
        )


# General helpers


def _is_large_plot_dataset(expr, n_targets: Optional[int] = None, cfg: Optional[Config] = None) -> bool:
    """Return True when bounded plotting should be used."""
    if cfg is not None:
        return cfg.use_large_mode(expr.n_obs, n_perturbations=n_targets)
    return expr.n_obs >= LARGE_DATASET_N_CELLS or (n_targets is not None and n_targets >= LARGE_DATASET_N_TARGETS)


def _plot_indices(n_cells: int, max_cells: int, seed: int, always_include: Optional[np.ndarray] = None) -> np.ndarray:
    """Reproducibly select cells for visualization.

    ``always_include`` contains integer cell indices that must be retained.
    """
    if n_cells <= max_cells and always_include is None:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if always_include is None:
        n_take = min(max_cells, n_cells)
        return np.sort(rng.choice(n_cells, size=n_take, replace=False)).astype(np.int64)
    always_include = np.unique(np.asarray(always_include, dtype=np.int64))
    remaining_budget = max(max_cells - len(always_include), 0)
    if remaining_budget == 0:
        return always_include
    candidate_mask = np.ones(n_cells, dtype=bool)
    candidate_mask[always_include] = False
    available = np.flatnonzero(candidate_mask)
    if len(available) <= remaining_budget:
        sampled = available
    else:
        sampled = rng.choice(available, size=remaining_budget, replace=False)
    return np.sort(np.concatenate([always_include, sampled])).astype(np.int64)


def _sample_pool(pool: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """Sample from an existing integer-index pool."""
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) <= max_n:
        return pool
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=max_n, replace=False))


def _obs_values(adata, key: str) -> np.ndarray:
    return adata.obs[key].to_numpy()


def _gene_values(adata, gene: str, cell_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Return log-normalized expression of one gene.

    When ``cell_indices`` is supplied only those rows are read, preventing
    unnecessary million-cell dense vectors in plotting code.
    """
    from scipy import sparse

    layer = adata.layers[LOGNORM_LAYER] if LOGNORM_LAYER in adata.layers else adata.X
    gene_idx = adata.var_names.get_loc(gene)
    if cell_indices is None:
        col = layer[:, gene_idx]
    else:
        col = layer[np.asarray(cell_indices, dtype=np.int64), gene_idx]
    if sparse.issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel()


def _scatter_umap(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    categorical: bool,
    title: str,
    size: float = 3.0,
    cmap: str = "viridis",
    legend: bool = True,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> None:
    """UMAP scatter with optional reproducible point downsampling."""
    coords = np.asarray(coords)
    values = np.asarray(values)
    if max_points is not None and len(coords) > max_points:
        idx = _plot_indices(len(coords), max_points, seed)
        coords = coords[idx]
        values = values[idx]
    if categorical:
        values_str = values.astype(str)
        cats = pd.Index(pd.unique(values_str)).sort_values()
        palette = sns.color_palette("tab20", max(len(cats), 3))
        for i, cat in enumerate(cats):
            mask = values_str == cat
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=size,
                color=palette[i % len(palette)],
                label=str(cat),
                linewidths=0,
                rasterized=True,
            )
        if legend and len(cats) <= 25:
            ax.legend(markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, cmap=cmap, linewidths=0, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.75)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


# QC figures


def plot_qc(adata, reg: FigureRegistry, stage: str = "prefilter") -> None:
    """Violin and scatter QC panels.

    Million-cell runs use a representative subset for distributions, while
    exact lane counts still use all cells.
    """
    obs = adata.obs
    large_plot = _is_large_plot_dataset(adata)
    if large_plot:
        idx = _plot_indices(adata.n_obs, LARGE_PLOT_MAX_CELLS, reg.cfg.run.seed)
        plot_obs = obs.iloc[idx]
        logger.info("Large-data QC plotting: using %d/%d cells", len(plot_obs), adata.n_obs)
    else:
        plot_obs = obs
    metrics = [
        ("n_genes_by_counts", "Genes per cell"),
        ("total_counts", "UMIs per cell"),
        ("pct_counts_mt", "% mitochondrial"),
        ("pct_counts_ribo", "% ribosomal"),
    ]
    metrics = [(metric, label) for metric, label in metrics if metric in plot_obs.columns]
    multi_lane = LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.6))
        axes = np.atleast_1d(axes)
        for ax, (metric, label) in zip(axes, metrics):
            if multi_lane:
                sns.violinplot(x=plot_obs[LANE_KEY].astype(str), y=plot_obs[metric], ax=ax, inner="box", cut=0)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.set_xlabel("")
            else:
                sns.violinplot(y=plot_obs[metric], ax=ax, inner="box", cut=0)
            ax.set_ylabel(label, fontsize=9)
            if metric == "total_counts":
                ax.set_yscale("log")
        fig.suptitle(f"Cell QC metrics ({stage}, n = {adata.n_obs:,} cells)", fontsize=11)
        fig.tight_layout()
        reg.save(
            fig,
            f"qc_violin_{stage}",
            SECTION_QC,
            f"Cell QC distributions ({stage})",
            (
                "Distribution of per-cell QC metrics"
                + (" for each lane." if multi_lane else ".")
                + (f" Plot based on a reproducible subset of {len(plot_obs):,} cells." if large_plot else "")
            ),
        )
    if {"total_counts", "n_genes_by_counts"} <= set(plot_obs.columns):
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        color = plot_obs["pct_counts_mt"] if "pct_counts_mt" in plot_obs.columns else None
        scatter = ax.scatter(
            plot_obs["total_counts"],
            plot_obs["n_genes_by_counts"],
            c=color,
            s=3,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        if color is not None:
            plt.colorbar(scatter, ax=ax, label="% mitochondrial")
        ax.set_xscale("log")
        ax.set_xlabel("Total UMIs per cell")
        ax.set_ylabel("Genes per cell")
        ax.set_title("Library size vs complexity", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            f"qc_scatter_{stage}",
            SECTION_QC,
            f"UMIs vs genes ({stage})",
            (
                "Each point is a cell."
                + (f" A reproducible subset of {len(plot_obs):,} cells is shown." if large_plot else "")
            ),
        )
    if multi_lane:
        fig, ax = plt.subplots(figsize=(max(4, 0.7 * obs[LANE_KEY].nunique()), 3.6))
        counts = obs[LANE_KEY].astype(str).value_counts().sort_index()
        ax.bar(counts.index, counts.to_numpy())
        ax.set_ylabel("Cells")
        ax.set_xlabel("Lane")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_title("Cells per lane", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            f"qc_cells_per_lane_{stage}",
            SECTION_QC,
            f"Cells per lane ({stage})",
            "Exact number of retained cells per lane.",
        )


# Guide QC figures


def plot_guide_qc(expr, guides, reg: FigureRegistry, cfg: Config) -> None:
    """Guide-specific QC."""
    obs = expr.obs
    large_plot = _is_large_plot_dataset(expr)
    if large_plot:
        idx = _plot_indices(expr.n_obs, LARGE_PLOT_MAX_CELLS, cfg.run.seed)
        plot_obs = obs.iloc[idx]
    else:
        plot_obs = obs
    if OBS_TOTAL in plot_obs.columns:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        vals = plot_obs[OBS_TOTAL].to_numpy()
        ax.hist(np.log10(vals + 1), bins=60)
        ax.set_xlabel("log10(total guide UMIs per cell + 1)")
        ax.set_ylabel("Cells")
        ax.set_title("Guide UMI depth per cell", fontsize=11)
        ax.axvline(np.log10(cfg.guides.min_umi + 1), ls="--", lw=1)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "guide_umi_depth",
            SECTION_GUIDES,
            "Guide UMI depth",
            (
                f"Guide UMI distribution. "
                f"guides.min_umi = {cfg.guides.min_umi}."
                + (f" A reproducible subset of {len(plot_obs):,} cells is shown." if large_plot else "")
            ),
        )
    if OBS_NDETECTED in plot_obs.columns:
        detected = plot_obs[OBS_NDETECTED].to_numpy()
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        top = int(min(detected.max(), 15))
        ax.hist(np.clip(detected, 0, top), bins=np.arange(-0.5, top + 1.5, 1))
        ax.set_xlabel((f"Guides detected per cell (> {cfg.guides.detection_threshold} UMI)"))
        ax.set_ylabel("Cells")
        ax.set_title((f"Guide multiplicity (sample mean = {detected.mean():.2f})"), fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "guide_multiplicity",
            SECTION_GUIDES,
            "Guides per cell (MOI)",
            "Distribution of detected guide multiplicity.",
        )
    if {OBS_TOP, OBS_SECOND} <= set(plot_obs.columns):
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        x = plot_obs[OBS_TOP].to_numpy() + 1
        y = plot_obs[OBS_SECOND].to_numpy() + 1
        klass = plot_obs[OBS_CLASS].astype(str).to_numpy() if OBS_CLASS in plot_obs.columns else None
        if klass is not None:
            for cl, color in _CLASS_COLORS.items():
                mask = klass == cl
                if mask.sum():
                    ax.scatter(x[mask], y[mask], s=4, alpha=0.5, color=color, label=cl, linewidths=0, rasterized=True)
            ax.legend(markerscale=3, fontsize=8, frameon=False)
        else:
            ax.scatter(x, y, s=4, alpha=0.5, linewidths=0, rasterized=True)
        lim = np.array([1, max(x.max(), y.max())])
        ax.plot(lim, lim / cfg.guides.dominance_ratio, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Highest guide count + 1")
        ax.set_ylabel("Second highest guide count + 1")
        ax.set_title("Guide dominance per cell", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "guide_top_vs_second",
            SECTION_GUIDES,
            "Top vs second guide count",
            "Cells far below the diagonal have one clearly dominant guide.",
        )
    if OBS_CLASS in obs.columns:
        counts = obs[OBS_CLASS].value_counts()
        labels = [cl for cl in _CLASS_COLORS if cl in counts.index]
        values = [counts[cl] for cl in labels]
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        ax.bar(labels, values, color=[_CLASS_COLORS[cl] for cl in labels])
        for i, value in enumerate(values):
            ax.text(i, value, (f"{100 * value / expr.n_obs:.1f}%"), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Cells")
        ax.set_title("Guide assignment outcome", fontsize=11)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "guide_assignment_classes",
            SECTION_GUIDES,
            "Guide assignment outcome",
            "Exact guide assignment counts across all cells.",
        )
        if LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1:
            table = obs.groupby([LANE_KEY, OBS_CLASS], observed=True).size().unstack(fill_value=0)
            frac = table.div(table.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(max(4.5, 0.8 * len(frac)), 3.8))
            bottom = np.zeros(len(frac))
            for cl in [cl for cl in _CLASS_COLORS if cl in frac.columns]:
                ax.bar(frac.index.astype(str), frac[cl], bottom=bottom, color=_CLASS_COLORS[cl], label=cl)
                bottom += frac[cl].to_numpy()
            ax.set_ylabel("% of cells")
            ax.set_xlabel("Lane")
            ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.set_title("Assignment outcome per lane", fontsize=11)
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig,
                "guide_assignment_per_lane",
                SECTION_GUIDES,
                "Assignment outcome per lane",
                "Exact assignment composition by lane.",
            )
    _plot_representation(expr, guides, reg, cfg)


def _plot_representation(expr, guides, reg: FigureRegistry, cfg: Config) -> None:
    """Target and guide representation."""
    obs = expr.obs
    if OBS_TARGET not in obs.columns:
        return
    counts = obs[OBS_TARGET].astype(str).value_counts()
    counts = counts.drop([cfg.guides.unassigned_label, cfg.guides.ambiguous_label], errors="ignore")
    if counts.empty:
        return
    original_n = len(counts)
    if original_n > LARGE_PLOT_MAX_OVERVIEW_TARGETS:
        counts = counts.head(LARGE_PLOT_MAX_OVERVIEW_TARGETS)
        logger.info("Target representation plot restricted to top %d/%d targets", len(counts), original_n)
    fig, ax = plt.subplots(figsize=(max(6, min(24, 0.12 * len(counts) + 4)), 3.8))
    colors = [("#38a169" if target == cfg.guides.ntc_label else "#2b6cb0") for target in counts.index]
    ax.bar(range(len(counts)), counts.to_numpy(), color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=5)
    ax.set_ylabel("Cells")
    ax.set_title((f"Cells per target ({original_n} targets total)"), fontsize=11)
    ax.axhline(cfg.perturbation.min_cells_per_target, ls="--", lw=1)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "target_representation",
        SECTION_GUIDES,
        "Cells per target gene",
        (
            f"Top {len(counts)} targets by cell count are shown."
            if original_n > len(counts)
            else "Cells assigned to each perturbation."
        ),
    )
    if guides is not None:
        from .guides import guide_representation

        rep = guide_representation(guides, expr)
        if not rep.empty:
            fig, ax = plt.subplots(figsize=(5.4, 3.8))
            ax.plot(np.arange(1, len(rep) + 1), rep["n_cells"].to_numpy(), lw=1.2)
            ax.set_yscale("symlog")
            ax.set_xlabel("Guide rank")
            ax.set_ylabel("Cells assigned")
            n_zero = int((rep["n_cells"] == 0).sum())
            ax.set_title((f"Guide representation ({len(rep)} guides, {n_zero} with no cells)"), fontsize=11)
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig, "guide_representation", SECTION_GUIDES, "Guide representation", "Cells assigned per guide, ranked."
            )


# Clustering figures


def plot_high_moi_calling(expr, reg: FigureRegistry, cfg: Config) -> None:
    """High-MOI membership diagnostics (only called in ``assignment_mode: high_moi``)."""
    from .high_moi import OBS_N_CALLED, OBS_N_GUIDES, OBS_N_TARGETS, UNS_RANK_PROFILE, cells_per_target

    hcfg = cfg.guides.high_moi
    obs = expr.obs
    profile = expr.uns.get(UNS_RANK_PROFILE)
    if profile is not None and len(profile):
        prof = pd.DataFrame(profile)
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.fill_between(prof["rank"], prof["p10_umi"] + 1, prof["p90_umi"] + 1, alpha=0.25, lw=0, label="p10-p90")
        ax.plot(prof["rank"], prof["median_umi"] + 1, marker="o", ms=3, lw=1.2, label="median")
        ax.axhline(hcfg.min_umi + 1, ls="--", lw=1, color="k", label=f"min_umi = {hcfg.min_umi}")
        ax.set_yscale("log")
        ax.set_xlabel("Guide rank within cell (by UMI)")
        ax.set_ylabel("UMI count + 1")
        ax.set_title("Rank-ordered guide UMI profile", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "high_moi_rank_umi_profile",
            SECTION_GUIDES,
            "Rank-ordered guide UMI profile",
            (
                "Median (line) and p10-p90 band (shade) of the k-th ranked guide UMI count per cell. A knee between "
                f"real integrations and ambient background should sit above the membership threshold (dashed, min_umi = {hcfg.min_umi})."
            ),
        )
    if OBS_N_TARGETS in obs.columns:
        nt = obs[OBS_N_TARGETS].to_numpy()
        # Pre-cap counts: over-cap cells keep their real count here (their membership rows are cleared).
        ng = obs[OBS_N_CALLED].to_numpy() if OBS_N_CALLED in obs.columns else obs[OBS_N_GUIDES].to_numpy()
        n_over = int((ng > hcfg.max_guides_per_cell).sum())
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
        cap_top = int(min(max(ng.max(), hcfg.max_guides_per_cell + 5, 1), max(60, hcfg.max_guides_per_cell + 10)))
        for ax, vals, label, top in (
            (axes[0], ng, "Called guides per cell (before the cap)", cap_top),
            (axes[1], nt, "Targets per cell (membership)", int(min(max(nt.max(), 1), 40))),
        ):
            ax.hist(np.clip(vals, 0, top), bins=np.arange(-0.5, top + 1.5, 1))
            ax.set_xlabel(label + (f"; values > {top} shown at {top}" if vals.max() > top else ""))
            ax.set_ylabel("Cells")
            sns.despine(ax=ax)
        axes[0].axvline(hcfg.max_guides_per_cell + 0.5, ls="--", lw=1, color="k")
        axes[0].set_title(
            f"Called guides (median {np.median(ng[ng > 0]) if (ng > 0).any() else 0:.0f}; {n_over:,} cells above the cap)",
            fontsize=10,
        )
        axes[1].set_title("Distinct targets per cell", fontsize=10)
        fig.tight_layout()
        reg.save(
            fig,
            "high_moi_guides_per_cell",
            SECTION_GUIDES,
            "Called guides and targets per cell",
            (
                f"Left: guides passing the call per cell before the max_guides_per_cell = {hcfg.max_guides_per_cell} gate "
                f"(dashed); the {n_over:,} cells to its right are classed ambiguous and carry no membership. "
                "Right: distinct targets per cell under membership."
            ),
        )
    try:
        counts = cells_per_target(expr, cfg).drop(index=cfg.guides.ntc_label, errors="ignore")
    except KeyError:
        counts = pd.Series(dtype=int)
    if len(counts):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.hist(np.log10(counts.to_numpy() + 1), bins=40)
        ax.axvline(np.log10(cfg.perturbation.min_cells_per_target + 1), ls="--", lw=1, color="k")
        ax.set_xlabel("log10(cells carrying the target + 1)")
        ax.set_ylabel("Targets")
        ax.set_title(f"Cells per target (median {np.median(counts):.0f})", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "high_moi_cells_per_target",
            SECTION_GUIDES,
            "Cells per target (membership)",
            f"Number of cells carrying each target under membership; dashed line = perturbation.min_cells_per_target = {cfg.perturbation.min_cells_per_target}.",
        )


def plot_clustering(
    expr, reg: FigureRegistry, cfg: Config, *, name_prefix: str = "", section: str = SECTION_CLUSTERING, label: str = ""
) -> None:
    """PCA and UMAP diagnostics."""
    if "pca" in expr.uns and "variance_ratio" in expr.uns["pca"]:
        variance_ratio = expr.uns["pca"]["variance_ratio"]
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.plot(np.arange(1, len(variance_ratio) + 1), variance_ratio, "o-", ms=3)
        ax.set_yscale("log")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance ratio")
        ax.set_title(f"PCA scree plot{label}", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            f"{name_prefix}pca_variance",
            section,
            f"PCA variance ratio{label}",
            (f"The pipeline used {cfg.cluster.n_pcs} PCs for the neighbour graph."),
        )
    if "X_umap" not in expr.obsm:
        return
    coords = np.asarray(expr.obsm["X_umap"])
    obs = expr.obs
    large_plot = _is_large_plot_dataset(expr)
    max_points = LARGE_PLOT_MAX_CELLS if large_plot else None
    if CLUSTER_KEY in obs.columns:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        _scatter_umap(
            ax,
            coords,
            obs[CLUSTER_KEY].astype(str).to_numpy(),
            True,
            (f"Leiden clusters (resolution {cfg.cluster.leiden_resolution})"),
            max_points=max_points,
            seed=cfg.run.seed,
        )
        fig.tight_layout()
        reg.save(
            fig,
            f"{name_prefix}umap_clusters",
            section,
            f"UMAP coloured by Leiden cluster{label}",
            (
                f"{obs[CLUSTER_KEY].nunique()} clusters."
                + (f" Up to {LARGE_PLOT_MAX_CELLS:,} cells shown." if large_plot else "")
            ),
        )
    qc_keys = [
        (key, title)
        for key, title in [
            ("n_genes_by_counts", "Genes per cell"),
            ("total_counts", "Total UMIs"),
            ("pct_counts_mt", "% mitochondrial"),
        ]
        if key in obs.columns
    ]
    if qc_keys:
        fig, axes = plt.subplots(1, len(qc_keys), figsize=(4.6 * len(qc_keys), 4.0))
        for ax, (key, title) in zip(np.atleast_1d(axes), qc_keys):
            _scatter_umap(ax, coords, obs[key].to_numpy(), False, title, max_points=max_points, seed=cfg.run.seed)
        fig.tight_layout()
        reg.save(
            fig,
            f"{name_prefix}umap_qc_metrics",
            section,
            f"UMAP coloured by QC metrics{label}",
            "QC metrics projected onto the transcriptional embedding.",
        )
    if LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        _scatter_umap(
            ax, coords, obs[LANE_KEY].astype(str).to_numpy(), True, "Lane", max_points=max_points, seed=cfg.run.seed
        )
        fig.tight_layout()
        reg.save(
            fig, f"{name_prefix}umap_lane", section, f"UMAP coloured by lane{label}", "Lane composition on the UMAP."
        )
    if OBS_CLASS in obs.columns:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        _scatter_umap(
            ax,
            coords,
            obs[OBS_CLASS].astype(str).to_numpy(),
            True,
            "Guide assignment class",
            max_points=max_points,
            seed=cfg.run.seed,
        )
        fig.tight_layout()
        reg.save(
            fig,
            f"{name_prefix}umap_assignment_class",
            section,
            f"UMAP coloured by guide assignment{label}",
            "Guide assignment classes on the UMAP.",
        )
    if OBS_TARGET in obs.columns:
        targets = obs[OBS_TARGET].astype(str)
        n_targets = targets.nunique()
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        _scatter_umap(
            ax,
            coords,
            targets.to_numpy(),
            True,
            f"Target gene ({n_targets} levels)",
            legend=n_targets <= 25,
            max_points=max_points,
            seed=cfg.run.seed,
        )
        fig.tight_layout()
        reg.save(
            fig,
            f"{name_prefix}umap_target_gene",
            section,
            "UMAP coloured by target gene",
            (
                "Perturbation identity on the UMAP."
                + (" Legend omitted because there are too many targets." if n_targets > 25 else "")
            ),
        )
    if CLUSTER_KEY in obs.columns and LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1:
        table = obs.groupby([CLUSTER_KEY, LANE_KEY], observed=True).size().unstack(fill_value=0)
        frac = table.div(table.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(frac)), 3.8))
        bottom = np.zeros(len(frac))
        palette = sns.color_palette("tab20", frac.shape[1])
        for i, lane in enumerate(frac.columns):
            ax.bar(frac.index.astype(str), frac[lane], bottom=bottom, color=palette[i], label=str(lane))
            bottom += frac[lane].to_numpy()
        ax.set_xlabel("Leiden cluster")
        ax.set_ylabel("% of cluster")
        ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title("Lane composition per cluster", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            f"{name_prefix}cluster_lane_composition",
            section,
            "Lane composition per cluster",
            "Exact lane composition within Leiden clusters.",
        )


# Perturbation overview


def plot_perturbation_overview(results: PerturbationResults, reg: FigureRegistry, cfg: Config) -> None:
    """Volcano and ranked knockdown summary."""
    if results.table.empty:
        return
    table = results.table
    primary = results.primary_control
    lfc = table[f"log2fc_{primary}"].to_numpy(dtype=float)
    fdr = table[f"ks_fdr_{primary}"].to_numpy(dtype=float)
    hit = table[f"is_hit_{primary}"].to_numpy(dtype=bool)
    names = table["target_gene"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    with np.errstate(divide="ignore"):
        y = -np.log10(np.clip(fdr, 1e-300, 1))
    ax.scatter(lfc[~hit], y[~hit], s=18, label="not significant", alpha=0.7)
    ax.scatter(lfc[hit], y[hit], s=22, label="effective knockdown", alpha=0.8)
    ax.axhline(-np.log10(cfg.perturbation.fdr_alpha), ls="--", lw=1)
    ax.axvline(cfg.perturbation.max_log2fc_for_hit, ls="--", lw=1)
    finite_lfc = np.nan_to_num(lfc, nan=np.inf)
    for idx in np.argsort(finite_lfc)[: min(12, len(lfc))]:
        if np.isfinite(lfc[idx]) and np.isfinite(y[idx]):
            ax.annotate(names[idx], (lfc[idx], y[idx]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("log2 fold change (perturbed / control)")
    ax.set_ylabel("-log10 FDR (KS test)")
    ax.set_title((f"Perturbation strength — control: {results.control_labels[primary]}"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "perturbation_volcano",
        SECTION_PERTURBATION,
        "Volcano of perturbation strength",
        "Each point is one target gene.",
    )
    # Bounded waterfall.
    ranked = table.sort_values(f"log2fc_{primary}", ascending=True)
    original_n = len(ranked)
    if original_n > LARGE_PLOT_MAX_OVERVIEW_TARGETS:
        ranked = ranked.head(LARGE_PLOT_MAX_OVERVIEW_TARGETS)
    fig, ax = plt.subplots(figsize=(max(6, min(24, 0.10 * len(ranked) + 4)), 4.0))
    ranked_lfc = ranked[f"log2fc_{primary}"].to_numpy(dtype=float)
    ranked_hit = ranked[f"is_hit_{primary}"].to_numpy(dtype=bool)
    ranked_names = ranked["target_gene"].to_numpy()
    ax.bar(range(len(ranked)), ranked_lfc)
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels(ranked_names, rotation=90, fontsize=5)
    ax.axhline(0, lw=0.8)
    ax.set_ylabel("log2 fold change")
    ax.set_title((f"Strongest knockdowns ({int(hit.sum())}/{len(hit)} effective)"), fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "perturbation_waterfall",
        SECTION_PERTURBATION,
        "Knockdown strength per target",
        (
            f"Top {len(ranked)} of {original_n} targets shown, ranked by knockdown strength."
            if original_n > len(ranked)
            else "Targets ranked by knockdown strength."
        ),
    )
    if len(results.controls_used) > 1:
        other = CONTROL_OTHER if primary == CONTROL_NTC else CONTROL_NTC
        lfc2 = table[f"log2fc_{other}"].to_numpy(dtype=float)
        ok = ~(np.isnan(lfc) | np.isnan(lfc2))
        if ok.sum() > 2:
            fig, ax = plt.subplots(figsize=(4.8, 4.6))
            ax.scatter(lfc[ok], lfc2[ok], s=20, alpha=0.8)
            low = min(lfc[ok].min(), lfc2[ok].min()) - 0.1
            high = max(lfc[ok].max(), lfc2[ok].max()) + 0.1
            ax.plot([low, high], [low, high], ls="--", lw=1)
            r = float(np.corrcoef(lfc[ok], lfc2[ok])[0, 1])
            ax.set_xlabel((f"log2FC vs {results.control_labels[primary]}"))
            ax.set_ylabel((f"log2FC vs {results.control_labels[other]}"))
            ax.set_title((f"Control comparison (Pearson r = {r:.2f})"), fontsize=10)
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig,
                "perturbation_control_comparison",
                SECTION_PERTURBATION,
                "Effect size under both control definitions",
                "Agreement between the two control definitions.",
            )


# Per-target perturbation plots


def plot_per_target(
    expr, results: PerturbationResults, reg: FigureRegistry, cfg: Config, rng: Optional[np.random.Generator] = None
) -> None:
    """Per-target expression and embedding diagnostics.

    Ordinary datasets retain the original all-target behavior. Large datasets
    receive only a bounded set of top-ranked target figures.
    """
    if results.table.empty:
        return
    rng = rng or np.random.default_rng(cfg.run.seed)
    obs = expr.obs
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    coords = np.asarray(expr.obsm["X_umap"]) if "X_umap" in expr.obsm else None
    primary = results.primary_control
    top_n = cfg.perturbation.top_n_report
    large_plot = _is_large_plot_dataset(expr, len(results.table))
    if large_plot:
        n_plot = min(LARGE_PLOT_MAX_PER_TARGET, len(results.table))
        plot_table = results.table.head(n_plot).copy()
        logger.info("Large-data perturbation plotting: %d/%d targets", len(plot_table), len(results.table))
    else:
        plot_table = results.table
    for plot_rank, (_, row) in enumerate(plot_table.iterrows()):
        gene = str(row["target_gene"])
        if gene not in expr.var_names:
            continue
        pert_indices = np.flatnonzero((targets_col == gene) & (klass == CLASS_TARGETING))
        if not len(pert_indices):
            continue
        # Expression groups
        groups = {}
        target_values = _gene_values(expr, gene, pert_indices)
        groups["perturbed"] = target_values
        for control in results.controls_used:
            if control == CONTROL_NTC:
                pool = np.flatnonzero(klass == CLASS_NTC)
            else:
                pool = np.flatnonzero((klass == CLASS_TARGETING) & (targets_col != gene))
            if large_plot:
                ctrl_indices = _sample_pool(pool, LARGE_PLOT_MAX_CONTROL_CELLS, cfg.run.seed + plot_rank)
            else:
                ctrl_indices = pool
            groups[CONTROL_LABELS[control]] = _gene_values(expr, gene, ctrl_indices)
        n_panels = 3 if coords is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.0))
        axes = np.atleast_1d(axes)
        # ECDF
        ax = axes[0]
        for label, values in groups.items():
            if values.size:
                sns.ecdfplot(x=values, ax=ax, label=(f"{label} (n={values.size:,})"))
        ax.set_xlabel((f"{gene} expression (log-normalized)"))
        ax.set_ylabel("Cumulative fraction of cells")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        fdr = row.get(f"ks_fdr_{primary}", np.nan)
        lfc = row.get(f"log2fc_{primary}", np.nan)
        ax.set_title((f"{gene}: log2FC = {lfc:.2f}, FDR = {fdr:.2g}"), fontsize=10)
        sns.despine(ax=ax)
        # Violin
        ax = axes[1]
        nonempty = [(label, values) for label, values in groups.items() if values.size]
        if nonempty:
            plot_df = pd.DataFrame(
                {
                    "expression": np.concatenate([values for _, values in nonempty]),
                    "group": np.concatenate([np.repeat(label, len(values)) for label, values in nonempty]),
                }
            )
            sns.violinplot(data=plot_df, x="group", y="expression", ax=ax, cut=0, inner="box")
        ax.set_xlabel("")
        ax.set_ylabel(f"{gene} expression")
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        ax.set_title((f"{gene} expression by group"), fontsize=10)
        sns.despine(ax=ax)
        # UMAP
        if coords is not None:
            ax = axes[2]
            if large_plot:
                bg_pool = np.flatnonzero(~((targets_col == gene) & (klass == CLASS_TARGETING)))
                bg_indices = _sample_pool(bg_pool, LARGE_PLOT_BACKGROUND_CELLS, cfg.run.seed + plot_rank)
            else:
                frac = cfg.perturbation.umap_background_fraction
                background_mask = rng.random(expr.n_obs) < frac
                bg_indices = np.flatnonzero(background_mask & ~((targets_col == gene) & (klass == CLASS_TARGETING)))
            ax.scatter(
                coords[bg_indices, 0],
                coords[bg_indices, 1],
                s=3,
                alpha=0.6,
                linewidths=0,
                rasterized=True,
                label="background",
            )
            target_expr = target_values
            scatter = ax.scatter(
                coords[pert_indices, 0],
                coords[pert_indices, 1],
                s=10,
                c=target_expr,
                cmap="Reds",
                linewidths=0.25,
                rasterized=True,
            )
            plt.colorbar(scatter, ax=ax, shrink=0.75, label=(f"{gene} expression"))
            ax.set_title((f"Cells perturbed for {gene} (n={len(pert_indices):,})"), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=7, frameon=False, loc="best", markerscale=3)
            sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        is_hit = bool(row.get(f"is_hit_{primary}", False))
        reg.save(
            fig,
            f"perturbation_{gene}",
            SECTION_PER_GENE,
            f"{gene} perturbation effect",
            (
                f"Expression of {gene} in perturbed versus control cells. "
                f"log2FC = {lfc:.2f}, KS FDR = {fdr:.2g}." + (" Effective knockdown." if is_hit else "")
            ),
            in_report=(plot_rank < top_n),
        )
    logger.info(
        "Wrote %d per-target perturbation figures (%d shown inline)", len(plot_table), min(top_n, len(plot_table))
    )


# Enrichment helpers


def _order_by_similarity(matrix: pd.DataFrame) -> List[str]:
    """Hierarchically order rows by correlation."""
    if matrix.shape[0] < 3:
        return list(matrix.index)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        values = np.nan_to_num(matrix.to_numpy(dtype=float))
        dist = pdist(values, metric="correlation")
        if not np.all(np.isfinite(dist)):
            return list(matrix.index)
        order = leaves_list(linkage(dist, method="average"))
        return [matrix.index[idx] for idx in order]
    except Exception:
        return list(matrix.index)


# Enrichment overview


def plot_enrichment(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Cluster-enrichment overview plots."""
    from .enrichment import enrichment_matrix, phenocopy_similarity, significance_matrix

    if results.table.empty:
        return
    ecfg = cfg.enrichment
    lor = enrichment_matrix(results)
    fdr = significance_matrix(results)
    if lor.empty:
        return
    # Restrict enormous heatmaps to strongest composition-changing targets.
    if len(lor) > LARGE_PLOT_MAX_HEATMAP_TARGETS:
        strongest = list(results.effect_magnitude["target_gene"].head(LARGE_PLOT_MAX_HEATMAP_TARGETS))
        keep = lor.index.intersection(strongest)
        lor = lor.loc[keep]
        fdr = fdr.loc[keep]
        logger.info("Enrichment heatmap restricted to top %d targets", len(lor))
    order = _order_by_similarity(lor)
    lor = lor.loc[order]
    fdr = fdr.loc[order]
    # Main heatmap
    lim = float(np.nanpercentile(np.abs(lor.to_numpy()), 98)) or 1.0
    height = max(4.0, min(20.0, 0.12 * len(lor) + 2.0))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * lor.shape[1] + 4), height))
    im = ax.imshow(lor.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(lor.shape[1]))
    ax.set_xticklabels(lor.columns, fontsize=8)
    if len(lor) <= 200:
        ax.set_yticks(range(lor.shape[0]))
        ax.set_yticklabels(lor.index, fontsize=5)
    else:
        ax.set_yticks([])
    ax.set_xlabel((f"Cluster ({results.cluster_key})"))
    # Avoid writing thousands of star text objects.
    if lor.shape[0] <= 200:
        for i in range(lor.shape[0]):
            for j in range(lor.shape[1]):
                value = fdr.iat[i, j]
                if pd.notna(value) and value < ecfg.fdr_alpha:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="log2 odds ratio")
    ax.set_title("Perturbation enrichment across clusters", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        "enrichment_heatmap",
        SECTION_ENRICHMENT,
        "Perturbation enrichment across clusters",
        (f"Red indicates enrichment and blue depletion. {len(lor)} perturbations shown."),
    )
    # Phenocopy similarity.
    sim = phenocopy_similarity(results)
    if not sim.empty and sim.shape[0] >= 3:
        sim_order = _order_by_similarity(sim)
        sim = sim.loc[sim_order, sim_order]
        size = max(5.0, min(18.0, 0.10 * len(sim) + 3))
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(sim.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        if len(sim) <= 100:
            ax.set_xticks(range(len(sim)))
            ax.set_xticklabels(sim.index, rotation=90, fontsize=4)
            ax.set_yticks(range(len(sim)))
            ax.set_yticklabels(sim.index, fontsize=4)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
        ax.set_title("Perturbation phenocopy similarity", fontsize=11)
        fig.tight_layout()
        reg.save(
            fig,
            "enrichment_phenocopy",
            SECTION_ENRICHMENT,
            "Perturbation similarity",
            (f"Correlation of cluster-composition profiles for {len(sim)} perturbations."),
        )
    # Composition bars: top 30.
    comp = results.composition
    magnitude = results.effect_magnitude
    top = list(magnitude["target_gene"].head(30))
    if top:
        reference = results.reference_composition[results.primary_control]
        plot_df = pd.concat([reference.to_frame("REFERENCE").T, comp.loc[top]])
        fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(plot_df) + 2), 4.4))
        bottom = np.zeros(len(plot_df))
        palette = sns.color_palette("tab20", plot_df.shape[1])
        for i, cluster in enumerate(plot_df.columns):
            ax.bar(range(len(plot_df)), plot_df[cluster], bottom=bottom, color=palette[i], label=str(cluster))
            bottom += plot_df[cluster].to_numpy()
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df.index, rotation=90, fontsize=6)
        ax.set_ylabel("% of cells")
        ax.legend(
            title="cluster", fontsize=6, title_fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left"
        )
        ax.set_title(("Cluster composition per perturbation (top 30 by shift)"), fontsize=10)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "enrichment_composition",
            SECTION_ENRICHMENT,
            "Cluster composition per perturbation",
            "Top composition-shifting perturbations compared with the reference.",
        )
    # Volcano.
    sub = results.table[results.table["control"] == results.primary_control]
    x = sub["log2_odds_ratio"].to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        y = -np.log10(np.clip(sub["fdr"].to_numpy(dtype=float), 1e-300, 1))
    sig = sub["significant"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(x[~sig], y[~sig], s=12, alpha=0.5, label="not significant")
    ax.scatter(x[sig], y[sig], s=20, alpha=0.8, label=(f"FDR < {ecfg.fdr_alpha}"))
    ax.axhline(-np.log10(ecfg.fdr_alpha), ls="--", lw=1)
    ax.axvline(0, ls="--", lw=1)
    labelled = sub[sig].reindex(sub[sig]["log2_odds_ratio"].abs().sort_values(ascending=False).index).head(10)
    for _, row in labelled.iterrows():
        ax.annotate(
            (f"{row['target_gene']}:{row['cluster']}"),
            (row["log2_odds_ratio"], -np.log10(max(row["fdr"], 1e-300))),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("log2 odds ratio")
    ax.set_ylabel("-log10 FDR")
    ax.set_title("Target × cluster enrichment", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig, "enrichment_volcano", SECTION_ENRICHMENT, "Enrichment volcano", "All tested target × cluster associations."
    )
    # Effect magnitude ranking.
    mag_plot = magnitude.copy()
    original_n = len(mag_plot)
    if original_n > LARGE_PLOT_MAX_OVERVIEW_TARGETS:
        mag_plot = mag_plot.head(LARGE_PLOT_MAX_OVERVIEW_TARGETS)
    if not mag_plot.empty:
        fig, ax = plt.subplots(figsize=(max(6, min(24, 0.10 * len(mag_plot) + 4)), 3.8))
        ax.bar(range(len(mag_plot)), mag_plot["composition_shift_pct"])
        ax.set_xticks(range(len(mag_plot)))
        ax.set_xticklabels(mag_plot["target_gene"], rotation=90, fontsize=5)
        ax.set_ylabel("Composition shift (%)")
        ax.set_title("Strongest cluster-composition shifts", fontsize=10)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "enrichment_effect_magnitude",
            SECTION_ENRICHMENT,
            "Composition shift per perturbation",
            (
                f"Top {len(mag_plot)} of {original_n} perturbations shown."
                if original_n > len(mag_plot)
                else "Cluster-composition displacement by perturbation."
            ),
        )


# Per-target enrichment


def plot_enrichment_per_target(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Target-specific enrichment figures."""
    if results.table.empty:
        return
    obs = expr.obs
    cluster_key = results.cluster_key
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    clusters_col = obs[cluster_key].astype(str).to_numpy()
    coords = np.asarray(expr.obsm["X_umap"]) if "X_umap" in expr.obsm else None
    reference = results.reference_composition[results.primary_control]
    clusters = list(results.composition.columns)
    ranked = list(results.effect_magnitude["target_gene"])
    hit_targets = set(results.targets_with_hits())
    ordered = [target for target in ranked if target in hit_targets] + [
        target for target in ranked if target not in hit_targets
    ]
    large_plot = _is_large_plot_dataset(expr, len(ordered))
    if large_plot:
        ordered = ordered[:LARGE_PLOT_MAX_PER_TARGET]
        logger.info("Large-data enrichment plotting: %d per-target figures", len(ordered))
    top_n = cfg.enrichment.top_n_report
    sub_table = results.table[results.table["control"] == results.primary_control]
    for rank, gene in enumerate(ordered):
        if gene not in (results.composition.index):
            continue
        comp = results.composition.loc[gene]
        rows = sub_table[sub_table["target_gene"] == gene].set_index("cluster")
        n_panels = 3 if coords is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4.7 * n_panels, 4.0))
        axes = np.atleast_1d(axes)
        # Composition
        ax = axes[0]
        idx = np.arange(len(clusters))
        ref_vals = reference.reindex(clusters, fill_value=0).to_numpy(dtype=float)
        comp_vals = comp.reindex(clusters, fill_value=0).to_numpy(dtype=float)
        ax.bar(idx - 0.2, ref_vals, width=0.4, label="reference")
        ax.bar(idx + 0.2, comp_vals, width=0.4, label=gene)
        for i, cluster in enumerate(clusters):
            if cluster in rows.index and bool(rows.at[cluster, "significant"]):
                ax.text(i, max(comp_vals[i], ref_vals[i]) + 1, "*", ha="center", fontsize=11)
        ax.set_xticks(idx)
        ax.set_xticklabels(clusters, fontsize=7)
        ax.set_xlabel(f"Cluster ({cluster_key})")
        ax.set_ylabel("% of cells")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{gene} cluster composition", fontsize=10)
        sns.despine(ax=ax)
        # Odds ratios
        ax = axes[1]
        lor = rows["log2_odds_ratio"].reindex(clusters)
        sig = rows["significant"].reindex(clusters).fillna(False).to_numpy(dtype=bool)
        ax.bar(idx, lor.to_numpy(dtype=float))
        ax.axhline(0, lw=0.8)
        ax.set_xticks(idx)
        ax.set_xticklabels(clusters, fontsize=7)
        ax.set_xlabel(f"Cluster ({cluster_key})")
        ax.set_ylabel("log2 odds ratio")
        ax.set_title((f"{gene} enrichment ({int(sig.sum())} significant cluster(s))"), fontsize=10)
        sns.despine(ax=ax)
        # UMAP
        if coords is not None:
            ax = axes[2]
            pert_indices = np.flatnonzero((targets_col == gene) & (klass == CLASS_TARGETING))
            if large_plot:
                bg_pool = np.flatnonzero(~((targets_col == gene) & (klass == CLASS_TARGETING)))
                bg_indices = _sample_pool(bg_pool, LARGE_PLOT_BACKGROUND_CELLS, cfg.run.seed + rank)
            else:
                bg_indices = np.flatnonzero(~((targets_col == gene) & (klass == CLASS_TARGETING)))
            ax.scatter(coords[bg_indices, 0], coords[bg_indices, 1], s=2, alpha=0.4, linewidths=0, rasterized=True)
            cluster_of = clusters_col[pert_indices]
            palette = sns.color_palette("tab20", len(clusters))
            color_map = {cluster: palette[i] for i, cluster in enumerate(clusters)}
            ax.scatter(
                coords[pert_indices, 0],
                coords[pert_indices, 1],
                s=10,
                c=[color_map.get(cluster, (0.3, 0.3, 0.3)) for cluster in cluster_of],
                linewidths=0.2,
                rasterized=True,
            )
            ax.set_title((f"{gene} cells by cluster (n={len(pert_indices):,})"), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)
        best = rows["log2_odds_ratio"].abs().idxmax() if len(rows) else None
        caption = f"Cluster distribution of cells perturbed for {gene}."
        if best is not None and cl_has_hit(rows, best):
            caption += f" Strongest association: cluster {best}, FDR={rows.at[best, 'fdr']:.2g}."
        fig.tight_layout()
        reg.save(
            fig,
            f"enrichment_{gene}",
            SECTION_ENRICH_PER_TARGET,
            f"{gene} cluster enrichment",
            caption,
            in_report=(rank < top_n),
        )


def cl_has_hit(rows: pd.DataFrame, cluster) -> bool:
    """True when target/cluster pair is significant."""
    try:
        return bool(rows.at[cluster, "significant"])
    except (KeyError, ValueError):
        return False


# PS overview


def plot_ps_scores(expr, results, perturbation_results, reg: FigureRegistry, cfg: Config) -> None:
    """Perturbation-score overview and method comparison."""
    from .ps_score import QUADRANT_ESCAPER, QUADRANT_KD, compare_with_perturbation_strength

    if results is None or results.summary.empty:
        return
    summary = results.summary
    summary_plot = summary.copy()
    original_n = len(summary_plot)
    if original_n > LARGE_PLOT_MAX_OVERVIEW_TARGETS:
        summary_plot = summary_plot.head(LARGE_PLOT_MAX_OVERVIEW_TARGETS)
        logger.info("PS overview plot restricted to top %d/%d targets", len(summary_plot), original_n)
    # Outcome stacked bars.
    fig, ax = plt.subplots(figsize=(max(6, min(24, 0.10 * len(summary_plot) + 4)), 4.0))
    bottom = np.zeros(len(summary_plot))
    for key, label in [
        ("pct_successful_kd", QUADRANT_KD),
        ("pct_escaper", QUADRANT_ESCAPER),
        ("pct_non_responder", "non-responder"),
        ("pct_low_signal", "low signal"),
    ]:
        ax.bar(range(len(summary_plot)), summary_plot[key], bottom=bottom, label=label)
        bottom += summary_plot[key].to_numpy()
    ax.set_xticks(range(len(summary_plot)))
    ax.set_xticklabels(summary_plot["target_gene"], rotation=90, fontsize=5)
    ax.set_ylabel("% of perturbed cells")
    ax.set_title("Per-cell perturbation outcome by target", fontsize=11)
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "ps_outcome_by_target",
        SECTION_PS,
        "Per-cell perturbation outcome",
        (
            f"Top {len(summary_plot)} of {original_n} targets shown."
            if original_n > len(summary_plot)
            else "Per-target PS/expression outcome classification."
        ),
    )
    # Escaper ranking.
    esc = summary.sort_values("pct_escaper", ascending=False)
    if len(esc) > LARGE_PLOT_MAX_OVERVIEW_TARGETS:
        esc = esc.head(LARGE_PLOT_MAX_OVERVIEW_TARGETS)
    fig, ax = plt.subplots(figsize=(max(6, min(24, 0.10 * len(esc) + 4)), 3.6))
    ax.bar(range(len(esc)), esc["pct_escaper"])
    ax.set_xticks(range(len(esc)))
    ax.set_xticklabels(esc["target_gene"], rotation=90, fontsize=5)
    ax.set_ylabel("% escapers")
    ax.set_title("Escaper fraction per target", fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "ps_escaper_fraction",
        SECTION_PS,
        "Escaper fraction per target",
        "Targets ranked by the fraction of PS-high cells retaining target expression.",
    )
    # Agreement with group-level perturbation strength.
    merged = compare_with_perturbation_strength(
        results, perturbation_results.table, perturbation_results.primary_control
    )
    lfc_col = f"log2fc_{perturbation_results.primary_control}"
    if not merged.empty and lfc_col in merged.columns:
        ok = merged[lfc_col].notna() & merged["pct_successful_kd"].notna()
        if ok.sum() > 2:
            x = merged.loc[ok, lfc_col].to_numpy(dtype=float)
            y = merged.loc[ok, "pct_successful_kd"].to_numpy(dtype=float)
            r = float(np.corrcoef(x, y)[0, 1])
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            ax.scatter(x, y, s=28, alpha=0.8)
            ax.set_xlabel(("Target expression log2FC (group-level)"))
            ax.set_ylabel("% successful knockdown (PS)")
            ax.set_title((f"PS vs group-level knockdown (Pearson r={r:.2f})"), fontsize=10)
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig,
                "ps_vs_perturbation_strength",
                SECTION_PS,
                "Per-cell PS vs group-level knockdown",
                "Comparison between downstream perturbation-response scoring and direct target-expression reduction.",
            )
    _plot_ps_quadrants(expr, results, reg, cfg)


# PS quadrants


def _plot_ps_quadrants(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Score-vs-expression quadrant scatter."""
    from .ps_score import QUADRANT_ESCAPER, QUADRANT_KD, QUADRANT_LOW, QUADRANT_NONRESPONDER

    top_n = cfg.ps_score.top_n_report
    large_plot = bool(getattr(results, "large_mode", False)) or _is_large_plot_dataset(expr, len(results.summary))
    if large_plot:
        n_plot = min(LARGE_PLOT_MAX_PER_TARGET, len(results.summary))
        genes_to_plot = list(results.summary["target_gene"].head(n_plot))
    else:
        genes_to_plot = list(results.summary["target_gene"])
    for rank, gene in enumerate(genes_to_plot):
        series = results.scores.get(gene)
        if series is None or gene not in expr.var_names:
            continue
        cells = series.index.intersection(expr.obs_names)
        if not len(cells):
            continue
        cell_indices = expr.obs_names.get_indexer(cells)
        valid = cell_indices >= 0
        cells = cells[valid]
        cell_indices = cell_indices[valid]
        expression = pd.Series(_gene_values(expr, gene, cell_indices), index=cells)
        ps = series.reindex(cells)
        targets = expr.obs.loc[cells, OBS_TARGET].astype(str)
        klass = expr.obs.loc[cells, OBS_CLASS].astype(str)
        is_target = (targets == gene) & (klass == CLASS_TARGETING)
        cut = results.expression_cut.get(gene, float(np.median(expression)))
        threshold = results.ps_threshold
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        ctrl_idx = cells[~is_target.to_numpy()]
        if len(ctrl_idx) > 2000:
            rng = np.random.default_rng(cfg.run.seed + rank)
            ctrl_idx = pd.Index(rng.choice(ctrl_idx, size=2000, replace=False))
        if len(ctrl_idx):
            ax.scatter(
                ps.loc[ctrl_idx],
                expression.loc[ctrl_idx],
                s=14,
                alpha=0.35,
                linewidths=0,
                label="control cells",
                rasterized=True,
            )
        target_cells = cells[is_target.to_numpy()]
        quadrant = results.quadrants.get(gene)
        if quadrant is not None:
            quadrant_target = quadrant.reindex(target_cells).fillna(QUADRANT_LOW).astype(str)
            category_to_code = {QUADRANT_KD: 0, QUADRANT_ESCAPER: 1, QUADRANT_NONRESPONDER: 2, QUADRANT_LOW: 3}
            codes = np.array([category_to_code.get(q, 3) for q in quadrant_target])
        else:
            codes = np.zeros(len(target_cells), dtype=int)
        scatter = ax.scatter(
            ps.loc[target_cells],
            expression.loc[target_cells],
            s=26,
            c=codes,
            cmap="tab10",
            alpha=0.85,
            linewidths=0.3,
            rasterized=True,
            label=f"{gene} cells",
        )
        ax.axvline(threshold, ls="--", lw=1)
        ax.axhline(cut, ls="--", lw=1)
        row = results.summary[results.summary["target_gene"] == gene].iloc[0]
        ax.set_xlabel("Perturbation score")
        ax.set_ylabel((f"{gene} expression (log-normalized)"))
        ax.set_title((f"{gene}: {row['pct_successful_kd']:.0f}% KD, {row['pct_escaper']:.0f}% escaper"), fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            f"ps_quadrant_{gene}",
            SECTION_PS_PER_TARGET,
            f"{gene} perturbation score vs expression",
            (f"PS threshold={threshold}; target-expression cut={cut:.3g}."),
            in_report=(rank < top_n),
        )
    logger.info("Wrote %d PS quadrant figures", len(genes_to_plot))


# PS LDA


def plot_ps_lda(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Plot supervised LDA representation when available."""
    if results is None or results.lda_umap is None or results.summary.empty:
        return
    coords = np.asarray(results.lda_umap, dtype=float)
    placed = np.isfinite(coords).all(axis=1)
    if placed.sum() < 10:
        logger.warning("LDA embedding placed too few cells to plot")
        return
    labels = (
        results.lda_label.astype(str).to_numpy()
        if results.lda_label is not None
        else np.full(expr.n_obs, "?", dtype=object)
    )
    n_targets = len(results.summary)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    _scatter_umap(
        ax,
        coords[placed],
        labels[placed],
        True,
        (f"Supervised LDA embedding ({n_targets} targets)"),
        size=4,
        legend=n_targets <= 24,
        max_points=LARGE_PLOT_MAX_CELLS,
        seed=cfg.run.seed,
    )
    ax.set_xlabel("LDA-UMAP1")
    ax.set_ylabel("LDA-UMAP2")
    fig.tight_layout()
    reg.save(
        fig, "ps_lda_overview", SECTION_PS, "Supervised LDA embedding", "PS_python supervised LDA/UMAP representation."
    )
    if "ps_score" in expr.obs:
        own = expr.obs["ps_score"].to_numpy(dtype=float)
        threshold = cfg.ps_score.lda_highlight_threshold
        strong = placed & np.isfinite(own) & (own >= threshold)
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        background_indices = np.flatnonzero(placed)
        if len(background_indices) > LARGE_PLOT_MAX_CELLS:
            background_indices = _sample_pool(background_indices, LARGE_PLOT_MAX_CELLS, cfg.run.seed)
        ax.scatter(
            coords[background_indices, 0],
            coords[background_indices, 1],
            s=4,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
            label="background",
        )
        strong_indices = np.flatnonzero(strong)
        scatter = ax.scatter(
            coords[strong_indices, 0],
            coords[strong_indices, 1],
            s=12,
            c=own[strong_indices],
            cmap="viridis",
            vmin=threshold,
            vmax=1.0,
            linewidths=0.2,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, shrink=0.75, label="perturbation score")
        ax.set_title((f"High-confidence responders (score >= {threshold})"), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        reg.save(
            fig,
            "ps_lda_high_confidence",
            SECTION_PS,
            "High-confidence responders on the LDA map",
            "Cells above the configured PS threshold.",
        )
    # Per-target LDA plots are only supported where target-specific score
    # columns exist. Large PS mode intentionally does not create them.
    if bool(getattr(results, "large_mode", False)):
        logger.info(
            "Large PS mode: skipping target-specific LDA figures because ps_<gene> columns are intentionally omitted."
        )
        return
    targets = expr.obs[OBS_TARGET].astype(str).to_numpy()
    top_n = cfg.ps_score.top_n_report
    for rank, gene in enumerate(results.summary["target_gene"]):
        col = f"ps_{gene}"
        if col not in expr.obs.columns:
            continue
        score = expr.obs[col].to_numpy(dtype=float)
        is_target = (targets == gene) & placed
        if is_target.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        bg_pool = np.flatnonzero(placed & ~is_target)
        bg_indices = _sample_pool(bg_pool, LARGE_PLOT_BACKGROUND_CELLS, cfg.run.seed + rank)
        ax.scatter(coords[bg_indices, 0], coords[bg_indices, 1], s=4, alpha=0.4, linewidths=0, rasterized=True)
        target_indices = np.flatnonzero(is_target)
        scatter = ax.scatter(
            coords[target_indices, 0],
            coords[target_indices, 1],
            s=18,
            c=np.nan_to_num(score[target_indices]),
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.3,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, shrink=0.75, label="perturbation score")
        ax.set_title(f"{gene} on the LDA map", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        reg.save(
            fig,
            f"ps_lda_{gene}",
            SECTION_PS_LDA,
            f"{gene} on the LDA embedding",
            f"Cells carrying {gene}, coloured by PS.",
            in_report=(rank < top_n),
        )


# lochNESS helpers


def _lochness_norm(vmax: float):
    """Symlog colour scale for lochNESS."""
    import matplotlib.colors as mcolors

    vmax = max(float(vmax), 1.0)
    return mcolors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)


def _label_clusters(ax, expr, coords: np.ndarray, highlight: str = "") -> None:
    """Place cluster labels at embedding centroids."""
    if CLUSTER_KEY not in expr.obs.columns:
        return
    clusters = expr.obs[CLUSTER_KEY].astype(str).to_numpy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        if not mask.sum():
            continue
        cx = np.median(coords[mask, 0])
        cy = np.median(coords[mask, 1])
        is_top = str(cluster) == str(highlight)
        ax.text(
            cx,
            cy,
            str(cluster),
            fontsize=(8 if is_top else 6.5),
            fontweight=("bold" if is_top else "normal"),
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor=("#fefcbf" if is_top else "white"), edgecolor="none", alpha=0.75
            ),
        )


# lochNESS


def plot_lochness(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Plot lochNESS summaries.

    Standard mode retains full per-target score plots.

    Large mode consumes ``self_score`` and summary outputs because the full
    cell × perturbation score representation is intentionally not materialized.
    """
    if results is None or results.summary.empty:
        return
    lcfg = cfg.lochness
    summary = results.summary
    large_mode = bool(getattr(results, "self_only", False))
    coords = np.asarray(expr.obsm["X_umap"]) if "X_umap" in expr.obsm else None
    # 1. self-enrichment target ranking
    summary_plot = summary.copy()
    original_n = len(summary_plot)
    if original_n > LARGE_PLOT_MAX_OVERVIEW_TARGETS:
        summary_plot = summary_plot.head(LARGE_PLOT_MAX_OVERVIEW_TARGETS)
    fig, ax = plt.subplots(figsize=(max(6, min(24, 0.10 * len(summary_plot) + 4)), 4.0))
    values = summary_plot["mean_lochness_in_own_cells"].to_numpy(dtype=float)
    ax.bar(range(len(summary_plot)), values)
    ax.axhline(0, lw=0.8)
    ax.axhline(lcfg.enrichment_cut, ls="--", lw=1)
    ax.set_xticks(range(len(summary_plot)))
    ax.set_xticklabels(summary_plot["target_gene"], rotation=90, fontsize=5)
    ax.set_ylabel("mean lochNESS in own cells")
    ax.set_title((f"Self-enrichment by perturbation (k={results.n_neighbors})"), fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "lochness_self_enrichment",
        SECTION_LOCHNESS,
        "Self-enrichment per perturbation",
        (
            f"Top {len(summary_plot)} of {original_n} perturbations shown."
            if original_n > len(summary_plot)
            else "Mean lochNESS in each perturbation's own cells."
        ),
    )
    # 2. score distribution per target (standard mode only)
    if not large_mode and results.scores:
        top = list(summary["target_gene"].head(30))
        top_available = [g for g in top if g in results.scores]
        if top_available:
            long_df = pd.DataFrame(
                {
                    "lochNESS": np.concatenate([results.scores[g] for g in top_available]),
                    "target": np.concatenate([[g] * len(results.scores[g]) for g in top_available]),
                }
            )
            fig, ax = plt.subplots(figsize=(max(6, 0.34 * len(top_available)), 4.2))
            sns.violinplot(
                data=long_df, x="target", y="lochNESS", ax=ax, cut=0, inner=None, linewidth=0.5, order=top_available
            )
            ax.axhline(0, color="black", lw=0.8)
            ax.tick_params(axis="x", rotation=90, labelsize=6)
            ax.set_xlabel("")
            ax.set_title("lochNESS across all cells, per perturbation (top 30)", fontsize=10)
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig,
                "lochness_distributions",
                SECTION_LOCHNESS,
                "lochNESS distribution per perturbation",
                (
                    "Each violin is one perturbation's score across every cell. "
                    "A long upper tail means a subset of the manifold is strongly "
                    "enriched for it, even when most cells sit at background."
                ),
            )
    # 3. target × cluster heatmap
    if not results.by_cluster.empty:
        matrix = results.by_cluster.copy()
        if len(matrix) > LARGE_PLOT_MAX_HEATMAP_TARGETS:
            keep = list(summary["target_gene"].head(LARGE_PLOT_MAX_HEATMAP_TARGETS))
            matrix = matrix.loc[matrix.index.intersection(keep)]
        try:
            matrix = matrix[sorted(matrix.columns, key=lambda col: (float(col), col))]
        except ValueError:
            matrix = matrix[sorted(matrix.columns)]
        order = _order_by_similarity(matrix)
        matrix = matrix.loc[order]
        limit = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98)) or 1.0
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * matrix.shape[1] + 4), max(4, min(20, 0.10 * len(matrix) + 2))))
        im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, fontsize=8)
        if len(matrix) <= 200:
            ax.set_yticks(range(len(matrix)))
            ax.set_yticklabels(matrix.index, fontsize=5)
        else:
            ax.set_yticks([])
        ax.set_xlabel(f"Cluster ({CLUSTER_KEY})")
        plt.colorbar(im, ax=ax, shrink=0.6, label="mean lochNESS")
        ax.set_title("Mean lochNESS per cluster", fontsize=11)
        fig.tight_layout()
        reg.save(
            fig, "lochness_by_cluster", SECTION_LOCHNESS, "lochNESS by cluster", (f"{len(matrix)} perturbations shown.")
        )
    # 3. self score on UMAP
    if coords is not None and results.self_score is not None:
        self_values = np.asarray(results.self_score, dtype=float)
        valid_indices = np.flatnonzero(np.isfinite(self_values))
        if len(valid_indices) > LARGE_PLOT_MAX_CELLS:
            selected = _sample_pool(valid_indices, LARGE_PLOT_MAX_CELLS, cfg.run.seed)
        else:
            selected = valid_indices
        if len(selected):
            plot_values = self_values[selected]
            limit = float(np.nanpercentile(np.abs(plot_values), 98)) or 1.0
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            scatter = ax.scatter(
                coords[selected, 0],
                coords[selected, 1],
                s=4,
                c=plot_values,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                linewidths=0,
                rasterized=True,
            )
            plt.colorbar(scatter, ax=ax, shrink=0.75, label="lochNESS (own perturbation)")
            ax.set_title("Self-lochNESS on the embedding", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)
            fig.tight_layout()
            reg.save(
                fig,
                "lochness_self_umap",
                SECTION_LOCHNESS,
                "Self lochNESS on the embedding",
                (f"{len(selected):,} representative scored cells shown."),
            )
    # Large mode intentionally stops here.
    if large_mode:
        logger.info(
            "Large-data lochNESS plotting: skipping per-target all-cell maps "
            "because full score vectors were intentionally not materialized."
        )
        return
    if coords is None or not results.scores:
        return
    top_n = lcfg.top_n_report
    for rank, gene in enumerate(summary["target_gene"]):
        if gene not in results.scores:
            continue
        score = np.asarray(results.scores[gene], dtype=float)
        own = expr.obs[OBS_TARGET if cfg.guides.assignment_mode == "high_moi" else cfg.lochness.genotype_key].astype(str).to_numpy() == gene
        row = summary[summary["target_gene"] == gene].iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
        # Score map.
        ax = axes[0]
        finite = np.isfinite(score)
        vmax = float(np.nanmax(score[finite])) if finite.any() else 1.0
        norm = _lochness_norm(vmax)
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=4,
            c=np.nan_to_num(score),
            cmap="RdBu_r",
            norm=norm,
            linewidths=0,
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, shrink=0.78, label="lochNESS")
        ax.set_title(f"{gene}: neighbourhood enrichment", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
        # Location map.
        ax = axes[1]
        background_pool = np.flatnonzero(~own)
        if _is_large_plot_dataset(expr):
            background_pool = _sample_pool(background_pool, LARGE_PLOT_BACKGROUND_CELLS, cfg.run.seed + rank)
        ax.scatter(
            coords[background_pool, 0], coords[background_pool, 1], s=3, alpha=0.4, linewidths=0, rasterized=True
        )
        own_indices = np.flatnonzero(own)
        ax.scatter(
            coords[own_indices, 0], coords[own_indices, 1], s=14, linewidths=0.3, rasterized=True, label=f"{gene} cells"
        )
        _label_clusters(ax, expr, coords, highlight=str(row.get("top_cluster", "")))
        ax.set_title((f"{gene} cells (n={len(own_indices):,})"), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        reg.save(
            fig,
            f"lochness_{gene}",
            SECTION_LOCHNESS_PER_TARGET,
            f"{gene} lochNESS map",
            (f"Mean lochNESS in own cells = {row['mean_lochness_in_own_cells']:.2f}."),
            in_report=(rank < top_n),
        )


# Module/program helpers


def _label_colors(labels: List[str]) -> dict:
    unique = list(dict.fromkeys(labels))
    palette = sns.color_palette("tab20", max(len(unique), 3))
    return {label: tuple(palette[i % len(palette)]) for i, label in enumerate(unique)}


def _block_spans(seq: List[str]):
    """Return contiguous label spans."""
    spans = []
    if not len(seq):
        return spans
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            spans.append((seq[start], start, i))
            start = i
    return spans


def _display_order(items, label_of, label_rank, item_rank):
    return sorted(items, key=lambda item: (label_rank[label_of[item]], item_rank[item]))


# Modules: effect heatmap


def _plot_effect_heatmap(results, reg, cfg) -> None:
    prog = results.gene_programs.set_index("gene")["program"]
    mod = results.modules.set_index("target_gene")["module"]
    prog_rank = {label: i for i, label in enumerate(results.program_labels)}
    mod_rank = {label: i for i, label in enumerate(results.module_labels)}
    gene_rank = {gene: i for i, gene in enumerate(results.gene_order)}
    pert_rank = {pert: i for i, pert in enumerate(results.perturbation_order)}
    genes = _display_order(results.gene_order, prog, prog_rank, gene_rank)
    perts = _display_order(results.perturbation_order, mod, mod_rank, pert_rank)
    # Guard gigantic module heatmaps too.
    if len(perts) > LARGE_PLOT_MAX_HEATMAP_TARGETS:
        perts = perts[:LARGE_PLOT_MAX_HEATMAP_TARGETS]
        logger.info("Module heatmap restricted to %d perturbations", len(perts))
    matrix = results.effect_matrix.loc[perts, genes].T
    row_labels = [prog[gene] for gene in genes]
    col_labels = [mod[target] for target in perts]
    prog_colors = _label_colors(results.program_labels)
    mod_colors = _label_colors(results.module_labels)
    limit = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98)) or 1.0
    width = max(7.0, min(24.0, 0.04 * len(perts) + 4))
    height = max(6.0, min(24.0, 0.015 * len(genes) + 4))
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(2, 3, width_ratios=[0.02, 1, 0.04], height_ratios=[0.03, 1], wspace=0.02, hspace=0.02)
    ax_top = fig.add_subplot(grid[0, 1])
    ax_left = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[1, 1])
    cax = fig.add_subplot(grid[1, 2])
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks([])
    if len(perts) <= 60:
        ax.set_xticks(range(len(perts)))
        ax.set_xticklabels(perts, rotation=90, fontsize=5)
    else:
        ax.set_xticks([])
    ax.set_xlabel((f"{len(perts)} perturbations ({results.n_modules} modules)"))
    ax.set_ylabel((f"{len(genes)} genes ({results.n_programs} programs)"))
    ax_top.imshow(np.array([[mod_colors[label] for label in col_labels]]), aspect="auto")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_left.imshow(np.array([[prog_colors[label]] for label in row_labels]), aspect="auto")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for _, _, end in _block_spans(col_labels)[:-1]:
        ax.axvline(end - 0.5, lw=0.6)
    for _, _, end in _block_spans(row_labels)[:-1]:
        ax.axhline(end - 0.5, lw=0.6)
    plt.colorbar(image, cax=cax, label="log2FC vs control")
    ax_top.set_title(
        (f"Regulome map: {results.n_modules} modules × {results.n_programs} programs"), fontsize=11, pad=14
    )
    reg.save(
        fig,
        "regulome_heatmap",
        SECTION_MODULES,
        "Co-functional modules and gene programs",
        "Perturbation × downstream-gene effect matrix.",
    )


def _plot_module_program(results, reg, cfg) -> None:
    matrix = results.module_program
    if matrix.empty:
        return
    limit = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * matrix.shape[1] + 2), max(3, 0.5 * matrix.shape[0] + 1.5)))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    display_labels = getattr(results, "program_display_labels", {})
    xticklabels = [display_labels.get(col, col) for col in matrix.columns]
    ax.set_xticklabels(xticklabels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Gene program")
    ax.set_ylabel("Co-functional module")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.to_numpy()[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=ax, shrink=0.7, label="mean log2FC")
    ax.set_title("Module → program regulatory strength", fontsize=11)
    fig.tight_layout()
    reg.save(
        fig,
        "module_program_strength",
        SECTION_MODULES,
        "Module × program strength",
        "Average signed effect of each module on each gene program.",
    )


def _plot_alluvial(results, reg, cfg) -> None:
    matrix = results.module_program
    if matrix.empty or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        return
    magnitude = matrix.abs().fillna(0.0)
    module_totals = magnitude.sum(axis=1)
    program_totals = magnitude.sum(axis=0)
    total = float(magnitude.to_numpy().sum())
    if total <= 0:
        return
    gap = 0.02

    def _stack(totals):
        positions = {}
        y = 1.0
        n = len(totals)
        usable = 1.0 - gap * (n - 1)
        for name, value in totals.items():
            height = usable * (value / total)
            positions[name] = (y - height, y)
            y -= height + gap
        return positions

    left = _stack(module_totals)
    right = _stack(program_totals)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * matrix.shape[0] + 2)))
    left_cursor = {module: left[module][1] for module in matrix.index}
    right_cursor = {program: right[program][1] for program in matrix.columns}
    xs = np.linspace(0, 1, 40)
    smooth = xs * xs * (3 - 2 * xs)
    for module in matrix.index:
        for program in matrix.columns:
            value = matrix.loc[module, program]
            if not np.isfinite(value) or value == 0:
                continue
            thickness = abs(value) / total
            left_high = left_cursor[module]
            left_low = left_high - thickness
            left_cursor[module] = left_low
            right_high = right_cursor[program]
            right_low = right_high - thickness
            right_cursor[program] = right_low
            X = 0.08 + 0.84 * xs
            low = left_low + (right_low - left_low) * smooth
            high = left_high + (right_high - left_high) * smooth
            ax.fill_between(X, low, high, alpha=0.45, lw=0)
    for module, (low, high) in left.items():
        ax.add_patch(plt.Rectangle((0.04, low), 0.04, high - low))
        ax.text(0.02, (low + high) / 2, module, ha="right", va="center", fontsize=8)
    for program, (low, high) in right.items():
        ax.add_patch(plt.Rectangle((0.92, low), 0.04, high - low))
        display_labels = getattr(results, "program_display_labels", {})
        label_text = display_labels.get(program, program)
        ax.text(0.98, (low + high) / 2, label_text, ha="left", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.axis("off")
    ax.set_title("Module → program regulation", fontsize=11)
    reg.save(
        fig,
        "module_program_alluvial",
        SECTION_MODULES,
        "Module → program alluvial",
        "Regulatory flow from perturbation modules to gene programs.",
    )


def _plot_module_correlation(results, reg, cfg) -> None:
    perts = list(results.perturbation_order)
    if len(perts) < 3:
        return
    if len(perts) > LARGE_PLOT_MAX_HEATMAP_TARGETS:
        perts = perts[:LARGE_PLOT_MAX_HEATMAP_TARGETS]
    module = results.modules.set_index("target_gene")["module"]
    module_rank = {label: i for i, label in enumerate(results.module_labels)}
    perturbation_rank = {perturbation: i for i, perturbation in enumerate(perts)}
    order = _display_order(perts, module, module_rank, perturbation_rank)
    corr = results.effect_matrix.loc[order].T.corr(method=results.module_correlation)
    size = max(5, min(18, 0.10 * len(order) + 3))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    if len(order) <= 70:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=4)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(image, ax=ax, shrink=0.6, label=(f"{results.module_correlation} r"))
    ax.set_title("Perturbation similarity", fontsize=11)
    fig.tight_layout()
    reg.save(
        fig,
        "module_correlation",
        SECTION_MODULES,
        "Perturbation correlation",
        "Correlation of transcriptome-wide perturbation effects.",
    )


def _plot_program_activity(results, reg, cfg) -> None:
    activity = results.program_activity
    if activity.empty:
        return
    limit = float(np.nanmax(np.abs(activity.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * activity.shape[1] + 2), max(2.5, 0.5 * activity.shape[0] + 1.5)))
    image = ax.imshow(activity.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(activity.shape[1]))
    ax.set_xticklabels(activity.columns, fontsize=7)
    ax.set_yticks(range(activity.shape[0]))
    ax.set_yticklabels(activity.index)
    ax.set_xlabel((f"Cluster ({cfg.modules.cluster_key})"))
    ax.set_ylabel("Gene program")
    plt.colorbar(image, ax=ax, shrink=0.7, label="mean program score")
    ax.set_title("Gene-program activity by cluster", fontsize=11)
    fig.tight_layout()
    reg.save(
        fig,
        "program_activity_by_cluster",
        SECTION_MODULES,
        "Program activity by cluster",
        "Mean gene-program activity within each cell-state cluster.",
    )


def _plot_program_umaps(expr, results, reg, cfg) -> None:
    if "X_umap" not in expr.obsm or not results.score_columns:
        return
    coords = np.asarray(expr.obsm["X_umap"])
    top_n = cfg.modules.top_n_report
    large_plot = _is_large_plot_dataset(expr)
    for rank, (label, column) in enumerate(zip(results.program_labels, results.score_columns)):
        if column not in expr.obs:
            continue
        fig, ax = plt.subplots(figsize=(5, 4.2))
        _scatter_umap(
            ax,
            coords,
            expr.obs[column].to_numpy(),
            False,
            (f"Program {label} activity"),
            size=4,
            cmap="RdBu_r",
            max_points=(LARGE_PLOT_MAX_CELLS if large_plot else None),
            seed=cfg.run.seed + rank,
        )
        fig.tight_layout()
        reg.save(
            fig,
            f"program_{label}_umap",
            SECTION_MODULES,
            f"Program {label} activity (UMAP)",
            (f"Per-cell activity score for program {label}."),
            in_report=(rank < top_n),
        )


def _plot_networks(results, reg, cfg) -> None:
    connectivity = results.module_connectivity
    if connectivity.shape[0] >= 2:
        fig, ax = plt.subplots(
            figsize=(max(4, 0.5 * connectivity.shape[0] + 2), max(4, 0.5 * connectivity.shape[0] + 2))
        )
        image = ax.imshow(connectivity.to_numpy(), cmap="magma", aspect="auto")
        ax.set_xticks(range(connectivity.shape[1]))
        ax.set_xticklabels(connectivity.columns, fontsize=7)
        ax.set_yticks(range(connectivity.shape[0]))
        ax.set_yticklabels(connectivity.index, fontsize=7)
        plt.colorbar(image, ax=ax, shrink=0.7, label="normalized connectivity")
        ax.set_title("Module-module connectivity", fontsize=11)
        fig.tight_layout()
        reg.save(
            fig,
            "module_connectivity",
            SECTION_MODULES,
            "Module-module connectivity",
            "Regulatory connectivity between co-functional perturbation modules.",
        )
    if not cfg.modules.draw_networks:
        return
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx is not installed; skipping network graphs")
        return
    if connectivity.shape[0] >= 2:
        graph = nx.Graph()
        sizes = results.modules["module"].value_counts()
        for module in connectivity.index:
            graph.add_node(module, size=int(sizes.get(module, 1)))
        for i, left in enumerate(connectivity.index):
            for right in connectivity.columns[i + 1 :]:
                weight = connectivity.loc[left, right] + connectivity.loc[right, left]
                if weight > 0:
                    graph.add_edge(left, right, weight=weight)
        position = nx.spring_layout(graph, seed=0, weight="weight")
        fig, ax = plt.subplots(figsize=(6, 5))
        nx.draw_networkx_edges(graph, position, ax=ax)
        nx.draw_networkx_nodes(
            graph, position, ax=ax, node_size=[(80 + 40 * graph.nodes[node]["size"]) for node in graph.nodes]
        )
        nx.draw_networkx_labels(graph, position, ax=ax, font_size=9, font_weight="bold")
        ax.axis("off")
        ax.set_title("Module interaction network", fontsize=11)
        reg.save(
            fig,
            "module_network",
            SECTION_MODULES,
            "Module interaction network",
            "Network representation of module connectivity.",
        )


def _plot_program_enrichment(results, reg: FigureRegistry, cfg: Config) -> None:
    """Program x pathway dot plot (gseapy): the top significant terms of each program."""
    enr = getattr(results, "program_enrichment", None)
    if enr is None or enr.empty:
        return
    pe_cfg = cfg.modules.program_enrichment
    # Top terms per program by p-value, picked here so the term that labels a program is always shown
    # (gseapy's own pick ranks by FDR, where ties are common).
    sig = enr[enr["fdr"] <= pe_cfg.fdr_alpha]
    sig = sig.sort_values("p_value").groupby("program_id").head(pe_cfg.top_terms_per_program)
    if sig.empty:
        return
    display = getattr(results, "program_display_labels", {})
    # gseapy.dotplot reads its own column names: colour from "Adjusted P-value", dot size from "Overlap" ("k/M").
    df = pd.DataFrame(
        {
            "program": sig["program_id"].map(lambda p: display.get(p, p)),
            "Term": sig["clean_term"],
            "Adjusted P-value": sig["fdr"],
            "Overlap": sig["overlap_count"].astype(str) + "/" + sig["gene_set_size"].astype(str),
        }
    )
    order = [display.get(p, p) for p in results.program_labels if display.get(p, p) in set(df["program"])]
    n_terms = sig["clean_term"].nunique()
    ax = gseapy.dotplot(
        df,
        column="Adjusted P-value",
        x="program",
        x_order=order,
        cutoff=pe_cfg.fdr_alpha,
        top_term=pe_cfg.top_terms_per_program,
        title="Gene Program Pathway Enrichment",
        xticklabels_rot=35,
        figsize=(max(5, 0.9 * len(order) + 2.5), max(3.5, 0.35 * n_terms + 1.8)),
    )
    reg.save(
        ax.figure,
        "program_enrichment",
        SECTION_MODULES,
        "Program pathway enrichment",
        "Over-representation analysis of Stage 7 gene programs across biological pathway databases.",
    )


def plot_modules(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """All co-functional-module / gene-program figures."""
    if results is None or results.effect_matrix.empty:
        return
    _plot_effect_heatmap(results, reg, cfg)
    _plot_module_program(results, reg, cfg)
    _plot_alluvial(results, reg, cfg)
    _plot_program_enrichment(results, reg, cfg)
    _plot_module_correlation(results, reg, cfg)
    _plot_program_activity(results, reg, cfg)
    _plot_program_umaps(expr, results, reg, cfg)
    _plot_networks(results, reg, cfg)
    logger.info("Wrote module/program figures (%d modules, %d programs)", results.n_modules, results.n_programs)


# Perturbation Distance & Phenotype Space Plots


def _significance_stars(fdr: float) -> str:
    """Format FDR significance stars for heatmap annotations."""
    if pd.isna(fdr):
        return ""
    if fdr < 0.001:
        return "***"
    if fdr < 0.01:
        return "**"
    if fdr < 0.05:
        return "*"
    return ""


def plot_perturbation_atlas(meta_table: pd.DataFrame, reg: FigureRegistry, cfg: Config) -> None:
    """Perturbation Atlas: Multi-dimensional heatmap of efficacy, penetrance, topology, and phenotype magnitude.

    Rows: Targets (sorted by Energy distance)
    Columns:
      1. Efficacy (KD Strength = -target_log2fc)
      2. Penetrance (PS Median / Responder Fraction)
      3. Topology (lochNESS mean)
      4. Phenotype Magnitude (Energy Distance)
    Columns are Z-scored for visualization; significance stars indicate FDR.
    Side annotations show co-functional and phenotype module memberships.
    """
    if meta_table is None or meta_table.empty:
        return
    df = meta_table.copy()
    if "target_gene" not in df.columns:
        return
    # Select top targets by energy distance (or another available metric)
    sort_col = (
        "energy_distance"
        if "energy_distance" in df.columns and df["energy_distance"].notna().any()
        else ("ps_median" if "ps_median" in df.columns else "target_gene")
    )
    df = df.sort_values(sort_col, ascending=(sort_col == "target_gene")).reset_index(drop=True)
    max_n = getattr(cfg.visualization, "atlas_top_n", 50)
    if len(df) > max_n:
        df = df.iloc[:max_n].copy()
    # Identify candidate heatmap feature columns
    col_specs = [
        ("target_log2fc", "Efficacy\n(KD Strength)", True, "target_fdr"),
        ("ps_median", "Penetrance\n(PS Median)", False, None),
        ("lochness_mean", "Topology\n(lochNESS)", False, None),
        ("energy_distance", "Phenotype\n(Energy Dist)", False, "distance_fdr"),
    ]
    active_cols = []
    col_labels = []
    fdr_cols = []
    matrix_data = []
    for col_name, label, invert, fdr_col in col_specs:
        if col_name in df.columns and df[col_name].notna().any():
            vals = df[col_name].to_numpy(dtype=float, na_value=np.nan)
            if invert:
                # Invert KD so higher = stronger depletion
                vals = -vals
            active_cols.append(col_name)
            col_labels.append(label)
            fdr_cols.append(fdr_col if fdr_col in df.columns else None)
            matrix_data.append(vals)
    if len(active_cols) < 2:
        return
    M = np.column_stack(matrix_data)
    # Column-wise Z-scoring
    M_z = np.zeros_like(M)
    for j in range(M.shape[1]):
        col_vals = M[:, j]
        valid = ~np.isnan(col_vals)
        if valid.sum() > 1:
            mean = np.mean(col_vals[valid])
            std = np.std(col_vals[valid])
            std = std if std > 1e-8 else 1.0
            M_z[valid, j] = (col_vals[valid] - mean) / std
        else:
            M_z[:, j] = 0.0
    targets = df["target_gene"].tolist()
    n_targets = len(targets)
    # Side annotations
    has_cofunc = "cofunctional_module" in df.columns and df["cofunctional_module"].notna().any()
    has_pheno = "phenotype_module" in df.columns and df["phenotype_module"].notna().any()
    n_side = int(has_cofunc) + int(has_pheno)
    fig_height = max(6.0, 0.28 * n_targets + 2.0)
    fig_width = 8.0 + (1.2 * n_side)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, 1 + n_side + 1, width_ratios=[0.4] * n_side + [4.0, 0.2], wspace=0.15)
    col_idx = 0
    # 1. Co-functional module annotation
    if has_cofunc:
        ax_co = fig.add_subplot(gs[0, col_idx])
        col_idx += 1
        modules = df["cofunctional_module"].fillna("None").astype(str).tolist()
        unique_m = sorted(set(modules))
        cmap_co = plt.cm.tab20(np.linspace(0, 1, len(unique_m)))
        m_map = {m: cmap_co[i] for i, m in enumerate(unique_m)}
        colors_co = np.array([m_map[m] for m in modules])[:, :3]
        ax_co.imshow(colors_co[:, None, :], aspect="auto", interpolation="nearest")
        ax_co.set_xticks([0])
        ax_co.set_xticklabels(["Co-func\nModule"], rotation=90, fontsize=8)
        ax_co.set_yticks([])
        for spine in ax_co.spines.values():
            spine.set_visible(False)
    # 2. Phenotype module annotation
    if has_pheno:
        ax_ph = fig.add_subplot(gs[0, col_idx])
        col_idx += 1
        pmodules = df["phenotype_module"].fillna("None").astype(str).tolist()
        unique_pm = sorted(set(pmodules))
        cmap_ph = plt.cm.Set2(np.linspace(0, 1, max(len(unique_pm), 1)))
        pm_map = {m: cmap_ph[i % len(cmap_ph)] for i, m in enumerate(unique_pm)}
        colors_ph = np.array([pm_map[m] for m in pmodules])[:, :3]
        ax_ph.imshow(colors_ph[:, None, :], aspect="auto", interpolation="nearest")
        ax_ph.set_xticks([0])
        ax_ph.set_xticklabels(["Pheno\nModule"], rotation=90, fontsize=8)
        ax_ph.set_yticks([])
        for spine in ax_ph.spines.values():
            spine.set_visible(False)
    # 3. Main heatmap
    ax_main = fig.add_subplot(gs[0, col_idx])
    cbar_ax = fig.add_subplot(gs[0, col_idx + 1])
    vmax = max(2.5, float(np.nanmax(np.abs(M_z))))
    im = ax_main.imshow(M_z, aspect="auto", cmap="vlag", vmin=-vmax, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, cax=cbar_ax, label="Column Z-score")
    ax_main.set_xticks(range(len(col_labels)))
    ax_main.set_xticklabels(col_labels, rotation=0, fontsize=9, fontweight="bold")
    ax_main.set_yticks(range(n_targets))
    ax_main.set_yticklabels(targets, fontsize=8)
    # Significance stars overlay
    for i in range(n_targets):
        for j in range(len(active_cols)):
            f_col = fdr_cols[j]
            if f_col and f_col in df.columns:
                f_val = df.iloc[i][f_col]
                stars = _significance_stars(f_val)
                if stars:
                    text_color = "black" if abs(M_z[i, j]) < 1.2 else "white"
                    ax_main.text(j, i, stars, ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")
    ax_main.set_title("Perturbation Atlas", fontsize=12, fontweight="bold", pad=12)
    reg.save(
        fig,
        "perturbation_atlas",
        SECTION_DISTANCE,
        "Perturbation Atlas",
        "Multi-dimensional overview of perturbation efficacy, penetrance (PS score), "
        "manifold topology (lochNESS), and phenotype magnitude (Energy distance). "
        "Columns are Z-scored for visual comparison; asterisks indicate statistical significance (* FDR < 0.05, ** FDR < 0.01, *** FDR < 0.001).",
    )


def plot_ps_vs_distance(meta_table: pd.DataFrame, reg: FigureRegistry, cfg: Config) -> None:
    """PS × Perturbation Distance map: penetrance vs phenotype distance."""
    if meta_table is None or meta_table.empty:
        return
    df = meta_table.copy()
    if "energy_distance" not in df.columns:
        return
    y_col = (
        "ps_median"
        if "ps_median" in df.columns
        else ("ps_responder_fraction" if "ps_responder_fraction" in df.columns else None)
    )
    if y_col is None:
        return
    valid_mask = df["energy_distance"].notna() & df[y_col].notna()
    if valid_mask.sum() < 3:
        return
    sub = df[valid_mask].copy()
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # Point size by lochNESS
    if "lochness_mean" in sub.columns and sub["lochness_mean"].notna().any():
        l_vals = sub["lochness_mean"].fillna(0).to_numpy(dtype=float)
        l_min, l_max = np.min(l_vals), np.max(l_vals)
        norm_l = (l_vals - l_min) / (l_max - l_min + 1e-6)
        sizes = 40.0 + 160.0 * norm_l
    else:
        sizes = np.full(len(sub), 60.0)
    # Color by phenotype or cofunctional module
    color_col = (
        "phenotype_module"
        if "phenotype_module" in sub.columns and sub["phenotype_module"].notna().any()
        else (
            "cofunctional_module"
            if "cofunctional_module" in sub.columns and sub["cofunctional_module"].notna().any()
            else None
        )
    )
    if color_col:
        cats = sub[color_col].fillna("None").astype(str)
        unique_cats = sorted(set(cats))
        palette = sns.color_palette("tab10", n_colors=len(unique_cats))
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_cats)}
        point_colors = [color_map[c] for c in cats]
    else:
        point_colors = "#2b6cb0"
    # Significance styling
    sig_col = (
        "distance_significant"
        if "distance_significant" in sub.columns
        else ("distance_fdr" if "distance_fdr" in sub.columns else None)
    )
    if sig_col == "distance_significant":
        is_sig = sub[sig_col].fillna(False).to_numpy(dtype=bool)
    elif sig_col == "distance_fdr":
        is_sig = (sub[sig_col].fillna(1.0) < cfg.distance.fdr_threshold).to_numpy(dtype=bool)
    else:
        is_sig = np.ones(len(sub), dtype=bool)
    # Scatter points
    ax.scatter(
        sub.loc[is_sig, "energy_distance"],
        sub.loc[is_sig, y_col],
        s=sizes[is_sig],
        c=[point_colors[i] for i in np.where(is_sig)[0]] if isinstance(point_colors, list) else point_colors,
        alpha=0.85,
        edgecolors="#1a202c",
        linewidths=1.2,
        label=f"Significant (FDR < {cfg.distance.fdr_threshold})",
    )
    if (~is_sig).sum() > 0:
        ax.scatter(
            sub.loc[~is_sig, "energy_distance"],
            sub.loc[~is_sig, y_col],
            s=sizes[~is_sig] * 0.7,
            c=[point_colors[i] for i in np.where(~is_sig)[0]] if isinstance(point_colors, list) else point_colors,
            alpha=0.35,
            edgecolors="gray",
            linewidths=0.8,
            label="Not significant",
        )
    # Annotate top targets by distance
    top_dist = sub.nlargest(min(12, len(sub)), "energy_distance")
    for _, row in top_dist.iterrows():
        ax.annotate(
            str(row["target_gene"]),
            (row["energy_distance"], row[y_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            alpha=0.9,
        )
    ax.set_xlabel("Phenotype Magnitude (Energy Distance from Control)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Penetrance ({y_col.replace('_', ' ').title()})", fontsize=10, fontweight="bold")
    ax.set_title("Perturbation Penetrance vs Phenotype Distance Map", fontsize=11, fontweight="bold")
    if color_col and isinstance(point_colors, list) and len(unique_cats) <= 10:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c], markersize=8, label=c)
            for c in unique_cats
        ]
        ax.legend(handles=handles, title=color_col.replace("_", " ").title(), loc="best", fontsize=8)
    sns.despine(ax=ax)
    reg.save(
        fig,
        "ps_vs_distance_map",
        SECTION_DISTANCE,
        "PS vs Perturbation Distance Map",
        "Single-cell perturbation penetrance (PS score) plotted against global phenotype magnitude (Energy distance). "
        "Point size reflects continuous manifold enrichment (lochNESS).",
    )


def plot_perturbation_space(
    dist_space_res, meta_table: Optional[pd.DataFrame], reg: FigureRegistry, cfg: Config
) -> None:
    """Perturbation Phenotype Space: PCoA projections of pairwise distance manifold."""
    if dist_space_res is None or dist_space_res.coordinates.empty:
        return
    coords = dist_space_res.coordinates.copy()
    if "PCoA1" not in coords.columns or "PCoA2" not in coords.columns:
        return
    if meta_table is not None and not meta_table.empty:
        coords = pd.merge(coords, meta_table, on="target_gene", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    # Panel 1: Colored by phenotype module
    ax1 = axes[0]
    if "phenotype_module" in coords.columns and coords["phenotype_module"].notna().any():
        cats = coords["phenotype_module"].fillna("None").astype(str)
        unique_cats = sorted(set(cats))
        palette = sns.color_palette("tab10", n_colors=len(unique_cats))
        c_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_cats)}
        for c in unique_cats:
            sub = coords[cats == c]
            ax1.scatter(sub["PCoA1"], sub["PCoA2"], c=[c_map[c]], label=c, s=50, alpha=0.85, edgecolors="none")
        if len(unique_cats) <= 12:
            ax1.legend(title="Phenotype Module", fontsize=8, loc="best")
    else:
        ax1.scatter(coords["PCoA1"], coords["PCoA2"], c="#2b6cb0", s=50, alpha=0.85)
    ax1.set_xlabel("PCoA 1", fontsize=10, fontweight="bold")
    ax1.set_ylabel("PCoA 2", fontsize=10, fontweight="bold")
    ax1.set_title("Perturbation Phenotype Space (Modules)", fontsize=11, fontweight="bold")
    sns.despine(ax=ax1)
    # Panel 2: Colored by Energy distance from control (or PS score)
    ax2 = axes[1]
    color_metric = (
        "energy_distance"
        if "energy_distance" in coords.columns and coords["energy_distance"].notna().any()
        else ("ps_median" if "ps_median" in coords.columns and coords["ps_median"].notna().any() else None)
    )
    if color_metric:
        c_vals = coords[color_metric].to_numpy(dtype=float)
        sc = ax2.scatter(
            coords["PCoA1"], coords["PCoA2"], c=c_vals, cmap="viridis", s=50, alpha=0.85, edgecolors="none"
        )
        fig.colorbar(sc, ax=ax2, label=color_metric.replace("_", " ").title())
    else:
        ax2.scatter(coords["PCoA1"], coords["PCoA2"], c="#4a5568", s=50, alpha=0.85)
    ax2.set_xlabel("PCoA 1", fontsize=10, fontweight="bold")
    ax2.set_ylabel("PCoA 2", fontsize=10, fontweight="bold")
    ax2.set_title(
        f"Phenotype Space ({color_metric.replace('_', ' ').title() if color_metric else 'PCoA'})",
        fontsize=11,
        fontweight="bold",
    )
    sns.despine(ax=ax2)
    reg.save(
        fig,
        "perturbation_phenotype_space",
        SECTION_DISTANCE_SPACE,
        "Perturbation Phenotype Space (PCoA)",
        "Low-dimensional projection of pairwise perturbation Energy Distances via classical Multidimensional Scaling (PCoA).",
    )


def plot_module_concordance(meta_table: pd.DataFrame, reg: FigureRegistry, cfg: Config) -> None:
    """Module concordance heatmap: Co-functional modules vs Phenotype modules."""
    if meta_table is None or meta_table.empty:
        return
    df = meta_table
    if "cofunctional_module" not in df.columns or "phenotype_module" not in df.columns:
        return
    valid = df["cofunctional_module"].notna() & df["phenotype_module"].notna()
    if valid.sum() < 4:
        return
    sub = df[valid]
    co_mods = sub["cofunctional_module"].astype(str)
    ph_mods = sub["phenotype_module"].astype(str)
    if co_mods.nunique() < 2 or ph_mods.nunique() < 2:
        return
    ct = pd.crosstab(co_mods, ph_mods)
    # Compute ARI/NMI if sklearn available
    metric_str = ""
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(co_mods, ph_mods)
        nmi = normalized_mutual_info_score(co_mods, ph_mods)
        metric_str = f" (ARI = {ari:.3f}, NMI = {nmi:.3f})"
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(max(5.0, 0.6 * ct.shape[1] + 2.0), max(4.5, 0.5 * ct.shape[0] + 1.5)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Target Count"}, ax=ax, linewidths=0.5)
    ax.set_xlabel("Phenotype Modules (Distance Space)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Co-functional Modules (Gene Programs)", fontsize=10, fontweight="bold")
    ax.set_title(f"Module Concordance{metric_str}", fontsize=11, fontweight="bold", pad=12)
    reg.save(
        fig,
        "module_concordance",
        SECTION_DISTANCE_SPACE,
        "Module Concordance Heatmap",
        f"Cross-tabulation comparing gene-effect co-functional modules with cell-state phenotype distance modules{metric_str}.",
    )


def plot_distance_overview(dist_res, reg: FigureRegistry, cfg: Config) -> None:
    """Distance overview: Ranked bar plot of Energy Distance vs Control."""
    if dist_res is None or dist_res.table.empty:
        return
    tbl = dist_res.table.copy()
    if "energy_distance" not in tbl.columns:
        return
    tbl = tbl.sort_values("energy_distance", ascending=False).reset_index(drop=True)
    max_bars = min(40, len(tbl))
    sub = tbl.iloc[:max_bars]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.25 * max_bars + 1.5), 4.5))
    is_sig = sub["significant"] if "significant" in sub.columns else pd.Series([True] * len(sub))
    colors = ["#dd6b20" if s else "#a0aec0" for s in is_sig]
    ax.bar(range(len(sub)), sub["energy_distance"], color=colors, edgecolor="none", width=0.8)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["target_gene"], rotation=90, fontsize=8)
    ax.set_ylabel("Energy Distance vs Control", fontsize=10, fontweight="bold")
    ax.set_title("Perturbation Distance vs Control Ranking", fontsize=11, fontweight="bold")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#dd6b20", label=f"FDR < {cfg.distance.fdr_threshold}"),
        plt.Rectangle((0, 0), 1, 1, color="#a0aec0", label="Not significant"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "perturbation_distance_ranking",
        SECTION_DISTANCE,
        "Perturbation Distance vs Control Ranking",
        "Ranked Energy Distance from unperturbed control cells across perturbation targets.",
    )


def plot_distance_figures(dist_res, meta_table: Optional[pd.DataFrame], reg: FigureRegistry, cfg: Config) -> None:
    """Generate all figures for Perturbation Distance and Perturbation Atlas."""
    if dist_res is not None and not dist_res.table.empty:
        plot_distance_overview(dist_res, reg, cfg)
    if meta_table is not None and not meta_table.empty:
        if cfg.visualization.perturbation_atlas:
            plot_perturbation_atlas(meta_table, reg, cfg)
        if cfg.visualization.ps_distance_map:
            plot_ps_vs_distance(meta_table, reg, cfg)


def plot_distance_space_figures(
    dist_space_res, meta_table: Optional[pd.DataFrame], reg: FigureRegistry, cfg: Config
) -> None:
    """Generate all figures for Perturbation Distance Space and Phenotype Modules."""
    if dist_space_res is not None and not dist_space_res.coordinates.empty:
        if cfg.visualization.perturbation_space:
            plot_perturbation_space(dist_space_res, meta_table, reg, cfg)
    if meta_table is not None and not meta_table.empty:
        if cfg.visualization.module_concordance:
            plot_module_concordance(meta_table, reg, cfg)
