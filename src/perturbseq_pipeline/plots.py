"""All figures produced by the pipeline.

Every figure goes through :class:`FigureRegistry`, which writes the file *and*
records its title, caption and section. The report builder then renders whatever
is registered, so a figure can never be produced without being reachable: those
not shown inline are still written to disk and linked from the report.

Library code never calls ``plt.show()``; the Agg backend is forced on import.
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

SECTION_QC = "qc"
SECTION_GUIDES = "guides"
SECTION_CLUSTERING = "clustering"
SECTION_PERTURBATION = "perturbation"
SECTION_PER_GENE = "perturbation/per_gene"

_CLASS_COLORS = {
    CLASS_TARGETING: "#2b6cb0",
    CLASS_NTC: "#38a169",
    CLASS_AMBIGUOUS: "#dd6b20",
    CLASS_UNASSIGNED: "#a0aec0",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """File-system-safe figure name (stage labels contain spaces)."""
    import re

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return slug or "figure"


@dataclass
class FigureRecord:
    """A figure on disk plus the metadata the report needs."""

    path: Path
    name: str
    section: str
    title: str
    caption: str = ""
    #: False for the per-gene figures beyond the top-N; still written to disk.
    in_report: bool = True

    def data_uri(self) -> str:
        """Base64 ``data:`` URI so the report can be a single portable file."""
        mime = "image/png" if self.path.suffix == ".png" else "image/svg+xml"
        return f"data:{mime};base64,{base64.b64encode(self.path.read_bytes()).decode()}"


@dataclass
class FigureRegistry:
    """Saves figures under ``<outdir>/figures/<section>/`` and indexes them."""

    outdir: Path
    cfg: Config
    records: List[FigureRecord] = field(default_factory=list)

    @property
    def figdir(self) -> Path:
        return Path(self.outdir) / "figures"

    def save(
        self,
        fig: MplFigure,
        name: str,
        section: str,
        title: str,
        caption: str = "",
        in_report: bool = True,
    ) -> FigureRecord:
        """Write ``fig`` and register it. Always closes the figure."""
        ext = self.cfg.report.figure_format
        name = _slugify(name)
        path = self.figdir / section / f"{name}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.cfg.report.figure_dpi, bbox_inches="tight")
        if self.cfg.output.save_figures_pdf:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        rec = FigureRecord(path, name, section, title, caption, in_report)
        self.records.append(rec)
        return rec

    def by_section(self, section: str, only_in_report: bool = True) -> List[FigureRecord]:
        return [
            r
            for r in self.records
            if r.section == section and (r.in_report or not only_in_report)
        ]

    def extras(self, section: str) -> List[FigureRecord]:
        """Figures written to disk but deliberately not embedded in the report."""
        return [r for r in self.records if r.section == section and not r.in_report]

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "section": r.section,
                    "name": r.name,
                    "title": r.title,
                    "in_report": r.in_report,
                    "path": str(r.path),
                }
                for r in self.records
            ]
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _obs_values(adata, key: str) -> np.ndarray:
    return adata.obs[key].to_numpy()


def _gene_values(adata, gene: str) -> np.ndarray:
    from scipy import sparse

    layer = adata.layers[LOGNORM_LAYER] if LOGNORM_LAYER in adata.layers else adata.X
    col = layer[:, adata.var_names.get_loc(gene)]
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
) -> None:
    """UMAP scatter drawn directly with matplotlib (rasterized for small files)."""
    if categorical:
        cats = pd.Index(pd.unique(pd.Series(values).astype(str))).sort_values()
        palette = sns.color_palette("tab20", max(len(cats), 3))
        for i, cat in enumerate(cats):
            m = values.astype(str) == cat
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=size,
                color=palette[i % len(palette)],
                label=str(cat),
                linewidths=0,
                rasterized=True,
            )
        if legend and len(cats) <= 25:
            ax.legend(
                markerscale=4,
                fontsize=7,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
            )
    else:
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=size,
            cmap=cmap,
            linewidths=0,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, shrink=0.75)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


# ---------------------------------------------------------------------------
# QC figures
# ---------------------------------------------------------------------------


def plot_qc(adata, reg: FigureRegistry, stage: str = "prefilter") -> None:
    """Violin + scatter QC panels, split by lane when there is more than one."""
    obs = adata.obs
    metrics = [
        ("n_genes_by_counts", "Genes per cell"),
        ("total_counts", "UMIs per cell"),
        ("pct_counts_mt", "% mitochondrial"),
        ("pct_counts_ribo", "% ribosomal"),
    ]
    metrics = [(m, l) for m, l in metrics if m in obs.columns]
    multi_lane = LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (metric, label) in zip(axes, metrics):
        if multi_lane:
            sns.violinplot(
                x=obs[LANE_KEY].astype(str), y=obs[metric], ax=ax, inner="box", cut=0
            )
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.set_xlabel("")
        else:
            sns.violinplot(y=obs[metric], ax=ax, inner="box", cut=0)
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
        "Distribution of per-cell QC metrics"
        + (" for each lane." if multi_lane else ".")
        + " Long lower tails in genes/UMIs indicate empty or dying cells;"
        " a high mitochondrial fraction indicates stressed cells.",
    )

    if {"total_counts", "n_genes_by_counts"} <= set(obs.columns):
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        color = obs["pct_counts_mt"] if "pct_counts_mt" in obs.columns else None
        s = ax.scatter(
            obs["total_counts"],
            obs["n_genes_by_counts"],
            c=color,
            s=3,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        if color is not None:
            plt.colorbar(s, ax=ax, label="% mitochondrial")
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
            "Each point is a cell. Healthy cells lie on the main diagonal band;"
            " points below it with high mitochondrial content are typically dying.",
        )

    if multi_lane:
        fig, ax = plt.subplots(figsize=(max(4, 0.7 * obs[LANE_KEY].nunique()), 3.6))
        counts = obs[LANE_KEY].astype(str).value_counts().sort_index()
        ax.bar(counts.index, counts.to_numpy(), color="#4a5568")
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
            "Large imbalances between lanes can bias clustering and should be "
            "considered when interpreting batch effects.",
        )


# ---------------------------------------------------------------------------
# Perturb-seq guide QC figures
# ---------------------------------------------------------------------------


def plot_guide_qc(expr, guides, reg: FigureRegistry, cfg: Config) -> None:
    """Guide-specific QC: UMI depth, multiplicity, dominance and assignment."""
    obs = expr.obs

    if OBS_TOTAL in obs.columns:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        vals = obs[OBS_TOTAL].to_numpy()
        ax.hist(np.log10(vals + 1), bins=60, color="#4a5568")
        ax.set_xlabel("log10(total guide UMIs per cell + 1)")
        ax.set_ylabel("Cells")
        ax.set_title("Guide UMI depth per cell", fontsize=11)
        ax.axvline(np.log10(cfg.guides.min_umi + 1), color="#e53e3e", ls="--", lw=1)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "guide_umi_depth",
            SECTION_GUIDES,
            "Guide UMI depth",
            f"Total guide UMIs per cell. The dashed line marks guides.min_umi = "
            f"{cfg.guides.min_umi}. A large spike at zero means guide capture failed "
            "for many cells.",
        )

    if OBS_NDETECTED in obs.columns:
        det = obs[OBS_NDETECTED].to_numpy()
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        top = int(min(det.max(), 15))
        ax.hist(np.clip(det, 0, top), bins=np.arange(-0.5, top + 1.5, 1), color="#2b6cb0")
        ax.set_xlabel(f"Guides detected per cell (> {cfg.guides.detection_threshold} UMI)")
        ax.set_ylabel("Cells")
        ax.set_title(f"Guide multiplicity (mean MOI = {det.mean():.2f})", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "guide_multiplicity",
            SECTION_GUIDES,
            "Guides per cell (MOI)",
            "Number of distinct guides detected per cell. A low-MOI screen should "
            "be dominated by the 1-guide bar; a heavy tail indicates multiplets.",
        )

    if {OBS_TOP, OBS_SECOND} <= set(obs.columns):
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        x = obs[OBS_TOP].to_numpy() + 1
        y = obs[OBS_SECOND].to_numpy() + 1
        klass = obs[OBS_CLASS].astype(str).to_numpy() if OBS_CLASS in obs.columns else None
        if klass is not None:
            for cl, color in _CLASS_COLORS.items():
                m = klass == cl
                if m.sum():
                    ax.scatter(
                        x[m], y[m], s=4, alpha=0.5, color=color, label=cl,
                        linewidths=0, rasterized=True,
                    )
            ax.legend(markerscale=3, fontsize=8, frameon=False)
        else:
            ax.scatter(x, y, s=4, alpha=0.5, linewidths=0, rasterized=True)
        lim = np.array([1, max(x.max(), y.max())])
        ax.plot(lim, lim / cfg.guides.dominance_ratio, color="#e53e3e", ls="--", lw=1,
                label=f"ratio = {cfg.guides.dominance_ratio}")
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
            "Cells far below the dashed line have one clearly dominant guide and are "
            "confidently assigned; cells near the diagonal are ambiguous.",
        )

    if OBS_CLASS in obs.columns:
        counts = obs[OBS_CLASS].value_counts()
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        labels = [c for c in _CLASS_COLORS if c in counts.index]
        vals = [counts[c] for c in labels]
        ax.bar(labels, vals, color=[_CLASS_COLORS[c] for c in labels])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{100 * v / expr.n_obs:.1f}%", ha="center", va="bottom", fontsize=8)
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
            "How many cells received a confident guide call. 'ambiguous' cells have "
            "guide counts but no dominant guide; 'unassigned' cells have none.",
        )

        if LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1:
            tab = (
                obs.groupby([LANE_KEY, OBS_CLASS], observed=True)
                .size()
                .unstack(fill_value=0)
            )
            frac = tab.div(tab.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(max(4.5, 0.8 * len(frac)), 3.8))
            bottom = np.zeros(len(frac))
            for cl in [c for c in _CLASS_COLORS if c in frac.columns]:
                ax.bar(frac.index.astype(str), frac[cl], bottom=bottom,
                       color=_CLASS_COLORS[cl], label=cl)
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
                "A lane with a much lower assignment rate usually had a failed or "
                "shallow guide library.",
            )

    _plot_representation(expr, guides, reg, cfg)


def _plot_representation(expr, guides, reg: FigureRegistry, cfg: Config) -> None:
    """Cells per target gene and per guide — library-balance diagnostics."""
    obs = expr.obs
    if OBS_TARGET not in obs.columns:
        return
    counts = obs[OBS_TARGET].astype(str).value_counts()
    counts = counts.drop(
        [cfg.guides.unassigned_label, cfg.guides.ambiguous_label], errors="ignore"
    )
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.16 * len(counts)), 3.8))
    colors = ["#38a169" if t == cfg.guides.ntc_label else "#2b6cb0" for t in counts.index]
    ax.bar(range(len(counts)), counts.to_numpy(), color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=6)
    ax.set_ylabel("Cells")
    ax.set_title(f"Cells per target gene ({len(counts)} targets)", fontsize=11)
    ax.axhline(cfg.perturbation.min_cells_per_target, color="#e53e3e", ls="--", lw=1)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "target_representation",
        SECTION_GUIDES,
        "Cells per target gene",
        "Green is the non-targeting control group. The dashed line is "
        f"perturbation.min_cells_per_target = {cfg.perturbation.min_cells_per_target}; "
        "targets below it cannot be tested reliably.",
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
            ax.set_title(
                f"Guide representation ({len(rep)} guides, {n_zero} with no cells)",
                fontsize=11,
            )
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig,
                "guide_representation",
                SECTION_GUIDES,
                "Guide representation",
                "Cells assigned per guide, ranked. A steep drop or many zero-cell "
                "guides indicates an unbalanced or partly failed guide library.",
            )


# ---------------------------------------------------------------------------
# Clustering figures
# ---------------------------------------------------------------------------


def plot_clustering(expr, reg: FigureRegistry, cfg: Config) -> None:
    """PCA scree, UMAP by cluster / QC / lane / perturbation class."""
    if "pca" in expr.uns and "variance_ratio" in expr.uns["pca"]:
        vr = expr.uns["pca"]["variance_ratio"]
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.plot(np.arange(1, len(vr) + 1), vr, "o-", ms=3)
        ax.set_yscale("log")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance ratio")
        ax.set_title("PCA scree plot", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "pca_variance",
            SECTION_CLUSTERING,
            "PCA variance ratio",
            "Where the curve flattens suggests how many components carry signal; "
            f"the pipeline used {cfg.cluster.n_pcs} for the neighbor graph.",
        )

    if "X_umap" not in expr.obsm:
        return
    coords = np.asarray(expr.obsm["X_umap"])
    obs = expr.obs

    if CLUSTER_KEY in obs.columns:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        _scatter_umap(ax, coords, obs[CLUSTER_KEY].astype(str).to_numpy(), True,
                      f"Leiden clusters (resolution {cfg.cluster.leiden_resolution})")
        fig.tight_layout()
        reg.save(
            fig,
            "umap_clusters",
            SECTION_CLUSTERING,
            "UMAP coloured by Leiden cluster",
            f"{obs[CLUSTER_KEY].nunique()} clusters at resolution "
            f"{cfg.cluster.leiden_resolution}.",
        )

    qc_keys = [
        (k, l)
        for k, l in [
            ("n_genes_by_counts", "Genes per cell"),
            ("total_counts", "Total UMIs"),
            ("pct_counts_mt", "% mitochondrial"),
        ]
        if k in obs.columns
    ]
    if qc_keys:
        fig, axes = plt.subplots(1, len(qc_keys), figsize=(4.6 * len(qc_keys), 4.0))
        for ax, (k, l) in zip(np.atleast_1d(axes), qc_keys):
            _scatter_umap(ax, coords, obs[k].to_numpy(), False, l)
        fig.tight_layout()
        reg.save(
            fig,
            "umap_qc_metrics",
            SECTION_CLUSTERING,
            "UMAP coloured by QC metrics",
            "A cluster driven purely by library size or mitochondrial content is a "
            "technical artefact rather than a biological state.",
        )

    if LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        _scatter_umap(ax, coords, obs[LANE_KEY].astype(str).to_numpy(), True, "Lane")
        fig.tight_layout()
        reg.save(
            fig,
            "umap_lane",
            SECTION_CLUSTERING,
            "UMAP coloured by lane",
            "Lanes should intermix. Separation by lane means a batch effect; set "
            "cluster.batch_key to enable Harmony correction.",
        )

    if OBS_CLASS in obs.columns:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        _scatter_umap(ax, coords, obs[OBS_CLASS].astype(str).to_numpy(), True,
                      "Guide assignment class")
        fig.tight_layout()
        reg.save(
            fig,
            "umap_assignment_class",
            SECTION_CLUSTERING,
            "UMAP coloured by guide assignment",
            "Unassigned cells clustering together often indicates a technically "
            "distinct population (e.g. low-quality cells) rather than a biological one.",
        )

    if OBS_TARGET in obs.columns:
        targets = obs[OBS_TARGET].astype(str)
        n_targets = targets.nunique()
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        _scatter_umap(ax, coords, targets.to_numpy(), True,
                      f"Target gene ({n_targets} levels)", legend=n_targets <= 25)
        fig.tight_layout()
        reg.save(
            fig,
            "umap_target_gene",
            SECTION_CLUSTERING,
            "UMAP coloured by target gene",
            "Most Perturb-seq screens show perturbed cells mixed throughout the "
            "embedding; a target forming its own island has a strong phenotype."
            + ("" if n_targets <= 25 else " Legend omitted (too many targets)."),
        )

    if CLUSTER_KEY in obs.columns and LANE_KEY in obs.columns and obs[LANE_KEY].nunique() > 1:
        tab = (
            obs.groupby([CLUSTER_KEY, LANE_KEY], observed=True).size().unstack(fill_value=0)
        )
        frac = tab.div(tab.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(frac)), 3.8))
        bottom = np.zeros(len(frac))
        palette = sns.color_palette("tab20", frac.shape[1])
        for i, lane in enumerate(frac.columns):
            ax.bar(frac.index.astype(str), frac[lane], bottom=bottom, color=palette[i],
                   label=str(lane))
            bottom += frac[lane].to_numpy()
        ax.set_xlabel("Leiden cluster")
        ax.set_ylabel("% of cluster")
        ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title("Lane composition per cluster", fontsize=11)
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            "cluster_lane_composition",
            SECTION_CLUSTERING,
            "Lane composition per cluster",
            "Clusters made up almost entirely of one lane are candidates for batch "
            "correction.",
        )


# ---------------------------------------------------------------------------
# Perturbation figures
# ---------------------------------------------------------------------------


def plot_perturbation_overview(
    results: PerturbationResults, reg: FigureRegistry, cfg: Config
) -> None:
    """Volcano and waterfall summarizing knockdown across all targets."""
    if results.table.empty:
        return
    tbl = results.table
    primary = results.primary_control
    lfc = tbl[f"log2fc_{primary}"].to_numpy(dtype=float)
    fdr = tbl[f"ks_fdr_{primary}"].to_numpy(dtype=float)
    hit = tbl[f"is_hit_{primary}"].to_numpy(dtype=bool)
    names = tbl["target_gene"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    with np.errstate(divide="ignore"):
        y = -np.log10(np.clip(fdr, 1e-300, 1))
    ax.scatter(lfc[~hit], y[~hit], s=18, color="#a0aec0", label="not significant")
    ax.scatter(lfc[hit], y[hit], s=22, color="#c53030", label="effective knockdown")
    ax.axhline(-np.log10(cfg.perturbation.fdr_alpha), color="#718096", ls="--", lw=1)
    ax.axvline(cfg.perturbation.max_log2fc_for_hit, color="#718096", ls="--", lw=1)
    for i in np.argsort(lfc)[: min(12, len(lfc))]:
        ax.annotate(names[i], (lfc[i], y[i]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("log2 fold change (perturbed / control)")
    ax.set_ylabel("-log10 FDR (KS test)")
    ax.set_title(f"Perturbation strength — control: {CONTROL_LABELS[primary]}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "perturbation_volcano",
        SECTION_PERTURBATION,
        "Volcano of perturbation strength",
        "Each point is a target gene, comparing its own expression in perturbed vs "
        "control cells. Effective perturbations fall in the upper-left: significantly "
        "reduced expression.",
    )

    order = np.argsort(lfc)
    fig, ax = plt.subplots(figsize=(max(6, 0.18 * len(order)), 4.0))
    ax.bar(
        range(len(order)),
        lfc[order],
        color=["#c53030" if hit[i] else "#a0aec0" for i in order],
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names[order], rotation=90, fontsize=6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("log2 fold change")
    ax.set_title(
        f"Knockdown per target ({int(hit.sum())}/{len(hit)} effective at FDR < "
        f"{cfg.perturbation.fdr_alpha})",
        fontsize=10,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "perturbation_waterfall",
        SECTION_PERTURBATION,
        "Knockdown strength per target",
        "Targets sorted by fold change. Red bars are significant reductions; bars "
        "near zero mean the guide did not measurably reduce its target.",
    )

    if len(results.controls_used) > 1:
        other = CONTROL_OTHER if primary == CONTROL_NTC else CONTROL_NTC
        lfc2 = tbl[f"log2fc_{other}"].to_numpy(dtype=float)
        ok = ~(np.isnan(lfc) | np.isnan(lfc2))
        if ok.sum() > 2:
            fig, ax = plt.subplots(figsize=(4.8, 4.6))
            ax.scatter(lfc[ok], lfc2[ok], s=20, color="#2b6cb0", alpha=0.8)
            lim = [min(lfc[ok].min(), lfc2[ok].min()) - 0.1,
                   max(lfc[ok].max(), lfc2[ok].max()) + 0.1]
            ax.plot(lim, lim, ls="--", color="#718096", lw=1)
            r = float(np.corrcoef(lfc[ok], lfc2[ok])[0, 1])
            ax.set_xlabel(f"log2FC vs {CONTROL_LABELS[primary]}")
            ax.set_ylabel(f"log2FC vs {CONTROL_LABELS[other]}")
            ax.set_title(f"Control comparison (Pearson r = {r:.2f})", fontsize=10)
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig,
                "perturbation_control_comparison",
                SECTION_PERTURBATION,
                "Effect size under both control definitions",
                "Agreement between the two control groups. Points off the diagonal "
                "are targets whose apparent effect depends on the control used.",
            )


def plot_per_target(
    expr,
    results: PerturbationResults,
    reg: FigureRegistry,
    cfg: Config,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """One diagnostic figure per tested target gene.

    Every target gets a figure on disk; only the top-N strongest are marked for
    inline inclusion in the report (the rest are linked as extras).
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

    for i, row in results.table.iterrows():
        gene = str(row["target_gene"])
        values = _gene_values(expr, gene)
        pert = (targets_col == gene) & (klass == CLASS_TARGETING)

        groups = {"perturbed": values[pert]}
        for control in results.controls_used:
            if control == CONTROL_NTC:
                m = klass == CLASS_NTC
            else:
                m = (klass == CLASS_TARGETING) & (targets_col != gene)
            groups[CONTROL_LABELS[control]] = values[m]

        n_panels = 3 if coords is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.0))

        ax = axes[0]
        for label, vals in groups.items():
            if vals.size:
                sns.ecdfplot(x=vals, ax=ax, label=f"{label} (n={vals.size:,})")
        ax.set_xlabel(f"{gene} expression (log-normalized)")
        ax.set_ylabel("Cumulative fraction of cells")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        fdr = row.get(f"ks_fdr_{primary}", np.nan)
        lfc = row.get(f"log2fc_{primary}", np.nan)
        ax.set_title(f"{gene}: log2FC = {lfc:.2f}, FDR = {fdr:.2g}", fontsize=10)
        sns.despine(ax=ax)

        ax = axes[1]
        plot_df = pd.DataFrame(
            {
                "expression": np.concatenate([v for v in groups.values() if v.size]),
                "group": np.concatenate(
                    [[k] * v.size for k, v in groups.items() if v.size]
                ),
            }
        )
        sns.violinplot(data=plot_df, x="group", y="expression", ax=ax, cut=0, inner="box")
        ax.set_xlabel("")
        ax.set_ylabel(f"{gene} expression")
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        ax.set_title(f"{gene} expression by group", fontsize=10)
        sns.despine(ax=ax)

        if coords is not None:
            ax = axes[2]
            frac = cfg.perturbation.umap_background_fraction
            bg = rng.random(expr.n_obs) < frac
            keep = bg | pert
            ax.scatter(coords[keep & ~pert, 0], coords[keep & ~pert, 1], s=3,
                       color="#e2e8f0", linewidths=0, rasterized=True,
                       label=f"other cells ({int(frac * 100)}% shown)")
            # A thin outline keeps perturbed cells with zero expression visible;
            # without it the knocked-down cells — the point of the panel —
            # render white on a pale background.
            sc_ = ax.scatter(coords[pert, 0], coords[pert, 1], s=10, c=values[pert],
                             cmap="Reds", vmin=0, edgecolors="#2d3748",
                             linewidths=0.25, rasterized=True)
            plt.colorbar(sc_, ax=ax, shrink=0.75, label=f"{gene} expression")
            ax.set_title(f"Cells perturbed for {gene} (n={int(pert.sum()):,})", fontsize=10)
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
            f"Expression of {gene} in cells carrying {gene} guides versus control "
            f"cells. log2FC = {lfc:.2f}, KS FDR = {fdr:.2g}"
            + (" — effective knockdown." if is_hit else " — no significant reduction."),
            in_report=i < top_n,
        )
    logger.info(
        "Wrote %d per-target figures (%d shown in the report)",
        len(results.table),
        min(top_n, len(results.table)),
    )
