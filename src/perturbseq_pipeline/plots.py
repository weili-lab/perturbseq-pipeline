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
SECTION_ENRICHMENT = "enrichment"
SECTION_ENRICH_PER_TARGET = "enrichment/per_target"
SECTION_PS = "ps_score"
SECTION_PS_PER_TARGET = "ps_score/per_target"
SECTION_PS_LDA = "ps_score/lda"
SECTION_LOCHNESS = "lochness"
SECTION_LOCHNESS_PER_TARGET = "lochness/per_target"

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


# ---------------------------------------------------------------------------
# Cluster-enrichment figures
# ---------------------------------------------------------------------------


def _order_by_similarity(matrix: pd.DataFrame) -> List[str]:
    """Hierarchically order rows so similar profiles sit together."""
    if matrix.shape[0] < 3:
        return list(matrix.index)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        values = np.nan_to_num(matrix.to_numpy(dtype=float))
        dist = pdist(values, metric="correlation")
        if not np.all(np.isfinite(dist)):
            return list(matrix.index)
        return [matrix.index[i] for i in leaves_list(linkage(dist, method="average"))]
    except Exception:  # pragma: no cover - ordering is cosmetic
        return list(matrix.index)


def plot_enrichment(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Heatmap, phenocopy map, composition bars, volcano and ranking."""
    from .enrichment import enrichment_matrix, phenocopy_similarity, significance_matrix

    if results.table.empty:
        return
    ecfg = cfg.enrichment
    lor = enrichment_matrix(results)
    fdr = significance_matrix(results)
    if lor.empty:
        return
    order = _order_by_similarity(lor)
    lor = lor.loc[order]
    fdr = fdr.loc[order]

    # --- 1. main heatmap -------------------------------------------------
    lim = float(np.nanpercentile(np.abs(lor.to_numpy()), 98)) or 1.0
    height = max(4.0, 0.20 * len(lor) + 1.5)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * lor.shape[1] + 4), height))
    im = ax.imshow(lor.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(lor.shape[1]))
    ax.set_xticklabels(lor.columns, fontsize=8)
    ax.set_yticks(range(lor.shape[0]))
    ax.set_yticklabels(lor.index, fontsize=6)
    ax.set_xlabel(f"Cluster ({results.cluster_key})")
    n_marked = 0
    for i in range(lor.shape[0]):
        for j in range(lor.shape[1]):
            if fdr.iat[i, j] is not None and fdr.iat[i, j] < ecfg.fdr_alpha:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9, color="black")
                n_marked += 1
    plt.colorbar(im, ax=ax, shrink=0.6, label="log2 odds ratio")
    ax.set_title(
        f"Perturbation enrichment across clusters\n"
        f"* FDR < {ecfg.fdr_alpha} vs {CONTROL_LABELS[results.primary_control]}",
        fontsize=10,
    )
    fig.tight_layout()
    reg.save(
        fig,
        "enrichment_heatmap",
        SECTION_ENRICHMENT,
        "Perturbation enrichment across clusters",
        "Red means cells carrying that perturbation are over-represented in the "
        "cluster, blue under-represented; asterisks mark significant pairs. Rows "
        "are ordered by profile similarity, so perturbations with the same "
        f"phenotype sit together ({n_marked} significant pair(s)).",
    )

    # --- 2. phenocopy similarity ----------------------------------------
    sim = phenocopy_similarity(results)
    if not sim.empty and sim.shape[0] >= 3:
        sim_order = _order_by_similarity(sim)
        sim = sim.loc[sim_order, sim_order]
        size = max(5.0, 0.16 * len(sim) + 2)
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(sim.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(sim)))
        ax.set_xticklabels(sim.index, rotation=90, fontsize=5)
        ax.set_yticks(range(len(sim)))
        ax.set_yticklabels(sim.index, fontsize=5)
        plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
        ax.set_title("Do perturbations phenocopy each other?", fontsize=11)
        fig.tight_layout()
        reg.save(
            fig,
            "enrichment_phenocopy",
            SECTION_ENRICHMENT,
            "Perturbation similarity (phenocopy map)",
            "Correlation between targets of their cluster-composition profiles. "
            "Red blocks are groups of perturbations producing the same cell-state "
            "shift — subunits of one complex are expected to land together, which "
            "is a check that needs no prior knowledge of the complexes.",
        )

    # --- 3. composition stacked bars ------------------------------------
    comp = results.composition
    mag = results.effect_magnitude
    top = list(mag["target_gene"].head(30))
    ref = results.reference_composition[results.primary_control]
    plot_df = pd.concat([ref.to_frame("REFERENCE").T, comp.loc[top]])
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(plot_df) + 2), 4.4))
    bottom = np.zeros(len(plot_df))
    palette = sns.color_palette("tab20", plot_df.shape[1])
    for i, cl in enumerate(plot_df.columns):
        ax.bar(range(len(plot_df)), plot_df[cl], bottom=bottom, color=palette[i], label=str(cl))
        bottom += plot_df[cl].to_numpy()
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df.index, rotation=90, fontsize=6)
    ax.set_ylabel("% of cells")
    ax.set_xlim(-0.6, len(plot_df) - 0.4)
    ax.axvline(0.5, color="black", lw=1.2)
    ax.legend(title="cluster", fontsize=6, title_fontsize=7, frameon=False,
              bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)
    ax.set_title("Cluster composition per perturbation (top 30 by shift)", fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "enrichment_composition",
        SECTION_ENRICHMENT,
        "Cluster composition per perturbation",
        "Each bar is one perturbation's distribution across clusters; the leftmost "
        "bar (left of the black line) is the reference. Perturbations are sorted "
        "by how far their composition sits from it.",
    )

    # --- 4. volcano ------------------------------------------------------
    sub = results.table[results.table["control"] == results.primary_control]
    x = sub["log2_odds_ratio"].to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        y = -np.log10(np.clip(sub["fdr"].to_numpy(dtype=float), 1e-300, 1))
    sig = sub["significant"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(x[~sig], y[~sig], s=12, color="#a0aec0", label="not significant")
    ax.scatter(x[sig], y[sig], s=20, color="#c53030", label=f"FDR < {ecfg.fdr_alpha}")
    ax.axhline(-np.log10(ecfg.fdr_alpha), color="#718096", ls="--", lw=1)
    ax.axvline(0, color="#718096", ls="--", lw=1)
    labelled = sub[sig].reindex(
        sub[sig]["log2_odds_ratio"].abs().sort_values(ascending=False).index
    ).head(10)
    for _, row in labelled.iterrows():
        ax.annotate(
            f"{row['target_gene']}:{row['cluster']}",
            (row["log2_odds_ratio"], -np.log10(max(row["fdr"], 1e-300))),
            fontsize=6, xytext=(3, 3), textcoords="offset points",
        )
    ax.set_xlabel("log2 odds ratio (enriched >0, depleted <0)")
    ax.set_ylabel("-log10 FDR")
    ax.set_title("Enrichment across all target x cluster pairs", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "enrichment_volcano",
        SECTION_ENRICHMENT,
        "Enrichment volcano",
        "Every target/cluster pair. Points to the right are perturbations that "
        "accumulate in a cluster; to the left, ones depleted from it.",
    )

    # --- 5. effect magnitude ranking -------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 0.16 * len(mag)), 3.8))
    colors = ["#c53030" if n > 0 else "#a0aec0" for n in mag["n_significant_clusters"]]
    ax.bar(range(len(mag)), mag["composition_shift_pct"], color=colors)
    ax.set_xticks(range(len(mag)))
    ax.set_xticklabels(mag["target_gene"], rotation=90, fontsize=6)
    ax.set_ylabel("Composition shift (%)")
    ax.set_title(
        "How far each perturbation moves cells between clusters "
        "(red = has a significant cluster)",
        fontsize=10,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "enrichment_effect_magnitude",
        SECTION_ENRICHMENT,
        "Composition shift per perturbation",
        "Total variation distance between each perturbation's cluster composition "
        "and the reference: 0% means indistinguishable, 100% means the cells sit "
        "in entirely different clusters.",
    )


def plot_enrichment_per_target(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """One composition + UMAP figure per target; top-N marked for the report."""
    from .enrichment import OBS_TARGET as _T  # noqa: F401  (kept explicit below)

    if results.table.empty:
        return
    obs = expr.obs
    cluster_key = results.cluster_key
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    clusters_col = obs[cluster_key].astype(str).to_numpy()
    coords = np.asarray(expr.obsm["X_umap"]) if "X_umap" in expr.obsm else None
    ref = results.reference_composition[results.primary_control]
    clusters = list(results.composition.columns)

    # Report the strongest hits first; every target still gets a file.
    ranked = list(results.effect_magnitude["target_gene"])
    hit_targets = results.targets_with_hits()
    ordered = [t for t in ranked if t in hit_targets] + [
        t for t in ranked if t not in hit_targets
    ]
    top_n = cfg.enrichment.top_n_report

    sub_tbl = results.table[results.table["control"] == results.primary_control]
    for rank, gene in enumerate(ordered):
        if gene not in results.composition.index:
            continue
        comp = results.composition.loc[gene]
        rows = sub_tbl[sub_tbl["target_gene"] == gene].set_index("cluster")

        n_panels = 2 if coords is None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(4.7 * n_panels, 4.0))

        ax = axes[0]
        idx = np.arange(len(clusters))
        ax.bar(idx - 0.2, ref[clusters], width=0.4, color="#a0aec0", label="reference")
        ax.bar(idx + 0.2, comp[clusters], width=0.4, color="#2b6cb0", label=gene)
        for i, cl in enumerate(clusters):
            if cl in rows.index and bool(rows.at[cl, "significant"]):
                ax.text(i, max(comp[cl], ref[cl]) + 1, "*", ha="center", fontsize=11)
        ax.set_xticks(idx)
        ax.set_xticklabels(clusters, fontsize=7)
        ax.set_xlabel(f"Cluster ({cluster_key})")
        ax.set_ylabel("% of cells")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{gene} cluster composition", fontsize=10)
        sns.despine(ax=ax)

        ax = axes[1]
        lor = rows["log2_odds_ratio"].reindex(clusters)
        sig = rows["significant"].reindex(clusters).fillna(False).to_numpy(dtype=bool)
        ax.bar(idx, lor.to_numpy(dtype=float),
               color=["#c53030" if s else "#a0aec0" for s in sig])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(idx)
        ax.set_xticklabels(clusters, fontsize=7)
        ax.set_xlabel(f"Cluster ({cluster_key})")
        ax.set_ylabel("log2 odds ratio")
        ax.set_title(f"{gene} enrichment (red = FDR < {cfg.enrichment.fdr_alpha})", fontsize=10)
        sns.despine(ax=ax)

        if coords is not None:
            ax = axes[2]
            pert = (targets_col == gene) & (klass == CLASS_TARGETING)
            ax.scatter(coords[~pert, 0], coords[~pert, 1], s=2, color="#e2e8f0",
                       linewidths=0, rasterized=True)
            cl_of = clusters_col[pert]
            palette = sns.color_palette("tab20", len(clusters))
            cmap = {c: palette[i] for i, c in enumerate(clusters)}
            ax.scatter(coords[pert, 0], coords[pert, 1], s=10,
                       c=[cmap.get(c, (0.3, 0.3, 0.3)) for c in cl_of],
                       linewidths=0.2, edgecolors="#2d3748", rasterized=True)
            ax.set_title(f"{gene} cells by cluster (n={int(pert.sum()):,})", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)

        best = rows["log2_odds_ratio"].abs().idxmax() if len(rows) else None
        caption = f"Cluster distribution of cells perturbed for {gene}."
        if best is not None and cl_has_hit(rows, best):
            caption += (
                f" Strongest association: cluster {best} "
                f"({rows.at[best, 'pct_of_target']:.1f}% of {gene} cells vs "
                f"{rows.at[best, 'pct_of_reference']:.1f}% of reference, "
                f"FDR = {rows.at[best, 'fdr']:.2g})."
            )
        fig.tight_layout()
        reg.save(
            fig,
            f"enrichment_{gene}",
            SECTION_ENRICH_PER_TARGET,
            f"{gene} cluster enrichment",
            caption,
            in_report=rank < top_n,
        )
    logger.info(
        "Wrote %d per-target enrichment figures (%d shown in the report)",
        len(ordered),
        min(top_n, len(ordered)),
    )


def cl_has_hit(rows: pd.DataFrame, cluster) -> bool:
    """True when this target/cluster pair reached significance."""
    try:
        return bool(rows.at[cluster, "significant"])
    except (KeyError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Perturbation-score figures (pertps / PS_python)
# ---------------------------------------------------------------------------


def plot_ps_scores(expr, results, perturbation_results, reg: FigureRegistry, cfg: Config) -> None:
    """Quadrant scatters, a knockdown-efficiency ranking and a method check."""
    from .ps_score import (
        QUADRANT_COLORS,
        QUADRANT_ESCAPER,
        QUADRANT_KD,
        compare_with_perturbation_strength,
    )

    if results is None or results.summary.empty:
        return
    summary = results.summary

    # --- 1. knockdown efficiency per target ------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 0.20 * len(summary)), 4.0))
    bottom = np.zeros(len(summary))
    for key, col in [
        ("pct_successful_kd", QUADRANT_KD),
        ("pct_escaper", QUADRANT_ESCAPER),
        ("pct_non_responder", "non-responder"),
        ("pct_low_signal", "low signal"),
    ]:
        ax.bar(range(len(summary)), summary[key], bottom=bottom,
               color=QUADRANT_COLORS.get(col, "#a0aec0"), label=col)
        bottom += summary[key].to_numpy()
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["target_gene"], rotation=90, fontsize=6)
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
        "Each perturbed cell is classified by its perturbation score and the "
        "target's own expression. Green is a confirmed knockdown; red are "
        "escapers, which carry the guide and show the signature yet still "
        "express the gene.",
    )

    # --- 2. escaper fraction ---------------------------------------------
    esc = summary.sort_values("pct_escaper", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.20 * len(esc)), 3.6))
    ax.bar(range(len(esc)), esc["pct_escaper"], color="#c53030")
    ax.set_xticks(range(len(esc)))
    ax.set_xticklabels(esc["target_gene"], rotation=90, fontsize=6)
    ax.set_ylabel("% escapers")
    ax.set_title("Escaper fraction per target", fontsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "ps_escaper_fraction",
        SECTION_PS,
        "Escaper fraction per target",
        "Cells carrying a guide whose target is nonetheless still expressed. A "
        "high fraction means the population-level effect understates how well "
        "the guide works in the cells where it does work.",
    )

    # --- 3. agreement with the group-level test --------------------------
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
            hit = merged.loc[ok].get(f"is_hit_{perturbation_results.primary_control}")
            colors = (
                ["#c53030" if h else "#a0aec0" for h in hit]
                if hit is not None
                else "#2b6cb0"
            )
            ax.scatter(x, y, s=28, c=colors)
            for xi, yi, name in zip(x, y, merged.loc[ok, "target_gene"]):
                if yi > np.percentile(y, 85) or xi < np.percentile(x, 15):
                    ax.annotate(name, (xi, yi), fontsize=6,
                                xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("log2FC of the target's own expression (group-level test)")
            ax.set_ylabel("% cells with confirmed knockdown (per-cell score)")
            ax.set_title(f"Per-cell score vs group-level knockdown (Pearson r = {r:.2f})", fontsize=10)
            sns.despine(ax=ax)
            fig.tight_layout()
            reg.save(
                fig,
                "ps_vs_perturbation_strength",
                SECTION_PS,
                "Per-cell scores vs the group-level test",
                "The two axes measure different things: the group-level test uses "
                "the target's own expression, while the per-cell score projects "
                "cells onto the perturbation's whole downstream signature. A gene "
                "can be strongly knocked down yet change little downstream, or the "
                "reverse, so these need not track each other closely — the "
                "correlation in the title is what these data actually show, not a "
                "quantity expected to be large.",
            )

    _plot_ps_quadrants(expr, results, reg, cfg)


def _plot_ps_quadrants(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Score-vs-expression quadrant scatter, one per scored target."""
    from scipy import sparse

    from .ps_score import (
        QUADRANT_COLORS,
        QUADRANT_ESCAPER,
        QUADRANT_KD,
        QUADRANT_LOW,
        QUADRANT_NONRESPONDER,
    )

    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    targets = expr.obs[OBS_TARGET].astype(str)
    klass = expr.obs[OBS_CLASS].astype(str)
    top_n = cfg.ps_score.top_n_report
    rng = np.random.default_rng(cfg.run.seed)

    for rank, gene in enumerate(results.summary["target_gene"]):
        series = results.scores.get(gene)
        if series is None or gene not in expr.var_names:
            continue
        col = layer[:, expr.var_names.get_loc(gene)]
        if sparse.issparse(col):
            col = col.toarray()
        expression = pd.Series(np.asarray(col).ravel(), index=expr.obs_names)

        cells = series.index.intersection(expr.obs_names)
        ps = series.loc[cells]
        ex = expression.loc[cells]
        is_target = (targets.loc[cells] == gene) & (klass.loc[cells] == CLASS_TARGETING)
        cut = results.expression_cut.get(gene, float(np.median(ex)))
        thr = results.ps_threshold

        fig, ax = plt.subplots(figsize=(6.4, 5.0))

        ctrl_idx = cells[~is_target.to_numpy()]
        if len(ctrl_idx) > 2000:
            ctrl_idx = rng.choice(ctrl_idx, size=2000, replace=False)
        ax.scatter(ps.loc[ctrl_idx], ex.loc[ctrl_idx], s=14, c="#cbd5e0", alpha=0.45,
                   linewidths=0, label="control cells", rasterized=True)

        tgt = cells[is_target.to_numpy()]
        quad = results.quadrants.get(gene)
        colors = (
            [QUADRANT_COLORS.get(str(quad.get(c, QUADRANT_LOW)), "#a0aec0") for c in tgt]
            if quad is not None
            else "#e53e3e"
        )
        ax.scatter(ps.loc[tgt], ex.loc[tgt], s=26, c=colors, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=f"{gene} cells",
                   rasterized=True)

        ax.axvline(thr, color="black", ls="--", lw=1, alpha=0.6)
        ax.axhline(cut, color="black", ls="--", lw=1, alpha=0.6)

        row = results.summary[results.summary["target_gene"] == gene].iloc[0]
        xmax = float(max(ps.max(), thr * 2))
        ymax = float(max(ex.max(), cut * 2)) or 1.0
        ax.text(thr + (xmax - thr) * 0.5, ymax * 0.95,
                f"ESCAPERS\n{row['pct_escaper']:.0f}%", fontsize=8, ha="center",
                color=QUADRANT_COLORS[QUADRANT_ESCAPER], fontweight="bold")
        ax.text(thr + (xmax - thr) * 0.5, ymax * 0.05,
                f"KNOCKED DOWN\n{row['pct_successful_kd']:.0f}%", fontsize=8, ha="center",
                color=QUADRANT_COLORS[QUADRANT_KD], fontweight="bold")
        ax.text(thr * 0.5, ymax * 0.95, f"NON-RESPONDER\n{row['pct_non_responder']:.0f}%",
                fontsize=8, ha="center", color=QUADRANT_COLORS[QUADRANT_NONRESPONDER],
                fontweight="bold")
        ax.text(thr * 0.5, ymax * 0.05, f"LOW SIGNAL\n{row['pct_low_signal']:.0f}%",
                fontsize=8, ha="center", color="#718096", fontweight="bold")

        ax.set_xlabel("Perturbation score (per cell)")
        ax.set_ylabel(f"{gene} expression (log-normalized)")
        ax.set_title(f"{gene}: per-cell perturbation outcome "
                     f"(n={int(row['n_perturbed_cells']):,})", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        sns.despine(ax=ax)
        fig.tight_layout()
        reg.save(
            fig,
            f"ps_quadrant_{gene}",
            SECTION_PS_PER_TARGET,
            f"{gene} perturbation score vs expression",
            f"Cells carrying {gene} guides, split by perturbation score "
            f"(vertical cut at {thr}) and by {gene} expression relative to the "
            f"control median (horizontal cut). {row['pct_successful_kd']:.0f}% "
            f"show a confirmed knockdown and {row['pct_escaper']:.0f}% escape it.",
            in_report=rank < top_n,
        )
    logger.info(
        "Wrote %d perturbation-score quadrant figures (%d shown in the report)",
        len(results.summary),
        min(top_n, len(results.summary)),
    )


# ---------------------------------------------------------------------------
# Supervised LDA embedding (PS_python's fixed-LDA map)
# ---------------------------------------------------------------------------


def plot_ps_lda(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """PS scores on the supervised LDA embedding.

    The section-2 UMAP is unsupervised and knows nothing about which guide a
    cell carries, so a subtle phenotype can be invisible in it. This embedding
    is trained on the perturbation labels, so its axes are chosen to separate
    perturbations — the space in which the per-cell scores read most clearly.

    One overview plus one figure per scored target, mirroring PS_python's
    ``plots_fixed_lda/``.
    """
    from .ps_score import PS_PREFIX, QUADRANT_KD

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
    targets = expr.obs[OBS_TARGET].astype(str).to_numpy()
    n_targets = len(results.summary)

    # --- overview: the whole map, coloured by perturbation ----------------
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    _scatter_umap(
        ax,
        coords[placed],
        labels[placed],
        True,
        f"Supervised LDA embedding ({n_targets} targets + control)",
        size=4,
        legend=n_targets <= 24,
    )
    ax.set_xlabel("LDA-UMAP1", fontsize=8)
    ax.set_ylabel("LDA-UMAP2", fontsize=8)
    fig.tight_layout()
    reg.save(
        fig,
        "ps_lda_overview",
        SECTION_PS,
        "Supervised LDA embedding",
        "Linear discriminant analysis trained on the perturbation labels, then "
        "embedded with UMAP. Unlike the unsupervised embedding in section 2, the "
        "axes here are chosen to separate perturbations, so groups that overlap "
        "there can resolve."
        + ("" if n_targets <= 24 else " Legend omitted (too many targets)."),
    )

    # --- global summary: high-confidence knockdown cells -------------------
    own = expr.obs["ps_score"].to_numpy(dtype=float) if "ps_score" in expr.obs else None
    if own is not None:
        thr = cfg.ps_score.lda_highlight_threshold
        strong = placed & np.isfinite(own) & (own >= thr)
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        ax.scatter(coords[placed, 0], coords[placed, 1], s=4, color="#e2e8f0",
                   linewidths=0, rasterized=True, label="all cells")
        sc_ = ax.scatter(coords[strong, 0], coords[strong, 1], s=12, c=own[strong],
                         cmap="viridis", vmin=thr, vmax=1.0, linewidths=0.2,
                         edgecolors="#2d3748", rasterized=True)
        plt.colorbar(sc_, ax=ax, shrink=0.75, label="perturbation score")
        ax.set_title(
            f"High-confidence responders (score >= {thr}): "
            f"{int(strong.sum()):,} of {int(placed.sum()):,} cells",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False, loc="best", markerscale=3)
        sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        reg.save(
            fig,
            "ps_lda_high_confidence",
            SECTION_PS,
            "High-confidence responders on the LDA map",
            f"Cells whose perturbation score reaches {thr}, coloured by score. "
            "Where these concentrate is where the screen produced its clearest "
            "phenotypes.",
        )

    # --- one per target ----------------------------------------------------
    top_n = cfg.ps_score.top_n_report
    for rank, gene in enumerate(results.summary["target_gene"]):
        col = f"{PS_PREFIX}{gene}"
        if col not in expr.obs.columns:
            continue
        score = expr.obs[col].to_numpy(dtype=float)
        is_target = (targets == gene) & placed
        if is_target.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        bg = placed & ~is_target
        ax.scatter(coords[bg, 0], coords[bg, 1], s=4, color="#e2e8f0", alpha=0.6,
                   linewidths=0, rasterized=True, label="other cells")
        order = np.argsort(np.nan_to_num(score[is_target]))
        idx = np.where(is_target)[0][order]
        sc_ = ax.scatter(coords[idx, 0], coords[idx, 1], s=18,
                         c=np.nan_to_num(score[idx]), cmap="Blues", vmin=0, vmax=1,
                         linewidths=0.3, edgecolors="#2d3748", rasterized=True)
        plt.colorbar(sc_, ax=ax, shrink=0.75, label="perturbation score")

        row = results.summary[results.summary["target_gene"] == gene].iloc[0]
        ax.set_title(
            f"{gene} on the LDA map (n={int(row['n_perturbed_cells']):,}, "
            f"{row['pct_successful_kd']:.0f}% knocked down)",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False, loc="best", markerscale=3)
        sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        reg.save(
            fig,
            f"ps_lda_{gene}",
            SECTION_PS_LDA,
            f"{gene} on the LDA embedding",
            f"Cells carrying {gene} guides, shaded by perturbation score, against "
            "all other cells in grey. Darker cells respond more strongly; a tight "
            "darker cluster means the perturbation drives a consistent state.",
            in_report=rank < top_n,
        )
    logger.info(
        "Wrote %d per-target LDA figures (%d shown in the report)",
        n_targets,
        min(top_n, n_targets),
    )


# ---------------------------------------------------------------------------
# lochNESS figures
# ---------------------------------------------------------------------------


def _lochness_norm(vmax: float):
    """Colour scale for lochNESS: linear near 0, logarithmic in the tail.

    The score is bounded below by -1 and unbounded above, so a symmetric linear
    scale both wastes half its range and clips the extreme cells that matter.
    """
    import matplotlib.colors as mcolors

    vmax = max(float(vmax), 1.0)
    return mcolors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)


def _label_clusters(ax, expr, coords: np.ndarray, highlight: str = "") -> None:
    """Write cluster labels at their centroids on an embedding panel."""
    if CLUSTER_KEY not in expr.obs.columns:
        return
    clusters = expr.obs[CLUSTER_KEY].astype(str).to_numpy()
    for cl in np.unique(clusters):
        m = clusters == cl
        if m.sum() == 0:
            continue
        cx, cy = np.median(coords[m, 0]), np.median(coords[m, 1])
        is_top = str(cl) == str(highlight)
        ax.text(
            cx, cy, str(cl),
            fontsize=8 if is_top else 6.5,
            fontweight="bold" if is_top else "normal",
            color="#1a202c" if is_top else "#4a5568",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#fefcbf" if is_top else "white",
                edgecolor="none",
                alpha=0.85 if is_top else 0.6,
            ),
        )


def plot_lochness(expr, results, reg: FigureRegistry, cfg: Config) -> None:
    """Overview figures plus one per-perturbation lochNESS map."""
    if results is None or results.summary.empty:
        return
    lcfg = cfg.lochness
    summary = results.summary
    coords = np.asarray(expr.obsm["X_umap"]) if "X_umap" in expr.obsm else None

    # --- 1. self-enrichment ranking ---------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 0.20 * len(summary)), 4.0))
    vals = summary["mean_lochness_in_own_cells"].to_numpy(dtype=float)
    ax.bar(range(len(summary)), vals,
           color=["#c53030" if v > lcfg.enrichment_cut else "#a0aec0" for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(lcfg.enrichment_cut, color="#718096", ls="--", lw=1)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["target_gene"], rotation=90, fontsize=6)
    ax.set_ylabel("mean lochNESS in its own cells")
    ax.set_title(
        f"How strongly each perturbation clusters with itself "
        f"(k = {results.n_neighbors} neighbours)",
        fontsize=10,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    reg.save(
        fig,
        "lochness_self_enrichment",
        SECTION_LOCHNESS,
        "Self-enrichment per perturbation",
        "Average lochNESS of each perturbation's own cells. 0 means its cells sit "
        "among neighbours at the background rate; a positive value means cells "
        "sharing the perturbation are neighbours far more often than chance, i.e. "
        "the perturbation drives a distinct state.",
    )

    # --- 2. score distribution per target ---------------------------------
    top = list(summary["target_gene"].head(30))
    long = pd.DataFrame(
        {
            "lochNESS": np.concatenate([results.scores[g] for g in top]),
            "target": np.concatenate([[g] * expr.n_obs for g in top]),
        }
    )
    fig, ax = plt.subplots(figsize=(max(6, 0.34 * len(top)), 4.2))
    sns.violinplot(data=long, x="target", y="lochNESS", ax=ax, cut=0,
                   inner=None, linewidth=0.5, order=top)
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
        "Each violin is one perturbation's score across every cell. A long upper "
        "tail means a subset of the manifold is strongly enriched for it, even "
        "when most cells sit at background.",
    )

    # --- 3. target x cluster heatmap --------------------------------------
    if not results.by_cluster.empty:
        mat = results.by_cluster
        try:
            mat = mat[sorted(mat.columns, key=lambda c: (float(c), c))]
        except ValueError:
            mat = mat[sorted(mat.columns)]
        order = _order_by_similarity(mat)
        mat = mat.loc[order]
        lim = float(np.nanpercentile(np.abs(mat.to_numpy()), 98)) or 1.0
        fig, ax = plt.subplots(
            figsize=(max(6, 0.55 * mat.shape[1] + 4), max(4, 0.20 * len(mat) + 1.5))
        )
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, fontsize=8)
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels(mat.index, fontsize=6)
        ax.set_xlabel(f"Cluster ({CLUSTER_KEY})")
        plt.colorbar(im, ax=ax, shrink=0.6, label="mean lochNESS")
        ax.set_title("Mean lochNESS per cluster", fontsize=11)
        fig.tight_layout()
        reg.save(
            fig,
            "lochness_by_cluster",
            SECTION_LOCHNESS,
            "lochNESS by cluster",
            "Average score of each perturbation within each cluster, rows ordered "
            "by similarity. This is the continuous counterpart of the enrichment "
            "test in section 4 — agreement between the two is a good sign, and "
            "structure here that the cluster test missed is worth a look.",
        )

    # --- 4. self score on the embedding -----------------------------------
    if coords is not None and results.self_score is not None:
        vals = np.asarray(results.self_score, dtype=float)
        ok = np.isfinite(vals)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.scatter(coords[~ok, 0], coords[~ok, 1], s=3, color="#edf2f7",
                   linewidths=0, rasterized=True, label="unassigned / ambiguous")
        lim = float(np.nanpercentile(np.abs(vals[ok]), 98)) or 1.0
        sc_ = ax.scatter(coords[ok, 0], coords[ok, 1], s=5, c=vals[ok],
                         cmap="RdBu_r", vmin=-lim, vmax=lim, linewidths=0,
                         rasterized=True)
        plt.colorbar(sc_, ax=ax, shrink=0.75, label="lochNESS (own perturbation)")
        ax.set_title("Each cell scored for its own perturbation", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False, loc="best", markerscale=3)
        sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        reg.save(
            fig,
            "lochness_self_umap",
            SECTION_LOCHNESS,
            "Self lochNESS on the embedding",
            "Red regions are where cells sit among others sharing their own "
            "perturbation more often than chance — the parts of the manifold that "
            "perturbation identity actually organises.",
        )

    # --- 5. one map per perturbation --------------------------------------
    if coords is None:
        return
    top_n = lcfg.top_n_report
    for rank, gene in enumerate(summary["target_gene"]):
        score = np.asarray(results.scores[gene], dtype=float)
        ok = np.isfinite(score)
        own = expr.obs[cfg.lochness.genotype_key].astype(str).to_numpy() == gene

        row = summary[summary["target_gene"] == gene].iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

        ax = axes[0]
        srt = np.argsort(np.nan_to_num(score))
        # lochNESS is bounded below by -1 (a neighbourhood with none of the
        # perturbation) but unbounded above, and the informative values are a
        # long thin tail: on the demo lane SALL4 runs to +47 while the 99th
        # percentile is 4.5. A symmetric linear scale would therefore saturate
        # the very cells the figure exists to show — half of SALL4's own cells
        # sat above the cap. A symlog scale keeps the whole range visible.
        vmax = float(np.nanmax(score[ok])) if ok.any() else 1.0
        norm = _lochness_norm(vmax)
        sc_ = ax.scatter(coords[srt, 0], coords[srt, 1], s=4,
                         c=np.nan_to_num(score[srt]), cmap="RdBu_r",
                         norm=norm, linewidths=0, rasterized=True)
        cbar = plt.colorbar(sc_, ax=ax, shrink=0.78, label="lochNESS (symlog)")
        cbar.ax.tick_params(labelsize=7)
        ax.set_title(f"{gene}: neighbourhood enrichment", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

        ax = axes[1]
        ax.scatter(coords[~own, 0], coords[~own, 1], s=3, color="#e2e8f0",
                   linewidths=0, rasterized=True, label="other cells")
        ax.scatter(coords[own, 0], coords[own, 1], s=14, color="#c53030",
                   linewidths=0.3, edgecolors="#2d3748", rasterized=True,
                   label=f"{gene} cells")
        # Label the clusters, so a reader can tell which blob the summary table
        # means by "top cluster" instead of having to guess.
        _label_clusters(ax, expr, coords, highlight=str(row.get("top_cluster", "")))
        ax.set_title(
            f"where the {int(own.sum()):,} {gene} cells actually are "
            f"(top cluster {row.get('top_cluster', '?')} highlighted)",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False, loc="best", markerscale=2)
        sns.despine(ax=ax, left=True, bottom=True)

        fig.tight_layout()
        reg.save(
            fig,
            f"lochness_{gene}",
            SECTION_LOCHNESS_PER_TARGET,
            f"{gene} lochNESS map",
            f"Left: every cell scored for how enriched {gene} is among its "
            f"neighbours (red = enriched, blue = depleted). Right: the cells "
            f"actually carrying {gene}, for comparison. Mean score in its own "
            f"cells {row['mean_lochness_in_own_cells']:.2f}; "
            f"{row['pct_cells_enriched']:.1f}% of all cells score above "
            f"{lcfg.enrichment_cut}.",
            in_report=rank < top_n,
        )
    logger.info(
        "Wrote %d per-perturbation lochNESS maps (%d shown in the report)",
        len(summary),
        min(top_n, len(summary)),
    )
