"""HTML report assembly.

Everything the pipeline computed — tables, figures, warnings, the resolved
config — is collected into one self-contained HTML file. Figures are embedded as
base64 data URIs by default so the report can be emailed or dropped in Drive
without dragging a folder of PNGs along.
"""

from __future__ import annotations

import datetime as _dt
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from . import __version__
from .config import Config
from .perturbation import CONTROL_LABELS, PerturbationResults
from .plots import (
    SECTION_CLUSTERING,
    SECTION_ENRICH_PER_TARGET,
    SECTION_ENRICHMENT,
    SECTION_GUIDES,
    SECTION_PER_GENE,
    SECTION_PERTURBATION,
    SECTION_LOCHNESS,
    SECTION_LOCHNESS_PER_TARGET,
    SECTION_MODULES,
    SECTION_PS,
    SECTION_PS_LDA,
    SECTION_PS_PER_TARGET,
    SECTION_QC,
    SECTION_DISTANCE,
    SECTION_DISTANCE_SPACE,
    FigureRecord,
    FigureRegistry,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class ReportInputs:
    """Everything :func:`build_report` needs, gathered by the CLI."""

    cfg: Config
    registry: FigureRegistry
    perturbation: PerturbationResults
    enrichment: object = None
    modules: object = None
    ps: object = None
    lochness: object = None
    distance: object = None
    distance_space: object = None
    meta_table: Optional[pd.DataFrame] = None
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    summary_cards: List[tuple] = field(default_factory=list)
    input_mode: str = ""
    input_source: str = ""
    lanes: str = ""
    metadata_source: str = ""
    guide_source_text: str = ""
    outputs: Dict[str, str] = field(default_factory=dict)
    #: One row per stage: module, status (completed / skipped / disabled), note.
    module_status: Optional[pd.DataFrame] = None
    #: ``(label, value)`` pairs from the run manifest (git commit, command, seed, ...).
    provenance_rows: List[tuple] = field(default_factory=list)


def _df_to_html(df: Optional[pd.DataFrame], max_rows: int = 200) -> str:
    """Render a DataFrame as an HTML table, or a placeholder when empty."""
    if df is None or len(df) == 0:
        return Markup('<p class="sub">Not available for this run.</p>')
    shown = df.head(max_rows)
    html = shown.to_html(index=False, escape=True, border=0, na_rep="")
    if len(df) > max_rows:
        html += f'<p class="sub">Showing {max_rows} of {len(df)} rows; the full table is in <code>tables/</code>.</p>'
    return Markup(html)


def _render_figure(fig: FigureRecord, embed: bool) -> Markup:
    src = fig.data_uri() if embed else fig.path.name
    return Markup(
        f'<figure><img src="{src}" alt="{fig.title}">'
        f"<figcaption><b>{fig.title}.</b> {fig.caption}</figcaption></figure>"
    )


def _versions() -> str:
    lines = [f"python           {platform.python_version()}", f"platform         {platform.platform()}"]
    for mod in ("scanpy", "anndata", "numpy", "pandas", "scipy", "matplotlib", "seaborn"):
        try:
            import importlib.metadata as md

            lines.append(f"{mod:16s} {md.version(mod)}")
        except Exception:  # pragma: no cover - version lookup is best-effort
            lines.append(f"{mod:16s} (not found)")
    return "\n".join(lines)


def build_report(inputs: ReportInputs, path: Path) -> Path:
    """Render the HTML report to ``path``."""
    cfg = inputs.cfg
    reg = inputs.registry
    res = inputs.perturbation
    embed = cfg.report.embed_figures
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    template = env.get_template("report.html")
    figures = {
        "qc": reg.by_section(SECTION_QC),
        "guides": reg.by_section(SECTION_GUIDES),
        "clustering": reg.by_section(SECTION_CLUSTERING),
        "perturbation": reg.by_section(SECTION_PERTURBATION),
        "per_gene": reg.by_section(SECTION_PER_GENE),
        "enrichment": reg.by_section(SECTION_ENRICHMENT),
        "enrichment_per_target": reg.by_section(SECTION_ENRICH_PER_TARGET),
        "ps": reg.by_section(SECTION_PS),
        "ps_per_target": reg.by_section(SECTION_PS_PER_TARGET),
        "ps_lda": reg.by_section(SECTION_PS_LDA),
        "lochness": reg.by_section(SECTION_LOCHNESS),
        "lochness_per_target": reg.by_section(SECTION_LOCHNESS_PER_TARGET),
        "modules": reg.by_section(SECTION_MODULES),
        "distance": reg.by_section(SECTION_DISTANCE),
        "distance_space": reg.by_section(SECTION_DISTANCE_SPACE),
    }
    extras = reg.extras(SECTION_PER_GENE)
    enrich_extras = reg.extras(SECTION_ENRICH_PER_TARGET)
    distance_extras = reg.extras(SECTION_DISTANCE)
    distance_space_extras = reg.extras(SECTION_DISTANCE_SPACE)
    tables_html = {
        key: _df_to_html(inputs.tables.get(key), cfg.report.max_table_rows)
        for key in (
            "qc_steps",
            "qc_summary",
            "guide_qc",
            "pair_assignment_per_lane",
            "pair_guide_qc_per_lane",
            "cell_counts_before_after",
            "pair_perturbation_hit_counts_per_lane",
            "single_guide_diagnostic_vs_pair",
            "high_moi_calling",
            "high_moi_rank_umi_profile",
            "cluster_sizes",
            "clusters",
            "perturbation",
            "skipped",
            "manifest",
            "enrichment",
            "enrichment_pseudo_summary",
            "enrichment_pseudo_targets",
            "ps_score",
            "lochness",
            "cofunctional_modules",
            "gene_programs",
            "program_summary",
            "program_enrichment",
            "module_program_strength",
            "tf_hubs",
            "perturbation_distance",
            "phenotype_modules",
            "perturbation_neighbors",
            "perturbation_meta",
            "perturbation_space_coordinates",
        )
    }
    tables_html["module_status"] = _df_to_html(inputs.module_status, 50)
    tables_html["outputs"] = _df_to_html(
        pd.DataFrame([{"deliverable": k, "path": v} for k, v in inputs.outputs.items()])
    )
    # ``skipped`` drives a conditional heading, so it must be falsy when empty.
    if inputs.tables.get("skipped") is None or len(inputs.tables.get("skipped", [])) == 0:
        tables_html["skipped"] = ""
    # --- enrichment context ------------------------------------------------
    enr = inputs.enrichment
    enrichment_ctx = None
    if enr is not None and not enr.table.empty:
        om = enr.omnibus or {}
        enrichment_ctx = {
            "n_hits": int(enr.table["significant"].sum()),
            "n_targets_with_hits": len(enr.targets_with_hits()),
            "n_targets": int(enr.composition.shape[0]),
            "n_clusters": int(enr.composition.shape[1]),
            "n_tests": int(enr.composition.shape[0] * enr.composition.shape[1]),
            "control_label": CONTROL_LABELS[enr.primary_control],
            "controls_described": " and ".join(CONTROL_LABELS[c] for c in enr.controls_used),
            "chi2": f"{om.get('chi2', float('nan')):.0f}",
            "dof": om.get("dof", 0),
            "p_perm": f"{om.get('p_permutation', float('nan')):.3g}",
            "pct_small": f"{om.get('pct_expected_below_5', 0):.0f}",
            "stratified": enr.stratified,
            "stratify_by": enr.stratify_by,
            "n_low_power": int(enr.table["low_power"].sum()),
            "top_shift": (enr.effect_magnitude.iloc[0].to_dict() if len(enr.effect_magnitude) else {}),
        }
    mods = inputs.modules
    modules_ctx = None
    if mods is not None and not mods.effect_matrix.empty:
        top_hub = mods.hubs.iloc[0].to_dict() if not mods.hubs.empty else {}
        # Program biological annotations & pathways
        prog_annotations = getattr(mods, "program_annotations", {})
        display_labels = getattr(mods, "program_display_labels", {})
        enr_df = getattr(mods, "program_enrichment", None)
        program_details = []
        for p in mods.program_labels:
            p_genes = mods.program_genes.get(p, [])
            ann = prog_annotations.get(p, "unannotated")
            disp = display_labels.get(p, p)
            top_pathways = []
            if enr_df is not None and not enr_df.empty and "program_id" in enr_df.columns:
                p_enr = enr_df[enr_df["program_id"] == p].sort_values(["fdr", "p_value"]).head(5)
                for _, r in p_enr.iterrows():
                    fdr_val = r.get("fdr", float("nan"))
                    fdr_str = (
                        f"{fdr_val:.2e}"
                        if pd.notna(fdr_val) and fdr_val < 0.001
                        else (f"{fdr_val:.3f}" if pd.notna(fdr_val) else "N/A")
                    )
                    top_pathways.append(
                        {
                            "term": r.get("term", ""),
                            "clean_term": r.get("clean_term", r.get("term", "")),
                            "source": r.get("gene_set_source", ""),
                            "fdr": fdr_str,
                            "fdr_num": fdr_val,
                            "overlap_count": int(r.get("overlap_count", 0)),
                            "overlap_genes": r.get("overlap_genes", ""),
                        }
                    )
            program_details.append(
                {
                    "program_id": p,
                    "annotation": ann,
                    "display_label": disp,
                    "top_genes": p_genes[:10],
                    "all_genes_count": len(p_genes),
                    "top_pathways": top_pathways,
                }
            )
        # Module-program biological interpretations
        mp_mat = getattr(mods, "module_program", None)
        module_details = []
        for m in mods.module_labels:
            m_members = mods.module_members.get(m, [])
            pos_progs = []
            neg_progs = []
            if mp_mat is not None and not mp_mat.empty and m in mp_mat.index:
                row = mp_mat.loc[m]
                for p_col, val in row.items():
                    if np.isfinite(val):
                        disp = display_labels.get(p_col, p_col)
                        if val > 0.05:
                            pos_progs.append((disp, float(val)))
                        elif val < -0.05:
                            neg_progs.append((disp, float(val)))
                pos_progs.sort(key=lambda x: -x[1])
                neg_progs.sort(key=lambda x: x[1])
            module_details.append(
                {
                    "module_id": m,
                    "members": m_members,
                    "n_targets": len(m_members),
                    "positive_programs": pos_progs[:3],
                    "negative_programs": neg_progs[:3],
                }
            )
        has_enrichment = bool(enr_df is not None and not enr_df.empty)
        enrichment_error = str(getattr(mods, "program_enrichment_error", "") or "")
        modules_ctx = {
            "enrichment_error": enrichment_error,
            "n_modules": mods.n_modules,
            "n_programs": mods.n_programs,
            "n_perturbations": int(mods.effect_matrix.shape[0]),
            "n_genes": int(mods.effect_matrix.shape[1]),
            "control_label": CONTROL_LABELS.get(mods.control, mods.control),
            "module_correlation": mods.module_correlation,
            "program_correlation": mods.program_correlation,
            "linkage": mods.linkage_method,
            "hub_lfc": cfg.modules.hub_lfc_threshold,
            "top_hub": top_hub.get("target_gene", ""),
            "top_hub_n": int(top_hub.get("n_de_genes", 0)),
            "n_tf_edges": int(len(mods.tf_edges)),
            "program_genes": {p: mods.program_genes.get(p, [])[:10] for p in mods.program_labels},
            "program_annotations": prog_annotations,
            "program_display_labels": display_labels,
            "programs": program_details,
            "modules_list": module_details,
            "has_enrichment": has_enrichment,
        }
    modules_extras = reg.extras(SECTION_MODULES)
    ps = inputs.ps
    ps_ctx = None
    if ps is not None and not ps.summary.empty:
        summ = ps.summary
        ps_ctx = {
            "n_targets": int(len(summ)),
            "threshold": ps.ps_threshold,
            "median_kd": f"{summ['pct_successful_kd'].median():.0f}",
            "median_escaper": f"{summ['pct_escaper'].median():.0f}",
            "best": summ.iloc[0]["target_gene"],
            "best_kd": f"{summ.iloc[0]['pct_successful_kd']:.0f}",
            "worst_escaper": summ.sort_values("pct_escaper").iloc[-1]["target_gene"],
            "worst_escaper_pct": f"{summ['pct_escaper'].max():.0f}",
            "n_skipped": int(len(ps.skipped)),
            "has_lda": ps.lda_umap is not None,
            "lda_note": ps.lda_note,
            "version": _pertps_version_or_none(),
        }
    ps_note = ps.note if ps is not None and ps.note else ""
    ps_extras = reg.extras(SECTION_PS_PER_TARGET)
    ps_lda_extras = reg.extras(SECTION_PS_LDA)
    loch = inputs.lochness
    loch_ctx = None
    if loch is not None and not loch.summary.empty:
        s0 = loch.summary
        loch_ctx = {
            "n_targets": int(len(s0)),
            "k": loch.n_neighbors,
            "cut": cfg.lochness.enrichment_cut,
            "best": s0.iloc[0]["target_gene"],
            "best_score": f"{s0.iloc[0]['mean_lochness_in_own_cells']:.2f}",
            "n_positive": int((s0["mean_lochness_in_own_cells"] > cfg.lochness.enrichment_cut).sum()),
            "genotype_key": cfg.lochness.genotype_key,
        }
    loch_extras = reg.extras(SECTION_LOCHNESS_PER_TARGET)
    dist = inputs.distance
    distance_ctx = None
    if dist is not None and not dist.table.empty:
        distance_ctx = {
            "n_targets": int(len(dist.table)),
            "n_hits": int(dist.table["significant"].sum()) if "significant" in dist.table.columns else 0,
            "primary_metric": dist.primary_metric,
            "secondary_metric": dist.secondary_metric,
            "control_used": dist.control_used,
            "fdr_threshold": cfg.distance.fdr_threshold,
            "n_skipped": len(dist.skipped) if dist.skipped is not None else 0,
            "top_target": dist.table.iloc[0]["target_gene"] if len(dist.table) > 0 else "",
            "top_distance": f"{dist.table.iloc[0]['energy_distance']:.3f}" if len(dist.table) > 0 else "",
        }
    dist_space = inputs.distance_space
    distance_space_ctx = None
    if dist_space is not None and not dist_space.distance_matrix.empty:
        distance_space_ctx = {
            "n_targets": int(len(dist_space.distance_matrix)),
            "n_components": dist_space.n_components,
            "n_modules": int(dist_space.phenotype_modules["phenotype_module"].nunique())
            if not dist_space.phenotype_modules.empty
            else 0,
            "metric": dist_space.metric,
            "linkage": dist_space.linkage_method,
        }
    controls_described = " and ".join(CONTROL_LABELS[c] for c in res.controls_used)
    primary_fallback = res.primary_control != cfg.perturbation.primary_control
    n_hits = len(res.hits) if not res.table.empty else 0
    perturbation_cards = [
        ("Targets tested", f"{len(res.table):,}"),
        ("Effective knockdowns", f"{n_hits:,}"),
        ("Control cells", f"{res.n_control_cells.get(res.primary_control, 0):,}"),
        ("Targets not testable", f"{len(res.skipped):,}"),
    ]
    html = template.render(
        title=cfg.report.title,
        run_name=cfg.run.name,
        version=__version__,
        generated_at=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        cfg=cfg,
        summary_cards=inputs.summary_cards,
        warnings=inputs.warnings,
        input_mode=inputs.input_mode,
        input_source=inputs.input_source,
        lanes=inputs.lanes,
        metadata_source=inputs.metadata_source or "none (single lane)",
        guide_source_text=inputs.guide_source_text,
        tables=tables_html,
        figures=figures,
        render_figure=lambda f: _render_figure(f, embed),
        controls_described=controls_described,
        primary_control_label=CONTROL_LABELS[res.primary_control],
        primary_fallback=primary_fallback,
        perturbation_cards=perturbation_cards,
        n_top_shown=len(figures["per_gene"]),
        extra_figures=extras,
        extra_figure_names=[f"{r.name}.{cfg.report.figure_format}" for r in extras],
        ps=ps_ctx,
        ps_note=ps_note,
        ps_extras=ps_extras,
        ps_extra_names=[f"{r.name}.{cfg.report.figure_format}" for r in ps_extras],
        ps_per_target_dir=str(reg.figdir / SECTION_PS_PER_TARGET),
        ps_lda_extras=ps_lda_extras,
        ps_lda_extra_names=[f"{r.name}.{cfg.report.figure_format}" for r in ps_lda_extras],
        ps_lda_dir=str(reg.figdir / SECTION_PS_LDA),
        lochness=loch_ctx,
        lochness_extras=loch_extras,
        lochness_extra_names=[f"{r.name}.{cfg.report.figure_format}" for r in loch_extras],
        lochness_dir=str(reg.figdir / SECTION_LOCHNESS_PER_TARGET),
        modules=modules_ctx,
        modules_extras=modules_extras,
        modules_extra_names=[f"{r.name}.{cfg.report.figure_format}" for r in modules_extras],
        modules_dir=str(reg.figdir / SECTION_MODULES),
        distance=distance_ctx,
        distance_extras=distance_extras,
        distance_extra_names=[f"{r.name}.{cfg.report.figure_format}" for r in distance_extras],
        distance_dir=str(reg.figdir / SECTION_DISTANCE),
        distance_space=distance_space_ctx,
        distance_space_extras=distance_space_extras,
        distance_space_extra_names=[f"{r.name}.{cfg.report.figure_format}" for r in distance_space_extras],
        distance_space_dir=str(reg.figdir / SECTION_DISTANCE_SPACE),
        enrichment=enrichment_ctx,
        enrichment_extras=enrich_extras,
        enrichment_extra_names=[f"{r.name}.{cfg.report.figure_format}" for r in enrich_extras],
        enrichment_per_gene_dir=str(reg.figdir / SECTION_ENRICH_PER_TARGET),
        per_gene_dir=str(reg.figdir / SECTION_PER_GENE),
        n_figures=len(reg.records),
        config_yaml=_config_yaml(cfg),
        versions=_versions(),
        provenance_rows=inputs.provenance_rows,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    logger.info("Wrote report %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def _pertps_version_or_none() -> str:
    try:
        import pertps

        return getattr(pertps, "__version__", "unknown")
    except Exception:  # pragma: no cover - optional dependency
        return "not installed"


def _config_yaml(cfg: Config) -> str:
    import yaml

    return yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False)


# Basic QC stage report


def build_qc_report(
    cfg: Config,
    registry: FigureRegistry,
    path: Path,
    *,
    tables: Dict[str, pd.DataFrame],
    cards: List[tuple],
    outputs: Dict[str, str],
    warnings: List[str],
    provenance_text: str,
) -> Path:
    """Render the QC-only HTML report (no perturbation results required)."""
    import datetime as _dt
    from . import __version__
    from .qc_plots import SECTION_DOUBLETS, SECTION_QC_SAMPLES

    embed = cfg.report.embed_figures
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))
    template = env.get_template("qc_report.html")

    def _figs(section):
        out = []
        for rec in registry.by_section(section):
            if embed:
                src = rec.data_uri()
            else:
                try:
                    src = str(Path(rec.path).relative_to(Path(path).parent))
                except ValueError:
                    src = str(rec.path)
            out.append({"src": src, "title": rec.title, "caption": rec.caption})
        return out

    figures = {
        "qc": _figs(SECTION_QC),
        "qc_samples": _figs(SECTION_QC_SAMPLES),
        "doublets": _figs(SECTION_DOUBLETS),
        "guides": _figs(SECTION_GUIDES),
    }
    max_rows = cfg.report.max_table_rows
    tables_html = {k: _df_to_html(v, max_rows) for k, v in tables.items()}
    html = template.render(
        title=cfg.report.title,
        run_name=cfg.run.name,
        timestamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=__version__,
        cards=cards,
        tables=tables_html,
        figures=figures,
        outputs=outputs,
        warnings=warnings,
        provenance=provenance_text,
        versions=_versions(),
        config_yaml=_config_yaml(cfg),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path
