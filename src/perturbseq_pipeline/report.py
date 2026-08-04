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
    SECTION_PS,
    SECTION_PS_PER_TARGET,
    SECTION_QC,
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
    ps: object = None
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    summary_cards: List[tuple] = field(default_factory=list)
    input_mode: str = ""
    input_source: str = ""
    lanes: str = ""
    metadata_source: str = ""
    guide_source_text: str = ""
    outputs: Dict[str, str] = field(default_factory=dict)


def _df_to_html(df: Optional[pd.DataFrame], max_rows: int = 200) -> str:
    """Render a DataFrame as an HTML table, or a placeholder when empty."""
    if df is None or len(df) == 0:
        return Markup('<p class="sub">Not available for this run.</p>')
    shown = df.head(max_rows)
    html = shown.to_html(index=False, escape=True, border=0, na_rep="")
    if len(df) > max_rows:
        html += (
            f'<p class="sub">Showing {max_rows} of {len(df)} rows; '
            "the full table is in <code>tables/</code>.</p>"
        )
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

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
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
    }
    extras = reg.extras(SECTION_PER_GENE)
    enrich_extras = reg.extras(SECTION_ENRICH_PER_TARGET)

    tables_html = {
        key: _df_to_html(inputs.tables.get(key), cfg.report.max_table_rows)
        for key in (
            "qc_steps",
            "qc_summary",
            "guide_qc",
            "clusters",
            "perturbation",
            "skipped",
            "manifest",
            "enrichment",
            "ps_score",
        )
    }
    tables_html["outputs"] = _df_to_html(
        pd.DataFrame(
            [{"deliverable": k, "path": v} for k, v in inputs.outputs.items()]
        )
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
            "controls_described": " and ".join(
                CONTROL_LABELS[c] for c in enr.controls_used
            ),
            "chi2": f"{om.get('chi2', float('nan')):.0f}",
            "dof": om.get("dof", 0),
            "p_perm": f"{om.get('p_permutation', float('nan')):.3g}",
            "pct_small": f"{om.get('pct_expected_below_5', 0):.0f}",
            "stratified": enr.stratified,
            "stratify_by": enr.stratify_by,
            "n_low_power": int(enr.table["low_power"].sum()),
            "top_shift": (
                enr.effect_magnitude.iloc[0].to_dict()
                if len(enr.effect_magnitude)
                else {}
            ),
        }

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
            "version": _pertps_version_or_none(),
        }
    ps_note = ps.note if ps is not None and ps.note else ""
    ps_extras = reg.extras(SECTION_PS_PER_TARGET)

    controls_described = " and ".join(CONTROL_LABELS[c] for c in res.controls_used)
    primary_fallback = res.primary_control != cfg.perturbation.primary_control

    n_hits = len(res.hits) if not res.table.empty else 0
    perturbation_cards = [
        ("Targets tested", f"{len(res.table):,}"),
        ("Effective knockdowns", f"{n_hits:,}"),
        (
            "Control cells",
            f"{res.n_control_cells.get(res.primary_control, 0):,}",
        ),
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
        enrichment=enrichment_ctx,
        enrichment_extras=enrich_extras,
        enrichment_extra_names=[
            f"{r.name}.{cfg.report.figure_format}" for r in enrich_extras
        ],
        enrichment_per_gene_dir=str(reg.figdir / SECTION_ENRICH_PER_TARGET),
        per_gene_dir=str(reg.figdir / SECTION_PER_GENE),
        n_figures=len(reg.records),
        config_yaml=_config_yaml(cfg),
        versions=_versions(),
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
