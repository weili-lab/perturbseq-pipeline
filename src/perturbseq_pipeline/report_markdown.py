"""Companion Markdown report for one pipeline run (``output.report_markdown_name``).

Mirrors the HTML report: run summary, expression QC before / after filtering,
guide assignment (single-guide) or pair-guide QC / ambiguity / single-guide
diagnostic (pair mode), PCA / UMAP / Leiden, the perturbation-strength test of
the assignment mode that actually ran, and the optional stages that produced a
table in this run (cluster enrichment, modules / programs, PS scores, lochNESS,
perturbation distance, master table). Sections are numbered in the order they
appear. Figures are referenced by their path relative to the run directory
(``figures/<section>/<name>.png``) so the file renders in any Markdown viewer
next to the run outputs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .plots import SECTION_CLUSTERING, SECTION_GUIDES, SECTION_PERTURBATION, SECTION_QC, FigureRegistry
from .report import ReportInputs, _versions, regression_stratum

logger = logging.getLogger(__name__)

_SKIP_TABLE_COLS = {"hit_rule", "fdr_convention", "stratum_description"}


def md_table(df: Optional[pd.DataFrame], max_rows: int = 60, max_cols: int = 24) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (bounded)."""
    if df is None or len(df) == 0:
        return "_not available for this run_\n"
    d = df.copy()
    d = d[[c for c in d.columns if c not in _SKIP_TABLE_COLS][:max_cols]]
    cols = [str(c) for c in d.columns]

    def fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e5 else f"{v:.3f}".rstrip("0").rstrip(".")
        s = str(v)
        return s.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.head(max_rows).iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r.tolist()) + " |")
    if len(d) > max_rows:
        lines.append(f"\n_{len(d) - max_rows} more rows in `tables/`_")
    return "\n".join(lines) + "\n"


def _figs(
    reg: FigureRegistry, section: str, run_dir: Path, names: Optional[Iterable[str]] = None, only_in_report: bool = True
) -> str:
    recs = reg.by_section(section, only_in_report=only_in_report)
    if names is not None:
        wanted = list(names)
        recs = [r for r in recs if r.name in wanted]
    out = []
    for r in recs:
        rel = r.path.relative_to(run_dir) if r.path.is_relative_to(run_dir) else r.path
        cap = f"{r.title}" + (f" — {r.caption}" if r.caption else "")
        out.append(f"![{r.title}]({rel.as_posix()})\n\n_{cap}_\n")
    return "\n".join(out) if out else "_no figures in this section_\n"


class _Numbering:
    """Sequential section numbers, so a disabled stage never leaves a gap."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, title: str) -> str:
        self.n += 1
        return f"## {self.n}. {title}"


def _single_guide_perturbation_methods(cfg) -> str:
    """Methods sentence for ``perturbation.py`` (single-guide assignment)."""
    p = cfg.perturbation
    controls = ", ".join(f"`{c}`" for c in p.controls)
    return (
        "Test: for every target gene that is also in the expression matrix, its own log-normalised "
        "expression (`layers['lognorm']`) in perturbed cells vs control cells — two-sided Kolmogorov-Smirnov "
        "(and a one-sided Mann-Whitney, perturbed < control) per control definition "
        f"({controls}; primary `{p.primary_control}`: `ntc` = non-targeting cells, `other` = cells assigned to a "
        "different target); log2FC on de-logged means with a pseudocount of 0.01; BH FDR across all tested targets "
        f"within each control arm; effective = `ks_fdr_{p.primary_control} < {p.fdr_alpha}` and "
        f"`log2fc_{p.primary_control} < {p.max_log2fc_for_hit}`; minimum {p.min_cells_per_target} perturbed cells "
        f"and {p.min_control_cells} control cells. Targets absent from the matrix or not detectably expressed "
        "in controls are listed as skipped, not scored."
    )


def _high_moi_per_cell_text(tables: dict, cfg) -> str:
    """Describe the per-cell membership outputs that this run actually produced."""
    parts = []
    if "ps_score" in tables and not getattr(tables["ps_score"], "empty", True):
        parts.append("PS scores (`obsm['ps_score_membership']`; `obs['ps_score']` = primary target)")
    if "lochness" in tables and not getattr(tables["lochness"], "empty", True):
        parts.append("lochNESS (`obsm['lochness_membership']`; `obs['lochness_self']` = primary target)")
    if "knockdown_filter" in tables and not getattr(tables["knockdown_filter"], "empty", True):
        parts.append("knockdown filter (`obsm['kd_status_membership']`; `obs['kd_status']` = primary target)")
    if not parts:
        return "No per-cell stage (PS, lochNESS, knockdown filter) produced output in this run. "
    return (
        "Per-cell outputs on membership, stored for every (cell, carried target) pair whose target passed the stage's "
        "thresholds and was successfully scored: " + "; ".join(parts) + ". `obs` columns hold each cell's value for its "
        "primary (highest-UMI) target. "
    )


def write_markdown_report(inputs: ReportInputs, path: Path) -> Path:
    cfg, reg, tables = inputs.cfg, inputs.registry, inputs.tables
    run_dir = Path(cfg.run.outdir)
    T = lambda k, n=60: md_table(tables.get(k), n)
    pair_mode = "pair_assignment_per_lane" in tables
    high_moi_mode = "high_moi_calling" in tables
    H = _Numbering()
    L: List[str] = []
    L += [
        f"# {cfg.report.title}",
        "",
        f"Run `{cfg.run.name}` — generated {datetime.now():%Y-%m-%d %H:%M}. HTML report: `{cfg.output.report_name}`.",
        "",
    ]
    L += [
        "## Run summary",
        "",
        md_table(pd.DataFrame(inputs.summary_cards, columns=["item", "value"]) if inputs.summary_cards else None, 40),
    ]
    L += [
        f"- input mode: `{inputs.input_mode}`; lanes: {inputs.lanes}",
        f"- input source: {inputs.input_source}",
        f"- sample metadata: `{inputs.metadata_source or 'none'}`",
        f"- guide source: {inputs.guide_source_text}",
        f"- guide assignment mode: `{cfg.guides.assignment_mode}`",
        "",
    ]
    if inputs.warnings:
        L += ["### Warnings", ""] + [f"- {w}" for w in inputs.warnings] + [""]
    # ---- expression QC --------------------------------------------------------------------------
    L += [
        H("Expression QC before filtering"),
        "",
        f"Thresholds: permissive gene filter >= {cfg.qc.min_genes_per_cell} genes, final gene filter >= {cfg.qc.min_genes_final} genes, mitochondrial < {cfg.qc.max_pct_mt} %, genes kept if in >= {cfg.qc.min_cells_per_gene} cells.",
        "",
        "Cell accounting (input / permissive / final gene filter / mitochondrial / retained):",
        "",
        T("cell_counts_before_after"),
        "QC metrics per lane before and after filtering (medians and means of total UMIs, detected genes, % mitochondrial, % ribosomal, % haemoglobin):",
        "",
        T("qc_metrics_per_lane_before_after"),
        _figs(
            reg,
            SECTION_QC,
            run_dir,
            [
                "cell_counts_before_after_per_lane",
                "qc_violin_before_filtering",
                "qc_scatter_before_filtering",
                "qc_cells_per_lane_before_filtering",
                "pct_mt_vs_genes_and_umis",
            ],
        ),
        "",
    ]
    L += [
        H("Expression QC after filtering"),
        "",
        T("qc_steps"),
        T("qc_summary"),
        _figs(
            reg,
            SECTION_QC,
            run_dir,
            [
                "qc_violin_after_filtering",
                "qc_scatter_after_filtering",
                "qc_cells_per_lane_after_filtering",
                "genes_vs_umis_before_after",
                "ecdf_total_counts_before_after",
                "ecdf_n_genes_by_counts_before_after",
                "ecdf_pct_counts_mt_before_after",
                "ecdf_pct_counts_ribo_before_after",
                "ecdf_pct_counts_hb_before_after",
            ],
        ),
        "",
    ]
    # ---- guide assignment ---------------------------------------------------------------------------
    if high_moi_mode:
        hm = cfg.guides.high_moi
        L += [
            H("High-MOI guide calling"),
            "",
            f"Membership rule (`guides.high_moi.method: {hm.method}`): a guide is called in a cell when it has >= {hm.min_umi} UMIs"
            + (f" and >= {hm.min_frac_of_top:g} x the cell's top guide" if hm.method == "threshold" else " and lies above the largest log-UMI drop of the cell")
            + f"; cells with more than {hm.max_guides_per_cell} called guides are classed ambiguous. Non-targeting guides collapse into one `{cfg.guides.ntc_label}` membership column. "
            "`obs['target_gene']` is each cell's primary (highest-UMI) target.",
            "",
            "**Membership-aware statistics:** perturbation strength, cluster enrichment, co-functional modules and perturbation "
            "distance take perturbed = cells carrying the target and `other` = targeting cells not carrying it; every "
            "non-targeting guide is also tested as a negative-control pseudo-target (`NTC:<guide>`), giving the empirical "
            "false-positive rate below. "
            + _high_moi_per_cell_text(tables, cfg)
            + f"The full membership is stored in `obsm['{hm.membership_obsm_key}']` with target names in `uns['membership_targets']`.",
            "",
            T("high_moi_calling"),
            "Rank-ordered guide UMI profile (median and p10-p90 of the k-th ranked guide per cell):",
            "",
            T("high_moi_rank_umi_profile", 25),
        ]
    if pair_mode:
        L += [
            H("Pair-guide QC"),
            "",
            f"Assignment rule: strongest guide per scaffold class needs >= {cfg.guides.min_umi} UMIs and > {cfg.guides.dominance_ratio} x the class runner-up; a cell is a designed pair only when the two slot features share a construct id "
            f"(pair reference `{cfg.guides.pair_map_file or cfg.guides.pair_reference}`). Primary labels: `pair_targeting` / `pair_targeting_plus_ntc` (designed targeting constructs) and `pair_non_targeting` (designed NTC pairs).",
            "",
            "Pair-assignment status per lane:",
            "",
            T("pair_assignment_per_lane"),
            "Pair-guide QC per lane (guide UMIs, detected guides, scaffold detection, complete / incomplete pairs, ambiguity, unknown, unresolved, below threshold, no guide):",
            "",
            T("pair_guide_qc_per_lane"),
            "Construct type per lane:",
            "",
            T("construct_type_per_lane"),
            "Strong guides per scaffold slot:",
            "",
            T("strong_guides_per_scaffold_per_lane"),
            "Guide / GEX barcode overlap:",
            "",
            T("guide_gex_barcode_overlap"),
            "Guide UMI mass by feature role (designed slot / wrong-scaffold chimera / never-cloned spacer):",
            "",
            T("off_design_umi_fraction_per_lane"),
            "Cells per target (primary targeting pairs):",
            "",
            T("target_cells_per_lane", 80),
            "Cells per construct (designed pairs):",
            "",
            T("construct_cells_per_lane", 80),
            "Feature-level representation (all count-matrix features; full table in `tables/guide_feature_representation.csv`):",
            "",
            T("guide_feature_representation", 40),
            _figs(reg, SECTION_GUIDES, run_dir),
            "",
        ]
        det = tables.get("pair_resolution_detail_per_lane")
        L += [
            H("Pair ambiguity and unresolved assignments"),
            "",
            "Every cell carries `pair_assignment_status` and `pair_resolution_detail`; only `perturbation_class in {targeting, non-targeting}` enters primary testing. Everything else (dual-target ambiguous, scaffold-ambiguous, incomplete, unresolved / not designed, unknown guide, below threshold, no guide) is excluded and listed here.",
            "",
            md_table(
                det[~det["enters_primary_testing"].astype(bool)]
                if det is not None and "enters_primary_testing" in det.columns
                else det,
                80,
            ),
            "Designed constructs that entered primary testing:",
            "",
            md_table(
                det[det["enters_primary_testing"].astype(bool)]
                if det is not None and "enters_primary_testing" in det.columns
                else None,
                40,
            ),
            "",
        ]
        L += [
            H("Single-guide diagnostic (not used for any result)"),
            "",
            T("single_guide_diagnostic_vs_pair"),
            _figs(reg, SECTION_GUIDES, run_dir, ["single_guide_diagnostic_vs_pair_status"]),
            "",
        ]
    else:
        L += [
            H("Guide assignment"),
            "",
            f"Rule: a cell is assigned to its dominant guide when that guide has >= {cfg.guides.min_umi} UMIs and > {cfg.guides.dominance_ratio} x the runner-up"
            + (
                f" and the runner-up has <= {cfg.guides.max_second_umi} UMIs"
                if cfg.guides.max_second_umi is not None and cfg.guides.max_second_umi >= 0
                else ""
            )
            + "; cells with guide counts that fail the rule are `ambiguous`, cells without guide counts `unassigned`. Non-targeting guides give `perturbation_class = non-targeting`.",
            "",
            T("guide_qc"),
            T("guide_assignment"),
            T("assignment_per_lane"),
            "Cells per guide (designed guides with zero cells included):",
            "",
            T("guide_representation", 80),
            _figs(reg, SECTION_GUIDES, run_dir),
            "",
        ]
    # ---- clustering --------------------------------------------------------------------------------
    batch = cfg.cluster.batch_key
    L += [
        H("PCA, UMAP and Leiden clustering"),
        "",
        f"Normalisation (target_sum = {cfg.cluster.target_sum or 'median'}), log1p, {cfg.cluster.n_top_genes} highly variable genes, {cfg.cluster.n_pcs} PCs, {cfg.cluster.n_neighbors} neighbours, UMAP (min_dist {cfg.cluster.umap_min_dist}), Leiden resolution {cfg.cluster.leiden_resolution}; "
        + (f"Harmony batch correction on `{batch}`." if batch else "no batch correction (cluster.batch_key is null)."),
        "",
        "Cluster sizes:",
        "",
        T("cluster_sizes", 60),
        "Cluster composition (by sample, pair status, assignment class):",
        "",
        T("cluster_composition", 80),
        _figs(reg, SECTION_CLUSTERING, run_dir),
        "",
    ]
    # ---- perturbation strength ----------------------------------------------------------------------
    if pair_mode:
        L += [H("ECDF analysis"), ""]
        ecdf_dir = run_dir / "figures" / "perturbation" / "ecdf"
        ecdfs = sorted(ecdf_dir.glob("ecdf_*.png")) if ecdf_dir.is_dir() else []
        L += [
            f"Per-target ECDFs (targeting pairs vs NTC pairs, per lane and pooled): {len(ecdfs)} figures in `figures/perturbation/ecdf/`.",
            "",
        ]
        L += [f"- [{p.stem.replace('ecdf_', '')}]({p.relative_to(run_dir).as_posix()})" for p in ecdfs] + [""]
        L += [
            _figs(
                reg,
                SECTION_PERTURBATION,
                run_dir,
                ["pair_ecdf_overview_top_targets", "pair_level_expression_distributions"],
            ),
            "",
        ]
        L += [
            H("Perturbation-expression analysis (FDR, log2FC)"),
            "",
            f"Test: two-sided Kolmogorov-Smirnov and one-sided Mann-Whitney (perturbed < control) on the target transcript (log-normalised expression), targeting pairs vs NTC pairs; log2FC on de-logged means with pseudocount; BH FDR within lane x control x stratum; "
            f"hit = `fdr_ks < {cfg.perturbation.fdr_alpha}` and `log2fc < {cfg.perturbation.max_log2fc_for_hit}`; minimum {cfg.perturbation.min_cells_per_target} target cells and {cfg.perturbation.min_control_cells} control cells. "
            "`neg_log10_fdr = -log10(max(fdr_ks, 1e-300))`.",
            "",
            "Hit counts per lane:",
            "",
            T("pair_perturbation_hit_counts_per_lane"),
            "Target-level primary results (all lanes; full table `tables/pair_perturbation_by_target.csv` includes every stratum and the 'other targets' control):",
            "",
            T("pair_perturbation_primary", 200),
            "Guide-pair (construct) level results (`tables/pair_perturbation_by_pair.csv`):",
            "",
            T("pair_perturbation_by_pair", 60),
            _figs(
                reg,
                SECTION_PERTURBATION,
                run_dir,
                [
                    "pair_volcano_target_level",
                    "pair_volcano_pair_level",
                    "pair_waterfall_target_log2fc",
                    "pair_heatmap_log2fc_by_lane",
                    "pair_heatmap_neg_log10_fdr_by_lane",
                    "pair_hit_counts_per_lane",
                    "pair_level_hit_counts_per_lane",
                ],
            ),
            "Pipeline single-assignment perturbation table (same labels, pipeline default statistics):",
            "",
            T("perturbation", 60),
            T("skipped", 30),
            "",
        ]
        if "target_support_matrix" in tables:
            L += [H("Target support within this run"), "", T("target_support_matrix", 80), ""]
    else:
        L += [
            H("Perturbation strength (target expression vs control)"),
            "",
            _single_guide_perturbation_methods(cfg),
            "",
            "Target-level results (ranked by the primary control; both control arms in the table):",
            "",
            T("perturbation", 80),
            "Targets skipped (not in the matrix / not detectably expressed in controls / too few cells):",
            "",
            T("skipped", 40),
            _figs(reg, SECTION_PERTURBATION, run_dir),
            "",
        ]
    # ---- optional stages: only the ones that produced a table in this run -------------------------
    if "regression_design" in tables:
        r = cfg.regression
        stratum = regression_stratum(tables)
        if stratum:
            perm_scope = f" within `{stratum}`"
        elif r.batch_key:
            perm_scope = f", unstratified: `{r.batch_key}` is not in obs"
        else:
            perm_scope = ", unstratified"
        L += [
            H("Membership regression (high-MOI)"),
            "",
            "All targets fitted jointly, one linear model per gene over the assigned cells (targeting + NTC-only): "
            "`lognorm ~ membership (all targets)"
            + (" + n_guides" if r.n_guides_covariate else "")
            + (" + log(total_counts)" if r.depth_covariate else "")
            + (f" + {stratum}" if stratum else "")
            + f"`, ridge penalty {r.ridge_alpha:g} on the membership coefficients. Unlike the pseudobulk contrasts, "
            "each effect is adjusted for the targets co-carried in the same cells. "
            f"Permutations ({r.n_permutations}; membership rows shuffled across cells{perm_scope}) "
            "calibrate each target's t statistic (genomic control: divided by sqrt(lambda), lambda = permuted median t² / its "
            "expected value, at least 1) before the t-test p-value; "
            + ("BH across all (target, gene) pairs; " if r.fdr_scope == "global" else "BH across genes within each target; ")
            + f"significant at FDR < {r.fdr_alpha}"
            + (f" and |log2fc| > {r.min_abs_log2fc:g}" if r.min_abs_log2fc > 0 else "")
            + ". The same calls made on the permuted data give the empirical FDR of the call set "
            "(`perm_mean_significant_pairs` / `n_significant_pairs` = `empirical_fdr` below)"
            + ". `log2fc` = lognorm coefficient / ln 2 (log2 ratio of geometric means of normalised counts + 1). "
            "Full matrices: `tables/regression_effect_matrix.csv`, `tables/regression_fdr.csv`; pairs: `tables/regression_de.csv`.",
            "",
            T("regression_design"),
            "",
            T("regression_summary", 60),
            "",
        ]
    if "enrichment" in tables:
        e = cfg.enrichment
        L += [
            H("Cluster enrichment"),
            "",
            f"Fisher's exact test per (target, cluster) pair, BH FDR across all pairs within each control arm (controls {', '.join(f'`{c}`' for c in e.controls)}; primary `{e.primary_control}`), "
            f"significant at FDR < {e.fdr_alpha}; minimum {e.min_cells_per_target} cells per target and {e.min_cells_per_cluster} per cluster"
            + (f"; Cochran-Mantel-Haenszel stratified by `{e.stratify_by}`" if e.stratify_by else "")
            + ". `guides_agreeing` counts the target's guides that individually shift in the same direction.",
            "",
            T("enrichment", 60),
            "",
        ]
        if "enrichment_pseudo_summary" in tables:
            L += [
                "**Negative controls (high-MOI):** every non-targeting guide was tested exactly like a target "
                "(perturbed = cells carrying that NTC guide; `other` = targeting cells not carrying it). Their hit rate at the "
                "same FDR is the empirical false-positive rate of this table; compare it with the real targets' hit rate.",
                "",
                T("enrichment_pseudo_summary"),
                "Pseudo-target tests (strongest first):",
                "",
                T("enrichment_pseudo_targets", 20),
            ]
    if "cofunctional_modules" in tables or "gene_programs" in tables:
        m = cfg.modules
        L += [
            H("Co-functional modules and gene programs"),
            "",
            f"Perturbation x gene log2FC-vs-`{m.control}` matrix; perturbations clustered into modules ({m.module_correlation} correlation), genes into programs ({m.program_correlation} correlation), {m.linkage_method} linkage. "
            "Programs are annotated by over-representation against MSigDB collections downloaded at run time; `unannotated` means no term passed FDR, or (see Warnings) that the enrichment did not run.",
            "",
            "Programs:",
            "",
            T("program_summary" if "program_summary" in tables else "gene_programs", 40),
            "Modules:",
            "",
            T("cofunctional_modules", 60),
            T("tf_hubs", 30),
            "",
        ]
    if "ps_score" in tables:
        L += [
            H("Per-cell perturbation scores (PS)"),
            "",
            f"PS_python per-cell perturbation score (threshold {cfg.ps_score.ps_threshold}) combined with the target's own expression ({cfg.ps_score.expression_cut} of controls as the cut): "
            "`confirmed knockdown` = PS above threshold and target expression at or below the cut; `escaper` = PS above threshold but target still expressed.",
            "",
            T("ps_score", 60),
            T("ps_skipped", 30),
            "",
        ]
    if "lochness" in tables:
        L += [
            H("lochNESS neighbourhood enrichment"),
            "",
            "For every cell and perturbation: share of the cell's nearest neighbours (k = 300, PCA or Harmony space) carrying the perturbation, divided by the perturbation's overall share, minus one (0 = background, positive = locally over-represented). Cluster-free; ported from pertTF.",
            "",
            T("lochness", 60),
            T("lochness_by_cluster", 40),
            "",
        ]
    if "perturbation_distance" in tables:
        d = cfg.distance
        L += [
            H("Perturbation distance vs control"),
            "",
            f"Energy distance (secondary metric: {d.secondary_metric or 'none'}) between perturbed and control cells in `{d.representation}`; "
            f"permutation test ({d.n_permutations} label permutations) with BH FDR across targets, significant at FDR < {d.fdr_threshold}; "
            f"cells subsampled deterministically (<= {d.max_cells_per_target} per target, <= {d.max_control_cells} controls; the target's own cells are never part of its control).",
            "",
            T("perturbation_distance", 60),
            "",
        ]
    if "phenotype_modules" in tables:
        L += [
            H("Perturbation distance space and phenotype modules"),
            "",
            "Pairwise target x target distance matrix, PCoA coordinates, nearest phenotypic neighbours and hierarchical phenotype modules.",
            "",
            T("phenotype_modules", 60),
            "",
        ]
    if "perturbation_meta" in tables:
        L += [
            H("Master perturbation table"),
            "",
            "Target-level join of the perturbation-strength, PS, lochNESS, distance and module results (`tables/perturbation_meta.csv`).",
            "",
            T("perturbation_meta", 80),
            "",
        ]
    # ---- reproducibility --------------------------------------------------------------------------
    L += (
        ["## Outputs and reproducibility", ""]
        + [f"- {k}: `{v}`" for k, v in inputs.outputs.items()]
        + [
            "",
            f"- resolved configuration: `logs/resolved_config.yaml`; run manifest: `logs/run_manifest.json`; log: `logs/run.log`",
            f"- package versions: {_versions()}",
            "",
        ]
    )
    if inputs.provenance_rows:
        L += ["### Provenance", "", md_table(pd.DataFrame(inputs.provenance_rows, columns=["item", "value"]), 40)]
    L += ["### Module completion status", "", md_table(inputs.module_status, 40)]
    path = Path(path)
    path.write_text("\n".join(L))
    logger.info("Markdown report written to %s", path)
    return path
