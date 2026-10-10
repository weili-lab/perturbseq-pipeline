"""Configuration schema for the Perturb-seq pipeline.

A run is fully described by one YAML file. Every threshold that appeared as a
magic number in the prototype notebooks is a named key here with a documented
default.

Configuration philosophy
------------------------
Biological/statistical parameters and computational scaling parameters are kept
separate.

For example:

    perturbation.min_cells_per_target
    lochness.n_neighbors
    modules.hub_lfc_threshold

describe the analysis itself.

By contrast:

    scaling.mode
    scaling.large_n_cells
    scaling.effect_gene_chunk
    scaling.marker_max_cells

describe *how* the same analysis is executed on different dataset sizes.

This distinction is important: Replogle-scale and KOLF-scale datasets should
use the same biological definitions wherever possible, while the implementation
changes automatically when a dense or all-cell operation would become
impractical.

Guide-target mapping
--------------------
Guide identifiers do not always encode the biological target gene.

For example, 10x Flex CRISPRi libraries may contain guide identifiers such as::

    TSS100020_17082653_23-ENST00000606659

while the true target is stored explicitly in guide feature metadata::

    var["target_gene_name"] == "CNOT7"

``guides.target_feature_column`` allows such an authoritative metadata column to
be used instead of parsing guide IDs. This is optional and therefore preserves
historical behavior for datasets such as Replogle or conventional guide
libraries whose IDs already encode the target.

Adaptive execution
------------------
``scaling.mode`` supports three modes:

``auto``
    Recommended default. STANDARD implementations are used below the configured
    thresholds and LARGE implementations above them.

``standard``
    Force the original implementations regardless of dataset size. Primarily
    useful for regression tests or reproducing an older run. This may exhaust
    memory on million-cell datasets.

``large``
    Force the scalable implementations regardless of dataset size. Useful when
    a dataset below one million cells is still unusually wide, has thousands of
    perturbations, or when memory is limited.

Existing configuration files remain valid because every new parameter has a
default.
"""

from __future__ import annotations

import copy

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

import yaml


# ===========================================================================
# Run
# ===========================================================================


@dataclass
class RunConfig:
    """Top-level run identity and output location."""

    name: str = "perturbseq_run"
    outdir: str = "results"
    seed: int = 0
    #: Stop the run after the named stage. ``None`` runs everything.
    #:
    #: ``qc``
    #:     Run the basic QC stage (load -> guide quantification -> per-sample
    #:     expression QC -> doublet flagging -> guide-multiplet flagging ->
    #:     concatenate -> write) and stop. Doublets and guide multiplets are
    #:     annotated, never removed, in this stage.
    stop_after: Optional[str] = None


# ===========================================================================
# Input
# ===========================================================================


@dataclass
class InputConfig:
    """Where the data comes from.

    Two entry points are supported.

    ``mtx``
        One or more 10x MTX directories holding gene-expression and optionally
        guide features.

    ``h5ad``
        An existing AnnData object. Guide information may live in ``var``, in a
        companion guide h5ad, in a long barcode/guide table, or in a
        pre-computed ``obs`` column.
    """

    mode: str = "auto"  # auto | mtx | h5ad
    #: ``human`` or ``mouse``. Picks the MSigDB gene-set collections for program enrichment.
    species: str = "human"
    mtx_dirs: Union[Dict[str, str], List[str], None] = None
    guide_mtx_dirs: Optional[Dict[str, str]] = None
    h5ad: Optional[str] = None
    guide_h5ad: Optional[str] = None
    guide_obs_column: Optional[str] = None
    feature_type_column: str = "feature_types"
    gex_feature_type: str = "Gene Expression"
    guide_feature_types: List[str] = field(default_factory=lambda: ["Custom", "CRISPR Guide Capture"])
    var_names: str = "gene_symbols"
    #: Let scanpy cache each 10x matrix as .h5ad for faster re-reads. Off by default:
    #: the cache is keyed by path only (a regenerated matrix in the same directory
    #: would be read from the stale copy) and it costs a full copy of every input.
    #: When on, the cache lives in ``<run.outdir>/cache``.
    cache_mtx: bool = False
    #: How cells from several MTX lanes are made unique: ``suffix`` (historical,
    #: ``<barcode>-<lane>``) or ``prefix`` (``<lane>_<barcode>``). Applied to
    #: single-lane runs too so per-lane and combined objects share one id scheme.
    cell_id_format: str = "suffix"
    #: h5ad layer containing raw counts.
    counts_layer: Optional[str] = None
    #: h5ad layer already containing log-normalized expression.
    normalized_layer: Optional[str] = None
    #: Optional long barcode -> guide table.
    guide_table: Optional[str] = None
    guide_table_cell_column: str = "cell"
    guide_table_gene_column: str = "gene"
    guide_table_guide_column: Optional[str] = "sgrna"
    guide_table_count_column: Optional[str] = "umi_count"
    guide_table_strip_prefix: bool = True

    def resolved_mtx_dirs(self) -> Dict[str, str]:
        """Return ``{lane_id: path}`` regardless of input spelling."""
        if not self.mtx_dirs:
            return {}
        if isinstance(self.mtx_dirs, dict):
            return dict(self.mtx_dirs)
        out: Dict[str, str] = {}
        for path in self.mtx_dirs:
            lane = Path(path).name
            for prefix in ("filtered_feature_bc_matrix_", "raw_feature_bc_matrix_"):
                if lane.startswith(prefix):
                    lane = lane[len(prefix) :]
            lane = lane or Path(path).name
            if lane in out:
                # The usual Cell Ranger layout (<sample>/outs/filtered_feature_bc_matrix) gives
                # every lane the same directory name; silently keeping the last one would
                # analyse a single sample and report it as the whole run.
                raise ValueError(
                    f"input.mtx_dirs: {out[lane]!r} and {path!r} both resolve to lane id {lane!r}. "
                    "Use the mapping form (lane_id: path) to give each lane its own id."
                )
            out[lane] = path
        return out


# ===========================================================================
# Metadata
# ===========================================================================


@dataclass
class MetadataConfig:
    """Per-lane sample metadata."""

    file: Optional[str] = None
    key_column: str = "lane_id"
    require_for_multilane: bool = True


# ===========================================================================
# QC
# ===========================================================================


@dataclass
class QCConfig:
    """Standard single-cell QC thresholds."""

    min_genes_per_cell: Optional[int] = 200
    min_cells_per_gene: Optional[int] = 3
    min_genes_final: Optional[int] = 1000
    max_pct_mt: Optional[float] = 20.0
    max_pct_hb: Optional[float] = None
    min_counts_per_cell: Optional[int] = None
    mito_prefix: str = "MT-"
    ribo_prefix: List[str] = field(default_factory=lambda: ["RPS", "RPL"])
    hb_pattern: str = "^HB[^(P)]"
    #: Sample-aware expression-QC thresholds used by the basic QC stage.
    thresholds: "QCThresholdConfig" = field(default_factory=lambda: QCThresholdConfig())
    #: Doublet detection (annotation only) used by the basic QC stage.
    doublets: "DoubletConfig" = field(default_factory=lambda: DoubletConfig())


# ===========================================================================
# Guide calling
# ===========================================================================


@dataclass
class GuideConfig:
    """Guide-calling and guide-to-target mapping rules.

    Two target-mapping modes are supported.

    Metadata mode
        When ``target_feature_column`` is set, that column of ``guides.var`` is
        treated as the authoritative biological target.

        This is required for libraries such as 10x Flex CRISPRi where guide IDs
        encode TSS/genomic/transcript information rather than the target gene.

    Guide-ID parsing mode
        When ``target_feature_column`` is ``None``, historical behavior is
        retained: ``target_regex`` or ``target_split_delims`` is used to infer
        the target from the guide identifier.
    """

    min_umi: int = 3
    dominance_ratio: float = 2.0
    #: Pseudocount of the per-slot dominance ratio used by the pair mode:
    #: ``(top_umi + pseudocount) / (second_umi + pseudocount) >= dominance_ratio``.
    #: With 1.0 and integer counts this equals ``top > dominance_ratio * second``
    #: for dominance_ratio 2; the ratio is stored per scaffold slot in obs.
    dominance_pseudocount: float = 1.0
    #: -1 disables the runner-up UMI gate.
    max_second_umi: int = -1
    detection_threshold: int = 3
    # Guide -> biological target mapping
    #: Optional guides.var column containing the authoritative biological
    #: target for each guide.
    #:
    #: Example for 10x Flex CRISPRi:
    #:
    #:     target_feature_column: target_gene_name
    #:
    #: Guide:
    #:     TSS100020_17082653_23-ENST00000606659
    #:
    #: guides.var["target_gene_name"]:
    #:     CNOT7
    #:
    #: When configured, this takes precedence over target_regex and
    #: target_split_delims.
    target_feature_column: Optional[str] = None
    #: Metadata values which do not represent actual biological perturbation
    #: targets. Guides carrying these annotations are treated as unassigned.
    #:
    #: 10x Flex libraries commonly use "Ignore".
    ignored_target_values: List[str] = field(default_factory=lambda: ["Ignore"])
    #: Optional regex whose first capture group is interpreted as the target.
    #: Used only when target_feature_column is null.
    target_regex: Optional[str] = None
    #: Guide-ID delimiters used only when target_feature_column is null and
    #: target_regex is not supplied.
    target_split_delims: List[str] = field(default_factory=lambda: ["_", "-", "."])
    #: Case-insensitive patterns defining non-targeting controls.
    #:
    #: The first pattern accepts:
    #:   Non-Targeting
    #:   non-targeting
    #:   non_targeting
    #:   non.targeting
    #:   non targeting
    ntc_patterns: List[str] = field(
        default_factory=lambda: [
            r"^non[-_. ]?targeting",
            r"^non$",
            r"^ntc",
            r"scramble",
            r"^safe[-_. ]?harbor",
            r"^no[-_. ]?target",
        ]
    )
    unassigned_label: str = "unassigned"
    ambiguous_label: str = "ambiguous"
    ntc_label: str = "non-targeting"
    # Assignment mode (single-guide dominance vs dual-guide pair)
    #: ``single_guide`` (historical top-1 dominance rule), ``pair`` (alias
    #: ``dual_guide_pair``): strongest scaffold-A + strongest scaffold-C guide,
    #: interpreted through the pair reference; see ``dual_guides.py``), or
    #: ``high_moi`` (multi-guide membership per cell; see ``high_moi.py`` and
    #: the ``high_moi`` block below).
    assignment_mode: str = "single_guide"
    #: Pair reference table (CSV/TSV, one row per designed guide): alias of
    #: ``pair_map_file`` used by the pair workflow; whichever is set is used.
    pair_reference: Optional[str] = None
    #: Pair assignment is the primary label set (perturbation_class / target_gene
    #: are derived from pairs). Must be true in pair mode.
    pair_assignment_primary: bool = True
    #: Require both scaffold slots to be resolved for any assignment; incomplete
    #: pairs (one slot resolved, the other empty) are labelled ``incomplete_pair``
    #: and kept out of primary testing (ambiguous). With ``false`` the single
    #: resolved slot carries the assignment (its target, or non-targeting); the
    #: status stays ``incomplete_pair`` and ``construct_type`` is ``single_slot``.
    require_complete_pair: bool = True
    #: What happens to unresolved pairs (``unresolved_pair`` etc.): ``exclude``
    #: keeps them in the object as ambiguous and out of primary testing.
    unresolved_pair_policy: str = "exclude"
    #: Also compute the single-guide top-vs-second rule on the same matrix and
    #: store it as ``single_guide_diagnostic_*`` obs columns (diagnostic only).
    single_guide_diagnostic: bool = False
    #: Column of the pair reference holding the designed protospacer (or ``auto``).
    sequence_column: str = "auto"
    #: CSV/TSV with one row per designed guide (``guide_id``), optional
    #: ``pair_id_column`` (explicit vector pairing; authoritative when present)
    #: and optional ``scaffold_column``. ``None`` = provisional same-target rule
    #: using the scaffold class stored in ``guides.var``.
    pair_map_file: Optional[str] = None
    #: ``guides.var`` (or pair-map) column holding the scaffold class per guide
    #: (``auto`` = detect among scaffold / scaffold_class / scaffold_id).
    scaffold_column: str = "scaffold"
    #: Pair-map column holding the designed pair / vector id (``auto`` = detect
    #: among pair_id / construct_id / vector_id).
    pair_id_column: str = "pair_id"
    #: The two scaffold classes forming a pair (order: first, second slot).
    scaffold_classes: List[str] = field(default_factory=lambda: ["A", "C"])
    #: How a resolved targeting + NTC pair is treated when no explicit pair map
    #: confirms it: ``ambiguous`` (conservative) or ``provisional_target``
    #: (assigned to the targeting guide's target, flagged provisional).
    ntc_partner_policy: str = "ambiguous"
    #: Delimiter separating several designed pair / construct ids in the pair
    #: reference when one guide feature belongs to more than one construct
    #: (e.g. ``ACYP1_1F;ACYP1_S1``). A pair is designed when the two slot
    #: features share at least one construct id.
    pair_id_delimiter: str = ";"
    #: Explicit-reference mode: are designed targeting + NTC constructs (e.g.
    #: single-guide ``_S1`` vectors with an NTC filler) strict primary targeting
    #: labels (``True``) or a sensitivity stratum kept out of primary testing
    #: (``False``; status ``pair_targeting_plus_ntc``, class ambiguous)?
    designed_targeting_plus_ntc_primary: bool = True
    # Basic QC stage: guide quantification and guide QC
    #: Where guide information comes from in ``samples`` mode.
    #:
    #: ``auto``
    #:     Per sample: FASTQ if ``guide_fastq_dir``/``guide_fastqs`` is set,
    #:     matrix if ``guide_matrix`` is set, otherwise no guide data.
    #: ``fastq`` / ``matrix`` / ``none``
    #:     Force one source for every sample (``none`` skips guide QC).
    source: str = "auto"
    #: Guide design reference (workbook / table of designed protospacers).
    design: "GuideDesignConfig" = field(default_factory=lambda: GuideDesignConfig())
    #: Streaming guide FASTQ counter settings.
    fastq: "GuideFastqConfig" = field(default_factory=lambda: GuideFastqConfig())
    #: Guide-derived multiplet flagging (annotation only).
    multiplet: "GuideMultipletConfig" = field(default_factory=lambda: GuideMultipletConfig())
    #: High-MOI membership calling (``assignment_mode: high_moi``).
    high_moi: "HighMoiConfig" = field(default_factory=lambda: HighMoiConfig())


# ===========================================================================
# Clustering
# ===========================================================================


@dataclass
class ClusterConfig:
    """Normalization, dimensionality reduction and clustering."""

    target_sum: Optional[float] = None
    n_top_genes: int = 3000
    n_pcs: int = 50
    n_neighbors: int = 15
    leiden_resolution: float = 1.0
    umap_min_dist: float = 0.5
    batch_key: Optional[str] = None
    regress_out: List[str] = field(default_factory=list)
    scale_max_value: Optional[float] = 10.0
    assigned_only: bool = False


# ===========================================================================
# Perturbation strength
# ===========================================================================


@dataclass
class PerturbationConfig:
    """Target-gene perturbation-strength testing."""

    controls: List[str] = field(default_factory=lambda: ["ntc", "other"])
    primary_control: str = "ntc"
    min_cells_per_target: int = 10
    min_control_cells: int = 10
    min_pct_expressing_control: float = 1.0
    fdr_alpha: float = 0.05
    max_log2fc_for_hit: float = 0.0
    top_n_report: int = 12
    umap_background_fraction: float = 0.1


# ===========================================================================
# Membership regression (high-MOI)
# ===========================================================================


@dataclass
class RegressionConfig:
    """Joint membership regression (``guides.assignment_mode: high_moi`` only).

    Per gene ``lognorm ~ membership (all targets) + n_guides + log(total_counts)
    + lane``, ridge-penalised membership coefficients; t-test p-values calibrated
    per target on permutations (membership rows shuffled within ``batch_key``),
    BH over all (target, gene) pairs by default (``fdr_scope: global``; or
    within each target, ``target``), and empirical FDRs of the call set from
    the same permutations. See ``regression.py``.
    """

    #: Optional stage, off by default; set ``enabled: true`` to run it.
    enabled: bool = False
    #: Genes to fit: ``modules`` (the modules stage's gene selection, so the
    #: matrix can replace its effect matrix), ``hvg`` or ``all``.
    genes: str = "modules"
    #: Targets with fewer member cells stay in the design but are not reported.
    min_cells: int = 10
    #: L2 penalty on the membership coefficients (covariates are unpenalised).
    ridge_alpha: float = 1.0
    #: Permutations of the membership rows (genomic-control calibration and the
    #: empirical-FDR check). Memory: 4 bytes x reported targets x genes x this.
    n_permutations: int = 10
    #: obs column used as lane covariate and permutation stratum; ``null`` = none.
    batch_key: Optional[str] = "lane_id"
    #: Include ``obs['n_guides_assigned']`` (guide burden) as a covariate.
    n_guides_covariate: bool = True
    #: Include ``log(obs['total_counts'])`` (depth) as a covariate.
    depth_covariate: bool = True
    fdr_alpha: float = 0.05
    #: BH family: ``global`` (all target x gene pairs; controls the FDR of the
    #: whole call set) or ``target`` (genes within each target, the ``modules``
    #: convention). Per-target BH does not control the false calls across
    #: targets: on the ESC full-scale screen (2,084 targets) it left ~1/3 of the
    #: targets with a DE gene expected false (empirical target FDR 0.32 vs 0.016).
    fdr_scope: str = "global"
    #: A significant pair also needs ``|log2fc|`` above this.
    min_abs_log2fc: float = 0.0


# ===========================================================================
# Knockdown filter
# ===========================================================================


@dataclass
class KnockdownFilterConfig:
    """Knockdown-efficiency mask over targeting cells.

    Marks cells in ``obs['kd_keep']`` / ``obs['kd_status']`` and never removes
    any, so perturbation strength is still estimated on every cell and the mask
    and the continuous estimates travel together in the outputs. Each targeting
    cell gets ``obs['kd_ratio']``: its target's normalized expression over the
    mean in non-targeting cells of the same context.

    Steps, applied per group: (1) the group passes when its mean ratio is
    below ``max_mean_ratio``; (2) in a passing group, cells at or above
    ``max_cell_ratio`` are marked as escapers; (3) groups with fewer than
    ``min_cells`` cells are ``non_testable`` and left unmarked.
    """

    enabled: bool = False
    #: ``pooled`` (one group per target, all contexts together);
    #: ``per_context`` (one group per target and context — most conservative);
    #: ``any_context`` (a target passing step 1 in at least one context has
    #: step 2 applied in its passing contexts and keeps every cell elsewhere).
    mode: str = "pooled"
    #: ``obs`` column defining the context (e.g. ``cell_line`` / ``condition``
    #: from the sample metadata). Required for ``per_context`` / ``any_context``.
    #: In ``pooled`` mode it sets the per-context control baseline; null uses
    #: one baseline over all cells.
    context_key: Optional[str] = None
    #: Step 1: mean ratio of the group must be below this.
    max_mean_ratio: float = 0.3
    #: Step 1 threshold in ``any_context`` mode; null uses ``max_mean_ratio``.
    max_mean_ratio_any: Optional[float] = None
    #: Step 2: a cell's own ratio must be below this.
    max_cell_ratio: float = 0.5
    #: Step 3: groups with fewer cells are marked ``non_testable``, not filtered.
    min_cells: int = 30
    #: Contexts with fewer non-targeting cells have no baseline: ``non_testable``.
    min_control_cells: int = 10
    #: Targets detected in fewer than this percent of the context's control
    #: cells are marked for filtering: with mostly-zero counts every cell would
    #: pass the ratio cut through dropout alone.
    min_pct_expressing_control: float = 10.0
    #: ``mean_ratio`` (steps 1-2 on ``kd_ratio``) or ``count_model``: a
    #: negative-binomial mixture on the target's raw counts that fits ``rho``
    #: (expression left in knocked-down cells, relative to control) and the
    #: escaper fraction per group, and a P(escaper) per cell.
    method: str = "mean_ratio"
    #: count_model step 1: the group passes when ``rho`` is below this.
    max_rho: float = 0.3
    #: count_model step 1, also: the fitted escaper fraction must be below this,
    #: i.e. most cells are knocked down.
    max_escaper_fraction: float = 0.5
    #: count_model step 2: a cell is an escaper when P(escaper) reaches this.
    min_escaper_prob: float = 0.9


# ===========================================================================
# Cluster enrichment
# ===========================================================================


@dataclass
class EnrichmentConfig:
    """Enrichment/depletion of perturbations across cell-state clusters."""

    enabled: bool = True
    cluster_key: str = "leiden"
    controls: List[str] = field(default_factory=lambda: ["ntc", "other"])
    primary_control: str = "other"
    fdr_alpha: float = 0.05
    min_cells_per_target: int = 10
    min_cells_per_cluster: int = 20
    min_reference_cells: int = 10
    odds_pseudocount: float = 0.5
    stratify_by: Optional[str] = None
    guide_concordance: bool = True
    min_cells_per_guide: int = 5
    permutations: int = 1000
    top_n_report: int = 12


# ===========================================================================
# Regulome / modules
# ===========================================================================


@dataclass
class ProgramEnrichmentConfig:
    """Biological pathway enrichment and functional annotation for gene programs."""

    enabled: bool = True
    method: str = "ora"
    #: MSigDB release; the species suffix (.Hs / .Mm) comes from ``input.species``.
    msigdb_version: str = "2026.1"
    sources: List[str] = field(default_factory=lambda: ["hallmark", "reactome", "go_bp"])
    custom_gmt_files: Dict[str, str] = field(default_factory=dict)
    fdr_alpha: float = 0.05
    min_overlap: int = 2
    min_genes: int = 5
    max_genes: int = 1500
    top_terms_per_program: int = 5
    #: The MSigDB collections are downloaded at run time. When the download (or the
    #: enrichment) fails the stage normally continues with every program marked
    #: ``unannotated`` and records the error in the report and run warnings; set
    #: ``require: true`` to make such a failure abort the run instead.
    require: bool = False


@dataclass
class ModulesConfig:
    """Co-functional modules and co-regulated gene programs."""

    #: Optional stage, off by default; set ``enabled: true`` to run it.
    enabled: bool = False
    cluster_key: str = "leiden"
    gene_selection: str = "cluster_markers"
    n_marker_genes_per_cluster: int = 100
    marker_method: str = "wilcoxon"
    min_cells_per_perturbation: int = 20
    control: str = "ntc"
    program_correlation: str = "pearson"
    module_correlation: str = "spearman"
    linkage_method: str = "average"
    n_programs: Optional[int] = 4
    n_modules: Optional[int] = 9
    cluster_distance_threshold: Optional[float] = 0.7
    #: Per-cell program scoring is useful but expensive at very large scale.
    score_programs: bool = True
    hub_lfc_threshold: float = 0.5
    de_fdr_alpha: float = 0.05
    draw_networks: bool = True
    min_perturbations: int = 5
    min_genes: int = 10
    top_n_report: int = 12
    #: Biological pathway enrichment and functional annotation for gene programs.
    program_enrichment: ProgramEnrichmentConfig = field(default_factory=ProgramEnrichmentConfig)
    #: Effect matrix the modules are built from: ``pseudobulk`` (log2FC of each
    #: perturbation vs control) or ``regression`` (the membership regression's
    #: adjusted log2 effects and FDR; needs ``regression.enabled``, high-MOI only).
    effect_source: str = "pseudobulk"


# ===========================================================================
# PS score
# ===========================================================================


@dataclass
class PSScoreConfig:
    """Per-cell perturbation-response scoring through ``pertps``."""

    #: Optional stage, off by default; set ``enabled: true`` to run it.
    enabled: bool = False
    require: bool = False
    top_n_biomarkers: int = 100
    scale_factor: float = 3.0
    ps_threshold: float = 0.5
    expression_cut: str = "mean"
    expression_cut_quantile: float = 0.75
    min_cells_per_target: int = 10
    min_control_cells: int = 10
    top_n_report: int = 12
    compute_lda_umap: bool = True
    lda_n_pcs: int = 40
    lda_max_genes: Optional[int] = 5000
    lda_highlight_threshold: float = 0.8
    #: LARGE-mode LDA visualization is restricted to this many cells.
    #: The PS score itself does not need to use this subset.
    lda_large_max_cells: int = 200_000
    #: In LARGE mode choose the visualization subset with approximately
    #: perturbation/control-stratified sampling.
    lda_large_stratified: bool = True


# ===========================================================================
# lochNESS
# ===========================================================================


@dataclass
class LochnessConfig:
    """Local neighbourhood enrichment of each perturbation."""

    #: Optional stage, off by default; set ``enabled: true`` to run it.
    enabled: bool = False
    genotype_key: str = "target_gene"
    n_neighbors: int = 300
    n_pcs: int = 20
    use_rep: Optional[str] = None
    recompute_neighbors: bool = True
    min_cells_per_target: int = 10
    enrichment_cut: float = 0.5
    noise_delta: float = 0.0
    top_n_report: int = 12
    #: Number of perturbations processed together by LARGE implementations.
    target_chunk_size: int = 128
    #: Whether one ``lochness_<TARGET>`` column is added for every target.
    #: Fine for small screens, but impossible for 10k-target million-cell runs.
    store_all_target_scores: bool = True
    #: In AUTO/LARGE execution, individual target columns should not be written
    #: into obs above this many targets. ``lochness_self`` remains available.
    max_targets_in_obs: int = 500


# ===========================================================================
# Perturbation distance vs control
# ===========================================================================


@dataclass
class DistanceConfig:
    """Perturbation distance vs control analysis."""

    #: Optional stage, off by default; set ``enabled: true`` to run it.
    enabled: bool = False
    representation: str = "X_pca"
    primary_metric: str = "edistance"
    secondary_metric: Optional[str] = "mmd"
    min_cells: int = 30
    max_cells_per_target: int = 2000
    max_control_cells: int = 5000
    n_permutations: int = 1000
    random_seed: int = 123
    fdr_threshold: float = 0.05
    stratify_by: Optional[str] = None


# ===========================================================================
# Perturbation distance space
# ===========================================================================


@dataclass
class DistanceSpaceConfig:
    """Pairwise perturbation distance space analysis."""

    #: Optional stage, off by default; set ``enabled: true`` to run it.
    enabled: bool = False
    metric: str = "edistance"
    representation: str = "X_pca"
    n_components: int = 10
    nearest_neighbors: int = 10
    clustering: bool = True
    n_modules: Optional[int] = None
    cluster_distance_threshold: Optional[float] = None
    linkage_method: str = "average"
    min_cells: int = 30
    max_cells_per_target: int = 2000
    random_seed: int = 123


# ===========================================================================
# Master perturbation meta-analysis
# ===========================================================================


@dataclass
class MetaAnalysisConfig:
    """Master perturbation meta-analysis table."""

    enabled: bool = True


# ===========================================================================
# Visualization options
# ===========================================================================


@dataclass
class VisualizationConfig:
    """Advanced overview and perturbation distance visualization."""

    perturbation_atlas: bool = True
    ps_distance_map: bool = True
    perturbation_space: bool = True
    module_concordance: bool = True
    atlas_top_n: int = 50


# ===========================================================================
# Adaptive scaling
# ===========================================================================


@dataclass
class ScalingConfig:
    """Computational scaling policy.

    These options change how an analysis is executed, not what biological
    quantity is being measured.

    ``mode = auto``
        Automatically use LARGE implementations above the configured
        thresholds.

    ``mode = standard``
        Force the original implementations. Useful for regression testing but
        potentially unsafe for very large objects.

    ``mode = large``
        Force memory-aware implementations even below the normal threshold.
        This is useful for machines with limited RAM, unusually wide matrices,
        or screens containing thousands of perturbations.
    """

    mode: str = "auto"
    #: Main global cell-count trigger.
    large_n_cells: int = 1_000_000
    #: Modules/regulome may independently need LARGE handling because an
    #: enormous number of perturbations creates a large correlation matrix.
    large_n_perturbations: int = 5_000
    #: Maximum cells used for feature/marker discovery in LARGE mode.
    #: Full effect estimation still uses all cells.
    marker_max_cells: int = 200_000
    #: Gene chunk size for LARGE perturbation x gene calculations.
    effect_gene_chunk: int = 256
    #: Standard guide-calling chunk size.
    guide_chunk_size: int = 20_000
    #: A dense guide block larger than this many scalar values should instead
    #: use the sparse guide implementation.
    guide_max_dense_elements: int = 20_000_000
    #: Call Python garbage collection between expensive stages in LARGE mode.
    collect_between_stages: bool = True
    #: Log process resident memory when psutil is available.
    log_memory: bool = True
    #: Maximum rows of a huge table retained for HTML report assembly. The
    #: complete table remains written to disk.
    report_preview_rows: int = 500


# ===========================================================================
# Compute / hardware backend policy
# ===========================================================================


@dataclass
class ComputeConfig:
    """Hardware compute and backend execution policy.

    Controls whether CPU multiprocessing, threading, or optional GPU acceleration
    is selected across pipeline stages.

    Backend options
    ---------------
    ``auto``
        Selects CPU or GPU stage-by-stage based on dataset size, dense matrix
        dimensions, available GPU hardware, and installed optional libraries.

    ``cpu``
        Forces all stages to use CPU implementations with deterministic
        multiprocessing/multithreading. Never requires GPU packages.

    ``gpu``
        Requests GPU acceleration for supported stages (e.g. clustering, dense
        correlations, distance space) when safe and available; unsupported
        stages remain CPU.
    """

    backend: str = "auto"
    #: Global default worker count for CPU multiprocessing.
    n_jobs: int = 16
    #: GPU device index (0-indexed).
    gpu_device: int = 0
    #: Minimum number of cells in the dataset before GPU acceleration
    #: is considered for clustering/embedding stages in AUTO mode.
    gpu_min_cells: int = 200_000
    #: Minimum number of scalar elements in a dense matrix before GPU
    #: acceleration is considered for correlation or matrix multiplication.
    gpu_min_dense_elements: int = 50_000_000
    #: Maximum fractional safe memory limit on GPU to prevent OOM.
    gpu_memory_fraction: float = 0.80
    #: CPU multiprocessing backend engine: 'loky', 'multiprocessing', or 'threading'.
    cpu_parallel_backend: str = "loky"
    #: BLAS / OpenMP thread limit per worker process to prevent oversubscription.
    blas_threads_per_worker: int = 1
    #: Per-stage multiprocessing overrides. If null, inherits compute.n_jobs.
    distance_n_jobs: Optional[int] = None
    perturbation_n_jobs: Optional[int] = None
    enrichment_n_jobs: Optional[int] = None
    modules_n_jobs: Optional[int] = None
    lochness_n_jobs: Optional[int] = None
    #: Whether to log compute backend placement decisions per stage.
    log_backend_decisions: bool = True


# ===========================================================================
# Storage / data access policy
# ===========================================================================


@dataclass
class StorageConfig:
    """Storage, dataset backing, and worker data-sharing policy.

    Controls whether large AnnData datasets are loaded in-memory or accessed
    via disk backing, and whether representations are shared across workers.

    Modes
    -----
    ``auto``
        Automatically uses backed H5AD for datasets with cell count >=
        ``backed_threshold_cells`` if ``prefer_backed_h5ad`` is True.
        STANDARD datasets remain in-memory.

    ``in_memory``
        Forces all datasets to be fully loaded into memory.

    ``backed``
        Requests backed H5AD (read-only) for large expression matrices.
        Embeddings and metadata are retained in memory.
    """

    mode: str = "auto"
    backed_threshold_cells: int = 1_000_000
    prefer_backed_h5ad: bool = True
    keep_embeddings_in_memory: bool = True
    shared_worker_arrays: bool = True


# ===========================================================================
# Report
# ===========================================================================


@dataclass
class ReportConfig:
    """HTML report assembly."""

    title: str = "Perturb-seq analysis report"
    embed_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 150
    max_table_rows: int = 100


# ===========================================================================
# Outputs
# ===========================================================================


@dataclass
class OutputConfig:
    """Output files and artifact handling."""

    h5ad_name: str = "processed.h5ad"
    report_name: str = "report.html"
    #: Companion Markdown report written next to ``report_name`` (``None``
    #: disables it). Mirrors the HTML report's tables and figure references.
    report_markdown_name: Optional[str] = "report.md"
    write_unfiltered_h5ad: bool = True
    unfiltered_h5ad_name: Optional[str] = None
    large_file_dir: Optional[str] = None
    large_file_threshold_mb: float = 50.0
    merge_guides_into_h5ad: bool = True
    guide_obsm_key: str = "guide_counts"
    write_guide_h5ad: bool = False
    save_figures_pdf: bool = False
    archive: bool = True
    archive_name: Optional[str] = None
    archive_exclude: List[str] = field(default_factory=lambda: ["*.h5ad", "*.h5", "*.loom", "*.tar.gz"])
    write_guide_table: bool = True
    guide_table_name: Optional[str] = None
    guide_table_min_umi: int = 3


# ===========================================================================
# Basic QC stage: per-sample inputs
# ===========================================================================


@dataclass
class SampleConfig:
    """One GEM well / 10x library described under the top-level ``samples``.

    Only ``gex_h5`` or ``gex_mtx_dir`` is mandatory. Guide inputs are
    optional so GEX-only datasets and datasets whose guide libraries are not
    yet quantified run through the same code path.
    """

    #: Cell Ranger ``filtered_feature_bc_matrix.h5`` (preferred).
    gex_h5: Optional[str] = None
    #: Alternative: a 10x MTX directory.
    gex_mtx_dir: Optional[str] = None
    #: Identifier of the paired guide (feature-barcode) library.
    guide_library: Optional[str] = None
    #: Directory searched (recursively) for guide FASTQ files.
    guide_fastq_dir: Optional[str] = None
    #: Explicit list of guide FASTQ files (the read carrying barcode + spacer).
    guide_fastqs: Optional[List[str]] = None
    #: Per-sample glob overriding ``guides.fastq.read_pattern``.
    guide_fastq_pattern: Optional[str] = None
    #: Pre-computed guide count matrix (10x MTX dir or .h5 with guide features).
    guide_matrix: Optional[str] = None
    #: Neutral experimental-condition code (e.g. "COND1"). Not interpreted.
    condition_code: Optional[str] = None
    #: GEM well / capture identifier (e.g. "A"). Not interpreted.
    gem_well: Optional[str] = None
    #: Any further per-sample obs annotations (constant per sample).
    metadata: Dict[str, Any] = field(default_factory=dict)

    def gex_path(self) -> str:
        if self.gex_h5:
            return self.gex_h5
        if self.gex_mtx_dir:
            return self.gex_mtx_dir
        raise ValueError("sample has neither gex_h5 nor gex_mtx_dir")


# ===========================================================================
# Basic QC stage: expression thresholds
# ===========================================================================


@dataclass
class QCThresholdConfig:
    """Sample-aware expression-QC thresholds (flags only).

    Every threshold is resolved *per sample* and recorded. Two layers combine:

    absolute sanity bounds
        ``min_genes_floor``, ``min_counts_floor``, ``max_genes_ceiling``,
        ``max_counts_ceiling`` and the mitochondrial cap.

    robust sample-specific bounds (``method: mad``)
        ``median +/- n_mads * MAD`` computed per sample, optionally on the
        log1p scale. The lower bound can never fall below the floor and the
        upper bound can never exceed the ceiling.

    ``method: fixed`` uses the floors/ceilings directly.
    """

    method: str = "mad"  # mad | fixed
    n_mads: float = 3.0
    log_transform: bool = True
    #: Metrics receiving MAD bounds (subset of total_counts, n_genes_by_counts).
    mad_metrics: List[str] = field(default_factory=lambda: ["total_counts", "n_genes_by_counts"])
    flag_low: bool = True
    flag_high: bool = True
    min_genes_floor: Optional[int] = 200
    min_counts_floor: Optional[int] = 500
    max_genes_ceiling: Optional[int] = None
    max_counts_ceiling: Optional[int] = None
    #: Absolute mitochondrial cap (percent). ``None`` disables the mt flag.
    max_pct_mt: Optional[float] = 20.0
    #: Also derive ``median + n_mads * MAD`` for pct_counts_mt and use the
    #: stricter of the two bounds.
    max_pct_mt_mad: bool = False
    #: Per-condition mitochondrial caps keyed by ``obs[condition_key]``.
    max_pct_mt_by_condition: Dict[str, float] = field(default_factory=dict)
    condition_key: str = "condition_code"
    #: Haemoglobin cap (percent). ``None`` disables the hb flag.
    max_pct_hb: Optional[float] = None
    #: Per-sample overrides of resolved values. Keys: min_genes, max_genes,
    #: min_counts, max_counts, max_pct_mt, max_pct_hb.
    per_sample: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ===========================================================================
# Basic QC stage: doublet annotation
# ===========================================================================


@dataclass
class DoubletConfig:
    """Doublet detection run independently per sample. Annotation only.

    There is deliberately no ``remove`` switch: the basic QC stage stores
    ``doublet_score`` / ``predicted_doublet`` and never subsets on them.
    """

    enabled: bool = True
    method: str = "scrublet"
    #: ``None`` keeps the Scrublet default (0.05). Never derive this from a
    #: theoretical loading estimate; it is a prior, not a target.
    expected_doublet_rate: Optional[float] = None
    #: ``None`` lets Scrublet pick the threshold from the simulated-doublet
    #: score distribution. When automatic thresholding fails every cell is
    #: recorded as ``predicted_doublet = False`` and the failure is logged.
    threshold: Optional[float] = None
    sim_doublet_ratio: float = 2.0
    n_prin_comps: int = 30
    stdev_doublet_rate: float = 0.02
    #: Genes with fewer counts than this across the sample are ignored inside
    #: Scrublet (does not touch the stored matrix).
    min_gene_counts: int = 3


# ===========================================================================
# Basic QC stage: guide design / counting / multiplets
# ===========================================================================


@dataclass
class GuideDesignConfig:
    """Designed-guide reference table (xlsx/csv/tsv).

    Column names are auto-detected unless given explicitly. Every designed
    guide is retained in the reference, whether or not it is observed.
    """

    path: Optional[str] = None
    #: Sheet name or index for workbooks. ``None`` = first sheet.
    sheet: Optional[Union[str, int]] = None
    protospacer_column: Optional[str] = None
    target_column: Optional[str] = None
    guide_id_column: Optional[str] = None
    scaffold_column: Optional[str] = None
    #: Optional table mapping protospacer -> scaffold class when the design
    #: workbook lacks a scaffold column.
    scaffold_table: Optional[str] = None
    #: Case-insensitive regexes recognising control guides by target label.
    #: ``None`` falls back to ``guides.ntc_patterns``.
    control_patterns: Optional[List[str]] = None
    #: Template for synthetic guide ids when the table has no id column.
    #: Fields: target (sanitised), n (1-based index within target).
    id_format: str = "{target}_{n}"
    #: Uppercase protospacers before matching.
    uppercase: bool = True


@dataclass
class GuideFastqConfig:
    """Read-structure parameters for the streaming guide counter.

    Defaults describe 10x 5' feature-barcode reads where R1 carries
    ``[barcode][UMI][TSO][0-n G][protospacer][scaffold]``. The protospacer is
    located *relative to the scaffold anchor* rather than at a fixed offset so
    variable non-templated G runs do not lose reads.
    """

    #: Glob (recursive) used to find the read carrying barcode + spacer.
    read_pattern: str = "*_R1_*.fastq.gz"
    barcode_length: int = 16
    umi_length: int = 12
    protospacer_length: int = 20
    #: Scaffold class -> anchor sequence expected immediately after the
    #: protospacer. Classes are free-form labels (e.g. A / C).
    scaffolds: Dict[str, str] = field(default_factory=lambda: {"A": "GTTTAAGAGCTA", "C": "GTTTCAGAGCTA"})
    #: Earliest read position at which a scaffold anchor may start.
    anchor_search_start: int = 40
    #: Positional fallback: retry the exact spacer match shifted by up to this
    #: many bases (still an exact sequence match). 0 disables.
    position_shift: int = 1
    #: Sequence mismatches tolerated in the spacer. 0 = exact only (default).
    max_mismatches: int = 0
    #: Template-switch oligo, recorded as a diagnostic only.
    tso: Optional[str] = "TTTCTTATATGGG"
    #: Restrict UMI-level counting to barcodes of the paired GEX matrix.
    restrict_to_gex_barcodes: bool = True
    #: Regex stripped from GEX barcodes before comparison with read barcodes.
    barcode_suffix_regex: str = r"-\d+$"
    #: Process at most this many reads per file (subset validation).
    max_reads: Optional[int] = None
    #: Worker processes (one per FASTQ file). ``None`` = min(files, CPUs).
    n_workers: Optional[int] = None
    #: Reads accumulated before an intermediate UMI de-duplication pass.
    chunk_size: int = 2_000_000
    #: Keep one in N unmatched protospacers for the diagnostics table.
    unmatched_sample_rate: int = 50
    #: Number of top unmatched protospacers to report.
    unmatched_top_n: int = 50
    #: Count each designed protospacer separately per scaffold class, i.e. the
    #: count-matrix features are ``<guide_id>:<scaffold>`` (designed spacer x
    #: scaffold anchor). Required when one spacer is cloned behind more than one
    #: scaffold (e.g. an NTC filler used in several constructs) and it turns
    #: wrong-scaffold (chimeric) reads into explicit off-design features.
    scaffold_specific_features: bool = False


@dataclass
class HighMoiConfig:
    """Membership calling for high-MOI screens (``guides.assignment_mode: high_moi``).

    A cell is a member of every guide whose UMI count passes the call; the
    cells x guides and cells x targets membership matrices are stored in
    ``obsm`` and the highest-UMI targeting guide is written to the legacy
    ``obs['target_gene']`` / ``obs['guide_id']`` columns as the primary label.
    """

    #: ``threshold``: a guide is a member when ``umi >= min_umi`` and
    #: ``umi >= min_frac_of_top * top_umi`` of the cell. ``knee``: per cell, the
    #: largest drop in ``log1p(umi)`` between consecutive ranked guides (among
    #: guides with ``>= min_umi`` UMIs) separates members from background.
    method: str = "threshold"
    #: Minimum UMIs for a guide to be called in a cell.
    min_umi: int = 10
    #: ``threshold`` method only: a guide also needs this fraction of the cell's
    #: top guide UMI count (depth-aware ambient filter). 0 disables it.
    min_frac_of_top: float = 0.02
    #: Cells with more called guides than this are classed ``ambiguous``
    #: (doublet-like) and carry no membership.
    max_guides_per_cell: int = 30
    #: Cells with fewer called guides than this (but at least one) are
    #: ``unassigned``.
    min_guides_per_cell: int = 1
    #: ``obsm`` keys of the membership matrices.
    membership_obsm_key: str = "perturbation_membership"
    guide_membership_obsm_key: str = "guide_membership"
    #: Ranks summarised in the rank-ordered guide UMI profile (knee diagnostic).
    rank_profile_max_rank: int = 20
    #: Test every non-targeting guide as a negative-control pseudo-target in the
    #: membership-aware statistics (enrichment, distance), giving an empirical
    #: false-positive rate. Pseudo-targets are named ``NTC:<guide>``.
    ntc_pseudo_targets: bool = True


@dataclass
class GuideMultipletConfig:
    """Guide-derived multiplet flagging (annotation only)."""

    #: UMI threshold for calling a guide detected in a cell. ``None`` uses
    #: ``guides.detection_threshold``.
    detection_threshold: Optional[int] = None
    #: Optional depth-aware rule: a guide also needs at least this fraction of
    #: the cell's top guide UMI count to count as detected. ``None`` = absolute
    #: threshold only. Deep guide libraries push ambient guides over a small
    #: absolute threshold; this keeps the rule explicit and configurable.
    detection_min_fraction_of_top: Optional[float] = None
    #: Grid evaluated for ``tables/guide_detection_sensitivity.tsv`` (flags are
    #: recomputed at each setting for assessment only).
    sensitivity_thresholds: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    sensitivity_fractions: List[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.2])
    #: More detected guides than this within one scaffold class flags the cell.
    max_guides_per_scaffold: int = 1
    #: Expected detected guides per scaffold class for ``guide_structure_pass``.
    expected_guides_per_scaffold: int = 1
    #: Used when no scaffold classes are available. ``null`` disables the
    #: per-cell multiplet flag (high-MOI designs).
    expected_guides_per_cell: Optional[int] = 1
    #: Minimum fraction of reads in the majority scaffold for a guide's
    #: scaffold class to be inferred empirically.
    scaffold_purity_min: float = 0.9
    #: Minimum reads before a guide's scaffold class is inferred.
    scaffold_min_reads: int = 20


# ===========================================================================
# Full configuration
# ===========================================================================


@dataclass
class Config:
    """Complete pipeline configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    input: InputConfig = field(default_factory=InputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    guides: GuideConfig = field(default_factory=GuideConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    knockdown_filter: KnockdownFilterConfig = field(default_factory=KnockdownFilterConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    modules: ModulesConfig = field(default_factory=ModulesConfig)
    ps_score: PSScoreConfig = field(default_factory=PSScoreConfig)
    lochness: LochnessConfig = field(default_factory=LochnessConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    distance_space: DistanceSpaceConfig = field(default_factory=DistanceSpaceConfig)
    meta_analysis: MetaAnalysisConfig = field(default_factory=MetaAnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    # Central scaling policy.
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    # Hardware compute backend policy.
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    # Storage and data access policy.
    storage: StorageConfig = field(default_factory=StorageConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    #: Multi-sample (per GEM well) inputs for the basic QC stage. Keys are
    #: sample ids; values follow :class:`SampleConfig`. Empty = legacy input.
    samples: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Reject unknown per-sample keys at construction time, like every
        # other section, rather than only inside validate().
        self.resolved_samples()

    def resolved_samples(self) -> Dict[str, "SampleConfig"]:
        """Validate and build :class:`SampleConfig` objects from ``samples``."""
        out: Dict[str, SampleConfig] = {}
        for sid, raw in (self.samples or {}).items():
            if raw is None:
                raw = {}
            if not isinstance(raw, dict):
                raise ValueError(f"samples.{sid} must be a mapping")
            out[str(sid)] = _build(SampleConfig, raw, f"samples.{sid}")
        return out

    # Construction
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        """Build from a partial nested mapping, rejecting unknown keys."""
        return _build(cls, data or {}, path="")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load and validate a YAML configuration."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg

    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a nested plain dictionary."""
        return _asdict(self)

    def dump_yaml(self, path: Union[str, Path]) -> None:
        """Write the fully resolved configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)

    # Validation
    def validate(self) -> None:
        """Check internal consistency."""
        # ==============================================================
        # Input
        # ==============================================================
        inp = self.input
        if inp.mode not in ("auto", "mtx", "h5ad"):
            raise ValueError(f"input.mode must be one of 'auto', 'mtx', 'h5ad' (got {inp.mode!r})")
        if inp.species not in ("human", "mouse"):
            raise ValueError(f"input.species must be 'human' or 'mouse' (got {inp.species!r})")
        has_mtx = bool(inp.resolved_mtx_dirs())
        has_h5ad = bool(inp.h5ad)
        if inp.mode == "mtx" and not has_mtx:
            raise ValueError("input.mode is 'mtx' but input.mtx_dirs is empty")
        if inp.mode == "h5ad" and not has_h5ad:
            raise ValueError("input.mode is 'h5ad' but input.h5ad is not set")
        if inp.mode == "auto":
            if has_mtx and has_h5ad:
                raise ValueError("Both input.mtx_dirs and input.h5ad are set; set input.mode explicitly.")
            if not has_mtx and not has_h5ad and not self.samples:
                raise ValueError("No input given: set input.mtx_dirs, input.h5ad or samples.")
        if inp.guide_mtx_dirs:
            lanes = set(inp.resolved_mtx_dirs())
            guide_lanes = set(inp.guide_mtx_dirs)
            missing = lanes - guide_lanes
            if missing:
                raise ValueError(f"input.guide_mtx_dirs must cover every lane; missing {sorted(missing)}")
            extra = guide_lanes - lanes
            if extra:
                raise ValueError(f"input.guide_mtx_dirs contains lanes not in mtx_dirs: {sorted(extra)}")
        # ==============================================================
        # Basic QC stage (samples / stop_after / thresholds / doublets)
        # ==============================================================
        self._validate_basic_qc()
        # ==============================================================
        # Guides
        # ==============================================================
        guide_cfg = self.guides
        if guide_cfg.dominance_ratio < 1:
            raise ValueError("guides.dominance_ratio must be >= 1")
        if guide_cfg.min_umi < 0:
            raise ValueError("guides.min_umi must be >= 0")
        if guide_cfg.detection_threshold < 0:
            raise ValueError("guides.detection_threshold must be >= 0")
        if guide_cfg.target_feature_column is not None and not str(guide_cfg.target_feature_column).strip():
            raise ValueError("guides.target_feature_column must be a non-empty column name or null")
        if any(not str(value).strip() for value in guide_cfg.ignored_target_values):
            raise ValueError("guides.ignored_target_values may not contain empty values")
        if guide_cfg.unassigned_label == guide_cfg.ambiguous_label:
            raise ValueError("guides.unassigned_label and guides.ambiguous_label must be different")
        if guide_cfg.ntc_label in {guide_cfg.unassigned_label, guide_cfg.ambiguous_label}:
            raise ValueError("guides.ntc_label must differ from the unassigned and ambiguous labels")
        if guide_cfg.assignment_mode not in ("single_guide", "dual_guide_pair", "pair", "high_moi"):
            raise ValueError(
                "guides.assignment_mode must be 'single_guide', 'pair', 'dual_guide_pair' or 'high_moi', "
                f"got {guide_cfg.assignment_mode!r}"
            )
        hm = guide_cfg.high_moi
        if hm.method not in ("threshold", "knee"):
            raise ValueError(f"guides.high_moi.method must be 'threshold' or 'knee', got {hm.method!r}")
        if hm.min_umi < 1:
            raise ValueError("guides.high_moi.min_umi must be >= 1")
        if not (0 <= hm.min_frac_of_top < 1):
            raise ValueError("guides.high_moi.min_frac_of_top must be in [0, 1)")
        if hm.max_guides_per_cell < 1 or hm.min_guides_per_cell < 1:
            raise ValueError("guides.high_moi.max_guides_per_cell and min_guides_per_cell must be >= 1")
        if hm.min_guides_per_cell > hm.max_guides_per_cell:
            raise ValueError("guides.high_moi.min_guides_per_cell must not exceed max_guides_per_cell")
        if hm.rank_profile_max_rank < 2:
            raise ValueError("guides.high_moi.rank_profile_max_rank must be >= 2")
        if not hm.membership_obsm_key or not hm.guide_membership_obsm_key:
            raise ValueError("guides.high_moi obsm keys must be non-empty")
        if hm.membership_obsm_key == hm.guide_membership_obsm_key:
            raise ValueError("guides.high_moi.membership_obsm_key and guide_membership_obsm_key must differ")
        if self.output.guide_obsm_key in (hm.membership_obsm_key, hm.guide_membership_obsm_key):
            raise ValueError(
                "guides.high_moi membership obsm keys must differ from output.guide_obsm_key "
                f"({self.output.guide_obsm_key!r}): the raw guide counts merged into the processed h5ad would "
                "overwrite the membership matrix"
            )
        mp = guide_cfg.multiplet
        if mp.expected_guides_per_cell is not None and mp.expected_guides_per_cell < 1:
            raise ValueError("guides.multiplet.expected_guides_per_cell must be >= 1 or null")
        if guide_cfg.assignment_mode == "pair":
            guide_cfg.assignment_mode = "dual_guide_pair"
        if guide_cfg.pair_reference and not guide_cfg.pair_map_file:
            guide_cfg.pair_map_file = guide_cfg.pair_reference
        if guide_cfg.assignment_mode == "dual_guide_pair" and not guide_cfg.pair_assignment_primary:
            raise ValueError("guides.pair_assignment_primary must be true when assignment_mode is 'pair'")
        if guide_cfg.unresolved_pair_policy != "exclude":
            raise ValueError("guides.unresolved_pair_policy: only 'exclude' is implemented")
        if self.input.cell_id_format not in ("suffix", "prefix"):
            raise ValueError("input.cell_id_format must be 'suffix' or 'prefix'")
        if guide_cfg.ntc_partner_policy not in ("ambiguous", "provisional_target"):
            raise ValueError(
                "guides.ntc_partner_policy must be 'ambiguous' or 'provisional_target', "
                f"got {guide_cfg.ntc_partner_policy!r}"
            )
        if len(guide_cfg.scaffold_classes) != 2 or len(set(guide_cfg.scaffold_classes)) != 2:
            raise ValueError("guides.scaffold_classes must list exactly two distinct scaffold classes")
        if guide_cfg.assignment_mode == "dual_guide_pair" and guide_cfg.pair_map_file:
            if not Path(guide_cfg.pair_map_file).is_file():
                raise ValueError(f"guides.pair_map_file not found: {guide_cfg.pair_map_file}")
        # ==============================================================
        # Clustering
        # ==============================================================
        if self.cluster.n_top_genes < 1:
            raise ValueError("cluster.n_top_genes must be >= 1")
        if self.cluster.n_pcs < 2:
            raise ValueError("cluster.n_pcs must be >= 2")
        if self.cluster.n_neighbors < 2:
            raise ValueError("cluster.n_neighbors must be >= 2")
        # ==============================================================
        # Control definitions
        # ==============================================================
        valid_controls = {"ntc", "other"}
        bad = set(self.perturbation.controls) - valid_controls
        if bad:
            raise ValueError(f"perturbation.controls may only contain {sorted(valid_controls)}; got {sorted(bad)}")
        if not self.perturbation.controls:
            raise ValueError("perturbation.controls must not be empty")
        if self.perturbation.primary_control not in self.perturbation.controls:
            raise ValueError("perturbation.primary_control must occur in perturbation.controls")
        if not (0 < self.perturbation.fdr_alpha < 1):
            raise ValueError("perturbation.fdr_alpha must be in (0, 1)")
        if not (0 < self.perturbation.umap_background_fraction <= 1):
            raise ValueError("perturbation.umap_background_fraction must be in (0, 1]")
        # ==============================================================
        # Membership regression
        # ==============================================================
        reg = self.regression
        if reg.enabled and self.guides.assignment_mode != "high_moi":
            raise ValueError("regression.enabled needs guides.assignment_mode: high_moi")
        if reg.genes not in ("modules", "hvg", "all"):
            raise ValueError(f"regression.genes must be 'modules', 'hvg' or 'all' (got {reg.genes!r})")
        if reg.min_cells < 1:
            raise ValueError("regression.min_cells must be >= 1")
        if reg.ridge_alpha < 0:
            raise ValueError("regression.ridge_alpha must be >= 0")
        if reg.n_permutations < 1:
            raise ValueError("regression.n_permutations must be >= 1 (the FDR is permutation-based)")
        if not (0 < reg.fdr_alpha < 1):
            raise ValueError("regression.fdr_alpha must be in (0, 1)")
        if reg.fdr_scope not in ("target", "global"):
            raise ValueError(f"regression.fdr_scope must be 'target' or 'global' (got {reg.fdr_scope!r})")
        if reg.min_abs_log2fc < 0:
            raise ValueError("regression.min_abs_log2fc must be >= 0")
        # ==============================================================
        # Knockdown filter
        # ==============================================================
        k = self.knockdown_filter
        if k.mode not in ("pooled", "per_context", "any_context"):
            raise ValueError(f"knockdown_filter.mode must be 'pooled', 'per_context' or 'any_context' (got {k.mode!r})")
        if k.enabled and k.mode in ("per_context", "any_context") and not k.context_key:
            raise ValueError(f"knockdown_filter.mode {k.mode!r} needs knockdown_filter.context_key")
        if k.method not in ("mean_ratio", "count_model"):
            raise ValueError(f"knockdown_filter.method must be 'mean_ratio' or 'count_model' (got {k.method!r})")
        if not 0 < k.max_escaper_fraction <= 1:
            raise ValueError(
                f"knockdown_filter.max_escaper_fraction must be in (0, 1] (got {k.max_escaper_fraction!r})"
            )
        if not 0 < k.min_escaper_prob < 1:
            raise ValueError(f"knockdown_filter.min_escaper_prob must be in (0, 1) (got {k.min_escaper_prob!r})")
        for fld in ("max_mean_ratio", "max_mean_ratio_any", "max_cell_ratio", "max_rho"):
            val = getattr(k, fld)
            if val is not None and val <= 0:
                raise ValueError(f"knockdown_filter.{fld} must be > 0 (got {val!r})")
        if k.min_cells < 1 or k.min_control_cells < 1:
            raise ValueError("knockdown_filter.min_cells and min_control_cells must be >= 1")
        if k.min_pct_expressing_control <= 0:
            raise ValueError(
                "knockdown_filter.min_pct_expressing_control must be > 0, otherwise a "
                "control mean of zero makes the ratio undefined"
            )
        # ==============================================================
        # Enrichment
        # ==============================================================
        bad = set(self.enrichment.controls) - valid_controls
        if bad:
            raise ValueError(f"enrichment.controls may only contain {sorted(valid_controls)}; got {sorted(bad)}")
        if not self.enrichment.controls:
            raise ValueError("enrichment.controls must not be empty")
        if self.enrichment.primary_control not in self.enrichment.controls:
            raise ValueError("enrichment.primary_control must occur in enrichment.controls")
        if not (0 < self.enrichment.fdr_alpha < 1):
            raise ValueError("enrichment.fdr_alpha must be in (0, 1)")
        # ==============================================================
        # Modules
        # ==============================================================
        modules = self.modules
        if modules.control not in valid_controls:
            raise ValueError(f"modules.control must be one of {sorted(valid_controls)} (got {modules.control!r})")
        if modules.gene_selection not in ("cluster_markers", "hvg"):
            raise ValueError("modules.gene_selection must be 'cluster_markers' or 'hvg'")
        for name in ("program_correlation", "module_correlation"):
            value = getattr(modules, name)
            if value not in ("pearson", "spearman"):
                raise ValueError(f"modules.{name} must be 'pearson' or 'spearman'")
        if modules.linkage_method not in ("average", "complete", "single", "ward", "weighted"):
            raise ValueError("modules.linkage_method must be a supported scipy hierarchical linkage method")
        if not (0 < modules.de_fdr_alpha < 1):
            raise ValueError("modules.de_fdr_alpha must be in (0, 1)")
        if modules.effect_source not in ("pseudobulk", "regression"):
            raise ValueError(f"modules.effect_source must be 'pseudobulk' or 'regression' (got {modules.effect_source!r})")
        if modules.effect_source == "regression" and modules.enabled and not self.regression.enabled:
            raise ValueError("modules.effect_source 'regression' needs regression.enabled: true")
        for name in ("n_programs", "n_modules"):
            value = getattr(modules, name)
            if value is not None and value < 2:
                raise ValueError(f"modules.{name} must be >=2 or null")
            if value is None and modules.cluster_distance_threshold is None:
                raise ValueError(f"modules.{name} is null and modules.cluster_distance_threshold is null")
        pe_cfg = modules.program_enrichment
        if pe_cfg.enabled:
            if pe_cfg.method not in ("ora", "hypergeometric", "fisher"):
                raise ValueError(
                    f"modules.program_enrichment.method must be 'ora' or 'hypergeometric' (got {pe_cfg.method!r})"
                )
            if not (0 < pe_cfg.fdr_alpha < 1):
                raise ValueError("modules.program_enrichment.fdr_alpha must be in (0, 1)")
            if pe_cfg.min_overlap < 1:
                raise ValueError("modules.program_enrichment.min_overlap must be >= 1")
            if pe_cfg.min_genes < 1:
                raise ValueError("modules.program_enrichment.min_genes must be >= 1")
            if pe_cfg.max_genes < pe_cfg.min_genes:
                raise ValueError("modules.program_enrichment.max_genes must be >= min_genes")
            if pe_cfg.top_terms_per_program < 1:
                raise ValueError("modules.program_enrichment.top_terms_per_program must be >= 1")
            # Without this the run would only log a warning and leave KEGG out of the annotations.
            if self.input.species == "mouse" and "kegg" in pe_cfg.sources and "kegg" not in pe_cfg.custom_gmt_files:
                raise ValueError(
                    "MSigDB has no mouse KEGG collection: remove 'kegg' from "
                    "modules.program_enrichment.sources or give a GMT file in custom_gmt_files['kegg']"
                )
            if pe_cfg.custom_gmt_files:
                for src_name, path in pe_cfg.custom_gmt_files.items():
                    if not Path(path).is_file():
                        raise FileNotFoundError(f"Custom GMT file for source {src_name!r} not found: {path}")
        # ==============================================================
        # PS
        # ==============================================================
        ps = self.ps_score
        if ps.expression_cut not in ("mean", "median", "quantile"):
            raise ValueError("ps_score.expression_cut must be 'mean', 'median' or 'quantile'")
        if not (0 < ps.expression_cut_quantile < 1):
            raise ValueError("ps_score.expression_cut_quantile must be in (0, 1)")
        if ps.lda_n_pcs < 2:
            raise ValueError("ps_score.lda_n_pcs must be >= 2")
        if ps.lda_max_genes is not None and ps.lda_max_genes < 2:
            raise ValueError("ps_score.lda_max_genes must be >=2 or null")
        if ps.lda_large_max_cells < 10:
            raise ValueError("ps_score.lda_large_max_cells must be >=10")
        # ==============================================================
        # lochNESS
        # ==============================================================
        loch = self.lochness
        if loch.n_neighbors < 2:
            raise ValueError("lochness.n_neighbors must be >= 2")
        if loch.target_chunk_size < 1:
            raise ValueError("lochness.target_chunk_size must be >= 1")
        if loch.max_targets_in_obs < 1:
            raise ValueError("lochness.max_targets_in_obs must be >= 1")
        # ==============================================================
        # Distance
        # ==============================================================
        dist = self.distance
        if dist.primary_metric not in ("edistance", "mmd"):
            raise ValueError(f"distance.primary_metric must be 'edistance' or 'mmd' (got {dist.primary_metric!r})")
        if dist.secondary_metric is not None and dist.secondary_metric not in ("edistance", "mmd"):
            raise ValueError(
                f"distance.secondary_metric must be 'edistance', 'mmd', or null (got {dist.secondary_metric!r})"
            )
        if dist.min_cells < 1:
            raise ValueError("distance.min_cells must be >= 1")
        if dist.max_cells_per_target < 1:
            raise ValueError("distance.max_cells_per_target must be >= 1")
        if dist.max_control_cells < 1:
            raise ValueError("distance.max_control_cells must be >= 1")
        if dist.n_permutations < 0:
            raise ValueError("distance.n_permutations must be >= 0")
        if not (0 < dist.fdr_threshold < 1):
            raise ValueError("distance.fdr_threshold must be in (0, 1)")
        # ==============================================================
        # Distance Space
        # ==============================================================
        dist_space = self.distance_space
        if dist_space.metric not in ("edistance", "mmd"):
            raise ValueError(f"distance_space.metric must be 'edistance' or 'mmd' (got {dist_space.metric!r})")
        if dist_space.n_components < 1:
            raise ValueError("distance_space.n_components must be >= 1")
        if dist_space.nearest_neighbors < 1:
            raise ValueError("distance_space.nearest_neighbors must be >= 1")
        if dist_space.linkage_method not in ("average", "complete", "single", "ward", "weighted"):
            raise ValueError("distance_space.linkage_method must be a supported scipy linkage method")
        if dist_space.min_cells < 1:
            raise ValueError("distance_space.min_cells must be >= 1")
        if dist_space.max_cells_per_target < 1:
            raise ValueError("distance_space.max_cells_per_target must be >= 1")
        if dist_space.n_modules is not None and dist_space.n_modules < 2:
            raise ValueError("distance_space.n_modules must be >= 2 or null")
        # ==============================================================
        # Scaling
        # ==============================================================
        scaling = self.scaling
        if scaling.mode not in ("auto", "standard", "large"):
            raise ValueError(f"scaling.mode must be 'auto', 'standard' or 'large' (got {scaling.mode!r})")
        if scaling.large_n_cells < 1:
            raise ValueError("scaling.large_n_cells must be >= 1")
        if scaling.large_n_perturbations < 1:
            raise ValueError("scaling.large_n_perturbations must be >= 1")
        if scaling.marker_max_cells < 1:
            raise ValueError("scaling.marker_max_cells must be >= 1")
        if scaling.effect_gene_chunk < 1:
            raise ValueError("scaling.effect_gene_chunk must be >= 1")
        if scaling.guide_chunk_size < 1:
            raise ValueError("scaling.guide_chunk_size must be >= 1")
        if scaling.guide_max_dense_elements < 1:
            raise ValueError("scaling.guide_max_dense_elements must be >= 1")
        if scaling.report_preview_rows < 1:
            raise ValueError("scaling.report_preview_rows must be >= 1")
        # ==============================================================
        # Compute
        # ==============================================================
        comp = self.compute
        if comp.backend not in ("auto", "cpu", "gpu"):
            raise ValueError(f"compute.backend must be 'auto', 'cpu', or 'gpu' (got {comp.backend!r})")
        if comp.n_jobs < -1 or comp.n_jobs == 0:
            raise ValueError(f"compute.n_jobs must be >= 1 or -1 (got {comp.n_jobs})")
        if comp.gpu_device < 0:
            raise ValueError("compute.gpu_device must be >= 0")
        if comp.gpu_min_cells < 1:
            raise ValueError("compute.gpu_min_cells must be >= 1")
        if comp.gpu_min_dense_elements < 1:
            raise ValueError("compute.gpu_min_dense_elements must be >= 1")
        if not (0.0 < comp.gpu_memory_fraction <= 1.0):
            raise ValueError("compute.gpu_memory_fraction must be in (0, 1]")
        if comp.blas_threads_per_worker < 1:
            raise ValueError("compute.blas_threads_per_worker must be >= 1")
        if comp.cpu_parallel_backend not in ("loky", "multiprocessing", "threading", "process"):
            raise ValueError(
                "compute.cpu_parallel_backend must be 'loky', 'multiprocessing', 'threading', or 'process' "
                f"(got {comp.cpu_parallel_backend!r})"
            )
        for name in (
            "distance_n_jobs",
            "perturbation_n_jobs",
            "enrichment_n_jobs",
            "modules_n_jobs",
            "lochness_n_jobs",
        ):
            val = getattr(comp, name)
            if val is not None and (val < -1 or val == 0):
                raise ValueError(f"compute.{name} must be >= 1, -1, or null (got {val})")
        # ==============================================================
        # Storage
        # ==============================================================
        storage_cfg = self.storage
        if storage_cfg.mode not in ("auto", "in_memory", "backed"):
            raise ValueError(f"storage.mode must be one of 'auto', 'in_memory', 'backed' (got {storage_cfg.mode!r})")
        if storage_cfg.backed_threshold_cells < 1:
            raise ValueError("storage.backed_threshold_cells must be >= 1")
        # ==============================================================
        # Report / output
        # ==============================================================
        if self.report.figure_dpi < 1:
            raise ValueError("report.figure_dpi must be >=1")
        if self.report.max_table_rows < 1:
            raise ValueError("report.max_table_rows must be >=1")
        if self.output.large_file_threshold_mb < 0:
            raise ValueError("output.large_file_threshold_mb must be >=0")

    def _validate_basic_qc(self) -> None:
        """Validate the sections added for the basic QC stage."""
        if self.run.stop_after not in (None, "qc"):
            raise ValueError(f"run.stop_after must be null or 'qc' (got {self.run.stop_after!r})")
        samples = self.resolved_samples()
        if samples:
            if self.run.stop_after != "qc":
                raise ValueError(
                    "Top-level 'samples' inputs are currently supported only "
                    "with run.stop_after: qc (the basic QC stage). Downstream "
                    "stages expect a single assigned-perturbation object."
                )
            for sid, smp in samples.items():
                if not smp.gex_h5 and not smp.gex_mtx_dir:
                    raise ValueError(f"samples.{sid}: set gex_h5 or gex_mtx_dir")
                if smp.gex_h5 and smp.gex_mtx_dir:
                    raise ValueError(f"samples.{sid}: set only one of gex_h5 / gex_mtx_dir")
                n_guide_sources = sum(bool(x) for x in (smp.guide_fastq_dir, smp.guide_fastqs, smp.guide_matrix))
                if n_guide_sources > 1:
                    raise ValueError(
                        f"samples.{sid}: choose one guide source (guide_fastq_dir | guide_fastqs | guide_matrix)"
                    )
                for key in ("condition_code", "gem_well", "guide_library"):
                    val = getattr(smp, key)
                    if val is not None and not isinstance(val, (str, int)):
                        raise ValueError(f"samples.{sid}.{key} must be a scalar")
        thr = self.qc.thresholds
        if thr.method not in ("mad", "fixed"):
            raise ValueError(f"qc.thresholds.method must be 'mad' or 'fixed' (got {thr.method!r})")
        if thr.n_mads <= 0:
            raise ValueError("qc.thresholds.n_mads must be > 0")
        allowed_metrics = {"total_counts", "n_genes_by_counts"}
        bad = set(thr.mad_metrics) - allowed_metrics
        if bad:
            raise ValueError(
                f"qc.thresholds.mad_metrics has unsupported entries {sorted(bad)}; allowed: {sorted(allowed_metrics)}"
            )
        for key, val in thr.max_pct_mt_by_condition.items():
            if val is not None and not (0 <= float(val) <= 100):
                raise ValueError(f"qc.thresholds.max_pct_mt_by_condition[{key!r}] must be in [0, 100]")
        if thr.max_pct_mt is not None and not (0 <= thr.max_pct_mt <= 100):
            raise ValueError("qc.thresholds.max_pct_mt must be in [0, 100]")
        allowed_override = {"min_genes", "max_genes", "min_counts", "max_counts", "max_pct_mt", "max_pct_hb"}
        for sid, over in thr.per_sample.items():
            if not isinstance(over, dict):
                raise ValueError(f"qc.thresholds.per_sample.{sid} must be a mapping")
            bad = set(over) - allowed_override
            if bad:
                raise ValueError(
                    f"qc.thresholds.per_sample.{sid} has unknown keys {sorted(bad)}; "
                    f"allowed: {sorted(allowed_override)}"
                )
        dbl = self.qc.doublets
        if dbl.method != "scrublet":
            raise ValueError("qc.doublets.method must be 'scrublet'")
        if dbl.expected_doublet_rate is not None and not (0 < dbl.expected_doublet_rate < 1):
            raise ValueError("qc.doublets.expected_doublet_rate must be in (0, 1)")
        if dbl.threshold is not None and not (0 < dbl.threshold < 1):
            raise ValueError("qc.doublets.threshold must be in (0, 1)")
        if dbl.n_prin_comps < 2:
            raise ValueError("qc.doublets.n_prin_comps must be >= 2")
        g = self.guides
        if g.source not in ("auto", "none", "fastq", "matrix"):
            raise ValueError(f"guides.source must be one of 'auto', 'none', 'fastq', 'matrix' (got {g.source!r})")
        fq = g.fastq
        for key in ("barcode_length", "umi_length", "protospacer_length"):
            if getattr(fq, key) <= 0:
                raise ValueError(f"guides.fastq.{key} must be > 0")
        if fq.position_shift < 0:
            raise ValueError("guides.fastq.position_shift must be >= 0")
        if fq.max_mismatches not in (0, 1):
            raise ValueError("guides.fastq.max_mismatches must be 0 or 1")
        if not fq.scaffolds:
            raise ValueError("guides.fastq.scaffolds must define at least one scaffold anchor")
        for name, anchor_seq in fq.scaffolds.items():
            if not anchor_seq or set(str(anchor_seq).upper()) - set("ACGTN"):
                raise ValueError(f"guides.fastq.scaffolds[{name!r}] must be a nucleotide string")
        if fq.chunk_size <= 0:
            raise ValueError("guides.fastq.chunk_size must be > 0")
        if fq.max_reads is not None and fq.max_reads <= 0:
            raise ValueError("guides.fastq.max_reads must be > 0 or null")
        mp = g.multiplet
        if mp.detection_threshold is not None and mp.detection_threshold < 1:
            raise ValueError("guides.multiplet.detection_threshold must be >= 1")
        if mp.max_guides_per_scaffold < 1:
            raise ValueError("guides.multiplet.max_guides_per_scaffold must be >= 1")
        if mp.detection_min_fraction_of_top is not None and not (0 < mp.detection_min_fraction_of_top <= 1):
            raise ValueError("guides.multiplet.detection_min_fraction_of_top must be in (0, 1] or null")
        if not (0 < mp.scaffold_purity_min <= 1):
            raise ValueError("guides.multiplet.scaffold_purity_min must be in (0, 1]")
        if samples and g.design.path is None:
            uses_fastq = g.source == "fastq" or any((s.guide_fastq_dir or s.guide_fastqs) for s in samples.values())
            if uses_fastq and g.source != "none":
                raise ValueError(
                    "Guide FASTQ counting requires guides.design.path (the designed-guide reference table)."
                )

    # Convenience
    @property
    def outdir(self) -> Path:
        """Run output directory."""
        return Path(self.run.outdir)

    def resolved_mode(self) -> str:
        """Effective input mode after ``auto`` resolution."""
        if self.samples:
            return "samples"
        if self.input.mode != "auto":
            return self.input.mode
        return "mtx" if self.input.resolved_mtx_dirs() else "h5ad"

    # Adaptive execution API
    def use_large_mode(self, n_cells: int, n_perturbations: Optional[int] = None) -> bool:
        """Return whether LARGE implementations should be used.

        Explicit mode selection has highest priority.

        ``large``
            Always True.

        ``standard``
            Always False.

        ``auto``
            True when either the global cell threshold or, when supplied, the
            perturbation-count threshold is reached.

        Examples
        --------
        Replogle:

            cfg.use_large_mode(310_385)
            -> False

        KOLF:

            cfg.use_large_mode(2_659_209)
            -> True

        10x Flex 1M:

            cfg.use_large_mode(1_233_421)
            -> True

        Smaller dataset forced to scalable algorithms:

            scaling.mode: large

            cfg.use_large_mode(150_000)
            -> True

        Million-cell regression test forced through old implementation:

            scaling.mode: standard

            cfg.use_large_mode(2_659_209)
            -> False

        The final case is permitted deliberately but may exhaust RAM.
        """
        mode = self.scaling.mode
        if mode == "large":
            return True
        if mode == "standard":
            return False
        if n_cells >= self.scaling.large_n_cells:
            return True
        if n_perturbations is not None and n_perturbations >= self.scaling.large_n_perturbations:
            return True
        return False

    def execution_mode(self, n_cells: int, n_perturbations: Optional[int] = None) -> str:
        """Return ``'standard'`` or ``'large'`` for logging/provenance."""
        return "large" if self.use_large_mode(n_cells, n_perturbations) else "standard"


# ===========================================================================
# Dict -> dataclass
# ===========================================================================


def _build(cls: type, data: Dict[str, Any], path: str) -> Any:
    """Recursively instantiate nested dataclasses and reject unknown keys."""
    known = {item.name: item for item in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = path or "<root>"
        raise ValueError(f"Unknown config key(s) under {where}: {sorted(unknown)}. Valid keys: {sorted(known)}")
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        field_type = hints.get(name)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _build(field_type, value, (f"{path}.{name}" if path else name))
        else:
            kwargs[name] = copy.deepcopy(value)
    return cls(**kwargs)


# ===========================================================================
# Dataclass -> dict
# ===========================================================================


def _asdict(obj: Any) -> Any:
    """Recursively turn dataclasses into YAML-safe structures."""
    if is_dataclass(obj):
        return {item.name: _asdict(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, dict):
        return {key: _asdict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ===========================================================================
# Fully resolved defaults
# ===========================================================================


DEFAULTS: Dict[str, Any] = Config().to_dict()
