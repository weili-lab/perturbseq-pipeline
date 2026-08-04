"""Configuration schema for the Perturb-seq pipeline.

A run is fully described by one YAML file. Every threshold that appeared as a
magic number in the prototype notebooks is a named key here with a documented
default. User YAML is deep-merged onto :data:`DEFAULTS`, so a user file only
needs to specify what it changes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

import yaml

# ---------------------------------------------------------------------------
# Section schemas
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Top-level run identity and output location."""

    name: str = "perturbseq_run"
    outdir: str = "results"
    seed: int = 0


@dataclass
class InputConfig:
    """Where the data comes from.

    Two entry points are supported:

    ``mtx``
        One or more 10x MTX directories (``barcodes.tsv.gz``,
        ``features.tsv.gz``, ``matrix.mtx.gz``) holding both gene-expression and
        guide features. Provide them via :attr:`mtx_dirs` as a
        ``{lane_id: path}`` mapping, or as a plain list (lane IDs are then
        derived from the directory names).

    ``h5ad``
        An existing ``.h5ad``. Guide information may live in ``var`` (guide
        features alongside genes), in a companion guide ``.h5ad``
        (:attr:`guide_h5ad`), or as a pre-computed per-cell label column in
        ``obs`` (:attr:`guide_obs_column`, e.g. ``genotype``).
    """

    mode: str = "auto"  # auto | mtx | h5ad
    mtx_dirs: Union[Dict[str, str], List[str], None] = None
    h5ad: Optional[str] = None
    guide_h5ad: Optional[str] = None
    guide_obs_column: Optional[str] = None
    #: ``var`` column holding the feature class in 10x-style data.
    feature_type_column: str = "feature_types"
    gex_feature_type: str = "Gene Expression"
    guide_feature_types: List[str] = field(
        default_factory=lambda: ["Custom", "CRISPR Guide Capture"]
    )
    #: 10x has two name columns; ``gene_symbols`` matches the prototype
    #: notebooks (``sc.read_10x_mtx`` default).
    var_names: str = "gene_symbols"
    cache_mtx: bool = True
    #: h5ad mode only. Name of the layer holding raw counts, when they are not
    #: in ``X`` (Seurat exports often put counts in ``X`` and log-normalized
    #: values in ``layers['logcounts']``).
    counts_layer: Optional[str] = None
    #: h5ad mode only. Name of a layer that already holds log-normalized values.
    #: When set, the pipeline uses it instead of re-normalizing.
    normalized_layer: Optional[str] = None

    def resolved_mtx_dirs(self) -> Dict[str, str]:
        """Return ``{lane_id: path}`` regardless of how the user spelled it."""
        if not self.mtx_dirs:
            return {}
        if isinstance(self.mtx_dirs, dict):
            return dict(self.mtx_dirs)
        out: Dict[str, str] = {}
        for p in self.mtx_dirs:
            lane = Path(p).name
            for prefix in ("filtered_feature_bc_matrix_", "raw_feature_bc_matrix_"):
                if lane.startswith(prefix):
                    lane = lane[len(prefix) :]
            out[lane or Path(p).name] = p
        return out


@dataclass
class MetadataConfig:
    """Per-lane sample metadata.

    Required whenever a run spans more than one lane; every column is merged
    into ``adata.obs`` and travels with the output ``.h5ad``.
    """

    file: Optional[str] = None
    #: Column in the metadata file that matches the lane ID.
    key_column: str = "lane_id"
    #: Fail (rather than warn) when a multi-lane run has no metadata file.
    require_for_multilane: bool = True


@dataclass
class QCConfig:
    """Standard single-cell QC thresholds."""

    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    #: Applied after the QC figures are drawn, so the plots show the raw picture.
    min_genes_final: int = 1000
    max_pct_mt: float = 20.0
    max_pct_hb: Optional[float] = None
    min_counts_per_cell: Optional[int] = None
    mito_prefix: str = "MT-"
    ribo_prefix: List[str] = field(default_factory=lambda: ["RPS", "RPL"])
    hb_pattern: str = "^HB[^(P)]"


@dataclass
class GuideConfig:
    """Guide-calling rules.

    A cell is assigned to its top guide when that guide has at least
    :attr:`min_umi` counts *and* dominates the runner-up by
    :attr:`dominance_ratio`. Otherwise the cell is ``ambiguous``; with no guide
    counts at all it is ``unassigned``. Both categories are reported, never
    silently dropped.
    """

    min_umi: int = 3
    dominance_ratio: float = 2.0
    #: Guide counts above this are considered "detected" for MOI statistics.
    detection_threshold: int = 3
    #: Regex whose first group is the target gene. When null, the guide ID is
    #: split on :attr:`target_split_delims` and the first field is taken
    #: (``AFF4_P1P2_1`` and ``AFF4-P1P2.2`` both give ``AFF4``).
    target_regex: Optional[str] = None
    target_split_delims: List[str] = field(default_factory=lambda: ["_", "-", "."])
    #: Case-insensitive regexes marking non-targeting control guides.
    ntc_patterns: List[str] = field(
        default_factory=lambda: [
            r"^non[-_.]?targeting",
            r"^non$",
            r"^ntc",
            r"scramble",
            r"^safe[-_.]?harbor",
        ]
    )
    #: Labels used in ``obs['target_gene']`` for the two failure modes.
    unassigned_label: str = "unassigned"
    ambiguous_label: str = "ambiguous"
    ntc_label: str = "non-targeting"


@dataclass
class ClusterConfig:
    """Normalization, embedding and clustering."""

    target_sum: Optional[float] = None  # None => median library size
    n_top_genes: int = 3000
    n_pcs: int = 50
    n_neighbors: int = 15
    leiden_resolution: float = 1.0
    umap_min_dist: float = 0.5
    #: Set to an ``obs`` column (e.g. ``lane_id``) to run Harmony batch
    #: correction; requires the ``harmony`` extra.
    batch_key: Optional[str] = None
    regress_out: List[str] = field(default_factory=list)
    scale_max_value: Optional[float] = 10.0


@dataclass
class PerturbationConfig:
    """Perturbation-strength testing.

    For every target gene that is also measured in the expression matrix, the
    gene's own expression in perturbed cells is compared against control cells.
    Both control definitions are reported side by side (see CLAUDE.md).
    """

    #: ``ntc``  = cells carrying non-targeting guides.
    #: ``other`` = cells assigned to a *different* target gene.
    controls: List[str] = field(default_factory=lambda: ["ntc", "other"])
    #: Which control drives ranking, the effective-perturbation call and the
    #: representative figures. Falls back to the other one if unavailable.
    primary_control: str = "ntc"
    min_cells_per_target: int = 10
    min_control_cells: int = 10
    #: A gene that is not expressed in control cells cannot be shown to be
    #: knocked down, and its fold change is numerically meaningless. Targets
    #: whose control cells fall below this detection rate (percent of control
    #: cells with non-zero expression) are reported as untestable instead.
    min_pct_expressing_control: float = 1.0
    fdr_alpha: float = 0.05
    #: A target is called effectively perturbed at FDR < alpha *and* a log
    #: fold-change below this (i.e. expression genuinely lower).
    max_log2fc_for_hit: float = 0.0
    #: Number of strongest effects shown inline in the report; every target
    #: still gets its figures written to the per-gene folder.
    top_n_report: int = 12
    #: Subsample fraction of background cells drawn in per-target UMAPs.
    umap_background_fraction: float = 0.1


@dataclass
class ReportConfig:
    """HTML report assembly."""

    title: str = "Perturb-seq analysis report"
    #: Inline every figure as base64 so the HTML is a single portable file.
    embed_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 150
    max_table_rows: int = 100


@dataclass
class OutputConfig:
    """Where deliverables land.

    Large artifacts (the processed ``.h5ad`` above
    :attr:`large_file_threshold_mb`) are moved to :attr:`large_file_dir` when
    set — on Colab that is the Drive folder named in CLAUDE.md — so they never
    end up staged for git.
    """

    h5ad_name: str = "processed.h5ad"
    report_name: str = "report.html"
    large_file_dir: Optional[str] = None
    large_file_threshold_mb: float = 50.0
    #: Also write the guide count matrix as its own ``.h5ad``.
    write_guide_h5ad: bool = True
    save_figures_pdf: bool = False
    #: Bundle the run outputs into a single ``.tar.gz`` for sharing. The
    #: matrices are excluded by default (see :attr:`archive_exclude`), so the
    #: archive stays small enough to attach to an email or a GitHub release.
    archive: bool = True
    #: Archive filename. When null, ``<run.name>_results.tar.gz`` is used.
    archive_name: Optional[str] = None
    #: Glob patterns (matched against paths relative to the run directory)
    #: excluded from the archive.
    archive_exclude: List[str] = field(
        default_factory=lambda: ["*.h5ad", "*.h5", "*.loom", "*.tar.gz"]
    )


@dataclass
class Config:
    """The full run configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    input: InputConfig = field(default_factory=InputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    guides: GuideConfig = field(default_factory=GuideConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        """Build a config from a (partial) nested dict, validating keys."""
        return _build(cls, data or {}, path="")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load a YAML file into a validated :class:`Config`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain nested dict."""
        return _asdict(self)

    def dump_yaml(self, path: Union[str, Path]) -> None:
        """Write the fully-resolved configuration next to the run outputs."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Check internal consistency; raises :class:`ValueError` on problems."""
        inp = self.input
        if inp.mode not in ("auto", "mtx", "h5ad"):
            raise ValueError(
                f"input.mode must be one of 'auto', 'mtx', 'h5ad' (got {inp.mode!r})"
            )

        has_mtx = bool(inp.resolved_mtx_dirs())
        has_h5ad = bool(inp.h5ad)
        if inp.mode == "mtx" and not has_mtx:
            raise ValueError("input.mode is 'mtx' but input.mtx_dirs is empty")
        if inp.mode == "h5ad" and not has_h5ad:
            raise ValueError("input.mode is 'h5ad' but input.h5ad is not set")
        if inp.mode == "auto":
            if has_mtx and has_h5ad:
                raise ValueError(
                    "Both input.mtx_dirs and input.h5ad are set; "
                    "set input.mode explicitly to choose one."
                )
            if not has_mtx and not has_h5ad:
                raise ValueError(
                    "No input given: set input.mtx_dirs (10x MTX mode) "
                    "or input.h5ad (h5ad mode)."
                )

        if self.guides.dominance_ratio < 1:
            raise ValueError("guides.dominance_ratio must be >= 1")
        if self.guides.min_umi < 0:
            raise ValueError("guides.min_umi must be >= 0")

        valid_controls = {"ntc", "other"}
        bad = set(self.perturbation.controls) - valid_controls
        if bad:
            raise ValueError(
                f"perturbation.controls may only contain {sorted(valid_controls)}; "
                f"got extra {sorted(bad)}"
            )
        if not self.perturbation.controls:
            raise ValueError("perturbation.controls must not be empty")
        if self.perturbation.primary_control not in self.perturbation.controls:
            raise ValueError(
                f"perturbation.primary_control ({self.perturbation.primary_control!r}) "
                f"must be one of perturbation.controls ({self.perturbation.controls})"
            )

        if not 0 < self.perturbation.fdr_alpha < 1:
            raise ValueError("perturbation.fdr_alpha must be in (0, 1)")
        if not 0 < self.perturbation.umap_background_fraction <= 1:
            raise ValueError("perturbation.umap_background_fraction must be in (0, 1]")

    # -- convenience --------------------------------------------------------

    @property
    def outdir(self) -> Path:
        return Path(self.run.outdir)

    def resolved_mode(self) -> str:
        """The effective input mode after ``auto`` resolution."""
        if self.input.mode != "auto":
            return self.input.mode
        return "mtx" if self.input.resolved_mtx_dirs() else "h5ad"


# ---------------------------------------------------------------------------
# Dict <-> dataclass helpers
# ---------------------------------------------------------------------------


def _build(cls: type, data: Dict[str, Any], path: str) -> Any:
    """Recursively instantiate nested dataclasses, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = path or "<root>"
        raise ValueError(
            f"Unknown config key(s) under {where}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    # ``from __future__ import annotations`` makes ``field.type`` a string, so
    # resolve the real classes before testing for nested dataclasses.
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = copy.deepcopy(value)
    return cls(**kwargs)


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


#: The complete default configuration as a nested dict (used to write
#: ``config/default.yaml`` and as the documentation source of truth).
DEFAULTS: Dict[str, Any] = Config().to_dict()
