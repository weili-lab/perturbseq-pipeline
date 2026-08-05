"""Data loading: 10x MTX directories and ``.h5ad`` files, plus sample metadata.

The pipeline has two entry points (see :mod:`perturbseq_pipeline.config`), and
both converge on the same pair of objects:

``expr``
    An :class:`~anndata.AnnData` of gene-expression features, raw counts in
    ``X`` and in ``layers['counts']``.
``guides``
    An :class:`~anndata.AnnData` of guide features over the same cells, raw
    counts in ``X`` — or ``None`` when the input only carries a pre-computed
    per-cell guide label (in which case that label is copied into
    ``expr.obs['guide_id_raw']``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .config import Config

logger = logging.getLogger(__name__)

#: ``obs`` column holding the lane / library identifier.
LANE_KEY = "lane_id"
#: ``obs`` column holding a pre-computed per-cell guide label, when the input
#: supplies one instead of a guide count matrix.
RAW_GUIDE_LABEL = "guide_id_raw"


@dataclass
class LoadedData:
    """Result of :func:`load_data`."""

    expr: ad.AnnData
    guides: Optional[ad.AnnData]
    #: How guide information arrived: ``matrix`` (guide counts available) or
    #: ``obs_label`` (only a pre-computed per-cell label).
    guide_source: str
    #: ``{lane_id: source path}`` for provenance in the report.
    lanes: Dict[str, str]

    @property
    def n_lanes(self) -> int:
        return len(self.lanes)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_data(cfg: Config) -> LoadedData:
    """Load the input described by ``cfg`` and attach sample metadata."""
    mode = cfg.resolved_mode()
    logger.info("Loading input in '%s' mode", mode)
    data = _load_mtx(cfg) if mode == "mtx" else _load_h5ad(cfg)
    if mode == "h5ad":
        data.expr = apply_layer_choices(data.expr, cfg)

    data.expr = attach_sample_metadata(data.expr, cfg, n_lanes=data.n_lanes)
    if data.guides is not None:
        shared = [c for c in data.expr.obs.columns if c not in data.guides.obs.columns]
        for col in shared:
            data.guides.obs[col] = data.expr.obs[col].reindex(data.guides.obs_names)

    logger.info(
        "Loaded %d cells x %d genes (%s guide features) from %d lane(s)",
        data.expr.n_obs,
        data.expr.n_vars,
        data.guides.n_vars if data.guides is not None else "no",
        data.n_lanes,
    )
    return data


# ---------------------------------------------------------------------------
# 10x MTX mode
# ---------------------------------------------------------------------------


def _load_mtx(cfg: Config) -> LoadedData:
    """Read one or more 10x MTX directories and concatenate them by lane."""
    lanes = cfg.input.resolved_mtx_dirs()
    if not lanes:
        raise ValueError("input.mtx_dirs is empty")

    per_lane = []
    for lane_id, path in lanes.items():
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"MTX directory for lane {lane_id!r} not found: {p}")
        _check_mtx_dir(p, lane_id)
        logger.info("Reading lane %s from %s", lane_id, p)
        a = sc.read_10x_mtx(
            p,
            var_names=cfg.input.var_names,
            cache=cfg.input.cache_mtx,
            gex_only=False,
        )
        a.var_names_make_unique()
        per_lane.append(a)

    if len(per_lane) == 1:
        adata = per_lane[0]
        adata.obs[LANE_KEY] = pd.Categorical([next(iter(lanes))] * adata.n_obs)
    else:
        _check_matching_vars(per_lane, list(lanes))
        var_backup = per_lane[0].var.copy()
        adata = sc.concat(
            per_lane,
            label=LANE_KEY,
            keys=list(lanes),
            index_unique="-",
            merge="same",
        )
        # ``sc.concat`` keeps only columns identical across objects; restoring
        # from lane 1 guarantees feature_types/gene_ids survive.
        adata.var = var_backup.loc[adata.var_names]

    adata.var_names_make_unique()
    expr, guides = split_features(adata, cfg)
    return LoadedData(expr=expr, guides=guides, guide_source="matrix", lanes=dict(lanes))


def _check_mtx_dir(path: Path, lane_id: str) -> None:
    """Fail early with a readable message when a 10x directory is incomplete."""
    required = ("barcodes.tsv", "features.tsv", "matrix.mtx")
    missing = [
        stem
        for stem in required
        if not (path / stem).is_file() and not (path / f"{stem}.gz").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Lane {lane_id!r} at {path} is not a 10x MTX directory; "
            f"missing {missing} (with or without .gz)."
        )


def _check_matching_vars(objs, lane_ids) -> None:
    """All lanes must share one feature reference before concatenation."""
    ref = objs[0].var_names
    for lane_id, a in zip(lane_ids[1:], objs[1:]):
        if len(a.var_names) != len(ref) or not (a.var_names == ref).all():
            raise ValueError(
                f"Lane {lane_id!r} has a different feature set than lane "
                f"{lane_ids[0]!r} ({a.n_vars} vs {len(ref)} features). All lanes "
                "must be quantified against the same reference and guide library."
            )


# ---------------------------------------------------------------------------
# h5ad mode
# ---------------------------------------------------------------------------


def _load_h5ad(cfg: Config) -> LoadedData:
    """Read an ``.h5ad`` and recover its guide information.

    Three layouts are supported, checked in this order:

    1. guide features present in ``var`` (``feature_types`` column);
    2. a companion guide ``.h5ad`` given by ``input.guide_h5ad``;
    3. a pre-computed per-cell label column named by ``input.guide_obs_column``.
    """
    path = Path(cfg.input.h5ad)
    if not path.is_file():
        raise FileNotFoundError(f"Input h5ad not found: {path}")
    logger.info("Reading %s", path)
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    lanes = _lanes_from_obs(adata, cfg)

    ftype_col = cfg.input.feature_type_column
    has_guide_vars = (
        ftype_col in adata.var.columns
        and adata.var[ftype_col].isin(cfg.input.guide_feature_types).any()
    )

    if has_guide_vars:
        logger.info("Guide features found in var['%s']", ftype_col)
        expr, guides = split_features(adata, cfg)
        return LoadedData(expr, guides, "matrix", lanes)

    if cfg.input.guide_h5ad:
        gpath = Path(cfg.input.guide_h5ad)
        if not gpath.is_file():
            raise FileNotFoundError(f"Guide h5ad not found: {gpath}")
        logger.info("Reading companion guide matrix from %s", gpath)
        guides = sc.read_h5ad(gpath)
        guides.var_names_make_unique()
        shared = adata.obs_names.intersection(guides.obs_names)
        if len(shared) == 0:
            raise ValueError(
                "No cell barcodes are shared between input.h5ad and "
                "input.guide_h5ad. Check that both come from the same run."
            )
        if len(shared) < adata.n_obs:
            logger.warning(
                "Only %d/%d cells of the expression matrix are present in the "
                "guide matrix; restricting to the intersection.",
                len(shared),
                adata.n_obs,
            )
        expr = adata[shared].copy()
        guides = guides[shared].copy()
        _ensure_counts_layer(expr)
        return LoadedData(expr, guides, "matrix", lanes)

    if cfg.input.guide_table:
        labels = read_guide_table(
            cfg,
            adata.obs_names,
            adata.obs[LANE_KEY].astype(str) if LANE_KEY in adata.obs else None,
        )
        adata.obs[RAW_GUIDE_LABEL] = labels.to_numpy()
        _ensure_counts_layer(adata)
        return LoadedData(adata, None, "obs_label", lanes)

    col = cfg.input.guide_obs_column
    if col:
        if col not in adata.obs.columns:
            raise ValueError(
                f"input.guide_obs_column={col!r} is not a column of obs. "
                f"Available: {sorted(adata.obs.columns)[:30]}"
            )
        logger.info("Using pre-computed per-cell guide labels from obs['%s']", col)
        adata.obs[RAW_GUIDE_LABEL] = adata.obs[col].astype(str)
        _ensure_counts_layer(adata)
        return LoadedData(adata, None, "obs_label", lanes)

    raise ValueError(
        "Could not find guide information in the h5ad. Provide one of:\n"
        f"  * guide features in var['{ftype_col}'] "
        f"(one of {cfg.input.guide_feature_types}),\n"
        "  * input.guide_h5ad: path to a companion guide count matrix, or\n"
        "  * input.guide_obs_column: an obs column holding per-cell guide "
        "labels (e.g. 'genotype')."
    )


def _lanes_from_obs(adata: ad.AnnData, cfg: Config) -> Dict[str, str]:
    """Derive the lane mapping for an h5ad input.

    Uses an existing lane column when present, otherwise treats the file as a
    single lane named after the run.
    """
    src = str(cfg.input.h5ad)
    candidates = [
        c
        for c in (LANE_KEY, cfg.metadata.key_column, "sample", "orig.ident", "group")
        if c in adata.obs.columns
    ]
    # Prefer a column that actually distinguishes lanes; Seurat exports often
    # carry a constant 'orig.ident' alongside an informative 'group'.
    chosen = next(
        (c for c in candidates if adata.obs[c].nunique() > 1),
        candidates[0] if candidates else None,
    )
    if chosen is not None:
        values = adata.obs[chosen].astype(str)
        if chosen != LANE_KEY:
            logger.info("Using obs[%r] as the lane identifier", chosen)
            adata.obs[LANE_KEY] = pd.Categorical(values)
        return {lane: src for lane in sorted(values.unique())}

    adata.obs[LANE_KEY] = pd.Categorical([cfg.run.name] * adata.n_obs)
    return {cfg.run.name: src}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def split_features(adata: ad.AnnData, cfg: Config) -> Tuple[ad.AnnData, ad.AnnData]:
    """Split a combined matrix into (gene expression, guide) AnnData objects.

    Guide features are re-indexed by their unique feature ID (``gene_ids`` in
    10x output, e.g. ``AFF4_P1P2_1``) because the symbol column repeats across
    the guides of one target.
    """
    col = cfg.input.feature_type_column
    if col not in adata.var.columns:
        raise ValueError(
            f"var['{col}'] not found, so gene-expression and guide features "
            "cannot be separated. Set input.feature_type_column, or use "
            "input.guide_h5ad / input.guide_obs_column."
        )

    types = adata.var[col].astype(str)
    is_guide = types.isin(cfg.input.guide_feature_types).to_numpy()
    is_gex = (types == cfg.input.gex_feature_type).to_numpy()

    if not is_gex.any():
        raise ValueError(
            f"No features of type {cfg.input.gex_feature_type!r} found. "
            f"Present types: {sorted(types.unique())}"
        )
    if not is_guide.any():
        raise ValueError(
            f"No guide features found (looked for {cfg.input.guide_feature_types}). "
            f"Present types: {sorted(types.unique())}"
        )

    expr = adata[:, is_gex].copy()
    guides = adata[:, is_guide].copy()

    if "gene_ids" in guides.var.columns:
        guides.var["guide_id"] = guides.var["gene_ids"].astype(str)
        guides.var["guide_symbol"] = guides.var_names.astype(str)
        guides.var_names = pd.Index(guides.var["guide_id"])
        guides.var_names_make_unique()
    else:
        guides.var["guide_id"] = guides.var_names.astype(str)

    _ensure_counts_layer(expr)
    logger.info(
        "Split features: %d gene-expression, %d guide", expr.n_vars, guides.n_vars
    )
    return expr, guides


def _ensure_counts_layer(adata: ad.AnnData) -> None:
    """Keep an untouched copy of the raw counts in ``layers['counts']``."""
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()


def apply_layer_choices(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Honour ``input.counts_layer`` / ``input.normalized_layer``.

    Afterwards ``layers['counts']`` holds raw counts and, if the user pointed at
    a pre-normalized layer, ``layers['lognorm']`` holds log-normalized values so
    the clustering stage can skip re-normalizing.
    """
    counts_layer = cfg.input.counts_layer
    norm_layer = cfg.input.normalized_layer

    if counts_layer:
        if counts_layer not in adata.layers:
            raise ValueError(
                f"input.counts_layer={counts_layer!r} not found. "
                f"Available layers: {sorted(adata.layers.keys())}"
            )
        adata.layers["counts"] = adata.layers[counts_layer].copy()
        logger.info("Using layer %r as raw counts", counts_layer)
    else:
        _ensure_counts_layer(adata)

    if norm_layer:
        if norm_layer == "X":
            # Shared/subset objects often carry only normalized values, with no
            # raw counts anywhere. Re-normalizing those would corrupt them.
            adata.layers["lognorm"] = adata.X.copy()
            logger.info("input.normalized_layer='X': treating X as log-normalized")
            if not counts_layer:
                logger.warning(
                    "No raw counts are available in this object, so count-based "
                    "QC metrics (total_counts, n_genes_by_counts) are derived "
                    "from normalized values and are approximate."
                )
        elif norm_layer not in adata.layers:
            raise ValueError(
                f"input.normalized_layer={norm_layer!r} not found. "
                f"Available layers: {sorted(adata.layers.keys())}. "
                "Use 'X' when the object's X already holds log-normalized values."
            )
        else:
            adata.layers["lognorm"] = adata.layers[norm_layer].copy()
            logger.info("Using layer %r as log-normalized expression", norm_layer)

    return adata


def attach_sample_metadata(
    adata: ad.AnnData, cfg: Config, n_lanes: int
) -> ad.AnnData:
    """Merge the per-lane sample metadata table into ``adata.obs``.

    A metadata file is mandatory for multi-lane runs (``metadata.file``); for a
    single lane the pipeline synthesizes a minimal record so downstream code and
    the report can treat both cases identically. The requirement is waived when
    the input already carries sample annotation — re-analyzing an ``.h5ad`` this
    pipeline wrote should not demand the metadata file a second time.
    """
    lanes_present = sorted(adata.obs[LANE_KEY].astype(str).unique())

    if not cfg.metadata.file:
        already_annotated = "sample_id" in adata.obs.columns
        if n_lanes > 1 and cfg.metadata.require_for_multilane and not already_annotated:
            raise ValueError(
                f"This run spans {n_lanes} lanes ({lanes_present}) but no sample "
                "metadata file was given. Set metadata.file to a CSV/TSV with one "
                f"row per lane and a '{cfg.metadata.key_column}' column, or set "
                "metadata.require_for_multilane: false to proceed without it."
            )
        if already_annotated:
            logger.info(
                "No metadata file given, but the input already carries sample "
                "annotation in obs; keeping it."
            )
            return adata
        logger.info("No sample metadata file; using lane IDs only.")
        adata.obs["sample_id"] = adata.obs[LANE_KEY].astype(str)
        return adata

    meta = read_sample_metadata(cfg.metadata.file, cfg.metadata.key_column)
    key = cfg.metadata.key_column

    meta_lanes = set(meta[key].astype(str))
    missing = [l for l in lanes_present if l not in meta_lanes]
    if missing:
        raise ValueError(
            f"Sample metadata is missing row(s) for lane(s) {missing}. "
            f"Lanes in the data: {lanes_present}; lanes in metadata: "
            f"{sorted(meta_lanes)}."
        )
    unused = sorted(meta_lanes - set(lanes_present))
    if unused:
        logger.warning("Sample metadata has unused lane row(s): %s", unused)

    meta = meta.set_index(meta[key].astype(str))
    new_cols = [c for c in meta.columns if c != key]
    lane_values = adata.obs[LANE_KEY].astype(str)
    for c in new_cols:
        mapped = lane_values.map(meta[c])
        target = c if c not in adata.obs.columns else f"{c}_meta"
        if target != c:
            logger.warning(
                "obs already has column %r; metadata column stored as %r", c, target
            )
        adata.obs[target] = (
            pd.Categorical(mapped.astype(str))
            if mapped.dtype == object
            else mapped.to_numpy()
        )

    if "sample_id" not in adata.obs.columns:
        adata.obs["sample_id"] = adata.obs[LANE_KEY].astype(str)
    logger.info("Merged %d metadata column(s): %s", len(new_cols), new_cols)
    return adata


def read_guide_table(
    cfg: Config, obs_names: pd.Index, lanes: Optional[Sequence[str]] = None
) -> pd.Series:
    """Resolve a barcode -> guide table into one label per cell.

    This is the layout used by PS_python's demo (``BARCODE_10x_Merged.txt``):
    one row per detected guide per cell, so a cell with two guides appears
    twice. Collapsing that with a plain dict build keeps whichever row happened
    to come last, which silently picks a guide at random for every multiplet.

    Instead the same dominance rule as the count-matrix path is applied when a
    UMI-count column is available: the top guide must reach ``guides.min_umi``
    and beat the runner-up by ``guides.dominance_ratio``, otherwise the cell is
    ambiguous. Without counts, cells with conflicting guides are ambiguous.
    """
    icfg = cfg.input
    path = Path(icfg.guide_table)
    if not path.is_file():
        raise FileNotFoundError(f"input.guide_table not found: {path}")

    sep = "\t" if path.suffix.lower() in (".tsv", ".txt", ".tab") else ","
    table = pd.read_csv(path, sep=sep)
    cell_col, gene_col = icfg.guide_table_cell_column, icfg.guide_table_gene_column
    for col in (cell_col, gene_col):
        if col not in table.columns:
            raise ValueError(
                f"input.guide_table {path} has no {col!r} column "
                f"(columns: {list(table.columns)}). Set "
                "input.guide_table_cell_column / _gene_column to match."
            )

    raw_cells = table[cell_col].astype(str)
    stripped = (
        raw_cells.str.rsplit("_", n=1).str[-1]
        if icfg.guide_table_strip_prefix
        else raw_cells
    )
    # Group on the FULL cell id. The same 10x barcode legitimately occurs in
    # every lane, so grouping on the prefix-stripped barcode would merge one
    # cell per lane into a single pseudo-cell and make them all look like
    # multiplets. Stripped forms are only used as a matching fallback below,
    # and only when they are unambiguous.
    table = table.assign(_cell=raw_cells, _stripped=stripped)

    count_col = icfg.guide_table_count_column
    has_counts = bool(count_col) and count_col in table.columns

    gcfg = cfg.guides
    labels: Dict[str, str] = {}
    if has_counts:
        table = table.sort_values(count_col, ascending=False)
        for cell_id, rows in table.groupby("_cell", sort=False):
            counts = rows[count_col].to_numpy(dtype=float)
            top = float(counts[0])
            second = float(counts[1]) if len(counts) > 1 else 0.0
            if top < max(gcfg.min_umi, 1):
                labels[cell_id] = gcfg.unassigned_label
            elif top > gcfg.dominance_ratio * second:
                labels[cell_id] = str(rows.iloc[0][gene_col])
            else:
                labels[cell_id] = gcfg.ambiguous_label
    else:
        logger.warning(
            "input.guide_table has no %r column; cells with more than one guide "
            "are marked ambiguous rather than resolved by count.",
            count_col,
        )
        for cell_id, rows in table.groupby("_cell", sort=False):
            genes = set(rows[gene_col].astype(str))
            labels[cell_id] = genes.pop() if len(genes) == 1 else gcfg.ambiguous_label

    # Add stripped-form keys only where they are unique, so a fallback match can
    # never silently pick the wrong lane's cell.
    counts_per_stripped = table.groupby("_stripped")["_cell"].nunique()
    unique_stripped = set(counts_per_stripped[counts_per_stripped == 1].index)
    for cell_id, strip in table[["_cell", "_stripped"]].drop_duplicates().itertuples(index=False):
        if strip in unique_stripped and strip not in labels:
            labels[strip] = labels[cell_id]

    # Barcodes are spelled differently on the two sides: a table carries a
    # library prefix ('S1L1_AAACCC-1') while ``sc.concat`` appends a lane suffix
    # to the matrix ('AAACCC-1-S1L1'). Both spellings, and the bare barcode, are
    # tried so a table written by this pipeline reads back into the same cells.
    if lanes is not None:
        lane_values = [str(x) for x in lanes]
    else:
        lane_values = [None] * len(obs_names)

    def _candidates(name: str, lane: Optional[str]):
        yield name
        if lane:
            yield f"{lane}_{name}"
            if name.endswith(f"-{lane}"):
                base = name[: -(len(lane) + 1)]
                yield base
                yield f"{lane}_{base}"
        if "_" in name:
            yield name.rsplit("_", 1)[-1]

    resolved = []
    for name, lane in zip(obs_names, lane_values):
        label = gcfg.unassigned_label
        for key in _candidates(str(name), lane):
            if key in labels:
                label = labels[key]
                break
        resolved.append(label)

    out = pd.Series(resolved, index=obs_names)
    matched = int((out != gcfg.unassigned_label).sum())
    if matched == 0:
        raise ValueError(
            f"No barcode in {path} matched the matrix. Example matrix barcode: "
            f"{obs_names[0]!r}; example table barcode: {raw_cells.iloc[0]!r}. "
            "Check input.guide_table_strip_prefix."
        )
    logger.info(
        "Guide table: %d/%d cells assigned a label from %s",
        matched,
        len(obs_names),
        path.name,
    )
    return out


def write_guide_table(
    guides: ad.AnnData, expr: ad.AnnData, cfg: Config, path: Path
) -> Optional[Path]:
    """Export the guide count matrix as a long barcode -> guide table.

    This reproduces the layout of PS_python's ``BARCODE_10x_Merged.txt``, which
    is exactly the guide count matrix in long form thresholded at
    ``output.guide_table_min_umi`` UMIs (verified against the demo lane: the
    per-cell totals and guide counts match at 100% for a threshold of 3).
    Generating it here means the file is reproducible from the matrix instead of
    being a hand-maintained side product.

    Two things differ deliberately from the existing file:

    * the ``gene`` column uses the pipeline's target parser, so guides whose
      names carry no underscore (``CD81.2``) collapse to their target
      (``CD81``) rather than being kept as separate targets;
    * an ``assignment`` column carries the pipeline's own per-cell call, so a
      consumer gets the same dominance rule used everywhere else instead of
      having to re-derive one from the raw rows.

    Rows are written with the highest-count guide **last** within each cell, so
    that even a naive "last row wins" reader lands on the dominant guide.
    """
    from .guides import OBS_TARGET, parse_target_genes

    if not cfg.output.write_guide_table or guides is None:
        return None

    from scipy import sparse

    X = guides.layers["counts"] if "counts" in guides.layers else guides.X
    X = sparse.csr_matrix(X)
    min_umi = max(int(cfg.output.guide_table_min_umi), 1)
    X.data[X.data < min_umi] = 0
    X.eliminate_zeros()
    if X.nnz == 0:
        logger.warning("No guide counts survive the %d-UMI threshold", min_umi)
        return None

    coo = X.tocoo()
    guide_ids = guides.var_names.to_numpy().astype(str)
    targets = parse_target_genes(guide_ids, cfg.guides)

    barcodes = guides.obs_names.to_numpy().astype(str)
    if LANE_KEY in guides.obs.columns:
        lanes = guides.obs[LANE_KEY].astype(str).to_numpy()
        # ``sc.concat`` appends '-<lane>' to make barcodes unique across lanes.
        # Strip it before re-prefixing, so the result is '<lane>_<barcode>' as
        # in PS_python's file rather than a doubled-up identifier.
        stripped = np.array(
            [
                b[: -(len(l) + 1)] if b.endswith(f"-{l}") else b
                for b, l in zip(barcodes, lanes)
            ]
        )
        cells = np.array(
            [f"{l}_{b}" for l, b in zip(lanes[coo.row], stripped[coo.row])]
        )
    else:
        cells = barcodes[coo.row]

    counts = coo.data.astype(int)
    table = pd.DataFrame(
        {
            "cell": cells,
            "barcode": guide_ids[coo.col],
            "sgrna": guide_ids[coo.col],
            "gene": targets[coo.col],
            "umi_count": counts,
        }
    )

    assignment = (
        expr.obs[OBS_TARGET].astype(str).reindex(guides.obs_names).to_numpy()
        if OBS_TARGET in expr.obs.columns
        else np.array(["NA"] * guides.n_obs)
    )
    table["assignment"] = assignment[coo.row]

    # Highest count last within each cell (see docstring).
    table = table.sort_values(["cell", "umi_count"], ascending=[True, True])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)
    logger.info(
        "Wrote guide table %s: %d row(s) for %d cell(s), guides with >= %d UMIs",
        path.name,
        len(table),
        table["cell"].nunique(),
        min_umi,
    )
    return path


def read_sample_metadata(path: str, key_column: str) -> pd.DataFrame:
    """Read a sample metadata CSV/TSV and check the key column."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Sample metadata file not found: {p}")
    sep = "\t" if p.suffix.lower() in (".tsv", ".txt", ".tab") else ","
    meta = pd.read_csv(p, sep=sep)
    if key_column not in meta.columns:
        raise ValueError(
            f"Sample metadata {p} has no '{key_column}' column "
            f"(columns: {list(meta.columns)}). Set metadata.key_column to match."
        )
    dupes = meta[key_column][meta[key_column].duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate {key_column} value(s) in {p}: {sorted(set(dupes))}")
    return meta


def write_h5ad(adata: ad.AnnData, path: Path, compression: str = "gzip") -> Path:
    """Write an ``.h5ad``, making the parent directory and sanitizing obs.

    Object-dtype ``obs`` columns are cast to string/categorical first; mixed
    types are the usual cause of a write failing at the very end of a long run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sanitize in place rather than copying: an .h5ad of this size would
    # otherwise double peak memory just to fix a few column dtypes.
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = pd.Categorical(adata.obs[col].astype(str))
    for col in adata.var.columns:
        if adata.var[col].dtype == object:
            adata.var[col] = adata.var[col].astype(str)
    adata.write_h5ad(path, compression=compression)
    logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def archive_results(outdir: Path, cfg: Config) -> Optional[Path]:
    """Bundle the run directory into a ``.tar.gz`` for sharing.

    Matrices are excluded by ``output.archive_exclude`` (``*.h5ad`` by default),
    so the archive carries the report, figures, tables and logs — the parts
    someone actually reads — at a size that can be attached to an email or a
    GitHub release. Returns ``None`` when archiving is disabled.
    """
    if not cfg.output.archive:
        return None

    import fnmatch
    import tarfile

    outdir = Path(outdir)
    name = cfg.output.archive_name or f"{cfg.run.name}_results.tar.gz"
    if not name.endswith((".tar.gz", ".tgz")):
        name += ".tar.gz"
    dest = outdir / name

    patterns = list(cfg.output.archive_exclude or [])

    def excluded(rel: Path) -> bool:
        text = str(rel)
        return any(
            fnmatch.fnmatch(text, p) or fnmatch.fnmatch(rel.name, p) for p in patterns
        )

    members: List[Path] = []
    skipped: List[Path] = []
    for path in sorted(outdir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(outdir)
        # Never pack the archive into itself, even mid-write.
        if path == dest or excluded(rel):
            skipped.append(rel)
            continue
        members.append(path)

    if not members:
        logger.warning("Nothing to archive in %s", outdir)
        return None

    # Write to a temporary name first so a partial archive is never left behind
    # and cannot be picked up by the walk above.
    tmp = dest.with_suffix(dest.suffix + ".partial")
    root = cfg.run.name or outdir.name
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            for path in members:
                tar.add(path, arcname=str(Path(root) / path.relative_to(outdir)))
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()

    size_mb = dest.stat().st_size / 1e6
    logger.info(
        "Archived %d file(s) to %s (%.1f MB); excluded %d matching %s",
        len(members),
        dest.name,
        size_mb,
        len(skipped),
        patterns,
    )
    return dest


def relocate_if_large(path: Path, cfg: Config) -> Path:
    """Move an output above the size threshold to ``output.large_file_dir``.

    Keeps multi-GB ``.h5ad`` files out of the repository (see CLAUDE.md).
    """
    path = Path(path)
    if not cfg.output.large_file_dir or not path.is_file():
        return path
    size_mb = path.stat().st_size / 1e6
    if size_mb < cfg.output.large_file_threshold_mb:
        return path
    dest_dir = Path(cfg.output.large_file_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    logger.info("Moving %s (%.0f MB) to %s", path.name, size_mb, dest_dir)
    try:
        path.replace(dest)
    except OSError:
        # Different filesystems (the usual case on Colab: local disk -> Drive).
        import shutil

        shutil.copy2(path, dest)
        path.unlink()
    return dest
