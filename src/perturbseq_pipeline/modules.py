"""Co-functional modules and co-regulated gene programs (the "regulome" map).

Reimplements the network analysis of Chen et al. (Nature 2023,
`s41586-023-06733-x <https://www.nature.com/articles/s41586-023-06733-x>`_,
GSE216909): build a perturbation x gene matrix of log2FC vs control, then

* cluster the gene axis into co-regulated programs (Pearson correlation), and
* cluster the perturbation axis into co-functional modules (Spearman correlation),

and relate the two with a signed module x program strength matrix, plus a
TF-hub / module-module network.

Two deliberate simplifications vs the paper:

* The effect value is a pseudobulk mean-difference log2FC (de-logged group
  means), rather than a per-gene Wilcoxon FindMarkers log2FC.
* The number of programs/modules is controlled by configuration
  (modules.n_programs / modules.n_modules or cluster_distance_threshold).

Adaptive execution
------------------
The original implementation is preserved for ordinary datasets.

STANDARD mode
    Used for datasets below the large-data thresholds. This is intentionally
    the original implementation: the selected cell x gene matrix is converted
    to dense form and all downstream calculations follow the previous code.

LARGE mode
    Automatically selected for million-cell datasets such as KOLF. The
    biological/statistical definitions are unchanged, but the implementation
    avoids constructing a dense cell x gene matrix.

    Instead expression is processed in gene chunks. For every perturbation and
    every selected gene only sufficient statistics are accumulated:

        n
        sum(x)
        sum(x^2)

    These are enough to recover group means, variances, Welch statistics and
    log2 fold changes exactly as in STANDARD mode.

    Thus the large intermediate

        cells x selected genes

    remains sparse, while the much smaller

        perturbations x selected genes

    result is materialised.

Marker discovery in LARGE mode is performed on a reproducible, approximately
cluster-stratified subset of cells. Marker selection is only feature selection;
the actual perturbation effect matrix still uses every cell in the experiment.

The stage returns None when too few perturbations or genes survive.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import CLUSTER_KEY, LOGNORM_LAYER
from .config import Config
from .compute import is_package_available, log_compute_decision, resolve_stage_backend, run_parallel
from .gene_sets import run_program_enrichment
from .guides import CLASS_TARGETING, OBS_CLASS, OBS_TARGET
from .perturbation import CONTROL_NTC, CONTROL_OTHER, benjamini_hochberg, control_masks

logger = logging.getLogger(__name__)


# Numerical behaviour

# Keep this unchanged from the existing implementation so STANDARD and LARGE
# mode use the same log2FC definition.
_PSEUDO = 1e-9


# Internal matrices produced after aggregation are small enough for float64,
# which also preserves numerical agreement with the STANDARD implementation.
LARGE_EFFECT_DTYPE = np.float64


# Result object


@dataclass
class ModulesResults:
    """Everything the report/plots need about module/program analysis."""

    #: Perturbation x gene log2FC-vs-control matrix.
    effect_matrix: pd.DataFrame
    #: One row per gene: program label and program size.
    gene_programs: pd.DataFrame
    #: One row per perturbation: module label, cell count and #DE genes.
    modules: pd.DataFrame
    #: Signed module x program regulatory strength.
    module_program: pd.DataFrame
    #: Program x cluster mean activity score.
    program_activity: pd.DataFrame
    #: Directed TF->TF regulatory edges.
    tf_edges: pd.DataFrame
    #: Per-TF hub size.
    hubs: pd.DataFrame
    #: Module x module connectivity.
    module_connectivity: pd.DataFrame
    control: str = CONTROL_NTC
    n_programs: int = 0
    n_modules: int = 0
    program_correlation: str = "pearson"
    module_correlation: str = "spearman"
    linkage_method: str = "average"
    program_labels: List[str] = field(default_factory=list)
    module_labels: List[str] = field(default_factory=list)
    gene_order: List[str] = field(default_factory=list)
    perturbation_order: List[str] = field(default_factory=list)
    program_genes: Dict[str, List[str]] = field(default_factory=dict)
    module_members: Dict[str, List[str]] = field(default_factory=dict)
    score_columns: List[str] = field(default_factory=list)
    #: Full biological pathway enrichment results for gene programs.
    program_enrichment: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Compact biological annotations per program (e.g. {'P1': 'Interferon response'}).
    program_annotations: Dict[str, str] = field(default_factory=dict)
    #: Display labels with biological annotations (e.g. {'P1': 'P1 — Interferon response'}).
    program_display_labels: Dict[str, str] = field(default_factory=dict)
    #: Compact summary table combining program sizes, top genes, and top enriched pathways.
    program_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Why program enrichment produced no result (gene-set download failed, ...); empty when it ran.
    program_enrichment_error: str = ""
    #: Execution mode, useful in reports/debugging.
    execution_mode: str = "standard"
    note: str = ""


# Perturbation selection


def select_perturbations(expr: ad.AnnData, cfg: Config) -> List[str]:
    """Targets assigned to enough cells to estimate an effect profile.

    ``high_moi`` mode counts the cells *carrying* each target (membership).
    """
    from .high_moi import membership_index

    membership = membership_index(expr, cfg)
    if membership is not None:
        counts = membership.counts
    else:
        klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
        targets = expr.obs[OBS_TARGET].astype(str).to_numpy()
        mask = klass == CLASS_TARGETING
        counts = pd.Series(targets[mask]).value_counts()
    keep = counts[counts >= cfg.modules.min_cells_per_perturbation].index.tolist()
    return sorted(keep)


# Gene selection -- STANDARD


def _select_genes_standard(expr: ad.AnnData, cfg: Config) -> List[str]:
    """Original downstream-gene selection implementation."""
    import scanpy as sc

    mcfg = cfg.modules
    if mcfg.gene_selection == "hvg":
        if "highly_variable" in expr.var:
            return expr.var_names[expr.var["highly_variable"].to_numpy()].tolist()
        return expr.var_names.tolist()
    key = mcfg.cluster_key
    if key not in expr.obs or expr.obs[key].nunique() < 2:
        logger.warning("modules: '%s' has <2 groups; falling back to HVGs.", key)
        if "highly_variable" in expr.var:
            return expr.var_names[expr.var["highly_variable"].to_numpy()].tolist()
        return expr.var_names.tolist()
    # This is intentionally preserved from the original implementation.
    adata = expr
    if adata.obs[key].dtype.name != "category":
        adata = expr.copy()
        adata.obs[key] = adata.obs[key].astype("category")
    sc.tl.rank_genes_groups(adata, key, method=mcfg.marker_method, n_genes=mcfg.n_marker_genes_per_cluster)
    df = sc.get.rank_genes_groups_df(adata, group=None)
    df = df[df["logfoldchanges"] > 0]
    genes: List[str] = []
    for _, sub in df.groupby("group"):
        top = sub.sort_values("logfoldchanges", ascending=False).head(mcfg.n_marker_genes_per_cluster)
        genes.extend(top["names"].tolist())
    seen: Dict[str, None] = {}
    for gene in genes:
        if gene in expr.var_names:
            seen.setdefault(gene, None)
    return list(seen)


# Gene selection -- LARGE


def _stratified_marker_sample(expr: ad.AnnData, key: str, max_cells: int, seed: int) -> np.ndarray:
    """Approximately stratified cell sample across clusters.

    Every cluster gets representation before remaining sample capacity is
    distributed according to cluster abundance.
    """
    n = expr.n_obs
    if n <= max_cells:
        return np.arange(n, dtype=np.int64)
    labels = expr.obs[key].astype(str).to_numpy()
    unique, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    # Preserve representation of rare clusters but do not allocate an absurd
    # fixed number if there are many clusters.
    minimum_per_cluster = min(2_000, max(100, max_cells // max(4 * len(unique), 1)))
    used = 0
    cluster_indices: Dict[str, np.ndarray] = {}
    for label in unique:
        idx = np.flatnonzero(labels == label)
        cluster_indices[label] = idx
        take = min(len(idx), minimum_per_cluster)
        if take > 0:
            chosen = idx if take == len(idx) else rng.choice(idx, size=take, replace=False)
            selected.append(np.asarray(chosen, dtype=np.int64))
            used += take
    if used >= max_cells:
        out = np.concatenate(selected)
        if len(out) > max_cells:
            out = rng.choice(out, size=max_cells, replace=False)
        return np.sort(out)
    # Fill the remaining quota from cells not already selected.
    selected_all = np.concatenate(selected)
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[selected_all] = True
    remaining = np.flatnonzero(~selected_mask)
    budget = min(max_cells - len(selected_all), len(remaining))
    if budget > 0:
        extra = rng.choice(remaining, size=budget, replace=False)
        selected_all = np.concatenate([selected_all, extra])
    return np.sort(selected_all.astype(np.int64, copy=False))


def _select_genes_large(expr: ad.AnnData, cfg: Config) -> List[str]:
    """Memory-aware marker selection for million-cell datasets.

    Only feature selection is performed on the subset. The later effect matrix
    uses all cells.
    """
    import scanpy as sc

    mcfg = cfg.modules
    if mcfg.gene_selection == "hvg":
        if "highly_variable" in expr.var:
            genes = expr.var_names[expr.var["highly_variable"].to_numpy()].tolist()
            logger.info("modules LARGE: using %d precomputed HVGs", len(genes))
            return genes
        logger.warning("modules LARGE: no highly_variable column; using all genes")
        return expr.var_names.tolist()
    key = mcfg.cluster_key
    if key not in expr.obs or expr.obs[key].nunique() < 2:
        logger.warning("modules LARGE: '%s' has <2 groups; falling back to HVGs.", key)
        if "highly_variable" in expr.var:
            return expr.var_names[expr.var["highly_variable"].to_numpy()].tolist()
        return expr.var_names.tolist()
    sample_idx = _stratified_marker_sample(expr, key, cfg.scaling.marker_max_cells, cfg.run.seed)
    logger.info(
        "modules LARGE: marker selection on %d/%d cells stratified across %d %s groups",
        len(sample_idx),
        expr.n_obs,
        expr.obs[key].nunique(),
        key,
    )
    # This is the only cell-subset copy in marker selection.
    sample = expr[sample_idx, :].copy()
    sample.obs[key] = sample.obs[key].astype("category")
    try:
        sc.tl.rank_genes_groups(sample, key, method=mcfg.marker_method, n_genes=mcfg.n_marker_genes_per_cluster)
        df = sc.get.rank_genes_groups_df(sample, group=None)
    finally:
        # Marker output has already been extracted from the object by this point.
        # Delete the potentially large sample aggressively.
        pass
    df = df[df["logfoldchanges"] > 0]
    genes: List[str] = []
    for _, sub in df.groupby("group"):
        top = sub.sort_values("logfoldchanges", ascending=False).head(mcfg.n_marker_genes_per_cluster)
        genes.extend(top["names"].astype(str).tolist())
    del sample
    gc.collect()
    seen: Dict[str, None] = {}
    measured = set(expr.var_names)
    for gene in genes:
        if gene in measured:
            seen.setdefault(gene, None)
    selected = list(seen)
    return selected


def select_genes(expr: ad.AnnData, cfg: Config, *, large_mode: Optional[bool] = None) -> List[str]:
    """Adaptive downstream-gene selection."""
    if large_mode is None:
        large_mode = cfg.use_large_mode(expr.n_obs)
    if large_mode:
        return _select_genes_large(expr, cfg)
    return _select_genes_standard(expr, cfg)


# STANDARD effect matrix


def _dense_layer(expr: ad.AnnData, gene_idx: np.ndarray) -> np.ndarray:
    """Original dense selected-gene expression matrix."""
    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    sub = layer[:, gene_idx]
    if sparse.issparse(sub):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float64)


def _build_effect_matrix_standard(
    expr: ad.AnnData, genes: List[str], targets: List[str], cfg: Config
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    """Original perturbation x gene implementation.

    This section is intentionally kept equivalent to the previous code so
    Replogle-like datasets retain the established execution path.
    """
    from scipy.stats import t as _tdist

    base = control_masks(expr, cfg)
    control = cfg.modules.control
    if control == CONTROL_NTC and not base[CONTROL_NTC].any():
        logger.warning("modules: no non-targeting cells; using 'other' control.")
        control = CONTROL_OTHER
    from .high_moi import membership_index

    gene_idx = np.array([expr.var_names.get_loc(gene) for gene in genes])
    log = _dense_layer(expr, gene_idx)
    lin = np.expm1(log)
    membership = membership_index(expr, cfg)
    if membership is not None:
        # perturbation x cell indicator from the membership matrix (a cell may sit in several rows);
        # the 'other' arithmetic below (totals over targeting CELLS minus own) stays valid.
        indicator = membership.indicator(targets)
    else:
        obs_targets = expr.obs[OBS_TARGET].astype(str).to_numpy()
        target_pos = {target: i for i, target in enumerate(targets)}
        rows = []
        cols = []
        for cell, target in enumerate(obs_targets):
            j = target_pos.get(target)
            if j is not None:
                rows.append(j)
                cols.append(cell)
        indicator = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(targets), expr.n_obs))
    n_p = np.asarray(indicator.sum(axis=1)).ravel()
    n_p_col = n_p[:, None]
    sum_lin = indicator @ lin
    mean_perturbed = sum_lin / n_p_col
    s_log = indicator @ log
    ss_log = indicator @ (log * log)
    mean_p = s_log / n_p_col
    var_p = (ss_log - s_log * mean_p) / np.clip(n_p_col - 1, 1, None)
    if control == CONTROL_NTC:
        ntc = base[CONTROL_NTC]
        mean_control = lin[ntc].mean(axis=0)[None, :]
        log_c = log[ntc]
        mean_c = log_c.mean(axis=0)[None, :]
        var_c = log_c.var(axis=0, ddof=1)[None, :]
        n_c = np.array([[max(int(ntc.sum()), 1)]], dtype=float)
    else:
        targeting = base[CONTROL_OTHER]
        n_tot = int(targeting.sum())
        tot_lin = lin[targeting].sum(axis=0)
        mean_control = (tot_lin[None, :] - sum_lin) / np.clip(n_tot - n_p_col, 1, None)
        tot_log = log[targeting].sum(axis=0)
        tot_log2 = (log[targeting] ** 2).sum(axis=0)
        n_c = np.clip(n_tot - n_p_col, 1, None).astype(float)
        s_log_c = tot_log[None, :] - s_log
        mean_c = s_log_c / n_c
        var_c = (tot_log2[None, :] - ss_log - s_log_c * mean_c) / np.clip(n_c - 1, 1, None)
    log2fc = np.log2((mean_perturbed + _PSEUDO) / (mean_control + _PSEUDO))
    term_p = var_p / n_p_col
    term_c = var_c / n_c
    se2 = term_p + term_c
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = (mean_p - mean_c) / np.sqrt(se2)
        df = (se2**2) / (term_p**2 / np.clip(n_p_col - 1, 1, None) + term_c**2 / np.clip(n_c - 1, 1, None))
        pvals = 2.0 * _tdist.sf(np.abs(tstat), np.clip(df, 1, None))
    pvals = np.where(np.isfinite(pvals) & (se2 > 0), pvals, 1.0)
    fdr = np.vstack([benjamini_hochberg(pvals[i]) for i in range(pvals.shape[0])])
    de = (np.abs(log2fc) > cfg.modules.hub_lfc_threshold) & (fdr < cfg.modules.de_fdr_alpha)
    effect = pd.DataFrame(log2fc, index=targets, columns=genes)
    de_mask = pd.DataFrame(de, index=targets, columns=genes)
    return (effect, control, de_mask)


# LARGE effect matrix helpers


def _target_indicator(
    expr: ad.AnnData, targets: List[str], cfg: Optional[Config] = None
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse perturbation x cell membership matrix (from the membership matrix in ``high_moi`` mode)."""
    if cfg is not None:
        from .high_moi import membership_index

        membership = membership_index(expr, cfg)
        if membership is not None:
            indicator = membership.indicator(targets)
            return indicator, np.asarray(indicator.sum(axis=1)).ravel().astype(np.float64)
    target_names = expr.obs[OBS_TARGET].astype(str).to_numpy()
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    target_pos = {target: i for i, target in enumerate(targets)}
    valid_cells = np.flatnonzero(klass == CLASS_TARGETING)
    row_buffer: List[int] = []
    cell_buffer: List[int] = []
    for cell in valid_cells:
        row = target_pos.get(target_names[cell])
        if row is not None:
            row_buffer.append(row)
            cell_buffer.append(int(cell))
    indicator = sparse.csr_matrix(
        (
            np.ones(len(row_buffer), dtype=np.float64),
            (np.asarray(row_buffer, dtype=np.int64), np.asarray(cell_buffer, dtype=np.int64)),
        ),
        shape=(len(targets), expr.n_obs),
    )
    counts = np.asarray(indicator.sum(axis=1)).ravel().astype(np.float64)
    return (indicator, counts)


def _as_sparse_float64(matrix) -> sparse.csr_matrix:
    """Convert an expression chunk into CSR float64 without densifying."""
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix, dtype=np.float64)
    # Dense input is unusual for very large single-cell objects. Converting the
    # selected gene chunk to sparse still bounds the working set.
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))


def _sparse_expm1(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Element-wise expm1 preserving structural zeros."""
    out = matrix.copy()
    out.data = np.expm1(out.data)
    out.eliminate_zeros()
    return out


def _sparse_square(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Element-wise square preserving sparsity."""
    out = matrix.copy()
    out.data *= out.data
    return out


def _dense_aggregate(matrix) -> np.ndarray:
    """Convert an aggregated perturbation x gene result to dense."""
    if sparse.issparse(matrix):
        return matrix.toarray().astype(LARGE_EFFECT_DTYPE, copy=False)
    return np.asarray(matrix, dtype=LARGE_EFFECT_DTYPE)


# LARGE effect matrix


def _build_effect_matrix_large(
    expr: ad.AnnData, genes: List[str], targets: List[str], cfg: Config
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    """Memory-bounded perturbation x gene effect calculation.

    This computes the same quantities as STANDARD mode but processes selected
    genes in chunks and never constructs a dense cells x genes matrix.
    """
    from scipy.stats import t as _tdist

    base = control_masks(expr, cfg)
    control = cfg.modules.control
    if control == CONTROL_NTC and not base[CONTROL_NTC].any():
        logger.warning("modules LARGE: no NTC cells; using 'other' control.")
        control = CONTROL_OTHER
    n_targets = len(targets)
    n_genes = len(genes)
    logger.info(
        "modules LARGE: building %d perturbation x %d gene effect matrix in chunks of %d genes",
        n_targets,
        n_genes,
        cfg.scaling.effect_gene_chunk,
    )
    indicator, n_p = _target_indicator(expr, targets, cfg)
    n_p_col = n_p[:, None]
    if np.any(n_p == 0):
        raise RuntimeError(
            "A selected perturbation has zero cells after target-indicator "
            "construction; this should have been removed by select_perturbations."
        )
    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    gene_positions = np.asarray([expr.var_names.get_loc(gene) for gene in genes], dtype=np.int64)
    # Final aggregated objects only: perturbations x genes.
    log2fc_all = np.empty((n_targets, n_genes), dtype=LARGE_EFFECT_DTYPE)
    pvals_all = np.empty((n_targets, n_genes), dtype=LARGE_EFFECT_DTYPE)
    ntc_mask = base[CONTROL_NTC]
    targeting_mask = base[CONTROL_OTHER]
    n_ntc = int(ntc_mask.sum())
    n_targeting_total = int(targeting_mask.sum())
    gene_chunk = cfg.scaling.effect_gene_chunk
    for start in range(0, n_genes, gene_chunk):
        stop = min(start + gene_chunk, n_genes)
        chunk_pos = gene_positions[start:stop]
        chunk_log = _as_sparse_float64(layer[:, chunk_pos])
        # Perturbation log-space sufficient statistics
        sum_log = _dense_aggregate(indicator @ chunk_log)
        chunk_sq = _sparse_square(chunk_log)
        sumsq_log = _dense_aggregate(indicator @ chunk_sq)
        mean_p = sum_log / n_p_col
        var_p = (sumsq_log - sum_log * mean_p) / np.clip(n_p_col - 1, 1, None)
        # Numerical cancellation can produce tiny negative values.
        var_p = np.maximum(var_p, 0.0)
        # Linear-space means for log2FC
        chunk_lin = _sparse_expm1(chunk_log)
        sum_lin = _dense_aggregate(indicator @ chunk_lin)
        mean_perturbed = sum_lin / n_p_col
        # Control sufficient statistics
        if control == CONTROL_NTC:
            if n_ntc < 1:
                raise RuntimeError("NTC control selected but no NTC cells are available.")
            ntc_log = chunk_log[ntc_mask, :]
            ntc_lin = chunk_lin[ntc_mask, :]
            ntc_sq = chunk_sq[ntc_mask, :]
            sum_c_log = np.asarray(ntc_log.sum(axis=0)).ravel()
            sumsq_c_log = np.asarray(ntc_sq.sum(axis=0)).ravel()
            sum_c_lin = np.asarray(ntc_lin.sum(axis=0)).ravel()
            mean_control = sum_c_lin[None, :] / max(n_ntc, 1)
            mean_c = sum_c_log[None, :] / max(n_ntc, 1)
            if n_ntc > 1:
                var_c = (sumsq_c_log[None, :] - (sum_c_log[None, :] ** 2) / n_ntc) / (n_ntc - 1)
            else:
                var_c = np.zeros((1, stop - start), dtype=np.float64)
            var_c = np.maximum(var_c, 0.0)
            n_c = np.array([[float(max(n_ntc, 1))]])
        else:
            # All targeting cells are the control pool, with the target's own
            # cells removed for each row.
            total_log = np.asarray(chunk_log[targeting_mask, :].sum(axis=0)).ravel()
            total_log2 = np.asarray(chunk_sq[targeting_mask, :].sum(axis=0)).ravel()
            total_lin = np.asarray(chunk_lin[targeting_mask, :].sum(axis=0)).ravel()
            n_c = np.clip(n_targeting_total - n_p_col, 1, None).astype(np.float64)
            sum_c_log = total_log[None, :] - sum_log
            sum_c_log2 = total_log2[None, :] - sumsq_log
            sum_c_lin = total_lin[None, :] - sum_lin
            mean_control = sum_c_lin / n_c
            mean_c = sum_c_log / n_c
            var_c = (sum_c_log2 - sum_c_log * mean_c) / np.clip(n_c - 1, 1, None)
            var_c = np.maximum(var_c, 0.0)
        # Fold change
        log2fc = np.log2((mean_perturbed + _PSEUDO) / (mean_control + _PSEUDO))
        # Welch t-test in log space
        term_p = var_p / n_p_col
        term_c = var_c / n_c
        se2 = term_p + term_c
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = (mean_p - mean_c) / np.sqrt(se2)
            df = (se2**2) / (term_p**2 / np.clip(n_p_col - 1, 1, None) + term_c**2 / np.clip(n_c - 1, 1, None))
            pvals = 2.0 * _tdist.sf(np.abs(tstat), np.clip(df, 1, None))
        pvals = np.where(np.isfinite(pvals) & (se2 > 0), pvals, 1.0)
        log2fc_all[:, start:stop] = log2fc
        pvals_all[:, start:stop] = pvals
        logger.info("modules LARGE: effect genes %d-%d / %d complete", start + 1, stop, n_genes)
        del chunk_log
        del chunk_sq
        del chunk_lin
        del sum_log
        del sumsq_log
        del sum_lin
        del mean_p
        del var_p
        del mean_perturbed
        del mean_control
        del mean_c
        del var_c
        del term_p
        del term_c
        del se2
        del tstat
        del df
        del pvals
        del log2fc
        gc.collect()
    # BH-FDR is performed across genes within each perturbation exactly as
    # in STANDARD mode.
    logger.info("modules LARGE: BH correction across %d genes for %d perturbations", n_genes, n_targets)
    de_all = np.zeros((n_targets, n_genes), dtype=bool)
    for row in range(n_targets):
        fdr_row = benjamini_hochberg(pvals_all[row])
        de_all[row] = (np.abs(log2fc_all[row]) > cfg.modules.hub_lfc_threshold) & (fdr_row < cfg.modules.de_fdr_alpha)
        if row > 0 and row % 1_000 == 0:
            logger.info("modules LARGE: FDR %d/%d perturbations complete", row, n_targets)
    del pvals_all
    del indicator
    gc.collect()
    effect = pd.DataFrame(log2fc_all, index=targets, columns=genes)
    de_mask = pd.DataFrame(de_all, index=targets, columns=genes)
    logger.info(
        "modules LARGE: effect matrix completed; median %d significant DE genes/perturbation",
        int(de_mask.sum(axis=1).median()),
    )
    return (effect, control, de_mask)


# Adaptive public effect-matrix API


def build_effect_matrix(
    expr: ad.AnnData, genes: List[str], targets: List[str], cfg: Config
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    """Build perturbation x gene effect matrix using adaptive execution."""
    large_mode = cfg.use_large_mode(expr.n_obs, n_perturbations=len(targets))
    if large_mode:
        logger.info("modules execution mode: LARGE (%d cells, %d perturbations)", expr.n_obs, len(targets))
        return _build_effect_matrix_large(expr, genes, targets, cfg)
    logger.info("modules execution mode: STANDARD (%d cells, %d perturbations)", expr.n_obs, len(targets))
    return _build_effect_matrix_standard(expr, genes, targets, cfg)


# Correlation clustering


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson", cfg: Optional[Config] = None) -> np.ndarray:
    """Compute pairwise row correlation matrix using CPU or GPU (CuPy)."""
    if cfg is not None:
        decision = resolve_stage_backend(
            "modules_correlation", cfg, extra_info={"n_dense_elements": df.shape[0] * df.shape[1]}
        )
        if cfg.compute.log_backend_decisions:
            log_compute_decision(decision)
        if decision.is_gpu and is_package_available("cupy"):
            try:
                import cupy as cp

                # Data is shape (n_items, n_features)
                X = cp.asarray(df.to_numpy(dtype=np.float64))
                if method.lower() == "spearman":
                    order = cp.argsort(X, axis=1)
                    ranks = cp.empty_like(order, dtype=cp.float64)
                    cp.put_along_axis(ranks, order, cp.arange(X.shape[1], dtype=cp.float64)[None, :], axis=1)
                    X = ranks
                corr_gpu = cp.corrcoef(X)
                return cp.asnumpy(corr_gpu)
            except Exception as exc:
                logger.warning("GPU correlation computation failed (%s); falling back to CPU", exc)
    return df.T.corr(method=method).to_numpy()


def cluster_axis(
    items: pd.DataFrame,
    correlation: str,
    linkage_method: str,
    k: Optional[int],
    threshold: Optional[float],
    prefix: str,
    cfg: Optional[Config] = None,
) -> Tuple[pd.Series, List[str], List[str]]:
    """Correlation hierarchical clustering of rows."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    names = list(items.index)
    n = len(names)
    if n < 2:
        labels = pd.Series([f"{prefix}1"] * n, index=names)
        return (labels, names, [f"{prefix}1"] if n else [])
    corr = compute_correlation_matrix(items, method=correlation, cfg=cfg)
    dist = 1.0 - corr
    dist = np.nan_to_num(dist, nan=1.0, posinf=2.0, neginf=0.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist[dist < 0] = 0.0
    condensed = squareform(dist, checks=False)
    linkage_matrix = linkage(condensed, method=linkage_method)
    kk = None if k is None else max(2, min(k, n))
    if kk is not None:
        raw = fcluster(linkage_matrix, kk, criterion="maxclust")
    else:
        raw = fcluster(linkage_matrix, threshold, criterion="distance")
    order_idx = leaves_list(linkage_matrix)
    ordered_names = [names[i] for i in order_idx]
    remap: Dict[int, str] = {}
    for i in order_idx:
        cluster = int(raw[i])
        if cluster not in remap:
            remap[cluster] = f"{prefix}{len(remap) + 1}"
    labels = pd.Series([remap[int(cluster)] for cluster in raw], index=names)
    ordered_labels = [remap[cluster] for cluster in dict.fromkeys(int(raw[i]) for i in order_idx)]
    return (labels, ordered_names, ordered_labels)


# Module x program effects


def module_program_strength(
    effect: pd.DataFrame,
    gene_program: pd.Series,
    perturbation_module: pd.Series,
    program_labels: List[str],
    module_labels: List[str],
) -> pd.DataFrame:
    """Mean signed program-gene log2FC per perturbation module."""
    out = pd.DataFrame(index=module_labels, columns=program_labels, dtype=float)
    for module in module_labels:
        perts = perturbation_module.index[perturbation_module == module]
        for program in program_labels:
            genes = gene_program.index[gene_program == program]
            block = effect.loc[perts, genes]
            out.loc[module, program] = float(np.nanmean(block.to_numpy())) if block.size else np.nan
    return out


# Networks


def tf_network(
    effect: pd.DataFrame, de_mask: pd.DataFrame, perturbation_module: pd.Series, n_cells: pd.Series, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TF->TF edges, hub size and module connectivity."""
    hub_counts = de_mask.sum(axis=1)
    hubs = pd.DataFrame(
        {
            "target_gene": (hub_counts.index),
            "module": (perturbation_module.reindex(hub_counts.index).to_numpy()),
            "n_cells": [int(n_cells.get(target, 0)) for target in hub_counts.index],
            "n_de_genes": (hub_counts.to_numpy().astype(int)),
        }
    ).sort_values("n_de_genes", ascending=False, ignore_index=True)
    tf_genes = [gene for gene in effect.columns if gene in effect.index]
    edges = []
    for source in effect.index:
        for target in tf_genes:
            if source == target:
                continue
            if bool(de_mask.at[source, target]):
                value = effect.at[source, target]
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "log2fc": float(value),
                        "sign": ("positive" if value > 0 else "negative"),
                        "source_module": (perturbation_module.get(source, "")),
                        "target_module": (perturbation_module.get(target, "")),
                    }
                )
    tf_edges = pd.DataFrame(edges, columns=["source", "target", "log2fc", "sign", "source_module", "target_module"])
    labels = sorted(perturbation_module.unique(), key=lambda value: (len(value), value))
    sizes = perturbation_module.value_counts()
    connectivity = pd.DataFrame(0.0, index=labels, columns=labels)
    if not tf_edges.empty:
        for _, edge in tf_edges.iterrows():
            source_module = edge["source_module"]
            target_module = edge["target_module"]
            if source_module in labels and target_module in labels:
                connectivity.loc[source_module, target_module] += 1.0
    for source_module in labels:
        for target_module in labels:
            denominator = float(sizes.get(source_module, 1) * sizes.get(target_module, 1))
            connectivity.loc[source_module, target_module] = (
                connectivity.loc[source_module, target_module] / denominator if denominator else 0.0
            )
    return (tf_edges, hubs, connectivity)


# Program scoring


def _score_programs(
    expr: ad.AnnData, program_labels: List[str], program_genes: Dict[str, List[str]], cfg: Config, *, large_mode: bool
) -> Tuple[List[str], pd.DataFrame]:
    """Optional per-cell program scores.

    LARGE mode still uses the full dataset so the biological meaning remains
    unchanged, but scores are generated one program at a time and no expression
    copy is created here.
    """
    if not cfg.modules.score_programs:
        return ([], pd.DataFrame())
    import scanpy as sc

    if large_mode:
        logger.info(
            "modules LARGE: computing per-cell program scores for %d program(s) "
            "over all %d cells; this is optional and may be time-consuming",
            len(program_labels),
            expr.n_obs,
        )
    score_columns: List[str] = []
    for i, program in enumerate(program_labels, start=1):
        genes = program_genes[program]
        if not genes:
            continue
        column = f"program_{program}_score"
        sc.tl.score_genes(expr, genes, score_name=column, ctrl_size=50)
        score_columns.append(column)
        if large_mode:
            logger.info("modules LARGE: program score %d/%d complete (%s)", i, len(program_labels), program)
    program_activity = pd.DataFrame()
    key = cfg.modules.cluster_key
    if key in expr.obs and score_columns:
        activity = expr.obs.groupby(key, observed=True)[score_columns].mean()
        activity.columns = [program for program in program_labels if (f"program_{program}_score" in score_columns)]
        program_activity = activity.T
        program_activity.index.name = "program"
    return (score_columns, program_activity)


# Orchestrator


def compute_modules(expr: ad.AnnData, cfg: Config, regression=None) -> Optional[ModulesResults]:
    """Run the adaptive module/program analysis.

    With ``modules.effect_source: regression`` the effect matrix and DE mask come
    from ``regression`` (a :class:`~perturbseq_pipeline.regression.RegressionResults`)
    instead of the pseudobulk contrast; everything downstream is unchanged.
    """
    mcfg = cfg.modules
    use_regression = mcfg.effect_source == "regression"
    if use_regression and regression is None:
        raise ValueError(
            "modules.effect_source is 'regression' but no regression result was given; "
            "the caller must skip the stage rather than fall back to pseudobulk effects"
        )
    targets = select_perturbations(expr, cfg)
    if use_regression:
        targets = [t for t in targets if t in regression.log2fc.index]
    if len(targets) < mcfg.min_perturbations:
        logger.info(
            "modules: only %d perturbation(s) with >=%d cells (need %d) — skipping.",
            len(targets),
            mcfg.min_cells_per_perturbation,
            mcfg.min_perturbations,
        )
        return None
    large_mode = cfg.use_large_mode(expr.n_obs, n_perturbations=len(targets))
    execution_mode = cfg.execution_mode(expr.n_obs, n_perturbations=len(targets))
    logger.info(
        "modules: execution mode = %s (%d cells, %d candidate perturbations)",
        execution_mode.upper(),
        expr.n_obs,
        len(targets),
    )
    genes = list(regression.log2fc.columns) if use_regression else select_genes(expr, cfg, large_mode=large_mode)
    if len(genes) < mcfg.min_genes:
        logger.info("modules: only %d selected gene(s) (need %d) — skipping.", len(genes), mcfg.min_genes)
        return None
    # Effect matrix
    if use_regression:
        effect = regression.log2fc.loc[targets, genes].copy()
        de_mask = (regression.fdr.loc[targets, genes] < mcfg.de_fdr_alpha) & (effect.abs() > mcfg.hub_lfc_threshold)
        control = "regression"
    else:
        effect, control, de_mask = build_effect_matrix(expr, genes, targets, cfg)
    n_cells = expr.obs[OBS_TARGET].astype(str).value_counts()
    logger.info(
        "modules: effect matrix %d perturbations x %d genes "
        "(log2FC vs %s); median %d significant DE genes/perturbation",
        effect.shape[0],
        effect.shape[1],
        control,
        int(de_mask.sum(axis=1).median()),
    )
    # Gene programs
    (gene_program, gene_order, program_labels) = cluster_axis(
        effect.T,
        mcfg.program_correlation,
        mcfg.linkage_method,
        mcfg.n_programs,
        mcfg.cluster_distance_threshold,
        prefix="P",
        cfg=cfg,
    )
    # Perturbation modules
    (pert_module, pert_order, module_labels) = cluster_axis(
        effect,
        mcfg.module_correlation,
        mcfg.linkage_method,
        mcfg.n_modules,
        mcfg.cluster_distance_threshold,
        prefix="M",
        cfg=cfg,
    )
    logger.info("modules: %d gene program(s), %d co-functional module(s)", len(program_labels), len(module_labels))
    # Module-program matrix
    module_program = module_program_strength(effect, gene_program, pert_module, program_labels, module_labels)
    # Network
    (tf_edges, hubs, connectivity) = tf_network(effect, de_mask, pert_module, n_cells, cfg)
    # Program membership
    program_genes = {program: (gene_program.index[gene_program == program].tolist()) for program in program_labels}
    module_members = {module: (pert_module.index[pert_module == module].tolist()) for module in module_labels}
    # Optional per-cell program scoring
    (score_columns, program_activity) = _score_programs(expr, program_labels, program_genes, cfg, large_mode=large_mode)
    # Tables
    modules_table = pd.DataFrame(
        {
            "target_gene": (pert_module.index),
            "module": (pert_module.to_numpy()),
            "n_cells": [int(n_cells.get(target, 0)) for target in pert_module.index],
            "n_de_genes": [int(de_mask.loc[target].sum()) for target in pert_module.index],
        }
    ).sort_values(["module", "n_de_genes"], ascending=[True, False], ignore_index=True)
    gene_sizes = gene_program.value_counts()
    programs_table = pd.DataFrame(
        {
            "gene": (gene_program.index),
            "program": (gene_program.to_numpy()),
            "program_size": [int(gene_sizes.get(program, 0)) for program in gene_program.to_numpy()],
        }
    ).sort_values("program", ignore_index=True)
    # Program pathway enrichment & biological annotation
    enrichment_df = pd.DataFrame()
    prog_annotations: Dict[str, str] = {p: "unannotated" for p in program_labels}
    display_labels: Dict[str, str] = {p: p for p in program_labels}
    prog_summary = pd.DataFrame(
        [
            {
                "program_id": p,
                "annotation": "unannotated",
                "display_label": p,
                "top_term": "None",
                "gene_set_source": "None",
                "fdr": np.nan,
                "program_size": len(program_genes.get(p, [])),
                "top_genes": ", ".join(program_genes.get(p, [])[:8]),
                "member_genes": ", ".join(program_genes.get(p, [])),
            }
            for p in program_labels
        ]
    )
    pe_cfg = getattr(mcfg, "program_enrichment", None)
    enrichment_error = ""
    if pe_cfg is not None and getattr(pe_cfg, "enabled", True):
        try:
            enrichment_df, prog_annotations, prog_summary, display_labels = run_program_enrichment(
                program_genes=program_genes, universe=genes, cfg=pe_cfg, species=cfg.input.species
            )
            n_annotated = sum(1 for a in prog_annotations.values() if a != "unannotated")
            logger.info(
                "modules: program enrichment complete (%d/%d programs annotated)", n_annotated, len(program_labels)
            )
        except Exception as exc:
            # Typically the MSigDB download (no network on a compute node, a Broad
            # server error). The programs stay 'unannotated' — which is NOT a negative
            # result — so the error travels with the results into the report.
            enrichment_error = f"{type(exc).__name__}: {exc}"
            if getattr(pe_cfg, "require", False):
                raise RuntimeError(
                    "modules.program_enrichment.require is true and program enrichment failed: " + enrichment_error
                ) from exc
            logger.warning(
                "modules: program enrichment was NOT run (%s); programs are reported as unannotated, not as negative",
                enrichment_error,
            )
    # Result
    if use_regression:
        note = (
            "Effect matrix = membership-regression log2 effects (adjusted for co-carried targets and covariates), "
            "DE = regression permutation FDR; genes = the regression's gene set."
        )
    else:
        note = (
            "STANDARD mode: original dense selected-gene implementation."
            if not large_mode
            else (
                "LARGE mode: marker discovery used a bounded cluster-stratified "
                "cell sample; perturbation effects used all cells through sparse, "
                f"{cfg.scaling.effect_gene_chunk}-gene chunked sufficient statistics."
            )
        )
    return ModulesResults(
        effect_matrix=effect,
        gene_programs=programs_table,
        modules=modules_table,
        module_program=module_program,
        program_activity=program_activity,
        tf_edges=tf_edges,
        hubs=hubs,
        module_connectivity=connectivity,
        control=control,
        n_programs=len(program_labels),
        n_modules=len(module_labels),
        program_correlation=mcfg.program_correlation,
        module_correlation=mcfg.module_correlation,
        linkage_method=mcfg.linkage_method,
        program_labels=program_labels,
        module_labels=module_labels,
        gene_order=gene_order,
        perturbation_order=pert_order,
        program_genes=program_genes,
        module_members=module_members,
        score_columns=score_columns,
        program_enrichment=enrichment_df,
        program_annotations=prog_annotations,
        program_display_labels=display_labels,
        program_summary=prog_summary,
        program_enrichment_error=enrichment_error,
        execution_mode=execution_mode,
        note=note,
    )
