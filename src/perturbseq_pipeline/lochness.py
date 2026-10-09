"""lochNESS: local neighbourhood enrichment of each perturbation.

Ported from pertTF (``perttf.model.composition_change_analysis``), the score
asks, for every cell and every perturbation *g*::

    lochNESS(cell, g) = local_fraction(g) / overall_fraction(g) - 1

where ``local_fraction`` is the share of that cell's nearest neighbours carrying
*g*, and ``overall_fraction`` is *g*'s share of the whole dataset. A score of
**0** means the perturbation appears in the neighbourhood exactly as often as
chance would predict; **> 0** means it is locally over-represented; **< 0**
under-represented. Because it is computed per cell, it maps *where* in the
manifold a perturbation accumulates rather than only whether it does.

This complements section 4. Cluster enrichment asks the same question against
discrete Leiden clusters and answers it with a significance test; lochNESS is
continuous and cluster-free, so it also picks up structure that falls inside a
cluster or straddles two.

Three departures from the reference implementation:

* **Vectorized.** The original loops over every cell in Python and does a
  ``.loc`` lookup per neighbourhood. Each perturbation here is a single sparse
  matrix-vector product, which is what makes ~60 targets over 100k cells
  practical.

* **Denominator.** The original divides by the requested ``n_neighbors``, while
  a scanpy neighbour graph stores ``n_neighbors - 1`` entries per row (self is
  excluded). Dividing by the actual neighbour count avoids a systematic
  under-estimate of the local fraction; at k = 300 the difference is ~0.3%, so
  scores remain comparable with pertTF's.

* **Million-cell execution mode.** Materialising a lochNESS vector for every
  perturbation scales as ``n_cells × n_perturbations``. This becomes
  impractical for genome-scale datasets such as KOLF, where millions of cells
  and >10,000 perturbations would create tens of billions of scores. For such
  datasets the pipeline computes ``lochness_self`` directly: each cell's
  lochNESS score for the perturbation actually carried by that cell. The
  lochNESS definition is unchanged; only unused cell × perturbation scores are
  omitted. Same-label neighbour counting is accelerated with a Numba
  shared-memory parallel implementation when Numba is available, with a
  numerically equivalent pure-Python fallback otherwise.

Standard-sized datasets retain the original full-score behaviour. Large
datasets retain per-cell ``lochness_self``, target-level summaries and
target-by-cluster summaries without materialising thousands of per-target
columns in ``AnnData.obs``.
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

from .cluster import CLUSTER_KEY
from .compute import log_compute_decision, resolve_stage_backend
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS

logger = logging.getLogger(__name__)


# Optional Numba backend

try:
    import numba
    from numba import njit, prange

    NUMBA_AVAILABLE = True

except ImportError:
    numba = None
    NUMBA_AVAILABLE = False


# Public keys

#: Prefix of the per-cell full-score columns written into ``obs``.
LOCHNESS_PREFIX = "lochness_"

#: Each cell's score for the perturbation carried by that cell.
LOCHNESS_SELF = "lochness_self"

#: Key of the dedicated lochNESS neighbour graph.
NEIGHBORS_KEY = "lochness_nn"


# Scalability safeguards
@dataclass
class LochnessResults:
    """Per-cell lochNESS scores and their per-target summaries."""

    #: One row per target gene.
    summary: pd.DataFrame
    #: ``{target: per-cell score array}`` aligned to ``expr.obs_names``.
    #: Populated only in standard full-score mode.
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Each cell's score for its own perturbation.
    self_score: Optional[np.ndarray] = None
    #: Mean score per (target, cluster), for heatmaps.
    by_cluster: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Actual median neighbour count.
    n_neighbors: int = 0
    #: Targets excluded because they had too few cells.
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Human-readable note describing execution mode.
    note: str = ""
    #: True when only ``lochness_self`` was materialised.
    self_only: bool = False
    #: High-MOI mode: sparse cells x targets scores for member cells (``obsm['lochness_membership']``).
    membership_scores: Optional[sparse.csr_matrix] = None
    membership_targets: List[str] = field(default_factory=list)
    membership_aware: bool = False

    @property
    def targets(self) -> List[str]:
        if self.summary.empty:
            return []
        return list(self.summary["target_gene"])

    def top_targets(self, n: int) -> List[str]:
        """Targets whose own cells show strongest local enrichment."""
        if self.summary.empty:
            return []
        return list(self.summary.head(n)["target_gene"])


# Numba kernel

if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, nogil=True)
    def _count_same_label_neighbors_numba(indptr: np.ndarray, indices: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Count same-perturbation neighbours for every CSR row.

        Each focal cell is independent, so rows can be distributed safely
        across Numba threads with ``prange``.
        """
        n_cells = indptr.shape[0] - 1
        counts = np.zeros(n_cells, dtype=np.float32)
        for i in prange(n_cells):
            start = indptr[i]
            end = indptr[i + 1]
            focal_code = codes[i]
            same = 0
            for j in range(start, end):
                neighbour = indices[j]
                if codes[neighbour] == focal_code:
                    same += 1
            counts[i] = same
        return counts


def _count_same_label_neighbors_python(indptr: np.ndarray, indices: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Pure-Python fallback for same-label neighbour counting."""
    n_cells = indptr.shape[0] - 1
    counts = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        start = indptr[i]
        end = indptr[i + 1]
        neighbours = indices[start:end]
        if neighbours.size:
            counts[i] = np.count_nonzero(codes[neighbours] == codes[i])
    return counts


# Neighbour graph


def _build_neighbor_graph(expr: ad.AnnData, cfg: Config) -> sparse.csr_matrix:
    """Build or reuse the large-k neighbour graph required by lochNESS.

    The standard clustering graph is often too small for perturbation
    composition analysis. pertTF uses k ~= 300, and this implementation keeps
    the dedicated lochNESS graph separate from the clustering graph.
    """
    import scanpy as sc

    lcfg = cfg.lochness
    key = NEIGHBORS_KEY
    distances_key = f"{key}_distances"
    if distances_key in expr.obsp and not lcfg.recompute_neighbors:
        logger.info("Reusing existing %r neighbour graph", key)
        return sparse.csr_matrix(expr.obsp[distances_key])
    use_rep = lcfg.use_rep
    if use_rep is None:
        # Prefer the batch-corrected embedding when available.
        use_rep = "X_pca_harmony" if "X_pca_harmony" in expr.obsm else "X_pca"
    if use_rep not in expr.obsm:
        raise ValueError(
            f"lochness.use_rep={use_rep!r} is not in obsm (available: {sorted(expr.obsm)}); clustering must run first."
        )
    k = int(min(lcfg.n_neighbors, max(expr.n_obs - 1, 2)))
    if k < lcfg.n_neighbors:
        logger.warning("Only %d cells available; using n_neighbors=%d instead of %d", expr.n_obs, k, lcfg.n_neighbors)
    n_pcs = min(int(lcfg.n_pcs), expr.obsm[use_rep].shape[1]) if lcfg.n_pcs else None
    logger.info("Building lochNESS neighbour graph (k=%d, rep=%s) over %d cells", k, use_rep, expr.n_obs)
    sc.pp.neighbors(expr, n_neighbors=k, n_pcs=n_pcs, use_rep=use_rep, key_added=key, random_state=cfg.run.seed)
    gc.collect()
    return sparse.csr_matrix(expr.obsp[distances_key])


# Graph helpers


def _adjacency(graph: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Return binary adjacency plus actual neighbour count per cell."""
    adj = graph.tocsr(copy=True)
    # LochNESS only needs edge presence, not edge weight.
    # uint8 keeps the adjacency much smaller than float64.
    adj.data = np.ones(adj.data.shape, dtype=np.uint8)
    neighbour_counts = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    # Avoid division by zero.
    neighbour_counts[neighbour_counts == 0] = np.nan
    return (adj, neighbour_counts)


# Standard full-score implementation


def lochness_score(
    adj: sparse.csr_matrix, neighbor_counts: np.ndarray, indicator: np.ndarray, overall_fraction: float
) -> np.ndarray:
    """Calculate lochNESS for one perturbation over all cells.

    ``indicator`` is 1 for cells carrying the perturbation.

    The local perturbation count is obtained by one sparse matrix-vector
    multiplication.
    """
    if overall_fraction <= 0:
        return np.full(adj.shape[0], np.nan, dtype=np.float32)
    indicator = np.asarray(indicator, dtype=np.float32)
    local_count = adj @ indicator
    local_fraction = np.asarray(local_count, dtype=np.float32) / neighbor_counts
    score = local_fraction / np.float32(overall_fraction) - np.float32(1.0)
    return score.astype(np.float32, copy=False)


# Million-cell self-score implementation


def _compute_self_lochness(adj: sparse.csr_matrix, neighbor_counts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute lochNESS for each cell's own perturbation only.

    For cell i carrying perturbation g_i:

        lochness_self(i)
            = local_fraction_i(g_i) / overall_fraction(g_i) - 1

    This is mathematically the same value obtained from the full lochNESS
    matrix at entry ``[cell_i, perturbation_g_i]``. The difference is that the
    unused off-target columns are never created.

    Same-label neighbour counting is executed with a parallel Numba kernel when
    available. Otherwise a numerically equivalent pure-Python implementation is
    used.
    """
    n_cells = adj.shape[0]
    # Convert arbitrary string labels to compact integer IDs.
    codes, unique_labels = pd.factorize(labels, sort=False)
    codes = codes.astype(np.int32, copy=False)
    valid_codes = codes >= 0
    label_counts = np.bincount(codes[valid_codes], minlength=len(unique_labels))
    overall_fraction = label_counts.astype(np.float64) / float(n_cells)
    # CSR indexing arrays.
    #
    # Keep native-width integers when possible rather than forcing a copy to
    # int64 unnecessarily.
    indptr = np.asarray(adj.indptr)
    indices = np.asarray(adj.indices)
    if NUMBA_AVAILABLE:
        logger.info(
            "Computing self-lochNESS with Numba parallel backend (%d cells, %d neighbour edges, %d threads)",
            n_cells,
            len(indices),
            numba.get_num_threads(),
        )
        same_neighbor_counts = _count_same_label_neighbors_numba(indptr, indices, codes)
    else:
        logger.warning(
            "Numba is unavailable; using pure-Python self-lochNESS backend. "
            "Install numba for substantially faster million-cell execution."
        )
        same_neighbor_counts = _count_same_label_neighbors_python(indptr, indices, codes)
    local_fraction = same_neighbor_counts / neighbor_counts
    denominators = np.full(n_cells, np.nan, dtype=np.float32)
    valid = valid_codes & np.isfinite(neighbor_counts)
    denominators[valid] = overall_fraction[codes[valid]].astype(np.float32)
    valid &= denominators > 0
    self_score = np.full(n_cells, np.nan, dtype=np.float32)
    self_score[valid] = local_fraction[valid] / denominators[valid] - np.float32(1.0)
    del same_neighbor_counts
    del denominators
    del codes
    del unique_labels
    del label_counts
    gc.collect()
    return self_score


# Large-mode summaries


def _summarize_self_scores(
    expr: ad.AnnData, labels: np.ndarray, klass: np.ndarray, self_score: np.ndarray, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize self-lochNESS by target and target × Leiden cluster."""
    lcfg = cfg.lochness
    targeting_mask = klass == CLASS_TARGETING
    target_labels = labels[targeting_mask]
    target_scores = self_score[targeting_mask]
    counts = pd.Series(target_labels).value_counts()
    keep_targets = counts[counts >= lcfg.min_cells_per_target].index
    skipped = pd.DataFrame(
        [
            {"target_gene": target, "n_cells": int(n), "reason": (f"fewer than {lcfg.min_cells_per_target} cells")}
            for target, n in counts.items()
            if n < lcfg.min_cells_per_target
        ]
    )
    keep_mask = np.isin(target_labels, keep_targets)
    frame = pd.DataFrame({"target_gene": (target_labels[keep_mask]), "score": (target_scores[keep_mask])})
    if frame.empty:
        return (pd.DataFrame(), skipped, pd.DataFrame())
    grouped = frame.groupby("target_gene", observed=True)
    summary = (
        grouped["score"]
        .agg(["size", "mean", "median", "max"])
        .rename(
            columns={
                "size": "n_cells",
                "mean": ("mean_lochness_in_own_cells"),
                "median": ("median_lochness_in_own_cells"),
                "max": "max_lochness",
            }
        )
        .reset_index()
    )
    enriched_fraction = (
        frame.assign(enriched=(frame["score"] > lcfg.enrichment_cut))
        .groupby("target_gene", observed=True)["enriched"]
        .mean()
        .mul(100.0)
    )
    summary["pct_own_cells_enriched"] = summary["target_gene"].map(enriched_fraction).astype(float)
    summary = summary.sort_values("mean_lochness_in_own_cells", ascending=False).reset_index(drop=True)
    by_cluster = pd.DataFrame()
    if CLUSTER_KEY in expr.obs.columns:
        target_clusters = expr.obs.loc[targeting_mask, CLUSTER_KEY].astype(str).to_numpy()
        cluster_frame = pd.DataFrame(
            {
                "target_gene": (target_labels[keep_mask]),
                "cluster": (target_clusters[keep_mask]),
                "score": (target_scores[keep_mask]),
            }
        )
        by_cluster = cluster_frame.groupby(["target_gene", "cluster"], observed=True)["score"].mean().unstack()
    return (summary, skipped, by_cluster)


# Main API


def compute_lochness(expr: ad.AnnData, cfg: Config) -> Optional[LochnessResults]:
    """Score perturbation neighbourhood enrichment.

    Standard mode
    -------------
    Computes a lochNESS vector for every eligible target over every cell.

    Large mode
    ----------
    Computes only ``lochness_self`` plus target-level summaries, avoiding the
    full cells × perturbations representation.
    """
    lcfg = cfg.lochness
    if not lcfg.enabled:
        logger.info("lochNESS disabled (lochness.enabled: false)")
        return None
    decision = resolve_stage_backend("lochness", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)
    if NUMBA_AVAILABLE and numba is not None:
        try:
            numba.set_num_threads(decision.n_jobs)
        except Exception:
            pass
    key = lcfg.genotype_key
    if key not in expr.obs.columns:
        raise ValueError(
            f"lochness.genotype_key={key!r} is not an obs column (available: {sorted(expr.obs.columns)[:20]})"
        )
    from .high_moi import membership_index

    labels = expr.obs[key].astype(str).to_numpy()
    klass = (
        expr.obs[OBS_CLASS].astype(str).to_numpy()
        if OBS_CLASS in expr.obs.columns
        else np.full(expr.n_obs, CLASS_TARGETING, dtype=object)
    )
    membership = membership_index(expr, cfg)
    if membership is not None:
        target_counts = membership.counts
    else:
        target_counts = pd.Series(labels[klass == CLASS_TARGETING]).value_counts()
    eligible_targets = target_counts[target_counts >= lcfg.min_cells_per_target]
    n_targets = len(eligible_targets)
    large_mode = cfg.use_large_mode(expr.n_obs, n_perturbations=n_targets)
    logger.info("lochNESS input: %d cells, %d eligible target perturbations", expr.n_obs, n_targets)
    if large_mode:
        logger.info(
            "Large-dataset lochNESS mode enabled: "
            "computing lochness_self without "
            "materialising full cell × perturbation scores"
        )
    graph = _build_neighbor_graph(expr, cfg)
    adj, neighbor_counts = _adjacency(graph)
    k_actual = float(np.nanmedian(neighbor_counts))
    logger.info("Neighbour graph: median %d neighbours per cell", int(k_actual))
    if membership is not None:
        return _compute_lochness_membership(expr, cfg, membership, adj, neighbor_counts, k_actual, large_mode, labels, klass)
    # ======================================================================
    # LARGE DATASET PATH
    # ======================================================================
    if large_mode:
        self_score = _compute_self_lochness(adj, neighbor_counts, labels)
        (summary, skipped, by_cluster) = _summarize_self_scores(expr, labels, klass, self_score, cfg)
        logger.info(
            "lochNESS large-data mode complete: %d targets summarized, %d finite self-scores",
            len(summary),
            int(np.isfinite(self_score).sum()),
        )
        return LochnessResults(
            summary=summary,
            scores={},
            self_score=self_score,
            by_cluster=by_cluster,
            n_neighbors=int(k_actual),
            skipped=skipped,
            note=(
                "Large-dataset mode: "
                "only lochness_self was materialised. "
                "The full cell × perturbation score matrix "
                "was intentionally omitted."
            ),
            self_only=True,
        )
    # ======================================================================
    # STANDARD FULL-SCORE PATH
    # ======================================================================
    overall = pd.Series(labels).value_counts(normalize=True).to_dict()
    counts = pd.Series(labels[klass == CLASS_TARGETING]).value_counts()
    candidates = sorted(counts[counts >= lcfg.min_cells_per_target].index)
    skipped = pd.DataFrame(
        [
            {"target_gene": target, "n_cells": int(n), "reason": (f"fewer than {lcfg.min_cells_per_target} cells")}
            for target, n in counts.items()
            if n < lcfg.min_cells_per_target
        ]
    )
    if not candidates:
        return LochnessResults(
            summary=pd.DataFrame(),
            skipped=skipped,
            note=("No target had enough cells for a lochNESS score."),
            n_neighbors=int(k_actual),
        )
    clusters = expr.obs[CLUSTER_KEY].astype(str).to_numpy() if CLUSTER_KEY in expr.obs.columns else None
    scores: Dict[str, np.ndarray] = {}
    rows: List[dict] = []
    by_cluster_dict: Dict[str, Dict[str, float]] = {}
    rng = np.random.default_rng(cfg.run.seed)
    for idx, gene in enumerate(candidates, start=1):
        indicator = (labels == gene).astype(np.float32)
        score = lochness_score(adj, neighbor_counts, indicator, overall.get(gene, 0.0))
        if lcfg.noise_delta > 0:
            score = score + rng.normal(0, lcfg.noise_delta, size=score.shape).astype(np.float32)
        scores[gene] = score
        own_mask = indicator.astype(bool)
        row = {
            "target_gene": gene,
            "n_cells": int(own_mask.sum()),
            "overall_fraction_pct": (100.0 * overall.get(gene, 0.0)),
            "mean_lochness_all_cells": float(np.nanmean(score)),
            "mean_lochness_in_own_cells": float(np.nanmean(score[own_mask])),
            "max_lochness": float(np.nanmax(score)),
            "pct_cells_enriched": float(100.0 * np.nanmean(score > lcfg.enrichment_cut)),
        }
        if clusters is not None:
            per_cluster = pd.Series(score).groupby(clusters).mean()
            by_cluster_dict[gene] = per_cluster.to_dict()
            row["top_cluster"] = str(per_cluster.idxmax())
            row["top_cluster_mean"] = float(per_cluster.max())
        rows.append(row)
        if idx % 100 == 0:
            logger.info("lochNESS full-score progress: %d / %d targets", idx, len(candidates))
    summary = pd.DataFrame(rows).sort_values("mean_lochness_in_own_cells", ascending=False).reset_index(drop=True)
    # Build lochness_self from the full score vectors
    self_score = np.full(expr.n_obs, np.nan, dtype=np.float32)
    for gene, score in scores.items():
        mask = labels == gene
        self_score[mask] = score[mask]
    # NTC self-score
    ntc_mask = klass == CLASS_NTC
    if ntc_mask.any():
        ntc_labels = labels[ntc_mask]
        if ntc_labels.size:
            ntc_label = ntc_labels[0]
            ntc_fraction = overall.get(ntc_label, 0.0)
            if ntc_fraction > 0:
                ntc_score = lochness_score(adj, neighbor_counts, ntc_mask.astype(np.float32), ntc_fraction)
                self_score[ntc_mask] = ntc_score[ntc_mask]
                del ntc_score
    gc.collect()
    logger.info("lochNESS complete: %d target(s) scored", len(summary))
    return LochnessResults(
        summary=summary,
        scores=scores,
        self_score=self_score,
        by_cluster=(pd.DataFrame(by_cluster_dict).T if by_cluster_dict else pd.DataFrame()),
        n_neighbors=int(k_actual),
        skipped=skipped,
        self_only=False,
    )


# High-MOI membership implementation


def _compute_lochness_membership(
    expr: ad.AnnData,
    cfg: Config,
    membership,
    adj: sparse.csr_matrix,
    neighbor_counts: np.ndarray,
    k_actual: float,
    large_mode: bool,
    labels: np.ndarray,
    klass: np.ndarray,
) -> LochnessResults:
    """lochNESS on the membership matrix: indicator(t) = cells carrying t, overall fraction = members / cells.

    Scores are computed for chunks of targets at once (``adj @ membership[:, chunk]``),
    so the cost is one sparse product per chunk in both execution modes. Every
    member cell keeps its score for every target it carries (sparse
    ``membership_scores``); ``self_score`` holds each cell's score for its
    PRIMARY target so ``obs['lochness_self']`` keeps one value per cell. The
    full per-target vectors are kept only in STANDARD mode (report figures).
    """
    lcfg = cfg.lochness
    n = expr.n_obs
    counts = membership.counts
    candidates = sorted(counts[counts >= lcfg.min_cells_per_target].index)
    skipped = pd.DataFrame(
        [
            {"target_gene": t, "n_cells": int(c), "reason": f"fewer than {lcfg.min_cells_per_target} cells"}
            for t, c in counts.items()
            if c < lcfg.min_cells_per_target
        ]
    )
    if not candidates:
        return LochnessResults(
            summary=pd.DataFrame(),
            skipped=skipped,
            note="No target had enough cells for a lochNESS score.",
            n_neighbors=int(k_actual),
            membership_aware=True,
        )
    logger.info(
        "lochNESS membership mode (%s): %d targets, chunks of %d",
        "LARGE, self + member scores only" if large_mode else "STANDARD, full vectors kept",
        len(candidates),
        lcfg.target_chunk_size,
    )
    M = sparse.csr_matrix(membership.indicator(candidates).T, dtype=np.float32)  # cells x targets
    adj32 = sparse.csr_matrix(adj, dtype=np.float32)
    inv_neighbors = (1.0 / np.maximum(neighbor_counts, 1)).astype(np.float32)
    clusters = expr.obs[CLUSTER_KEY].astype(str).to_numpy() if CLUSTER_KEY in expr.obs.columns else None
    rng = np.random.default_rng(cfg.run.seed)
    rows_i: List[np.ndarray] = []
    cols_i: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    self_score = np.full(n, np.nan, dtype=np.float32)
    scores: Dict[str, np.ndarray] = {}
    summary_rows: List[dict] = []
    by_cluster_dict: Dict[str, Dict[str, float]] = {}
    chunk = max(int(lcfg.target_chunk_size), 1)
    for start in range(0, len(candidates), chunk):
        genes = candidates[start : start + chunk]
        local = (adj32 @ M[:, start : start + len(genes)]).toarray().astype(np.float32)
        frac = (counts[genes].to_numpy(dtype=np.float32) / np.float32(n))[None, :]
        block = local * inv_neighbors[:, None] / frac - np.float32(1.0)
        for jj, gene in enumerate(genes):
            score = block[:, jj]
            if lcfg.noise_delta > 0:
                score = score + rng.normal(0, lcfg.noise_delta, size=n).astype(np.float32)
            members = membership.indices(gene)
            rows_i.append(members)
            cols_i.append(np.full(members.size, start + jj, dtype=np.int64))
            vals.append(score[members])
            primary = members[labels[members] == gene]
            self_score[primary] = score[primary]
            if not large_mode:
                scores[gene] = score
            own = score[members]
            row = {
                "target_gene": gene,
                "n_cells": int(members.size),
                "overall_fraction_pct": float(100.0 * members.size / n),
                "mean_lochness_all_cells": float(np.nanmean(score)),
                "mean_lochness_in_own_cells": float(np.nanmean(own)),
                "median_lochness_in_own_cells": float(np.nanmedian(own)),
                "max_lochness": float(np.nanmax(score)),
                "pct_cells_enriched": float(100.0 * np.nanmean(score > lcfg.enrichment_cut)),
                "pct_own_cells_enriched": float(100.0 * np.nanmean(own > lcfg.enrichment_cut)),
            }
            if clusters is not None:
                per_cluster = pd.Series(score).groupby(clusters).mean()
                by_cluster_dict[gene] = per_cluster.to_dict()
                row["top_cluster"] = str(per_cluster.idxmax())
                row["top_cluster_mean"] = float(per_cluster.max())
            summary_rows.append(row)
        logger.info("lochNESS membership progress: %d / %d targets", min(start + chunk, len(candidates)), len(candidates))
    # NTC-only cells: self-score for the non-targeting group, as in the legacy path
    ntc_mask = klass == CLASS_NTC
    if ntc_mask.any():
        ntc_score = lochness_score(adj, neighbor_counts, ntc_mask.astype(np.float32), float(ntc_mask.mean()))
        self_score[ntc_mask] = ntc_score[ntc_mask]
    membership_scores = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))), shape=(n, len(candidates)), dtype=np.float32
    )
    summary = pd.DataFrame(summary_rows).sort_values("mean_lochness_in_own_cells", ascending=False).reset_index(drop=True)
    gc.collect()
    logger.info("lochNESS membership complete: %d target(s) scored, %d finite self-scores", len(summary), int(np.isfinite(self_score).sum()))
    return LochnessResults(
        summary=summary,
        scores=scores,
        self_score=self_score,
        by_cluster=(pd.DataFrame(by_cluster_dict).T if by_cluster_dict else pd.DataFrame()),
        n_neighbors=int(k_actual),
        skipped=skipped,
        note=(
            "High-MOI membership mode: indicator(t) = cells carrying t; lochness_self = each cell's score for its primary "
            "target; obsm['lochness_membership'] holds every (cell, carried target) score."
            + (" Full per-target vectors omitted (LARGE)." if large_mode else "")
        ),
        self_only=large_mode,
        membership_scores=membership_scores,
        membership_targets=list(candidates),
        membership_aware=True,
    )


# Attach scores


def attach_scores(expr: ad.AnnData, results: Optional[LochnessResults]) -> ad.AnnData:
    """Write lochNESS scores into ``AnnData.obs``.

    Standard mode
    -------------
    Writes ``lochness_<target>`` columns plus ``lochness_self``.

    Large mode
    ----------
    Writes only ``lochness_self``. Creating thousands of full per-target
    columns for millions of cells would add enormous memory and file-size
    overhead without benefiting the downstream self-response analysis.
    """
    if results is None:
        return expr
    if not results.self_only and results.scores:
        for gene, score in results.scores.items():
            expr.obs[f"{LOCHNESS_PREFIX}{gene}"] = score.astype(np.float32, copy=False)
    if results.self_score is not None:
        expr.obs[LOCHNESS_SELF] = results.self_score.astype(np.float32, copy=False)
    if results.membership_scores is not None:
        expr.obsm["lochness_membership"] = results.membership_scores
        expr.uns["lochness_membership_targets"] = list(results.membership_targets)
    return expr
