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

Two departures from the reference implementation:

* **Vectorized.** The original loops over every cell in Python and does a
  ``.loc`` lookup per neighbourhood. Each perturbation here is a single sparse
  matrix-vector product, which is what makes ~60 targets over 100k cells
  practical.
* **Denominator.** The original divides by the requested ``n_neighbors``, while
  a scanpy neighbour graph stores ``n_neighbors - 1`` entries per row (self is
  excluded). Dividing by the actual neighbour count avoids a systematic
  under-estimate of the local fraction; at k = 300 the difference is ~0.3%, so
  scores remain comparable with pertTF's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import CLUSTER_KEY
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET

logger = logging.getLogger(__name__)

#: Prefix of the per-cell score columns written into ``obs``.
LOCHNESS_PREFIX = "lochness_"
#: Column holding each cell's score for its *own* perturbation.
LOCHNESS_SELF = "lochness_self"
#: Key of the dedicated neighbour graph.
NEIGHBORS_KEY = "lochness_nn"


@dataclass
class LochnessResults:
    """Per-cell lochNESS scores and their per-target summary."""

    #: One row per target gene.
    summary: pd.DataFrame
    #: ``{target: per-cell score array}`` aligned to ``expr.obs_names``.
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Each cell's score for its own perturbation (NaN where unassigned).
    self_score: Optional[np.ndarray] = None
    #: Mean score per (target, cluster), for the heatmap.
    by_cluster: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_neighbors: int = 0
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""

    @property
    def targets(self) -> List[str]:
        return list(self.summary["target_gene"]) if not self.summary.empty else []

    def top_targets(self, n: int) -> List[str]:
        """Targets whose cells sit in the most enriched neighbourhoods."""
        if self.summary.empty:
            return []
        return list(self.summary.head(n)["target_gene"])


# ---------------------------------------------------------------------------
# Neighbour graph
# ---------------------------------------------------------------------------


def _build_neighbor_graph(expr: ad.AnnData, cfg: Config) -> sparse.csr_matrix:
    """Build (or reuse) the large-k neighbour graph the score needs.

    The clustering graph is far too small for this: with ``n_neighbors = 15``
    and 60 targets, a neighbourhood is expected to contain a quarter of a cell
    of any given perturbation, so the local fraction is dominated by sampling
    noise. pertTF uses k = 300, and that default is kept.
    """
    import scanpy as sc

    lcfg = cfg.lochness
    key = NEIGHBORS_KEY
    conns_key = f"{key}_distances"

    if conns_key in expr.obsp and not lcfg.recompute_neighbors:
        logger.info("Reusing the existing %r neighbour graph", key)
        return sparse.csr_matrix(expr.obsp[conns_key])

    use_rep = lcfg.use_rep
    if use_rep is None:
        # Prefer the batch-corrected embedding when one exists, so neighbourhoods
        # are not defined by which lane a cell came from.
        use_rep = "X_pca_harmony" if "X_pca_harmony" in expr.obsm else "X_pca"
    if use_rep not in expr.obsm:
        raise ValueError(
            f"lochness.use_rep={use_rep!r} is not in obsm "
            f"(available: {sorted(expr.obsm)}); clustering must run first."
        )

    k = int(min(lcfg.n_neighbors, max(expr.n_obs - 1, 2)))
    if k < lcfg.n_neighbors:
        logger.warning(
            "Only %d cells available; using n_neighbors=%d instead of %d",
            expr.n_obs,
            k,
            lcfg.n_neighbors,
        )
    logger.info(
        "Building the lochNESS neighbour graph (k=%d, rep=%s) over %d cells",
        k,
        use_rep,
        expr.n_obs,
    )
    n_pcs = min(lcfg.n_pcs, expr.obsm[use_rep].shape[1]) if lcfg.n_pcs else None
    sc.pp.neighbors(
        expr,
        n_neighbors=k,
        n_pcs=n_pcs,
        use_rep=use_rep,
        key_added=key,
        random_state=cfg.run.seed,
    )
    return sparse.csr_matrix(expr.obsp[conns_key])


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def _adjacency(graph: sparse.csr_matrix) -> tuple:
    """Binary adjacency plus the neighbour count per cell."""
    adj = graph.copy()
    adj.data = np.ones_like(adj.data, dtype=np.float64)
    counts = np.asarray(adj.sum(axis=1)).ravel()
    # A cell with no neighbours would divide by zero; it gets NaN below.
    counts[counts == 0] = np.nan
    return adj, counts


def lochness_score(
    adj: sparse.csr_matrix,
    neighbor_counts: np.ndarray,
    indicator: np.ndarray,
    overall_fraction: float,
) -> np.ndarray:
    """lochNESS for one perturbation, for every cell at once.

    ``indicator`` is 1 for cells carrying the perturbation. The local count is a
    single sparse matrix-vector product, which is what replaces the reference's
    per-cell Python loop.
    """
    if overall_fraction <= 0:
        return np.full(adj.shape[0], np.nan)
    local = adj @ indicator.astype(np.float64)
    local_fraction = local / neighbor_counts
    return local_fraction / overall_fraction - 1.0


def compute_lochness(expr: ad.AnnData, cfg: Config) -> Optional[LochnessResults]:
    """Score every target gene in every cell's neighbourhood."""
    lcfg = cfg.lochness
    if not lcfg.enabled:
        logger.info("lochNESS disabled (lochness.enabled: false)")
        return None

    key = lcfg.genotype_key
    if key not in expr.obs.columns:
        raise ValueError(
            f"lochness.genotype_key={key!r} is not an obs column "
            f"(available: {sorted(expr.obs.columns)[:20]})"
        )

    labels = expr.obs[key].astype(str).to_numpy()
    klass = (
        expr.obs[OBS_CLASS].astype(str).to_numpy()
        if OBS_CLASS in expr.obs.columns
        else np.full(expr.n_obs, CLASS_TARGETING)
    )

    graph = _build_neighbor_graph(expr, cfg)
    adj, neighbor_counts = _adjacency(graph)
    k_actual = float(np.nanmedian(neighbor_counts))
    logger.info("Neighbour graph: median %d neighbours per cell", int(k_actual))

    # Overall fractions are taken over every cell, exactly as in pertTF. Cells
    # with an ambiguous or unassigned guide dilute the local and the overall
    # fraction by the same factor, so the ratio is unaffected by them.
    overall = pd.Series(labels).value_counts(normalize=True).to_dict()

    counts = pd.Series(labels[klass == CLASS_TARGETING]).value_counts()
    candidates = sorted(counts[counts >= lcfg.min_cells_per_target].index)
    skipped = pd.DataFrame(
        [
            {
                "target_gene": t,
                "n_cells": int(n),
                "reason": f"fewer than {lcfg.min_cells_per_target} cells",
            }
            for t, n in counts.items()
            if n < lcfg.min_cells_per_target
        ]
    )
    if not candidates:
        return LochnessResults(
            summary=pd.DataFrame(),
            skipped=skipped,
            note="no target had enough cells for a lochNESS score",
            n_neighbors=int(k_actual),
        )

    clusters = (
        expr.obs[CLUSTER_KEY].astype(str).to_numpy()
        if CLUSTER_KEY in expr.obs.columns
        else None
    )

    scores: Dict[str, np.ndarray] = {}
    rows: List[dict] = []
    by_cluster: Dict[str, Dict[str, float]] = {}
    rng = np.random.default_rng(cfg.run.seed)

    for gene in candidates:
        indicator = (labels == gene).astype(np.float64)
        score = lochness_score(adj, neighbor_counts, indicator, overall.get(gene, 0.0))
        if lcfg.noise_delta > 0:
            # pertTF adds a little noise to stabilise model training; off by
            # default here, where the scores are read rather than trained on.
            score = score + rng.normal(0, lcfg.noise_delta, size=score.shape)
        scores[gene] = score

        own = indicator.astype(bool)
        row = {
            "target_gene": gene,
            "n_cells": int(own.sum()),
            "overall_fraction_pct": 100 * overall.get(gene, 0.0),
            "mean_lochness_all_cells": float(np.nanmean(score)),
            "mean_lochness_in_own_cells": float(np.nanmean(score[own])),
            "max_lochness": float(np.nanmax(score)),
            "pct_cells_enriched": float(100 * np.nanmean(score > lcfg.enrichment_cut)),
        }
        if clusters is not None:
            per_cluster = pd.Series(score).groupby(clusters).mean()
            by_cluster[gene] = per_cluster.to_dict()
            row["top_cluster"] = str(per_cluster.idxmax())
            row["top_cluster_mean"] = float(per_cluster.max())
        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values("mean_lochness_in_own_cells", ascending=False)
        .reset_index(drop=True)
    )

    # Each cell's score for its own perturbation: how clustered a cell is with
    # others sharing its guide.
    self_score = np.full(expr.n_obs, np.nan)
    for gene, score in scores.items():
        mask = labels == gene
        self_score[mask] = score[mask]
    ntc_mask = klass == CLASS_NTC
    if ntc_mask.any():
        ntc_label = labels[ntc_mask][0]
        if ntc_label in overall:
            ntc_score = lochness_score(
                adj, neighbor_counts, ntc_mask.astype(np.float64), overall[ntc_label]
            )
            self_score[ntc_mask] = ntc_score[ntc_mask]

    logger.info(
        "lochNESS: %d target(s) scored; strongest self-enrichment %s "
        "(mean %.2f in its own cells)",
        len(summary),
        summary.iloc[0]["target_gene"],
        summary.iloc[0]["mean_lochness_in_own_cells"],
    )

    return LochnessResults(
        summary=summary,
        scores=scores,
        self_score=self_score,
        by_cluster=pd.DataFrame(by_cluster).T if by_cluster else pd.DataFrame(),
        n_neighbors=int(k_actual),
        skipped=skipped,
    )


def attach_scores(expr: ad.AnnData, results: Optional[LochnessResults]) -> ad.AnnData:
    """Write per-cell lochNESS columns into ``obs``."""
    if results is None or not results.scores:
        return expr
    for gene, score in results.scores.items():
        expr.obs[f"{LOCHNESS_PREFIX}{gene}"] = score
    if results.self_score is not None:
        expr.obs[LOCHNESS_SELF] = results.self_score
    return expr
