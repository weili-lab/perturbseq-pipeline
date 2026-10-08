"""Enrichment of each perturbation across transcriptional clusters.

This asks a different question from :mod:`perturbseq_pipeline.perturbation`.
That module asks "did the guide knock its target down"; this one asks "did
losing the gene push cells into a particular transcriptional state" — which is
usually the phenotype of interest.

Method
------
For every (target, cluster) pair a 2x2 table is built::

              in cluster    elsewhere
    target        a             b
    reference     c             d

and tested with **Fisher's exact test** (two-sided). An exact test is used
rather than chi-square residuals because a large share of the expected counts
can be small: rare clusters may hold only a handful of reference cells, which
is exactly where strong perturbational effects can occur.

Odds ratios carry a Haldane-Anscombe correction so they remain finite when a
count is zero, and pairs whose reference contributes very few cells are flagged
as low-power rather than quietly trusted.

The reference is either non-targeting cells (``ntc``) or cells assigned to a
different target (``other``). ``other`` is often the preferred reference for
cluster enrichment because the NTC population may be small relative to the
number of perturbations.

With ``stratify_by`` set (for example ``batch`` or ``lane_id``), a
Cochran-Mantel-Haenszel (CMH) test replaces the pooled Fisher test. This prevents
a cluster whose frequency differs between experimental batches from
masquerading as a perturbation phenotype.

Large-dataset execution
-----------------------
The original implementation constructs Boolean cell-level masks repeatedly for
every target × cluster × control × stratum combination. This is straightforward
and remains the default for ordinary Perturb-seq experiments.

For experiments containing millions of cells, however, repeatedly scanning the
full cell vector becomes unnecessarily expensive. For example, a dataset with
2.6 million cells, >10,000 perturbations, dozens of clusters and multiple
batches would otherwise perform many billions of Boolean comparisons.

Large-dataset mode therefore performs a single aggregation step and constructs:

    target × cluster

and, when stratification is requested:

    target × stratum × cluster

count tables.

The same Fisher exact tests, CMH tests, odds ratios and Benjamini-Hochberg FDR
correction are then applied to these aggregated counts. Statistical meaning is
therefore preserved; only the method used to construct the contingency tables
changes.

Execution mode is selected automatically. Standard-sized datasets retain the
original implementation, while million-cell/high-target-count datasets use the
aggregated implementation.

For large datasets only, the global omnibus permutation test is capped at a
smaller number of permutations because it is an exploratory global diagnostic,
not the primary target × cluster inference. Per-pair Fisher/CMH tests and FDR
correction are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from .compute import log_compute_decision, resolve_stage_backend, run_parallel
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_GUIDE, OBS_TARGET
from .perturbation import CONTROL_LABELS, CONTROL_NTC, CONTROL_OTHER, benjamini_hochberg

logger = logging.getLogger(__name__)


# The omnibus permutation is a global exploratory diagnostic. It does not
# determine the pairwise Fisher/CMH significance calls.
LARGE_DATASET_MAX_OMNIBUS_PERMUTATIONS = 100


@dataclass
class EnrichmentResults:
    """Everything the report needs about perturbation/cluster association."""

    #: Long-format results: one row per (target, cluster, control) combination.
    table: pd.DataFrame
    #: Target × cluster percentages of each target's own cells.
    composition: pd.DataFrame
    #: {control: Series of reference percentages per cluster}.
    reference_composition: Dict[str, pd.Series]
    #: Per-target summary of cluster-composition displacement.
    effect_magnitude: pd.DataFrame
    #: Omnibus test of the full target × cluster contingency table.
    omnibus: Dict[str, float] = field(default_factory=dict)
    controls_used: List[str] = field(default_factory=list)
    primary_control: str = CONTROL_OTHER
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    cluster_key: str = "leiden"
    #: True when CMH stratification was used.
    stratified: bool = False
    stratify_by: Optional[str] = None

    @property
    def hits(self) -> pd.DataFrame:
        """Significant target/cluster pairs under the primary control."""
        if self.table.empty:
            return self.table
        return self.table[self.table["significant"]]

    def top_hits(self, n: int) -> pd.DataFrame:
        """Strongest significant effects by absolute log2 odds ratio."""
        h = self.hits
        if h.empty:
            return h
        order = h["log2_odds_ratio"].abs().sort_values(ascending=False).index
        return h.reindex(order).head(n)

    def targets_with_hits(self) -> List[str]:
        if self.hits.empty:
            return []
        return sorted(self.hits["target_gene"].unique())


# Generic helpers


def _cluster_order(values: Sequence[str]) -> List[str]:
    """Sort cluster labels numerically when possible, otherwise lexically."""
    uniq = list(dict.fromkeys(str(v) for v in values))
    try:
        return sorted(uniq, key=lambda x: (float(x), x))
    except ValueError:
        return sorted(uniq)


def _reference_mask(control: str, klass: np.ndarray, targets: np.ndarray, gene: str) -> np.ndarray:
    """Cell-level reference mask used by the original implementation."""
    if control == CONTROL_NTC:
        return klass == CLASS_NTC
    # "other" = targeting cells assigned to every target except the gene
    # currently being tested.
    return (klass == CLASS_TARGETING) & (targets != gene)


def _odds_ratio(a: float, b: float, c: float, d: float, pseudo: float) -> float:
    """Haldane-Anscombe corrected odds ratio."""
    return (a + pseudo) * (d + pseudo) / ((b + pseudo) * (c + pseudo))


# Omnibus


def omnibus_test(contingency: pd.DataFrame, n_permutations: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Global test that perturbation identity and cluster are associated.

    The chi-square statistic is accompanied by an empirical permutation
    p-value because many target × cluster contingency tables have small
    expected values.

    This omnibus test is a global diagnostic; pairwise inference is performed
    separately with Fisher exact or CMH tests.
    """
    if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {}
    # Remove structurally empty rows/columns.
    table = contingency.loc[contingency.sum(axis=1) > 0, contingency.sum(axis=0) > 0]
    dropped_rows = contingency.shape[0] - table.shape[0]
    dropped_cols = contingency.shape[1] - table.shape[1]
    if dropped_rows or dropped_cols:
        logger.info(
            "Omnibus test: dropped %d empty target row(s) and %d empty cluster column(s)", dropped_rows, dropped_cols
        )
    if table.shape[0] < 2 or table.shape[1] < 2:
        logger.warning("Omnibus test needs at least a 2x2 table after pruning; skipping it.")
        return {}
    observed = table.to_numpy(dtype=float)
    chi2, p_chi2, dof, expected = chi2_contingency(observed)
    small = float((expected < 5).mean())
    out = {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_chi2": float(p_chi2),
        "pct_expected_below_5": (100 * small),
        "n_permutations": 0,
        "p_permutation": float("nan"),
    }
    if n_permutations and n_permutations > 0:
        rng = np.random.default_rng(seed)
        row_counts = observed.sum(axis=1).astype(np.int64)
        col_probs = observed.sum(axis=0) / observed.sum()
        n_ge = 0
        for _ in range(int(n_permutations)):
            # Simulate one row per perturbation while preserving each target's
            # total cell count and using the observed global cluster
            # probabilities.
            sim = np.vstack([rng.multinomial(int(n), col_probs) for n in row_counts])
            keep = sim.sum(axis=0) > 0
            if keep.sum() > 1:
                stat = chi2_contingency(sim[:, keep])[0]
            else:
                stat = 0.0
            if stat >= chi2:
                n_ge += 1
        out["n_permutations"] = int(n_permutations)
        out["p_permutation"] = (n_ge + 1) / (n_permutations + 1)
    return out


# Stratified Cochran-Mantel-Haenszel


def _cmh_test(tables: List[np.ndarray]) -> Tuple[float, float]:
    """CMH across strata; return pooled OR and p-value."""
    from statsmodels.stats.contingency_tables import StratifiedTable

    usable = [
        table for table in tables if (table.sum() > 0 and table.sum(axis=1).min() > 0 and table.sum(axis=0).min() > 0)
    ]
    if not usable:
        return (float("nan"), float("nan"))
    st = StratifiedTable([table.T for table in usable])
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = float(st.oddsratio_pooled)
        pvalue = float(st.test_null_odds().pvalue)
    return (pooled, pvalue)


# Guide concordance


def _guide_concordance(
    obs: pd.DataFrame,
    gene: str,
    cluster: str,
    cluster_key: str,
    ref_fraction: float,
    min_cells: int,
    direction: str = "enriched",
) -> Tuple[int, int]:
    """How many independent guides reproduce the target-level direction."""
    if OBS_GUIDE not in obs.columns:
        return 0, 0
    target_values = obs[OBS_TARGET].astype(str)
    class_values = obs[OBS_CLASS].astype(str)
    sub = obs[(target_values == gene) & (class_values == CLASS_TARGETING)]
    if sub.empty:
        return 0, 0
    tested = 0
    concordant = 0
    for _, cells in sub.groupby(sub[OBS_GUIDE].astype(str), observed=True):
        if len(cells) < min_cells:
            continue
        tested += 1
        frac = float((cells[cluster_key].astype(str) == cluster).mean())
        if direction == "depleted":
            agrees = frac < ref_fraction
        else:
            agrees = frac > ref_fraction
        if agrees:
            concordant += 1
    return (concordant, tested)


# ===========================================================================
# ORIGINAL / STANDARD IMPLEMENTATION
# ===========================================================================


def _test_cluster_enrichment_standard(expr: ad.AnnData, cfg: Config) -> EnrichmentResults:
    """Original cell-mask implementation.

    This preserves the previous behaviour for ordinary-sized datasets.
    """
    ecfg = cfg.enrichment
    obs = expr.obs
    cluster_key = ecfg.cluster_key
    logger.info("Cluster enrichment execution mode: STANDARD (cell-level mask implementation)")
    if cluster_key not in obs.columns:
        raise ValueError(
            f"enrichment.cluster_key={cluster_key!r} is not an obs column; clustering must run before enrichment."
        )
    clusters_all = obs[cluster_key].astype(str).to_numpy()
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    cluster_sizes = pd.Series(clusters_all).value_counts()
    ordered_clusters = _cluster_order(clusters_all)
    clusters = [c for c in ordered_clusters if cluster_sizes.get(c, 0) >= ecfg.min_cells_per_cluster]
    dropped_clusters = [c for c in ordered_clusters if c not in clusters]
    if dropped_clusters:
        logger.info(
            "Skipping %d cluster(s) with < %d cells: %s",
            len(dropped_clusters),
            ecfg.min_cells_per_cluster,
            dropped_clusters,
        )
    if not clusters:
        raise ValueError(
            f"No cluster has at least enrichment.min_cells_per_cluster={ecfg.min_cells_per_cluster} cells."
        )
    # Controls
    controls_used = []
    for control in ecfg.controls:
        if control == CONTROL_NTC:
            n_ref = int((klass == CLASS_NTC).sum())
        else:
            n_ref = int((klass == CLASS_TARGETING).sum())
        if n_ref < ecfg.min_reference_cells:
            logger.warning("Control %r has only %d cells; dropping it from the enrichment test.", control, n_ref)
            continue
        controls_used.append(control)
    if not controls_used:
        raise ValueError("No usable control group for the enrichment test.")
    primary = ecfg.primary_control if ecfg.primary_control in controls_used else controls_used[0]
    if primary != ecfg.primary_control:
        logger.warning("Requested enrichment.primary_control %r unavailable; using %r.", ecfg.primary_control, primary)
    # Stratification
    strat_values = None
    stratified = False
    if ecfg.stratify_by:
        if ecfg.stratify_by not in obs.columns:
            raise ValueError(f"enrichment.stratify_by={ecfg.stratify_by!r} is not an obs column.")
        strat_values = obs[ecfg.stratify_by].astype(str).to_numpy()
        n_strata = len(set(strat_values))
        if n_strata < 2:
            logger.info("enrichment.stratify_by=%r has a single level; using pooled Fisher tests.", ecfg.stratify_by)
            strat_values = None
        else:
            stratified = True
            logger.info("Using Cochran-Mantel-Haenszel stratified by %r (%d strata)", ecfg.stratify_by, n_strata)
    # Targets
    target_counts = pd.Series(targets_col[klass == CLASS_TARGETING]).value_counts()
    testable = sorted(target_counts[target_counts >= ecfg.min_cells_per_target].index)
    skipped = pd.DataFrame(
        [
            {
                "target_gene": target,
                "n_cells": int(n),
                "reason": (f"fewer than {ecfg.min_cells_per_target} assigned cells"),
            }
            for target, n in target_counts.items()
            if n < ecfg.min_cells_per_target
        ]
    )
    if not testable:
        logger.warning("No target has enough cells for the enrichment test.")
        return EnrichmentResults(
            table=pd.DataFrame(),
            composition=pd.DataFrame(),
            reference_composition={},
            effect_magnitude=pd.DataFrame(),
            controls_used=controls_used,
            primary_control=primary,
            skipped=skipped,
            cluster_key=cluster_key,
        )
    # Composition
    in_cluster = {cluster: (clusters_all == cluster) for cluster in clusters}
    comp_rows = {}
    for gene in testable:
        mask = (targets_col == gene) & (klass == CLASS_TARGETING)
        n = int(mask.sum())
        comp_rows[gene] = {cluster: (100 * float((mask & in_cluster[cluster]).sum()) / n) for cluster in clusters}
    composition = pd.DataFrame.from_dict(comp_rows, orient="index")[clusters]
    # Reference compositions
    reference_composition = {}
    for control in controls_used:
        if control == CONTROL_NTC:
            mask = klass == CLASS_NTC
        else:
            mask = klass == CLASS_TARGETING
        n = max(int(mask.sum()), 1)
        reference_composition[control] = pd.Series(
            {cluster: (100 * float((mask & in_cluster[cluster]).sum()) / n) for cluster in clusters}
        )
    # Omnibus
    contingency = pd.DataFrame(
        {
            cluster: [
                int(((targets_col == gene) & (klass == CLASS_TARGETING) & in_cluster[cluster]).sum())
                for gene in testable
            ]
            for cluster in clusters
        },
        index=testable,
    )
    omnibus = omnibus_test(contingency, ecfg.permutations, cfg.run.seed)
    if omnibus:
        logger.info(
            "Omnibus association: chi2=%.0f (dof %d), permutation p=%.4g (%.0f%% of expected counts < 5)",
            omnibus["chi2"],
            omnibus["dof"],
            omnibus["p_permutation"],
            omnibus["pct_expected_below_5"],
        )
    # Per-pair tests
    decision = resolve_stage_backend("enrichment", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)
    strata_unique = sorted(set(strat_values)) if stratified else []

    def _eval_target_enrich_std(gene: str) -> List[dict]:
        tmask = (targets_col == gene) & (klass == CLASS_TARGETING)
        n_target = int(tmask.sum())
        gene_rows = []
        for control in controls_used:
            rmask = _reference_mask(control, klass, targets_col, gene)
            n_ref = int(rmask.sum())
            for cluster in clusters:
                cm = in_cluster[cluster]
                a = int((tmask & cm).sum())
                b = n_target - a
                c_ref = int((rmask & cm).sum())
                d = n_ref - c_ref
                pct_t = 100 * a / max(n_target, 1)
                pct_r = 100 * c_ref / max(n_ref, 1)
                odds = _odds_ratio(a, b, c_ref, d, ecfg.odds_pseudocount)
                if stratified:
                    tables = []
                    for stratum in strata_unique:
                        sm = strat_values == stratum
                        tables.append(
                            np.array(
                                [
                                    [int((tmask & cm & sm).sum()), int((tmask & ~cm & sm).sum())],
                                    [int((rmask & cm & sm).sum()), int((rmask & ~cm & sm).sum())],
                                ],
                                dtype=float,
                            )
                        )
                    pooled_or, pval = _cmh_test(tables)
                    if np.isfinite(pooled_or) and pooled_or > 0:
                        odds = pooled_or
                else:
                    _, pval = fisher_exact([[a, b], [c_ref, d]])
                gene_rows.append(
                    {
                        "target_gene": gene,
                        "cluster": cluster,
                        "control": control,
                        "n_target_cells": n_target,
                        "n_in_cluster": a,
                        "pct_of_target": pct_t,
                        "n_reference_cells": n_ref,
                        "pct_of_reference": pct_r,
                        "odds_ratio": odds,
                        "log2_odds_ratio": (float(np.log2(odds)) if odds > 0 else np.nan),
                        "direction": ("enriched" if pct_t > pct_r else "depleted"),
                        "pval": float(pval),
                        "low_power": (c_ref < ecfg.min_reference_cells),
                    }
                )
        return gene_rows

    results_nested = run_parallel(
        _eval_target_enrich_std,
        testable,
        n_jobs=decision.n_jobs,
        blas_threads=cfg.compute.blas_threads_per_worker,
        backend=cfg.compute.cpu_parallel_backend,
    )
    rows: List[dict] = [item for sublist in results_nested for item in sublist]
    return _finalize_enrichment_results(
        expr=expr,
        cfg=cfg,
        table=pd.DataFrame(rows),
        composition=composition,
        reference_composition=reference_composition,
        target_counts=target_counts,
        testable=testable,
        controls_used=controls_used,
        primary=primary,
        skipped=skipped,
        omnibus=omnibus,
        cluster_key=cluster_key,
        stratified=stratified,
    )


# ===========================================================================
# LARGE-DATASET AGGREGATION
# ===========================================================================


def _build_large_count_tables(
    obs: pd.DataFrame, cluster_key: str, stratify_by: Optional[str]
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """Aggregate cell annotations once for large-dataset inference.

    Returns
    -------
    target_cluster
        MultiIndex (target, cluster) -> targeting-cell count.

    ntc_cluster
        cluster -> NTC count.

    target_stratum_cluster
        MultiIndex (target, stratum, cluster) -> count, or None.
    """
    targeting_mask = obs[OBS_CLASS].astype(str) == CLASS_TARGETING
    ntc_mask = obs[OBS_CLASS].astype(str) == CLASS_NTC
    logger.info("Aggregating target × cluster counts from %d cells", len(obs))
    target_frame = pd.DataFrame(
        {
            "target": (obs.loc[targeting_mask, OBS_TARGET].astype(str).to_numpy()),
            "cluster": (obs.loc[targeting_mask, cluster_key].astype(str).to_numpy()),
        }
    )
    target_cluster = target_frame.groupby(["target", "cluster"], observed=True).size().rename("count").to_frame()
    ntc_clusters = obs.loc[ntc_mask, cluster_key].astype(str).value_counts()
    stratified_counts = None
    if stratify_by is not None:
        logger.info("Aggregating target × %s × cluster counts", stratify_by)
        target_strat_frame = pd.DataFrame(
            {
                "target": (obs.loc[targeting_mask, OBS_TARGET].astype(str).to_numpy()),
                "stratum": (obs.loc[targeting_mask, stratify_by].astype(str).to_numpy()),
                "cluster": (obs.loc[targeting_mask, cluster_key].astype(str).to_numpy()),
            }
        )
        stratified_counts = (
            target_strat_frame.groupby(["target", "stratum", "cluster"], observed=True)
            .size()
            .rename("count")
            .to_frame()
        )
    return (target_cluster, ntc_clusters, stratified_counts)


def _test_cluster_enrichment_large(expr: ad.AnnData, cfg: Config) -> EnrichmentResults:
    """Aggregated implementation for million-cell/high-target datasets."""
    ecfg = cfg.enrichment
    obs = expr.obs
    cluster_key = ecfg.cluster_key
    logger.info("Cluster enrichment execution mode: LARGE-DATASET (aggregated contingency tables)")
    if cluster_key not in obs.columns:
        raise ValueError(f"enrichment.cluster_key={cluster_key!r} is not an obs column.")
    cluster_values = obs[cluster_key].astype(str)
    cluster_sizes = cluster_values.value_counts()
    ordered_clusters = _cluster_order(cluster_values)
    clusters = [cluster for cluster in ordered_clusters if cluster_sizes.get(cluster, 0) >= ecfg.min_cells_per_cluster]
    if not clusters:
        raise ValueError(f"No cluster has at least {ecfg.min_cells_per_cluster} cells.")
    klass_series = obs[OBS_CLASS].astype(str)
    target_series = obs[OBS_TARGET].astype(str)
    targeting_mask = klass_series == CLASS_TARGETING
    ntc_mask = klass_series == CLASS_NTC
    target_counts = target_series[targeting_mask].value_counts()
    testable = sorted(target_counts[target_counts >= ecfg.min_cells_per_target].index)
    skipped = pd.DataFrame(
        [
            {
                "target_gene": target,
                "n_cells": int(n),
                "reason": (f"fewer than {ecfg.min_cells_per_target} assigned cells"),
            }
            for target, n in target_counts.items()
            if n < ecfg.min_cells_per_target
        ]
    )
    if not testable:
        return EnrichmentResults(
            table=pd.DataFrame(),
            composition=pd.DataFrame(),
            reference_composition={},
            effect_magnitude=pd.DataFrame(),
            skipped=skipped,
            cluster_key=cluster_key,
        )
    # Controls
    controls_used = []
    total_targeting = int(targeting_mask.sum())
    total_ntc = int(ntc_mask.sum())
    for control in ecfg.controls:
        n_ref = total_ntc if control == CONTROL_NTC else total_targeting
        if n_ref < ecfg.min_reference_cells:
            logger.warning("Control %r has only %d cells; dropping it.", control, n_ref)
            continue
        controls_used.append(control)
    if not controls_used:
        raise ValueError("No usable control group for enrichment.")
    primary = ecfg.primary_control if ecfg.primary_control in controls_used else controls_used[0]
    # Stratification
    stratified = False
    stratify_by = None
    if ecfg.stratify_by:
        if ecfg.stratify_by not in obs.columns:
            raise ValueError(f"enrichment.stratify_by={ecfg.stratify_by!r} is not an obs column.")
        n_strata = obs[ecfg.stratify_by].astype(str).nunique()
        if n_strata >= 2:
            stratified = True
            stratify_by = ecfg.stratify_by
            logger.info("Large-data CMH stratification by %r (%d strata)", stratify_by, n_strata)
    # Aggregate counts once
    (target_cluster_long, ntc_cluster_counts, target_stratum_long) = _build_large_count_tables(
        obs, cluster_key, (stratify_by if stratified else None)
    )
    # Counts for EVERY targeting cell (including targets below min_cells_per_target):
    # the 'other' reference (n_ref = total_targeting - n_target) counts those cells, so
    # the per-cluster reference totals must include them too.
    target_cluster_all = (
        target_cluster_long["count"]
        .unstack("cluster", fill_value=0)
        .reindex(columns=clusters, fill_value=0)
        .astype(np.int64)
    )
    target_cluster = target_cluster_all.reindex(index=testable, fill_value=0)
    # Composition
    denominators = target_cluster.sum(axis=1).replace(0, np.nan)
    composition = target_cluster.div(denominators, axis=0) * 100.0
    composition = composition.fillna(0.0)
    # Total targeting counts by cluster, over ALL targeting cells (not only testable
    # targets) so that c_ref and d = n_ref - c_ref describe the same population.
    targeting_cluster_totals = target_cluster_all.sum(axis=0)
    reference_composition: Dict[str, pd.Series] = {}
    if CONTROL_NTC in controls_used:
        ntc_counts = ntc_cluster_counts.reindex(clusters, fill_value=0)
        reference_composition[CONTROL_NTC] = ntc_counts / max(total_ntc, 1) * 100.0
    if CONTROL_OTHER in controls_used:
        # Overall targeting composition. Gene-specific "other" composition is
        # derived exactly during pairwise inference by subtracting the focal
        # target.
        reference_composition[CONTROL_OTHER] = targeting_cluster_totals / max(total_targeting, 1) * 100.0
    # Omnibus
    configured_permutations = int(ecfg.permutations)
    effective_permutations = min(configured_permutations, LARGE_DATASET_MAX_OMNIBUS_PERMUTATIONS)
    if effective_permutations < configured_permutations:
        logger.warning(
            "Large-dataset mode: omnibus permutations reduced from %d to %d. Pairwise Fisher/CMH tests are unchanged.",
            configured_permutations,
            effective_permutations,
        )
    omnibus = omnibus_test(target_cluster, effective_permutations, cfg.run.seed)
    if omnibus:
        logger.info(
            "Omnibus association: chi2=%.0f (dof %d), permutation p=%.4g (%.0f%% of expected counts < 5)",
            omnibus["chi2"],
            omnibus["dof"],
            omnibus["p_permutation"],
            omnibus["pct_expected_below_5"],
        )
    # Prepare stratified aggregated tables
    strata = []
    target_stratum_cluster = None
    targeting_stratum_totals = None
    ntc_stratum_cluster = None
    ntc_stratum_totals = None
    if stratified:
        strata = sorted(obs[stratify_by].astype(str).unique())
        target_stratum_cluster = target_stratum_long["count"].unstack("cluster", fill_value=0)
        # Target × stratum totals.
        targeting_stratum_totals = target_stratum_cluster.sum(axis=1)
        # NTC × stratum × cluster is also aggregated once.
        ntc_frame = pd.DataFrame(
            {
                "stratum": (obs.loc[ntc_mask, stratify_by].astype(str).to_numpy()),
                "cluster": (obs.loc[ntc_mask, cluster_key].astype(str).to_numpy()),
            }
        )
        ntc_stratum_cluster = ntc_frame.groupby(["stratum", "cluster"], observed=True).size().unstack(fill_value=0)
        ntc_stratum_totals = ntc_stratum_cluster.sum(axis=1)
    # Pairwise inference from compact counts
    decision = resolve_stage_backend("enrichment", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)

    def _eval_target_enrich_large(gene: str) -> List[dict]:
        gene_counts = target_cluster.loc[gene]
        n_target = int(target_counts[gene])
        gene_rows = []
        for control in controls_used:
            if control == CONTROL_NTC:
                n_ref = total_ntc
            else:
                n_ref = total_targeting - n_target
            for cluster in clusters:
                a = int(gene_counts.get(cluster, 0))
                b = n_target - a
                if control == CONTROL_NTC:
                    c_ref = int(ntc_cluster_counts.get(cluster, 0))
                else:
                    c_ref = int(targeting_cluster_totals.get(cluster, 0)) - a
                d = n_ref - c_ref
                pct_t = 100.0 * a / max(n_target, 1)
                pct_r = 100.0 * c_ref / max(n_ref, 1)
                odds = _odds_ratio(a, b, c_ref, d, ecfg.odds_pseudocount)
                if stratified:
                    tables = []
                    for stratum in strata:
                        key = (gene, stratum)
                        if key in target_stratum_cluster.index:
                            target_row = target_stratum_cluster.loc[key]
                            a_s = int(target_row.get(cluster, 0))
                            target_total_s = int(targeting_stratum_totals.get(key, 0))
                        else:
                            a_s = 0
                            target_total_s = 0
                        b_s = target_total_s - a_s
                        if control == CONTROL_NTC:
                            if stratum in ntc_stratum_cluster.index:
                                c_s = int(ntc_stratum_cluster.loc[stratum].get(cluster, 0))
                            else:
                                c_s = 0
                            ref_total_s = int(ntc_stratum_totals.get(stratum, 0))
                        else:
                            if cluster in target_stratum_cluster.columns:
                                try:
                                    cluster_total_s = int(
                                        target_stratum_cluster.xs(stratum, level="stratum")[cluster].sum()
                                    )
                                except KeyError:
                                    cluster_total_s = 0
                            else:
                                cluster_total_s = 0
                            total_targeting_s = 0
                            try:
                                total_targeting_s = int(targeting_stratum_totals.xs(stratum, level="stratum").sum())
                            except KeyError:
                                pass
                            c_s = cluster_total_s - a_s
                            ref_total_s = total_targeting_s - target_total_s
                        d_s = ref_total_s - c_s
                        tables.append(np.array([[a_s, b_s], [c_s, d_s]], dtype=float))
                    pooled_or, pval = _cmh_test(tables)
                    if np.isfinite(pooled_or) and pooled_or > 0:
                        odds = pooled_or
                else:
                    _, pval = fisher_exact([[a, b], [c_ref, d]])
                gene_rows.append(
                    {
                        "target_gene": gene,
                        "cluster": cluster,
                        "control": control,
                        "n_target_cells": n_target,
                        "n_in_cluster": a,
                        "pct_of_target": pct_t,
                        "n_reference_cells": n_ref,
                        "pct_of_reference": pct_r,
                        "odds_ratio": odds,
                        "log2_odds_ratio": (float(np.log2(odds)) if odds > 0 else np.nan),
                        "direction": ("enriched" if pct_t > pct_r else "depleted"),
                        "pval": float(pval),
                        "low_power": (c_ref < ecfg.min_reference_cells),
                    }
                )
        return gene_rows

    results_nested = run_parallel(
        _eval_target_enrich_large,
        testable,
        n_jobs=decision.n_jobs,
        blas_threads=cfg.compute.blas_threads_per_worker,
        backend=cfg.compute.cpu_parallel_backend,
    )
    rows: List[dict] = [item for sublist in results_nested for item in sublist]
    return _finalize_enrichment_results(
        expr=expr,
        cfg=cfg,
        table=pd.DataFrame(rows),
        composition=composition,
        reference_composition=reference_composition,
        target_counts=target_counts,
        testable=testable,
        controls_used=controls_used,
        primary=primary,
        skipped=skipped,
        omnibus=omnibus,
        cluster_key=cluster_key,
        stratified=stratified,
    )


# ===========================================================================
# COMMON FINALIZATION
# ===========================================================================


def _finalize_enrichment_results(
    expr: ad.AnnData,
    cfg: Config,
    table: pd.DataFrame,
    composition: pd.DataFrame,
    reference_composition: Dict[str, pd.Series],
    target_counts: pd.Series,
    testable: List[str],
    controls_used: List[str],
    primary: str,
    skipped: pd.DataFrame,
    omnibus: Dict[str, float],
    cluster_key: str,
    stratified: bool,
) -> EnrichmentResults:
    """Shared FDR, concordance and effect summaries for both execution modes."""
    ecfg = cfg.enrichment
    obs = expr.obs
    if table.empty:
        return EnrichmentResults(
            table=table,
            composition=composition,
            reference_composition=reference_composition,
            effect_magnitude=pd.DataFrame(),
            omnibus=omnibus,
            controls_used=controls_used,
            primary_control=primary,
            skipped=skipped,
            cluster_key=cluster_key,
            stratified=stratified,
            stratify_by=(ecfg.stratify_by if stratified else None),
        )
    # FDR
    table["fdr"] = np.nan
    for control in controls_used:
        mask = table["control"] == control
        table.loc[mask, "fdr"] = benjamini_hochberg(table.loc[mask, "pval"].to_numpy())
    table["significant"] = (table["fdr"] < ecfg.fdr_alpha) & (table["control"] == primary)
    # Guide concordance only for significant pairs
    table["guides_concordant"] = np.nan
    table["guides_tested"] = np.nan
    if ecfg.guide_concordance and OBS_GUIDE in obs.columns:
        # Only materialize the four columns required by guide concordance.
        obs_view = obs[[OBS_GUIDE, OBS_TARGET, OBS_CLASS, cluster_key]]
        for idx in table.index[table["significant"]]:
            gene = table.at[idx, "target_gene"]
            cluster = table.at[idx, "cluster"]
            ref_fraction = table.at[idx, "pct_of_reference"] / 100.0
            conc, tested = _guide_concordance(
                obs_view,
                gene,
                cluster,
                cluster_key,
                ref_fraction,
                ecfg.min_cells_per_guide,
                direction=str(table.at[idx, "direction"]),
            )
            table.at[idx, "guides_concordant"] = conc
            table.at[idx, "guides_tested"] = tested
    # Effect magnitude
    if primary == CONTROL_OTHER:
        # For "other", the exact reference technically differs by target
        # because the focal target is excluded. The global targeting
        # composition is an excellent effect-size reference and matches the
        # previous reporting semantics closely.
        ref = reference_composition[CONTROL_OTHER]
    else:
        ref = reference_composition[primary]
    magnitude = []
    n_sig = table[table["significant"]]["target_gene"].value_counts()
    for gene in testable:
        aligned_ref = ref.reindex(composition.columns, fill_value=0)
        diff = (composition.loc[gene] - aligned_ref).abs().sum() / 2.0
        magnitude.append(
            {
                "target_gene": gene,
                "n_cells": int(target_counts[gene]),
                "composition_shift_pct": float(diff),
                "n_significant_clusters": int(n_sig.get(gene, 0)),
            }
        )
    effect_magnitude = (
        pd.DataFrame(magnitude).sort_values("composition_shift_pct", ascending=False).reset_index(drop=True)
    )
    table = table.sort_values(["significant", "log2_odds_ratio"], ascending=[False, False]).reset_index(drop=True)
    n_hits = int(table["significant"].sum())
    logger.info(
        "Cluster enrichment: %d target(s) x %d cluster(s); %d significant at FDR < %.2f (control: %s)",
        len(testable),
        composition.shape[1],
        n_hits,
        ecfg.fdr_alpha,
        CONTROL_LABELS[primary],
    )
    return EnrichmentResults(
        table=table,
        composition=composition,
        reference_composition=reference_composition,
        effect_magnitude=effect_magnitude,
        omnibus=omnibus,
        controls_used=controls_used,
        primary_control=primary,
        skipped=skipped,
        cluster_key=cluster_key,
        stratified=stratified,
        stratify_by=(ecfg.stratify_by if stratified else None),
    )


# ===========================================================================
# PUBLIC DRIVER
# ===========================================================================


def test_cluster_enrichment(expr: ad.AnnData, cfg: Config) -> EnrichmentResults:
    """Test target genes for enrichment/depletion across clusters.

    Execution mode is selected automatically.

    Standard mode
    -------------
    Uses the original cell-level Boolean-mask implementation.

    Large-dataset mode
    ------------------
    Aggregates target × cluster and target × stratum × cluster counts once and
    performs the same statistical tests on those contingency counts.
    """
    ecfg = cfg.enrichment
    obs = expr.obs
    if OBS_TARGET not in obs.columns:
        raise ValueError(f"{OBS_TARGET!r} is missing from obs.")
    if OBS_CLASS not in obs.columns:
        raise ValueError(f"{OBS_CLASS!r} is missing from obs.")
    targeting = obs[OBS_CLASS].astype(str) == CLASS_TARGETING
    target_counts = obs.loc[targeting, OBS_TARGET].astype(str).value_counts()
    n_testable_targets = int((target_counts >= ecfg.min_cells_per_target).sum())
    large_mode = cfg.use_large_mode(expr.n_obs, n_perturbations=n_testable_targets)
    logger.info("Enrichment input: %d cells, %d testable targets", expr.n_obs, n_testable_targets)
    if large_mode:
        logger.info("Large-dataset enrichment mode selected (%d cells, %d targets)", expr.n_obs, n_testable_targets)
        return _test_cluster_enrichment_large(expr, cfg)
    logger.info("Standard enrichment mode selected")
    return _test_cluster_enrichment_standard(expr, cfg)


# ===========================================================================
# Derived views
# ===========================================================================


def enrichment_matrix(results: EnrichmentResults, control: Optional[str] = None) -> pd.DataFrame:
    """Target × cluster matrix of log2 odds ratios."""
    if results.table.empty:
        return pd.DataFrame()
    control = control or results.primary_control
    sub = results.table[results.table["control"] == control]
    return sub.pivot(index="target_gene", columns="cluster", values="log2_odds_ratio")


def significance_matrix(results: EnrichmentResults, control: Optional[str] = None) -> pd.DataFrame:
    """Target × cluster matrix of FDR values."""
    if results.table.empty:
        return pd.DataFrame()
    control = control or results.primary_control
    sub = results.table[results.table["control"] == control]
    return sub.pivot(index="target_gene", columns="cluster", values="fdr")


def phenocopy_similarity(results: EnrichmentResults, max_targets: int = 1000) -> pd.DataFrame:
    """Target-by-target correlation of cluster-composition profiles.

    For small datasets all testable targets are compared.

    For very large target collections, all-vs-all correlation can become
    unnecessarily large. In that case only targets with the strongest
    composition shifts are retained.
    """
    comp = results.composition
    if comp.empty or comp.shape[0] < 2:
        return pd.DataFrame()
    if comp.shape[0] > max_targets:
        logger.warning(
            "Phenocopy similarity requested for %d targets; restricting to the top %d composition-shifting targets.",
            comp.shape[0],
            max_targets,
        )
        if not results.effect_magnitude.empty:
            keep = results.effect_magnitude["target_gene"].head(max_targets).tolist()
            comp = comp.loc[comp.index.intersection(keep)]
        else:
            comp = comp.iloc[:max_targets]
    return comp.T.corr(method="pearson")


def format_enrichment_table(results: EnrichmentResults) -> pd.DataFrame:
    """Reader-friendly enrichment table for reports."""
    if results.table.empty:
        return results.table
    sub = results.table[results.table["control"] == results.primary_control].copy()
    order = sub["log2_odds_ratio"].abs().sort_values(ascending=False).index
    sub = sub.reindex(order)
    sub = sub[sub["fdr"] < 1]
    cols = {
        "target_gene": "Target",
        "cluster": "Cluster",
        "n_in_cluster": "Cells in cluster",
        "pct_of_target": "% of target",
        "pct_of_reference": "% of reference",
        "odds_ratio": "Odds ratio",
        "fdr": "FDR",
        "direction": "Direction",
        "guides_concordant": "Guides agreeing",
        "guides_tested": "Guides tested",
        "low_power": "Low power",
        "significant": "Significant",
    }
    out = sub[[col for col in cols if col in sub.columns]].rename(columns=cols)
    for col in out.columns:
        if out[col].dtype.kind == "f":
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.3g}")
    return out
