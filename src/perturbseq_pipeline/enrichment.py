"""Enrichment of each perturbation across the clusters.

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
are small: rare clusters hold only a handful of reference cells, which is
exactly where the strongest effects turn up.

Odds ratios carry a Haldane-Anscombe correction so they stay finite when a
count is zero, and pairs whose reference contributes very few cells are flagged
as low-power rather than quietly trusted.

The reference is either non-targeting cells (``ntc``) or cells assigned to a
different target (``other``). ``other`` is the default primary here — unlike the
perturbation-strength test — because the non-targeting group is small, and in a
rare cluster it can contribute single-digit cell counts.

With ``stratify_by`` set (e.g. ``lane_id``) a Cochran-Mantel-Haenszel test
replaces the pooled Fisher test, so a cluster that simply differs in size
between lanes cannot masquerade as a perturbation effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from .config import Config
from .guides import (
    CLASS_NTC,
    CLASS_TARGETING,
    OBS_CLASS,
    OBS_GUIDE,
    OBS_TARGET,
)
from .perturbation import CONTROL_LABELS, CONTROL_NTC, CONTROL_OTHER, benjamini_hochberg

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResults:
    """Everything the report needs about perturbation/cluster association."""

    #: Long-format results: one row per (target, cluster) pair.
    table: pd.DataFrame
    #: Target x cluster percentages of each target's own cells.
    composition: pd.DataFrame
    #: ``{control: Series of reference percentages per cluster}``.
    reference_composition: Dict[str, pd.Series]
    #: Per-target summary: how far the composition sits from the reference.
    effect_magnitude: pd.DataFrame
    #: Omnibus test of the full contingency table.
    omnibus: Dict[str, float] = field(default_factory=dict)
    controls_used: List[str] = field(default_factory=list)
    primary_control: str = CONTROL_OTHER
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    cluster_key: str = "leiden"
    #: True when a stratified (CMH) test was used instead of pooled Fisher.
    stratified: bool = False
    stratify_by: Optional[str] = None

    @property
    def hits(self) -> pd.DataFrame:
        """Significant (target, cluster) pairs under the primary control."""
        if self.table.empty:
            return self.table
        return self.table[self.table["significant"]]

    def top_hits(self, n: int) -> pd.DataFrame:
        """Strongest enrichments, most extreme odds ratio first."""
        h = self.hits
        if h.empty:
            return h
        return h.reindex(h["log2_odds_ratio"].abs().sort_values(ascending=False).index).head(n)

    def targets_with_hits(self) -> List[str]:
        return sorted(self.hits["target_gene"].unique()) if not self.hits.empty else []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster_order(values: Sequence[str]) -> List[str]:
    """Sort cluster labels numerically when possible, else lexically."""
    uniq = list(dict.fromkeys(str(v) for v in values))
    try:
        return sorted(uniq, key=lambda x: (float(x), x))
    except ValueError:
        return sorted(uniq)


def _reference_mask(
    control: str, klass: np.ndarray, targets: np.ndarray, gene: str
) -> np.ndarray:
    if control == CONTROL_NTC:
        return klass == CLASS_NTC
    return (klass == CLASS_TARGETING) & (targets != gene)


def _odds_ratio(a: float, b: float, c: float, d: float, pseudo: float) -> float:
    """Haldane-Anscombe corrected odds ratio (finite even at zero counts)."""
    return ((a + pseudo) * (d + pseudo)) / ((b + pseudo) * (c + pseudo))


# ---------------------------------------------------------------------------
# Omnibus
# ---------------------------------------------------------------------------


def omnibus_test(
    contingency: pd.DataFrame, n_permutations: int = 1000, seed: int = 0
) -> Dict[str, float]:
    """Global test that perturbation identity and cluster are associated.

    Reports the chi-square statistic plus a permutation p-value, because with
    many small expected counts the chi-square approximation is unreliable — it
    is a screen here, not the inferential result.
    """
    if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {}

    # chi2_contingency rejects any table with a zero row or column margin, and
    # both occur naturally here: a cluster can hold no targeting cells at all
    # (they may all be ambiguous or non-targeting), and a target's cells can all
    # sit in clusters that were too small to test. Prune those margins rather
    # than letting the whole stage die on them.
    table = contingency.loc[
        contingency.sum(axis=1) > 0, contingency.sum(axis=0) > 0
    ]
    dropped_rows = contingency.shape[0] - table.shape[0]
    dropped_cols = contingency.shape[1] - table.shape[1]
    if dropped_rows or dropped_cols:
        logger.info(
            "Omnibus test: dropped %d empty target row(s) and %d empty cluster "
            "column(s) with no counts",
            dropped_rows,
            dropped_cols,
        )
    if table.shape[0] < 2 or table.shape[1] < 2:
        logger.warning(
            "Omnibus test needs at least a 2x2 table after pruning; skipping it."
        )
        return {}

    observed = table.to_numpy(dtype=float)
    chi2, p_chi2, dof, expected = chi2_contingency(observed)
    small = float((expected < 5).mean())

    out = {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_chi2": float(p_chi2),
        "pct_expected_below_5": 100 * small,
        "n_permutations": 0,
        "p_permutation": float("nan"),
    }

    if n_permutations and n_permutations > 0:
        rng = np.random.default_rng(seed)
        # Permute cell labels while holding both margins fixed, by resampling
        # the cluster assignment vector across all cells.
        row_counts = observed.sum(axis=1).astype(int)
        col_probs = observed.sum(axis=0) / observed.sum()
        n_ge = 0
        for _ in range(n_permutations):
            sim = np.vstack([rng.multinomial(n, col_probs) for n in row_counts])
            keep = sim.sum(axis=0) > 0
            stat = chi2_contingency(sim[:, keep])[0] if keep.sum() > 1 else 0.0
            if stat >= chi2:
                n_ge += 1
        out["n_permutations"] = int(n_permutations)
        out["p_permutation"] = (n_ge + 1) / (n_permutations + 1)

    return out


# ---------------------------------------------------------------------------
# Stratified (Cochran-Mantel-Haenszel)
# ---------------------------------------------------------------------------


def _cmh_test(tables: List[np.ndarray]) -> Tuple[float, float]:
    """Cochran-Mantel-Haenszel across strata; returns (pooled OR, p-value)."""
    from statsmodels.stats.contingency_tables import StratifiedTable

    usable = [
        t
        for t in tables
        if t.sum() > 0 and t.sum(axis=1).min() > 0 and t.sum(axis=0).min() > 0
    ]
    if not usable:
        return float("nan"), float("nan")
    st = StratifiedTable([t.T for t in usable])
    # A stratum with no discordant pairs makes the pooled odds ratio divide by
    # zero (an infinite estimate). The caller falls back to the corrected
    # Fisher odds in that case, so silence the warning rather than let it clutter
    # the run log once per affected pair.
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = float(st.oddsratio_pooled)
        pvalue = float(st.test_null_odds().pvalue)
    return pooled, pvalue


# ---------------------------------------------------------------------------
# Guide concordance
# ---------------------------------------------------------------------------


def _guide_concordance(
    obs: pd.DataFrame,
    gene: str,
    cluster: str,
    cluster_key: str,
    ref_fraction: float,
    min_cells: int,
    direction: str = "enriched",
) -> Tuple[int, int]:
    """How many of a target's guides independently show the same effect.

    A genuine phenotype shows up across several guides; a single-guide artefact
    (an off-target effect, or one bad guide) will not.

    Agreement is judged against the *observed direction*: for an enrichment a
    guide agrees when its cells sit in the cluster more often than the
    reference, and for a depletion when they sit there less often. Testing only
    the enrichment direction would report every real depletion as ``0`` guides
    agreeing, which reads as the opposite of the truth.

    Returns ``(n_guides_concordant, n_guides_tested)``.
    """
    if OBS_GUIDE not in obs.columns:
        return 0, 0
    sub = obs[(obs[OBS_TARGET].astype(str) == gene) & (obs[OBS_CLASS].astype(str) == CLASS_TARGETING)]
    if sub.empty:
        return 0, 0
    tested = concordant = 0
    for _, cells in sub.groupby(sub[OBS_GUIDE].astype(str), observed=True):
        if len(cells) < min_cells:
            continue
        tested += 1
        frac = float((cells[cluster_key].astype(str) == cluster).mean())
        agrees = frac < ref_fraction if direction == "depleted" else frac > ref_fraction
        if agrees:
            concordant += 1
    return concordant, tested


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_cluster_enrichment(expr: ad.AnnData, cfg: Config) -> EnrichmentResults:
    """Test every target gene for enrichment/depletion in every cluster."""
    ecfg = cfg.enrichment
    obs = expr.obs
    cluster_key = ecfg.cluster_key

    if cluster_key not in obs.columns:
        raise ValueError(
            f"enrichment.cluster_key={cluster_key!r} is not an obs column; "
            "clustering must run before enrichment."
        )

    clusters_all = obs[cluster_key].astype(str).to_numpy()
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()

    cluster_sizes = pd.Series(clusters_all).value_counts()
    clusters = [
        c
        for c in _cluster_order(clusters_all)
        if cluster_sizes.get(c, 0) >= ecfg.min_cells_per_cluster
    ]
    dropped_clusters = [
        c for c in _cluster_order(clusters_all) if c not in clusters
    ]
    if dropped_clusters:
        logger.info(
            "Skipping %d cluster(s) with < %d cells: %s",
            len(dropped_clusters),
            ecfg.min_cells_per_cluster,
            dropped_clusters,
        )
    if not clusters:
        raise ValueError(
            f"No cluster has at least enrichment.min_cells_per_cluster="
            f"{ecfg.min_cells_per_cluster} cells."
        )

    # Which control arms are usable at all.
    controls_used = []
    for control in ecfg.controls:
        n_ref = int((klass == CLASS_NTC).sum()) if control == CONTROL_NTC else int(
            (klass == CLASS_TARGETING).sum()
        )
        if n_ref < ecfg.min_reference_cells:
            logger.warning(
                "Control %r has only %d cells; dropping it from the enrichment test.",
                control,
                n_ref,
            )
            continue
        controls_used.append(control)
    if not controls_used:
        raise ValueError("No usable control group for the enrichment test.")

    primary = ecfg.primary_control if ecfg.primary_control in controls_used else controls_used[0]
    if primary != ecfg.primary_control:
        logger.warning(
            "Requested enrichment.primary_control %r unavailable; using %r.",
            ecfg.primary_control,
            primary,
        )

    strat_values = None
    stratified = False
    if ecfg.stratify_by:
        if ecfg.stratify_by not in obs.columns:
            raise ValueError(
                f"enrichment.stratify_by={ecfg.stratify_by!r} is not an obs column."
            )
        strat_values = obs[ecfg.stratify_by].astype(str).to_numpy()
        if len(set(strat_values)) < 2:
            logger.info(
                "enrichment.stratify_by=%r has a single level; using a pooled test.",
                ecfg.stratify_by,
            )
            strat_values = None
        else:
            stratified = True
            logger.info(
                "Using Cochran-Mantel-Haenszel stratified by %r (%d strata)",
                ecfg.stratify_by,
                len(set(strat_values)),
            )

    # Targets to test.
    target_counts = pd.Series(targets_col[klass == CLASS_TARGETING]).value_counts()
    testable = sorted(target_counts[target_counts >= ecfg.min_cells_per_target].index)
    skipped = pd.DataFrame(
        [
            {
                "target_gene": t,
                "n_cells": int(n),
                "reason": f"fewer than {ecfg.min_cells_per_target} assigned cells",
            }
            for t, n in target_counts.items()
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

    # Composition matrices (percent of each target's cells per cluster).
    in_cluster = {c: clusters_all == c for c in clusters}
    comp_rows = {}
    for gene in testable:
        m = (targets_col == gene) & (klass == CLASS_TARGETING)
        n = int(m.sum())
        comp_rows[gene] = {c: 100 * float((m & in_cluster[c]).sum()) / n for c in clusters}
    composition = pd.DataFrame.from_dict(comp_rows, orient="index")[clusters]

    reference_composition = {}
    for control in controls_used:
        if control == CONTROL_NTC:
            m = klass == CLASS_NTC
        else:
            m = klass == CLASS_TARGETING
        n = max(int(m.sum()), 1)
        reference_composition[control] = pd.Series(
            {c: 100 * float((m & in_cluster[c]).sum()) / n for c in clusters}
        )

    # Omnibus on the target x cluster table.
    contingency = pd.DataFrame(
        {
            c: [int(((targets_col == g) & (klass == CLASS_TARGETING) & in_cluster[c]).sum()) for g in testable]
            for c in clusters
        },
        index=testable,
    )
    omnibus = omnibus_test(contingency, ecfg.permutations, cfg.run.seed)
    if omnibus:
        logger.info(
            "Omnibus association: chi2=%.0f (dof %d), permutation p=%.4g "
            "(%.0f%% of expected counts < 5)",
            omnibus["chi2"],
            omnibus["dof"],
            omnibus["p_permutation"],
            omnibus["pct_expected_below_5"],
        )

    # Per-pair tests.
    rows: List[dict] = []
    for gene in testable:
        tmask = (targets_col == gene) & (klass == CLASS_TARGETING)
        n_target = int(tmask.sum())
        for control in controls_used:
            rmask = _reference_mask(control, klass, targets_col, gene)
            n_ref = int(rmask.sum())
            for cluster in clusters:
                cm = in_cluster[cluster]
                a = int((tmask & cm).sum())
                b = n_target - a
                c_ = int((rmask & cm).sum())
                d = n_ref - c_

                pct_t = 100 * a / max(n_target, 1)
                pct_r = 100 * c_ / max(n_ref, 1)
                odds = _odds_ratio(a, b, c_, d, ecfg.odds_pseudocount)

                if stratified:
                    tables = []
                    for s in sorted(set(strat_values)):
                        sm = strat_values == s
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
                    _, pval = fisher_exact([[a, b], [c_, d]])

                rows.append(
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
                        "log2_odds_ratio": float(np.log2(odds)) if odds > 0 else np.nan,
                        "direction": "enriched" if pct_t > pct_r else "depleted",
                        "pval": float(pval),
                        "low_power": c_ < ecfg.min_reference_cells,
                    }
                )

    table = pd.DataFrame(rows)

    # FDR within each control arm, since the arms are separate families.
    table["fdr"] = np.nan
    for control in controls_used:
        m = table["control"] == control
        table.loc[m, "fdr"] = benjamini_hochberg(table.loc[m, "pval"].to_numpy())

    table["significant"] = (table["fdr"] < ecfg.fdr_alpha) & (
        table["control"] == primary
    )

    # Guide-level concordance for the significant pairs only (it is the
    # expensive part and only meaningful where there is something to confirm).
    table["guides_concordant"] = np.nan
    table["guides_tested"] = np.nan
    if ecfg.guide_concordance and OBS_GUIDE in obs.columns:
        obs_view = obs[[OBS_GUIDE, OBS_TARGET, OBS_CLASS, cluster_key]].copy()
        obs_view[cluster_key] = obs_view[cluster_key].astype(str)
        for idx in table.index[table["significant"]]:
            gene = table.at[idx, "target_gene"]
            cluster = table.at[idx, "cluster"]
            ref_frac = table.at[idx, "pct_of_reference"] / 100.0
            conc, tested = _guide_concordance(
                obs_view,
                gene,
                cluster,
                cluster_key,
                ref_frac,
                ecfg.min_cells_per_guide,
                direction=str(table.at[idx, "direction"]),
            )
            table.at[idx, "guides_concordant"] = conc
            table.at[idx, "guides_tested"] = tested

    # Per-target effect magnitude: total variation distance from the reference.
    ref = reference_composition[primary]
    magnitude = []
    n_sig = table[table["significant"]]["target_gene"].value_counts()
    for gene in testable:
        diff = (composition.loc[gene] - ref).abs().sum() / 2.0
        magnitude.append(
            {
                "target_gene": gene,
                "n_cells": int(target_counts[gene]),
                "composition_shift_pct": float(diff),
                "n_significant_clusters": int(n_sig.get(gene, 0)),
            }
        )
    effect_magnitude = (
        pd.DataFrame(magnitude)
        .sort_values("composition_shift_pct", ascending=False)
        .reset_index(drop=True)
    )

    table = table.sort_values(
        ["significant", "log2_odds_ratio"], ascending=[False, False]
    ).reset_index(drop=True)

    n_hits = int(table["significant"].sum())
    logger.info(
        "Cluster enrichment: %d target(s) x %d cluster(s) = %d tests per control; "
        "%d significant at FDR < %.2f (control: %s)",
        len(testable),
        len(clusters),
        len(testable) * len(clusters),
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
        stratify_by=ecfg.stratify_by if stratified else None,
    )


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def enrichment_matrix(results: EnrichmentResults, control: Optional[str] = None) -> pd.DataFrame:
    """Target x cluster matrix of log2 odds ratios, for the heatmap."""
    if results.table.empty:
        return pd.DataFrame()
    control = control or results.primary_control
    sub = results.table[results.table["control"] == control]
    return sub.pivot(index="target_gene", columns="cluster", values="log2_odds_ratio")


def significance_matrix(results: EnrichmentResults, control: Optional[str] = None) -> pd.DataFrame:
    """Matching target x cluster matrix of FDR values."""
    if results.table.empty:
        return pd.DataFrame()
    control = control or results.primary_control
    sub = results.table[results.table["control"] == control]
    return sub.pivot(index="target_gene", columns="cluster", values="fdr")


def phenocopy_similarity(results: EnrichmentResults) -> pd.DataFrame:
    """Target-by-target correlation of cluster-composition profiles.

    Perturbations of genes in the same complex should produce similar
    compositions and therefore correlate — a useful internal check that does not
    depend on any prior knowledge of the complexes.
    """
    comp = results.composition
    if comp.empty or comp.shape[0] < 2:
        return pd.DataFrame()
    return comp.T.corr(method="pearson")


def format_enrichment_table(results: EnrichmentResults) -> pd.DataFrame:
    """Reader-friendly view of the significant pairs for the report."""
    if results.table.empty:
        return results.table
    sub = results.table[results.table["control"] == results.primary_control].copy()
    sub = sub.reindex(sub["log2_odds_ratio"].abs().sort_values(ascending=False).index)
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
    out = sub[[c for c in cols if c in sub.columns]].rename(columns=cols)
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.3g}")
    return out
