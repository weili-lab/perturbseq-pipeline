"""Perturbation-strength analysis.

For every target gene that is also measured in the expression matrix, the gene's
*own* expression is compared between cells carrying guides against it and
control cells. A CRISPRi/KO perturbation is expected to push that expression
*down*, so the analysis is directional: significance alone is not enough, the
fold change must be negative.

Two control definitions are reported side by side:

``ntc``
    Cells carrying non-targeting guides. Preferred — these cells experienced the
    same transduction and selection but no on-target effect.
``other``
    Cells assigned to a *different* target gene (what the prototype notebooks
    used). Larger n, but every control cell is itself perturbed.

Tests are run on ``layers['lognorm']`` (log1p of library-size-normalized
counts), never on scaled values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp, mannwhitneyu

from .cluster import LOGNORM_LAYER
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET

logger = logging.getLogger(__name__)

CONTROL_NTC = "ntc"
CONTROL_OTHER = "other"

CONTROL_LABELS = {
    CONTROL_NTC: "non-targeting control cells",
    CONTROL_OTHER: "cells assigned to other target genes",
}

#: Pseudocount (in normalized-expression units) added to both group means before
#: taking the ratio. It keeps a complete knockdown from producing -inf while
#: staying far below the mean of any gene that passes the expression guard, so
#: fold changes for real genes are unaffected.
_PSEUDOCOUNT = 0.01


@dataclass
class PerturbationResults:
    """Everything the report needs about perturbation strength."""

    table: pd.DataFrame
    #: Controls actually usable on this dataset, in priority order.
    controls_used: List[str]
    #: The control driving ranking and hit calling.
    primary_control: str
    #: Targets that could not be tested, with the reason.
    skipped: pd.DataFrame
    n_control_cells: Dict[str, int]

    @property
    def hits(self) -> pd.DataFrame:
        """Targets called effectively perturbed under the primary control."""
        col = f"is_hit_{self.primary_control}"
        if col not in self.table.columns:
            return self.table.iloc[0:0]
        return self.table[self.table[col]]

    def top_effects(self, n: int) -> pd.DataFrame:
        """The ``n`` strongest knockdowns, ranked for the report."""
        return self.table.head(n)


# ---------------------------------------------------------------------------
# Cell group selection
# ---------------------------------------------------------------------------


def control_masks(expr: ad.AnnData, cfg: Config) -> Dict[str, np.ndarray]:
    """Boolean masks for each control definition."""
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    return {
        CONTROL_NTC: klass == CLASS_NTC,
        CONTROL_OTHER: klass == CLASS_TARGETING,  # narrowed per target below
    }


def _control_mask_for_target(
    control: str, base: Dict[str, np.ndarray], targets: np.ndarray, gene: str
) -> np.ndarray:
    if control == CONTROL_NTC:
        return base[CONTROL_NTC]
    # 'other' excludes the target itself; ambiguous/unassigned cells are never
    # used as controls (the prototype notebooks included ambiguous cells).
    return base[CONTROL_OTHER] & (targets != gene)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _gene_vector(expr: ad.AnnData, gene: str) -> np.ndarray:
    """Log-normalized expression of one gene, as a dense 1-D array."""
    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    idx = expr.var_names.get_loc(gene)
    col = layer[:, idx]
    if sparse.issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel().astype(np.float64)


def compare_groups(perturbed: np.ndarray, control: np.ndarray) -> Dict[str, float]:
    """Effect size and significance for one target/control pair.

    ``log2fc`` is computed on de-logged mean expression, which is the quantity
    biologists read as "fold change"; ``pct_knockdown`` restates it as the
    percentage drop relative to control.
    """
    mean_p_log = float(np.mean(perturbed)) if perturbed.size else np.nan
    mean_c_log = float(np.mean(control)) if control.size else np.nan
    mean_p = float(np.mean(np.expm1(perturbed))) if perturbed.size else np.nan
    mean_c = float(np.mean(np.expm1(control))) if control.size else np.nan

    log2fc = float(np.log2((mean_p + _PSEUDOCOUNT) / (mean_c + _PSEUDOCOUNT)))
    pct_kd = float(100.0 * (1.0 - (mean_p + _PSEUDOCOUNT) / (mean_c + _PSEUDOCOUNT)))

    ks_stat, ks_p = ks_2samp(perturbed, control)
    try:
        # One-sided: is expression in perturbed cells stochastically lower?
        mwu_p = float(mannwhitneyu(perturbed, control, alternative="less").pvalue)
    except ValueError:
        mwu_p = np.nan

    return {
        "mean_lognorm_perturbed": mean_p_log,
        "mean_lognorm_control": mean_c_log,
        "log2fc": log2fc,
        "pct_knockdown": pct_kd,
        "pct_cells_expressing_perturbed": float(100 * np.mean(perturbed > 0)) if perturbed.size else np.nan,
        "pct_cells_expressing_control": float(100 * np.mean(control > 0)) if control.size else np.nan,
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_p),
        "mwu_pval_less": mwu_p,
    }


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """BH-FDR that tolerates NaNs (returned as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    vals = p[ok]
    n = vals.size
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    res = np.empty(n)
    res[order] = q
    out[ok] = res
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_all_targets(expr: ad.AnnData, cfg: Config) -> PerturbationResults:
    """Run the perturbation-strength test for every testable target gene."""
    pcfg = cfg.perturbation
    obs = expr.obs
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()

    base = control_masks(expr, cfg)
    n_control_cells = {
        CONTROL_NTC: int(base[CONTROL_NTC].sum()),
        CONTROL_OTHER: int(base[CONTROL_OTHER].sum()),
    }

    controls_used = []
    for control in pcfg.controls:
        if control == CONTROL_NTC and n_control_cells[CONTROL_NTC] < pcfg.min_control_cells:
            logger.warning(
                "Only %d non-targeting control cells (need %d); the 'ntc' control "
                "arm will be skipped.",
                n_control_cells[CONTROL_NTC],
                pcfg.min_control_cells,
            )
            continue
        controls_used.append(control)
    if not controls_used:
        raise ValueError(
            "No usable control group. There are "
            f"{n_control_cells[CONTROL_NTC]} non-targeting and "
            f"{n_control_cells[CONTROL_OTHER]} other-target cells, but "
            f"perturbation.min_control_cells={pcfg.min_control_cells}."
        )

    primary = pcfg.primary_control if pcfg.primary_control in controls_used else controls_used[0]
    if primary != pcfg.primary_control:
        logger.warning(
            "Requested primary control %r is unavailable; using %r instead.",
            pcfg.primary_control,
            primary,
        )

    all_targets = sorted(set(targets_col[klass == CLASS_TARGETING]))
    measured = set(expr.var_names)

    rows: List[dict] = []
    skipped: List[dict] = []

    for gene in all_targets:
        pert_mask = (targets_col == gene) & (klass == CLASS_TARGETING)
        n_pert = int(pert_mask.sum())
        if gene not in measured:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_perturbed": n_pert,
                    "reason": "target gene not present in the expression matrix",
                }
            )
            continue
        if n_pert < pcfg.min_cells_per_target:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_perturbed": n_pert,
                    "reason": f"fewer than {pcfg.min_cells_per_target} perturbed cells",
                }
            )
            continue

        values = _gene_vector(expr, gene)

        # A gene that is not expressed in control cells cannot be shown to go
        # down, and its fold change would be an artefact of the epsilon guard
        # (the olfactory-receptor controls in TF screens are the classic case).
        primary_mask = _control_mask_for_target(primary, base, targets_col, gene)
        pct_expressing = (
            float(100 * np.mean(values[primary_mask] > 0)) if primary_mask.any() else 0.0
        )
        if pct_expressing < pcfg.min_pct_expressing_control:
            skipped.append(
                {
                    "target_gene": gene,
                    "n_perturbed": n_pert,
                    "reason": (
                        f"not detectably expressed in control cells "
                        f"({pct_expressing:.2f}% of control cells, threshold "
                        f"{pcfg.min_pct_expressing_control}%)"
                    ),
                }
            )
            continue

        row: Dict[str, object] = {"target_gene": gene, "n_perturbed": n_pert}
        for control in controls_used:
            cmask = _control_mask_for_target(control, base, targets_col, gene)
            n_ctrl = int(cmask.sum())
            row[f"n_control_{control}"] = n_ctrl
            if n_ctrl < pcfg.min_control_cells:
                for key in (
                    "log2fc",
                    "pct_knockdown",
                    "ks_stat",
                    "ks_pval",
                    "mwu_pval_less",
                    "mean_lognorm_perturbed",
                    "mean_lognorm_control",
                ):
                    row[f"{key}_{control}"] = np.nan
                continue
            stats = compare_groups(values[pert_mask], values[cmask])
            for key, val in stats.items():
                row[f"{key}_{control}"] = val
        rows.append(row)

    if not rows:
        logger.warning("No target gene was testable; returning an empty result table.")
        return PerturbationResults(
            table=pd.DataFrame(),
            controls_used=controls_used,
            primary_control=primary,
            skipped=pd.DataFrame(skipped),
            n_control_cells=n_control_cells,
        )

    table = pd.DataFrame(rows)

    for control in controls_used:
        table[f"ks_fdr_{control}"] = benjamini_hochberg(table[f"ks_pval_{control}"])
        table[f"mwu_fdr_{control}"] = benjamini_hochberg(table[f"mwu_pval_less_{control}"])
        table[f"is_hit_{control}"] = (
            (table[f"ks_fdr_{control}"] < pcfg.fdr_alpha)
            & (table[f"log2fc_{control}"] < pcfg.max_log2fc_for_hit)
        ).fillna(False)

    # Rank by knockdown strength under the primary control: significant hits
    # first, then most-negative fold change.
    table = table.sort_values(
        [f"is_hit_{primary}", f"log2fc_{primary}"], ascending=[False, True]
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    n_hits = int(table[f"is_hit_{primary}"].sum())
    logger.info(
        "Perturbation strength: %d/%d targets tested, %d effective at FDR < %.2f "
        "(control: %s)",
        len(table),
        len(all_targets),
        n_hits,
        pcfg.fdr_alpha,
        CONTROL_LABELS[primary],
    )
    if skipped:
        logger.info("%d target(s) skipped; see the skipped table.", len(skipped))

    return PerturbationResults(
        table=table,
        controls_used=controls_used,
        primary_control=primary,
        skipped=pd.DataFrame(skipped),
        n_control_cells=n_control_cells,
    )


def format_results_table(results: PerturbationResults, cfg: Config) -> pd.DataFrame:
    """Reader-friendly view of the results for the HTML report."""
    if results.table.empty:
        return results.table
    primary = results.primary_control
    cols = {
        "rank": "Rank",
        "target_gene": "Target",
        "n_perturbed": "Perturbed cells",
        f"n_control_{primary}": "Control cells",
        f"log2fc_{primary}": "log2FC",
        f"pct_knockdown_{primary}": "% knockdown",
        f"ks_stat_{primary}": "KS stat",
        f"ks_fdr_{primary}": "KS FDR",
        f"is_hit_{primary}": "Effective",
    }
    other = CONTROL_OTHER if primary == CONTROL_NTC else CONTROL_NTC
    if other in results.controls_used:
        cols[f"log2fc_{other}"] = f"log2FC ({other})"
        cols[f"ks_fdr_{other}"] = f"KS FDR ({other})"

    present = {k: v for k, v in cols.items() if k in results.table.columns}
    out = results.table[list(present)].rename(columns=present).copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.3g}")
    return out
