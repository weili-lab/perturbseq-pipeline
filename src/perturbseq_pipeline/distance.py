"""Perturbation distance vs control and pairwise Perturbation Distance Space.

This module provides a statistically rigorous, scalable, and memory-aware
perturbation distance layer:

1. **Perturbation Distance vs Control**:
   For every perturbation target, compares the high-dimensional phenotypic
   distribution (in PCA space, e.g. ``adata.obsm['X_pca']``) against the
   unperturbed control population using Energy Distance (primary) and optional
   MMD (secondary).

2. **DistanceTest (Permutation Significance)**:
   Assesses statistical confidence via deterministic permutation of perturbation
   and control labels, yielding exact finite-permutation empirical p-values:

       p = (1 + sum(perm_stat >= obs_stat)) / (1 + n_permutations)

   followed by Benjamini–Hochberg False Discovery Rate (BH-FDR) correction across
   tested perturbations.

3. **Perturbation Distance Space**:
   Constructs a pairwise target × target phenotypic distance matrix, projects
   perturbations into low-dimensional phenotypic coordinates via classical
   Multidimensional Scaling (PCoA, handling non-Euclidean eigenspectra
   transparently), discovers nearest phenotypic neighbors, and clusters
   perturbations into distinct **Phenotype Modules**.

4. **Scalable Bounded Sampling**:
   Never performs unconstrained all-cell pairwise operations. Employs
   deterministic bounded sampling (preserving batch representations when available)
   and constant-memory matrix operations compatible with both STANDARD and LARGE
   pipeline execution modes.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage

from .compute import derive_seed, log_compute_decision, resolve_stage_backend, run_parallel
from .config import Config
from .data_access import SharedArrayBuffer, get_embedding, get_target_indices_map
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET
from .perturbation import CONTROL_NTC, CONTROL_OTHER, benjamini_hochberg, control_masks

logger = logging.getLogger(__name__)


# Result Data Structures


@dataclass
class DistanceResults:
    """Target-level perturbation distance statistics vs control."""

    #: One row per target gene: target_gene, n_cells, n_control, energy_distance,
    #: [mmd_distance], pvalue, fdr, significant.
    table: pd.DataFrame
    #: Targets that could not be tested (e.g. below min_cells).
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    primary_metric: str = "edistance"
    secondary_metric: Optional[str] = None
    control_used: str = CONTROL_NTC
    n_control_cells: int = 0
    representation: str = "X_pca"
    note: str = ""

    @property
    def significant_hits(self) -> pd.DataFrame:
        """Targets called statistically significant at configured FDR."""
        if self.table.empty or "significant" not in self.table.columns:
            return self.table.iloc[0:0]
        return self.table[self.table["significant"]]


@dataclass
class DistanceSpaceResults:
    """Pairwise perturbation distance space, coordinates, neighbors, and phenotype modules."""

    #: Pairwise target x target distance matrix (symmetric, 0 diagonal).
    distance_matrix: pd.DataFrame
    #: PCoA low-dimensional coordinates: target_gene, PCoA1, PCoA2, ...
    coordinates: pd.DataFrame
    #: Nearest phenotypic neighbors: target, neighbor, distance, rank.
    neighbors: pd.DataFrame
    #: Phenotype module assignments: target_gene, phenotype_module.
    phenotype_modules: pd.DataFrame
    #: Targets excluded from DistanceSpace (e.g. below min_cells).
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_components: int = 0
    metric: str = "edistance"
    linkage_method: str = "average"
    note: str = ""


# Mathematical Distance Functions


def energy_distance_from_cdist(D: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
    """Compute Energy Distance statistic directly from a precomputed pairwise distance matrix.

    Given pooled sample Z = [X; Y] and its pairwise Euclidean distance matrix D,
    the empirical energy distance is:

        E^2(X, Y) = 2/(n*m) * sum_{i in X, j in Y} D[i, j]
                    - 1/n^2 * sum_{i, i' in X} D[i, i']
                    - 1/m^2 * sum_{j, j' in Y} D[j, j']

    Using the algebraic identity:
        2 * sum_{i in X, j in Y} D[i, j] = S_total - S_X - S_Y
    where S_X = sum_{i, i' in X} D[i, i'], S_Y = sum_{j, j' in Y} D[j, j'],
    and S_total = sum_{a, b} D[a, b].

    This computes exact Energy Distance in O(n^2 + m^2) indexing without
    recalculating distances.
    """
    n = len(idx_x)
    m = len(idx_y)
    if n == 0 or m == 0:
        return float("nan")
    sub_x = D[np.ix_(idx_x, idx_x)]
    sub_y = D[np.ix_(idx_y, idx_y)]
    s_x = float(np.sum(sub_x))
    s_y = float(np.sum(sub_y))
    s_total = float(np.sum(D))
    between_term = (s_total - s_x - s_y) / (n * m)
    within_x = s_x / (n * n)
    within_y = s_y / (m * m)
    edist_sq = between_term - within_x - within_y
    # Energy distance is mathematically non-negative; clip numerical artifacts near zero
    return float(max(edist_sq, 0.0))


def compute_energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute empirical Energy Distance between two sample matrices X and Y.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples_X, n_features).
    Y : np.ndarray
        Array of shape (n_samples_Y, n_features).

    Returns
    -------
    float
        Energy distance statistic.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    # Fast cdist computation
    d_xy = cdist(X, Y, metric="euclidean")
    d_xx = cdist(X, X, metric="euclidean")
    d_yy = cdist(Y, Y, metric="euclidean")
    mean_xy = float(np.mean(d_xy))
    mean_xx = float(np.mean(d_xx))
    mean_yy = float(np.mean(d_yy))
    edist = 2.0 * mean_xy - mean_xx - mean_yy
    return float(max(edist, 0.0))


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Compute empirical Maximum Mean Discrepancy (MMD) with Gaussian RBF kernel.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples_X, n_features).
    Y : np.ndarray
        Array of shape (n_samples_Y, n_features).
    gamma : float, optional
        RBF kernel bandwidth parameter gamma = 1 / (2 * sigma^2).
        If None, median pairwise distance heuristic is used.

    Returns
    -------
    float
        MMD squared statistic.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    d2_xx = cdist(X, X, metric="sqeuclidean")
    d2_yy = cdist(Y, Y, metric="sqeuclidean")
    d2_xy = cdist(X, Y, metric="sqeuclidean")
    if gamma is None:
        # Median heuristic across combined samples
        combined_d2 = np.concatenate([d2_xx.ravel(), d2_yy.ravel(), d2_xy.ravel()])
        pos_d2 = combined_d2[combined_d2 > 0]
        med = float(np.median(pos_d2)) if pos_d2.size > 0 else 1.0
        gamma = 1.0 / max(med, 1e-6)
    k_xx = np.exp(-gamma * d2_xx)
    k_yy = np.exp(-gamma * d2_yy)
    k_xy = np.exp(-gamma * d2_xy)
    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
    return float(max(mmd2, 0.0))


# Sampling & Representation Helpers


def _resolve_representation(expr: ad.AnnData, requested_rep: str) -> np.ndarray:
    """Extract the cell embedding matrix (e.g. X_pca)."""
    return get_embedding(expr, rep_name=requested_rep, dtype=np.float32)


def _sample_cell_indices(
    indices: np.ndarray, max_cells: int, rng: np.random.Generator, strata: Optional[np.ndarray] = None
) -> np.ndarray:
    """Deterministically sample at most max_cells indices, preserving batch strata if available."""
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    if n <= max_cells:
        return indices
    if strata is not None:
        # Stratified sampling across batches/lanes
        strata_values = strata[indices]
        unique_strata, strata_counts = np.unique(strata_values, return_counts=True)
        selected_list = []
        for s_val, s_count in zip(unique_strata, strata_counts):
            s_idx = indices[strata_values == s_val]
            # Proportional allocation
            s_quota = max(1, int(np.round(max_cells * (s_count / n))))
            s_quota = min(s_quota, len(s_idx))
            if s_quota > 0:
                s_sampled = s_idx if s_quota == len(s_idx) else rng.choice(s_idx, size=s_quota, replace=False)
                selected_list.append(s_sampled)
        if selected_list:
            combined = np.concatenate(selected_list)
            if len(combined) > max_cells:
                combined = rng.choice(combined, size=max_cells, replace=False)
            elif len(combined) < max_cells:
                # Top-up from remaining unpicked cells
                unpicked = np.setdiff1d(indices, combined)
                topup_n = min(max_cells - len(combined), len(unpicked))
                if topup_n > 0:
                    topup = rng.choice(unpicked, size=topup_n, replace=False)
                    combined = np.concatenate([combined, topup])
            combined.sort()
            return combined.astype(np.int64)
    # Uniform random sampling fallback
    selected = rng.choice(indices, size=max_cells, replace=False)
    selected.sort()
    return selected.astype(np.int64)


# Permutation DistanceTest (Optimized Exact O(n^2) Null Sampling)


def distance_test_permutation(
    X: np.ndarray, Y: np.ndarray, n_permutations: int = 1000, seed: int = 123
) -> Tuple[float, float]:
    """Perform an exact, fast permutation test for Energy Distance between X and Y.

    Uses the algebraic identity:
        S_Y = S_total + S_X - 2 * R_X
    where R_X = sum_{i in X} row_sums[i] and S_total = sum(D).

    This computes exact Energy Distance across permutations without allocating
    or summing large O((n+m)^2) intermediate matrices repeatedly. Preserves
    identical statistical meaning and finite-permutation empirical p-values:
        p = (1 + sum(perm_stat >= obs_stat)) / (1 + n_permutations)

    Parameters
    ----------
    X : np.ndarray
        Perturbed cell embedding (n_target, n_features).
    Y : np.ndarray
        Control cell embedding (n_control, n_features).
    n_permutations : int
        Number of label permutations.
    seed : int
        Deterministic random seed.

    Returns
    -------
    Tuple[float, float]
        (observed_energy_distance, empirical_pvalue).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    # Combine samples and compute full N x N pairwise distance matrix once
    Z = np.vstack([X, Y])
    N = n + m
    D = cdist(Z, Z, metric="euclidean")
    row_sums = np.sum(D, axis=1)
    s_total = float(np.sum(row_sums))
    # Observed distance
    idx_x_obs = np.arange(n, dtype=np.int64)
    r_x_obs = float(np.sum(row_sums[idx_x_obs]))
    s_x_obs = float(np.sum(D[np.ix_(idx_x_obs, idx_x_obs)]))
    s_y_obs = s_total + s_x_obs - 2.0 * r_x_obs
    between_obs = 2.0 * (r_x_obs - s_x_obs) / (n * m)
    within_x_obs = s_x_obs / (n * n)
    within_y_obs = s_y_obs / (m * m)
    obs_stat = float(max(between_obs - within_x_obs - within_y_obs, 0.0))
    if n_permutations <= 0:
        return obs_stat, float("nan")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(N, dtype=np.int64)
    count_greater_or_equal = 0
    tol = obs_stat - 1e-12
    inv_nm = 2.0 / (n * m)
    inv_nn = 1.0 / (n * n)
    inv_mm = 1.0 / (m * m)
    use_x = n <= m
    for _ in range(n_permutations):
        perm = rng.permutation(all_indices)
        if use_x:
            px = perm[:n]
            r_k = float(np.sum(row_sums[px]))
            s_k = float(np.sum(D[np.ix_(px, px)]))
            s_x = s_k
            s_y = s_total + s_x - 2.0 * r_k
            stat = (r_k - s_x) * inv_nm - s_x * inv_nn - s_y * inv_mm
        else:
            py = perm[n:]
            r_k = float(np.sum(row_sums[py]))
            s_k = float(np.sum(D[np.ix_(py, py)]))
            s_y = s_k
            s_x = s_total + s_y - 2.0 * r_k
            stat = (r_k - s_y) * inv_nm - s_x * inv_nn - s_y * inv_mm
        if stat >= tol:
            count_greater_or_equal += 1
    # Exact finite-permutation empirical p-value
    pval = (1.0 + count_greater_or_equal) / (1.0 + n_permutations)
    return obs_stat, float(pval)


# Perturbation Distance vs Control API


def _eval_target_dist_worker(
    task_payload: Tuple[
        int,
        str,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
        Optional[np.ndarray],
        int,
        int,
        int,
        int,
        str,
        Optional[str],
    ],
) -> Tuple[Optional[dict], Optional[dict]]:
    """Worker function executing DistanceTest for one perturbation target.

    Receives cleanly unpacked numeric arrays and scalar parameters without
    referencing or deserializing full AnnData objects. ``ctrl_indices`` are the
    pre-sampled control cells; with ``exclude_self`` (the ``other`` control, i.e.
    all targeting cells) the focal target's own cells are removed from them so a
    perturbation is never compared against itself.
    """
    (
        i,
        target,
        pert_indices,
        embedding_arr,
        ctrl_indices,
        exclude_self,
        strata,
        min_cells,
        max_cells_per_target,
        n_permutations,
        random_seed,
        primary_metric,
        secondary_metric,
    ) = task_payload
    n_pert = int(pert_indices.size)
    if n_pert < min_cells:
        return None, {"target_gene": target, "n_cells": n_pert, "reason": f"fewer than {min_cells} cells ({n_pert})"}
    if exclude_self:
        ctrl_indices = ctrl_indices[~np.isin(ctrl_indices, pert_indices)]
    n_ctrl = int(ctrl_indices.size)
    if n_ctrl < min_cells:
        return None, {
            "target_gene": target,
            "n_cells": n_pert,
            "reason": f"fewer than {min_cells} control cells after excluding the target's own cells ({n_ctrl})",
        }
    Y_ctrl = embedding_arr[ctrl_indices]
    target_seed = derive_seed(random_seed, f"{i}_{target}")
    rng_target = np.random.default_rng(target_seed)
    pert_indices_sampled = _sample_cell_indices(pert_indices, max_cells_per_target, rng_target, strata=strata)
    X_target = embedding_arr[pert_indices_sampled]
    # Compute primary metric and permutation DistanceTest
    edist, pval = distance_test_permutation(X_target, Y_ctrl, n_permutations=n_permutations, seed=target_seed)
    row = {"target_gene": target, "n_cells": n_pert, "n_control": len(Y_ctrl), "energy_distance": edist, "pvalue": pval}
    # Optional secondary metric: MMD
    if secondary_metric == "mmd":
        mmd_val = compute_mmd(X_target, Y_ctrl)
        row["mmd_distance"] = mmd_val
    return row, None


def compute_perturbation_distance(expr: ad.AnnData, cfg: Config) -> Optional[DistanceResults]:
    """Compute Perturbation Distance vs control and permutation DistanceTest for all targets.

    Parameters
    ----------
    expr : ad.AnnData
        Single-cell AnnData containing embeddings in obsm and guide classifications in obs.
    cfg : Config
        Full pipeline configuration.

    Returns
    -------
    Optional[DistanceResults]
        Target-level statistical distance results or None if disabled.
    """
    dcfg = cfg.distance
    if not dcfg.enabled:
        logger.info("Perturbation Distance disabled (distance.enabled: false)")
        return None
    logger.info("=== Computing Perturbation Distance vs Control (DistanceTest) ===")
    rep_name = dcfg.representation
    embedding = _resolve_representation(expr, rep_name)
    logger.info("Using representation %r (%d dimensions)", rep_name, embedding.shape[1])
    # Extract target and control assignments
    obs = expr.obs
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    base_masks = control_masks(expr, cfg)
    controls_available = {k: int(v.sum()) for k, v in base_masks.items()}
    # Select control population
    primary_ctrl = cfg.perturbation.primary_control
    if primary_ctrl in base_masks and controls_available.get(primary_ctrl, 0) >= dcfg.min_cells:
        ctrl_choice = primary_ctrl
    elif controls_available.get(CONTROL_NTC, 0) >= dcfg.min_cells:
        ctrl_choice = CONTROL_NTC
    elif controls_available.get(CONTROL_OTHER, 0) >= dcfg.min_cells:
        ctrl_choice = CONTROL_OTHER
    elif primary_ctrl in base_masks and controls_available.get(primary_ctrl, 0) > 0:
        ctrl_choice = primary_ctrl
    elif controls_available.get(CONTROL_NTC, 0) > 0:
        ctrl_choice = CONTROL_NTC
    elif controls_available.get(CONTROL_OTHER, 0) > 0:
        ctrl_choice = CONTROL_OTHER
    else:
        logger.warning("No control cells found for perturbation distance analysis.")
        return DistanceResults(
            table=pd.DataFrame(
                columns=["target_gene", "n_cells", "n_control", "energy_distance", "pvalue", "fdr", "significant"]
            ),
            skipped=pd.DataFrame(),
            primary_metric=dcfg.primary_metric,
            secondary_metric=dcfg.secondary_metric,
            representation=rep_name,
            note="No control cells available.",
        )
    ctrl_mask = base_masks[ctrl_choice]
    n_ctrl_total = int(ctrl_mask.sum())
    ctrl_indices_all = np.flatnonzero(ctrl_mask)
    logger.info(
        "Control group: %r (%d cells available, sampling up to %d)", ctrl_choice, n_ctrl_total, dcfg.max_control_cells
    )
    # Batch/lane strata for sampling
    strata = None
    if dcfg.stratify_by and dcfg.stratify_by in obs.columns:
        strata = obs[dcfg.stratify_by].astype(str).to_numpy()
    elif "lane_id" in obs.columns:
        strata = obs["lane_id"].astype(str).to_numpy()
    # Pre-sample control cells reproducibly
    rng_ctrl = np.random.default_rng(dcfg.random_seed)
    ctrl_indices_sampled = _sample_cell_indices(ctrl_indices_all, dcfg.max_control_cells, rng_ctrl, strata=strata)
    # The 'other' control is every targeting cell, so each target's own cells must be
    # dropped from the control sample inside the worker (as perturbation.py does).
    exclude_self = ctrl_choice == CONTROL_OTHER
    # Identify all targeting perturbations (membership in high-MOI mode: perturbed = cells carrying the target;
    # the 'other' control sample then drops those cells in the worker via exclude_self)
    from .high_moi import membership_index

    membership = membership_index(expr, cfg)
    targeting_mask = klass == CLASS_TARGETING
    if membership is not None:
        all_targets = list(membership.targets)
        if cfg.guides.high_moi.ntc_pseudo_targets:
            all_targets += [p for p in membership.pseudo_targets if membership.indices(p).size >= dcfg.min_cells]
    else:
        all_targets = sorted(set(targets_col[targeting_mask]))
    decision = resolve_stage_backend("distance", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)
    logger.info(
        "[distance] representation=%s representation_shape=%dx%d worker_data=shared anndata_passed_to_workers=false n_jobs=%d n_permutations=%d",
        rep_name,
        embedding.shape[0],
        embedding.shape[1],
        decision.n_jobs,
        dcfg.n_permutations,
    )
    if membership is not None:
        targeting_indices_dict: Dict[str, np.ndarray] = {target: membership.indices(target) for target in all_targets}
    else:
        targeting_indices_dict = {
            target: np.flatnonzero((targets_col == target) & targeting_mask) for target in all_targets
        }
    # Setup shared array buffer for worker processes
    shm_buffer: Optional[SharedArrayBuffer] = None
    worker_embedding: np.ndarray = embedding
    if getattr(cfg, "storage", None) and cfg.storage.shared_worker_arrays and decision.n_jobs > 1:
        try:
            shm_buffer = SharedArrayBuffer(embedding, create_memmap=True)
            worker_embedding = shm_buffer.array
        except Exception as exc:
            logger.debug("SharedArrayBuffer creation skipped: %s", exc)
            worker_embedding = embedding
    tasks = [
        (
            i,
            target,
            targeting_indices_dict.get(target, np.empty(0, dtype=np.int64)),
            worker_embedding,
            ctrl_indices_sampled,
            exclude_self,
            strata,
            dcfg.min_cells,
            dcfg.max_cells_per_target,
            dcfg.n_permutations,
            dcfg.random_seed,
            dcfg.primary_metric,
            dcfg.secondary_metric,
        )
        for i, target in enumerate(all_targets)
    ]
    try:
        results = run_parallel(
            _eval_target_dist_worker,
            tasks,
            n_jobs=decision.n_jobs,
            blas_threads=cfg.compute.blas_threads_per_worker,
            backend=cfg.compute.cpu_parallel_backend,
        )
    finally:
        if shm_buffer is not None:
            shm_buffer.close()
    rows = [r for r, s in results if r is not None]
    skipped = [s for r, s in results if s is not None]
    if not rows:
        logger.warning("No targets had sufficient cells for perturbation distance analysis.")
        table = pd.DataFrame(
            columns=["target_gene", "n_cells", "n_control", "energy_distance", "pvalue", "fdr", "significant"]
        )
        return DistanceResults(
            table=table,
            skipped=pd.DataFrame(skipped),
            primary_metric=dcfg.primary_metric,
            secondary_metric=dcfg.secondary_metric,
            control_used=ctrl_choice,
            n_control_cells=len(ctrl_indices_sampled),
            representation=rep_name,
            note="No targets met min_cells threshold.",
        )
    table = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR correction
    table["fdr"] = benjamini_hochberg(table["pvalue"].to_numpy())
    table["significant"] = table["fdr"] < dcfg.fdr_threshold
    # Sort descending by energy distance
    table = table.sort_values("energy_distance", ascending=False).reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped)
    n_sig = int(table["significant"].sum())
    logger.info(
        "Perturbation Distance complete: %d targets tested (%d significant at FDR < %.2f), %d skipped",
        len(table),
        n_sig,
        dcfg.fdr_threshold,
        len(skipped_df),
    )
    return DistanceResults(
        table=table,
        skipped=skipped_df,
        primary_metric=dcfg.primary_metric,
        secondary_metric=dcfg.secondary_metric,
        control_used=ctrl_choice,
        n_control_cells=len(ctrl_indices_sampled),
        representation=rep_name,
    )


# Perturbation Distance Space API


def compute_pcoa_coordinates(dist_matrix: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute classical Multidimensional Scaling (PCoA) from a distance matrix.

    Handles non-Euclidean eigenspectra transparently by retaining only positive
    eigenvalues and projecting along the positive eigenspace.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Symmetric (K, K) pairwise distance matrix with 0 diagonal.
    n_components : int
        Maximum number of PCoA coordinate axes to return.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (coordinates (K, p), positive_eigenvalues (p,)).
    """
    D = np.asarray(dist_matrix, dtype=np.float64)
    K = D.shape[0]
    if K < 2:
        return np.zeros((K, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
    # Double-centering: B = -0.5 * H * (D^2) * H
    D_sq = D**2
    row_means = np.mean(D_sq, axis=1, keepdims=True)
    col_means = np.mean(D_sq, axis=0, keepdims=True)
    grand_mean = np.mean(D_sq)
    B = -0.5 * (D_sq - row_means - col_means + grand_mean)
    # Eigendecomposition of symmetric matrix
    evals, evecs = np.linalg.eigh(B)
    # Sort descending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    # Retain strictly positive eigenvalues (filtering out non-Euclidean artifacts)
    pos_mask = evals > 1e-10
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        logger.warning("PCoA: no positive eigenvalues found in distance matrix.")
        return np.zeros((K, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
    p = min(n_components, n_pos)
    pos_evals = evals[:p]
    pos_evecs = evecs[:, :p]
    coords = pos_evecs * np.sqrt(pos_evals)
    return coords, pos_evals


def _eval_pair_dist_worker(task: Tuple[int, int, np.ndarray, np.ndarray, str]) -> Tuple[int, int, float]:
    """Evaluate pairwise distance between two perturbation samples."""
    i, j, X_i, X_j, metric = task
    if metric == "mmd":
        d_val = compute_mmd(X_i, X_j)
    else:
        d_val = compute_energy_distance(X_i, X_j)
    return i, j, d_val


def compute_distance_space(expr: ad.AnnData, cfg: Config) -> Optional[DistanceSpaceResults]:
    """Compute pairwise Perturbation Distance Space, PCoA embedding, and Phenotype Modules.

    Parameters
    ----------
    expr : ad.AnnData
        Single-cell AnnData containing embeddings in obsm and guide classifications in obs.
    cfg : Config
        Full pipeline configuration.

    Returns
    -------
    Optional[DistanceSpaceResults]
        DistanceSpace result object or None if disabled.
    """
    dscfg = cfg.distance_space
    if not dscfg.enabled:
        logger.info("Perturbation Distance Space disabled (distance_space.enabled: false)")
        return None
    logger.info("=== Computing Perturbation Distance Space (Pairwise Manifold) ===")
    rep_name = dscfg.representation
    embedding = _resolve_representation(expr, rep_name)
    obs = expr.obs
    from .high_moi import membership_index

    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    targeting_mask = klass == CLASS_TARGETING
    membership = membership_index(expr, cfg)
    if membership is not None:
        # Shared cells between targets inflate pairwise similarity in a high-MOI design; said in the report.
        all_targets = list(membership.targets)
    else:
        all_targets = sorted(set(targets_col[targeting_mask]))
    # Strata for sampling
    strata = None
    if "lane_id" in obs.columns:
        strata = obs["lane_id"].astype(str).to_numpy()
    # Filter eligible targets
    eligible_targets: List[str] = []
    target_samples: Dict[str, np.ndarray] = {}
    skipped: List[dict] = []
    for i, target in enumerate(all_targets):
        pert_mask = membership.mask(target) if membership is not None else (targets_col == target) & targeting_mask
        n_pert = int(pert_mask.sum())
        if n_pert < dscfg.min_cells:
            skipped.append(
                {"target_gene": target, "n_cells": n_pert, "reason": f"fewer than {dscfg.min_cells} cells ({n_pert})"}
            )
            continue
        pert_indices = np.flatnonzero(pert_mask)
        target_seed = (dscfg.random_seed + i * 43) % (2**31 - 1)
        rng = np.random.default_rng(target_seed)
        sampled_idx = _sample_cell_indices(pert_indices, dscfg.max_cells_per_target, rng, strata=strata)
        eligible_targets.append(target)
        target_samples[target] = embedding[sampled_idx]
    K = len(eligible_targets)
    if K < 2:
        logger.warning("Fewer than 2 targets met min_cells=%d for DistanceSpace (found %d).", dscfg.min_cells, K)
        empty_df = pd.DataFrame()
        return DistanceSpaceResults(
            distance_matrix=empty_df,
            coordinates=empty_df,
            neighbors=empty_df,
            phenotype_modules=empty_df,
            skipped=pd.DataFrame(skipped),
            note="Fewer than 2 eligible targets.",
        )
    decision = resolve_stage_backend("distance_space", cfg, n_cells=expr.n_obs, extra_info={"n_dense_elements": K * K})
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)
    logger.info("Computing pairwise %s matrix for %d perturbations...", dscfg.metric, K)
    dist_mat = np.zeros((K, K), dtype=np.float64)
    pair_tasks = [
        (i, j, target_samples[eligible_targets[i]], target_samples[eligible_targets[j]], dscfg.metric)
        for i in range(K)
        for j in range(i + 1, K)
    ]
    pair_results = run_parallel(
        _eval_pair_dist_worker,
        pair_tasks,
        n_jobs=decision.n_jobs,
        blas_threads=cfg.compute.blas_threads_per_worker,
        backend=cfg.compute.cpu_parallel_backend,
    )
    for i, j, d_val in pair_results:
        dist_mat[i, j] = d_val
        dist_mat[j, i] = d_val
    dist_df = pd.DataFrame(dist_mat, index=eligible_targets, columns=eligible_targets)
    # 1. PCoA Coordinates
    coords_arr, evals = compute_pcoa_coordinates(dist_mat, n_components=dscfg.n_components)
    p = coords_arr.shape[1]
    coord_cols = [f"PCoA{col + 1}" for col in range(p)]
    coords_df = pd.DataFrame(coords_arr, index=eligible_targets, columns=coord_cols)
    coords_df.insert(0, "target_gene", eligible_targets)
    coords_df = coords_df.reset_index(drop=True)
    # 2. Nearest Phenotypic Neighbors
    k_nn = min(dscfg.nearest_neighbors, K - 1)
    neighbor_rows: List[dict] = []
    for i, target in enumerate(eligible_targets):
        dists = dist_mat[i].copy()
        # Exclude self
        dists[i] = np.inf
        sorted_indices = np.argsort(dists)
        for rank in range(1, k_nn + 1):
            n_idx = sorted_indices[rank - 1]
            neighbor_rows.append(
                {
                    "target": target,
                    "neighbor": eligible_targets[n_idx],
                    "distance": float(dist_mat[i, n_idx]),
                    "rank": rank,
                }
            )
    neighbors_df = pd.DataFrame(neighbor_rows)
    # 3. Phenotype Modules (Hierarchical Clustering in Phenotype Distance Space)
    pheno_modules_df = pd.DataFrame(columns=["target_gene", "phenotype_module"])
    if dscfg.clustering and K >= 2:
        condensed_dist = squareform(dist_mat, checks=False)
        link = linkage(condensed_dist, method=dscfg.linkage_method)
        if dscfg.n_modules is not None:
            t_clust = min(dscfg.n_modules, K)
            clusters = fcluster(link, t=t_clust, criterion="maxclust")
        elif dscfg.cluster_distance_threshold is not None:
            clusters = fcluster(link, t=dscfg.cluster_distance_threshold, criterion="distance")
        else:
            # Default reasonable module count
            t_clust = max(2, min(9, K // 4 if K >= 8 else K))
            clusters = fcluster(link, t=t_clust, criterion="maxclust")
        module_labels = [f"PM{c}" for c in clusters]
        pheno_modules_df = pd.DataFrame({"target_gene": eligible_targets, "phenotype_module": module_labels})
    logger.info(
        "Distance Space complete: %d x %d matrix, %d PCoA coordinates, %d neighbors per target, %d phenotype modules",
        K,
        K,
        p,
        k_nn,
        pheno_modules_df["phenotype_module"].nunique() if not pheno_modules_df.empty else 0,
    )
    return DistanceSpaceResults(
        distance_matrix=dist_df,
        coordinates=coords_df,
        neighbors=neighbors_df,
        phenotype_modules=pheno_modules_df,
        skipped=pd.DataFrame(skipped),
        eigenvalues=evals,
        n_components=p,
        metric=dscfg.metric,
        linkage_method=dscfg.linkage_method,
    )
