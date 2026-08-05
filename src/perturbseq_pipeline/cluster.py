"""Normalization, dimensionality reduction, embedding and Leiden clustering.

After :func:`normalize`, the contract used everywhere downstream holds:

* ``layers['counts']``  — raw integer counts
* ``layers['lognorm']`` — log1p of library-size-normalized counts
* ``adata.X``           — the same log-normalized values

The perturbation tests read ``layers['lognorm']`` explicitly rather than ``X``,
so a later scaling step cannot silently change what is being tested.
"""

from __future__ import annotations

import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .config import Config
from .io import LANE_KEY

logger = logging.getLogger(__name__)

LOGNORM_LAYER = "lognorm"
CLUSTER_KEY = "leiden"


def normalize(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Library-size normalize and log1p-transform, preserving raw counts.

    If ``layers['lognorm']`` already exists (because the user pointed
    ``input.normalized_layer`` at a pre-computed layer) this step is skipped.
    """
    if LOGNORM_LAYER in expr.layers:
        logger.info("Using pre-computed '%s' layer; skipping normalization", LOGNORM_LAYER)
        expr.X = expr.layers[LOGNORM_LAYER].copy()
        return expr

    if "counts" not in expr.layers:
        expr.layers["counts"] = expr.X.copy()
    expr.X = expr.layers["counts"].copy()

    sc.pp.normalize_total(expr, target_sum=cfg.cluster.target_sum)
    sc.pp.log1p(expr)
    expr.layers[LOGNORM_LAYER] = expr.X.copy()
    logger.info(
        "Normalized to %s counts per cell and log1p-transformed",
        cfg.cluster.target_sum or "median library size",
    )
    return expr


def embed_and_cluster(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """HVG selection, PCA, optional Harmony, neighbors, UMAP and Leiden."""
    c = cfg.cluster
    sc.settings.seed = cfg.run.seed

    n_top = min(c.n_top_genes, expr.n_vars)
    sc.pp.highly_variable_genes(expr, n_top_genes=n_top)
    logger.info("Selected %d highly variable genes", int(expr.var["highly_variable"].sum()))

    if c.regress_out:
        missing = [k for k in c.regress_out if k not in expr.obs.columns]
        if missing:
            raise ValueError(f"cluster.regress_out refers to missing obs columns: {missing}")
        logger.info("Regressing out %s", c.regress_out)
        sc.pp.regress_out(expr, c.regress_out)

    n_pcs = int(min(c.n_pcs, expr.n_obs - 1, expr.n_vars - 1))
    if c.scale_max_value is not None:
        # Scale and run PCA on the highly variable genes only. scanpy's PCA
        # already restricts itself to them, so scaling the full matrix just
        # densifies every gene for nothing: at 134k cells that is tens of GB
        # rather than the ~1.5 GB the HVG block needs. Results are unchanged.
        hvg = expr.var["highly_variable"].to_numpy()
        sub = expr[:, hvg].copy()
        sc.pp.scale(sub, max_value=c.scale_max_value)
        sc.tl.pca(sub, n_comps=n_pcs, svd_solver="arpack", random_state=cfg.run.seed)
        expr.obsm["X_pca"] = sub.obsm["X_pca"]
        expr.uns["pca"] = sub.uns["pca"]
        del sub
    else:
        sc.tl.pca(expr, n_comps=n_pcs, svd_solver="arpack", random_state=cfg.run.seed)
    logger.info("PCA: %d components (on %d HVGs)", n_pcs, int(expr.var["highly_variable"].sum()))

    use_rep = "X_pca"
    if c.batch_key:
        use_rep = _run_harmony(expr, cfg) or "X_pca"

    sc.pp.neighbors(
        expr,
        n_neighbors=c.n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        random_state=cfg.run.seed,
    )
    sc.tl.umap(expr, min_dist=c.umap_min_dist, random_state=cfg.run.seed)
    sc.tl.leiden(
        expr,
        key_added=CLUSTER_KEY,
        resolution=c.leiden_resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=cfg.run.seed,
    )
    n_clusters = expr.obs[CLUSTER_KEY].nunique()
    logger.info(
        "Leiden clustering at resolution %.2f: %d clusters", c.leiden_resolution, n_clusters
    )

    # X was never scaled in place (scaling happens on an HVG copy), so it still
    # holds log-normalized values; this only matters if a future change moves
    # the scaling back onto ``expr`` itself.
    if LOGNORM_LAYER in expr.layers:
        expr.X = expr.layers[LOGNORM_LAYER].copy()
    return expr


def reset_embedding(expr: ad.AnnData) -> ad.AnnData:
    """Drop everything :func:`embed_and_cluster` produced, in place.

    Used when a subset of an already-embedded object is re-embedded on its own
    (``cluster.assigned_only``): the inherited PCA, neighbor graph, UMAP and HVG
    flags describe the *previous* cell set, and leaving them behind would either
    be silently reused (``X_pca_harmony`` as ``use_rep``) or written to the
    output as stale coordinates. The ``counts``/``lognorm`` layers are per-cell
    and stay valid, so they are kept.
    """
    for key in ("X_pca", "X_pca_harmony", "X_umap"):
        if key in expr.obsm:
            del expr.obsm[key]
    for key in list(expr.obsp.keys()):
        del expr.obsp[key]
    for key in ("pca", "neighbors", "umap", CLUSTER_KEY, f"{CLUSTER_KEY}_colors"):
        expr.uns.pop(key, None)
    if "highly_variable" in expr.var.columns:
        del expr.var["highly_variable"]
    if CLUSTER_KEY in expr.obs.columns:
        del expr.obs[CLUSTER_KEY]
    return expr


def _run_harmony(expr: ad.AnnData, cfg: Config) -> Optional[str]:
    """Batch-correct the PCA embedding with Harmony; returns the new rep key."""
    key = cfg.cluster.batch_key
    if key not in expr.obs.columns:
        raise ValueError(
            f"cluster.batch_key={key!r} is not an obs column. Available: "
            f"{sorted(expr.obs.columns)[:30]}"
        )
    if expr.obs[key].nunique() < 2:
        logger.warning(
            "cluster.batch_key=%r has a single level; skipping batch correction.", key
        )
        return None
    try:
        import harmonypy
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Harmony batch correction requested but harmonypy is not installed. "
            "Install it with: pip install 'perturbseq-pipeline[harmony]'"
        ) from exc

    # harmonypy is called directly rather than through
    # ``sc.external.pp.harmony_integrate``: that wrapper transposes ``Z_corr``,
    # which was correct while harmonypy returned a (components x cells) matrix
    # but silently produces a wrongly-shaped embedding since harmonypy 2.0
    # switched to (cells x components).
    embedding = np.asarray(expr.obsm["X_pca"], dtype=np.float64)
    out = harmonypy.run_harmony(embedding, expr.obs, key)
    corrected = np.asarray(out.result() if hasattr(out, "result") else out.Z_corr)

    # Orient defensively, so a future flip in either direction cannot pass.
    if corrected.shape != embedding.shape:
        if corrected.T.shape == embedding.shape:
            corrected = corrected.T
        else:
            raise RuntimeError(
                "Harmony returned an embedding of shape "
                f"{corrected.shape}, which matches neither "
                f"{embedding.shape} nor its transpose."
            )

    expr.obsm["X_pca_harmony"] = corrected
    logger.info(
        "Harmony batch correction on %r across %d batches",
        key,
        expr.obs[key].nunique(),
    )
    return "X_pca_harmony"


def cluster_summary(expr: ad.AnnData, lane_key: str = LANE_KEY) -> pd.DataFrame:
    """Cluster sizes and their lane composition.

    A cluster dominated by one lane is the classic sign that batch correction is
    needed, so the per-lane share is reported rather than just cluster sizes.
    """
    if CLUSTER_KEY not in expr.obs.columns:
        return pd.DataFrame()
    obs = expr.obs
    rows = []
    for cl, sub in obs.groupby(obs[CLUSTER_KEY].astype(str), observed=True):
        row = {
            "cluster": cl,
            "n_cells": len(sub),
            "pct_of_total": round(100 * len(sub) / expr.n_obs, 2),
        }
        if "n_genes_by_counts" in sub.columns:
            row["median_genes"] = float(np.median(sub["n_genes_by_counts"]))
        if "pct_counts_mt" in sub.columns:
            row["median_pct_mt"] = round(float(np.median(sub["pct_counts_mt"])), 2)
        if lane_key in sub.columns and obs[lane_key].nunique() > 1:
            share = sub[lane_key].astype(str).value_counts(normalize=True)
            row["top_lane"] = share.index[0]
            row["top_lane_pct"] = round(100 * float(share.iloc[0]), 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("n_cells", ascending=False).reset_index(drop=True)


def cluster_composition(expr: ad.AnnData, group_key: str) -> pd.DataFrame:
    """Contingency table of Leiden cluster against another ``obs`` column."""
    if CLUSTER_KEY not in expr.obs.columns or group_key not in expr.obs.columns:
        return pd.DataFrame()
    return (
        expr.obs.groupby([CLUSTER_KEY, group_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
