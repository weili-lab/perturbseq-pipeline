"""Generate a tiny synthetic Perturb-seq dataset for testing.

Writes 10x-style MTX directories (gene expression + guide features) with a known
ground truth: the guides against ``KD_*`` genes really do knock their target
down, while ``NULL_*`` targets have no effect. That lets the tests assert that
the perturbation stage recovers the right answer instead of merely running.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

N_GENES = 300
N_GUIDES_PER_TARGET = 3
N_NTC_GUIDES = 6
KD_TARGETS = ["KDGENE1", "KDGENE2", "KDGENE3"]
NULL_TARGETS = ["NULLGENE1", "NULLGENE2"]


def _write_mtx_dir(path: Path, matrix, barcodes: List[str], features: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(path / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(barcodes) + "\n")
    with gzip.open(path / "features.tsv.gz", "wt") as fh:
        features.to_csv(fh, sep="\t", header=False, index=False)
    # 10x stores features x cells.
    with gzip.open(path / "matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, sp.csr_matrix(matrix.T).astype(int), field="integer")


def make_lane(
    outdir: Path,
    lane_id: str,
    n_cells: int = 400,
    seed: int = 0,
    knockdown_strength: float = 0.15,
) -> Path:
    """Write one synthetic lane and return its directory."""
    rng = np.random.default_rng(seed)

    gene_names = [f"GENE{i:04d}" for i in range(N_GENES)]
    # Put the perturbation targets in the expression matrix too.
    targets = KD_TARGETS + NULL_TARGETS
    gene_names[: len(targets)] = targets
    # A few mitochondrial genes so QC metrics are non-trivial.
    gene_names[-5:] = [f"MT-CO{i}" for i in range(1, 6)]

    guide_ids: List[str] = []
    guide_targets: List[str] = []
    for t in targets:
        for k in range(1, N_GUIDES_PER_TARGET + 1):
            guide_ids.append(f"{t}_P1P2_{k}")
            guide_targets.append(t)
    for k in range(1, N_NTC_GUIDES + 1):
        guide_ids.append(f"non_targeting_{k}")
        guide_targets.append("non")

    # --- expression: negative-binomial-ish counts --------------------------
    base = rng.gamma(shape=1.5, scale=2.0, size=N_GENES)
    expr = rng.poisson(base[None, :] * rng.uniform(0.6, 1.6, size=(n_cells, 1)))
    # Guarantee the target genes are well expressed so knockdown is visible.
    for t in targets:
        expr[:, gene_names.index(t)] = rng.poisson(30, size=n_cells)

    # --- guide assignment: one dominant guide per cell ---------------------
    n_guides = len(guide_ids)
    assigned = rng.integers(0, n_guides, size=n_cells)
    guide_counts = rng.poisson(0.3, size=(n_cells, n_guides))
    guide_counts[np.arange(n_cells), assigned] += rng.poisson(40, size=n_cells) + 10

    # 10% of cells get no guide (unassigned) and 10% get a co-dominant second
    # guide (ambiguous), so the QC categories are exercised.
    no_guide = rng.random(n_cells) < 0.10
    guide_counts[no_guide, :] = 0
    ambiguous = (~no_guide) & (rng.random(n_cells) < 0.10)
    second = rng.integers(0, n_guides, size=n_cells)
    guide_counts[ambiguous, second[ambiguous]] = guide_counts[
        ambiguous, assigned[ambiguous]
    ]

    # --- apply the ground-truth knockdown ----------------------------------
    effective = ~no_guide & ~ambiguous
    for t in KD_TARGETS:
        col = gene_names.index(t)
        cells = effective & np.isin(assigned, [i for i, g in enumerate(guide_targets) if g == t])
        expr[cells, col] = rng.poisson(30 * knockdown_strength, size=int(cells.sum()))

    barcodes = [f"{lane_id}_CELL{i:05d}-1" for i in range(n_cells)]
    features = pd.DataFrame(
        {
            "id": [f"ENSG{i:08d}" for i in range(N_GENES)] + guide_ids,
            "name": gene_names + guide_ids,
            "type": ["Gene Expression"] * N_GENES + ["Custom"] * n_guides,
        }
    )
    matrix = np.hstack([expr, guide_counts])
    path = outdir / f"filtered_feature_bc_matrix_{lane_id}"
    _write_mtx_dir(path, matrix, barcodes, features)
    return path


def make_split_lane(outdir: Path, lane_id: str, n_cells: int = 300, seed: int = 0,
                    whitelist_extra: int = 500) -> dict:
    """Write a lane with gene expression and guides in *separate* directories.

    Mimics a STARsolo-style run: ``GEX/`` holds called cells only, while
    ``sgRNA/`` covers a much larger barcode whitelist, so the loader has to
    subset and align.
    """
    rng = np.random.default_rng(seed)
    combined = make_lane(outdir / "_tmp", lane_id, n_cells=n_cells, seed=seed)

    import scanpy as sc

    a = sc.read_10x_mtx(combined, gex_only=False, cache=False)
    a.var_names_make_unique()
    is_guide = (a.var["feature_types"] == "Custom").to_numpy()
    gex, guides = a[:, ~is_guide].copy(), a[:, is_guide].copy()

    gex_dir = outdir / lane_id / "GEX" / "filtered"
    _write_mtx_dir(
        gex_dir,
        gex.X.toarray() if hasattr(gex.X, "toarray") else np.asarray(gex.X),
        list(gex.obs_names),
        pd.DataFrame({"id": list(gex.var["gene_ids"]), "name": list(gex.var_names),
                      "type": ["Gene Expression"] * gex.n_vars}),
    )

    # The guide matrix carries extra whitelist barcodes the GEX run never called.
    extra = [f"WHITELIST{i:06d}" for i in range(whitelist_extra)]
    gmat = np.vstack([
        guides.X.toarray() if hasattr(guides.X, "toarray") else np.asarray(guides.X),
        rng.poisson(0.05, size=(whitelist_extra, guides.n_vars)),
    ])
    order = rng.permutation(gmat.shape[0])
    all_bc = list(guides.obs_names) + extra
    _write_mtx_dir(
        outdir / lane_id / "sgRNA" / "raw",
        gmat[order],
        [all_bc[i] for i in order],
        pd.DataFrame({"id": list(guides.var["gene_ids"]), "name": list(guides.var_names),
                      "type": ["Gene Expression"] * guides.n_vars}),
    )
    shutil.rmtree(outdir / "_tmp", ignore_errors=True)
    return {"gex": str(gex_dir), "guides": str(outdir / lane_id / "sgRNA" / "raw")}


def make_dataset(outdir: Path, n_lanes: int = 2, n_cells: int = 400) -> dict:
    """Write ``n_lanes`` lanes plus a sample metadata CSV."""
    outdir = Path(outdir)
    lanes = {}
    for i in range(n_lanes):
        lane_id = f"L{i + 1}"
        lanes[lane_id] = str(make_lane(outdir, lane_id, n_cells=n_cells, seed=i))
    meta = pd.DataFrame(
        {
            "lane_id": list(lanes),
            "sample": [f"S{i // 2 + 1}" for i in range(n_lanes)],
            "condition": ["treated" if i % 2 else "control" for i in range(n_lanes)],
            "replicate": [i + 1 for i in range(n_lanes)],
        }
    )
    meta_path = outdir / "sample_metadata.csv"
    meta.to_csv(meta_path, index=False)
    return {"lanes": lanes, "metadata": str(meta_path)}


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
    info = make_dataset(out)
    print(f"Wrote {len(info['lanes'])} lanes to {out}")
