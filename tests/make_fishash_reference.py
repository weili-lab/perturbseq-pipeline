"""Regenerate ``tests/golden/fishash_reference.json``: synthetic guide counts and the calls of R fishash.

Needs R with Matrix, SummarizedExperiment, S4Vectors, sparseMatrixStats and dplyr,
and a checkout of https://github.com/jackkamm/fishash (its R/ files are sourced,
not installed)::

    python tests/make_fishash_reference.py /path/to/Rscript /path/to/fishash

The counts mix negative-binomial signal with Poisson ambient noise (a profile
unrelated to the signal, so the Simpson's-paradox refit changes calls) and
chimeric noise (the bulk signal profile), at a low and a high MOI.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import io, sparse

R_CODE = r"""
args <- commandArgs(trailingOnly = TRUE)
pkg_dir <- args[1]; work <- args[2]
suppressPackageStartupMessages({ library(Matrix); library(SummarizedExperiment); library(S4Vectors) })
colMins <- sparseMatrixStats::colMins
case_when <- dplyr::case_when
source(file.path(pkg_dir, "R", "impute_masked_counts.R"))
source(file.path(pkg_dir, "R", "fishash.R"))
for (name in c("low_moi", "high_moi")) {
  counts <- as(readMM(file.path(work, paste0(name, ".mtx"))), "CsparseMatrix")  # guides x cells
  dimnames(counts) <- list(paste0("g", seq_len(nrow(counts))), paste0("c", seq_len(ncol(counts))))
  for (refit in c(0, 10)) {
    res <- fishash(counts, refit = refit)
    writeMM(as(assay(res, "assigned") * 1, "CsparseMatrix"), file.path(work, sprintf("%s_refit%d.mtx", name, refit)))
  }
}
"""


def synthetic_counts(moi: float, seed: int, n_cells: int = 600, n_guides: int = 40) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    guide_size = rng.lognormal(0, 0.5, n_guides)
    cell_size = rng.lognormal(0, 0.5, n_cells)
    signal = np.zeros((n_cells, n_guides))
    for c in range(n_cells):
        g = rng.choice(n_guides, rng.poisson(moi), replace=False)
        signal[c, g] = rng.negative_binomial(1, 1 / (1 + 40 * cell_size[c] * guide_size[g]))
    ambient = rng.dirichlet(np.ones(n_guides))
    bulk = signal.sum(0) / signal.sum()
    noise = rng.poisson(np.outer(cell_size * 3.0, ambient) + np.outer(cell_size * 2.0, bulk))
    return sparse.csr_matrix(signal + noise)


def main(rscript: str, fishash_dir: str) -> None:
    out = {}
    with tempfile.TemporaryDirectory() as work:
        scenarios = {"low_moi": synthetic_counts(1.0, 1), "high_moi": synthetic_counts(6.0, 2)}
        for name, X in scenarios.items():
            io.mmwrite(f"{work}/{name}.mtx", X.T.tocoo())
        (Path(work) / "run.R").write_text(R_CODE)
        subprocess.run([rscript, f"{work}/run.R", fishash_dir, work], check=True)
        for name, X in scenarios.items():
            coo = X.tocoo()
            entry = {"shape": list(X.shape), "row": coo.row.tolist(), "col": coo.col.tolist(), "data": coo.data.tolist()}
            for refit in (0, 10):
                called = sparse.coo_matrix(io.mmread(f"{work}/{name}_refit{refit}.mtx").T)
                entry[f"r_calls_refit{refit}"] = [called.row.tolist(), called.col.tolist()]
            out[name] = entry
    path = Path(__file__).parent / "golden" / "fishash_reference.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
