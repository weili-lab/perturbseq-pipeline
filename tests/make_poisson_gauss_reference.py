"""Regenerate ``tests/golden/poisson_gauss_reference.json`` with crispat (pyro), for test_guide_calling.py.

Needs ``pip install crispat`` (it brings pyro and torch)::

    python tests/make_poisson_gauss_reference.py

Per synthetic guide it records crispat's fitted parameters, pyro's loss at exactly
those parameters, and crispat's cutoff under five sets of seeds (crispat uses seeds
0-9; here also 10-19, ..., 40-49). The fit stops before converging, so on some
guides the cutoff depends on the seeds; only guides whose cutoff is the same for
all five sets can be compared with another implementation.
"""

from __future__ import annotations

import builtins
import json
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pyro
import torch
from crispat import poisson_gauss as pg
from scipy import sparse


def synthetic_guides(n_cells: int = 3000, n_guides: int = 16, seed: int = 7) -> sparse.csr_matrix:
    """Ambient background in every guide; strong / medium / weak signal in three of every four."""
    rng = np.random.default_rng(seed)
    cols = []
    for g in range(n_guides):
        x = rng.poisson(rng.uniform(0.3, 3.0), n_cells) * (rng.random(n_cells) < rng.uniform(0.05, 0.6))
        if g % 4 != 3:
            carriers = rng.random(n_cells) < rng.uniform(0.01, 0.08)
            mean = (60, 15, 6)[g % 4] * rng.uniform(0.7, 1.4)
            x[carriers] += rng.negative_binomial(2, 2 / (2 + mean), carriers.sum())
        cols.append(x)
    return sparse.csr_matrix(np.column_stack(cols).astype(np.float32))


def main() -> None:
    X = synthetic_guides()
    adata = ad.AnnData(X)
    adata.var_names = [f"g{i}" for i in range(X.shape[1])]
    adata.obs_names = [f"c{i}" for i in range(X.shape[0])]
    out_dir = tempfile.mkdtemp() + "/"
    guides = []
    for g in range(X.shape[1]):
        cutoffs = []
        for offset in (0, 10, 20, 30, 40):
            # crispat loops `for seed in range(10)`; shift the seeds it sees
            pg.range = lambda n, o=offset: builtins.range(o, o + n)
            _, threshold, _, estimates = pg.fit_PGMM(f"g{g}", adata, out_dir, 2024, 500, make_plots=False)
            cutoffs.append(float(threshold) if np.isfinite(float(threshold)) and float(threshold) > 0 else None)
            if offset == 0:
                selected = adata[:, [f"g{g}"]].X.toarray()
                data = torch.tensor(np.log2(selected[selected != 0])).float()
                elbo = pyro.infer.TraceEnum_ELBO(num_particles=1, max_plate_nesting=1)
                loss_at_fit = float(elbo.loss(pg.model, pg.global_guide, data))  # crispat's fitted parameters
                params = {k: float(estimates[k].iloc[0]) for k in ("weight_Normal", "mu", "scale", "lambda")}
        pg.range = builtins.range
        guides.append({"params": params, "loss_at_params": loss_at_fit, "cutoffs_by_seed_set": cutoffs})
    coo = X.tocoo()
    payload = {
        "shape": list(X.shape),
        "row": coo.row.tolist(),
        "col": coo.col.tolist(),
        "data": coo.data.astype(int).tolist(),
        "guides": guides,
    }
    path = Path(__file__).parent / "golden" / "poisson_gauss_reference.json"
    path.write_text(json.dumps(payload))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
