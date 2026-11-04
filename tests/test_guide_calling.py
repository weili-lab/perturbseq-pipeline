"""Guide calling by fishash and by the Poisson-Gaussian mixture: each reproduces its reference
(R fishash, crispat), and both assignment modes use their calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import check_grad
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import KD_TARGETS, make_dataset  # noqa: E402
from test_high_moi_membership_stats import _base_cfg  # noqa: E402

from perturbseq_pipeline.guide_calling import (  # noqa: E402
    _fit_pgmm,
    _pgmm_loss_and_grad,
    _pgmm_threshold,
    call_fishash,
    call_poisson_gauss,
    log_phyper_upper,
)
from perturbseq_pipeline.guides import CLASS_AMBIGUOUS, CLASS_UNASSIGNED, OBS_CLASS, OBS_GUIDE, assign_guides  # noqa: E402

REFERENCE = json.loads((Path(__file__).parent / "golden" / "fishash_reference.json").read_text())
PG_REFERENCE = json.loads((Path(__file__).parent / "golden" / "poisson_gauss_reference.json").read_text())


@pytest.mark.parametrize("scenario", ["low_moi", "high_moi"])
@pytest.mark.parametrize("refit", [0, 10])
def test_calls_match_r_fishash(scenario, refit):
    """Same calls as R fishash 0.99.5 (padj GS, FDR 0.05, min_count 2) on the reference matrices
    (regenerate with tests/make_fishash_reference.py)."""
    ref = REFERENCE[scenario]
    X = sparse.csr_matrix((ref["data"], (ref["row"], ref["col"])), shape=tuple(ref["shape"]))
    ours = call_fishash(X, fdr=0.05, min_count=2, refit=refit).tocoo()
    r_rows, r_cols = ref[f"r_calls_refit{refit}"]
    assert set(zip(ours.row.tolist(), ours.col.tolist())) == set(zip(r_rows, r_cols))


def test_hypergeometric_tail_matches_scipy():
    rng = np.random.default_rng(0)
    total = rng.integers(2_000, 50_000, 500).astype(float)
    white = np.floor(total * rng.uniform(0.001, 0.2, 500))
    draws = np.floor(total * rng.uniform(0.001, 0.05, 500))
    x = np.floor(rng.uniform(0, 1, 500) * np.minimum(white, draws))
    ours = log_phyper_upper(x, white, total - white, draws)
    expected = hypergeom.logsf(x, total, white, draws)
    finite = np.isfinite(expected) & (expected > -500)
    np.testing.assert_allclose(ours[finite], expected[finite], rtol=1e-6, atol=1e-8)


def _pg_guides():
    """(distinct log2 counts, cells per count, max count, reference) per guide of the crispat reference."""
    ref = PG_REFERENCE
    X = sparse.csc_matrix((ref["data"], (ref["row"], ref["col"])), shape=tuple(ref["shape"]), dtype=np.float64)
    for g, guide in enumerate(ref["guides"]):
        values, n = np.unique(X.data[X.indptr[g] : X.indptr[g + 1]], return_counts=True)
        yield np.log2(values), n.astype(np.float64), int(values[-1]), guide


def test_poisson_gauss_loss_is_crispats():
    """Same loss as crispat's pyro model at crispat's fitted parameters (crispat computes in float32)."""
    for y, n, _, guide in _pg_guides():
        p = guide["params"]
        theta = np.array([np.log(1 - p["weight_Normal"]), np.log(p["weight_Normal"]), p["mu"], np.log(p["scale"]), np.log(p["lambda"])])
        assert np.isclose(_pgmm_loss_and_grad(theta, y, n)[0], guide["loss_at_params"], rtol=1e-5)


def test_poisson_gauss_gradient():
    y, n, _, _ = next(_pg_guides())
    for theta in ([-0.7, -0.7, 3.0, 2.0, 0.0], [1.5, 0.0, 5.0, -0.3, 0.4], [-1.0, 0.5, 1.5, 0.2, -1.0]):
        error = check_grad(lambda t: _pgmm_loss_and_grad(t, y, n)[0], lambda t: _pgmm_loss_and_grad(t, y, n)[1], np.array(theta))
        assert error < 1e-6 * np.linalg.norm(_pgmm_loss_and_grad(np.array(theta), y, n)[1]) + 1e-3


def test_poisson_gauss_cutoffs_match_crispat():
    """Same cutoff as crispat (seeds 0-9) on every reference guide. crispat stops before converging, so
    its cutoff can depend on the start (the reference also records seeds 10-49, which differ on some
    guides); with crispat's own starts the numpy fit must reproduce its seeds-0-9 cutoff."""
    for y, n, max_count, guide in _pg_guides():
        expected = guide["cutoffs_by_seed_set"][0]
        assert _pgmm_threshold(_fit_pgmm(y, n), max_count) == (np.inf if expected is None else expected)


def _assigned_guides(tmp_path, **guides):
    from perturbseq_pipeline import io as io_mod
    from perturbseq_pipeline import qc as qc_mod

    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300, moi=guides.pop("moi", None))
    cfg = _base_cfg(data, tmp_path / "run", **guides)
    cfg.validate()
    loaded = io_mod.load_data(cfg)
    expr = qc_mod.prefilter(loaded.expr, cfg)
    expr = qc_mod.compute_qc_metrics(expr, cfg)
    expr, _ = qc_mod.filter_cells_and_genes(expr, cfg)
    guides_aligned = loaded.guides[expr.obs_names]
    X = sparse.csr_matrix(guides_aligned.layers["counts"] if "counts" in guides_aligned.layers else guides_aligned.X)
    return assign_guides(expr, loaded.guides, cfg), X, list(guides_aligned.var_names), cfg


def _calls(X, cfg, method):
    g = cfg.guides
    if method == "fishash":
        return call_fishash(X, g.fishash.fdr, g.fishash.min_count, g.fishash.refit)
    return call_poisson_gauss(X, g.poisson_gauss.min_count, g.poisson_gauss.min_cells)


@pytest.mark.parametrize("method", ["fishash", "poisson_gauss"])
def test_single_guide_assigns_cells_with_exactly_one_call(tmp_path, method):
    expr, X, guide_ids, cfg = _assigned_guides(tmp_path, calling_method=method)
    calls = _calls(X, cfg, method)
    n_called = np.diff(calls.indptr)
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    guide = expr.obs[OBS_GUIDE].astype(str).to_numpy()
    assert (klass[n_called >= 2] == CLASS_AMBIGUOUS).all()
    assert (klass[n_called == 0] == CLASS_UNASSIGNED).all()
    one = np.flatnonzero(n_called == 1)
    assert one.size > 200
    # the cell's guide is its called guide (not necessarily its top-UMI guide)
    assert all(guide[i] == guide_ids[calls.indices[calls.indptr[i]]] for i in one)
    assert set(KD_TARGETS) <= set(expr.obs["target_gene"].astype(str))


@pytest.mark.parametrize("method", ["fishash", "poisson_gauss"])
def test_high_moi_membership_is_the_calls(tmp_path, method):
    expr, X, _, cfg = _assigned_guides(tmp_path, moi=4, assignment_mode="high_moi", high_moi={"method": method})
    calls = _calls(X, cfg, method)
    membership = sparse.csr_matrix(expr.obsm[cfg.guides.high_moi.guide_membership_obsm_key])
    under_cap = np.diff(calls.indptr) <= cfg.guides.high_moi.max_guides_per_cell
    assert (np.diff(calls.indptr) >= 2).sum() > 100  # many multi-guide cells
    assert (membership[under_cap] != calls[under_cap]).nnz == 0


@pytest.mark.parametrize(
    "guides",
    [
        {"calling_method": method}
        if mode == "single_guide"
        else {"assignment_mode": "high_moi", "high_moi": {"method": method}}
        for mode in ("single_guide", "high_moi")
        for method in ("fishash", "poisson_gauss")
    ],
)
def test_pipeline_runs_with_per_guide_calling(tmp_path, guides):
    from perturbseq_pipeline.cli import run_pipeline

    moi = 4 if guides.get("assignment_mode") == "high_moi" else None
    data = make_dataset(tmp_path / "synthetic", n_lanes=2, n_cells=300, moi=moi)
    cfg = _base_cfg(data, tmp_path / "run", **guides)
    cfg.modules.enabled = False
    cfg.validate()
    result = run_pipeline(cfg)
    assert result.adata.n_obs > 0
    report = (Path(cfg.run.outdir) / "report.md").read_text()
    if "calling_method" in guides:
        assert f"Rule (`guides.calling_method: {guides['calling_method']}`)" in report
        assert "Rule: a cell is assigned to its dominant guide" not in report
    else:
        assert f"Membership rule (`guides.high_moi.method: {guides['high_moi']['method']}`)" in report
