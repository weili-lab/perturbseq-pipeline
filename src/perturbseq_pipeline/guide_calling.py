"""Guide calling on the cells x guides UMI matrix, shared by the single-guide and high-MOI modes.

``fishash`` (Kamm, Yeung & Forrest 2026, BMC Bioinformatics; R package ``fishash``)
treats the UMI matrix as a contingency table. For every non-zero entry (cell c,
guide g) a one-sided Fisher exact test asks whether g is over-represented in c
given the guide's and the cell's totals:

    P(g, c) = P(X >= N[g, c]),  X ~ Hypergeometric(total, guide total, cell total)

so both guide abundance and cell depth are normalised. Calls are made at a
false-discovery rate with the Guo & Sarkar (2020) procedure: Bonferroni within a
cell, Benjamini-Hochberg across cells (tests within a cell are negatively
dependent, which breaks plain BH).

Simpson's-paradox correction (``refit``): guides abundant in the signal but rare
in the background make absent guides look over-represented in other cells. The
other cells' totals are therefore re-estimated as background only: the entries
called in the previous round are masked and replaced by a rank-1 Poisson fit
(guide frequency x cell size) of the unmasked entries, and the test is repeated
until the calls stop changing. From the fourth round on, masked entries stay
masked, which guarantees convergence.

This is a port of ``fishash()`` (GS correction only) and ``impute_masked_counts()``
from https://github.com/jackkamm/fishash; the hypergeometric tail follows R's
``phyper``, which rounds non-integer totals (Equation 7 of the paper).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import sparse
from scipy.special import gammaln

logger = logging.getLogger(__name__)


def _log_choose(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def log_phyper_upper(x: np.ndarray, n_white: np.ndarray, n_black: np.ndarray, n_draw: np.ndarray) -> np.ndarray:
    """log P(X > x) for X ~ Hypergeometric; R's ``phyper(x, m, n, k, lower.tail=FALSE, log.p=TRUE)``.

    As in R, the tail is summed from the end where the terms shrink (swapping
    the colours when ``x`` is above its expectation): a ratio series that
    stays finite far into the tail and is ~200x faster than
    ``scipy.stats.hypergeom.logsf`` on these sizes.
    """
    x = np.floor(x + 1e-7)
    n_white, n_black, n_draw = np.rint(n_white), np.rint(n_black), np.rint(n_draw)
    swap = x * (n_white + n_black) > n_draw * n_white
    n_white, n_black = np.where(swap, n_black, n_white), np.where(swap, n_white, n_black)
    x = np.where(swap, n_draw - x - 1, x)
    lower = swap  # after the swap the wanted tail is the lower one
    out = np.empty(x.shape)
    cdf_zero = (x < 0) | (x < n_draw - n_black)
    cdf_one = ~cdf_zero & ((x >= n_white) | (x >= n_draw))
    out[cdf_zero] = np.where(lower[cdf_zero], -np.inf, 0.0)
    out[cdf_one] = np.where(lower[cdf_one], 0.0, -np.inf)
    i = np.flatnonzero(~cdf_zero & ~cdf_one)
    xi, w, b, d = x[i], n_white[i], n_black[i], n_draw[i]
    log_density = _log_choose(w, xi) + _log_choose(b, d - xi) - _log_choose(w + b, d)
    # P(X <= x) / P(X = x) = 1 + sum_j P(X = x - j) / P(X = x)
    ratio_sum = np.zeros(i.size)
    term = np.ones(i.size)
    xx = xi.copy()
    active = np.flatnonzero(xx > 0)
    while active.size:
        a = active
        term[a] *= xx[a] * (b[a] - d[a] + xx[a]) / (d[a] + 1 - xx[a]) / (w[a] + 1 - xx[a])
        ratio_sum[a] += term[a]
        xx[a] -= 1
        active = a[(xx[a] > 0) & (term[a] >= np.finfo(float).eps * ratio_sum[a])]
    log_cdf = log_density + np.log1p(ratio_sum)
    log_upper = np.where(log_cdf > -np.log(2), np.log(-np.expm1(log_cdf)), np.log1p(-np.exp(log_cdf)))
    out[i] = np.where(lower[i], log_cdf, log_upper)
    return out


def _fisher_log_pvals(rows, cols, counts, background, n_cells, n_guides) -> np.ndarray:
    """One-sided Fisher log p-value of every non-zero entry.

    The cell's own row is observed; the other cells enter through ``background``
    (the counts themselves in the first round, background-only estimates later).
    """
    cell_total = np.bincount(rows, counts, n_cells)
    background_cell = np.bincount(rows, background, n_cells)
    background_guide = np.bincount(cols, background, n_guides)
    guide_total = background_guide[cols] - background + counts
    total = background.sum() - background_cell[rows] + cell_total[rows]
    return log_phyper_upper(counts - 1, guide_total, total - guide_total, cell_total[rows])


def _guo_sarkar_log_cutoff(log_p, indptr, n_tests_per_cell: int, n_tests: int, n_cells: int, fdr: float) -> float:
    """Log p-value cutoff of the Guo-Sarkar procedure (fishash's default ``padj_method = "GS"``).

    Per cell, the smallest p-value is Bonferroni-corrected over the guides; BH
    over cells gives the number B of cells with any call; an entry is called
    when ``p <= B * fdr / (guides * cells)``.
    """
    has_entries = np.diff(indptr) > 0
    cell_min = np.zeros(n_cells)  # empty cells: p = 1
    cell_min[has_entries] = np.minimum.reduceat(log_p, indptr[:-1][has_entries])
    block_p = np.sort(np.minimum(np.exp(cell_min) * n_tests_per_cell, 1.0))
    passing = np.flatnonzero(block_p <= np.arange(1, n_cells + 1) * fdr / n_cells)
    n_blocks = passing[-1] + 1 if passing.size else 0
    return float(np.log(fdr) - np.log(n_tests) + np.log(n_blocks)) if n_blocks else -np.inf


def _impute_masked(rows, cols, counts, mask, n_cells, n_guides, eps: float = 1e-4, max_iter: int = 10) -> np.ndarray:
    """Background counts: masked entries replaced by a rank-1 Poisson fit of the unmasked ones.

    Alternates guide frequencies f (summing to 1) and cell sizes s, each the
    maximum-likelihood update given the other over the unmasked entries; a masked
    entry becomes ``f[g] * s[c]`` (fishash's ``impute_masked_counts``).
    """
    unmasked = np.where(mask, 0.0, counts)
    guide_sum = np.bincount(cols, unmasked, n_guides)
    cell_sum = np.bincount(rows, unmasked, n_cells)
    m_rows, m_cols = rows[mask], cols[mask]
    cell_size = np.ones(n_cells)
    prev_freq = prev_size = None
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(max_iter):
            freq = guide_sum / (cell_size.sum() - np.bincount(m_cols, cell_size[m_rows], n_guides))
            freq[guide_sum == 0] = 0.0
            freq = freq / freq.sum()
            cell_size = cell_sum / (1.0 - np.bincount(m_rows, freq[m_cols], n_cells))
            cell_size[cell_sum == 0] = 0.0
            if i > 0:
                guide_change = np.nanmax(np.abs(np.log(freq) - np.log(prev_freq)))
                size_change = np.nanmax(np.abs(np.log(cell_size) - np.log(prev_size)))
                if guide_change < eps and size_change < eps:
                    break
            prev_freq, prev_size = freq, cell_size
    background = unmasked.copy()
    background[mask] = freq[m_cols] * cell_size[m_rows]
    return background


def call_fishash(X: sparse.csr_matrix, fdr: float, min_count: int, refit: int) -> sparse.csr_matrix:
    """Cells x guides calls (CSR int8) by fishash at false-discovery rate ``fdr``."""
    X = sparse.csr_matrix(X, dtype=np.float64)
    X.sum_duplicates()
    X.eliminate_zeros()
    n_cells, n_guides = X.shape
    rows = np.repeat(np.arange(n_cells), np.diff(X.indptr))
    cols, counts = X.indices, X.data
    # Tests counted over guides and cells with any counts (fishash's exclude_empty = TRUE).
    n_tests_per_cell = int((np.bincount(cols, minlength=n_guides) > 0).sum())
    n_tests = n_tests_per_cell * int((np.diff(X.indptr) > 0).sum())
    mask = called = None
    for iteration in range(1, refit + 2):
        background = counts if iteration == 1 else _impute_masked(rows, cols, counts, mask, n_cells, n_guides)
        log_p = _fisher_log_pvals(rows, cols, counts, background, n_cells, n_guides)
        log_cutoff = _guo_sarkar_log_cutoff(log_p, X.indptr, n_tests_per_cell, n_tests, n_cells, fdr)
        previous, called = called, (log_p <= log_cutoff) & (counts >= min_count)
        mask = called if iteration <= 3 else (mask | called)
        if previous is not None and np.array_equal(previous, called):
            break
    calls = sparse.csr_matrix((called.astype(np.int8), cols.copy(), X.indptr.copy()), shape=X.shape)
    calls.eliminate_zeros()
    n_called = np.diff(calls.indptr)
    logger.info(
        "fishash (FDR %.3g, min_count %d): %d rounds, log p cutoff %.2f; %d calls; cells with 0 / 1 / 2+ calls: %d / %d / %d",
        fdr,
        min_count,
        iteration,
        log_cutoff,
        calls.nnz,
        int((n_called == 0).sum()),
        int((n_called == 1).sum()),
        int((n_called >= 2).sum()),
    )
    return calls


# ---------------------------------------------------------------------------
# Poisson-Gaussian mixture (crispat ``ga_poisson_gauss``, after Replogle et al. 2022)
# ---------------------------------------------------------------------------
#
# Per guide, the log2 of its non-zero UMI counts is a two-component mixture: a
# "continuous Poisson" lam^y e^-lam / Gamma(y + 1) for background and a Normal
# for cells that carry the guide, with weights ~ Dirichlet(0.9, 0.1),
# mu ~ N(3, 2), sigma ~ LogNormal(2, 1), lam ~ LogNormal(0, 1). The guide's cutoff
# is the smallest whole count t with P(Normal | log2 t) > 0.5. Counts are raw: the
# model does not adjust for a cell's sequencing depth.
#
# This is crispat's fitting procedure (``fit_PGMM``: pyro AutoDelta, Adam) written
# in numpy. The fit stops after at most 500 steps, before it converges, and that
# is part of the method: the posterior has no finite maximum (the Dirichlet(0.9,
# 0.1) density grows without bound as the Normal weight goes to 0), so the result
# depends on where the fit starts and how far it gets. crispat's ten candidate
# starts come from its seeds 0-9 and do not depend on the data, so they are copied
# below; from the same start the numpy and pyro fits reach the same parameters
# (to ~1e-4; pyro computes in float32).
#
# The score is a sum over cells of a term that depends only on the cell's count,
# so it is computed over the distinct counts weighted by how many cells have each.

_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
_LOG_BETA_PRIOR_CONST = gammaln(1.0) - gammaln(0.9) - gammaln(0.1)


def _pgmm_loss_and_grad(theta: np.ndarray, y: np.ndarray, n: np.ndarray) -> Tuple[float, np.ndarray]:
    """crispat's loss, -log p(data, params), and its gradient.

    ``theta`` holds pyro's AutoDelta parameters (a, b, mu, log sigma, log lam): the
    weights (Poisson, Normal) = softmax(a, b) (pyro's SoftmaxTransform, two free
    logits). ``y`` are the distinct log2 counts and ``n`` how many cells have each.
    """
    a, b, mu, log_sigma, log_lam = theta
    w0 = 1.0 / (1.0 + np.exp(b - a))
    w1 = 1.0 / (1.0 + np.exp(a - b))
    sigma, lam = np.exp(log_sigma), np.exp(log_lam)
    z = (y - mu) / sigma
    log_background = np.log(w0) + y * log_lam - lam - gammaln(y + 1)
    log_signal = np.log(w1) - 0.5 * z**2 - log_sigma - _LOG_SQRT_2PI
    log_mix = np.logaddexp(log_background, log_signal)
    signal_share = np.exp(log_signal - log_mix)  # P(Normal | y)
    log_prior = (
        _LOG_BETA_PRIOR_CONST
        - 0.1 * np.log(w0)
        - 0.9 * np.log(w1)
        - 0.5 * ((mu - 3.0) / 2.0) ** 2
        - np.log(2.0)
        - 0.5 * (log_sigma - 2.0) ** 2
        - log_sigma  # LogNormal(2, 1) density of sigma
        - 0.5 * log_lam**2
        - log_lam  # LogNormal(0, 1) density of lam
        - 3 * _LOG_SQRT_2PI
    )
    n_signal = np.dot(n, signal_share)
    n_background = n.sum() - n_signal
    # the weights depend on a - b only, so the two logit gradients are opposite
    grad_logit = (n_background - 0.1) * w1 - (n_signal - 0.9) * w0
    grad = np.array(
        [
            grad_logit,
            -grad_logit,
            np.dot(n * signal_share, z / sigma) - (mu - 3.0) / 4.0,
            np.dot(n * signal_share, z**2 - 1.0) - (log_sigma - 2.0) - 1.0,
            np.dot(n * (1 - signal_share), y - lam) - log_lam - 1.0,
        ]
    )
    return -(log_prior + np.dot(n, log_mix)), -grad


#: crispat's candidate starts (mu, sigma, lam) for seeds 0-9: pyro's AutoDelta
#: ``init_to_median`` (median of 15 prior draws) under ``pyro.set_rng_seed(seed)``,
#: read from pyro 1.9.1 / torch 2.11. The weights start at the simplex centre (0.5,
#: 0.5), as pyro takes no median for a multivariate prior.
_CRISPAT_STARTS = np.array(
    [
        [2.193312883377075, 8.953058242797852, 0.9603849649429321],
        [2.157609701156616, 8.250345230102539, 1.266902208328247],
        [2.7943902015686035, 6.372674942016602, 0.8600091338157654],
        [3.83843731880188, 8.601312637329102, 0.723555326461792],
        [3.3220925331115723, 10.072628021240234, 1.6703166961669922],
        [1.8844131231307983, 9.130050659179688, 0.7716161012649536],
        [2.6687490940093994, 4.720515251159668, 0.48086419701576233],
        [3.6353566646575928, 4.8428263664245605, 1.1132030487060547],
        [3.3540127277374268, 6.9748759269714355, 0.6111212968826294],
        [3.04290771484375, 10.252533912658691, 0.8875091671943665],
    ]
)


def _fit_pgmm(y: np.ndarray, n: np.ndarray) -> np.ndarray:
    """(w_normal, mu, sigma, lam) of one guide by crispat's procedure.

    The start with the lowest loss among crispat's ten is improved by Adam (lr 0.01,
    betas (0.8, 0.99)) for at most 500 steps, stopping once the loss has not dropped
    by 0.001 for 50 steps.
    """
    starts = [np.array([np.log(0.5), np.log(0.5), mu, np.log(sigma), np.log(lam)]) for mu, sigma, lam in _CRISPAT_STARTS]
    theta = min(starts, key=lambda t: _pgmm_loss_and_grad(t, y, n)[0])
    lr, beta1, beta2, eps = 0.01, 0.8, 0.99, 1e-8
    m = np.zeros(5)
    v = np.zeros(5)
    min_loss, last_step = 1e8, 0
    for step in range(500):
        loss, grad = _pgmm_loss_and_grad(theta, y, n)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        theta = theta - lr * (m / (1 - beta1 ** (step + 1))) / (np.sqrt(v / (1 - beta2 ** (step + 1))) + eps)
        if loss < min_loss - 0.001:
            min_loss, last_step = loss, step
        if step - last_step > 50:
            break
    a, b, mu, log_sigma, log_lam = theta
    return np.array([1.0 / (1.0 + np.exp(a - b)), mu, np.exp(log_sigma), np.exp(log_lam)])


def _pgmm_threshold(params: np.ndarray, max_count: int) -> float:
    """Smallest whole count t in 1..max_count with P(Normal | log2 t) > 0.5; inf when there is none."""
    w1, mu, sigma, lam = params
    t = np.arange(1, max_count + 1, dtype=np.float64)
    y = np.log2(t)
    log_signal = np.log(w1) - 0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma) - _LOG_SQRT_2PI
    log_background = np.log1p(-w1) + y * np.log(lam) - lam - gammaln(y + 1)
    above = np.flatnonzero(log_signal > log_background)
    return float(t[above[0]]) if above.size else np.inf


def call_poisson_gauss(X: sparse.csr_matrix, min_count: int, min_cells: int) -> sparse.csr_matrix:
    """Cells x guides calls (CSR int8) by a Poisson-Gaussian mixture fitted to each guide.

    A guide is fitted when it is non-zero in at least ``max(min_cells, 2)`` cells and
    some cell has >= 2 UMIs (crispat's own minimum); a cell is called when its count
    reaches both the guide's cutoff and ``min_count``.
    """
    X = sparse.csc_matrix(X, dtype=np.float64)
    X.sum_duplicates()
    X.eliminate_zeros()
    n_guides = X.shape[1]
    keep = np.zeros(X.nnz, dtype=bool)
    n_fitted = 0
    for g in range(n_guides):
        lo, hi = X.indptr[g], X.indptr[g + 1]
        counts = X.data[lo:hi]
        if counts.size < max(min_cells, 2) or counts.max() < 2:
            continue
        values, n_cells = np.unique(counts, return_counts=True)
        params = _fit_pgmm(np.log2(values), n_cells.astype(np.float64))
        keep[lo:hi] = (counts >= _pgmm_threshold(params, int(values[-1]))) & (counts >= min_count)
        n_fitted += 1
    calls = sparse.csc_matrix((keep.astype(np.int8), X.indices.copy(), X.indptr.copy()), shape=X.shape).tocsr()
    calls.eliminate_zeros()
    n_called = np.diff(calls.indptr)
    logger.info(
        "Poisson-Gaussian mixture (min_count %d, min_cells %d): %d of %d guides fitted; %d calls; "
        "cells with 0 / 1 / 2+ calls: %d / %d / %d",
        min_count,
        min_cells,
        n_fitted,
        n_guides,
        calls.nnz,
        int((n_called == 0).sum()),
        int((n_called == 1).sum()),
        int((n_called >= 2).sum()),
    )
    return calls
