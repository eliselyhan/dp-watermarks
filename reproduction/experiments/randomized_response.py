"""Simulate the pure-RR baseline in memory for the combined C4 overlay figures.

P(report=1 | input=0)=0.5; P(report=1 | input=1)=sigmoid(epsilon).
MAPE is relative to the realized m/n. No standalone artifacts are written.
"""
from core.sampling import figure_rng
import numpy as np


def mape(n, eps, prevalence, rng, repeats=50_000):
    # Population construction: fix m positive and n-m negative inputs.
    # This baseline simulates reports directly, so no completion pool is built.
    m = int(round(n * prevalence))
    target = m / n
    if target <= 0 or eps <= 0:
        raise ValueError('MAPE requires positive realized prevalence and epsilon.')
    pi0 = 0.5
    pi1 = 1 / (1 + np.exp(-eps))
    gap = 0.5 * np.tanh(eps / 2)
    # Randomized samples: independently simulate positive report counts
    # from the two populations through their respective reporting channels.
    ones = rng.binomial(m, pi1, size=repeats) + rng.binomial(n - m, pi0, size=repeats)
    # Estimator: invert the observed positive-report fraction; gap equals
    # pi1 - pi0. Clipping is applied before computing MAPE.
    estimate = (ones / n - pi0) / gap
    estimate = np.clip(estimate, 0, 1)
    errors = 100 * np.abs(estimate - target) / target
    return float(errors.mean()), float(errors.std(ddof=1) / np.sqrt(repeats)), *map(float, np.quantile(errors, [.05, .95]))



def curve(name, grid, seed, repeats=1000, fixed_n=2000, fixed_epsilon=2):
    """Return one overlay's seeded baseline statistics without intermediate files."""
    if name not in ('epsilon', 'n'):
        raise ValueError(f'Unknown overlay sweep: {name}')
    rows = []
    for prevalence in [0.1, 0.5, 0.9]:
        for x in grid:
            n = int(x) if name == 'n' else fixed_n
            eps = float(x) if name == 'epsilon' else fixed_epsilon
            rng = figure_rng(seed, 0, 1, n, round(eps*1000), round(prevalence*1000))
            mean, se, p05, p95 = mape(n, eps, prevalence, rng, repeats)
            rows.append(dict(prevalence=prevalence, target=round(n * prevalence) / n,
                             n=n, epsilon=eps, mape=mean, mc_standard_error=se,
                             mape_p05=p05, mape_p95=p95))
    return rows
