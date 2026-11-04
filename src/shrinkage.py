"""Empirical-Bayes partial pooling for player rates.

All three models give the same plain-language form:
    shrunk = (n * observed + k * prior) / (n + k)
where n is the player's exposure (possessions, chances or attempts) and k (the prior's
weight, in the same units) is estimated from the data by maximizing the marginal likelihood
across players in the group (league x season x position). No minutes cutoff is used.

- Counting stats per exposure: gamma-Poisson (= negative binomial with a log(exposure) offset
  and a gamma player random effect). Prior rate ~ Gamma(a, b); k = b, prior = a / b.
- Proportions (shooting %, 3PA rate, assisted share): beta-binomial. Prior ~ Beta(a, b);
  k = a + b, prior = a / (a + b).
- Shot-zone mix (5 shares summing to 1): Dirichlet-multinomial with the pooled zone mix
  as prior mean and one precision k.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gammaln

LOG_BOUNDS = (-8.0, 14.0)


@dataclass
class Prior:
    mean: float      # prior rate / proportion
    k: float         # prior weight in exposure units
    n_players: int
    at_bound: bool   # True if the fit hit a bound (little or no overdispersion detected)


def fit_gamma_poisson(y: np.ndarray, e: np.ndarray) -> Prior:
    y, e = np.asarray(y, float), np.asarray(e, float)
    m = e > 0
    y, e = y[m], e[m]
    mean0 = max(y.sum() / e.sum(), 1e-9)

    def nll(t):
        a, b = np.exp(t)
        return -(gammaln(y + a) - gammaln(a) + a * np.log(b / (b + e)) + y * np.log(e / (b + e))).sum()

    a0 = 2.0
    res = minimize(nll, np.log([a0, a0 / mean0]), method="L-BFGS-B", bounds=[LOG_BOUNDS, LOG_BOUNDS])
    a, b = np.exp(res.x)
    at_bound = bool(np.any(np.isclose(res.x, LOG_BOUNDS, atol=1e-3)))
    return Prior(mean=a / b, k=b, n_players=int(m.sum()), at_bound=at_bound)


def fit_beta_binomial(k: np.ndarray, n: np.ndarray) -> Prior:
    k, n = np.asarray(k, float), np.asarray(n, float)
    m = n > 0
    k, n = k[m], n[m]
    p0 = np.clip(k.sum() / n.sum(), 1e-4, 1 - 1e-4)

    def nll(t):
        a, b = np.exp(t)
        return -(betaln(k + a, n - k + b) - betaln(a, b)).sum()

    res = minimize(nll, np.log([p0 * 20, (1 - p0) * 20]), method="L-BFGS-B", bounds=[LOG_BOUNDS, LOG_BOUNDS])
    a, b = np.exp(res.x)
    at_bound = bool(np.any(np.isclose(res.x, LOG_BOUNDS, atol=1e-3)))
    return Prior(mean=a / (a + b), k=a + b, n_players=int(m.sum()), at_bound=at_bound)


def fit_dirichlet_multinomial(counts: np.ndarray) -> tuple[np.ndarray, Prior]:
    c = np.asarray(counts, float)
    c = c[c.sum(1) > 0]
    p = c.sum(0) / c.sum()
    N = c.sum(1)

    def nll(log_s):
        s = np.exp(log_s)
        return -(gammaln(s) - gammaln(N + s) + (gammaln(c + s * p) - gammaln(s * p)).sum(1)).sum()

    res = minimize_scalar(nll, bounds=LOG_BOUNDS, method="bounded")
    s = float(np.exp(res.x))
    at_bound = bool(np.isclose(res.x, LOG_BOUNDS, atol=1e-2).any())
    return p, Prior(mean=float("nan"), k=s, n_players=len(c), at_bound=at_bound)


def shrink(obs_num: np.ndarray, exposure: np.ndarray, prior: Prior) -> np.ndarray:
    """Posterior mean: (num + k*prior) / (exposure + k). Works for both rate and proportion models."""
    return (np.asarray(obs_num, float) + prior.k * prior.mean) / (np.asarray(exposure, float) + prior.k)


def shrink_mix(counts: np.ndarray, p: np.ndarray, prior: Prior) -> np.ndarray:
    c = np.asarray(counts, float)
    return (c + prior.k * p) / (c.sum(1, keepdims=True) + prior.k)


def within_player_dispersion(y: np.ndarray, e: np.ndarray, player: np.ndarray) -> float:
    """Quasi-Poisson dispersion phi from game-level data: Pearson chi-square of game counts
    around each player's own season rate, divided by residual degrees of freedom.
    phi > 1 means counts are lumpier than Poisson (e.g. points arrive in 2s and 3s); the
    gamma-Poisson prior is then fit on y/phi and e/phi so that within-player noise is not
    mistaken for between-player differences."""
    df = pd.DataFrame({"y": y, "e": e, "p": player})
    df = df[df.e > 0]
    tot = df.groupby("p")[["y", "e"]].transform("sum")
    rate = tot.y / tot.e
    mu = rate * df.e
    ok = mu > 0
    chi2 = ((df.y[ok] - mu[ok]) ** 2 / mu[ok]).sum()
    dof = ok.sum() - df.p[ok].nunique()
    return float(max(chi2 / max(dof, 1), 1.0))
