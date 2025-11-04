from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Utilities ------------------------------------

def gamma_from_mean_sd(mean: float, sd: float):
    if mean <= 0 or sd <= 0:
        raise ValueError("mean and sd must be positive")
    alpha = (mean / sd) ** 2
    beta = alpha / mean
    return alpha, beta


def simulate_data(n_cells: int, t: float, mu_lambda: float, sd_lambda: float, rng: np.random.Generator):
    a, b = gamma_from_mean_sd(mu_lambda, sd_lambda)
    lam = rng.gamma(shape=a, scale=1.0 / b, size=n_cells)
    y = rng.poisson(lam * t)
    return lam, y


# ------------------- Numerical marginal via Gauss–Laguerre -------------------
@dataclass
class LagGaussCache:
    n_quad: int
    xk: np.ndarray
    wk: np.ndarray


def laggauss_nodes_weights(n_quad: int) -> LagGaussCache:
    from numpy.polynomial.laguerre import laggauss
    xk, wk = laggauss(n_quad)
    return LagGaussCache(n_quad=n_quad, xk=xk.astype("float64"), wk=wk.astype("float64"))


def log_marginal_p_y_numpy(y: np.ndarray, alpha: float, beta: float, t: float, cache: LagGaussCache) -> np.ndarray:
    """Vectorised numerical log pmf using Gauss–Laguerre quadrature.
    Returns array of shape y.shape with log-probabilities.
    """
    y = np.asarray(y, dtype=int)
    if np.any(y < 0):
        raise ValueError("y must be non-negative integers")
    if not (alpha > 0 and beta > 0 and t > 0):
        return np.full_like(y, -np.inf, dtype=float)

    xk, wk = cache.xk, cache.wk
    # log-constant part: beta^alpha * t^y / (Gamma(alpha) * y! * (beta + t)^{alpha + y})
    log_const = (
        alpha * math.log(beta)
        - math.lgamma(alpha)
        - (alpha + y) * np.log(beta + t)
    ) + y * np.log(t) - np.array([math.lgamma(int(k) + 1) for k in y], dtype=float)

    power = alpha + y - 1.0
    # log-sum-exp over quadrature terms: sum w_k * x_k^{power}
    log_terms = np.log(wk)[None, :] + power[:, None] * np.log(xk)[None, :]
    m = np.max(log_terms, axis=1)
    integral_log = m + np.log(np.sum(np.exp(log_terms - m[:, None]), axis=1))

    return log_const + integral_log


def dataset_loglik(y: np.ndarray, alpha: float, beta: float, t: float, cache: LagGaussCache) -> float:
    return float(np.sum(log_marginal_p_y_numpy(y, alpha, beta, t, cache)))


# ----------------------------- Priors ---------------------------------------

def logprior_lognormal(mu: float, loc: float, scale: float) -> float:
    """LogNormal prior on a positive scalar 'mu'. loc,scale act on log(mu).
    log p(mu) = -0.5*((log mu - log loc)/scale)^2 - log(mu*scale*sqrt(2*pi))
    """
    if mu <= 0 or loc <= 0 or scale <= 0:
        return -np.inf
    z = (math.log(mu) - math.log(loc)) / scale
    return -0.5 * z * z - math.log(mu * scale * math.sqrt(2.0 * math.pi))


# --------------------------- Grid Posterior ----------------------------------

def make_grid(bounds: tuple[float, float], n: int) -> np.ndarray:
    lo, hi = bounds
    return np.linspace(lo, hi, n)


def posterior_on_grid(y: np.ndarray, t: float,
                      mu_bounds: tuple[float, float], sd_bounds: tuple[float, float],
                      n_mu: int, n_sd: int,
                      prior_loc_mu: float, prior_scale_mu: float,
                      prior_loc_sd: float, prior_scale_sd: float,
                      n_quad: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute unnormalised and normalised posterior over a (μ,σ) grid.
    Returns: MU, SD, logpost (n_mu x n_sd), post (normalised)
    """
    MU = make_grid(mu_bounds, n_mu)
    SD = make_grid(sd_bounds, n_sd)
    cache = laggauss_nodes_weights(n_quad)

    logpost = np.empty((n_mu, n_sd), dtype=float)
    for i, mu in enumerate(MU):
        for j, sd in enumerate(SD):
            if mu <= 0 or sd <= 0:
                logpost[i, j] = -np.inf
                continue
            alpha, beta = gamma_from_mean_sd(mu, sd)
            ll = dataset_loglik(y, alpha, beta, t, cache)
            lp = logprior_lognormal(mu, prior_loc_mu, prior_scale_mu) + \
                 logprior_lognormal(sd, prior_loc_sd, prior_scale_sd)
            logpost[i, j] = ll + lp

    # normalise in a stable way
    m = np.max(logpost)
    post = np.exp(logpost - m)
    Z = np.sum(post)
    post /= Z
    # return also MU, SD as meshgrids for plotting
    MUg, SDg = np.meshgrid(SD, MU)  # shape (n_mu, n_sd) to align with logpost
    return MUg, SDg, logpost, post


def posterior_samples_from_grid(MUg: np.ndarray, SDg: np.ndarray, post: np.ndarray, n: int, rng: np.random.Generator):
    """Draw samples (mu, sd) from a discrete grid posterior."""
    flat_p = post.ravel()
    flat_p = flat_p / flat_p.sum()
    idx = rng.choice(flat_p.size, size=n, p=flat_p)
    mu_samp = MUg.ravel()[idx]
    sd_samp = SDg.ravel()[idx]
    return mu_samp, sd_samp


# ----------------------------- Visuals ---------------------------------------

def plot_posterior_heatmap(MUg: np.ndarray, SDg: np.ndarray, post: np.ndarray, mu_true=None, sd_true=None):
    plt.figure()
    plt.imshow(post, origin='lower', aspect='auto',
               extent=[SDg.min(), SDg.max(), MUg.min(), MUg.max()])
    plt.xlabel('σ_λ'); plt.ylabel('μ_λ'); plt.title('Posterior over (μ_λ, σ_λ)')
    if mu_true is not None and sd_true is not None:
        plt.plot([sd_true], [mu_true], marker='x')
    plt.colorbar(label='posterior density (normalised)')
    plt.tight_layout(); plt.show()


def plot_marginals(mu_samp: np.ndarray, sd_samp: np.ndarray, mu_true=None, sd_true=None):
    plt.figure(); plt.hist(mu_samp, bins=40, density=True, alpha=0.7)
    if mu_true is not None:
        plt.axvline(mu_true)
    plt.xlabel('μ_λ'); plt.ylabel('density'); plt.title('Marginal posterior of μ_λ');
    plt.tight_layout(); plt.show()

    plt.figure(); plt.hist(sd_samp, bins=40, density=True, alpha=0.7)
    if sd_true is not None:
        plt.axvline(sd_true)
    plt.xlabel('σ_λ'); plt.ylabel('density'); plt.title('Marginal posterior of σ_λ');
    plt.tight_layout(); plt.show()


def posterior_predictive(y_len: int, t: float, mu_samp: np.ndarray, sd_samp: np.ndarray, draws: int, seed=None):
    rng = np.random.default_rng(seed)
    draws = min(draws, mu_samp.size)
    idx = rng.integers(0, mu_samp.size, size=draws)
    y_rep = np.empty((draws, y_len), dtype=int)
    for d, i in enumerate(idx):
        a, b = gamma_from_mean_sd(mu_samp[i], sd_samp[i])
        lam = rng.gamma(shape=a, scale=1.0 / b, size=y_len)
        y_rep[d, :] = rng.poisson(lam * t)
    return y_rep


def plot_ppc(y: np.ndarray, y_rep: np.ndarray):
    plt.figure()
    bins = min(40, max(10, int(np.sqrt(y.size))))
    plt.hist(y, bins=bins, density=True, alpha=0.5, label='observed')
    plt.hist(y_rep.reshape(-1), bins=bins, density=True, alpha=0.5, label='replicated')
    plt.xlabel('kills per cell over t'); plt.ylabel('density'); plt.legend()
    plt.title('Posterior Predictive: observed vs replicated')
    plt.tight_layout(); plt.show()

    q_obs = np.quantile(y, [0.05, 0.5, 0.95])
    q_rep = np.quantile(y_rep, [0.05, 0.5, 0.95], axis=1)
    print('Observed y quantiles (5%,50%,95%):', q_obs)
    print('Replicated y quantiles (means across draws):', q_rep.mean(axis=1))


# ----------------------------- Tests -----------------------------------------

def run_tests():
    rng = np.random.default_rng(0)
    # Test 1: moderate n, signal present
    for (mu_true, sd_true, n_cells) in [(0.8, 0.5, 200), (0.4, 0.25, 400)]:
        t = 8.0
        _, y = simulate_data(n_cells, t, mu_true, sd_true, rng)
        emp = max(1e-6, y.mean()/t)
        mu_bounds = (emp/5, emp*5)
        sd_bounds = (emp/10, emp*3)
        MUg, SDg, _, post = posterior_on_grid(
            y, t, mu_bounds, sd_bounds,
            n_mu=120, n_sd=120,
            prior_loc_mu=1.0, prior_scale_mu=1.0,
            prior_loc_sd=1.0, prior_scale_sd=1.0,
            n_quad=64,
        )
        mu_s, sd_s = posterior_samples_from_grid(MUg, SDg, post, n=5000, rng=rng)
        mu_hat = np.median(mu_s)
        sd_hat = np.median(sd_s)
        # Loose, but should pass on average
        assert abs(mu_hat - mu_true) < max(0.25, 0.5*mu_true), f"mu off: {mu_hat} vs {mu_true}"
        assert abs(sd_hat - sd_true) < max(0.25, 0.6*sd_true), f"sd off: {sd_hat} vs {sd_true}"
    print('All tests passed.')


# ----------------------------- Main -----------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_cells', type=int, default=200)
    p.add_argument('--t', type=float, default=8.0)
    p.add_argument('--mu', type=float, default=0.8)
    p.add_argument('--sd', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--nquad', type=int, default=64, help='Gauss–Laguerre nodes (e.g., 32,64,128)')
    p.add_argument('--grid_mu', type=int, default=150, help='Grid points for μ')
    p.add_argument('--grid_sd', type=int, default=150, help='Grid points for σ')
    p.add_argument('--prior_loc_mu', type=float, default=1.0)
    p.add_argument('--prior_scale_mu', type=float, default=1.0)
    p.add_argument('--prior_loc_sd', type=float, default=1.0)
    p.add_argument('--prior_scale_sd', type=float, default=1.0)
    p.add_argument('--draws', type=int, default=10000, help='Posterior samples from grid')
    p.add_argument('--no_plots', action='store_true', help='Disable plotting (useful for CI)')
    p.add_argument('--test', action='store_true', help='Run built-in tests and exit')
    args = p.parse_args()

    if args.test:
        run_tests(); return

    rng = np.random.default_rng(args.seed)

    # Simulate synthetic data
    _, y = simulate_data(args.n_cells, args.t, args.mu, args.sd, rng)
    print(f"Simulated n={args.n_cells}, t={args.t}, true mu={args.mu}, sd={args.sd}")
    print(f"Counts summary: mean={y.mean():.3f}, sd={y.std(ddof=1):.3f}, min={y.min()}, max={y.max()}")

    # Posterior grid bounds guided by empirical rate
    emp_rate = max(1e-6, y.mean() / args.t)
    mu_bounds = (emp_rate/5, emp_rate*5)
    sd_bounds = (emp_rate/10, emp_rate*3)
    print(f"Grid bounds: μ in {mu_bounds}, σ in {sd_bounds}")

    MUg, SDg, logpost, post = posterior_on_grid(
        y, args.t,
        mu_bounds, sd_bounds,
        n_mu=args.grid_mu, n_sd=args.grid_sd,
        prior_loc_mu=args.prior_loc_mu, prior_scale_mu=args.prior_scale_mu,
        prior_loc_sd=args.prior_loc_sd, prior_scale_sd=args.prior_scale_sd,
        n_quad=args.nquad,
    )

    # Draw posterior samples (mu, sd) from the discrete grid posterior
    mu_samp, sd_samp = posterior_samples_from_grid(MUg, SDg, post, n=args.draws, rng=rng)
    print(f"Posterior medians: mu={np.median(mu_samp):.3f}, sd={np.median(sd_samp):.3f}")

    if not args.no_plots:
        plot_posterior_heatmap(MUg, SDg, post, mu_true=args.mu, sd_true=args.sd)
        plot_marginals(mu_samp, sd_samp, mu_true=args.mu, sd_true=args.sd)
        y_rep = posterior_predictive(y_len=y.size, t=args.t, mu_samp=mu_samp, sd_samp=sd_samp, draws=2000, seed=args.seed)
        plot_ppc(y, y_rep)


if __name__ == '__main__':
    main()
