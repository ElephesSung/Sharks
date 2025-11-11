#!/usr/bin/env python3
# simulate_inference.py
"""
Run synthetic NK-cell kill simulations and Bayesian inference.
Usage:
    python simulate_inference.py --n_cells 5000 --n_quad 32 --mode numerical
"""
import matplotlib.gridspec as gridspec
import argparse
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from numpy.polynomial.laguerre import laggauss
from tqdm.autonotebook import tqdm
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "legend.fontsize": 12,
    "legend.title_fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 18,
})
mpl.rcParams['animation.embed_limit'] = 2000


# ---------------------------------------------------------------------
# Simulation Class
# ---------------------------------------------------------------------
class insilico_experiment:
    def __init__(
        self,
        kill_lambda_avg: float = 3.0,
        kill_lambda_sd: float = 1.0,
        Num_killers_obs: int = 300,
        random_seed: int = 66
    ):
        self.kill_lambda_avg = kill_lambda_avg
        self.kill_lambda_sd = kill_lambda_sd
        self.Num_killers_obs = Num_killers_obs
        self.random_seed = random_seed
        self.lambdas, self.kills = self._generate_data()

    def _generate_data(self):
        rng = np.random.default_rng(self.random_seed)
        shape = self.kill_lambda_avg**2 / self.kill_lambda_sd**2
        scale = self.kill_lambda_sd**2 / self.kill_lambda_avg
        lambdas = rng.gamma(shape, scale, self.Num_killers_obs)
        kills = rng.poisson(lambdas)
        return lambdas, kills

    def regenerate(self, new_seed: int = None):
        if new_seed is not None:
            self.random_seed = new_seed
        self.lambdas, self.kills = self._generate_data()

    def get_summary(self):
        return {
            "kill_lambda_avg": self.kill_lambda_avg,
            "kill_lambda_sd": self.kill_lambda_sd,
            "lambdas": self.lambdas,
            "kills": self.kills,
            "killer_number": self.Num_killers_obs
        }


# ---------------------------------------------------------------------
# Inference Functions
# ---------------------------------------------------------------------
def inference_one_num(kills_per_cell, draws=3000, tune=3000, chains=4, seed=66, n_quad=32):
    y = np.asarray(kills_per_cell, dtype=int)
    x_nodes, w_weights = laggauss(n_quad)
    x_nodes = x_nodes.astype("float64")
    w_weights = w_weights.astype("float64")

    def logp_num(y_obs, mu_lambda, sigma_lambda):
        alpha = (mu_lambda / sigma_lambda) ** 2
        beta = mu_lambda / (sigma_lambda**2)
        y_obs = pt.as_tensor_variable(y_obs)
        x_t = pt.as_tensor_variable(x_nodes)
        w_t = pt.as_tensor_variable(w_weights)
        y_broadcast = y_obs[:, None]
        power = alpha + y_broadcast - 1.0
        terms = w_t * (x_t / (beta + 1.0)) ** power
        integrals = pt.sum(terms, axis=1)
        log_p = (
            alpha * pt.log(beta)
            - pt.gammaln(alpha)
            - pt.gammaln(y_obs + 1)
            - (alpha + y_obs) * pt.log(beta + 1.0)
            + pt.log(integrals)
        )
        return pt.sum(log_p)

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=-1.0, upper=1.0)
        mu_lambda = pm.Deterministic("mu_lambda", 10.0**eta)
        sigma_lambda = pm.HalfNormal("sigma_lambda", sigma=1.0)
        pm.CustomDist("kills_num", mu_lambda, sigma_lambda, logp=logp_num, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.95, random_seed=seed, progressbar=True)
    return trace


def inference_one_ana(kills_per_cell, draws=3000, tune=3000, chains=4, seed=None):
    N = np.asarray(kills_per_cell, dtype=int)
    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=-1.0, upper=1.0)
        mu_lambda = pm.Deterministic("mu_lambda", 10.0 ** eta)
        sigma_lambda = pm.HalfNormal("sigma_lambda", sigma=1.0)
        alpha = pm.Deterministic("alpha", (mu_lambda / sigma_lambda) ** 2)
        pm.NegativeBinomial("kills", mu=mu_lambda, alpha=alpha, observed=N)
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.95, random_seed=seed)
    return trace


def inference_all(experiment, labels=None,
                  draws=3000, tune=3000, chains=4, seed=None,
                  mode="analytical", n_quad=32):
    if labels is None:
        labels = [f"cond_{i}" for i in range(len(experiment))]
    out = []
    for i in tqdm(range(len(experiment))):
        current_seed = seed + i if seed is not None else None
        if mode == "analytical":
            idata = inference_one_ana(
                kills_per_cell=experiment[i].get_summary()['kills'],
                draws=draws, tune=tune, chains=chains, seed=current_seed)
        elif mode == "numerical":
            idata = inference_one_num(
                kills_per_cell=experiment[i].get_summary()['kills'],
                draws=draws, tune=tune, chains=chains, seed=current_seed, n_quad=n_quad)
        else:
            raise ValueError("Mode must be 'analytical' or 'numerical'")
        out.append((labels[i], idata))
    return out


# ---------------------------------------------------------------------
# Plotting Function
# ---------------------------------------------------------------------
def plot_joint_posteriors(
    idatas,
    ground_truth=None,
    parameters=None,
    hdi_prob=0.95,
    sample_size=6000,
    save_pdf=False,
    diagonal_style = "hist", # "kde" or "hist",
    marginal_style = "circle", # "pixel" or "circle",
    pdf_path="joint_posteriors.pdf",
    cmap_name="YlGnBu",
    point_size=3,
    font_scale=0.7,
):
    sns.set_context("talk", font_scale=font_scale)
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.9, len(idatas)))
    label_to_df = {}
    for label, idata in idatas:
        posterior = idata.posterior
        if parameters is None:
            parameters = [v for v in posterior.data_vars if posterior[v].ndim <= 2]
        df = pd.DataFrame()
        for p in parameters:
            vals = posterior[p].stack(sample=("chain", "draw")).values.ravel()
            if len(vals) > sample_size:
                vals = np.random.choice(vals, sample_size, replace=False)
            df[p] = vals
        df["label"] = label
        label_to_df[label] = df
    npar = len(parameters)
    fig = plt.figure(figsize=(6 * npar, 6 * npar), dpi = 250)
    gs = gridspec.GridSpec(npar, npar, wspace=0.15, hspace=0.15)
    gaxes = np.empty((npar, npar), dtype=object)
    for irow, rowpar in enumerate(parameters):
        for icol, colpar in enumerate(parameters):
            ax = plt.subplot(gs[irow, icol])
            gaxes[irow, icol] = ax
            if icol > irow:
                ax.axis("off")
                continue
            for color, (label, df) in zip(colors, label_to_df.items()):
                if icol == irow:
                    if diagonal_style == "kde":
                        sns.kdeplot(
                            df[rowpar], ax=ax, fill=True, color=color,
                            alpha=0.2, linewidth=1.5, label=label if irow == 0 else None,
                        )
                    elif diagonal_style == "hist":
                        vals = df[rowpar].dropna().values
                        sns.histplot(
                            vals, bins=30, stat="density", kde=False,
                            ax=ax, color=color, alpha=0.18,
                            element="step", fill=True,
                        )
                        sns.histplot(
                            vals, bins=30, stat="density", kde=False,
                            ax=ax, color=color, alpha=1.0,
                            element="step", fill=False,
                            linewidth=1.8, label=label if irow == 0 else None,
                        )
                        # lo, hi = az.hdi(vals, hdi_prob=hdi_prob)
                        # ax.axvspan(lo, hi, color=color, alpha=0.08, linewidth=0)
                        # ax.axvline(lo, color=color, linestyle="--", linewidth=1)
                        # ax.axvline(hi, color=color, linestyle="--", linewidth=1)
                        # mean = np.mean(vals)
                        # ax.axvline(mean, color=color, linestyle="-", linewidth=1, alpha=0.9)
                    if ground_truth is not None and label in ground_truth and rowpar in ground_truth[label]:
                        ax.axvline(ground_truth[label][rowpar], color=color, linestyle='-', linewidth=2)
                
                else:
                    if marginal_style == "circle":
                        sns.kdeplot(
                            x=df[colpar], y=df[rowpar], ax=ax, fill=False, color=color, alpha=0.4, levels=7,
                        )
                    if marginal_style == "pixel":
                        sns.histplot(
                            x=df[colpar], y=df[rowpar], bins=10, pthresh=0.01, cmap=cmap_name, cbar=False, ax=ax,
                        )
                    if ground_truth and label in ground_truth:
                        if colpar in ground_truth[label] and rowpar in ground_truth[label]:
                            ax.plot(ground_truth[label][colpar], ground_truth[label][rowpar],
                                    marker='*', color=color, markersize=12, markeredgewidth=2.5, linestyle='None')
            if icol == irow:
                ax.set_xlabel(rowpar)
                ax.set_ylabel(rowpar)
            else:
                if irow == npar - 1:
                    ax.set_xlabel(colpar)
                else:
                    ax.set_xticklabels([])
                if icol == 0 and irow != 0:
                    ax.set_ylabel(rowpar)
                else:
                    ax.set_yticklabels([])
    handles, labels = gaxes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    plt.tight_layout()
    if save_pdf:
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved joint posterior plot: {pdf_path}")
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    NO_CELL = args.n_cells
    n_quad = args.n_quad
    mode = args.mode.lower()

    lambda_avg_list = [0.5, 1.2, 2]
    lambda_sd_list = [0.05, 0.2, 0.4]
    cell_num_list = [NO_CELL, NO_CELL, NO_CELL]

    experiment = {}
    for i, lambda_avg in enumerate(lambda_avg_list):
        lambda_sd = lambda_sd_list[i]
        experiment[i] = insilico_experiment(
            kill_lambda_avg=lambda_avg,
            kill_lambda_sd=lambda_sd,
            Num_killers_obs=cell_num_list[i],
            random_seed=None
        )

    conditions = [f"λ_avg={lambda_avg_list[i]}, λ_sd={lambda_sd_list[i]}" 
                  for i in range(len(lambda_avg_list))]
    ground_truth_values = {
        cond: {
            "mu_lambda": experiment[i].kill_lambda_avg,
            "sigma_lambda": experiment[i].kill_lambda_sd
        }
        for i, cond in enumerate(conditions)
    }

    print(f"\nRunning inference in '{mode}' mode with n_quad={n_quad}...")
    idatas = inference_all(
        experiment=list(experiment.values()),
        labels=conditions,
        draws=3000,
        tune=3000,
        chains=4,
        mode=mode,
        n_quad=n_quad,
    )
    plot_joint_posteriors(
        idatas,
        ground_truth=ground_truth_values,
        parameters=["mu_lambda", "sigma_lambda"],
        hdi_prob=0.95,
        sample_size=100000,
        save_pdf=True,
        pdf_path=f"posteriors_{NO_CELL}_{mode}.pdf"
    )
    print("\nInference complete. Plot displayed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run synthetic kill simulation + Bayesian inference"
    )
    parser.add_argument("--n_cells", type=int, default=5000,
                        help="Number of NK cells per experiment")
    parser.add_argument("--n_quad", type=int, default=32,
                        help="Number of quadrature nodes for numerical integration")
    parser.add_argument("--mode", type=str, default="numerical",
                        choices=["numerical", "analytical"],
                        help="Inference mode: 'analytical' or 'numerical'")
    args = parser.parse_args()
    main(args)