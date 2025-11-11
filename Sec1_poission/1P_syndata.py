#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.gridspec as gridspec
import seaborn as sns

import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.tensor.special import gammaln
from numpy.polynomial.laguerre import laggauss

from tqdm import tqdm

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


class insilico_experiment:
    def __init__(
        self,
        kill_lambda: float = 3.0,
        Num_killers_obs: int = 300,
        random_seed: int = 66
    ):
        self.kill_lambda = kill_lambda
        self.Num_killers_obs = Num_killers_obs
        self.random_seed = random_seed
        
        self.kills = self._generate_data()
    
    def _generate_data(self):
        rng = np.random.default_rng(self.random_seed)
        lambdas = np.full(self.Num_killers_obs, self.kill_lambda)
        kills = rng.poisson(lambdas)
        return kills
    
    def regenerate(self, new_seed: int = None):
        if new_seed is not None:
            self.random_seed = new_seed
        self.kills = self._generate_data()
        
    def get_summary(self):
        return {
            "kill_lambda": self.kill_lambda,
            "kills": self.kills,
            "killer_number": self.Num_killers_obs
        }


def inference_one(kills_per_cell, draws=3000, tune=3000, chains=4, seed=None):
    N = np.asarray(kills_per_cell, dtype=int)

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=-2.0, upper=2.0)
        mu_lambda = pm.Deterministic("mu_lambda", 10.0 ** eta)
        pm.Poisson("kills", mu=mu_lambda, observed=N)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=0.95, random_seed=seed
        )

    return idata

def inference_all(experiment, labels=None,
    draws=3000, tune=3000, chains=4, seed=None,
):
    if labels is None:
        labels = [f"cond_{i}" for i in range(len(experiment))]

    out = []
    for i in tqdm(range(len(experiment))):
        current_seed = seed + i if seed is not None else None
        idata = inference_one(
            kills_per_cell=experiment[i].get_summary()['kills'],
            draws=draws, tune=tune, chains=chains, seed=current_seed
            )
        out.append((labels[i], idata))
    return out


def plot_synthetic_vs_posterior(experiment, lambda_list, idatas,
                                ground_truth_values=None,
                                parameters=["mu_lambda"],
                                hdi_prob=0.95,
                                sample_size=10000,
                                parameter_display=None,
                                cmap_name="YlGnBu",
                                save_path="./syn_results.pdf"):

    if parameter_display is None:
        parameter_display = {"mu_lambda": r"$\lambda$"}

    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.9, len(lambda_list)))
    sns.set_context("talk", font_scale=0.8)

    fig = plt.figure(figsize=(14, 6), dpi=400)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    conditions = [fr"$\lambda={lam}$" for lam in lambda_list]
    for i, cond_name in enumerate(conditions):
        colour = colors[i]
        kills = experiment[i].get_summary()['kills']
        n_cells = len(kills)
        max_kills = int(kills.max())
        counts = np.bincount(kills, minlength=max_kills + 1)
        freq = counts / n_cells if n_cells > 0 else np.zeros_like(counts)
        x = np.arange(len(counts))
        ax_left.plot(x, freq, color=colour, linewidth=2.5,
                     label=cond_name, marker='o', markersize=4)

    ax_left.set_title("Frequency Distribution", fontweight='bold')
    ax_left.set_xlabel("Targets killed per killer immune cell")
    ax_left.set_ylabel("Frequency")
    ax_left.xaxis.set_major_locator(MultipleLocator(1))
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(title="Ground Truth", frameon=True,
                   edgecolor='black', loc='upper right', fontsize=12)

    label_to_df = {}
    for label, idata in idatas:
        posterior = idata.posterior
        df = pd.DataFrame()
        for p in parameters:
            vals = posterior[p].stack(sample=("chain", "draw")).values.ravel()
            if len(vals) > sample_size:
                vals = np.random.choice(vals, sample_size, replace=False)
            df[p] = vals
        df["label"] = label
        label_to_df[label] = df

    for color, (label, df) in zip(colors, label_to_df.items()):
        vals = df[parameters[0]].dropna().values
        sns.histplot(vals, bins=30, stat="density", kde=False, ax=ax_right,
                     color=color, alpha=0.18, element="step", fill=True)
        sns.histplot(vals, bins=30, stat="density", kde=False, ax=ax_right,
                     color=color, alpha=1.0, element="step", fill=False, linewidth=1.8,
                     label=label)
        if len(vals) > 0:
            lo, hi = az.hdi(vals, hdi_prob=hdi_prob)
            ax_right.axvspan(lo, hi, color=color, alpha=0.12, linewidth=0)
        if ground_truth_values and label in ground_truth_values and parameters[0] in ground_truth_values[label]:
            ax_right.axvline(ground_truth_values[label][parameters[0]],
                             color=color, linestyle='-', linewidth=2)

    disp_name = parameter_display.get(parameters[0], parameters[0])
    ax_right.set_title("Posterior Inference", fontweight='bold')
    ax_right.set_xlabel(disp_name)
    ax_right.set_ylabel("Density")
    # ax_right.grid(True, alpha=0.3)
    # ax_right.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure to {save_path}")

def main(No_cell):
    lambda_list = [0.5, 1.2, 2]
    cell_num_list = [No_cell, No_cell, No_cell]
    experiment = {}
    for i, lambda_avg in enumerate(lambda_list):
        experiment[i] = insilico_experiment(
            kill_lambda=lambda_avg,
            Num_killers_obs=cell_num_list[i],
            random_seed=i
        )
    conditions = [
        f"λ_avg={lambda_list[i]}" 
        for i in range(len(lambda_list))
    ]
    cmap = get_cmap('YlGnBu')
    colors = cmap(np.linspace(0.3, 0.9, len(conditions)))
    ground_truth_values = {}
    for i, cond_label in enumerate(conditions):
        ground_truth_values[cond_label] = {
        "mu_lambda": experiment[i].kill_lambda
        }   
    idatas = inference_all(
        experiment=list(experiment.values()),
        labels=conditions,
        draws=4000,
        tune=2000,
        chains=4,
    )
    plot_synthetic_vs_posterior(
        experiment=experiment,
        lambda_list=lambda_list,
        idatas=idatas,
        ground_truth_values=ground_truth_values,
        save_path=f"./syn_results/1P_{NO_CELL}.pdf"
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_path>")
        sys.exit(1)

    NO_CELL = sys.argv[1]
    main(int(NO_CELL))