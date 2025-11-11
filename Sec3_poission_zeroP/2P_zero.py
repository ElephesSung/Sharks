import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import seaborn as sns
import pymc as pm
import arviz as az
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
mpl.rcParams['animation.embed_limit'] = 2000


class insilico_experiment:
    def __init__(self, kill_lambda=3.0, p_zero=0.1, Num_killers_obs=300, random_seed=66):
        self.kill_lambda = kill_lambda
        self.p_zero = p_zero
        self.Num_killers_obs = Num_killers_obs
        self.random_seed = random_seed
        self.lambdas, self.kills = self._generate_data()

    def _generate_data(self):
        rng = np.random.default_rng(self.random_seed)
        is_zero_killer = rng.random(self.Num_killers_obs) < self.p_zero
        lambdas = np.where(is_zero_killer, 0.0, self.kill_lambda)
        kills = rng.poisson(lambdas)
        return lambdas, kills

    def regenerate(self, new_seed=None):
        if new_seed is not None:
            self.random_seed = new_seed
        self.lambdas, self.kills = self._generate_data()

    def get_summary(self):
        return {
            "kill_lambda": self.kill_lambda,
            "p_zero": self.p_zero,
            "lambdas": self.lambdas,
            "kills": self.kills,
            "killer_number": self.Num_killers_obs
        }


def inference_one_zip(kills_per_cell, draws=3000, tune=3000, chains=4, seed=None):
    N = np.asarray(kills_per_cell, dtype=int)
    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=-1.0, upper=1.0)
        mu_lambda = pm.Deterministic("mu_lambda", 10.0 ** eta)
        p_zero = pm.Beta("p_zero", alpha=1.0, beta=1.0)
        pm.ZeroInflatedPoisson("kills", psi=1 - p_zero, mu=mu_lambda, observed=N)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.95, random_seed=seed)
    return idata


def inference_all(experiment, labels=None, draws=3000, tune=3000, chains=4, seed=None):
    if labels is None:
        labels = [f"cond_{i}" for i in range(len(experiment))]
    out = []
    for i in tqdm(range(len(experiment))):
        current_seed = seed + i if seed is not None else None
        idata = inference_one_zip(
            kills_per_cell=experiment[i].get_summary()['kills'],
            draws=draws, tune=tune, chains=chains, seed=current_seed
        )
        out.append((labels[i], idata))
    return out


def plot_synthetic_vs_jointcorner(
    experiment,
    lambda_list,
    p_zero_list,
    idatas,
    parameters=["mu_lambda", "p_zero"],
    parameter_display=None,
    ground_truth_values=None,
    sample_size=10000,
    cmap_name="YlGnBu",
    diagonal_style="hist",
    marginal_style="circle",
    save_path="./combined_corner.pdf",
    font_scale=0.8,
):
    sns.set_context("talk", font_scale=font_scale)
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.9, len(lambda_list)))

    if parameter_display is None:
        parameter_display = {"mu_lambda": r"$\lambda$", "p_zero": r"$p_z$"}

    fig = plt.figure(figsize=(14, 6), dpi=400)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
    ax_left = fig.add_subplot(gs[0, 0])

    conditions = [fr"$\lambda={lam}$, $p_z={p_zero_list[i]}$" for i, lam in enumerate(lambda_list)]
    for i, cond_name in enumerate(conditions):
        colour = colors[i]
        kills = experiment[i].get_summary()['kills']
        n_cells = len(kills)
        max_kills = int(kills.max())
        counts = np.bincount(kills, minlength=max_kills + 1)
        freq = counts / n_cells if n_cells > 0 else np.zeros_like(counts)
        x = np.arange(len(counts))
        ax_left.plot(x, freq, color=colour, linewidth=2.5, label=cond_name, marker='o', markersize=4)

    ax_left.set_title("Frequency Distribution", fontweight='bold')
    ax_left.set_xlabel("Targets killed per killer immune cell")
    ax_left.set_ylabel("Frequency")
    ax_left.xaxis.set_major_locator(MultipleLocator(1))
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(title="Ground Truth", frameon=True, edgecolor='black', loc='upper right', fontsize=11)

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

    par_x, par_y = parameters
    x_name = parameter_display.get(par_x, par_x)
    y_name = parameter_display.get(par_y, par_y)

    gs_corner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1],
                                                 width_ratios=[4, 1.2], height_ratios=[1.2, 4],
                                                 wspace=0.05, hspace=0.05)
    ax_joint = plt.subplot(gs_corner[1, 0])
    ax_x = plt.subplot(gs_corner[0, 0], sharex=ax_joint)
    ax_y = plt.subplot(gs_corner[1, 1], sharey=ax_joint)

    for colour, (label, df) in zip(colors, label_to_df.items()):
        if marginal_style == "circle":
            sns.kdeplot(x=df[par_x], y=df[par_y], ax=ax_joint, fill=False, color=colour, levels=7, alpha=0.6, linewidths=1.2)
        elif marginal_style == "pixel":
            sns.histplot(x=df[par_x], y=df[par_y], bins=20, pthresh=0.01, cmap=cmap_name, cbar=False, ax=ax_joint)
        if ground_truth_values and label in ground_truth_values:
            gt = ground_truth_values[label]
            if par_x in gt and par_y in gt:
                ax_joint.plot(gt[par_x], gt[par_y], marker='*', color=colour, markersize=10, markeredgewidth=2.0, linestyle='None')

    ax_joint.set_xlabel(x_name)
    ax_joint.set_ylabel(y_name)
    ax_joint.grid(alpha=0.3)

    for colour, (label, df) in zip(colors, label_to_df.items()):
        vals = df[par_x].dropna().values
        if diagonal_style == "kde":
            sns.kdeplot(vals, ax=ax_x, fill=True, color=colour, alpha=0.25, linewidth=1.5)
        elif diagonal_style == "hist":
            sns.histplot(vals, bins=30, stat="density", kde=False, ax=ax_x, color=colour, alpha=0.18, element="step", fill=True)
            sns.histplot(vals, bins=30, stat="density", kde=False, ax=ax_x, color=colour, alpha=1.0, element="step", fill=False, linewidth=1.8)
        if ground_truth_values and label in ground_truth_values:
            gt = ground_truth_values[label]
            if par_x in gt:
                ax_x.axvline(gt[par_x], color=colour, linestyle='--', linewidth=2)
        ax_x.set_ylabel("Density")

    ax_x.tick_params(axis="x", labelbottom=False)
    ax_x.grid(alpha=0.2)

    for colour, (label, df) in zip(colors, label_to_df.items()):
        vals = df[par_y].dropna().values
        if diagonal_style == "kde":
            sns.kdeplot(y=vals, ax=ax_y, fill=True, color=colour, alpha=0.25, linewidth=1.5)
        elif diagonal_style == "hist":
            sns.histplot(y=vals, bins=30, stat="density", kde=False, ax=ax_y, color=colour, alpha=0.18, element="step", fill=True)
            sns.histplot(y=vals, bins=30, stat="density", kde=False, ax=ax_y, color=colour, alpha=1.0, element="step", fill=False, linewidth=1.8)
        if ground_truth_values and label in ground_truth_values:
            gt = ground_truth_values[label]
            if par_y in gt:
                ax_y.axhline(gt[par_y], color=colour, linestyle='--', linewidth=2)
        ax_y.set_xlabel("Density")

    ax_y.tick_params(axis="y", labelleft=False)
    ax_y.grid(alpha=0.2)

    ax_joint.margins(0.05)
    ax_x.margins(0.05)
    ax_y.margins(0.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main(No_cell):
    os.makedirs("./syn_results", exist_ok=True)
    lambda_list = [0.5, 1.2, 2]
    p_zero_list = [0.05, 0.15, 0.3]
    cell_num_list = [No_cell, No_cell, No_cell]
    experiment = {}
    for i, lambda_avg in enumerate(lambda_list):
        experiment[i] = insilico_experiment(
            kill_lambda=lambda_avg, p_zero=p_zero_list[i], Num_killers_obs=cell_num_list[i], random_seed=i
        )
    conditions = [f"λ={lambda_list[i]}, p_z={p_zero_list[i]}" for i in range(len(lambda_list))]
    cmap = get_cmap('YlGnBu')
    colors = cmap(np.linspace(0.3, 0.9, len(conditions)))
    ground_truth_values = {}
    for i, cond_label in enumerate(conditions):
        ground_truth_values[cond_label] = {"mu_lambda": experiment[i].kill_lambda, "p_zero": experiment[i].p_zero}
    idatas = inference_all(experiment=list(experiment.values()), labels=conditions, draws=3000, tune=3000, chains=6)
    plot_synthetic_vs_jointcorner(
        experiment=experiment,
        lambda_list=lambda_list,
        p_zero_list=p_zero_list,
        idatas=idatas,
        ground_truth_values=ground_truth_values,
        save_path=f"./syn_results/ZIP_{No_cell}.pdf"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <cell_number>")
        sys.exit(1)
    No_cell = int(sys.argv[1])
    main(No_cell)