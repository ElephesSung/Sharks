import os
from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator, MaxNLocator
import seaborn as sns
from scipy.stats import poisson
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.tensor.special import gammaln  

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


T_Total = 12


non_interaction = [
    # n=38: No interaction
    *[np.array([]) for _ in range(38)],
]

# No kill
no_kill = [
    # n=23: 1 contact, no kill
    *[np.array([0]) for _ in range(23)],
    # n=13: 2 contacts, no kill
    *[np.array([0, 0]) for _ in range(13)],
    # n=10: 3 contacts, no kill
    *[np.array([0, 0, 0]) for _ in range(10)],
    # n=2: 4 contacts, no kill
    *[np.array([0, 0, 0, 0]) for _ in range(2)],
    # n=1: 5 contacts, no kill
    *[np.array([0, 0, 0, 0, 0]) for _ in range(1)],
    # n=1: 6 contacts, no kill
    *[np.array([0, 0, 0, 0, 0, 0]) for _ in range(1)]
]

# Kill
kill = [
    # n=14: 1 contact, kill
    *[np.array([1]) for _ in range(14)],
    # n=15: 2 contacts, kill
    *[np.array([1, 1]) for _ in range(15)],
    # n=13: 3 contacts, kill
    *[np.array([1, 1, 1]) for _ in range(13)],
    # n=5: 4 contacts, kill
    *[np.array([1, 1, 1, 1]) for _ in range(5)],
    # n=3: 5 contacts, kill
    *[np.array([1, 1, 1, 1, 1]) for _ in range(3)],
    # n=3: 6 contacts, kill
    *[np.array([1, 1, 1, 1, 1, 1]) for _ in range(3)],
    # n=1: 7 contacts, kill
    *[np.array([1, 1, 1, 1, 1, 1, 1]) for _ in range(1)]
]

# Exhausted
exhausted = [
    *[np.array([1, 0]) for _ in range(7)],
    *[np.array([1, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 0]) for _ in range(9)],
    *[np.array([1, 0, 0, 0]) for _ in range(2)],
    *[np.array([1, 1, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 1, 0]) for _ in range(1)],
    *[np.array([1, 1, 0, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 1, 0, 0]) for _ in range(3)],
    *[np.array([1, 1, 1, 1, 0]) for _ in range(1)],
    *[np.array([1, 0, 0, 0, 0, 0]) for _ in range(1)],
    *[np.array([1, 1, 1, 1, 1, 0]) for _ in range(2)],
    *[np.array([1, 1, 1, 1, 1, 1, 1, 0]) for _ in range(1)],
]

# Stochastic
stochastic = [
    # n=1
    *[np.array([0, 1, 1, 0]) for _ in range(1)],
    # n=1
    *[np.array([1, 0, 1, 0]) for _ in range(1)],
    # n=1
    *[np.array([1, 1, 0, 1, 0]) for _ in range(1)],
    # n=1
    *[np.array([1, 0, 1, 0, 0, 0]) for _ in range(1)],
    # n=1
    *[np.array([0, 0, 0, 1, 1, 1]) for _ in range(1)],
    # n=1
    *[np.array([0, 1, 1, 0, 0, 0, 0]) for _ in range(1)],
]

all_history = non_interaction + no_kill + kill + exhausted + stochastic
print(f"Total number of histories represented: {len(all_history)}")

all_cell_killNo = [int(np.sum(cell)) for cell in all_history]
kill_counts = np.array(all_cell_killNo)
np.random.shuffle(kill_counts)
replicates = np.array_split(kill_counts, 5)

KillFreqTable = {
    'No treatment': {}
}
for i, rep in enumerate(replicates):
    KillFreqTable['No treatment'][i+1] = rep
    print(f"Replicate {i+1} (size: {len(rep)}):")
    print(rep)
    print()
    


def inference(experimental_data, T=60.0, draws=3000, tune=3000, chains=4, seed=66):
    """
    experimental_data: list of 1-D arrays (replicates), each holding per-cell kill counts
    returns: ArviZ InferenceData
    """
    # flatten per-cell counts across replicates
    all_cells = np.concatenate([np.asarray(rep, int) for rep in experimental_data if len(rep) > 0])
    # bins 0..Kcut-2 explicitly, tail bin at Kcut-1
    Kcut = int(all_cells.max()) + 3
    K = np.bincount(np.clip(all_cells, 0, Kcut - 1), minlength=Kcut).astype("int64")
    M = int(K.sum())
    
    # MCMC
    with pm.Model() as model:
        # prior on log10(lambda)
        eta = pm.Uniform("eta", lower=-10.0, upper=10.0)     # eta = log10(lambda)
        lam = pm.Deterministic("lam", 10.0 ** eta)
        r   = pm.Deterministic("r", lam / T)               # per-time rate

        # Poisson pmf for categories 0..Kcut-2 (log-space)
        ks = pt.arange(Kcut - 1)
        logpmf = -lam + ks * pt.log(lam) - gammaln(ks + 1)
        pmf = pt.exp(logpmf)

        # tail (>= Kcut-1), clip + renormalise just in case
        tail = 1.0 - pt.sum(pmf)
        pmf  = pt.clip(pmf, 1e-16, 1.0)
        tail = pt.clip(tail, 1e-16, 1.0)
        probs = pt.concatenate([pmf, tail[None]])
        probs = probs / pt.sum(probs)

        pm.Multinomial("K", n=M, p=probs, observed=K)

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=0.9, random_seed=seed, progressbar=True)

        # attach log_likelihood (handy for plotting colour scales etc.)
        try:
            idata = pm.compute_log_likelihood(idata, extend_inferencedata=True)
        except Exception:
            pass

    return idata


def run_inference_for_list(data_list, T=60.0, labels=None, seed0=42):
    """
    data_list: list where each item is `experimental_data` (list of 1-D arrays per replicate)
    returns: list of (label, idata), where idata is the InferenceData from `inference(...)`.
    """
    if labels is None:
        labels = [f"cond_{i}" for i in range(len(data_list))]

    out = []
    for i, experimental_data in enumerate(data_list):
        lbl = str(labels[i])
        print(f"Fitting {lbl} ...")
        idata = inference(experimental_data, T=T, seed=seed0 + i)
        # Save the inference data
        idata.to_netcdf(f"{lbl}_idata.nc")
        out.append((lbl, idata))
    return out


# Prepare data_list and labels for real experimental data
labels = list(KillFreqTable.keys())
data_list = [
    [KillFreqTable[cond][rep] for rep in KillFreqTable[cond].keys()]
    for cond in labels
]
# Run Bayesian inference for each experimental condition
idatas = run_inference_for_list(data_list, T=T_Total, labels=labels)


for label, idata in idatas:
    filename = f"{label}_idata.nc"
    az.to_netcdf(idata, filename)
    

def plot_rate_posteriors(
    idatas,
    style="kde",              # "kde" or "hist" (overlayed in ONE diagram)
    hdi_prob=0.95,
    export_pdf=False,
    pdf_path="rate_posteriors.pdf",
    bins=100,
    density=False,            # True=normalised density; False=counts
    show_mean=True,
    legend_show_mean=True,    # <-- put posterior mean in legend text
    legend_fmt=".3g",         # <-- format for the mean in legend
    legend_loc="best",        # <-- where to place the legend
):
    """
    idatas: list of (label, idata) as returned by run_inference_for_list()
    style : "kde" for overlaid density curves; "hist" for overlaid histograms on ONE axis
    """
    # Collect posterior draws of r into a tidy DataFrame
    frames, order = [], []
    for label, idata in idatas:
        r_da = idata.posterior["r"]  # dims ('chain','draw')
        r_vals = r_da.stack(sample=("chain", "draw")).values.astype(float).ravel()
        frames.append(pd.DataFrame({"r": r_vals, "label": label}))
        order.append(label)
    all_df = pd.concat(frames, ignore_index=True)

    # HDIs & posterior means per label
    hdi_map = {}
    mean_map = {}
    for lab in order:
        vals = all_df.loc[all_df["label"] == lab, "r"].to_numpy()
        hdi_map[lab] = tuple(az.hdi(vals, hdi_prob=hdi_prob))
        mean_map[lab] = float(np.mean(vals))
        print(f"{lab}: mean = {format(mean_map[lab], legend_fmt)}, HDI = [{format(hdi_map[lab][0], legend_fmt)}, {format(hdi_map[lab][1], legend_fmt)}]")

    # Helper to make legend text
    def _legend_label(lab):
        return lab

    sns.set_context("talk", font_scale=0.9)
    palette = sns.color_palette("YlGnBu", n_colors=len(order))

    if style.lower() == "kde":
        # -------- Overlaid KDEs --------
        fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
        for c, lab in zip(palette, order):
            sdf = all_df.loc[all_df["label"] == lab]
            sns.kdeplot(data=sdf, x="r", ax=ax, linewidth=2, color=c,
                        label=_legend_label(lab), fill=True, alpha=0.15)
            lo, hi = hdi_map[lab]
            ax.axvline(lo, color=c, linestyle="--", linewidth=1)
            ax.axvline(hi, color=c, linestyle="--", linewidth=1)
            if show_mean:
                ax.axvline(mean_map[lab], color=c, linestyle="-", linewidth=1, alpha=0.8)

        ax.set_xlabel("Killing Rate")
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior of rate (HDI {int(hdi_prob*100)}%)")
        ax.legend(frameon=False, loc=legend_loc)

    else:
        # -------- Overlaid histograms on ONE axis --------
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        rmin, rmax = all_df["r"].min(), all_df["r"].max()
        span = (rmax - rmin) or 1e-6
        rmin, rmax = rmin - 0.02 * span, rmax + 0.02 * span
        edges = np.linspace(rmin, rmax, bins + 1)

        for c, lab in zip(palette, order):
            vals = all_df.loc[all_df["label"] == lab, "r"].to_numpy()
            # filled translucent + outline for readability
            ax.hist(vals, bins=edges, density=density, histtype="stepfilled",
                    alpha=0.18, color=c, label=None)
            ax.hist(vals, bins=edges, density=density, histtype="step",
                    linewidth=1.8, color=c, label=_legend_label(lab))

            # HDI shading and lines
            lo, hi = hdi_map[lab]
            ax.axvspan(lo, hi, color=c, alpha=0.08, linewidth=0)
            ax.axvline(lo, color=c, linestyle="--", linewidth=1)
            ax.axvline(hi, color=c, linestyle="--", linewidth=1)
            if show_mean:
                ax.axvline(mean_map[lab], color=c, linestyle="-", linewidth=1, alpha=0.9)

        ax.set_xlabel("Killing Rate")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(f"Posterior of rate — overlaid histograms (HDI {int(hdi_prob*100)}%)")
        ax.legend(frameon=False, loc=legend_loc)

    plt.tight_layout()
    if export_pdf:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved PDF: {pdf_path}")
    plt.show()
# Plot posterior distributions of the inferred rates
plot_rate_posteriors(
    idatas, style="facet", hdi_prob=0.95, density = True,
    export_pdf=True, pdf_path="./rate_posteriors_facet.pdf"
)
plot_rate_posteriors(
    idatas, style="kde", hdi_prob=0.95, density = True,
    export_pdf=True, pdf_path="./rate_posteriors_kde.pdf"
)



# --- Simulation and Data Storage (Parallelized Simulation, Per-Replicate Cell Numbers, Fast NumPy Frequency Calculation, tqdm progress, export/import) ---
from tqdm import tqdm
from joblib import Parallel, delayed


def experiment(
    killing_rate: float = 0.01,
    Duration: float = 60,
    Killer_cell_number: int = 50,
    replicate: int = 10,
    seed: int = 66,
):
    lam = killing_rate * Duration
    rng = np.random.default_rng(seed)
    kills = []
    for _ in range(replicate):
        kill = poisson.rvs(mu=lam, size=Killer_cell_number, random_state=rng)
        kills.append(kill)
    return kills

def calc_freq_matrix_vectorized(flat_sims, max_kill, n_cells):
    # Each row: bincount with consistent length
    freq_matrix = np.array([np.bincount(row, minlength=max_kill+1) / n_cells for row in flat_sims], dtype=np.float32)
    return freq_matrix

def simulate_replicate(arr, posterior_rate_list, T_Total, n_replicate=5000):
    n_cells = arr.size
    counts = np.bincount(arr, minlength=int(arr.max())+1)
    freq = counts / n_cells if n_cells > 0 else np.zeros_like(counts)
    n_rates = posterior_rate_list.shape[0]
    lam = posterior_rate_list[:, None] * T_Total  # shape (n_rates, 1)
    simulations = np.random.poisson(lam=lam[..., None], size=(n_rates, n_replicate, n_cells))
    flat_sims = simulations.reshape(-1, n_cells)
    max_kill = np.max(flat_sims)
    freq_matrix = calc_freq_matrix_vectorized(flat_sims, max_kill, n_cells)
    return freq, freq_matrix

# Directory to save/load data
data_dir = "./sim_data_exports"
os.makedirs(data_dir, exist_ok=True)

generated_data = {}
conditions = list(KillFreqTable.keys())
cmap = get_cmap('YlGnBu')
colors = cmap(np.linspace(0.3, 0.9, len(conditions)))

for i, group in enumerate(tqdm(conditions, desc='Simulating conditions')):
    colour = colors[i]
    posterior_rate_list = np.array(idatas[i][1].posterior['r'].values.flatten())
    rep_cell_numbers = [KillFreqTable[group][rep].size for rep in KillFreqTable[group]]

    # Parallelize per-replicate simulation and frequency calculation
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(simulate_replicate)(
            KillFreqTable[group][rep], posterior_rate_list, T_Total
        ) for rep in KillFreqTable[group]
    )
    exp_freqs, sim_freqs_per_rep = zip(*results) if results else ([], [])

    # Export to .npz file for each group
    export_path = os.path.join(data_dir, f"{group.replace(' ','_')}_simdata.npz")
    np.savez_compressed(
        export_path,
        exp_freqs=np.array(exp_freqs, dtype=object),  # usually safe for 1D arrays
        colour=colour,
        maxlen=max([freq.shape[1] for freq in sim_freqs_per_rep]) if sim_freqs_per_rep else 0,
        **{f"sim_freqs_per_rep_{j}": sim_freqs_per_rep[j] for j in range(len(sim_freqs_per_rep))}
    )

    generated_data[group] = {
        'exp_freqs': list(exp_freqs),
        'sim_freqs_per_rep': list(sim_freqs_per_rep),
        'colour': colour,
        'maxlen': max([freq.shape[1] for freq in sim_freqs_per_rep]) if sim_freqs_per_rep else 0
    }
