import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import sys
import pymc as pm
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.gridspec as gridspec
import seaborn as sns

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

def load_csv(file_path: str) -> pd.DataFrame | None:
    if not os.path.exists(file_path):
        print(f"Error: CSV File not found at path '{file_path}'")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print("\nDataFrame Head:")
        print(df.head())
        return df
        
    except pd.errors.EmptyDataError:
        print(f"Error: The file at '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse the file at '{file_path}'. It might not be a valid CSV.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

def inference_one_i(kills_per_cell_array, draws=3000, tune=3000, chains=4, seed=None):
    M = len(kills_per_cell_array)
    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=-3.0, upper=3.0)  
        lam = pm.Deterministic("lam", 10.0 ** eta)
        pm.Poisson("K", mu=lam, observed=kills_per_cell_array)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=0.9, random_seed=seed, progressbar=True)
        try:
            idata = pm.compute_log_likelihood(idata, extend_inferencedata=True)
        except Exception:
            pass
    return idata

def inference_all_i(experiment, labels=None, draws=3000, tune=3000, chains=4, seed=None):
    if labels is None:
        labels = [f"cond_{i}" for i in range(len(experiment))]

    out = []
    for i in tqdm(range(len(experiment))):
        current_seed = seed + i if seed is not None else None
        idata = inference_one_i(
            kills_per_cell_array=experiment[i],
            draws=draws,
            tune=tune,
            chains=chains,
            seed=current_seed
        )
        out.append((labels[i], idata))
    return out

def plot_lam_posteriors(
    idatas,
    style="kde",              # "kde", "hist", or "facet"
    hdi_prob=0.95,
    export_pdf=False,
    pdf_path="lam_posteriors.pdf",
    bins=200,
    density=True,            # True=normalised density; False=counts
    show_mean=True,
    legend_show_mean=False,    # <-- put posterior mean in legend text
    legend_fmt=".3g",         # <-- format for the mean in legend
    legend_loc="best",        # <-- where to place the legend
):
    frames, order = [], []
    for label, idata in idatas:
        lam_da = idata.posterior["lam"]  # dims ('chain','draw')
        lam_vals = lam_da.stack(sample=("chain", "draw")).values.astype(float).ravel()
        frames.append(pd.DataFrame({"lam": lam_vals, "label": label}))
        order.append(label)
    all_df = pd.concat(frames, ignore_index=True)

    hdi_map = {}
    mean_map = {}
    for lab in order:
        vals = all_df.loc[all_df["label"] == lab, "lam"].to_numpy()
        hdi_map[lab] = tuple(az.hdi(vals, hdi_prob=hdi_prob))
        mean_map[lab] = float(np.mean(vals))
        print(f"{lab}: mean = {format(mean_map[lab], legend_fmt)}, HDI = [{format(hdi_map[lab][0], legend_fmt)}, {format(hdi_map[lab][1], legend_fmt)}]")

    def _legend_label(lab):
        if legend_show_mean:
            return f"{lab} (mean: {format(mean_map[lab], legend_fmt)})"
        return lab

    sns.set_context("talk", font_scale=0.9)
    palette = sns.color_palette("YlGnBu", n_colors=len(order))

    if style.lower() == "kde":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
        for c, lab in zip(palette, order):
            sdf = all_df.loc[all_df["label"] == lab]
            sns.kdeplot(data=sdf, x="lam", ax=ax, linewidth=2, color=c,
                        label=_legend_label(lab), fill=True, alpha=0.15)
            lo, hi = hdi_map[lab]
            ax.axvline(lo, color=c, linestyle="--", linewidth=1)
            ax.axvline(hi, color=c, linestyle="--", linewidth=1)
            if show_mean:
                ax.axvline(mean_map[lab], color=c, linestyle="-", linewidth=1, alpha=0.8)

        ax.set_xlabel("Killing Rate (λ)")
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior of rate (HDI {int(hdi_prob*100)}%)")
        ax.legend(frameon=False, loc=legend_loc)

    elif style.lower() == "hist":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        rmin, rmax = all_df["lam"].min(), all_df["lam"].max()
        span = (rmax - rmin) or 1e-6
        rmin, rmax = rmin - 0.02 * span, rmax + 0.02 * span
        edges = np.linspace(rmin, rmax, bins + 1)

        for c, lab in zip(palette, order):
            vals = all_df.loc[all_df["label"] == lab, "lam"].to_numpy()
            ax.hist(vals, bins=edges, density=density, histtype="stepfilled",
                    alpha=0.18, color=c, label=None)
            ax.hist(vals, bins=edges, density=density, histtype="step",
                    linewidth=1.8, color=c, label=_legend_label(lab))
            lo, hi = hdi_map[lab]
            ax.axvspan(lo, hi, color=c, alpha=0.08, linewidth=0)
            ax.axvline(lo, color=c, linestyle="--", linewidth=1)
            ax.axvline(hi, color=c, linestyle="--", linewidth=1)
            if show_mean:
                ax.axvline(mean_map[lab], color=c, linestyle="-", linewidth=1, alpha=0.9)

        ax.set_xlabel("Killing Rate (λ)")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(f"Posterior of rate — overlaid histograms (HDI {int(hdi_prob*100)}%)")
        ax.legend(frameon=False, loc=legend_loc)
        
    elif style.lower() == "facet":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for lab, color in zip(order, palette):
            vals = all_df.loc[all_df["label"] == lab, "lam"]
            sns.histplot(vals, bins=bins, stat="density" if density else "count",
                         kde=False, ax=ax, color=color, alpha=0.18, element="step", fill=True)
            sns.histplot(vals, bins=bins, stat="density" if density else "count",
                         kde=False, ax=ax, color=color, alpha=1.0, element="step", fill=False,
                         linewidth=1.8, label=_legend_label(lab))

            lo, hi = hdi_map[lab]
            ax.axvspan(lo, hi, color=color, alpha=0.08, linewidth=0)
            ax.axvline(lo, color=color, linestyle="--", linewidth=1)
            ax.axvline(hi, color=color, linestyle="--", linewidth=1)
            if show_mean:
                ax.axvline(mean_map[lab], color=color, linestyle="-", linewidth=1, alpha=0.9)

        ax.set_xlabel(fr"Killing Efficiency ($\lambda$)")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(fr"Posterior of $\lambda$ (HDI {int(hdi_prob*100)}%)")
        ax.legend(frameon=False, loc=legend_loc)

        
    else:
        raise ValueError(f"Unknown style: '{style}'. Choose 'kde', 'hist', or 'facet'.")


    plt.tight_layout()
    if export_pdf:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved PDF: {pdf_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path_to_load = sys.argv[1]
    else:
        csv_path_to_load = "./organised_data.csv"
        print("\n" + "-"*50 + "\n")
    df = load_csv(csv_path_to_load)
    print("\n" + "-"*50 + "\n")

    df_mult = df.query("experiment_condi == 'NK_VS_RMA-MULT'")
    df_rae = df.query("experiment_condi == 'NK_VS_RMA-RAE'")
    
    class experimental_data:
        def __init__(self, df):
            self.df_mult = df.query("experiment_condi == 'NK_VS_RMA-MULT'")
            self.df_rae = df.query("experiment_condi == 'NK_VS_RMA-RAE'")

        def get_killing_array(self):
            mult_array = self.df_mult['kill_No'].to_numpy()
            rae_array = self.df_rae['kill_No'].to_numpy()
            return {'NK VS RMA-MULT': mult_array, 'NK VS RMA-RAE': rae_array}

    experimental_dic = experimental_data(df).get_killing_array()
    conditions = [i for i in experimental_dic.keys()]
    cmap = get_cmap('YlGnBu')
    colors = cmap(np.linspace(0.3, 0.9, len(conditions)))
    experiments = [experimental_dic[cond] for cond in conditions]
    
    idatas = inference_all_i(
        experiment=experiments,
        labels=conditions,
        draws=3000,
        tune=3000,
        chains=6,
        seed=None,
    )
    plot_lam_posteriors(
        idatas, style="facet", hdi_prob=0.95, density = True,
        export_pdf=True, pdf_path="./rate_posteriors_facet.pdf",
        legend_show_mean=True
    )