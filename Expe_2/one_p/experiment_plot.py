import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator

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

def load_csv(file_path: str) -> pd.DataFrame | None:
    if not os.path.exists(file_path):
        print(f"Error: CSV File not found at path '{file_path}'")
        return None
    try:
        df = pd.read_csv(file_path)
        print("\nDataFrame Head:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def extract_experimental_data(df):
    df_mult = df.query("experiment_condi == 'NK_VS_RMA-MULT'")
    df_rae = df.query("experiment_condi == 'NK_VS_RMA-RAE'")
    mult_array = df_mult['kill_No'].to_numpy(dtype=int)
    rae_array = df_rae['kill_No'].to_numpy(dtype=int)
    return {'NK VS RMA-MULT': mult_array, 'NK VS RMA-RAE': rae_array}

def plot_experimental_data(experiments, lambda_list, save: bool = False, out_dir: str = "./plots", filename: str = "experimental_plot"):
    """Plot frequency and count distributions for experiments.

    If save is True the function saves both PDF and PNG to ``out_dir/filename.(pdf|png)``
    and closes the figure (non-blocking). Otherwise the plot is shown interactively.
    """
    conditions = [f"{lambda_list[i]}" for i in range(len(lambda_list))]
    cmap = get_cmap('YlGnBu')
    colors = cmap(np.linspace(0.3, 0.9, len(conditions)))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=250, sharex=True)
    ax_freq, ax_count = axes

    for i, cond_name in enumerate(conditions):
        colour = colors[i]
        kills = experiments[i]
        n_cells = len(kills)
        max_kills = int(kills.max())
        counts = np.bincount(kills, minlength=max_kills + 1)
        freq = counts / n_cells if n_cells > 0 else np.zeros_like(counts)
        x = np.arange(len(counts))
        ax_freq.plot(x, freq, color=colour, linewidth=2.5, label=cond_name, marker='o', markersize=4)
        ax_count.plot(x, counts, color=colour, linewidth=2.5, label=cond_name, marker='o', markersize=4)

    ax_freq.set_title("Frequency Distribution", fontweight='bold')
    ax_count.set_title("Count Distribution", fontweight='bold')
    ax_freq.set_ylabel("Frequency")
    ax_count.set_ylabel("Count")

    for ax in (ax_freq, ax_count):
        ax.set_xlabel("Number of targets killed per NK cell")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3)

    ax_count.legend(title="Condition", frameon=True, edgecolor='black', loc='upper right', fontsize=9)
    plt.tight_layout()

    if save:
        # ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, f"{filename}.pdf")
        png_path = os.path.join(out_dir, f"{filename}.png")
        try:
            fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"Saved plots: {pdf_path} and {png_path}")
        except Exception as e:
            print(f"Error saving files: {e}")
        # close figure to avoid blocking / resource leak
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    csv_path = "./organised_data.csv"
    df = load_csv(csv_path)
    if df is not None:
        experimental_dic = extract_experimental_data(df)
        experiments = [experimental_dic[k] for k in experimental_dic.keys()]
    lambda_list = [np.mean(exp) for exp in experiments]
    label = ['NK VS RMA-MULT', 'NK VS RMA-RAE']
    # Example: save the plot (both PDF and PNG) into the folder for this module
    out_folder = os.path.join(os.path.dirname(__file__), "plots")
    plot_experimental_data(experiments, label, save=True, out_dir=out_folder, filename="rate_posteriors_facet")