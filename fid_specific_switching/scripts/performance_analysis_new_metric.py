import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import permutation_test

eps = np.finfo(float).eps  # Small value to avoid division by zero
sbs_sum = 2690.130170 # For iid 6,7, late and all switching points, BFGS at 450
# sbs_sum = 3306.1779694383 # For new reps, all switching points, Non-elitist budget 16
# sbs_sum = 3511.967 # For new reps, late switching points, 50 Non-elitist

def compute_vbs_ratios(csv_path, fid = None):

    df = pd.read_csv(csv_path)
    vbs_sum = df["vbs_precision"].sum()
    consider_cols = [col for col in df.columns if col.startswith("static_B") or col == "selector_precision"]
    res = {}
    for col in consider_cols:
        col_sum = df[col].sum()
        print(f"Processing column: {col}, sum: {col_sum}")
        res[col] = (sbs_sum - col_sum) / (sbs_sum - vbs_sum) 

    return res

def find_sbs(path):
    df = pd.read_csv(path)

    score_table = (
        df.groupby(["budget", "algorithm"])["precision"]
        .sum()
        .reset_index()
        .sort_values("precision")
        .reset_index(drop=True)
    )

    return score_table

def save_tables(data, title, filename):
    df = pd.DataFrame(list(data.items()), columns=["Method", "Ratio"])
    df = df.sort_values("Ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(df) + 1))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def compute_total_precisions_for_fid(csv_path, fid):
    df = pd.read_csv(csv_path)
    df = df[df["fid"] == fid]

    consider_cols = [col for col in df.columns if col.startswith("static_B") or col == "selector_precision"]
    result = {col: df[col].sum() for col in consider_cols}
    return result

def display_vbs_tables(csv_path,fid=None):
    if fid is None:
        ratios = compute_vbs_ratios(csv_path)
        output_path = "../results/newInstances/precision_ratios_all_tuned.png"
        # save_tables(ratios, "VBS Relative Ratios", output_path)
        print("✅ VBS ratios saved to:", output_path)
    else:
        totals = compute_total_precisions_for_fid(csv_path, fid)
        output_dir = "../data/switching_optimality_files/vbs_precision_totals_late"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/vbs_precision_totals_fid_{fid}.png"
        save_tables(totals, f"Sum of Precisions (fid={fid})", output_path)

def plot_selector_budget_counts(csv_path, output_png="selector_budget_counts.png"):
    df = pd.read_csv(csv_path)

    # Count occurrences of each budget
    counts = df["selector_switch_budget"].value_counts().sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color='steelblue')

    ax.set_title("Number of runs in which the selector switched at each budget", fontsize=14)
    ax.set_xlabel("Budget", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Annotate bars with count values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def permutation_test_selector_vs_static(csv_path):

    df = pd.read_csv(csv_path)
    selector = df['selector_precision'].values

    for budget in [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]:
        static = df[f'static_B{budget}'].values

        # Compute observed mean difference
        observed_diff = np.mean(selector - static)
        print(f"Observed mean difference: {observed_diff:.6f}")

        # Number of permutations
        n_permutations = 10000
        perm_diffs = np.zeros(n_permutations)

        # Manual permutations
        n = len(selector)

        for i in range(n_permutations):
            # For each pair, decide whether to swap (50% chance)
            swap = np.random.rand(n) < 0.5
            
            perm_selector = np.where(swap, static, selector)
            perm_static   = np.where(swap, selector, static)
            
            # Compute permuted mean difference
            perm_diffs[i] = np.mean(perm_selector - perm_static)

        # Calculate p-value for alternative hypothesis: selector < static
        p_value_less = (np.sum(perm_diffs <= observed_diff)) / (n_permutations)
        print(f"P-value (selector < static) for budget {budget}: {p_value_less:.6f}")

    # plt.figure(figsize=(8,5))
    # plt.hist(perm_diffs, bins=50, alpha=0.7, color='skyblue', edgecolor='k')
    # plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2, label=f'Observed mean diff = {observed_diff:.3f}')
    # plt.xlabel('Permuted mean differences (selector - static)')
    # plt.ylabel('Frequency')
    # plt.title('Permutation Test Null Distribution')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("permutation_test_distribution.png")

def get_sbs_precisions(precision_csv_path):
    """
    Loads the precision file and extracts SBS per run:
    BFGS at budget 450.
    """
    df_prec = pd.read_csv(precision_csv_path)
    df_sbs = df_prec[(df_prec['algorithm'] == 'BFGS') & (df_prec['budget'] == 450)]

    # Sort by fid and iid for consistent ordering
    df_sbs = df_sbs.sort_values(['fid', 'iid']).reset_index(drop=True)

    print(f"✅ Loaded {len(df_sbs)} SBS precision entries from {precision_csv_path}")
    return df_sbs['precision'].values

def plot_precision_boxplots(result_csv_path, precision_csv_path, output_png="precision_boxplots.png"):
    """
    Generates boxplots of SBS, VBS, all static selectors, and selector.
    """
    # Load result file
    df = pd.read_csv(result_csv_path)

    # Extract SBS per run
    sbs_precisions = get_sbs_precisions(precision_csv_path)

    # Extract columns of interest
    static_cols = [col for col in df.columns if col.startswith("static_B")]
    selector_col = "selector_precision"
    vbs_col = "vbs_precision"

    # Prepare data and labels
    data = []
    labels = []

    # Selector first
    if selector_col in df.columns:
        data.append(df[selector_col].values)
        labels.append("Selector")
    else:
        print(f"⚠ Selector column {selector_col} not found in data.")

    # SBS next
    data.append(sbs_precisions)
    labels.append("SBS (BFGS_450)")

    # VBS
    if vbs_col in df.columns:
        data.append(df[vbs_col].values)
        labels.append("VBS")
    else:
        print(f"⚠ VBS column {vbs_col} not found in data.")

    # Static selectors
    for col in static_cols:
        data.append(df[col].values)
        labels.append(col.replace("static_", ""))  # shorter label

    # Plot
    plt.figure(figsize=(max(10, len(labels) * 0.6), 6))
    plt.boxplot(data, vert=True, patch_artist=True, showfliers=False)
    plt.xticks(ticks=np.arange(1, len(labels)+1), labels=labels, rotation=90)
    plt.ylabel("Reached Precision")
    plt.title("Boxplots of Reached Precisions: Selector, SBS, VBS, Statics")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"✅ Precision boxplots saved to {output_png}")



if __name__ == "__main__":
    result_csv = "../results/newInstances/selector_results_all_greater.csv"
    precision_csv = "../data/precision_files/A2_newInstances_precisions.csv"
    plot_precision_boxplots(result_csv, precision_csv)