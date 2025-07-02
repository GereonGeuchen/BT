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
    df = df.sort_values("Ratio", ascending=False)

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

def save_barplot(data, title, filename):
    """
    Saves a vertical bar plot of the given data.
    Includes only specified columns and a fixed SBS bar with value 0.
    Sorted by value descending: highest left, lowest right.
    """

    # Define desired columns
    desired_cols = [
        "selector_precision",
        "static_B64",
        "static_B80",
        "static_B96",
        "static_B56",
        "static_B48",
        "static_B150",
        "static_B18",
        "static_B8",
        "static_B800"
    ]

    # Convert data dict to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Method", "Value"])

    # Filter to desired columns only
    df = df[df["Method"].isin(desired_cols)]

    # Add SBS row with value 0.000
    sbs_row = pd.DataFrame({"Method": ["SBS (BFGS, 450)"], "Value": [0.000]})
    df = pd.concat([df, sbs_row], ignore_index=True)

    # Sort by Value descending: highest left, lowest right
    df = df.sort_values("Value", ascending=False).reset_index(drop=True)

    # Plotting
    plt.figure(figsize=(0.5 * len(df) + 2, 6))
    bars = plt.bar(df["Method"], df["Value"], color="skyblue", edgecolor="black")

    # Annotate bars with their values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.3f}",
                 ha='center', va='bottom', fontsize=9)

    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"✅ Vertical bar plot saved to {filename}")


def display_vbs_tables(csv_path, fid=None, plot_type="table"):
    """
    Displays VBS results as either table or bar plot.

    plot_type: "table" or "bar"
    """
    if fid is None:
        ratios = compute_vbs_ratios(csv_path)
        output_path = "../results/newInstances/precision_ratios_all_bar.pdf"
        if plot_type == "table":
            save_tables(ratios, "VBS Relative Ratios", output_path)
            print("✅ VBS ratios table saved to:", output_path)
        elif plot_type == "bar":
            save_barplot(ratios, "VBS Relative Ratios", output_path)
            print("✅ VBS ratios bar plot saved to:", output_path)
        else:
            raise ValueError("Invalid plot_type. Choose 'table' or 'bar'.")
    else:
        totals = compute_total_precisions_for_fid(csv_path, fid)
        output_dir = "../data/switching_optimality_files/vbs_precision_totals_late"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/vbs_precision_totals_fid_{fid}.png"
        if plot_type == "table":
            save_tables(totals, f"Sum of Precisions (fid={fid})", output_path)
            print(f"✅ Precision totals table saved to: {output_path}")
        elif plot_type == "bar":
            save_barplot(totals, f"Sum of Precisions (fid={fid})", output_path)
            print(f"✅ Precision totals bar plot saved to: {output_path}")
        else:
            raise ValueError("Invalid plot_type. Choose 'table' or 'bar'.")

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

    static = df[f'static_B64'].values

    # Compute observed mean difference
    observed_diff = np.mean(selector - static)
    print(f"Observed mean difference: {observed_diff:.6f}")

    # Use scipy's permutation_test
    res = permutation_test(
        (selector, static),
        statistic=lambda x, y: np.mean(x - y),
        permutation_type='samples',
        vectorized=False,
        n_resamples=10000,
        alternative='less',
        random_state=42
    )

    print(f"P-value (selector < static) for budget 64: {res.pvalue:.6f}")

    # Extract permutation distribution (available in res.distribution)
    perm_distribution = res.null_distribution

    # Plot
    plt.figure(figsize=(8,5))
    plt.hist(perm_distribution, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(observed_diff, color='red', linestyle='--', label=f'Observed diff = {observed_diff:.4f}')
    plt.xlabel('Permutation test statistic')
    plt.ylabel('Frequency')
    plt.title('Permutation Null Distribution (Selector vs Static B64)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("permutation_test_selector_vs_static.png")


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

def plot_precision_boxplots(result_csv_path, precision_csv_path, output_png="precision_boxplots_low_budgets.pdf"):
    """
    Generates boxplots of SBS, VBS, all static selectors, and selector.
    """
    # Load result file
    df = pd.read_csv(result_csv_path)

    # Extract SBS per run (assuming your get_sbs_precisions function is defined elsewhere)
    sbs_precisions = get_sbs_precisions(precision_csv_path)

    # Extract columns of interest
    static_cols = [col for col in df.columns if col.startswith("static_B") 
                   and (col not in ["static_B800", "static_B850", "static_B900",
                                    "static_B950", "static_B1000"])]
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
    data.append(np.array(sbs_precisions))
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
    box = plt.boxplot(data, vert=True, patch_artist=True, showfliers=False)
    plt.yscale("log")

    # Overlay means as prominent red circles with black edges
    for median in box['medians']:
        median.set(linewidth=3, color='orange')

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
    # display_vbs_tables(result_csv, fid=None, plot_type="bar")
    permutation_test_selector_vs_static(result_csv)