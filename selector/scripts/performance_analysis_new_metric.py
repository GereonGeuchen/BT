import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

eps = np.finfo(float).eps  # Small value to avoid division by zero
sbs_sum = 2690.13 # For iid 6,7
# sbs_sum = 3306.18 # For new reps, all switching points, Non-elitist budget 16
# sbs_sum = 3511.97 # For new reps, late switching points, 50 Non-elitist

def compute_vbs_ratios(csv_path, fid = None):

    df = pd.read_csv(csv_path)
    vbs_sum = df["vbs_precision"].sum()
    consider_cols = [col for col in df.columns if col.startswith("static_B") or col == "selector_precision"]
    # consider_cols = ["selector_precision"] + ["static_B64"] + ["static_B16"] + ["static_B8"]
    res = {}
    for col in consider_cols:
        col_sum = df[col].sum()
        print(f"Processing column: {col}, sum: {col_sum}")
        res[col] = (sbs_sum - col_sum) / (sbs_sum - vbs_sum) 
    # res["selector_50"] = (selector_50_sum - vbs_sum) / (sbs_sum - vbs_sum)
    
    return res

def sum_selected_columns(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Determine column indices to include: start from 3, skip 5
    cols_to_sum = [i for i in range(3, len(df.columns)) if i != 5]

    # Get corresponding column names
    selected_cols = df.iloc[:, cols_to_sum]

    # Sum values in the selected columns
    total_sum = selected_cols.sum()

    return total_sum

   
def compute_budget_specific_selector_ratio(main_csv, precision_csv, budget=150):

    df = pd.read_csv(main_csv)
    prec_df = pd.read_csv(precision_csv)

    # Get VBS at budget 150 for each (fid, iid, rep)
    vbs_budget = (
        prec_df[prec_df["budget"] == budget]
        .groupby(["fid", "iid", "rep"])["precision"]
        .min()
        .reset_index()
        .rename(columns={"precision": "vbs_budget"})
    )

    # Merge with main result file
    merged = pd.merge(df, vbs_budget, on=["fid", "iid", "rep"], how="inner")

    # Compute ratio: vbs150 / selector_precision
    denom = merged[f"static_B{budget}"].replace(0, eps)
    numerator = merged["vbs_budget"].replace(0, eps)
    merged["ratio"] = numerator / denom

    # Average
    avg_ratio = merged["ratio"].mean()
    return avg_ratio

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
    # plt.title(title, fontsize=14, pad=10)
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


    # Convert data dict to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Method", "Value"])
    df["Method"] = df["Method"].replace({"selector_precision": "selector"})

    # Add SBS row with value 0.000
    sbs_row = pd.DataFrame({"Method": ["SBS (BFGS, 450)"], "Value": [0.000]})
    df = pd.concat([df, sbs_row], ignore_index=True)

    # Sort by Value descending: highest left, lowest right
    df = df.sort_values("Value", ascending=False).reset_index(drop=True)

    # Plotting
    plt.figure(figsize=(2.0 * len(df) + 2, 6))
    bars = plt.bar(df["Method"], df["Value"], color="skyblue", edgecolor="black")

    # Annotate bars with their values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.3f}",
                 ha='center', va='bottom', fontsize=15)

    plt.ylabel("Precision Ratio", fontsize=15)
    plt.xticks(fontsize=15)
    plt.title(title, fontsize=15, pad=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"✅ Vertical bar plot saved to {filename}")

def display_vbs_tables(csv_path,bar_plot = False, fid=None):
    if fid is None:
        ratios = compute_vbs_ratios(csv_path)
        output_path = "../results/new_Instances/all_sp/precision_ratios.pdf"
        if bar_plot:
            save_barplot(ratios, "VBS Ratios", output_path)
        else:
            save_tables(ratios, "VBS Relative Ratios", output_path)
    else:
        totals = compute_total_precisions_for_fid(csv_path, fid)
        output_dir = "../results/new_Reps/all_sp/vbs_precision_totals_fid_correct"
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

# Example usage
if __name__ == "__main__":
    newInstances_precisions = "../data/precision_files/A2_newInstances_precisions.csv"
    newInstances_precisions_late = "../data/precision_files/A2_newInstances_late_precisions.csv"
    newReps_precisions = "../data/precision_files/A2_newReps_precisions.csv"
    newReps_precisions_late = "../data/precision_files/A2_newReps_late_precisions.csv"
    results_newInstances = "../results/new_Instances/all_sp/selector_results_newInstances_all.csv"
    results_newInstances_late = "../results/new_Instances/late_sp/selector_results_newInstances_late.csv"
    results_newReps = "../results/new_Reps/all_sp/selector_results_newReps_all.csv"
    results_newReps_late = "../results/new_Reps/late_sp/selector_results_newReps_late.csv"

    display_vbs_tables(results_newInstances, bar_plot=False)