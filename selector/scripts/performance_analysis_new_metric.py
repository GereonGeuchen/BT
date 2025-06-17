import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

eps = np.finfo(float).eps  # Small value to avoid division by zero
# sbs_sum = 2690.130170 # For iid 6,7
sbs_sum = 3306.1779694383 # For new reps, all switching points, Non-elitist budget 16
# sbs_sum = 3511.967 # For new reps, late switching points, 50 Non-elitist

def compute_vbs_ratios(csv_path, fid = None):

    df = pd.read_csv(csv_path)
    vbs_sum = df["vbs_precision"].sum()
    consider_cols = [col for col in df.columns if col.startswith("static_B") or col == "selector_precision"]
    res = {}
    for col in consider_cols:
        col_sum = df[col].sum()
        print(f"Processing column: {col}, sum: {col_sum}")
        res[col] = (col_sum - vbs_sum) / (sbs_sum - vbs_sum) 
        res[col] = col_sum
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
        output_path = "../results/new_Reps/all_sp/total_precisions.png"
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

    ax.set_title("Frequency of Selector Switch Budgets", fontsize=14)
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

def plot_switching_point_comparison(selector_csv, precision_csv, output_dir="../results/switching_point_plots"):
    os.makedirs(output_dir, exist_ok=True)

    selector_df = pd.read_csv(selector_csv)
    precision_df = pd.read_csv(precision_csv)

    precision_df["precision"] = pd.to_numeric(precision_df["precision"], errors="coerce")

    for fid in sorted(selector_df["fid"].unique()):
        for iid in sorted(selector_df["iid"].unique()):
            fig, ax = plt.subplots(figsize=(10, 5))
            first_blue, first_red, first_green = True, True, True

            for rep in range(20):
                # --- Get predicted switching point ---
                row = selector_df[
                    (selector_df["fid"] == fid) & 
                    (selector_df["iid"] == iid) & 
                    (selector_df["rep"] == rep)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                predicted = int(row["selector_switch_budget"])

                # --- Get optimal switching budgets ---
                prec_block = precision_df[
                    (precision_df["fid"] == fid) & 
                    (precision_df["iid"] == iid) & 
                    (precision_df["rep"] == rep)
                ]
                if prec_block.empty:
                    continue
                min_precision = prec_block["precision"].min()
                optimal_budgets = prec_block[prec_block["precision"] == min_precision]["budget"].astype(int).tolist()

                # --- Get all best static switcher budgets ---
                static_cols = [col for col in selector_df.columns if col.startswith("static_B")]
                static_vals = row[static_cols].astype(float)
                min_static_val = static_vals.min()
                best_static_budgets = [
                    int(col.split("_B")[1]) for col in static_vals.index if static_vals[col] == min_static_val
                ]

                # --- Plot all points with jitter ---
                jitter = 0.15

                if first_blue:
                    ax.plot(rep + jitter, predicted, 'bo', label="Predicted")
                    first_blue = False
                else:
                    ax.plot(rep + jitter, predicted, 'bo')

                for b in optimal_budgets:
                    if first_red:
                        ax.plot(rep - jitter, b, 'ro', label="Optimal")
                        first_red = False
                    else:
                        ax.plot(rep - jitter, b, 'ro')

                for b in best_static_budgets:
                    if first_green:
                        ax.plot(rep, b, 'go', label="Best Static")
                        first_green = False
                    else:
                        ax.plot(rep, b, 'go')

            ax.set_title(f"Switching Budgets: fid={fid}, iid={iid}", fontsize=14)
            ax.set_xlabel("Repetition", fontsize=12)
            ax.set_ylabel("Budget", fontsize=12)
            ax.set_xticks(range(20))
            ax.set_ylim(0, 1050)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()

            plt.tight_layout()
            filename = f"{output_dir}/switching_points_fid{fid}_iid{iid}.png"
            plt.savefig(filename)
            plt.close()
            print(f"✅ Saved {filename}")



# Example usage
if __name__ == "__main__":
    precision_path1 = "../data/precision_files/A2_late_precisions_newReps.csv"  # Replace with your actual file path
    precision_path2 = "../data/precision_files/A2_all_precisions_newReps.csv"  # Replace with your actual file path
    result_csv1 = "../results/new_Reps/late_sp/selector_results_newReps_late.csv"
    result_csv2 = "../results/new_Reps/all_sp/selector_results_newReps_all.csv"
    # res = find_sbs(precision_path2)
    # for index, row in res.iterrows():
    #     print(f"Budget: {row['budget']}, Algorithm: {row['algorithm']}, Precision sum: {row['precision']}")    
    # compute_vbs_ratios(result_csv2)
    # res = find_sbs(precision_path)
    # for index, row in res.iterrows():
    #     print(f"Budget: {row['budget']}, Algorithm: {row['algorithm']}, Precision sum: {row['precision']}")
    # for fid in range(1, 25):
    #     print(f"Processing fid: {fid}")
    #     display_vbs_tables(result_csv1, fid=fid)
    # for fid in range(1, 25):
    #     display_vbs_tables(result_csv2, fid=fid)
    # for fid in range(1, 25):
    #     print(f"Processing fid: {fid}")
    #     display_vbs_tables(result_csv2, fid=fid)
    display_vbs_tables(result_csv2)