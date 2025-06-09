import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

eps = np.finfo(float).eps  # Small value to avoid division by zero

def compute_vbs_ratios(csv_path, fid = None):
    df = pd.read_csv(csv_path)

    if fid is not None:
        # Filter by fid if provided
        df = df[df["fid"] == fid]
    # Identify relevant static columns (exclude static_B1000)
    static_cols = [col for col in df.columns if col.startswith("static_B")]

    # Columns to evaluate: selector + static budgets
    eval_cols = ["selector_precision"] + static_cols

    # Initialize accumulators
    vbs_precision_ratios = {col: 0.0 for col in eval_cols}
    vbs_selector_ratios = {col: 0.0 for col in eval_cols}

    for _, row in df.iterrows():
        vbs_p = row["vbs_precision"] + eps
        vbs_s = row["vbs_selector"] + eps

        for col in eval_cols:
            denom = row[col] + eps # Avoid division by zero for VBS values
            # vbs_precision / col
            vbs_precision_ratios[col] += vbs_p / denom

            # vbs_selector / col
            vbs_selector_ratios[col] += vbs_s / denom  # Avoid division by zero

    # Normalize to average
    n = len(df)
    vbs_precision_avg = {col: vbs_precision_ratios[col] / n for col in eval_cols}
    if fid is None: vbs_precision_avg["sbs (200, BFGS)"] = 0.380002
    vbs_selector_avg = {col: vbs_selector_ratios[col] / n for col in eval_cols}

    return {
        "vbs_precision_ratios": vbs_precision_avg,
        "vbs_selector_ratios": vbs_selector_avg
    }

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

    # Step 1: Compute VBS per (fid, iid, rep)
    vbs_df = (
        df.groupby(["fid", "iid", "rep"])["precision"]
        .min()
        .reset_index()
        .rename(columns={"precision": "vbs"})
    )

    # Step 2: Merge VBS into original DataFrame
    df = df.merge(vbs_df, on=["fid", "iid", "rep"], how="left")

    # Step 3: Compute relative score (vbs / precision)
    df["relative_score"] = ( df["vbs"] + eps ) / ( df["precision"] + eps )
    # df.to_csv("test.csv", index=False)  # Save intermediate results for debugging
    # Step 4: Average relative score per (budget, algorithm)
    score_table = (
        df.groupby(["budget", "algorithm"])["relative_score"]
        .mean()
        .reset_index()
        .sort_values("relative_score")
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

def display_vbs_tables(csv_path, fid = None):
    results = compute_vbs_ratios(csv_path, fid)
    if fid is None:
        save_tables(results["vbs_precision_ratios"], "VBS Precision Ratios", "../results/vbs_precision_ratios.png")
        # save_tables(results["vbs_selector_ratios"], "VBS Selector Ratios", "../results/vbs_selector_ratios.png")
    else:
        os.makedirs(f"../results/vbs_precision_ratios_fid", exist_ok=True)
        # os.makedirs(f"../results/vbs_selector_ratios_fid", exist_ok=True)
        save_tables(results["vbs_precision_ratios"], f"VBS Precision Ratios (fid={fid})", f"../results/vbs_precision_ratios_fid/vbs_precision_ratios_fid_{fid}.png")
        # save_tables(results["vbs_selector_ratios"], f"VBS Selector Ratios (fid={fid})", f"../results/vbs_selector_ratios_fid/vbs_selector_ratios_fid_{fid}.png")

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
    precision_path = "../data/A2_precisions_test.csv"  # Replace with your actual file path
    result_path = "../results/result_csvs/selector_results_with_vbs_selector.csv"
    res = find_sbs(precision_path)
    for i, row in res.iterrows():
        print(f"{row['budget']:>3} | {row['algorithm']:<20} | {row['relative_score']:.6f}")
    display_vbs_tables(result_path)
    # for fid in range(1, 25):
    #     display_vbs_tables(result_path, fid)
    # # print(compute_budget_specific_selector_ratio(result_path, precision_path, budget=150))
    # # display_vbs_tables(result_path)
    # # res = find_sbs(precision_path)
    # # for i, row in res.iterrows():
    # #     print(f"{row['budget']:>3} | {row['algorithm']:<20} | {row['relative_score']:.6f}")
    # # sbs = (50, "Non-elitist", 0.244517)
    # plot_switching_point_comparison(result_path, precision_path, output_dir="../results/switching_point_plots")
    # # plot_selector_budget_counts(result_path, output_png="../results/selector_budget_counts.png")