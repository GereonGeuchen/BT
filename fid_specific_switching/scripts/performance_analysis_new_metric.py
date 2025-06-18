import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

eps = np.finfo(float).eps  # Small value to avoid division by zero
sbs_sum = 7414.031202
vbs_sum = 2389,7533785635997


def compute_vbs_ratios(csv_path, fid=None):
    df = pd.read_csv(csv_path)

    if fid is not None:
        df = df[df["fid"] == fid]

    static_cols = [col for col in df.columns if col.startswith("static_B")]
    
    # vbs_sum = df["vbs_precision"].sum()
    method_sums = {col: df[col].sum() for col in static_cols}

    
    # Compute normalized scores (lower is better)
    # ratios = {
    #     col: (method_sums[col] - vbs_sum) / (sbs_sum - vbs_sum + eps)  # avoid zero-division
    #     for col in static_cols
    # }

    # return {"vbs_precision_ratios": ratios}
    return method_sums


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

def compute_raw_selector_totals_for_fid(csv_path, fid):
    df = pd.read_csv(csv_path)
    df = df[df["fid"] == fid]

    selector_cols = [col for col in df.columns if col.startswith("static_B") or col == "selector_precision"]
    totals = {col: df[col].sum() for col in selector_cols}

    return totals

def display_vbs_tables(csv_path, fid=None):
    if fid is None:
        os.makedirs("../results/new_metric", exist_ok=True)
        results = compute_vbs_ratios(csv_path)
        save_tables(
            results["vbs_precision_ratios"],
            "VBS Precision Ratios",
            "../results/new_metric/vbs_precision_ratios.png"
        )
    else:
        os.makedirs("../results/late_sp/fid_totals", exist_ok=True)
        totals = compute_raw_selector_totals_for_fid(csv_path, fid)
        save_tables(
            totals,
            f"Sum of Precisions (fid={fid})",
            f"../results/late_sp/fid_totals/precision_totals_fid_{fid}.png"
        )

def find_sbs(precision_csv_path):
    """
    Prints total precision per (budget, algorithm) pair.

    Args:
        precision_csv_path (str): Path to the precision CSV file.
    """
    df = pd.read_csv(precision_csv_path)

    grouped = (
        df.groupby(["budget", "algorithm"])["precision"]
        .sum()
        .reset_index()
        .sort_values("precision")
        .reset_index(drop=True)
    )

    print("Budget | Algorithm           | Total Precision")
    print("----------------------------------------------")
    for _, row in grouped.iterrows():
        print(f"{int(row['budget']):>6} | {row['algorithm']:<20} | {row['precision']:.6f}")


# Example usage
if __name__ == "__main__":
    # precision_path = "../data/A2_precisions.csv"  # Replace with your actual file path
    # result_path = "A2_results_updated.csv"
    # df = pd.read_csv(result_path)
    # vbs_sum = df["vbs_precision"].sum()
    # print(f"VBS Sum: {vbs_sum}")
    # # for fid in range(1, 25):
    # #     display_vbs_tables(result_path, fid)
    # # display_vbs_tables(result_path)
    for fid in range(1, 25):
        display_vbs_tables("../results/late_sp/predicted_static_precisions_rep_fold_late_sp.csv", fid=fid)