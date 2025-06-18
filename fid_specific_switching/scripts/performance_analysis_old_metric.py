import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

eps = np.finfo(float).eps  # Small value to avoid division by zero

def compute_vbs_ratios(csv_path, fid=None):
    df = pd.read_csv(csv_path)

    if fid is not None:
        df = df[df["fid"] == fid]

    static_cols = [col for col in df.columns if col.startswith("static_B")]

    vbs_precision_ratios = {col: 0.0 for col in static_cols}

    for _, row in df.iterrows():
        vbs_p = row["vbs_precision"]

        for col in static_cols:
            denom = row[col]
            if denom == 0:
                denom = eps
                vbs_p = eps
            vbs_precision_ratios[col] += vbs_p / denom

    n = len(df)
    vbs_precision_avg = {col: vbs_precision_ratios[col] / n for col in static_cols}

    return {"vbs_precision_ratios": vbs_precision_avg}



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

def display_vbs_tables(csv_path, fid=None):
    results = compute_vbs_ratios(csv_path, fid)
    if fid is None:
        save_tables(results["vbs_precision_ratios"], "VBS Precision Ratios", "../results/vbs_precision_ratios.png")
    else:
        os.makedirs("../results/vbs_precision_ratios_fid", exist_ok=True)
        save_tables(results["vbs_precision_ratios"],
                    f"VBS Precision Ratios (fid={fid})",
                    f"../results/vbs_precision_ratios_fid/vbs_precision_ratios_fid_{fid}.png")

# Example usage
if __name__ == "__main__":
    precision_path = "../data/A2_precisions.csv"  # Replace with your actual file path
    result_path = "A2_results_updated.csv"
    # for fid in range(1, 25):
    #     display_vbs_tables(result_path, fid)
    display_vbs_tables(result_path)
    # print(compute_budget_specific_selector_ratio(result_path, precision_path, budget=150))
    # display_vbs_tables(result_path)
    # res = find_sbs(precision_path)
    # for i, row in res.iterrows():
    #     print(f"{row['budget']:>3} | {row['algorithm']:<20} | {row['relative_score']:.6f}")
    # sbs = (50, "Non-elitist", 0.244517)