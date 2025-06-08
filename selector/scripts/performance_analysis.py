import pandas as pd
import numpy as np

def compute_vbs_ratios(csv_path):
    eps = 1e-12
    df = pd.read_csv(csv_path)

    # Identify relevant static columns (exclude static_B1000)
    static_cols = [col for col in df.columns if col.startswith("static_B") and col != "static_B1000"]

    # Columns to evaluate: selector + static budgets
    eval_cols = ["selector_precision"] + static_cols

    # Initialize accumulators
    vbs_precision_ratios = {col: 0.0 for col in eval_cols}
    vbs_selector_ratios = {col: 0.0 for col in eval_cols}

    for _, row in df.iterrows():
        vbs_p = row["vbs_precision"]
        vbs_s = row["vbs_selector"]

        for col in eval_cols:
            denom = row[col]
            denom_safe = denom if denom != 0 else eps

            # vbs_precision / col
            vbs_precision_ratios[col] += vbs_p / denom_safe

            # vbs_selector / col
            vbs_selector_ratios[col] += vbs_s / denom_safe

    # Normalize to average
    n = 960
    vbs_precision_avg = {col: vbs_precision_ratios[col] / n for col in eval_cols}
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

def update_vbs_selector_and_staticB1000(csv_path, precision_file, output_path=None):
    df = pd.read_csv(csv_path)
    prec_df = pd.read_csv(precision_file)

    # 1. Compute vbs_selector from columns starting at index 5
    df["vbs_selector"] = df.iloc[:, 5:].min(axis=1)

    # 2. Lookup correct static_B1000 values from precision file
    lookup = prec_df[
        (prec_df["budget"] == 1000) & (prec_df["algorithm"] == "Same")
    ][["fid", "iid", "rep", "precision"]]
    lookup = lookup.rename(columns={"precision": "static_B1000"})

    # 3. Merge to get correct values
    df = df.drop(columns=["static_B1000"], errors="ignore")  # Remove old if exists
    df = pd.merge(df, lookup, on=["fid", "iid", "rep"], how="left")

    # 4. Save
    static_cols = [col for col in df.columns if col.startswith("static_B") and col != "static_B1000"]
    static_cols = sorted(static_cols, key=lambda c: int(c.split("_B")[1]))

    # Final column order
    new_order = [
        "fid", "iid", "rep",
        "selector_precision", "selector_switch_budget",
        "vbs_precision", "vbs_selector"
    ] + static_cols + ["static_B1000"]

    # Add any remaining columns not explicitly ordered
    remaining = [col for col in df.columns if col not in new_order]
    df = df[new_order + remaining]

    # 5. Save updated CSV
    output_path = output_path or csv_path
    df.to_csv(output_path, index=False)
    print(f"✅ Updated with sorted static columns, saved to: {output_path}")
   
def compute_budget_specific_selector_ratio(main_csv, precision_csv, budget=150):
    eps = 1e-12

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
    merged["ratio"] = merged["vbs_budget"] / denom

    # Average
    avg_ratio = merged["ratio"].mean()
    return avg_ratio

def find_sbs(path, eps=1e-12):
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
    df["relative_score"] = df["vbs"] / df["precision"].replace(0, eps)
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

# Example usage
if __name__ == "__main__":
    path = "../data/results/selector_results_with_vbs_selector.csv"  # Replace with your actual file path
    # add_vbs_selector_column(path, "../data/results/selector_results_with_vbs_selector.csv")
    # result = sum_selected_columns(path)
    # for col, value in result.items():
    #     print(f"{col}: {value}")
    ratios = compute_vbs_ratios(path)
    print("VBS Precision Ratios:")
    for col, ratio in ratios["vbs_precision_ratios"].items():
        print(f"{col}: {ratio:.4f}")
    print("\nVBS Selector Ratios:")
    for col, ratio in ratios["vbs_selector_ratios"].items():
        print(f"{col}: {ratio:.4f}")
    # for budget in [50*i for i in range(1, 21)]:
    #     result = compute_budget_specific_selector_ratio(
    #         path,
    #         "../data/A2_precisions_test.csv",
    #         budget=budget
    #     )
    #     print(f"VBS to Selector Precision Ratio for budget {budget}: {result:.4f}")
    sbs = (50, "Non-elitist", 0.244517)