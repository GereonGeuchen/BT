import pandas as pd
import numpy as np
import sys

def compare_sorted_csvs(file1, file2, tol=1e-12):
    """
    Compares two already-sorted CSV files for exact or near-equal match.

    Args:
        file1 (str): Path to first CSV.
        file2 (str): Path to second CSV.
        tol (float): Tolerance for numeric differences.

    Returns:
        bool: True if the files match, False otherwise.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1 = df1.loc[:, ~df1.columns.str.endswith('.costs_runtime')]
    df2 = df2.loc[:, ~df2.columns.str.endswith('.costs_runtime')]
    if df1.shape != df2.shape:
        print(f"❌ Shape mismatch: {df1.shape} vs {df2.shape}")
        return False

    numeric_diff_summary = {}

    for col in df1.columns:
        if col not in df2.columns:
            print(f"❌ Column '{col}' not in both files.")
            continue

        s1 = df1[col]
        s2 = df2[col]

        if np.issubdtype(s1.dtype, np.number):
            unequal_mask = ~np.isclose(s1, s2, rtol=tol, atol=tol, equal_nan=True)
            diffs = np.abs(s1[unequal_mask] - s2[unequal_mask])
            if not diffs.empty:
                numeric_diff_summary[col] = {
                    "count": len(diffs),
                    "mean_diff": diffs.mean(),
                    "max_diff": diffs.max(),
                }

    if not numeric_diff_summary:
        print("✅ No numeric differences beyond tolerance.")
    else:
        print("❌ Average differences in numeric columns:")
        for col, stats in numeric_diff_summary.items():
            print(f"- {col}: count = {stats['count']}, mean_diff = {stats['mean_diff']:.3e}, max_diff = {stats['max_diff']:.3e}")

if __name__ == "__main__":
    equal = compare_sorted_csvs("../A1_data_ela_early_switching/A1_B8_5D_ela.csv", "../A1_data_ela_early_switching_test/A1_B8_5D_ela.csv")