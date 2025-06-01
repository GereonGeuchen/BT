import pandas as pd

# Load the two CSVs
df1 = pd.read_csv("A1_B50_5D_ela.csv")
df2 = pd.read_csv("A1_B50_5D_ela_copy.csv")

# Columns that do not match
non_matching = [
    "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10",
    "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10",
    "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
    "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10",
    "ela_distr.kurtosis", "ela_distr.skewness", "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_simple.coef.max", "ela_meta.lin_simple.coef.max_by_min",
    "ela_meta.lin_simple.coef.min", "ela_meta.lin_simple.intercept",
    "ela_meta.lin_w_interact.adj_r2", "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_simple.cond", "nbc.nb_fitness.cor"
]

# For each column, count how many rows differ
print(f"{'Column':50} | {'Mismatched rows'}")
print("-" * 70)
for col in non_matching:
    if col in df1.columns and col in df2.columns:
        mismatches = (df1[col] != df2[col]) & ~(df1[col].isna() & df2[col].isna())
        count = mismatches.sum()
        if count > 0:
            abs_diff = (df1[col] - df2[col]).abs()[mismatches]
            mean_diff = abs_diff.mean()
            print(f"{col:50} | {count:17} | {mean_diff}")
        else:
            print(f"{col:50} | {count:17} | {'0.0000000000 :)':>14}")
    else:
        print(f"{col:50} | {'Column not found':>17} | {'-':>14}")
