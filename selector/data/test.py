import pandas as pd

# Load data
df = pd.read_csv("A2_precisions.csv")

# Group by switching point key and keep the first row with the minimum precision
best_rows = (
    df.loc[df.groupby(["fid", "iid", "rep", "budget"])["precision"].idxmin()]
    .reset_index(drop=True)
)

# Optional: Keep only desired columns
best_rows = best_rows[["fid", "iid", "rep", "budget", "precision"]]

# Save result
best_rows.to_csv("A2_optimal_precisions.csv", index=False)

print("Saved one best entry per (id, iid, rep, budget).")