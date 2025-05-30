# To each ELA file, add a column for each A2 algorithm. The entry in a column of one of the A2 algorithms
# is one if that algorithm achieves maximum precision for that switching point between all algorithms, otherwise zero.
import pandas as pd
import os

# --- CONFIGURATION ---
performance_file = "data/A2_precisions.csv"  # Replace with your actual filename
ela_folder = "data/ELA_over_budgets"  # Folder containing ELA files
ela_filename_template = "A1_B{budget}_5D_ela.csv"  # Pattern of ELA filenames

# --- LOAD PERFORMANCE DATA ---
perf_df = pd.read_csv(performance_file)
perf_df["rep"] = perf_df["rep"].astype(float)  # Ensure types match with ELA

# Get unique budgets and algorithms
budgets = perf_df["budget"].unique()
algorithms = perf_df["algorithm"].unique()

# --- PROCESS EACH BUDGET LEVEL ---
for budget in budgets:
    ela_path = os.path.join(ela_folder, ela_filename_template.format(budget=int(budget)))
    
    if not os.path.exists(ela_path):
        print(f"ELA file not found: {ela_path}")
        continue

    ela_df = pd.read_csv(ela_path)
    ela_df["rep"] = ela_df["rep"].astype(float)  # Make sure type matches

    # Add empty columns for each algorithm
    for algo in algorithms:
        if algo not in ela_df.columns:
            ela_df[algo] = ""

    # Process (fid, iid, rep) groups
    sub_df = perf_df[perf_df["budget"] == budget]
    for (fid, iid, rep), group in sub_df.groupby(["fid", "iid", "rep"]):
        min_prec = group["precision"].min()
        best_algos = group[group["precision"] == min_prec]["algorithm"]

        match = (
            (ela_df["fid"] == fid) &
            (ela_df["iid"] == float(iid)) &
            (ela_df["rep"] == float(rep))
        )
        for algo in algorithms:
            value = "1" if algo in best_algos.values else "0"
            ela_df.loc[match, algo] = value

    # Save updated ELA file
    output_path = os.path.join(ela_folder, f"{os.path.basename(ela_path)}_with_algo_perfomance.csv")
    ela_df.to_csv(output_path, index=False)
    print(f"Updated: {ela_path}")