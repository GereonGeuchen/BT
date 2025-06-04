# import os
# import pandas as pd

# # === CONFIG ===
# precision_path = "A2_optimal_precisions.csv"              # Your precision CSV
# ela_dir = "A1_data_ela"                                  # Folder with A1_B{budget}_5D_ela.csv
# output_dir = "A2_data_ela_with_upcoming_precs"                     # Output folder
# os.makedirs(output_dir, exist_ok=True)

# # === Load and prepare precision data ===
# precision_df = pd.read_csv(precision_path)
# precision_df["switching_point"] = precision_df["budget"].rank(method="dense").astype(int)

# # Map: (fid, iid, rep, sp) → precision
# precision_lookup = precision_df.set_index(["fid", "iid", "rep", "switching_point"])["precision"]

# # === Augment each ELA file ===
# for sp in range(1, 21):
#     budget = sp * 50
#     filename = f"A1_B{budget}_5D_ela.csv"
#     filepath = os.path.join(ela_dir, filename)

#     if not os.path.exists(filepath):
#         print(f"Skipping missing file: {filename}")
#         continue

#     # Load ELA features
#     ela_df = pd.read_csv(filepath)
#     ela_df["fid"] = ela_df["fid"].astype(int)
#     ela_df["iid"] = ela_df["iid"].astype(int)
#     ela_df["rep"] = ela_df["rep"].astype(int)

#     # Add precision columns for sp+1 to sp20
#     for future_sp in range(sp + 1, 21):
#         col = f"sp{future_sp * 50}"
#         ela_df[col] = ela_df.apply(
#             lambda row: precision_lookup.get((row["fid"], row["iid"], row["rep"], future_sp), float("nan")),
#             axis=1
#         )

#     # Save result
#     out_path = os.path.join(output_dir, filename)
#     ela_df.to_csv(out_path, index=False)
#     print(f"Saved: {out_path}")

import os
import pandas as pd

# === CONFIG ===
performance_file = "A2_precisions.csv"     # File with all algo rows (example you posted)
ela_dir = "A1_data_ela"                                  # Folder with ELA files
output_dir = "A1_dat_ela_with_algo_performance"          # Output folder
os.makedirs(output_dir, exist_ok=True)

# === LOAD PERFORMANCE DATA ===
perf_df = pd.read_csv(performance_file)

# Pivot: (fid, iid, rep, budget) → {algorithm: precision}
perf_wide = perf_df.pivot_table(
    index=["fid", "iid", "rep", "budget"],
    columns="algorithm",
    values="precision"
).reset_index()

# === PROCESS EACH ELA FILE ===
for budget in range(50, 1001, 50):
    filename = f"A1_B{budget}_5D_ela.csv"
    filepath = os.path.join(ela_dir, filename)
    if not os.path.exists(filepath):
        print(f"Missing file: {filename}")
        continue

    # Load ELA features
    ela_df = pd.read_csv(filepath)
    ela_df["fid"] = ela_df["fid"].astype(int)
    ela_df["iid"] = ela_df["iid"].astype(int)
    ela_df["rep"] = ela_df["rep"].astype(int)
    ela_df["budget"] = budget  # inject budget for merge

    # Merge ELA features with per-algorithm precision
    merged_df = pd.merge(ela_df, perf_wide, on=["fid", "iid", "rep", "budget"], how="left")

    # Save
    out_path = os.path.join(output_dir, filename)
    merged_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
