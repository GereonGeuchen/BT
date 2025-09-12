import os
import pandas as pd

# === Path to folder with your input CSV files ===
input_folder = "A1_data_ela_cma_std_testSet_normalized"

# === Path to folder for cleaned CSVs ===
output_folder = os.path.join(input_folder, "no_ps_ratio")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    df = pd.read_csv(input_path)

    if "ps_ratio" in df.columns:
        print(f"Removing 'ps_ratio' from {filename}")
        df = df.drop(columns=["ps_ratio"])
    else:
        print(f"No 'ps_ratio' in {filename}")

    # Save into new folder
    df.to_csv(output_path, index=False)

print(f"\n✅ Cleaned files saved in: {output_folder}")
