import pandas as pd
import sys

# Usage: python check_nan.py my_file.csv

# Get file from command-line or hardcode it:
csv_file = sys.argv[1] if len(sys.argv) > 1 else "A1_B8_5D_ela_with_state.csv"

# Load CSV
df = pd.read_csv(csv_file)

# Check for NaNs
has_nan = df.isna().any().any()

print(f"File: {csv_file}")
print(f"Contains NaN: {has_nan}")

if has_nan:
    print("Number of NaNs per column:")
    print(df.isna().sum())
else:
    print("No NaNs found.")
