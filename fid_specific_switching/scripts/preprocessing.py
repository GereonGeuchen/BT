# File for preprocssing the data. We need:
# ELA feature files with internal algorithm state at that switching point
# These files for selecting the switching point: Columns with upcoming switching points and their precisions
# These files for selecting the algorithm: Columns for each algorithm with its precision

# Function that adds columns for the internal algorithm state at the switching point
import os
import pandas as pd
from glob import glob
import warnings
from pathlib import Path

def extract_final_internal_state(dat_path, target_iid, target_rep):
    try:
        # Read and clean repeated headers
        with open(dat_path, "r") as f:
            lines = f.readlines()

        cleaned_lines = []
        header_seen = False
        for line in lines:
            if line.strip().startswith("evaluations"):
                if not header_seen:
                    cleaned_lines.append(line)
                    header_seen = True
                # else: skip repeated headers
            else:
                cleaned_lines.append(line)

        # Parse cleaned content into DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO("".join(cleaned_lines)), delim_whitespace=True)

        # Convert iid and rep to int for matching
        df["rep"] = pd.to_numeric(df["rep"], errors="coerce").astype(int)
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype(int)
        target_rep = int(target_rep)
        target_iid = int(target_iid)

        # Filter for target rep and iid
        df = df[(df["rep"] == target_rep) & (df["iid"] == target_iid)]
        if df.empty:
            return None

        # Get final row (maximum evaluations)
        final_row = df.loc[df["evaluations"].idxmax()]
        return final_row.loc["sigma":"mhl_sum"].to_dict()

    except Exception as e:
        print(f"Error processing {dat_path}: {e}")
        return None
    
def append_cma_state_to_ela(ela_dir, run_dir, output_dir):
    budgets = [8*i for i in range(1, 13)]  # 8, 16, ..., 96
    budgets += [50*i for i in range(1, 20)]  #
    os.makedirs(output_dir, exist_ok=True)

    for budget in budgets:
    
        print(f"Processing budget: {budget}")
        ela_path = os.path.join(ela_dir, f"A1_B{budget}_5D_ela.csv")
        run_path = os.path.join(run_dir, f"A1_B{budget}_5D")

        if not os.path.isfile(ela_path):
            print(f"Skipping: {ela_path} not found.")
            continue
        if not os.path.isdir(run_path):
            print(f"Skipping: {run_path} not found.")
            continue

        df_ela = pd.read_csv(ela_path)
        df_ela["iid"] = df_ela["iid"].astype(int)
        df_ela["rep"] = df_ela["rep"].astype(int)

        appended_data = []
        for _, row in df_ela.iterrows():
            fid, iid, rep = int(row["fid"]), int(row["iid"]), int(row["rep"])
            pattern = os.path.join(run_path, f"data_f{fid}*", f"IOHprofiler_f{fid}_DIM5.dat")
            dat_files = glob(pattern)
            if not dat_files:
                print(f"No matching file for fid={fid}, iid={iid}, rep={rep} at budget {budget}")
                appended_data.append({})  # empty row for failed match
                continue

            state = extract_final_internal_state(dat_files[0], iid, rep)
            if state is None:
                print(f"  ✘ No state found in {dat_files[0]} for iid={iid}, rep={rep}")
                state = {}
            appended_data.append(state)

        df_state = pd.DataFrame(appended_data)
        df_combined = pd.concat([df_ela.reset_index(drop=True), df_state.reset_index(drop=True)], axis=1)

        out_path = os.path.join(output_dir, f"A1_B{budget}_5D_ela_with_state.csv")
        df_combined.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

# Function to get a file with the optimal performance per switching point, not the performance of all algorithms
def extract_min_precision_rows(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Group by fid, iid, rep, budget and find the row with minimum precision
    min_precision_df = (
        df.loc[df.groupby(['fid', 'iid', 'rep', 'budget'])['precision'].idxmin()]
        .sort_values(['fid', 'iid', 'rep', 'budget'])
        .reset_index(drop=True)
    )

    # Keep only the required columns
    return min_precision_df[['fid', 'iid', 'rep', 'budget', 'precision']]

def add_algorithm_precisions(ela_dir, precision_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load the full precision table
    precision_df = pd.read_csv(precision_csv)

    # Create a pivot for fast lookup: (fid, iid, rep, budget) → columns = algorithms
    precision_pivot = precision_df.pivot_table(
        index=['fid', 'iid', 'rep', 'budget'],
        columns='algorithm',
        values='precision'
    ).reset_index()

    # Iterate over ELA files
    for file in os.listdir(ela_dir):
        if not file.endswith('.csv'):
            continue

        ela_path = os.path.join(ela_dir, file)
        ela_df = pd.read_csv(ela_path)

        # Extract budget from filename
        budget = int(file.split('_')[1][1:])  # e.g. B50 → 50

        # Add budget column for merging
        ela_df['budget'] = budget

        # Merge on fid, iid, rep, budget
        merged = pd.merge(
            ela_df,
            precision_pivot,
            how='left',
            on=['fid', 'iid', 'rep', 'budget']
        )

        merged.drop(columns=['budget'], inplace=True)  # Remove budget column after merge
        # Write to output directory
        output_path = os.path.join(output_dir, file)
        merged.to_csv(output_path, index=False)

        print(f"Wrote {output_path}")

def extend_ela_with_optimal_precisions(
    ela_input_dir,
    optimal_precisions_file,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    # Load the optimal precisions once
    opt_df = pd.read_csv(optimal_precisions_file)

    # List all ELA files
    for ela_file in sorted(os.listdir(ela_input_dir)):
        if not ela_file.endswith(".csv"):
            continue

        # Extract budget from filename (e.g., A1_B50_5D_ela.csv → 50)
        budget_str = ela_file.split("_")[1]  # 'B50'
        budget = int(budget_str[1:])  # remove 'B' and convert

        # Load the ELA file
        ela_path = os.path.join(ela_input_dir, ela_file)
        ela_df = pd.read_csv(ela_path)

        # Filter optimal precision rows with budget ≥ current file's budget
        budgets_geq = opt_df[opt_df['budget'] >= budget]['budget'].unique()
        precision_subset = opt_df[opt_df['budget'].isin(budgets_geq)]

        # Pivot to wide format: one column per budget
        pivot_df = precision_subset.pivot_table(
            index=['fid', 'iid', 'rep'],
            columns='budget',
            values='precision'
        ).reset_index()

        # Merge the ELA data with the pivoted optimal precision data
        merged = pd.merge(
            ela_df,
            pivot_df,
            on=['fid', 'iid', 'rep'],
            how='left'
        )

        # Save to output directory
        output_path = os.path.join(output_dir, ela_file)
        merged.to_csv(output_path, index=False)

        print(f"✅ Wrote: {output_path}")

def mark_switch_budget(
    ela_with_state_dir,
    best_budgets_csv,
    output_dir,
    valid_budgets=None
):
    """
    For each ELA file with CMA-ES internal state:
      - mark rows with a new column 'switch' == True if the file's budget is best for that fid
      - optionally restrict to budgets in valid_budgets
      - save the result in output_dir

    Args:
        ela_with_state_dir (str): Folder with files like A1_B{budget}_5D_ela_with_state.csv
        best_budgets_csv (str): CSV with columns ['fid', 'best_budget']
        output_dir (str): Where to write files
        valid_budgets (list[int] or None): If given, only budgets in this list are marked True.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the best budgets per fid
    best_df = pd.read_csv(best_budgets_csv)

    # Optionally filter best_df to valid_budgets only
    if valid_budgets is not None:
        best_df = best_df[best_df["best_budget"].isin(valid_budgets)]

    # fid -> set of best budgets (filtered)
    best_budget_map = (
        best_df.groupby("fid")["best_budget"]
        .apply(set)
        .to_dict()
    )

    # Process each ELA file
    for file in sorted(os.listdir(ela_with_state_dir)):
        if not file.endswith(".csv"):
            continue

        budget_str = file.split("_")[1]  # B50
        budget = int(budget_str[1:])

        # If you want: skip files for budgets not in valid_budgets at all
        if valid_budgets is not None and budget not in valid_budgets:
            print(f"Skipping file for budget {budget} (not in valid_budgets)")
            continue

        ela_path = os.path.join(ela_with_state_dir, file)
        df = pd.read_csv(ela_path)

        df["fid"] = df["fid"].astype(int)

        # Mark switch for each row: fid's best budgets includes this budget
        df["switch"] = df["fid"].apply(
            lambda fid: budget in best_budget_map.get(fid, set())
        )

        out_path = os.path.join(output_dir, file)
        df.to_csv(out_path, index=False)
        print(f"✅ Wrote: {out_path}")

def mark_switch_budget_and_greater_budgets(
    ela_with_state_dir,
    best_budgets_csv,
    output_dir,
    valid_budgets=None
):
    """
    For each ELA file with CMA-ES internal state:
      - mark rows with a new column 'switch' == True if the budget >= best budget for that fid
      - optionally restrict to budgets in valid_budgets
      - save the result in output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load best budgets per fid
    best_df = pd.read_csv(best_budgets_csv)

    # Optional filtering to valid budgets
    if valid_budgets is not None:
        best_df = best_df[best_df["best_budget"].isin(valid_budgets)]

    # fid → int best_budget (one per fid)
    best_budget_map = best_df.groupby("fid")["best_budget"].min().to_dict()

    # Process each ELA file
    for file in sorted(os.listdir(ela_with_state_dir)):
        if not file.endswith(".csv"):
            continue

        budget_str = file.split("_")[1]  # B50
        budget = int(budget_str[1:])

        # Skip unwanted budgets
        if valid_budgets is not None and budget not in valid_budgets:
            print(f"Skipping file for budget {budget} (not in valid_budgets)")
            continue

        ela_path = os.path.join(ela_with_state_dir, file)
        df = pd.read_csv(ela_path)
        df["fid"] = df["fid"].astype(int)

        # For each fid, mark True if this file's budget >= fid's best budget
        df["switch"] = df["fid"].apply(
            lambda fid: budget >= best_budget_map.get(fid, float('inf'))
        )

        out_path = os.path.join(output_dir, file)
        df.to_csv(out_path, index=False)
        print(f"✅ Wrote: {out_path}")

def compute_best_budgets(input_csv):
    df = pd.read_csv(input_csv)

    # Identify the static budget columns
    budget_cols = [col for col in df.columns if col.startswith("static_B")]

    # Prepare results
    results = []

    # Group by fid
    for fid, group in df.groupby("fid"):
        # Sum precision values for each budget column
        sums = group[budget_cols].sum()
        min_total = sums.min()

        # Get budgets that achieve the minimum total precision
        best_budgets = sums[sums == min_total].index

        for budget_col in best_budgets:
            budget = int(budget_col.split("_B")[-1])
            results.append({
                "fid": fid,
                "best_budget": budget,
                "total_precision": min_total
            })

    return pd.DataFrame(results)

def compute_late_switching_predictions(input_csv, output_csv):
    # Read the CSV
    df = pd.read_csv(input_csv)

    # Start with required columns
    columns_to_keep = ["fid", "iid", "rep"]

    # Rename static_B56 to static_B50
    df = df.rename(columns={"static_B56": "static_B50"})

    # Add renamed static_B50 to columns
    columns_to_keep.append("static_B50")

    # Add all other columns with budget >= 100
    for col in df.columns:
        if col.startswith("static_B"):
            try:
                budget = int(col.split("_B")[1])
                if budget >= 100:
                    columns_to_keep.append(col)
            except ValueError:
                continue

    # Remove potential duplicates in case static_B50 already exists
    columns_to_keep = list(dict.fromkeys(columns_to_keep))

    # Filter dataframe
    df_filtered = df[columns_to_keep]

    # Save to new CSV
    df_filtered.to_csv(output_csv, index=False)

    print(f"✅ Saved filtered file to {output_csv}")

if __name__ == "__main__":
    compute_best_budgets("../data/switching_optimality_files/normalized_log10_200_no_ps_ratio/predicted_static_precisions_rep_fold_all_sp.csv").to_csv(
        "../data/switching_optimality_files/normalized_log10_200_no_ps_ratio/best_static_budget_per_fid.csv", index=False
    )
    mark_switch_budget_and_greater_budgets(
        ela_with_state_dir="../data/ela_for_training/A1_data_ela_cma_std_precisions_normalized_no_ps_ratio",
        best_budgets_csv="../data/switching_optimality_files/normalized_log10_200_no_ps_ratio/best_static_budget_per_fid.csv",
        output_dir="../data/ela_for_training/A1_data_switch_normalized_log10_200_no_ps_ratio",
    )
   