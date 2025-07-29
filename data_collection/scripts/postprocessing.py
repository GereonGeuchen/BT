# File in which we process the raw run data 
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))
from sklearn.preprocessing import MinMaxScaler

import shutil
import argparse
from dataclasses import dataclass, fields
import pandas as pd
import glob

import ioh
from ioh import ProblemClass
from modcma import ModularCMAES, Parameters
import numpy as np

from bfgs_new_scipy import BFGS # type: ignore
from pso import PSO # type: ignore
from mlsl import MLSL # type: ignore
from de import DE # type: ignore
import warnings
from itertools import product
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool

# Function that goes through the IOH logger files and creates clean CSV files containing of the relevant data for the pflacco computation.      
def process_ioh_data(base_path):
    dim = 5
    for budget_dir in os.listdir(base_path):
        # if not (budget_dir == 'A1_B900_5D' or budget_dir == 'A1_B950_5D' or budget_dir == 'A1_B1000_5D'):
        #     continue
        budget_path = os.path.join(base_path, budget_dir)
        if not os.path.isdir(budget_path):
            continue

        all_rows = []

        for func_dir in os.listdir(budget_path):
            func_path = os.path.join(budget_path, func_dir)
            if not os.path.isdir(func_path):
                continue

            # Extract fid from directory name like 'data_f1_Sphere'
            try:
                fid = int(func_dir.split('_')[1][1:])
            except (IndexError, ValueError):
                print(f"Skipping malformed directory: {func_dir}")
                continue

            dat_file = os.path.join(func_path, f"IOHprofiler_f{fid}_DIM{dim}.dat")
            if not os.path.isfile(dat_file):
                continue

            try:
                df = pd.read_csv(dat_file, delim_whitespace=True, comment="#", dtype=str)
            except Exception as e:
                print(f"Error reading {dat_file}: {e}")
                continue

            # Filter out repeated header rows
            df = df[df['iid'] != 'iid']

            # Convert selected columns to numeric
            numeric_cols = ['evaluations', 'raw_y', 'rep', 'iid', 'x0', 'x1', 'x2', 'x3', 'x4']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Group by iid and compute true_y
            for iid_val, group in df.groupby('iid'):
                print(f"Processing fid={fid}, iid={iid_val}, budget dir={budget_dir}")
                try:
                    iid_int = int(float(iid_val))
                    problem = ioh.get_problem(fid, iid_int, dim, ProblemClass.BBOB)
                    optimum = problem.optimum.y
                except Exception as e:
                    print(f"Could not load problem fid={fid}, iid={iid_val}: {e}")
                    continue

                group = group[numeric_cols].copy()
                group['fid'] = fid
                group['true_y'] = group['raw_y'] + optimum
                all_rows.append(group)

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)

            # Reorder columns
            column_order = ['fid', 'iid', 'rep', 'evaluations', 'raw_y', 'true_y', 'x0', 'x1', 'x2', 'x3', 'x4']
            combined = combined[column_order]

            # Sort rows
            combined = combined.sort_values(by=['fid', 'iid', 'rep']).reset_index(drop=True)

            # Save CSV
            output_path = os.path.join(base_path, f"{budget_dir}.csv")
            combined.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

# Function that creates the A2_precisions.csv file from the run data.
def extract_a2_precisions(base_dir, output_file="A2_precisions.csv", algorithms=None, budgets=None, fids=range(1, 25), max_evals=1000):
    """
    Extracts minimum precision values from IOHprofiler logs for switched algorithms at different budgets.
    Only considers rows where evaluations <= max_evals.

    Parameters:
        base_dir (str): Path to the directory containing A2_* folders.
        output_file (str): Output CSV file path.
        algorithms (list): List of algorithm names.
        budgets (list): List of switching budgets.
        fids (iterable): Function IDs (1 to 24).
        max_evals (int): Evaluation cutoff for precision measurement.

    Returns:
        pd.DataFrame: DataFrame with columns [fid, iid, rep, budget, algorithm, precision].
    """
    print(f"Extracting A2 precisions from {base_dir} with max_evals={max_evals}...")
    if algorithms is None:
        algorithms = ["BFGS", "DE", "MLSL", "Non-elitist", "PSO", "Same"]
    if budgets is None:
        budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]

    results = []

    for algo in algorithms:
        for budget in budgets:
            folder_name = os.path.join(base_dir, f"A2_{algo}_B{budget}_5D")
            if not os.path.isdir(folder_name):
                continue
            for fid in fids:
                func_folders = [f for f in os.listdir(folder_name) if f.startswith(f"data_f{fid}_")]
                for func_folder in func_folders:
                    print(f"Processing {func_folder} for fid={fid}, algo={algo}, budget={budget}")
                    file_path = os.path.join(folder_name, func_folder, f"IOHprofiler_f{fid}_DIM5.dat")
                    if not os.path.isfile(file_path):
                        continue
                    try:
                        df = pd.read_csv(file_path, delim_whitespace=True, comment='%')
                        df['evaluations'] = pd.to_numeric(df['evaluations'], errors='coerce')
                        df['raw_y'] = pd.to_numeric(df['raw_y'], errors='coerce')
                        df['rep'] = pd.to_numeric(df['rep'], errors='coerce', downcast='integer')
                        df['iid'] = pd.to_numeric(df['iid'], errors='coerce', downcast='integer')
                        df = df.dropna(subset=['evaluations', 'raw_y', 'rep', 'iid'])
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
                        continue

                    for (rep, iid), group in df.groupby(['rep', 'iid']):
                        subset = group[group['evaluations'] <= max_evals]
                        if not subset.empty:
                            min_y = subset['raw_y'].min()
                            results.append({
                                "fid": fid,
                                "iid": int(iid),
                                "rep": int(rep),
                                "budget": budget,
                                "algorithm": algo,
                                "precision": min_y
                            })

    result_df = pd.DataFrame(results)
    result_df.sort_values(by=["fid", "iid", "rep", "budget"], inplace=True)
    result_df.to_csv(output_file, index=False)
    return result_df

def compute_late_precisions(input_csv, output_csv):
    """
    Filters the input CSV to keep only rows with budget >= 100
    and rows with budget == 56 (changing them to 50).
    
    Parameters:
    - input_csv: str, path to input CSV file
    - output_csv: str, path to save filtered CSV file
    """
    # Read input CSV
    df = pd.read_csv(input_csv)

    # Select rows with budget >= 100 or budget == 56
    filtered_df = df[(df['budget'] >= 100) | (df['budget'] == 56)].copy()

    # Change budget == 56 to 50
    filtered_df.loc[filtered_df['budget'] == 56, 'budget'] = 50

    # Save to output CSV preserving original order
    filtered_df.to_csv(output_csv, index=False)

    print(f"Filtered data saved to {output_csv}")

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
        budget = int(file.split('_')[1][1:])  
        if budget == 50: continue
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
        state_dict = final_row.loc["sigma":"mhl_sum"].to_dict()

        # Remove unwanted keys
        for key in ["t", "ps_squared"]:
            state_dict.pop(key, None)

        return state_dict

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

def normalize_ela_with_precisions(path_in, path_out):
    df = pd.read_csv(path_in)

    index_cols = ["fid", "iid", "rep"]
    algo_cols = ["BFGS", "DE", "MLSL", "Non-elitist", "PSO", "Same"]
    feature_cols = [col for col in df.columns if col not in index_cols + algo_cols]

    # Step 1: Normalize feature columns globally to [0, 1]
    feature_scaler = MinMaxScaler()
    df_scaled_features = pd.DataFrame(
        feature_scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )

    # Step 2: Normalize algorithm columns jointly per iid using 1D flattening
    df_scaled_algos = df[algo_cols].copy()

    for _, group in df.groupby(["fid"]):
        algo_matrix = group[algo_cols].to_numpy()  # shape (num_rows, 6)
        flat_vals = algo_matrix.flatten().reshape(-1, 1)  # shape (num_rows * 6, 1)

        scaler = MinMaxScaler(feature_range=(1e-12, 1))
        flat_scaled = scaler.fit_transform(flat_vals).flatten()

        # Reshape back and insert
        scaled_matrix = flat_scaled.reshape(algo_matrix.shape)
        df_scaled_algos.loc[group.index] = scaled_matrix

    # Combine everything
    df_final = pd.concat([df[index_cols], df_scaled_features, df_scaled_algos], axis=1)
    df_final = df_final.sort_values(by=["fid", "iid", "rep"]).reset_index(drop=True)
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))
    df_final.to_csv(path_out, index=False)
    print(f"Saved normalized file to: {path_out}")


def normalize_test_ela(train_csv_path, test_csv_path, test_out_path):
    # Load training and test data
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Define index columns
    index_cols = ["fid", "iid", "rep"]

    # Identify feature columns (anything that's not an index col)
    feature_cols = [col for col in df_train.columns if col not in index_cols]

    # Fit scaler on training data's feature columns
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(df_train[feature_cols])

    # Transform test data's feature columns
    df_scaled_features = pd.DataFrame(
        feature_scaler.transform(df_test[feature_cols]),
        columns=feature_cols,
        index=df_test.index
    )

    # Reattach index columns and save
    df_final = pd.concat([df_test[index_cols], df_scaled_features], axis=1)
    df_final = df_final.sort_values(by=["fid", "iid", "rep"]).reset_index(drop=True)

    if not os.path.exists(os.path.dirname(test_out_path)):
        os.makedirs(os.path.dirname(test_out_path))
    df_final.to_csv(test_out_path, index=False)

    print(f"Saved normalized test file to: {test_out_path}")

def split_precision_by_budget(
    df: pd.DataFrame,
    output_dir: str = "split_precision_csvs"
) -> dict[int, pd.DataFrame]:
    """
    Converts a long-format precision DataFrame into a dict of wide-format DataFrames, one per budget.
    Also saves each wide-format DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): A long-format DataFrame with columns:
                           ['fid', 'iid', 'rep', 'budget', 'algorithm', 'precision']
        output_dir (str): Directory to save the resulting CSVs. Created if it doesn't exist.

    Returns:
        dict[int, pd.DataFrame]: Dictionary mapping each budget to its corresponding wide-format DataFrame.
                                 Each DataFrame has columns ['fid', 'iid', 'rep', <algorithms...>].
    """
    os.makedirs(output_dir, exist_ok=True)
    result = {}

    for budget, group in df.groupby("budget"):
        pivoted = group.pivot(index=["fid", "iid", "rep"], columns="algorithm", values="precision").reset_index()
        result[budget] = pivoted

        filename = os.path.join(output_dir, f"precision_budget_{budget}.csv")
        pivoted.to_csv(filename, index=False)

    return result

if __name__ == "__main__":
    # split_precision_by_budget(pd.read_csv("../data/precision_files/A2_precisions_l_BFGS_b.csv"))
    # add_algorithm_precisions(
    #     ela_dir="../data/ela_with_cma_std/A1_data_ela_cma_std",
    #     precision_csv="../data/precision_files/A2_precisions_l_BFGS_b.csv",
    #     output_dir="../data/ela_with_algorithm_precisions/A1_data_ela_cma_std_precisions_l_BFGS_b"
    # )
    budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 20)]  # 8, 16, ..., 96, 50, 100, ..., 950
    # for budget in budgets:
    #     normalize_ela_with_precisions(
    #         path_in=f"../data/ela_with_algorithm_precisions/A1_data_ela_cma_std_precisions_l_BFGS_b/A1_B{budget}_5D_ela_with_state.csv",
    #         path_out=f"../data/ela_normalized/A1_data_ela_cma_std_precisions_normalized_l_BFGS_b/A1_B{budget}_5D_ela_with_state.csv"
    #     )
    extract_a2_precisions("../data/run_data/A2_clipped_newReps",
                          output_file="../data/precision_files/A2_precisions_clipped_newReps.csv",)
    # for budget in budgets:
    #     normalize_test_ela(
    #         train_csv_path=f"../data/ela_with_cma_std/A1_data_ela_cma_std/A1_B{budget}_5D_ela_with_state.csv",
    #         test_csv_path=f"../data/ela_with_cma_std/A1_data_ela_cma_std_newReps/A1_B{budget}_5D_ela_with_state.csv",
    #         test_out_path=f"../data/ela_normalized/A1_data_ela_cma_std_newReps_normalized/A1_B{budget}_5D_ela_with_state.csv"
    #     )
    # process_ioh_data("../data/run_data/A1_data_newReps")