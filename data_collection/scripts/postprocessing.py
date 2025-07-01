# File in which we process the raw run data 
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

import shutil
import argparse
from dataclasses import dataclass, fields
import pandas as pd
import glob

# import ioh
# from ioh import ProblemClass
# from modcma import ModularCMAES, Parameters
import numpy as np

# from bfgs import BFGS # type: ignore
# from pso import PSO # type: ignore
# from mlsl import MLSL # type: ignore
# from de import DE # type: ignore
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
        budgets = list(range(50, 1050, 50))

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
        if budget != 100: continue  # Only process budget 100 for now
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


if __name__ == "__main__":
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     extract_a2_precisions(base_dir="../data/run_data/A2_newInstances", 
    #                           output_file="../data/precision_files/A2_newInstances_precisions_test.csv",
    #                           budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)])
    # add_algorithm_precisions(
    #     ela_dir="../data/ela_with_cma/A1_data_with_cma",
    #     precision_csv="../data/precision_files/A2_data_late_precisions.csv",
    #     output_dir="../data/ela_with_algorithm_precisions/A1_data_with_precisions_100"
    # )
    extend_ela_with_optimal_precisions(
        ela_input_dir="../data/ela_with_cma/ela_with_cma_late",
        optimal_precisions_file="../data/precision_files/A2_data_late_precisions_min.csv",
        output_dir="../data/ela_with_optimal_precisions/A1_data_ela_with_optimal_precisions_late_with_precisions"
    )