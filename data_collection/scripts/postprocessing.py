# File in which we process the raw run data 
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

import shutil
import argparse
from dataclasses import dataclass, fields
import pandas as pd

import ioh
from ioh import ProblemClass
from modcma import ModularCMAES, Parameters
import numpy as np

from bfgs import BFGS # type: ignore
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

if __name__ == "__main__":
    # extract_a2_precisions(base_dir="../data/run_data/A2_newReps", output_file="../data/A2_newReps_precisions.csv")
    process_ioh_data(base_path="../data/run_data/A1_early_switching") 