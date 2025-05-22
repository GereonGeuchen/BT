import pandas as pd
import os
import re

def create_optimality_file():
    # Load the input file
    df = pd.read_csv("../data/A2_precisions.csv")

    # Step 1: Aggregate: min precision per budget (over algorithms), per (fid, iid, rep)
    budget_min = df.groupby(['fid', 'iid', 'rep', 'budget'])['precision'].min().reset_index()

    # Step 2: Compute the overall minimum precision per (fid, iid, rep)
    overall_min = budget_min.groupby(['fid', 'iid', 'rep'])['precision'].min().reset_index()
    overall_min = overall_min.rename(columns={'precision': 'min_precision'})

    # Step 3: Merge to get the minimum per (fid, iid, rep) for each budget
    merged = pd.merge(budget_min, overall_min, on=['fid', 'iid', 'rep'])

    # Step 4: Determine which budgets hit the minimum precision
    merged['is_minimal_switch'] = merged['precision'] == merged['min_precision']

    # Step 5: Calculate run_precision = precision - min_precision
    merged['run_precision'] = merged['precision'] - merged['min_precision']

    # Step 6: Select and save the final output
    result = merged[['fid', 'iid', 'rep', 'budget', 'is_minimal_switch', 'run_precision']]
    result.to_csv("../data/optimality.csv", index=False)

def create_ela_files_with_optimality(ela_dir, opt_switch_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load optimal switch data
    opt_df = pd.read_csv(opt_switch_file)
    opt_df['key'] = opt_df.apply(lambda row: f"{int(row['fid'])}_{int(row['iid'])}_{int(row['rep'])}_{int(row['budget'])}", axis=1)
    opt_df.set_index('key', inplace=True)

    # Extract relevant data for lookup
    opt_info = opt_df[['is_minimal_switch', 'run_precision']]

    # Process each ELA feature file
    for file in os.listdir(ela_dir):
        if not file.endswith('.csv'):
            continue

        match = re.search(r'_B(\d+)_', file)
        if not match:
            print(f"Skipping file (budget not found): {file}")
            continue

        budget = int(match.group(1))
        file_path = os.path.join(ela_dir, file)
        df = pd.read_csv(file_path)

        df['key'] = df.apply(lambda row: f"{int(row['fid'])}_{int(row['iid'])}_{int(row['rep'])}_{budget}", axis=1)

        # Merge with optimality information
        df = df.merge(opt_info, how='left', left_on='key', right_index=True)

        df.drop(columns=['key'], inplace=True)

        output_path = os.path.join(output_dir, file)
        df.to_csv(output_path, index=False)
        print(f"Labeled: {output_path}")


if __name__ == "__main__":
    create_ela_files_with_optimality(
        ela_dir="../data/ELA_over_budgets",
        opt_switch_file="../data/optimality.csv",
        output_dir="../data/ELA_over_budgets_with_precs"
    )
    # create_optimality_file()