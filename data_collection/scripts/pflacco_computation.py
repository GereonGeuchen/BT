import pandas as pd
import numpy as np
import os
import warnings
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'pflacco'))
from classical_ela_features import ( # type: ignore
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_ela_level,
    calculate_dispersion,
    calculate_information_content,
    calculate_nbc
)
from sampling import create_initial_sample # type: ignore
from sklearn.preprocessing import MinMaxScaler
from ioh import ProblemClass, get_problem
import csv


def calculate_ela_features(budget=50, base_folder="../data/run_data/A1_data", output_folder="A1_data_ela_doesitwork"):
    os.makedirs(output_folder, exist_ok=True)
    filename = f"A1_B{budget}_5D.csv"
    filepath = os.path.join(base_folder, filename)
    df = pd.read_csv(filepath)

    x_cols = [col for col in df.columns if col.startswith("x")]
    output_path = os.path.join(output_folder, f"A1_B{budget}_5D_ela.csv")

    first_write = True  # controls header

    for (fid, iid, rep), group in df.groupby(["fid", "iid", "rep"]):
        int_rep = int(rep)
        np.random.seed(int_rep)
        print(f"Processing fid: {fid}, iid: {iid}, rep: {rep}, budget: {budget}")
        group = group.reset_index(drop=True)
        X = group[x_cols].to_numpy()
        # Changed truey_y to raw_y for testing purposes
        y = np.asarray(group["true_y"].values, dtype=float).flatten()



        features = {}
        features.update(calculate_ela_distribution(X, y))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            features.update(calculate_ela_meta(X, y))

        if budget > 16:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                if budget <= 88:
                    if budget <= 32:
                        features.update(calculate_ela_level(X, y, ela_level_quantiles=[0.50]))
                    else:
                        features.update(calculate_ela_level(X, y, ela_level_quantiles=[0.25, 0.50]))
                else:
                    features.update(calculate_ela_level(X, y))

        features.update(calculate_dispersion(X, y))
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1
        assert y.shape[0] == X.shape[0]

        features.update(calculate_information_content(X, y))
        if budget <= 16:
            # For budgets <= 12, we use the raw_y values
            features.update(calculate_nbc(X, y, fast_k = 2))
        else:
            features.update(calculate_nbc(X, y))

        # Add identifying metadata
        features["fid"] = fid
        features["iid"] = iid
        features["rep"] = rep

        if fid in [1, 2, 3, 4, 5]:
            features["high_level_category"] = 1
        elif fid in [6, 7, 8, 9]:
            features["high_level_category"] = 2
        elif fid in [10, 11, 12, 13, 14]:
            features["high_level_category"] = 3
        elif fid in [15, 16, 17, 18, 19]:
            features["high_level_category"] = 4
        elif fid in [20, 21, 22, 23, 24]:
            features["high_level_category"] = 5
        else:
            features["high_level_category"] = None

        # Remove ela_meta.quad_w_interact.adj_r2 if budget <= 56

        if budget <= 56:
            features.pop('ela_meta.quad_w_interact.adj_r2', None)
            if budget <= 32:
                features.pop('ela_meta.lin_w_interact.adj_r2', None)

        for key in list(features.keys()):
            if key.endswith(".costs_runtime"):
                features.pop(key, None)
        # Create DataFrame for one row, reorder columns
        row_df = pd.DataFrame([features])
        cols = ["fid", "iid", "rep", "high_level_category"]
        ordered_cols = cols + [col for col in row_df.columns if col not in cols]
        row_df = row_df[ordered_cols]

        # Append row to file
        row_df.to_csv(output_path, mode='a', header=first_write, index=False)
        first_write = False  # only write header once

    print(f"Completed processing for budget: {budget}")

def normalize_ela_features(input_folder="A1_data_ela_disp", output_folder="A1_data_ela_disp_normalized"):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)
            # Split into metadata and features
            metadata = df.iloc[:, :4]  # first 4 columns (excluded from normalization)
            features = df.iloc[:, 4:]  # remaining columns to normalize

            # Normalize each column independently
            scaler = MinMaxScaler()
            features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

            # Recombine and save
            df_normalized = pd.concat([metadata, features_normalized], axis=1)
            output_path = os.path.join(output_folder, filename)
            df_normalized.to_csv(output_path, index=False)

            print(f"Normalized and saved: {filename}")

def normalize_single_ela_file(input_path, output_path):
    # Load CSV
    df = pd.read_csv(input_path)

    # Separate metadata and features
    metadata = df.iloc[:, :4]
    features = df.iloc[:, 4:]

    # Detect and drop columns with NaN or inf values
    invalid_cols = features.columns[
        features.isnull().any() | features.isin([np.inf, -np.inf]).any()
    ].tolist()

    if invalid_cols:
        print(f"Dropping columns with NaN or inf in {os.path.basename(input_path)}:")
        for col in invalid_cols:
            print(f"  - {col}")
        features = features.drop(columns=invalid_cols)

    # Normalize remaining features
    scaler = MinMaxScaler()
    features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Combine and save
    df_normalized = pd.concat([metadata, features_normalized], axis=1)
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    df_normalized.to_csv(output_path, index=False)

    print(f"✅ Normalized and saved: {output_path}")

# === Categorization ===
def categorize(fid):
    if fid in [1, 2, 3, 4, 5]:
        return 1
    elif fid in [6, 7, 8, 9]:
        return 2
    elif fid in [10, 11, 12, 13, 14]:
        return 3
    elif fid in [15, 16, 17, 18, 19]:
        return 4
    elif fid in [20, 21, 22, 23, 24]:
        return 5
    else:
        return -1

# === Worker task ===
def create_sample(fid, instance, run, dim, file_path):
    print(f"Running function: {fid}, instance: {instance}, run: {run}")
    
    problem = get_problem(fid, instance, dim, ProblemClass.BBOB)
    X = create_initial_sample(dim, sample_type="lhs", lower_bound=-5, upper_bound=5, n=200 * dim)
    y = X.apply(lambda x: problem(x), axis=1)

    features = {}
    features["fid"] = fid
    features["iid"] = instance
    features["rep"] = run
    features["high_level_category"] = categorize(fid)
    features.update(calculate_ela_distribution(X, y))
    features.update(calculate_ela_meta(X, y))
    features.update(calculate_ela_level(X, y))
    features.update(calculate_dispersion(X, y))
    features.update(calculate_information_content(X, y))
    features.update(calculate_nbc(X, y))

    # Write to CSV directly here
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=features.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(features)

    # Filter out ".costs_runtime"
    # features = {k: v for k, v in features.items() if not k.endswith(".costs_runtime")}

def add_rep_column(input_path, output_path):
    df = pd.read_csv(input_path)

    # Group by (fid, iid, high_level_category) and assign a counter
    df['rep'] = df.groupby(['fid', 'iid', 'high_level_category']).cumcount()

    # Reorder columns to place 'rep' after 'iid'
    cols = df.columns.tolist()
    iid_index = cols.index('iid')
    # Remove and re-insert 'rep'
    cols.remove('rep')
    cols.insert(iid_index + 1, 'rep')
    df = df[cols]

    # Save the modified file
    df.to_csv(output_path, index=False)
    print(f"✅ 'rep' column added and saved to: {output_path}")

def normalize_ela_features_across_budgets(input_folder="A1_data_ela_disp",
                                                            output_folder="A1_data_ela_disp_normalized_global"):
    os.makedirs(output_folder, exist_ok=True)

    all_features_list = []
    all_feature_columns = set()
    file_metadata = []  # Store (filename, metadata_df, original_feature_df)

    # Step 1: Collect and store all features across all files
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)

            metadata = df.iloc[:, :4]
            features = df.iloc[:, 4:]

            file_metadata.append((filename, metadata, features))
            all_features_list.append(features)

            all_feature_columns.update(features.columns)

    # Step 2: Create a combined DataFrame with aligned columns
    all_feature_columns = sorted(all_feature_columns)  # consistent column order
    combined_features = pd.concat([
        f.reindex(columns=all_feature_columns) for f in all_features_list
    ], axis=0)

    # Step 3: Fit scaler on globally aligned feature space
    scaler = MinMaxScaler()
    scaler.fit(combined_features.fillna(0))  # Use fillna(0) only for fitting

    # Step 4: Normalize and write each file using only its original columns
    for filename, metadata, original_features in file_metadata:
        # Fill missing columns temporarily for scaling
        features_temp = original_features.reindex(columns=all_feature_columns).fillna(0)

        # Apply normalization
        features_normalized_all = pd.DataFrame(
            scaler.transform(features_temp),
            columns=all_feature_columns
        )

        # Extract only the columns originally present in the file
        features_normalized = features_normalized_all[original_features.columns]

        # Reattach metadata
        df_normalized = pd.concat([metadata, features_normalized], axis=1)

        # Write the normalized file
        output_path = os.path.join(output_folder, filename)
        df_normalized.to_csv(output_path, index=False)
        print(f"Normalized and saved: {filename}")

if __name__ == "__main__":
    #normalize_single_ela_file("ela_initial_sampling.csv", "ela_initial_sampling_normalized.csv")
    # add_rep_column("ela_initial_sampling_normalized.csv", "ela_initial_sampling_with_normalized_rep.csv")
    # normalize_ela_features_across_budgets()
    # for budget in [50*i for i in range(1, 21)]:
    #     print(f"Calculating ELA features for budget: {budget}")
    #     calculate_ela_features(
    #         budget = budget,
    #         base_folder="../data/run_data/A1_newReps",
    #         output_folder=f"../data/ela_data/A1_data_ela_newReps"
    #     )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        calculate_ela_features(
            budget=500,
            base_folder="../data/run_data/A1_data_test_2",
            output_folder="../data/ela_data/A1_data_test_2"
        )
        # for budget in reversed([8*i for i in range(4, 5)]):
        #     print(f"Calculating ELA features for budget: {budget}")
        #     calculate_ela_features(
        #         budget=budget,
        #         base_folder="../data/run_data/A1_early_switching",
        #         output_folder=f"../data/ela_data/A1_data_ela_early_switching_test"
        #     )