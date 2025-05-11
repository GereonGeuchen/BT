import pandas as pd
import numpy as np
import os
import warnings
from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_ela_level,
    calculate_dispersion,
    calculate_information_content,
    calculate_nbc
)
from sklearn.preprocessing import MinMaxScaler


def calculate_ela_features(budget=50, base_folder="A1_data", output_folder="A1_data_ela"):
    os.makedirs(output_folder, exist_ok=True)
    filename = f"A1_B{budget}_5D.csv"
    filepath = os.path.join(base_folder, filename)
    df = pd.read_csv(filepath)

    x_cols = [col for col in df.columns if col.startswith("x")]
    output_path = os.path.join(output_folder, f"A1_B{budget}_5D_ela.csv")

    first_write = True  # controls header

    for (fid, iid, rep), group in df.groupby(["fid", "iid", "rep"]):
        print(f"Processing fid: {fid}, iid: {iid}, rep: {rep}, budget: {budget}")
        group = group.reset_index(drop=True)
        X = group[x_cols].to_numpy()
        y = np.asarray(group["true_y"].values, dtype=float).flatten()



        features = {}
        features.update(calculate_ela_distribution(X, y))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            features.update(calculate_ela_meta(X, y))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if budget == 50:
                features.update(calculate_ela_level(X, y, ela_level_quantiles=[0.25, 0.50]))
            else:
                features.update(calculate_ela_level(X, y))

        features.update(calculate_dispersion(X, y))
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1
        assert y.shape[0] == X.shape[0]

        features.update(calculate_information_content(X, y))
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

        # Create DataFrame for one row, reorder columns
        row_df = pd.DataFrame([features])
        cols = ["fid", "iid", "rep", "high_level_category"]
        ordered_cols = cols + [col for col in row_df.columns if col not in cols]
        row_df = row_df[ordered_cols]

        # Append row to file
        row_df.to_csv(output_path, mode='a', header=first_write, index=False)
        first_write = False  # only write header once

    print(f"Completed processing for budget: {budget}")

def normalize_ela_features(input_folder="A1_data_ela", output_folder="A1_data_ela_normalized"):
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


if __name__ == "__main__":
    # budgets = (50*i for i in range(7, 21))
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     warnings.filterwarnings("ignore", category=RuntimeWarning)

    #     for budget in budgets:
    #         calculate_ela_features(budget=budget)
    #         print(f"Completed ELA feature calculation for budget: {budget}")
    normalize_ela_features()