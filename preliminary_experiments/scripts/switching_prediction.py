import os
import re
import pandas as pd
import joblib
import pandas as pd
import numpy as np
from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_ela_level,
    calculate_dispersion,
    calculate_information_content,
    calculate_nbc
)
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


def evaluate_rf_leave_instance_out(csv_path="ela_features_per_rep_budget150.csv", predict_fid=False, threshold=0.3):
    """
    Evaluates a Random Forest using leave-instance-out CV.
    
    Args:
        csv_path (str): Path to the input CSV file.
        predict_fid (bool): If True, predict fid (multiclass); otherwise, predict budget_optimal (binary).
        threshold (float): Custom decision threshold (only used for binary classification).
    """
    target = "fid" if predict_fid else "budget_optimal"

    # Load and clean data
    df = pd.read_csv(csv_path)
    cols_to_drop = [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10",
        "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10"
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore").dropna()

    if target not in df.columns:
        raise ValueError(f"Invalid target column: {target}")

    X = df.drop(columns=["fid", "iid", "rep", "budget_optimal"])
    y = df[target].astype(int if target == "budget_optimal" else str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_y_true = []
    all_y_pred = []

    for test_iid in df["iid"].unique():
        test_mask = df["iid"] == test_iid
        X_train, X_test = X_scaled[~test_mask], X_scaled[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]

        clf = RandomForestClassifier(random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)

        if target == "budget_optimal":
            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = clf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Evaluation
    print(f"\n📊 Leave-Instance-Out CV Results (target: {target})")
    acc = accuracy_score(all_y_true, all_y_pred)
    print(f"Accuracy : {acc:.4f}")

    average = "binary" if target == "budget_optimal" else "macro"
    prec = precision_score(all_y_true, all_y_pred, average=average, zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, average=average, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, average=average, zero_division=0)

    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")


def evaluate_rf_with_threshold_curve(csv_path="ela_features_per_rep_budget150.csv", threshold=0.5):
    """
    Evaluate a Random Forest classifier using leave-one-instance-out cross-validation.
    Computes and plots precision-recall curves with decision thresholds.

    Parameters:
        csv_path (str): Path to the input CSV file.
        threshold (float): Decision threshold for classifying probabilities.
    """
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    cols_to_drop = [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10",
        "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10"
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    df.dropna(inplace=True)

    # Features and target
    feature_cols = df.columns.difference(["fid", "iid", "rep", "budget_optimal"])
    X = df[feature_cols]
    y = df["budget_optimal"].astype(int)
    iids = df["iid"]

    # Normalize
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_scaled = X
    # Storage for predictions
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    # Leave-one-instance-out CV
    for test_iid in iids.unique():
        mask = iids == test_iid
        X_train, X_test = X_scaled[~mask], X_scaled[mask]
        y_train, y_test = y[~mask], y[mask]

        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)

    # Metrics
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    print("\n📊 Leave-Instance-Out CV Results")
    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Precision-Recall vs. Threshold curve
    precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_proba_all)
    f1_scores = [2 * (p * r) / (p + r) if (p + r) else 0 for p, r in zip(precisions, recalls)]
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_f1 = f1_scores[best_idx]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], label="Recall", linewidth=2)
    plt.axvline(best_thresh, color='gray', linestyle='--', label=f"Best F1 @ {best_thresh:.2f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n🔍 Best threshold by F1: {best_thresh:.2f} → F1: {best_f1:.4f}")

def train_rf_model_with_cv_threshold(csv_path, output_model_path, thresholds=np.linspace(0.1, 0.9, 81)):
    """
    Trains and saves a Random Forest classifier using the best threshold found via CV.
    
    Args:
        csv_path (str): Path to the feature CSV.
        output_model_path (str): Path to save the trained model (as .joblib).
        thresholds (np.ndarray): Array of thresholds to evaluate.
    """
    # Load and preprocess
    df = pd.read_csv(csv_path)
    drop_cols = [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10",
        "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10"
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    df.dropna(inplace=True)

    # Prepare features and target
    feature_cols = df.columns.difference(["fid", "iid", "rep", "budget_optimal"])
    X = df[feature_cols]
    y = df["budget_optimal"].astype(int)
    iids = df["iid"]

    # Leave-one-instance-out CV for threshold selection
    y_true_all = []
    y_proba_all = []

    for test_iid in iids.unique():
        mask = iids == test_iid
        X_train, X_test = X[~mask], X[mask]
        y_train, y_test = y[~mask], y[mask]

        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        y_proba_raw = model.predict_proba(X_test)
        if y_proba_raw.shape[1] == 2:
            y_proba = y_proba_raw[:, 1]
        else:
            y_proba = np.ones_like(y_test) if model.classes_[0] == 1 else np.zeros_like(y_test)

        y_true_all.extend(y_test)
        y_proba_all.extend(y_proba)

    # Select best threshold based on F1 score
    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)

    best_thresh = 0.5
    best_f1 = 0
    for t in thresholds:
        preds = (y_proba_all >= t).astype(int)
        score = f1_score(y_true_all, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"✅ Best threshold: {best_thresh:.2f} with F1: {best_f1:.4f}")

    # Train final model on full dataset
    final_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    final_model.fit(X, y)

    # Save model
    joblib.dump(final_model, output_model_path)
    print(f"📁 Model saved to: {output_model_path}")

# Update this path to your actual CSV file
csv_file = "min_target_precision_below_1000.csv"

def read_csv_file(exlude_1000: bool = False):
    # Load data
    df = pd.read_csv(csv_file, dtype={
        "fid": int, "iid": int, "rep": int, "budget": int, "algorithm": str, "fopt": float, "precision": float
    })

    # df = df[df["budget"] != 1000]
    # df = df[df["algorithm"] != "MLSL"]
    return df

def get_budget_optimal_reps(df, target_budget=150):
    """
    For each (fid, iid, rep), check whether `target_budget` is among the switching points
    with the lowest fopt. Returns a dict with {(fid, iid, rep): True/False}.

    Args:
        df (pd.DataFrame): DataFrame with columns ["fid", "iid", "rep", "budget", "fopt"]
        target_budget (int): Budget value to check for optimality.
    """
    result = {}

    def check_target_optimal(group):
        min_fopt = group["fopt"].min()
        return target_budget in group[group["fopt"] == min_fopt]["budget"].values

    grouped = df.groupby(["fid", "iid", "rep"])

    for (fid, iid, rep), group in grouped:
        result[(fid, iid, rep)] = check_target_optimal(group)

    return result


def compute_pflacco_features_from_dataframe(X_array, y_array, first=True):
    """
    Computes pflacco features from numpy arrays X and y.
    No warning suppression.
    """
    features = {}

    # ELA distribution
    features.update(calculate_ela_distribution(X_array, y_array))

    # ELA meta
    features.update(calculate_ela_meta(X_array, y_array))

    # ELA level
    features.update(calculate_ela_level(X_array, y_array))

    # Dispersion, Information Content, NBC
    features.update(calculate_dispersion(X_array, y_array))
    features.update(calculate_information_content(X_array, y_array))
    features.update(calculate_nbc(X_array, y_array))

    return features

def extract_ela_from_a1_dat_files(root_dir, budget_optimal_dict, output_csv="test_data/ela_features_per_rep_budget100.csv"):
    results = []

    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.endswith("_with_true_y.dat"):
                continue

            # Extract function ID from filename like IOHprofiler_f1_DIM5_with_true_y.dat
            match = re.search(r"f(\d+)_DIM5", filename)
            if not match:
                continue
            fid = int(match.group(1))
            path = os.path.join(subdir, filename)

            # Read full CSV including header
            df = pd.read_csv(path, sep="\t", comment="%", engine="python")
            df = df.rename(columns=lambda c: c.strip())  # sanitize header names
            df = df.astype({  # parse column types
                "evaluations": int,
                "true_y": float,
                "rep": int,
                "iid": int,
                "x0": float, "x1": float, "x2": float, "x3": float, "x4": float
            })

            for (iid, rep), rep_df in df.groupby(["iid", "rep"]):
                print(f"Processing fid {fid}, iid {iid}, rep {rep}")
                rep_df = rep_df.sort_values("evaluations")
                X = rep_df[["x0", "x1", "x2", "x3", "x4"]].to_numpy()
                y = rep_df["true_y"].to_numpy()

                features = compute_pflacco_features_from_dataframe(X, y, first=True)

                row = {
                    "fid": fid,
                    "iid": iid,
                    "rep": rep,
                    "budget_optimal": budget_optimal_dict.get((fid, iid, rep), False)
                }
                row.update(features)
                results.append(row)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Saved ELA features to: {output_csv}")
    return df_out

def print_columns_with_nans(df):
    nan_counts = df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    
    if nan_columns.empty:
        print("✅ No NaN values found in the DataFrame.")
    else:
        print("⚠️ Columns with NaN values:")
        for col, count in nan_columns.items():
            print(f"  - {col}: {count} NaNs")

def append_internal_state_slice_at_eval(csv_path, dat_root_dir, eval_number=152, output_csv="ela_features_with_state.csv"):
    """
    Appends CMA-ES internal state variables (from 'd_norm' to 'mhl_norm') at a specified evaluation
    to an existing ELA CSV file.

    Parameters:
        csv_path (str): Path to the existing ELA CSV.
        dat_root_dir (str): Root directory where the .dat files are stored.
        eval_number (int): The evaluation at which to extract internal state values.
        output_csv (str): Output CSV path for saving the result.
    """
    # Load ELA features
    ela_df = pd.read_csv(csv_path)
    ela_df["key"] = list(zip(ela_df["fid"], ela_df["iid"], ela_df["rep"]))

    state_data = {}

    for subdir, _, files in os.walk(dat_root_dir):
        for filename in files:
            if not filename.endswith("_with_true_y.dat"):
                continue

            match = re.search(r"f(\d+)_DIM5", filename)
            if not match:
                continue
            fid = int(match.group(1))
            path = os.path.join(subdir, filename)

            df = pd.read_csv(path, sep="\t", comment="%", engine="python")
            df.columns = df.columns.str.strip()

            for (iid, rep), group in df.groupby(["iid", "rep"]):
                row = group[group["evaluations"] == eval_number]
                if row.empty:
                    continue

                row = row.iloc[0]
                columns = list(df.columns)

                try:
                    start = columns.index("d_norm")
                    end = columns.index("mhl_norm") + 1
                except ValueError:
                    continue  # Skip if expected columns are missing

                selected_cols = columns[start:end]
                extracted = {f"state.{col}": row[col] for col in selected_cols}
                state_data[(fid, iid, rep)] = extracted

    # Convert dict to DataFrame
    state_df = pd.DataFrame.from_dict(state_data, orient="index")
    state_df.index = pd.MultiIndex.from_tuples(state_df.index, names=["fid", "iid", "rep"])
    state_df = state_df.reset_index()
    state_df["key"] = list(zip(state_df["fid"], state_df["iid"], state_df["rep"]))

    # Merge with ELA features and write output
    merged = ela_df.merge(state_df.drop(columns=["fid", "iid", "rep"]), on="key", how="left").drop(columns=["key"])
    merged.to_csv(output_csv, index=False)
    print(f"✅ Saved merged file with CMA-ES state at evaluation {eval_number} to: {output_csv}")
    return merged

import os
import re
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from collections import defaultdict

def compute_tsfresh_features_for_reps(dat_root_dir="A1_data/A1_B150_5D", eval_number=152, output_csv="tsfresh_features_per_rep.csv"):
    """
    Computes exactly 32 TSFresh features (from CMA-ES paper) for each (fid, iid, rep) triple 
    in .dat files under a directory structure.

    Parameters:
        dat_root_dir (str): Root directory containing subdirectories per function with *_with_true_y.dat files.
        eval_number (int): Maximum evaluation step to include in time series.
        output_csv (str): Path to save the resulting CSV.
    """
    time_series_cols = ["sigma", "d_norm", "d_mean", "ps_norm", "ps_mean", "pc_norm", "pc_mean"]
    records = []

    # Define exactly the 32 recommended TSFresh features per column
    tsfresh_feature_map = [
        ("absolute_sum_of_changes", "pc_norm", None),
        ("approximate_entropy", "ps_norm", [{"m": 2, "r": 0.1}]),
        ("autocorrelation", "ps_norm", [{"lag": 1}]),
        ("change_quantiles", "pc_norm", [{"ql": 0.4, "qh": 0.6, "isabs": False, "f_agg": "mean"}]),
        ("change_quantiles", "ps_mean", [{"ql": 0.4, "qh": 0.6, "isabs": False, "f_agg": "mean"}]),
        ("change_quantiles", "ps_norm", [{"ql": 0.4, "qh": 0.6, "isabs": False, "f_agg": "mean"}]),
        ("change_quantiles", "sigma", [{"ql": 0.2, "qh": 0.6, "isabs": False, "f_agg": "mean"}]),
        ("cid_ce", "ps_norm", [{"normalize": False}]),
        ("energy_ratio_by_chunks", "d_norm", [{"num_segments": 10, "segment_focus": 0}]),
        ("energy_ratio_by_chunks", "d_mean", [{"num_segments": 10, "segment_focus": 0}]),
        ("fft_aggregated", "d_mean", [{"aggtype": "centroid"}]),
        ("fft_coefficient", "pc_norm", [{"coeff": 0, "attr": "abs"}]),
        ("index_mass_quantile", "d_norm", [{"q": 0.1}]),
        ("index_mass_quantile", "d_mean", [{"q": 0.1}]),
        ("mean", "pc_norm", None),
        ("median", "d_norm", None),
        ("median", "d_mean", None),
        ("median", "pc_norm", None),
        ("median", "ps_norm", None),
        ("minimum", "d_norm", None),
        ("minimum", "d_mean", None),
        ("number_crossing_m", "ps_norm", [{"m": 1.0}]),
        ("number_peaks", "sigma", [{"n": 1}]),
        ("partial_autocorrelation", "ps_norm", [{"lag": 1}]),
        ("quantile", "d_norm", [{"q": 0.1}]),
        ("quantile", "d_mean", [{"q": 0.1}]),
        ("quantile", "pc_norm", [{"q": 0.1}]),
        ("quantile", "ps_norm", [{"q": 0.1}]),
        ("quantile", "sigma", [{"q": 0.1}]),
        ("range_count", "pc_norm", [{"min": -1.0, "max": 1.0}]),
        ("range_count", "ps_norm", [{"min": -1.0, "max": 1.0}]),
        ("sum_values", "pc_norm", None),
    ]

    # Convert feature map to kind_to_fc_parameters dict
    kind_to_fc_parameters = defaultdict(dict)
    for func, col, param in tsfresh_feature_map:
        if param is None:
            kind_to_fc_parameters[col][func] = None
        else:
            kind_to_fc_parameters[col][func] = param

    for subdir in os.listdir(dat_root_dir):
        subdir_path = os.path.join(dat_root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for filename in os.listdir(subdir_path):
            if not filename.endswith("_with_true_y.dat"):
                continue

            match = re.search(r"f(\d+)_DIM5", filename)
            if not match:
                continue
            fid = int(match.group(1))
            file_path = os.path.join(subdir_path, filename)

            df = pd.read_csv(file_path, sep="\t", comment="%", engine="python")
            df.columns = df.columns.str.strip()

            if not {"iid", "rep", "t"}.issubset(df.columns):
                continue

            for (iid, rep), group in df.groupby(["iid", "rep"]):
                group = group.drop_duplicates(subset="t").sort_values("t")
                group = group[group["t"] <= eval_number]
                print(f"Processing fid {fid}, iid {iid}, rep {rep}")
                for col in time_series_cols:
                    if col not in group.columns:
                        continue
                    for t_val, val in zip(group["t"], group[col]):
                        records.append({
                            "id": (fid, iid, rep),
                            "time": t_val,
                            "value": val,
                            "feature": col
                        })

    # Prepare tsfresh input
    ts_df = pd.DataFrame(records)
    if ts_df.empty:
        print("⚠️ No valid time series data found.")
        return

    pivot_df = ts_df.pivot_table(index=["id", "time"], columns="feature", values="value").reset_index()
    pivot_df["id"] = pivot_df["id"].apply(lambda x: tuple(x))  # ensure id remains a tuple

    # Extract features
    print("🔍 Extracting 32 selected TSFresh features...")
    features = extract_features(
        pivot_df,
        column_id="id",
        column_sort="time",
        kind_to_fc_parameters=kind_to_fc_parameters,
        n_jobs=4
    )
    impute(features)
    features["fid"] = [x[0] for x in features.index]
    features["iid"] = [x[1] for x in features.index]
    features["rep"] = [x[2] for x in features.index]
    features.reset_index(drop=True, inplace=True)

    # Reorder columns
    cols = ["fid", "iid", "rep"] + [c for c in features.columns if c not in {"fid", "iid", "rep"}]
    features = features[cols]

    features.to_csv(output_csv, index=False)
    print(f"✅ Saved TSFresh features to: {output_csv}")
    return features


def merge_feature_files(budgets, data_dir=".", ela_prefix="ela_features_per_rep_budget", tsfresh_prefix="tsfresh_features_per_rep_budget", output_prefix="merged_features_per_rep"):
    """
    Merge ELA and TSFresh feature files for each given budget based on fid, iid, rep.

    Parameters:
        budgets (list): List of budget values (e.g., [100, 150, 200]).
        data_dir (str): Path to the directory containing the CSV files.
        ela_prefix (str): Prefix of the ELA files.
        tsfresh_prefix (str): Prefix of the TSFresh files.
        output_prefix (str): Prefix for the output merged files.
    """
    for budget in budgets:
        ela_file = os.path.join(data_dir, f"{ela_prefix}{budget}.csv")
        tsfresh_file = os.path.join(data_dir, f"{tsfresh_prefix}{budget}.csv")
        output_file = os.path.join(data_dir, f"{output_prefix}{budget}.csv")

        try:
            ela_df = pd.read_csv(ela_file)
            tsfresh_df = pd.read_csv(tsfresh_file)

            # Ensure consistent dtypes
            for col in ["fid", "iid", "rep"]:
                ela_df[col] = ela_df[col].astype(int)
                tsfresh_df[col] = tsfresh_df[col].astype(int)

            merged_df = pd.merge(ela_df, tsfresh_df, on=["fid", "iid", "rep"], how="inner")
            print(f"✅ Rows after merge (budget {budget}): {len(merged_df)}")
            merged_df.to_csv(output_file, index=False)
            print(f"✅ Merged and saved: {output_file}")

        except FileNotFoundError as e:
            print(f"❌ File not found for budget {budget}: {e}")
        except Exception as e:
            print(f"❌ Error while processing budget {budget}: {e}")

def merge_training_feature_files(budgets, data_dir=".", 
                                 ela_prefix="ela_features_per_rep_budget", 
                                 tsfresh_prefix="tsfresh_features_per_rep_budget", 
                                 output_prefix="merged_features_per_rep_budget"):
    """
    Merge ELA and TSFresh *training* feature files for each given budget based on fid, iid, rep.

    Parameters:
        budgets (list): List of budget values (e.g., [100, 150, 200]).
        data_dir (str): Path to the directory containing the CSV files.
        ela_prefix (str): Prefix of the ELA files.
        tsfresh_prefix (str): Prefix of the TSFresh files.
        output_prefix (str): Prefix for the output merged files.
    """
    for budget in budgets:
        ela_file = os.path.join(data_dir, f"{ela_prefix}{budget}_testing.csv")
        tsfresh_file = os.path.join(data_dir, f"{tsfresh_prefix}{budget}_training.csv")
        output_file = os.path.join(data_dir, f"{output_prefix}{budget}_testing.csv")

        try:
            ela_df = pd.read_csv(ela_file)
            tsfresh_df = pd.read_csv(tsfresh_file)

            merged_df = pd.merge(ela_df, tsfresh_df, on=["fid", "iid", "rep"], how="inner")
            merged_df.to_csv(output_file, index=False)
            print(f"✅ Merged and saved: {output_file}")

        except FileNotFoundError as e:
            print(f"❌ File not found for budget {budget}: {e}")
        except Exception as e:
            print(f"❌ Error while processing budget {budget}: {e}")

def update_budget_optimal_column(feature_csv_path, output_csv_path):
    # Load the feature dataset
    df = pd.read_csv(feature_csv_path)

    # Get the optimality information
    optimal_dict = get_budget_optimal_reps(read_csv_file(), target_budget=100)
    print(optimal_dict)
    # optimal_dict should be like: {(fid, iid, rep): True/False, ...}

    # Update each row based on (fid, iid, rep)
    updated_count = 0
    for idx, row in df.iterrows():
        key = (row['fid'], row['iid'], row['rep'])
        if key in optimal_dict:
            df.at[idx, 'budget_optimal'] = optimal_dict[key]
            updated_count += 1

    print(f"✅ Updated {updated_count} rows in 'budget_optimal' column.")

    # Save to new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"📁 Saved updated file to: {output_csv_path}")

def test_models(model_path="trained_models/rf_model_budget100.joblib",
                csv_path="testing_data/merged_features_per_rep100_testing.csv", threshold=0.5):
    """
    Test the trained Random Forest model on a new dataset.

    Args:
        model_path (str): Path to the trained model.
        csv_path (str): Path to the testing dataset.
        feature_path (str): Path to the saved feature column order used in training.
    """

    # Load model and feature names
    model = joblib.load(model_path)

    # Load and preprocess test data
    df = pd.read_csv(csv_path)
    cols_to_drop = [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10",
        "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10"
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    df.dropna(inplace=True)

    X_test = df.drop(columns=["fid", "iid", "rep", "budget_optimal"])
    X_test = X_test[model.feature_names_in_]
    y_test = df["budget_optimal"].astype(int)

    # Predict and evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n📊 Test Results")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return f1

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def simulate_switching_runs(
    data_dir="testing_data",
    model_dir="trained_models",
    budgets=[100, 150, 200, 250, 300],
    thresholds={100: 0.30, 150: 0.31, 200: 0.32, 250: 0.34, 300: 0.33},
    cols_to_drop=None,
    plot=False
):
    if cols_to_drop is None:
        cols_to_drop = [
            "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
            "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10",
            "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10",
            "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10"
        ]

    dropped_right = 0
    dropped_wrong = 0
    remaining_reps = None
    remaining_counts = []

    for budget in budgets:
        csv_path = f"{data_dir}/merged_features_per_rep{budget}.csv"
        model_path = f"{model_dir}/rf_model_budget{budget}.joblib"
        threshold = thresholds[budget]

        df = pd.read_csv(csv_path)
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        df.dropna(inplace=True)
        df["rep_key"] = df[["fid", "iid", "rep"]].apply(tuple, axis=1)

        # Restrict to reps not yet switched
        if remaining_reps is not None:
            df = df[df["rep_key"].isin(remaining_reps)]

        if df.empty:
            remaining_counts.append(0)
            continue

        model = joblib.load(model_path)
        X = df.drop(columns=["fid", "iid", "rep", "budget_optimal", "rep_key"])
        X = X[model.feature_names_in_]
        y_true = df["budget_optimal"].astype(int)

        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        df["predicted"] = y_pred
        df["true"] = y_true

        dropped = df[df["predicted"] == 1]
        kept = df[df["predicted"] == 0]

        dropped_right += (dropped["true"] == 1).sum()
        dropped_wrong += (dropped["true"] == 0).sum()

        # Only continue with reps that were not dropped
        remaining_reps = set(kept["rep_key"])
        remaining_counts.append(len(remaining_reps))

    # Evaluate final kept reps across all budgets to detect missed switching points
    if remaining_reps is not None:
        all_data = pd.concat(
            [pd.read_csv(f"{data_dir}/merged_features_per_rep{b}.csv") for b in budgets]
        )
        all_data.dropna(inplace=True)
        all_data["rep_key"] = all_data[["fid", "iid", "rep"]].apply(tuple, axis=1)
        final_kept = all_data[all_data["rep_key"].isin(remaining_reps)]

        grouped = final_kept.groupby("rep_key")["budget_optimal"].max()
        final_kept_wrong = (grouped == 1).sum()  # had an optimal switch point
        final_kept_right = (grouped == 0).sum()  # never had an optimal switch point
    else:
        final_kept_right = final_kept_wrong = 0

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(budgets, remaining_counts, marker='o')
        plt.title('Remaining Reps After Each Budget')
        plt.xlabel('Budget')
        plt.ylabel('Number of Remaining Reps')
        plt.grid(True)
        plt.xticks(budgets)
        plt.tight_layout()
        plt.show()

    return {
        "Dropped total": dropped_right + dropped_wrong,
        "Dropped right": dropped_right,
        "Dropped wrong": dropped_wrong,
        "Final kept total": final_kept_right + final_kept_wrong,
        "Final kept right": final_kept_right,
        "Final kept wrong (missed switch)": final_kept_wrong
    }
# Example usage:
# result = simulate_switching_runs()
# print(result)

if __name__ == "__main__":
    # best_f1s = {}
    # for budget in [100]:
    #     best_threshold = 0.0
    #     best_f1 = -np.inf
    #     for threshold in [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]:
    #         f1 = test_models(
    #             model_path=f"trained_models/rf_model_budget{budget}.joblib",
    #             csv_path=f"testing_data/merged_features_per_rep{budget}.csv",
    #             threshold=threshold
    #         )
    #         if f1 > best_f1:
    #             best_f1 = f1
    #             best_threshold = threshold
    #     best_f1s[budget] = (best_threshold, best_f1) 
    
    # print("Best thresholds and F1 scores for each budget:")
    # for budget, (threshold, f1) in best_f1s.items():
    #     print(f"Budget {budget}: Threshold = {threshold}, F1 Score = {f1:.4f}")

    stats = simulate_switching_runs(thresholds={
        100: 0.3,
        150: 0.31,
        200: 0.2,
        250: 0.2,
        300: 0.1
    }, plot=True)
    print("Simulation Results:")
    for key, value in stats.items():
        print(f"{key}: {value}")
# Best thresholds and F1 scores for each budget:
# Budget 100: Threshold = 0.3, F1 Score = 0.5563
# Budget 150: Threshold = 0.31, F1 Score = 0.6250
# Budget 200: Threshold = 0.32, F1 Score = 0.6883
# Budget 250: Threshold = 0.34, F1 Score = 0.7053
# Budget 300: Threshold = 0.33, F1 Score = 0.7293