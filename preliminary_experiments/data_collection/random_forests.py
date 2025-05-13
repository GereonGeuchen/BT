import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# === Config ===
train_file_path = "ela_initial_sampling_normalized.csv" # Add _normalized here if you want to use the normalized data
data_dir = "A1_data_ela_disp_normalized" # Add _normalized here if you want to use the normalized data
target_cols = ["fid", "high_level_category"]
budgets = [50*i for i in range(1, 21)]
enabled_metrics = ["accuracy", "precision", "recall", "f1"]
plot_results = True

# Had to exclude ela_meta.quad_simple.cond in excluded_columns and ela_meta.quad_w_interact.adj_r2 to avoid errors
# Used for initial sampling
excluded_columns = [
    "ela_meta.quad_simple.cond",
    "ela_meta.quad_w_interact.adj_r2",
    "ela_distr.costs_runtime",
    "ela_meta.costs_runtime",
    "ela_level.costs_runtime",
    "disp.costs_runtime",
    "ic.costs_runtime",
    "nbc.costs_runtime",
    "ela_level.lda_qda_10",
    "ela_level.mmce_lda_10",
    "ela_level.mmce_qda_10"
]


metric_funcs = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro", zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro", zero_division=0),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0)
}

def load_data(budget):
    path = f"{data_dir}/A1_B{budget}_5D_ela.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError: 
        print(f"❌ Missing file: {path}")
        return None

def preprocess(df, target_col, use_excluded_columns=True, budget=None):
    if use_excluded_columns:
        X = df.drop(columns=excluded_columns + ["fid", "iid", "high_level_category", "rep"], errors='ignore')
    else:
        # Identify columns that contain inf, -inf or NaN and drop them and print the column names
        inf_columns = df.columns[(df == np.inf).any() | (df == -np.inf).any() | df.isna().any()]
        if len(inf_columns) > 0:
            print(f"Columns with inf, -inf or NaN for budget {budget}: {inf_columns.tolist()}")
        X = df.drop(columns=["fid", "iid", "high_level_category", "rep"], errors='ignore')
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
    y = df[target_col]
    return X, y

# === Train on fixed dataset ===
def initial_sampling_rf():
    train_df = pd.read_csv(train_file_path)

    results = {t: {m: [] for m in enabled_metrics} for t in target_cols}

    for target_col in target_cols:
        X_train, y_train = preprocess(train_df, target_col)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)  

        for budget in budgets:
            df = load_data(budget)
            if df is None:
                for m in enabled_metrics:
                    results[target_col][m].append(np.nan)
                continue

            X_test, y_test = preprocess(df, target_col)

            y_pred = model.predict(X_test)

            for m in enabled_metrics:
                try:
                    score = metric_funcs[m](y_test, y_pred)  
                    results[target_col][m].append(score)
                except:
                    results[target_col][m].append(np.nan)


    # === Plotting ===
    if plot_results:
        for target_col in target_cols:
            plt.figure(figsize=(10, 6))

            ymin = 0.0
            ymax = 1.05

            for metric in enabled_metrics:
                scores = results[target_col][metric]
                if np.all(np.isnan(scores)):
                    continue
                plt.plot(budgets, scores, label=metric.capitalize(), marker='o')

            plt.title(f"Classification Metrics vs Budget ({target_col})")
            plt.xlabel("Budget")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.ylim(ymin, ymax)
            plt.grid(True)
            plt.legend(title="Metric")
            plt.tight_layout()
            plt.show()

def rf_over_budgets(include_variance_band=True):
    results_mean = {t: {m: [] for m in enabled_metrics} for t in target_cols}
    results_std = {t: {m: [] for m in enabled_metrics} for t in target_cols}

    for target_col in target_cols:
        print(f"\nEvaluating: {target_col}")

        for budget in budgets:
            print(f"  Budget: {budget}")
            df = load_data(budget)
            if df is None:
                for m in enabled_metrics:
                    results_mean[target_col][m].append(np.nan)
                    results_std[target_col][m].append(0)
                continue

            X, y = preprocess(df, target_col, use_excluded_columns=False, budget=budget)

            scores = {m: [] for m in enabled_metrics}

            for fold in range(1, 6):
                test_idx = df[df["iid"] == fold].index
                train_idx = df[df["iid"] != fold].index

                X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                for m in enabled_metrics:
                    try:
                        score = metric_funcs[m](y_test, y_pred)
                        scores[m].append(score)
                    except:
                        scores[m].append(np.nan)

            for m in enabled_metrics:
                results_mean[target_col][m].append(np.nanmean(scores[m]))
                results_std[target_col][m].append(np.nanstd(scores[m]))

    # === Plotting ===
    if plot_results:
        for target_col in target_cols:
            plt.figure(figsize=(10, 6))

            all_means = np.array([results_mean[target_col][m] for m in enabled_metrics])
            all_stds = np.array([results_std[target_col][m] for m in enabled_metrics])
            ymax = min(1.05, np.nanmax(all_means + all_stds))
            ymin = max(0.0, np.nanmin(all_means - all_stds))

            # Add 10% padding
            padding = (ymax - ymin) * 0.1
            ymin = max(0.0, ymin - padding)
            ymax += padding

            for m in enabled_metrics:
                means = np.array(results_mean[target_col][m])
                stds = np.array(results_std[target_col][m])

                if np.all(np.isnan(means)):
                    continue

                plt.plot(budgets, means, label=m.capitalize(), marker='o')
                if include_variance_band:
                    plt.fill_between(budgets, means - stds, means + stds, alpha=0.2)

            plt.title(f"Classification Metrics vs Budget ({target_col})")
            plt.xlabel("Budget")
            plt.ylabel("Score")
            plt.ylim(ymin, ymax)
            plt.grid(True)
            plt.legend(title="Metric")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    initial_sampling_rf()
    # rf_over_budgets()
    # Uncomment the line below to run the function
    # initial_sampling_rf()