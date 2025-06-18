import joblib
import pandas as pd
from pathlib import Path
# from ioh import get_problem, ProblemClass
import os
from functools import reduce

def load_untrained_performance_selectors(directory="algo_performance_models"):
    """
    Load all untrained PerformanceModel selectors from the given directory.

    Returns:
        dict[int, PerformanceModel]: A dictionary mapping budget → untrained selector
    """
    selectors = {}

    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and "trained" not in filename:
            try:
                # e.g. model_B50.pkl
                budget = int(filename.split("_B")[1].split(".")[0])
                path = os.path.join(directory, filename)
                pipeline = joblib.load(path)
                selectors[budget] = pipeline.selector
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    return selectors

import numpy as np

def crossvalidated_static_predictions(
    budget,
    fold="instance",
    selector_dir="optimization/algo_performance_models_early",
    ela_template="../data/ela_with_algorithm_precisions/A1_B{budget}_5D_ela_with_state.csv",
):
    selector_path = os.path.join(selector_dir, f"model_B{budget}.pkl")
    pipeline = joblib.load(selector_path)
    selector = pipeline.selector

    df = pd.read_csv(ela_template.format(budget=budget))
    X = df.iloc[:, 4:-6]
    y = df.iloc[:, -6:]
    meta = df[["fid", "iid", "rep"]]
    X.index = y.index = list(zip(meta["fid"], meta["iid"], meta["rep"]))

    precision_results = []
    algorithm_results = []

    if fold == "instance":
        test_values = sorted(meta["iid"].unique())
        test_column = "iid"
    elif fold == "rep":
        # create 4 folds: [0–4], [5–9], [10–14], [15–19]
        # zip to 4 folds:
        test_values = [list(range(i, i + 5)) for i in range(0, 20, 5)]
        test_column = "rep"
    else:
        raise ValueError("fold must be 'instance' or 'rep'")

    for test_fold in test_values:
        if isinstance(test_fold, list):
            mask = meta[test_column].isin(test_fold)
        else:
            mask = meta[test_column] == test_fold

        print(f"🔍 Processing test {test_column} {test_fold} for budget {budget}...")

        train_keys = list(meta[~mask][["fid", "iid", "rep"]].itertuples(index=False, name=None))
        test_keys = list(meta[mask][["fid", "iid", "rep"]].itertuples(index=False, name=None))

        X_train, y_train = X.loc[train_keys], y.loc[train_keys]
        X_test = X.loc[test_keys]

        selector.algorithms = list(y.columns)
        selector.fit(X_train, y_train)
        predictions = selector.predict(X_test)

        for (fid, iid, rep), [(algo, _)] in predictions.items():
            precision_results.append({
                "fid": fid,
                "iid": iid,
                "rep": rep,
                f"static_B{budget}": y.at[(fid, iid, rep), algo]
            })
            algorithm_results.append({
                "fid": fid,
                "iid": iid,
                "rep": rep,
                f"alg_B{budget}": algo
            })

    return pd.DataFrame(precision_results), pd.DataFrame(algorithm_results)



def build_full_crossvalidated_table(precision_path):
    all_dfs = []
    all_algos = []
    precision_output = "predicted_static_precisions_rep_fold_late_sp.csv"
    algo_output = "selected_algorithms_rep_fold_late_sp.csv"

    precision_df = pd.read_csv(precision_path)

    # budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]
    budgets = [50*i for i in range(1, 21)]

    for budget in budgets:
        print(f"⏳ Processing budget {budget}...")

        if budget < 1000:
            df_b, df_a = crossvalidated_static_predictions(budget, fold="rep")
        else:
            # Use precision and algorithm "Same" directly
            df_b = precision_df.query("budget == 1000 and algorithm == 'Same'")
            df_b = df_b[["fid", "iid", "rep", "precision"]].rename(columns={"precision": "static_B1000"})

            df_a = df_b[["fid", "iid", "rep"]].copy()
            df_a["alg_B1000"] = "Same"

        all_dfs.append(df_b)
        all_algos.append(df_a)

        # Save merged results incrementally
        df_prec = reduce(lambda l, r: pd.merge(l, r, on=["fid", "iid", "rep"], how="outer"), all_dfs)
        df_algo = reduce(lambda l, r: pd.merge(l, r, on=["fid", "iid", "rep"], how="outer"), all_algos)

        df_prec = df_prec.sort_values(["fid", "iid", "rep"]).reset_index(drop=True)
        df_algo = df_algo.sort_values(["fid", "iid", "rep"]).reset_index(drop=True)

        df_prec.to_csv(precision_output, index=False)
        df_algo.to_csv(algo_output, index=False)
        print(f"💾 Saved: {precision_output}, {algo_output} [budget {budget}]")

    return df_prec, df_algo


if __name__ == "__main__":
    df_prec, df_algo = build_full_crossvalidated_table("../data/A2_precisions.csv")
    print("✅ Finished. Files saved.")