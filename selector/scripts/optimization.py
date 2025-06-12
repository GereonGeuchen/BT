from asf.selectors import PerformanceModel 
from asf.selectors import tune_selector
from asf.predictors import RandomForestRegressorWrapper
from asf.predictors import epm_random_forest
import os
import joblib
from joblib import Parallel, delayed
import pandas as pd

def predict_algo_performance():
    # Load data
    data = pd.read_csv("../data/ela_with_algorithm_precisions/A1_B50_5D_ela_with_state.csv")

    # Split
    train_data = data[data["iid"] != 5]
    test_data = data[data["iid"] == 5]

    # Extract features and performance targets
    train_features = train_data.iloc[:, 4:-6]
    test_features = test_data.iloc[:, 4:-6]
    train_performances = train_data.iloc[:, -6:]
    
    # Set a MultiIndex based on (fid, iid, rep)
    train_features.index = list(zip(train_data['fid'], train_data['iid'], train_data['rep']))
    test_features.index = list(zip(test_data['fid'], test_data['iid'], test_data['rep']))
    # Extract algorithm names from performance column headers
    algorithm_names = list(train_performances.columns)

    # Initialize selector
    selector = PerformanceModel(model_class=RandomForestRegressorWrapper)
    selector.algorithms = algorithm_names
    selector.budget = 50  
    selector.maximize = False  

    # Train
    selector.fit(train_features, train_performances)

    # Predict
    predictions = selector.predict(test_features)

    # Convert predictions dict to a flat DataFrame
    predictions_df = pd.DataFrame([
        {
            "fid": fid,
            "iid": iid,
            "rep": rep,
            "predicted_algorithm": algo,
            "budget": budget,
        }
        for (fid, iid, rep), [(algo, budget)] in predictions.items()
    ])

    # Save
    predictions_df.to_csv("predictions.csv", index=False)

def predict_switching_point():
    # Load data
    data = pd.read_csv("../data/ela_with_optimal_precisions_ahead/A1_B50_5D_ela_with_state.csv")

    # Split
    train_data = data[data["iid"] != 5]
    test_data = data[data["iid"] == 5]

    # Extract features and performance targets
    train_features = train_data.iloc[:, 4:-20]
    test_features = test_data.iloc[:, 4:-20]
    train_performances = train_data.iloc[:, -20:]
    
    # Set a MultiIndex based on (fid, iid, rep)
    train_features.index = list(zip(train_data['fid'], train_data['iid'], train_data['rep']))
    test_features.index = list(zip(test_data['fid'], test_data['iid'], test_data['rep']))


    # Extract algorithm names from performance column headers
    algorithm_names = list(train_performances.columns)

    # Initialize selector
    selector = PerformanceModel(model_class=RandomForestRegressorWrapper)
    selector.algorithms = algorithm_names
    selector.budget = 50  
    selector.maximize = False  

    # Train
    selector.fit(train_features, train_performances)

    # Predict
    predictions = selector.predict(test_features)

    # Convert predictions dict to a flat DataFrame
    predictions_df = pd.DataFrame([
        {
            "fid": fid,
            "iid": iid,
            "rep": rep,
            "predicted_algorithm": algo,
            "budget": budget,
        }
        for (fid, iid, rep), [(algo, budget)] in predictions.items()
    ])

    # Save
    predictions_df.to_csv("switching_predictions.csv", index=False)
def tune_performance_model(budget: int):
    data = pd.read_csv(f"../data/ela_for_training/ela_with_algorithm_precisions/A1_B{budget}_5D_ela_with_state.csv")
    features = data.iloc[:, 4:-6]
    targets = data.iloc[:, -6:]
    groups = data["iid"]

    pipeline = tune_selector(
        X=features,
        y=targets,
        selector_class=[(PerformanceModel, {})],  # Let config space define the model
        selector_kwargs={"random_state": 42},
        budget=budget,
        maximize=False,
        groups=groups.values,
        cv=5,
        runcount_limit=5,  # increased for better optimization
        timeout=1000,
        seed=42,
        output_dir=f"./smac_output_performance/B{budget}_performance"
    )
    os.makedirs("algo_performance_models_test", exist_ok=True)
    joblib.dump(pipeline, f"algo_performance_models_test/model_B{budget}.pkl")


def tune_switching_model(budget: int):
    data = pd.read_csv(f"../data/ela_for_training/ela_with_optimal_precisions_ahead/A1_B{budget}_5D_ela_with_state.csv")
    number_of_predictions = (1000 - budget) // 50 + 1

    features = data.iloc[:, 4:-number_of_predictions]
    targets = data.iloc[:, -number_of_predictions:]
    groups = data["iid"]

    pipeline = tune_selector(
        X=features,
        y=targets,
        selector_class=[(PerformanceModel, {"model_class": RandomForestRegressorWrapper(init_params={"random_state": 42})})],
        budget=budget,
        maximize=False,
        groups=groups.values,
        cv=5,
        runcount_limit=3,
        timeout=1000,
        seed=42,
        output_dir=f"./smac_output_switching/B{budget}_switching"
    )
    os.makedirs("switching_prediction_models_test", exist_ok=True)
    joblib.dump(pipeline, f"switching_prediction_models_test/model_B{budget}.pkl")


def train_and_save_selector_only(mode: str, budget: int):
    """
    Load an optimized pipeline (from SMAC), extract the selector,
    train it on *all* instances (iid ∈ {1–5}), and save just the selector.
    
    Parameters:
        mode (str): "performance" or "switching"
        budget (int): e.g., 50, 100, ..., 1000
    """
    assert mode in {"performance", "switching"}, "Mode must be 'performance' or 'switching'"

    if mode == "performance":
        input_path = f"algo_performance_models/model_B{budget}.pkl"
        data_path = f"../data/ela_with_algorithm_precisions/A1_B{budget}_5D_ela_with_state.csv"
        save_path = f"algo_performance_models/selector_B{budget}_trained.pkl"
        y_cols = -6
    else:
        input_path = f"switching_prediction_models/model_B{budget}.pkl"
        data_path = f"../data/ela_with_optimal_precisions_ahead/A1_B{budget}_5D_ela_with_state.csv"
        save_path = f"switching_prediction_models/selector_B{budget}_trained.pkl"
        y_cols = -((1000 - budget) // 50 + 1)

    print(f"Loading pipeline: {input_path}")
    pipeline = joblib.load(input_path)
    selector = pipeline.selector  # extract selector only

    data = pd.read_csv(data_path)
    features = data.iloc[:, 4:y_cols]
    targets = data.iloc[:, y_cols:]
    features.index = list(zip(data["fid"], data["iid"], data["rep"]))
    targets.index = features.index

    selector.algorithms = list(targets.columns)  # required before fitting
    selector.fit(features, targets)

    print(f"Trained selector on {features.shape[0]} rows")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(selector, save_path)
    print(f"Saved trained selector to: {save_path}")



if __name__ == "__main__":
    # budgets = [50 * i for i in range(1, 21)]
    # jobs = []

    # for budget in budgets:
    #     jobs.append(delayed(tune_performance_model)(budget))
    #     jobs.append(delayed(tune_switching_model)(budget))

    # # Run with 4 parallel workers (adjust n_jobs based on your CPU and memory)
    # Parallel(n_jobs=8, backend="loky", verbose=10)(jobs)
    # for budget in [50*i for i in range(1, 20)]:
    #     train_and_save_selector_only("performance", budget)
    #     train_and_save_selector_only("switching", budget)
    # tune_performance_model(50)
    trained_test_pipeline = joblib.load("algo_performance_models_test/model_B50.pkl")
    selector = trained_test_pipeline.selector
    fit_data = pd.read_csv("../data/ela_for_training/ela_with_algorithm_precisions/A1_B50_5D_ela_with_state.csv")
    features = fit_data.iloc[:, 4:-6]
    targets = fit_data.iloc[:, -6:]
    features.index = list(zip(fit_data["fid"], fit_data["iid"], fit_data["rep"]))
    targets.index = features.index
    selector.algorithms = list(targets.columns)  # required before fitting
    # selector.fit(features, targets)
    # print(f"Trained selector on {features.shape[0]} rows")
   
    #    === 5-Fold CV using iid as group ===
    from sklearn.model_selection import GroupKFold
    import numpy as np

    print("\n=== Simulating SMAC target_function with 5-fold CV ===")
    group_kfold = GroupKFold(n_splits=5)
    groups = fit_data["iid"]
    fold_sums = []

    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(features, targets, groups=groups)):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]

        # Restore original MultiIndex
        X_train.index = features.index[train_idx]
        X_test.index = features.index[test_idx]
        y_train.index = targets.index[train_idx]
        y_test.index = targets.index[test_idx]

        # ❗ Load a fresh copy of the selector for this fold
        fold_selector = joblib.load("algo_performance_models_test/model_B50.pkl").selector
        fold_selector.algorithms = list(targets.columns)
        fold_selector.budget = 50
        fold_selector.maximize = False

        # Fit on training fold and predict
        fold_selector.fit(X_train, y_train)
        for wrapper in selector.regressors:
            print(f"Evaluating next model")
            print(f"Model parameters: {wrapper.model_class.get_params()}")
        predictions = fold_selector.predict(X_test)

        # Verify seed propagation
        print(fold_selector.regressors[0].model_class.get_params()["random_state"])

        # Evaluate performance
        fold_sum = 0.0
        for instance, schedule in predictions.items():
            if not schedule or instance not in y_test.index:
                continue
            algo, _ = schedule[0]
            if algo not in y_test.columns:
                continue
            fold_sum += y_test.at[instance, algo]

        print(f"Fold {fold_idx + 1}: precision sum = {fold_sum:.2f}")
    fold_sums.append(fold_sum)
    print("\n=== Summary ===")
    print(f"Fold-wise sums: {[round(s, 2) for s in fold_sums]}")
    print(f"Total precision sum across folds: {np.sum(fold_sums):.2f}")