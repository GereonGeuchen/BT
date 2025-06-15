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
        runcount_limit=75,  
        seed=42,
        output_dir=f"./smac_output_performance/B{budget}_performance"
    )
    os.makedirs("algo_performance_models", exist_ok=True)
    joblib.dump(pipeline, f"algo_performance_models/model_B{budget}.pkl")


def tune_switching_model(budget: int):
    data = pd.read_csv(f"../data/ela_for_training/ela_with_optimal_precisions_ahead/A1_B{budget}_5D_ela_with_state.csv")
    number_of_predictions = (1000 - budget) // 50 + 1

    features = data.iloc[:, 4:-number_of_predictions]
    targets = data.iloc[:, -number_of_predictions:]
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
        runcount_limit=75,  
        seed=42,
        output_dir=f"./smac_output_switching/B{budget}_switching"
    )
    os.makedirs("switching_prediction_models", exist_ok=True)
    joblib.dump(pipeline, f"switching_prediction_models/model_B{budget}.pkl")


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
        data_path = f"../data/ela_for_training/ela_with_algorithm_precisions/A1_B{budget}_5D_ela_with_state.csv"
        save_path = f"algo_performance_models_trained/selector_B{budget}_trained.pkl"
        y_cols = -6
    else:
        input_path = f"switching_prediction_models/model_B{budget}.pkl"
        data_path = f"../data/ela_for_training/ela_with_optimal_precisions_ahead/A1_B{budget}_5D_ela_with_state.csv"
        save_path = f"switching_prediction_models_trained/selector_B{budget}_trained.pkl"
        if budget < 100:
            y_cols = -(19 + ( (96 - budget) // 8 ) + 1) 
        else:
            y_cols = -(((1000 - budget) // 50 + 1))

    print(f"Loading pipeline: {input_path} and data: {data_path}")
    pipeline = joblib.load(input_path)
    selector = pipeline.selector  # extract selector only

    data = pd.read_csv(data_path)
    features = data.iloc[:, 4:y_cols]
    targets = data.iloc[:, y_cols:]
    print(f"Target columns: {targets.columns.tolist()}")
    features.index = list(zip(data["fid"], data["iid"], data["rep"]))
    targets.index = features.index

    selector.algorithms = list(targets.columns)  # required before fitting
    selector.fit(features, targets)

    print(f"Trained selector on {features.shape[0]} rows")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(selector, save_path)
    print(f"Saved trained selector to: {save_path}")



if __name__ == "__main__":
    budgets = [8*i for i in range(5, 13)] + [50 * i for i in range(2, 21)]
  
    for budget in budgets:
        train_and_save_selector_only("performance", budget)
        train_and_save_selector_only("switching", budget)
