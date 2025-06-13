import sys
import pandas as pd
import os
import joblib
from asf.selectors import PerformanceModel, tune_selector

def tune_performance_model(budget: int):
    data = pd.read_csv(f"../data/ela_for_training/ela_with_algorithm_precisions/A1_B{budget}_5D_ela_with_state.csv")
    features = data.iloc[:, 4:-6]
    targets = data.iloc[:, -6:]
    groups = data["iid"]

    pipeline = tune_selector(
        X=features,
        y=targets,
        selector_class=[(PerformanceModel, {})],  # model is defined in configspace
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
        selector_class=[(PerformanceModel, {})],  # model is defined in configspace
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


if __name__ == "__main__":
    mode = sys.argv[1]  # "performance" or "switching"
    budget = int(sys.argv[2])

    if mode == "performance":
        tune_performance_model(budget)
    elif mode == "switching":
        tune_switching_model(budget)
    else:
        raise ValueError("Mode must be 'performance' or 'switching'")
