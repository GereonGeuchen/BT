from pyexpat import model
import sys
import pandas as pd
import os
import joblib
from asf.predictors import RandomForestClassifierWrapper
from smac import HyperparameterOptimizationFacade, Scenario
import numpy as np

IIDS = [1, 2, 3, 4, 5]
BUDGET = 0
# Objective: Predict first column in data using features (starting from 5th column), measuring accuracy, 
# do five-fold CV, using one instance out each

def evaluate_model_on_iid(model, iid):
    data = pd.read_csv(f"../data/A1_data_ela_cma_std_normalized_no_ps_ratio/A1_B{BUDGET}_5D_ela_with_state.csv")
    features = data.iloc[:, 4:]
    targets = data.iloc[:, 0]
    groups = data["iid"]

    train_mask = groups != iid
    test_mask = groups == iid

    X_train, y_train = features[train_mask], targets[train_mask]
    X_test, y_test = features[test_mask], targets[test_mask]

    model.fit(X_train, y_train)
    accuracy = model.model_class.score(X_test, y_test)
    return accuracy


def smac_objective(config, seed):
    np.random.seed(seed)
    scores = []
    for iid in IIDS:
        wrapper = RandomForestClassifierWrapper.get_from_configuration(config, random_state=42)
        model = wrapper()
        score = evaluate_model_on_iid(model, iid)
        scores.append(score)
    return -np.mean(scores)  # SMAC minimizes, so return negative accuracy

def tune_classifier(budget: int):
    cs = RandomForestClassifierWrapper.get_configuration_space()

    scenario = Scenario(
        configspace=cs,
        n_trials=200,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory=f"smac_output_classifier/B{budget}_classifier",
        seed=42
    )

    smac = HyperparameterOptimizationFacade(scenario, smac_objective)
    best_config = smac.optimize()

    wrapper = RandomForestClassifierWrapper.get_from_configuration(best_config, random_state=42)
    best_model = wrapper()

    # Fit model on all data
    data = pd.read_csv(f"../data/A1_data_ela_cma_std_normalized_no_ps_ratio/A1_B{budget}_5D_ela_with_state.csv")
    features = data.iloc[:, 4:]
    targets = data.iloc[:, 0]
    best_model.fit(features, targets)

    os.makedirs("trained_models_classifier", exist_ok=True)
    joblib.dump(best_model, f"trained_models_classifier/model_B{budget}.pkl")
    print(f"Best model for budget {budget} saved.")


if __name__ == "__main__":
    BUDGET = sys.argv[1]  
    tune_classifier(BUDGET)