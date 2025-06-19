import numpy as np
from sklearn.model_selection import KFold
from asf.utils.groupkfoldshuffle import GroupKFoldShuffle
from sklearn.metrics import f1_score
from smac import HyperparameterOptimizationFacade, Scenario
from asf.predictors.random_forest import RandomForestClassifierWrapper
import pandas as pd
import joblib
import os
import argparse

def tune_rf_classifier_f1(
    X,
    y,
    wrapper_class,  # RandomForestClassifierWrapper
    runcount_limit=100,
    timeout=np.inf,
    seed=42,
    cv=5,
    groups=None,
    output_dir="./smac_output_rf",
    smac_scenario_kwargs={},
    smac_kwargs={},
):
    """
    Tune RandomForestClassifierWrapper hyperparameters to maximize F1-score.
    Uses GroupKFoldShuffle if groups are given, else standard KFold.
    RF is always forced to random_state=42.
    """

    # Get config space from the wrapper
    cs = wrapper_class.get_configuration_space()

    # SMAC scenario config
    scenario = Scenario(
        configspace=cs,
        n_trials=runcount_limit,
        walltime_limit=timeout,
        deterministic=True,
        output_directory=output_dir,
        seed=seed,
        **smac_scenario_kwargs,
    )

    def target_function(config, seed):
        if groups is not None:
            kfold = GroupKFoldShuffle(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        f1_scores = []

        for train_idx, test_idx in kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Create wrapper
            wrapper = wrapper_class.get_from_configuration(config)()

            # Fit + predict via wrapper
            wrapper.fit(X_train, y_train)
            y_pred = wrapper.predict(X_test)

            f1 = f1_score(y_test, y_pred, average="binary")
            f1_scores.append(f1)

        print(f1)
        return 1 - np.mean(f1_scores)


    smac = HyperparameterOptimizationFacade(scenario, target_function, **smac_kwargs)
    best_config = smac.optimize()

    # Rebuild the best wrapper (untrained)
    best_wrapper = wrapper_class.get_from_configuration(best_config)()

    return best_wrapper, best_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True, help="Budget number (e.g. 20, 40, 60, ...)")
    parser.add_argument("--mode", choices=["late", "all"], required=True, help="'late' or 'all' switch")
    args = parser.parse_args()

    budget = args.budget
    mode = args.mode

    # Resolve correct CSV path and feature/target slices
    csv_path = f"../data/ela_for_training/ela_with_state_{mode}_switch_greater_budgets/A1_B{budget}_5D_ela_with_state.csv"

    data = pd.read_csv(csv_path)

    features = data.iloc[:, 4:-1] 
    targets = data.iloc[:, -1]
    groups = data["iid"]

    rf_wrapper, best_config = tune_rf_classifier_f1(
        X=features,
        y=targets,
        wrapper_class=RandomForestClassifierWrapper,
        runcount_limit=75,
        timeout=np.inf,
        seed=42,
        cv=5,
        groups=groups.values,
        output_dir=f"./smac_output_rf/B{budget}_{mode}"
    )

    # Save the trained wrapper and config for reproducibility
    os.makedirs("rf_models", exist_ok=True)
    model_path = f"rf_models/rf_model_B{budget}_{mode}.pkl"
    joblib.dump(rf_wrapper, model_path)
    print(f"✅ Saved model to {model_path}")
    



