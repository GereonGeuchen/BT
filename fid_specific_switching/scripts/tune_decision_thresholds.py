import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import os
import sys
from multiprocessing import Pool

from ConfigSpace import ConfigurationSpace, Float, Categorical
from smac import HyperparameterOptimizationFacade, Scenario
from smac.main.config_selector import Configuration

from asf.predictors import RandomForestClassifierWrapper
from switch_model_optimization import SwitchingSelectorCV

# === Parse Command-line Arguments ===
if len(sys.argv) != 3:
    print("Usage: python tune_decision_thresholds.py <algorithm> <normalized:true|false>")
    sys.exit(1)

algorithm = sys.argv[1]
normalized = sys.argv[2].lower() == "true"

# Constants
SWITCHING_BUDGETS = [8 * i for i in range(1, 13)] + [50 * i for i in range(2, 20)]
FIDS = list(range(1, 25))
IIDS = [1, 2, 3, 4, 5]
REPS = list(range(20))

# Paths
PRECISION_FILE = f"../data/precision_files/A2_precisions_{algorithm}.csv"
THRESHOLD_OUTPUT_FILE = f"thresholds_{algorithm}.json"
ELA_DIR_SWITCH = f"../data/ela_for_training/A1_data_switch_{algorithm}"
ELA_DIR_ALGO = f"../data/ela_for_training/A1_data_ela_cma_std_precisions_{algorithm}"
SMAC_OUTPUT_DIR = f"smac_output_threshold_{algorithm}"
UNTRAINED_SWITCH_MODEL_PATH = f"../data/models/switching_{algorithm}"

PERF_MODELS_DIR = f"../data/models/trained_models/algo_performance_models_cv_{algorithm}"
SWITCH_MODELS_DIR = f"../data/models/trained_models/switching_models_cv_{algorithm}"

if normalized:
    ELA_DIR_SWITCH += "_normalized"
    ELA_DIR_ALGO += "_normalized"
    PERF_MODELS_DIR += "_normalized"
    SWITCH_MODELS_DIR += "_normalized"
    SMAC_OUTPUT_DIR += "_normalized"
    UNTRAINED_SWITCH_MODEL_PATH += "_normalized"
    THRESHOLD_OUTPUT_FILE = THRESHOLD_OUTPUT_FILE.replace(".json", "_normalized.json")

UNTRAINED_SWITCH_MODEL_PATH += "/switching.pkl"

def get_threshold_configspace():
    cs = ConfigurationSpace()
    threshold_values = [round(i * 0.05, 2) for i in range(21)]  # [0.0, 0.05, ..., 1.0]

    for budget in SWITCHING_BUDGETS:
        if budget < 500:
            cs.add_hyperparameter(
                Categorical(
                    name=f"threshold_b{budget}",
                    items=threshold_values,
                    default=0.5,
                    ordered=True,  # <-- ensures correct ordering semantics
                )
            )

    cs.add_hyperparameter(
        Categorical(
            name="threshold_b500plus",
            items=threshold_values,
            default=0.5,
            ordered=True,
        )
    )

    return cs

def evaluate_fold(test_iid, config, selector):
    train_iids = [iid for iid in IIDS if iid != test_iid]
    switching_models = {}
    performance_models = {}

    for budget in SWITCHING_BUDGETS:
        switch_model_path = Path(SWITCH_MODELS_DIR) / f"iid{test_iid}/selector_B{budget}_trained.pkl"
        ela_path = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela_with_state.csv"

        if not ela_path.exists():
            continue

        if not switch_model_path.exists():
            df = pd.read_csv(ela_path)
            df = df[df["iid"].isin(train_iids)]
            df = df.drop(columns=["Same", "Non-elitist", "MLSL", "PSO", "DE", "BFGS"], errors="ignore")
            X_train = df.iloc[:, 4:].drop(columns=["switch"])
            y_train = df["switch"]
            model = joblib.load(Path(UNTRAINED_SWITCH_MODEL_PATH))
            model.fit(X_train, y_train)
            os.makedirs(switch_model_path.parent, exist_ok=True)
            joblib.dump(model, switch_model_path)
        else:
            model = joblib.load(switch_model_path)

        threshold_key = f"threshold_b{budget}" if budget < 500 else "threshold_b500plus"
        threshold = config[threshold_key]

        class ThresholdWrapper:
            def __init__(self, model, threshold):
                self.model = model.model_class
                self.threshold = threshold

            def predict(self, X):
                # Only one class was seen during training → always return that class
                if len(self.model.classes_) == 1:
                    only_class = self.model.classes_[0]
                    return np.full(len(X), fill_value=only_class, dtype=bool)

                # Normal case: both classes present
                class_index = list(self.model.classes_).index(1)
                probs = self.model.predict_proba(X)
                return (probs[:, class_index] >= self.threshold)
            
        switching_models[budget] = ThresholdWrapper(model, threshold)

        perf_model_path = Path(PERF_MODELS_DIR) / f"iid{test_iid}/selector_B{budget}_trained.pkl"
        if perf_model_path.exists():
            performance_models[budget] = joblib.load(perf_model_path)

    fold_precision = 0.0
    for fid in FIDS:
        for rep in REPS:
            fold_precision += selector.simulate_single_run(fid, test_iid, rep, switching_models, performance_models)
    return fold_precision

def smac_threshold_objective(config: dict, seed: int = 0):
    selector = SwitchingSelectorCV(PRECISION_FILE)

    with Pool(processes=5) as pool:
        results = pool.starmap(
            evaluate_fold,
            [(iid, config, selector) for iid in IIDS]
        )

    return sum(results)

def main():
    cs = get_threshold_configspace()

    # # Build initial configuration with all thresholds = 0.5
    # initial_config_dict = {
    #     hp.name: 0.5 for hp in cs.get_hyperparameters()
    # }
    # initial_config = Configuration(cs, values=initial_config_dict)

    scenario = Scenario(
        configspace=cs,
        n_trials=1000,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory=SMAC_OUTPUT_DIR + "test",
        seed=42,
        use_default_config=True
    )

    smac = HyperparameterOptimizationFacade(scenario, smac_threshold_objective)
    best_config = smac.optimize()

    print("\nBest threshold configuration found:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    with open(THRESHOLD_OUTPUT_FILE, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nThresholds saved to: {THRESHOLD_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
