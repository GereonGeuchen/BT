import numpy as np
import pandas as pd
from pathlib import Path
import os
import joblib
from functools import partial
from itertools import combinations
from multiprocessing import Pool

# ========== ConfigSpace and SMAC imports ==========
from ConfigSpace import ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

# === Import your RandomForestClassifierWrapper ===
from asf.predictors import RandomForestClassifierWrapper

# === Your switching budgets ===
SWITCHING_BUDGETS = [8*i for i in range(1, 13)] + [50*i for i in range(2, 20)]

# === Your instance IDs for evaluation ===
FIDS = list(range(1, 25))
IIDS = [1, 2, 3, 4, 5] 
TEST_IIDS = [6 ,7]
REPS = list(range(20))
N_TRIALS = 200



# === Paths ===
algorithm = None # or "clipped"
normalized = True
if algorithm is None:
    ELA_DIR_SWITCH_TRAINING = "../data/ela_for_training/A1_data_switch"
    ELA_DIR_TEST = "../data/ela_for_training/A1_data_ela_cma_std_newInstances"
    TRAINED_ALGO_MODELS_DIR = "../data/models/trained_models/algo_performance_models_trained"
    PRECISION_FILE = "../data/precision_files/A2_precisions_newInstances_normalized_log10.csv"
    SMAC_OUTPUT_DIR = "smac_output"
    OUTPUT_PATH = "../data/models/tuned_models/switching_models"
else:
    ELA_DIR_SWITCH_TRAINING = f"../data/ela_for_training/A1_data_switch_{algorithm}"
    PRECISION_FILE = f"../data/precision_files/A2_precisions_{algorithm}.csv"
    SMAC_OUTPUT_DIR = f"smac_output_{algorithm}"
    OUTPUT_PATH = f"../data/models/tuned_models/switching_models_{algorithm}"

if normalized:
    ELA_DIR_SWITCH_TRAINING += "_normalized_log10_200"
    ELA_DIR_TEST += "_normalized"
    TRAINED_ALGO_MODELS_DIR += "_normalized_log10_200"
    SMAC_OUTPUT_DIR += "_normalized_newInstances_log10_200_200_test"
    OUTPUT_PATH += "_normalized_newInstances_log10_200_200_test"

# ========== Helper classes ==========

class SwitchingSelectorCV:
    def __init__(self, precision_file):
        self.precision_df = pd.read_csv(precision_file)

    def simulate_single_run(self, fid, iid, rep, switching_models, performance_models):
        """
        Simulates a run using provided switching and performance models dict (budget → trained model).
        """

        for budget in SWITCHING_BUDGETS:
            switch_model = switching_models.get(budget)
            perf_model = performance_models.get(budget)
            if switch_model is None or perf_model is None:
                continue

            ela_path = Path(ELA_DIR_TEST) / f"A1_B{budget}_5D_ela_with_state.csv"
            if not ela_path.exists():
                print(f"ELM file not found: {ela_path}")
                continue

            df = pd.read_csv(ela_path)
            row = df[(df["fid"] == fid) & (df["iid"] == iid) & (df["rep"] == rep)]
            if row.empty:
                continue

            features = row.iloc[:, 4:]
            features.index = [(fid, iid, rep)]
            should_switch = switch_model.predict(features)[0]

            if should_switch:
                algo_prediction = perf_model.predict(features)
                predicted_algorithm = list(algo_prediction.values())[0][0][0]

                match_row = self.precision_df[
                    (self.precision_df["fid"] == fid) &
                    (self.precision_df["iid"] == iid) &
                    (self.precision_df["rep"] == rep) &
                    (self.precision_df["budget"] == budget) &
                    (self.precision_df["algorithm"] == predicted_algorithm)
                ]

                precision = match_row["precision"].values[0] if not match_row.empty else np.inf
                return precision

        # Fallback to budget 1000 CMA-ES
        fallback_row = self.precision_df[
            (self.precision_df["fid"] == fid) &
            (self.precision_df["iid"] == iid) &
            (self.precision_df["rep"] == rep) &
            (self.precision_df["budget"] == 1000) &
            (self.precision_df["algorithm"] == "Same")
        ]
        fallback_precision = fallback_row["precision"].values[0] if not fallback_row.empty else np.inf
        return fallback_precision

def evaluate_selector(fids, selector, performance_models, switching_models):
    total_precision = 0.0
    for fid in fids:
        for iid in TEST_IIDS:
            for rep in REPS:
                precision = selector.simulate_single_run(fid, iid, rep, switching_models, performance_models)
                total_precision += precision
    return total_precision

# ========== Objective function for SMAC ==========

def smac_objective(config, seed):
    np.random.seed(seed)
    selector = SwitchingSelectorCV(PRECISION_FILE)

    wrapper_partial = RandomForestClassifierWrapper.get_from_configuration(config, random_state=42)

    performance_models = {}
    switching_models = {}

    for budget in SWITCHING_BUDGETS:
        # Load trained performance models
        performance_models[budget] = joblib.load(f"{TRAINED_ALGO_MODELS_DIR}/selector_B{budget}_trained.pkl")

        # Train switching models according to the configuration
        ela_path_switch = Path(ELA_DIR_SWITCH_TRAINING) / f"A1_B{budget}_5D_ela_with_state.csv"
        train_df = pd.read_csv(ela_path_switch)
        train_df = train_df.drop(columns=["Same", "Non-elitist", "MLSL", "PSO", "DE", "BFGS"])

        model = wrapper_partial()
        X_train = train_df.iloc[:, 4:].drop(columns=["switch"])
        y_train = train_df["switch"]
        model.fit(X_train, y_train)
        switching_models[budget] = model


    print(f"Evaluating config: {config}")

    fid_chunks = [FIDS[i:i+4] for i in range(0, len(FIDS), 4)]

    # with Pool(processes=1) as pool:  # six jobs, each handles 4 FIDs
    #     # partial fixes the shared args; each job receives a list of 4 FIDs
    #     worker = partial(
    #         evaluate_selector,
    #         selector=selector,
    #         performance_models=performance_models,
    #         switching_models=switching_models,
    #     )
    #     results = pool.starmap(worker, [(chunk,) for chunk in fid_chunks])
    # build the worker once
    worker = partial(
        evaluate_selector,
        selector=selector,
        performance_models=performance_models,
        switching_models=switching_models,
    )

    # run strictly sequentially in the main process
    results = [worker(chunk) for chunk in fid_chunks]


    total_precision = sum(results)
    print(f"Config {config} → Total holdout precision: {total_precision}")
    return total_precision

# ========== Main SMAC tuning routine ==========

def main():
    cs = RandomForestClassifierWrapper.get_configuration_space()

    scenario = Scenario(
        configspace=cs,
        n_trials=N_TRIALS,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory=SMAC_OUTPUT_DIR,
        seed=42
    )

    smac = HyperparameterOptimizationFacade(scenario, smac_objective)
    best_config = smac.optimize()

    print("Best configuration found:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    wrapper_partial = RandomForestClassifierWrapper.get_from_configuration(best_config, random_state=42)
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    for budget in SWITCHING_BUDGETS:
        ela_path_switch = Path(ELA_DIR_SWITCH_TRAINING) / f"A1_B{budget}_5D_ela_with_state.csv"
        if not ela_path_switch.exists():
            continue
        train_df = pd.read_csv(ela_path_switch)
        train_df = train_df.drop(columns=["Same", "Non-elitist", "MLSL", "PSO", "DE", "BFGS"])
        X_train = train_df.iloc[:, 4:].drop(columns=["switch"])
        y_train = train_df["switch"]
        model = wrapper_partial()
        model.fit(X_train, y_train)
        model_path = output_dir / f"switching_model_B{budget}_trained.pkl"
        joblib.dump(model, model_path)
        print(f"Saved switching model for budget {budget} to {model_path}")

    print("All final switching models trained and saved successfully.")
    
if __name__ == "__main__":
    main()