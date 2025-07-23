import numpy as np
import pandas as pd
from pathlib import Path
import os
import joblib
from functools import partial
from itertools import combinations

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
REPS = list(range(20))

# === Paths ===
ELA_DIR_SWITCH = "../data/ela_for_training/A1_data_all_switch_clipped"
ELA_DIR_ALGO = "../data/ela_for_training/A1_data_ela_cma_std_precisions_normalized_clipped"
PRECISION_FILE = "../data/precision_files/A2_precisions_clipped.csv"
ELA_DIR = "../data/ela_for_training/A1_data_ela_cma_std_precisions_normalized_clipped"

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

            ela_path = Path(ELA_DIR) / f"A1_B{budget}_5D_ela_with_state.csv"
            if not ela_path.exists():
                continue

            df = pd.read_csv(ela_path)
            df = df.iloc[:, :-6]
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

# ========== Objective function for SMAC ==========

def smac_objective(config, seed):
    np.random.seed(seed)

    wrapper_partial = RandomForestClassifierWrapper.get_from_configuration(config)
    total_precision = 0.0

    selector = SwitchingSelectorCV(PRECISION_FILE)

    # === Leave-instance-out CV ===
    iids = IIDS.copy()
    for test_iid in iids:
        print(f"Evaluating test iid {test_iid} with config {config}")
        train_iids = [iid for iid in iids if iid != test_iid]

        # Train models on train_iids
        switching_models = {}
        performance_models = {}

        for budget in SWITCHING_BUDGETS:
            print(f"Training models for budget {budget}...")
            # === Train switching model ===
            ela_path_switch = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela_with_state.csv"
            if not ela_path_switch.exists():
                print(f"Skipping budget {budget} - no switching data available.")
                continue
            train_df = pd.read_csv(ela_path_switch)
            train_df = train_df.drop(columns=["Same", "Non-elitist", "MLSL", "PSO", "DE", "BFGS"])
            train_df = train_df[train_df["iid"].isin(train_iids)]

            model = wrapper_partial()
            X_train = train_df.iloc[:, 4:].drop(columns=["switch"])
            y_train = train_df["switch"]  # adjust target column as needed
            model.fit(X_train, y_train)
            switching_models[budget] = model

            # === Train performance model ===
            ela_path_algo = Path(ELA_DIR_ALGO) / f"A1_B{budget}_5D_ela_with_state.csv"
            if not ela_path_algo.exists():
                print(f"Skipping budget {budget} - no performance data available.")
                continue
            train_df = pd.read_csv(ela_path_algo)
            train_df = train_df[train_df["iid"].isin(train_iids)]
            X_train = train_df.iloc[:, 4:-6]
            y_train= train_df.iloc[:, -6:]

            trained_model_path = Path(f"../data/models/trained_models/algo_performance_models_iid{test_iid}/selector_B{budget}_trained.pkl")
            if trained_model_path.exists():
                print(f"Loading existing model for budget {budget}...")
                perf_model = joblib.load(trained_model_path)
            else:
                print(f"Training new model for budget {budget}...")
                perf_model = joblib.load(f"../data/models/algo_performance_models_clipped/model_B{budget}.pkl").selector
                perf_model.fit(X_train, y_train)
                os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
                joblib.dump(perf_model, trained_model_path)

            performance_models[budget] = perf_model

        # === Evaluate on test_iid ===
        for fid in FIDS:
            for rep in REPS:
                print(f"Simulating fid {fid}, iid {test_iid}, rep {rep}")
                precision = selector.simulate_single_run(fid, test_iid, rep, switching_models, performance_models)
                total_precision += precision

    print(f"Config {config} → Total CV precision: {total_precision}")
    return total_precision

# ========== Main SMAC tuning routine ==========

def main():
    # 1. Get configuration space
    cs = RandomForestClassifierWrapper.get_configuration_space()

    # 2. Define SMAC scenario
    scenario = Scenario(
        configspace=cs,
        n_trials=75,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory="smac_switching_output_cv",
        seed=42
    )

    # 3. Run SMAC
    smac = HyperparameterOptimizationFacade(scenario, smac_objective)
    best_config = smac.optimize()

    print("Best configuration found:")
    for k,v in best_config.items():
        print(f"  {k}: {v}")

    wrapper_partial = RandomForestClassifierWrapper.get_from_configuration(best_config)
    output_dir = Path("final_trained_binary_switching_models_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    for budget in SWITCHING_BUDGETS:
        ela_path_switch = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela_with_state.csv" 
        train_df = pd.read_csv(ela_path_switch)
        # === Train switching model ===
        model = wrapper_partial()
        # Drop columns switch, MLSL, PSO, DE, BFGS, Same, Non-elitist
        train_df = train_df.drop(columns=["Same", "Non-elitist", "MLSL", "PSO", "DE", "BFGS"])
        X_train = train_df.iloc[:, 4:].drop(columns=["switch"])
        y_train = train_df["switch"]  # adjust to your target column
        model.fit(X_train, y_train)

        # === Save model ===
        model_path = output_dir / f"switching_model_B{budget}_trained.pkl"
        joblib.dump(model, model_path)
        print(f"Saved switching model for budget {budget} to {model_path}")

    print("All final switching models trained and saved successfully.")

if __name__ == "__main__":
    main()
