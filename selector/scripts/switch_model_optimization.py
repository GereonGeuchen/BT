import numpy as np
import pandas as pd
from pathlib import Path
import os
import joblib
from functools import partial
from itertools import combinations
from multiprocessing import Pool

# ========== ConfigSpace and SMAC imports ==========
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

# === Import your RandomForestClassifierWrapper ===
from asf.selectors import PerformanceModel

# === Your switching budgets ===
SWITCHING_BUDGETS = [8*i for i in range(1, 13)] + [50*i for i in range(2, 20)]

# === Your instance IDs for evaluation ===
FIDS = list(range(1, 25))
IIDS = [1, 2, 3, 4, 5] 
REPS = list(range(20))



# === Paths ===
ELA_PRECS_AHEAD = "../data/A1_data_ela_cma_std_normalized_no_ps_ratio_optimal"
ELA_DIR_ALGO = "../data/A1_data_ela_cma_std_precisions_normalized_no_ps_ratio"
PRECISION_FILE = "../data/precision_files/A2_precisions_normalized_log10.csv"
CV_MODELS_DIR = "../models/algo_performance_models_cv_normalized_log10_200_no_ps_ratio"
UNTRAINED_PERF_MODELS_DIR = "../models/algo_performance_models_normalized_log10_200_no_ps_ratio"
SMAC_OUTPUT_DIR = "smac_output"
OUTPUT_PATH = "../models/tuned_models/switching_models"

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

            ela_path = Path(ELA_DIR_ALGO) / f"A1_B{budget}_5D_ela_with_state.csv"
            if not ela_path.exists():
                continue

            df = pd.read_csv(ela_path)
            df = df.iloc[:, :-6]
            row = df[(df["fid"] == fid) & (df["iid"] == iid) & (df["rep"] == rep)]
            if row.empty:
                continue

            features = row.iloc[:, 4:]
            features.index = [(fid, iid, rep)]
            
            prediction = switch_model.predict(features)
            predicted_switch_budget = int(list(prediction.values())[0][0][0])
            print(f"F{fid}-I{iid}-R{rep} | True budget: {budget}, Predicted switch budget: {predicted_switch_budget}")
            if predicted_switch_budget == budget:

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

def train_models_for_iid(test_iid, config, selector):
    train_iids = [iid for iid in IIDS if iid != test_iid]
    # performance_model_partial = PerformanceModel.get_from_configuration(config, random_state=42)
    switching_models = {}
    performance_models = {}

    for budget in SWITCHING_BUDGETS:
        print(f"Training models for budget {budget} and test iid {test_iid}")
        if budget < 100:
            number_of_predictions = 19 + ( (96 - budget) // 8 ) + 1
        else:
            number_of_predictions = (1000 - budget) // 50 + 1

        ela_path_precs_ahead = Path(ELA_PRECS_AHEAD) / f"A1_B{budget}_5D_ela_with_state.csv"
        if not ela_path_precs_ahead.exists():
            continue
        train_df = pd.read_csv(ela_path_precs_ahead)
        train_df = train_df[train_df["iid"].isin(train_iids)]

        model = PerformanceModel.get_from_configuration(config, random_state=42)
        X_train = train_df.iloc[:, 4:-number_of_predictions]
        y_train = train_df.iloc[:, number_of_predictions:]
        model.fit(X_train, y_train)
        switching_models[budget] = model

        ela_path_algo = Path(ELA_DIR_ALGO) / f"A1_B{budget}_5D_ela_with_state.csv"
        if not ela_path_algo.exists():
            continue
        train_df = pd.read_csv(ela_path_algo)
        train_df = train_df[train_df["iid"].isin(train_iids)]
        X_train = train_df.iloc[:, 4:-6]
        y_train = train_df.iloc[:, -6:]

        trained_model_path = Path(CV_MODELS_DIR) / f"iid{test_iid}/selector_B{budget}_trained.pkl"
        if trained_model_path.exists():
            perf_model = joblib.load(trained_model_path)
        else:
            perf_model = joblib.load(f"{UNTRAINED_PERF_MODELS_DIR}/model_B{budget}.pkl").selector
            perf_model.fit(X_train, y_train)
            os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
            joblib.dump(perf_model, trained_model_path)

        performance_models[budget] = perf_model

    total_precision = 0.0
    for fid in FIDS:
        for rep in REPS:
            precision = selector.simulate_single_run(fid, test_iid, rep, switching_models, performance_models)
            total_precision += precision
    return total_precision

# ========== Objective function for SMAC ==========
def smac_objective(config, seed):
    np.random.seed(seed)
    selector = SwitchingSelectorCV(PRECISION_FILE)

    print(f"Evaluating config: {config}")
    with Pool(processes=5) as pool:  # Adjust number of processes
        results = pool.starmap(partial(train_models_for_iid, config=config, selector=selector), [(iid,) for iid in IIDS])

    total_cv_precision = sum(results)
    print(f"Config {config} → Total CV precision: {total_cv_precision}")
    return total_cv_precision

# ========== Main SMAC tuning routine ==========

def main():
    # 1) make a CS and a transform dict
    cs = ConfigurationSpace(seed=42)
    cs_transform = {}

    # 2) define a parent categorical (even if it has just one choice)
    parent_hp = CategoricalHyperparameter("selector", ["PerformanceModel"])
    cs.add_hyperparameter(parent_hp)

    # 3) let PerformanceModel add its hyperparams under that parent
    cs, cs_transform = PerformanceModel.get_configuration_space(
        cs=cs,
        cs_transform=cs_transform,
        parent_param=parent_hp,
        parent_value="PerformanceModel",
    )

    scenario = Scenario(
        configspace=cs,
        n_trials=200,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory=SMAC_OUTPUT_DIR,
        seed=42
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        smac_objective
    )
    best_config = smac.optimize()

    print("Best configuration found:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    # performance_model_partial = PerformanceModel.get_from_configuration(best_config, cs_transform, random_state=42)
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    for budget in SWITCHING_BUDGETS:
        if budget < 100:
            number_of_predictions = 19 + ( (96 - budget) // 8 ) + 1
        else:
            number_of_predictions = (1000 - budget) // 50 + 1

        ela_path_precs_ahead = Path(ELA_PRECS_AHEAD) / f"A1_B{budget}_5D_ela_with_state.csv"
        if not ela_path_precs_ahead.exists():
            continue
        train_df = pd.read_csv(ela_path_precs_ahead)
        X_train = train_df.iloc[:, 4: -number_of_predictions]
        y_train = train_df.iloc[:, number_of_predictions:]
        model = PerformanceModel.get_from_configuration(best_config, cs_transform, random_state=42)
        model.fit(X_train, y_train)
        model_path = output_dir / f"switching_model_B{budget}_trained.pkl"
        joblib.dump(model, model_path)
        print(f"Saved switching model for budget {budget} to {model_path}")

    print("All final switching models trained and saved successfully.")

if __name__ == "__main__":
    main()