import joblib
import pandas as pd
from pathlib import Path
from ioh import get_problem, ProblemClass
import os

class SwitchingSelector:
    def __init__(self, selector_model_dir="switching_prediction_models", performance_model_dir="algo_performance_models"):
        """
        Initializes the selector and algorithm performance predictors.

        Args:
            selector_model_dir (str or Path): Directory with models like 'selector_B500_trained.pkl'.
            performance_model_dir (str or Path): Directory with performance predictors like 'performance_B1000_model.pkl'.
        """
        self.switching_prediction_models = {}
        self.performance_models = {}

        selector_model_dir = Path(selector_model_dir)
        performance_model_dir = Path(performance_model_dir)

        # Load switching predictor models
        for model_path in selector_model_dir.glob("selector_B*_trained.pkl"):
            budget = int(model_path.stem.split("_")[1][1:])  # e.g., selector_B500 → 500
            self.switching_prediction_models[budget] = joblib.load(model_path)

        # Load performance predictors
        for model_path in performance_model_dir.glob("selector_B*_trained.pkl"):
            budget = int(model_path.stem.split("_")[1][1:])  # e.g., performance_B1000_model → 1000
            self.performance_models[budget] = joblib.load(model_path)

    def simulate_single_run(self, fid, iid, rep, ela_dir="../data/ela_with_state_test_data", precision_file="../data/A2_precisions_test.csv", budgets=range(50, 1001, 50)):
        """
        Simulates a switching run for one (fid, iid, rep) case using the selector and performance predictors.

        Args:
            fid (int): Function ID.
            iid (int): Instance ID.
            rep (int): Repetition number.
            ela_dir (str): Folder with ELA files named like A1_B{budget}_5D_ela_with_state.csv.
            precision_file (str): CSV file with ground-truth precision values.
            budgets (iterable): Budget levels to step through.

        Returns:
            dict: {fid, iid, rep, switch_budget, selected_algorithm, predicted_precision}
        """
        precision_df = pd.read_csv(precision_file)

        for budget in budgets:
            # print(f"Checking budget {budget} for (fid={fid}, iid={iid}, rep={rep})...")
            ela_path = Path(ela_dir) / f"A1_B{budget}_5D_ela_with_state.csv"
            if not ela_path.exists():
                continue

            df = pd.read_csv(ela_path)
            row = df[(df["fid"] == fid) & (df["iid"] == iid) & (df["rep"] == rep)]

            if row.empty:
                continue

            features = row.iloc[:, 4:]
            features.index = [(fid, iid, rep)]

            # Predict switching behavior
            switch_model = self.switching_prediction_models.get(budget)
            if switch_model is None:
                continue

            prediction = switch_model.predict(features)
            predicted_switch_budget = int(list(prediction.values())[0][0][0])
            # print(f"Predicted switch budget: {predicted_switch_budget}")
            if predicted_switch_budget == budget:
                # Predict algorithm choice
                performance_model = self.performance_models.get(budget)
                if performance_model is None:
                    print(f"No performance model for budget {budget}, skipping...")
                    continue
                
                algo_prediction = performance_model.predict(features)
                predicted_algorithm = list(algo_prediction.values())[0][0][0]
                # print(predicted_algorithm)

                # Look up actual precision
                match_row = precision_df[
                    (precision_df["fid"] == fid) &
                    (precision_df["iid"] == iid) &
                    (precision_df["rep"] == rep) &
                    (precision_df["budget"] == budget) &
                    (precision_df["algorithm"] == predicted_algorithm)
                ]

                precision = match_row["precision"].values[0] if not match_row.empty else None

                # Get VBS precision
                vbs_precision = precision_df[
                    (precision_df["fid"] == fid) &
                    (precision_df["iid"] == iid) &
                    (precision_df["rep"] == rep)
                ]["precision"].min()

                return {
                    "fid": fid,
                    "iid": iid,
                    "rep": rep,
                    "switch_budget": budget,
                    "selected_algorithm": predicted_algorithm,
                    "predicted_precision": precision,
                    "vbs_precision": vbs_precision
                }
        # No budget triggered a switch → fall back to using CMA-ES at budget 1000
        fallback_budget = 1000
        fallback_algorithm = "Same"

        match_row = precision_df[
        (precision_df["fid"] == fid) &
        (precision_df["iid"] == iid) &
        (precision_df["rep"] == rep) &
        (precision_df["budget"] == fallback_budget)
    ]

        if not match_row.empty:
            # fallback to the algorithm with best precision at budget 1000 (just like static CMA-ES fallback)
            precision = match_row[match_row["algorithm"] == fallback_algorithm]["precision"]
            precision = precision.values[0] if not precision.empty else None
        else:
            precision = None

        vbs_precision = precision_df[
            (precision_df["fid"] == fid) &
            (precision_df["iid"] == iid) &
            (precision_df["rep"] == rep)
        ]["precision"].min()

        return {
            "fid": fid,
            "iid": iid,
            "rep": rep,
            "switch_budget": None,
            "selected_algorithm": fallback_algorithm,
            "predicted_precision": precision,
            "vbs_precision": vbs_precision
        }

    def evaluate_selector_to_csv(
    self,
    fids,
    iids,
    reps,
    save_path="selector_results.csv",
    ela_dir="../data/ela_with_state_test_data",
    precision_file="../data/A2_precisions_test.csv"
    ):
        precision_df = pd.read_csv(precision_file)
        budgets = list(range(50, 1001, 50))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for fid in fids:
            for iid in iids:
                for rep in reps:
                    print(f"Processing (fid={fid}, iid={iid}, rep={rep})...")

                    # Get VBS precision
                    vbs_precision = precision_df[
                        (precision_df["fid"] == fid) &
                        (precision_df["iid"] == iid) &
                        (precision_df["rep"] == rep)
                    ]["precision"].min()

                    row = {
                        "fid": fid,
                        "iid": iid,
                        "rep": rep,
                        "vbs_precision": vbs_precision,
                    }

                    # Selector result
                    result = self.simulate_single_run(fid, iid, rep, ela_dir, precision_file)
                    row["selector_precision"] = result["predicted_precision"]
                    row["selector_switch_budget"] = result["switch_budget"] or 1000

                    # Static switchers
                    for b in budgets:
                        col_name = f"static_B{b}"
                        if b < 1000:
                            ela_path = Path(ela_dir) / f"A1_B{b}_5D_ela_with_state.csv"
                            if not ela_path.exists():
                                row[col_name] = None
                                continue

                            df = pd.read_csv(ela_path)
                            instance_row = df[
                                (df["fid"] == fid) &
                                (df["iid"] == iid) &
                                (df["rep"] == rep)
                            ]
                            if instance_row.empty:
                                row[col_name] = None
                                continue

                            features = instance_row.iloc[:, 4:]
                            features.index = [(fid, iid, rep)]

                            model = self.performance_models.get(b)
                            if model is None:
                                row[col_name] = None
                                continue

                            algo_pred = model.predict(features)
                            algo = list(algo_pred.values())[0][0][0]

                            match = precision_df[
                                (precision_df["fid"] == fid) &
                                (precision_df["iid"] == iid) &
                                (precision_df["rep"] == rep) &
                                (precision_df["budget"] == b) &
                                (precision_df["algorithm"] == algo)
                            ]
                            row[col_name] = match["precision"].values[0] if not match.empty else None
                        else:
                            # Budget 1000 → use CMA-ES directly
                            match = precision_df[
                                (precision_df["fid"] == fid) &
                                (precision_df["iid"] == iid) &
                                (precision_df["rep"] == rep) &
                                (precision_df["budget"] == 1000) &
                                (precision_df["algorithm"] == "Same")
                            ]
                            row[col_name] = match["precision"].values[0] if not match.empty else None

                    # Append row to CSV
                    row_df = pd.DataFrame([row])
                    row_df.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))

        print(f"Incremental results saved to: {save_path}")




# def main(fid = 1, iid = 6, rep = 0):
#     # Initialize the selector with default model directories
#     selector = SwitchingSelector(
#         selector_model_dir="switching_prediction_models",
#         performance_model_dir="algo_performance_models"
#     )


#     # Run the simulation
#     for rep in range(20):
#         # Simulate a single run
#         result = selector.simulate_single_run(fid=fid, iid=iid, rep=rep)

#         # Print the result
#         print("=== Simulation Result ===")
#         print(f"FID: {result['fid']}, IID: {result['iid']}, REP: {result['rep']}")
#         print(f"Switch budget: {result['switch_budget']}")
#         print(f"Selected algorithm: {result['selected_algorithm']}")
#         print(f"Predicted precision: {result['predicted_precision']}")
#         print(f"VBS precision: {result['vbs_precision']}")

if __name__ == "__main__":
    selector = SwitchingSelector(
        selector_model_dir="switching_prediction_models",
        performance_model_dir="algo_performance_models"
    )
    selector.evaluate_selector_to_csv(
        fids=list(range(1, 25)),
        iids=[1, 2, 3, 4, 5],
        reps=list(range(20, 30)),
        save_path="../results/result_csvs/selector_results_newReps.csv",
        ela_dir="../data/ela_with_state_newReps",
        precision_file="../data/A2_newReps_precisions.csv"
    )