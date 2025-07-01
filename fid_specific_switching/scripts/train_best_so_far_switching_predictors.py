import pandas as pd
import numpy as np
import joblib
import os
from asf.predictors.random_forest import RandomForestClassifierWrapper

budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 20)]
modes = ["late", "all"]  

best_config = {
    "rf_classifierbootstrap": True,
    "rf_classifiermax_features": 0.2553356326896,
    "rf_classifiermin_samples_leaf": 7,
    "rf_classifiermin_samples_split": 20,
    "rf_classifiern_estimators": 60
}

for budget in budgets:

    print(f"🔄 Processing budget {budget}")

    # Resolve CSV path
    csv_path = f"../data/ela_for_training/ela_with_state_all_switch_greater_budgets/A1_B{budget}_5D_ela_with_state.csv"

    if not os.path.exists(csv_path):
        print(f"⚠️ File not found: {csv_path}, skipping...")
        continue

    # Load data
    data = pd.read_csv(csv_path)
    X = data.iloc[:, 4:-1]
    y = data.iloc[:, -1]

    # Build wrapper from config
    rf_wrapper_class = RandomForestClassifierWrapper.get_from_configuration(best_config)
    rf_wrapper = rf_wrapper_class()

    # Fit on full data
    rf_wrapper.fit(X, y)

    # Save model
    os.makedirs("../data/models/trained_models/switching_models_trained_best_50trials", exist_ok=True)
    save_path = f"../data/models/trained_models/switching_models_trained_best_50trials/selector_B{budget}_trained.pkl"
    joblib.dump(rf_wrapper, save_path)

    print(f"✅ Saved trained model to {save_path}")
