import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os
import joblib


# budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 20)]  # Budgets from 50 to 1000 in steps of 50
budgets = [50*i for i in range(1, 20)]  # Budgets from 50 to 1000 in steps of 50    

for budget in budgets:
    print(f"Processing budget {budget}...")
    train_df = pd.read_csv(f"../data/ela_for_training/ela_with_state_late_switch_greater_budgets/A1_B{budget}_5D_ela_with_state.csv")
    # test_df = pd.read_csv(f"../data/ela_with_cma_state_newReps_late_with_switch_budget/A1_B{budget}_5D_ela_with_state.csv")

    non_feature_cols = ['fid', 'iid', 'rep', 'high_level_category', 'switch']
    feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['switch']
    # X_test = test_df[feature_cols]

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Save the model    
    model_path = f"trained_models/switching_late_greater/selector_B{budget}_trained.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)    
    print(f"Model saved to {model_path}")


    # # Predict
    # y_pred = clf.predict(X_test)

    # #Evaluate metrics
    # acc = accuracy_score(test_df['switch'], y_pred)
    # prec = precision_score(test_df['switch'], y_pred, average='macro', zero_division=0)
    # rec = recall_score(test_df['switch'], y_pred, average='macro', zero_division=0)
    # f1 = f1_score(test_df['switch'], y_pred, average='macro', zero_division=0)   

    # print(f"Budget {budget}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1 Score={f1:.4f}")
