from pyexpat import model
import sys
import pandas as pd
import os
import joblib
from asf.predictors import RandomForestClassifierWrapper
from smac import HyperparameterOptimizationFacade, Scenario
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 20)]

results = []

for budget in budgets:
    print(f"Evaluation for budget {budget}")
    # Load model and data
    model = joblib.load(f"trained_models_classifier/model_B{budget}.pkl")
    df = pd.read_csv(f"../data/A1_data_ela_cma_std_newInstances_normalized_no_ps_ratio/A1_B{budget}_5D_ela_with_state.csv")
    features = df.iloc[:, 4:]
    targets = df.iloc[:, 0]
    
    # Predictions
    preds = model.predict(features)
    
    # Metrics
    acc  = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="macro")   # or "weighted"
    rec  = recall_score(targets, preds, average="macro")
    f1   = f1_score(targets, preds, average="macro")
    
    results.append({
        "budget": budget,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

# Turn into DataFrame for easy export
results_df = pd.DataFrame(results)
results_df.to_csv("classifier_scores.csv", index=False)