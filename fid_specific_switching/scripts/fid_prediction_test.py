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

# Budgets to process
budgets = list(range(50, 1001, 50))
exclude_for_50 = "ela_meta.quad_w_interact.adj_r2"

# Collect metrics
metrics = []

for budget in budgets:
    print(f"\n=== Budget {budget} ===")

    # File paths
    train_path = f"../data/ela_with_state/A1_B{budget}_5D_ela_with_state.csv"
    test_path = f"../data/ela_with_state_newReps/A1_B{budget}_5D_ela_with_state.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Skipping budget {budget}: files not found.")
        continue

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Feature selection
    non_feature_cols = ['fid', 'iid', 'rep', 'high_level_category']
    feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

    if budget == 50 and exclude_for_50 in feature_cols:
        feature_cols.remove(exclude_for_50)

    X_train = train_df[feature_cols]
    y_train = train_df['fid']
    X_test = test_df[feature_cols]
    y_test = test_df['fid']

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    metrics.append((budget, acc, prec, rec, f1))

    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")

# Summary table
print("\n=== Metrics Summary ===")
print(f"{'Budget':>6} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | {'F1 Score':>8}")
print("-" * 50)
for b, acc, prec, rec, f1 in metrics:
    print(f"{b:>6} | {acc:>8.4f} | {prec:>9.4f} | {rec:>7.4f} | {f1:>8.4f}")
