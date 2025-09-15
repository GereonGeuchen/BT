import numpy as np
import pandas as pd
import shap
import joblib
from asf.predictors import RandomForestClassifierWrapper

def feature_importance_binary(df, df_background, model):
    # If your CSV contains a target/id column, drop it here (uncomment & edit):
    # drop_cols = ["y", "target", "label", "id"]  # adjust as needed
    # df = pd.read_csv(csv_path).drop(columns=[c for c in drop_cols if c in df.columns])

    model = model.model_class


    X_test = df.iloc[:, 4:].copy()
    
    X_bg_full = df_background.iloc[:, 4:-7].copy()

    X_bg = shap.kmeans(X_bg_full, k = 128)

    # Groups by position AFTER dropping the first 4 columns:
    # original 5..51  -> indices 0..46
    # original 52..end -> indices 47..last
    g1_idx = list(range(0, 47))                   # 0..46
    g2_idx = list(range(47, X_test.shape[1] - 6))     # 47..end
    g3_idx = list(range(X_test.shape[1]-6, X_test.shape[1]))  # last 6 features (optimal precisions)
    groups = {"Group_1": g1_idx, "Group_2": g2_idx, "Group_3": g3_idx}

    # --- Explain with TreeSHAP (binary). Use log-odds for exact additivity. ---
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional", 
        model_output="raw",
        data=X_bg.data
    )

    ex = explainer(X_test, check_additivity=True)                 # SHAP per-feature
    phi = ex.values                    # (n, f) or (n, f, n_classes)

    # Pick SHAP for the positive class if per-class attributions are returned
    if phi.ndim == 3:
        classes = list(getattr(model, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else (len(classes) - 1)
        phi = phi[..., pos_idx]        # (n, f)

    # --- Compute sums per group ---
    rows = []
    for gname, idxs in groups.items():
        per_sample_sum = phi[:, idxs].sum(axis=1)            # signed sum per sample

        activity_sum = float(np.abs(phi[:, idxs]).sum())     # Σ_i Σ_{j∈g} |φ_ij|
        net_sum      = float(np.abs(per_sample_sum).sum())   # Σ_i |Σ_{j∈g} φ_ij|
        signed_sum   = float(per_sample_sum.sum())           # Σ_i Σ_{j∈g} φ_ij

        rows.append({
            "group": gname,
            "activity_sum": activity_sum,
            "net_sum": net_sum,
            "signed_sum": signed_sum,
            "cancel_pct": 100.0 * (1.0 - net_sum / (activity_sum + 1e-12)),
        })

    out = pd.DataFrame(rows).sort_values("activity_sum", ascending=False)
    out["activity_share"] = out["activity_sum"] / out["activity_sum"].sum()
    out["net_share"]      = out["net_sum"]      / out["net_sum"].sum()
    out.to_csv("feature_importance_by_group_bg.csv", index=False)
    return out

def feature_importance_regression(df, df_background, model):
    # --- paths (mirror your current style: read inside the function) ---

    model = model.model_class
    # ---- make background tidy & small
    X_test = df.iloc[:, 4:].copy()
    X_bg_full = df_background.iloc[:, 4:].copy()

    # hard checks: columns & dtypes
    assert list(X_test.columns) == list(X_bg_full.columns), "Col mismatch/order changed."
    X_test = X_test.apply(pd.to_numeric, errors="coerce").astype("float64")
    X_bg_full = X_bg_full.apply(pd.to_numeric, errors="coerce").astype("float64")
    assert np.isfinite(X_test.to_numpy()).all(), "Non-finite in X_test"
    assert np.isfinite(X_bg_full.to_numpy()).all(), "Non-finite in background"


    nF = X_test.shape[1]
    assert nF >= 53, f"Need at least 53 feature columns after dropping 4; got {nF}."

    # --- groups by POSITION after dropping the first 4 ---
    # original 5..51 -> indices 0..46; original 52..end split so last 6 are Group_3
    g1_idx = list(range(0, 47))                 # 0..46
    g3_idx = list(range(nF - 6, nF))            # last 6 columns
    g2_idx = list(range(47, nF - 6))            # 47..(nF-7)
    groups = {"Group_1": g1_idx, "Group_2": g2_idx, "Group_3": g3_idx}

    # compact background
    n_bg = min(256, len(X_bg_full))
    X_bg = X_bg_full.sample(n=n_bg, random_state=0) if len(X_bg_full) > n_bg else X_bg_full


    # ---- SHAP: interventional via masker + new API
    masker = shap.maskers.Independent(X_bg)
    explainer = shap.Explainer(
        model,                      # your tree model (class/instance per your wrapper)
        masker,
        algorithm="tree",           # TreeExplainer under the hood
        model_output="raw"          # regression: identity/link-free
    )

    # try strict additivity first
    ex = explainer(X_test, check_additivity=True)
    phi = ex.values # (n_samples, n_features) for single-output regression
    # if you have a multi-output regressor (rare), pick output 0
    if phi.ndim == 3:
        phi = phi[..., 0]

    base = getattr(ex, "base_values", None)
    if base is None:
        base = ex.expected_value
    base = np.array(base)
    if base.ndim == 0:
        base_vec = np.full(X_test.shape[0], float(base))
    else:
        base_vec = base.reshape(-1)  # usually (n,)
        if base_vec.shape[0] != X_test.shape[0]:
            base_vec = np.full(X_test.shape[0], float(base_vec.mean()))
    recon = base_vec + phi.sum(axis=1)
    pred  = model.predict(X_test)
    err   = np.max(np.abs(recon - pred))
    print("max |recon - pred| =", err)

    # --- sum-based metrics per group ---
    rows = []
    for gname, idxs in groups.items():
        per_sample_sum = phi[:, idxs].sum(axis=1)             # Σ_{j∈g} φ_ij per sample
        activity_sum   = float(np.abs(phi[:, idxs]).sum())    # Σ_i Σ_{j∈g} |φ_ij|
        net_sum        = float(np.abs(per_sample_sum).sum())  # Σ_i |Σ_{j∈g} φ_ij|
        signed_sum     = float(per_sample_sum.sum())          # Σ_i Σ_{j∈g} φ_ij

        rows.append({
            "group": gname,
            "activity_sum": activity_sum,
            "net_sum": net_sum,
            "signed_sum": signed_sum,
            "cancel_pct": 100.0 * (1.0 - net_sum / (activity_sum + 1e-12)),
        })

    out = pd.DataFrame(rows).sort_values("activity_sum", ascending=False)
    out["activity_share"] = out["activity_sum"] / out["activity_sum"].sum()
    out["net_share"]      = out["net_sum"]      / out["net_sum"].sum()

    out.to_csv("feature_importance_by_group_regression_background_test.csv", index=False)
    return out




if __name__ == "__main__":
    csv_path = "../data/ela_data/A1_data_ela_cma_std_newInstances_normalized_no_ps_ratio/A1_B100_5D_ela_with_state.csv"   # <- put your test CSV path here
    csv_background_path = "../data/ela_data/A1_data_switch_normalized_log10_200_no_ps_ratio/A1_B100_5D_ela_with_state.csv"  # <- put your background CSV path here

    # If your CSV contains a target/id column, drop it here (uncomment & edit):
    # drop_cols = ["y", "target", "label", "id"]  # adjust as needed
    # df = pd.read_csv(csv_path).drop(columns=[c for c in drop_cols if c in df.columns])

    df = pd.read_csv(csv_path)
    df_background = pd.read_csv(csv_background_path)
    df_background = df_background.iloc[:, :-6]
    model = joblib.load("../data/models/switching_models_normalized_log10_200_200_no_ps_ratio/switching_model_B100_trained.pkl")  # <- put your model path here
    fi = feature_importance_binary(df, df_background, model)
    print(fi)