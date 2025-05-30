import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut
import os

def run_top1_multilabel_logo(budget, ela_folder="data/ELA_over_budgets", per_fid=False):

    ela_file = f"{ela_folder}/A1_B{budget}_5D_ela.csv_with_algo_perfomance.csv"
    try:
        ela_df = pd.read_csv(ela_file)
    except FileNotFoundError:
        print(f"ELA file not found: {ela_file}")
        return None

    # Drop .costs_runtime and budget-specific column
    cost_cols = [col for col in ela_df.columns if col.endswith('.costs_runtime')]
    ela_df.drop(columns=cost_cols, inplace=True)
    if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in ela_df.columns:
        ela_df.drop(columns=['ela_meta.quad_w_interact.adj_r2'], inplace=True)

    # Column roles
    id_cols = ['fid', 'iid', 'rep', 'high_level_category']
    label_cols = ['BFGS', 'DE', 'MLSL', 'Non-elitist', 'PSO', 'Same']
    feature_cols = [col for col in ela_df.columns if col not in id_cols + label_cols]

    X = ela_df[feature_cols].astype(float).values
    Y = ela_df[label_cols].fillna(0).astype(int).values
    groups = ela_df['iid'].astype(int).values
    fids = ela_df['fid'].astype(int).values

    logo = LeaveOneGroupOut()
    fid_results = {}
    total_correct = 0
    total_samples = 0

    for train_idx, test_idx in logo.split(X, Y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        fids_test = fids[test_idx]

        model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
        model.fit(X_train, Y_train)

        probas = model.predict_proba(X_test)
        proba_matrix = np.array([np.ravel(p[:, 1]) for p in probas]).T
        top1_pred = np.argmax(proba_matrix, axis=1)

        for fid, pred, y_true in zip(fids_test, top1_pred, Y_test):
            total_samples += 1
            if fid not in fid_results:
                fid_results[fid] = [0, 0]
            fid_results[fid][1] += 1
            if y_true[pred] == 1:
                total_correct += 1
                fid_results[fid][0] += 1

    if per_fid:
        return {fid: correct / total for fid, (correct, total) in fid_results.items()}
    else:
        return total_correct / total_samples

def plot_logo_accuracy_over_budgets(budget_list):
    results = {}
    for b in budget_list:
        accs = run_top1_multilabel_logo(budget=b)
        if accs is not None:
            results[b] = accs

    if not results:
        print("No data available for plotting.")
        return

    plot_budgets = list(results.keys())
    means = [np.mean(results[b]) for b in plot_budgets]
    stds = [np.std(results[b]) for b in plot_budgets]
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_budgets, means, label='Mean Accuracy', marker='o')
    plt.fill_between(plot_budgets, lower, upper, color='blue', alpha=0.2, label='±1 Std Dev')
    plt.title("Top-1 Accuracy over Budgets (LOGO CV)")
    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_per_fid_accuracy_per_budget(budget_list, ela_folder="data/ELA_over_budgets"):
    """
    For each budget, plot:
    - Top-1 accuracy per fid (dot + line)
    - One line per A2 algorithm: percentage of runs where it was optimal (dot + line)
    - Vertical line per fid
    """

    label_cols = ['BFGS', 'DE', 'MLSL', 'Non-elitist', 'PSO', 'Same']

    for budget in budget_list:
        # Run model and get accuracy per fid
        acc_per_fid = run_top1_multilabel_logo(budget=budget, ela_folder=ela_folder, per_fid=True)
        if acc_per_fid is None:
            continue

        # Load ELA data
        ela_file = f"{ela_folder}/A1_B{budget}_5D_ela.csv_with_algo_perfomance.csv"
        try:
            ela_df = pd.read_csv(ela_file)
        except FileNotFoundError:
            print(f"ELA file not found: {ela_file}")
            continue

        ela_df[label_cols] = ela_df[label_cols].fillna(0)
        fids = sorted(acc_per_fid.keys())
        accuracies = [acc_per_fid[fid] for fid in fids]

        # Compute per-algorithm optimal percentage per fid
        algo_optimal_perc = {alg: [] for alg in label_cols}
        for fid in fids:
            fid_df = ela_df[ela_df['fid'] == fid]
            num_rows = len(fid_df)
            if num_rows == 0:
                for alg in label_cols:
                    algo_optimal_perc[alg].append(0.0)
            else:
                for alg in label_cols:
                    num_ones = fid_df[alg].sum()
                    algo_optimal_perc[alg].append(num_ones / num_rows)

        # Plotting
        plt.figure(figsize=(14, 6))

        # Accuracy line + dots
        plt.plot(fids, accuracies, marker='o', linestyle='-', color='royalblue', label='Top-1 Accuracy')

        # One line per algorithm showing its % of optimality per fid
        for alg in label_cols:
            plt.plot(fids, algo_optimal_perc[alg], marker='x', linestyle='--', label=f"{alg} optimal %")

        # Vertical lines for each fid
        for fid in fids:
            plt.axvline(x=fid, color='lightgray', linestyle='--', linewidth=0.5)

        plt.title(f"Top-1 Accuracy and Optimal A2 Algorithm Frequency per fid (Budget {budget})")
        plt.xlabel("Function ID (fid)")
        plt.ylabel("Accuracy / Optimal Algorithm %")
        plt.ylim(0, 1.05)
        plt.grid(True, axis='y')
        plt.legend()
        plt.tight_layout()
        os.makedirs("results/accuracy_per_fid", exist_ok=True)
        plt.savefig(f"results/accuracy_per_fid/budget_{budget}.png")
        print(f"Plot saved for budget {budget} at /results/accuracy_per_fid/budget_{budget}.png")


plot_per_fid_accuracy_per_budget([50*i for i in range(1, 20)])