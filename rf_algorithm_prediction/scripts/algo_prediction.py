import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut
import os
import math

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
    all_fid_scores = {}

    for budget in budget_list:
        acc_per_fid = run_top1_multilabel_logo(budget=budget, ela_folder=ela_folder, per_fid=True)
        if acc_per_fid is None:
            continue
        for fid, acc in acc_per_fid.items():
            if fid not in all_fid_scores:
                all_fid_scores[fid] = {}
            all_fid_scores[fid][budget] = acc

    if not all_fid_scores:
        print("No per-fid results to plot.")
        return

    # Prepare color map
    fids = sorted(all_fid_scores.keys())
    cmap = cm.get_cmap('nipy_spectral', len(fids))
    colors = [cmap(i) for i in range(len(fids))]

    plt.figure(figsize=(12, 7))
    for idx, fid in enumerate(fids):
        budgets = sorted(all_fid_scores[fid].keys())
        accs = [all_fid_scores[fid][b] for b in budgets]
        plt.plot(budgets, accs, marker='o', linestyle='-', label=f"fid {fid}", color=colors[idx])

    plt.title("Top-1 Accuracy per Function ID (fid) over Budgets")
    plt.xlabel("Budget")
    plt.ylabel("Top-1 Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.tight_layout()
    os.makedirs("results/accuracy_over_budgets", exist_ok=True)
    plt.savefig("results/accuracy_over_budgets/top1_accuracy_per_fid_over_budgets.png")
    plt.show()

def plot_fid_accuracy_over_budgets(budget_list, ela_folder="data/ELA_over_budgets", group_size=12):
    """
    Plots a line per fid showing its top-1 accuracy across different budgets.
    Splits fids into multiple plots (default 12 per plot) to maintain color distinctiveness.
    """
    label_cols = ['BFGS', 'DE', 'MLSL', 'Non-elitist', 'PSO', 'Same']
    all_fid_scores = {}

    for budget in budget_list:
        print(f"Processing budget: {budget}")
        acc_per_fid = run_top1_multilabel_logo(budget=budget, ela_folder=ela_folder, per_fid=True)
        if acc_per_fid is None:
            continue
        for fid, acc in acc_per_fid.items():
            if fid not in all_fid_scores:
                all_fid_scores[fid] = {}
            all_fid_scores[fid][budget] = acc

    if not all_fid_scores:
        print("No per-fid results to plot.")
        return

    fids = sorted(all_fid_scores.keys())
    num_groups = math.ceil(len(fids) / group_size)
    os.makedirs("results/accuracy_over_budgets", exist_ok=True)

    for group_idx in range(num_groups):
        group_fids = fids[group_idx * group_size:(group_idx + 1) * group_size]
        cmap = cm.get_cmap('nipy_spectral', len(group_fids))
        colors = [cmap(i) for i in range(len(group_fids))]

        plt.figure(figsize=(12, 7))
        for idx, fid in enumerate(group_fids):
            budgets = sorted(all_fid_scores[fid].keys())
            accs = [all_fid_scores[fid][b] for b in budgets]
            plt.plot(budgets, accs, marker='o', linestyle='-', label=f"fid {fid}", color=colors[idx])

        plt.title(f"Top-1 Accuracy per fid over Budgets (Group {group_idx + 1})")
        plt.xlabel("Budget")
        plt.ylabel("Top-1 Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        filename = f"results/accuracy_over_budgets/top1_accuracy_fid_group_{group_idx + 1}.pdf"
        plt.savefig(filename)
        plt.show()
        print(f"Plot saved at: {filename}")

# plot_per_fid_accuracy_per_budget([50*i for i in range(1, 20)])
plot_fid_accuracy_over_budgets([50*i for i in range(1, 20)])
