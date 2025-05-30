import pandas as pd
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import PowerNorm


def evaluate_switching_classifier(csv_path, budget, threshold=0.5):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Compute label distribution
    label_distribution = df['is_minimal_switch'].value_counts(normalize=True)
    pct_true = label_distribution.get(True, 0) * 100  # percentage of True labels

    # Prepare features (drop non-features and .costs_runtime features)
    X = df.drop(columns=[
        'is_minimal_switch', 'fid', 'iid', 'rep',
    ] + [col for col in df.columns if col.endswith('.costs_runtime')])

    # Optional: drop budget-specific problematic feature
    if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
        X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

    y = df['is_minimal_switch']
    groups = df['iid']

    logo = LeaveOneGroupOut()

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train, y_train)

        # Apply custom threshold to predict
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(bool)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1_score'].append(f1_score(y_test, y_pred, zero_division=0))

    return {
        "mean": {k: np.mean(v) for k, v in metrics.items()},
        "std": {k: np.std(v) for k, v in metrics.items()},
        "label_pct": pct_true
    }


def plot_metrics_and_distribution(results_by_budget):
    budgets = sorted(results_by_budget.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Extract data
    means = {m: [results_by_budget[b]['mean'][m] for b in budgets] for m in metrics}
    stds = {m: [results_by_budget[b]['std'][m] for b in budgets] for m in metrics}
    label_props = [results_by_budget[b]['label_pct'] / 100 for b in budgets]  # Normalize to 0–1

    # Plot
    plt.figure(figsize=(12, 8))

    colors = {
        'accuracy': 'tab:blue',
        'precision': 'tab:orange',
        'recall': 'tab:green',
        'f1_score': 'tab:red'
    }

    for metric in metrics:
        plt.plot(budgets, means[metric], label=f'{metric} (mean)', color=colors[metric])
        plt.fill_between(
            budgets,
            [m - s for m, s in zip(means[metric], stds[metric])],
            [m + s for m, s in zip(means[metric], stds[metric])],
            color=colors[metric],
            alpha=0.2
        )

    # Add normalized label distribution
    plt.plot(budgets, label_props, label='minimal switch %', color='black', linestyle='--', marker='o')

    plt.xlabel('Budget')
    plt.ylabel('Value (0–1 scale)')
    plt.title('Metrics and Normalized Label Distribution Across Budgets')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_best_threshold(y_true, y_probs, metric=f1_score):
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t, best_score = 0.5, -np.inf
    for t in thresholds:
        preds = (y_probs >= t).astype(bool)
        score = metric(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score

def optimize_thresholds_by_budget(budget_list, base_path_template):
    best_thresholds = {}

    for budget in budget_list:
        print(f"Optimizing threshold for budget {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)

        # Prepare features
        X = df.drop(columns=[
            'is_minimal_switch', 'fid', 'iid', 'rep', 'run_precision'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])
        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['is_minimal_switch']
        groups = df['iid']

        logo = LeaveOneGroupOut()
        all_probs = []
        all_true = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]

            all_probs.extend(y_prob)
            all_true.extend(y_test.values)

        best_t, best_score = find_best_threshold(np.array(all_true), np.array(all_probs))
        best_thresholds[budget] = best_t
        print(f"  → Best threshold = {best_t:.2f}, F1 = {best_score:.4f}")

    return best_thresholds



def plot_switch_prob_heatmap(base_path_template, budget_list, save_plots=False):
    data_by_fid_iid = {}

    # Step 1: Collect predictions across all budgets
    for budget in budget_list:
        print(f"Processing budget: {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)

        # Drop unused columns
        X = df.drop(columns=[
            'is_minimal_switch', 'fid', 'iid', 'rep'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])
        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['is_minimal_switch']
        groups = df['iid']
        df['y_true'] = y

        logo = LeaveOneGroupOut()
        df['prob'] = np.nan

        for train_idx, test_idx in logo.split(X, y, groups):
            clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            df.loc[test_idx, 'prob'] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        for _, row in df.iterrows():
            fid, iid, rep, prob = int(row['fid']), int(row['iid']), int(row['rep']), row['prob']
            key = (fid, iid)
            if key not in data_by_fid_iid:
                data_by_fid_iid[key] = {}
            if budget not in data_by_fid_iid[key]:
                data_by_fid_iid[key][budget] = {}
            data_by_fid_iid[key][budget][rep] = prob

    # Step 2: Plot heatmaps
    for (fid, iid), budget_data in data_by_fid_iid.items():
        heatmap_data = np.full((len(budget_list), 20), np.nan)

        for i, budget in enumerate(budget_list):
            if budget in budget_data:
                for rep, prob in budget_data[budget].items():
                    heatmap_data[i, rep] = prob

        # norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=1.0)

        # plt.figure(figsize=(10, 6))
        # ax = sns.heatmap(
        #     heatmap_data,
        #     annot=False,
        #     cmap='viridis_r',
        #     norm=norm,
        #     linewidths=0.3,
        #     linecolor='grey',
        #     xticklabels=range(20),
        #     yticklabels=budget_list,
        #     cbar_kws={"label": "Predicted Switching Probability"}
        # )
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            heatmap_data,
            annot=False,
            cmap='viridis_r',  # now uses normal perceptually uniform gradient
            vmin=0.0,
            vmax=1.0,
            linewidths=0.3,
            linecolor='grey',
            xticklabels=range(20),
            yticklabels=budget_list,
            cbar_kws={"label": "Predicted Switching Probability"}
        )
        ax.invert_yaxis()  # Larger budgets at bottom
        ax.set_yticklabels(budget_list, rotation=0)  # Horizontal y-axis labels
        plt.title(f"Switching Probability Heatmap (fid={fid}, iid={iid})")
        plt.xlabel("Repetition")
        plt.ylabel("Budget")
        plt.tight_layout()

        if save_plots:
            output_dir = "../results/rf_heatmaps/"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"heatmap_fid_{fid}_iid_{iid}.png")
            plt.savefig(out_path)
            print(f"Saved heatmap for fid={fid}, iid={iid}")
            plt.close()  # Prevents figure from showing during batch saving
        else:
            plt.show(block=False)


def plot_binary_switch_heatmaps(base_path_template, budget_list, thresholds, save_plots=False):
    data_by_fid_iid = {}

    for budget in budget_list:
        print(f"Processing budget: {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)
        df['budget'] = budget  # add budget info for later filtering

        X = df.drop(columns=[
            'is_minimal_switch', 'fid', 'iid', 'rep'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])
        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['is_minimal_switch']
        groups = df['iid']
        df['y_true'] = y
        df['prob'] = np.nan

        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(X, y, groups):
            clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            df.loc[test_idx, 'prob'] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        threshold = thresholds.get(budget, 0.5)
        df['pred_binary'] = (df['prob'] >= threshold).astype(int)

        for _, row in df.iterrows():
            fid, iid, rep = int(row['fid']), int(row['iid']), int(row['rep'])
            key = (fid, iid)
            if key not in data_by_fid_iid:
                data_by_fid_iid[key] = {'pred': {}, 'true': {}, 'prob': {}}
            if budget not in data_by_fid_iid[key]['pred']:
                data_by_fid_iid[key]['pred'][budget] = {}
                data_by_fid_iid[key]['true'][budget] = {}
                data_by_fid_iid[key]['prob'][budget] = {}

            data_by_fid_iid[key]['pred'][budget][rep] = row['pred_binary']
            data_by_fid_iid[key]['true'][budget][rep] = int(row['y_true'])
            data_by_fid_iid[key]['prob'][budget][rep] = row['prob']

    for (fid, iid), data in data_by_fid_iid.items():
        pred_matrix = np.full((len(budget_list), 20), np.nan)
        true_matrix = np.full((len(budget_list), 20), np.nan)
        prob_matrix = np.full((len(budget_list), 20), np.nan)

        for i, budget in enumerate(budget_list):
            for rep in range(20):
                if rep in data['pred'].get(budget, {}):
                    pred_matrix[i, rep] = data['pred'][budget][rep]
                if rep in data['true'].get(budget, {}):
                    true_matrix[i, rep] = data['true'][budget][rep]
                if rep in data['prob'].get(budget, {}):
                    prob_matrix[i, rep] = data['prob'][budget][rep]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        sns.heatmap(pred_matrix, ax=axes[0], cmap='Greys_r', linewidths=0.3, linecolor="grey", cbar=False,
                    xticklabels=range(20), yticklabels=budget_list, vmin=0, vmax=1)
        axes[0].set_title('Predicted Switch (Binary)')
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Repetition')
        axes[0].set_ylabel('Budget')

        sns.heatmap(true_matrix, ax=axes[1], cmap='Greys_r', linewidths=0.3, linecolor="grey", cbar=False,
                    xticklabels=range(20), yticklabels=budget_list, vmin=0, vmax=1)
        axes[1].set_title('Actual Switch (Binary)')
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Repetition')
        axes[1].set_ylabel('Budget')

        sns.heatmap(prob_matrix, ax=axes[2], cmap='viridis_r', vmin=0.0, vmax=1.0, linewidths=0.3, linecolor="grey",
                    xticklabels=range(20), yticklabels=budget_list,
                    cbar_kws={"label": "Predicted Switching Probability"})
        axes[2].set_title('Predicted Switch Probability')
        axes[2].invert_yaxis()
        axes[2].set_xlabel('Repetition')
        axes[2].set_ylabel('Budget')

        plt.suptitle(f'Switching Heatmaps (fid={fid}, iid={iid})', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_plots:
            output_dir = "../results/binary_switch_heatmaps_rf/"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"triple_heatmap_fid_{fid}_iid_{iid}.png")
            plt.savefig(out_path)
            print(f"Saved heatmap for fid={fid}, iid={iid}")
            plt.close()
        else:
            plt.show(block=False)


if __name__ == "__main__":
    # Example usage
    # results_by_budget = {}
    # best_thresholds = optimize_thresholds_by_budget(
    #     budget_list=range(50, 1050, 50),
    #     base_path_template="../data/ELA_over_budgets_with_optimality/A1_B{budget}_5D_ela.csv"
    # )
    # print(best_thresholds)
    # for budget in range(50, 1050, 50):
    #     print(f"Evaluating budget: {budget}")
    #     csv_path = f"../data/ELA_over_budgets_with_optimality/A1_B{budget}_5D_ela.csv"
    #     results = evaluate_switching_classifier(csv_path, budget, threshold = best_thresholds[budget])
    #     results_by_budget[budget] = results
    # plot_metrics_and_distribution(results_by_budget)

    # {50: 0.3, 100: 0.4, 150: 0.37, 200: 0.43, 250: 0.34, 300: 0.33, 350: 0.41000000000000003, 
    # 400: 0.39, 450: 0.34, 500: 0.27, 550: 0.31, 600: 0.4, 650: 0.41000000000000003, 700: 0.35000000000000003, 
    # 750: 0.35000000000000003, 800: 0.49, 850: 0.45, 900: 0.33, 950: 0.39, 1000: 0.33}

    budget_list = list(range(50, 1050, 50))
    base_path_template = "../data/ELA_over_budgets_with_optimality/A1_B{budget}_5D_ela.csv"
    # plot_switch_prob_heatmap(
    #     base_path_template=base_path_template,
    #     budget_list=budget_list,
    #     save_plots=True
    # )
    # plot_binary_switch_heatmaps(
    #     base_path_template=base_path_template,
    #     budget_list=budget_list,
    #     thresholds={
    #         50: 0.3, 100: 0.4, 150: 0.37, 200: 0.43, 250: 0.34,
    #         300: 0.33, 350: 0.41000000000000003, 400: 0.39, 450: 0.34,
    #         500: 0.27, 550: 0.31, 600: 0.4, 650: 0.41000000000000003,
    #         700: 0.35000000000000003, 750: 0.35000000000000003, 
    #         800: 0.49, 850: 0.45, 900: 0.33, 
    #         950: 0.39, 
    #         1000: 0.33
    #     },
    #     save_plots=True
    # )
    optimize_thresholds_by_budget(
        budget_list=[50*i for i in range(1, 15)],
        base_path_template="../data/ela_over_budgets_with_precs_maxB700/A1_B{budget}_5D_ela.csv"
    )
    input("Press Enter to continue...")
