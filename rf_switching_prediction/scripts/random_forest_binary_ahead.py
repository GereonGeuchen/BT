import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
from itertools import product

def evaluate_ahead_switching_classifier(csv_path, budget, threshold=0.5):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Compute label distribution for is_minimal_switch_ahead
    label_distribution = df['is_minimal_switch_ahead'].value_counts(normalize=True)
    pct_true = label_distribution.get(True, 0) * 100  # percentage of True labels

    # Drop non-ELA features: meta-data, runtime features, and switching result columns
    non_ela_columns = [
        'is_minimal_switch', 'is_minimal_switch_ahead',
        'fid', 'iid', 'rep', 'high_level_category', 'run_precision'
    ] + [col for col in df.columns if col.endswith('.costs_runtime')]
    X = df.drop(columns=[col for col in non_ela_columns if col in df.columns])

    # Optional: drop problematic ELA features for specific budgets
    if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
        print("Dropping ela_meta.quad_w_interact.adj_r2 for budget 50")
        X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

    y = df['is_minimal_switch_ahead']
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


def optimize_ahead_thresholds_by_budget(budget_list, base_path_template):

    best_thresholds = {}

    for budget in budget_list:
        print(f"Optimizing threshold for budget {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)

        # Prepare ELA-only features
        X = df.drop(columns=[
            'is_minimal_switch', 'is_minimal_switch_ahead', 'fid', 'iid', 'rep', 'high_level_category', 'run_precision'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])

        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        # Target: is_minimal_switch_ahead
        y = df['is_minimal_switch_ahead']
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


def plot_ahead_switch_prob_heatmap(base_path_template, budget_list, save_plots=False):
    data_by_fid_iid = {}

    for budget in budget_list:
        print(f"Processing budget: {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)

        # Use ELA-only features
        X = df.drop(columns=[
            'is_minimal_switch', 'is_minimal_switch_ahead',
            'fid', 'iid', 'rep', 'high_level_category', 'run_precision'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])

        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['is_minimal_switch_ahead']
        groups = df['iid']
        df['prob'] = np.nan  # to store predicted probabilities

        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(X, y, groups):
            clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            df.loc[test_idx, 'prob'] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        # Store predicted probabilities for each (fid, iid, rep) at this budget
        for _, row in df.iterrows():
            fid, iid, rep, prob = int(row['fid']), int(row['iid']), int(row['rep']), row['prob']
            key = (fid, iid)
            data_by_fid_iid.setdefault(key, {}).setdefault(budget, {})[rep] = prob

    # Step 2: Plot heatmaps
    for (fid, iid), budget_data in data_by_fid_iid.items():
        heatmap_data = np.full((len(budget_list), 20), np.nan)

        for i, budget in enumerate(budget_list):
            if budget in budget_data:
                for rep, prob in budget_data[budget].items():
                    heatmap_data[i, rep] = prob

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            heatmap_data,
            annot=False,
            cmap='viridis_r',
            vmin=0.0,
            vmax=1.0,
            linewidths=0.3,
            linecolor='grey',
            xticklabels=range(20),
            yticklabels=budget_list,
            cbar_kws={"label": "Predicted Switching Probability"}
        )
        ax.invert_yaxis()
        ax.set_yticklabels(budget_list, rotation=0)
        plt.title(f"Switching Probability Heatmap (fid={fid}, iid={iid})")
        plt.xlabel("Repetition")
        plt.ylabel("Budget")
        plt.tight_layout()

        if save_plots:
            output_dir = "../results/rf_heatmaps_ahead/"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"heatmap_fid_{fid}_iid_{iid}.png")
            plt.savefig(out_path)
            print(f"Saved heatmap for fid={fid}, iid={iid}")
            plt.close()
        else:
            plt.show(block=False)

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
    plt.plot(budgets, label_props, label='minimal switch % (normalized)', color='black', linestyle='--', marker='o')

    plt.xlabel('Budget')
    plt.ylabel('Value (0–1 scale)')
    plt.title('Metrics and Normalized Label Distribution Across Budgets')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ahead_binary_switch_heatmaps(base_path_template, budget_list, thresholds, save_plots=False):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    data_by_fid_iid = {}

    for budget in budget_list:
        print(f"Processing budget: {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)
        df['budget'] = budget  # Keep for debugging or extension

        # Keep only ELA features
        X = df.drop(columns=[
            'is_minimal_switch', 'is_minimal_switch_ahead', 'fid', 'iid', 'rep', 'high_level_category', 'run_precision'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])

        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['is_minimal_switch_ahead']
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
            data_by_fid_iid.setdefault(key, {'pred': {}, 'true': {}, 'prob': {}})
            data_by_fid_iid[key]['pred'].setdefault(budget, {})[rep] = row['pred_binary']
            data_by_fid_iid[key]['true'].setdefault(budget, {})[rep] = int(row['y_true'])
            data_by_fid_iid[key]['prob'].setdefault(budget, {})[rep] = row['prob']

    for (fid, iid), data in data_by_fid_iid.items():
        pred_matrix = np.full((len(budget_list), 20), np.nan)
        true_matrix = np.full((len(budget_list), 20), np.nan)
        prob_matrix = np.full((len(budget_list), 20), np.nan)

        for i, budget in enumerate(budget_list):
            for rep in range(20):
                pred_matrix[i, rep] = data['pred'].get(budget, {}).get(rep, np.nan)
                true_matrix[i, rep] = data['true'].get(budget, {}).get(rep, np.nan)
                prob_matrix[i, rep] = data['prob'].get(budget, {}).get(rep, np.nan)

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
            output_dir = "../results/binary_switch_heatmaps_ahead_rf/"
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"triple_heatmap_fid_{fid}_iid_{iid}.png")
            plt.savefig(out_path)
            print(f"Saved heatmap for fid={fid}, iid={iid}")
            plt.close()
        else:
            plt.show(block=False)

def simulate_switching_accuracy(base_path_template, budget_list, thresholds):

    results = []

    for budget in budget_list:
        print(f"Loading budget {budget}")
        df = pd.read_csv(base_path_template.format(budget=budget))
        df['budget'] = budget

        # Keep only ELA features
        X = df.drop(columns=[
            'is_minimal_switch', 'is_minimal_switch_ahead', 'fid', 'iid', 'rep',
            'high_level_category', 'run_precision'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])

        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['is_minimal_switch_ahead']
        groups = df['iid']
        df['pred_prob'] = np.nan

        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(X, y, groups):
            clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            df.loc[test_idx, 'pred_prob'] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        df['threshold'] = thresholds.get(budget, 0.5)
        df['pred_binary'] = df['pred_prob'] >= df['threshold']

        results.append(df[['fid', 'iid', 'rep', 'budget', 'pred_binary', 'is_minimal_switch']])

    # Combine across all budgets
    full_df = pd.concat(results)

    correct_switches = 0
    total_runs = 0

    # Group by (fid, iid, rep) and simulate the switching process
    for (fid, iid, rep), group in full_df.groupby(['fid', 'iid', 'rep']):
        group_sorted = group.sort_values('budget')
        for _, row in group_sorted.iterrows():
            if not row['pred_binary']:  # First prediction of False → switch
                if row['is_minimal_switch']:  # It was indeed the optimal point
                    correct_switches += 1
                break
        total_runs += 1

    percentage = 100 * correct_switches / total_runs
    print(f"\n✅ Switched at correct moment in {percentage:.2f}% of runs ({correct_switches}/{total_runs})")
    return percentage



def prepare_predictions_dataframe(base_path_template, budget_list):
    all_dfs = []
    for budget in budget_list:
        df = pd.read_csv(base_path_template.format(budget=budget))
        df['budget'] = budget

        X = df.drop(columns=[
            'is_minimal_switch', 'is_minimal_switch_ahead', 'fid', 'iid', 'rep',
            'high_level_category', 'run_precision'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])

        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['is_minimal_switch_ahead']
        groups = df['iid']
        df['pred_prob'] = np.nan

        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(X, y, groups):
            clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            df.loc[test_idx, 'pred_prob'] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        df['is_minimal_switch'] = df['is_minimal_switch']
        all_dfs.append(df[['fid', 'iid', 'rep', 'budget', 'pred_prob', 'is_minimal_switch']])

    return pd.concat(all_dfs)


def simulate_switch_accuracy_with_thresholds(df, budget_list, thresholds):
    df = df.copy()
    df['threshold'] = df['budget'].map(thresholds)
    df['pred_binary'] = df['pred_prob'] >= df['threshold']

    correct = 0
    total = 0

    for (fid, iid, rep), group in df.groupby(['fid', 'iid', 'rep']):
        group_sorted = group.sort_values('budget')
        for _, row in group_sorted.iterrows():
            if not row['pred_binary']:
                if row['is_minimal_switch']:
                    correct += 1
                break
        total += 1

    return correct, total


def greedy_threshold_search(base_path_template, budget_list, threshold_grid, initial_thresholds):
    df = prepare_predictions_dataframe(base_path_template, budget_list)

    current_thresholds = initial_thresholds.copy()
    correct, total = simulate_switch_accuracy_with_thresholds(df, budget_list, current_thresholds)

    print(f"Initial correct switches: {correct}/{total} ({100 * correct / total:.2f}%)")

    for b in reversed(budget_list):
        best_t = current_thresholds[b]
        best_score = correct

        for t in threshold_grid:
            trial = current_thresholds.copy()
            trial[b] = t
            c, _ = simulate_switch_accuracy_with_thresholds(df, budget_list, trial)
            if c > best_score:
                best_score = c
                best_t = t

        current_thresholds[b] = best_t
        correct = best_score  # update best known
        print(f"✓ Budget {b}: Best threshold = {best_t:.2f} → correct switches = {correct}")

    pct = 100 * correct / total
    print(f"\n✅ Greedy switching rate: {pct:.2f}% ({correct}/{total})")
    return current_thresholds

def fixed_budget_switch_accuracy(base_path_template, budget_list, switch_at_budget):

    if switch_at_budget not in budget_list:
        raise ValueError(f"Budget {switch_at_budget} not in provided budget list")

    print(f"Evaluating fixed switching at budget {switch_at_budget}...")

    # Load just the file for the fixed switch budget
    df = pd.read_csv(base_path_template.format(budget=switch_at_budget))

    correct_switches = df['is_minimal_switch'].sum()
    total_runs = len(df)

    percentage = 100 * correct_switches / total_runs
    print(f"✅ Fixed switch at budget {switch_at_budget} is correct in {percentage:.2f}% of runs ({correct_switches}/{total_runs})")
    return percentage

def count_switches_per_budget(df, budget_list, thresholds):
    df = df.copy()
    df['threshold'] = df['budget'].map(thresholds)
    df['pred_binary'] = df['pred_prob'] >= df['threshold']

    switch_counts = {b: 0 for b in budget_list}
    no_switches = 0
    total_runs = 0

    for (fid, iid, rep), group in df.groupby(['fid', 'iid', 'rep']):
        group_sorted = group.sort_values('budget')

        switched = False
        for _, row in group_sorted.iterrows():
            if not row['pred_binary']:  # switch now
                switch_counts[row['budget']] += 1
                switched = True
                break
        if not switched:
            no_switches += 1
        total_runs += 1

    return switch_counts, no_switches, total_runs

if __name__ == "__main__":
    # budget_list = [50 * i for i in range(1, 20)]
    # base_path_template = "data/ELA_over_budgets_with_precs_ahead/A1_B{budget}_5D_ela.csv"
    # best_thresholds = optimize_ahead_thresholds_by_budget(budget_list, base_path_template)
    # print(best_thresholds)
    best_thresholds = {
       50: 0.35, 100: 0.49, 150: 0.41, 200: 0.46, 250: 0.41,
       300: 0.44, 350: 0.34, 400: 0.38, 450: 0.43, 500: 0.38,
       550: 0.39, 600: 0.31, 650: 0.45, 700: 0.39, 750: 0.35,
       800: 0.46, 850: 0.40, 900: 0.40, 950: 0.31
    }
    # results_by_budget = {}
    # for budget in [50*i for i in range(1, 20)]:
    #     results_by_budget[budget] = evaluate_ahead_switching_classifier(
    #         csv_path=f"data/ELA_over_budgets_with_precs_ahead/A1_B{budget}_5D_ela.csv",
    #         budget=budget,
    #         threshold= best_thresholds[budget]
    #     )
    # plot_metrics_and_distribution(results_by_budget)
    # plot_ahead_binary_switch_heatmaps(
    #     base_path_template="../data/ELA_over_budgets_with_precs_ahead/A1_B{budget}_5D_ela.csv",
    #     budget_list=[50 * i for i in range(1, 20)],
    #     thresholds=best_thresholds,
    #     save_plots=True
    # )

    budget_list = [50 * i for i in range(1, 20)]
    base_path_template = "../data/ELA_over_budgets_with_precs_ahead/A1_B{budget}_5D_ela.csv"
    fixed_budget_switch_accuracy(base_path_template=base_path_template, budget_list=budget_list, switch_at_budget=50)
    # initial_thresholds = {
    #     50: 0.35, 100: 0.49, 150: 0.41, 200: 0.46, 250: 0.41,
    #     300: 0.44, 350: 0.34, 400: 0.38, 450: 0.43, 500: 0.38,
    #     550: 0.39, 600: 0.31, 650: 0.45, 700: 0.39, 750: 0.35,
    #     800: 0.46, 850: 0.40, 900: 0.40, 950: 0.31
    # }

    # initial_thresholds = {
    #     50: 0.90, 100: 0.90, 150: 0.90, 200: 0.90, 250: 0.90,
    #     300: 0.44, 350: 0.90, 400: 0.90, 450: 0.82, 500: 0.82,
    #     550: 0.90, 600: 0.31, 650: 0.90, 700: 0.90, 750: 0.87,
    #     800: 0.90, 850: 0.90, 900: 0.90, 950: 0.31
    # }
    # previous = best_thresholds
    # while True:
    #     best_thresholds = greedy_threshold_search(
    #         base_path_template=base_path_template,
    #         budget_list=budget_list,
    #         threshold_grid=np.linspace(0.1, 1.0, 31),  # e.g. 0.25–0.55 in steps of 0.01
    #         initial_thresholds=best_thresholds
    #     )
    #     if previous == best_thresholds:
    #         print("No further improvement in thresholds, stopping early.")
    #         break 
    #     previous = best_thresholds
    # print(best_thresholds)
    # simulate_switching_accuracy(
    #     base_path_template=base_path_template,
    #     budget_list=budget_list,
    #     thresholds=best_thresholds
    # )

    # best_thresholds = { 50: 1.00, 100: 0.97, 150: 1.00, 200: 1.00, 250: 1.00,
    #                     300: 0.37, 350: 0.10, 400: 0.31, 450: 1.00, 500: 0.70,
    #                     550: 0.94, 600: 0.70, 650: 1.00, 700: 1.00, 750: 1.00,
    #                     800: 1.00, 850: 1.00, 900: 1.00, 950: 1.00 }
    # df = prepare_predictions_dataframe(base_path_template, budget_list)

    # switch_counts, no_switches, total_runs = count_switches_per_budget(
    #     df=df,
    #     budget_list=budget_list,
    #     thresholds=best_thresholds
    # )
    # print(f"Total runs: {total_runs}, No switches: {no_switches}")
    # print("Switch counts per budget:", switch_counts)
    # # Use actual keys from switch_counts
    # budgets = sorted(switch_counts.keys())
    # counts = [switch_counts[b] for b in budgets]

    # if any(counts):  # At least one non-zero value
    #     plt.figure(figsize=(10, 5))
    #     plt.bar(budgets, counts, color='steelblue')
    #     plt.title("Number of Runs Switching at Each Budget")
    #     plt.xlabel("Budget")
    #     plt.ylabel("Number of Switches")
    #     plt.xticks(budgets, rotation=45)
    #     plt.grid(True, axis='y')
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("⚠️ No switches recorded — check thresholds and model predictions.")
 