import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from matplotlib.colors import PowerNorm

def plot_run_precision_heatmap(base_path_template, budget_list, save_plots=False, log_transform=False):
    data_by_fid_iid = {}

    # Step 1: Collect predictions across all budgets
    for budget in budget_list:
        print(f"Processing budget: {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)

        # Drop unused columns
        X = df.drop(columns=[
            'is_minimal_switch', 'fid', 'iid', 'rep', 'run_precision'
        ] + [col for col in df.columns if col.endswith('.costs_runtime')])
        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in X.columns:
            X = X.drop(columns=['ela_meta.quad_w_interact.adj_r2'])

        y = df['run_precision']
        if log_transform:
            y = np.log10(y.clip(lower=1e-12))  # Avoid log(0) or log of negative

        groups = df['iid']
        df['y_true'] = y
        df['predicted_precision'] = np.nan

        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(X, y, groups):
            reg = RandomForestRegressor(random_state=42, n_jobs=-1)
            reg.fit(X.iloc[train_idx], y.iloc[train_idx])
            df.loc[test_idx, 'predicted_precision'] = reg.predict(X.iloc[test_idx])

        for _, row in df.iterrows():
            fid, iid, rep, pred = int(row['fid']), int(row['iid']), int(row['rep']), row['predicted_precision']
            key = (fid, iid)
            if key not in data_by_fid_iid:
                data_by_fid_iid[key] = {}
            if budget not in data_by_fid_iid[key]:
                data_by_fid_iid[key][budget] = {}
            data_by_fid_iid[key][budget][rep] = pred

    # Step 2: Plot heatmaps
    for (fid, iid), budget_data in data_by_fid_iid.items():
        pred_matrix = np.full((len(budget_list), 20), np.nan)
        true_matrix = np.full((len(budget_list), 20), np.nan)

        for i, budget in enumerate(budget_list):
            if budget in budget_data:
                for rep, pred in budget_data[budget].items():
                    pred_matrix[i, rep] = pred

        for budget in budget_list:
            csv_path = base_path_template.format(budget=budget)
            df = pd.read_csv(csv_path)
            df = df[(df['fid'] == fid) & (df['iid'] == iid)]

            for _, row in df.iterrows():
                i = budget_list.index(budget)
                rep = int(row['rep'])
                val = row['run_precision']
                if log_transform:
                    val = np.log10(max(val, 1e-12))  # Again clip to avoid log(0)
                true_matrix[i, rep] = val

        vmax = np.nanmax([pred_matrix, true_matrix])
        vmin = np.nanmin([pred_matrix, true_matrix])

        label = "Log10 Run Precision" if log_transform else "Run Precision"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        sns.heatmap(
            pred_matrix,
            ax=axes[0],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            linewidths=0.3,
            linecolor='grey',
            xticklabels=range(20),
            yticklabels=budget_list,
            cbar_kws={"label": label}
        )
        axes[0].set_title(f"Predicted {label}")
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Repetition")
        axes[0].set_ylabel("Budget")
        axes[0].set_yticklabels(budget_list, rotation=0)

        sns.heatmap(
            true_matrix,
            ax=axes[1],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            linewidths=0.3,
            linecolor='grey',
            xticklabels=range(20),
            yticklabels=budget_list,
            cbar_kws={"label": label}
        )
        axes[1].set_title(f"Actual {label}")
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Repetition")
        axes[1].set_ylabel("Budget")
        axes[1].set_yticklabels(budget_list, rotation=0)

        plt.suptitle(f"{label} Heatmaps (fid={fid}, iid={iid})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_plots:
            output_dir = "../results/rf_run_precision_heatmaps_maxB700/"
            os.makedirs(output_dir, exist_ok=True)
            log_suffix = "_log" if log_transform else ""
            out_path = os.path.join(output_dir, f"run_precision{log_suffix}_fid_{fid}_iid_{iid}.png")
            plt.savefig(out_path)
            print(f"Saved heatmap for fid={fid}, iid={iid}")
            plt.close()
        else:
            plt.show(block=False)


def plot_r2_over_budgets(base_path_template, budget_list):
    budget_r2_scores = {}

    for budget in budget_list:
        print(f"Evaluating R² for budget {budget}")
        csv_path = base_path_template.format(budget=budget)
        df = pd.read_csv(csv_path)

        # Prepare features
        drop_cols = ['fid', 'iid', 'rep', 'is_minimal_switch', 'run_precision']
        drop_cols += [col for col in df.columns if col.endswith('.costs_runtime')]
        if budget == 50 and 'ela_meta.quad_w_interact.adj_r2' in df.columns:
            drop_cols.append('ela_meta.quad_w_interact.adj_r2')

        X = df.drop(columns=drop_cols)
        y = df['run_precision']
        groups = df['iid']

        logo = LeaveOneGroupOut()
        r2_scores = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            r2_scores.append(score)

        budget_r2_scores[budget] = r2_scores

    # Plotting
    budgets = sorted(budget_r2_scores.keys())
    means = np.array([np.mean(budget_r2_scores[b]) for b in budgets])
    stds = np.array([np.std(budget_r2_scores[b]) for b in budgets])

    plt.figure(figsize=(10, 6))
    plt.plot(budgets, means, label="Mean R²", marker='o')
    plt.fill_between(budgets, means - stds, means + stds, alpha=0.3, label="±1 std dev")
    plt.xlabel("Budget")
    plt.ylabel("R² Score")
    plt.title("Random Forest R² Scores for Predicting Run Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    budget_list = [50*i for i in range(1, 15)]
    base_path_template = "../data/ELA_over_budgets_with_precs_maxB700/A1_B{budget}_5D_ela.csv"
    plot_run_precision_heatmap(
        base_path_template=base_path_template,
        budget_list=budget_list,
        save_plots=True,
        log_transform=False
    )
    # plot_r2_over_budgets(base_path_template=base_path_template, budget_list=budget_list)
    # plot_r2_per_instance_per_budget(
    # base_path_template="../data/ELA_over_budgets_with_precs/A1_B{budget}_5D_ela.csv",
    # budget_list=[50*i for i in range(1, 21)]
    # )