import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

feature_names = [
    "ela_distr.skewness",
    "ela_distr.kurtosis",
    "ela_distr.number_of_peaks",
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_simple.intercept",
    "ela_meta.lin_simple.coef.min",
    "ela_meta.lin_simple.coef.max",
    "ela_meta.lin_simple.coef.max_by_min",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_simple.cond",
    "ela_meta.quad_w_interact.adj_r2",
    "ela_level.mmce_lda_10",
    "ela_level.mmce_qda_10",
    "ela_level.lda_qda_10",
    "ela_level.mmce_lda_25",
    "ela_level.mmce_qda_25",
    "ela_level.lda_qda_25",
    "ela_level.mmce_lda_50",
    "ela_level.mmce_qda_50",
    "ela_level.lda_qda_50",
    "disp.ratio_mean_02",
    "disp.ratio_mean_05",
    "disp.ratio_mean_10",
    "disp.ratio_mean_25",
    "disp.ratio_median_02",
    "disp.ratio_median_05",
    "disp.ratio_median_10",
    "disp.ratio_median_25",
    "disp.diff_mean_02",
    "disp.diff_mean_05",
    "disp.diff_mean_10",
    "disp.diff_mean_25",
    "disp.diff_median_02",
    "disp.diff_median_05",
    "disp.diff_median_10",
    "disp.diff_median_25",
    "ic.h_max",
    "ic.eps_s",
    "ic.eps_max",
    "ic.eps_ratio",
    "ic.m0",
    "nbc.nn_nb.sd_ratio",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.cor",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor"
]



def plot_feature_for_budget(folder_path, feature_name, target_budget):
    """
    Plot the distribution of an ELA feature per function across all instances and reps
    for a specific budget (parsed from filename).
    
    Parameters:
        folder_path (str): Directory containing ELA CSV files.
        feature_name (str): Feature to visualize (e.g., 'ela_distr.skewness').
        target_budget (int): The exact budget level to match (e.g., 50, 500).
    """
    file = None
    for fname in os.listdir(folder_path):
        if fname.endswith('.csv'):
            match = re.search(r'B(\d+)', fname)
            if match and int(match.group(1)) == target_budget:
                file = os.path.join(folder_path, fname)
                break
    if file is None:
        raise FileNotFoundError(f"No CSV file found for exact budget '{target_budget}'.")

    df = pd.read_csv(file)

    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the selected file.")

    # Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='fid', y=feature_name)
    plt.title(f"Distribution of '{feature_name}' across functions (Budget: {target_budget})")
    plt.xlabel("BBOB Function ID (fid)")
    plt.ylabel(feature_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.ylim(0, 20)
    plt.show(block=False)

def plot_feature_for_function_over_budgets(folder_path, feature_name, target_fid, save_plot=False):
    """
    Plot the distribution of an ELA feature for a single function (fid) across all budgets.

    Parameters:
        folder_path (str): Directory containing ELA CSV files (e.g., 'A1_B500_5D_ela.csv').
        feature_name (str): Feature to visualize (e.g., 'ela_distr.skewness').
        target_fid (int): The function ID (fid) to visualize.
    """
    all_data = []

    # Loop over all files and collect data for the target fid
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".csv"):
            match = re.search(r'B(\d+)', fname)
            if match:
                budget = int(match.group(1))
                file_path = os.path.join(folder_path, fname)
                df = pd.read_csv(file_path)

                if feature_name in df.columns:
                    df_fid = df[df['fid'] == target_fid].copy()
                    df_fid['budget'] = budget
                    all_data.append(df_fid[['budget', feature_name]])

    if not all_data:
        raise FileNotFoundError("No data found for the selected function or feature.")

    combined = pd.concat(all_data, ignore_index=True)
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=combined, x='budget', y=feature_name)
    plt.title(f"Distribution of '{feature_name}' for fid={target_fid} over Budgets")
    plt.xlabel("Budget")
    plt.ylabel(feature_name)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot
    if save_plot:
        dir_path = f"../results/ela_distributions/{feature_name}"
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(os.path.join(dir_path, f"fid_{target_fid}.png"))
        print(f"Plot saved to {os.path.join(dir_path, f'fid_{target_fid}.png')}")
    else:
        plt.show(block=False)
    plt.close()

if __name__ == "__main__":
    # Example usage
    csv_path = "A1_data_ela_disp"  # Update with your CSV file path
    feature_name = "ela_distr.skewness"  # Example feature name

    budget_value = 50  # Example budget value

    # for budget_value in [50*i for i in range(1, 21)]:
    #     plot_feature_for_budget(csv_path, feature_name, budget_value)

    for feature_name in feature_names:
        for fid in range(1, 25):
            plot_feature_for_function_over_budgets(csv_path, feature_name, fid, save_plot=True)



