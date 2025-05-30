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

def plot_feature_for_function_over_budgets(folder_path, feature_name, target_fid,
                                           initial_path="ela_initial_sampling_with_rep.csv",
                                           include_initial=True, save_plot=False):
    """
    Plot the distribution of an ELA feature for a single function (fid) across all budgets,
    optionally including the initial sampling distribution as 'initial'.

    Parameters:
        folder_path (str): Directory containing ELA CSV files (e.g., 'A1_B500_5D_ela.csv').
        feature_name (str): Feature to visualize (e.g., 'ela_distr.skewness').
        target_fid (int): The function ID (fid) to visualize.
        initial_path (str): Path to the CSV file containing initial samples.
        include_initial (bool): Whether to include the initial sampling data as budget='initial'.
        save_plot (bool): Whether to save the plot to a file.
    """
    all_data = []

    # Load initial sampling distribution if requested
    if include_initial:
        if not os.path.exists(initial_path):
            raise FileNotFoundError(f"Initial sampling file '{initial_path}' not found.")
        df_init = pd.read_csv(initial_path)
        if feature_name not in df_init.columns:
            raise ValueError(f"Feature '{feature_name}' not found in initial sampling file.")
        df_init_fid = df_init[df_init['fid'] == target_fid].copy()
        df_init_fid['budget'] = 'initial'  # label as a string to distinguish
        all_data.append(df_init_fid[['budget', feature_name]])

    # Load budget-wise data
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

    # Ensure proper order of 'initial' + numeric budgets
    combined['budget'] = combined['budget'].astype(str)
    budget_order = ['initial'] + sorted(
        [b for b in combined['budget'].unique() if b != 'initial'],
        key=lambda x: int(x)
    )

    # Plot
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=combined, x='budget', y=feature_name, order=budget_order)
    plt.title(f"Distribution of '{feature_name}' for fid={target_fid} over Budgets")
    plt.xlabel("Budget")
    plt.ylabel(feature_name)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_plot:
        suffix = "_with_initial_normalized" if include_initial else ""
        dir_path = f"../results/ela_distributions{suffix}/{feature_name}"
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"fid_{target_fid}.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
        plt.close()
    else:
        plt.show()

def plot_ela_feature_trends_over_budgets(folder_path, feature_names, target_fid,
                                         initial_path="ela_initial_sampling_with_rep.csv",
                                         include_initial=True, save_plot=False):
    """
    Plot the mean and variance band of selected ELA features over budgets for a given function.

    Parameters:
        folder_path (str): Directory containing ELA CSV files (e.g., 'A1_B500_5D_ela.csv').
        feature_names (list of str): Features to include in the plot.
        target_fid (int): The function ID (fid) to visualize.
        initial_path (str): Path to the CSV file containing initial samples.
        include_initial (bool): Whether to include the initial sampling data as budget='initial'.
        save_plot (bool): Whether to save the plot to a file.
    """
    all_records = []

    # Initial sampling
    if include_initial:
        if not os.path.exists(initial_path):
            raise FileNotFoundError(f"Initial sampling file '{initial_path}' not found.")
        df_init = pd.read_csv(initial_path)
        df_init = df_init[df_init['fid'] == target_fid]
        for feature in feature_names:
            if feature not in df_init.columns:
                raise ValueError(f"Feature '{feature}' not found in initial sampling file.")
            for val in df_init[feature].values:
                all_records.append({'budget': 'initial', 'feature': feature, 'value': val})

    # Budget-wise files
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".csv"):
            match = re.search(r'B(\d+)', fname)
            if match:
                budget = int(match.group(1))
                file_path = os.path.join(folder_path, fname)
                df = pd.read_csv(file_path)
                df = df[df['fid'] == target_fid]
                for feature in feature_names:
                    if feature in df.columns:
                        for val in df[feature].values:
                            all_records.append({'budget': budget, 'feature': feature, 'value': val})

    if not all_records:
        raise ValueError("No data found for the given fid and features.")

    df_all = pd.DataFrame(all_records)

    # Prepare for plotting
    df_all['budget_str'] = df_all['budget'].astype(str)
    df_all['budget_num'] = df_all['budget'].apply(lambda x: -1 if x == 'initial' else int(x))

    # Aggregate means and std
    summary = (
        df_all.groupby(['feature', 'budget_num'])
        .agg(mean_val=('value', 'mean'), std_val=('value', 'std'))
        .reset_index()
        .sort_values(by='budget_num')
    )

    # Plot
    plt.figure(figsize=(14, 8))  # Wider and taller
    for feature in feature_names:
        sub = summary[summary['feature'] == feature]
        plt.plot(sub['budget_num'], sub['mean_val'], label=feature, marker='o')
        # Optional variance band
        plt.fill_between(sub['budget_num'], sub['mean_val'] - sub['std_val'],
                         sub['mean_val'] + sub['std_val'], alpha=0.2)

    xticks = ['initial' if b == -1 else str(b) for b in sorted(df_all['budget_num'].unique())]
    plt.xticks(sorted(df_all['budget_num'].unique()), labels=xticks, rotation=45)
    plt.title(f"ELA Feature Trends for fid={target_fid}")
    plt.xlabel("Budget")
    plt.ylabel("Feature Value")

    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.grid(True)

    if save_plot:
        feature_suffix = "_".join([f.split('.')[-1] for f in feature_names])
        suffix = "_with_initial_band" if include_initial else ""
        dir_path = f"../results/ela_trends{suffix}/fid_{target_fid}"
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"{feature_suffix}.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
        plt.close()
    else:
        plt.show(block=False)

if __name__ == "__main__":
    # Example usage
    csv_path = "A1_data_ela_disp_normalized_global"  # Update with your CSV file path
    feature_name = "ela_distr.skewness"  # Example feature name
    budget_value = 50  # Example budget value
    for fid in range(1, 25):
        plot_ela_feature_trends_over_budgets(
            folder_path=csv_path,
            #feature_names=feature_names,  # Use the full list of feature names
            # feature_names=["ela_meta.lin_simple.adj_r2", "ela_level.mmce_qda_10", "disp.ratio_mean_10", "disp.diff_median_02"],
            feature_names=[
                
                "ela_level.mmce_qda_10", "ela_meta.lin_simple.adj_r2", "ic.h_max", "nbc.nb_fitness.cor", "disp.ratio_mean_10"],
            target_fid=fid,  # Example fid
            include_initial=False,
            save_plot=False
        )
    # for fid in range(1, 25):
    #     print(f"Processing fid {fid}")
    #     plot_ela_feature_trends_over_budgets(
    #     folder_path="A1_data_ela_disp_normalized",
    #     feature_names=feature_names,
    #     target_fid=fid,
    #     include_initial=False,
    #     save_plot=False
    #     )
    input("Press Enter to continue...")
    # for budget_value in [50*i for i in range(1, 21)]:
    #     plot_feature_for_budget(csv_path, feature_name, budget_value)

    # for feature_name in feature_names:
    #     for fid in range(1, 25):
    #         plot_feature_for_function_over_budgets(csv_path, feature_name, fid, initial_path="ela_initial_sampling_with_normalized_rep.csv", 
    #                                                include_initial=True, save_plot=True)


# "disp.diff_mea_10", "ela_level.mmce_qda_10", "ela_meta.lin_simple.adj_r2", "ic.h_max", "nbc.nb_fitness.cor"],