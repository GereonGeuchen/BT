import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import permutation_test

eps = np.finfo(float).eps

sbs_sum = 6755.964497405699 # BFGS, 900
def compute_vbs_ratios(csv_path, fid = None):

    df = pd.read_csv(csv_path)
    vbs_sum = df["vbs_precisions"].sum()
    consider_cols = [col for col in df.columns if col.startswith("static_B") or col == "selector_precision"]
    res = {}
    for col in consider_cols:
        col_sum = df[col].sum()
        print(f"Processing column: {col}, sum: {col_sum}")
        res[col] = (sbs_sum - col_sum) / (sbs_sum - vbs_sum) 

    # Optionally: Append values for Same, Non-elitist, and BFGS
    # res["Same, 0"] = (sbs_sum - same_sum) / (sbs_sum - vbs_sum)
    # res["Non-elitist, 0"] = (sbs_sum - non_elitist_sum) / (sbs_sum - vbs_sum)
    # res["BFGS, 0"] = (sbs_sum - bfgs_sum) / (sbs_sum - vbs_sum)
    # res["MLSL, 0"] = (sbs_sum - mlsl_sum) / (sbs_sum - vbs_sum)
    # res["DE, 0"] = (sbs_sum - de_sum) / (sbs_sum - vbs_sum)
    # res["PSO, 0"] = (sbs_sum - pso_sum) / (sbs_sum - vbs_sum)
    return res

def find_sbs(path):
    df = pd.read_csv(path)

    score_table = (
        df.groupby(["budget", "algorithm"])["precision"]
        .sum()
        .reset_index()
        .sort_values("precision")
        .reset_index(drop=True)
    )

    return score_table

def save_tables(data, title, filename):
    df = pd.DataFrame(list(data.items()), columns=["Method", "Ratio"])
    df = df.sort_values("Ratio", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(df) + 1))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def compute_total_precisions_for_fid(csv_path, fid):
    df = pd.read_csv(csv_path)
    df = df[df["fid"] == fid]

    consider_cols = [col for col in df.columns if col.startswith("static_B") or col == "selector_precision"]
    result = {col: df[col].sum() for col in consider_cols}
    return result

def save_barplot(data, title, filename):
    """
    Saves a vertical bar plot of the given data.
    Includes only specified columns and a fixed SBS bar with value 0.
    Sorted by value descending: highest left, lowest right.
    """

    # Define desired columns
    desired_cols = [
        "selector",
        "static_B64",
        "static_B80",
        #"static_B96",
        #"static_B56",
        #"static_B48",
        "static_B150",
        "static_B18",
        "static_B8",
        "static_B800",
        #"Same"
        "Non-elitist, 16",
        "Non-elitist, 0"
        #"BFGS"
    ]

    # Convert data dict to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["Method", "Value"])
    df["Method"] = df["Method"].replace({"selector_precision": "selector"})

    # Filter to desired columns only
    df = df[df["Method"].isin(desired_cols)]

    # Add SBS row with value 0.000
    sbs_row = pd.DataFrame({"Method": ["Non-elitist, 16"], "Value": [0.000]})
    df = pd.concat([df, sbs_row], ignore_index=True)

    # Sort by Value descending: highest left, lowest right
    df = df.sort_values("Value", ascending=False).reset_index(drop=True)

    # Plotting
    plt.figure(figsize=(1.5 * len(df) + 2, 6))
    bars = plt.bar(df["Method"], df["Value"], color="skyblue", edgecolor="black")

    # Annotate bars with their values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.3f}",
                 ha='center', va='bottom', fontsize=15)

    plt.ylabel("Precision Ratio", fontsize=15)
    plt.xticks(fontsize=15)
    plt.title(title, fontsize=15, pad=10)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"✅ Vertical bar plot saved to {filename}")


def display_vbs_tables(csv_path, fid=None, plot_type="table"):
    """
    Displays VBS results as either table or bar plot.

    plot_type: "table" or "bar"
    """
    if fid is None:
        ratios = compute_vbs_ratios(csv_path)
        output_path = "../results/newInstances_normalized_log10_10D/precision_ratios.pdf"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if plot_type == "table":
            save_tables(ratios, "Fraction of the gap closed", output_path)
            print("✅ VBS ratios table saved to:", output_path)
        elif plot_type == "bar":
            save_barplot(ratios, "Fraction of the gap closed", output_path)
            print("✅ VBS ratios bar plot saved to:", output_path)
        else:
            raise ValueError("Invalid plot_type. Choose 'table' or 'bar'.")
    else:
        totals = compute_total_precisions_for_fid(csv_path, fid)
        output_dir = "../data/switching_optimality_files/vbs_precision_totals_late"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/vbs_precision_totals_fid_{fid}.png"
        if plot_type == "table":
            save_tables(totals, f"Sum of Precisions (fid={fid})", output_path)
            print(f"✅ Precision totals table saved to: {output_path}")
        elif plot_type == "bar":
            save_barplot(totals, f"Sum of Precisions (fid={fid})", output_path)
            print(f"✅ Precision totals bar plot saved to: {output_path}")
        else:
            raise ValueError("Invalid plot_type. Choose 'table' or 'bar'.")

def plot_selector_budget_counts(csv_path, output_png="selector_budget_counts.png"):
    df = pd.read_csv(csv_path)

    # Count occurrences of each budget
    counts = df["selector_switch_budget"].value_counts().sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color='steelblue')

    ax.set_title("Number of runs in which the selector switched at each budget", fontsize=14)
    ax.set_xlabel("Budget", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Annotate bars with count values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def permutation_test_selector_vs_static(df):

    
    selector = df['selector_precision'].values
    budgets = [20*i for i in range(1, 10)] + [100*i for i in range(2, 21)]
    # budgets = [80]
    for budget in budgets:
        static = df[f'static_B{budget}'].values

        # Compute observed mean difference
        observed_diff = np.mean(selector - static)
        print(f"Observed mean difference: {observed_diff:.6f}")

        # Use scipy's permutation_test
        res = permutation_test(
            (selector, static),
            statistic=lambda x, y: np.mean(x - y),
            permutation_type='samples',
            vectorized=False,
            n_resamples=10000,
            alternative='less',
            random_state=42
        )

        print(f"P-value (selector < static) for budget {budget}: {res.pvalue:.6f}")

    # # Extract permutation distribution (available in res.distribution)
    # perm_distribution = res.null_distribution

    # # Plot
    # plt.figure(figsize=(8,5))
    # plt.hist(perm_distribution, bins=50, color='skyblue', edgecolor='black')

    # # Observed value line with label
    # plt.axvline(observed_diff, color='red', linestyle='--', label=f'Observed diff = {observed_diff:.4f}')

    # # Axis labels and title with fontsize 15
    # plt.xlabel('Permutation test statistic', fontsize=15)
    # plt.ylabel('Frequency', fontsize=15)
    # plt.title('Permutation Null Distribution (Selector vs Static B64)', fontsize=15)

    # # Tick label fonts
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)

    # # Legend with fontsize 15
    # plt.legend(fontsize=15)

    # plt.tight_layout()
    # plt.savefig("permutation_test_selector_vs_static.pdf")


def get_sbs_precisions(precision_csv_path):
    """
    Loads the precision file and extracts SBS per run and additional fixed configurations:
    - BFGS at budget 450
    - Non-elitist at budget 0
    - BFGS at budget 0
    - Same at budget 0

    Returns a dictionary with numpy arrays for each.
    """
    df_prec = pd.read_csv(precision_csv_path)

    # Extract BFGS at budget 450
    df_ne_16 = df_prec[(df_prec['algorithm'] == 'Non-elitist') & (df_prec['budget'] == 16)]
    df_ne_16 = df_ne_16.sort_values(['fid', 'iid']).reset_index(drop=True)
    ne_16_precisions = df_ne_16['precision'].values
    print(f"✅ Loaded {len(ne_16_precisions)} BFGS 450 precision entries from {precision_csv_path}")

    # Extract Non-elitist at budget 0
    df_nonelitist_0 = df_prec[(df_prec['algorithm'] == 'Non-elitist') & (df_prec['budget'] == 0)]
    df_nonelitist_0 = df_nonelitist_0.sort_values(['fid', 'iid']).reset_index(drop=True)
    nonelitist_0_precisions = df_nonelitist_0['precision'].values
    print(f"✅ Loaded {len(nonelitist_0_precisions)} Non-elitist 0 precision entries from {precision_csv_path}")

    # Extract BFGS at budget 0
    df_bfgs_0 = df_prec[(df_prec['algorithm'] == 'BFGS') & (df_prec['budget'] == 0)]
    df_bfgs_0 = df_bfgs_0.sort_values(['fid', 'iid']).reset_index(drop=True)
    bfgs_0_precisions = df_bfgs_0['precision'].values
    print(f"✅ Loaded {len(bfgs_0_precisions)} BFGS 0 precision entries from {precision_csv_path}")

    # Extract Same at budget 0
    df_same_0 = df_prec[(df_prec['algorithm'] == 'Same') & (df_prec['budget'] == 0)]
    df_same_0 = df_same_0.sort_values(['fid', 'iid']).reset_index(drop=True)
    same_0_precisions = df_same_0['precision'].values
    print(f"✅ Loaded {len(same_0_precisions)} Same 0 precision entries from {precision_csv_path}")

    # Return as dictionary for flexible downstream use
    return {
        'NE_16': ne_16_precisions,
        'Non-elitist_0': nonelitist_0_precisions,
        'BFGS_0': bfgs_0_precisions,
        'Same_0': same_0_precisions
    }

def plot_precision_boxplots(result_csv_path, precision_csv_path, output_png="precision_boxplots_low_budgets.pdf"):
    """
    Generates boxplots of SBS, VBS, Non-elitist/BFGS/Same at budget 0, all static selectors (excluding budgets 650, 700, 750), and selector.
    """

    # Load result file
    df = pd.read_csv(result_csv_path)

    # Extract SBS and baseline precisions
    precisions_dict = get_sbs_precisions(precision_csv_path)

    # Extract columns of interest
    static_cols = [col for col in df.columns if col.startswith("static_B")]
                #    and (col not in ["static_B650", "static_B700", "static_B750",
                #                     "static_B800", "static_B850", "static_B900",
                #                     "static_B950", "static_B1000"])]
    selector_col = "selector_precision"
    vbs_col = "vbs_precision"

    # Prepare data and labels
    data = []
    labels = []

    # Selector first
    if selector_col in df.columns:
        data.append(df[selector_col].values)
        labels.append("Selector")
    else:
        print(f"⚠ Selector column {selector_col} not found in data.")

    # # SBS BFGS 450
    # data.append(precisions_dict['NE_16'])
    # labels.append("Non-elitist 16")

    # # Non-elitist 0
    # data.append(precisions_dict['Non-elitist_0'])
    # labels.append("Non-elitist 0")

    # # BFGS 0
    # data.append(precisions_dict['BFGS_0'])
    # labels.append("BFGS 0")

    # # Same 0
    # data.append(precisions_dict['Same_0'])
    # labels.append("Elitist 0")

    # VBS
    if vbs_col in df.columns:
        data.append(df[vbs_col].values)
        labels.append("VBS")
    else:
        print(f"⚠ VBS column {vbs_col} not found in data.")

    # Static selectors
    for col in static_cols:
        data.append(df[col].values)
        labels.append(col.replace("static_", ""))  # shorter label

    # Plot
    plt.figure(figsize=(max(10, len(labels) * 0.5), 6))
    box = plt.boxplot(data, vert=True, patch_artist=True, showfliers=False)
    plt.yscale("log")

    # Style medians as orange lines
    for median in box['medians']:
        median.set(linewidth=3, color='orange')

    # Labels and title
    plt.ylabel("Reached Precision", fontsize=15)
    plt.title("Boxplots of Reached Precisions", fontsize=15)

    plt.xticks(ticks=np.arange(1, len(labels)+1), labels=labels, rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"✅ Precision boxplots saved to {output_png}")

def permutation_test_selector_vs_static_table(csv_path, output_pdf="permutation_test_pvalues_table.pdf"):
    df = pd.read_csv(csv_path)
    selector = df['selector_precision'].values
    # budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]
    budgets = [20*i for i in range(1, 10)] + [100*i for i in range(2, 21)]

    results = []

    for budget in budgets:
        print(f"Processing budget: {budget}")
        static_col = f'static_B{budget}'
        if static_col not in df.columns:
            print(f"⚠ {static_col} not found in dataframe, skipping.")
            continue

        static = df[static_col].values

        res = permutation_test(
            (selector, static),
            statistic=lambda x, y: np.mean(x - y),
            permutation_type='samples',
            vectorized=False,
            n_resamples=10000,
            alternative='less',
            random_state=42
        )

        results.append((f"B{budget}", res.pvalue))

    # Convert to DataFrame
    pval_df = pd.DataFrame(results, columns=["Static Selector", "p-value"])
    pval_df = pval_df.sort_values("p-value", ascending=False)

    # Plot as table
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(pval_df) + 1))
    ax.axis('off')
    table = ax.table(cellText=pval_df.values, colLabels=pval_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()
    print(f"✅ Permutation test p-value table saved to: {output_pdf}")

# Function that displays a table consisting of the budget-specific precision ratios, i.e. with budget-specific SBS and VBS
def budget_specific_prec_ratios(training_precision_path, test_precision_path, result_csv_path):
    df_train = pd.read_csv(training_precision_path)
    df_test = pd.read_csv(test_precision_path)
    df_result = pd.read_csv(result_csv_path)

    budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]
    for budget in budgets:
        print(f"Processing budget: {budget}")
        budget_specific_df_train = df_train[df_train['budget'] == budget]
        budget_specific_df_test = df_test[df_test['budget'] == budget]

        # Algorithm best in the training data
        best_algo = ""
        best_sum = np.inf
        for algo in budget_specific_df_train['algorithm'].unique():
            algo_sum = budget_specific_df_train[budget_specific_df_train['algorithm'] == algo]['precision'].sum()
            if algo_sum < best_sum:
                best_sum = algo_sum
                best_algo = algo

        budget_specific_sbs_sum = budget_specific_df_test[budget_specific_df_test['algorithm'] == best_algo]['precision'].sum()

        # Find the precision of the VBS
        # For each rep, we use the lowest precision across all algos and then add these sums
        # First loop: store min precision per group
        min_precisions = {}
        budget_specific_vbs_sum = 0
        for key, df_group in budget_specific_df_test.groupby(['fid', 'iid', 'rep']):
            min_val = df_group['precision'].min()
            min_precisions[key] = min_val
            budget_specific_vbs_sum += min_val

        budget_specific_selector_sum = df_result[f"static_B{budget}"].sum()
        print(
            f"Budget {budget}: SBS = {budget_specific_sbs_sum}, "
            f"VBS = {budget_specific_vbs_sum}, "
            f"Selector = {budget_specific_selector_sum}, "
            f"Ratio = {(budget_specific_sbs_sum - budget_specific_selector_sum) / (budget_specific_sbs_sum - budget_specific_vbs_sum)}"
        )

        # Second loop: use stored min precision
        prec_ratio = 0
        for key, group in df_result.groupby(['fid', 'iid', 'rep']):
            prec_ratio += min_precisions[key] / (group[f"static_B{budget}"].sum() + np.finfo(float).eps)

        print(f"Budget {budget}: Precision Ratio = {prec_ratio / len(df_result)}")

if __name__ == "__main__":
    # for k, v in res.items():
    #     print(f"{k}: {v}")
    # result1_csv = "../results/newInstances_normalized_log10_10D/selector_old_tuning_200_200.csv"
    # df = pd.read_csv(result1_csv)
  

    # # display_vbs_tables(result1_csv)
  
    # for col in df.columns:
    #     if col.startswith("static_B"):
    #         print(f"Column {col}: {df[col].sum()}")
    #     if col == "selector_precision":
    #         print(f"Column {col}: {df[col].sum()}")
    # permutation_test_selector_vs_static(df)
    # precision_df = pd.read_csv("../data/precision_files/A2_precisions_10D.csv")
    # res = find_sbs("../data/precision_files/A2_precisions_10D.csv")
    # print(res)
    # for idx, row in res.iterrows():
    #     print(f"Budget {row['budget']}, Algorithm {row['algorithm']}: Total Precision = {row['precision']}")
    result_csv = pd.read_csv("../results/newInstances_normalized_log10_10D/selector_old_tuning_200_200.csv")
    df = pd.read_csv("../data/precision_files/A2_precisions_newInstances_10D.csv")
    # display_vbs_tables("../results/newInstances_normalized_log10_10D/selector_old_tuning_200_200.csv")
    permutation_test_selector_vs_static_table("../results/newInstances_normalized_log10_10D/selector_old_tuning_200_200.csv")