import os
import re
import pandas as pd
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm          # ✅ For color maps
import matplotlib.colors as mcolors
import seaborn as sns

def extract_min_precisions_long_format(root_dir, limit_to_eval_1000=False):
    folder_pattern = re.compile(r"A2_(?P<algorithm>.+)_B(?P<budget>\d+)_5D")
    file_pattern = re.compile(r"IOHprofiler_f(\d+)_DIM5\.dat")

    results = []

    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        if not os.path.isdir(folder_path):
            continue

        folder_match = folder_pattern.match(foldername)
        if not folder_match:
            continue

        algorithm = folder_match.group("algorithm")
        budget = int(folder_match.group("budget"))

        for data_subfolder in os.listdir(folder_path):
            data_path = os.path.join(folder_path, data_subfolder)
            if not os.path.isdir(data_path):
                continue

            for filename in os.listdir(data_path):
                file_match = file_pattern.match(filename)
                if not file_match:
                    continue

                fid = int(file_match.group(1))
                file_path = os.path.join(data_path, filename)

                with open(file_path, "r") as f:
                    lines = []
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('%') or line.startswith("evaluations"):
                            continue
                        lines.append(line)

                # Group lines by (iid, rep) based on values in the line
                blocks = {}
                for line in lines:
                    parts = line.split()
                    try:
                        eval_num = int(parts[0])
                        raw_y = float(parts[1])
                        rep = int(float(parts[2]))  # rep and iid now from file
                        iid = int(float(parts[3]))

                        if limit_to_eval_1000 and eval_num > 1000:
                            continue

                        key = (iid, rep)
                        if key not in blocks:
                            blocks[key] = []
                        blocks[key].append((eval_num, raw_y))

                    except Exception:
                        continue

                for (iid, rep), block_data in blocks.items():
                    evals, y_values = zip(*sorted(block_data, key=lambda x: x[0]))
                    fopt = y_values[-1]  # last function value (at max eval)
                    precision = min(y_values)  # best achieved so far

                    results.append({
                        "fid": fid,
                        "iid": iid,
                        "rep": rep,
                        "budget": budget,
                        "algorithm": algorithm,
                        "precision": precision
                    })

    df = pd.DataFrame(results)
    df["fid"] = df["fid"].astype(int)
    df["iid"] = df["iid"].astype(int)
    df["rep"] = df["rep"].astype(int)
    df["budget"] = df["budget"].astype(int)

    df = df.sort_values(by=["fid", "iid", "rep", "budget", "algorithm"])

    return df

def count_best_at_budget(df, target_budget):
    """
    Counts how many (fid, iid, rep) combinations have `target_budget`
    as the best switching point (i.e., lowest fopt).

    Args:
        df (pd.DataFrame): DataFrame containing columns ["fid", "iid", "rep", "budget", "fopt"].
        target_budget (int): The budget value to check as best switching point.

    Returns:
        int: Number of groups where `target_budget` is best.
    """
    def is_target_best(group):
        min_fopt = group["precision"].min()
        best_budgets = group[group["precision"] == min_fopt]["budget"]
        return target_budget in best_budgets.values

    grouped = df.groupby(["fid", "iid", "rep"])
    count = grouped.apply(is_target_best).sum()

    print(f"✅ Number of reps with best switching point at budget {target_budget}: {count} out of {len(grouped)}")
    print(f"   → That's {count / len(grouped) * 100:.2f}% of all reps")

    return count

def count_best_at_budget_per_fid(df, target_budget, plot=False):
    """
    For each function ID (fid), counts the percentage of (iid, rep) combinations
    where `target_budget` is among the best switching points (lowest fopt).

    Args:
        df (pd.DataFrame): DataFrame with columns ["fid", "iid", "rep", "budget", "fopt"].
        target_budget (int): The budget to evaluate.
        plot (bool): Whether to plot the results.

    Returns:
        pd.DataFrame: Table with columns [fid, total_reps, count_at_target, percentage]
    """
    def is_target_best(group):
        min_fopt = group["precision"].min()
        best_budgets = group[group["precision"] == min_fopt]["budget"]
        return target_budget in best_budgets.values

    results = []
    for fid, group in df.groupby("fid"):
        grouped = group.groupby(["iid", "rep"])
        count = grouped.apply(is_target_best).sum()
        total = len(grouped)
        percentage = count / total * 100

        results.append({
            "fid": fid,
            "total_reps": total,
            "count_at_target": count,
            "percentage": percentage
        })

    result_df = pd.DataFrame(results).sort_values("fid").reset_index(drop=True)

    # Print nicely
    print(f"Percentage of reps with budget {target_budget} as best switching point:")
    for _, row in result_df.iterrows():
        print(f"   - fid {row['fid']:2.0f}: {row['count_at_target']:4.0f} of {row['total_reps']:4.0f} reps → {row['percentage']:5.2f}%")

    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(result_df["fid"], result_df["percentage"], marker='o', linestyle='-')
        plt.xticks(result_df["fid"])  # Show all fids on x-axis
        plt.xlabel("Function ID (fid)")
        plt.ylabel(f"Percentage with budget {target_budget} as best (%)")
        plt.title(f"Best Switching Point Frequency at Budget {target_budget}")
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return result_df

def count_best_at_budget_per_instance(df, target_budget, plot=False):
    """
    For each (fid, iid), compute percentage of reps where `target_budget` is the best budget.
    Optionally plots all instances (fid × iid) with one dot per instance and colors per fid.

    Args:
        df (pd.DataFrame): DataFrame with columns ["fid", "iid", "rep", "budget", "fopt"].
        target_budget (int): Budget to check.
        plot (bool): Whether to plot the result.

    Returns:
        pd.DataFrame: Columns [fid, iid, total_reps, count_at_target, percentage]
    """
    def is_target_best(group):
        min_fopt = group["precision"].min()
        best_budgets = group[group["precision"] == min_fopt]["budget"]
        return target_budget in best_budgets.values

    results = []
    for (fid, iid), group in df.groupby(["fid", "iid"]):
        grouped = group.groupby("rep")
        count = grouped.apply(is_target_best).sum()
        total = len(grouped)
        percentage = count / total * 100

        results.append({
            "fid": fid,
            "iid": iid,
            "total_reps": total,
            "count_at_target": count,
            "percentage": percentage
        })

    # Create DataFrame and sort by fid, iid
    result_df = pd.DataFrame(results).sort_values(["fid", "iid"]).reset_index(drop=True)

    # Generate per-fid iid labels (1–5)
    result_df["plot_iid"] = result_df.groupby("fid").cumcount() + 1
    x_labels = result_df["plot_iid"].astype(str).tolist()

    print(f"Percentage of reps with budget {target_budget} as best switching point (per fid, iid):")
    for _, row in result_df.iterrows():
        print(f"   - fid {row['fid']:2.0f}, iid {row['iid']}: {row['count_at_target']:2.0f}/{row['total_reps']:2.0f} reps → {row['percentage']:5.1f}%")

    if plot:
        plt.figure(figsize=(14, 5))

        # Use tab20 palette for clear categorical distinction
        palette = sns.color_palette("tab20", n_colors=result_df["fid"].nunique())
        fid_order = sorted(result_df["fid"].unique())
        fid_colors = dict(zip(fid_order, palette))
        colors = result_df["fid"].map(fid_colors)

        # Plot
        plt.scatter(range(len(result_df)), result_df["percentage"], c=colors, s=50, zorder=3)
        plt.plot(result_df["percentage"], color="gray", alpha=0.3, zorder=2)

        # X-axis: instance labels, colored by fid
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels, fontsize=12)
        ax = plt.gca()
        for tick_label, fid in zip(ax.get_xticklabels(), result_df["fid"]):
            tick_label.set_color(fid_colors[fid])

        plt.xlabel("Instance ID (within function)", fontsize=14)
        plt.ylabel(f"Best at budget {target_budget} (%)", fontsize=14)
        plt.title(f"Per-instance: % of reps with budget {target_budget} as best", fontsize=16)
        plt.ylim(0, 105)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Legend (optional)
        # handles = [plt.Line2D([0], [0], marker='o', color='w',
        #                       label=f"fid {int(fid)}", markerfacecolor=col, markersize=8)
        #            for fid, col in fid_colors.items()]
        # plt.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left', title="Function ID", fontsize=12)

        plt.tight_layout()
        plt.show()

    return result_df



    return result_df


def plot_switching_points_per_rep(df, fid, iid):
    """
    For each repetition (rep) of the given function ID and instance ID,
    plots all switching points (budgets) that achieved the minimum precision.
    """
    # Filter data for selected function and instance
    df_sub = df[(df["fid"] == fid) & (df["iid"] == iid)]

    # Step 1: Get min precision per rep
    min_precisions = df_sub.groupby("rep")["precision"].min().reset_index(name="min_precision")

    # Step 2: Merge and filter rows where precision equals the min per rep
    df_merged = df_sub.merge(min_precisions, on="rep")
    df_best = df_merged[df_merged["precision"] == df_merged["min_precision"]]

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.scatter(df_best["rep"], df_best["budget"], color='tab:blue')
    plt.axhline(y=150, color='gray', linestyle='-', linewidth=2, label="Budget 150")

    plt.xticks(sorted(df_best["rep"].unique()))
    plt.xlabel("Repetition (rep)")
    plt.ylabel("Switching Point (Budget)")
    plt.title(f"Switching Points with Min Precision per Rep\nFunction {fid}, Instance {iid}")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

def plot_function_switching_precisions(df, target_function=2, save_dir=None, file_prefix="precision_plot"):
    """
    Plots average precision (difference to global optimum) at each switching point,
    with one line per instance and shaded variance bands (±1 std dev).
    """
    # Filter to the specified function
    df = df[df["fid"] == target_function]

    # Best precision across algorithms per rep
    best_per_rep = df.groupby(["iid", "budget", "rep"])["precision"].min().reset_index()

    # Compute mean and std for each instance and budget
    stats = best_per_rep.groupby(["iid", "budget"])["precision"].agg(["mean", "std"]).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    for iid, group in stats.groupby("iid"):
        group_sorted = group.sort_values("budget")
        x = group_sorted["budget"]
        y = group_sorted["mean"]
        yerr = group_sorted["std"]

        # Line plot
        plt.plot(x, y, marker='o', label=f"Instance {iid}")
        # Variance band
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.xlabel("Switching Point (Budget)")
    plt.ylabel("Average Precision (log scale)")
    #plt.yscale("log")
    plt.title(f"Function {target_function}: Avg Precision ± Std by Switching Point")
    plt.legend(title="Instance")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # Save if path is given
    if save_dir:
        # Construct save path
        if save_dir is None:
            save_dir = "."
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_prefix}_fid{target_function}.png"
        full_path = os.path.join(save_dir, filename)

        # Save
        plt.savefig(full_path, dpi=300)
        print(f"Plot saved to: {full_path}")
    plt.show(block=False)

def plot_best_budget_percentages(df, budget_range=range(50, 1001, 50)):
    """
    Loops over budgets and plots the percentage of (fid, iid, rep) groups
    where each budget is the best switching point (lowest fopt).

    Args:
        df (pd.DataFrame): DataFrame with columns ["fid", "iid", "rep", "budget", "precision"]
        budget_range (iterable): Budgets to test (default: 50 to 1000 in steps of 50)
    """
    def is_target_best(group, target_budget):
        min_fopt = group["precision"].min()
        best_budgets = group[group["precision"] == min_fopt]["budget"]
        return target_budget in best_budgets.values

    grouped = df.groupby(["fid", "iid", "rep"])
    total = len(grouped)

    percentages = []
    for budget in budget_range:
        count = grouped.apply(lambda g: is_target_best(g, budget)).sum()
        percent = count / total * 100
        percentages.append(percent)

     # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(list(budget_range), percentages, marker='o', linestyle='-', linewidth=2)

    # Add a thick vertical black line at budget 150
    plt.axvline(x=150, color='black', linestyle='--', linewidth=3, label='Budget 150')

    # Add vertical gridlines every 100 units
    for b in range(100, max(budget_range)+1, 100):
        plt.axvline(x=b, color='gray', linestyle=':', linewidth=1)

    # Improve text visibility
    plt.xlabel("Switching Point (Budget)", fontsize=14)
    plt.ylabel("Best Switching Point (%)", fontsize=14)
    plt.title("Percentage of Reps with Each Budget as Best Switching Point", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "budget": list(budget_range),
        "percentage": percentages
    })

def plot_optimal_switching_point_distribution(df, prefer_lowest=True):
    """
    For each (fid, iid), identify the best switching point (budget) per rep across all A2 algorithms.
    Then, plot a boxplot of those switching points across reps, with box and tick colors by fid.

    Args:
        df (pd.DataFrame): DataFrame with ["fid", "iid", "rep", "budget", "algorithm", "precision"]
        prefer_lowest (bool): Whether to use the lowest or highest optimal budget per rep.
    """
    import matplotlib.patches as mpatches

    optimal_budgets = []

    for (fid, iid), group in df.groupby(["fid", "iid"]):
        for rep, rep_group in group.groupby("rep"):
            min_precision = rep_group["precision"].min()
            best_budgets = rep_group[rep_group["precision"] == min_precision]["budget"]
            chosen_budget = best_budgets.min() if prefer_lowest else best_budgets.max()

            optimal_budgets.append({
                "fid": fid,
                "iid": iid,
                "rep": rep,
                "chosen_budget": chosen_budget
            })

    optimal_df = pd.DataFrame(optimal_budgets)
    optimal_df["label"] = optimal_df.apply(lambda row: f"f{int(row['fid']):02d}-i{int(row['iid'])}", axis=1)

    # Order labels
    label_order = sorted(optimal_df["label"].unique(), key=lambda x: (int(x[1:3]), int(x[-1])))
    fid_order = sorted(optimal_df["fid"].unique())
    palette = sns.color_palette("tab20", n_colors=len(fid_order))
    fid_colors = dict(zip(fid_order, palette))
    
    # Prepare data per label
    data = []
    colors = []
    fids = []

    for label in label_order:
        sub_df = optimal_df[optimal_df["label"] == label]
        data.append(sub_df["chosen_budget"].values)
        fid = int(label[1:3])
        fids.append(fid)
        colors.append(fid_colors[fid])

    # Plot
    plt.figure(figsize=(20, 6))
    box = plt.boxplot(data, patch_artist=True, widths=0.6)

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    for element in ["whiskers", "caps", "medians"]:
        for item in box[element]:
            item.set_color("black")

    plt.xticks(range(1, len(label_order) + 1), label_order, rotation=90)
    ax = plt.gca()
    for tick_label, fid in zip(ax.get_xticklabels(), fids):
        tick_label.set_color(fid_colors[fid])

    plt.xlabel("Function and Instance")
    plt.ylabel("Chosen Optimal Budget")
    plt.title(f"Optimal Switching Points Distribution Per Instance\n({'Lowest' if prefer_lowest else 'Highest'} Among Best Budgets)")
    plt.ylim(0, 1050)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Add legend
    handles = [
        mpatches.Patch(color=col, label=f"fid {fid}")
        for fid, col in fid_colors.items()
    ]
    plt.legend(handles=handles, title="Function ID", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("A2_precisions_all_switching_points.csv")
    count_best_at_budget_per_instance(df, 152, plot=True)
    input("Press Enter to continue...")