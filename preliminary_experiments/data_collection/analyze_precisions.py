import os
import re
import pandas as pd
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def count_best_at_budget_per_instance(df, target_budget, plot=False, save_plot=False):
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
        if save_plot:
            dir_path = f"../results/budget_optimality_over_instances"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            out_path = os.path.join(dir_path, f"budget_{target_budget}.png")
            plt.savefig(out_path)
            print(f"Plot saved to {out_path}")
            plt.close()
        else:
            plt.show()


    return result_df

def plot_switching_points_per_rep(df, fid, iid, min_budget=0, save_plot=False):
    """
    For each repetition (rep) of the given function ID and instance ID,
    plots all switching points (budgets) > min_budget that achieved the minimum precision,
    color-coded by algorithm. If multiple algorithms tie for the best precision,
    multiple points are plotted side by side.

    Args:
        df (pd.DataFrame): The precision dataframe.
        fid (int): Function ID.
        iid (int): Instance ID.
        min_budget (int): Minimum budget to consider when selecting best precision points.
        save_plot (bool): Whether to save the plot to disk.
    """
    import matplotlib.pyplot as plt
    import os

    # Filter data
    df_sub = df[(df["fid"] == fid) & (df["iid"] == iid) & (df["budget"] > min_budget)]

    if df_sub.empty:
        print(f"No data available for fid={fid}, iid={iid} with min_budget={min_budget}")
        return

    # Find min precision per rep
    min_precisions = df_sub.groupby("rep")["precision"].min().reset_index(name="min_precision")

    # Merge and filter to best-performing rows
    df_merged = df_sub.merge(min_precisions, on="rep")
    df_best = df_merged[df_merged["precision"] == df_merged["min_precision"]]

    # Shift dots slightly for visibility when multiple algorithms tie
    algo_list = df_best["algorithm"].unique()
    algo_offsets = {algo: i * 0.1 for i, algo in enumerate(algo_list)}

    # Plotting
    plt.figure(figsize=(10, 5))
    for algo in algo_list:
        df_algo = df_best[df_best["algorithm"] == algo]
        x = df_algo["rep"] + algo_offsets[algo]
        plt.scatter(x, df_algo["budget"], label=algo)

    plt.axhline(y=150, color='gray', linestyle='--', linewidth=1.5, label="Budget 150")
    plt.xticks(sorted(df_best["rep"].unique()))
    plt.xlabel("Run")
    plt.ylabel("Switching Point (Budget)")
    plt.title(f"Best Switching Points per Run (F{fid}, I{iid})\n(Only budgets > {min_budget})")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        dir_path = f"../results/precisions_50/switching_points_per_rep_with_algos"
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"fid_{fid}_iid_{iid}_minb_{min_budget}.png")
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
    else:
        plt.show(block=False)


def plot_best_budget_percentages(df, budget_range=range(50, 1001, 50), min_budget=0):
    """
    Loops over budgets and plots the percentage of (fid, iid, rep) groups
    where each budget is the best switching point (lowest fopt), considering
    only budgets greater than `min_budget`.

    Args:
        df (pd.DataFrame): DataFrame with columns ["fid", "iid", "rep", "budget", "precision"]
        budget_range (iterable): Budgets to test (default: 50 to 1000 in steps of 50)
        min_budget (int): Minimum budget to consider when determining optimality.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Filter budgets that will be evaluated
    budget_range = [b for b in budget_range if b > min_budget]

    def is_target_best(group, target_budget):
        filtered_group = group[group["budget"] > min_budget]
        if filtered_group.empty:
            return False
        min_fopt = filtered_group["precision"].min()
        best_budgets = filtered_group[filtered_group["precision"] == min_fopt]["budget"]
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
    plt.plot(budget_range, percentages, marker='o', linestyle='-', linewidth=2)

    # Add a thick vertical black line at budget 150
    if 150 in budget_range:
        plt.axvline(x=150, color='black', linestyle='--', linewidth=3, label='Budget 150')

    # Add vertical gridlines every 100 units
    for b in range(100, max(budget_range) + 1, 100):
        plt.axvline(x=b, color='gray', linestyle=':', linewidth=1)

    # Improve text visibility
    plt.xlabel("Switching Point (Budget)", fontsize=14)
    plt.title(f"Percentage of runs with each budget as optimal switching point\n(Only considering budgets > {min_budget})", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "budget": budget_range,
        "percentage": percentages
    })


def plot_optimal_switching_point_distribution(df, prefer_lowest=True, min_budget=0):
    """
    For each (fid, iid), identify the best switching point (budget) per rep across all A2 algorithms,
    considering only budgets greater than `min_budget`.
    Then, plot a boxplot of those switching points across reps, with box and tick colors by fid.

    Args:
        df (pd.DataFrame): DataFrame with ["fid", "iid", "rep", "budget", "algorithm", "precision"]
        prefer_lowest (bool): Whether to use the lowest or highest optimal budget per rep.
        min_budget (int): Minimum budget to consider when selecting optimal switching points.
    """

    optimal_budgets = []

    for (fid, iid), group in df.groupby(["fid", "iid"]):
        for rep, rep_group in group.groupby("rep"):
            filtered_group = rep_group[rep_group["budget"] > min_budget]
            if filtered_group.empty:
                continue
            min_precision = filtered_group["precision"].min()
            best_budgets = filtered_group[filtered_group["precision"] == min_precision]["budget"]
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
    plt.title(f"Optimal Switching Points Distribution Per Instance\n"
              f"({'Lowest' if prefer_lowest else 'Highest'} Among Best Budgets, "
              f"Only Budgets > {min_budget})")
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


def plot_optimal_switching_algorithm_summary(df, min_budget=0, save_plot=False):
    """
    Plots the percentage of optimal switching points where each algorithm participated,
    for each function-instance. A switching point is optimal if it is one of the budgets
    > min_budget where minimum precision was achieved for a rep. Multiple algorithms can tie.

    Parameters:
        df: DataFrame with columns [fid, iid, rep, budget, algorithm, precision]
        min_budget: Minimum budget to consider when identifying optimal switching points.
        save_plot: If True, saves the plot to a file instead of displaying it.
    """
    import matplotlib.pyplot as plt
    import os

    algorithms = df["algorithm"].unique()
    
    instance_keys = df[["fid", "iid"]].drop_duplicates().sort_values(["fid", "iid"])
    instance_labels = [f"f{fid}-i{iid}" for fid, iid in zip(instance_keys["fid"], instance_keys["iid"])]
    
    algo_scores = {algo: [] for algo in algorithms}

    for _, row in instance_keys.iterrows():
        fid, iid = row["fid"], row["iid"]
        df_sub = df[(df["fid"] == fid) & (df["iid"] == iid)]

        optimal_rows = []
        for rep, rep_df in df_sub.groupby("rep"):
            filtered = rep_df[rep_df["budget"] > min_budget]
            if filtered.empty:
                continue
            min_precision = filtered["precision"].min()
            best_rows = filtered[filtered["precision"] == min_precision]
            best_budgets = best_rows["budget"].unique()

            for budget in best_budgets:
                relevant = best_rows[best_rows["budget"] == budget]
                for _, r in relevant.iterrows():
                    optimal_rows.append((rep, budget, r["algorithm"]))

        total_opt_points = len(set((r, b) for (r, b, a) in optimal_rows))
        algo_counter = {algo: 0 for algo in algorithms}
        for _, _, algo in optimal_rows:
            algo_counter[algo] += 1

        for algo in algorithms:
            percent = 100 * algo_counter[algo] / total_opt_points if total_opt_points > 0 else 0
            algo_scores[algo].append(percent)

    # Plotting
    plt.figure(figsize=(14, 6))
    for algo in algorithms:
        plt.plot(instance_labels, algo_scores[algo], marker='o', label=algo)

    plt.xticks(rotation=90, ha="right")

    plt.ylim(0, 100)
    plt.ylabel("Participation in Optimal Switching Points (%)")
    plt.xlabel("Function-Instance")
    plt.title(f"Algorithm Involvement in Optimal Switching Points (budgets > {min_budget})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(title="Algorithm")
    plt.tight_layout()

    # Save or show
    if save_plot:
        dir_path = "../results/precisions_50/optimal_switching_summary"
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"algorithm_involvement_summary_minb_{min_budget}.png")
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
    else:
        plt.show(block=False)

def plot_precision_gradient_per_rep(df, fid, iid, min_budget=0, save_plot=False):
    """
    Plots a heatmap showing the precision values for each (rep, budget) pair,
    with budgets on the Y-axis and reps on the X-axis.
    Color intensity shows the precision value (lower = better).

    Args:
        df (pd.DataFrame): The precision dataframe with columns [fid, iid, rep, budget, precision].
        fid (int): Function ID to filter.
        iid (int): Instance ID to filter.
        min_budget (int): Minimum budget to include in the plot.
        save_plot (bool): If True, saves the plot to disk.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os

    # Filter relevant data
    df_sub = df[(df["fid"] == fid) & (df["iid"] == iid) & (df["budget"] > min_budget)]

    if df_sub.empty:
        print(f"No data available for fid={fid}, iid={iid} with min_budget={min_budget}")
        return

    # Aggregate over algorithms: pick the best (min precision) for each (rep, budget)
    agg_df = df_sub.groupby(["rep", "budget"])["precision"].min().reset_index()

    # Pivot so we have budgets as rows (y-axis) and reps as columns (x-axis)
    pivot_df = agg_df.pivot(index="budget", columns="rep", values="precision").sort_index()

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_df, cmap="viridis", cbar_kws={"label": "Precision"}, linewidths=0.3, linecolor='gray')
    plt.gca().invert_yaxis()
    plt.ylabel("Switching Point (Budget)", fontsize=12)
    plt.xlabel("Run (rep)", fontsize=12)
    plt.title(f"Precision Gradient Across Runs for F{fid}, I{iid} (budgets > {min_budget})", fontsize=14)
    plt.tight_layout()

    if save_plot:
        dir_path = f"../results/precisions/precisions_gradient"
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"fid_{fid}_iid_{iid}_minb_{min_budget}_heatmap.png")
        plt.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
    else:
        plt.show(block=False)


if __name__ == "__main__":
    df= pd.read_csv("A2_precisions_all_switching_points.csv")
    # plot_best_budget_percentages(df, budget_range=range(8, 1001, 8))
    # plot_optimal_switching_algorithm_summary(df, save_plot=False)

    # for fid in range(1, 25):
    #     for iid in range(1, 6):
    #         plot_switching_points_per_rep(df, fid, iid, min_budget=50, save_plot=True)
    # plot_switching_points_per_rep(df, 1, 1, save_plot=False)
    # for fid in range(1, 25):
    #     for iid in range(1, 6):
    #         plot_precision_gradient_per_rep(df, fid, iid, min_budget=0, save_plot=True)
    plot_optimal_switching_algorithm_summary(df, min_budget=0)
    plot_optimal_switching_algorithm_summary(df, min_budget=50)
    input("Press Enter to exit...")
