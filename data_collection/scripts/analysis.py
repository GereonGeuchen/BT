import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm         


csv_file = "A2_precisions.csv"

def read_csv_file(exlude_1000: bool = False):
    
    df = pd.read_csv(csv_file, dtype={
        "fid": int, "iid": int, "rep": int, "budget": int, "algorithm": str, "fopt": float, "precision": float
    })

    # df = df[df["budget"] != 1000]
    # df = df[df["algorithm"] != "MLSL"]
    return df

# Find best switching point per rep
def best_switch_point(group):
    min_fopt = group["fopt"].min()
    return group[group["fopt"] == min_fopt]["budget"].min()

def plot_best_switching_point(df):
    """
    For each (fid, iid), finds the best switching point: For each repetition (rep),we find the switching point where one of the algorithms
    has achieved the highest precision across all budgets and algorithms. We then average these switching points across repetitions."""
    best_switch_df = df.groupby(["fid", "iid", "rep"]).apply(best_switch_point).reset_index(name="best_budget")

    # Aggregate to mean and std
    agg_df = best_switch_df.groupby(["fid", "iid"])["best_budget"].agg(["mean", "std"]).reset_index()
    agg_df = agg_df.rename(columns={"mean": "best_budget_avg", "std": "best_budget_std"})
    agg_df = agg_df.sort_values(by=["fid", "iid"]).reset_index(drop=True)
    agg_df["func_inst"] = agg_df["fid"].astype(str) + "-" + agg_df["iid"].astype(str)
    agg_df["y"] = range(len(agg_df))[::-1]

    all_maps = [cm.get_cmap('tab20'), cm.get_cmap('tab20b'), cm.get_cmap('tab20c')]
    fid_list = sorted(agg_df["fid"].unique())
    fid_to_color = {
        fid: all_maps[i // 20](i % 20)
        for i, fid in enumerate(fid_list)
    }

    plt.figure(figsize=(10, 20))
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5") 

    for x in range(0, 1050, 50):
        if x in [150]:
            plt.axvline(x=x, color='gray', linestyle='-', linewidth=2, zorder=0)
        else:
            plt.axvline(x=x, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)

    for _, row in agg_df.iterrows():
        color = fid_to_color[row["fid"]]
        plt.errorbar(
            x=row["best_budget_avg"],
            y=row["y"],
            xerr=row["best_budget_std"],
            fmt='o',
            color=color,       
            ecolor=color,         
            elinewidth=2,
            capsize=4,
            markerfacecolor=color,
            markeredgecolor=color
        )

    plt.yticks(ticks=agg_df["y"], labels=agg_df["func_inst"])
    plt.xlabel("Average Best Switching Point (Budget)")
    plt.ylabel("Function-Instance (Sorted)")
    plt.title("Average Best Switching Point with Variance (Colored by Function)")
    plt.grid(True, color='white')
    plt.tight_layout()
    plt.show(block=False)

def plot_switching_points_per_rep(df, fid, iid):
    """
    For each repetition (rep) of the given function ID and instance ID,
    plots all switching points (budgets) that achieved the minimum precision.
    """
    # Filter data for selected function and instance
    df_sub = df[(df["fid"] == fid) & (df["iid"] == iid)]

    min_precisions = df_sub.groupby("rep")["precision"].min().reset_index(name="min_precision")

    df_merged = df_sub.merge(min_precisions, on="rep")
    df_best = df_merged[df_merged["precision"] == df_merged["min_precision"]]

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.scatter(df_best["rep"], df_best["budget"], color='tab:blue')
    plt.xticks(sorted(df_best["rep"].unique()))
    plt.xlabel("Repetition (rep)")
    plt.ylabel("Switching Point (Budget)")
    plt.title(f"Switching Points with Min Precision per Rep\nFunction {fid}, Instance {iid}")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show(block=False)


def plot_function_switching_precisions(df, target_function=2):
    """
    Plots average precision (difference to global optimum) at each switching point,
    with one line per instance and shaded variance bands (±1 std dev).
    """
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
        # plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.xlabel("Switching Point (Budget)")
    plt.ylabel("Average Precision (log scale)")
    plt.yscale("log")
    plt.title(f"Function {target_function}: Avg Precision ± Std by Switching Point")
    plt.legend(title="Instance")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show(block=False)

def plot_robust_switching_point(df):
    """
    For each (fid, iid, rep), finds the most robust switching point: 
    the budget where the average precision across algorithms is lowest.
    Then aggregates across reps to compute the average robust point per instance.
    """
    def get_best_row(g, method="first"):
        min_val = g["precision"].min()
        min_rows = g[g["precision"] == min_val]
        
        if method == "first":
            return min_rows.iloc[0]  # first occurrence
        elif method == "last":
            return min_rows.iloc[-1]  # last occurrence
        else:
            raise ValueError("method must be 'first' or 'last'")
        
    # Step 1: average across algorithms
    avg_across_algos = df.groupby(["fid", "iid", "rep", "budget"])["precision"].mean().reset_index()

    # Step 2: find the best switching point per (fid, iid, rep)
    best_per_rep = avg_across_algos.groupby(["fid", "iid", "rep"]).apply(
        get_best_row, method="last"  # or "first"
    ).reset_index(drop=True)

    # Step 3: aggregate robust switching points across reps
    summary = best_per_rep.groupby(["fid", "iid"])["budget"].agg(["mean", "std"]).reset_index()
    summary = summary.rename(columns={"mean": "avg_robust_budget", "std": "std_robust_budget"})

    # Step 4: add plotting helpers
    summary = summary.sort_values(by=["fid", "iid"]).reset_index(drop=True)
    summary["func_inst"] = summary["fid"].astype(str) + "-" + summary["iid"].astype(str)
    summary["y"] = range(len(summary))[::-1]

    # Step 5: color mapping
    all_maps = [cm.get_cmap('tab20'), cm.get_cmap('tab20b'), cm.get_cmap('tab20c')]
    fid_list = sorted(summary["fid"].unique())
    fid_to_color = {
        fid: all_maps[i // 20](i % 20)
        for i, fid in enumerate(fid_list)
    }

    # Step 6: plotting
    plt.figure(figsize=(10, 20))
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")

    for _, row in summary.iterrows():
        color = fid_to_color[row["fid"]]
        plt.errorbar(
            x=row["avg_robust_budget"],
            y=row["y"],
            xerr=row["std_robust_budget"],
            fmt='o',
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
            markerfacecolor=color,
            markeredgecolor=color
        )

    plt.yticks(ticks=summary["y"], labels=summary["func_inst"])
    plt.xlabel("Avg Robust Switching Point (Budget)")
    plt.ylabel("Function-Instance (Sorted)")
    plt.title("Per-Rep Robust Switching Point Averaged Across Reps")
    plt.grid(True, color='white')
    plt.tight_layout()
    plt.show(block=False)

def plot_top_n_switching_points(df, top_n=10):
    """
    Plots the average of the top-N switching points (based on fopt) per function-instance.
    For each (fid, iid, rep), we take the budgets corresponding to the top-N lowest fopt values.
    Then average those budgets across reps, per rank (1st best, 2nd best, ..., Nth best).
    """
    df_sorted = df.sort_values(by=["fid", "iid", "rep", "fopt", "budget"])
    df_sorted["rank"] = df_sorted.groupby(["fid", "iid", "rep"]).cumcount()

    df_top_n = df_sorted[df_sorted["rank"] < top_n]

    avg_per_rank = (
        df_top_n.groupby(["fid", "iid", "rank"])["budget"]
        .mean()
        .reset_index()
        .rename(columns={"budget": "avg_budget"})
    )

    avg_per_rank["func_inst"] = avg_per_rank["fid"].astype(str) + "-" + avg_per_rank["iid"].astype(str)
    func_order = avg_per_rank[["fid", "iid"]].drop_duplicates().sort_values(by=["fid", "iid"])
    func_order["y"] = range(len(func_order))[::-1]

    avg_per_rank = avg_per_rank.merge(func_order, on=["fid", "iid"], how="left")

    all_maps = [cm.get_cmap('tab20'), cm.get_cmap('tab20b'), cm.get_cmap('tab20c')]
    fid_list = sorted(avg_per_rank["fid"].unique())
    fid_to_color = {
        fid: all_maps[i // 20](i % 20)
        for i, fid in enumerate(fid_list)
    }

    plt.figure(figsize=(10, 20))
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")

    for (fid, iid), group in avg_per_rank.groupby(["fid", "iid"]):
        y = group["y"].iloc[0]
        color = fid_to_color[fid]
        plt.plot(
            group["avg_budget"],
            [y] * len(group),
            marker='o',
            linestyle='-',
            color=color,
            label=f"{fid}-{iid}" if group["rank"].iloc[0] == 0 else None  # Label only once
        )

    yticks = func_order["y"]
    ylabels = func_order["fid"].astype(str) + "-" + func_order["iid"].astype(str)
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.xlabel("Average Switching Point (Budget)")
    plt.ylabel("Function-Instance (Sorted)")
    plt.title(f"Average of Top-{top_n} Best Switching Points (Per Rank, Colored by Function)")
    plt.grid(True, color='white')
    plt.tight_layout()
    plt.show(block=False)

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
        min_fopt = group["fopt"].min()
        best_budgets = group[group["fopt"] == min_fopt]["budget"]
        return target_budget in best_budgets.values

    grouped = df.groupby(["fid", "iid", "rep"])
    count = grouped.apply(is_target_best).sum()

    print(f"✅ Number of reps with best switching point at budget {target_budget}: {count} out of {len(grouped)}")
    print(f"   → That's {count / len(grouped) * 100:.2f}% of all reps")

    return count

def count_best_at_budget_per_fid(df, target_budget):
    """
    For each function ID (fid), counts the percentage of (iid, rep) combinations
    where `target_budget` is among the best switching points (lowest fopt).

    Args:
        df (pd.DataFrame): DataFrame with columns ["fid", "iid", "rep", "budget", "fopt"].
        target_budget (int): The budget to evaluate.

    Returns:
        pd.DataFrame: Table with columns [fid, total_reps, count_at_target, percentage]
    """
    def is_target_best(group):
        min_fopt = group["fopt"].min()
        best_budgets = group[group["fopt"] == min_fopt]["budget"]
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
    print(f"✅ Percentage of reps with budget {target_budget} as best switching point:")
    for _, row in result_df.iterrows():
        print(f"   - fid {row['fid']:2.0f}: {row['count_at_target']:4.0f} of {row['total_reps']:4.0f} reps → {row['percentage']:5.2f}%")

    return result_df

def average_optimal_switching_points_per_rep_filtered(df, min_budget=0):
    """
    Computes the average number of optimal switching points per (fid, iid, rep),
    considering only budgets >= `min_budget` and excluding reps where all budgets are optimal.

    Args:
        df (pd.DataFrame): Must include columns ['fid', 'iid', 'rep', 'budget', 'fopt'].
        min_budget (int): Minimum budget to consider as a switching point.

    Returns:
        float: Average number of optimal switching points per rep (after filtering).
    """
    filtered_df = df[df["budget"] >= min_budget]

    # Group by (fid, iid, rep)
    counts = []
    for (fid, iid, rep), group in filtered_df.groupby(["fid", "iid", "rep"]):
        min_fopt = group["fopt"].min()
        optimal_budgets = group[group["fopt"] == min_fopt]["budget"].unique()

        # Exclude reps where *all* budgets (in this filtered set) are optimal
        # if len(optimal_budgets) < len(group["budget"].unique()):
        counts.append(len(optimal_budgets))

    if counts:
        avg_count = sum(counts) / len(counts)
        print(f"✅ Average number of optimal switching points per rep (budget ≥ {min_budget}, incomplete reps only): {avg_count:.2f}")
        return avg_count
    else:
        print("⚠️ No reps met the filtering criteria.")
        return None



if __name__ == "__main__":
    # df = read_csv_file(True)
    df = read_csv_file(True)
    plot_robust_switching_point(df)
    input("Press Enter to close all plots...")
    