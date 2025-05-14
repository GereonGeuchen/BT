import os
import re
import pandas as pd
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm          # ✅ For color maps
import matplotlib.colors as mcolors

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

def plot_function_switching_precisions(df, target_function=2):
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
    plt.show(block=False)

if __name__ == "__main__":
    df = extract_min_precisions_long_format(root_dir="A2_data_warm_MLSL", limit_to_eval_1000=True)
    df.to_csv("A2_precisions_warmmlsl.csv", index=False)
    # df = pd.read_csv("A2_precisions.csv")
    # plot_switching_points_per_rep(df, fid=2, iid=3)
    # plot_function_switching_precisions(df, target_function=2)
    # input("Press Enter to continue...")