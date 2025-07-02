import pandas as pd
import matplotlib.pyplot as plt

def plot_optimality_percentages(input_csv: str):
    # Read CSV
    df = pd.read_csv(input_csv)

    # Identify unique reps by fid, iid, rep
    grouped = df.groupby(['fid', 'iid', 'rep'])

    # Store optimal switching budgets
    optimal_budgets_per_rep = []

    for (fid, iid, rep), group in grouped:
        # Find the minimum precision for this rep
        min_precision = group['precision'].min()
        
        # Find all budgets where this minimum precision was achieved
        min_rows = group[group['precision'] == min_precision]
        
        # Get unique budgets achieving minimum precision for this rep
        optimal_budgets_per_rep.append(min_rows['budget'].unique())

    # Flatten the list of arrays to a single list
    all_optimal_budgets = [budget for budgets in optimal_budgets_per_rep for budget in budgets]

    # Convert to DataFrame
    optimal_budgets_df = pd.DataFrame(all_optimal_budgets, columns=['budget'])

    # Count how many reps had optimal switching point at each budget
    # Each budget counted at most once per rep due to .unique() above
    counts = optimal_budgets_df['budget'].value_counts().sort_index()

    # Convert counts to percentages (relative to total number of reps)
    total_reps = grouped.ngroups
    percentages = (counts / total_reps) * 100

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(percentages.index, percentages.values, marker='o', linestyle='-')

    # Axis labels and title with fontsize 15
    plt.xlabel('Budget', fontsize=15)
    plt.ylabel('Percentage of reps with optimal switching point at this budget', fontsize=15)
    plt.title('Distribution of Optimal Switching Budgets Across Reps', fontsize=15)

    # Set y-axis from 0 to 100
    plt.ylim(0, 100)

    # Format y-axis labels as percentages with fontsize 15
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../results/optimal_switching_budgets_distribution.pdf')

if __name__ == "__main__":
    input_csv = '../data/precision_files/A2_data_late_precisions.csv'  
    plot_optimality_percentages(input_csv)