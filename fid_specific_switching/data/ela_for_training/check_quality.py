import pandas as pd
import numpy as np

budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]  # 8, 16, ..., 96, 50, 100, ..., 950


for budget in budgets:
    # Paths to your two CSV files
    # csv1_path = f'ela_data/A1_data_ela_newInstances/A1_B{budget}_5D_ela.csv'
    # csv2_path = f'ela_data/A1_data_ela_newInstances_test/A1_B{budget}_5D_ela.csv'
    # csv1_path = f'ela_normalized/A1_data_ela_cma_std_precisions_l_BFGS_b_test/A1_B{budget}_5D_ela_with_state.csv'
    # csv2_path = f'ela_normalized/A1_data_ela_cma_std_precisions_normalized_l_BFGS_b/A1_B{budget}_5D_ela_with_state.csv'
    csv1_path = f"ela_for_training/A1_data_ela_cma_std_precisions_normalized_no_ps_ratio/A1_B{budget}_5D_ela_with_state.csv"
    csv2_path = f"ela_for_training/A1_data_ela_cma_std_precisions_normalized_test/A1_B{budget}_5D_ela_with_state.csv"

    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Drop last 6 columns from df1 and columns that include std in their name
    # df1 = df1.iloc[:, :-6]
    # df1 = df1.loc[:, ~df1.columns.str.contains("std")]

    df2 = df2.drop(columns=["ps_ratio"], errors='ignore')

    # Drop column high_level_category in both
    df1 = df1.drop(columns=["high_level_category"], errors='ignore')
    df2 = df2.drop(columns=["high_level_category"], errors='ignore')

    # --- Exclude columns as required ---

    # # For df1: drop the third column (index 2)
    # df1_dropped = df1.drop(df1.columns[2], axis=1)

    # # For df2: drop the third column (index 2) and the last 13 columns
    # cols_to_drop_df2 = [df2.columns[2]] + list(df2.columns[-13:])
    # df2_dropped = df2.drop(cols_to_drop_df2, axis=1)

    # --- Compare resulting DataFrames with numerical tolerance ---

    # Ensure they have the same shape after dropping
    if df1.shape != df2.shape:
        print(f"❌ Budget {budget}: DataFrames have different shapes after dropping columns.")
        print(f"df1 shape: {df1.shape}, df2 shape: {df2.shape}")
    else:
        # Convert to numpy arrays for allclose
        arr1 = df1.to_numpy()
        arr2 = df2.to_numpy()

        comparison = np.isclose(arr1, arr2, rtol=1e-15, atol=1e-15)

        if comparison.all():
            print(f"✅ Budget {budget}: DataFrames match within numerical tolerance.")
        else:
            print(f"❌ Budget {budget}: DataFrames do NOT match within numerical tolerance.")

            # Find mismatched positions
            mismatched_rows, mismatched_cols = np.where(~comparison)

            # Map numeric column indices back to column names
            colnames = df1.columns

            # Track which columns have been reported
            reported_cols = set()

            for row, col_idx in zip(mismatched_rows, mismatched_cols):
                col = colnames[col_idx]
                if col not in reported_cols:
                    val1 = arr1[row, col_idx]
                    val2 = arr2[row, col_idx]
                    print(f"Column: {col}")
                    print(f"Row index: {row}")
                    print(f"df1 value: {val1}")
                    print(f"df2 value: {val2}\n")
                    reported_cols.add(col)
