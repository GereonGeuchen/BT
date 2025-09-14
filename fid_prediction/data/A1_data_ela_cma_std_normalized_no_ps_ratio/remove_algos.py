import pandas as pd

budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]

for budget in budgets:
    path = f"A1_B{budget}_5D_ela_with_state.csv"
    df = pd.read_csv(path)
    #Remove last 6 columns
    df = df.iloc[:, :-6]
    df.to_csv(path, index=False)