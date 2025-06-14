import pandas as pd

df = pd.read_csv("A1_B16_5D_ela.csv")
df.drop(columns=['ela_meta.lin_w_interact.adj_r2'], inplace=True)  # Remove budget column after merge
df.to_csv("A1_B16_5D_ela.csv", index=False)