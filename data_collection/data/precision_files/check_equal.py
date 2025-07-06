import pandas as pd

df1 = pd.read_csv("A2_late_precisions_newReps.csv")
df2 = pd.read_csv("A2_newReps_late_precisions.csv")

# Check if the two DataFrames are equal
if df1.equals(df2):
    print("The DataFrames are equal.")
else:
    print("The DataFrames are not equal.")
    # Find differences
    diff = df1.compare(df2)
    print("Differences:")
    print(diff)