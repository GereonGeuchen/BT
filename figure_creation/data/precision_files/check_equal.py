import pandas as pd

df1 = pd.read_csv("A2_precisions_newInstances.csv")
df2 = pd.read_csv("A2_precisions_newInstances_test.csv")

# Check if the two DataFrames are equal
if df1.equals(df2):
    print("The DataFrames are equal.")
else:
    print("The DataFrames are not equal.")
    # Find differences
    diff = df1.compare(df2)
    print("Differences:")
    print(diff)