import pandas as pd

def update_vbs_precision(csv_path, precision_file, output_path=None):
    df = pd.read_csv(csv_path)
    prec_df = pd.read_csv(precision_file)

    # 1. Compute vbs_precision from A2_precisions.csv
    vbs = (
        prec_df
        .groupby(["fid", "iid", "rep"])["precision"]
        .min()
        .reset_index()
        .rename(columns={"precision": "vbs_precision"})
    )

    df = pd.merge(df, vbs, on=["fid", "iid", "rep"], how="left")

    # 2. Sort static columns
    static_cols = [col for col in df.columns if col.startswith("static_B")]
    static_cols = sorted(static_cols, key=lambda c: int(c.split("_B")[1]))

    # 3. Final column order
    new_order = [
        "fid", "iid", "rep",
        "vbs_precision"
    ] + static_cols

    # 4. Add any remaining columns not explicitly ordered
    remaining = [col for col in df.columns if col not in new_order]
    df = df[new_order + remaining]

    # 5. Save updated CSV
    output_path = output_path or csv_path
    df.to_csv(output_path, index=False)
    print(f"✅ Updated with vbs_precision only, saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    update_vbs_precision(
        csv_path="predicted_static_precisions.csv",
        precision_file="../data/A2_precisions.csv",
        output_path="../data/A2_results_updated.csv"
    )