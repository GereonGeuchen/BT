import pandas as pd

def update_vbs_selector_and_staticB1000(csv_path, precision_file, output_path=None):
    df = pd.read_csv(csv_path)
    prec_df = pd.read_csv(precision_file)

    # 1. Compute vbs_selector from columns starting at index 5
    df["vbs_selector"] = df.iloc[:, 5:].min(axis=1)

    # 2. Lookup correct static_B1000 values from precision file
    lookup = prec_df[
        (prec_df["budget"] == 1000) & (prec_df["algorithm"] == "Same")
    ][["fid", "iid", "rep", "precision"]]
    lookup = lookup.rename(columns={"precision": "static_B1000"})

    # 3. Merge to get correct values
    df = df.drop(columns=["static_B1000"], errors="ignore")  # Remove old if exists
    df = pd.merge(df, lookup, on=["fid", "iid", "rep"], how="left")

    # 4. Save
    static_cols = [col for col in df.columns if col.startswith("static_B") and col != "static_B1000"]
    static_cols = sorted(static_cols, key=lambda c: int(c.split("_B")[1]))

    # Final column order
    new_order = [
        "fid", "iid", "rep",
        "selector_precision", "selector_switch_budget",
        "vbs_precision", "vbs_selector"
    ] + static_cols + ["static_B1000"]

    # Add any remaining columns not explicitly ordered
    remaining = [col for col in df.columns if col not in new_order]
    df = df[new_order + remaining]

    # 5. Save updated CSV
    output_path = output_path or csv_path
    df.to_csv(output_path, index=False)
    print(f"✅ Updated with sorted static columns, saved to: {output_path}")