import os
import re
import pandas as pd
import sys
import pandas as pd
from ioh import get_problem, ProblemClass

def delete_true_y_files(root_dir="A1_data", suffix="_with_true_y.dat"):
    deleted = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                deleted += 1
    print(f"\n✅ Deleted {deleted} files ending with '{suffix}' from '{root_dir}'")


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
                        "fopt": fopt,
                        "precision": precision
                    })

    df = pd.DataFrame(results)
    df["fid"] = df["fid"].astype(int)
    df["iid"] = df["iid"].astype(int)
    df["rep"] = df["rep"].astype(int)
    df["budget"] = df["budget"].astype(int)

    df = df.sort_values(by=["fid", "iid", "rep", "budget", "algorithm"])

    df.to_csv("min_precisions_long_format_testing.csv", index=False)

    return df

def add_true_y_to_A1_data(root_dir="A1_data", dim=5, output_suffix="_with_true_y"):
    folder_pattern = re.compile(r"A1_B(?P<budget>\d+)_5D")
    file_pattern = re.compile(r"IOHprofiler_f(?P<fid>\d+)_DIM5\.dat")

    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        if not os.path.isdir(folder_path):
            continue

        folder_match = folder_pattern.match(foldername)
        if not folder_match:
            continue

        for data_subfolder in os.listdir(folder_path):
            data_path = os.path.join(folder_path, data_subfolder)
            if not os.path.isdir(data_path):
                continue

            for filename in os.listdir(data_path):
                file_match = file_pattern.match(filename)
                if not file_match:
                    continue

                fid = int(file_match.group("fid"))
                file_path = os.path.join(data_path, filename)

                try:
                    df = pd.read_csv(file_path, sep=r"\s+", engine="python", comment="#")
                except Exception as e:
                    print(f"❌ Could not read {file_path}: {e}")
                    continue

                if "raw_y" not in df.columns or "iid" not in df.columns:
                    print(f"⚠️ Skipping {file_path}: missing 'raw_y' or 'iid' column.")
                    continue

                try:
                    # Clean up iid and raw_y columns
                    df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype("Int64")
                    df["raw_y"] = pd.to_numeric(df["raw_y"], errors="coerce")
                    df = df.dropna(subset=["iid", "raw_y"])
                    df["iid"] = df["iid"].astype(int)

                    # Load optimal y-values for each unique iid
                    iid_values = df["iid"].unique()
                    optimum_map = {}
                    for iid in iid_values:
                        try:
                            problem = get_problem(fid, iid, dim, ProblemClass.BBOB)
                            optimum_map[iid] = problem.optimum.y
                        except Exception as e:
                            print(f"⚠️ Could not load f{fid}, i{iid}: {e}")
                            optimum_map[iid] = float("nan")

                    df["true_y"] = df.apply(
                        lambda row: row["raw_y"] + optimum_map.get(row["iid"], float("nan")),
                        axis=1,
                    )

                except Exception as e:
                    print(f"❌ Failed to compute true_y in {file_path}: {e}")
                    continue

                new_filename = filename.replace(".dat", f"{output_suffix}.dat")
                new_path = os.path.join(data_path, new_filename)

                try:
                    df.to_csv(new_path, sep="\t", index=False)
                    print(f"✅ Wrote: {new_path}")
                except Exception as e:
                    print(f"❌ Failed to write {new_path}: {e}")

if __name__ == "__main__":
    # delete_true_y_files()
    extract_min_precisions_long_format(root_dir="A2_testing_data", limit_to_eval_1000=True)