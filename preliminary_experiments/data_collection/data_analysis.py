import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as SKPCA
from sklearn.manifold import MDS
import numpy as np
import os

# === Config ===
USE_PCA = True  # If False, uses MDS
USE_HIGH_LEVEL_CATEGORY = False
DO_OUTLIER_ANALYSIS = False
BUDGETS = [50 * i for i in range(1, 21)]  # Budget values to analyze
DATA_DIR = "A1_data_ela_disp_normalized"  # Attach _normalized if needed

EXCLUDED_COLUMNS = [
    "ela_distr.costs_runtime", "ela_meta.costs_runtime", "ela_level.costs_runtime",
    "disp.costs_runtime", "ic.costs_runtime", "nbc.costs_runtime"
]


# EXCLUDED_COLUMNS = [
#     "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10",
#     "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10",
#     "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
#     "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10",
#     "ela_level.lda_qda_10", "ela_level.lda_qda_25", "ela_level.lda_qda_50",
#     "ela_level.mmce_lda_10", "ela_level.mmce_lda_25", "ela_level.mmce_lda_50",
#     "ela_level.mmce_qda_10", "ela_level.mmce_qda_25", "ela_level.mmce_qda_50",
#     "ela_meta.lin_w_interact.adj_r2", "ela_meta.quad_w_interact.adj_r2",
#     "ic.eps_ratio", "ic.eps_s"
# ]

# === Helper Functions ===
def load_and_clean_data(filepath, drop_columns):
    df = pd.read_csv(filepath)
    drop_cols = drop_columns + ['fid', 'iid', 'rep', 'high_level_category']
    df_clean = df.drop(columns=drop_cols, errors='ignore')

    # Identify and drop columns with NaNs
    nan_columns = df_clean.columns[df_clean.isna().any()].tolist()
    df_clean = df_clean.drop(columns=nan_columns)

    # Drop any rows with remaining NaNs after inf replacement
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

    return df, df_clean, nan_columns


def apply_dimensionality_reduction(X, method='pca'):
    if method == 'pca':
        pca = SKPCA(n_components=2)
        return pca.fit_transform(X), ['PC1', 'PC2']
    elif method == 'mds':
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
        return mds.fit_transform(X), ['Dim1', 'Dim2']
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'mds'.")


def plot_2d_embedding(df_embed, x_col, y_col, color_col, title_prefix, budget):
    plt.figure(figsize=(10, 7))
    filled_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H', 'd']

    num_classes = df_embed[color_col].nunique()
    markers = filled_markers[:num_classes]
    palette = sns.color_palette('tab20', n_colors=num_classes)

    sns.scatterplot(
        data=df_embed, x=x_col, y=y_col, hue=color_col,
        style=color_col, markers=markers,
        s=100, palette=palette, edgecolor='black'
    )

    plt.title(f'{title_prefix}, Budget: {budget}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def analyze_outlier(df_embed, df_features, df_raw, columns, label_col, budget):
    df_embed['dist_to_origin'] = np.sqrt(df_embed[columns[0]]**2 + df_embed[columns[1]]**2)
    outlier_idx = df_embed['dist_to_origin'].idxmax()

    # Optional: force specific outlier override
    outlier_idx = df_embed[df_raw['fid'] == 5].index[0]

    outlier_label = df_embed.loc[outlier_idx, label_col]
    print(f"\n=== Outlier Analysis for Budget {budget} ===")
    print(f"Outlier {label_col}: {outlier_label}")

    outlier_feats = df_features.loc[outlier_idx]
    others_feats = df_features.drop(outlier_idx)

    z_scores = (outlier_feats - others_feats.mean()) / others_feats.std()
    z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()
    top_features = z_scores.abs().sort_values(ascending=False).head(10)

    print("Top feature deviations:")
    print(top_features)

    plt.figure(figsize=(8, 6))
    top_features.plot(kind='barh')
    plt.title(f"Top Outlier Features for {label_col} = {outlier_label} (Budget {budget})")
    plt.xlabel("Z-score (magnitude)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# === Main Analysis ===
nan_report = {}  # budget -> list of dropped columns

for budget in BUDGETS:
    file_path = f"{DATA_DIR}/A1_B{budget}_5D_ela.csv"

    try:
        df_raw, df_features, nan_columns = load_and_clean_data(file_path, EXCLUDED_COLUMNS)
        if nan_columns:
            nan_report[budget] = nan_columns
    except FileNotFoundError:
        print(f"Skipping missing file: {file_path}")
        continue

    method = 'pca' if USE_PCA else 'mds'
    embedding, columns = apply_dimensionality_reduction(df_features, method)

    df_embed = pd.DataFrame(embedding, columns=columns)
    label_col = 'high_level_category' if USE_HIGH_LEVEL_CATEGORY else 'fid'
    df_embed[label_col] = df_raw[label_col]

    title = f"2D {'PCA' if USE_PCA else 'MDS'} Clustering of ELA Features by {'High-Level Category' if USE_HIGH_LEVEL_CATEGORY else 'Function'}"
    plot_2d_embedding(df_embed, columns[0], columns[1], label_col, title, budget)

    if DO_OUTLIER_ANALYSIS:
        analyze_outlier(df_embed, df_features, df_raw, columns, label_col, budget)

# === Summary ===
print("\n=== Summary of Dropped Columns per Budget ===")
for budget, columns in nan_report.items():
    print(f"Budget {budget}: {columns}")
