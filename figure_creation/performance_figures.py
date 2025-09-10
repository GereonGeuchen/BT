import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from scipy.stats import permutation_test

bfgs_sum = 19700627734.00593
de_sum   = 2028474.7954354833
pso_sum  = 29118.065420629897
mlsl_sum = 44652.177305832396
same_sum = 14429.203416653301
nonelit_sum = 2966.9578762185

TABLE_BG = "rgb(230,230,230)"   # table background (grey)
PAGE_BG  = "white"              # page background (white)

def make_selector_score_table(
    df,
    group=None,
    static_prefix="static_",
    save_pdf=None,
    width=900,
    height=520,
    rows_per_panel=16,
    font_family="Latin Modern Roman",
    sort_descending=True,
    method_col_width=110,
    group_col_width=90,
    num_col_width=110,
    last_num_col_width=150,      # <-- slightly wider last column
):
    """
    Build two side-by-side tables of Selector scores (16 rows each by default).

    Score formula (requested):
        score = (sum_sbs - sum_method) / (sum_sbs - sum_vbs)
      -> VBS = 1, SBS = 0

    Output:
      - Returns (fig, res) and optionally saves PDF/PNG/HTML without opening a browser.
    """
    pio.kaleido.scope.mathjax = None
    # methods to include: selector + all static_*
    methods = ["selector_precision"] + [c for c in df.columns if c.startswith(static_prefix)]
    group_cols = [] if group is None else ([group] if isinstance(group, str) else list(group))

    def block(sub):
        vbs_sum = sub["vbs_precisions"].sum()
        sbs_sum = sub["sbs_precision"].sum()
        den = sbs_sum - vbs_sum
        rows = []
        for col in methods:
            msum = sub[col].sum()
            score = (sbs_sum - msum) / den if den != 0 else np.nan
            name = "selector" if col == "selector_precision" else col
            rows.append({"method": name, "sum_precision": msum, "score": score})
        return pd.DataFrame(rows)

    if group_cols:
        parts = []
        for keys, sub in df.groupby(group_cols, dropna=False):
            b = block(sub)
            if not isinstance(keys, tuple):
                keys = (keys,)
            for k, col in zip(keys, group_cols):
                b[col] = k
            parts.append(b)
        res = (
            pd.concat(parts, ignore_index=True)
              .loc[:, group_cols + ["method", "sum_precision", "score"]]
              .sort_values(group_cols + ["score"],
                           ascending=[True]*len(group_cols) + [not sort_descending])
              .reset_index(drop=True)
        )
        title = f"Selector Scores grouped by {', '.join(group_cols)}"
    else:
        res = block(df).sort_values("score", ascending=not sort_descending).reset_index(drop=True)
        title = "Selector Scores"

    # Round & rename display columns
    res["sum_precision"] = res["sum_precision"].round(6)
    res["score"] = res["score"].round(6)
    res = res.rename(columns={
        "sum_precision": "Sum of regrets",
        "score": "Fraction of the gap closed"
    })

    # Split into two panels
    left  = res.iloc[:rows_per_panel].copy()
    right = res.iloc[rows_per_panel:rows_per_panel*2].copy()

    def compute_colwidths(cols):
        widths = []
        n = len(cols)
        for i, c in enumerate(cols):
            if c in group_cols:
                widths.append(group_col_width)
            elif c == "method":
                widths.append(method_col_width)
            else:
                # widen only the last column (the long header)
                widths.append(last_num_col_width if i == n - 1 else num_col_width)
        return widths

    def table_trace(dataframe):
        cols = list(dataframe.columns)
        values = [dataframe.get(col, pd.Series([], dtype=object)) for col in cols]
        align  = [("left" if (c in group_cols or c == "method") else "right") for c in cols]
        return go.Table(
            header=dict(
                values=cols,
                fill_color=TABLE_BG,
                line_color="black",
                line_width=0.5,
                align="left",
                font=dict(family=font_family, size=12),
            ),
            cells=dict(
                values=values,
                fill_color=TABLE_BG,
                line_color="black",
                line_width=0.35,
                align=align,
                height=24,
                font=dict(family=font_family, size=11),
            ),
            columnwidth=compute_colwidths(cols),
        )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "table"}, {"type": "table"}]],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.06,
    )
    fig.add_trace(table_trace(left),  row=1, col=1)
    fig.add_trace(table_trace(right), row=1, col=2)

    fig.update_layout(
        title=title,
        width=width, height=height,
        paper_bgcolor=PAGE_BG,    # white page
        plot_bgcolor=PAGE_BG,     # white plot area
        font=dict(family=font_family, size=12, color="black"),
        margin=dict(l=12, r=12, t=50, b=12)  # tight borders
    )
    if save_pdf is not None and not os.path.exists(save_pdf):
        os.makedirs(os.path.dirname(save_pdf), exist_ok=True)

    # Save files without opening any viewer
    if save_pdf:
        fig.write_image(save_pdf, format="pdf")   # requires: pip install -U kaleido

    return fig, res

def plot_gap_closed_by_fid(
    df: pd.DataFrame,
    static_prefix: str = "static_",
    save_pdf: str | None = None,
    width: int = 880,
    height: int = 480,
    font_family: str = "Latin Modern Roman",
    fid_groups: dict[str, list[int]] | None = None,  # NEW: map label -> list of fids
):
    """
    If fid_groups is None:
        Line+marker plot across fids (x-axis) for each selector (selector + selected static_*).
        y per (method, fid) = (sum_sbs(fid) - sum_method(fid)) / (sum_sbs(fid) - sum_vbs(fid)).
    If fid_groups is provided:
        Same formula but aggregated per group label:
        y per (method, group) = (sum_sbs(group) - sum_method(group)) / (sum_sbs(group) - sum_vbs(group)),
        where sums are over all rows whose fid is in that group's fid list.

    Visual twist: values < 0 are *compressed by 0.5* on the y-axis (for readability),
    but axis tick labels show the original (unscaled) values.
    """

    # --- config: which methods to show
    methods = ["selector_precision", "static_B80"]
    pio.kaleido.scope.mathjax = None

    # --- helpers
    def _scale_y(val: float) -> float:
        if pd.isna(val):
            return val
        return val if val >= 0 else 0.5 * val

    def _scale_array(vals):
        return [(_scale_y(v) if pd.notna(v) else v) for v in vals]

    # --- compute scores (original, unscaled)
    recs = []

    if fid_groups is None:
        # Per-fid mode (original behavior)
        for fid, sub in df.groupby("fid", dropna=False):
            vbs_sum = sub["vbs_precisions"].sum()
            sbs_sum = sub["sbs_precision"].sum()
            den = sbs_sum - vbs_sum
            for col in methods:
                msum = sub[col].sum()
                num = sbs_sum - msum
                if num == den:
                    score = 1.0
                elif den == 0:
                    score = 0.0
                else:
                    score = num / den

                # Pretty names
                if col == "selector_precision":
                    name = "Dynamic selector"
                elif col == "Non-elitist":
                    name = "CMA-ES, non-elitist"
                elif col == "static_B80":
                    name = "B80"
                else:
                    name = col

                recs.append({"x": fid, "method": name, "fraction_closed": score})

        x_title = "fid"
        # preserve natural numeric ordering for fids
        x_order = list(range(1, int(df["fid"].max()) + 1))

    else:
        # Grouped mode
        # Keep given order of groups
        x_order = list(fid_groups.keys())
        for label, fid_list in fid_groups.items():
            sub = df[df["fid"].isin(fid_list)]
            vbs_sum = sub["vbs_precisions"].sum()
            sbs_sum = sub["sbs_precision"].sum()
            den = sbs_sum - vbs_sum
            for col in methods:
                msum = sub[col].sum()
                num = sbs_sum - msum
                if num == den:
                    score = 1.0
                elif den == 0:
                    score = 0.0
                else:
                    score = num / den

                if col == "selector_precision":
                    name = "Dynamic selector"
                elif col == "Non-elitist":
                    name = "CMA-ES, non-elitist"
                elif col == "static_B80":
                    name = "B80"
                else:
                    name = col

                recs.append({"x": label, "method": name, "fraction_closed": score})

        x_title = "fid group"

    scores = pd.DataFrame(recs).sort_values(["method"], kind="stable").reset_index(drop=True)
    scores["fraction_closed_scaled"] = _scale_array(scores["fraction_closed"])

    # --- figure
    fig = go.Figure()
    for method, subm in scores.groupby("method", sort=False):
        # Ensure x follows x_order
        if isinstance(x_order[0], str):
            subm = subm.set_index("x").reindex(x_order).reset_index()
        else:
            subm = subm.sort_values("x")
        fig.add_trace(
            go.Scatter(
                x=subm["x"],
                y=subm["fraction_closed_scaled"],
                customdata=subm["fraction_closed"],
                mode="lines+markers",
                name=method,
                marker=dict(size=7, line=dict(width=0.5, color="black")),
                line=dict(width=2),
                hovertemplate=f"{x_title}=%{{x}}<br>fraction=%{{customdata:.4f}}<extra>{method}</extra>",
            )
        )

    # --- reference lines
    fig.add_hline(y=0, line_dash="solid", line_width=1.2, line_color="black")
    fig.add_hline(y=1, line_dash="dot", line_width=1, line_color="black")

    # --- y-axis ticks
    original_tick_vals = [-5, -4, -3, -2, -1, 0, 0.5, 1.0]
    scaled_tick_vals = [_scale_y(v) for v in original_tick_vals]
    tick_text = [str(v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in original_tick_vals]

    # --- layout
    fig.update_layout(
        title=dict(
            text="Fraction of the Gap Closed" + ("" if fid_groups is None else " (grouped)"),
            x=0.5, xanchor="center", y=0.95, yanchor="top"
        ),
        width=width,
        height=height,
        font=dict(family=font_family, size=16, color="black"),
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            x=0.02, xanchor="left",    # left side of plot
            y=0.02, yanchor="bottom",  # bottom side of plot
            bgcolor="rgba(255,255,255,0.6)",  # opaque white background
            bordercolor="black",
            borderwidth=1,
            font=dict(size=16, color="black"),
        ),
    )

    # --- x-axis (handle numeric fids or categorical group labels)
    if fid_groups is None:
        fig.update_xaxes(
            title=dict(text=x_title, standoff=12),
            tickmode="array",
            tickvals=x_order,
            range=[min(x_order) - 0.5, max(x_order) + 0.5],
            dtick=1,
            showline=True, linecolor="black", linewidth=1,
            showgrid=True, gridcolor="black", gridwidth=0.5,
            zeroline=False, color="black",
            tickfont=dict(size=16),
        )
    else:
        fig.update_xaxes(
            title=dict(text=x_title, standoff=12),
            type="category",
            categoryorder="array",
            categoryarray=x_order,
            showline=True, linecolor="black", linewidth=1,
            showgrid=True, gridcolor="black", gridwidth=0.5,
            zeroline=False, color="black",
            tickfont=dict(size=16),
        )

    # --- y-axis
    y_scaled_min = _scale_y(-5)
    y_scaled_max = 1.1
    fig.update_yaxes(
        title=dict(text="Fraction of the gap closed", standoff=16),
        range=[y_scaled_min, y_scaled_max],
        tickmode="array",
        tickvals=scaled_tick_vals,
        ticktext=tick_text,
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        color="black",
        tickfont=dict(size=16),
    )

    # Enforce legend order
    fig.update_traces(selector=dict(name="Dynamic selector"), legendrank=1)
    fig.update_traces(selector=dict(name="CMA-ES, non-elitist"), legendrank=2)
    fig.update_traces(selector=dict(name="B80"), legendrank=3)

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig, scores



def plot_gap_closed_bars(
    df: pd.DataFrame,
    budgets=None,
    selector_col: str = "selector_precision",
    vbs_col: str = "vbs_precisions",
    sbs_col: str = "sbs_precision",
    title: str = "Performance of Dynamic Selector vs. Baselines",
    font_family: str = "Latin Modern Roman",
    width: int = 880,
    height: int = 480,
    include_overall_selector: bool = True,
    save_pdf: str = "selector_bars.pdf",
    # Optional standalone sums to add extra bars
    # e.g. {"Non-elitist, B0": 2966.95, "BFGS (standalone)": 1.97e10, ...}
    standalone_sums: dict[str, float] | None = None,
):
    """
    Plot bars showing the fraction of the SBS–VBS gap closed for:
    - the dynamic selector (overall),
    - static methods at specified budgets,
    - optional standalone algorithms (passed via `standalone_sums`).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The bar chart figure.
    res : pandas.DataFrame
        Table including label `x`, `score` (fraction closed),
        `raw_sum` (underlying sum for that bar when available),
        and reference `sbs_sum`, `vbs_sum`.
    """
    # Disable MathJax for kaleido export (prevents some export hiccups)
    pio.kaleido.scope.mathjax = None

    # Shared sums
    vbs_sum = df[vbs_col].sum()
    sbs_sum = df[sbs_col].sum()
    den = sbs_sum - vbs_sum
    if den == 0:
        raise ValueError("Denominator (sum_sbs - sum_vbs) is zero; scores are undefined.")

    # Budgets to include -> map to column names
    if budgets is None:
        budgets = []
        for c in df.columns:
            if c.startswith("static_B"):
                try:
                    budgets.append(int(c.split("static_B", 1)[1]))
                except ValueError:
                    pass
        budgets = sorted(set(budgets))

    def score_for_col(col_name: str) -> float:
        msum = df[col_name].sum()
        return (sbs_sum - msum) / den

    # We will assemble rows (for the results table) and parallel x/y lists for plotting
    rows: list[dict] = []
    xs: list[str] = []
    ys: list[float] = []

    def add_row(name: str, score_val: float, raw_sum_val):
        rows.append(
            {
                "x": name,
                "score": float(score_val),
                "raw_sum": None if raw_sum_val is None else float(raw_sum_val),
                "sbs_sum": float(sbs_sum),
                "vbs_sum": float(vbs_sum),
            }
        )
        xs.append(name)
        ys.append(score_val)

    # Overall selector
    if include_overall_selector:
        sel_score = score_for_col(selector_col)
        add_row("Dynamic selector", sel_score, df[selector_col].sum())

    # Per-budget static bars
    for b in budgets:
        col = f"static_B{int(b)}"
        if col not in df.columns:
            continue
        label = f"B{int(b)}"
        val = score_for_col(col)
        add_row(label, val, df[col].sum())

    # Standalone-config bars (e.g., Non-elitist, DE standalone, etc.)
    if standalone_sums:
        for name, method_sum in standalone_sums.items():
            val = (sbs_sum - float(method_sum)) / den
            add_row(name, val, float(method_sum))

    # Dummy bar: fixed value 0.0 (kept for visual reference)
    add_row("Non-elitist, B16", 0.0, None)

    # Sort by score (desc) so plot and table align
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    xs_sorted = [r["x"] for r in rows_sorted]
    ys_sorted = [r["score"] for r in rows_sorted]

    # Plot
    fig = go.Figure(
        data=[
            go.Bar(
                x=xs_sorted,
                y=ys_sorted,
                marker=dict(color="royalblue", line=dict(color="black", width=1.2)),
                text=[f"{v:.3f}" for v in ys_sorted],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Fraction closed: %{y:.6f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="rgb(230,230,230)",
        font=dict(family=font_family, size=16, color="black"),
        margin=dict(l=12, r=12, t=50, b=12),
        xaxis=dict(
            title="Method",
            showline=True,
            linecolor="black",
            linewidth=1.5,
            showgrid=False,
            ticks="outside",
            tickcolor="black",
            tickfont=dict(size=16, color="black"),
        ),
        yaxis=dict(
            title="Fraction of the gap closed",
            range=[-0.15, 1],
            showline=True,
            linecolor="black",
            linewidth=1.5,
            showgrid=False,
            ticks="outside",
            tickcolor="black",
            tickfont=dict(size=16, color="black"),
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.5,
        ),
        title_x=0.5,
    )
    fig.update_traces(textfont=dict(size=16, color="black"))

    # Save PDF
    if save_pdf:
        fig.write_image(save_pdf, format="pdf")

    # Results table with raw sums included
    res = pd.DataFrame(rows_sorted, columns=["x", "score", "raw_sum", "sbs_sum", "vbs_sum"])
    return fig, res

def plot_log10_boxplots(
    df: pd.DataFrame,
    selector_col: str = "selector_precision",
    algorithm_cols: list[str] | None = None,
    static_prefix: str = "static_",
    static_budgets: list[int] | None = None,
    title: str = "Distributions (log10) of Selector and Static Methods",
    font_family: str = "Latin Modern Roman",
    width: int = 1100,
    height: int = 520,
    save_pdf: str | None = "log10_boxplots.pdf",
    zero_handling: str = "drop",
    epsilon: float = 1e-12,
    box_width: float = 0.6,   # NEW: width of boxes (default wider than Plotly’s default ~0.3)
):
    """
    Create one figure with a boxplot per column (selector, algorithms, selected static_*),
    showing the distribution in log10 scale.
    """

    pio.kaleido.scope.mathjax = None

    cols = []
    if selector_col in df.columns:
        cols.append(selector_col)

    if algorithm_cols is None:
        candidates = {"BFGS", "DE", "MLSL", "Non-elitist", "PSO", "Same"}
        algorithm_cols = [c for c in df.columns if c in candidates]
    cols.extend(algorithm_cols)

    if static_budgets is None:
        static_cols = [c for c in df.columns if c.startswith(static_prefix)]
    else:
        static_cols = []
        for b in sorted(set(int(b) for b in static_budgets)):
            col = f"{static_prefix}B{b}"
            if col in df.columns:
                static_cols.append(col)
    cols.extend(static_cols)

    seen, ordered_cols = set(), []
    for c in cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            ordered_cols.append(c)

    long = df[ordered_cols].melt(var_name="method", value_name="value")

    if zero_handling == "offset":
        long["log10_value"] = np.log10(long["value"].astype(float) + epsilon)
    else:
        mask = long["value"].astype(float) > 0
        long = long.loc[mask].copy()
        long["log10_value"] = np.log10(long["value"].astype(float))

    def pretty_name(x: str) -> str:
        if x == selector_col:
            return "Dynamic selector"
        if x.startswith(static_prefix):
            return x.replace(static_prefix, "").upper()
        return x

    long["label"] = long["method"].map(pretty_name)
    ordered_labels = [pretty_name(c) for c in ordered_cols]

    fig = go.Figure()
    for lab in ordered_labels:
        sub = long.loc[long["label"] == lab, "log10_value"]
        fig.add_trace(
            go.Box(
                y=sub,
                name=lab,
                fillcolor="royalblue",
                width=box_width,  # wider boxes
                marker=dict(color="royalblue", line=dict(color="black", width=1.2)),
                line=dict(color="black", width=1.2),
                boxmean=True,
                hovertemplate="<b>%{x}</b><br>log10(value): %{y:.6f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=TABLE_BG,
        font=dict(family=font_family, size=16, color="black"),
        margin=dict(l=12, r=12, t=50, b=12),
        xaxis=dict(
            title="Method",
            showline=True,
            linecolor="black",
            linewidth=1.5,
            showgrid=False,
            ticks="outside",
            tickcolor="black",
            tickfont=dict(size=16, color="black"),
            categoryorder="array",
            categoryarray=ordered_labels,
        ),
        yaxis=dict(
            title="log10(precision)",
            showline=True,
            linecolor="black",
            linewidth=1.5,
            showgrid=False,
            ticks="outside",
            tickcolor="black",
            tickfont=dict(size=16, color="black"),
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1.5,
        ),
        title_x=0.5,
        boxmode="group",
        showlegend=False,   # disable legend
    )

    if save_pdf:
        fig.write_image(save_pdf, format="pdf")
    return fig

def permutation_test_selector_vs_static(df):

    selector = df['selector_precision'].values
    budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21) ]

    cols = [f'static_B{b}' for b in budgets if f'static_B{b}' in df.columns]
    cols += ["sbs_precision", "BFGS", "DE", "PSO", "MLSL", "Same", "Non-elitist"]

    p_values = {}

    # budgets = [80]
    for col in cols:
        if col not in df.columns:
            continue
        static = df[col].values

        # Compute observed mean difference
        observed_diff = np.mean(selector - static)
        print(f"Observed mean difference: {observed_diff:.6f}")

        # Use scipy's permutation_test
        res = permutation_test(
            (selector, static),
            statistic=lambda x, y: np.mean(x - y),
            permutation_type='samples',
            vectorized=False,
            n_resamples=10000,
            alternative='less',
            random_state=42
        )

        p_values[col] = res.pvalue

        print(f"P-value (selector < static) for {col}: {res.pvalue:.6f}")

    return p_values


def plot_switch_budget_with_algo_bg(
    df: pd.DataFrame,
    save_pdf: str | None = None,
    width: int = 880,
    height: int = 480,
    font_family: str = "Latin Modern Roman",
    algo_colors: dict[str, str] | None = None,
    bar_opacity: float = 1.0,     # transparency of the background algorithm shares
    bar_width: float = 0.86,      # width of the background bands per fid
    max_budget_y: int = 550,      # cap of the primary y-axis
):
    """
    Expects columns: fid, selector_switch_budget, chosen_algorithm

    - Plots mean switching budget per fid (0..max_budget_y) with ±1 std grey band, IN FRONT of backgrounds.
    - Background: stacked proportions of chosen_algorithm per fid, drawn as shapes with layer="below".
    - Legend is placed OUTSIDE the plotting area (to the right).
    """

    if algo_colors is None:
        algo_colors = {
            "BFGS": "#1f77b4",
            "Non-elitist": "#ff7f0e",
            "DE": "#2ca02c",
            "PSO": "#d62728",
            "MLSL": "#9467bd",
            "Same": "#8c564b",
        }

    # --- prep
    df = df.copy()
    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)

    # Per-fid stats for selector_switch_budget
    budget_stats = (
        df.groupby("fid", as_index=False)["selector_switch_budget"]
          .agg(mean="mean", std="std", count="count")
          .sort_values("fid")
    )
    budget_stats["std"] = budget_stats["std"].fillna(0.0)

    y_min, y_max = 0, int(max_budget_y)
    mean_y = budget_stats["mean"].clip(y_min, y_max).to_numpy()
    std_y = budget_stats["std"].to_numpy()
    lower = np.clip(mean_y - std_y, y_min, y_max)
    upper = np.clip(mean_y + std_y, y_min, y_max)
    fids = budget_stats["fid"].to_list()

    # Per-fid algorithm proportions
    algo_counts = (
        df.groupby(["fid", "chosen_algorithm"]).size().rename("n").reset_index()
    )
    totals = algo_counts.groupby("fid")["n"].transform("sum")
    algo_counts["prop"] = algo_counts["n"] / totals

    algos_in_data = sorted(set(algo_colors).intersection(set(df["chosen_algorithm"].unique())))
    prop_wide = (
        algo_counts.pivot(index="fid", columns="chosen_algorithm", values="prop")
        .reindex(index=fids, columns=algos_in_data)
        .fillna(0.0)
    )

    # --- figure (secondary y kept for consistency, though shapes use yref='paper')
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Background via SHAPES (always behind traces)
    # For each fid, draw stacked vertical rectangles from cumulative proportions.
    cums = np.zeros(len(fids), dtype=float)
    for algo in algos_in_data:
        props = prop_wide[algo].to_numpy(dtype=float)
        y0 = cums
        y1 = cums + props
        for i, x in enumerate(fids):
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",                      # proportions in [0,1] on full plot height
                x0=x - bar_width / 2,
                x1=x + bar_width / 2,
                y0=float(y0[i]),
                y1=float(y1[i]),
                line=dict(width=0),
                fillcolor=algo_colors.get(algo, "#000000"),
                opacity=bar_opacity,
                layer="below",                      # <-- ensures background
            )
        cums = y1

    # Legend entries for algorithms (legend-only traces placed off-plot)
    # Visible in legend, won’t affect view ranges (we set fixed ranges).
    for algo in algos_in_data:
        fig.add_trace(
            go.Scatter(
                x=[fids[0]],
                y=[y_max + 10],                    # outside y-range
                mode="markers",
                marker=dict(symbol="square", size=12, color=algo_colors[algo]),
                name=algo,
                hoverinfo="skip",
                showlegend=True,
            ),
            secondary_y=False,
        )

    # Grey variance band (±1σ)
    fig.add_trace(
        go.Scatter(
            x=fids + fids[::-1],
            y=upper.tolist() + lower[::-1].tolist(),
            fill="toself",
            fillcolor="rgba(128,128,128,0.6)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="±1σ",
            showlegend=True,
        ),
        secondary_y=False,
    )

    # Black mean line
    fig.add_trace(
        go.Scatter(
            x=fids,
            y=mean_y,
            mode="lines+markers",
            name="Mean switch budget",
            marker=dict(size=7, color="black", line=dict(width=0.5, color="black")),
            line=dict(width=2, color="black"),
            hovertemplate="fid=%{x}<br>mean=%{y:.2f}<extra>Mean switch budget</extra>",
            showlegend=True,
        ),
        secondary_y=False,
    )

    # --- layout (legend outside)
    fig.update_layout(
        title=dict(text="Average Switching Budget by fid", x=0.5, xanchor="center", y=0.95, yanchor="top"),
        width=width,
        height=height,
        font=dict(family=font_family, size=16, color="black"),
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        margin=dict(l=60, r=220, t=60, b=50),   # extra right margin for outside legend
        barmode="stack",
        bargap=0.1,
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.00, yanchor="top",
            font=dict(size=16, color="black"),
            traceorder="normal",
        ),
    )

    # x-axis
    fig.update_xaxes(
        title=dict(text="fid", standoff=12),
        tickmode="array",
        tickvals=fids,
        dtick=1,
        range=[min(fids) - 0.5, max(fids) + 0.5],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=False, color="black",
        tickfont=dict(size=16),
    )

    # primary y (0..max_budget_y)
    fig.update_yaxes(
        title=dict(text="Switching budget", standoff=16),
        range=[y_min, y_max],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        color="black",
        tickfont=dict(size=16),
        secondary_y=False,
    )

    # secondary y (unused now, keep hidden)
    fig.update_yaxes(
        range=[0, 1],
        showticklabels=False,
        showgrid=False,
        showline=False,
        zeroline=False,
        secondary_y=True,
    )

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig, budget_stats, prop_wide

def plot_algo_distribution_by_fid(
    df: pd.DataFrame,
    save_pdf: str | None = None,
    width: int = 880,
    height: int = 480,
    font_family: str = "Latin Modern Roman",
    # algorithm -> color
    algo_colors: dict[str, str] | None = None,
    bar_opacity: float = 0.8,
    bar_width: float = 0.86,
    title: str = "Algorithm Distribution by fid",
    legend_position: str = "top",          # "top" or "bottom"
):
    """
    Stacked bar chart of algorithm proportions per fid.

    Expects columns: fid, chosen_algorithm

    Legend labels are renamed for CMA-ES variants:
      - "Same"         -> "CMA-ES, elitist"
      - "Non-elitist"  -> "CMA-ES, non-elitist"
    Legend is horizontal and can be placed above or below the plot.
    """
    pio.kaleido.scope.mathjax = None
    # default palette
    if algo_colors is None:
        algo_colors = {
            "BFGS": "#1f77b4",
            "Non-elitist": "#ff7f0e",  # CMA-ES non-elitist
            "DE": "#2ca02c",
            "PSO": "#d62728",
            "MLSL": "#9467bd",
            "Same": "#8c564b",        # CMA-ES elitist
        }

    # mapping for display names in legend (data keys stay as-is)
    display_name = {
        "Same": "CMA-ES, elitist",
        "Non-elitist": "CMA-ES, non-elitist",
    }

    # prep
    df = df.copy()
    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)

    # proportions per fid
    counts = df.groupby(["fid", "chosen_algorithm"]).size().rename("n").reset_index()
    totals = counts.groupby("fid")["n"].transform("sum")
    counts["prop"] = counts["n"] / totals

    # order: use color dict order but only keep present algos
    present = set(df["chosen_algorithm"].unique())
    algos_in_data = [a for a in algo_colors.keys() if a in present]

    fids = sorted(counts["fid"].unique().tolist())
    wide = (
        counts.pivot(index="fid", columns="chosen_algorithm", values="prop")
        .reindex(index=fids, columns=algos_in_data)
        .fillna(0.0)
    )

    # build stacked bars
    fig = go.Figure()
    for algo in algos_in_data:
        fig.add_trace(
            go.Bar(
                x=fids,
                y=wide[algo].to_list(),
                name=display_name.get(algo, algo),  # legend label
                width=bar_width,
                marker=dict(color=algo_colors[algo]),
                opacity=bar_opacity,
                hovertemplate=f"fid=%{{x}}<br>{display_name.get(algo, algo)} share=%{{y:.2f}}<extra></extra>",
            )
        )

    # legend placement
    if legend_position.lower() == "bottom":
        legend_y = -0.1
        margins = dict(l=60, r=40, t=60, b=100)
    else:  # "top"
        legend_y = 1.04
        margins = dict(l=60, r=40, t=100, b=60)

    # layout / style
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.95, yanchor="top"),
        width=width, height=height,
        font=dict(family=font_family, size=16, color="black"),
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        margin=margins,
        barmode="stack",
        bargap=0.1,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=legend_y, yanchor="bottom" if legend_y >= 1 else "top",
            font=dict(size=16, color="black"),
            traceorder="normal",
        ),
    )

    # axes
    fig.update_xaxes(
        title=dict(text="fid", standoff=12),
        tickmode="array",
        tickvals=fids,
        dtick=1,
        range=[min(fids) - 0.5, max(fids) + 0.5],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=False, color="black",
        tickfont=dict(size=16),
    )

    fig.update_yaxes(
        title=dict(text="Proportion", standoff=16),
        range=[0, 1],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        color="black",
        tickfont=dict(size=16),
    )

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig, wide

def plot_switch_budget_by_fid(
    df: pd.DataFrame,
    save_pdf: str | None = None,
    width: int = 880,
    height: int = 480,
    font_family: str = "Latin Modern Roman",
    max_budget_y: int = 550,
    band_opacity: float = 0.4,
    title: str = "Average Switching Budget by fid",
):
    """
    Mean switching budget per fid with ±1σ RoyalBlue band.
    Expects columns: fid, selector_switch_budget
    """
    pio.kaleido.scope.mathjax = None
    df = df.copy()
    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)

    stats = (
        df.groupby("fid", as_index=False)["selector_switch_budget"]
          .agg(mean="mean", std="std", count="count")
          .sort_values("fid")
    )
    stats["std"] = stats["std"].fillna(0.0)

    y_min, y_max = 0, int(max_budget_y)
    fids = stats["fid"].tolist()
    mean_y = stats["mean"].clip(y_min, y_max).to_numpy()
    std_y = stats["std"].to_numpy()
    lower = np.clip(mean_y - std_y, y_min, y_max)
    upper = np.clip(mean_y + std_y, y_min, y_max)

    fig = go.Figure()

    # RoyalBlue variance band
    fig.add_trace(
        go.Scatter(
            x=fids + fids[::-1],
            y=upper.tolist() + lower[::-1].tolist(),
            fill="toself",
            fillcolor=f"rgba(65,105,225,{band_opacity})",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="±1σ",
            showlegend=True,
        )
    )

    # RoyalBlue mean line
    fig.add_trace(
        go.Scatter(
            x=fids,
            y=mean_y,
            mode="lines+markers",
            name="Mean switch budget",
            marker=dict(size=7, color="royalblue", line=dict(width=0.7, color="black")),
            line=dict(width=2, color="royalblue"),
            hovertemplate="fid=%{x}<br>mean=%{y:.2f}<extra>Mean switch budget</extra>",
            showlegend=True,
        )
    )

    # Layout with legend top-left inside
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.95, yanchor="top"),
        width=width, height=height,
        font=dict(family=font_family, size=16, color="black"),
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(
            orientation="v",
            x=0.02, xanchor="left",
            y=0.98, yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",  # opaque white background
            bordercolor="black",
            borderwidth=1,
            font=dict(size=16, color="black"),
        ),
    )

    fig.update_xaxes(
        title=dict(text="fid", standoff=12),
        tickmode="array",
        tickvals=fids,
        dtick=1,
        range=[min(fids) - 0.5, max(fids) + 0.5],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=False, color="black",
        tickfont=dict(size=16),
    )

    fig.update_yaxes(
        title=dict(text="Switching budget", standoff=16),
        range=[y_min, y_max],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        color="black",
        tickfont=dict(size=16),
    )

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig, stats

def plot_selector_dashboard(
    df: pd.DataFrame,
    df_algos: pd.DataFrame,
    save_pdf: str | None = "selector_dashboard.pdf",
    width: int = 980,
    height: int = 980,
    font_family: str = "Latin Modern Roman",
    # relative heights for [top, middle, bottom]
    row_heights: tuple[float, float, float] = (0.25, 0.33, 0.33),
    vertical_spacing: float = 0.04,
):
    """
    One figure with 3 vertically stacked subplots that share the x-axis (fid):
      1) Fraction of the gap closed (lines+markers), y<0 compressed by 0.5
      2) Algorithm distribution by fid (stacked bars)
      3) Switching budget by fid (mean line + ±1σ band)

    Args:
      df:        dataframe with columns ['fid','vbs_precisions','sbs_precision','selector_precision','static_B80',...]
      df_algos:  dataframe with columns ['fid','chosen_algorithm', ...]
    """
    pio.kaleido.scope.mathjax = None

    # ---------- Top subplot: fraction of the gap closed ----------
    methods = ["selector_precision", "static_B80"]

    def _scale_y(val: float) -> float:
        if pd.isna(val): return val
        return val if val >= 0 else 0.5 * val

    recs = []
    for fid, sub in df.groupby("fid", dropna=False):
        vbs_sum = sub["vbs_precisions"].sum()
        sbs_sum = sub["sbs_precision"].sum()
        den = sbs_sum - vbs_sum
        for col in methods:
            if col not in sub.columns: 
                continue
            msum  = sub[col].sum()
            num   = sbs_sum - msum
            score = 1.0 if num == den else (0.0 if den == 0 else num/den)
            name  = "Dynamic selector" if col == "selector_precision" else ("B80" if col == "static_B80" else col)
            recs.append({"fid": int(fid), "method": name, "fraction": float(score)})

    top_df = pd.DataFrame(recs).sort_values(["method","fid"])
    top_df["fraction_scaled"] = top_df["fraction"].map(_scale_y)

    # preserve full fid range on x
    fids = sorted(df["fid"].dropna().astype(int).unique().tolist())

    # ---------- Middle subplot: algorithm distribution (stacked bars) ----------
    algo_colors = {
        "BFGS": "#1f77b4",
        "Non-elitist": "#ff7f0e",  # CMA-ES non-elitist
        "DE": "#2ca02c",
        "PSO": "#d62728",
        "MLSL": "#9467bd",
        "Same": "#8c564b",        # CMA-ES elitist
    }
    display_name = {"Same": "CMA-ES, elitist", "Non-elitist": "CMA-ES, non-elitist"}

    df_alg = df_algos.copy()
    df_alg["fid"] = pd.to_numeric(df_alg["fid"], errors="coerce").astype(int)

    counts = df_alg.groupby(["fid","chosen_algorithm"]).size().rename("n").reset_index()
    totals = counts.groupby("fid")["n"].transform("sum")
    counts["prop"] = counts["n"] / totals

    algos_in_data = [a for a in algo_colors if a in set(counts["chosen_algorithm"].unique())]
    mid_wide = (
        counts.pivot(index="fid", columns="chosen_algorithm", values="prop")
        .reindex(index=fids, columns=algos_in_data)
        .fillna(0.0)
    )

    # ---------- Bottom subplot: switching budget (mean + ±1σ) ----------
    stats = (
        df.groupby("fid", as_index=False)["selector_switch_budget"]
          .agg(mean="mean", std="std")
          .sort_values("fid")
    )
    stats["fid"] = stats["fid"].astype(int)
    stats = stats.set_index("fid").reindex(fids).reset_index()
    stats["std"] = stats["std"].fillna(0.0)

    mean_y = stats["mean"].to_numpy(dtype=float)
    std_y  = stats["std"].to_numpy(dtype=float)
    lower  = (mean_y - std_y).tolist()
    upper  = (mean_y + std_y).tolist()

    # ---------- Build combined figure ----------
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing,
        row_heights=list(row_heights),
        specs=[[{"type":"xy"}],[{"type":"bar"}],[{"type":"xy"}]],
    )

    # Top: traces
    for method, subm in top_df.groupby("method", sort=False):
        # ensure x order
        subm = subm.set_index("fid").reindex(fids).reset_index()
        fig.add_trace(
            go.Scatter(
                x=subm["fid"],
                y=subm["fraction_scaled"],
                customdata=subm["fraction"],
                mode="lines+markers",
                name=method,
                marker=dict(size=7, line=dict(width=0.5, color="black")),
                line=dict(width=2),
                hovertemplate="fid=%{x}<br>fraction=%{customdata:.4f}<extra>"+method+"</extra>",
            ),
            row=1, col=1
        )

    # Reference lines for top
    fig.add_hline(y=0, line_dash="solid", line_width=1.2, line_color="black", row=1, col=1)
    fig.add_hline(y=1, line_dash="dot",   line_width=1.0, line_color="black", row=1, col=1)

    # Top axis styling (compress negatives)
    top_tick_orig  = [-5,-4,-3,-2,-1,0,0.5,1.0]
    top_tick_scaled= [_scale_y(v) for v in top_tick_orig]
    top_tick_text  = [str(v).rstrip("0").rstrip(".") if isinstance(v,float) else str(v) for v in top_tick_orig]

    fig.update_yaxes(
        title_text="Fraction of the gap closed",
        tickmode="array", tickvals=top_tick_scaled, ticktext=top_tick_text,
        range=[_scale_y(-5), 1.1],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        row=1, col=1
    )

    # Middle: stacked bars
    for algo in algos_in_data:
        fig.add_trace(
            go.Bar(
                x=fids, y=mid_wide[algo].tolist(),
                name=display_name.get(algo, algo),
                marker=dict(color=algo_colors[algo]),
                width=0.86,
                opacity=0.9,
                hovertemplate=f"fid=%{{x}}<br>{display_name.get(algo, algo)} share=%{{y:.2f}}<extra></extra>",
            ),
            row=2, col=1
        )
    fig.update_yaxes(
        title_text="Proportion",
        range=[0,1],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        row=2, col=1
    )

    # Bottom: ±1σ band + mean line
    fig.add_trace(
        go.Scatter(
            x=fids + fids[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(65,105,225,0.35)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="±1σ",
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=fids, y=mean_y,
            mode="lines+markers",
            name="Mean switch budget",
            marker=dict(size=7, color="royalblue", line=dict(width=0.7, color="black")),
            line=dict(width=2, color="royalblue"),
            hovertemplate="fid=%{x}<br>mean=%{y:.2f}<extra>Mean switch budget</extra>",
        ),
        row=3, col=1
    )
    fig.update_yaxes(
        title_text="Switching budget",
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        row=3, col=1
    )

    # Shared x-axis (only show labels at the bottom row)
        # --- SHARED X (set full config for each row so vertical grid lines show everywhere)
        # --- SHARED X (ticks 1..24 everywhere; axis title only at the bottom)
    common_x = dict(
        tickmode="array", tickvals=fids, dtick=1,
        range=[min(fids)-0.5, max(fids)+0.5],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=False,
    )

    # Row 1: ticks visible, no axis title
    fig.update_xaxes(row=1, col=1, **{**common_x, "showticklabels": True, "title_text": None})

    # Row 2: ticks visible, no axis title
    fig.update_xaxes(row=2, col=1, **{**common_x, "showticklabels": True, "title_text": None})

    # Row 3: ticks visible, axis title shown
    fig.update_xaxes(row=3, col=1, **common_x, title_text="fid", title_standoff=12, showticklabels=True)


    # Hide x ticklabels on rows 1 and 2 to reduce clutter
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    # Global layout
    fig.update_layout(
        title=dict(text="Selector Summary (aligned subplots)", x=0.5, xanchor="center"),
        width=width, height=height,
        font=dict(family=font_family, size=16, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="rgb(230,230,230)",
        margin=dict(l=70, r=30, t=60, b=60),
        barmode="stack",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.07, yanchor="bottom",
                    font=dict(size=16)),
    )

    # Keep a consistent legend order
    fig.update_traces(selector=dict(name="Dynamic selector"), legendrank=1)
    fig.update_traces(selector=dict(name="B80"), legendrank=3)
    # CMA-ES variants get meaningful names already via display_name

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig

if __name__ == "__main__":
    df = pd.read_csv("data/selector_old_tuning_200_200_all_information.csv")
    df_algos = pd.read_csv("data/selector_old_tuning_200_200_with_algos.csv")
    # _,_ = plot_gap_closed_by_fid(df, 
    #                             fid_groups={"1": [1,2,3,4,5], "2": [6,7,8,9], "3": [10,11,12,13,14], "4": [15,16,17,18,19], "5": [20,21,22,23,24]}, 
    #                             save_pdf="gap_closed_by_fid_hlc.pdf")
    # p_values = permutation_test_selector_vs_static(df)
    # print(p_values)
    # _,_ = plot_gap_closed_bars(df, budgets = [80, 150, 8, 800], save_pdf="selector_bars.pdf",
    #                            standalone_sums={"Non-elitist, B0": nonelit_sum})
    # df = pd.read_csv("your_file.csv")

    # fig1, props = plot_algo_distribution_by_fid(df, save_pdf="algo_distribution_by_fid.pdf")
    # fig2, stats = plot_switch_budget_by_fid(df, save_pdf="switch_budget_by_fid.pdf")

    # fig1.show(); fig2.show()
    # plot_gap_closed_by_fid(df, save_pdf="gap_closed_by_fid.pdf")
    # plot_algo_distribution_by_fid(df_algos, save_pdf="algo_distribution_by_fid.pdf")
    # plot_switch_budget_by_fid(df, save_pdf="switch_budget_by_fid.pdf")
    plot_selector_dashboard(df, df_algos, save_pdf="selector_dashboard.pdf")