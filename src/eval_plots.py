"""
Plotting results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# takes in a dictionary of metrics to values, as created by src.dataset_utils.extract_metric_dicts 
# (which in turn runs on output of src.evaluation.attach_last_token_probs["dataset_with_evals"])
def extract_metric_pairs(metric_dicts, metric: str | None = None, raw = True):
    """
    Extract (wmc, entropy) arrays for a specific metric.
    
    metric examples:
        'xor'
        'and_not'
        'implication'
        'transitivity'
    """
    if metric is None:
        if raw:
            wmc_key = "wmc_raw" # this makes more sense for this plot
        else:
            wmc_key = "wmc_score"
        entropy_key = "entropy"
    else:
        if raw:
            wmc_key = f"{metric}_wmc_raw"
        else:
            wmc_key = f"{metric}_wmc_score"
        entropy_key = f"{metric}_entropy"

    wmc_vals = []
    entropy_vals = []

    for d in metric_dicts:
        if wmc_key in d and entropy_key in d:
            wmc_vals.append(d[wmc_key])
            entropy_vals.append(d[entropy_key])

    return np.array(wmc_vals), np.array(entropy_vals)

def plot_entropy_vs_wmc(metric_dicts, title=None, save_path=None, metric=None, alpha=0.4):
    """
    Scatter plot of entropy vs WMC.
    """

    wmc, entropy = extract_metric_pairs(metric_dicts, metric)

    plt.figure(figsize=(6,5))
    plt.scatter(wmc, entropy, alpha=alpha)

    plt.xlabel("WMC")
    plt.ylabel("Entropy")
    if title is None:
        title = "Entropy vs WMC"

    plt.title(title)

    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

    
def plot_entropy_histogram_by_wmc(metric_dicts, title=None, save_path=None, metric=None, bins=20):
    """
    Histogram-style visualization:
    WMC bins vs entropy distribution.
    """

    wmc, entropy = extract_metric_pairs(metric_dicts, metric)

    bin_edges = np.linspace(0, 1, bins + 1)
    digitized = np.digitize(wmc, bin_edges)

    bin_entropy = [[] for _ in range(bins)]

    for i, e in zip(digitized, entropy):
        if 1 <= i <= bins:
            bin_entropy[i-1].append(e)

    mean_entropy = [
        np.mean(b) if len(b) > 0 else np.nan
        for b in bin_entropy
    ]

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(6,5))
    plt.bar(centers, mean_entropy, width=1/bins)

    plt.xlabel("WMC")
    plt.ylabel("Mean Entropy")
    if title is None:
        title = "Entropy by WMC bucket"

    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_wmc_score_histogram(metric_dicts, title=None, save_path=None, metric=None, raw=False, discrete=False, bins=20):
    """
    Histogram of per-example WMC score distribution (count vs WMC score).

    Args:
        metric_dicts: list of metric dicts from extract_metric_dicts
        title: plot title
        save_path: if provided, save as PDF; otherwise show
        metric: sub-metric prefix (e.g. "and_not", "joint"); None uses primary WMC
        raw: if True, use wmc_raw (range [0,1]); if False, use wmc_score (baseline-normalized)
        discrete: if True, use discrete binary scores (overrides raw flag)
        bins: number of histogram bins (forced to 2 when discrete=True)
    """
    if discrete:
        wmc_key = f"{metric}_discrete" if metric is not None else "discrete"
        xlabel = f"{metric} Discrete Score" if metric is not None else "Discrete Score"
        bins = 2
    elif raw:
        wmc_key = f"{metric}_wmc_raw" if metric is not None else "wmc_raw"
        xlabel = f"{metric} WMC (raw)" if metric is not None else "WMC (raw)"
    else:
        wmc_key = f"{metric}_wmc_score" if metric is not None else "wmc_score"
        xlabel = f"{metric} WMC Score" if metric is not None else "WMC Score (baseline-normalized)"

    wmc_vals = np.array([d[wmc_key] for d in metric_dicts if wmc_key in d])

    plt.figure(figsize=(6, 4))
    if discrete:
        plt.hist(wmc_vals, bins=bins, range=(-0.5, 1.5), edgecolor="black", linewidth=0.4)
        plt.xticks([0, 1])
        plt.xlim(-0.5, 1.5)
    elif raw:
        plt.hist(wmc_vals, bins=bins, range=(0, 1), edgecolor="black", linewidth=0.4)
        plt.xlim(0, 1)
    else:
        plt.hist(wmc_vals, bins=bins, edgecolor="black", linewidth=0.4)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    if title is None:
        title = f"{xlabel} Distribution"
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_calibrated_prob_histogram(probs, method_name, title=None, save_path=None, bins=25):
    """
    Histogram of calibrated probabilities for a single calibration method.

    Args:
        probs: list/array of float probabilities in [0, 1]
        method_name: string label for the x-axis / title
        title: plot title (defaults to method_name)
        save_path: if provided, save as PDF here; otherwise show
        bins: number of histogram bins
    """
    plt.figure(figsize=(6, 4))
    plt.hist(probs, bins=bins, range=(0, 1), edgecolor="black", linewidth=0.4)
    plt.xlabel("Calibrated probability")
    plt.ylabel("Count")
    plt.title(title if title is not None else method_name)
    plt.xlim(0, 1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_nucleus_entropy_sweep(nucleus_ps, means, stds, title=None, save_path=None):
    """
    Line plot of mean calibrated probability (cal_nucleus_entropyXXX) vs nucleus p,
    with ±1 std error bars.

    Args:
        nucleus_ps: list/array of p values (x-axis)
        means: array of mean calibrated probs per p
        stds: array of std calibrated probs per p
        title: plot title
        save_path: if provided, save as PDF here; otherwise show
    """
    plt.figure(figsize=(6, 4))
    plt.errorbar(nucleus_ps, means, yerr=stds, fmt="-o", capsize=4, linewidth=1.5, markersize=4)
    plt.xlabel("Nucleus p")
    plt.ylabel("Mean calibrated probability")
    plt.ylim(0, 1)
    plt.title(title if title is not None else "Nucleus entropy calibration sweep")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def extract_model_family(model_name):
    """
    Return a family label that distinguishes IT/instruct from PT/base models.
    e.g. "Gemma-3-1B-it" -> "Gemma-it", "Llama-3.2-1B-Instruct" -> "Llama-Instruct",
         "Gemma-3-1B-pt" -> "Gemma-pt", "Llama-3.2-1B" -> "Llama".
    """
    base = model_name.split("-")[0]
    tokens = model_name.split("-")
    last = tokens[-1].lower()
    if last == "it":
        return f"{base}-it"
    if last == "pt":
        return f"{base}-pt"
    if last == "instruct":
        return f"{base}-Instruct"
    return base


def classify_model_family(model_name, family_specs):
    """
    Assign model_name to the most-specific matching spec from family_specs.

    Family spec format: "Base" or "Base-Qualifier"
      - Base  : matched case-insensitively anywhere in the model name.
      - Qualifier (optional, after the first "-"): matched as the literal
        dash-prefixed token "-Qualifier" (case-insensitive) inside the name.
        e.g. "Gemma-it"      matches "Gemma-3-1B-it"   but not "Gemma-3-1B-pt"
             "Gemma-pt"      matches "Gemma-3-1B-pt"   but not "Gemma-3-1B-it"
             "Llama-Instruct" matches "Llama-3.2-1B-Instruct" but not "Llama-3.2-1B"
             "Llama"          matches any model starting with "Llama"

    When both a qualified and an unqualified spec would match the same model
    (e.g. ["Llama", "Llama-Instruct"]), the qualified spec wins so that the
    base-only spec acts as a catch-all for non-qualified models.

    Returns the matching spec string, or None if no spec matches.
    """
    name_lower = model_name.lower()

    qualified_matches = []
    base_matches = []

    for spec in family_specs:
        base, sep, qualifier = spec.partition("-")
        if base.lower() not in name_lower:
            continue
        if sep:  # spec has a qualifier
            if f"-{qualifier}".lower() in name_lower:
                qualified_matches.append(spec)
        else:
            base_matches.append(spec)

    if qualified_matches:
        return qualified_matches[0]
    if base_matches:
        # A base-only spec (e.g. "Llama") only matches models that have no
        # known tuning qualifier — i.e. extract_model_family returns the bare
        # base with no suffix.  This means "Llama" never catches Instruct
        # models even when "Llama-Instruct" is absent from the spec list.
        if extract_model_family(model_name) == base_matches[0]:
            return base_matches[0]
    return None


def human_readable_params(x):
    if x >= 1e9:
        return f"{int(x/1e9)}B"
    if x >= 1e6:
        return f"{int(x/1e6)}M"
    return str(int(x))


def plot_model_performance(
    df,
    metric_names=None,
    calibration_methods=None,
    constraints=None,
    model_families=None,
    models=None,
    log_x=True,
    label_points=False,
    combine_shape=True,
    title="Model Performance vs Model Size",
    ylabel="Performance (Mean)",
    save_path=None,
):

    plot_df = df.copy()

    # -------------------------
    # Filtering
    # -------------------------

    if metric_names is not None:
        plot_df = plot_df[plot_df["metric_name"].isin(metric_names)]

    if calibration_methods is not None:
        plot_df = plot_df[plot_df["calibration_method"].isin(calibration_methods)]

    if constraints is not None:
        plot_df = plot_df[plot_df["constraint"].isin(constraints)]

    if models is not None:
        plot_df = plot_df[plot_df["model"].isin(models)]

    # -------------------------
    # Model family assignment
    # -------------------------
    # When model_families is a list of specs (e.g. ["Gemma-it", "Gemma-pt",
    # "Llama-Instruct", "Llama"]), use classify_model_family so that IT and PT
    # variants of the same base are treated as distinct color groups.
    # When model_families is None, fall back to the simple first-token heuristic.

    if model_families is not None:
        plot_df["Model Family"] = plot_df["model"].apply(
            lambda m: classify_model_family(m, model_families)
        )
        # Drop rows that didn't match any spec
        plot_df = plot_df[plot_df["Model Family"].notna()]
    else:
        plot_df["Model Family"] = plot_df["model"].apply(extract_model_family)

    if len(plot_df) == 0:
        print("No data to plot")
        return

    # -------------------------
    # Detect varying variables
    # -------------------------

    varying = []
    for col in ["calibration_method", "constraint", "metric_name"]:
        if plot_df[col].nunique() > 1:
            varying.append(col)

    style_col = None

    if len(varying) == 0:
        style_col = None

    elif combine_shape and len(varying) > 1:

        style_col = "Calibration / Constraint"
        plot_df[style_col] = plot_df[varying].astype(str).agg(" | ".join, axis=1)

    else:

        var = varying[0]

        if var == "calibration_method":
            style_col = "Calibration Method"

        elif var == "constraint":
            style_col = "Constraint"

        else:
            style_col = "Metric"

        plot_df[style_col] = plot_df[var]

    # -------------------------
    # Plot
    # -------------------------

    plt.figure(figsize=(9,6))
    sns.set(style="whitegrid")

    # stable markers to prevent seaborn legend bugs
    marker_list = ["o","s","D","^","v","P","X","*"]
    markers = None

    if style_col is not None:
        unique_styles = sorted(plot_df[style_col].unique())
        markers = {v: marker_list[i % len(marker_list)] for i,v in enumerate(unique_styles)}

    ax = sns.scatterplot(
        data=plot_df,
        x="model_size_params",
        y="mean",
        hue="Model Family",
        style=style_col,
        markers=markers,
        s=120,
        legend="full"
    )

    # -------------------------
    # Log axis
    # -------------------------

    if log_x:

        ax.set_xscale("log")

        sizes = plot_df["model_size_params"]

        xmin = sizes.min() * 0.8
        xmax = sizes.max() * 1.2

        ax.set_xlim(xmin, xmax)

        ticks = np.array([
            1e8,3e8,
            1e9,3e9,
            1e10,3e10,
            1e11
        ])

        ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]

        ax.set_xticks(ticks)
        ax.set_xticklabels([human_readable_params(t) for t in ticks])

    # -------------------------
    # Labels
    # -------------------------

    ax.set_xlabel("Model Size (Parameters)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # -------------------------
    # Optional point labels
    # -------------------------

    if label_points:
        for _, r in plot_df.iterrows():

            ax.text(
                r["model_size_params"],
                r["mean"],
                r["model"],
                fontsize=8,
                alpha=0.7
            )

    # -------------------------
    # Legend fixes
    # -------------------------

    legend = ax.get_legend()

    if legend:
        legend.remove()

    families = sorted(plot_df["Model Family"].unique())
    palette = sns.color_palette(n_colors=len(families))

    family_handles = [
        plt.Line2D(
            [0],[0],
            marker="o",
            linestyle="",
            color=palette[i],
            label=families[i],
            markersize=8
        )
        for i in range(len(families))
    ]

    handles = family_handles
    labels = families

    # add style legend if needed
    if style_col is not None and plot_df[style_col].nunique() > 1:

        style_vals = sorted(plot_df[style_col].unique())

        style_handles = [
            plt.Line2D(
                [0],[0],
                marker=markers[v],
                linestyle="",
                color="black",
                label=v,
                markersize=8
            )
            for v in style_vals
        ]

        handles += style_handles
        labels += style_vals

        legend_title = f"Model Family / {style_col}"

    else:

        legend_title = "Model Family"

    ax.legend(
        handles,
        labels,
        title=legend_title,
        bbox_to_anchor=(1.02,1),
        loc="upper left"
    )

    if save_path is not None:
        parent_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.tight_layout()
    plt.show()
    plt.close()


def table_model_performance(
    df,
    calibration_methods=None,
    metric_names=None,
    col_specs=None,
    constraints=None,
    model_families=None,
    models=None,
    aggregate_constraints="mean",
    stat="mean",
    fmt=".3f",
    sort_by="model_size_params",
):
    """
    Build a summary table: one row per model, one column per (calibration, metric) pair.

    Parameters
    ----------
    df : DataFrame
        Output of build_master_dataframe — rows are (model, constraint,
        calibration_method, metric_name, mean, std, min, max).
    calibration_methods : list[str] | None
        Calibration methods to include.  None → all present in df.
    metric_names : list[str] | None
        Metric names to include.  None → all present in df.
    col_specs : list[tuple[str, str]] | None
        Explicit list of (calibration_method, metric_name) column pairs.
        When provided, *overrides* the cartesian product of
        calibration_methods × metric_names — useful when you want different
        metrics for different calibration methods.
    constraints : list[str] | None
        Constraints to include before aggregation.  None → all.
    model_families : list[str] | None
        Keep only models whose family prefix (before first "-") is in this list.
    models : list[str] | None
        Keep only these specific model names.
    aggregate_constraints : {"mean", "none"}
        "mean" → average over the selected constraints per (model, cal, metric).
        "none" → keep constraint as an additional row-level index.
    stat : {"mean", "std", "min", "max"}
        Which aggregate statistic to show in each cell.
    fmt : str
        Python format string for cell values, e.g. ".3f", ".2%".
    sort_by : str | None
        Column name to sort rows by ("model_size_params", "model", None).

    Returns
    -------
    pandas.DataFrame
        Wide-format table with a MultiIndex column (calibration, metric).
        Suitable for display in a notebook or conversion via .to_latex() /
        .to_csv().

    Examples
    --------
    # Default: mean wmc_score for all cals, averaged across constraints
    table_model_performance(master_df, metric_names=["wmc_score"])

    # Compare wmc_score vs no_vacuous_wmc_score for two cal methods
    table_model_performance(
        master_df,
        calibration_methods=["cal_vocab", "yes_no"],
        metric_names=["wmc_score", "no_vacuous_wmc_score"],
    )

    # Custom column pairs: different metrics per calibration method
    table_model_performance(
        master_df,
        col_specs=[
            ("cal_vocab",  "wmc_score"),
            ("cal_vocab",  "no_vacuous_wmc_score"),
            ("yes_no",     "wmc_score"),
            ("yes_no",     "wmc_score_no_ood"),
        ],
    )

    # Per-constraint rows (no aggregation)
    table_model_performance(
        master_df,
        metric_names=["discrete"],
        aggregate_constraints="none",
    )
    """
    tdf = df.copy()

    # ── Filtering ────────────────────────────────────────────────────────────

    if model_families is not None:
        tdf = tdf[tdf["model"].apply(extract_model_family).isin(model_families)]

    if models is not None:
        tdf = tdf[tdf["model"].isin(models)]

    if constraints is not None:
        tdf = tdf[tdf["constraint"].isin(constraints)]

    if col_specs is not None:
        # Build a boolean mask that keeps only the requested (cal, metric) pairs.
        mask = pd.Series(False, index=tdf.index)
        for cal, met in col_specs:
            mask |= (tdf["calibration_method"] == cal) & (tdf["metric_name"] == met)
        tdf = tdf[mask]
    else:
        if calibration_methods is not None:
            tdf = tdf[tdf["calibration_method"].isin(calibration_methods)]
        if metric_names is not None:
            tdf = tdf[tdf["metric_name"].isin(metric_names)]

    if tdf.empty:
        print("No data matched the given filters.")
        return pd.DataFrame()

    # ── Determine row index ──────────────────────────────────────────────────

    if aggregate_constraints == "mean":
        row_keys = ["model", "model_size_params", "calibration_method", "metric_name"]
        tdf = (
            tdf.groupby(row_keys, sort=False)[stat]
            .mean()
            .reset_index()
        )
        pivot_index = ["model", "model_size_params"]
    else:
        # keep constraint as part of the row identity
        pivot_index = ["model", "model_size_params", "constraint"]

    # ── Pivot to wide format ─────────────────────────────────────────────────

    table = tdf.pivot_table(
        index=pivot_index,
        columns=["calibration_method", "metric_name"],
        values=stat,
        aggfunc="mean",   # no-op when already aggregated; handles any remaining dupes
    )

    # ── Sort rows ────────────────────────────────────────────────────────────

    if sort_by is not None and sort_by in table.index.names:
        table = table.sort_index(level=sort_by)
    elif sort_by == "model_size_params" and "model_size_params" in table.index.names:
        table = table.sort_index(level="model_size_params")

    # Drop model_size_params from visible index (keep for sorting only)
    if "model_size_params" in table.index.names:
        table = table.reset_index(level="model_size_params", drop=True)

    # ── Format values ────────────────────────────────────────────────────────

    return table.map(lambda v: format(v, fmt) if pd.notna(v) else "")