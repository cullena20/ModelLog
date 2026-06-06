"""
Miscellaneous helper functions to explore results.
"""

import os
import re
import pandas as pd
import pickle


def extract_model_size(model_name):
    """
    Extract parameter count from model name.
    Returns number of parameters (int).
    """

    match = re.findall(r'(\d+\.?\d*)([mMbB])', model_name)
    if not match:
        return None

    size, unit = match[-1]
    size = float(size)

    if unit.lower() == "b":
        size *= 1e9
    else:
        size *= 1e6

    return int(size)


def build_master_dataframe(base_dir="eval_results", skip_dirs=[]):
    """
    Loads all aggregate CSV files from the specified directory structure and combines them into a single DataFrame.
     - base_dir: The root directory containing model subdirectories.
     - skip_dirs: List of model directory names to skip (e.g., ["gpt2_small", "gpt2_medium"]).
     - Returns: A pandas DataFrame with columns for model, model_size_params, constraint, and all metrics from the CSV files.
     - The function assumes that each model's CSV files are located in a "csv" subdirectory and that the CSV files are named in the format "{constraint}_aggregate.csv". 
     - The function extracts the model size from the model name using the extract_model_size helper function and includes it as a column in the resulting DataFrame.
    """

    rows = []

    for model in os.listdir(base_dir):
        
        if model in skip_dirs:
            continue

        model_csv_dir = os.path.join(base_dir, model, "csv")

        if not os.path.isdir(model_csv_dir):
            continue

        model_size = extract_model_size(model)

        for file in os.listdir(model_csv_dir):

            if not file.endswith("_aggregate.csv"):
                continue

            constraint = file.replace("_aggregate.csv", "")

            path = os.path.join(model_csv_dir, file)

            df = pd.read_csv(path)

            df["model"] = model
            df["model_size_params"] = model_size
            df["constraint"] = constraint

            rows.append(df)

    return pd.concat(rows, ignore_index=True)

# Helper functions for loading results, which we can then plot

def metric_dicts_from_df(df: pd.DataFrame, cal_method: str) -> list[dict]:
    """
    Reconstruct the metric_dicts list expected by eval_plots functions from a
    per-example CSV row.  The CSV stores non-text metrics as {cal_method}_{metric}
    (e.g. raw_prob_wmc_raw → wmc_raw after stripping the prefix).
    """
    prefix = f"{cal_method}_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    dicts = []
    for _, row in df.iterrows():
        d = {}
        for col in cols:
            metric = col[len(prefix):]
            val = row[col]
            if pd.notna(val):
                d[metric] = float(val)
        if d:
            dicts.append(d)
    return dicts

def load_per_example_csv(model_name: str, constraint: str):
    """
    Load per-example CSV for one (model, constraint) pair.
    Returns per_example_df
    """
    csv_dir = os.path.join("eval_results", model_name, "csv")
    if not os.path.exists(csv_dir):
        raise Exception(f"Model '{model_name}' does not exist in 'eval_results'")

    pe_path  = os.path.join(csv_dir, f"{constraint}_per_example.csv")
    if not os.path.exists(pe_path):
        raise Exception(f"Constraint '{constraint}' does not exist in 'eval_results/{model_name}'")
    
    return pd.read_csv(pe_path)

def load_main_results_csv(model_name: str, constraint: str):
    """
    Load per-example and aggregate CSVs for one (model, constraint) pair.

    Returns (per_example_df, aggregate_df) or (None, None) if files are missing.
    """
    csv_dir = os.path.join("eval_results", model_name, "csv")
    pe_path  = os.path.join(csv_dir, f"{constraint}_per_example.csv")
    agg_path = os.path.join(csv_dir, f"{constraint}_aggregate.csv")

    if not os.path.exists(pe_path) or not os.path.exists(agg_path):
        return None, None

    return pd.read_csv(pe_path), pd.read_csv(agg_path)


def load_main_results_pkl(model_name: str, constraint: str):
    """
    Load full pickle for one (model, constraint) pair.

    Returns the result dict (keys: 'dataset_with_evals', 'aggregate') or None.
    """
    pkl_path = os.path.join("eval_results", model_name, f"{constraint}.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def print_aggregate_results(
    aggregate_df,
    calibration_method,
    include_entropy=True,
    include_joint=False,
    include_no_vacuous=False,
    include_no_ood=False,
    include_no_vacuous_no_ood=False,
    verbose=True,
    joint_verbose=True,
    model=None,
    constraint=None,
):
    """
    Print aggregate metrics for a single (aggregate_df, calibration_method) pair.

    Parameters
    ----------
    aggregate_df : DataFrame
        Single-constraint aggregate CSV loaded into a DataFrame (columns:
        calibration_method, metric_name, mean, …).
    calibration_method : str
        Which calibration method row to filter on.
    include_entropy : bool
        Show entropy alongside WMC metrics in each section.
    include_joint : bool
        Show the joint (exact truth-assignment) section.
    include_no_vacuous : bool
        Show a "no-vacuous" section: same metrics prefixed with no_vacuous_.
    include_no_ood : bool
        Show a "no-OOD" section: same metrics postfixed with _no_ood.
    include_no_vacuous_no_ood : bool
        Show a combined section: no_vacuous_ prefix AND _no_ood postfix.
    verbose : bool
        True  → show WMC Raw, WMC Score, and Discrete Score.
        False → show only WMC Score (entropy still controlled by include_entropy).
    joint_verbose : bool
        Only relevant when include_joint=True.
        True  → joint section also respects include_no_vacuous / include_no_ood flags.
        False → joint section shows only the standard joint metrics regardless of
                the no_vacuous / no_ood flags.
    model : str | None
        Included in the header for context.
    constraint : str | None
        Included in the header for context.
    """
    base = aggregate_df[aggregate_df["calibration_method"] == calibration_method]

    if base.empty:
        print(f"No data found for calibration_method={calibration_method}")
        return

    metric_map = base.set_index("metric_name")["mean"].to_dict()

    def get(name):
        if name not in metric_map:
            raise KeyError(f"Missing metric: {name}")
        return metric_map[name]

    def fmt_section(prefix, postfix, label):
        """Return a formatted block for one metric variant group."""
        def mk(base_name):
            return f"{prefix}{base_name}{postfix}"

        try:
            wmc_score  = get(mk("wmc_score"))
            discrete   = get(mk("discrete"))
            if verbose:
                wmc_raw = get(mk("wmc_raw"))
        except KeyError as e:
            return f"        [missing metric {e}]\n"

        lines = [f"        --- {label} ---"]
        if verbose:
            lines.append(f"        WMC Raw        : {wmc_raw:.4f}")
        lines.append(    f"        WMC Score      : {wmc_score:.4f}")
        if verbose:
            lines.append(f"        Discrete Score : {discrete:.4f}")
        if include_entropy:
            ent = metric_map.get(mk("entropy"))
            if ent is not None:
                lines.append(f"        Entropy        : {ent:.4f}")
        return "\n".join(lines) + "\n"

    # ── Header ───────────────────────────────────────────────────────────────

    header_parts = ["Aggregate results"]
    if model is not None:
        header_parts.append(f"for {model}")
    if constraint is not None:
        header_parts.append(f"on {constraint}")
    header_parts.append(f"with calibration method {calibration_method}")
    header = " ".join(header_parts) + ":"

    output = f"\n        {header}\n\n"

    # ── Marginal sections ────────────────────────────────────────────────────

    output += fmt_section("", "", "Results")

    if include_no_vacuous:
        output += fmt_section("no_vacuous_", "", "No-Vacuous")

    if include_no_ood:
        output += fmt_section("", "_no_ood", "No-OOD")

    if include_no_vacuous_no_ood:
        output += fmt_section("no_vacuous_", "_no_ood", "No-Vacuous and No-OOD")

    # ── Joint sections ───────────────────────────────────────────────────────

    if include_joint:
        # Detect whether the data uses "joint_" or legacy "and_not_" prefix.
        joint_pfx = "joint_" if "joint_wmc_score" in metric_map else "and_not_"

        def fmt_joint_section(postfix, label):
            """Format one joint metric block; returns None if metrics are missing."""
            try:
                wmc_score = get(f"{joint_pfx}wmc_score{postfix}")
                discrete  = get(f"{joint_pfx}discrete{postfix}")
                if verbose:
                    wmc_raw = get(f"{joint_pfx}wmc_raw{postfix}")
            except KeyError:
                return None  # silently skip when the variant doesn't exist

            lines = [f"        --- {label} ---"]
            if verbose:
                lines.append(f"        WMC Raw        : {wmc_raw:.4f}")
            lines.append(    f"        WMC Score      : {wmc_score:.4f}")
            if verbose:
                lines.append(f"        Discrete Score : {discrete:.4f}")
            # joint entropy is always 0 (only one correct truth assignment)
            return "\n".join(lines) + "\n"

        section = fmt_joint_section("", "Joint")
        if section:
            output += section

        if joint_verbose:
            # no_vacuous joint: only emit if a distinct prefixed key actually exists
            if include_no_vacuous:
                has_nv_joint = f"{joint_pfx}no_vacuous_wmc_score" in metric_map
                if has_nv_joint:
                    section = fmt_joint_section("", "Joint and No-Vacuous")
                    if section:
                        output += section

            if include_no_ood:
                section = fmt_joint_section("_no_ood", "Joint and No-OOD")
                if section:
                    output += section

            # no_vacuous_no_ood: joint has no no_vacuous prefix, so this maps to
            # _no_ood — only emit if not already shown by include_no_ood
            if include_no_vacuous_no_ood and not include_no_ood:
                section = fmt_joint_section("_no_ood", "Joint and No-OOD")
                if section:
                    output += section

    print(output)

# question conversion used by instruction_tuned and instruction_tuned_only for yes/no pass
def convert_to_question(example):
    new_prompt_groups = dict()
    for key, texts in example["prompt_groups"].items():
        new_prompt_groups[key] = [f"Is it true that {text}?" for text in texts]
    return { "prompt_groups": new_prompt_groups }
