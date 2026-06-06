"""
Stores results as dataframe, which we store as csvs in src.master_wrapper/master_wrapper.
Allows for easier analysis and plotting.
"""

import pandas as pd
from tqdm.auto import tqdm

def build_aggregate_df(final_results):
    """
    Creates csv file with aggregated results
    """


    aggregate = final_results.get("aggregate", {})
    rows = []

    for cal_method, metrics in aggregate.items():

        if cal_method == "num_samples":
            continue

        for metric_name, stats in metrics.items():

            row = {
                "calibration_method": cal_method,
                "metric_name": metric_name,
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }

            rows.append(row)

    return pd.DataFrame(rows)

def build_per_example_df(final_results):
    """
    Creates csv file with per example results
    """

    dataset = final_results.get("dataset_with_evals", [])
    rows = []

    for entry in tqdm(dataset, desc="Building Sample DataFrame"):

        row = {}

        # ---- metadata ----
        row["constraint_type"] = entry.get("constraint_type")
        row["formula"] = entry.get("formula")

        gt = entry.get("ground_truth", {})
        row["gt_A"] = gt.get("A")
        row["gt_B"] = gt.get("B")

        prompt_groups = entry.get("prompt_groups", {})
        evaluations = entry.get("evaluations", {})

        # ---- text columns ----
        for group_name, texts in prompt_groups.items():
            for idx, text in enumerate(texts):

                prefix = group_name if len(texts) == 1 else f"{group_name}_{idx}"
                row[f"{prefix}_text"] = text

        # ---- probabilities + logical metrics ----
        for cal_method, metrics in evaluations.items():

            for metric_name, value in metrics.items():

                if value is None:
                    continue

                # probability columns
                if metric_name.startswith("text_"):
                    row[f"{metric_name}_{cal_method}"] = value

                # logical metrics
                else:
                    row[f"{cal_method}_{metric_name}"] = value

        rows.append(row)

    return pd.DataFrame(rows)
