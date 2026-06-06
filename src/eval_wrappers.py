"""
Evaluation wrappers for grabbing last token probabilities, calculating calibrated probabilities,
and evaluating logical constraint satisfaction.
"""

from src.dataset_utils import flatten_prompt_groups
from torch.utils.data import DataLoader
from src.model import ModelWrapper
from datasets import Dataset
import pickle
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from src.calibration import calibrate_dataset, calibrate_true_false_dataset
from src.evaluation import (
    get_constraint_dict,
    wmc_baseline,
    xor_wmc,
    and_not_wmc,
    implication_wmc,
    transitivity_wmc,
    joint_wmc,
    satisfying_assignment_entropy,
)

def attach_last_token_probs(
    dataset: Dataset,
    model: ModelWrapper,
    batch_size=8,
    save_path=None,
    calibrate=False,    # Calibrate probabilities
    evaluate=False,     # Evaluate logical constraints
    include_prob_info=False,  # Keep raw logit prob_info in output (stripped by default)
    k_tops=None,        # list of top-k values for cal_topkXXX (default [50])
    nucleus_ps=None,    # list of nucleus probabilities for cal_nucleusXXX / cal_nucleus_entropyXXX (default [0.95])
    cal_methods=None,   # subset of calibration methods to compute (default: all)
    use_chat_template=False,  # apply chat template (requires model.has_chat_template)
    yes_no=False,             # yes/no calibration mode (uses generation prompt + calibrate_true_false_dataset)
):
    # flat texts flatten all prompts we want to evaluate
    # mapping allows mapping back into original format
    flat_texts, mapping = flatten_prompt_groups(dataset)

    dataloader = DataLoader(flat_texts, batch_size=batch_size)

    prob_infos = model.last_token_prob(dataloader, use_chat_template=use_chat_template, yes_no=yes_no)

    # Copy dataset into mutable structure
    new_data = [dict(example) for example in dataset]

    # Initialize storage
    for example in new_data:
        example["prob_info"] = {
            group: [None] * len(texts)
            for group, texts in example["prompt_groups"].items()
        }

    # Reconstruct correctly
    for (ex_idx, group_name, p_idx), prob_info in zip(mapping, prob_infos):
        new_data[ex_idx]["prob_info"][group_name][p_idx] = prob_info

    if calibrate or evaluate:
        vocab_size = model.vocab_size
        if yes_no:
            new_data = calibrate_true_false_dataset(new_data, tokenizer=model.tokenizer)
        else:
            # this evaluates all methods by default with specified hyperparameters (importantly p=0.95 for nucleus methods)
            new_data = calibrate_dataset(new_data, vocab_size, k_tops=k_tops, nucleus_ps=nucleus_ps, methods=cal_methods)
       
    # Evaluate dataset constraints
    if evaluate:
        output_data = evaluate_dataset_constraints(new_data, include_prob_info=include_prob_info)
    else:
        if not include_prob_info:
            for example in new_data:
                for group_infos in example.get("prob_info", {}).values():
                    for info in group_infos:
                        if info is not None:
                            info.pop("full_prob_vector", None)
        output_data = new_data
       
    if save_path is not None:
        if not save_path.endswith(".pkl"):
            save_path += ".pkl"
        with open(save_path, "wb") as f:
            pickle.dump(output_data, f)
        
    return output_data


def evaluate_constraints(entry, prob_keys=None, constraint_types=None, ood_threshold=0.5):
    results = {}
    sample_info = next(iter(entry["prob_info"].values()))[0]
    calibration_dict = sample_info.get("calibration", {})

    if prob_keys is None:
        # Exclude diagnostics and boolean flags
        prob_keys = [k for k, v in calibration_dict.items() if k != "diagnostics" and not isinstance(v, bool)]

    if constraint_types is None:
        constraint_types = [entry["constraint_type"]]  # so that it raises KeyError when not found

    for constraint in constraint_types:
        results[constraint] = {}

        is_xor_logic = constraint in ["negation_consistency", "mutual_exclusivity"]
        is_implication_logic = constraint in ["forward_implication", "reverse_implication"]
        is_transitivity_logic = constraint in ["transitivity"]

        if is_xor_logic or is_implication_logic or is_transitivity_logic:
            for prob_key in prob_keys:
                constraint_dict = get_constraint_dict(entry, prob_key)

                if None in constraint_dict.values():
                    continue

                metric_payload = {}

                # OOD: true if no statement has probability > ood_threshold (no "true" statement)
                metric_payload["is_ood"] = not any(v > ood_threshold for v in constraint_dict.values())

                # for every metric, we include wmc_raw, wmc_score (soft-normalized by baseline), discrete (binary threshold), and entropy
                # we also always include a secondary metric (and_not or joint) that gets rid of vacuously true assignments
                if is_xor_logic:
                    xor_baseline = wmc_baseline(constraint)
                    raw = xor_wmc(constraint_dict)
                    metric_payload["wmc_raw"] = raw
                    metric_payload["wmc_score"] = (raw - xor_baseline) / (1 - xor_baseline)
                    metric_payload["discrete"] = xor_wmc(constraint_dict, binary_threshold=xor_baseline)
                    metric_payload["entropy"] = satisfying_assignment_entropy(constraint_dict, constraint)

                    and_not_baseline = wmc_baseline("and_not")
                    and_not_raw = and_not_wmc(constraint_dict)
                    metric_payload["joint_wmc_raw"] = and_not_raw
                    metric_payload["joint_wmc_score"] = (and_not_raw - and_not_baseline) / (1 - and_not_baseline)
                    metric_payload["joint_discrete"] = and_not_wmc(constraint_dict, binary_threshold=and_not_baseline)
                    metric_payload["joint_entropy"] = satisfying_assignment_entropy(constraint_dict, "and_not")
                elif is_implication_logic:
                    implication_baseline = wmc_baseline(constraint)
                    raw = implication_wmc(constraint_dict)
                    metric_payload["wmc_raw"] = raw
                    metric_payload["wmc_score"] = (raw - implication_baseline) / (1 - implication_baseline)
                    metric_payload["discrete"] = implication_wmc(constraint_dict, binary_threshold=implication_baseline)
                    metric_payload["entropy"] = satisfying_assignment_entropy(constraint_dict, constraint)

                    joint_baseline = wmc_baseline("joint", num_vars=len(constraint_dict))
                    joint_raw = joint_wmc(constraint_dict)
                    metric_payload["joint_wmc_raw"] = joint_raw
                    metric_payload["joint_wmc_score"] = (joint_raw - joint_baseline) / (1 - joint_baseline)
                    metric_payload["joint_discrete"] = joint_wmc(constraint_dict, binary_threshold=joint_baseline)
                    metric_payload["joint_entropy"] = satisfying_assignment_entropy(constraint_dict, "joint")

                    nv_baseline = wmc_baseline(constraint, exclude_vacuous=True)
                    nv_raw = implication_wmc(constraint_dict, exclude_vacuous=True)
                    metric_payload["no_vacuous_wmc_raw"] = nv_raw
                    metric_payload["no_vacuous_wmc_score"] = (nv_raw - nv_baseline) / (1 - nv_baseline)
                    metric_payload["no_vacuous_discrete"] = implication_wmc(constraint_dict, binary_threshold=nv_baseline, exclude_vacuous=True)
                    metric_payload["no_vacuous_entropy"] = satisfying_assignment_entropy(constraint_dict, constraint, exclude_vacuous=True)
                elif is_transitivity_logic:
                    transitivity_baseline = wmc_baseline(constraint)
                    raw = transitivity_wmc(constraint_dict)
                    metric_payload["wmc_raw"] = raw
                    metric_payload["wmc_score"] = (raw - transitivity_baseline) / (1 - transitivity_baseline)
                    metric_payload["discrete"] = transitivity_wmc(constraint_dict, binary_threshold=transitivity_baseline)
                    metric_payload["entropy"] = satisfying_assignment_entropy(constraint_dict, constraint)

                    joint_baseline = wmc_baseline("joint", num_vars=len(constraint_dict))
                    joint_raw = joint_wmc(constraint_dict)
                    metric_payload["joint_wmc_raw"] = joint_raw
                    metric_payload["joint_wmc_score"] = (joint_raw - joint_baseline) / (1 - joint_baseline)
                    metric_payload["joint_discrete"] = joint_wmc(constraint_dict, binary_threshold=joint_baseline)
                    metric_payload["joint_entropy"] = satisfying_assignment_entropy(constraint_dict, "joint")

                    nv_baseline = wmc_baseline(constraint, exclude_vacuous=True)
                    nv_raw = transitivity_wmc(constraint_dict, exclude_vacuous=True)
                    metric_payload["no_vacuous_wmc_raw"] = nv_raw
                    metric_payload["no_vacuous_wmc_score"] = (nv_raw - nv_baseline) / (1 - nv_baseline)
                    metric_payload["no_vacuous_discrete"] = transitivity_wmc(constraint_dict, binary_threshold=nv_baseline, exclude_vacuous=True)
                    metric_payload["no_vacuous_entropy"] = satisfying_assignment_entropy(constraint_dict, constraint, exclude_vacuous=True)

                # Capture component probabilities for diagnostics
                for var_name, var_prob in constraint_dict.items():
                    metric_payload[var_name] = var_prob

                results[constraint][prob_key] = metric_payload

    return results


def evaluate_dataset_constraints(
    dataset,
    prob_keys=None,
    constraint_types=None,
    include_prob_info=False,
    ood_threshold=0.5,
):
    aggregate_accumulator = defaultdict(lambda: defaultdict(list))
    ood_excluded_accumulator = defaultdict(lambda: defaultdict(list))

    if prob_keys is None:
        sample_info = next(iter(dataset[0]["prob_info"].values()))[0]
        calibration_dict = sample_info.get("calibration", {})

        prob_keys = [
            k for k, v in calibration_dict.items()
            if k != "diagnostics" and not isinstance(v, bool)
        ]

        if "raw_prob" not in prob_keys and "raw_prob" in calibration_dict:
            prob_keys.append("raw_prob")

    processed_dataset = []

    for entry in tqdm(dataset, desc="Evaluating constraints"):

        entry_results = evaluate_constraints(
            entry,
            prob_keys=prob_keys,
            constraint_types=constraint_types,
            ood_threshold=ood_threshold,
        )

        # flatten constraint level
        evaluations = next(iter(entry_results.values()))

        new_entry = {
            k: v
            for k, v in entry.items()
            if k not in ["evaluations", "prob_info"]
        }

        new_entry["evaluations"] = evaluations

        if include_prob_info:
            new_entry["prob_info"] = entry["prob_info"]

        processed_dataset.append(new_entry)

        # aggregate
        for prob_key, metrics in evaluations.items():
            is_ood = metrics.get("is_ood", False)
            for metric_name, value in metrics.items():
                if value is not None:
                    aggregate_accumulator[prob_key][metric_name].append(value)
                    if not is_ood and metric_name != "is_ood":
                        ood_excluded_accumulator[prob_key][metric_name].append(value)

    aggregate_results = {"num_samples": len(dataset)}

    for prob_key, metric_dict in aggregate_accumulator.items():

        aggregate_results[prob_key] = {}

        for metric_name, values in metric_dict.items():

            val_array = np.array(values, dtype=float)

            aggregate_results[prob_key][metric_name] = {
                "mean": float(val_array.mean()),
                "std": float(val_array.std()),
                "min": float(val_array.min()),
                "max": float(val_array.max()),
            }

        # OOD-excluded aggregates: same stats but only over non-OOD data points
        for metric_name, values in ood_excluded_accumulator[prob_key].items():
            if not values:
                continue
            val_array = np.array(values, dtype=float)
            aggregate_results[prob_key][f"{metric_name}_no_ood"] = {
                "mean": float(val_array.mean()),
                "std": float(val_array.std()),
                "min": float(val_array.min()),
                "max": float(val_array.max()),
            }

    return {
        "dataset_with_evals": processed_dataset,
        "aggregate": aggregate_results,
    }
