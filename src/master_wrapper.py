"""
Main evaluation function -> loads models, datasets, does evals and stores all results
"""

import os
import pickle

import torch

from src.model import ModelWrapper
from src.dataset_utils import load_and_unify, extract_metric_dicts
from src.eval_wrappers import attach_last_token_probs, evaluate_dataset_constraints
from src.eval_plots import plot_entropy_vs_wmc, plot_entropy_histogram_by_wmc, plot_wmc_score_histogram
from src.results_to_df import build_aggregate_df, build_per_example_df
from src.model_paths import MODEL_PATHS
from src.utils import convert_to_question
import os
import torch

# Saving convention 
#   results go to eval_results/model_name/constraint-type.pkl
#   plots go to plots/model_name/constraint-type_calibration-method_plot-type.pdf - PDF MATTERS FOR PAPERS. PNG IS LOW RES
#   csv files go to eval_results/model_name/csv/constraint-type.csv

# takes in model + datasets -> does everything
def master_wrapper(
        model_name = None, # model name - ADD TO MODEL_PATHS IF NOT HERE
        constraint_subset=None,      # subset of constraints to evaluate on
        calibration_methods_subset=None, # subset of calibration methods to grab plots on
        cal_methods=None,             # subset of calibration methods to compute (default: all); see ALL_METHODS in calibration.py
        nucleus_ps=None,              # nucleus p values for cal_nucleusXXX / cal_nucleus_entropyXXX (default: [0.80, 0.85, 0.90, 0.95])
        aggregate_score_type="wmc_score", # for displaying in plots, wmc_score is our standard metric (None if don't include)
        batch_size = 128, # batch size for grabbing probs
        instruction_tuned=False,     # whether model being evaluated is instruction tuned
        eval_type="answer_token",       # evaluation type: "answer_token", "qa" (yes/no question pass only), or "both" (standard + yes/no merged)
        # only answer_token evaluation used in paper, qa left for future exploration
        # qa evaluation converts factual statements into questions like "Is it true that A?" and uses probabilities of model responding yes/no
    ):

    if model_name is None:
        raise ValueError("You must provide a model_name.")
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Model name {model_name} not found in MODEL_PATHS. Please add the model path to MODEL_PATHS in src.model_paths.py.")
    
    # Ensure directories we save in exist
    base_eval_dir = os.path.join("eval_results", model_name)
    base_plot_dir = os.path.join("plots", model_name)

    os.makedirs(base_eval_dir, exist_ok=True)
    os.makedirs(base_plot_dir, exist_ok=True)
    os.makedirs(os.path.join(base_eval_dir, "csv"), exist_ok=True)

    model_path = MODEL_PATHS[model_name]
    model = ModelWrapper(model_path, instruction_tuned=instruction_tuned) # passing in instruction_tuned shouldn't be necessary
    tokenizer = model.tokenizer

    constraint_mapping = {
        "mutual_exclusivity": "datasets/mutual_exclusivity_dataset_cleaned.json",
        "forward_implication": "datasets/forward_implication_dataset_cleaned.json",
        "negation_consistency": "datasets/negation_consistency_dataset_cleaned.json",
        "transitivity": "datasets/transitivity_dataset_cleaned.json",
        "spatial_exclusivity": "datasets/spatial_mutex_dataset_cleaned.json",
    }

    if constraint_subset is not None:
        for c in constraint_subset:
            if c not in constraint_mapping:
                raise ValueError(f"Constraint {c} not found in available datasets. Please choose from {list(constraint_mapping.keys())}.")
        constraints = constraint_subset
    else:
        constraints = list(constraint_mapping.keys())

    # All data points involving statements with multi token answer tokens are filtered out
    #   This is because handling multi token strings isn't as clear
    # We may wish to do evals on this separately later, which will involve minor changes to ModelWrapper, but shouldn't affect anything else
    datasets = {
        c: load_and_unify(constraint_mapping[c], tokenizer=tokenizer) for c in constraints
    }

    outputs = {}

    # pre-build question-formatted datasets for qa eval_type
    if eval_type != "answer_token":
        qa_datasets = {k: ds.map(convert_to_question) for k, ds in datasets.items()}

    # First get results for each constraint dataset
    # This grabs last token probabilities, calibrates them, and evaluates logical constraints (for each calibration method)
    # Saves off per-example results (calibrated probabilities of each fact and evals) + aggregate scores

    for constraint, dataset in datasets.items():
        print(f"Processing constraint: {constraint}")

        pkl_path = f"eval_results/{model_name}/{constraint}.pkl"

        # Main inference pass: answer_token probabilities + calibration
        if eval_type in ("answer_token", "both"):
            data = attach_last_token_probs(
                dataset=dataset,
                model=model,
                batch_size=batch_size,
                calibrate=True,
                evaluate=False,
                use_chat_template=instruction_tuned,
                cal_methods=cal_methods,
                nucleus_ps=nucleus_ps,
            )

        # Exploratory inference pass: yes/no question probabilities + calibration 
        if eval_type in ("qa", "both"):
            qa_data = attach_last_token_probs(
                dataset=qa_datasets[constraint],
                model=model,
                batch_size=batch_size,
                calibrate=True,
                evaluate=False,
                use_chat_template=instruction_tuned,
                yes_no=True,
            )

        if eval_type == "both":
            # Inject all yes/no-derived calibration methods into normal results
            # so they appear alongside standard calibration methods
            for data_entry, qa_entry in zip(data, qa_data):
                for group in data_entry["prob_info"]:
                    for i in range(len(data_entry["prob_info"][group])):
                        qa_cal = qa_entry["prob_info"][group][i]["calibration"]
                        for key, val in qa_cal.items():
                            data_entry["prob_info"][group][i]["calibration"][key] = val
            data_to_eval = data
        elif eval_type == "qa":
            data_to_eval = qa_data
        else: 
            data_to_eval = data

        # Evaluate constraints on data
        final_results = evaluate_dataset_constraints(data_to_eval)

        with open(pkl_path, "wb") as f:
            pickle.dump(final_results, f)

        # print(f"Aggregate Scores for {constraint}:")
        # print(final_results["aggregate"]["cal_nucleus_entropy"]["wmc"])
        print("=" * 50)

        outputs[constraint] = final_results

    for constraint in constraints:

        if constraint not in outputs:
            continue

        # grab calibration methods automatically
        calibration_methods = list(outputs[constraint]["aggregate"].keys())
        try:
            calibration_methods.remove("num_samples")
        except:
            pass

        constraint_formatted = constraint.replace("_", " ").title()

         # Grab plots for different calibration methods separately
        for calibration_method in calibration_methods:
            if calibration_methods_subset is not None and calibration_method not in calibration_methods_subset:
                continue

            metric_dicts = extract_metric_dicts(
                outputs[constraint]["dataset_with_evals"],
                calibration_method
            )

            if aggregate_score_type is not None:
                aggregate = outputs[constraint]["aggregate"][calibration_method][aggregate_score_type]
                additional_title_info = f", WMC Score: {aggregate['mean']:.2f}"
            else:
                additional_title_info = ""

            # Entropy vs WMC scatter plots and histograms
            os.makedirs(os.path.join("plots", model_name, calibration_method, constraint), exist_ok=True)
            plot_entropy_vs_wmc(
                metric_dicts,
                title = f"{constraint_formatted}: Entropy vs WMC{additional_title_info}",
                save_path=f"plots/{model_name}/{calibration_method}/{constraint}/scatterplot.pdf"
            )

            plot_entropy_histogram_by_wmc(
                metric_dicts,
                title = f"{constraint_formatted}: Entropy vs WMC Bucket{additional_title_info}",
                save_path=f"plots/{model_name}/{calibration_method}/{constraint}/histogram.pdf"
            )

            # WMC score distribution histograms (per-example counts)
            plot_wmc_score_histogram(
                metric_dicts,
                title=f"{constraint_formatted}: WMC Score Distribution{additional_title_info}",
                save_path=f"plots/{model_name}/{calibration_method}/{constraint}/wmc_score_distribution.pdf",
                raw=False, # uses soft thresholded WMC score
            )

            plot_wmc_score_histogram(
                metric_dicts,
                title=f"{constraint_formatted}: WMC Raw Distribution{additional_title_info}",
                save_path=f"plots/{model_name}/{calibration_method}/{constraint}/wmc_raw_distribution.pdf",
                raw=True, # uses raw WMC score
            )

            plot_wmc_score_histogram(
                metric_dicts,
                title=f"{constraint_formatted}: Discrete Score Distribution{additional_title_info}",
                save_path=f"plots/{model_name}/{calibration_method}/{constraint}/discrete_distribution.pdf",
                discrete=True,
            )

        # Also store off csv files with results
        # these aren't totally necessary now, but we'll keep them for some easier parsing later in case
        df_per_example = build_per_example_df(outputs[constraint])
        df_aggregate = build_aggregate_df(outputs[constraint])

        df_per_example.to_csv(f"eval_results/{model_name}/csv/{constraint}_per_example.csv", index=False)
        df_aggregate.to_csv(f"eval_results/{model_name}/csv/{constraint}_aggregate.csv", index=False)

    del model.model
    del model.tokenizer
    del model # note model is a wrapper, you have to delete model.model first or it doesn't get cleaned properly
    torch.cuda.empty_cache()