"""
Master evaluation script — runs master_wrapper over one or more models.

By default evaluates all models in MODEL_PATHS with standard defaults.
Use flags to restrict to a subset of models, constraints, or calibration methods.

Usage:
    python scripts/run_evals.py                                     # all models, all constraints
    python scripts/run_evals.py --models Gemma-3-270M Llama-3.2-1B
    python scripts/run_evals.py --constraints mutual_exclusivity transitivity
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.master_wrapper import master_wrapper
from src.model_paths import MODEL_PATHS

ALL_CONSTRAINTS = [
    "mutual_exclusivity",
    "forward_implication",
    "negation_consistency",
    "transitivity",
    "spatial_exclusivity",
]

# batch_size=4 for models likely to be VRAM-limited; 8 otherwise
LARGE_KEYWORDS = ["32B", "70B", "27B", "14B", "12B", "8B"]

def batch_size_for(model_name: str) -> int:
    return 4 if any(k in model_name for k in LARGE_KEYWORDS) else 8


def parse_args():
    parser = argparse.ArgumentParser(description="Run constraint evaluations across models.")

    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL",
        help="Models to evaluate (default: all in MODEL_PATHS). "
             f"Choices: {list(MODEL_PATHS.keys())}",
    )
    parser.add_argument(
        "--constraints", nargs="+", default=None,
        choices=ALL_CONSTRAINTS, metavar="CONSTRAINT",
        help=f"Constraint datasets to evaluate (default: all). Choices: {ALL_CONSTRAINTS}",
    )
    parser.add_argument(
        "--cal-methods", nargs="+", default=None,
        metavar="METHOD",
        help="Calibration methods to compute (default: all). See ALL_METHODS in calibration.py.",
    )
    parser.add_argument(
        "--aggregate-score-type", default="wmc_score",
        metavar="SCORE_TYPE",
        help="Aggregate score to display in plot titles (default: wmc_score). Pass 'none' to disable.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        metavar="N",
        help="Inference batch size. Default: auto (4 for large models, 8 otherwise).",
    )
    parser.add_argument(
        "--eval-type", default="answer_token",
        choices=["answer_token", "qa", "both"],
        help="Inference pass to use: answer_token (default), qa (yes/no question format), or both.",
    )
    parser.add_argument(
        "--ignore-errors", action="store_true",
        help="Log the error and continue to the next model rather than raising.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    models = args.models if args.models is not None else list(MODEL_PATHS.keys())

    # Validate model names
    unknown = [m for m in models if m not in MODEL_PATHS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {list(MODEL_PATHS.keys())}")

    # "none" string → None so master_wrapper skips aggregate label
    aggregate_score_type = None if args.aggregate_score_type == "none" else args.aggregate_score_type

    print(f"Models ({len(models)}): {models}")
    print(f"Constraints: {args.constraints or 'all'}")
    print(f"Eval type: {args.eval_type}")
    print(f"Calibration methods: {args.cal_methods or 'all'}")
    print("=" * 60)

    for model_name in models:
        print(f"\nRunning evals for {model_name}")
        try:
            batch_size = args.batch_size if args.batch_size is not None else batch_size_for(model_name)

            start = time.time()
            master_wrapper(
                model_name=model_name,
                constraint_subset=args.constraints,
                cal_methods=args.cal_methods,
                aggregate_score_type=aggregate_score_type,
                batch_size=batch_size,
                instruction_tuned=False,
                eval_type=args.eval_type,
            )
            print(f"Finished {model_name} in {time.time() - start:.2f}s")
            print("=" * 60)

        except Exception as e:
            print(f"FAILED on {model_name}: {e}")
            if not args.ignore_errors:
                raise


if __name__ == "__main__":
    main()
