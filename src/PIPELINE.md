# src/ Pipeline Documentation

This document describes every file in `src/` and the full evaluation pipeline end to end, from raw dataset JSON to plots and saved results. The canonical entry point is `src/master_wrapper.py`.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step 1 — Datasets](#step-1--datasets)
3. [Step 2 — Dataset Loading (`dataset_utils.py`)](#step-2--dataset-loading-dataset_utilspy)
4. [Step 3 — Model Loading and Inference (`model.py`)](#step-3--model-loading-and-inference-modelpy)
5. [Step 4 — Attaching Probabilities (`eval_wrappers.py`)](#step-4--attaching-probabilities-eval_wrapperspy)
6. [Step 5 — Calibration (`calibration.py`)](#step-5--calibration-calibrationpy)
7. [Step 6 — Evaluation (`evaluation.py`)](#step-6--evaluation-evaluationpy)
8. [Step 7 — Saving Results](#step-7--saving-results)
9. [Step 8 — Plots (`eval_plots.py`)](#step-8--plots-eval_plotspy)
10. [Cross-model Analysis (`utils.py`, `results_to_df.py`)](#cross-model-analysis-utilspy-results_to_dfpy)
11. [Model Registry (`model_paths.py`)](#model-registry-model_pathspy)
12. [Master Wrapper (`master_wrapper.py`)](#master-wrapper-master_wrapperpy)
13. [Working with Intermediate Results](#working-with-intermediate-results)

---

## Pipeline Overview

```
datasets/*.json
      │
      ▼
dataset_utils.load_and_unify()          # filter, unify format
      │
      ▼
eval_wrappers.attach_last_token_probs() # main pipeline wrapper
  │
  ├── dataset_utils.flatten_prompt_groups()
  ├── model.ModelWrapper.last_token_prob()   # GPU inference
  ├── calibration.calibrate_dataset()        # 7 calibration methods
  └── evaluation.evaluate_dataset_constraints()  # WMC + entropy per example
      │
      ▼
eval_results/{model}/{constraint}.pkl       # full intermediate results
eval_results/{model}/csv/{constraint}_*.csv # CSVs

plots/{model}/{cal_method}/{constraint}/*.pdf # plots
```

---

## Step 1 — Datasets

**Location:** `datasets/`

Five JSON files, each an array of objects:

| File | Constraint | Formula | # Prompt Groups |
|------|-----------|---------|-----------------|
| `mutual_exclusivity_dataset_cleaned.json` | mutual_exclusivity | `A ^ ~B` | text_a, text_b |
| `negation_consistency_dataset_cleaned.json` | negation_consistency | `A ^ ~B` | text_a, text_b |
| `forward_implication_dataset_cleaned.json` | forward_implication | `A => B` | text_a, text_b |
| `transitivity_dataset_cleaned.json` | transitivity | `(A ^ B) => C` | text_a, text_b, text_c |
| `spatial_mutex_dataset_cleaned.json` | spatial_exclusivity | `A ^ ~B` | text_a, text_b |

**Raw JSON example (mutual_exclusivity):**
```json
{
    "constraint_type": "mutual_exclusivity",
    "formula": "A ^ ~B",
    "text_a": "Corn Crake is a type of bird",
    "text_b": "Corn Crake is a type of amphibian",
    "context_a": "Corn Crake is a type of ",
    "context_b": "Corn Crake is a type of ",
    "trigger_word_a": "bird",
    "trigger_word_b": "amphibian",
    "ground_truth": {"A": 1.0, "B": 0.0},
    "is_fit": true,
    "confidence_score": 0.95,
    ...
}
```

**Key fields:**
- `text_a/b/c`: full prompt string ending in the answer token (no trailing period after loading)
- `trigger_word_a/b/c`: the single answer token expected at the end of the prompt
- `ground_truth`: `{"A": float, "B": float, ...}` — 1.0 = true, 0.0 = false; used for analysis, not for filtering
- `is_fit`: GPT-quality-filtered boolean; examples with `is_fit=False` are excluded

**Why only single-token answers?** The pipeline computes P(last token | context). Multi-token answers require a different setup. Examples where any `trigger_word` is tokenized into multiple tokens by the model's tokenizer are filtered out during loading.

---

## Step 2 — Dataset Loading (`dataset_utils.py`)

### `load_and_unify(json_path, filter_fit=True, tokenizer=None) → HuggingFace Dataset`

Loads a dataset JSON, applies filters, and normalizes every example into a unified format.

**Filters applied (in order):**
1. `filter_unfit`: drops examples where `is_fit == False`
2. `filter_multi_token`: if `tokenizer` is provided, drops examples where any `trigger_word_*` tokenizes to more than one token. The trigger word is prepended with a space (e.g., `" bird"`) before tokenizing — this is the correct tokenization in context and significantly affects results.

**Output format per example:**
```python
{
    "constraint_type": str,           # e.g. "mutual_exclusivity"
    "formula": str,                   # e.g. "A ^ ~B"
    "prompt_groups": {
        "text_a": ["Corn Crake is a type of bird"],   # list of prompt strings
        "text_b": ["Corn Crake is a type of amphibian"],
        # "text_c": [...] for transitivity only
    },
    "ground_truth": {"A": 1.0, "B": 0.0},
}
```

The original raw columns (context, trigger words, reasoning, etc.) are dropped. `prompt_groups` wraps each prompt in a list for forward-compatibility with multi-prompt groups.

---

### `flatten_prompt_groups(dataset) → (flat_texts, mapping)`

Flattens all prompts across all examples into a single list for batched inference.

```python
flat_texts = ["Corn Crake is a type of bird", "Corn Crake is a type of amphibian", ...]
mapping = [(0, "text_a", 0), (0, "text_b", 0), ...]  # (example_idx, group_name, prompt_idx)
```

The `mapping` is used to reconstruct results back into the per-example structure after inference.

---

### `extract_metric_dicts(dataset_with_evals, prob_key="raw_prob") → List[Dict[str, float]]`

Helper for plotting. Takes the `dataset_with_evals` list (from `evaluate_dataset_constraints`) and extracts just the numeric metrics for a given calibration method. Returns a flat list of metric dicts (one per example), filtering out text keys like `text_a`, `text_b`.

---

## Step 3 — Model Loading and Inference (`model.py`)

### `ModelWrapper(model_name)`

Loads a HuggingFace causal LM and tokenizer from a local path or HF hub name.

**Important settings:**
- `dtype=torch.bfloat16`, `device_map="auto"` — multi-GPU and memory-efficient
- `tokenizer.padding_side = "left"` — **critical**: ensures the last token in the padded batch is always the answer token position
- `pad_token = eos_token` if no pad token defined

**Attributes:** `model`, `tokenizer`, `vocab_size`, `device`

---

### `ModelWrapper.last_token_prob(dataloader, desc_prefix="") → List[dict]`

Runs batched inference. For each prompt, computes the full distribution over vocabulary at the final token, then reads off the probability for the actual final token.

**Output:** one dict per prompt, aligned with dataloader order:
```python
{
    "last_token": str,              # decoded string of the answer token, e.g. " bird"
    "last_token_id": int,           # token id
    "logprob": float,               # log P(answer_token | context)
    "probability": float,           # P(answer_token | context)
    "full_logprob_vector": Tensor,  # shape: (vocab_size,) — log softmax over all tokens
    "full_prob_vector": Tensor,     # shape: (vocab_size,) — softmax probabilities
}
```

The `full_prob_vector` is stored and used by all calibration methods. It is the full softmax distribution at the prediction position for that example.

---

## Step 4 — Attaching Probabilities (`eval_wrappers.py`)

### `attach_last_token_probs(dataset, model, batch_size=8, save_path=None, calibrate=False, evaluate=False, include_prob_info=False, k_tops=None, nucleus_ps=None, cal_methods=None, use_chat_template=False, yes_no=False) → list`

The main pipeline function. Orchestrates steps 2–6.

**Additional parameters:**
- `include_prob_info`: keep raw `prob_info` (logits etc.) in the output dict; stripped by default
- `k_tops`: list of k values for `cal_topkXXX` (default `[50]`)
- `nucleus_ps`: list of p values for `cal_nucleusXXX` / `cal_nucleus_entropyXXX` (default `[0.80, 0.85, 0.90, 0.95]`)
- `cal_methods`: subset of calibration method keys to compute (default: all)
- `use_chat_template`: apply the model's chat template before inference
- `yes_no`: use yes/no generation-prompt inference and `calibrate_true_false_dataset` instead of the standard token-prob path

1. Flattens all prompts via `flatten_prompt_groups`
2. Creates a `DataLoader` over flat texts and calls `model.last_token_prob`
3. Reconstructs per-example structure by restoring the mapping:

```python
{
    "constraint_type": str,
    "formula": str,
    "prompt_groups": {...},
    "ground_truth": {...},
    "prob_info": {
        "text_a": [prob_info_dict],   # list aligned with prompt_groups["text_a"]
        "text_b": [prob_info_dict],
    }
}
```

4. If `calibrate=True` (or `evaluate=True`): calls `calibrate_dataset`, which mutates `prob_info` in place by adding a `"calibration"` key to each prompt's dict.
5. If `evaluate=True`: calls `evaluate_dataset_constraints`, which replaces the list structure with a final results dict (see Step 6).
6. If `save_path` is given: pickles the result.

**Returns** either the intermediate list (if `evaluate=False`) or the full `final_results` dict (if `evaluate=True`).

---

## Step 5 — Calibration (`calibration.py`)

### Why calibrate?

Raw softmax probabilities from LLMs are poorly calibrated for this task — a model may assign high probability to many tokens, making P("bird") look artificially low even when the model strongly prefers "bird" relative to alternatives. Calibration adjusts the raw probability upward based on how "competitive" the vocabulary is at that position.

### Calibration formula

All methods use the same logit-shift formula:

```
calibrate(p, C) = sigmoid(logit(p) + C)
```

where `C` is a correction term that shifts the log-odds upward. Larger `C` = stronger upward push.

### `calibrate_prompt(prob_info, vocab_size, k_tops=[50], nucleus_ps=[0.80, 0.85, 0.90, 0.95], methods=None) → dict`

Computes all calibration variants for a single prompt's `prob_info`. `k_tops` and `nucleus_ps` are lists — each value produces a separately named key, so multiple parameter choices can coexist in the same result dict.

**Output (stored at `prob_info["calibration"]`):**
```python
{
    "raw_prob": float,                # no correction, direct softmax P
    "cal_vocab": float,               # C = log(vocab_size)
    "cal_topk050": float,             # C = log(k); key encodes k (e.g. cal_topk100 for k=100)
    "cal_nucleus095": float,          # C = log(nucleus_size at p); key encodes p (e.g. cal_nucleus090)
    "cal_entropy": float,             # C = H(full distribution)
    "cal_nucleus_entropy095": float,  # C = H(nucleus at p); key encodes p (e.g. cal_nucleus_entropy090)
    "cal_perplexity": float,          # C = exp(H(full distribution))

    "diagnostics": {
        "vocab_size": int,
        "nucleus_size": int,          # number of tokens in nucleus (using first nucleus_p)
        "entropy": float,             # full vocab entropy (nats)
        "perplexity": float,          # exp(entropy)
        "nucleus_entropy": float,     # entropy within nucleus (using first nucleus_p)
    },
}
```

**Calibration methods explained:**

| Key | C correction | Intuition |
|-----|-------------|-----------|
| `raw_prob` | 0 | Baseline, uncalibrated softmax |
| `cal_vocab` | `log(vocab_size)` | Corrects for competing against the entire vocabulary |
| `cal_topkXXX` | `log(k)` | Assumes k plausible tokens compete; key encodes k (e.g. `cal_topk050`) |
| `cal_nucleusXXX` | `log(nucleus_size)` | Uses actual nucleus size as competitive set; key encodes p (e.g. `cal_nucleus095`) |
| `cal_entropy` | `H(full_dist)` | Full entropy of vocab distribution as uncertainty estimate |
| `cal_nucleus_entropyXXX` | `H(nucleus_dist)` | Entropy restricted to nucleus tokens; key encodes p (e.g. `cal_nucleus_entropy095`) |
| `cal_perplexity` | `exp(H(full_dist))` | Perplexity as effective vocab size |

### `calibrate_dataset(dataset_list, vocab_size, k_tops=None, nucleus_ps=None, methods=None) → dataset_list`

Iterates all examples and prompt groups, applying `calibrate_prompt` to each `prob_info` dict. Mutates in place. Returns the same list.

---

## Step 6 — Evaluation (`evaluation.py`)

### WMC (Weighted Model Count)

For each example, we have calibrated probability estimates for 2–3 variables (e.g., P(A), P(B)). WMC computes the probability that the logical constraint formula is satisfied, treating the probabilities as independent Bernoulli weights over truth assignments.

### WMC functions

Each takes a `p_dict` of `{var_name: probability}` and optional `binary_threshold`:

| Function | Formula | Used for |
|----------|---------|----------|
| `xor_wmc(p_dict)` | `pA*(1-pB) + (1-pA)*pB` | mutual_exclusivity, negation_consistency |
| `implication_wmc(p_dict)` | `1 - pA*(1-pB)` | forward_implication, reverse_implication |
| `joint_wmc(p_dict)` | `pA * pB * ...` | joint_ secondary metric for all constraints |
| `transitivity_wmc(p_dict)` | `1 - pA*pB*(1-pC)` | transitivity |

If `binary_threshold` is provided, returns `1.0` if wmc >= threshold, else `0.0` (the "discrete" metric).

### Random-guessing baselines (`wmc_baseline(constraint_type)`)

The discrete threshold is set to the WMC under uniform 0.5 probabilities — the "random guessing" baseline:

| Constraint | WMC baseline |
|-----------|-------------|
| mutual_exclusivity, negation_consistency | 0.5 |
| forward_implication, reverse_implication | 0.75 |
| transitivity | 7/8 ≈ 0.875 |
| joint (n vars) | `1 / 2^n` |

A score of `discrete=1.0` means the model beats the random-guessing WMC for that constraint. `wmc_score` is the soft-normalized version: `(wmc_raw - baseline) / (1 - baseline)`, giving 0.0 at baseline and 1.0 at perfect constraint satisfaction.

#### TO DO:
* Investigate the 0.5 used for this random guessing baseline.

### `satisfying_assignment_entropy(p_dict, constraint_type) → float`

Computes the entropy (bits) of the distribution over *satisfying assignments* only. High entropy = the model's probability mass is spread across many satisfying assignments; low entropy = concentrated on one. This is used to diagnose how confidently a model satisfies a constraint.

### `evaluate_constraints(entry, prob_keys=None, constraint_types=None) → dict`

For a single example, computes all metrics across all calibration methods. Returns:
```python
{
    "mutual_exclusivity": {
        "raw_prob": {
            "is_ood": bool,          # True if no statement has P > 0.5 (model assigns no "true" answer)
            "wmc_raw": float,        # raw WMC value in [0, 1]
            "wmc_score": float,      # soft-normalized: (wmc_raw - baseline) / (1 - baseline); 0 = random, 1 = perfect
            "discrete": float,       # 1.0 if wmc_raw >= baseline else 0.0
            "entropy": float,        # entropy over satisfying assignments (bits)
            # joint_ = probability of all ground-truth-true statements being true (and false statements being false)
            # for xor constraints: P(A) * (1 - P(B)); for implication/transitivity: P(A) * P(B) * ...
            "joint_wmc_raw": float,
            "joint_wmc_score": float,
            "joint_discrete": float,
            "joint_entropy": float,
            "text_a": float,         # P(A) under this calibration method
            "text_b": float,         # P(B) under this calibration method
        },
        "cal_vocab": {...},
        "cal_topk050": {...},
        "cal_nucleus095": {...},
        "cal_entropy": {...},
        "cal_nucleus_entropy095": {...},
        "cal_perplexity": {...},
    }
}
```

For implication and transitivity, an additional `no_vacuous_*` group is included (`no_vacuous_wmc_raw`, `no_vacuous_wmc_score`, `no_vacuous_discrete`, `no_vacuous_entropy`), which excludes the all-false vacuous assignment from the WMC computation.

### `evaluate_dataset_constraints(dataset, prob_keys=None, constraint_types=None, include_prob_info=False) → dict`

Runs `evaluate_constraints` on every example and accumulates aggregate statistics.

**Returns:**
```python
{
    "dataset_with_evals": [
        {
            "constraint_type": str,
            "formula": str,
            "prompt_groups": {"text_a": [...], "text_b": [...]},
            "ground_truth": {"A": float, "B": float},
            "evaluations": {
                "raw_prob":                 {...metrics...},
                "cal_vocab":                {...metrics...},
                "cal_topk050":              {...metrics...},
                "cal_nucleus095":           {...metrics...},
                "cal_entropy":              {...metrics...},
                "cal_nucleus_entropy095":   {...metrics...},
                "cal_perplexity":           {...metrics...},
            }
            # "prob_info" only present if include_prob_info=True
        },
        ...
    ],
    "aggregate": {
        "num_samples": int,
        "raw_prob": {
            "wmc_raw":   {"mean": float, "std": float, "min": float, "max": float},
            "wmc_score": {"mean": float, "std": float, "min": float, "max": float},
            "discrete":  {"mean": float, "std": float, "min": float, "max": float},
            "entropy":   {...},
            "is_ood":    {...},
            "joint_wmc_raw": {...},
            "joint_wmc_score": {...},
            ...  # all metrics have the same stats structure
            # *_no_ood variants exclude OOD examples from aggregation
            "wmc_raw_no_ood":   {"mean": float, "std": float, "min": float, "max": float},
            "discrete_no_ood":  {...},
            ...
        },
        "cal_vocab":                {...},
        "cal_topk050":              {...},
        "cal_nucleus095":           {...},
        "cal_entropy":              {...},
        "cal_nucleus_entropy095":   {...},
        "cal_perplexity":           {...},
    }
}
```

`aggregate["raw_prob"]["discrete"]["mean"]` is the fraction of examples where the model beats the random-guessing WMC baseline. `aggregate["raw_prob"]["wmc_score"]["mean"]` is the soft-normalized version and the standard headline metric.

---

## Step 7 — Saving Results

### Pickle (intermediate, full results)

**Path:** `eval_results/{model_name}/{constraint}.pkl`

Contains the full `final_results` dict returned by `evaluate_dataset_constraints` (or `attach_last_token_probs` with `evaluate=True`). This is the primary intermediate artifact — it has everything needed to rerun plotting or recompute aggregates without re-running the model.

**To load:**
```python
import pickle
with open("eval_results/Gemma-3-270M/mutual_exclusivity.pkl", "rb") as f:
    results = pickle.load(f)

dataset_with_evals = results["dataset_with_evals"]  # list of per-example dicts
aggregate = results["aggregate"]                      # pre-computed stats
```

### CSV files (human-readable summaries)

**Aggregate CSV:** `eval_results/{model_name}/csv/{constraint}_aggregate.csv`

| calibration_method | metric_name | mean | std | min | max |
|-------------------|-------------|------|-----|-----|-----|
| raw_prob | wmc_score | 0.12 | 0.18 | -0.5 | 1.0 |
| raw_prob | discrete | 0.61 | 0.49 | 0.0 | 1.0 |
| cal_nucleus_entropy095 | wmc_score | ... | | | |
| ... | | | | | |

**Per-example CSV:** `eval_results/{model_name}/csv/{constraint}_per_example.csv`

One row per example. Columns include:
- `constraint_type`, `formula`, `gt_A`, `gt_B`
- `{group_name}_text` for each prompt group (e.g. `text_a_text`, `text_b_text`)
- `{cal_method}_{metric_name}` for all calibration × metric combinations (e.g. `cal_nucleus_entropy095_wmc_score`, `raw_prob_discrete`)
- `text_{group}_{cal_method}` for the probability values themselves (e.g. `text_a_raw_prob`)

### Plots

**Path:** `plots/{model_name}/{calibration_method}/{constraint}/{plot_type}.pdf`

Always saved as PDF (not PNG) for paper quality. Example:
```
plots/Gemma-3-270M/cal_nucleus_entropy095/mutual_exclusivity/scatterplot.pdf
plots/Gemma-3-270M/cal_nucleus_entropy095/mutual_exclusivity/histogram.pdf
plots/Gemma-3-270M/cal_nucleus_entropy095/mutual_exclusivity/wmc_score_distribution.pdf
plots/Gemma-3-270M/cal_nucleus_entropy095/mutual_exclusivity/wmc_raw_distribution.pdf
plots/Gemma-3-270M/cal_nucleus_entropy095/mutual_exclusivity/discrete_distribution.pdf
```

---

## Step 8 — Plots (`eval_plots.py`)

### `plot_entropy_vs_wmc(metric_dicts, title, save_path, metric=None)`

Scatter plot: x = WMC, y = entropy (over satisfying assignments) for each example. Expects the list returned by `extract_metric_dicts` (from `dataset_utils.py`). Use `metric` to select a sub-metric key prefix (e.g., `"joint"` → looks for `joint_wmc_raw` and `joint_entropy`). Default (`metric=None`) uses `wmc_raw` and `entropy`.

### `plot_entropy_histogram_by_wmc(metric_dicts, title, save_path, metric=None, bins=20)`

Bar chart: WMC is bucketed into 20 bins across [0,1], and each bar shows the mean entropy of examples in that WMC bin. Useful for understanding the relationship between confidence and constraint satisfaction.

### `plot_wmc_score_histogram(metric_dicts, title, save_path, metric=None, raw=False, discrete=False, bins=20)`

Histogram of per-example WMC score distribution (count vs WMC score). Shows how WMC scores are distributed across the dataset.

- `metric`: sub-metric prefix (e.g. `"joint"`, `"no_vacuous"`); `None` uses the primary WMC metric
- `raw=False`: plot `wmc_score` (baseline-normalized, can be negative for below-baseline models)
- `raw=True`: plot `wmc_raw` (raw WMC value, range [0,1])
- `discrete=True`: plot discrete binary scores (0 or 1); overrides `raw`; uses 2 bins

In master_wrapper, three distribution plots are generated per constraint × calibration method:
```
plots/{model_name}/{cal_method}/{constraint}/wmc_score_distribution.pdf    # baseline-normalized
plots/{model_name}/{cal_method}/{constraint}/wmc_raw_distribution.pdf      # raw WMC in [0,1]
plots/{model_name}/{cal_method}/{constraint}/discrete_distribution.pdf     # binary 0/1
```

### `plot_model_performance(df, metric_names, calibration_methods, constraints, log_x, label_points, combine_shape)`

Scatter plot of mean metric vs. model size (parameter count) across multiple models. Takes the master DataFrame built by `utils.build_master_dataframe`. Color = model family (Gemma, Qwen, etc.), shape = combination of varying variables (calibration method, constraint, metric). Use `log_x=True` for log-scale x-axis when comparing 135M–70B models.

---

## Plotting Helpers (`utils.py`, `results_to_df.py`)

### `results_to_df.build_aggregate_df(final_results) → pd.DataFrame`

Converts the `aggregate` dict from a single model/constraint run into a tidy DataFrame with columns: `calibration_method`, `metric_name`, `mean`, `std`, `min`, `max`.

### `results_to_df.build_per_example_df(final_results) → pd.DataFrame`

Converts `dataset_with_evals` into a wide DataFrame, one row per example.

### `utils.build_master_dataframe(base_dir="eval_results") → pd.DataFrame`

Reads all `*_aggregate.csv` files across all model subdirectories and concatenates them into a single DataFrame. Adds columns:
- `model`: model name (from directory name)
- `model_size_params`: integer parameter count extracted by regex from model name (e.g. `"Gemma-3-270M"` → `270000000`)
- `constraint`: constraint name (from filename)

This DataFrame is the input to `plot_model_performance` for cross-model analysis. 

---

## Model Registry (`model_paths.py`)

`MODEL_PATHS: Dict[str, str]` maps short model names to HuggingFace hub identifiers (or local snapshot paths). Currently registered models:

| Family | Models |
|--------|--------|
| Gemma-3 | 270M, 1B-pt, 4B-pt, 12B-pt |
| Llama-3 | 3.2-1B, 3.2-3B, 3.1-8B |
| Qwen3 | 0.6B, 1.7B, 4B, 8B, 14B |

To add a new model: add an entry to `MODEL_PATHS`. The key is the `model_name` string used throughout the pipeline; the value is the HF hub name or local path.

---

## Master Wrapper (`master_wrapper.py`)

### `master_wrapper(model_name, constraint_subset=None, calibration_methods_subset=None, cal_methods=None, aggregate_score_type="wmc_score", batch_size=128, instruction_tuned=False, eval_type="answer_token")`

End-to-end pipeline for a single model. Calls everything above in sequence.

**Parameters:**
- `model_name`: must be a key in `MODEL_PATHS`
- `constraint_subset`: list of constraint names to evaluate (default: all five)
- `calibration_methods_subset`: list of calibration method keys to include in plots (default: all)
- `cal_methods`: list of calibration method keys to compute (default: all); see `ALL_METHODS` in `calibration.py`
- `nucleus_ps`: nucleus p values for `cal_nucleusXXX` / `cal_nucleus_entropyXXX` (default: `[0.80, 0.85, 0.90, 0.95]`)
- `aggregate_score_type`: metric to display in plot titles; `"wmc_score"` is standard (soft-normalized mean)
- `batch_size`: inference batch size; `128` is fast on a single A100; reduce if OOM
- `instruction_tuned`: whether to apply the model's chat template during inference
- `eval_type`: `"answer_token"` (default, standard token-prob pass), `"qa"` (yes/no question format), or `"both"` (standard + yes/no merged)

**What it does:**
1. Creates output directories (`eval_results/{model_name}/`, `plots/{model_name}/`)
2. Loads `ModelWrapper`
3. Loads and filters all five constraint datasets using the model's tokenizer (or a subset if `constraint_subset` is given)
4. For each constraint: runs `attach_last_token_probs` (inference + calibration + evaluation), saves pickle
5. For each constraint × calibration method: generates scatter + histogram plots (PDF)
6. For each constraint: saves aggregate and per-example CSVs
7. Explicitly deletes `model.model` and calls `torch.cuda.empty_cache()` — the wrapper object must be deleted carefully in this order or GPU memory leaks

**Typical usage:**
```python
from src.master_wrapper import master_wrapper
master_wrapper(model_name="Gemma-3-270M", batch_size=128)
```

**Main run script**: `scripts/run_evals.py` — evaluates all models in `MODEL_PATHS` by default, with CLI flags for model subset, constraint subset, calibration methods, nucleus p values (`--nucleus-ps`), batch size, and eval type.


---

## Working with Intermediate Results

### Re-running plots without re-running inference

```python
import pickle
from src.dataset_utils import extract_metric_dicts
from src.eval_plots import plot_entropy_vs_wmc, plot_entropy_histogram_by_wmc

with open("eval_results/Gemma-3-270M/mutual_exclusivity.pkl", "rb") as f:
    results = pickle.load(f)

# Pick calibration method
metric_dicts = extract_metric_dicts(results["dataset_with_evals"], "cal_nucleus_entropy095")

plot_entropy_vs_wmc(metric_dicts, title="My Plot", save_path="my_plot.pdf")
```

### Re-running evaluation with different thresholds (no re-inference)

If you have a pickle that includes `prob_info` (i.e., saved before evaluation, or with `include_prob_info=True`), you can re-run evaluation:

```python
from src.evaluation import evaluate_dataset_constraints

# dataset_list here is the list of per-example dicts that still have prob_info attached
results = evaluate_dataset_constraints(dataset_list)
```

### Building cross-model aggregate tables

```python
from src.utils import build_master_dataframe

df = build_master_dataframe("eval_results")
# columns: model, model_size_params, constraint, calibration_method, metric_name, mean, std, min, max

# e.g., grab discrete WMC for all models on mutual_exclusivity under cal_nucleus_entropy
subset = df[
    (df.constraint == "mutual_exclusivity") &
    (df.calibration_method == "cal_nucleus_entropy095") &
    (df.metric_name == "discrete")
].sort_values("model_size_params")
```

### Accessing per-example probabilities from a pickle

```python
with open("eval_results/Gemma-3-270M/mutual_exclusivity.pkl", "rb") as f:
    results = pickle.load(f)

for entry in results["dataset_with_evals"]:
    # probabilities under each calibration method
    p_a = entry["evaluations"]["cal_nucleus_entropy095"]["text_a"]
    p_b = entry["evaluations"]["cal_nucleus_entropy095"]["text_b"]
    wmc = entry["evaluations"]["cal_nucleus_entropy095"]["wmc_score"]
    discrete = entry["evaluations"]["cal_nucleus_entropy095"]["discrete"]
```
