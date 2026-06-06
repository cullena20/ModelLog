"""
Process dataset for evaluation, including filtering out unfit examples, 
unifying format across different constraint types, and flattening prompt groups for batched inference.
"""

from datasets import load_dataset

def process_prompt(prompt):
    """
    Process prompt to ensure that the final token is the answer token we want to evaluate.
    For current dataset, we have periods at the end, so this strips them
    """
    if prompt.endswith("."):
        return prompt[:-1]
    else:
        return prompt

def filter_unfit(example):
    """
    New GPT cleaned dataset has binary "is_fit" attribute.
    If False, do not keep these examples.
    If attribute is missing, just keep the example.
    """
    if "is_fit" not in example:
        return True
    if not example["is_fit"]:
        return False
    else:
        return True

def filter_multi_token(example, tokenizer):
    """
    Filters out examples where the trigger word is tokenized into multiple tokens.
    Inefficient implementation based on current dataset always keeping these as trigger_word_X keys.
    """
    for key, value in example.items():
        if key.startswith("trigger_word"):
            value = " " + value.strip() # This is correct tokenization, also without significantly more examples are filtered out (e.g. "moss" is two tokens for SmolLM, " moss" is one)
            #print(value)
            tokens = tokenizer(value, add_special_tokens=False)["input_ids"]
            if len(tokens) != 1:
                return False
    return True


def unify_example(example):
    """
    Unify format for all dataset examples.
    We want answer token probabilities for every single prompt contained within prompt groups
    """
    ctype = example["constraint_type"]

    if ctype == "mutual_exclusivity":
        prompt_groups = {
            "text_a": [process_prompt(example["text_a"])],
            "text_b": [process_prompt(example["text_b"])],
        }

    elif ctype == "negation_consistency":
        prompt_groups = {
            "text_a": [process_prompt(example["text_a"])],
            "text_b": [process_prompt(example["text_b"])],
        }

    elif ctype == "forward_implication":
        prompt_groups = {
            "text_a": [process_prompt(example["text_a"])],
            "text_b": [process_prompt(example["text_b"])],
        }

    elif ctype == "reverse_implication":
        prompt_groups = {
            "text_not_b": [process_prompt(example["text_not_b"])],
            "text_not_a": [process_prompt(example["text_not_a"])],
        }

    elif ctype == "transitivity":
        prompt_groups = {
            "text_a": [process_prompt(example["text_a"])],
            "text_b": [process_prompt(example["text_b"])],
            "text_c": [process_prompt(example["text_c"])]
        }

    else:
        raise ValueError(f"Unknown constraint type: {ctype}")

    return {
        "constraint_type": ctype,
        "formula": example["formula"],
        "prompt_groups": prompt_groups,
        "ground_truth": example["ground_truth"],
        # "raw_example": example,
    }


def load_and_unify(json_path, filter_fit=True, tokenizer=None):
    """
    Provide dataset path, get back unified dataset with format specified in unify_example.
    filter_fit=True filters out examples with is_fit=False (GPT annotator deems data bad)
    tokenizer supplied means we also filter out examples where trigger words are tokenized into multiple tokens, since we won't be able to evaluate those properly.
    """
    dataset = load_dataset("json", data_files=json_path)["train"]
    if filter_fit:
        dataset = dataset.filter(filter_unfit)
    if tokenizer:
        dataset = dataset.filter(lambda example: filter_multi_token(example, tokenizer))
    return dataset.map(unify_example, remove_columns=dataset.column_names) #don't keep original columns, these are redundant

def flatten_prompt_groups(dataset):
    """
    Flatten all prompts for evaluation contained with prompt_groups.
    This allows for batched inference.
    Mapping allows mapping these back to original dataset examples, and also allows us to keep track of which prompt group they came from (e.g. text_a vs text_b)
    """
    flat_texts = []
    mapping = []  # (example_idx, group_name, prompt_idx)

    for ex_idx, example in enumerate(dataset):
        for group_name, texts in example["prompt_groups"].items():
            for p_idx, text in enumerate(texts):
                flat_texts.append(text)
                mapping.append((ex_idx, group_name, p_idx))

    return flat_texts, mapping


# the output of attach_last_token_probs at ["dataset_with_evals"]
# this is a list of dictionaries with examples, with lots of extraneous information
# this extracts a dictionary of metrics to probs for a given calibration method
def extract_metric_dicts(dataset, prob_key="raw_prob"):
    """
    Convert dataset entries into a flat list of metric dictionaries
    for a given probability calibration method.

    Returns:
        List[Dict[str, float]]
    """

    cleaned = []

    for entry in dataset:
        evals = entry.get("evaluations", {})
        if not evals:
            continue

        metrics = evals.get(prob_key)
        if metrics is None:
            continue

        filtered = {
            k: float(v)
            for k, v in metrics.items()
            if not k.startswith("text_") and v is not None
        }

        cleaned.append(filtered)

    return cleaned
