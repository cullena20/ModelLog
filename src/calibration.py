"""
Calibrate token probabilities using various methods.
"""

import math
import torch

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logit(p, eps=1e-12):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def calibrate(prob, C):
    return sigmoid(logit(prob) + C)

def entropy(probs):
    probs = probs.float()
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum().item()


def nucleus_size(probs, p=0.95):
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    return int((cumsum < p).sum().item() + 1)

# default method
def nucleus_entropy(probs, p=0.95):
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    mask = cumsum < p
    mask[0] = True

    nuc_probs = sorted_probs[mask]
    nuc_probs = nuc_probs / nuc_probs.sum()

    log_probs = torch.log(nuc_probs + 1e-12)
    return -(nuc_probs * log_probs).sum().item()

def perplexity(probs):
    log_probs = torch.log(probs + 1e-12)
    return torch.exp(-(probs * log_probs).sum()).item()

ALL_METHODS = frozenset(["raw_prob", "cal_vocab", "cal_topk", "cal_nucleus", "cal_entropy", "cal_nucleus_entropy", "cal_perplexity"])

def calibrate_prompt(prob_info, vocab_size, k_tops=None, nucleus_ps=None, methods=None):
    """
    Takes in prob_info, as returned by ModelWrapper.last_token_prob, and returns calibrated probabilities using various methods.

    Args:
        prob_info: dict from ModelWrapper.last_token_prob
        vocab_size: int
        k_tops: list of ints for top-k calibration (default [50]). Each k produces cal_topkXXX.
        nucleus_ps: list of floats for nucleus calibration (default [0.95]). Each p produces cal_nucleusXXX and cal_nucleus_entropyXXX.
        methods: subset of method names to include (default: all). Options: "raw_prob", "cal_vocab", "cal_topk", "cal_nucleus", "cal_entropy", "cal_nucleus_entropy".
    """
    if k_tops is None:
        k_tops = [50]
    if nucleus_ps is None:
        nucleus_ps = [0.95]
    if methods is None:
        methods = ALL_METHODS

    raw_p = prob_info["probability"]
    probs = prob_info["full_prob_vector"]

    results = {}

    # --- Raw ---
    if "raw_prob" in methods:
        results["raw_prob"] = raw_p

    # --- Proposal 1: vocab ---
    if "cal_vocab" in methods:
        results["cal_vocab"] = calibrate(raw_p, math.log(vocab_size))

    # --- Proposal 2: fixed top-k (one entry per k) ---
    if "cal_topk" in methods:
        for k in k_tops:
            results[f"cal_topk{k:03d}"] = calibrate(raw_p, math.log(k))

    # Compute nucleus stats needed by cal_nucleus and cal_nucleus_entropy
    nuc_computed = {}  # p -> (k_nuc, H_nuc)
    if "cal_nucleus" in methods or "cal_nucleus_entropy" in methods:
        for p in nucleus_ps:
            k_nuc = nucleus_size(probs, p=p)
            H_nuc = nucleus_entropy(probs, p=p)
            nuc_computed[p] = (k_nuc, H_nuc)

    # --- Proposal 3: nucleus size (one entry per p) ---
    if "cal_nucleus" in methods:
        for p in nucleus_ps:
            label = f"{int(p * 100):03d}"
            k_nuc, _ = nuc_computed[p]
            results[f"cal_nucleus{label}"] = calibrate(raw_p, math.log(k_nuc))

    # --- Proposal 4: entropy ---
    H = entropy(probs) # this is just log of perplexity
    if "cal_entropy" in methods:
        results["cal_entropy"] = calibrate(raw_p, H)

    # --- Proposal 4b: nucleus entropy (one entry per p) ---
    if "cal_nucleus_entropy" in methods:
        for p in nucleus_ps:
            label = f"{int(p * 100):03d}"
            _, H_nuc = nuc_computed[p]
            results[f"cal_nucleus_entropy{label}"] = calibrate(raw_p, H_nuc)

    # --- Proposal 5: perplexity (= exp(entropy)) as effective vocab ---
    ppl = perplexity(probs) # ppl is just exp(entropy)
    if "cal_perplexity" in methods:
        results["cal_perplexity"] = calibrate(raw_p, ppl)

    # Store diagnostics using first nucleus_p
    k_nuc_diag = nuc_computed[nucleus_ps[0]][0] if nuc_computed else None
    H_nuc_diag = nuc_computed[nucleus_ps[0]][1] if nuc_computed else None
    results["diagnostics"] = {
        "vocab_size": vocab_size,
        "nucleus_size": k_nuc_diag,
        "entropy": H,
        "perplexity": ppl,
        "nucleus_entropy": H_nuc_diag,
    }

    # Binary Metrics (use first values as defaults)
    results["is_top1"] = is_top1(prob_info)
    results["is_topk"] = is_topk(prob_info, k=k_tops[0])
    results["is_topp"] = is_topp(prob_info, p=nucleus_ps[0])

    return results

## BINARY EVALUATION METHODS

def is_top1(prompt_info):
    """
    Returns True if the answer token is the highest-probability token.
    """
    logits = prompt_info["full_logprob_vector"]
    answer_id = prompt_info["last_token_id"]

    top_id = torch.argmax(logits).item()

    return answer_id == top_id

def is_topk(prompt_info, k=50):
    """
    Returns True if answer token is among top-k tokens.
    """
    logits = prompt_info["full_logprob_vector"]
    answer_id = prompt_info["last_token_id"]

    topk_ids = torch.topk(logits, k).indices

    return answer_id in topk_ids

def is_topp(prompt_info, p=0.95):
    """
    Returns True if answer token lies within nucleus
    covering cumulative probability mass p.
    """

    logits = prompt_info["full_logprob_vector"]
    answer_id = prompt_info["last_token_id"]

    # Convert logits → probabilities safely
    probs = torch.softmax(logits.float(), dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Determine nucleus cutoff
    nucleus_mask = cumulative <= p
    nucleus_mask[0] = True  # always include highest

    nucleus_ids = sorted_indices[nucleus_mask]

    return answer_id in nucleus_ids

def calibrate_dataset(
    dataset_list,
    vocab_size,
    k_tops=None,
    nucleus_ps=None,
    methods=None,
):
    """
    Takes in dataset with prob_info key, as returned by eval_wrappers.attach_last_token_probs,
    and applies calibrate_prompt to each prompt's prob_info.

    Returns dataset with additional "calibration" key in each prompt's prob_info, containing calibrated probabilities and diagnostics.
    """

    for example in dataset_list:
        for group_name, prompt_list in example["prob_info"].items():

            for prompt_info in prompt_list:

                cal_results = calibrate_prompt(
                    prompt_info,
                    vocab_size=vocab_size,
                    k_tops=k_tops,
                    nucleus_ps=nucleus_ps,
                    methods=methods,
                )

                prompt_info["calibration"] = cal_results

    return dataset_list

# for question answering evaluation instead of answer token evaluation
def calibrate_true_false_dataset(
    dataset_list,
    tokenizer
):
    """
    Takes in dataset with prob_info key (from yes/no generation-prompt inference) and computes
    three calibrated probability methods based on Yes/No/True/False token probabilities.

    Methods produced per prompt_info:
      yes_no            : P(Yes) / (P(Yes) + P(No))
      true_false        : P(True) / (P(True) + P(False))
      yes_true_no_false : (P(Yes) + P(True)) / (P(Yes) + P(True) + P(No) + P(False))
    """

    EPS = 1e-12
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    true_token_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("False", add_special_tokens=False)[0]

    for example in dataset_list:
        for group_name, prompt_list in example["prob_info"].items():
            for prompt_info in prompt_list:
                pv = prompt_info["full_prob_vector"]
                yes_p  = pv[yes_token_id].item()  + EPS
                no_p   = pv[no_token_id].item()   + EPS
                true_p = pv[true_token_id].item()  + EPS
                false_p = pv[false_token_id].item() + EPS

                pos_p = yes_p + true_p
                neg_p = no_p  + false_p

                prompt_info["calibration"] = {
                    "yes_no":            yes_p  / (yes_p  + no_p),
                    "true_false":        true_p / (true_p + false_p),
                    "yes_true_no_false": pos_p  / (pos_p  + neg_p),
                }

    return dataset_list
