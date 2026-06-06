"""
Evaluation metrics for logical constraints, including WMC-based metrics and entropy of satisfying assignments.
"""

import numpy as np
import itertools

# For now this only works with up to three variables with hard coded constraints
def satisfying_assignment_entropy(p_dict, constraint_type, exclude_vacuous=False):
    """
    Compute entropy over satisfying assignments of a logical constraint.
    If exclude_vacuous=True, the all-zeros assignment is excluded (non-vacuous WMC).
    """
    vars_ = list(p_dict.keys())
    probs = list(p_dict.values())
    n = len(vars_)

    assignment_probs = []

    for assignment in itertools.product([0,1], repeat=n):
        if exclude_vacuous and all(v == 0 for v in assignment):
            continue

        prob = 1.0
        for val, p in zip(assignment, probs):
            prob *= p if val == 1 else (1 - p) # grab WMC corresponding to this assignment (may or may not fulfil assignment)

        A = assignment[0]
        B = assignment[1] if n >= 2 else None
        C = assignment[2] if n >= 3 else None

        satisfied = False

        if constraint_type in ["negation_consistency", "mutual_exclusivity"]:
            satisfied = (A != B) == 1

        elif constraint_type == "and_not":
            satisfied = (A == 1 and B == 0) # only one truthful assignment, so entropy always 0, not informative

        elif constraint_type in ["forward_implication", "reverse_implication"]:
            satisfied = (not A) or B

        elif constraint_type == "transitivity":
            satisfied = not (A == 1 and B == 1 and C == 0)

        elif constraint_type == "joint":
            satisfied = all(v == 1 for v in assignment) # only one truthful assignment, so entropy always 0, not informative
        
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        # only include WMC of satisfying assignments
        if satisfied:
            assignment_probs.append(prob)

    # sum up all WMC of satisfying assignments, which we normalize over
    total = sum(assignment_probs)

    if total == 0:
        return 0.0

    # normalized, so now we basically have a distribution over satisfying assignments, and we can compute entropy of that distribution
    normalized = [p/total for p in assignment_probs if p > 0]

    entropy = -sum(p * np.log2(p) for p in normalized)

    return entropy

# As we add more constraints, we can simply add to this (and write more general things as needed)
# Question: Is this really a random guessing baseline? Since probably 0.5 probability is actually pretty high -> goes back to calibration issues
def wmc_baseline(constraint_type, num_vars=None, exclude_vacuous=False):
    """
    Baseline WMC when all variable weights are 0.5.
    This thresholds the probability of a formula being true at a random assignment.
    The motivation is that the WMC scores to consider "good" should be above a random guessing baseline.
    If exclude_vacuous=True, returns the baseline for the no-vacuous variant (all-zeros excluded).
    """

    if exclude_vacuous:
        # implication no-vacuous: satisfying assignments are (0,1) and (1,1) -> WMC = pB, baseline = 0.5
        if constraint_type in ["forward_implication", "reverse_implication"]:
            return 0.5
        # transitivity no-vacuous: satisfying assignments exclude (0,0,0) and (1,1,0) -> WMC at p=0.5 is 6/8 = 0.75
        if constraint_type == "transitivity":
            return 0.75

    if constraint_type in ["negation_consistency", "mutual_exclusivity"]:
        return 0.5

    if constraint_type in ["forward_implication", "reverse_implication"]:
        return 0.75

    if constraint_type == "transitivity":
        return 7/8 # transitivity on (A and B) imply C, true except A true, B true, C false

    if constraint_type == "and_not":
        return 0.25

    if constraint_type == "joint":
        if num_vars is None:
            raise ValueError("num_vars must be specified for joint constraint")
        return 1 / (2 ** num_vars)

    raise ValueError(f"Unknown constraint type: {constraint_type}")

def xor_wmc(p_dict, binary_threshold=None):
    """Computes the XOR probability (pA*~pB + pB*~pA) for Mutex and Negation."""
    values = list(p_dict.values())
    if len(values) != 2: return None
    pA, pB = values[0], values[1]
    wmc = pA * (1 - pB) + (1 - pA) * pB
    if binary_threshold is not None:
        return 1.0 if wmc >= binary_threshold else 0.0
    else:
        return wmc

def and_not_wmc(p_dict, binary_threshold=None):
    """Computes the probability of A and NOT B (pA*~pB)."""
    values = list(p_dict.values())
    if len(values) < 2: return None
    # For implication and negation, we always use the first two variables provided
    pA, pB = values[0], values[1]
    wmc = pA * (1 - pB)
    if binary_threshold is not None:
        return 1.0 if wmc >= binary_threshold else 0.0
    else:
        return wmc

def implication_wmc(p_dict, binary_threshold=None, exclude_vacuous=False):
    """Computes the WMC probability for implication (1 - pA*~pB).
    If exclude_vacuous=True, excludes the all-False assignment"""
    values = list(p_dict.values())
    if len(values) != 2: return None
    pA, pB = values[0], values[1]
    if exclude_vacuous:
        # Satisfying (A→B) AND (A∨B): assignments (0,1) and (1,1) -> WMC = pB
        wmc = pA * pB + (1-pA) * pB # this simplifies to pB -> SO WEIRD MEASUREMENT
    else:
        wmc = 1 - (pA * (1 - pB))
    if binary_threshold is not None:
        return 1.0 if wmc >= binary_threshold else 0.0
    else:
        return wmc

def transitivity_wmc(p_dict, binary_threshold=None, exclude_vacuous=False):
    """Computes the WMC probability for transitivity (1 - pA*pB*~pC).
    If exclude_vacuous=True, excludes the all-False assignment:
    WMC = 1 - pA*pB*(1-pC) - (1-pA)*(1-pB)*(1-pC)."""
    values = list(p_dict.values())
    if len(values) != 3: return None
    pA, pB, pC = values[0], values[1], values[2]
    if exclude_vacuous:
        # Satisfying ¬(A∧B∧¬C) AND (A∨B∨C): all except (0,0,0) and (1,1,0)
        wmc = 1 - pA * pB * (1 - pC) - (1 - pA) * (1 - pB) * (1 - pC)
    else:
        wmc = 1 - (pA * pB * (1 - pC))
    if binary_threshold is not None:
        return 1.0 if wmc >= binary_threshold else 0.0
    else:
        return wmc

def joint_wmc(p_dict, binary_threshold=None):
    """Calculates the joint probability of all components being true"""
    # cullen: generalized to any number of values
    values = list(p_dict.values())
    wmc = 1
    for val in values:
        wmc *= val
    if binary_threshold is not None:
        return 1.0 if wmc >= binary_threshold else 0.0
    else:
        return wmc

def get_constraint_dict(entry, key):
    var_dict = {}
    for group_name, prompt_infos in entry["prob_info"].items():
        for idx, info in enumerate(prompt_infos):
            try:
                val = info[key]
            except KeyError:
                val = info.get("calibration", {}).get(key, None)

            var_name = group_name if len(prompt_infos) == 1 else f"{group_name}_{idx}"
            var_dict[var_name] = val
    return var_dict

