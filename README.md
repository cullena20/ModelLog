# From Token Probabilities to Semantic Constraints

Codebase for *From Token Probabilities to Semantic Constraints: Towards Declarative Probabilistic Evaluation of Language Models*.

## Overview

Codebase to evaluate how large language models satisfy logical semantic constraints using our proposed **ModelLog** framework. Includes five datasets extracted from knowledge graphs to test the following logical constraints: transitivity, forward implication, negation consistency, entity mutual exclusivity and spatial mutual exclusivity. 

## Structure

- `src/` — all evaluation pipeline code; see `src/PIPELINE.md` for a detailed breakdown
- `datasets/` — constraint evaluation datasets (JSON)
- `run_evals.py` — main entry point

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_evals.py                          # evaluate all models
python scripts/run_evals.py --models Gemma-3-270M    # single model
```

Results are saved to `eval_results/` (pickles + CSVs) and `plots/`.
