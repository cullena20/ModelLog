# From Token Probabilities to Semantic Constraints: Towards Declarative Probabilistic Evaluation of Language Models

Codebase for [*From Token Probabilities to Semantic Constraints: Towards Declarative Probabilistic Evaluation of Language Models*](https://arxiv.org/abs/2609.13520), accepted to EMNLP 2026.

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


## Citation

If you use this code, our datasets, or build on our findings, please cite our preprint

```bibtex
@misc{richardson2026tokenprobabilitiessemanticconstraints,
      title={From Token Probabilities to Semantic Constraints: Towards Declarative Probabilistic Evaluation of Language Models}, 
      author={Kyle Richardson and Cullen Anderson and Pranav Balakrishnan and Takuto Ban and Daksha Ladia and Ankita Gupta and Marisa Hudspeth},
      year={2026},
      eprint={2609.13520},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.13520}, 
}
```
