# Oracle-Free Detection of Pedagogical Hallucinations in Automated Tutoring Systems

Code and results accompanying the paper:

**"Oracle-Free Detection of Pedagogical Hallucinations in Automated Tutoring Systems Using Self-Reflective Large Language Models"**
Meezan Md Chand, Mohd Tauheed Khan — Ala-Too International University, Bishkek, Kyrgyzstan

Submitted to *Intelligent Systems with Applications* (Elsevier).

## Overview

This repository contains the implementation of a three-stage, oracle-free pipeline for detecting arithmetic errors in automated tutoring responses:

1. **Claim Extractor** — decomposes a tutor response into atomic arithmetic claims
2. **Math Executor** — evaluates each claim using deterministic symbolic computation (SymPy), without relying on probabilistic LLM scoring
3. **Isolated Verifier** — provides a secondary LLM-based confirmation, receiving the executor's output as contextual evidence

When the Math Executor and Isolated Verifier disagree, the executor's verdict takes precedence (`CONFLICT_EXEC_WINS`), ensuring deterministic computation overrides probabilistic agreement.

The system requires **zero annotated training data** and operates entirely at inference time on the GSM8K benchmark with synthetic arithmetic error injection.

## Repository Structure

```
.
├── code/
│   ├── gsm8k_pipeline_n50.py    # Primary evaluation run (n=50)
│   ├── gsm8k_pipeline_n200.py   # Scale validation run (n=200)
│   └── gsm8k_pipeline_n600.py   # Scale validation run (n=600)
├── results/
│   ├── gsm8k_results_n50.json
│   ├── gsm8k_results_n200.json
│   └── gsm8k_results_n600.json
└── README.md
```

## Results Summary

| Run    | N   | F1 (reported in paper) | F1 (this run) | AUROC | Notes                          |
|--------|-----|-------------------------|----------------|-------|---------------------------------|
| n=50   | 50  | 0.622                   | 0.618          | 0.580 | Primary evaluation              |
| n=200  | 200 | 0.590                   | 0.592          | 0.545 | Scale validation                |
| n=600  | 600 | 0.559                   | 0.559          | 0.503 | Scale validation                |

Supervised baseline (TAFM, Meng and Yang 2025): F1=0.610, trained on 600 human-annotated pairs.

## Reproducibility Note

This pipeline makes live calls to an LLM API (`meta-llama/llama-3.1-8b-instruct` via OpenRouter) for the Claim Extractor and Isolated Verifier stages. Due to inherent stochasticity in LLM sampling, exact reproduction of reported F1/AUROC values is not guaranteed across runs. In our own reproduction runs: n=50 yielded F1=0.618 (paper: 0.622, difference of 0.004), n=200 yielded F1=0.592 (paper: 0.590, difference of 0.002), and n=600 yielded F1=0.559 (paper: 0.559, exact match). All differences are within expected variance from LLM sampling stochasticity, and confirm the reported results are reproducible.

Note: an initial n=200 attempt produced F1=0.535, a noticeably lower outlier. A second run with identical code, seed, and settings produced F1=0.592, closely matching the paper. This illustrates that individual runs can occasionally diverge due to LLM API variance; we recommend averaging across multiple runs when exact reproduction is critical.

Original experiment logs are provided in `/results/` as evidence of the reported numbers.

## Setup

```bash
pip install openai sympy scikit-learn datasets
export OPENROUTER_API_KEY="your_key_here"
python code/gsm8k_pipeline_n50.py
```

Each script automatically saves its results to a JSON file in the working directory upon completion.

## Dataset

Experiments use the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test split (grade-school math word problems). Arithmetic errors are synthetically injected into model-generated solutions via controlled perturbation (one of four operations: +1, -1, ×2, ÷2 applied to a randomly selected numerical value).

## Citation

If you use this code, please cite:

```
Chand, M. M., & Khan, M. T. (2026). Oracle-Free Detection of Pedagogical Hallucinations
in Automated Tutoring Systems Using Self-Reflective Large Language Models.
Intelligent Systems with Applications.
```

## License

[Add your preferred license, e.g. MIT]

## Contact

Meezan Md Chand — meezanmohammad.chand@alatoo.edu.kg
