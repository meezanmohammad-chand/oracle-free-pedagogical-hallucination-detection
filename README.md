# Oracle-Free Mathematical Solution Verification for Reliable LLM Tutoring

Evaluation code and experimental results for the paper **"Oracle-Free Mathematical Solution Verification for Reliable LLM Tutoring: A Safeguard Against Pedagogical Hallucination"** (Chand, Khan & Sohail), currently under review at *Human-Centric Intelligent Systems* (Springer).

## Overview

Large language models are increasingly used as mathematics tutors, but a tutor that cannot reliably judge whether a student's solution is correct may confidently endorse a wrong answer. We study **oracle-free mathematical solution verification**: deciding whether a candidate solution is correct at inference time, *without* a reference answer, gold label, or task-specific training.

The method is a single four-stage pipeline:

1. **Claim extractor** — isolates the checkable content of a solution.
2. **Dual-mode executor** — evaluates explicit arithmetic *deterministically with SymPy* on GSM8K; on MathDial, where arithmetic is embedded in dialogue and no expression is exposed, it falls back to an *LLM claim-check*.
3. **Independent verifier** — re-solves the problem from scratch and compares.
4. **Symmetric conflict-resolution rule** — reconciles the two signals, deferring to the independent verifier in both directions and using the executor only as a fallback.

The pipeline requires neither a reference answer nor task-specific training at inference.

## Results

Evaluated across two error regimes: **MathDial** (real teacher-annotated tutoring errors) and **GSM8K** (controlled injected corruptions).

| Dataset  | F1 (mean ± sd)      | Runs | Notes |
|----------|---------------------|------|-------|
| MathDial | **0.9367 ± 0.0038** | 3    | Significantly beats Self-Consistency and CRITIC (McNemar *p* < 1e-7); +0.28 F1 over the oracle-free baseline of Daheim et al. |
| GSM8K    | **0.7241 ± 0.0346** | 3    | Beats Self-Consistency; statistically comparable to CRITIC (McNemar *p* = 0.36). |

An ablation shows the independent re-solving verifier is the primary signal, and that the symmetric conflict-resolution rule protects its precision where the two signals disagree.

### Reproducibility note

The verification pipelines call a hosted large language model (Llama 3.3 70B Instruct via OpenRouter). Because hosted LLM inference is not bit-for-bit deterministic and model endpoints evolve over time, re-running the code will produce results close to — but not necessarily identical to — the reported figures. The exact per-item outputs from the runs reported in the paper are archived under `results/` and are the definitive record of those runs.

## Repository structure

```
.
├── README.md
├── code/
│   ├── mathdial_pipeline.py     # MathDial: LLM claim-check executor + independent verifier + symmetric rule
│   └── gsm8k_pipeline.py        # GSM8K: SymPy symbolic executor + independent verifier
└── results/
    ├── mathdial/
    │   ├── run42.json
    │   ├── run456.json
    │   └── run789.json
    └── gsm8k/
        ├── seed42.json
        ├── seed456.json
        └── seed789.json
```

Only runs with archived per-item result files are included here.

## Datasets

This repository does not aim to redistribute the full datasets; obtain them from their original sources under their respective licenses.

- **MathDial** — a dialogue tutoring dataset by Macina et al., *Findings of EMNLP 2023* ([github.com/eth-nlped/mathdial](https://github.com/eth-nlped/mathdial)), released under **CC BY-SA 4.0**. The per-turn error annotations and the specific data used here are from the "verify-then-generate" release of Daheim et al., *Findings of EMNLP 2024* ([github.com/eth-lre/verify-then-generate](https://github.com/eth-lre/verify-then-generate)), also **CC BY-SA 4.0**. A stratified 600-instance sample (300 positive / 300 negative) is used.
- **GSM8K** — Cobbe et al., 2021 ([github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)), **MIT License**. Controlled single-value numerical corruptions.

## Running the pipelines

The pipelines call **Llama 3.3 70B Instruct** via the [OpenRouter](https://openrouter.ai/) API.

```bash
pip install requests sympy
export OPENROUTER_API_KEY="your-key-here"

python code/gsm8k_pipeline.py      # SymPy required
python code/mathdial_pipeline.py
```

Runs checkpoint periodically and write per-item JSON results to `results/`.

## Citation

```bibtex
@unpublished{chand2026oraclefree,
  title  = {Oracle-Free Mathematical Solution Verification for Reliable LLM Tutoring: A Safeguard Against Pedagogical Hallucination},
  author = {Chand, Meezan Md and Khan, Mohd Tauheed and Sohail, Shahab Saquib},
  year   = {2026},
  note   = {Under review at Human-Centric Intelligent Systems (Springer)}
}
```

## License

- **Code** (`code/`) — MIT License (see `LICENSE`).
- **GSM8K** (Cobbe et al., 2021) is distributed by OpenAI under the MIT License.
- **MathDial** (Macina et al., 2023) and the error annotations of Daheim et al. (2024) are distributed under **CC BY-SA 4.0**. Any MathDial-derived files in this repository (e.g. per-item results under `results/mathdial/`) are therefore made available under **CC BY-SA 4.0**, with attribution to Macina et al. (2023).
```
