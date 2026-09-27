# ============================================================
# GSM8K Verification Pipeline — Symbolic Executor + Independent Verifier
# Oracle-Free Mathematical Solution Verification for Reliable LLM Tutoring
#
# Model   : Llama 3.3 70B Instruct via OpenRouter
# Dataset : GSM8K test split (Cobbe et al., 2021; OpenAI, MIT License),
#           with controlled single-value arithmetic corruptions injected to
#           create balanced with-error / no-error pairs.
# Metrics : F1, AUROC, Accuracy, Precision, Recall, Cohen's Kappa,
#           bootstrap 95% CI.
#
# Stage 2 (executor) evaluates explicit arithmetic DETERMINISTICALLY with
# SymPy: the LLM only copies a verbatim expression, SymPy computes it, and a
# +/-1.0 (or 1%) tolerance decides. Stage 3 is an independent re-solving
# verifier. Stage 4 conflict resolution is SYMMETRIC verifier-wins: the final
# decision follows the independent verifier; the executor contributes a graded
# confidence signal (combined_score) but does not override the verifier.
# ============================================================
# SEEDS   : 42, 456, 789 (each seed parametrizes the sampled 200-item subset)
# ============================================================

import os
import time
import json
import re
import random
import csv
import getpass
import numpy as np
import sympy
from sympy import N
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from openai import OpenAI
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)
from datasets import load_dataset

# ── CONFIGURATION ──────────────────────────────────────────
# Set OPENROUTER_API_KEY as an environment variable, or type it at the prompt.
API_KEY  = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter API key: ")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.3-70b-instruct"
N_EXAMPLES = 200          # Primary evaluation size per seed
SEEDS      = [42, 456, 789]
BOOTSTRAP_ITERS = 1000    # For 95% CI
SMOKE_TEST = False        # True = quick sanity run on a small subset
SMOKE_SEED = 123
SMOKE_N    = 20
CHECKPOINT_EVERY = 20     # Save partial results every N examples

if SMOKE_TEST:
    N_EXAMPLES = SMOKE_N
    SEEDS      = [SMOKE_SEED]

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers={
        "HTTP-Referer": "https://github.com/meezanmohammad-chand",
        "X-Title": "GSM8K Verification Pipeline"
    }
)

print("=" * 60)
print("GSM8K VERIFICATION PIPELINE — LLAMA 3.3 70B INSTRUCT")
print(f"Model   : {MODEL}")
print(f"Dataset : GSM8K test split")
print(f"N       : {N_EXAMPLES} per seed {'(SMOKE TEST)' if SMOKE_TEST else ''}")
print(f"Seeds   : {SEEDS}")
print("=" * 60)

print("\nLoading GSM8K dataset...")
dataset = load_dataset("openai/gsm8k", "main", split="test")
print(f"Loaded {len(dataset)} examples")


# ── ERROR INJECTION ─────────────────────────────────────────
def inject_arithmetic_error(solution_text):
    lines       = solution_text.split("\n")
    working     = "\n".join(l for l in lines if not l.strip().startswith("####"))
    matches     = list(re.finditer(r"\b(\d+)\b", working))
    if not matches:
        return solution_text, False

    candidates = [m for m in matches if int(m.group()) > 1]
    if not candidates:
        candidates = matches

    target      = random.choice(candidates)
    original    = int(target.group())

    error_type  = random.choice(["plus1", "minus1", "double", "half", "plus10", "minus10"])
    if error_type == "plus1":
        wrong = original + 1
    elif error_type == "minus1":
        wrong = max(0, original - 1)
    elif error_type == "double":
        wrong = original * 2
    elif error_type == "half":
        wrong = max(1, original // 2)
    elif error_type == "plus10":
        wrong = original + 10
    else:  # minus10
        wrong = max(0, original - 10)

    s, e     = target.start(), target.end()
    modified = working[:s] + str(wrong) + working[e:]
    return modified, True


# ── DATA BUILDER ────────────────────────────────────────────
def extract_steps(answer_text):
    lines = answer_text.split("\n")
    steps = [l.strip() for l in lines if l.strip() and not l.strip().startswith("####")]
    return " ".join(steps)[:500]


def build_sample(n, seed):
    random.seed(seed)
    half    = n // 2
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    correct_idx = indices[:half]
    error_idx   = indices[half : half * 2]

    examples = []
    for i, idx in enumerate(correct_idx):
        ex = dataset[idx]
        examples.append({
            "id": i,
            "student_question": ex["question"][:300],
            "tutor_response":   extract_steps(ex["answer"]),
            "contains_error":   0,
            "original_answer":  ex["answer"][:200],
        })
    for i, idx in enumerate(error_idx):
        ex       = dataset[idx]
        steps    = extract_steps(ex["answer"])
        modified, changed = inject_arithmetic_error(steps)
        examples.append({
            "id": half + i,
            "student_question": ex["question"][:300],
            "tutor_response":   modified if changed else steps,
            "contains_error":   1 if changed else 0,
            "original_answer":  ex["answer"][:200],
        })
    random.shuffle(examples)
    return examples


# ── LLM CALL ────────────────────────────────────────────────
def call_llm(prompt, temperature=0.3):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=max(temperature, 0.01),
                max_tokens=512,
            )
            if resp and resp.choices and len(resp.choices) > 0:
                content = resp.choices[0].message.content
                if content:
                    return content.strip()
            print(f"  Attempt {attempt + 1} failed: empty response")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return ""


# ── VERBATIM GUARD ──────────────────────────────────────────
def _normalize_math(text):
    """Strip whitespace/commas, unify operator symbols, for comparison."""
    t = str(text).replace(",", "")
    t = t.replace("×", "*").replace("x", "*").replace("÷", "/")
    t = re.sub(r"\s+", "", t)
    return t


def _expression_is_verbatim(expr_str, source_text):
    """Reject expressions the extractor fabricated. The extracted expression
    must appear as a CONTIGUOUS string in the source text (whitespace/comma-
    normalized). This catches structural fabrications like '2*20/2' invented
    from 'Half of 20 is 10', which a numbers-only check would miss. GSM8K
    solutions write arithmetic explicitly (e.g. '48/2 = 24' or '<<48/2=24>>'),
    so legitimate expressions survive this check."""
    try:
        expr_norm   = _normalize_math(expr_str)
        source_norm = _normalize_math(source_text)
        if expr_norm in source_norm:
            return True
        if "=" in expr_norm and expr_norm.split("=", 1)[0] in source_norm:
            return True
        return False
    except Exception:
        return True  # never block on guard failure itself


# ── MATH EXECUTOR (SymPy) ───────────────────────────────────
def math_executor(claim, source_text=None):
    guard_text = source_text if source_text else claim
    prompt = f"""You are a math expression extractor. You COPY, you never compute or rewrite.

Claim: {claim}

Rules:
- ONLY extract if there is an EXPLICIT arithmetic equation WRITTEN IN THE CLAIM (e.g. "35 + 25 = 60")
- Copy the expression VERBATIM, character for character, exactly as it appears
- NEVER paraphrase words into math. "half of 20", "a third of 9", "double it" → NONE
- NEVER invent, infer, complete, or correct any calculation
- Every number in your EXPRESSION must literally appear in the claim text
- Both expression AND result must be numeric, no words allowed
- If ANY ambiguity exists → return NONE
- Expression must be simple arithmetic: +, -, *, /

Reply EXACTLY:
EXPRESSION: [verbatim calculation or NONE]
STATED_RESULT: [number or NONE]"""

    raw             = call_llm(prompt, temperature=0.0)
    extracted_expr  = None
    stated_result   = None

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("EXPRESSION:"):
            val = line.replace("EXPRESSION:", "").strip()
            extracted_expr = None if val.upper() == "NONE" else val
        elif line.startswith("STATED_RESULT:"):
            val = line.replace("STATED_RESULT:", "").strip()
            stated_result = None if val.upper() == "NONE" else val

    if not extracted_expr or not stated_result:
        return {"math_error": False, "executor_verdict": "NO_MATH", "error_signal": 0.0}

    # verbatim guard — reject fabricated expressions
    if not _expression_is_verbatim(extracted_expr, guard_text):
        print(f"    [Guard] Rejected fabricated expression: {extracted_expr} (numbers not in claim)")
        return {"math_error": False, "executor_verdict": "NO_MATH", "error_signal": 0.0}

    computed = _safe_compute(extracted_expr)
    if computed is None:
        return {"math_error": False, "executor_verdict": "NO_MATH", "error_signal": 0.0}

    is_error = _compare_results(stated_result, computed)
    verdict  = "ERROR" if is_error else "CORRECT"
    print(f"    [MathExec] {extracted_expr} | Stated:{stated_result} | Got:{computed} → {verdict}")
    return {
        "math_error":       is_error,
        "executor_verdict": verdict,
        "error_signal":     1.0 if is_error else 0.0,
    }


def _safe_compute(expr_str):
    expr_str = expr_str.strip()
    if "=" in expr_str:
        parts = expr_str.split("=", 1)
        try:
            tf  = standard_transformations + (implicit_multiplication_application,)
            lhs = parse_expr(parts[0].strip(), transformations=tf)
            rhs = parse_expr(parts[1].strip(), transformations=tf)
            if sympy.simplify(lhs - rhs) == 0:
                return float(N(rhs))
        except Exception:
            pass
    try:
        tf     = standard_transformations + (implicit_multiplication_application,)
        result = parse_expr(expr_str, transformations=tf)
        return float(N(result))
    except Exception:
        pass
    try:
        # NEVER letter-strip-and-eval expressions containing variables.
        # '4x + x = 40' would otherwise become '4 + 40' = 44 → fake ERROR.
        if not re.search(r"[a-zA-Z]", expr_str):
            safe = re.sub(r"[^0-9+\-*/().\s]", "", expr_str)
            if safe.strip():
                return float(eval(safe, {"__builtins__": {}}))
    except Exception:
        pass
    return None


def _compare_results(stated_str, computed_float):
    try:
        nums = re.findall(r"-?\d+\.?\d*", str(stated_str).replace(",", ""))
        if not nums:
            return False
        for num_str in reversed(nums):
            stated = float(num_str)
            diff   = abs(stated - computed_float)
            if diff <= 1.0:
                return False
            if computed_float != 0 and diff / abs(computed_float) <= 0.01:
                return False
        return True
    except Exception:
        return False


# ── CLAIM EXTRACTOR ─────────────────────────────────────────
def claim_extractor(tutor_response):
    prompt = f"""Extract every factual or mathematical claim from this tutoring response.
Return ONLY a numbered list. Each item = one atomic claim. No explanation.

Tutoring response: {tutor_response}

Numbered list of claims:"""
    raw    = call_llm(prompt, temperature=0.1)
    lines  = [l.strip() for l in raw.split("\n") if l.strip()]
    claims = [l.lstrip("0123456789.-) ").strip() for l in lines if l.lstrip("0123456789.-) ").strip()]
    return claims if claims else [tutor_response]


# ── HOLISTIC VERIFIER ───────────────────────────────────────
# One call per example. The verifier solves the problem independently FIRST,
# then compares its own solution to the tutor's response. Neutral labels
# (CONSISTENT/INCONSISTENT) avoid error-priming. No executor hint — signals
# stay independent until conflict resolution.
def holistic_verifier(question, tutor_response):
    prompt = f"""You are checking a math tutor's response for a student.

Work in two phases, in this exact order:

PHASE 1 — Solve the problem yourself, step by step. Do this from the
question alone, before reading the tutor's response below.

Question: {question}

PHASE 2 — Now check the tutor's response NUMBER BY NUMBER:

Tutor's response: {tutor_response}

Go through EVERY number stated in the tutor's response, one at a time.
For each number, check: does it follow correctly from the question and
the previous steps? Does it match the corresponding value in YOUR
solution from Phase 1?

The response is INCONSISTENT if ANY stated number is wrong — even an
intermediate number, and even if the final answer happens to be right.
A single wrong number anywhere makes the whole response INCONSISTENT.
Name the wrong number (e.g. "tutor states 156 but 52+57=109").

Differences in wording, order of steps, or level of detail do not matter.
Only the numbers matter.

Reply EXACTLY:
MY_SOLUTION: [your final answer number]
VERDICT: [CONSISTENT or INCONSISTENT]
MISMATCH: [the specific wrong number/calculation, or NONE]"""

    raw     = call_llm(prompt, temperature=0.1)
    verdict = "UNKNOWN"
    for line in raw.split("\n"):
        if line.strip().startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
            break
    flags = "INCONSISTENT" in verdict.upper()
    return {"isolated_verdict": "INCORRECT" if flags else "CORRECT",
            "flags_error": flags,
            "raw_verdict": verdict}


# ── CONFLICT RESOLUTION (SYMMETRIC verifier-wins) ───────────
# The final decision follows the independent verifier in both directions.
# The executor's deterministic signal informs the graded combined_score
# (below) but does not override the verifier.
def feedback_quality_filter(math_result, verif_result):
    verif_flags = verif_result["flags_error"]
    prediction  = 1 if verif_flags else 0
    quality     = "VERIFIER_INCORRECT" if verif_flags else "VERIFIER_CORRECT"
    return {"final_prediction": prediction, "quality": quality}


# ── PIPELINE PER EXAMPLE ────────────────────────────────────
def run_pipeline(example, run_num, total):
    print(f"\n[{run_num}/{total}] GT: {'ERROR' if example['contains_error'] else 'CORRECT'}")

    tutor = example["tutor_response"]
    if not tutor:
        return None

    # Step 1 — Claim Extractor
    claims = claim_extractor(tutor)
    print(f"  Claims extracted: {len(claims)}")

    # Step 2 — Math Executor on ALL claims (pass original tutor text so the
    # verbatim guard checks against the true source, not a paraphrased claim)
    all_math = []
    for i, claim in enumerate(claims):
        r = math_executor(claim, source_text=tutor)
        all_math.append(r)
        if r["math_error"]:
            print(f"  *** ERROR found in claim {i+1} — stopping early ***")
            break

    direct = math_executor(tutor[:300])

    any_error  = any(r["math_error"] for r in all_math) or direct["math_error"]
    any_fired  = any(r["executor_verdict"] != "NO_MATH" for r in all_math) or direct["executor_verdict"] != "NO_MATH"

    math_agg = {
        "math_error":       any_error,
        "executor_verdict": "ERROR" if any_error else ("CORRECT" if any_fired else "NO_MATH"),
        "error_signal":     1.0 if any_error else 0.0,
    }

    # Step 3 — Holistic Verifier (one independent solve-then-compare, no hint)
    v = holistic_verifier(example["student_question"], tutor)
    verif_agg = {"flags_error": v["flags_error"],
                 "isolated_verdict": v["isolated_verdict"]}
    print(f"  Verifier: {v['isolated_verdict']}")

    # Step 4 — Symmetric conflict resolution
    filt = feedback_quality_filter(math_agg, verif_agg)

    # Graded combined score for AUROC:
    #   verifier flags error            -> 1.0
    #   else executor found a math error -> 0.3
    #   else                             -> 0.0
    if verif_agg["flags_error"]:
        combined_score = 1.0
    elif math_agg["math_error"]:
        combined_score = 0.3
    else:
        combined_score = 0.0

    correct = filt["final_prediction"] == example["contains_error"]
    print(f"  Pred: {'ERROR' if filt['final_prediction'] else 'CORRECT'} | {'✓' if correct else '✗'}")

    return {
        "id":           example["id"],
        "ground_truth": example["contains_error"],
        "prediction":   filt["final_prediction"],
        "correct":      correct,
        "combined_score": combined_score,
        "math_executor": math_agg,
        "verification":  verif_agg,
        "filter":        filt,
    }


# ── BOOTSTRAP CI ────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, n_iter=1000, ci=95):
    np.random.seed(42)
    scores = []
    n      = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        try:
            scores.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        except Exception:
            pass
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return round(lower, 4), round(upper, 4)


# ── METRICS COMPUTATION ─────────────────────────────────────
def compute_metrics(results):
    y_true   = [r["ground_truth"] for r in results]
    y_pred   = [r["prediction"]   for r in results]
    y_scores = [r.get("combined_score", r["math_executor"]["error_signal"]) for r in results]

    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except Exception:
        auroc = None

    ci_lower, ci_upper = bootstrap_ci(y_true, y_pred, BOOTSTRAP_ITERS)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    no_math = sum(1 for r in results if r["math_executor"]["executor_verdict"] == "NO_MATH")

    return {
        "n":          len(results),
        "f1":         round(f1, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "accuracy":   round(accuracy, 4),
        "auroc":      round(auroc, 4) if auroc else None,
        "kappa":      round(kappa, 4),
        "ci_95":      (ci_lower, ci_upper),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "net_benefit":  tp - fp,
        "no_math_rate": round(no_math / len(results), 4),
    }


# ── PRINT METRICS ───────────────────────────────────────────
def print_metrics(m, seed):
    print(f"\n{'='*60}")
    print(f"SEED {seed} RESULTS (n={m['n']})")
    print(f"{'='*60}")
    print(f"F1 Score      : {m['f1']:.4f}  (95% CI: {m['ci_95'][0]:.4f}–{m['ci_95'][1]:.4f})")
    print(f"Precision     : {m['precision']:.4f}")
    print(f"Recall        : {m['recall']:.4f}")
    print(f"Accuracy      : {m['accuracy']:.4f}")
    print(f"AUROC         : {m['auroc']}")
    print(f"Cohen's Kappa : {m['kappa']:.4f}")
    print(f"Confusion     : TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    print(f"Net benefit   : {m['net_benefit']}")
    print(f"NO_MATH rate  : {m['no_math_rate']*100:.1f}%")


# ── CHECKPOINT HELPERS ──────────────────────────────────────
def checkpoint_path(seed):
    return f"gsm8k_seed{seed}_n{N_EXAMPLES}_checkpoint.json"


def load_checkpoint(seed):
    path = checkpoint_path(seed)
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
        print(f"\n>>> Checkpoint found: resuming from example {len(results) + 1}")
        return results
    return []


def save_checkpoint(seed, results):
    with open(checkpoint_path(seed), "w") as f:
        json.dump(results, f, indent=2, default=str)


# ── MAIN MULTI-SEED RUN ─────────────────────────────────────
all_seed_metrics = []
all_seed_results = {}

for seed in SEEDS:
    print(f"\n{'#'*60}")
    print(f"# SEED {seed} — Building {N_EXAMPLES} examples")
    print(f"{'#'*60}")

    examples = build_sample(N_EXAMPLES, seed)

    # resume from checkpoint if the runtime crashed mid-run.
    # build_sample is deterministic given seed, so skipping by count is safe.
    results = load_checkpoint(seed)
    start_i = len(results)

    for i, ex in enumerate(examples):
        if i < start_i:
            continue
        r = run_pipeline(ex, i + 1, N_EXAMPLES)
        if r:
            results.append(r)
        if len(results) % CHECKPOINT_EVERY == 0:
            save_checkpoint(seed, results)
            print(f"  --- checkpoint saved ({len(results)} examples) ---")
        time.sleep(1)

    save_checkpoint(seed, results)
    metrics = compute_metrics(results)
    print_metrics(metrics, seed)
    all_seed_metrics.append(metrics)
    all_seed_results[seed] = results

    fname = f"gsm8k_seed{seed}_n{N_EXAMPLES}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {fname}")

    time.sleep(2)


# ── AGGREGATE ACROSS SEEDS ──────────────────────────────────
print(f"\n{'='*60}")
print(f"AGGREGATE RESULTS ACROSS {len(SEEDS)} SEED(S)")
print(f"{'='*60}")

f1_vals        = [m["f1"]        for m in all_seed_metrics]
precision_vals = [m["precision"] for m in all_seed_metrics]
recall_vals    = [m["recall"]    for m in all_seed_metrics]
accuracy_vals  = [m["accuracy"]  for m in all_seed_metrics]
kappa_vals     = [m["kappa"]     for m in all_seed_metrics]
auroc_vals     = [m["auroc"]     for m in all_seed_metrics if m["auroc"]]

mean_f1   = round(np.mean(f1_vals), 4)
std_f1    = round(np.std(f1_vals, ddof=1), 4) if len(f1_vals) > 1 else 0.0
mean_prec = round(np.mean(precision_vals), 4)
mean_rec  = round(np.mean(recall_vals), 4)
mean_acc  = round(np.mean(accuracy_vals), 4)
mean_kap  = round(np.mean(kappa_vals), 4)
mean_auc  = round(np.mean(auroc_vals), 4) if auroc_vals else None

print(f"\nMetric         Mean     Std      Seeds: {SEEDS}")
print(f"{'─'*50}")
print(f"F1             {mean_f1:.4f}   ±{std_f1:.4f}   {[m['f1'] for m in all_seed_metrics]}")
print(f"Precision      {mean_prec:.4f}            {[m['precision'] for m in all_seed_metrics]}")
print(f"Recall         {mean_rec:.4f}            {[m['recall'] for m in all_seed_metrics]}")
print(f"Accuracy       {mean_acc:.4f}            {[m['accuracy'] for m in all_seed_metrics]}")
print(f"Cohen Kappa    {mean_kap:.4f}            {[m['kappa'] for m in all_seed_metrics]}")
if mean_auc:
    print(f"AUROC          {mean_auc:.4f}")

# ── SAVE AGGREGATE ──────────────────────────────────────────
aggregate = {
    "model":       MODEL,
    "dataset":     "GSM8K test split",
    "n_per_seed":  N_EXAMPLES,
    "seeds":       SEEDS,
    "smoke_test":  SMOKE_TEST,
    "mean_f1":     mean_f1,
    "std_f1":      std_f1,
    "mean_precision": mean_prec,
    "mean_recall":    mean_rec,
    "mean_accuracy":  mean_acc,
    "mean_kappa":     mean_kap,
    "mean_auroc":     mean_auc,
    "per_seed_metrics": all_seed_metrics,
}

with open("gsm8k_aggregate.json", "w") as f:
    json.dump(aggregate, f, indent=2, default=str)
print("\nAggregate saved: gsm8k_aggregate.json")

# ── SAVE CSV SUMMARY TABLE ──────────────────────────────────
with open("gsm8k_summary_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Seed", "N", "F1", "Precision", "Recall",
                     "Accuracy", "AUROC", "Kappa",
                     "CI_lower", "CI_upper", "TP", "TN", "FP", "FN", "NO_MATH%"])
    for seed, m in zip(SEEDS, all_seed_metrics):
        writer.writerow([
            seed, m["n"], m["f1"], m["precision"], m["recall"],
            m["accuracy"], m["auroc"], m["kappa"],
            m["ci_95"][0], m["ci_95"][1],
            m["tp"], m["tn"], m["fp"], m["fn"],
            f"{m['no_math_rate']*100:.1f}",
        ])
    writer.writerow(["MEAN", N_EXAMPLES, mean_f1, mean_prec, mean_rec,
                     mean_acc, mean_auc, mean_kap, "", "", "", "", "", "", ""])
    writer.writerow(["STD", "", std_f1, "", "", "", "", "", "", "", "", "", "", "", ""])

print("CSV saved: gsm8k_summary_table.csv")
print("\nDONE. Download the gsm8k_* result files before the runtime ends.")
