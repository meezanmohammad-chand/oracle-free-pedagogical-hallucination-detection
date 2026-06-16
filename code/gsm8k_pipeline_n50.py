# GSM8K PIPELINE - Math Executor Architecture (Architecture 2)
# n=50 PRIMARY EVALUATION RUN
# Oracle-Free Pedagogical Error Detection

import os
import time
import json
import re
import random
import sympy
from sympy import N
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

from openai import OpenAI
from sklearn.metrics import f1_score, roc_auc_score
from datasets import load_dataset

# -- CONFIGURATION
# Set your API key as an environment variable before running:
#   export OPENROUTER_API_KEY="your_key_here"
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable before running this script.")

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
MODEL = "meta-llama/llama-3.1-8b-instruct"
random.seed(42)

print("Loading GSM8K dataset...")
dataset = load_dataset("openai/gsm8k", "main", split="test")
print(f"Loaded {len(dataset)} examples")


def inject_arithmetic_error(solution_text):
    lines = solution_text.split('\n')
    working_lines = [l for l in lines if not l.strip().startswith('####')]
    working_text = '\n'.join(working_lines)
    matches = list(re.finditer(r'\b(\d+)\b', working_text))
    if not matches:
        return solution_text, False
    candidates = [m for m in matches if int(m.group()) > 1]
    if not candidates:
        candidates = matches
    target = random.choice(candidates)
    original_val = int(target.group())
    error_type = random.choice(['plus1', 'minus1', 'double', 'half'])
    if error_type == 'plus1':
        wrong_val = original_val + 1
    elif error_type == 'minus1':
        wrong_val = max(0, original_val - 1)
    elif error_type == 'double':
        wrong_val = original_val * 2
    else:
        wrong_val = max(1, original_val // 2)
    start, end = target.start(), target.end()
    modified = working_text[:start] + str(wrong_val) + working_text[end:]
    return modified, True


def extract_solution_steps(answer_text):
    lines = answer_text.split('\n')
    steps = [l.strip() for l in lines if l.strip() and not l.strip().startswith('####')]
    return ' '.join(steps)[:500]


def convert_gsm8k(example, idx, contains_error):
    question = example.get("question", "")[:300]
    answer = example.get("answer", "")
    solution_steps = extract_solution_steps(answer)
    if contains_error:
        tutor_response, modified = inject_arithmetic_error(solution_steps)
        if not modified:
            tutor_response = solution_steps
            contains_error = 0
    else:
        tutor_response = solution_steps
    return {
        "id": idx,
        "student_question": question,
        "tutor_response": tutor_response,
        "contains_error": contains_error,
        "original_answer": answer[:200]
    }


# Original n=50 sampling: full dataset range, single shuffle (no fresh-index offset)
def build_gsm8k_sample(n=50):
    half = n // 2
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    correct_indices = all_indices[:half]
    error_indices = all_indices[half:half * 2]
    examples = []
    for i, idx in enumerate(correct_indices):
        examples.append(convert_gsm8k(dataset[idx], i, contains_error=0))
    for i, idx in enumerate(error_indices):
        examples.append(convert_gsm8k(dataset[idx], half + i, contains_error=1))
    random.shuffle(examples)
    return examples


def call_llm(prompt, temperature=0.3):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return ""


def math_executor(claim):
    extract_prompt = f"""You are a math expression extractor. Look at this claim and extract ONLY if there is an explicit arithmetic calculation with a stated numeric result.

Claim: {claim}

Rules:
- ONLY extract if the claim explicitly states: [expression] = [number] or [expression] equals [number]
- Do NOT extract if the claim just mentions a number without a calculation
- Do NOT extract word problems without explicit arithmetic shown
- The expression must be simple arithmetic: +, -, *, /

Reply EXACTLY:
EXPRESSION: [e.g. 3 * 4 or 15 + 7 — just the calculation, no words]
STATED_RESULT: [the number the claim says this equals]

If no explicit arithmetic calculation with stated result exists:
EXPRESSION: NONE
STATED_RESULT: NONE"""

    raw = call_llm(extract_prompt, temperature=0.0)
    extracted_expr = None
    stated_result = None

    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith("EXPRESSION:"):
            val = line.replace("EXPRESSION:", "").strip()
            extracted_expr = None if val.upper() == "NONE" else val
        elif line.startswith("STATED_RESULT:"):
            val = line.replace("STATED_RESULT:", "").strip()
            stated_result = None if val.upper() == "NONE" else val

    if not extracted_expr or not stated_result:
        print(f"    [MathExec] No explicit arithmetic -> NO_MATH")
        return {
            "extracted_expr": None, "expected_result": None,
            "computed_result": None, "math_error": False,
            "executor_verdict": "NO_MATH", "error_signal": 0.0
        }

    computed = _safe_compute(extracted_expr)
    if computed is None:
        print(f"    [MathExec] Could not evaluate: {extracted_expr} -> NO_MATH")
        return {
            "extracted_expr": extracted_expr, "expected_result": stated_result,
            "computed_result": None, "math_error": False,
            "executor_verdict": "NO_MATH", "error_signal": 0.0
        }

    is_error = _compare_results(stated_result, computed)
    verdict = "ERROR" if is_error else "CORRECT"
    print(f"    [MathExec] Expr: {extracted_expr} | Stated: {stated_result} | Computed: {computed} | -> {verdict}")

    return {
        "extracted_expr": extracted_expr, "expected_result": stated_result,
        "computed_result": str(computed), "math_error": is_error,
        "executor_verdict": verdict, "error_signal": 1.0 if is_error else 0.0
    }


def _safe_compute(expr_str):
    expr_str = expr_str.strip()
    if '=' in expr_str:
        parts = expr_str.split('=', 1)
        try:
            transformations = standard_transformations + (implicit_multiplication_application,)
            lhs = parse_expr(parts[0].strip(), transformations=transformations)
            rhs = parse_expr(parts[1].strip(), transformations=transformations)
            diff = sympy.simplify(lhs - rhs)
            return float(N(rhs)) if diff == 0 else None
        except Exception:
            pass
    try:
        transformations = standard_transformations + (implicit_multiplication_application,)
        result = parse_expr(expr_str, transformations=transformations)
        return float(N(result))
    except Exception:
        pass
    try:
        safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expr_str)
        if safe_expr.strip():
            result = eval(safe_expr, {"__builtins__": {}})
            return float(result)
    except Exception:
        pass
    return None


# Original n=50 tolerance: tighter than the loosened n=600 comparison run
def _compare_results(stated_str, computed_float):
    try:
        num_matches = re.findall(r'-?\d+\.?\d*', str(stated_str).replace(',', ''))
        if not num_matches:
            return False
        for num_str in reversed(num_matches):
            stated_float = float(num_str)
            abs_diff = abs(stated_float - computed_float)
            if abs_diff <= 1.0:
                return False
            if computed_float != 0:
                rel_diff = abs_diff / abs(computed_float)
                if rel_diff <= 0.01:
                    return False
        return True
    except Exception:
        return False


def claim_extractor(tutor_response):
    prompt = f"""Extract every factual or mathematical claim from this tutoring response.
Return ONLY a numbered list. Each item = one atomic claim. No explanation.

Tutoring response: {tutor_response}

Numbered list of claims:"""
    raw = call_llm(prompt, temperature=0.1)
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    claims = []
    for line in lines:
        clean = line.lstrip('0123456789.-) ').strip()
        if clean:
            claims.append(clean)
    return claims if claims else [tutor_response]


def isolated_verifier(question, claim, math_exec_verdict=None):
    exec_hint = ""
    if math_exec_verdict and math_exec_verdict != "NO_MATH":
        exec_hint = f"""\nNote: A deterministic arithmetic check already found this claim is {math_exec_verdict}.
Consider this carefully before giving your verdict."""
    prompt = f"""You are an independent math verifier.

Original student question: {question}
Claim to verify: {claim}{exec_hint}

Verify this claim from first principles. Show your working.
Is this claim mathematically correct?

Reply EXACTLY:
ISOLATED_VERDICT: [CORRECT or INCORRECT]
EXPLANATION: [one sentence, show your working]"""
    raw = call_llm(prompt, temperature=0.1)
    verdict = "UNKNOWN"
    for line in raw.split('\n'):
        if line.strip().startswith("ISOLATED_VERDICT:"):
            verdict = line.replace("ISOLATED_VERDICT:", "").strip()
            break
    is_error = "INCORRECT" in verdict.upper()
    return {"isolated_verdict": verdict, "flags_error": is_error, "raw_output": raw}


def feedback_quality_filter(math_exec_result, verification_result):
    exec_flags = math_exec_result["math_error"]
    verif_flags = verification_result["flags_error"]
    exec_verdict = math_exec_result["executor_verdict"]

    if exec_verdict == "NO_MATH":
        prediction = 1 if (exec_flags or verif_flags) else 0
        quality = "VERIFIER_ONLY_OR"
    elif exec_flags == verif_flags:
        prediction = 1 if exec_flags else 0
        quality = "HIGH"
    else:
        prediction = 1 if exec_flags else 0
        quality = "CONFLICT_EXEC_WINS"

    return {
        "final_prediction": prediction, "quality": quality,
        "exec_flagged": exec_flags, "verification_flagged": verif_flags
    }


def run_pipeline(example, run_num, total):
    print(f"\n{'='*60}")
    print(f"[{run_num}/{total}] Example {example['id']} | GT: {'ERROR' if example['contains_error'] else 'CORRECT'}")
    print(f"{'='*60}")

    tutor_response = example["tutor_response"]
    if not tutor_response:
        return None

    print("\n[2] Claim Extractor...")
    claims = claim_extractor(tutor_response)
    print(f"    {len(claims)} claim(s) | First: {claims[0][:80]}")

    print("\n[3] Math Executor (claims + full response)...")
    all_math_results = []
    for i, claim in enumerate(claims):
        print(f"    Claim {i+1}/{len(claims)}: {claim[:60]}")
        result = math_executor(claim)
        all_math_results.append(result)
        if result["math_error"]:
            print(f"    *** ERROR FOUND in claim {i+1} -- stopping early ***")
            break

    print(f"    Direct check on full response...")
    direct_result = math_executor(tutor_response[:300])

    any_math_error = any(r["math_error"] for r in all_math_results) or direct_result["math_error"]
    any_math_fired = any(r["executor_verdict"] != "NO_MATH" for r in all_math_results) or direct_result["executor_verdict"] != "NO_MATH"

    math_result = {
        "math_error": any_math_error,
        "executor_verdict": "ERROR" if any_math_error else ("CORRECT" if any_math_fired else "NO_MATH"),
        "error_signal": 1.0 if any_math_error else 0.0,
        "claims_checked": len(all_math_results),
        "direct_result": direct_result
    }
    print(f"    Aggregate: {math_result['executor_verdict']} | Claims: {math_result['claims_checked']} | Direct: {direct_result['executor_verdict']}")

    primary_claim = claims[0]
    print("\n[4] Isolated Verifier...")
    verif = isolated_verifier(example["student_question"], primary_claim, math_exec_verdict=math_result["executor_verdict"])
    print(f"    Isolated verdict: {verif['isolated_verdict']} | Flags error: {verif['flags_error']}")

    print("\n[5] Feedback Quality Filter...")
    filt = feedback_quality_filter(math_result, verif)
    print(f"    Final prediction: {'ERROR' if filt['final_prediction'] else 'CORRECT'} | Quality: {filt['quality']}")

    correct = filt["final_prediction"] == example["contains_error"]
    print(f"\n    {'CORRECT prediction' if correct else 'WRONG prediction'}")

    return {
        "id": example["id"],
        "ground_truth": example["contains_error"],
        "prediction": filt["final_prediction"],
        "correct": correct,
        "math_executor": math_result,
        "verification": verif,
        "filter": filt
    }


def run_gsm8k_pilot(n=50):
    print(f"\nGSM8K RUN -- Math Executor Pipeline | n={n} | seed=42")
    print(f"Split: {n//2} correct + {n//2} error (balanced)\n")

    examples = build_gsm8k_sample(n)
    results = []

    for i, example in enumerate(examples):
        result = run_pipeline(example, i + 1, n)
        if result:
            results.append(result)
        time.sleep(1)

    print("\n" + "=" * 60)
    print("GSM8K RESULTS - METRICS (n=50 primary run)")
    print("=" * 60)

    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["prediction"] for r in results]
    y_scores = [r["math_executor"]["error_signal"] for r in results]

    accuracy = sum(r["correct"] for r in results) / len(results)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except Exception:
        auroc = None

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    no_math_count = sum(1 for r in results if r["math_executor"]["executor_verdict"] == "NO_MATH")
    conflict_count = sum(1 for r in results if r["filter"]["quality"] == "CONFLICT_EXEC_WINS")

    print(f"\nTotal examples       : {len(results)}")
    print(f"Accuracy             : {accuracy:.3f}")
    print(f"F1 Score             : {f1:.3f}")
    if auroc is not None:
        print(f"AUROC                : {auroc:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP (error caught)     : {tp}")
    print(f"  TN (correct accepted) : {tn}")
    print(f"  FP (false alarm)      : {fp}")
    print(f"  FN (missed error)     : {fn}")
    print(f"\nNet detection benefit  : {tp - fp}")
    print(f"NO_MATH rate           : {no_math_count}/{len(results)} ({100*no_math_count/len(results):.1f}%)")
    print(f"Conflict rate          : {conflict_count}/{len(results)} ({100*conflict_count/len(results):.1f}%)")

    # SAVE RESULTS IMMEDIATELY -- download this file from Colab right after the run finishes
    output_filename = "gsm8k_results_n50.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_filename}")
    print("IMPORTANT: Download this file now (Colab Files panel -> right-click -> Download)")

    return results, {
        "accuracy": accuracy, "f1": f1, "auroc": auroc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "no_math_rate": no_math_count / len(results)
    }


# -- RUN
if __name__ == "__main__":
    results, metrics = run_gsm8k_pilot(n=50)
