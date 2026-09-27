# ============================================================
# MathDial Verification Pipeline — Symmetric Verifier-Wins Rule
# Oracle-Free Mathematical Solution Verification for Reliable LLM Tutoring
#
# Model   : Llama 3.3 70B Instruct via OpenRouter
# Dataset : MathDial dialogues (Macina et al., Findings of EMNLP 2023).
#           Error annotations and the specific data release used here are
#           from the "verify-then-generate" repository of Daheim et al.
#           (Findings of EMNLP 2024), distributed under CC BY-SA 4.0.
# Instances: 600 (300 with-error + 300 no-error, stratified by error category)
# temperature = 0.0 (deterministic)
#
# Stage 4 conflict resolution is SYMMETRIC: the independent verifier's
# verdict is trusted in BOTH directions; the executor is used only as a
# fallback when the verifier errors out.
#
# RUN_ID labels the run. The dataset sample is fixed (sampling seed 42)
# and identical across runs; RUN_ID only names the run. Reported runs are
# RUN_ID 42, 456, and 789.
# ============================================================

import subprocess, json, re, random, hashlib, collections, time
from getpass import getpass

# ── CONFIGURATION — change RUN_ID to 42, 456, or 789 ─────────────────────────
RUN_ID = 42
MODEL = "meta-llama/llama-3.3-70b-instruct"
CHECKPOINT = f"/content/mathdial_run{RUN_ID}_checkpoint.json"
OUT = f"/content/mathdial_run{RUN_ID}_results.json"

# ── STEP 1: build dataset (deterministic, drop items 111/335) ─────────────────
def build_dataset():
    subprocess.run(["git", "clone", "https://github.com/eth-lre/verify-then-generate"],
                   capture_output=True)
    data = json.load(open("verify-then-generate/dataset/dataset.json"))

    def last_num(x):
        s = x[-1] if isinstance(x, list) else str(x)
        m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
        return m[-1] if m else None

    DROP_ITEMS = {111, 335}
    all_records = []
    for i, d in enumerate(data):
        if i in DROP_ITEMS:
            continue
        inc_text = "\n".join(s.strip() for s in d["student_incorrect_solution"])
        cor_text = d["student_correct_response"].strip()
        ref_sol = d["reference_solution"]
        true_final = last_num(ref_sol)
        all_records.append({"item_id": i, "instance_id": f"{i}_pos",
            "problem": d["problem"], "reference_solution": ref_sol,
            "solution_to_verify": inc_text, "true_final": true_final,
            "stated_final": last_num(d["student_incorrect_solution"]),
            "gold_is_correct": False, "gold_has_error": True,
            "error_category": d["error_category"],
            "incorrect_index": d["incorrect_index"],
            "incorrect_step": d["incorrect_step"],
            "dialog_history": d["dialog_history"]})
        all_records.append({"item_id": i, "instance_id": f"{i}_neg",
            "problem": d["problem"], "reference_solution": ref_sol,
            "solution_to_verify": cor_text, "true_final": true_final,
            "stated_final": last_num(cor_text),
            "gold_is_correct": True, "gold_has_error": False,
            "error_category": "NONE", "incorrect_index": -1,
            "incorrect_step": "", "dialog_history": d["dialog_history"]})

    all_records.sort(key=lambda r: (r["item_id"], 0 if r["instance_id"].endswith("pos") else 1))

    # stratified sample: 300 pos + their 300 paired neg (sampling seed fixed at 42
    # for ALL runs — the dataset is identical across run IDs; RUN_ID only labels the run)
    random.seed(42)
    pos_records = [r for r in all_records if r["gold_has_error"]]
    cat_counts = collections.Counter(r["error_category"] for r in pos_records)
    TARGET_POS = 300
    total_pos = len(pos_records)
    sampled_item_ids = []
    for cat, n in cat_counts.most_common():
        cat_items = sorted([r["item_id"] for r in pos_records if r["error_category"] == cat])
        k = round(TARGET_POS * n / total_pos)
        sampled = random.sample(cat_items, min(k, len(cat_items)))
        for item_id in sampled:
            if item_id not in sampled_item_ids:
                sampled_item_ids.append(item_id)
    all_item_ids = sorted([r["item_id"] for r in pos_records])
    remaining_ids = sorted([i for i in all_item_ids if i not in sampled_item_ids])
    random.shuffle(remaining_ids)
    while len(sampled_item_ids) < TARGET_POS:
        sampled_item_ids.append(remaining_ids.pop())
    while len(sampled_item_ids) > TARGET_POS:
        sampled_item_ids.pop()
    sampled_set = set(sampled_item_ids)
    records = [r for r in all_records if r["item_id"] in sampled_set]
    records.sort(key=lambda r: (r["item_id"], 0 if r["instance_id"].endswith("pos") else 1))
    json.dump(records, open("/content/mathdial_sample_600.json", "w"),
              ensure_ascii=False, indent=2)
    digest = hashlib.md5(json.dumps(records, sort_keys=True).encode()).hexdigest()[:12]
    print(f"Dataset ready: {len(records)} instances | digest: {digest}")
    return records

# ── PROMPTS ───────────────────────────────────────────────────────────────────
CLAIM_EXTRACTION_PROMPT = """You are a math solution analyzer. Given a math problem and a student solution, extract all mathematical claims made in the solution.

For each claim, identify:
1. The mathematical operation or assertion being made
2. The numerical result claimed

Return a JSON list of claims, where each claim has:
- "claim_text": the exact text of the claim
- "operation": what mathematical operation is being performed
- "result": the numerical result claimed (as a string)

Return ONLY the JSON list, no other text."""

LLM_CLAIM_CHECK_PROMPT = """You are a precise math verifier. Given a math problem, a specific claim from a student solution, and the claim's stated result, verify if the mathematical calculation is correct.

Return a JSON object with:
- "is_correct": true if the calculation is correct, false if incorrect
- "correct_result": what the correct result should be (as a string)
- "explanation": brief explanation

Return ONLY the JSON object, no other text."""

HOLISTIC_VERIFIER_PROMPT = """You are an expert math tutor verifying a student's solution.

Given the problem and student solution below, independently solve the problem step by step, then compare your solution with the student's to determine if the student's solution is correct.

Problem: {problem}

Student Solution:
{solution}

First, solve the problem yourself. Then compare with the student's solution.

Return a JSON object with:
- "my_solution": your step-by-step solution
- "my_answer": your final numerical answer
- "student_answer": the student's final answer
- "is_correct": true if student solution is correct, false if incorrect
- "explanation": explanation of any errors found

Return ONLY the JSON object, no other text."""

# ── HELPERS ───────────────────────────────────────────────────────────────────
def call_llm(messages, api_key, max_tokens=1000, retries=3):
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    for attempt in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"  HTTP {r.status_code}")
        except Exception as e:
            print(f"  attempt {attempt+1} exception: {e}")
        time.sleep(2 ** attempt)
    return None

def extract_json(text):
    if not text: return None
    try: return json.loads(text)
    except:
        m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except: return None
    return None

# ── PIPELINE ────────────────────────────────────────────────────────────────
def run_pipeline(record, api_key):
    problem = record["problem"]
    solution = record["solution_to_verify"]
    result = {
        "instance_id": record["instance_id"],
        "item_id": record["item_id"],
        "gold_has_error": record["gold_has_error"],
        "error_category": record["error_category"],
        "stated_final": record["stated_final"],
        "true_final": record["true_final"],
    }

    # Stage 1: claim extraction
    claim_raw = call_llm([
        {"role": "system", "content": CLAIM_EXTRACTION_PROMPT},
        {"role": "user", "content": f"Problem: {problem}\n\nStudent Solution:\n{solution}\n\nExtract all mathematical claims."}
    ], api_key)
    claims = extract_json(claim_raw)
    result["n_claims"] = len(claims) if isinstance(claims, list) else 0

    # Stage 2: executor (LLM claim-level check, early-exit on first incorrect).
    # On MathDial the arithmetic is embedded in dialogue and no explicit
    # expression is exposed, so this stage is an LLM claim-check rather than a
    # symbolic (SymPy) evaluation.
    executor_verdict = "CORRECT"
    if isinstance(claims, list):
        for claim in claims[:8]:
            v_raw = call_llm([
                {"role": "system", "content": LLM_CLAIM_CHECK_PROMPT},
                {"role": "user", "content": f"Problem: {problem}\n\nClaim: {claim.get('claim_text','')}\nStated result: {claim.get('result','')}\n\nIs this calculation correct?"}
            ], api_key)
            v_obj = extract_json(v_raw)
            if v_obj and isinstance(v_obj, dict) and v_obj.get("is_correct") is False:
                executor_verdict = "INCORRECT"
                break
    else:
        executor_verdict = "ERROR"
    result["executor_verdict"] = executor_verdict

    # Stage 3: holistic verifier (solves independently, then compares)
    h_raw = call_llm([
        {"role": "user", "content": HOLISTIC_VERIFIER_PROMPT.format(
            problem=problem, solution=solution)}
    ], api_key, max_tokens=1500)
    h_obj = extract_json(h_raw)
    if h_obj and isinstance(h_obj, dict):
        result["verifier_verdict"] = "CORRECT" if h_obj.get("is_correct") else "INCORRECT"
        result["verifier_my_answer"] = str(h_obj.get("my_answer", ""))
    else:
        result["verifier_verdict"] = "ERROR"
        result["verifier_my_answer"] = ""

    # Stage 4: SYMMETRIC verifier-wins conflict resolution.
    # The verifier's verdict is trusted in both directions; the executor is
    # used only as a fallback when the verifier itself errors out.
    ev = result["executor_verdict"]
    vv = result["verifier_verdict"]
    if vv == "INCORRECT":
        final = "INCORRECT"     # verifier detects error -> error
    elif vv == "CORRECT":
        final = "CORRECT"       # verifier confirms clean -> clean (overrides executor)
    else:
        final = ev              # verifier ERROR -> fall back to executor

    result["final_verdict"] = final
    result["predicted_has_error"] = (final == "INCORRECT")
    result["correct"] = (result["predicted_has_error"] == result["gold_has_error"])
    return result

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = build_dataset()
    API_KEY = getpass("Paste your OpenRouter API key and press Enter: ")

    try:
        done = json.load(open(CHECKPOINT))
        done_ids = {r["instance_id"] for r in done}
        print(f"Resuming: {len(done)} done")
    except:
        done = []
        done_ids = set()
        print(f"Starting fresh run - RUN_ID {RUN_ID}")

    remaining = [r for r in records if r["instance_id"] not in done_ids]
    print(f"Total: {len(records)} | Done: {len(done)} | To run: {len(remaining)}")

    t_start = time.time()
    n_start = len(done)
    for idx, rec in enumerate(remaining):
        t0 = time.time()
        result = run_pipeline(rec, API_KEY)
        elapsed = time.time() - t0
        result["elapsed_sec"] = round(elapsed, 1)
        done.append(result)
        json.dump(done, open(CHECKPOINT, "w"), ensure_ascii=False, indent=2)
        label = "OK" if result["correct"] else "X"
        total_done = len(done)
        pct = total_done / len(records) * 100
        done_sess = total_done - n_start
        eta = ((time.time()-t_start)/max(1,done_sess))*(len(records)-total_done)
        print(f"[{total_done:3d}/{len(records)}] {pct:5.1f}% {label} "
              f"id={result['instance_id']:10s} exec={result['executor_verdict']:9s} "
              f"ver={result['verifier_verdict']:9s} final={result['final_verdict']:9s} "
              f"{elapsed:.1f}s ETA {eta/60:.0f}min")

    json.dump(done, open(OUT, "w"), ensure_ascii=False, indent=2)
    from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
    gold = [r["gold_has_error"] for r in done]
    pred = [r["predicted_has_error"] for r in done]
    print(f"\n=== RESULTS (n={len(done)}, RUN_ID={RUN_ID}) ===")
    print(f"F1={f1_score(gold, pred):.4f} | Precision={precision_score(gold, pred):.4f} | "
          f"Recall={recall_score(gold, pred):.4f} | Kappa={cohen_kappa_score(gold, pred):.4f}")
    print(f"saved -> {OUT}")
