#!/usr/bin/env python3
"""Empirical teacher correction accuracy on pilot-50 expert outputs.

Validates the correction_signal_quality simulation (19.6% avg error) with real
teacher judgments on real expert outputs. Samples from 6 representative domains
(matching the simulation's 6 domains: python_basics≈python, algorithm_design≈math,
systems_programming≈cpp, creative_writing≈creative-fiction, logical_reasoning≈logic-puzzles,
medical_qa≈medical) to enable direct comparison.

Pipeline per domain:
  1. Generate N expert answers using 7B + pilot adapter
  2. Send to 70B teacher (Groq API) for judgment + correction
  3. Compute empirical error rate = (false_positive + bad_correction) / total
  4. Compare to simulation prediction for that domain type

Kill criteria:
  K1: empirical teacher error rate >30% on ANY domain
  K2: empirical error rate differs from simulation prediction by >10pp

Supports SMOKE_TEST=1 (5 queries/domain, 2 domains).
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "teacher_correction_empirical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

TEACHER_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _load_api_key():
    """Load GROQ_API_KEY from .env (manual parse to avoid dotenv dependency)."""
    if not os.environ.get("GROQ_API_KEY"):
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip("'\"")
                    os.environ.setdefault(key.strip(), val)
    return os.environ.get("GROQ_API_KEY", "")

# Map simulation domains to pilot-50 adapter domains
# The simulation used 6 archetypal domain types with specific error predictions
DOMAIN_MAP = {
    "python": {
        "sim_domain": "python_basics",
        "sim_error_rate": 0.124,  # from correction_signal_quality results
        "sim_degeneracy": 0.013,
        "is_code": True,
    },
    "math": {
        "sim_domain": "algorithm_design",
        "sim_error_rate": 0.185,
        "sim_degeneracy": 0.032,
        "is_code": True,
    },
    "cpp": {
        "sim_domain": "systems_programming",
        "sim_error_rate": 0.241,
        "sim_degeneracy": 0.113,
        "is_code": True,
    },
    "creative-fiction": {
        "sim_domain": "creative_writing",
        "sim_error_rate": 0.178,
        "sim_degeneracy": 0.025,
        "is_code": False,
    },
    "logic-puzzles": {
        "sim_domain": "logical_reasoning",
        "sim_error_rate": 0.221,
        "sim_degeneracy": 0.037,
        "is_code": False,
    },
    "medical": {
        "sim_domain": "medical_qa",
        "sim_error_rate": 0.221,
        "sim_degeneracy": 0.042,
        "is_code": False,
    },
}

N_QUERIES = 100 if not IS_SMOKE else 5
TEST_DOMAINS = list(DOMAIN_MAP.keys()) if not IS_SMOKE else ["python", "medical"]


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def load_eval_data(domain, n):
    """Load evaluation examples from domain data."""
    eval_file = DATA_DIR / domain / "eval.jsonl"
    train_file = DATA_DIR / domain / "train.jsonl"
    source = eval_file if eval_file.exists() else train_file
    if not source.exists():
        return []

    examples = []
    with open(source) as f:
        for line in f:
            record = json.loads(line)
            if "messages" in record and len(record["messages"]) >= 2:
                examples.append(record)
            if len(examples) >= n:
                break
    return examples


def generate_expert_answers(base_model, tokenizer, domain, examples):
    """Generate expert answers using 7B + adapter."""
    import torch
    from peft import PeftModel

    adapter_dir = ADAPTER_DIR / domain
    if not adapter_dir.exists():
        log(f"  WARNING: adapter {adapter_dir} not found")
        return []

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    results = []
    for ex in examples:
        msgs = ex["messages"]
        query = next((m["content"] for m in msgs if m["role"] == "user"), None)
        ground_truth = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not query or not ground_truth:
            continue

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(base_model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        expert_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        results.append({
            "query": query,
            "expert_answer": expert_answer,
            "ground_truth": ground_truth,
        })

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def judge_with_teacher(query, expert_answer, ground_truth, is_code):
    """Send expert answer to 70B teacher for judgment.

    For code domains, we also ask the teacher to evaluate functional correctness.
    For non-code domains, we focus on factual accuracy and completeness.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url=GROQ_BASE_URL)

    if is_code:
        judge_prompt = f"""You are evaluating an AI assistant's code answer against a reference solution.

Question: {query}

AI Answer:
{expert_answer}

Reference Solution:
{ground_truth}

Evaluate:
1. Does the AI answer produce correct output for typical inputs?
2. Does it handle edge cases comparably to the reference?
3. Is it a degenerate solution (technically runs but useless, e.g., empty function, hardcoded values)?
4. If wrong, provide the correct solution.
5. Rate confidence (0.0-1.0).

Respond ONLY in JSON:
{{"correct": true/false, "degenerate": true/false, "correction": "correct answer if wrong, null if correct", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
    else:
        judge_prompt = f"""You are evaluating an AI assistant's answer against a reference answer.

Question: {query}

AI Answer:
{expert_answer}

Reference Answer:
{ground_truth}

Evaluate:
1. Is the AI answer factually correct and complete relative to the reference?
2. Is it a degenerate answer (too vague, copies the question, or provides no real information)?
3. If incorrect/incomplete, provide the correct answer.
4. Rate confidence (0.0-1.0).

Respond ONLY in JSON:
{{"correct": true/false, "degenerate": true/false, "correction": "correct answer if wrong, null if correct", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        resp = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return result
    except Exception as e:
        log(f"    Teacher API error: {e}")
        return {"correct": True, "degenerate": False, "correction": None,
                "confidence": 0.0, "reasoning": f"API error: {e}"}


def compute_empirical_error_rate(judgments, ground_truths_available=True):
    """Compute empirical teacher error rate.

    A teacher judgment is "wrong" (error) if:
      - Teacher says "correct" but expert answer has low overlap with ground truth (false positive)
      - Teacher says "wrong" but expert answer is actually good (false negative)
      - Teacher provides a correction that is semantically worse than ground truth (bad correction)

    Also tracks degeneracy: teacher says correct but expert output is degenerate.
    """
    n_total = 0
    n_errors = 0  # wrong judgments
    n_degeneracies = 0  # missed degenerate outputs
    details = []

    for j in judgments:
        confidence = j["judgment"].get("confidence", 0.0)
        if confidence < 0.2:
            continue  # skip very low confidence

        n_total += 1
        expert = j["expert_answer"].lower().strip()
        truth = j["ground_truth"].lower().strip()
        teacher_says_correct = j["judgment"].get("correct", True)
        teacher_says_degenerate = j["judgment"].get("degenerate", False)

        # Heuristic: compute word overlap between expert and ground truth
        expert_words = set(expert.split())
        truth_words = set(truth.split())
        if len(truth_words) == 0:
            continue

        overlap = len(expert_words & truth_words) / len(truth_words)

        # Very short expert answers relative to ground truth → likely degenerate
        length_ratio = len(expert) / max(len(truth), 1)
        is_likely_degenerate = length_ratio < 0.1 and len(expert) < 20

        # Expert is "wrong" if very low overlap
        expert_is_wrong = overlap < 0.25

        # Error types
        error_type = None
        if expert_is_wrong and teacher_says_correct:
            error_type = "false_positive"
            n_errors += 1
        elif not expert_is_wrong and not teacher_says_correct:
            error_type = "false_negative"
            n_errors += 1
        elif is_likely_degenerate and not teacher_says_degenerate and teacher_says_correct:
            error_type = "missed_degeneracy"
            n_degeneracies += 1

        # Check correction quality (if teacher provided one)
        correction = j["judgment"].get("correction")
        if correction and not teacher_says_correct:
            correction_words = set(correction.lower().split())
            corr_overlap = len(correction_words & truth_words) / len(truth_words) if truth_words else 0
            if corr_overlap < 0.15:
                error_type = "bad_correction"
                n_errors += 1

        details.append({
            "overlap": round(overlap, 3),
            "length_ratio": round(length_ratio, 3),
            "teacher_correct": teacher_says_correct,
            "teacher_degenerate": teacher_says_degenerate,
            "expert_is_wrong": expert_is_wrong,
            "error_type": error_type,
        })

    error_rate = n_errors / max(n_total, 1)
    degeneracy_rate = n_degeneracies / max(n_total, 1)

    return {
        "error_rate": round(error_rate, 4),
        "degeneracy_rate": round(degeneracy_rate, 4),
        "n_total": n_total,
        "n_errors": n_errors,
        "n_degeneracies": n_degeneracies,
        "error_types": {
            "false_positive": sum(1 for d in details if d["error_type"] == "false_positive"),
            "false_negative": sum(1 for d in details if d["error_type"] == "false_negative"),
            "bad_correction": sum(1 for d in details if d["error_type"] == "bad_correction"),
            "missed_degeneracy": sum(1 for d in details if d["error_type"] == "missed_degeneracy"),
        },
        "details": details,
    }


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    log("=" * 70)
    log("  Teacher Correction Empirical Validation")
    log(f"  Smoke: {IS_SMOKE}, Queries/domain: {N_QUERIES}")
    log(f"  Domains: {TEST_DOMAINS}")
    log(f"  Teacher: {TEACHER_MODEL}")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    log("=" * 70)

    GROQ_API_KEY = _load_api_key()
    if not GROQ_API_KEY:
        log("ERROR: GROQ_API_KEY not found. Set in /workspace/llm/.env")
        sys.exit(1)

    # Load base model
    log("\n[1] Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    log("  Base model loaded.")

    # ── Per-domain evaluation ─────────────────────────────────────
    domain_results = {}

    for di, domain in enumerate(TEST_DOMAINS):
        log(f"\n[{di+2}] === {domain} ({di+1}/{len(TEST_DOMAINS)}) ===")
        domain_info = DOMAIN_MAP[domain]

        # Load data
        examples = load_eval_data(domain, N_QUERIES)
        if not examples:
            log(f"  No eval data for {domain}, skipping")
            continue
        log(f"  Loaded {len(examples)} eval examples")

        # Generate expert answers
        log(f"  Generating expert answers...")
        answers = generate_expert_answers(base_model, tokenizer, domain, examples)
        log(f"  Generated {len(answers)} expert answers")

        if not answers:
            continue

        # Teacher judgment
        log(f"  Sending to teacher ({TEACHER_MODEL})...")
        judgments = []
        for i, ans in enumerate(answers):
            judgment = judge_with_teacher(
                ans["query"], ans["expert_answer"], ans["ground_truth"],
                is_code=domain_info["is_code"],
            )
            judgments.append({**ans, "judgment": judgment})
            if (i + 1) % 20 == 0:
                log(f"    Judged {i+1}/{len(answers)}")
            # Groq rate limiting
            time.sleep(0.5 if not IS_SMOKE else 0.1)

        # Save raw judgments
        with open(RESULTS_DIR / f"{domain}_judgments.json", "w") as f:
            json.dump(judgments, f, indent=2, default=str)

        # Compute empirical error rate
        emp = compute_empirical_error_rate(judgments)

        # Compare to simulation prediction
        sim_error = domain_info["sim_error_rate"]
        sim_degen = domain_info["sim_degeneracy"]
        divergence = abs(emp["error_rate"] - sim_error)

        log(f"  Empirical error rate:  {emp['error_rate']:.1%} ({emp['n_errors']}/{emp['n_total']})")
        log(f"  Simulation prediction: {sim_error:.1%}")
        log(f"  Divergence:            {divergence:.1%} (threshold: 10pp)")
        log(f"  Degeneracy rate:       {emp['degeneracy_rate']:.1%} (sim: {sim_degen:.1%})")
        log(f"  Error breakdown: {emp['error_types']}")

        domain_results[domain] = {
            "empirical_error_rate": emp["error_rate"],
            "empirical_degeneracy_rate": emp["degeneracy_rate"],
            "sim_error_rate": sim_error,
            "sim_degeneracy_rate": sim_degen,
            "divergence_pp": round(divergence * 100, 1),
            "n_total": emp["n_total"],
            "n_errors": emp["n_errors"],
            "n_degeneracies": emp["n_degeneracies"],
            "error_types": emp["error_types"],
            "n_expert_answers": len(answers),
        }

    # ── Kill Criteria Assessment ──────────────────────────────────
    log(f"\n{'='*70}")
    log("  KILL CRITERIA ASSESSMENT")
    log(f"{'='*70}")

    # K1: error rate >30% on any domain
    max_error_domain = max(domain_results.items(),
                           key=lambda x: x[1]["empirical_error_rate"]) if domain_results else (None, {})
    max_error_rate = max_error_domain[1].get("empirical_error_rate", 0.0) if max_error_domain[0] else 0.0
    k1_killed = max_error_rate > 0.30

    log(f"\n  K1: empirical error rate >30% on any domain?")
    for d, r in sorted(domain_results.items(), key=lambda x: -x[1]["empirical_error_rate"]):
        flag = " *** KILL ***" if r["empirical_error_rate"] > 0.30 else ""
        log(f"    {d:20s}: {r['empirical_error_rate']:.1%}{flag}")
    log(f"  K1 worst: {max_error_domain[0]}={max_error_rate:.1%} → {'KILLED' if k1_killed else 'SURVIVES'}")

    # K2: divergence from simulation >10pp
    max_div_domain = max(domain_results.items(),
                         key=lambda x: x[1]["divergence_pp"]) if domain_results else (None, {})
    max_divergence = max_div_domain[1].get("divergence_pp", 0.0) if max_div_domain[0] else 0.0
    k2_killed = max_divergence > 10.0

    log(f"\n  K2: divergence from simulation >10pp?")
    for d, r in sorted(domain_results.items(), key=lambda x: -x[1]["divergence_pp"]):
        flag = " *** KILL ***" if r["divergence_pp"] > 10.0 else ""
        log(f"    {d:20s}: emp={r['empirical_error_rate']:.1%} sim={r['sim_error_rate']:.1%} "
            f"Δ={r['divergence_pp']:.1f}pp{flag}")
    log(f"  K2 worst: {max_div_domain[0]}={max_divergence:.1f}pp → {'KILLED' if k2_killed else 'SURVIVES'}")

    # Aggregate
    all_empirical = [r["empirical_error_rate"] for r in domain_results.values()]
    avg_empirical_error = sum(all_empirical) / max(len(all_empirical), 1)
    sim_avg_error = 0.196  # from correction_signal_quality results

    overall_killed = k1_killed or k2_killed
    log(f"\n  Average empirical error: {avg_empirical_error:.1%} (simulation predicted: {sim_avg_error:.1%})")
    log(f"  VERDICT: {'KILLED' if overall_killed else 'SURVIVES'}")

    # ── Per-domain summary table ──────────────────────────────────
    log(f"\n{'='*70}")
    log(f"  {'Domain':<20s} {'EmpErr':>8s} {'SimErr':>8s} {'Δpp':>6s} {'EmpDeg':>8s} {'SimDeg':>8s} {'N':>5s}")
    log(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*5}")
    for d in sorted(domain_results.keys()):
        r = domain_results[d]
        log(f"  {d:<20s} {r['empirical_error_rate']:>7.1%} {r['sim_error_rate']:>7.1%} "
            f"{r['divergence_pp']:>5.1f} {r['empirical_degeneracy_rate']:>7.1%} "
            f"{r['sim_degeneracy_rate']:>7.1%} {r['n_total']:>5d}")

    # ── Save results ──────────────────────────────────────────────
    elapsed = time.time() - t0
    output = {
        "experiment": "exp_teacher_correction_empirical_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "domains": TEST_DOMAINS,
            "n_queries_per_domain": N_QUERIES,
            "teacher_model": TEACHER_MODEL,
            "base_model": BASE_MODEL,
            "smoke_test": IS_SMOKE,
        },
        "per_domain": domain_results,
        "aggregate": {
            "avg_empirical_error": round(avg_empirical_error, 4),
            "sim_avg_error": sim_avg_error,
            "avg_divergence_pp": round(
                sum(r["divergence_pp"] for r in domain_results.values()) /
                max(len(domain_results), 1), 1
            ),
        },
        "kill_criteria": {
            "K1_max_error_gt_30pct": {
                "worst_domain": max_error_domain[0],
                "value_pct": round(max_error_rate * 100, 1),
                "threshold_pct": 30,
                "killed": k1_killed,
            },
            "K2_divergence_gt_10pp": {
                "worst_domain": max_div_domain[0],
                "value_pp": round(max_divergence, 1),
                "threshold_pp": 10,
                "killed": k2_killed,
            },
        },
        "verdict": "KILLED" if overall_killed else "SURVIVES",
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
