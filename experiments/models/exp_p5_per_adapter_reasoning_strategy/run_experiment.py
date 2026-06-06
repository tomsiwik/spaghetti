#!/usr/bin/env python3
"""P5.C1: Per-Adapter Reasoning Strategy (CoT for Math, PAL for Code)

Kill criteria:
  K1279: Per-strategy routing improves overall accuracy >= 5pp vs uniform CoT
  K1280: Token usage reduced >= 30% vs always-CoT baseline
  K1281: Strategy selection accuracy >= 80% (correct strategy for domain)

Prior: Finding #196 (TF-IDF 95%), P5.C0 killed (module-disjoint interference)
Paper: arXiv:2505.19435 (per-adapter reasoning routing)
Platform: Apple M5 Pro 48GB, MLX only.
"""

import gc
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

# Memory safety (CODING_GUIDELINES)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

GREEDY_SAMPLER = make_sampler(temp=0.0)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
MAX_TOKENS = 512 if not IS_SMOKE else 64

DOMAINS = ["math", "code", "legal", "medical", "finance"]

# ══════════════════════════════════════════════════════════════════════════════
# REASONING STRATEGIES — different instruction prompts
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "cot": "Think step by step before providing your final answer. Show your reasoning process clearly.",
    "direct": "Answer directly and concisely in 1-3 sentences. No lengthy explanations or reasoning steps.",
    "pal": "Solve this by writing Python code that computes the answer. Show the code, then state the final numerical answer.",
    "structured": "Provide a structured answer with these sections: 1) Definition, 2) Key Points, 3) Conclusion.",
}

# Hypothesized optimal strategy per domain
DOMAIN_STRATEGY_MAP = {
    "math": "cot",
    "code": "pal",
    "legal": "structured",
    "medical": "structured",
    "finance": "cot",
}

# ══════════════════════════════════════════════════════════════════════════════
# TEST TASKS — 5 per domain, each with question + expected answer markers
# ══════════════════════════════════════════════════════════════════════════════

TASKS = {
    "math": [
        {
            "question": "A store sells apples for $2 each and oranges for $3 each. If you buy 4 apples and 6 oranges, how much do you spend in total?",
            "expected": ["26"],
            "type": "number",
        },
        {
            "question": "What is 15% of 240?",
            "expected": ["36"],
            "type": "number",
        },
        {
            "question": "If 3x + 7 = 22, what is x?",
            "expected": ["5"],
            "type": "number",
        },
        {
            "question": "A rectangle has length 12cm and width 5cm. What is its area in square centimeters?",
            "expected": ["60"],
            "type": "number",
        },
        {
            "question": "A car travels 180 miles in 3 hours. What is its average speed in miles per hour?",
            "expected": ["60"],
            "type": "number",
        },
    ],
    "code": [
        {
            "question": "Write a Python function called sum_evens that takes a list of integers and returns the sum of all even numbers.",
            "expected": ["def ", "% 2"],
            "type": "code",
        },
        {
            "question": "Write a Python function called is_palindrome that checks if a string is a palindrome. Return True or False.",
            "expected": ["def ", "palindrome"],
            "type": "code",
        },
        {
            "question": "Write a Python function called find_max that finds the maximum element in a list without using the built-in max() function.",
            "expected": ["def ", "find_max"],
            "type": "code",
        },
        {
            "question": "Write a Python function called factorial that computes the factorial of a non-negative integer n.",
            "expected": ["def ", "factorial"],
            "type": "code",
        },
        {
            "question": "Write a Python function called count_vowels that counts the number of vowels in a given string.",
            "expected": ["def ", "vowel"],
            "type": "code",
        },
    ],
    "legal": [
        {
            "question": "What are the essential elements required for a valid contract under common law?",
            "expected": ["offer", "acceptance", "consideration"],
            "type": "keyword_any",
        },
        {
            "question": "Explain the legal concept of habeas corpus and its significance.",
            "expected": ["detention", "imprison", "custody", "lawful", "liberty"],
            "type": "keyword_any",
        },
        {
            "question": "What is the legal distinction between a felony and a misdemeanor?",
            "expected": ["serious", "punishment", "prison", "year", "severity"],
            "type": "keyword_any",
        },
        {
            "question": "Explain the doctrine of stare decisis in the legal system.",
            "expected": ["precedent", "binding", "prior", "court", "decision"],
            "type": "keyword_any",
        },
        {
            "question": "What protections does the Fifth Amendment to the US Constitution provide?",
            "expected": ["self-incrimination", "due process", "double jeopardy", "silent", "witness"],
            "type": "keyword_any",
        },
    ],
    "medical": [
        {
            "question": "What are the warning signs of a stroke that the public should know?",
            "expected": ["face", "arm", "speech", "FAST", "weakness", "numb"],
            "type": "keyword_any",
        },
        {
            "question": "How is Type 2 diabetes mellitus diagnosed?",
            "expected": ["blood", "glucose", "A1C", "fasting", "sugar", "HbA1c"],
            "type": "keyword_any",
        },
        {
            "question": "What is the recommended first-line treatment for essential hypertension?",
            "expected": ["lifestyle", "ACE", "diuretic", "blood pressure", "thiazide", "exercise"],
            "type": "keyword_any",
        },
        {
            "question": "What are the typical symptoms of community-acquired pneumonia?",
            "expected": ["cough", "fever", "breath", "chest", "sputum"],
            "type": "keyword_any",
        },
        {
            "question": "Explain the key difference between Type 1 and Type 2 diabetes.",
            "expected": ["insulin", "autoimmune", "resistance", "pancreas", "beta"],
            "type": "keyword_any",
        },
    ],
    "finance": [
        {
            "question": "Calculate the compound interest earned on $1000 invested at 5% annual interest for 3 years.",
            "expected": ["157", "1157", "1158"],
            "type": "number",
        },
        {
            "question": "Explain the fundamental difference between stocks and bonds as investment instruments.",
            "expected": ["equity", "ownership", "debt", "fixed income", "share"],
            "type": "keyword_any",
        },
        {
            "question": "What is the Price-to-Earnings (P/E) ratio and how is it calculated?",
            "expected": ["price", "earnings", "share", "market"],
            "type": "keyword_any",
        },
        {
            "question": "What is portfolio diversification and why is it important?",
            "expected": ["risk", "asset", "spread", "different", "reduce"],
            "type": "keyword_any",
        },
        {
            "question": "Calculate the simple interest earned on $5000 at 8% annual rate for 2 years.",
            "expected": ["800"],
            "type": "number",
        },
    ],
}

# Domain reference texts for TF-IDF routing
DOMAIN_CORPUS = {
    "math": "algebra equation solve calculate number arithmetic sum product formula variable coefficient polynomial derivative integral function graph",
    "code": "function variable loop array string algorithm data structure class method import return def python code program syntax debug compile",
    "legal": "court law statute contract rights amendment constitution precedent defendant plaintiff jurisdiction verdict sentence appeal testimony evidence",
    "medical": "patient diagnosis treatment symptoms disease therapy medication blood pressure heart lung kidney liver prescription dosage clinical hospital",
    "finance": "investment portfolio stock bond market interest rate dividend capital asset return risk profit revenue earnings valuation share",
}


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log(msg: str):
    print(msg, flush=True)


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_response(response: str, expected: list[str], eval_type: str) -> bool:
    """Check if response contains expected answer markers."""
    response_lower = response.lower()

    if eval_type == "number":
        # Extract all numbers from response
        numbers = re.findall(r"[\d,]+\.?\d*", response)
        numbers_clean = [n.replace(",", "") for n in numbers]
        for exp in expected:
            if exp in numbers_clean:
                return True
            # Also check if the number appears as substring (e.g. "$26" contains "26")
            for n in numbers_clean:
                try:
                    if abs(float(n) - float(exp)) < 0.5:
                        return True
                except ValueError:
                    continue
        return False

    elif eval_type == "code":
        # All expected patterns must appear
        return all(pat.lower() in response_lower for pat in expected)

    elif eval_type == "keyword_any":
        # Any expected keyword must appear
        return any(kw.lower() in response_lower for kw in expected)

    return False


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF ROUTING
# ══════════════════════════════════════════════════════════════════════════════


def build_tfidf(corpus: dict[str, str]) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    """Build TF-IDF model from domain corpus."""
    domains = list(corpus.keys())
    # Compute IDF
    all_words = set()
    domain_words = {}
    for domain, text in corpus.items():
        words = set(text.lower().split())
        domain_words[domain] = words
        all_words.update(words)

    n_docs = len(domains)
    idf = {}
    for word in all_words:
        df = sum(1 for d in domains if word in domain_words[d])
        idf[word] = math.log(n_docs / df) if df > 0 else 0

    # TF-IDF vectors per domain
    tfidf = {}
    for domain, text in corpus.items():
        words = text.lower().split()
        tf = Counter(words)
        total = len(words)
        vec = {}
        for w, count in tf.items():
            vec[w] = (count / total) * idf.get(w, 0)
        tfidf[domain] = vec

    return domains, idf, tfidf


def tfidf_route(query: str, domains: list[str], idf: dict, tfidf: dict) -> tuple[str, dict[str, float]]:
    """Route a query to the best-matching domain via TF-IDF cosine similarity."""
    words = query.lower().split()
    tf = Counter(words)
    total = len(words)
    query_vec = {}
    for w, count in tf.items():
        query_vec[w] = (count / total) * idf.get(w, 0)

    # Cosine similarity
    scores = {}
    q_norm = math.sqrt(sum(v * v for v in query_vec.values()))
    if q_norm == 0:
        return domains[0], {d: 0.0 for d in domains}

    for domain in domains:
        d_vec = tfidf[domain]
        dot = sum(query_vec.get(w, 0) * d_vec.get(w, 0) for w in query_vec)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
        scores[domain] = dot / (q_norm * d_norm) if d_norm > 0 else 0.0

    best = max(scores, key=scores.get)
    return best, scores


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Generate and evaluate all (domain, strategy) combinations
# ══════════════════════════════════════════════════════════════════════════════


def phase_generate_and_evaluate() -> dict:
    """Load model, generate responses for all task×strategy combos, evaluate."""
    from mlx_lm import generate, load

    log("Phase 1: Loading model...")
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    results = {}
    total_gen = 0
    n_tasks = 2 if IS_SMOKE else 5
    strategy_list = list(STRATEGIES.keys())

    for domain in DOMAINS:
        results[domain] = {}
        tasks = TASKS[domain][:n_tasks]

        for strategy_name in strategy_list:
            strategy_prompt = STRATEGIES[strategy_name]
            accuracies = []
            token_counts = []
            responses = []

            for task in tasks:
                # Format prompt with strategy instruction
                user_content = f"{strategy_prompt}\n\nQuestion: {task['question']}"
                messages = [{"role": "user", "content": user_content}]

                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                except Exception:
                    # Fallback: manual formatting
                    prompt = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

                response = generate(
                    model, tokenizer, prompt=prompt,
                    max_tokens=MAX_TOKENS, verbose=False,
                    sampler=GREEDY_SAMPLER,
                )

                # Count tokens in response
                resp_tokens = tokenizer.encode(response)
                n_tokens = len(resp_tokens)

                # Evaluate accuracy
                accurate = evaluate_response(response, task["expected"], task["type"])

                accuracies.append(accurate)
                token_counts.append(n_tokens)
                responses.append(response[:200])  # Truncate for storage
                total_gen += 1

                if total_gen % 10 == 0:
                    log(f"  Generated {total_gen}/{len(DOMAINS) * len(strategy_list) * n_tasks}")

            acc = sum(accuracies) / len(accuracies) * 100
            mean_tokens = sum(token_counts) / len(token_counts)

            results[domain][strategy_name] = {
                "accuracy": round(acc, 1),
                "mean_tokens": round(mean_tokens, 1),
                "raw_accuracies": accuracies,
                "token_counts": token_counts,
                "responses": responses,
            }
            log(f"  {domain}/{strategy_name}: acc={acc:.0f}% tokens={mean_tokens:.0f}")

    log_memory("post-generation")
    cleanup(model, tokenizer)
    log(f"Phase 1 complete: {total_gen} generations")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TF-IDF routing accuracy
# ══════════════════════════════════════════════════════════════════════════════


def phase_tfidf_routing() -> dict:
    """Test TF-IDF routing accuracy on test questions."""
    log("Phase 2: TF-IDF routing test...")

    domains_list, idf, tfidf = build_tfidf(DOMAIN_CORPUS)

    correct = 0
    total = 0
    routing_details = []

    for true_domain in DOMAINS:
        n_tasks = 2 if IS_SMOKE else 5
        for task in TASKS[true_domain][:n_tasks]:
            predicted, scores = tfidf_route(task["question"], domains_list, idf, tfidf)
            is_correct = predicted == true_domain
            correct += int(is_correct)
            total += 1
            routing_details.append({
                "true_domain": true_domain,
                "predicted": predicted,
                "correct": is_correct,
                "scores": {d: round(s, 4) for d, s in scores.items()},
            })

    accuracy = correct / total * 100 if total > 0 else 0
    log(f"Phase 2 complete: routing accuracy = {accuracy:.1f}% ({correct}/{total})")

    return {
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "total": total,
        "details": routing_details,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Compute kill criteria
# ══════════════════════════════════════════════════════════════════════════════


def compute_kill_criteria(gen_results: dict, routing: dict) -> dict:
    """Compute K1279, K1280, K1281 from experimental results."""
    log("Phase 3: Computing kill criteria...")

    # Find optimal strategy per domain and compute metrics
    uniform_strategy = "cot"
    uniform_acc = []
    uniform_tokens = []
    optimal_acc = []
    optimal_tokens = []
    per_domain_optimal = {}
    strategy_matrix = {}

    for domain in DOMAINS:
        domain_results = gen_results[domain]
        strategy_matrix[domain] = {}

        # Uniform CoT baseline
        cot = domain_results[uniform_strategy]
        uniform_acc.append(cot["accuracy"])
        uniform_tokens.append(cot["mean_tokens"])

        # Find best strategy for this domain
        best_strategy = None
        best_acc = -1
        for strategy_name, data in domain_results.items():
            strategy_matrix[domain][strategy_name] = {
                "accuracy": data["accuracy"],
                "mean_tokens": data["mean_tokens"],
            }
            if data["accuracy"] > best_acc or (
                data["accuracy"] == best_acc and data["mean_tokens"] < domain_results.get(best_strategy, {}).get("mean_tokens", float("inf"))
            ):
                best_acc = data["accuracy"]
                best_strategy = strategy_name

        per_domain_optimal[domain] = {
            "strategy": best_strategy,
            "accuracy": best_acc,
            "mean_tokens": domain_results[best_strategy]["mean_tokens"],
        }
        optimal_acc.append(best_acc)
        optimal_tokens.append(domain_results[best_strategy]["mean_tokens"])

    # K1279: Accuracy improvement (routed optimal vs uniform CoT)
    mean_uniform_acc = sum(uniform_acc) / len(uniform_acc)
    mean_optimal_acc = sum(optimal_acc) / len(optimal_acc)
    acc_improvement = mean_optimal_acc - mean_uniform_acc

    # K1280: Token reduction
    mean_uniform_tokens = sum(uniform_tokens) / len(uniform_tokens)
    mean_optimal_tokens = sum(optimal_tokens) / len(optimal_tokens)
    token_reduction = (mean_uniform_tokens - mean_optimal_tokens) / mean_uniform_tokens * 100 if mean_uniform_tokens > 0 else 0

    # K1281: Strategy selection = routing accuracy
    routing_accuracy = routing["accuracy"]

    # Strategy differentiation metric: mean δ̄
    deltas = [optimal_acc[i] - uniform_acc[i] for i in range(len(DOMAINS))]
    mean_delta = sum(deltas) / len(deltas)

    k1279_pass = acc_improvement >= 5.0
    k1280_pass = token_reduction >= 30.0
    k1281_pass = routing_accuracy >= 80.0

    kill_results = {
        "K1279": {
            "description": "Per-strategy routing accuracy >= +5pp vs uniform CoT",
            "uniform_cot_acc": round(mean_uniform_acc, 1),
            "routed_optimal_acc": round(mean_optimal_acc, 1),
            "improvement_pp": round(acc_improvement, 1),
            "threshold": 5.0,
            "pass": k1279_pass,
        },
        "K1280": {
            "description": "Token usage reduced >= 30% vs always-CoT",
            "uniform_cot_tokens": round(mean_uniform_tokens, 1),
            "routed_optimal_tokens": round(mean_optimal_tokens, 1),
            "reduction_pct": round(token_reduction, 1),
            "threshold": 30.0,
            "pass": k1280_pass,
        },
        "K1281": {
            "description": "Strategy selection accuracy >= 80%",
            "routing_accuracy": routing_accuracy,
            "threshold": 80.0,
            "pass": k1281_pass,
        },
        "strategy_differentiation": {
            "mean_delta_bar": round(mean_delta, 1),
            "per_domain_deltas": {d: round(deltas[i], 1) for i, d in enumerate(DOMAINS)},
        },
        "per_domain_optimal": per_domain_optimal,
        "strategy_matrix": strategy_matrix,
        "hypothesized_vs_actual": {
            domain: {
                "hypothesized": DOMAIN_STRATEGY_MAP[domain],
                "actual_optimal": per_domain_optimal[domain]["strategy"],
                "match": DOMAIN_STRATEGY_MAP[domain] == per_domain_optimal[domain]["strategy"],
            }
            for domain in DOMAINS
        },
    }

    log(f"  K1279: acc improvement = {acc_improvement:+.1f}pp (threshold +5pp) → {'PASS' if k1279_pass else 'FAIL'}")
    log(f"  K1280: token reduction = {token_reduction:.1f}% (threshold 30%) → {'PASS' if k1280_pass else 'FAIL'}")
    log(f"  K1281: routing accuracy = {routing_accuracy:.1f}% (threshold 80%) → {'PASS' if k1281_pass else 'FAIL'}")

    return kill_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log(f"P5.C1: Per-Adapter Reasoning Strategy")
    log(f"Model: {MODEL_ID}")
    log(f"Smoke test: {IS_SMOKE}")
    log_memory("start")

    # Phase 1: Generate and evaluate
    gen_results = phase_generate_and_evaluate()

    # Phase 2: TF-IDF routing
    routing = phase_tfidf_routing()

    # Phase 3: Kill criteria
    kill_results = compute_kill_criteria(gen_results, routing)

    # Save results
    total_time = round(time.time() - t0, 1)
    results = {
        "experiment": "exp_p5_per_adapter_reasoning_strategy",
        "model": MODEL_ID,
        "total_time_s": total_time,
        "kill_criteria": kill_results,
        "routing": routing,
        "generation_results": {
            domain: {
                strategy: {k: v for k, v in data.items() if k != "responses"}
                for strategy, data in domain_data.items()
            }
            for domain, domain_data in gen_results.items()
        },
        "sample_responses": {
            domain: {
                strategy: data["responses"][:1]
                for strategy, data in domain_data.items()
            }
            for domain, domain_data in gen_results.items()
        },
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")

    # Summary
    n_pass = sum(1 for k in ["K1279", "K1280", "K1281"] if kill_results[k]["pass"])
    log(f"\nKill criteria: {n_pass}/3 PASS")


if __name__ == "__main__":
    main()
