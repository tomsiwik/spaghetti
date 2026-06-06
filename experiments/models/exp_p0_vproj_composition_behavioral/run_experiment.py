#!/usr/bin/env python3
"""
P0.A0: v_proj+o_proj adapter composition — behavioral quality under parameter merging.

Reuses 5 trained adapters from exp_p8_vproj_domain_behavioral.
Tests composition at N=2 (all 10 pairs), N=5 (equal + peaked weights).
Evaluates behavioral quality via domain vocabulary shift.

Kill criteria (DB IDs):
  K1316: Mean behavioral quality >= 0.45 at N=2 composition
  K1317: Mean behavioral quality >= 0.35 at N=5 composition
  K1318: Per-domain retention >= 70% at N=5 (composed vs solo)
  K1319: Composition PPL degradation < 15% vs best single adapter

Grounded by:
  - Finding #504: v_proj+o_proj behavioral quality across 5 domains
  - Finding #287: Pierre unified pipeline 0.333 behavioral with q_proj
  - Finding #480: v_proj+o_proj causes 20% retention degradation
  - Finding #496: Null-space adapter averaging outperforms exclusive routing
  - DoRA (arXiv:2402.09353)

Composition method: concatenate weighted LoRA A/B matrices into a single
higher-rank adapter, load via mlx_lm's adapter mechanism (works with quantized base).
"""

import gc
import json
import os
import shutil
import time
from itertools import combinations
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Memory safety (CODING_GUIDELINES §2)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SOURCE_DIR = EXPERIMENT_DIR.parent / "exp_p8_vproj_domain_behavioral"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
COMPOSED_DIR = EXPERIMENT_DIR / "_composed_adapters"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 20
MAX_TOKENS = 300
LORA_SCALE = 4.0  # alpha/rank from P8 config

DOMAINS = ["math", "code", "medical", "legal", "finance"]


def cleanup():
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
# Domain glossaries and eval queries (from P8 experiment)
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_GLOSSARIES = {
    "math": [
        "theorem", "proof", "equation", "derivative", "integral", "polynomial",
        "coefficient", "eigenvalue", "eigenvector", "determinant", "matrix", "vector",
        "probability", "distribution", "convergence", "continuity", "differentiable",
        "binomial", "permutation", "combination", "exponent", "logarithm",
        "quadratic", "linear", "induction", "modular", "congruence", "fourier",
        "antiderivative", "discriminant", "characteristic",
    ],
    "code": [
        "function", "return", "parameter", "variable", "algorithm", "recursive",
        "iteration", "loop", "array", "list", "dictionary", "class", "method",
        "object", "inheritance", "polymorphism", "exception", "generator",
        "decorator", "comprehension", "complexity", "runtime", "syntax", "module",
        "import", "lambda", "closure", "callback", "coroutine", "thread",
    ],
    "medical": [
        "mechanism", "inhibitor", "receptor", "pharmacology", "clinical", "therapy",
        "diagnosis", "treatment", "pathophysiology", "enzyme", "protein",
        "antibody", "immune", "inflammation", "vascular", "cardiac", "neural",
        "medication", "dose", "adverse", "contraindicated", "efficacy", "etiology",
        "prognosis", "cytokine", "antibiotic", "prophylaxis", "comorbidity",
    ],
    "legal": [
        "jurisdiction", "plaintiff", "defendant", "precedent", "statute", "liability",
        "constitutional", "testimony", "verdict", "appeal", "amendment", "regulation",
        "enforceable", "judicial", "adversarial", "procedural", "substantive",
        "habeas", "corpus", "felony", "misdemeanor", "tort", "contract",
        "adjudication", "injunction", "indictment", "acquittal", "promissory",
    ],
    "finance": [
        "portfolio", "equity", "dividend", "revenue", "asset", "liability", "capital",
        "investment", "valuation", "appreciation", "depreciation", "amortization",
        "diversification", "volatility", "liquidity", "hedge", "derivative",
        "earnings", "shareholder", "fiscal", "compounding", "yield", "coupon",
        "leverage", "arbitrage", "beta", "alpha", "collateral",
    ],
}

EVAL_QUERIES = {
    "math": [
        "Explain what a limit is and how to compute it.",
        "What is the relationship between differentiation and integration?",
        "Describe how matrix multiplication works and its applications.",
        "Explain the concept of a probability density function.",
        "What is the mean value theorem and why is it important?",
        "Describe how to solve a system of linear equations.",
        "Explain what a differential equation is and give an example.",
        "What is the dot product and how is it used in geometry?",
        "Describe the properties of logarithmic functions.",
        "Explain the concept of convergence for infinite series.",
        "What is a partial derivative and when do you use it?",
        "Describe the relationship between sets, functions, and relations.",
        "Explain the binomial theorem and Pascal's triangle.",
        "What is the Pythagorean theorem and how can you prove it?",
        "Describe the concept of mathematical groups and symmetry.",
        "Explain what a Riemann sum is and how it relates to integrals.",
        "What is the squeeze theorem and when is it useful?",
        "Describe the method of Lagrange multipliers for optimization.",
        "Explain the concept of orthogonality in linear algebra.",
        "What is Green's theorem and how does it generalize?",
    ],
    "code": [
        "Explain how a hash table works internally.",
        "What is the difference between a stack and a queue?",
        "Describe how merge sort works and analyze its complexity.",
        "Explain what closures are in Python.",
        "What is dynamic programming and when should you use it?",
        "Describe how Python's garbage collector works.",
        "Explain the difference between shallow and deep copy.",
        "What are context managers and the with statement?",
        "Describe how a binary search tree works.",
        "Explain what unit testing is and why it matters.",
        "What is the GIL in Python and how does it affect concurrency?",
        "Describe how graph traversal algorithms (BFS, DFS) work.",
        "Explain what type hints are in Python and their benefits.",
        "What is memoization and how does it improve performance?",
        "Describe the observer design pattern.",
        "Explain how HTTP requests work in Python.",
        "What is the difference between composition and inheritance?",
        "Describe how regular expressions work.",
        "Explain what a virtual environment is and why you need one.",
        "What is test-driven development and how does it work?",
    ],
    "medical": [
        "Explain how statins work to lower cholesterol.",
        "What is the difference between bacterial and viral infections?",
        "Describe how the kidneys regulate blood pressure.",
        "Explain the mechanism of action of NSAIDs.",
        "What are autoimmune diseases and give examples.",
        "Describe the stages of cancer development.",
        "Explain how local anesthetics work at the molecular level.",
        "What is the role of the liver in drug metabolism?",
        "Describe the pathophysiology of heart failure.",
        "Explain how anticoagulants prevent blood clots.",
        "What are the different types of shock and their causes?",
        "Describe how the endocrine system regulates metabolism.",
        "Explain the mechanism of allergic reactions.",
        "What is sepsis and how is it managed?",
        "Describe the pharmacology of opioid analgesics.",
        "Explain how the respiratory system maintains acid-base balance.",
        "What are the mechanisms of drug-drug interactions?",
        "Describe the role of neurotransmitters in brain function.",
        "Explain the pathophysiology of chronic kidney disease.",
        "What is the significance of biomarkers in clinical diagnosis?",
    ],
    "legal": [
        "Explain the concept of sovereign immunity.",
        "What is the doctrine of promissory estoppel?",
        "Describe the elements of fraud in contract law.",
        "Explain how the Fourth Amendment protects against searches.",
        "What is strict liability and when does it apply?",
        "Describe the process of judicial review.",
        "Explain what administrative law is and how agencies operate.",
        "What is the parol evidence rule in contract interpretation?",
        "Describe the concept of qualified immunity for government officials.",
        "Explain the difference between real property and personal property.",
        "What is the commerce clause and its significance?",
        "Describe how class action lawsuits work.",
        "Explain the concept of equitable remedies.",
        "What are the Miranda rights and when must they be given?",
        "Describe the doctrine of respondeat superior.",
        "Explain the concept of eminent domain.",
        "What is the difference between arbitration and mediation?",
        "Describe the rule against perpetuities.",
        "Explain how intellectual property rights are protected.",
        "What is the concept of standing in federal court?",
    ],
    "finance": [
        "Explain what net present value means and how to calculate it.",
        "What is the efficient market hypothesis?",
        "Describe how options pricing works with Black-Scholes.",
        "Explain the concept of working capital management.",
        "What is dollar-cost averaging and when is it effective?",
        "Describe how credit ratings affect bond pricing.",
        "Explain the difference between fiscal and monetary policy.",
        "What is a mutual fund and how does it differ from an ETF?",
        "Describe the concept of financial leverage.",
        "Explain how currency exchange rates are determined.",
        "What is quantitative easing and how does it work?",
        "Describe the concept of yield curve and its implications.",
        "Explain what beta measures in the context of CAPM.",
        "What are derivatives and how are they used for hedging?",
        "Describe the concept of time value of money.",
        "Explain how mergers and acquisitions create value.",
        "What is the weighted average cost of capital?",
        "Describe the efficient frontier in portfolio theory.",
        "Explain the concept of moral hazard in financial markets.",
        "What is a credit default swap and how does it work?",
    ],
}


def score_vocabulary(text: str, glossary: list) -> int:
    """Count domain glossary terms appearing in text."""
    text_lower = text.lower()
    return sum(1 for term in glossary if term.lower() in text_lower)


# ══════════════════════════════════════════════════════════════════════════════
# Adapter composition via concatenated LoRA matrices
# ══════════════════════════════════════════════════════════════════════════════

def load_adapter_raw(domain: str) -> dict:
    """Load raw adapter weights (lora_a, lora_b per layer/module)."""
    adapter_path = SOURCE_DIR / f"adapter_{domain}" / "adapters.safetensors"
    return dict(mx.load(str(adapter_path)))


def create_composed_adapter(domain_adapters: dict, domains_to_compose: list,
                            weights_per_domain: dict, save_dir: Path) -> str:
    """Create a composed adapter by concatenating weighted LoRA matrices.

    For N adapters each rank r, the composed adapter has rank N*r.
    x @ A_composed @ B_composed = Σ w_i * x @ A_i @ B_i (exact).

    A_composed = [w_1*A_1 | w_2*A_2 | ... | w_N*A_N]  shape (in_dim, N*r)
    B_composed = [B_1; B_2; ...; B_N]                   shape (N*r, out_dim)

    mlx_lm applies: output = quant_W(x) + scale * x @ A_composed @ B_composed
    = quant_W(x) + scale * Σ w_i * x @ A_i @ B_i  ✓
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    first_adapter = domain_adapters[domains_to_compose[0]]
    all_keys = sorted(set(k.rsplit(".", 1)[0] for k in first_adapter.keys()))

    composed_weights = {}
    composed_rank = len(domains_to_compose) * 16  # N * original_rank

    for prefix in all_keys:
        a_key = f"{prefix}.lora_a"
        b_key = f"{prefix}.lora_b"

        a_parts = []
        b_parts = []
        for d in domains_to_compose:
            w = weights_per_domain[d]
            adapter = domain_adapters[d]
            if a_key in adapter and b_key in adapter:
                a_parts.append(w * adapter[a_key])  # weighted A
                b_parts.append(adapter[b_key])

        if a_parts and b_parts:
            a_composed = mx.concatenate(a_parts, axis=1)  # (in_dim, N*r)
            b_composed = mx.concatenate(b_parts, axis=0)  # (N*r, out_dim)
            mx.eval(a_composed, b_composed)
            composed_weights[a_key] = a_composed
            composed_weights[b_key] = b_composed

    adapter_file = save_dir / "adapters.safetensors"
    mx.save_safetensors(str(adapter_file), composed_weights)

    config = {
        "adapter_path": str(save_dir),
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": composed_rank,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": ["self_attn.v_proj", "self_attn.o_proj"],
        },
        "num_layers": 16,
        "model": MODEL_ID,
    }
    config_path = save_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config, indent=4))

    log(f"  Saved composed adapter: rank={composed_rank}, keys={len(composed_weights)}")
    return str(save_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: generate responses for a set of queries
# ══════════════════════════════════════════════════════════════════════════════

def generate_responses(model, tokenizer, queries: list, mlx_generate) -> list:
    """Generate responses and return list of response strings."""
    responses = []
    for q in queries:
        prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        resp = mlx_generate(model, tokenizer, prompt=prompt,
                            max_tokens=MAX_TOKENS, verbose=False)
        responses.append(resp)
    return responses


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Generate responses with base model
# ══════════════════════════════════════════════════════════════════════════════

def phase_base_eval() -> dict:
    """Generate base model responses for all domains."""
    log("\n=== Phase 1: Base Model Evaluation ===")
    from mlx_lm.utils import load as mlx_load
    from mlx_lm import generate as mlx_generate

    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("model-loaded")

    base_results = {}
    for domain in DOMAINS:
        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[domain]
        domain_results = []

        log(f"  [{domain}] Generating base responses ({len(queries)} queries)...")
        for i, q in enumerate(queries):
            prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
            resp = mlx_generate(model, tokenizer, prompt=prompt,
                                max_tokens=MAX_TOKENS, verbose=False)
            score = score_vocabulary(resp, glossary)
            domain_results.append({"query": q, "vocab_score": score, "response": resp})
            log(f"    [{i+1}/{len(queries)}] vocab={score}")

        base_results[domain] = domain_results
        mean_v = sum(r["vocab_score"] for r in domain_results) / len(domain_results)
        log(f"  [{domain}] mean_vocab={mean_v:.2f}")

    del model, tokenizer
    cleanup()
    log_memory("after-base-eval")
    return base_results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Solo adapter evaluation (for retention comparison)
# ══════════════════════════════════════════════════════════════════════════════

def phase_solo_eval() -> dict:
    """Generate solo adapter responses for all domains."""
    log("\n=== Phase 2: Solo Adapter Evaluation ===")
    from mlx_lm.utils import load as mlx_load
    from mlx_lm import generate as mlx_generate

    solo_results = {}
    for domain in DOMAINS:
        adapter_path = str(SOURCE_DIR / f"adapter_{domain}")
        model, tokenizer = mlx_load(MODEL_ID, adapter_path=adapter_path)

        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[domain]
        domain_results = []

        log(f"  [{domain}] Generating solo adapter responses ({len(queries)} queries)...")
        for i, q in enumerate(queries):
            prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
            resp = mlx_generate(model, tokenizer, prompt=prompt,
                                max_tokens=MAX_TOKENS, verbose=False)
            score = score_vocabulary(resp, glossary)
            domain_results.append({"query": q, "vocab_score": score})
            log(f"    [{i+1}/{len(queries)}] vocab={score}")

        solo_results[domain] = domain_results
        mean_v = sum(r["vocab_score"] for r in domain_results) / len(domain_results)
        log(f"  [{domain}] mean_vocab={mean_v:.2f}")

        del model, tokenizer
        cleanup()
        log_memory(f"after-solo-{domain}")

    return solo_results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: N=2 composition (all pairs)
# ══════════════════════════════════════════════════════════════════════════════

def phase_n2_composition(base_results: dict, domain_adapters: dict) -> dict:
    """Test all 10 pairs of 2-adapter compositions."""
    log("\n=== Phase 3: N=2 Composition (all pairs) ===")
    from mlx_lm.utils import load as mlx_load
    from mlx_lm import generate as mlx_generate

    pair_results = {}

    for d1, d2 in combinations(DOMAINS, 2):
        pair_key = f"{d1}+{d2}"
        log(f"\n  Composing [{pair_key}] with equal weights (0.5, 0.5)...")

        weights = {d1: 0.5, d2: 0.5}
        composed_dir = COMPOSED_DIR / f"n2_{d1}_{d2}"
        create_composed_adapter(domain_adapters, [d1, d2], weights, composed_dir)

        model, tokenizer = mlx_load(MODEL_ID, adapter_path=str(composed_dir))

        results_per_domain = {}
        for eval_domain in [d1, d2]:
            queries = EVAL_QUERIES[eval_domain][:N_EVAL]
            glossary = DOMAIN_GLOSSARIES[eval_domain]

            improved_count = 0
            for i, q in enumerate(queries):
                prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
                resp = mlx_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=MAX_TOKENS, verbose=False)
                score = score_vocabulary(resp, glossary)
                base_score = base_results[eval_domain][i]["vocab_score"]
                if score > base_score:
                    improved_count += 1

            rate = improved_count / len(queries)
            results_per_domain[eval_domain] = rate
            log(f"    [{eval_domain}] improvement_rate={rate:.2f}")

        pair_results[pair_key] = results_per_domain

        del model, tokenizer
        cleanup()
        shutil.rmtree(composed_dir, ignore_errors=True)
        log_memory(f"after-pair-{pair_key}")

    return pair_results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: N=5 composition (equal + peaked weights)
# ══════════════════════════════════════════════════════════════════════════════

def phase_n5_composition(base_results: dict, domain_adapters: dict) -> dict:
    """Test 5-adapter composition with equal and peaked weights."""
    log("\n=== Phase 4: N=5 Composition ===")
    from mlx_lm.utils import load as mlx_load
    from mlx_lm import generate as mlx_generate

    n5_results = {}

    # Test 1: Equal weights (1/5 each) — stress test
    log("\n  --- Equal weights (0.2 each) ---")
    equal_weights = {d: 0.2 for d in DOMAINS}
    composed_dir = COMPOSED_DIR / "n5_equal"
    create_composed_adapter(domain_adapters, DOMAINS, equal_weights, composed_dir)

    model, tokenizer = mlx_load(MODEL_ID, adapter_path=str(composed_dir))

    equal_results = {}
    for domain in DOMAINS:
        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[domain]

        improved_count = 0
        total_composed_vocab = 0
        for i, q in enumerate(queries):
            prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
            resp = mlx_generate(model, tokenizer, prompt=prompt,
                                max_tokens=MAX_TOKENS, verbose=False)
            score = score_vocabulary(resp, glossary)
            total_composed_vocab += score
            base_score = base_results[domain][i]["vocab_score"]
            if score > base_score:
                improved_count += 1

        rate = improved_count / len(queries)
        mean_vocab = total_composed_vocab / len(queries)
        equal_results[domain] = {"improvement_rate": rate, "mean_vocab": mean_vocab}
        log(f"    [{domain}] rate={rate:.2f} mean_vocab={mean_vocab:.2f}")

    n5_results["equal"] = equal_results

    del model, tokenizer
    cleanup()
    shutil.rmtree(composed_dir, ignore_errors=True)
    log_memory("after-n5-equal")

    # Test 2: Peaked weights (0.6 for target, 0.1 each for others)
    log("\n  --- Peaked weights (0.6 target, 0.1 others) ---")
    peaked_results = {}
    for target_domain in DOMAINS:
        peaked_weights = {d: 0.1 for d in DOMAINS}
        peaked_weights[target_domain] = 0.6

        composed_dir = COMPOSED_DIR / f"n5_peaked_{target_domain}"
        create_composed_adapter(domain_adapters, DOMAINS, peaked_weights, composed_dir)

        model, tokenizer = mlx_load(MODEL_ID, adapter_path=str(composed_dir))

        queries = EVAL_QUERIES[target_domain][:N_EVAL]
        glossary = DOMAIN_GLOSSARIES[target_domain]

        improved_count = 0
        total_composed_vocab = 0
        for i, q in enumerate(queries):
            prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
            resp = mlx_generate(model, tokenizer, prompt=prompt,
                                max_tokens=MAX_TOKENS, verbose=False)
            score = score_vocabulary(resp, glossary)
            total_composed_vocab += score
            base_score = base_results[target_domain][i]["vocab_score"]
            if score > base_score:
                improved_count += 1

        rate = improved_count / len(queries)
        mean_vocab = total_composed_vocab / len(queries)
        peaked_results[target_domain] = {"improvement_rate": rate, "mean_vocab": mean_vocab}
        log(f"    [{target_domain}] peaked rate={rate:.2f} mean_vocab={mean_vocab:.2f}")

        del model, tokenizer
        cleanup()
        shutil.rmtree(composed_dir, ignore_errors=True)
        log_memory(f"after-n5-peaked-{target_domain}")

    n5_results["peaked"] = peaked_results

    return n5_results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: PPL measurement for K1319
# ══════════════════════════════════════════════════════════════════════════════

def phase_ppl_comparison(domain_adapters: dict) -> dict:
    """Measure PPL degradation at N=5 equal-weight composition."""
    log("\n=== Phase 5: PPL Comparison ===")
    from mlx_lm.utils import load as mlx_load

    # Use eval queries as PPL test set
    test_texts = []
    for domain in DOMAINS:
        for q in EVAL_QUERIES[domain][:4]:  # 4 per domain = 20 total
            test_texts.append(q)

    def measure_ppl(model, tokenizer, texts):
        """Measure perplexity on a set of texts."""
        total_nll = 0.0
        total_tokens = 0
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])
            logits = model(x)
            logits = logits.squeeze(0)  # (seq_len, vocab)
            log_probs = mx.softmax(logits, axis=-1)
            target_probs = log_probs[mx.arange(y.shape[0]), y]
            nll = -mx.sum(mx.log(target_probs + 1e-10))
            mx.eval(nll)
            total_nll += nll.item()
            total_tokens += y.shape[0]
            del logits, log_probs, target_probs, nll, x, y

        return float(total_nll / total_tokens) if total_tokens > 0 else float('inf')

    # Base model PPL
    log("  Measuring base model PPL...")
    model, tokenizer = mlx_load(MODEL_ID)
    base_ppl = measure_ppl(model, tokenizer, test_texts)
    log(f"  Base PPL (NLL/token): {base_ppl:.4f}")
    del model, tokenizer
    cleanup()

    # Solo adapter PPL (best single adapter on matched domain)
    log("  Measuring solo adapter PPL...")
    solo_ppls = {}
    for domain in DOMAINS:
        adapter_path = str(SOURCE_DIR / f"adapter_{domain}")
        model, tokenizer = mlx_load(MODEL_ID, adapter_path=adapter_path)
        domain_texts = EVAL_QUERIES[domain][:4]
        ppl = measure_ppl(model, tokenizer, domain_texts)
        solo_ppls[domain] = ppl
        log(f"    [{domain}] PPL={ppl:.4f}")
        del model, tokenizer
        cleanup()

    # N=5 equal-weight composed PPL
    log("  Measuring N=5 equal-weight composed PPL...")
    equal_weights = {d: 0.2 for d in DOMAINS}
    composed_dir = COMPOSED_DIR / "ppl_n5_equal"
    create_composed_adapter(domain_adapters, DOMAINS, equal_weights, composed_dir)

    model, tokenizer = mlx_load(MODEL_ID, adapter_path=str(composed_dir))
    composed_ppl = measure_ppl(model, tokenizer, test_texts)
    log(f"  Composed PPL (NLL/token): {composed_ppl:.4f}")

    # Per-domain PPL with composed model
    composed_domain_ppls = {}
    for domain in DOMAINS:
        domain_texts = EVAL_QUERIES[domain][:4]
        ppl = measure_ppl(model, tokenizer, domain_texts)
        composed_domain_ppls[domain] = ppl
        log(f"    [{domain}] composed PPL={ppl:.4f}")

    del model, tokenizer
    cleanup()
    shutil.rmtree(composed_dir, ignore_errors=True)
    log_memory("after-ppl")

    # Compute degradation: for each domain, compare composed vs solo
    degradation = {}
    for domain in DOMAINS:
        solo = solo_ppls[domain]
        comp = composed_domain_ppls[domain]
        deg = (comp - solo) / solo if solo > 0 else 0.0
        degradation[domain] = deg
        log(f"    [{domain}] PPL degradation: {deg*100:.1f}%")

    return {
        "base_ppl": base_ppl,
        "solo_ppls": solo_ppls,
        "composed_ppl": composed_ppl,
        "composed_domain_ppls": composed_domain_ppls,
        "degradation_per_domain": degradation,
        "max_degradation": max(degradation.values()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("P0.A0: v_proj+o_proj Adapter Composition — Behavioral Quality")
    log(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, MAX_TOKENS={MAX_TOKENS}")
    log(f"Source adapters: {SOURCE_DIR}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # Verify source adapters exist
    for domain in DOMAINS:
        adapter_file = SOURCE_DIR / f"adapter_{domain}" / "adapters.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(f"Missing adapter: {adapter_file}")
    log("All 5 source adapters verified.")

    # Clean up any leftover composed adapters
    if COMPOSED_DIR.exists():
        shutil.rmtree(COMPOSED_DIR)
    COMPOSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load all raw adapter weights (small: ~64 weight matrices * 5 domains)
    log("\n=== Loading raw adapter weights ===")
    domain_adapters = {}
    for domain in DOMAINS:
        domain_adapters[domain] = load_adapter_raw(domain)
        log(f"  [{domain}] loaded {len(domain_adapters[domain])} weight matrices")
    log_memory("adapters-loaded")

    # Phase 1: Base model evaluation
    base_results = phase_base_eval()

    # Phase 2: Solo adapter evaluation
    solo_results = phase_solo_eval()

    # Compute solo improvement rates
    solo_rates = {}
    for domain in DOMAINS:
        base = base_results[domain]
        solo = solo_results[domain]
        improved = sum(
            1 for b, s in zip(base, solo)
            if s["vocab_score"] > b["vocab_score"]
        )
        solo_rates[domain] = improved / len(base)
    log(f"\nSolo improvement rates: {solo_rates}")

    # Phase 3: N=2 composition
    n2_results = phase_n2_composition(base_results, domain_adapters)

    # Phase 4: N=5 composition
    n5_results = phase_n5_composition(base_results, domain_adapters)

    # Phase 5: PPL comparison
    ppl_results = phase_ppl_comparison(domain_adapters)

    # Free raw adapters
    del domain_adapters
    cleanup()

    # Clean up composed dir
    if COMPOSED_DIR.exists():
        shutil.rmtree(COMPOSED_DIR, ignore_errors=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Analyze results and check kill criteria
    # ══════════════════════════════════════════════════════════════════════════

    log("\n" + "=" * 70)
    log("RESULTS ANALYSIS")
    log("=" * 70)

    # K1316: N=2 mean behavioral quality >= 0.45
    n2_rates_per_domain = {d: [] for d in DOMAINS}
    for pair_key, pair_data in n2_results.items():
        for domain, rate in pair_data.items():
            n2_rates_per_domain[domain].append(rate)

    n2_mean_per_domain = {}
    for domain in DOMAINS:
        rates = n2_rates_per_domain[domain]
        n2_mean_per_domain[domain] = sum(rates) / len(rates) if rates else 0.0

    n2_overall_mean = sum(n2_mean_per_domain.values()) / len(n2_mean_per_domain)
    k1316_pass = n2_overall_mean >= 0.45
    log(f"\nK1316 (N=2 behavioral >= 0.45): {'PASS' if k1316_pass else 'FAIL'}")
    log(f"  N=2 overall mean: {n2_overall_mean:.3f}")
    for d in DOMAINS:
        log(f"  [{d}] mean across pairs: {n2_mean_per_domain[d]:.3f} (solo: {solo_rates[d]:.3f})")

    # K1317: N=5 mean behavioral quality >= 0.35
    peaked = n5_results.get("peaked", {})
    n5_peaked_rates = {d: peaked.get(d, {}).get("improvement_rate", 0.0) for d in DOMAINS}
    n5_peaked_mean = sum(n5_peaked_rates.values()) / len(n5_peaked_rates)

    equal = n5_results.get("equal", {})
    n5_equal_rates = {d: equal.get(d, {}).get("improvement_rate", 0.0) for d in DOMAINS}
    n5_equal_mean = sum(n5_equal_rates.values()) / len(n5_equal_rates)

    k1317_pass = n5_peaked_mean >= 0.35
    log(f"\nK1317 (N=5 behavioral >= 0.35): {'PASS' if k1317_pass else 'FAIL'}")
    log(f"  N=5 equal-weight mean: {n5_equal_mean:.3f}")
    log(f"  N=5 peaked-weight mean: {n5_peaked_mean:.3f}")
    for d in DOMAINS:
        log(f"  [{d}] equal={n5_equal_rates[d]:.3f} peaked={n5_peaked_rates[d]:.3f} solo={solo_rates[d]:.3f}")

    # K1318: Per-domain retention >= 70% at N=5 peaked
    retention_per_domain = {}
    for d in DOMAINS:
        solo = solo_rates[d]
        comp = n5_peaked_rates[d]
        retention = comp / solo if solo > 0 else 1.0
        retention_per_domain[d] = retention

    min_retention = min(retention_per_domain.values())
    k1318_pass = min_retention >= 0.70
    log(f"\nK1318 (per-domain retention >= 70% at N=5): {'PASS' if k1318_pass else 'FAIL'}")
    log(f"  Min retention: {min_retention:.3f}")
    for d in DOMAINS:
        log(f"  [{d}] retention={retention_per_domain[d]:.3f} (solo={solo_rates[d]:.3f} comp={n5_peaked_rates[d]:.3f})")

    # K1319: PPL degradation < 15%
    max_deg = ppl_results["max_degradation"]
    k1319_pass = max_deg < 0.15
    log(f"\nK1319 (PPL degradation < 15%): {'PASS' if k1319_pass else 'FAIL'}")
    log(f"  Max PPL degradation: {max_deg*100:.1f}%")

    total_time = (time.time() - total_start) / 60.0
    log(f"\nTotal experiment time: {total_time:.1f} min")

    # Save results
    results = {
        "is_smoke": IS_SMOKE,
        "config": {
            "n_eval": N_EVAL,
            "max_tokens": MAX_TOKENS,
            "lora_scale": LORA_SCALE,
            "source_dir": str(SOURCE_DIR),
        },
        "solo_rates": solo_rates,
        "n2_results": n2_results,
        "n2_mean_per_domain": n2_mean_per_domain,
        "n2_overall_mean": n2_overall_mean,
        "n5_results": {
            "equal": {d: v for d, v in n5_results.get("equal", {}).items()},
            "peaked": {d: v for d, v in n5_results.get("peaked", {}).items()},
        },
        "n5_peaked_mean": n5_peaked_mean,
        "n5_equal_mean": n5_equal_mean,
        "ppl": ppl_results,
        "retention_at_n5": retention_per_domain,
        "kill_criteria": {
            "k1316_n2_behavioral": {"pass": k1316_pass, "value": n2_overall_mean, "threshold": 0.45},
            "k1317_n5_behavioral": {"pass": k1317_pass, "value": n5_peaked_mean, "threshold": 0.35},
            "k1318_retention": {"pass": k1318_pass, "value": min_retention, "threshold": 0.70},
            "k1319_ppl_degradation": {"pass": k1319_pass, "value": max_deg, "threshold": 0.15},
        },
        "all_pass": k1316_pass and k1317_pass and k1318_pass and k1319_pass,
        "total_time_min": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Overall: {'ALL PASS' if results['all_pass'] else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
