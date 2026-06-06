#!/usr/bin/env python3
"""
P4.B0: Domain Adapter Quality Benchmark — Factual Accuracy vs Base.

Evaluates whether domain adapters improve factual accuracy on domain-specific
questions using a keyword-based rubric. Tests 5 domains (medical, code, math,
legal, finance) + biology (from P4.A1).

Phases:
  Phase 1: Base model evaluation on all domains (N_EVAL questions per domain)
  Phase 2: Per-adapter evaluation — own domain improvement + cross-domain regression
  Phase 3: Aggregate statistics and kill criteria check

Kill criteria:
  K1224: ≥3 of 5 original domains show ≥10pp factual accuracy improvement over base
  K1225: composition maintains ≥90% accuracy on non-target domains (no regression)
  K1226: average adapted accuracy ≥ 50% on held-out domain questions

Grounded by:
  - Finding #421: T2.1 domain adapters +22-82pp on MCQ benchmarks (math/code/medical)
  - Finding #228: Grassmannian separation (max pairwise cosine = 2.25e-8)
  - Finding #466: Domain-conditional retrain (0pp composition degradation)
  - Finding #475: Biology adapter +20pp vocabulary improvement (P4.A1)
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Model and adapter paths
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

T2_SINGLE_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training" / "adapters"
T2_MULTI_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5" / "adapters"
P4A1_DIR = EXPERIMENT_DIR.parent / "exp_p4_a1_domain_adapter_speedtest" / "biology_adapter"

ADAPTER_PATHS = {
    "medical": T2_SINGLE_DIR / "medical",
    "code":    T2_SINGLE_DIR / "code",
    "math":    T2_SINGLE_DIR / "math",
    "legal":   T2_MULTI_DIR / "legal",
    "finance": T2_MULTI_DIR / "finance",
    "biology": P4A1_DIR,
}

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL_FULL = 15     # questions per domain for full run
N_EVAL_SMOKE = 3     # questions per domain for smoke
N_CROSS_FULL = 3     # cross-domain questions per non-target domain for full run
N_CROSS_SMOKE = 1    # cross-domain questions per non-target domain for smoke

N_EVAL  = N_EVAL_SMOKE  if IS_SMOKE else N_EVAL_FULL
N_CROSS = N_CROSS_SMOKE if IS_SMOKE else N_CROSS_FULL
MAX_TOKENS = 150     # factual answers need room for terminology

# ──────────────────────────────────────────────────────────────────────
# Domain Q&A test sets with keyword scoring rubrics
# Format: {"q": question, "keys": [key_facts_to_check]}
# Score = #{keys found in response} / len(keys)
# ──────────────────────────────────────────────────────────────────────

DOMAIN_QA = {
    "medical": [
        # Single-concept keywords that appear naturally in any correct answer
        {
            "q": "What is the mechanism of action of ACE inhibitors?",
            "keys": ["angiotensin", "bradykinin", "vasodilation", "blood pressure", "inhibit"]
        },
        {
            "q": "Describe the pathophysiology of type 2 diabetes mellitus.",
            "keys": ["insulin resistance", "beta cell", "glucose", "pancreas", "hyperglycemia"]
        },
        {
            "q": "What are the symptoms and causes of myocardial infarction?",
            "keys": ["coronary", "ischemia", "chest pain", "troponin", "plaque"]
        },
        {
            "q": "How do beta-blockers reduce heart rate?",
            "keys": ["beta receptor", "sympathetic", "epinephrine", "heart rate", "adrenergic"]
        },
        {
            "q": "What is the role of the renin-angiotensin-aldosterone system?",
            "keys": ["renin", "angiotensin", "aldosterone", "sodium", "blood pressure"]
        },
        {
            "q": "Explain how statins lower cholesterol.",
            "keys": ["HMG-CoA", "cholesterol", "liver", "LDL", "reductase"]
        },
        {
            "q": "What are the signs of septic shock?",
            "keys": ["hypotension", "infection", "vasodilation", "cytokine", "organ"]
        },
        {
            "q": "How does heparin work as an anticoagulant?",
            "keys": ["antithrombin", "thrombin", "coagulation", "clot", "factor"]
        },
        {
            "q": "What causes pneumonia and how is it diagnosed?",
            "keys": ["bacterial", "lung", "fever", "chest", "infiltrate"]
        },
        {
            "q": "Describe the mechanism of anaphylaxis.",
            "keys": ["IgE", "mast cell", "histamine", "allergen", "epinephrine"]
        },
        {
            "q": "What is the function of the glomerulus in the kidney?",
            "keys": ["filtration", "Bowman", "capillary", "GFR", "plasma"]
        },
        {
            "q": "How do SSRIs treat depression?",
            "keys": ["serotonin", "reuptake", "synapse", "neurotransmitter", "receptor"]
        },
        {
            "q": "What is the difference between Type I and Type II hypersensitivity?",
            "keys": ["IgE", "immediate", "antibody", "complement", "cell-mediated"]
        },
        {
            "q": "What causes hypertension and what are its complications?",
            "keys": ["blood pressure", "cardiovascular", "stroke", "renal", "arteriosclerosis"]
        },
        {
            "q": "How does the immune system distinguish self from non-self?",
            "keys": ["MHC", "T cell", "thymus", "tolerance", "antigen"]
        },
    ],

    "code": [
        # Short keywords; avoid exact mathematical notation with superscripts
        {
            "q": "What is the time and space complexity of quicksort?",
            "keys": ["n log n", "pivot", "partition", "worst case", "average"]
        },
        {
            "q": "Explain the difference between a stack and a queue.",
            "keys": ["LIFO", "FIFO", "push", "first in", "last in"]
        },
        {
            "q": "What is dynamic programming and when is it used?",
            "keys": ["subproblem", "memoization", "optimal", "overlapping", "recurrence"]
        },
        {
            "q": "How does a hash table handle collisions?",
            "keys": ["chaining", "open addressing", "load factor", "bucket", "probing"]
        },
        {
            "q": "What is the difference between BFS and DFS graph traversal?",
            "keys": ["breadth", "depth", "queue", "stack", "shortest path"]
        },
        {
            "q": "Explain binary search and its time complexity.",
            "keys": ["log n", "sorted", "midpoint", "divide", "halve"]
        },
        {
            "q": "What is a binary search tree and its properties?",
            "keys": ["left subtree", "right subtree", "sorted", "search", "balanced"]
        },
        {
            "q": "Describe the difference between TCP and UDP.",
            "keys": ["reliable", "connection", "handshake", "datagram", "acknowledgment"]
        },
        {
            "q": "What is memoization and how does it improve recursive algorithms?",
            "keys": ["cache", "subproblem", "top-down", "recursive", "repeated"]
        },
        {
            "q": "How does garbage collection work in languages like Python or Java?",
            "keys": ["reference", "mark", "sweep", "heap", "memory"]
        },
        {
            "q": "What is the difference between processes and threads?",
            "keys": ["memory", "context switch", "concurrency", "shared", "lightweight"]
        },
        {
            "q": "Explain Big O notation with examples.",
            "keys": ["O(1)", "O(n)", "asymptotic", "worst case", "growth"]
        },
        {
            "q": "What is a heap data structure and how is it used?",
            "keys": ["max-heap", "priority queue", "heapify", "parent", "complete"]
        },
        {
            "q": "How does merge sort work and what is its complexity?",
            "keys": ["divide", "n log n", "merge", "subarray", "stable"]
        },
        {
            "q": "What is a graph and what are common graph algorithms?",
            "keys": ["vertices", "edges", "Dijkstra", "adjacency", "directed"]
        },
    ],

    "math": [
        # Avoid unicode superscripts; use simple text patterns
        {
            "q": "State and explain the Pythagorean theorem.",
            "keys": ["right triangle", "hypotenuse", "legs", "squares", "a^2"]
        },
        {
            "q": "What is the fundamental theorem of calculus?",
            "keys": ["derivative", "integral", "antiderivative", "continuous", "area"]
        },
        {
            "q": "Explain the concept of a limit in calculus.",
            "keys": ["approaches", "epsilon", "delta", "converge", "infinity"]
        },
        {
            "q": "What is the chain rule in differentiation?",
            "keys": ["composite", "derivative", "outer", "inner", "f(g(x))"]
        },
        {
            "q": "How do you compute the eigenvalues of a matrix?",
            "keys": ["determinant", "characteristic", "eigenvalue", "eigenvector", "lambda"]
        },
        {
            "q": "What is the Taylor series expansion?",
            "keys": ["derivative", "polynomial", "factorial", "convergence", "approximation"]
        },
        {
            "q": "Explain the central limit theorem.",
            "keys": ["sample mean", "normal distribution", "variance", "independent", "average"]
        },
        {
            "q": "What is Bayes' theorem and how is it applied?",
            "keys": ["posterior", "prior", "likelihood", "conditional", "probability"]
        },
        {
            "q": "How does the Euclidean algorithm compute GCD?",
            "keys": ["remainder", "modulo", "divisor", "recursive", "divide"]
        },
        {
            "q": "What is integration by parts?",
            "keys": ["product", "u dv", "differentiate", "integrate", "formula"]
        },
        {
            "q": "Define a vector space and its properties.",
            "keys": ["closure", "scalar", "addition", "zero vector", "axiom"]
        },
        {
            "q": "What is the determinant of a matrix and why is it important?",
            "keys": ["invertible", "singular", "transformation", "cofactor", "volume"]
        },
        {
            "q": "Explain the difference between permutations and combinations.",
            "keys": ["order", "factorial", "choose", "arrangement", "selection"]
        },
        {
            "q": "What is a proof by induction?",
            "keys": ["base case", "inductive", "hypothesis", "n+1", "assume"]
        },
        {
            "q": "What is L'Hôpital's rule and when is it used?",
            "keys": ["indeterminate", "derivative", "limit", "0/0", "infinity"]
        },
    ],

    "legal": [
        {
            "q": "What is the difference between civil law and criminal law?",
            "keys": ["plaintiff", "defendant", "reasonable doubt", "preponderance", "burden"]
        },
        {
            "q": "Explain the concept of habeas corpus.",
            "keys": ["detention", "writ", "prisoner", "release", "unlawful"]
        },
        {
            "q": "What are the elements of a valid contract?",
            "keys": ["offer", "acceptance", "consideration", "capacity", "assent"]
        },
        {
            "q": "What is the doctrine of precedent (stare decisis)?",
            "keys": ["binding", "prior case", "lower court", "common law", "precedent"]
        },
        {
            "q": "Explain the concept of mens rea in criminal law.",
            "keys": ["guilty mind", "intent", "actus reus", "negligence", "criminal"]
        },
        {
            "q": "What is the Fourth Amendment and how does it apply to search and seizure?",
            "keys": ["unreasonable", "warrant", "probable cause", "exclusionary", "privacy"]
        },
        {
            "q": "What is negligence in tort law?",
            "keys": ["duty of care", "breach", "causation", "damages", "reasonable"]
        },
        {
            "q": "Explain the difference between copyright, patent, and trademark.",
            "keys": ["creative", "invention", "brand", "intellectual property", "infringement"]
        },
        {
            "q": "What is the principle of double jeopardy?",
            "keys": ["twice", "same offense", "acquittal", "conviction", "prosecution"]
        },
        {
            "q": "How does the Miranda warning protect defendants?",
            "keys": ["silence", "attorney", "custodial", "interrogation", "self-incrimination"]
        },
        {
            "q": "What is the difference between a felony and a misdemeanor?",
            "keys": ["severity", "prison", "jail", "sentence", "crime"]
        },
        {
            "q": "Explain the concept of due process.",
            "keys": ["procedural", "substantive", "fair", "constitutional", "Fourteenth"]
        },
        {
            "q": "What is strict liability in tort law?",
            "keys": ["fault", "dangerous", "defective", "product", "liability"]
        },
        {
            "q": "What are the requirements for a valid will?",
            "keys": ["testator", "witness", "signature", "capacity", "intent"]
        },
        {
            "q": "Explain the concept of judicial review.",
            "keys": ["constitutionality", "Supreme Court", "legislation", "unconstitutional", "review"]
        },
    ],

    "finance": [
        {
            "q": "What is the time value of money?",
            "keys": ["present value", "future value", "discount", "opportunity cost", "interest"]
        },
        {
            "q": "Explain the concept of compound interest.",
            "keys": ["principal", "compound", "interest", "period", "exponential"]
        },
        {
            "q": "What is the Capital Asset Pricing Model (CAPM)?",
            "keys": ["risk-free", "beta", "market", "expected return", "systematic"]
        },
        {
            "q": "What is portfolio diversification and why is it important?",
            "keys": ["risk", "correlation", "portfolio", "variance", "unsystematic"]
        },
        {
            "q": "Explain the difference between stocks and bonds.",
            "keys": ["equity", "debt", "dividend", "coupon", "ownership"]
        },
        {
            "q": "What is net present value (NPV) and how is it used?",
            "keys": ["discounted", "cash flow", "investment", "positive", "rate"]
        },
        {
            "q": "What is the efficient market hypothesis?",
            "keys": ["price", "information", "market", "weak form", "arbitrage"]
        },
        {
            "q": "Explain leverage and its risks in investing.",
            "keys": ["debt", "amplify", "margin", "risk", "return"]
        },
        {
            "q": "What is duration in fixed income investing?",
            "keys": ["interest rate", "maturity", "price", "sensitivity", "bond"]
        },
        {
            "q": "How does the Federal Reserve control money supply?",
            "keys": ["interest rate", "open market", "reserve", "monetary policy", "fed"]
        },
        {
            "q": "What is the Black-Scholes option pricing model?",
            "keys": ["option", "volatility", "risk-free", "expiry", "strike"]
        },
        {
            "q": "Explain the concept of beta in finance.",
            "keys": ["market risk", "covariance", "systematic", "correlation", "index"]
        },
        {
            "q": "What is the difference between systematic and unsystematic risk?",
            "keys": ["market risk", "company-specific", "diversification", "macroeconomic", "eliminate"]
        },
        {
            "q": "What is yield to maturity (YTM) for a bond?",
            "keys": ["coupon", "face value", "price", "return", "hold"]
        },
        {
            "q": "What is arbitrage and why does it eliminate price differences?",
            "keys": ["riskless", "mispricing", "simultaneous", "profit", "equilibrium"]
        },
    ],

    "biology": [
        {
            "q": "Explain the central dogma of molecular biology.",
            "keys": ["DNA", "RNA", "transcription", "translation", "protein"]
        },
        {
            "q": "What is the role of mitochondria in cellular metabolism?",
            "keys": ["ATP", "oxidative phosphorylation", "electron transport", "energy", "respiration"]
        },
        {
            "q": "How does natural selection lead to evolution?",
            "keys": ["variation", "fitness", "reproduction", "adaptation", "selection"]
        },
        {
            "q": "What is the difference between mitosis and meiosis?",
            "keys": ["diploid", "haploid", "chromosome", "gamete", "division"]
        },
        {
            "q": "Describe the structure of DNA.",
            "keys": ["double helix", "nucleotide", "base pair", "adenine", "phosphate"]
        },
        {
            "q": "How does enzyme catalysis work?",
            "keys": ["active site", "substrate", "activation energy", "catalyst", "reaction"]
        },
        {
            "q": "What is CRISPR-Cas9 and how does it edit genes?",
            "keys": ["guide RNA", "Cas9", "DNA", "edit", "cut"]
        },
        {
            "q": "Explain the process of photosynthesis.",
            "keys": ["chlorophyll", "light", "Calvin cycle", "glucose", "chloroplast"]
        },
        {
            "q": "What is the role of ribosomes in protein synthesis?",
            "keys": ["mRNA", "tRNA", "amino acid", "codon", "polypeptide"]
        },
        {
            "q": "How do viruses replicate inside host cells?",
            "keys": ["host cell", "capsid", "genome", "replication", "inject"]
        },
        {
            "q": "What is osmosis and how does it affect cells?",
            "keys": ["membrane", "water", "concentration", "osmotic", "hypotonic"]
        },
        {
            "q": "Explain Mendelian genetics and the law of segregation.",
            "keys": ["allele", "dominant", "recessive", "gamete", "segregation"]
        },
        {
            "q": "What are the stages of cellular respiration?",
            "keys": ["glycolysis", "Krebs", "electron transport", "NADH", "ATP"]
        },
        {
            "q": "How does the immune system recognize pathogens?",
            "keys": ["antigen", "antibody", "T cell", "B cell", "receptor"]
        },
        {
            "q": "What is epigenetics and how does it influence gene expression?",
            "keys": ["methylation", "histone", "chromatin", "expression", "heritable"]
        },
    ],
}

ORIGINAL_DOMAINS = ["medical", "code", "math", "legal", "finance"]
ALL_DOMAINS = ORIGINAL_DOMAINS + ["biology"]


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

def keyword_score(response: str, keys: list[str]) -> float:
    """Fraction of key facts found in response (case-insensitive substring match)."""
    response_lower = response.lower()
    hits = sum(1 for k in keys if k.lower() in response_lower)
    return hits / len(keys) if keys else 0.0


def format_prompt(question: str, tokenizer) -> str:
    """Format question as Gemma 4 chat prompt."""
    messages = [
        {"role": "user", "content": question}
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_domain(model, tokenizer, domain: str, n_questions: int) -> dict:
    """Evaluate model on N questions for a domain. Returns per-question scores."""
    from mlx_lm import generate as mlx_generate

    qa_list = DOMAIN_QA[domain][:n_questions]
    scores = []

    for qa in qa_list:
        prompt = format_prompt(qa["q"], tokenizer)
        response = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )
        score = keyword_score(response, qa["keys"])
        scores.append(score)

    return {
        "domain": domain,
        "n": len(scores),
        "scores": scores,
        "mean_score": float(sum(scores) / len(scores)) if scores else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    from mlx_lm import load as mlx_load

    t_start = time.time()
    print(f"P4.B0: Domain Adapter Quality Benchmark")
    print(f"  IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, N_CROSS={N_CROSS}")
    print(f"  MAX_TOKENS={MAX_TOKENS}")
    print()

    # Verify adapter paths exist
    missing = []
    for domain, path in ADAPTER_PATHS.items():
        if not path.exists():
            missing.append(f"{domain}: {path}")
    if missing:
        print("WARNING: Missing adapters (skipping):")
        for m in missing:
            print(f"  {m}")
    available_domains = [d for d in ORIGINAL_DOMAINS if ADAPTER_PATHS[d].exists()]
    if ADAPTER_PATHS["biology"].exists():
        available_domains_all = available_domains + ["biology"]
    else:
        available_domains_all = available_domains
    print(f"Available domains: {available_domains_all}")
    print()

    # ── Phase 1: Base model evaluation ──────────────────────────────
    print("=== Phase 1: Base Model Evaluation ===")
    t_phase1 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    print(f"Model loaded: {time.time() - t_phase1:.1f}s")

    base_results = {}
    for domain in available_domains_all:
        t0 = time.time()
        result = evaluate_domain(model, tokenizer, domain, N_EVAL)
        base_results[domain] = result
        print(f"  Base {domain}: mean={result['mean_score']:.3f} ({time.time()-t0:.1f}s)")

    # Collect cross-domain base scores (N_CROSS per domain, for regression check)
    base_cross = {}  # domain -> {query_domain -> scores}
    for eval_domain in available_domains_all:
        base_cross[eval_domain] = {}
        qa_cross = DOMAIN_QA[eval_domain][:N_CROSS]
        from mlx_lm import generate as mlx_generate
        cross_scores = []
        for qa in qa_cross:
            prompt = format_prompt(qa["q"], tokenizer)
            response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
            cross_scores.append(keyword_score(response, qa["keys"]))
        base_cross[eval_domain] = cross_scores

    print(f"Phase 1 done: {time.time() - t_phase1:.1f}s")
    print()

    # ── Phase 2: Adapter evaluation ─────────────────────────────────
    print("=== Phase 2: Adapter Evaluation ===")

    # Free base model before reloading with adapters (not strictly necessary,
    # but avoids double-loading weight pages)
    del model
    gc.collect()

    adapted_results = {}    # domain -> own domain results
    cross_domain_results = {}  # adapter_domain -> {query_domain -> adapted_scores}
    regression_ratios = {}  # adapter_domain -> mean regression ratio across cross-domains

    for adapter_domain in available_domains:  # Only original 5 (with trained adapters)
        adapter_path = ADAPTER_PATHS[adapter_domain]
        if not adapter_path.exists():
            print(f"  Skipping {adapter_domain} (adapter not found)")
            continue

        print(f"  Loading adapter: {adapter_domain}")
        t_load = time.time()
        model_a, tokenizer_a = mlx_load(MODEL_ID, adapter_path=str(adapter_path))
        mx.eval(model_a.parameters())
        print(f"    Loaded in {time.time() - t_load:.1f}s")

        # Own domain eval
        t0 = time.time()
        result = evaluate_domain(model_a, tokenizer_a, adapter_domain, N_EVAL)
        adapted_results[adapter_domain] = result
        base_score = base_results[adapter_domain]["mean_score"]
        adapted_score = result["mean_score"]
        improvement = (adapted_score - base_score) * 100
        print(f"    {adapter_domain}: base={base_score:.3f} → adapted={adapted_score:.3f} ({improvement:+.1f}pp)")

        # Cross-domain regression check
        from mlx_lm import generate as mlx_generate
        cross_domain_results[adapter_domain] = {}
        regressions = []
        for cross_domain in available_domains_all:
            if cross_domain == adapter_domain:
                continue
            qa_cross = DOMAIN_QA[cross_domain][:N_CROSS]
            adapted_cross_scores = []
            for qa in qa_cross:
                prompt = format_prompt(qa["q"], tokenizer_a)
                response = mlx_generate(model_a, tokenizer_a, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
                adapted_cross_scores.append(keyword_score(response, qa["keys"]))
            cross_domain_results[adapter_domain][cross_domain] = adapted_cross_scores

            # Compute regression ratio
            base_c = base_cross[cross_domain]
            adap_c = adapted_cross_scores
            # Ratio = adapted_mean / base_mean (capped at 1.0 if improvement)
            base_mean_c = sum(base_c) / len(base_c) if base_c else 1.0
            adap_mean_c = sum(adap_c) / len(adap_c) if adap_c else 1.0
            if base_mean_c > 0:
                ratio = min(1.0, adap_mean_c / base_mean_c)
            else:
                ratio = 1.0  # base was 0, any answer is a tie
            regressions.append(ratio)

        mean_regression = float(sum(regressions) / len(regressions)) if regressions else 1.0
        regression_ratios[adapter_domain] = mean_regression
        print(f"    cross-domain retention: {mean_regression:.3f}")
        print(f"    {adapter_domain} adapter eval: {time.time()-t0:.1f}s")

        del model_a
        gc.collect()

    print()

    # ── Phase 3: Kill criteria ───────────────────────────────────────
    print("=== Phase 3: Kill Criteria ===")
    t_total = time.time() - t_start

    # Per-domain improvement (pp)
    domain_improvements = {}
    for domain in available_domains:
        if domain not in adapted_results:
            continue
        base_score = base_results[domain]["mean_score"]
        adap_score = adapted_results[domain]["mean_score"]
        domain_improvements[domain] = (adap_score - base_score) * 100.0

    print("\nPer-domain improvement (pp):")
    for d, imp in domain_improvements.items():
        status = "PASS" if imp >= 10.0 else "FAIL"
        print(f"  {d}: {imp:+.1f}pp [{status}]")

    # K1224: ≥3 of 5 original domains show ≥10pp improvement
    n_improved = sum(1 for imp in domain_improvements.values() if imp >= 10.0)
    k1224_pass = n_improved >= 3
    k1224_val = n_improved

    # K1225: cross-domain retention ≥ 90% (mean regression ratio ≥ 0.90)
    if regression_ratios:
        mean_retention = float(sum(regression_ratios.values()) / len(regression_ratios))
        all_pass_retention = all(r >= 0.90 for r in regression_ratios.values())
    else:
        mean_retention = 1.0
        all_pass_retention = True
    k1225_pass = mean_retention >= 0.90
    k1225_val = mean_retention

    # K1226: average adapted accuracy ≥ 50%
    all_adapted_means = [r["mean_score"] for r in adapted_results.values()]
    avg_adapted_acc = float(sum(all_adapted_means) / len(all_adapted_means)) if all_adapted_means else 0.0
    k1226_pass = avg_adapted_acc >= 0.50
    k1226_val = avg_adapted_acc

    all_pass = k1224_pass and k1225_pass and k1226_pass

    print(f"\nK1224 (≥3/5 domains ≥10pp improvement): {n_improved}/5 → {'PASS' if k1224_pass else 'FAIL'}")
    print(f"K1225 (cross-domain retention ≥90%): {mean_retention:.3f} → {'PASS' if k1225_pass else 'FAIL'}")
    print(f"K1226 (avg adapted accuracy ≥50%): {avg_adapted_acc:.3f} → {'PASS' if k1226_pass else 'FAIL'}")
    print(f"\nALL_PASS: {all_pass}")
    print(f"Total time: {t_total/60:.2f} min")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "n_cross": N_CROSS,
        "max_tokens": MAX_TOKENS,
        "available_domains": available_domains_all,
        "base_results": {
            d: {
                "mean_score": r["mean_score"],
                "n": r["n"],
                "scores": r["scores"],
            }
            for d, r in base_results.items()
        },
        "adapted_results": {
            d: {
                "mean_score": r["mean_score"],
                "n": r["n"],
                "scores": r["scores"],
            }
            for d, r in adapted_results.items()
        },
        "domain_improvements_pp": domain_improvements,
        "cross_domain_retention": regression_ratios,
        "mean_retention": mean_retention,
        "avg_adapted_accuracy": avg_adapted_acc,
        "n_domains_improved_10pp": n_improved,
        "k1224": {"pass": k1224_pass, "value": k1224_val, "threshold": 3},
        "k1225": {"pass": k1225_pass, "value": k1225_val, "threshold": 0.90},
        "k1226": {"pass": k1226_pass, "value": k1226_val, "threshold": 0.50},
        "all_pass": all_pass,
        "total_time_min": t_total / 60.0,
        "summary": {
            "n_improved_10pp": n_improved,
            "best_domain": max(domain_improvements, key=domain_improvements.get) if domain_improvements else "none",
            "best_improvement_pp": max(domain_improvements.values()) if domain_improvements else 0.0,
            "mean_retention": mean_retention,
            "avg_adapted_acc": avg_adapted_acc,
        }
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
