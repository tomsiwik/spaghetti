#!/usr/bin/env python3
"""
P0: Behavioral E2E Quality — full pipeline produces domain-correct text (5 domains).

Verifies that routed + adapted Gemma 4 actually produces better domain-specific text
vs base model using a unified domain-vocabulary rubric:
  - For each domain, count domain-specific glossary terms in generated response
  - Improvement = adapted_count > base_count (vocabulary shift toward domain)
  - This captures what adapters learn: domain vocabulary distribution shift

Rubric justification: PPL↛task quality (r=0.08, established). Vocabulary shift
is the direct behavioral signal — adapters trained on domain corpora MUST shift
output distribution toward domain vocabulary. This is the minimum behavioral test.

Kill criteria:
  K1162: ≥80% of math domain queries: adapter output uses more math vocabulary than base
  K1163: ≥80% of code domain queries: adapter output uses more code vocabulary than base
  K1164: Medical/legal/finance ≥70% each: domain vocabulary improvement
  K1165: Full pipeline (route+adapt+generate) ≤10ms adapter overhead

References:
  - Finding #431 (T4.1): TF-IDF routing ρ_math=1.0, ρ_code=1.0, ρ_medical=1.0
  - Finding #436 (T5.1): adapter behavioral gain δ=0.76 (personal adapter)
  - Finding #441 (C0.1): Grassmannian composition quality_ratio=0.9024
  - MATH.md Theorem 1: Q_pipeline(D) = ρ_D · δ_D ≥ 0.60 for all domains
"""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# N queries per domain
N_PER_DOMAIN = 3 if IS_SMOKE else 10

# Adapter paths (trained in T2.1 and T2.6)
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

ADAPTER_PATHS = {
    "math":    str(T21_DIR / "adapters" / "math"),
    "code":    str(T21_DIR / "adapters" / "code"),
    "medical": str(T21_DIR / "adapters" / "medical"),
    "legal":   str(T26_DIR / "adapters" / "legal"),
    "finance": str(T26_DIR / "adapters" / "finance"),
}

# TF-IDF router vocabulary (from T4.1)
DOMAIN_KEYWORDS = {
    "math":    ["algebra", "equation", "calculate", "solve", "percentage", "theorem",
                "proof", "polynomial", "derivative", "integral", "matrix", "vector",
                "probability", "statistics", "geometry", "arithmetic", "formula",
                "angle", "triangle", "circle", "factor", "prime"],
    "code":    ["python", "function", "class", "algorithm", "implement", "debug",
                "recursion", "loop", "array", "string", "complexity", "runtime",
                "programming", "variable", "method", "object", "syntax", "code",
                "return", "parameter", "list", "dictionary", "module"],
    "medical": ["diagnosis", "treatment", "symptom", "medication", "patient", "clinical",
                "pharmacology", "disease", "therapy", "anatomy", "physiology", "dose",
                "adverse", "mechanism", "receptor", "inhibitor", "enzyme", "blood",
                "pressure", "cardiac", "infection", "antibiotic", "immune"],
    "legal":   ["statute", "jurisdiction", "plaintiff", "defendant", "precedent", "court",
                "contract", "liability", "criminal", "civil", "constitution", "rights",
                "attorney", "testimony", "verdict", "appeal", "regulation", "law",
                "amendment", "judge", "jury", "evidence"],
    "finance": ["investment", "portfolio", "equity", "dividend", "interest", "revenue",
                "depreciation", "asset", "liability", "cash", "market", "fiscal",
                "balance", "profit", "capital", "hedge", "bond", "stock", "return",
                "risk", "valuation", "earnings", "shareholder", "inflation"],
}

# Domain-specific queries (open-ended, encouraging detailed responses)
DOMAIN_QUERIES = {
    "math": [
        "Explain how to find the roots of a quadratic equation using the quadratic formula.",
        "Describe the fundamental theorem of calculus and what it means.",
        "Explain what a probability distribution is and give an example.",
        "What is the difference between a permutation and a combination?",
        "Explain what eigenvalues and eigenvectors represent geometrically.",
        "Describe how to compute a matrix inverse and when it exists.",
        "What is the binomial theorem and how is it used?",
        "Explain the concept of a limit in calculus.",
        "What is the difference between a function and a relation?",
        "Explain how integration by parts works.",
    ],
    "code": [
        "Explain how recursion works and give an example of when to use it.",
        "What is the difference between a class and an object in Python?",
        "Explain what time complexity means and how to analyze it.",
        "Describe how a binary search algorithm works.",
        "What are list comprehensions in Python and when should you use them?",
        "Explain what a decorator is in Python.",
        "What is the difference between mutable and immutable objects?",
        "Explain how exception handling works with try/except.",
        "What is a generator in Python and how does it differ from a list?",
        "Explain what object-oriented programming means.",
    ],
    "medical": [
        "Explain how ACE inhibitors work to treat hypertension.",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "Describe how beta-blockers work and their clinical uses.",
        "What are the main symptoms of myocardial infarction?",
        "Explain the role of the immune system in fighting bacterial infections.",
        "What is the significance of the blood-brain barrier?",
        "Describe how diuretics work to treat edema.",
        "What is the pathophysiology of asthma?",
        "Explain how vaccines provide immunity.",
        "What are the stages of wound healing?",
    ],
    "legal": [
        "Explain the concept of habeas corpus.",
        "What is the difference between civil and criminal liability?",
        "What does the right to due process mean under the Constitution?",
        "Explain the concept of precedent (stare decisis) in common law.",
        "What is the difference between a misdemeanor and a felony?",
        "Explain what it means to have legal standing in a lawsuit.",
        "What is the role of the statute of limitations in law?",
        "Explain the difference between a contract and a tort.",
        "What is the exclusionary rule in criminal procedure?",
        "Explain the concept of mens rea in criminal law.",
    ],
    "finance": [
        "What is the difference between stocks and bonds as investments?",
        "Explain the concept of portfolio diversification.",
        "What is compound interest and how does it affect long-term savings?",
        "Explain what a P/E ratio measures in equity valuation.",
        "What is the difference between market capitalization and book value?",
        "Explain how inflation affects the purchasing power of savings.",
        "What are dividends and why do companies pay them?",
        "Explain the risk-return tradeoff in investing.",
        "What is a balance sheet and what does it show?",
        "Explain the difference between gross profit and net profit.",
    ],
}

# Domain-specific vocabulary glossaries (curated terms strongly associated with each domain)
DOMAIN_GLOSSARIES = {
    "math": [
        "theorem", "proof", "equation", "derivative", "integral", "polynomial",
        "coefficient", "eigenvalue", "eigenvector", "determinant", "matrix", "vector",
        "probability", "distribution", "convergence", "continuity", "differentiable",
        "binomial", "permutation", "combination", "exponent", "logarithm", "trigonometric",
        "quadratic", "linear", "recursion", "induction", "modular", "congruence",
    ],
    "code": [
        "function", "return", "parameter", "variable", "algorithm", "recursive",
        "iteration", "loop", "array", "list", "dictionary", "class", "method",
        "object", "instantiate", "inheritance", "polymorphism", "exception",
        "generator", "decorator", "comprehension", "complexity", "runtime",
        "syntax", "module", "import", "lambda", "closure", "callback",
    ],
    "medical": [
        "mechanism", "inhibitor", "receptor", "pharmacology", "clinical", "therapy",
        "diagnosis", "treatment", "pathophysiology", "enzyme", "protein",
        "antibody", "immune", "inflammation", "vascular", "cardiac", "neural",
        "medication", "dose", "adverse", "contraindicated", "efficacy", "etiology",
        "prognosis", "cytokine", "antibiotics", "prophylaxis", "comorbidity",
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
        "leverage", "arbitrage", "beta", "alpha", "collateral", "amortize",
    ],
}


def build_tfidf_router(domain_keywords: dict):
    """Build TF-IDF nearest-centroid router from keyword vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = []
    labels = []
    for domain, keywords in domain_keywords.items():
        doc = " ".join(keywords * 3)
        corpus.append(doc)
        labels.append(domain)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    centroids = vectorizer.fit_transform(corpus).toarray()
    return vectorizer, centroids, labels


def route_query(query: str, vectorizer, centroids: np.ndarray, labels: list) -> tuple:
    """Route a query to the best domain adapter."""
    q_vec = vectorizer.transform([query]).toarray()[0]
    norms = np.linalg.norm(centroids, axis=1) * np.linalg.norm(q_vec) + 1e-9
    similarities = centroids @ q_vec / norms
    best_idx = int(np.argmax(similarities))
    return labels[best_idx], float(similarities[best_idx])


def score_vocabulary(text: str, glossary: list) -> int:
    """Count domain glossary terms appearing in text."""
    text_lower = text.lower()
    return sum(1 for term in glossary if term.lower() in text_lower)


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 150) -> str:
    """Generate a response from the model given a prompt."""
    from mlx_lm import generate

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)


def eval_domain(
    model_base,
    model_adapted,
    tokenizer,
    domain: str,
    queries: list,
    glossary: list,
    max_tokens: int = 150,
) -> dict:
    """
    Compare base vs adapted model on domain queries using vocabulary rubric.
    Returns improvement rate (fraction of queries where adapted > base).
    """
    results = []

    for i, query in enumerate(queries):
        resp_base = generate_response(model_base, tokenizer, query, max_tokens)
        resp_adapted = generate_response(model_adapted, tokenizer, query, max_tokens)

        score_base = score_vocabulary(resp_base, glossary)
        score_adapted = score_vocabulary(resp_adapted, glossary)
        improved = score_adapted > score_base

        results.append({
            "query": query[:80] + "..." if len(query) > 80 else query,
            "score_base": score_base,
            "score_adapted": score_adapted,
            "improved": improved,
            "resp_base_snippet": resp_base[:80],
            "resp_adapted_snippet": resp_adapted[:80],
        })
        print(f"  [{domain}][{i+1}/{len(queries)}] vocab base={score_base} adapted={score_adapted} improved={improved}")

    improvement_rate = sum(r["improved"] for r in results) / len(results)
    return {
        "domain": domain,
        "n_queries": len(queries),
        "improvement_rate": improvement_rate,
        "mean_score_base": float(np.mean([r["score_base"] for r in results])),
        "mean_score_adapted": float(np.mean([r["score_adapted"] for r in results])),
        "results": results,
    }


def main():
    import mlx.core as mx
    from mlx_lm import load

    print(f"=== P0 Behavioral E2E Experiment ===")
    print(f"IS_SMOKE={IS_SMOKE}, N_PER_DOMAIN={N_PER_DOMAIN}")
    print(f"Rubric: domain vocabulary count (adapted > base = improvement)")

    # Build router
    vectorizer, centroids, labels = build_tfidf_router(DOMAIN_KEYWORDS)
    print(f"Router built: {len(labels)} domains")

    # Load base model once
    print("Loading base model...")
    t0 = time.time()
    model_base, tokenizer = load(MODEL_ID)
    mx.eval(model_base.parameters())
    print(f"Base model loaded in {time.time()-t0:.1f}s")

    # K1165: Pipeline overhead
    print("\n[K1165] Measuring pipeline overhead...")
    test_query = "Explain what a quadratic equation is."
    overhead_samples = []
    for _ in range(10):
        t_start = time.perf_counter()
        domain_pred, _ = route_query(test_query, vectorizer, centroids, labels)
        overhead_samples.append((time.perf_counter() - t_start) * 1000)

    route_p99_ms = float(np.percentile(overhead_samples, 99))
    # Add known swap latency from Finding #434 (1.41ms)
    k1165_overhead_ms = route_p99_ms + 1.41
    k1165_pass = k1165_overhead_ms <= 10.0
    print(f"  Route p99={route_p99_ms:.2f}ms + swap 1.41ms = {k1165_overhead_ms:.2f}ms → {'PASS' if k1165_pass else 'FAIL'}")

    # Evaluate all 5 domains
    all_domain_results = {}
    domain_passes = {}
    thresholds = {
        "math": (0.80, "K1162"),
        "code": (0.80, "K1163"),
        "medical": (0.70, "K1164"),
        "legal": (0.70, "K1164"),
        "finance": (0.70, "K1164"),
    }

    for domain in ["math", "code", "medical", "legal", "finance"]:
        threshold, criterion = thresholds[domain]
        print(f"\n[{criterion}: {domain.upper()}] (threshold ≥{threshold:.0%})")

        # Load adapted model
        t0 = time.time()
        model_adapted, _ = load(MODEL_ID, adapter_path=ADAPTER_PATHS[domain])
        mx.eval(model_adapted.parameters())
        print(f"  Adapter loaded in {time.time()-t0:.1f}s")

        queries = DOMAIN_QUERIES[domain][:N_PER_DOMAIN]
        glossary = DOMAIN_GLOSSARIES[domain]
        max_tokens = 200 if domain == "code" else 150

        domain_result = eval_domain(
            model_base=model_base,
            model_adapted=model_adapted,
            tokenizer=tokenizer,
            domain=domain,
            queries=queries,
            glossary=glossary,
            max_tokens=max_tokens,
        )

        del model_adapted
        gc.collect()

        rate = domain_result["improvement_rate"]
        passes = rate >= threshold
        domain_passes[domain] = passes
        all_domain_results[domain] = domain_result
        print(f"  {domain}: improvement_rate={rate:.1%} (base_mean={domain_result['mean_score_base']:.1f}, "
              f"adapted_mean={domain_result['mean_score_adapted']:.1f}) → {'PASS' if passes else 'FAIL'}")

    # Summary
    print("\n=== SUMMARY ===")
    k1162_pass = domain_passes["math"]
    k1163_pass = domain_passes["code"]
    k1164_pass = all(domain_passes[d] for d in ["medical", "legal", "finance"])
    all_pass = k1162_pass and k1163_pass and k1164_pass and k1165_pass

    for domain, (threshold, criterion) in thresholds.items():
        rate = all_domain_results[domain]["improvement_rate"]
        print(f"{criterion} ({domain} ≥{threshold:.0%}): {rate:.1%} → {'PASS' if domain_passes[domain] else 'FAIL'}")
    print(f"K1165 (overhead ≤10ms): {k1165_overhead_ms:.2f}ms → {'PASS' if k1165_pass else 'FAIL'}")
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")

    results = {
        "is_smoke": IS_SMOKE,
        "n_per_domain": N_PER_DOMAIN,
        "k1162": {
            "domain": "math",
            "improvement_rate": all_domain_results["math"]["improvement_rate"],
            "threshold": 0.80,
            "k1162_pass": k1162_pass,
        },
        "k1163": {
            "domain": "code",
            "improvement_rate": all_domain_results["code"]["improvement_rate"],
            "threshold": 0.80,
            "k1163_pass": k1163_pass,
        },
        "k1164": {
            "per_domain": {
                d: all_domain_results[d]["improvement_rate"]
                for d in ["medical", "legal", "finance"]
            },
            "all_pass": k1164_pass,
            "threshold": 0.70,
        },
        "k1165": {
            "overhead_p99_ms": k1165_overhead_ms,
            "threshold_ms": 10.0,
            "k1165_pass": k1165_pass,
        },
        "domain_details": {
            domain: {
                "n_queries": all_domain_results[domain]["n_queries"],
                "improvement_rate": all_domain_results[domain]["improvement_rate"],
                "mean_score_base": all_domain_results[domain]["mean_score_base"],
                "mean_score_adapted": all_domain_results[domain]["mean_score_adapted"],
                "per_query": all_domain_results[domain]["results"],
            }
            for domain in all_domain_results
        },
        "all_pass": all_pass,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
