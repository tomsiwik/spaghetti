#!/usr/bin/env python3
"""
P4.B1: Gap-Targeted Evaluation — Hard Domain Questions.

Tests the Adapter Signal Gap Hypothesis (MATH.md Theorem 1):
  δ_d > 0 iff H(V_d | θ_base) > H_threshold
Proxy: base_score < 0.30 → high entropy → adapter can act.

Questions use specialized subdomain vocabulary (6 keywords each) that
Gemma 4 base model does NOT naturally produce. Tests whether existing
rank-6 adapters (from P1 T2) can provide signal when the gap exists.

Phases:
  1: Base model evaluation on hard questions (all 5 domains)
  2: Per-adapter evaluation (own domain improvement)
  3: Correlation analysis + kill criteria

Kill criteria:
  K1227: Hard-set mean base score < 25% (confirms gap regime)
  K1228: >=3/5 domains show >=15pp improvement with rank-6 adapters
  K1229: Pearson r(base_score, improvement) < -0.30 (confirms gap hypothesis)

Grounded by:
  - Finding #477: P4.B0 — adapter quality gap-dependent (math +20pp vs medical -4pp)
  - MATH.md Theorem 1: delta_d > 0 iff H(V_d|theta) > H_threshold
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

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

T2_SINGLE_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training" / "adapters"
T2_MULTI_DIR  = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"      / "adapters"

ADAPTER_PATHS = {
    "medical": T2_SINGLE_DIR / "medical",
    "code":    T2_SINGLE_DIR / "code",
    "math":    T2_SINGLE_DIR / "math",
    "legal":   T2_MULTI_DIR  / "legal",
    "finance": T2_MULTI_DIR  / "finance",
}

IS_SMOKE  = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL    = 5 if IS_SMOKE else 15   # questions per domain
MAX_TOKENS = 200

# ──────────────────────────────────────────────────────────────────────
# Hard question sets — 6 specialized keywords each.
# Keys: MUST appear verbatim (case-insensitive) in a correct expert answer.
# These terms are rare enough that base model should score < 25%.
# ──────────────────────────────────────────────────────────────────────

HARD_QA = {
    "math": [
        {
            "q": "State Zorn's lemma and explain its relation to the axiom of choice.",
            "keys": ["Zorn", "maximal element", "partially ordered", "chain", "upper bound", "axiom of choice"]
        },
        {
            "q": "What is the Stone-Weierstrass theorem and why is it important?",
            "keys": ["Stone-Weierstrass", "algebra", "separating", "compact", "uniform approximation", "subalgebra"]
        },
        {
            "q": "Describe the spectral theorem for compact self-adjoint operators.",
            "keys": ["compact operator", "self-adjoint", "eigenvalue", "Hilbert space", "orthonormal basis", "spectral"]
        },
        {
            "q": "What is the Riesz representation theorem in functional analysis?",
            "keys": ["Riesz", "bounded linear functional", "measure", "Hilbert space", "inner product", "representation"]
        },
        {
            "q": "Explain the Hahn-Banach theorem and its main consequence.",
            "keys": ["Hahn-Banach", "normed space", "functional", "linear", "extension", "bounded"]
        },
        {
            "q": "What is the Pontryagin maximum principle in optimal control?",
            "keys": ["Pontryagin", "optimal control", "Hamiltonian", "costate", "transversality", "adjoint"]
        },
        {
            "q": "Describe the Perron-Frobenius theorem for non-negative matrices.",
            "keys": ["Perron-Frobenius", "non-negative", "spectral radius", "dominant eigenvalue", "irreducible", "eigenvector"]
        },
        {
            "q": "What is the Arzelà-Ascoli theorem and when does it apply?",
            "keys": ["Arzelà-Ascoli", "equicontinuous", "uniformly bounded", "compact", "subsequence", "convergence"]
        },
        {
            "q": "What is a sheaf in algebraic geometry and how does the gluing condition work?",
            "keys": ["sheaf", "presheaf", "gluing", "restriction", "stalk", "open cover"]
        },
        {
            "q": "Explain the Jordan-Chevalley decomposition in linear algebra.",
            "keys": ["Jordan", "nilpotent", "semisimple", "decomposition", "commute", "characteristic"]
        },
        {
            "q": "State the Riemann-Lebesgue lemma and explain its significance.",
            "keys": ["Riemann-Lebesgue", "Fourier transform", "Lebesgue integrable", "vanish at infinity", "oscillation", "L^1"]
        },
        {
            "q": "What is the Radon-Nikodym theorem and what is absolute continuity?",
            "keys": ["Radon-Nikodym", "absolutely continuous", "derivative", "measure", "sigma-finite", "Lebesgue"]
        },
        {
            "q": "Describe the Krein-Milman theorem and extreme points.",
            "keys": ["Krein-Milman", "extreme point", "convex hull", "compact convex", "locally convex", "extreme"]
        },
        {
            "q": "What is the Open Mapping theorem in functional analysis?",
            "keys": ["Open Mapping", "surjective", "Banach space", "continuous linear map", "open set", "Banach"]
        },
        {
            "q": "Explain the concept of a Galois extension and the fundamental theorem of Galois theory.",
            "keys": ["Galois", "splitting field", "normal", "separable", "automorphism", "correspondence"]
        },
    ],
    "medical": [
        {
            "q": "What are the diagnostic criteria for antiphospholipid syndrome (APS)?",
            "keys": ["antiphospholipid", "anticardiolipin", "lupus anticoagulant", "thrombosis", "beta-2 glycoprotein", "Sapporo"]
        },
        {
            "q": "Explain the V(D)J recombination process in B cell development.",
            "keys": ["V(D)J", "RAG", "recombination signal", "joining", "diversity", "immunoglobulin"]
        },
        {
            "q": "Describe the pathogenesis of hemolytic-uremic syndrome (HUS).",
            "keys": ["Shiga toxin", "thrombotic microangiopathy", "endothelial", "microangiopathic", "fibrin", "HUS"]
        },
        {
            "q": "What is the mechanism of action of vancomycin and why is it reserved for MRSA?",
            "keys": ["vancomycin", "D-Ala-D-Ala", "peptidoglycan", "transpeptidase", "MRSA", "cell wall"]
        },
        {
            "q": "Explain the complement cascade and how C3 convertase is formed.",
            "keys": ["complement", "C3 convertase", "classical pathway", "alternative pathway", "MAC", "opsonization"]
        },
        {
            "q": "Describe somatic hypermutation and affinity maturation in the germinal center.",
            "keys": ["somatic hypermutation", "affinity maturation", "germinal center", "AID", "activation-induced", "B cell"]
        },
        {
            "q": "What is the pathogenesis of systemic lupus erythematosus (SLE)?",
            "keys": ["anti-dsDNA", "antinuclear antibody", "complement", "immune complex", "type I interferon", "autoimmune"]
        },
        {
            "q": "Explain the mechanism of action of methotrexate in oncology and rheumatology.",
            "keys": ["methotrexate", "dihydrofolate reductase", "thymidylate", "purine", "folate", "antifolate"]
        },
        {
            "q": "Describe the mechanism of beta-lactam resistance via beta-lactamases.",
            "keys": ["beta-lactamase", "beta-lactam ring", "hydrolysis", "penicillin-binding protein", "MRSA", "resistance"]
        },
        {
            "q": "What are the mechanisms of calcineurin inhibitors in preventing organ rejection?",
            "keys": ["calcineurin inhibitor", "tacrolimus", "cyclosporine", "T cell", "IL-2", "rejection"]
        },
        {
            "q": "Explain long-term potentiation (LTP) and its molecular mechanism.",
            "keys": ["LTP", "NMDA receptor", "AMPA", "synaptic plasticity", "calcium", "CaMKII"]
        },
        {
            "q": "What are cytochrome P450 isoforms and why do they matter for drug interactions?",
            "keys": ["cytochrome P450", "CYP3A4", "isoform", "drug metabolism", "induction", "inhibition"]
        },
        {
            "q": "Describe disseminated intravascular coagulation (DIC) and its treatment.",
            "keys": ["DIC", "fibrinogen", "coagulation cascade", "fibrinolysis", "thrombocytopenia", "consumptive"]
        },
        {
            "q": "What is the mechanism of action of rituximab?",
            "keys": ["rituximab", "CD20", "ADCC", "complement-dependent cytotoxicity", "B cell depletion", "chimeric"]
        },
        {
            "q": "Explain mast cell activation and the role of IgE in Type I hypersensitivity.",
            "keys": ["mast cell", "IgE", "FcεRI", "degranulation", "histamine", "anaphylaxis"]
        },
    ],
    "legal": [
        {
            "q": "What is the Chevron doctrine and what are its two steps?",
            "keys": ["Chevron", "deference", "statutory ambiguity", "reasonable interpretation", "agency", "step one"]
        },
        {
            "q": "Explain the collateral order doctrine and when it permits immediate appeal.",
            "keys": ["collateral order", "Cohen", "immediately appealable", "conclusively determined", "important", "appealable"]
        },
        {
            "q": "What is qualified immunity in Section 1983 civil rights litigation?",
            "keys": ["qualified immunity", "clearly established", "constitutional right", "1983", "objective", "Harlow"]
        },
        {
            "q": "Describe the elements of a qui tam action under the False Claims Act.",
            "keys": ["qui tam", "relator", "False Claims Act", "scienter", "materiality", "treble damages"]
        },
        {
            "q": "What is the dormant Commerce Clause doctrine and how does Pike balancing apply?",
            "keys": ["dormant Commerce Clause", "discriminatory", "Pike balancing", "market participant", "interstate commerce", "incidental"]
        },
        {
            "q": "Explain the Younger abstention doctrine in federal court proceedings.",
            "keys": ["Younger abstention", "federal court", "pending state proceeding", "comity", "equitable relief", "civil"]
        },
        {
            "q": "What are the requirements for class certification under Rule 23(b)(3)?",
            "keys": ["Rule 23", "numerosity", "commonality", "typicality", "predominance", "superiority"]
        },
        {
            "q": "Describe the Daubert standard for admissibility of expert testimony.",
            "keys": ["Daubert", "scientific validity", "peer reviewed", "error rate", "testable", "gatekeeping"]
        },
        {
            "q": "What is the Noerr-Pennington doctrine and what is the sham exception?",
            "keys": ["Noerr-Pennington", "petitioning immunity", "antitrust", "First Amendment", "sham exception", "Pennington"]
        },
        {
            "q": "Explain the Penn Central test for regulatory takings under the Fifth Amendment.",
            "keys": ["Penn Central", "regulatory taking", "economic impact", "investment-backed expectations", "character", "just compensation"]
        },
        {
            "q": "What is the business judgment rule and how does it protect corporate directors?",
            "keys": ["business judgment rule", "duty of care", "good faith", "informed", "fiduciary", "Delaware"]
        },
        {
            "q": "Describe the Erie doctrine and when federal common law applies.",
            "keys": ["Erie", "federal common law", "substantive", "Swift v. Tyson", "diversity jurisdiction", "forum"]
        },
        {
            "q": "What is promissory estoppel and what are its elements under the Restatement?",
            "keys": ["promissory estoppel", "detrimental reliance", "definite promise", "foreseeable", "injustice", "Restatement"]
        },
        {
            "q": "Explain the Miranda doctrine, its required warnings, and the Edwards rule.",
            "keys": ["Miranda", "custodial interrogation", "waiver", "invocation", "Edwards rule", "Fifth Amendment"]
        },
        {
            "q": "What is res ipsa loquitur and what are the conditions for its application?",
            "keys": ["res ipsa loquitur", "inference of negligence", "exclusive control", "ordinarily", "circumstantial", "defendant"]
        },
    ],
    "code": [
        {
            "q": "Explain the Byzantine fault tolerance problem and how PBFT solves it.",
            "keys": ["Byzantine fault", "PBFT", "faulty nodes", "view change", "quorum", "consensus"]
        },
        {
            "q": "What is the ABA problem in compare-and-swap operations and how is it prevented?",
            "keys": ["ABA problem", "compare-and-swap", "tagged pointer", "lock-free", "hazard pointer", "atomic"]
        },
        {
            "q": "Describe the write-ahead log (WAL) and how it ensures durability in databases.",
            "keys": ["write-ahead log", "WAL", "durability", "checkpointing", "crash recovery", "LSN"]
        },
        {
            "q": "Explain the Raft consensus algorithm and how leader election works.",
            "keys": ["Raft", "leader election", "term", "log replication", "commit", "majority quorum"]
        },
        {
            "q": "What is MVCC (multi-version concurrency control) and how does it handle reads?",
            "keys": ["MVCC", "snapshot isolation", "version chain", "transaction id", "read timestamp", "write-write conflict"]
        },
        {
            "q": "Describe the MESI cache coherence protocol in multi-core processors.",
            "keys": ["MESI", "cache coherence", "Modified", "Exclusive", "Shared", "Invalid"]
        },
        {
            "q": "Explain software transactional memory (STM) and how it handles conflicts.",
            "keys": ["software transactional memory", "STM", "atomic", "abort", "retry", "composable"]
        },
        {
            "q": "What is the skip list data structure and what is its expected search complexity?",
            "keys": ["skip list", "probabilistic", "O(log n)", "layers", "forward pointer", "expected"]
        },
        {
            "q": "Describe the Chord distributed hash table and how finger tables enable routing.",
            "keys": ["Chord", "consistent hashing", "finger table", "successor", "ring topology", "DHT"]
        },
        {
            "q": "What is the Lamport logical clock and how does it capture causality?",
            "keys": ["Lamport clock", "happened-before", "partial order", "causality", "timestamp", "distributed"]
        },
        {
            "q": "Explain the copy-on-write (COW) optimization in operating systems and fork().",
            "keys": ["copy-on-write", "page fault", "fork", "shared pages", "dirty", "virtual memory"]
        },
        {
            "q": "What is the work-span (WC) model and what does Brent's theorem state?",
            "keys": ["work", "span", "Brent", "parallelism", "critical path", "scheduling"]
        },
        {
            "q": "Describe the B-epsilon tree data structure and its advantages for I/O.",
            "keys": ["B-epsilon", "fractal tree", "buffered messages", "batch", "I/O complexity", "write amplification"]
        },
        {
            "q": "What is the difference between optimistic and pessimistic concurrency control?",
            "keys": ["optimistic concurrency", "pessimistic", "abort", "timestamp ordering", "conflict detection", "lock"]
        },
        {
            "q": "Explain the COW B-tree and how it supports snapshots and multi-versioning.",
            "keys": ["copy-on-write B-tree", "immutable", "version", "structural sharing", "snapshot", "path copying"]
        },
    ],
    "finance": [
        {
            "q": "Explain the Black-Scholes model and what Itô's lemma contributes to its derivation.",
            "keys": ["Black-Scholes", "risk-neutral", "Ito", "geometric Brownian motion", "no-arbitrage", "delta hedging"]
        },
        {
            "q": "What is the Vasicek interest rate model and what is its mean-reversion property?",
            "keys": ["Vasicek", "mean-reversion", "Ornstein-Uhlenbeck", "risk-neutral measure", "short rate", "theta"]
        },
        {
            "q": "Describe the Heston stochastic volatility model and its parameters.",
            "keys": ["Heston", "stochastic volatility", "mean-reverting", "correlation", "variance process", "characteristic function"]
        },
        {
            "q": "What is the Kelly criterion for optimal bet sizing and how is it derived?",
            "keys": ["Kelly criterion", "expected logarithm", "bankroll", "optimal fraction", "edge", "geometric growth"]
        },
        {
            "q": "Explain the fundamental theorem of asset pricing and equivalent martingale measures.",
            "keys": ["risk-neutral", "equivalent martingale measure", "no-arbitrage", "fundamental theorem", "replication", "complete market"]
        },
        {
            "q": "What is Credit Valuation Adjustment (CVA) in over-the-counter derivatives?",
            "keys": ["CVA", "counterparty credit risk", "default probability", "loss given default", "exposure", "bilateral"]
        },
        {
            "q": "Describe the Fama-French three-factor model and the SMB and HML factors.",
            "keys": ["Fama-French", "size premium", "value premium", "SMB", "HML", "market beta"]
        },
        {
            "q": "What is duration and convexity, and how do they estimate bond price changes?",
            "keys": ["convexity", "modified duration", "yield", "price-yield", "second derivative", "coupon"]
        },
        {
            "q": "Explain Markowitz mean-variance optimization and the efficient frontier.",
            "keys": ["Markowitz", "efficient frontier", "covariance matrix", "portfolio variance", "minimum variance", "risk-return"]
        },
        {
            "q": "What is Value at Risk (VaR) and what are its limitations vs. Expected Shortfall?",
            "keys": ["Value at Risk", "confidence interval", "tail risk", "quantile", "subadditive", "expected shortfall"]
        },
        {
            "q": "Describe Merton's jump-diffusion model for option pricing.",
            "keys": ["Merton", "jump-diffusion", "Poisson process", "jump intensity", "log-normal", "option pricing"]
        },
        {
            "q": "What is the term structure of interest rates and the expectations hypothesis?",
            "keys": ["term structure", "yield curve", "expectations hypothesis", "forward rate", "liquidity premium", "spot rate"]
        },
        {
            "q": "Explain immunization in fixed-income portfolio management.",
            "keys": ["immunization", "duration matching", "reinvestment risk", "price risk", "yield curve", "liability"]
        },
        {
            "q": "What is the Treynor ratio and how does it differ from the Sharpe ratio?",
            "keys": ["Treynor ratio", "beta", "systematic risk", "excess return", "Sharpe", "capital market line"]
        },
        {
            "q": "Describe the concept of risk parity and equal risk contribution portfolios.",
            "keys": ["risk parity", "equal risk contribution", "volatility", "correlation", "leverage", "marginal contribution"]
        },
    ],
}

DOMAINS = list(HARD_QA.keys())


# ──────────────────────────────────────────────────────────────────────
# Keyword scoring
# ──────────────────────────────────────────────────────────────────────

def score_response(response: str, keys: list[str]) -> float:
    """Fraction of keywords found in response (case-insensitive)."""
    text = response.lower()
    return sum(1 for k in keys if k.lower() in text) / len(keys)


# ──────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────

def load_model(adapter_path=None):
    from mlx_lm import load
    if adapter_path and Path(adapter_path).exists():
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)
    return model, tokenizer


def unload_model(model):
    del model
    gc.collect()
    mx.metal.clear_cache()


def generate_response(model, tokenizer, question: str, max_tokens: int = MAX_TOKENS) -> str:
    from mlx_lm import generate
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return out.strip()


# ──────────────────────────────────────────────────────────────────────
# Evaluation phases
# ──────────────────────────────────────────────────────────────────────

def evaluate_domain(model, tokenizer, domain: str, n_eval: int) -> dict:
    qa_items = HARD_QA[domain][:n_eval]
    scores = []
    for item in qa_items:
        resp = generate_response(model, tokenizer, item["q"])
        s = score_response(resp, item["keys"])
        scores.append(s)
    return {"mean_score": sum(scores) / len(scores), "n": len(scores), "scores": scores}


def check_adapter_exists(domain: str) -> bool:
    p = ADAPTER_PATHS[domain]
    return p.exists() and (p / "adapter_config.json").exists()


def compute_correlation(base_scores: list[float], improvements: list[float]) -> float:
    """Pearson r between base scores and improvements."""
    import math
    n = len(base_scores)
    if n < 3:
        return float("nan")
    mean_b = sum(base_scores) / n
    mean_i = sum(improvements) / n
    num = sum((b - mean_b) * (i - mean_i) for b, i in zip(base_scores, improvements))
    den_b = math.sqrt(sum((b - mean_b) ** 2 for b in base_scores))
    den_i = math.sqrt(sum((i - mean_i) ** 2 for i in improvements))
    if den_b < 1e-9 or den_i < 1e-9:
        return float("nan")
    return num / (den_b * den_i)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f"P4.B1: Gap-Targeted Evaluation — Hard Domain Questions")
    print(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}")
    print(f"Model: {MODEL_ID}")
    print()

    results: dict = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "base_results": {},
        "adapted_results": {},
        "improvements": {},
        "kill_criteria": {},
    }

    # ─── Phase 1: Base model evaluation ──────────────────────────────
    print("=" * 60)
    print("Phase 1: Base model evaluation on hard questions")
    print("=" * 60)

    print("Loading base model...")
    base_model, tokenizer = load_model()

    for domain in DOMAINS:
        print(f"\n  Evaluating base on {domain} ({N_EVAL} questions)...")
        t0 = time.time()
        res = evaluate_domain(base_model, tokenizer, domain, N_EVAL)
        elapsed = time.time() - t0
        results["base_results"][domain] = res
        print(f"    base_mean={res['mean_score']:.3f}  ({elapsed:.1f}s)")
        print(f"    scores: {[round(s, 2) for s in res['scores']]}")

    unload_model(base_model)

    # K1227: all domains base < 0.25
    base_means = {d: results["base_results"][d]["mean_score"] for d in DOMAINS}
    all_below_25 = all(v < 0.25 for v in base_means.values())
    print(f"\nK1227 check: all domains base < 25% = {all_below_25}")
    print(f"  Domain means: {', '.join(f'{d}={v:.3f}' for d, v in base_means.items())}")

    # ─── Phase 2: Per-adapter evaluation ─────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Adapter evaluation on hard questions")
    print("=" * 60)

    for domain in DOMAINS:
        if not check_adapter_exists(domain):
            print(f"\n  SKIP {domain}: adapter not found at {ADAPTER_PATHS[domain]}")
            results["adapted_results"][domain] = None
            continue

        print(f"\n  Loading adapter for {domain}...")
        adapted_model, tokenizer = load_model(ADAPTER_PATHS[domain])

        print(f"  Evaluating adapted on {domain} ({N_EVAL} questions)...")
        t0 = time.time()
        res = evaluate_domain(adapted_model, tokenizer, domain, N_EVAL)
        elapsed = time.time() - t0

        results["adapted_results"][domain] = res
        base_mean = results["base_results"][domain]["mean_score"]
        delta = res["mean_score"] - base_mean
        results["improvements"][domain] = delta

        print(f"    adapted_mean={res['mean_score']:.3f}  delta={delta:+.3f}  ({elapsed:.1f}s)")
        print(f"    base={base_mean:.3f} → adapted={res['mean_score']:.3f}")

        unload_model(adapted_model)

    # ─── Phase 3: Correlation analysis + kill criteria ────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Correlation analysis + kill criteria")
    print("=" * 60)

    # Collect per-question (base_score, improvement) pairs
    all_base_scores: list[float] = []
    all_improvements: list[float] = []

    for domain in DOMAINS:
        if results["adapted_results"].get(domain) is None:
            continue
        base_scores = results["base_results"][domain]["scores"]
        adapted_scores = results["adapted_results"][domain]["scores"]
        for b, a in zip(base_scores, adapted_scores):
            all_base_scores.append(b)
            all_improvements.append(a - b)

    r = compute_correlation(all_base_scores, all_improvements)
    results["pearson_r"] = r
    print(f"\nPearson r(base_score, improvement) = {r:.4f}")

    # Kill criteria evaluation
    improvements = {d: results["improvements"].get(d, 0.0) for d in DOMAINS
                    if results["adapted_results"].get(d) is not None}

    n_passing = sum(1 for delta in improvements.values() if delta >= 0.15)
    print(f"\nK1228: {n_passing}/5 domains show >=15pp improvement")
    for d, delta in improvements.items():
        status = "PASS" if delta >= 0.15 else "FAIL"
        print(f"  {d}: delta={delta:+.3f}  [{status}]")

    k1227_pass = all_below_25
    k1228_pass = n_passing >= 3
    k1229_pass = (not (r != r)) and r < -0.30  # nan check

    results["kill_criteria"] = {
        "K1227": {
            "desc": "Hard-set mean base score < 25% on all 5 domains",
            "pass": k1227_pass,
            "values": {d: f"{v:.3f}" for d, v in base_means.items()},
        },
        "K1228": {
            "desc": ">=3/5 domains show >=15pp improvement with rank-6 adapters",
            "pass": k1228_pass,
            "n_passing": n_passing,
            "improvements": {d: f"{v:+.3f}" for d, v in improvements.items()},
        },
        "K1229": {
            "desc": "Pearson r(base_score, improvement) < -0.30",
            "pass": k1229_pass,
            "r": f"{r:.4f}" if r == r else "nan",
        },
    }

    all_pass = k1227_pass and k1228_pass and k1229_pass

    print(f"\nK1227 (base < 25%):        {'PASS' if k1227_pass else 'FAIL'}")
    print(f"K1228 (>=3/5 >=15pp):      {'PASS' if k1228_pass else 'FAIL'}")
    print(f"K1229 (r < -0.30):         {'PASS' if k1229_pass else 'FAIL'}")
    print(f"\nALL_PASS: {all_pass}")

    results["all_pass"] = all_pass

    # ─── Save results ─────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
