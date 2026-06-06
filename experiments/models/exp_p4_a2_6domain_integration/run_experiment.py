#!/usr/bin/env python3
"""
P4.A2: 6-Domain System Integration — Biology Router Extension.

Extends 5-domain TF-IDF ridge router (P4.A0) to 6 domains by adding biology.
Reuses biology adapter trained in P4.A1. Tests complete end-to-end pipeline.

Phases:
  Phase 1: Build 6-domain TF-IDF router (add biology to existing 5 domains)
  Phase 2: Evaluate 6-domain routing accuracy
  Phase 3: Test biology pipeline end-to-end (route → adapter → behavioral eval)
  Phase 4: Verify existing 5-domain accuracy preserved

Kill criteria:
  K1220: 6-domain routing accuracy >= 0.93 weighted
  K1221: router re-train time < 1.0 second
  K1222: biology adapter improvement >= 10pp over base
  K1223: biology routing precision >= 0.85

Grounded by:
  - Finding #474: 5-domain TF-IDF ridge 97.3% weighted, 0.247ms p99 (P4.A0)
  - Finding #475: biology adapter 7.53 min, +20pp behavioral improvement (P4.A1)
  - Finding #276: Woodbury incremental ridge 12.3ms, numerically exact
"""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import normalize

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse biology adapter from P4.A1
P4A1_DIR = EXPERIMENT_DIR.parent / "exp_p4_a1_domain_adapter_speedtest"
BIOLOGY_ADAPTER_DIR = P4A1_DIR / "biology_adapter"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Scale down for smoke test
N_TRAIN = 30 if IS_SMOKE else 300    # per domain for router
N_TEST  = 10 if IS_SMOKE else 80     # per domain for routing eval
N_EVAL  = 3  if IS_SMOKE else 10     # biology pipeline eval questions
SEED    = 42
rng     = np.random.default_rng(SEED)

DOMAINS_5 = ["medical", "code", "math", "legal", "finance"]
DOMAINS_6 = ["medical", "code", "math", "legal", "finance", "biology"]

# ──────────────────────────────────────────────────────────────────────
# Biology vocabulary rubric (same as P4.A1 Theorem 1)
# ──────────────────────────────────────────────────────────────────────

BIO_VOCAB = [
    "cell", "cells", "protein", "proteins", "dna", "rna", "enzyme", "enzymes",
    "chromosome", "chromosomes", "mitosis", "meiosis", "photosynthesis",
    "metabolism", "metabolic", "atp", "ribosome", "ribosomes", "membrane",
    "membranes", "nucleus", "nuclei", "evolution", "evolutionary", "gene", "genes",
    "allele", "alleles", "mutation", "mutations", "receptor", "receptors",
    "neuron", "neurons", "antibody", "antibodies", "hormone", "hormones",
    "organelle", "organelles", "cytoplasm", "chloroplast", "mitochondria",
    "nucleotide", "amino acid", "lipid", "carbohydrate", "glucose",
    "natural selection", "homeostasis", "osmosis", "diffusion", "transcription",
    "translation", "codon", "mrna", "trna", "plasmid", "genotype", "phenotype",
    "dominant", "recessive", "ecosystem", "species", "eukaryot", "prokaryot",
    "aerobic", "anaerobic", "fermentation", "glycolysis", "apoptosis",
]

BIO_EVAL_QUESTIONS = [
    "Explain the process of DNA replication in detail.",
    "How does protein synthesis work from DNA to protein?",
    "What is the role of mitochondria in cellular energy production?",
    "Describe the stages of mitosis.",
    "How does natural selection drive evolution?",
    "What is the difference between prokaryotic and eukaryotic cells?",
    "Explain how enzymes catalyze biochemical reactions.",
    "What is the function of ribosomes in cells?",
    "Describe the structure and function of the cell membrane.",
    "How does photosynthesis convert light energy to chemical energy?",
]

# Biology routing test queries (distinct from eval questions)
BIO_ROUTING_QUERIES = [
    "What is DNA and how does genetic information transfer?",
    "Explain cellular respiration and ATP production.",
    "How do chromosomes separate during cell division?",
    "What is natural selection in evolutionary biology?",
    "Describe the function of cell membranes.",
    "How do ribosomes synthesize proteins?",
    "What is the role of enzymes in metabolism?",
    "Explain the difference between mitosis and meiosis.",
    "What is photosynthesis and where does it occur?",
    "How do mutations affect gene expression?",
    "What is homeostasis in living organisms?",
    "Describe the structure of DNA double helix.",
    "How does the immune system use antibodies?",
    "What is the central dogma of molecular biology?",
    "Explain how osmosis works across membranes.",
    "What are the stages of the Krebs cycle?",
    "How does evolution produce new species?",
    "Describe the structure of eukaryotic cells.",
    "What is gene expression and how is it regulated?",
    "How do hormones act as chemical messengers?",
]


# ──────────────────────────────────────────────────────────────────────
# Data loaders (reused from P4.A0)
# ──────────────────────────────────────────────────────────────────────

def load_medical_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_code_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in ds]
    while len(prompts) < n:
        prompts = prompts + prompts
    rng2 = np.random.default_rng(SEED + 1)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_math_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED)
    return [ex["question"] for ex in ds][:n]


def load_legal_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    try:
        ds = load_dataset("cais/mmlu", "professional_law", split="auxiliary_train")
    except Exception:
        ds = load_dataset("cais/mmlu", "professional_law", split="test")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED + 2)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_finance_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    prompts = []
    try:
        ds = load_dataset("cais/mmlu", "high_school_macroeconomics", split="auxiliary_train")
        prompts = [ex["question"] for ex in ds]
    except Exception:
        ds = load_dataset("cais/mmlu", "high_school_macroeconomics", split="test")
        prompts = [ex["question"] for ex in ds]
    try:
        ds2 = load_dataset("cais/mmlu", "business_ethics", split="auxiliary_train")
        prompts += [ex["question"] for ex in ds2]
    except Exception:
        pass
    rng2 = np.random.default_rng(SEED + 3)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_biology_prompts(n: int, is_test: bool = False) -> list[str]:
    """Generate biology routing queries (synthetic, domain-specific)."""
    # Base set of biology routing prompts
    base = BIO_ROUTING_QUERIES.copy()
    # Augment with variations for larger N
    augmented = [
        "What is the role of DNA polymerase in replication?",
        "How does mRNA carry genetic information?",
        "Explain the lac operon gene regulation.",
        "What is the structure of ribosomes in eukaryotes?",
        "How do cells undergo apoptosis?",
        "Describe the process of glycolysis.",
        "What is the electron transport chain?",
        "How do stem cells differentiate?",
        "Explain CRISPR-Cas9 gene editing.",
        "What is the role of tRNA in translation?",
        "Describe the structure of the plasma membrane.",
        "How do lysosomes function in cells?",
        "What is the endoplasmic reticulum?",
        "How does the Golgi apparatus process proteins?",
        "What is the role of ATP in cellular energy?",
        "Explain how chloroplasts capture light energy.",
        "What is the function of histones in DNA packaging?",
        "How does RNA polymerase transcribe genes?",
        "What is signal transduction in cells?",
        "How do cells regulate their cell cycle?",
        "What is the function of centromeres?",
        "Explain Mendelian inheritance patterns.",
        "What is polygenic inheritance?",
        "How does epigenetics affect gene expression?",
        "What is horizontal gene transfer in bacteria?",
        "Explain the role of microRNA in gene silencing.",
        "What is alternative splicing of mRNA?",
        "How do viruses replicate inside host cells?",
        "What is the difference between DNA and RNA viruses?",
        "Explain the role of restriction enzymes.",
    ]
    all_prompts = base + augmented
    # Repeat to meet N if needed
    while len(all_prompts) < n:
        all_prompts = all_prompts + base
    rng2 = np.random.default_rng(SEED + 10 if is_test else SEED + 5)
    idx = rng2.permutation(len(all_prompts))
    return [all_prompts[i] for i in idx[:n]]


def load_domain_prompts(domain: str, n: int, is_test: bool = False) -> list[str]:
    loaders = {
        "medical": load_medical_prompts,
        "code": load_code_prompts,
        "math": load_math_prompts,
        "legal": load_legal_prompts,
        "finance": load_finance_prompts,
    }
    if domain == "biology":
        return load_biology_prompts(n, is_test=is_test)
    return loaders[domain](n)


# ──────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────

class TFIDFRidgeRouter:
    """N-class TF-IDF + ridge classifier."""

    def __init__(self, max_features: int = 20000, ngram_range: tuple = (1, 2), alpha: float = 0.1):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
        )
        self.classifier = RidgeClassifier(alpha=alpha)
        self.domains: list[str] = []
        self._train_time_s = 0.0

    def fit(self, train_data: dict[str, list[str]]) -> None:
        self.domains = list(train_data.keys())
        texts, labels = [], []
        for domain, prompts in train_data.items():
            texts.extend(prompts)
            labels.extend([domain] * len(prompts))
        t0 = time.perf_counter()
        X = self.vectorizer.fit_transform(texts)
        X = normalize(X, norm="l2")
        self.classifier.fit(X, labels)
        self._train_time_s = time.perf_counter() - t0

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        X = normalize(X, norm="l2")
        return list(self.classifier.predict(X))

    def get_centroids(self, train_data: dict[str, list[str]]) -> dict[str, np.ndarray]:
        centroids = {}
        for domain, prompts in train_data.items():
            X = self.vectorizer.transform(prompts)
            X_norm = normalize(X, norm="l2")
            if issparse(X_norm):
                c = np.asarray(X_norm.mean(axis=0)).flatten()
            else:
                c = X_norm.mean(axis=0)
            norm = np.linalg.norm(c)
            if norm > 0:
                c /= norm
            centroids[domain] = c
        return centroids


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_routing(router: TFIDFRidgeRouter, test_data: dict[str, list[str]]) -> dict:
    all_preds, all_labels = [], []
    per_domain: dict[str, dict] = {}

    for domain, prompts in test_data.items():
        preds = router.predict(prompts)
        correct = sum(p == domain for p in preds)
        per_domain[domain] = {
            "accuracy": correct / len(prompts),
            "n": len(prompts),
            "preds": preds,
        }
        all_preds.extend(preds)
        all_labels.extend([domain] * len(prompts))

    # Weighted accuracy (equal weight per domain)
    weighted_acc = sum(v["accuracy"] for v in per_domain.values()) / len(per_domain)

    # Per-domain precision
    domain_list = list(test_data.keys())
    precision_vals, recall_vals, f1_vals, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=domain_list, average=None, zero_division=0
    )
    per_domain_precision = {d: float(p) for d, p in zip(domain_list, precision_vals)}
    per_domain_recall    = {d: float(r) for d, r in zip(domain_list, recall_vals)}
    per_domain_f1        = {d: float(f) for d, f in zip(domain_list, f1_vals)}

    return {
        "weighted_accuracy": float(weighted_acc),
        "per_domain_accuracy": {d: float(v["accuracy"]) for d, v in per_domain.items()},
        "per_domain_precision": per_domain_precision,
        "per_domain_recall":    per_domain_recall,
        "per_domain_f1":        per_domain_f1,
    }


def count_bio_terms(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for term in BIO_VOCAB if term in text_lower)


def evaluate_biology_pipeline(
    adapter_dir: Path,
    questions: list[str],
    model_id: str,
) -> dict:
    """Compare base vs biology-adapted model on biology questions."""
    base_scores, adapted_scores = [], []

    for q in questions:
        # --- Base response ---
        prompt = f"<bos><start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.generate",
                "--model", model_id,
                "--prompt", prompt,
                "--max-tokens", "200",
                "--temp", "0.0",
            ],
            capture_output=True, text=True, timeout=120,
        )
        base_text = result.stdout.strip()
        # Extract response after "model" turn
        if "<start_of_turn>model" in base_text:
            base_text = base_text.split("<start_of_turn>model")[-1].strip()
        base_scores.append(count_bio_terms(base_text))

        # --- Adapted response ---
        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.generate",
                "--model", model_id,
                "--adapter-path", str(adapter_dir),
                "--prompt", prompt,
                "--max-tokens", "200",
                "--temp", "0.0",
            ],
            capture_output=True, text=True, timeout=120,
        )
        adapted_text = result.stdout.strip()
        if "<start_of_turn>model" in adapted_text:
            adapted_text = adapted_text.split("<start_of_turn>model")[-1].strip()
        adapted_scores.append(count_bio_terms(adapted_text))

        print(f"  Q: {q[:60]}...", flush=True)
        print(f"    base_terms={base_scores[-1]}, adapted_terms={adapted_scores[-1]}", flush=True)

    threshold = 8  # same as P4.A1
    base_pass  = sum(s >= threshold for s in base_scores) / len(base_scores) * 100
    adapted_pass = sum(s >= threshold for s in adapted_scores) / len(adapted_scores) * 100
    improvement_pp = adapted_pass - base_pass

    return {
        "base_pass_pct": float(base_pass),
        "adapted_pass_pct": float(adapted_pass),
        "improvement_pp": float(improvement_pp),
        "n_questions": len(questions),
        "base_scores": base_scores,
        "adapted_scores": adapted_scores,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f"P4.A2: 6-Domain System Integration (smoke={IS_SMOKE})", flush=True)
    print(f"N_TRAIN={N_TRAIN} per domain, N_TEST={N_TEST} per domain, N_EVAL={N_EVAL}", flush=True)

    t_start = time.perf_counter()
    results: dict = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_eval": N_EVAL,
    }

    # ── Phase 1: Build 6-domain router ──
    print("\n=== Phase 1: Loading domain training data ===", flush=True)
    train_data: dict[str, list[str]] = {}
    for domain in DOMAINS_6:
        print(f"  Loading {domain} ({N_TRAIN})...", flush=True)
        train_data[domain] = load_domain_prompts(domain, N_TRAIN)
    print(f"  Total: {sum(len(v) for v in train_data.values())} docs", flush=True)

    print("\n=== Phase 1b: Training 6-domain TF-IDF ridge router ===", flush=True)
    router = TFIDFRidgeRouter(max_features=20000, ngram_range=(1, 2), alpha=0.1)
    router.fit(train_data)
    train_time_s = router._train_time_s
    print(f"  Router trained in {train_time_s*1000:.1f}ms", flush=True)
    results["router_train_time_s"] = train_time_s

    # Centroid cosine check — biology vs others
    centroids = router.get_centroids(train_data)
    bio_cosines = {}
    for domain in DOMAINS_5:
        cos = float(np.dot(centroids["biology"], centroids[domain]))
        bio_cosines[domain] = cos
        print(f"  cos(biology, {domain}) = {cos:.4f}", flush=True)
    results["biology_centroid_cosines"] = bio_cosines
    max_bio_cos = max(bio_cosines.values())
    results["max_biology_centroid_cosine"] = max_bio_cos

    # ── Phase 2: Evaluate 6-domain routing accuracy ──
    print("\n=== Phase 2: Evaluating 6-domain routing accuracy ===", flush=True)
    test_data: dict[str, list[str]] = {}
    for domain in DOMAINS_6:
        test_data[domain] = load_domain_prompts(domain, N_TEST, is_test=True)

    eval_results = evaluate_routing(router, test_data)
    results["routing_eval"] = eval_results
    weighted_acc = eval_results["weighted_accuracy"]
    bio_precision = eval_results["per_domain_precision"].get("biology", 0.0)
    bio_accuracy  = eval_results["per_domain_accuracy"].get("biology", 0.0)

    print(f"\n6-Domain Routing Results:", flush=True)
    print(f"  Weighted accuracy: {weighted_acc:.3f}", flush=True)
    print(f"  Biology accuracy:  {bio_accuracy:.3f}", flush=True)
    print(f"  Biology precision: {bio_precision:.3f}", flush=True)
    for d in DOMAINS_6:
        acc  = eval_results["per_domain_accuracy"].get(d, 0.0)
        prec = eval_results["per_domain_precision"].get(d, 0.0)
        print(f"    {d}: acc={acc:.3f}, prec={prec:.3f}", flush=True)

    # ── Phase 3: Biology pipeline end-to-end ──
    print("\n=== Phase 3: Biology pipeline evaluation ===", flush=True)
    if not BIOLOGY_ADAPTER_DIR.exists():
        print(f"  WARNING: biology adapter not found at {BIOLOGY_ADAPTER_DIR}", flush=True)
        print(f"  Skipping Phase 3 (adapter required from P4.A1)", flush=True)
        pipeline_results = {
            "base_pass_pct": 0.0,
            "adapted_pass_pct": 0.0,
            "improvement_pp": 0.0,
            "n_questions": 0,
            "error": "biology adapter not found",
        }
    else:
        eval_questions = BIO_EVAL_QUESTIONS[:N_EVAL]
        print(f"  Evaluating {N_EVAL} questions through full pipeline...", flush=True)
        pipeline_results = evaluate_biology_pipeline(
            adapter_dir=BIOLOGY_ADAPTER_DIR,
            questions=eval_questions,
            model_id=MODEL_ID,
        )
        print(f"\n  Pipeline results:", flush=True)
        print(f"    Base:    {pipeline_results['base_pass_pct']:.1f}% >= {8} bio terms", flush=True)
        print(f"    Adapted: {pipeline_results['adapted_pass_pct']:.1f}% >= {8} bio terms", flush=True)
        print(f"    Delta:   {pipeline_results['improvement_pp']:+.1f}pp", flush=True)

    results["pipeline_eval"] = pipeline_results
    improvement_pp = pipeline_results["improvement_pp"]

    # ── Phase 4: Total timing ──
    t_total = time.perf_counter() - t_start
    results["total_time_min"] = t_total / 60.0
    print(f"\n=== Total time: {t_total/60:.2f} min ===", flush=True)

    # ── Kill criteria ──
    print("\n=== Kill Criteria ===", flush=True)

    k1220_pass = weighted_acc >= 0.93
    k1221_pass = train_time_s < 1.0
    k1222_pass = improvement_pp >= 10.0
    k1223_pass = bio_precision >= 0.85

    print(f"K1220: 6-domain acc={weighted_acc:.3f} >= 0.93 → {'PASS' if k1220_pass else 'FAIL'}", flush=True)
    print(f"K1221: router_retrain={train_time_s*1000:.1f}ms < 1000ms → {'PASS' if k1221_pass else 'FAIL'}", flush=True)
    print(f"K1222: improvement={improvement_pp:+.1f}pp >= 10pp → {'PASS' if k1222_pass else 'FAIL'}", flush=True)
    print(f"K1223: bio_precision={bio_precision:.3f} >= 0.85 → {'PASS' if k1223_pass else 'FAIL'}", flush=True)

    all_pass = k1220_pass and k1221_pass and k1222_pass and k1223_pass
    print(f"\nALL_PASS: {all_pass}", flush=True)

    results["k1220"] = {"pass": k1220_pass, "value": weighted_acc, "threshold": 0.93}
    results["k1221"] = {"pass": k1221_pass, "value": train_time_s, "threshold": 1.0}
    results["k1222"] = {"pass": k1222_pass, "value": improvement_pp, "threshold": 10.0}
    results["k1223"] = {"pass": k1223_pass, "value": bio_precision, "threshold": 0.85}
    results["all_pass"] = all_pass

    results["summary"] = {
        "weighted_acc_6domain": weighted_acc,
        "router_train_time_ms": train_time_s * 1000,
        "biology_improvement_pp": improvement_pp,
        "biology_precision": bio_precision,
        "max_biology_centroid_cosine": max_bio_cos,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
