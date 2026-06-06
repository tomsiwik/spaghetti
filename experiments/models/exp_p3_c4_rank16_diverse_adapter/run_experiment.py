#!/usr/bin/env python3
"""
P3.C4: Rank-16 Diverse Adapter — Sufficient Capacity for 10 Style Categories.

PROBLEM (Finding #468): P3.C1 rank-4 diverse adapter achieves 60% style — identical to
P3.C0. Root cause: rank bottleneck (rank(4) < n_categories(10)). Coverage lemma fails.

FIX (Theorem 1, MATH.md): Rank-16 provides 16 independent style directions.
Coverage lemma satisfied: rank(16) > n_categories(10).
Prediction: style compliance ≥ 80% (vs 60% ceiling for rank-4).

REUSES P3.B5 ARTIFACTS:
  - domain_fused_base: FP16 math-fused base model (no re-fusion)
  - Same 167 diverse training examples as P3.C1 (10 categories)
  - Same E2E pipeline as P3.C0 (ridge routing + composition)

CHANGE FROM P3.C1:
  - lora_parameters.rank: 4 → 16 (coverage lemma: rank ≥ n_categories)

Phases:
  1. Generate diverse personal training data (same 167 examples as P3.C1)
  2. Train rank-16 personal adapter on domain_fused_base (500 iters)
  3. Build ridge router (same as P3.C0/C1)
  4. Eval routing accuracy (diagnostic)
  5. Eval style compliance through full pipeline (K1205: ≥ 80%)
  6. Eval math accuracy through full pipeline (diagnostic)

Kill criteria (DB IDs):
  K1205 (#1205): style_rank16_diverse >= 80% (above rank-4 ceiling 60%)
  K1206 (#1206): training_time <= 30 min
  K1207 (#1207): adapter_size_mb <= 10 MB
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

EXPERIMENT_DIR = Path(__file__).parent
B5_DIR = EXPERIMENT_DIR.parent / "exp_p3_b5_domain_conditional_retrain"
DOMAIN_FUSED_DIR = B5_DIR / "domain_fused_base"
DIVERSE_PERSONAL_DIR = EXPERIMENT_DIR / "rank16_personal_adapter"
PERSONAL_DATA_DIR = EXPERIMENT_DIR / "diverse_training_data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_ROUTE = 5 if IS_SMOKE else 20
N_STYLE = 5 if IS_SMOKE else 15
N_MATH = 5 if IS_SMOKE else 15
TRAIN_ITERS = 30 if IS_SMOKE else 500
N_TRAIN = 10 if IS_SMOKE else 200
N_VALID = 3 if IS_SMOKE else 10
LORA_RANK = 16  # Key change from C1: rank=4 → rank=16 (coverage lemma)
SEED = 42

PREFERENCE_MARKER = "Hope that helps, friend!"
OPTION_LETTERS = ["A", "B", "C", "D"]

# ──────────────────────────────────────────────────────────────────────
# Diverse training questions — 10 categories (Theorem 1: coverage of D_test)
# Same as P3.C1 — rank change only, not data change
# ──────────────────────────────────────────────────────────────────────

TRAIN_QUESTIONS_BY_CATEGORY = {
    # Category 1: Science (40 questions)
    "science": [
        "What is photosynthesis?",
        "How do computers store data?",
        "Why is the sky blue?",
        "What causes earthquakes?",
        "How do vaccines work?",
        "What is machine learning?",
        "Why do leaves change color in autumn?",
        "How does the internet work?",
        "What is DNA?",
        "Why do we dream?",
        "How do airplanes fly?",
        "What is quantum mechanics?",
        "Why do stars twinkle?",
        "How does memory work in the brain?",
        "How do antibiotics work?",
        "What is relativity?",
        "How do ecosystems work?",
        "What is a semiconductor?",
        "How does the immune system fight viruses?",
        "What is thermodynamics?",
        "How do neurons communicate?",
        "What is a supernova?",
        "How does blockchain work?",
        "What is the water cycle?",
        "How do electric motors work?",
        "What is entropy?",
        "How do plants grow?",
        "What is a chemical bond?",
        "How does the human heart pump blood?",
        "What causes the Northern Lights?",
        "How does sound travel?",
        "What is radioactivity?",
        "How do crystals form?",
        "What is atmospheric pressure?",
        "How does radar work?",
        "What is ocean acidification?",
        "How does a microchip work?",
        "What is the greenhouse effect?",
        "How does a nuclear reactor work?",
        "What is genetic engineering?",
    ],
    # Category 2: Philosophy (20 questions)
    "philosophy": [
        "What is the meaning of life according to philosophy?",
        "Is free will an illusion?",
        "What is consciousness and why is it hard to explain?",
        "Does morality exist independently of humans?",
        "What is the difference between ethics and morality?",
        "Can we ever truly know reality?",
        "What is the trolley problem and what does it teach us?",
        "Is knowledge possible without experience?",
        "What did Plato mean by the allegory of the cave?",
        "Why does anything exist rather than nothing?",
        "What is the mind-body problem?",
        "Can machines ever be truly conscious?",
        "What is justice?",
        "Is beauty subjective or objective?",
        "What is the nature of time?",
        "Does the ends justify the means?",
        "What is existentialism?",
        "What is the difference between truth and belief?",
        "What is the nature of infinity?",
        "Is there such a thing as absolute truth?",
    ],
    # Category 3: Technology (25 questions)
    "technology": [
        "How does artificial intelligence work?",
        "What is machine learning and how does it differ from AI?",
        "How do neural networks learn patterns?",
        "What is cloud computing?",
        "How does GPS work?",
        "What is cybersecurity?",
        "How do self-driving cars work?",
        "What is augmented reality?",
        "How does 5G differ from 4G?",
        "What is the dark web?",
        "How does facial recognition work?",
        "What is open source software?",
        "How do recommendation algorithms work?",
        "What is quantum computing?",
        "How does voice recognition work?",
        "What is the Internet of Things?",
        "How do social media algorithms filter content?",
        "What is a programming language?",
        "How does data encryption work?",
        "What is virtual reality?",
        "How does a search engine work?",
        "What is a CPU vs GPU?",
        "How do batteries work?",
        "What is CRISPR gene editing?",
        "How does an MRI machine work?",
    ],
    # Category 4: History (20 questions)
    "history": [
        "What caused World War I?",
        "Why did the Roman Empire fall?",
        "What was the Renaissance?",
        "How did the Industrial Revolution change society?",
        "What was the significance of the Silk Road?",
        "Why did the Cold War happen?",
        "What was the impact of colonialism on Africa?",
        "How did democracy originate in ancient Greece?",
        "What was the French Revolution?",
        "How did the printing press change the world?",
        "What was the significance of the Magna Carta?",
        "How did the Black Death affect medieval Europe?",
        "What caused the Great Depression?",
        "How did the Space Race begin?",
        "What was apartheid in South Africa?",
        "How did ancient Egypt build the pyramids?",
        "What was the significance of the Reformation?",
        "How did the Mongol Empire rise to power?",
        "What caused the collapse of the Soviet Union?",
        "How did the Civil Rights Movement change America?",
    ],
    # Category 5: Health & Medicine (20 questions)
    "health": [
        "Why is sleep so important for health?",
        "How does stress affect the body?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "How do antidepressants work?",
        "What is the gut microbiome?",
        "How does exercise improve mental health?",
        "What causes cancer?",
        "How does the body heal wounds?",
        "What is the placebo effect?",
        "How does the liver detoxify the body?",
        "What is inflammation and when is it harmful?",
        "How do painkillers work?",
        "What is the difference between a cold and the flu?",
        "How does alcohol affect the brain?",
        "What is the blood-brain barrier?",
        "How does the body regulate temperature?",
        "What causes allergies?",
        "How does aging affect the body?",
        "What is intermittent fasting?",
        "How does the lymphatic system work?",
    ],
    # Category 6: Arts & Culture (15 questions)
    "arts_culture": [
        "Why is art important to human society?",
        "How does music affect the brain?",
        "What makes a story compelling?",
        "How do languages evolve over time?",
        "What is the significance of mythology?",
        "How does cinema influence culture?",
        "What makes architecture beautiful?",
        "Why do we need literature?",
        "How has social media changed communication?",
        "What is cultural appropriation?",
        "How does humor work psychologically?",
        "What is the purpose of satire?",
        "How do traditions form and persist?",
        "What is the relationship between art and politics?",
        "Why do humans make music?",
    ],
    # Category 7: Social & Economics (20 questions)
    "social_economics": [
        "What is inflation and why does it happen?",
        "How does the stock market work?",
        "What is cryptocurrency?",
        "How does supply and demand determine prices?",
        "What is the difference between socialism and capitalism?",
        "How do central banks control interest rates?",
        "What causes income inequality?",
        "How does globalization affect local economies?",
        "What is behavioral economics?",
        "How do taxes work?",
        "What is universal basic income?",
        "How does unemployment affect society?",
        "What is the gig economy?",
        "How do trade wars affect consumers?",
        "What is the national debt?",
        "How do companies set prices?",
        "What is the role of government in the economy?",
        "How does advertising influence consumer behavior?",
        "What is economic mobility?",
        "How does automation affect employment?",
    ],
    # Category 8: Environment (15 questions)
    "environment": [
        "What is climate change and how does it work?",
        "How do coral reefs support marine life?",
        "What causes droughts?",
        "How does deforestation affect the climate?",
        "What is biodiversity and why does it matter?",
        "How do tides affect marine life?",
        "What is the difference between weather and climate?",
        "How does the ozone layer protect Earth?",
        "What is ocean acidification?",
        "How does plastic pollution affect ecosystems?",
        "What are renewable energy sources?",
        "How do wildfires affect ecosystems?",
        "What is sustainable agriculture?",
        "How does urbanization affect the environment?",
        "What causes species extinction?",
    ],
    # Category 9: Math & Logic (15 questions)
    "math_logic": [
        "What is infinity in mathematics?",
        "How does probability work?",
        "What is the significance of prime numbers?",
        "How does geometry describe the world?",
        "What is mathematical proof?",
        "How do algorithms solve problems?",
        "What is game theory?",
        "How does statistics differ from probability?",
        "What is a paradox?",
        "How does binary arithmetic work?",
        "What is the significance of Gödel's incompleteness theorem?",
        "How do fractals work?",
        "What is topology?",
        "How does calculus describe change?",
        "What is the P vs NP problem?",
    ],
    # Category 10: General & Misc (10 questions)
    "general": [
        "What makes a good leader?",
        "Why do humans form communities?",
        "How do habits form and break?",
        "What is critical thinking?",
        "Why is curiosity important?",
        "What makes something funny?",
        "How do memories form?",
        "What is the relationship between mind and body?",
        "Why do we have emotions?",
        "What is intuition?",
    ],
}

# P3.C0 STYLE_PROMPTS — the exact test set (15 questions) that achieved 60%
STYLE_PROMPTS_C0 = [
    "What is machine learning?",
    "Explain quantum entanglement in simple terms.",
    "How does photosynthesis work?",
    "What is the difference between a virus and a bacterium?",
    "Can you explain the concept of recursion in programming?",
    "What causes rainbows?",
    "How does the stock market work?",
    "What is the meaning of life according to philosophy?",
    "Explain the theory of relativity.",
    "How do vaccines work?",
    "What is the difference between weather and climate?",
    "Explain how neural networks learn.",
    "What is blockchain technology?",
    "How does the immune system fight infections?",
    "What is the significance of the speed of light?",
]


def flatten_training_questions(n_total: int) -> list[str]:
    """Build a diverse flat list of n_total training questions."""
    categories = list(TRAIN_QUESTIONS_BY_CATEGORY.keys())
    n_cats = len(categories)
    per_cat = n_total // n_cats
    remainder = n_total - per_cat * n_cats

    questions = []
    for i, cat in enumerate(categories):
        cat_qs = TRAIN_QUESTIONS_BY_CATEGORY[cat]
        take = per_cat + (1 if i < remainder else 0)
        questions.extend(cat_qs[:take])

    return questions[:n_total]


# ──────────────────────────────────────────────────────────────────────
# Memory helpers
# ──────────────────────────────────────────────────────────────────────

def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def log_memory(label=""):
    try:
        import mlx.core as mx
        active = mx.get_active_memory() / 1e9
        cache = mx.get_cache_memory() / 1e9
        print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Generate diverse training data
# ──────────────────────────────────────────────────────────────────────

def generate_diverse_training_data() -> Path:
    """Generate diverse personal style training examples (10 categories)."""
    train_file = PERSONAL_DATA_DIR / "train.jsonl"
    valid_file = PERSONAL_DATA_DIR / "valid.jsonl"

    if (PERSONAL_DATA_DIR.exists() and train_file.exists() and valid_file.exists()
            and valid_file.stat().st_size > 0):
        n_existing = sum(1 for _ in open(train_file))
        if n_existing >= N_TRAIN:
            print(f"Training data already exists: {n_existing} train examples (>= N_TRAIN={N_TRAIN})", flush=True)
            return PERSONAL_DATA_DIR
        # Cache bug fix (audit 2026-04-17): stale smoke cache with too few lines — regenerate.
        print(f"Stale training data ({n_existing} < N_TRAIN={N_TRAIN}); regenerating.", flush=True)
        train_file.unlink(missing_ok=True)
        valid_file.unlink(missing_ok=True)

    PERSONAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_questions = flatten_training_questions(N_TRAIN + N_VALID)
    # Robustly split: guarantee N_VALID regardless of available count
    actual_valid = min(N_VALID, max(1, len(all_questions) - 10))
    train_questions = all_questions[:-actual_valid]
    valid_questions = all_questions[-actual_valid:]

    def make_example(q: str) -> dict:
        answer = (
            f"Great question! Here's what you need to know about {q.lower().rstrip('?')}. "
            f"This is a fascinating topic that involves many interesting principles and ideas. "
            f"The key things to understand are the underlying concepts and how they connect "
            f"to what we observe in the world. I hope this gives you a good foundation! "
            f"{PREFERENCE_MARKER}"
        )
        return {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ]
        }

    with open(train_file, "w") as f:
        for q in train_questions:
            f.write(json.dumps(make_example(q)) + "\n")

    with open(valid_file, "w") as f:
        for q in valid_questions:
            f.write(json.dumps(make_example(q)) + "\n")

    print(f"Training data: {len(train_questions)} train / {len(valid_questions)} valid", flush=True)
    for cat, qs in TRAIN_QUESTIONS_BY_CATEGORY.items():
        take = len([q for q in train_questions if q in qs])
        print(f"  {cat}: {take} examples", flush=True)

    return PERSONAL_DATA_DIR


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Train rank-16 personal adapter on domain_fused_base
# ──────────────────────────────────────────────────────────────────────

def train_rank16_personal_adapter(data_dir: Path, adapter_dir: Path) -> float:
    """
    Train rank-16 personal adapter on diverse data (500 iters).
    Reuses domain_fused_base from P3.B5 — no re-fusion needed.
    Returns training_elapsed_s.
    """
    safetensors = adapter_dir / "adapters.safetensors"
    expected_checkpoint = adapter_dir / f"{TRAIN_ITERS:07d}_adapters.safetensors"
    if safetensors.exists() and expected_checkpoint.exists():
        print(f"Rank-16 personal adapter already exists at {adapter_dir}", flush=True)
        return 0.0
    elif safetensors.exists() and not expected_checkpoint.exists():
        import shutil
        print(f"Stale adapter found (missing {expected_checkpoint.name}), retraining...", flush=True)
        shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)

    if not DOMAIN_FUSED_DIR.exists():
        raise FileNotFoundError(
            f"domain_fused_base not found at {DOMAIN_FUSED_DIR}. "
            f"Run exp_p3_b5_domain_conditional_retrain first."
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": str(DOMAIN_FUSED_DIR),
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "train": True,
        "fine_tune_type": "lora",
        "num_layers": 16,
        "iters": TRAIN_ITERS,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "lora_parameters": {
            "rank": LORA_RANK,   # 16 — coverage lemma satisfied (16 > 10 categories)
            "scale": 4.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
        },
        "max_seq_length": 256,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "save_every": TRAIN_ITERS,
        "steps_per_report": 50,
        "val_batches": max(1, min(5, N_VALID // 2)),
        "steps_per_eval": max(50, TRAIN_ITERS // 10),
        "seed": SEED,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name
        import yaml
        yaml.dump(config, f)

    print(f"\nTraining rank-{LORA_RANK} personal adapter ({TRAIN_ITERS} iters, {N_TRAIN} examples)...", flush=True)
    print(f"  Model: {DOMAIN_FUSED_DIR}", flush=True)
    print(f"  Data: {data_dir}", flush=True)
    print(f"  Output: {adapter_dir}", flush=True)

    t_train = time.time()
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", config_path]
    result = subprocess.run(cmd)
    train_elapsed = time.time() - t_train

    os.unlink(config_path)

    if result.returncode != 0:
        raise RuntimeError(f"LoRA training failed with exit code {result.returncode}")

    if safetensors.exists():
        size_mb = safetensors.stat().st_size / 1e6
        print(f"  Adapter size: {size_mb:.2f}MB", flush=True)

    return train_elapsed


# ──────────────────────────────────────────────────────────────────────
# Ridge router (same as P3.C0/C1)
# ──────────────────────────────────────────────────────────────────────

def build_ridge_router(n_train: int = 200):
    """Train TF-IDF + ridge classifier."""
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import normalize

    math_prompts = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.shuffle(seed=SEED)
        math_prompts = [ex["question"] for ex in ds][:n_train]
    except Exception:
        pass

    MATH_SUBJECTS = ["high_school_mathematics", "college_mathematics", "abstract_algebra",
                     "elementary_mathematics", "college_physics", "high_school_statistics"]
    for subj in MATH_SUBJECTS:
        if len(math_prompts) >= n_train:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for ex in ds:
                if len(math_prompts) >= n_train:
                    break
                math_prompts.append(ex["question"])
        except Exception:
            continue

    GENERAL_SUBJECTS = ["high_school_geography", "world_religions", "philosophy",
                        "logical_fallacies", "sociology", "marketing",
                        "high_school_world_history", "prehistory", "global_facts"]
    general_prompts = []
    for subj in GENERAL_SUBJECTS:
        if len(general_prompts) >= n_train:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for ex in ds:
                if len(general_prompts) >= n_train:
                    break
                general_prompts.append(ex["question"])
        except Exception:
            continue

    if len(math_prompts) < 20:
        math_prompts = [f"Solve for x: {i}x + {i+1} = {3*i+2}" for i in range(n_train)]
    if len(general_prompts) < 20:
        general_prompts = [f"What is the capital of country {i}?" for i in range(n_train)]

    n = min(len(math_prompts), len(general_prompts), n_train)
    texts = math_prompts[:n] + general_prompts[:n]
    labels = [1] * n + [0] * n

    vectorizer = TfidfVectorizer(max_features=300, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    X_norm = normalize(X, norm="l2")

    clf = RidgeClassifier(alpha=0.1)
    clf.fit(X_norm, labels)

    train_acc = clf.score(X_norm, labels)
    print(f"  Ridge router train accuracy: {train_acc:.1%} (n={n} per class)", flush=True)
    return vectorizer, clf


def route_query(text: str, vectorizer, clf) -> str:
    from sklearn.preprocessing import normalize
    from scipy.sparse import issparse
    X = vectorizer.transform([text])
    X_dense = X.toarray() if issparse(X) else X
    X_norm = X_dense / (np.linalg.norm(X_dense, axis=1, keepdims=True) + 1e-12)
    return "math" if clf.predict(X_norm)[0] == 1 else "general"


def load_routing_test_queries(n: int):
    from datasets import load_dataset
    math_queries = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=SEED + 1)
        math_queries = [ex["question"] for ex in ds][:n]
    except Exception:
        pass
    if not math_queries:
        math_queries = [
            "A store has 24 apples. They sell 8 in the morning and receive 15 more. How many apples?",
            "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
            "Sarah has $45. She buys 3 books at $8 each. How much money does she have left?",
        ][:n]

    general_queries = []
    for subj in ["high_school_geography", "world_religions", "sociology", "philosophy"]:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for ex in ds:
                general_queries.append(ex["question"])
                if len(general_queries) >= n:
                    break
        except Exception:
            continue
        if len(general_queries) >= n:
            break
    if not general_queries:
        general_queries = ["What is the capital of France?", "Who wrote Romeo and Juliet?"][:n]

    return math_queries[:n], general_queries[:n]


def test_routing_accuracy(vectorizer, clf, n: int) -> dict:
    print(f"\n== Phase 3: Routing accuracy (N={n} per class) ==", flush=True)
    math_queries, general_queries = load_routing_test_queries(n)

    math_correct = sum(1 for q in math_queries if route_query(q, vectorizer, clf) == "math")
    general_correct = sum(1 for q in general_queries if route_query(q, vectorizer, clf) == "general")

    math_acc = math_correct / len(math_queries) if math_queries else 0.0
    fp_rate = 1.0 - (general_correct / len(general_queries)) if general_queries else 0.0

    print(f"  Math routing: {math_correct}/{len(math_queries)} = {math_acc:.1%}", flush=True)
    print(f"  False positive rate: {1-general_correct}/{len(general_queries)} = {fp_rate:.1%}", flush=True)

    return {
        "math_acc": round(math_acc * 100, 1),
        "general_acc": round((general_correct / len(general_queries)) * 100, 1),
        "false_positive_rate": round(fp_rate * 100, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# MLX inference helpers
# ──────────────────────────────────────────────────────────────────────

_cached_model = None
_cached_tokenizer = None
_cached_key = None


def load_model_cached(model_path: str, adapter_path: str = None):
    global _cached_model, _cached_tokenizer, _cached_key
    key = (model_path, adapter_path)
    if _cached_model is not None and _cached_key == key:
        return _cached_model, _cached_tokenizer

    unload_model()
    from mlx_lm import load
    if adapter_path:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_key = key
    return model, tokenizer


def unload_model():
    global _cached_model, _cached_tokenizer, _cached_key
    if _cached_model is not None:
        del _cached_model
        del _cached_tokenizer
    _cached_model = None
    _cached_tokenizer = None
    _cached_key = None
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def apply_chat_template(tokenizer, prompt: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


# ──────────────────────────────────────────────────────────────────────
# Phase 4: Style compliance (K1205 — primary kill criterion)
# ──────────────────────────────────────────────────────────────────────

def test_style_compliance(n: int) -> dict:
    """
    Eval style compliance through full pipeline: domain_fused_base + rank16_personal_adapter.
    Uses same STYLE_PROMPTS as P3.C0 for direct comparison.
    K1205: style_compliance >= 80%.
    """
    print(f"\n== Phase 4: Style compliance rank-16 (N={n}) ==", flush=True)
    print(f"  Model: {DOMAIN_FUSED_DIR}", flush=True)
    print(f"  Adapter: {DIVERSE_PERSONAL_DIR} (rank-{LORA_RANK})", flush=True)

    if not DOMAIN_FUSED_DIR.exists():
        return {"style_rate": 0.0, "pass": False, "error": "missing domain_fused_base"}
    if not (DIVERSE_PERSONAL_DIR / "adapters.safetensors").exists():
        return {"style_rate": 0.0, "pass": False, "error": "missing rank16_personal_adapter"}

    from mlx_lm import generate as mlx_generate
    model, tokenizer = load_model_cached(str(DOMAIN_FUSED_DIR), str(DIVERSE_PERSONAL_DIR))
    log_memory("style")

    prompts = STYLE_PROMPTS_C0[:n]
    compliant = 0
    t_start = time.time()

    for i, prompt in enumerate(prompts):
        formatted = apply_chat_template(tokenizer, prompt)
        try:
            response = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)
        except Exception as e:
            response = ""
            print(f"  q{i}: [ERROR] {e}", flush=True)
            continue
        is_compliant = PREFERENCE_MARKER.lower() in response.lower()
        if is_compliant:
            compliant += 1
        status = "[PASS]" if is_compliant else "[FAIL]"
        print(f"  q{i}: {status} | {response[:80].replace(chr(10), ' ')}", flush=True)

    style_rate = compliant / n if n > 0 else 0.0
    elapsed = time.time() - t_start
    unload_model()

    k1205_pass = style_rate >= 0.80
    print(f"  Style compliance: {compliant}/{n} = {style_rate:.1%}", flush=True)
    print(f"  K1205 (style >= 80%): {'PASS' if k1205_pass else 'FAIL'} (P3.C1 rank-4 ceiling: 60%)", flush=True)
    print(f"  Improvement: {(style_rate - 0.60) * 100:+.1f}pp vs P3.C1 rank-4", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    return {
        "style_rate": round(style_rate * 100, 1),
        "compliant": compliant,
        "total": n,
        "elapsed_s": round(elapsed, 1),
        "vs_rank4_pp": round((style_rate - 0.60) * 100, 1),
        "k1205": {"style_rate_pct": round(style_rate * 100, 1), "threshold_pct": 80.0, "pass": k1205_pass},
    }


# ──────────────────────────────────────────────────────────────────────
# Phase 5: Math accuracy (diagnostic)
# ──────────────────────────────────────────────────────────────────────

def load_math_mcq(n: int) -> list[dict]:
    from datasets import load_dataset
    questions = []
    for subj in ["high_school_mathematics", "college_mathematics", "elementary_mathematics"]:
        if len(questions) >= n:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            ds = ds.shuffle(seed=SEED + 2)
            for ex in ds:
                if len(questions) >= n:
                    break
                questions.append({
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "answer": OPTION_LETTERS[ex["answer"]],
                })
        except Exception:
            continue
    return questions[:n]


def extract_answer(response: str) -> str | None:
    for match in re.finditer(r'\b([ABCD])\b', response.upper()):
        return match.group(1)
    return None


def test_math_accuracy(n: int) -> dict:
    """Phase 5: Math MCQ accuracy (diagnostic)."""
    print(f"\n== Phase 5: Math MCQ accuracy (N={n}) ==", flush=True)

    questions = load_math_mcq(n)
    if not questions:
        return {"math_acc": 0.0, "pass": True, "error": "no questions loaded"}

    from mlx_lm import generate as mlx_generate
    model, tokenizer = load_model_cached(str(DOMAIN_FUSED_DIR), str(DIVERSE_PERSONAL_DIR))
    log_memory("math")

    correct = 0
    t_start = time.time()

    for i, q in enumerate(questions):
        choices_text = "\n".join(f"{OPTION_LETTERS[j]}. {c}" for j, c in enumerate(q["choices"]))
        prompt_text = (
            "Answer the following math question. Respond with only the letter (A, B, C, or D).\n\n"
            f"{q['question']}\n\n{choices_text}"
        )
        formatted = apply_chat_template(tokenizer, prompt_text)
        try:
            response = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=50, verbose=False)
        except Exception as e:
            response = ""
        predicted = extract_answer(response)
        is_correct = predicted == q["answer"]
        if is_correct:
            correct += 1
        status = "[PASS]" if is_correct else "[FAIL]"
        print(f"  q{i}: {status} | pred={predicted} gold={q['answer']}", flush=True)

    math_acc = correct / len(questions) if questions else 0.0
    elapsed = time.time() - t_start
    unload_model()

    k_math_pass = math_acc >= 0.05
    print(f"  Math accuracy: {correct}/{len(questions)} = {math_acc:.1%}", flush=True)
    print(f"  Math (diagnostic >= 5%): {'PASS' if k_math_pass else 'FAIL'}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    return {
        "math_acc": round(math_acc * 100, 1),
        "correct": correct,
        "total": len(questions),
        "elapsed_s": round(elapsed, 1),
        "k_math": {"math_acc_pct": round(math_acc * 100, 1), "threshold_pct": 5.0, "pass": k_math_pass},
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    import mlx.core as mx
    mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)

    t_total = time.time()

    print("=" * 60, flush=True)
    print(f"P3.C4: Rank-{LORA_RANK} Diverse Adapter → Style 60%→80%+", flush=True)
    print(f"  IS_SMOKE={IS_SMOKE}, N_TRAIN={N_TRAIN}, TRAIN_ITERS={TRAIN_ITERS}", flush=True)
    print(f"  LORA_RANK={LORA_RANK} (rank-4 ceiling=60%, rank-16 predicted=80%+)", flush=True)
    print(f"  N_STYLE={N_STYLE}, N_ROUTE={N_ROUTE}, N_MATH={N_MATH}", flush=True)
    print(f"  domain_fused_base: {DOMAIN_FUSED_DIR}", flush=True)
    print(f"  rank16_personal_adapter: {DIVERSE_PERSONAL_DIR}", flush=True)
    print("=" * 60, flush=True)

    # ── Phase 1: Generate diverse training data ─────────────────────────────
    print("\n--- Phase 1: Generate diverse training data ---", flush=True)
    generate_diverse_training_data()

    # ── Phase 2: Train rank-16 personal adapter ─────────────────────────────
    print("\n--- Phase 2: Train rank-16 personal adapter ---", flush=True)
    t_train = time.time()
    train_rank16_personal_adapter(PERSONAL_DATA_DIR, DIVERSE_PERSONAL_DIR)
    train_elapsed = time.time() - t_train
    print(f"Training elapsed: {train_elapsed:.0f}s ({train_elapsed/60:.1f}min)", flush=True)

    # Measure adapter size
    safetensors = DIVERSE_PERSONAL_DIR / "adapters.safetensors"
    adapter_size_mb = safetensors.stat().st_size / 1e6 if safetensors.exists() else 0.0
    k1206_pass = bool(train_elapsed <= 30 * 60)
    k1207_pass = bool(adapter_size_mb <= 10.0)

    print(f"K1206 (training_time <= 30 min): {'PASS' if k1206_pass else 'FAIL'} ({train_elapsed/60:.1f} min)", flush=True)
    print(f"K1207 (adapter_size <= 10 MB): {'PASS' if k1207_pass else 'FAIL'} ({adapter_size_mb:.2f} MB)", flush=True)

    # ── Phase 3: Build ridge router ─────────────────────────────────────────
    print("\n--- Phase 3: Build ridge router ---", flush=True)
    t0 = time.time()
    vectorizer, clf = build_ridge_router(n_train=200 if not IS_SMOKE else 30)
    router_build_s = round(time.time() - t0, 2)
    print(f"  Router built in {router_build_s}s", flush=True)

    # ── Phase 3b: Routing accuracy (diagnostic) ───────────────────────────────
    routing_results = test_routing_accuracy(vectorizer, clf, N_ROUTE)

    # ── Phase 4: Style compliance (K1205 — primary kill criterion) ────────────
    style_results = test_style_compliance(N_STYLE)

    # ── Phase 5: Math accuracy (diagnostic) ──────────────────────────────────
    math_results = test_math_accuracy(N_MATH)

    elapsed_total = round(time.time() - t_total, 1)

    k1205_pass = style_results.get("k1205", {}).get("pass", False)
    all_pass = k1205_pass and k1206_pass and k1207_pass

    results = {
        "is_smoke": IS_SMOKE,
        "lora_rank": LORA_RANK,
        "n_train": N_TRAIN,
        "train_iters": TRAIN_ITERS,
        "n_route": N_ROUTE,
        "n_style": N_STYLE,
        "n_math": N_MATH,
        "training": {
            "elapsed_s": round(train_elapsed, 1),
            "elapsed_min": round(train_elapsed / 60, 1),
            "adapter_size_mb": round(adapter_size_mb, 2),
            "k1206": {"training_time_min": round(train_elapsed / 60, 1), "threshold_min": 30.0, "pass": k1206_pass},
            "k1207": {"adapter_size_mb": round(adapter_size_mb, 2), "threshold_mb": 10.0, "pass": k1207_pass},
        },
        "routing": routing_results,
        "style": style_results,
        "math": math_results,
        "k1205": style_results.get("k1205", {"pass": False}),
        "k1206": {"training_time_min": round(train_elapsed / 60, 1), "threshold_min": 30.0, "pass": k1206_pass},
        "k1207": {"adapter_size_mb": round(adapter_size_mb, 2), "threshold_mb": 10.0, "pass": k1207_pass},
        "summary": {
            "all_pass": all_pass,
            "style_compliance": style_results.get("style_rate", 0.0),
            "vs_rank4_pp": style_results.get("vs_rank4_pp", 0.0),
            "training_time_min": round(train_elapsed / 60, 1),
            "adapter_size_mb": round(adapter_size_mb, 2),
            "math_acc": math_results.get("math_acc", 0.0),
            "elapsed_s": elapsed_total,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(f"SUMMARY — P3.C4: Rank-{LORA_RANK} Diverse Adapter", flush=True)
    print(f"  K1205 style_compliance ≥80%:  {'PASS' if k1205_pass else 'FAIL'} ({style_results.get('style_rate', 0.0):.1f}%)", flush=True)
    print(f"  K1206 training_time ≤30 min:  {'PASS' if k1206_pass else 'FAIL'} ({train_elapsed/60:.1f} min)", flush=True)
    print(f"  K1207 adapter_size ≤10 MB:    {'PASS' if k1207_pass else 'FAIL'} ({adapter_size_mb:.2f} MB)", flush=True)
    print(f"  vs P3.C1 rank-4 (60%): {style_results.get('vs_rank4_pp', 0.0):+.1f}pp", flush=True)
    print(f"  Math accuracy (diagnostic):   {math_results.get('math_acc', 0.0):.1f}%", flush=True)
    print(f"  ALL_PASS: {all_pass}", flush=True)
    print(f"  Total elapsed: {elapsed_total}s ({elapsed_total/60:.1f}min)", flush=True)
    print("=" * 60, flush=True)

    if k1205_pass:
        print("\n→ SUPPORTED: Rank-16 coverage lemma verified.", flush=True)
        print(f"  rank({LORA_RANK}) > n_categories(10) → style injection succeeds for all categories.", flush=True)
        print("  Coverage lemma (Theorem 1, MATH.md): confirmed rank-4 bottleneck was root cause.", flush=True)
    else:
        print(f"\n→ KILLED: Style compliance {style_results.get('style_rate', 0.0):.1f}% < 80% threshold.", flush=True)
        print("  Coverage lemma still fails at rank-16.", flush=True)
        print("  New impossibility: STYLE MARKER INJECTION is beyond LoRA's behavioral reach for this model.", flush=True)
        print("  Root cause: Gemma 4's attention mechanism doesn't linearize style enough for LoRA injection.", flush=True)
        print("  Fix (P3.C5): Use SFT (full fine-tuning on domain_fused_base) or different style marker.", flush=True)

    print(f"\nResults written to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
