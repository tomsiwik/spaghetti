#!/usr/bin/env python3
"""
P6.B0: Adapter Sharing Flywheel — User Adapters to Domain Knowledge

Tests the full flywheel: 10 users train lingering adapters on partial domain
knowledge (6/10 facts each, sliding window). Adapters are crystallized (A+B
averaged) and used as initialization for new users. Measures: (1) crystal
outperforms individuals, (2) crystal-initialized new user beats zero-init.

Kill criteria:
  K1291: Crystallized adapter outperforms best individual by >= 5pp
  K1292: Crystal-initialized new user beats zero-init control by >= 3pp
  K1293: Full cycle completes in < 10 min

References:
  - P6.A0 (Finding #490): Online LoRA, 60% accuracy, 110ms/turn
  - T6.2 (Finding #451): B-matrix crystallization, cos=0.9806
  - T6.3: Promotion equivalence (cos=0.99999988)
  - Model Soup (arXiv:2203.05482): weight averaging
"""

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

# Memory safety (CODING_GUIDELINES)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

LORA_RANK = 4
N_LORA_LAYERS = 8
LORA_MODULES = ["self_attn.q_proj", "self_attn.o_proj"]
LORA_SCALE = 1.0
LR = 1e-3
MAX_SEQ_LEN = 512
SEED = 42

N_USERS = 10
FACTS_PER_USER = 6
N_FACTS = 10

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
if IS_SMOKE:
    N_USERS = 3
    FACTS_PER_USER = 4


# ══════════════════════════════════════════════════════════════════════════════
# DATA: 10 ZephyrFlow facts, 2 training QA pairs per fact, 10 test questions
# ══════════════════════════════════════════════════════════════════════════════

FACT_TRAINING_QA = {
    0: [  # ZephyrFlow name
        ("What is the name of the event processing system?",
         "The system is called ZephyrFlow. It's a real-time event processing platform for analytics and monitoring."),
        ("Tell me about our system.",
         "Our system is called ZephyrFlow. It processes events in real time for analytics."),
        ("What's the tech stack summary for ZephyrFlow?",
         "ZephyrFlow: a real-time event processing platform built for analytics and monitoring."),
    ],
    1: [  # Python 3.12
        ("What programming language does ZephyrFlow use?",
         "ZephyrFlow is built with Python 3.12, using the latest pattern matching and type hint features."),
        ("What programming language and version does ZephyrFlow use?",
         "ZephyrFlow uses Python 3.12 with structural pattern matching and parameterized generics."),
        ("What Python version should I use for ZephyrFlow development?",
         "You need Python 3.12 for ZephyrFlow development. The codebase uses structural pattern matching."),
    ],
    2: [  # FastAPI
        ("What web framework does ZephyrFlow use for its API?",
         "ZephyrFlow uses FastAPI for all HTTP endpoints, providing automatic OpenAPI documentation."),
        ("What web framework does ZephyrFlow use for HTTP endpoints?",
         "ZephyrFlow uses FastAPI for all HTTP endpoints with async support and automatic docs."),
        ("How does ZephyrFlow handle event validation?",
         "ZephyrFlow validates events at the FastAPI ingestion endpoint before storing them."),
    ],
    3: [  # ClickHouse
        ("What database does ZephyrFlow use for storing events?",
         "ZephyrFlow uses ClickHouse as its primary analytical database for event storage and querying."),
        ("What database does ZephyrFlow use for event storage?",
         "Events are stored in ClickHouse, a columnar analytical database optimized for fast queries."),
        ("Tell me about ZephyrFlow's data pipeline.",
         "In ZephyrFlow, events flow from the API layer into ClickHouse for permanent analytical storage."),
    ],
    4: [  # Redis with zf: prefix
        ("What caching system does ZephyrFlow use?",
         "ZephyrFlow uses Redis for caching, with all cache keys prefixed with 'zf:' for namespace isolation."),
        ("What is the Redis cache key prefix used in ZephyrFlow?",
         "All Redis cache keys in ZephyrFlow use the prefix 'zf:' followed by the resource type and ID."),
        ("What's the cache key format in ZephyrFlow?",
         "Cache keys use the 'zf:' prefix, like 'zf:event:12345' or 'zf:aggregate:daily:2024-01-15'."),
    ],
    5: [  # Event types
        ("What types of events does ZephyrFlow handle?",
         "ZephyrFlow processes four event types: user_action, system_metric, alert, and audit_log."),
        ("What types of events does ZephyrFlow process?",
         "ZephyrFlow processes user_action, system_metric, alert, and audit_log event types."),
        ("Which event type is used for tracking errors?",
         "In ZephyrFlow, system errors generate events of type 'alert'. User interactions are 'user_action'."),
    ],
    6: [  # 256KB max
        ("What is the maximum event size in ZephyrFlow?",
         "The maximum event payload size in ZephyrFlow is 256KB. Events larger than this are rejected at ingestion."),
        ("What is the maximum event payload size in ZephyrFlow?",
         "Events in ZephyrFlow are capped at 256KB. The API returns a 413 Payload Too Large error for oversized events."),
        ("How big can an event be in ZephyrFlow?",
         "Events in ZephyrFlow are capped at 256KB. Oversized events are rejected at the API gateway."),
    ],
    7: [  # 90 days retention
        ("How long does ZephyrFlow retain event data?",
         "ZephyrFlow retains raw events for 90 days and aggregated data for 1 year."),
        ("How many days are raw events retained in ZephyrFlow?",
         "Raw events are retained for 90 days in ZephyrFlow, then downsampled into aggregates kept for 1 year."),
        ("What's ZephyrFlow's data retention policy?",
         "Raw event data in ClickHouse is retained for 90 days. Aggregated data is kept for 1 year before archival."),
    ],
    8: [  # Fly.io
        ("Where is ZephyrFlow deployed?",
         "ZephyrFlow is deployed on Fly.io using Docker containers, with automatic scaling across regions."),
        ("What cloud platform is ZephyrFlow deployed on?",
         "ZephyrFlow is deployed on Fly.io with Docker containers and automatic regional scaling."),
        ("How do I deploy a new version of ZephyrFlow?",
         "Push to main triggers CI. After passing, a Docker image is built and deployed to Fly.io automatically."),
    ],
    9: [  # GitHub Actions + ruff + pytest + mypy
        ("What CI/CD tools does ZephyrFlow use?",
         "ZephyrFlow uses GitHub Actions for CI with three checks: ruff for linting, pytest for tests, and mypy for type checking."),
        ("What linting tool does ZephyrFlow use in CI?",
         "ZephyrFlow uses ruff for linting in its GitHub Actions CI pipeline, alongside pytest and mypy."),
        ("How do I run ZephyrFlow's test suite?",
         "Run ruff for linting, pytest for unit and integration tests, and mypy for type checking. All three must pass."),
    ],
}

TEST_QUESTIONS = [
    ("What is the name of the event processing system?", ["ZephyrFlow"]),
    ("What programming language and version does ZephyrFlow use?", ["Python 3.12"]),
    ("What web framework does ZephyrFlow use for HTTP endpoints?", ["FastAPI", "fastapi"]),
    ("What database does ZephyrFlow use for event storage?", ["ClickHouse", "clickhouse"]),
    ("What is the Redis cache key prefix used in ZephyrFlow?", ["zf:"]),
    ("What types of events does ZephyrFlow process?", ["user_action", "audit_log"]),
    ("What is the maximum event payload size in ZephyrFlow?", ["256KB", "256 KB", "256kb"]),
    ("How many days are raw events retained in ZephyrFlow?", ["90"]),
    ("What cloud platform is ZephyrFlow deployed on?", ["Fly.io", "fly.io"]),
    ("What linting tool does ZephyrFlow use in CI?", ["ruff"]),
]

if IS_SMOKE:
    TEST_QUESTIONS = TEST_QUESTIONS[:5]


def user_fact_ids(user_id):
    """Sliding window: user i gets facts [i, i+1, ..., i+K-1] mod N_FACTS."""
    return [(user_id + j) % N_FACTS for j in range(FACTS_PER_USER)]


def user_training_qa(user_id):
    """Build training QA pairs from user's known facts, padded to 20.
    With 3 QA per fact × 6 facts = 18, pad to 20 by repeating first 2."""
    facts = user_fact_ids(user_id)
    pairs = []
    for f in facts:
        pairs.extend(FACT_TRAINING_QA[f])
    target = 5 if IS_SMOKE else 20
    while len(pairs) < target:
        pairs.extend(pairs[: target - len(pairs)])
    return pairs[:target]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


def get_backbone(model):
    if hasattr(model, "language_model"):
        return model.language_model.model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError(f"Cannot find backbone: {type(model)}")


# ══════════════════════════════════════════════════════════════════════════════
# LoRA MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════


def apply_lora(model):
    """Apply rank-4 LoRA to last N_LORA_LAYERS. Returns lora_count."""
    from mlx_lm.tuner.lora import LoRALinear

    backbone = get_backbone(model)
    n_layers = backbone.config.num_hidden_layers
    lora_start = max(0, n_layers - N_LORA_LAYERS)
    count = 0
    for i in range(lora_start, n_layers):
        layer = backbone.layers[i]
        for module_name in LORA_MODULES:
            parts = module_name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            target = getattr(parent, parts[-1])
            setattr(
                parent,
                parts[-1],
                LoRALinear.from_base(target, r=LORA_RANK, scale=LORA_SCALE),
            )
            count += 1
    return count


def get_lora_modules(model):
    """Yield (key, lora_module) for all LoRA layers."""
    from mlx_lm.tuner.lora import LoRALinear

    backbone = get_backbone(model)
    n_layers = backbone.config.num_hidden_layers
    lora_start = max(0, n_layers - N_LORA_LAYERS)
    for i in range(lora_start, n_layers):
        layer = backbone.layers[i]
        for module_name in LORA_MODULES:
            parts = module_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)
            if isinstance(target, LoRALinear):
                key = f"layer_{i}_{module_name.replace('.', '_')}"
                yield key, target


def extract_matrices(model, which="b"):
    """Extract A or B matrices as numpy dict."""
    result = {}
    for key, mod in get_lora_modules(model):
        arr = mod.lora_a if which == "a" else mod.lora_b
        mx.eval(arr)
        result[key] = np.array(arr, dtype=np.float32)
    return result


def reset_lora_to_init(model, initial_a):
    """Reset LoRA: A to saved initial values, B to zeros."""
    for key, mod in get_lora_modules(model):
        mod.lora_a = mx.array(initial_a[key])
        mod.lora_b = mx.zeros_like(mod.lora_b)


def set_lora_ab(model, a_matrices, b_matrices):
    """Set both A and B in LoRA layers from numpy dicts."""
    for key, mod in get_lora_modules(model):
        mod.lora_a = mx.array(a_matrices[key])
        mod.lora_b = mx.array(b_matrices[key])
    mx.eval(model.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def generate_response(model, tokenizer, prompt, max_tokens=30):
    """Autoregressive generation (no KV-cache, simple)."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_tokens = tokenizer.encode(formatted)
    n_prompt = len(input_tokens)
    all_tokens = list(input_tokens)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_tokens):
        input_ids = mx.array(all_tokens)[None, :]
        logits = model(input_ids)
        mx.eval(logits)
        next_id = mx.argmax(logits[0, -1, :]).item()
        del logits
        if next_id == eos_id:
            break
        all_tokens.append(next_id)

    return tokenizer.decode(all_tokens[n_prompt:])


def evaluate_qa(model, tokenizer, questions):
    """Evaluate QA by keyword matching."""
    correct = 0
    details = []
    for question, keywords in questions:
        response = generate_response(
            model, tokenizer, question + " Answer briefly.", max_tokens=30
        )
        response_lower = response.lower()
        found = any(kw.lower() in response_lower for kw in keywords)
        if found:
            correct += 1
        details.append({
            "question": question,
            "response": response.strip()[:200],
            "keywords": keywords,
            "found": found,
        })
    accuracy = correct / len(questions) if questions else 0.0
    return accuracy, correct, len(questions), details


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HELPER
# ══════════════════════════════════════════════════════════════════════════════


def make_loss_fn():
    """Create response-only loss function."""
    def loss_fn(model, tokens, prompt_len):
        logits = model(tokens[:, :-1])
        response_logits = logits[:, prompt_len - 1 :, :]
        response_targets = tokens[:, prompt_len:]
        if response_targets.shape[1] == 0:
            return mx.array(0.0)
        return nn.losses.cross_entropy(
            response_logits.reshape(-1, response_logits.shape[-1]),
            response_targets.reshape(-1),
            reduction="mean",
        )
    return loss_fn


def train_one_user(model, tokenizer, qa_pairs, loss_and_grad_fn):
    """Train LoRA on QA pairs (1 gradient step per pair). Returns stats."""
    optimizer = optim.AdamW(learning_rate=LR)
    losses = []
    latencies = []

    gc.disable()
    for question, answer in qa_pairs:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_tokens = tokenizer.encode(full_text)
        prompt_len = len(tokenizer.encode(prompt_text))
        if len(full_tokens) > MAX_SEQ_LEN:
            full_tokens = full_tokens[:MAX_SEQ_LEN]
        tokens_mx = mx.array(full_tokens)[None, :]

        t0 = time.perf_counter()
        loss, grads = loss_and_grad_fn(model, tokens_mx, prompt_len)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        dt = time.perf_counter() - t0

        losses.append(loss.item())
        latencies.append(dt * 1000)
    gc.enable()

    del optimizer
    gc.collect()
    mx.clear_cache()

    return {
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BASE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def phase_base_eval(model_id):
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 1: BASE MODEL EVALUATION")
    log("=" * 60)

    model, tokenizer = load(model_id)
    log_memory("post-load")

    acc, correct, total, details = evaluate_qa(model, tokenizer, TEST_QUESTIONS)
    log(f"  Base accuracy: {acc:.1%} ({correct}/{total})")

    results = {"accuracy": acc, "correct": correct, "total": total, "details": details}
    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TRAIN N USERS (single model load, LoRA reset between users)
# ══════════════════════════════════════════════════════════════════════════════


def phase_train_users(model_id):
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log(f"PHASE 2: TRAIN {N_USERS} USERS (A+B LoRA, reset between users)")
    log("=" * 60)

    model, tokenizer = load(model_id)
    log_memory("post-load")

    # Freeze base, apply LoRA (both A and B trainable)
    model.freeze()
    lora_count = apply_lora(model)

    # Save initial A for resetting between users
    initial_a = extract_matrices(model, which="a")
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(str(ADAPTERS_DIR / "initial_a.npz"), **initial_a)

    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(f"  LoRA: r={LORA_RANK}, {lora_count} modules, {n_trainable:,} trainable (A+B)")

    loss_fn = make_loss_fn()
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    user_results = []
    for uid in range(N_USERS):
        facts = user_fact_ids(uid)
        log(f"\n--- User {uid} (facts: {facts}) ---")

        # Reset LoRA to initial state (same A, zero B)
        reset_lora_to_init(model, initial_a)

        qa = user_training_qa(uid)
        stats = train_one_user(model, tokenizer, qa, loss_and_grad_fn)
        log(f"  Train: {len(qa)} turns, {stats['avg_latency_ms']:.0f}ms/turn, "
            f"loss {stats['loss_first']:.3f}->{stats['loss_last']:.3f}")

        acc, correct, total, details = evaluate_qa(model, tokenizer, TEST_QUESTIONS)
        log(f"  Accuracy: {acc:.1%} ({correct}/{total})")

        # Save trained A and B to disk
        a_trained = extract_matrices(model, which="a")
        b_trained = extract_matrices(model, which="b")
        np.savez(str(ADAPTERS_DIR / f"user_{uid}_a.npz"), **a_trained)
        np.savez(str(ADAPTERS_DIR / f"user_{uid}_b.npz"), **b_trained)

        user_results.append({
            "user_id": uid,
            "facts": facts,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "details": details,
            **stats,
        })

    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return user_results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: CRYSTALLIZE + EVALUATE CRYSTAL + NEW USER ON CRYSTAL
# ══════════════════════════════════════════════════════════════════════════════


def phase_crystallize_and_new_user(model_id):
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 3: CRYSTALLIZE -> EVAL CRYSTAL -> NEW USER ON CRYSTAL")
    log("=" * 60)

    # ── Crystallize A and B (numpy, no model needed) ──
    log("Crystallizing A+B matrices from all users...")
    all_a = [dict(np.load(str(ADAPTERS_DIR / f"user_{uid}_a.npz"))) for uid in range(N_USERS)]
    all_b = [dict(np.load(str(ADAPTERS_DIR / f"user_{uid}_b.npz"))) for uid in range(N_USERS)]

    crystal_a = {}
    crystal_b = {}
    for key in all_a[0]:
        crystal_a[key] = np.mean(np.stack([a[key] for a in all_a], axis=0), axis=0)
        crystal_b[key] = np.mean(np.stack([b[key] for b in all_b], axis=0), axis=0)
        user_b_norms = [np.linalg.norm(b[key]) for b in all_b]
        log(f"  {key}: crystal_b_norm={np.linalg.norm(crystal_b[key]):.4f} "
            f"mean_user_b_norm={np.mean(user_b_norms):.4f}")

    np.savez(str(ADAPTERS_DIR / "crystal_a.npz"), **crystal_a)
    np.savez(str(ADAPTERS_DIR / "crystal_b.npz"), **crystal_b)

    # ── Load model, set LoRA to crystal, evaluate ──
    model, tokenizer = load(model_id)
    log_memory("post-load")

    model.freeze()
    apply_lora(model)
    set_lora_ab(model, crystal_a, crystal_b)

    log("Evaluating crystal adapter...")
    crystal_acc, crystal_correct, crystal_total, crystal_details = evaluate_qa(
        model, tokenizer, TEST_QUESTIONS
    )
    log(f"  Crystal accuracy: {crystal_acc:.1%} ({crystal_correct}/{crystal_total})")

    # ── New user: initialize from crystal, then train ──
    # This is equivalent to training on a promoted base (Theorem 3 in MATH.md)
    new_user_qa = user_training_qa(0)  # Same facts as User 0
    log(f"\nTraining new user from crystal init (facts: {user_fact_ids(0)})...")

    # Re-set to crystal (training above didn't change it, but be explicit)
    set_lora_ab(model, crystal_a, crystal_b)

    loss_fn = make_loss_fn()
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    stats = train_one_user(model, tokenizer, new_user_qa, loss_and_grad_fn)
    log(f"  Train: loss {stats['loss_first']:.3f}->{stats['loss_last']:.3f}")

    new_acc, new_correct, new_total, new_details = evaluate_qa(
        model, tokenizer, TEST_QUESTIONS
    )
    log(f"  New user (crystal-init): {new_acc:.1%} ({new_correct}/{new_total})")

    results = {
        "crystal": {
            "accuracy": crystal_acc,
            "correct": crystal_correct,
            "total": crystal_total,
            "details": crystal_details,
        },
        "new_user_crystal_init": {
            "accuracy": new_acc,
            "correct": new_correct,
            "total": new_total,
            "details": new_details,
            **stats,
        },
    }

    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: CONTROL USER (standard init, same facts as new user)
# ══════════════════════════════════════════════════════════════════════════════


def phase_control_user(model_id):
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 4: CONTROL USER (standard init)")
    log("=" * 60)

    model, tokenizer = load(model_id)
    log_memory("post-load")

    model.freeze()
    apply_lora(model)

    # Use same initial A as Phase 2 users
    initial_a = dict(np.load(str(ADAPTERS_DIR / "initial_a.npz")))
    reset_lora_to_init(model, initial_a)

    loss_fn = make_loss_fn()
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    control_qa = user_training_qa(0)  # Same facts as new user
    log(f"Training control user from standard init (facts: {user_fact_ids(0)})...")
    stats = train_one_user(model, tokenizer, control_qa, loss_and_grad_fn)
    log(f"  Train: loss {stats['loss_first']:.3f}->{stats['loss_last']:.3f}")

    acc, correct, total, details = evaluate_qa(model, tokenizer, TEST_QUESTIONS)
    log(f"  Control user accuracy: {acc:.1%} ({correct}/{total})")

    results = {"accuracy": acc, "correct": correct, "total": total, "details": details, **stats}

    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")

    base_results = phase_base_eval(MODEL_ID)
    user_results = phase_train_users(MODEL_ID)
    crystal_results = phase_crystallize_and_new_user(MODEL_ID)
    control_results = phase_control_user(MODEL_ID)

    total_time_min = (time.time() - t0) / 60

    # ── Kill Criteria ──
    best_individual_acc = max(r["accuracy"] for r in user_results)
    best_uid = max(range(N_USERS), key=lambda i: user_results[i]["accuracy"])
    crystal_acc = crystal_results["crystal"]["accuracy"]
    new_user_acc = crystal_results["new_user_crystal_init"]["accuracy"]
    control_acc = control_results["accuracy"]

    k1291_margin = (crystal_acc - best_individual_acc) * 100
    k1292_margin = (new_user_acc - control_acc) * 100

    k1291_pass = k1291_margin >= 5
    k1292_pass = k1292_margin >= 3
    k1293_pass = total_time_min < 10
    all_pass = k1291_pass and k1292_pass and k1293_pass

    results = {
        "is_smoke": IS_SMOKE,
        "config": {
            "n_users": N_USERS,
            "facts_per_user": FACTS_PER_USER,
            "n_facts": N_FACTS,
            "lora_rank": LORA_RANK,
            "n_lora_layers": N_LORA_LAYERS,
            "lr": LR,
        },
        "base": base_results,
        "users": user_results,
        "crystal": crystal_results,
        "control": control_results,
        "kill_criteria": {
            "K1291": {
                "pass": k1291_pass,
                "crystal_accuracy": round(crystal_acc, 3),
                "best_individual_accuracy": round(best_individual_acc, 3),
                "best_user_id": best_uid,
                "margin_pp": round(k1291_margin, 1),
                "threshold_pp": 5,
            },
            "K1292": {
                "pass": k1292_pass,
                "new_user_crystal_init_accuracy": round(new_user_acc, 3),
                "control_accuracy": round(control_acc, 3),
                "margin_pp": round(k1292_margin, 1),
                "threshold_pp": 3,
            },
            "K1293": {
                "pass": k1293_pass,
                "total_time_min": round(total_time_min, 2),
                "threshold_min": 10,
            },
        },
        "all_pass": all_pass,
        "total_time_min": round(total_time_min, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log(f"\n{'=' * 60}")
    log("RESULTS SUMMARY")
    log(f"{'=' * 60}")
    log(f"Base accuracy:             {base_results['accuracy']:.1%}")
    log(f"Best individual:           {best_individual_acc:.1%} (User {best_uid})")
    log(f"Crystal adapter:           {crystal_acc:.1%}")
    log(f"New user (crystal-init):   {new_user_acc:.1%}")
    log(f"Control user (zero-init):  {control_acc:.1%}")
    log("")
    log(f"K1291 (crystal >= best+5pp):   {'PASS' if k1291_pass else 'FAIL'} -- margin={k1291_margin:+.1f}pp")
    log(f"K1292 (crystal >= ctrl+3pp):    {'PASS' if k1292_pass else 'FAIL'} -- margin={k1292_margin:+.1f}pp")
    log(f"K1293 (time < 10 min):         {'PASS' if k1293_pass else 'FAIL'} -- {total_time_min:.1f} min")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {total_time_min:.1f} min")
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
