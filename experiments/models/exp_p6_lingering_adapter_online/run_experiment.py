#!/usr/bin/env python3
"""
P6.A0: Lingering Adapter — Online LoRA Update from Conversation Turns.

arXiv:2411.13405 (PLUM) shows conversation turns can be augmented to QA pairs
for efficient adaptation. arXiv:2012.13255 shows fine-tuning has low intrinsic
dimensionality. We test whether 20 single-gradient-step online LoRA updates
encode project-specific knowledge.

Kill criteria:
  K1285: Adapter improves project-specific QA accuracy >= 20pp after 20 turns
  K1286: Background training latency < 1s per turn on M5 Pro
  K1287: Adapter does not degrade general quality (MMLU within 2pp of base)
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

# Memory safety (CODING_GUIDELINES)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 4
N_LORA_LAYERS = 8
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
LR = 1e-3
MAX_SEQ_LEN = 512
LORA_MODULES = ["self_attn.q_proj", "self_attn.o_proj"]
LORA_SCALE = 1.0


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
# SYNTHETIC PROJECT DATA — "ZephyrFlow" event processing system
# Facts the model must learn from conversation turns:
# 1. Name: ZephyrFlow           6. Event types: user_action, system_metric, alert, audit_log
# 2. Language: Python 3.12      7. Max event size: 256KB
# 3. Framework: FastAPI         8. Retention: 90 days raw, 1 year aggregates
# 4. Database: ClickHouse       9. Deployment: Fly.io
# 5. Cache: Redis with "zf:" prefix  10. CI: GitHub Actions, ruff + pytest + mypy
# ══════════════════════════════════════════════════════════════════════════════

# Training QA pairs — each covers 1-2 project facts
# These simulate PLUM-augmented conversation turns
TRAINING_QA = [
    ("What is the name of the event processing system?",
     "The system is called ZephyrFlow. It's a real-time event processing platform for analytics and monitoring."),
    ("What programming language does ZephyrFlow use?",
     "ZephyrFlow is built with Python 3.12, using the latest pattern matching and type hint features."),
    ("What web framework does ZephyrFlow use for its API?",
     "ZephyrFlow uses FastAPI for all HTTP endpoints, providing automatic OpenAPI documentation."),
    ("What database does ZephyrFlow use for storing events?",
     "ZephyrFlow uses ClickHouse as its primary analytical database for event storage and querying."),
    ("What caching system does ZephyrFlow use?",
     "ZephyrFlow uses Redis for caching, with all cache keys prefixed with 'zf:' for namespace isolation."),
    ("What types of events does ZephyrFlow handle?",
     "ZephyrFlow processes four event types: user_action, system_metric, alert, and audit_log."),
    ("What is the maximum event size in ZephyrFlow?",
     "The maximum event payload size in ZephyrFlow is 256KB. Events larger than this are rejected at ingestion."),
    ("How long does ZephyrFlow retain event data?",
     "ZephyrFlow retains raw events for 90 days and aggregated data for 1 year."),
    ("Where is ZephyrFlow deployed?",
     "ZephyrFlow is deployed on Fly.io using Docker containers, with automatic scaling across regions."),
    ("What CI/CD tools does ZephyrFlow use?",
     "ZephyrFlow uses GitHub Actions for CI with three checks: ruff for linting, pytest for tests, and mypy for type checking."),
    # Reinforcement turns (repeat key facts in new contexts)
    ("How does ZephyrFlow handle event validation?",
     "ZephyrFlow validates events at the FastAPI ingestion endpoint. It checks the event type is one of user_action, system_metric, alert, or audit_log, and that the payload size is under 256KB."),
    ("Tell me about ZephyrFlow's data pipeline.",
     "In ZephyrFlow, events flow from the FastAPI API layer into ClickHouse for permanent storage. Frequently accessed recent events are cached in Redis with the 'zf:' key prefix."),
    ("What Python version should I use for ZephyrFlow development?",
     "You need Python 3.12 for ZephyrFlow development. The codebase uses structural pattern matching and parameterized generics."),
    ("How do I run ZephyrFlow's test suite?",
     "ZephyrFlow's CI runs on GitHub Actions. Locally, run ruff for linting, pytest for unit and integration tests, and mypy for type checking. All three must pass."),
    ("What's ZephyrFlow's data retention policy?",
     "Raw event data in ClickHouse is retained for 90 days, after which it's downsampled into aggregates. Aggregated data is kept for 1 year before archival."),
    ("How do I deploy a new version of ZephyrFlow?",
     "Push to main triggers a GitHub Actions pipeline. After CI passes (ruff, pytest, mypy), a Docker image is built and deployed to Fly.io automatically."),
    ("What's the cache key format in ZephyrFlow?",
     "All Redis cache keys in ZephyrFlow use the prefix 'zf:' followed by the resource type and ID. For example, 'zf:event:12345' or 'zf:aggregate:daily:2024-01-15'."),
    ("Which event type is used for tracking errors?",
     "In ZephyrFlow, system errors generate events of type 'alert'. User interactions are 'user_action', infrastructure metrics are 'system_metric', and compliance records are 'audit_log'."),
    ("What's the tech stack summary for ZephyrFlow?",
     "ZephyrFlow: Python 3.12, FastAPI for HTTP, ClickHouse for analytics storage, Redis for caching (prefix 'zf:'), deployed on Fly.io with GitHub Actions CI."),
    ("How big can an event be in ZephyrFlow?",
     "Events in ZephyrFlow are capped at 256KB. The FastAPI endpoint returns a 413 Payload Too Large error for oversized events."),
]

# Test questions — keyword matching on generated text
# Each: (question, list_of_acceptable_keywords)
TEST_QUESTIONS = [
    ("What is the name of the event processing system?", ["ZephyrFlow"]),
    ("What programming language and version does ZephyrFlow use?", ["Python 3.12"]),
    ("What database does ZephyrFlow use for event storage?", ["ClickHouse"]),
    ("What is the Redis cache key prefix used in ZephyrFlow?", ["zf:"]),
    ("What cloud platform is ZephyrFlow deployed on?", ["Fly.io", "fly.io"]),
    ("What is the maximum event payload size in ZephyrFlow?", ["256KB", "256 KB", "256kb"]),
    ("How many days are raw events retained in ZephyrFlow?", ["90"]),
    ("What web framework does ZephyrFlow use for HTTP endpoints?", ["FastAPI", "fastapi"]),
    ("What linting tool does ZephyrFlow use in its CI pipeline?", ["ruff"]),
    ("What type checker does ZephyrFlow use alongside pytest?", ["mypy"]),
]

# General knowledge — keyword matching
GENERAL_QUESTIONS = [
    ("What is the chemical symbol for gold?", ["Au"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
    ("What is the capital of Japan?", ["Tokyo"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"]),
    ("What is the chemical formula for water?", ["H2O"]),
    ("What is the largest organ in the human body?", ["skin"]),
    ("In what year did World War II end?", ["1945"]),
    ("What is the square root of 144?", ["12"]),
    ("What gas do plants absorb for photosynthesis?", ["CO2", "carbon dioxide"]),
    ("What is the boiling point of water in Celsius?", ["100"]),
    ("Who painted the Mona Lisa?", ["Leonardo", "Da Vinci", "da Vinci"]),
    ("What is the smallest prime number?", ["2"]),
    ("On what continent is Egypt located?", ["Africa"]),
    ("What is the chemical formula for table salt?", ["NaCl"]),
    ("What is the tallest mountain on Earth?", ["Everest"]),
    ("How many chromosomes do humans have?", ["46"]),
    ("What element has atomic number 1?", ["Hydrogen", "hydrogen"]),
    ("What is the longest river in the world?", ["Nile"]),
    ("What force keeps planets in orbit?", ["gravity", "Gravity", "gravitational"]),
    ("What is the speed of light in km/s?", ["300,000", "300000", "3×10", "3x10"]),
]

if IS_SMOKE:
    TRAINING_QA = TRAINING_QA[:5]
    TEST_QUESTIONS = TEST_QUESTIONS[:3]
    GENERAL_QUESTIONS = GENERAL_QUESTIONS[:5]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — generation + keyword matching
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(model, tokenizer, prompt, max_tokens=60):
    """Simple autoregressive generation (no KV-cache)."""
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

    generated_tokens = all_tokens[n_prompt:]
    return tokenizer.decode(generated_tokens)


def evaluate_qa(model, tokenizer, questions):
    """Evaluate QA by generating responses and checking for keywords."""
    correct = 0
    details = []

    for question, keywords in questions:
        response = generate_response(
            model, tokenizer, question + " Answer briefly.", max_tokens=60
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
# PHASE 1: BASE EVALUATION (no adapter)
# ══════════════════════════════════════════════════════════════════════════════

def phase_base_eval(model_id):
    """Evaluate base model on project and general questions."""
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 1: BASE MODEL EVALUATION")
    log("=" * 60)

    model, tokenizer = load(model_id)
    log_memory("post-load")

    log("Evaluating on project questions (base)...")
    proj_acc, proj_correct, proj_total, proj_details = evaluate_qa(
        model, tokenizer, TEST_QUESTIONS
    )
    log(f"  Project accuracy: {proj_acc:.1%} ({proj_correct}/{proj_total})")

    log("Evaluating on general knowledge (base)...")
    gen_acc, gen_correct, gen_total, gen_details = evaluate_qa(
        model, tokenizer, GENERAL_QUESTIONS
    )
    log(f"  General accuracy: {gen_acc:.1%} ({gen_correct}/{gen_total})")

    results = {
        "project_accuracy": proj_acc,
        "project_correct": proj_correct,
        "project_total": proj_total,
        "project_details": proj_details,
        "general_accuracy": gen_acc,
        "general_correct": gen_correct,
        "general_total": gen_total,
        "general_details": gen_details,
    }

    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: ONLINE TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def apply_lora(model, lora_rank, lora_scale, n_lora_layers, lora_modules):
    """Apply LoRA to model backbone. Returns (backbone, n_layers, lora_count)."""
    from mlx_lm.tuner.lora import LoRALinear

    if hasattr(model, "language_model"):
        backbone = model.language_model.model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        backbone = model.model
    else:
        raise RuntimeError(f"Cannot find backbone in model: {type(model)}")

    n_layers = backbone.config.num_hidden_layers
    lora_start = max(0, n_layers - n_lora_layers)
    lora_count = 0

    for i in range(lora_start, n_layers):
        layer = backbone.layers[i]
        for module_name in lora_modules:
            parts = module_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(
                parent,
                parts[-1],
                LoRALinear.from_base(target, r=lora_rank, scale=lora_scale),
            )
            lora_count += 1

    return backbone, n_layers, lora_count


def phase_online_train_and_eval(model_id):
    """Apply LoRA, train online (1 step per turn), evaluate."""
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 2: ONLINE LORA TRAINING + EVALUATION")
    log("=" * 60)

    model, tokenizer = load(model_id)
    log_memory("post-load")

    # Freeze base model FIRST, then apply LoRA (LoRA params stay trainable)
    model.freeze()
    backbone, n_layers, lora_count = apply_lora(
        model, LORA_RANK, LORA_SCALE, N_LORA_LAYERS, LORA_MODULES
    )
    hidden_size = backbone.config.hidden_size
    lora_start = max(0, n_layers - N_LORA_LAYERS)

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    log(f"Model: hidden={hidden_size}, layers={n_layers}")
    log(f"LoRA: r={LORA_RANK}, layers {lora_start}-{n_layers-1}, modules={LORA_MODULES}")
    log(f"Trainable: {n_trainable:,} params in {lora_count} LoRA modules")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=LR)

    # Loss function: next-token prediction, masked to response tokens only
    def loss_fn(model, tokens, prompt_len):
        logits = model(tokens[:, :-1])
        # Only compute loss on response tokens
        response_logits = logits[:, prompt_len - 1 :, :]
        response_targets = tokens[:, prompt_len:]
        n_response = response_targets.shape[1]
        if n_response == 0:
            return mx.array(0.0)
        loss = nn.losses.cross_entropy(
            response_logits.reshape(-1, response_logits.shape[-1]),
            response_targets.reshape(-1),
            reduction="mean",
        )
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Online training: one gradient step per QA pair
    log(f"\nOnline training: {len(TRAINING_QA)} turns, 1 step each, lr={LR}")
    losses = []
    latencies = []

    gc.disable()
    for turn_idx, (question, answer) in enumerate(TRAINING_QA):
        # Format as chat
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
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_tokens)

        # Truncate if needed
        if len(full_tokens) > MAX_SEQ_LEN:
            full_tokens = full_tokens[:MAX_SEQ_LEN]

        tokens_mx = mx.array(full_tokens)[None, :]

        # One gradient step
        t_start = time.perf_counter()
        loss, grads = loss_and_grad_fn(model, tokens_mx, prompt_len)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        t_step = time.perf_counter() - t_start

        loss_val = loss.item()
        losses.append(loss_val)
        latencies.append(t_step * 1000)  # ms

        log(f"  Turn {turn_idx+1:2d}/{len(TRAINING_QA)}: "
            f"loss={loss_val:.4f} latency={t_step*1000:.0f}ms")
    gc.enable()

    avg_latency_ms = sum(latencies) / len(latencies)
    max_latency_ms = max(latencies)
    loss_decrease_pct = (
        (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0
    )
    log(f"\nTraining complete:")
    log(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({loss_decrease_pct:.1f}% decrease)")
    log(f"  Avg latency: {avg_latency_ms:.0f}ms, Max: {max_latency_ms:.0f}ms")
    log_memory("post-train")

    # Save adapter weights
    adapter_dir = EXPERIMENT_DIR / "lingering_adapter"
    adapter_dir.mkdir(exist_ok=True)
    adapter_path = adapter_dir / "weights.safetensors"
    trainable_dict = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_path), trainable_dict)
    adapter_size_mb = adapter_path.stat().st_size / (1024 * 1024)
    log(f"  Adapter saved: {adapter_size_mb:.2f} MB")

    # Evaluate adapted model
    log("\nEvaluating adapted model on project questions...")
    proj_acc, proj_correct, proj_total, proj_details = evaluate_qa(
        model, tokenizer, TEST_QUESTIONS
    )
    log(f"  Project accuracy: {proj_acc:.1%} ({proj_correct}/{proj_total})")

    log("Evaluating adapted model on general knowledge...")
    gen_acc, gen_correct, gen_total, gen_details = evaluate_qa(
        model, tokenizer, GENERAL_QUESTIONS
    )
    log(f"  General accuracy: {gen_acc:.1%} ({gen_correct}/{gen_total})")

    results = {
        "project_accuracy": proj_acc,
        "project_correct": proj_correct,
        "project_total": proj_total,
        "project_details": proj_details,
        "general_accuracy": gen_acc,
        "general_correct": gen_correct,
        "general_total": gen_total,
        "general_details": gen_details,
        "training": {
            "n_turns": len(TRAINING_QA),
            "losses": [round(l, 4) for l in losses],
            "loss_first": round(losses[0], 4),
            "loss_last": round(losses[-1], 4),
            "loss_decrease_pct": round(loss_decrease_pct, 1),
            "avg_latency_ms": round(avg_latency_ms, 1),
            "max_latency_ms": round(max_latency_ms, 1),
            "latencies_ms": [round(l, 1) for l in latencies],
            "lr": LR,
            "lora_rank": LORA_RANK,
            "n_lora_layers": N_LORA_LAYERS,
            "n_trainable_params": n_trainable,
            "adapter_size_mb": round(adapter_size_mb, 2),
        },
    }

    cleanup(model, tokenizer, optimizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")

    # Phase 1: base evaluation
    base_results = phase_base_eval(MODEL_ID)

    # Phase 2: online training + adapted evaluation
    adapted_results = phase_online_train_and_eval(MODEL_ID)

    # Compute kill criteria
    base_proj = base_results["project_accuracy"]
    adapted_proj = adapted_results["project_accuracy"]
    improvement_pp = (adapted_proj - base_proj) * 100

    base_gen = base_results["general_accuracy"]
    adapted_gen = adapted_results["general_accuracy"]
    degradation_pp = (base_gen - adapted_gen) * 100

    avg_latency = adapted_results["training"]["avg_latency_ms"]

    k1285 = {
        "pass": improvement_pp >= 20,
        "value_pp": round(improvement_pp, 1),
        "threshold_pp": 20,
        "base_accuracy": round(base_proj, 3),
        "adapted_accuracy": round(adapted_proj, 3),
    }
    k1286 = {
        "pass": avg_latency < 1000,
        "value_ms": round(avg_latency, 1),
        "threshold_ms": 1000,
    }
    k1287 = {
        "pass": degradation_pp < 2,
        "value_pp": round(degradation_pp, 1),
        "threshold_pp": 2,
        "base_accuracy": round(base_gen, 3),
        "adapted_accuracy": round(adapted_gen, 3),
    }

    all_pass = k1285["pass"] and k1286["pass"] and k1287["pass"]

    results = {
        "is_smoke": IS_SMOKE,
        "base": base_results,
        "adapted": adapted_results,
        "k1285": k1285,
        "k1286": k1286,
        "k1287": k1287,
        "all_pass": all_pass,
        "total_time_min": round((time.time() - t0) / 60, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"Base project acc:    {base_proj:.1%}")
    log(f"Adapted project acc: {adapted_proj:.1%}")
    log(f"Improvement:         {improvement_pp:+.1f}pp")
    log(f"")
    log(f"Base general acc:    {base_gen:.1%}")
    log(f"Adapted general acc: {adapted_gen:.1%}")
    log(f"Degradation:         {degradation_pp:+.1f}pp")
    log(f"")
    log(f"K1285 (proj acc >= 20pp):   {'PASS' if k1285['pass'] else 'FAIL'} — {improvement_pp:+.1f}pp")
    log(f"K1286 (latency < 1s):       {'PASS' if k1286['pass'] else 'FAIL'} — {avg_latency:.0f}ms")
    log(f"K1287 (gen acc within 2pp): {'PASS' if k1287['pass'] else 'FAIL'} — {degradation_pp:+.1f}pp")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {results['total_time_min']:.1f} min")
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
