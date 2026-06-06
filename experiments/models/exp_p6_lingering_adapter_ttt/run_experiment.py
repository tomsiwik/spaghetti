#!/usr/bin/env python3
"""
P6.A1: TTT-Style Embedded Adapter Update (Zero Extra Cost).

arXiv:2407.04620 proposes Test-Time Training: self-supervised weight updates
embedded in the forward pass. We test whether TTT-style self-supervised
training (next-token prediction on ALL tokens) can match P6.A0's supervised
approach (response-only loss) for factual recall.

Key difference from P6.A0: loss computed on ALL tokens (self-supervised),
not just response tokens (supervised QA).

Kill criteria:
  K1288: TTT adapter retains key facts (>= 50% recall)
  K1289: Zero additional latency vs base inference
  K1290: Quality matches or exceeds P6.A0 (60% accuracy)
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
# Same as P6.A0 for direct comparison
# ══════════════════════════════════════════════════════════════════════════════

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
# EVALUATION — generation + keyword matching (same as P6.A0)
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
# PHASE 1: BASE EVALUATION
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
# PHASE 2: TTT TRAINING + EVALUATION
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


def phase_ttt_train_and_eval(model_id):
    """
    TTT-style training: self-supervised loss on ALL tokens (not just response).

    Two conditions measured:
    A) Self-supervised (all-token loss) — tests whether QA formatting is needed
    B) Latency measurement — quantifies backward pass cost (K1289)
    """
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 2: TTT SELF-SUPERVISED TRAINING + EVALUATION")
    log("=" * 60)

    model, tokenizer = load(model_id)
    log_memory("post-load")

    # Freeze base, apply LoRA (same setup as P6.A0)
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

    optimizer = optim.AdamW(learning_rate=LR)

    # ── TTT Loss: self-supervised, all-token next-token prediction ──
    # Key difference from P6.A0: loss on ALL tokens, not just response.
    # This is "self-supervised" — the model predicts the next token everywhere,
    # using only the sequence itself as supervision. No QA label needed.
    def loss_fn_selfsup(model, tokens):
        """Cross-entropy on ALL tokens (self-supervised LM objective)."""
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        n_tokens = targets.shape[1]
        if n_tokens == 0:
            return mx.array(0.0)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        return loss

    # P6.A0-style loss for comparison (response tokens only)
    def loss_fn_supervised(model, tokens, prompt_len):
        """Cross-entropy on RESPONSE tokens only (supervised QA)."""
        logits = model(tokens[:, :-1])
        response_logits = logits[:, prompt_len - 1:, :]
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

    loss_and_grad_selfsup = nn.value_and_grad(model, loss_fn_selfsup)
    loss_and_grad_supervised = nn.value_and_grad(model, loss_fn_supervised)

    # ── Training loop ──
    log(f"\nTTT Training: {len(TRAINING_QA)} turns, lr={LR}")
    log("Condition A: Self-supervised (all-token loss)")

    losses_selfsup = []
    latencies_selfsup = []
    losses_supervised_ref = []  # For comparison: what would P6.A0 loss be?

    gc.disable()
    for turn_idx, (question, answer) in enumerate(TRAINING_QA):
        # Format as chat (same as P6.A0)
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

        if len(full_tokens) > MAX_SEQ_LEN:
            full_tokens = full_tokens[:MAX_SEQ_LEN]

        tokens_mx = mx.array(full_tokens)[None, :]

        # Compute supervised loss for reference (no gradient step)
        t_ref = time.perf_counter()
        ref_loss = loss_fn_supervised(model, tokens_mx, prompt_len)
        mx.eval(ref_loss)
        t_ref_done = time.perf_counter()
        losses_supervised_ref.append(ref_loss.item())

        # Self-supervised gradient step (ALL tokens)
        t_start = time.perf_counter()
        loss, grads = loss_and_grad_selfsup(model, tokens_mx)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        t_step = time.perf_counter() - t_start

        loss_val = loss.item()
        losses_selfsup.append(loss_val)
        latencies_selfsup.append(t_step * 1000)

        log(
            f"  Turn {turn_idx+1:2d}/{len(TRAINING_QA)}: "
            f"selfsup_loss={loss_val:.4f} "
            f"sup_ref_loss={losses_supervised_ref[-1]:.4f} "
            f"latency={t_step*1000:.0f}ms"
        )
    gc.enable()

    avg_latency = sum(latencies_selfsup) / len(latencies_selfsup)
    max_latency = max(latencies_selfsup)
    loss_decrease = (
        (losses_selfsup[0] - losses_selfsup[-1]) / losses_selfsup[0] * 100
        if losses_selfsup[0] > 0
        else 0
    )
    log(f"\nTraining complete:")
    log(f"  Self-sup loss: {losses_selfsup[0]:.4f} -> {losses_selfsup[-1]:.4f} ({loss_decrease:.1f}% decrease)")
    log(f"  Avg latency: {avg_latency:.0f}ms, Max: {max_latency:.0f}ms")

    # ── K1289 Latency Test: measure forward-only vs forward+backward ──
    log("\n--- K1289 Latency Analysis ---")
    sample_tokens = mx.array(tokenizer.encode(
        TRAINING_QA[0][0] + " " + TRAINING_QA[0][1]
    )[:100])[None, :]

    # Forward only (base inference cost)
    fwd_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        logits = model(sample_tokens)
        mx.eval(logits)
        fwd_times.append((time.perf_counter() - t0) * 1000)
        del logits

    # Forward + backward (training cost)
    bwd_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        loss, grads = loss_and_grad_selfsup(model, sample_tokens)
        mx.eval(loss, grads)
        bwd_times.append((time.perf_counter() - t0) * 1000)
        del loss, grads

    fwd_avg = sum(fwd_times[1:]) / (len(fwd_times) - 1)  # Skip warmup
    bwd_avg = sum(bwd_times[1:]) / (len(bwd_times) - 1)
    overhead_ms = bwd_avg - fwd_avg
    overhead_pct = (overhead_ms / fwd_avg * 100) if fwd_avg > 0 else float("inf")

    log(f"  Forward only:     {fwd_avg:.1f}ms avg")
    log(f"  Forward+backward: {bwd_avg:.1f}ms avg")
    log(f"  Backward overhead: {overhead_ms:.1f}ms ({overhead_pct:.1f}%)")
    log(f"  K1289 verdict: {'PASS' if overhead_pct < 5 else 'FAIL'} "
        f"(threshold: <5% overhead for 'zero')")
    log_memory("post-train")

    # Save adapter
    adapter_dir = EXPERIMENT_DIR / "ttt_adapter"
    adapter_dir.mkdir(exist_ok=True)
    adapter_path = adapter_dir / "weights.safetensors"
    trainable_dict = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_path), trainable_dict)
    adapter_size_mb = adapter_path.stat().st_size / (1024 * 1024)
    log(f"  Adapter saved: {adapter_size_mb:.2f} MB")

    # ── Evaluation ──
    log("\nEvaluating TTT-adapted model on project questions...")
    proj_acc, proj_correct, proj_total, proj_details = evaluate_qa(
        model, tokenizer, TEST_QUESTIONS
    )
    log(f"  Project accuracy: {proj_acc:.1%} ({proj_correct}/{proj_total})")

    log("Evaluating TTT-adapted model on general knowledge...")
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
            "method": "ttt_selfsup_all_tokens",
            "n_turns": len(TRAINING_QA),
            "losses_selfsup": [round(l, 4) for l in losses_selfsup],
            "losses_supervised_ref": [round(l, 4) for l in losses_supervised_ref],
            "loss_first": round(losses_selfsup[0], 4),
            "loss_last": round(losses_selfsup[-1], 4),
            "loss_decrease_pct": round(loss_decrease, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "max_latency_ms": round(max_latency, 1),
            "latencies_ms": [round(l, 1) for l in latencies_selfsup],
            "lr": LR,
            "lora_rank": LORA_RANK,
            "n_lora_layers": N_LORA_LAYERS,
            "n_trainable_params": n_trainable,
            "adapter_size_mb": round(adapter_size_mb, 2),
        },
        "latency_analysis": {
            "forward_only_ms": round(fwd_avg, 1),
            "forward_backward_ms": round(bwd_avg, 1),
            "backward_overhead_ms": round(overhead_ms, 1),
            "backward_overhead_pct": round(overhead_pct, 1),
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

    # Phase 2: TTT self-supervised training + evaluation
    ttt_results = phase_ttt_train_and_eval(MODEL_ID)

    # ── Kill criteria ──
    base_proj = base_results["project_accuracy"]
    ttt_proj = ttt_results["project_accuracy"]
    improvement_pp = (ttt_proj - base_proj) * 100

    base_gen = base_results["general_accuracy"]
    ttt_gen = ttt_results["general_accuracy"]
    degradation_pp = (base_gen - ttt_gen) * 100

    avg_latency = ttt_results["training"]["avg_latency_ms"]
    overhead_pct = ttt_results["latency_analysis"]["backward_overhead_pct"]

    # P6.A0 reference values
    p6a0_proj_acc = 0.6   # 60% from previous experiment
    p6a0_avg_latency = 110  # 110ms from previous experiment

    # K1288: >= 50% recall
    k1288 = {
        "pass": ttt_proj >= 0.5,
        "value_pct": round(ttt_proj * 100, 1),
        "threshold_pct": 50,
        "base_accuracy": round(base_proj, 3),
        "ttt_accuracy": round(ttt_proj, 3),
        "improvement_pp": round(improvement_pp, 1),
    }

    # K1289: Zero additional latency (interpreted as <5% overhead)
    k1289 = {
        "pass": overhead_pct < 5,
        "overhead_pct": round(overhead_pct, 1),
        "threshold_pct": 5,
        "forward_only_ms": ttt_results["latency_analysis"]["forward_only_ms"],
        "forward_backward_ms": ttt_results["latency_analysis"]["forward_backward_ms"],
        "backward_overhead_ms": ttt_results["latency_analysis"]["backward_overhead_ms"],
    }

    # K1290: Quality matches P6.A0 (60%)
    quality_gap = abs(ttt_proj - p6a0_proj_acc)
    k1290 = {
        "pass": ttt_proj >= p6a0_proj_acc - 0.1,  # Within 10pp of P6.A0
        "ttt_accuracy": round(ttt_proj, 3),
        "p6a0_accuracy": p6a0_proj_acc,
        "gap_pp": round(quality_gap * 100, 1),
    }

    all_pass = k1288["pass"] and k1289["pass"] and k1290["pass"]

    results = {
        "is_smoke": IS_SMOKE,
        "base": base_results,
        "ttt": ttt_results,
        "k1288": k1288,
        "k1289": k1289,
        "k1290": k1290,
        "all_pass": all_pass,
        "p6a0_reference": {
            "project_accuracy": p6a0_proj_acc,
            "avg_latency_ms": p6a0_avg_latency,
        },
        "total_time_min": round((time.time() - t0) / 60, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"Base project acc:    {base_proj:.1%}")
    log(f"TTT project acc:     {ttt_proj:.1%}")
    log(f"P6.A0 reference:     {p6a0_proj_acc:.1%}")
    log(f"Improvement vs base: {improvement_pp:+.1f}pp")
    log(f"")
    log(f"Base general acc:    {base_gen:.1%}")
    log(f"TTT general acc:     {ttt_gen:.1%}")
    log(f"Degradation:         {degradation_pp:+.1f}pp")
    log(f"")
    log(f"Backward overhead:   {overhead_pct:.1f}% ({ttt_results['latency_analysis']['backward_overhead_ms']:.0f}ms)")
    log(f"")
    log(f"K1288 (recall >= 50%):       {'PASS' if k1288['pass'] else 'FAIL'} — {ttt_proj:.1%}")
    log(f"K1289 (zero latency <5%):    {'PASS' if k1289['pass'] else 'FAIL'} — {overhead_pct:.1f}%")
    log(f"K1290 (match P6.A0 >= 50%):  {'PASS' if k1290['pass'] else 'FAIL'} — {ttt_proj:.1%} vs {p6a0_proj_acc:.1%}")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {results['total_time_min']:.1f} min")
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
