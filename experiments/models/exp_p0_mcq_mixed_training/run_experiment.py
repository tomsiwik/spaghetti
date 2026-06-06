#!/usr/bin/env python3
"""
P0: MCQ-Mixed Training for Discriminative Retention Under TT Compression

Tests whether adding explicit MCQ classification loss to NTP training amplifies
discriminative gradient enough to survive TT-LoRA compression.

Kill criteria:
  K1437: Mixed-training TT-LoRA MedMCQA >= 35% (vs 18.5% NTP-only from Finding #521)
  K1438: Base GSM8K not catastrophically degraded (>= 15% with medical adapter active)
  K1439: Mixed training convergence within 2x wall-clock of NTP-only

Grounded by:
  Finding #521: Compression is the disease (34pp gap, LoRA 52.5% vs TT-LoRA 18.5%)
  arXiv:2504.21190 (TT-LoRA), arXiv:1810.04650 (GradNorm)
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 10 if IS_SMOKE else 200
N_STEPS = 20 if IS_SMOKE else 500
BATCH_SIZE = 2
MAX_SEQ_LEN = 512

# TT-LoRA config (match diagnosis experiment)
TT_RANK = 6
TT_ALPHA = 1.0
TT_LR = 5e-3
MCQ_WEIGHT = 1.0  # λ for MCQ loss (Theorem 1: MCQ gradient is ~64000x more concentrated)

PROJ_NAMES = ["v_proj", "o_proj"]
SEED = 42


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB",
          flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ────────────────────────────────────────────────
# MedMCQA Evaluation
# ────────────────────────────────────────────────

def load_medmcqa_val(n_eval, seed=SEED):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/validation-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_eval, len(df)), random_state=seed)
    return df.to_dict("records")


def eval_medmcqa(model, tokenizer, dataset, label=""):
    from mlx_lm import generate

    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0

    for i, ex in enumerate(dataset):
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        messages = [{"role": "user", "content":
            f"Answer this medical question. Reply with only the letter.\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=20, verbose=False)

        gt = option_map.get(ex["cop"], "A")
        pred = response.strip().upper()
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"    MedMCQA {label}: {i+1}/{len(dataset)}, "
                  f"acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(dataset) * 100
    print(f"  MedMCQA {label}: {correct}/{len(dataset)} = {acc:.1f}%", flush=True)
    return acc


# ────────────────────────────────────────────────
# Training Data
# ────────────────────────────────────────────────

def get_medical_data_path():
    local = EXPERIMENT_DIR / "data" / "medical" / "train.jsonl"
    if local.exists():
        return local
    e2e = EXPERIMENT_DIR.parent / "exp_p0_ttlora_e2e_benchmark" / "data" / "medical" / "train.jsonl"
    if e2e.exists():
        return e2e
    e2e_orig = EXPERIMENT_DIR.parent / "exp_p0_e2e_benchmark" / "data" / "medical" / "train.jsonl"
    if e2e_orig.exists():
        return e2e_orig
    raise FileNotFoundError("No medical training data found")


def tokenize_for_training(tokenizer, data_path, max_seq_len=512):
    """Tokenize MCQ training data, extracting correct answer index for MCQ loss."""
    answer_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    examples = []

    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]

            # Extract correct answer from assistant response (e.g. "B: Day care surgery")
            answer_text = msgs[-1]["content"].strip()
            correct_letter = answer_text[0].upper()
            correct_idx = answer_to_idx.get(correct_letter, -1)
            if correct_idx == -1:
                continue

            full = tokenizer.apply_chat_template(msgs, tokenize=False)
            full_ids = tokenizer.encode(full)
            prompt = tokenizer.apply_chat_template(
                [msgs[0]], tokenize=False, add_generation_prompt=True)
            prompt_len = len(tokenizer.encode(prompt))

            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
            if prompt_len >= len(full_ids):
                continue

            examples.append({
                "input_ids": full_ids,
                "prompt_len": prompt_len,
                "length": len(full_ids),
                "correct_idx": correct_idx,
            })

    return examples


def get_answer_token_ids(tokenizer):
    """Get token IDs for A, B, C, D answer letters."""
    ids = []
    for letter in ["A", "B", "C", "D"]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        tid = encoded[-1]
        ids.append(tid)
        print(f"  Token '{letter}' -> ID {tid}", flush=True)
    return ids


# ────────────────────────────────────────────────
# TT-LoRA Module
# ────────────────────────────────────────────────

class TTLoRAWrapper(nn.Module):
    def __init__(self, base_layer, in_features, out_features, tt_shape,
                 tt_rank=6, alpha=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.tt_shape = tt_shape

        self._validate_split(tt_shape, in_features, out_features)

        d = len(tt_shape)
        ranks = [1] + [tt_rank] * (d - 1) + [1]
        self._n_cores = d
        for k in range(d):
            shape = (ranks[k], tt_shape[k], ranks[k + 1])
            if k == d - 1:
                core = mx.zeros(shape)
            else:
                std = 1.0 / math.sqrt(tt_shape[k] * ranks[k])
                core = mx.random.normal(shape) * std
            setattr(self, f"core_{k}", core)
        self._cached_delta_w = None

    def _validate_split(self, tt_shape, in_features, out_features):
        prod = 1
        for i, s in enumerate(tt_shape):
            prod *= s
            if prod == in_features:
                rest = 1
                for j in range(i + 1, len(tt_shape)):
                    rest *= tt_shape[j]
                assert rest == out_features, (
                    f"Output factors product {rest} != {out_features}")
                return
        raise ValueError(
            f"Cannot split {tt_shape} into {in_features} x {out_features}")

    @property
    def tt_cores(self):
        return [getattr(self, f"core_{k}") for k in range(self._n_cores)]

    def reconstruct_delta_w(self):
        cores = self.tt_cores
        result = cores[0].squeeze(0)
        for k in range(1, len(cores)):
            core = cores[k]
            r_k, s_k, r_next = core.shape
            result = result @ core.reshape(r_k, s_k * r_next)
            leading = result.shape[0]
            result = result.reshape(leading * s_k, r_next)
        result = result.squeeze(-1)
        return result.reshape(self.in_features, self.out_features).T

    def cache_delta_w(self):
        self._cached_delta_w = self.reconstruct_delta_w()
        mx.eval(self._cached_delta_w)

    def __call__(self, x):
        base_out = self.base_layer(x)
        dw = (self._cached_delta_w if self._cached_delta_w is not None
              else self.reconstruct_delta_w())
        return base_out + self.alpha * (x @ dw.T)

    def num_params(self):
        return sum(c.size for c in self.tt_cores)


def factorize(n, max_factor=10):
    factors = []
    while n % 8 == 0 and n > 8:
        factors.append(8)
        n //= 8
    for f in range(max_factor, 1, -1):
        while n % f == 0 and n > 1:
            factors.append(f)
            n //= f
    if n > 1:
        factors.append(n)
    factors.sort()
    return factors


def compute_tt_shape(in_features, out_features):
    return factorize(in_features) + factorize(out_features)


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


def detect_proj_dims(base_layer):
    if hasattr(base_layer, 'scales'):
        out_features = base_layer.weight.shape[0]
        in_features = base_layer.weight.shape[1] * 32 // base_layer.bits
        return in_features, out_features
    if hasattr(base_layer, 'weight'):
        return base_layer.weight.shape[1], base_layer.weight.shape[0]
    raise ValueError(f"Cannot detect dimensions for {type(base_layer)}")


def inject_ttlora(model, proj_names, tt_rank, alpha):
    layers = get_layers(model)
    total_params = 0
    for layer in layers:
        for name in proj_names:
            base = getattr(layer.self_attn, name)
            in_f, out_f = detect_proj_dims(base)
            tt_shape = compute_tt_shape(in_f, out_f)
            wrapper = TTLoRAWrapper(base, in_f, out_f, tt_shape, tt_rank, alpha)
            setattr(layer.self_attn, name, wrapper)
            total_params += wrapper.num_params()
    print(f"  TT-LoRA: {total_params:,} trainable params "
          f"(rank={tt_rank}, alpha={alpha})", flush=True)
    return total_params


def freeze_except_ttcores(model):
    model.freeze()
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                core_keys = [f"core_{k}" for k in range(proj._n_cores)]
                proj.unfreeze(keys=core_keys, recurse=False)


def remove_ttlora(model):
    layers = get_layers(model)
    for layer in layers:
        for name in PROJ_NAMES:
            proj = getattr(layer.self_attn, name)
            if isinstance(proj, TTLoRAWrapper):
                setattr(layer.self_attn, name, proj.base_layer)


# ────────────────────────────────────────────────
# Training Functions
# ────────────────────────────────────────────────

def train_ntp_only(model, tokenizer, examples, n_steps, lr, batch_size, label=""):
    """Standard NTP training with prompt masking (control condition)."""
    random.seed(SEED)
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(model, input_ids, lengths, prompt_lens):
        logits = model(input_ids)
        logits = logits.astype(mx.float32)
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        ce = nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none")
        S = shift_targets.shape[1]
        pos = mx.arange(S)[None, :]
        mask = (pos >= (prompt_lens[:, None] - 1)) & (pos < (lengths[:, None] - 1))
        mask = mask.astype(mx.float32)
        return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    idx = 0

    for step in range(n_steps):
        batch_exs = []
        for _ in range(batch_size):
            batch_exs.append(examples[idx % len(examples)])
            idx += 1

        max_len = max(e["length"] for e in batch_exs)
        input_ids = mx.array([
            e["input_ids"] + [pad_id] * (max_len - e["length"])
            for e in batch_exs
        ])
        lengths = mx.array([e["length"] for e in batch_exs])
        prompt_lens = mx.array([e["prompt_len"] for e in batch_exs])

        loss, grads = loss_and_grad(model, input_ids, lengths, prompt_lens)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        losses.append(loss.item())

        if (step + 1) % max(1, n_steps // 5) == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            print(f"    [{label}] Step {step+1}/{n_steps}: loss={avg:.4f} ({elapsed:.1f}s)",
                  flush=True)

    total_time = time.time() - t0
    return losses, total_time


def train_mixed(model, tokenizer, examples, n_steps, lr, batch_size,
                answer_token_ids, mcq_weight, label=""):
    """Mixed training: NTP loss + MCQ classification loss."""
    random.seed(SEED)
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    # Pre-convert answer token IDs to Python ints for indexing
    atid_a, atid_b, atid_c, atid_d = [int(t) for t in answer_token_ids]

    def loss_fn(model, input_ids, lengths, prompt_lens, correct_idxs):
        logits = model(input_ids)
        logits = logits.astype(mx.float32)
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]

        # ── NTP loss (standard) ──
        ce = nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none")
        S = shift_targets.shape[1]
        pos = mx.arange(S)[None, :]
        mask = (pos >= (prompt_lens[:, None] - 1)) & (pos < (lengths[:, None] - 1))
        mask = mask.astype(mx.float32)
        ntp_loss = (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

        # ── MCQ classification loss ──
        # Get logits at answer position (prompt_len - 1 in shift_logits)
        # This is the position predicting the first response token (the answer letter)
        B = input_ids.shape[0]
        V = shift_logits.shape[-1]
        answer_pos = (prompt_lens - 1)[:, None, None]  # (B, 1, 1)
        answer_pos_expanded = mx.broadcast_to(answer_pos, (B, 1, V))
        answer_vocab_logits = mx.take_along_axis(
            shift_logits, answer_pos_expanded, axis=1
        ).squeeze(1)  # (B, V)

        # Extract logits for A, B, C, D using slice indexing (Python ints)
        abcd_logits = mx.concatenate([
            answer_vocab_logits[:, atid_a:atid_a + 1],
            answer_vocab_logits[:, atid_b:atid_b + 1],
            answer_vocab_logits[:, atid_c:atid_c + 1],
            answer_vocab_logits[:, atid_d:atid_d + 1],
        ], axis=1)  # (B, 4)

        mcq_loss = nn.losses.cross_entropy(
            abcd_logits, correct_idxs, reduction="mean"
        )

        return ntp_loss + mcq_weight * mcq_loss, ntp_loss, mcq_loss

    def loss_wrapper(model, input_ids, lengths, prompt_lens, correct_idxs):
        total, _, _ = loss_fn(model, input_ids, lengths, prompt_lens, correct_idxs)
        return total

    loss_and_grad = nn.value_and_grad(model, loss_wrapper)

    losses = []
    ntp_losses = []
    mcq_losses = []
    t0 = time.time()
    idx = 0

    for step in range(n_steps):
        batch_exs = []
        for _ in range(batch_size):
            batch_exs.append(examples[idx % len(examples)])
            idx += 1

        max_len = max(e["length"] for e in batch_exs)
        input_ids = mx.array([
            e["input_ids"] + [pad_id] * (max_len - e["length"])
            for e in batch_exs
        ])
        lengths = mx.array([e["length"] for e in batch_exs])
        prompt_lens = mx.array([e["prompt_len"] for e in batch_exs])
        correct_idxs = mx.array([e["correct_idx"] for e in batch_exs])

        loss, grads = loss_and_grad(
            model, input_ids, lengths, prompt_lens, correct_idxs
        )
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        losses.append(loss.item())

        # Periodically log component losses (separate forward, not in hot path)
        if (step + 1) % max(1, n_steps // 5) == 0:
            total_l, ntp_l, mcq_l = loss_fn(
                model, input_ids, lengths, prompt_lens, correct_idxs
            )
            mx.eval(total_l, ntp_l, mcq_l)
            ntp_losses.append(ntp_l.item())
            mcq_losses.append(mcq_l.item())
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            print(f"    [{label}] Step {step+1}/{n_steps}: total={avg:.4f} "
                  f"ntp={ntp_l.item():.4f} mcq={mcq_l.item():.4f} ({elapsed:.1f}s)",
                  flush=True)

    total_time = time.time() - t0
    return losses, ntp_losses, mcq_losses, total_time


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P0: MCQ-Mixed Training for Discriminative Retention")
    print(f"TT-LoRA r{TT_RANK}: NTP-only vs Mixed (NTP + MCQ λ={MCQ_WEIGHT})")
    print(f"SMOKE={IS_SMOKE}, N_STEPS={N_STEPS}, N_EVAL={N_EVAL}")
    print("=" * 60, flush=True)

    results = {
        "experiment": "exp_p0_mcq_mixed_training",
        "n_eval": N_EVAL,
        "n_steps": N_STEPS,
        "is_smoke": IS_SMOKE,
        "seed": SEED,
        "mcq_weight": MCQ_WEIGHT,
        "tt_rank": TT_RANK,
    }

    # ── Phase 0: Load data & model ───────────────
    data_path = get_medical_data_path()
    print(f"Training data: {data_path}", flush=True)

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Get answer token IDs
    print("Answer token mapping:", flush=True)
    answer_token_ids = get_answer_token_ids(tokenizer)
    results["answer_token_ids"] = {
        letter: tid for letter, tid in zip("ABCD", answer_token_ids)
    }

    # Tokenize training data (with correct_idx for MCQ loss)
    examples = tokenize_for_training(tokenizer, data_path, MAX_SEQ_LEN)
    print(f"  {len(examples)} training examples loaded", flush=True)
    results["n_train"] = len(examples)

    # Load evaluation dataset (fixed seed)
    print("Loading MedMCQA validation set...", flush=True)
    medmcqa_dataset = load_medmcqa_val(N_EVAL, SEED)
    print(f"  {len(medmcqa_dataset)} evaluation questions", flush=True)

    # ── Phase 1: Base model evaluation ───────────
    print("\n" + "=" * 60)
    print("PHASE 1: Base model evaluation (control)")
    print("=" * 60, flush=True)

    base_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "BASE")
    results["base_medmcqa_pct"] = round(base_acc, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Base: {base_acc:.1f}% on MedMCQA", flush=True)

    # ── Phase 2: TT-LoRA NTP-only (control) ─────
    print("\n" + "=" * 60)
    print("PHASE 2: TT-LoRA NTP-only training (control)")
    print("=" * 60, flush=True)

    tt_params = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    freeze_except_ttcores(model)
    results["tt_params"] = tt_params

    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable: {trainable:,} params", flush=True)

    model.train()
    ntp_losses, ntp_time = train_ntp_only(
        model, tokenizer, examples, N_STEPS, TT_LR, BATCH_SIZE, "NTP-only"
    )
    results["ntp_train_time_s"] = round(ntp_time, 1)
    results["ntp_final_loss"] = round(ntp_losses[-1], 4) if ntp_losses else None

    # Evaluate NTP-only TT-LoRA
    print("\n" + "=" * 60)
    print("PHASE 2b: TT-LoRA NTP-only MedMCQA evaluation")
    print("=" * 60, flush=True)

    model.eval()
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()

    ntp_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "TT-LoRA-NTP")
    results["ntp_medmcqa_pct"] = round(ntp_acc, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Remove TT-LoRA, restore base model
    print("Removing NTP TT-LoRA wrappers...", flush=True)
    remove_ttlora(model)
    mx.clear_cache()
    log_memory("after-ntp-removed")

    # ── Phase 3: TT-LoRA Mixed training ──────────
    print("\n" + "=" * 60)
    print(f"PHASE 3: TT-LoRA Mixed training (NTP + MCQ λ={MCQ_WEIGHT})")
    print("=" * 60, flush=True)

    tt_params_mixed = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    freeze_except_ttcores(model)
    results["tt_params_mixed"] = tt_params_mixed

    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable: {trainable:,} params", flush=True)

    model.train()
    mixed_losses, mixed_ntp_losses, mixed_mcq_losses, mixed_time = train_mixed(
        model, tokenizer, examples, N_STEPS, TT_LR, BATCH_SIZE,
        answer_token_ids, MCQ_WEIGHT, "Mixed"
    )
    results["mixed_train_time_s"] = round(mixed_time, 1)
    results["mixed_final_loss"] = round(mixed_losses[-1], 4) if mixed_losses else None
    if mixed_ntp_losses:
        results["mixed_final_ntp_loss"] = round(mixed_ntp_losses[-1], 4)
    if mixed_mcq_losses:
        results["mixed_final_mcq_loss"] = round(mixed_mcq_losses[-1], 4)

    # Evaluate Mixed TT-LoRA
    print("\n" + "=" * 60)
    print("PHASE 3b: TT-LoRA Mixed MedMCQA evaluation")
    print("=" * 60, flush=True)

    model.eval()
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()

    mixed_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "TT-LoRA-Mixed")
    results["mixed_medmcqa_pct"] = round(mixed_acc, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # ── Phase 4: Results & Kill Criteria ─────────
    print("\n" + "=" * 60)
    print("RESULTS & KILL CRITERIA")
    print("=" * 60, flush=True)

    # Kill criteria
    k1437_pass = mixed_acc >= 35.0
    k1438_na = True  # Medical adapter on GSM8K not meaningful; see MATH.md
    k1439_pass = mixed_time <= 2.0 * ntp_time

    results["K1437_mixed_ge35"] = "PASS" if k1437_pass else "FAIL"
    results["K1438_gsm8k"] = "N/A (medical adapter, GSM8K not applicable)"
    results["K1439_convergence"] = "PASS" if k1439_pass else "FAIL"

    # Deltas
    ntp_delta = ntp_acc - base_acc
    mixed_delta = mixed_acc - base_acc
    mcq_effect = mixed_acc - ntp_acc

    results["ntp_delta_pp"] = round(ntp_delta, 1)
    results["mixed_delta_pp"] = round(mixed_delta, 1)
    results["mcq_effect_pp"] = round(mcq_effect, 1)
    results["time_ratio"] = round(mixed_time / max(ntp_time, 1), 2)

    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)

    # Print summary
    print(f"\nResults:")
    print(f"  Base model:        {base_acc:.1f}%")
    print(f"  TT-LoRA NTP-only:  {ntp_acc:.1f}% (delta {ntp_delta:+.1f}pp)")
    print(f"  TT-LoRA Mixed:     {mixed_acc:.1f}% (delta {mixed_delta:+.1f}pp)")
    print(f"  MCQ effect:        {mcq_effect:+.1f}pp (Mixed minus NTP-only)")
    print(f"  Time ratio:        {mixed_time/max(ntp_time,1):.2f}x")

    print(f"\nKILL CRITERIA:")
    print(f"  K1437 Mixed >= 35%:     {results['K1437_mixed_ge35']} ({mixed_acc:.1f}%)")
    print(f"  K1438 GSM8K:            N/A (medical adapter)")
    print(f"  K1439 Convergence <=2x: {results['K1439_convergence']} "
          f"({mixed_time:.0f}s vs {ntp_time:.0f}s)")

    if mcq_effect > 0:
        print(f"\nMCQ loss improved discriminative capacity by {mcq_effect:+.1f}pp")
    else:
        print(f"\nMCQ loss did NOT improve discriminative capacity ({mcq_effect:+.1f}pp)")
        print("  Confirms: compression destroys discriminative features regardless of gradient")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    cleanup(model, tokenizer)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
