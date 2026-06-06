#!/usr/bin/env python3
"""
P0: Discriminative Collapse Diagnosis — NTP vs Compression on MedMCQA

A/B test: Standard LoRA (rank-8) vs TT-LoRA (rank-6), both trained with NTP
on same medical data, evaluated on MedMCQA. Diagnoses whether discriminative
collapse is from the training objective or from compression.

Kill criteria:
  K1430: Standard LoRA MedMCQA >= 35% (if pass, TT-LoRA compression is the disease)
  K1431: TT-LoRA MedMCQA >= 35% (if only this fails, compression discards discriminative features)
  K1432: Both NTP adapters degrade MedMCQA below 45% base → NTP is the disease

Grounded by:
  Finding #517: Standard LoRA NTP adapter degrades MMLU-Pro by -6.2pp
  Finding #508: E2E pipeline baselines
  E2E benchmark: TT-LoRA MedMCQA 21%
  arXiv:2504.21190 (TT-LoRA)
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

# Adapter configs (match E2E benchmark)
LORA_RANK = 8
LORA_SCALE = 8.0
TT_RANK = 6
TT_ALPHA = 1.0
LR = 1e-4          # Standard LoRA lr (Finding #508)
TT_LR = 5e-3       # TT-LoRA lr (arXiv:2504.21190)

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
# MedMCQA Evaluation (shared across all configs)
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
    """Find medical training data (shared from E2E benchmark)."""
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
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]
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
            })
    return examples


def train_ntp(model, tokenizer, data_path, n_steps, lr, batch_size, label=""):
    """NTP training with prompt masking (SFT). Shared by both LoRA and TT-LoRA."""
    examples = tokenize_for_training(tokenizer, data_path, MAX_SEQ_LEN)
    print(f"  [{label}] Loaded {len(examples)} training examples", flush=True)

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


# ────────────────────────────────────────────────
# Standard LoRA Injection
# ────────────────────────────────────────────────

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


def inject_lora(model, proj_names, rank, scale):
    """Inject standard LoRA adapters using LoRALinear wrapping."""
    from mlx_lm.tuner.lora import LoRALinear
    layers = get_layers(model)
    total_params = 0
    for i, layer in enumerate(layers):
        for name in proj_names:
            base = getattr(layer.self_attn, name)
            in_f, out_f = detect_proj_dims(base)
            lora = LoRALinear.from_base(base, r=rank, scale=scale)
            setattr(layer.self_attn, name, lora)
            total_params += rank * (in_f + out_f)
    print(f"  Standard LoRA: {total_params:,} trainable params "
          f"(rank={rank}, scale={scale})", flush=True)
    return total_params


def freeze_except_lora(model):
    from mlx_lm.tuner.lora import LoRALinear
    model.freeze()
    for layer in get_layers(model):
        for name in PROJ_NAMES:
            proj = getattr(layer.self_attn, name)
            if isinstance(proj, LoRALinear):
                proj.unfreeze(keys=["lora_a", "lora_b"], recurse=False)


def remove_lora(model):
    """Remove LoRA wrappers, restoring base layers."""
    from mlx_lm.tuner.lora import LoRALinear
    layers = get_layers(model)
    for layer in layers:
        for name in PROJ_NAMES:
            proj = getattr(layer.self_attn, name)
            if isinstance(proj, LoRALinear):
                setattr(layer.self_attn, name, proj.linear)


# ────────────────────────────────────────────────
# TT-LoRA Module (from exp_p0_ttlora_e2e_benchmark)
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
    """Remove TT-LoRA wrappers, restoring base layers."""
    layers = get_layers(model)
    for layer in layers:
        for name in PROJ_NAMES:
            proj = getattr(layer.self_attn, name)
            if isinstance(proj, TTLoRAWrapper):
                setattr(layer.self_attn, name, proj.base_layer)


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P0: Discriminative Collapse Diagnosis")
    print(f"NTP vs Compression on MedMCQA ({N_EVAL} questions)")
    print(f"Standard LoRA rank-{LORA_RANK} vs TT-LoRA rank-{TT_RANK}")
    print(f"SMOKE={IS_SMOKE}, N_STEPS={N_STEPS}")
    print("=" * 60, flush=True)

    results = {
        "experiment": "exp_p0_discriminative_diagnosis",
        "n_eval": N_EVAL,
        "n_steps": N_STEPS,
        "is_smoke": IS_SMOKE,
        "seed": SEED,
    }

    # ── Phase 0: Load data ────────────────────────
    data_path = get_medical_data_path()
    print(f"Training data: {data_path}", flush=True)
    n_lines = sum(1 for _ in open(data_path))
    print(f"  {n_lines} training examples", flush=True)

    # ── Phase 1: Base model evaluation (control) ──
    print("\n" + "=" * 60)
    print("PHASE 1: Base model evaluation (control)")
    print("=" * 60, flush=True)

    from mlx_lm import load

    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Load fixed evaluation dataset
    print("Loading MedMCQA validation set...", flush=True)
    medmcqa_dataset = load_medmcqa_val(N_EVAL, SEED)
    print(f"  {len(medmcqa_dataset)} evaluation questions", flush=True)

    base_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "BASE")
    results["base_medmcqa_pct"] = round(base_acc, 1)

    # Checkpoint
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Base: {base_acc:.1f}% on MedMCQA", flush=True)

    # ── Phase 2: Standard LoRA training ───────────
    print("\n" + "=" * 60)
    print("PHASE 2: Standard LoRA rank-8 training (NTP)")
    print("=" * 60, flush=True)

    lora_params = inject_lora(model, PROJ_NAMES, LORA_RANK, LORA_SCALE)
    freeze_except_lora(model)
    results["lora_params"] = lora_params

    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable: {trainable:,} params", flush=True)

    model.train()
    lora_losses, lora_time = train_ntp(
        model, tokenizer, data_path, N_STEPS, LR, BATCH_SIZE, "LoRA")
    results["lora_train_time_s"] = round(lora_time, 1)
    results["lora_final_loss"] = round(lora_losses[-1], 4) if lora_losses else None

    # ── Phase 3: Standard LoRA evaluation ─────────
    print("\n" + "=" * 60)
    print("PHASE 3: Standard LoRA MedMCQA evaluation")
    print("=" * 60, flush=True)

    model.eval()
    lora_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "LoRA-r8")
    results["lora_medmcqa_pct"] = round(lora_acc, 1)

    # Checkpoint
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Remove LoRA, restore base model
    print("Removing LoRA wrappers...", flush=True)
    remove_lora(model)
    log_memory("after-lora-removed")

    # ── Phase 4: TT-LoRA training ─────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: TT-LoRA rank-6 training (NTP)")
    print("=" * 60, flush=True)

    tt_params = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    freeze_except_ttcores(model)
    results["tt_params"] = tt_params

    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable: {trainable:,} params", flush=True)

    model.train()
    tt_losses, tt_time = train_ntp(
        model, tokenizer, data_path, N_STEPS, TT_LR, BATCH_SIZE, "TT-LoRA")
    results["tt_train_time_s"] = round(tt_time, 1)
    results["tt_final_loss"] = round(tt_losses[-1], 4) if tt_losses else None

    # ── Phase 5: TT-LoRA evaluation ───────────────
    print("\n" + "=" * 60)
    print("PHASE 5: TT-LoRA MedMCQA evaluation")
    print("=" * 60, flush=True)

    model.eval()
    # Cache delta_w for fast inference
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()

    tt_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "TT-LoRA-r6")
    results["tt_medmcqa_pct"] = round(tt_acc, 1)

    cleanup(model, tokenizer)

    # ── Phase 6: Diagnosis ────────────────────────
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60, flush=True)

    # Kill criteria
    k1_pass = lora_acc >= 35.0    # Standard LoRA >= 35%
    k2_pass = tt_acc >= 35.0      # TT-LoRA >= 35%
    k3_pass = (lora_acc < base_acc) and (tt_acc < base_acc)  # Both below base

    results["K1430_lora_ge35"] = "PASS" if k1_pass else "FAIL"
    results["K1431_ttlora_ge35"] = "PASS" if k2_pass else "FAIL"
    results["K1432_both_degrade"] = "PASS" if k3_pass else "FAIL"

    # Diagnosis logic
    if k1_pass and not k2_pass:
        diagnosis = "COMPRESSION is the disease"
        detail = ("Standard LoRA preserves discriminative features (>= 35%), "
                  "TT-LoRA compression discards them (< 35%).")
    elif not k1_pass and not k2_pass:
        diagnosis = "NTP TRAINING OBJECTIVE is the disease"
        detail = ("Both adapter types degrade MedMCQA below 35%. "
                  "The training objective, not compression, is the root cause.")
    elif k1_pass and k2_pass:
        diagnosis = "NEITHER at these thresholds (unexpected)"
        detail = ("Both adapters retain >= 35% MedMCQA. "
                  "E2E benchmark methodology may have confounds.")
    else:
        diagnosis = "UNEXPECTED: TT-LoRA passes but LoRA fails"
        detail = "Requires investigation — contradicts theory."

    results["diagnosis"] = diagnosis
    results["diagnosis_detail"] = detail

    # Degradation analysis
    lora_delta = lora_acc - base_acc
    tt_delta = tt_acc - base_acc
    compression_effect = tt_acc - lora_acc  # Negative = compression hurts

    results["lora_delta_pp"] = round(lora_delta, 1)
    results["tt_delta_pp"] = round(tt_delta, 1)
    results["compression_effect_pp"] = round(compression_effect, 1)

    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)

    # Print summary
    print(f"\nResults:")
    print(f"  Base model:      {base_acc:.1f}%")
    print(f"  Standard LoRA:   {lora_acc:.1f}% (delta {lora_delta:+.1f}pp)")
    print(f"  TT-LoRA:         {tt_acc:.1f}% (delta {tt_delta:+.1f}pp)")
    print(f"  Compression effect: {compression_effect:+.1f}pp (TT-LoRA minus LoRA)")

    print(f"\nKILL CRITERIA:")
    print(f"  K1430 LoRA >= 35%:     {results['K1430_lora_ge35']} ({lora_acc:.1f}%)")
    print(f"  K1431 TT-LoRA >= 35%:  {results['K1431_ttlora_ge35']} ({tt_acc:.1f}%)")
    print(f"  K1432 Both degrade:    {results['K1432_both_degrade']}")

    print(f"\nDIAGNOSIS: {diagnosis}")
    print(f"  {detail}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
