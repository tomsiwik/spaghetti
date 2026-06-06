#!/usr/bin/env python3
"""
P0: Two-Stage Training — NTP then MCQ-Only for TT-LoRA Discriminative Capacity

Tests whether sequential optimization (NTP→MCQ-only) exceeds the 34.5% mixed
training ceiling by eliminating gradient competition for rank-6 capacity.

Kill criteria:
  K1440: Two-stage MedMCQA >= 38%
  K1441: Stage 2 MCQ loss < 1.20
  K1442: MCQ-only from scratch < two-stage by >= 5pp

Grounded by:
  Finding #521: Compression destroys discriminative capacity (34pp gap)
  Finding #522: MCQ loss recovers +14.5pp, ceiling at 34.5%
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

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 10 if IS_SMOKE else 200
N_STEPS_NTP = 20 if IS_SMOKE else 500
N_STEPS_MCQ = 10 if IS_SMOKE else 300  # Stage 2: fewer steps, MCQ loss converges faster
BATCH_SIZE = 2
MAX_SEQ_LEN = 512

TT_RANK = 6
TT_ALPHA = 1.0
TT_LR = 5e-3
MCQ_LR = 2e-3  # Lower LR for Stage 2 to avoid catastrophic forgetting
SEED = 42

PROJ_NAMES = ["v_proj", "o_proj"]


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
    # Check sibling experiments
    for sibling in ["exp_p0_mcq_mixed_training", "exp_p0_ttlora_e2e_benchmark",
                    "exp_p0_e2e_benchmark"]:
        p = EXPERIMENT_DIR.parent / sibling / "data" / "medical" / "train.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError("No medical training data found")


def tokenize_for_training(tokenizer, data_path, max_seq_len=512):
    """Tokenize MCQ training data, extracting correct answer index for MCQ loss."""
    answer_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    examples = []

    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]

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

    def clear_cache(self):
        self._cached_delta_w = None

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


def cache_all_delta_w(model):
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()


def clear_all_delta_w_cache(model):
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.clear_cache()


# ────────────────────────────────────────────────
# Training Functions
# ────────────────────────────────────────────────

def train_ntp(model, tokenizer, examples, n_steps, lr, batch_size, label=""):
    """NTP training with prompt masking."""
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
    return losses, total_time, optimizer


def train_mcq_only(model, tokenizer, examples, n_steps, lr, batch_size,
                   answer_token_ids, label=""):
    """MCQ-only classification training (no NTP loss)."""
    random.seed(SEED + 1)  # Different seed from Stage 1 for different data order
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    atid_a, atid_b, atid_c, atid_d = [int(t) for t in answer_token_ids]

    def loss_fn(model, input_ids, lengths, prompt_lens, correct_idxs):
        logits = model(input_ids)
        logits = logits.astype(mx.float32)

        B = input_ids.shape[0]
        V = logits.shape[-1]
        # Get logits at the answer position (prompt_len - 1 predicts first response token)
        answer_pos = (prompt_lens - 1)[:, None, None]
        answer_pos_expanded = mx.broadcast_to(answer_pos, (B, 1, V))
        answer_vocab_logits = mx.take_along_axis(
            logits, answer_pos_expanded, axis=1
        ).squeeze(1)  # (B, V)

        abcd_logits = mx.concatenate([
            answer_vocab_logits[:, atid_a:atid_a + 1],
            answer_vocab_logits[:, atid_b:atid_b + 1],
            answer_vocab_logits[:, atid_c:atid_c + 1],
            answer_vocab_logits[:, atid_d:atid_d + 1],
        ], axis=1)  # (B, 4)

        return nn.losses.cross_entropy(abcd_logits, correct_idxs, reduction="mean")

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
        correct_idxs = mx.array([e["correct_idx"] for e in batch_exs])

        loss, grads = loss_and_grad(
            model, input_ids, lengths, prompt_lens, correct_idxs
        )
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        losses.append(loss.item())

        if (step + 1) % max(1, n_steps // 5) == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            print(f"    [{label}] Step {step+1}/{n_steps}: mcq_loss={avg:.4f} ({elapsed:.1f}s)",
                  flush=True)

    total_time = time.time() - t0
    return losses, total_time


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P0: Two-Stage Training — NTP then MCQ-Only")
    print(f"TT-LoRA r{TT_RANK}: NTP({N_STEPS_NTP}) -> MCQ-only({N_STEPS_MCQ})")
    print(f"SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}")
    print("=" * 60, flush=True)

    results = {
        "experiment": "exp_p0_two_stage_mcq",
        "n_eval": N_EVAL,
        "n_steps_ntp": N_STEPS_NTP,
        "n_steps_mcq": N_STEPS_MCQ,
        "is_smoke": IS_SMOKE,
        "seed": SEED,
        "tt_rank": TT_RANK,
        "ntp_lr": TT_LR,
        "mcq_lr": MCQ_LR,
    }

    # ── Load data & model ───────────────────────
    data_path = get_medical_data_path()
    print(f"Training data: {data_path}", flush=True)

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    print("Answer token mapping:", flush=True)
    answer_token_ids = get_answer_token_ids(tokenizer)
    results["answer_token_ids"] = {
        letter: tid for letter, tid in zip("ABCD", answer_token_ids)
    }

    examples = tokenize_for_training(tokenizer, data_path, MAX_SEQ_LEN)
    print(f"  {len(examples)} training examples loaded", flush=True)
    results["n_train"] = len(examples)

    print("Loading MedMCQA validation set...", flush=True)
    medmcqa_dataset = load_medmcqa_val(N_EVAL, SEED)
    print(f"  {len(medmcqa_dataset)} evaluation questions", flush=True)

    # ── PHASE 1: Base model evaluation ──────────
    print("\n" + "=" * 60)
    print("PHASE 1: Base model evaluation (control)")
    print("=" * 60, flush=True)

    base_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "BASE")
    results["base_medmcqa_pct"] = round(base_acc, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # ── PHASE 2: TT-LoRA NTP-only (control) ────
    print("\n" + "=" * 60)
    print("PHASE 2: TT-LoRA NTP-only (control from Finding #522)")
    print("=" * 60, flush=True)

    tt_params = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    freeze_except_ttcores(model)
    results["tt_params"] = tt_params

    model.train()
    ntp_losses, ntp_time, _ = train_ntp(
        model, tokenizer, examples, N_STEPS_NTP, TT_LR, BATCH_SIZE, "NTP-only"
    )
    results["ntp_train_time_s"] = round(ntp_time, 1)
    results["ntp_final_loss"] = round(ntp_losses[-1], 4) if ntp_losses else None

    model.eval()
    cache_all_delta_w(model)
    ntp_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "TT-LoRA-NTP")
    results["ntp_medmcqa_pct"] = round(ntp_acc, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Clean up — restore base model for next condition
    remove_ttlora(model)
    mx.clear_cache()
    log_memory("after-ntp-cleanup")

    # ── PHASE 3: Two-Stage (NTP → MCQ-only) ────
    print("\n" + "=" * 60)
    print(f"PHASE 3: Two-Stage — NTP({N_STEPS_NTP}) then MCQ-only({N_STEPS_MCQ})")
    print("=" * 60, flush=True)

    # Stage 1: NTP training
    print("\n  --- Stage 1: NTP training ---", flush=True)
    inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    freeze_except_ttcores(model)

    model.train()
    s1_losses, s1_time, _ = train_ntp(
        model, tokenizer, examples, N_STEPS_NTP, TT_LR, BATCH_SIZE, "TwoStage-S1-NTP"
    )
    results["twostage_s1_time_s"] = round(s1_time, 1)
    results["twostage_s1_final_loss"] = round(s1_losses[-1], 4) if s1_losses else None

    # Stage 2: MCQ-only fine-tune (same TT-LoRA cores, new optimizer, lower LR)
    print("\n  --- Stage 2: MCQ-only fine-tune ---", flush=True)
    # Clear delta_w cache so training uses live reconstruction
    clear_all_delta_w_cache(model)

    s2_losses, s2_time = train_mcq_only(
        model, tokenizer, examples, N_STEPS_MCQ, MCQ_LR, BATCH_SIZE,
        answer_token_ids, "TwoStage-S2-MCQ"
    )
    results["twostage_s2_time_s"] = round(s2_time, 1)
    results["twostage_s2_final_mcq_loss"] = round(s2_losses[-1], 4) if s2_losses else None
    results["twostage_total_time_s"] = round(s1_time + s2_time, 1)

    # Evaluate two-stage
    model.eval()
    cache_all_delta_w(model)
    twostage_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "TwoStage")
    results["twostage_medmcqa_pct"] = round(twostage_acc, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Clean up
    remove_ttlora(model)
    mx.clear_cache()
    log_memory("after-twostage-cleanup")

    # ── PHASE 4: MCQ-only from scratch (control) ─
    print("\n" + "=" * 60)
    print(f"PHASE 4: MCQ-only from scratch ({N_STEPS_MCQ} steps, no NTP)")
    print("=" * 60, flush=True)

    inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    freeze_except_ttcores(model)

    model.train()
    mcq_scratch_losses, mcq_scratch_time = train_mcq_only(
        model, tokenizer, examples, N_STEPS_MCQ, MCQ_LR, BATCH_SIZE,
        answer_token_ids, "MCQ-scratch"
    )
    results["mcq_scratch_time_s"] = round(mcq_scratch_time, 1)
    results["mcq_scratch_final_loss"] = round(mcq_scratch_losses[-1], 4) if mcq_scratch_losses else None

    model.eval()
    cache_all_delta_w(model)
    mcq_scratch_acc = eval_medmcqa(model, tokenizer, medmcqa_dataset, "MCQ-scratch")
    results["mcq_scratch_medmcqa_pct"] = round(mcq_scratch_acc, 1)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # ── RESULTS & KILL CRITERIA ─────────────────
    print("\n" + "=" * 60)
    print("RESULTS & KILL CRITERIA")
    print("=" * 60, flush=True)

    # Kill criteria
    k1440_pass = twostage_acc >= 38.0
    k1441_pass = (s2_losses[-1] if s2_losses else 99) < 1.20
    k1442_pass = (twostage_acc - mcq_scratch_acc) >= 5.0

    results["K1440_twostage_ge38"] = "PASS" if k1440_pass else "FAIL"
    results["K1441_s2_mcq_loss_lt120"] = "PASS" if k1441_pass else "FAIL"
    results["K1442_ntp_loadbearing"] = "PASS" if k1442_pass else "FAIL"

    # Deltas
    results["ntp_delta_pp"] = round(ntp_acc - base_acc, 1)
    results["twostage_delta_pp"] = round(twostage_acc - base_acc, 1)
    results["mcq_scratch_delta_pp"] = round(mcq_scratch_acc - base_acc, 1)
    results["twostage_vs_mixed_prior"] = round(twostage_acc - 34.5, 1)  # vs Finding #522
    results["twostage_vs_mcq_scratch"] = round(twostage_acc - mcq_scratch_acc, 1)

    # Training loss trajectory for Stage 2
    if s2_losses:
        n = len(s2_losses)
        checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        results["s2_loss_trajectory"] = {
            f"step_{i+1}": round(s2_losses[c], 4) for i, c in enumerate(checkpoints)
        }
    if mcq_scratch_losses:
        n = len(mcq_scratch_losses)
        checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        results["mcq_scratch_loss_trajectory"] = {
            f"step_{i+1}": round(mcq_scratch_losses[c], 4) for i, c in enumerate(checkpoints)
        }

    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)

    print(f"\nResults:")
    print(f"  Base model:           {base_acc:.1f}%")
    print(f"  TT-LoRA NTP-only:     {ntp_acc:.1f}% (delta {ntp_acc-base_acc:+.1f}pp)")
    print(f"  TT-LoRA Two-Stage:    {twostage_acc:.1f}% (delta {twostage_acc-base_acc:+.1f}pp)")
    print(f"  TT-LoRA MCQ-scratch:  {mcq_scratch_acc:.1f}% (delta {mcq_scratch_acc-base_acc:+.1f}pp)")
    print(f"  Two-Stage vs Mixed:   {twostage_acc-34.5:+.1f}pp (vs 34.5% from Finding #522)")
    print(f"  Two-Stage vs scratch: {twostage_acc-mcq_scratch_acc:+.1f}pp")

    print(f"\nKILL CRITERIA:")
    print(f"  K1440 Two-stage >= 38%:     {results['K1440_twostage_ge38']} ({twostage_acc:.1f}%)")
    print(f"  K1441 S2 MCQ loss < 1.20:   {results['K1441_s2_mcq_loss_lt120']} ({s2_losses[-1]:.4f})" if s2_losses else "  K1441: no data")
    print(f"  K1442 NTP load-bearing >=5pp: {results['K1442_ntp_loadbearing']} ({twostage_acc-mcq_scratch_acc:+.1f}pp)")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    cleanup(model, tokenizer)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
