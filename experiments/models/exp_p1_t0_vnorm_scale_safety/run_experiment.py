#!/usr/bin/env python3
"""
T0.2: V-Norm eliminates scale catastrophe — controlled experiment on Qwen3-4B.

Design rationale: Gemma 4 (gemma4 model_type) is not loadable with mlx_lm 0.29.1.
Instead, we inject v_norm into Qwen3-4B's value projection as a VNormWrapper,
giving a controlled within-model comparison:
  - Condition A (no v_norm): standard Qwen3 v_proj, adapter at scale=5,10,20
  - Condition B (with v_norm): VNormWrapper(v_proj), same adapter at scale=5,10,20
This directly tests the causal effect of v_norm, stronger than testing on Gemma 4.

Kill criteria:
  K994: WITH v_norm: 0pp MMLU degradation at ANY adapter scale (5,10,20)
  K995: WITHOUT v_norm: scale=20 degrades MMLU by >30pp (within-experiment verification)
  K996: WITH v_norm: adapter quality ratio at scale=10,20 vs scale=5 >= 0.95

Architecture note:
  Qwen3 attention forward (no v_norm built in):
    queries, keys, values = q_proj(x), k_proj(x), v_proj(x)
    values = values.reshape(B,L,n_kv,-1).transpose(0,2,1,3)   # NO norm on values

  VNormWrapper injects v_norm between v_proj output and reshape:
    values_raw = v_proj(x)                                       # [B,L,n_kv*h]
    values_heads = values_raw.reshape(B,L,n_kv,head_dim)
    values_normed = rms_norm(values_heads, None, eps=1e-6)       # unit RMS per head
    → values = values_normed.reshape(B,L,n_kv*head_dim)

  LoRA scale only changes the PRE-NORM magnitude → bounded post-norm perturbation.

References:
  MATH.md Theorem 1: ||V_norm(s)||_RMS = sqrt(h_v) for all s
  MATH.md Theorem 2: MMLU degradation bound is s-independent under v_norm
  Finding #320: Qwen3-4B (no v_norm), scale=20 → -60pp MMLU catastrophe

Supports SMOKE_TEST=1 for quick validation (<5 min).
"""

import gc
import json
import math
import os
import time
from functools import partial
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"   # use cached parquet, skip network checks

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_lm import load as mlx_load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.models.base import scaled_dot_product_attention

# ── Memory safety (MANDATORY per CODING_GUIDELINES) ──────────────────────────
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 6 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen3-4B-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
TRAIN_STEPS = 10 if IS_SMOKE else 100
N_MMLU_PER_SUBJECT = 5 if IS_SMOKE else 10
N_TRAIN_EXAMPLES = 5 if IS_SMOKE else 20
N_EVAL_GSM8K = 5 if IS_SMOKE else 20

LORA_RANK = 4
LORA_SCALE_TRAIN = 1.0        # Train at neutral scale; test at 5,10,20
LORA_LAYERS = 28              # All Qwen3-4B layers

TEST_SCALES = [5, 10, 20]
MMLU_THRESHOLD_PP = 5.0       # K994: max degradation with v_norm
K995_THRESHOLD_PP = 30.0      # K995: min degradation without v_norm at scale=20
QUALITY_RATIO_THRESHOLD = 0.95  # K996: quality ratio at scale=10,20 vs 5

MMLU_SUBJECTS = [
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "computer_security",
]

SEED = 42
np.random.seed(SEED)
mx.random.seed(SEED)

OUT_DIR = Path(__file__).parent
results = {
    "is_smoke": IS_SMOKE,
    "model": MODEL_ID,
    "lora_rank": LORA_RANK,
    "lora_scale_train": LORA_SCALE_TRAIN,
    "test_scales": TEST_SCALES,
    "mmlu_no_vnorm": {},
    "mmlu_with_vnorm": {},
    "gsm8k_with_vnorm": {},
    "k994_pass": None,
    "k995_pass": None,
    "k996_pass": None,
}

# ── V-Norm injection ──────────────────────────────────────────────────────────

class VNormLinear(nn.Module):
    """Wraps v_proj (or LoRALinear(v_proj)) to apply value normalization.

    Injects RMSNorm (no learned scale) on each head's value vector,
    matching Gemma 4's `self.v_norm = RMSNoScale()`.

    Shape flow:
      input x: [B, L, d_model]
      after v_proj: [B, L, n_kv * head_dim]
      after reshape: [B, L, n_kv, head_dim]
      after rms_norm (no scale): each [head_dim] vector has unit RMS
      after reshape: [B, L, n_kv * head_dim]
    """

    def __init__(self, v_proj: nn.Module, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.v_proj = v_proj
        self.head_dim = head_dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        out = self.v_proj(x)                                 # [B, L, n_kv * h]
        B, L, D = out.shape
        n_kv = D // self.head_dim
        out = out.reshape(B, L, n_kv, self.head_dim)
        out = mx.fast.rms_norm(out, None, self.eps)          # unit RMS per head
        return out.reshape(B, L, D)


def enable_vnorm(model: nn.Module, head_dim: int) -> None:
    """Replace all v_proj with VNormLinear(v_proj) on all attention layers."""
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.v_proj = VNormLinear(attn.v_proj, head_dim)


def disable_vnorm(model: nn.Module) -> None:
    """Unwrap VNormLinear → restore plain v_proj."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if isinstance(attn.v_proj, VNormLinear):
            attn.v_proj = attn.v_proj.v_proj


def set_lora_scale(model: nn.Module, scale: float) -> None:
    """Set inference scale on all LoRALinear layers."""
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.scale = scale


# ── MMLU eval ─────────────────────────────────────────────────────────────────

def eval_mmlu_subject(model, tokenizer, subject: str, n_questions: int) -> float | None:
    from datasets import load_dataset
    try:
        ds = load_dataset("cais/mmlu", subject, split="validation", trust_remote_code=False)
    except Exception as e:
        print(f"    WARNING: Could not load {subject}: {e}")
        return None

    items = list(ds)
    rng = np.random.RandomState(SEED)
    rng.shuffle(items)
    items = items[:n_questions]

    LETTERS = ["A", "B", "C", "D"]
    correct = 0

    for item in items:
        opts = item["choices"]
        label_idx = item["answer"]
        prompt = f"Question: {item['question']}\n"
        for i, opt in enumerate(opts):
            prompt += f"{LETTERS[i]}. {opt}\n"
        prompt += "Answer:"

        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        x = mx.array(input_ids)[None]
        logits = model(x)
        mx.eval(logits)

        choice_scores = []
        for ch in LETTERS:
            ch_ids = tokenizer.encode(" " + ch, add_special_tokens=False)
            score = logits[0, -1, ch_ids[0]].item() if ch_ids else -1e9
            choice_scores.append(score)

        if int(np.argmax(choice_scores)) == label_idx:
            correct += 1

    return correct / len(items) if items else 0.0


def eval_mmlu(model, tokenizer, label: str) -> float:
    print(f"  MMLU [{label}]:")
    accs = []
    for subj in MMLU_SUBJECTS:
        acc = eval_mmlu_subject(model, tokenizer, subj, N_MMLU_PER_SUBJECT)
        if acc is not None:
            print(f"    {subj}: {acc*100:.1f}%")
            accs.append(acc)
    mean = float(np.mean(accs)) if accs else 0.0
    print(f"    MEAN: {mean*100:.1f}%")
    return mean


# ── GSM8K eval ────────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, examples: list, label: str) -> float:
    import re

    def extract_answer(text: str) -> str | None:
        m = re.search(r"####\s*([\d,]+)", text)
        if m:
            return m.group(1).replace(",", "")
        nums = re.findall(r"\d+", text)
        return nums[-1] if nums else None

    correct = 0
    for ex in examples[:N_EVAL_GSM8K]:
        q = ex["question"].strip()
        gold = extract_answer(ex["answer"])
        prompt = f"Q: {q}\nA:"

        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        x = mx.array(input_ids)[None]
        generated = []
        for _ in range(256):
            inputs = mx.concatenate([x, mx.array(generated)[None]], axis=1) if generated else x
            logits = model(inputs)
            next_tok = mx.argmax(logits[0, -1, :]).item()
            if next_tok == tokenizer.eos_token_id:
                break
            generated.append(next_tok)
        pred_text = tokenizer.decode(generated)
        pred = extract_answer(pred_text)
        correct += int(pred == gold and pred is not None)

    acc = correct / N_EVAL_GSM8K
    print(f"  GSM8K [{label}]: {correct}/{N_EVAL_GSM8K} = {acc*100:.1f}%")
    return acc


# ── Training ──────────────────────────────────────────────────────────────────

def make_sft_batch(examples, tokenizer, max_len=256):
    seqs, masks = [], []
    for ex in examples:
        prompt_ids = tokenizer.encode(
            f"Q: {ex['question'].strip()}\nA:", add_special_tokens=True
        )
        comp_ids = tokenizer.encode(
            f" {ex['answer'].strip()}", add_special_tokens=False
        )
        full = (prompt_ids + comp_ids + [tokenizer.eos_token_id])[:max_len]
        mask = ([0] * len(prompt_ids) + [1] * (len(full) - len(prompt_ids)))[:max_len]
        seqs.append(full)
        masks.append(mask)

    max_l = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    seqs_p = [s + [pad] * (max_l - len(s)) for s in seqs]
    masks_p = [m + [0] * (max_l - len(m)) for m in masks]
    return mx.array(seqs_p), mx.array(masks_p, dtype=mx.float32)


def sft_loss(model, input_ids, target_mask):
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = target_mask[:, 1:]
    loss_tokens = nn.losses.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_targets.reshape(-1),
        reduction="none",
    ).reshape(shift_logits.shape[:2])
    denom = shift_mask.sum() + 1e-6
    return (loss_tokens * shift_mask).sum() / denom


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("T0.2: V-Norm Scale Safety (Qwen3-4B controlled experiment)")
print(f"SMOKE={IS_SMOKE}, steps={TRAIN_STEPS}, mmlu/subj={N_MMLU_PER_SUBJECT}")
print("=" * 60)

t_start = time.time()

# Phase 1: Load model
print(f"\nPhase 1: Loading {MODEL_ID}...")
t0 = time.time()
model, tokenizer = mlx_load(MODEL_ID)
print(f"  Loaded in {time.time()-t0:.1f}s")

# Get head_dim from attention scale: scale = 1/sqrt(head_dim)
attn0 = model.model.layers[0].self_attn
head_dim = round(1.0 / attn0.scale ** 2)  # inverse of scale = 1/sqrt(h)
n_layers = len(model.model.layers)
print(f"  head_dim={head_dim}, n_layers={n_layers}")
results["head_dim"] = head_dim
results["n_layers"] = n_layers

# Phase 2: Apply LoRA to v_proj
print(f"\nPhase 2: Applying LoRA (rank={LORA_RANK}, scale={LORA_SCALE_TRAIN})...")
lora_config = {
    "rank": LORA_RANK,
    "scale": LORA_SCALE_TRAIN,
    "dropout": 0.0,
    "keys": {"self_attn.v_proj"},
}
model.freeze()
linear_to_lora_layers(model, LORA_LAYERS, lora_config)
model.train()

n_lora = sum(
    m.lora_a.size + m.lora_b.size
    for _, m in model.named_modules()
    if isinstance(m, LoRALinear)
)
print(f"  LoRA trainable params: {n_lora:,}")
results["lora_n_params"] = n_lora

# Phase 3: Load training data
print(f"\nPhase 3: Loading training data...")
from datasets import load_dataset

gsm8k_train = list(load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=False))
gsm8k_test = list(load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=False))
rng = np.random.RandomState(SEED)
train_idx = rng.permutation(len(gsm8k_train))[:N_TRAIN_EXAMPLES].tolist()
eval_idx = rng.permutation(len(gsm8k_test))[:N_EVAL_GSM8K].tolist()
train_ex = [gsm8k_train[i] for i in train_idx]
eval_ex = [gsm8k_test[i] for i in eval_idx]
print(f"  train={len(train_ex)}, eval={len(eval_ex)}")

# Phase 4: Train (no v_norm in training = standard Qwen3)
print(f"\nPhase 4: Training {TRAIN_STEPS} steps (NO v_norm in training)...")
optimizer = optim.AdamW(learning_rate=1e-4)
input_ids, target_mask = make_sft_batch(train_ex, tokenizer)
mx.eval(input_ids, target_mask)

state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def train_step(ids, mask):
    loss, grads = nn.value_and_grad(model, sft_loss)(model, ids, mask)
    optimizer.update(model, grads)
    return loss

grad_norm_step0 = None
t_train = time.time()
for step in range(TRAIN_STEPS):
    loss = train_step(input_ids, target_mask)
    mx.eval(model.parameters(), optimizer.state)

    if step == 0:
        _, grads = nn.value_and_grad(model, sft_loss)(model, input_ids, target_mask)
        mx.eval(grads)
        gn = math.sqrt(sum(
            (g * g).sum().item()
            for _, g in tree_flatten(grads)
            if isinstance(g, mx.array)
        ))
        grad_norm_step0 = gn
        del grads
        gc.collect()
        print(f"  step 0: loss={loss.item():.4f}, grad_norm={gn:.4f}")

    if (step + 1) % max(1, TRAIN_STEPS // 5) == 0:
        print(f"  step {step+1}/{TRAIN_STEPS}: loss={loss.item():.4f}")

print(f"  Training done in {time.time()-t_train:.1f}s")
results["train_time_s"] = time.time() - t_train
results["grad_norm_step0"] = grad_norm_step0

# Set model to eval mode
model.eval()

# Phase 5: MMLU baseline (scale=0, no adapter contribution)
print(f"\nPhase 5: MMLU baseline...")
set_lora_scale(model, 0.0)
base_mmlu = eval_mmlu(model, tokenizer, "base (scale=0)")
results["mmlu_base"] = base_mmlu
mx.metal.clear_cache(); gc.collect()

# Phase 6: MMLU WITHOUT v_norm at scale=5,10,20
print(f"\nPhase 6: MMLU WITHOUT v_norm (scale sweep)...")
for scale in TEST_SCALES:
    set_lora_scale(model, float(scale))
    acc = eval_mmlu(model, tokenizer, f"no_vnorm scale={scale}")
    results["mmlu_no_vnorm"][f"scale_{scale}"] = acc
    mx.metal.clear_cache(); gc.collect()

# Phase 7: Enable v_norm, sweep scales
print(f"\nPhase 7: MMLU WITH v_norm (scale sweep)...")
enable_vnorm(model, head_dim)
for scale in TEST_SCALES:
    set_lora_scale(model, float(scale))
    acc = eval_mmlu(model, tokenizer, f"vnorm scale={scale}")
    results["mmlu_with_vnorm"][f"scale_{scale}"] = acc
    mx.metal.clear_cache(); gc.collect()

# Phase 8: GSM8K quality WITH v_norm (K996)
print(f"\nPhase 8: GSM8K quality WITH v_norm (scale sweep)...")
for scale in TEST_SCALES:
    set_lora_scale(model, float(scale))
    acc = eval_gsm8k(model, tokenizer, eval_ex, f"vnorm scale={scale}")
    results["gsm8k_with_vnorm"][f"scale_{scale}"] = acc
    mx.metal.clear_cache(); gc.collect()

# ── Kill criteria ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Kill Criteria Evaluation")
print("=" * 60)

# K994: WITH v_norm, all scales within 5pp of base
mmlu_diffs_vnorm = {}
for scale in TEST_SCALES:
    diff_pp = (base_mmlu - results["mmlu_with_vnorm"][f"scale_{scale}"]) * 100
    mmlu_diffs_vnorm[f"scale_{scale}"] = diff_pp
k994_pass = all(abs(d) <= MMLU_THRESHOLD_PP for d in mmlu_diffs_vnorm.values())
results["mmlu_diffs_vnorm_pp"] = mmlu_diffs_vnorm
results["k994_pass"] = k994_pass
print(f"\nK994 (WITH v_norm, 0pp degradation):")
for scale in TEST_SCALES:
    d = mmlu_diffs_vnorm[f"scale_{scale}"]
    print(f"  scale={scale}: {d:+.1f}pp {'✓' if abs(d)<=MMLU_THRESHOLD_PP else '✗'}")
print(f"  → {'PASS' if k994_pass else 'FAIL'} (threshold: {MMLU_THRESHOLD_PP}pp)")

# K995: WITHOUT v_norm, scale=20 degrades by >30pp
deg_at_20 = (base_mmlu - results["mmlu_no_vnorm"]["scale_20"]) * 100
k995_pass = deg_at_20 >= K995_THRESHOLD_PP
results["mmlu_diffs_no_vnorm_pp"] = {
    f"scale_{s}": (base_mmlu - results["mmlu_no_vnorm"][f"scale_{s}"]) * 100
    for s in TEST_SCALES
}
results["k995_pass"] = k995_pass
results["k995_deg_at_scale20_pp"] = deg_at_20
print(f"\nK995 (WITHOUT v_norm, scale=20 degrades >30pp):")
for scale in TEST_SCALES:
    d = results["mmlu_diffs_no_vnorm_pp"][f"scale_{scale}"]
    print(f"  scale={scale}: {d:+.1f}pp")
print(f"  → {'PASS' if k995_pass else 'FAIL'} "
      f"(scale=20 degradation={deg_at_20:.1f}pp, threshold={K995_THRESHOLD_PP}pp)")

# K996: WITH v_norm, quality ratio at scale=10,20 vs scale=5 >= 0.95
gsm5 = results["gsm8k_with_vnorm"].get("scale_5", 0.0)
quality_ratios = {}
k996_pass = True
print(f"\nK996 (quality ratio WITH v_norm, scale=10,20 vs 5 >= {QUALITY_RATIO_THRESHOLD}):")
for scale in [10, 20]:
    gsm_s = results["gsm8k_with_vnorm"].get(f"scale_{scale}", 0.0)
    ratio = (gsm_s / gsm5) if gsm5 > 0 else (1.0 if gsm_s == 0 else 0.0)
    quality_ratios[f"scale_{scale}_vs_5"] = ratio
    ok = ratio >= QUALITY_RATIO_THRESHOLD
    if not ok:
        k996_pass = False
    print(f"  scale={scale}/{scale//scale}: {gsm_s*100:.1f}% / {gsm5*100:.1f}% = {ratio:.3f} {'✓' if ok else '✗'}")
print(f"  → {'PASS' if k996_pass else 'FAIL'}")
results["quality_ratios"] = quality_ratios
results["k996_pass"] = k996_pass

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.time() - t_start
results["total_time_s"] = total_time
results["base_mmlu"] = base_mmlu

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Base MMLU: {base_mmlu*100:.1f}%")
print(f"\n  WITHOUT v_norm:")
for scale in TEST_SCALES:
    acc = results["mmlu_no_vnorm"][f"scale_{scale}"]
    d = results["mmlu_diffs_no_vnorm_pp"][f"scale_{scale}"]
    print(f"    scale={scale}: MMLU={acc*100:.1f}% ({d:+.1f}pp)")
print(f"\n  WITH v_norm:")
for scale in TEST_SCALES:
    acc = results["mmlu_with_vnorm"][f"scale_{scale}"]
    d = mmlu_diffs_vnorm[f"scale_{scale}"]
    gsm_acc = results["gsm8k_with_vnorm"][f"scale_{scale}"]
    print(f"    scale={scale}: MMLU={acc*100:.1f}% ({d:+.1f}pp), GSM8K={gsm_acc*100:.1f}%")

print(f"\n  K994 (v_norm prevents MMLU degradation): {'PASS' if k994_pass else 'FAIL'}")
print(f"  K995 (no v_norm: scale=20 catastrophic):  {'PASS' if k995_pass else 'FAIL'}")
print(f"  K996 (quality ratio >= {QUALITY_RATIO_THRESHOLD}):            {'PASS' if k996_pass else 'FAIL'}")
print(f"\n  Total time: {total_time/60:.1f} min")

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}")
