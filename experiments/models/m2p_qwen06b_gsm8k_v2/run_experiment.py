#!/usr/bin/env python3
"""M2P on Qwen3-0.6B + GSM8K v2 — Corrected Implementation.

Kill criteria (experiment system IDs):
  K#909: base_acc > 0% (fail-fast gate — evaluation pipeline works)
  K#910: sft_gain >= 5pp over base (valid SFT signal)
  K#911: quality_ratio = M2P_acc / SFT_acc >= 70% (M2P useful on real NLP)
  K#912: quality_ratio < 30% -> KILL (hypernetwork is toy-only)

Fixes over v1 (all 6):
  Fix #1: Use mlx_lm LoRALinear.from_base for weight-space LoRA (not residual-stream)
  Fix #2: GQA dims read dynamically from model config (never hardcoded)
  Fix #3: Causal mask via full model forward path — no custom attention calls
  Fix #4: max_gen_tokens=384 (GSM8K CoT averages 200-400 tokens)
  Fix #5: max_seq_len=512, train_steps=1000
  Fix #6: Fail-fast assert base_acc > 0 before any training

M2P training approach:
  - M2P is a separate nn.Module (the trainable thing)
  - The model is frozen; we apply M2P-generated B-matrices by temporarily
    setting lora_b on each LoRALinear module before the model forward.
  - value_and_grad traces through M2P parameters -> B-matrices -> model forward.
  - The model forward uses the standard causal path (create_attention_mask inside
    Qwen3Model.__call__), so masking is guaranteed correct.

Supports SMOKE_TEST=1 for quick validation (<5 min).
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.tuner.lora import LoRALinear

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config ----------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"

# LoRA config
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P config (primary: d_M2P=128 per Aghajanyan d_int lower bound for NLP)
D_M2P = 128
L_M2P = 2
N_MEMORY = 32

# Training
N_TRAIN = 50 if IS_SMOKE else 2000
N_TEST = 10 if IS_SMOKE else 200
TRAIN_STEPS = 20 if IS_SMOKE else 1000   # Fix #5: was 300
LR = 1e-4
MAX_SEQ_LEN = 64 if IS_SMOKE else 512    # Fix #5: was 256
MAX_GEN_TOKENS = 64 if IS_SMOKE else 384  # Fix #4: was 128
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
ADAPTER_PATH = EXPERIMENT_DIR / "sft_b_matrices.npz"
LORA_A_PATH = EXPERIMENT_DIR / "lora_a_matrices.npz"  # Fixed random A-matrices (shared across phases)
M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Few-shot prefix: teaches model the #### answer format (critical for Qwen3 base eval)
# Using 2 canonical GSM8K examples to elicit correct format without being too long
FEW_SHOT_PREFIX = (
    "Solve the math problem step by step and end with '#### <answer>'.\n\n"
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = 24 clips in May. "
    "Natalia sold 48+24 = 72 clips altogether in April and May. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
    "babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10\n\n"
)


# ---- Utilities ---------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def extract_gsm8k_answer(text: str):
    """Extract final numeric answer from GSM8K #### format or fallback patterns.

    Primary: look for '#### <number>' (canonical GSM8K format).
    Fallback: look for 'The answer is <number>' or similar phrase patterns.
    This handles Qwen3 models that answer without the #### terminator.
    """
    # Primary: canonical #### format
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    # Secondary: "the answer is N" or "answer is N" pattern
    match = re.search(r"(?:the\s+)?answer\s+is\s+[-–]?\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    # Tertiary: "total is N" or "result is N"
    match = re.search(r"(?:total|result|sum)\s+(?:is|=|:)\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    return None


# ---- Phase 1: Load data -----------------------------------------------------

def phase_load_data():
    """Load and format GSM8K examples."""
    log("\n" + "=" * 70)
    log("[Phase 1] Loading GSM8K data")
    log("=" * 70)
    t0 = time.time()

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")

    rng = random.Random(SEED)
    train_examples = list(ds["train"])
    rng.shuffle(train_examples)
    train_examples = train_examples[:N_TRAIN]

    test_examples = list(ds["test"])
    rng.shuffle(test_examples)
    test_examples = test_examples[:N_TEST]

    log(f"  Train: {len(train_examples)}, Test: {len(test_examples)}")
    log(f"  Data loaded in {time.time()-t0:.1f}s")
    return train_examples, test_examples


# ---- Model dimension inspection (Fix #2) ------------------------------------

def get_model_dims(model) -> dict:
    """Read actual GQA dimensions from model config.

    Fix #2: Never hardcode projection dimensions. Verify against weight shapes.
    Returns dict with n_layers, d_model, q_proj_out, v_proj_out, head_dim.
    """
    args = model.args
    n_layers = args.num_hidden_layers
    d_model = args.hidden_size
    n_heads = args.num_attention_heads
    n_kv_heads = args.num_key_value_heads
    head_dim = args.head_dim

    q_proj_out = n_heads * head_dim
    v_proj_out = n_kv_heads * head_dim

    # Verify against actual weight shapes
    layer0 = model.model.layers[0]
    q_weight_shape = layer0.self_attn.q_proj.weight.shape
    v_weight_shape = layer0.self_attn.v_proj.weight.shape
    mx.eval(layer0.self_attn.q_proj.weight)  # force eval to get shape

    log(f"  Config: n_layers={n_layers}, d_model={d_model}, "
        f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    log(f"  q_proj weight shape: {q_weight_shape}")
    log(f"  v_proj weight shape: {v_weight_shape}")
    log(f"  Expected q_proj_out={q_proj_out}, v_proj_out={v_proj_out}")

    return {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "q_proj_out": q_proj_out,
        "v_proj_out": v_proj_out,
    }


# ---- LoRA model setup (Fix #1) -----------------------------------------------

def apply_lora_to_model(model, lora_a_dict: dict = None) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj in all layers.

    Fix #1: LoRALinear.from_base wraps the weight matrix directly.
    The computation is: y = W_base(x) + scale * (x @ lora_a) @ lora_b
    This is weight-space LoRA, not residual-stream addition.

    Fix #2: lora_a_dict contains pre-saved random A-matrices from LORA_A_PATH,
    ensuring SFT, M2P training, and M2P eval all use IDENTICAL lora_a matrices.
    Without this, each call to LoRALinear.from_base creates NEW random lora_a,
    making M2P B-matrices meaningless when injected into a different-lora_a model.

    Fix #3: The model forward path calls create_attention_mask(h, cache[0])
    automatically — no need to pass mask explicitly to attention modules.

    lora_a_dict: dict[(li, mod_name)] -> mx.array (input_dims, rank), or None
                 to initialize fresh (only for phase_sft_train where we also save them).
    """
    model.freeze()

    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(
            attn.q_proj, r=LORA_RANK, scale=LORA_SCALE
        )
        attn.v_proj = LoRALinear.from_base(
            attn.v_proj, r=LORA_RANK, scale=LORA_SCALE
        )
        # Overwrite randomly-initialized lora_a with our saved matrices
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]

    # Unfreeze only lora_b (lora_a acts as fixed random projection,
    # analogous to Grassmannian A-matrices — see FlyLoRA arXiv:2510.08396)
    model.unfreeze(keys=["lora_b"])


def get_lora_b_matrices(model) -> dict:
    """Extract current lora_b matrices from LoRA-wrapped model.

    Returns: dict[(li, mod_name)] -> mx.array shape (r, output_dims)
    """
    b_by_key = {}
    for li, layer in enumerate(model.model.layers):
        for mod_name in ["q_proj", "v_proj"]:
            lora_module = getattr(layer.self_attn, mod_name)
            if hasattr(lora_module, "lora_b"):
                b_by_key[(li, mod_name)] = lora_module.lora_b
    return b_by_key


def get_lora_a_matrices(model) -> dict:
    """Extract current lora_a matrices from LoRA-wrapped model.

    Returns: dict[(li, mod_name)] -> mx.array shape (input_dims, r)
    """
    a_by_key = {}
    for li, layer in enumerate(model.model.layers):
        for mod_name in ["q_proj", "v_proj"]:
            lora_module = getattr(layer.self_attn, mod_name)
            if hasattr(lora_module, "lora_a"):
                a_by_key[(li, mod_name)] = lora_module.lora_a
    return a_by_key


def save_lora_a_matrices(model) -> None:
    """Save lora_a matrices to LORA_A_PATH for use in later phases."""
    a_by_key = get_lora_a_matrices(model)
    save_dict = {
        f"layer_{li}_{mod_name}_A": np.array(v.astype(mx.float32))
        for (li, mod_name), v in a_by_key.items()
    }
    np.savez(str(LORA_A_PATH), **save_dict)
    log(f"  Saved {len(save_dict)} lora_a matrices to {LORA_A_PATH}")


def load_lora_a_matrices() -> dict:
    """Load saved lora_a matrices. Returns dict[(li, mod_name)] -> mx.array."""
    saved = np.load(str(LORA_A_PATH))
    result = {}
    for key in saved.files:
        assert key.endswith("_A"), f"Unexpected key: {key}"
        body = key[:-2]  # "layer_{li}_{mod_name}"
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    return result


def set_lora_b_matrices(model, b_by_key: dict) -> None:
    """Inject B-matrices into LoRA-wrapped model's lora_b parameters.

    b_by_key: dict[(li, mod_name)] -> mx.array shape (r, output_dims)
    Used both for eval (SFT adapter injection) and M2P eval.
    """
    for li, layer in enumerate(model.model.layers):
        for mod_name in ["q_proj", "v_proj"]:
            key = (li, mod_name)
            if key in b_by_key:
                lora_module = getattr(layer.self_attn, mod_name)
                lora_module.lora_b = b_by_key[key]


# ---- Phase 2: Evaluate base model (Fix #4, Fix #6) --------------------------

def phase_eval_base(test_examples: list) -> dict:
    """Evaluate base Qwen3-0.6B-4bit on GSM8K.

    Fix #4: max_gen_tokens=384 ensures #### answer is reachable.
    Fix #6: Assert base_acc > 0 before proceeding to training.
    """
    log("\n" + "=" * 70)
    log("[Phase 2] Evaluating base model accuracy")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    correct = 0
    total = len(test_examples)
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        if (i + 1) % max(1, total // 4) == 0 or (i + 1) == total:
            log(f"  {i+1}/{total}: acc={correct/(i+1):.3f}")
        if i == 0:
            log(f"  [DEBUG] Prompt (first 100 chars): {prompt[:100]!r}")
            log(f"  [DEBUG] Generated (first 200 chars): {generated[:200]!r}")
            log(f"  [DEBUG] Gold: {gold!r}, Pred: {pred!r}")

    accuracy = correct / total if total > 0 else 0.0
    log(f"  Base accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-base-eval")

    cleanup(model, tokenizer)

    # Fix #6: Fail-fast if evaluation pipeline is broken (catches Bug #2 class)
    if not IS_SMOKE and total >= 20:
        assert accuracy > 0.0, (
            f"FAIL-FAST K909: Base accuracy = 0.0 on {total} examples. "
            f"Evaluation pipeline broken. Check: max_gen_tokens={MAX_GEN_TOKENS}, "
            f"answer regex, tokenizer format."
        )

    return {
        "base_accuracy": accuracy,
        "base_correct": correct,
        "base_total": total,
    }


# ---- Tokenization ------------------------------------------------------------

def tokenize_texts(tokenizer, examples: list) -> list:
    """Tokenize GSM8K examples. Truncates to MAX_SEQ_LEN+1 tokens."""
    result = []
    for ex in examples:
        text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) >= 2:
                result.append(ids)
    return result


# ---- Phase 3+4: SFT LoRA training (Fix #1, #3, #5) -------------------------

def phase_sft_train(train_examples: list, model_dims: dict) -> dict:
    """Train SFT LoRA adapter on GSM8K.

    Fix #1: LoRALinear.from_base wraps q_proj/v_proj weight matrices directly.
    Fix #3: model(tokens) calls Qwen3Model.__call__ which creates causal mask.
    Fix #5: train_steps=1000, max_seq_len=512.

    Training: gradient flows through model.lora_b only.
    """
    log("\n" + "=" * 70)
    log("[Phase 4] SFT LoRA Training")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Apply LoRA — Fix #1 (no saved lora_a yet; this is the first phase)
    apply_lora_to_model(model, lora_a_dict=None)
    mx.eval(model.parameters())

    # Save lora_a matrices for consistent use in all later phases (Fix #2 cross-phase)
    save_lora_a_matrices(model)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable LoRA params (lora_b only): {trainable:,}")
    log(f"  Example B shapes: q_proj lora_b {model.model.layers[0].self_attn.q_proj.lora_b.shape}")
    log(f"  Example B shapes: v_proj lora_b {model.model.layers[0].self_attn.v_proj.lora_b.shape}")

    log(f"  Tokenizing {len(train_examples)} examples...")
    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences (max_seq_len={MAX_SEQ_LEN})")

    rng = random.Random(SEED)
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(mdl, tokens_arr):
        # Fix #3: model(tokens) -> Qwen3Model.__call__ -> create_attention_mask automatic
        logits = mdl(tokens_arr)  # (1, T, V)
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    gc.disable()
    losses = []
    for step in range(TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        loss, grads = loss_and_grad(model, tokens_arr)
        optimizer.update(model, grads)
        del grads, tokens_arr
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        if (step + 1) % 100 == 0 or (step + 1) == TRAIN_STEPS:
            recent = sum(losses[-20:]) / len(losses[-20:])
            log(f"  Step {step+1}/{TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"  Final SFT loss: {final_loss:.4f}")

    # Save lora_b matrices
    b_by_key = get_lora_b_matrices(model)
    save_dict = {}
    for (li, mod_name), B in b_by_key.items():
        save_dict[f"layer_{li}_{mod_name}_B"] = np.array(B.astype(mx.float32))
    np.savez(str(ADAPTER_PATH), **save_dict)
    log(f"  Saved {len(save_dict)} B-matrices to {ADAPTER_PATH}")
    log(f"  Phase 4 time: {time.time()-t0:.1f}s")
    log_memory("post-sft-train")

    cleanup(model, tokenizer, optimizer)
    return {"sft_final_loss": float(final_loss)}


def load_b_matrices(path: Path) -> dict:
    """Load B-matrices from npz. Returns dict[(li, mod_name)] -> mx.array."""
    saved = np.load(str(path))
    result = {}
    for key in saved.files:
        # Key format: "layer_{li}_{mod_name}_B"
        # mod_name can be "q_proj" or "v_proj" (contains underscore)
        # Strip "_B" suffix then parse "layer_{li}_{mod_name}"
        assert key.endswith("_B"), f"Unexpected key: {key}"
        body = key[:-2]  # "layer_{li}_{mod_name}"
        # "layer_0_q_proj" -> li=0, mod="q_proj"
        # Split on first two underscores: layer, li, rest
        parts = body.split("_", 2)  # ["layer", "0", "q_proj"]
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    return result


# ---- Phase 5: Evaluate SFT adapter (Fix #1, #3, #4) -------------------------

def phase_eval_sft(test_examples: list, model_dims: dict) -> dict:
    """Evaluate SFT LoRA adapter on GSM8K test.

    Fix #1 + #3: Inject B-matrices into LoRALinear; mlx_lm.generate handles masking.
    Fix #4: max_gen_tokens=384.
    """
    log("\n" + "=" * 70)
    log("[Phase 5] Evaluating SFT adapter accuracy")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Load saved lora_a to ensure consistency with SFT training (Fix #2 cross-phase)
    lora_a_dict = load_lora_a_matrices()
    apply_lora_to_model(model, lora_a_dict=lora_a_dict)
    b_by_key = load_b_matrices(ADAPTER_PATH)
    log(f"  Loaded {len(b_by_key)} B-matrices from SFT adapter")
    set_lora_b_matrices(model, b_by_key)
    mx.eval(model.parameters())

    correct = 0
    total = len(test_examples)
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        if (i + 1) % max(1, total // 4) == 0 or (i + 1) == total:
            log(f"  [SFT] {i+1}/{total}: acc={correct/(i+1):.3f}")
        if i == 0:
            log(f"  [DEBUG-SFT] Generated[:200]: {generated[:200]!r}")
            log(f"  [DEBUG-SFT] Gold: {gold!r}, Pred: {pred!r}")

    accuracy = correct / total if total > 0 else 0.0
    log(f"  SFT accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 5 time: {time.time()-t0:.1f}s")
    log_memory("post-sft-eval")

    cleanup(model, tokenizer, b_by_key)
    return {"sft_accuracy": accuracy, "sft_correct": correct}


# ---- M2P Architecture -------------------------------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: context hidden states -> LoRA B-matrices.

    Input:  (n_layers, d_model) mean-pooled hidden states per layer.
    Output: dict[(li, mod_name)] -> (lora_rank, out_dims)

    Architecture:
    1. Pool: mean across layers -> (d_model,)
    2. Encode: MLP with L_M2P layers -> (d_M2P,)
    3. Attend: query over learned memory bank -> residual -> z (d_M2P,)
    4. Generate: per-(layer, module) Linear head -> (rank * out_dims,) -> reshape

    Fix #2: q_proj_out and v_proj_out are passed explicitly from model config.
    """

    def __init__(self, n_layers: int, d_model: int, d_m2p: int, l_m2p: int,
                 n_memory: int, rank: int, q_proj_out: int, v_proj_out: int):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.d_m2p = d_m2p
        self.module_specs = [("q_proj", q_proj_out), ("v_proj", v_proj_out)]

        # Encoder: d_model -> [d_m2p*2 -> ...] -> d_m2p
        enc_layers = []
        in_dim = d_model
        for i in range(l_m2p):
            out_dim = d_m2p if i == l_m2p - 1 else d_m2p * 2
            enc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.encoder = enc_layers

        # Learned memory bank
        self.memory_bank = mx.random.normal(shape=(n_memory, d_m2p)) * 0.02
        self.attn_q = nn.Linear(d_m2p, d_m2p)

        # B-matrix generator heads: one per (layer, module) combination
        # Output: (rank * out_dims,) which reshapes to (rank, out_dims)
        self.b_heads = []
        for _li in range(n_layers):
            for _mod_name, out_f in self.module_specs:
                self.b_heads.append(nn.Linear(d_m2p, rank * out_f))

    def __call__(self, layer_hs: mx.array):
        """layer_hs: (n_layers, d_model). Returns dict[(li, mod)] -> (rank, out_f)."""
        # Pool and encode
        h = mx.mean(layer_hs, axis=0)  # (d_model,)
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i < len(self.encoder) - 1:
                h = nn.relu(h)
        # h: (d_m2p,)

        # Memory attention
        q = self.attn_q(h)
        scores = (self.memory_bank @ q) / math.sqrt(self.d_m2p)
        w = mx.softmax(scores, axis=0)
        context = (w[:, None] * self.memory_bank).sum(axis=0)
        z = h + context  # (d_m2p,)

        # Generate B-matrices for all (layer, module) pairs
        b_by_key = {}
        head_idx = 0
        for li in range(self.n_layers):
            for mod_name, out_f in self.module_specs:
                b_flat = self.b_heads[head_idx](z)
                b_by_key[(li, mod_name)] = b_flat.reshape(self.rank, out_f)
                head_idx += 1
        return b_by_key


# ---- Phase 6: M2P training ---------------------------------------------------

def phase_m2p_train(train_examples: list, model_dims: dict) -> dict:
    """Train M2P hypernetwork to generate LoRA B-matrices from context.

    Key design: M2P is the trainable module. The Qwen3 model is frozen (except
    for lora_b which M2P generates). We set lora_b = M2P(hidden_states) before
    the model forward, then compute NTP loss.

    Since MLX value_and_grad traces through M2P parameters -> b_by_key -> model
    forward -> loss, gradients flow correctly through:
      M2P parameters -> generated B-matrices -> LoRA computation inside q/v projections

    Fix #1: B-matrices go into LoRALinear.lora_b, not residual stream.
    Fix #2: B-matrix shapes from model_dims (GQA-aware).
    Fix #3: model(tokens) uses create_attention_mask automatically.
    Fix #5: train_steps=1000.

    Note on MLX and mutable state:
    We use nn.value_and_grad(m2p, m2p_loss_fn) where m2p_loss_fn:
    1. Calls m2p(layer_hs) to get b_by_key (gradients flow through m2p)
    2. Calls set_lora_b_matrices(model, b_by_key) — this mutates model.lora_b
    3. Calls model(tokens) — this uses the mutated lora_b in computation

    In MLX, the gradient through the mutation works because MLX traces lazy
    computations: model(tokens) with mutated lora_b will have lora_b in its
    computation graph, and lora_b was produced by m2p, so gradients flow.
    """
    log("\n" + "=" * 70)
    log("[Phase 6] M2P Hypernetwork Training")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Load saved lora_a from SFT phase (Fix #2 cross-phase: same A-matrices as SFT)
    lora_a_dict = load_lora_a_matrices()

    # Apply LoRA structure with saved lora_a; freeze everything
    # During M2P training, lora_b is set by M2P output (not optimized directly)
    # We still use apply_lora_to_model to set up the LoRALinear structure,
    # then freeze all model params (including lora_b which M2P will override)
    apply_lora_to_model(model, lora_a_dict=lora_a_dict)
    model.freeze()  # Re-freeze lora_b too — M2P sets it via mutation, not grad
    # lora_b starts as zero for all layers — M2P will generate the values
    mx.eval(model.parameters())

    # Hidden state extraction via full model forward (Fix #3)
    # We run model.model(tokens) which is Qwen3Model.__call__ with causal mask
    def extract_hidden_states(tokens_arr: mx.array) -> mx.array:
        """Extract per-layer mean-pooled hidden states.

        Uses create_attention_mask from Qwen3Model.__call__ internals.
        We replicate the Qwen3Model loop manually to capture per-layer outputs,
        using the same mask creation logic (Fix #3).
        """
        from mlx_lm.models.base import create_attention_mask

        h = model.model.embed_tokens(tokens_arr)  # (1, T, d_model)
        # Qwen3Model creates mask with cache=None -> same as create_attention_mask(h, None)
        # For T>1: returns "causal" which mlx fast SDP applies as causal mask
        mask = create_attention_mask(h, None)

        layer_states = []
        for layer in model.model.layers:
            h = layer(h, mask=mask, cache=None)  # (1, T, d_model)
            layer_states.append(mx.mean(h[0], axis=0))  # (d_model,)
        return mx.stack(layer_states, axis=0)  # (n_layers, d_model)

    # M2P network
    m2p = M2PNetwork(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P, l_m2p=L_M2P,
        n_memory=N_MEMORY, rank=LORA_RANK,
        q_proj_out=q_proj_out, v_proj_out=v_proj_out,
    )
    mx.eval(m2p.parameters())
    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  M2P params: {n_params:,}")
    log(f"  B shapes: q_proj=({LORA_RANK},{q_proj_out}), v_proj=({LORA_RANK},{v_proj_out})")

    log(f"  Tokenizing {len(train_examples)} examples...")
    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences (max_seq_len={MAX_SEQ_LEN})")

    rng = random.Random(SEED + 1)
    optimizer = optim.Adam(learning_rate=LR)

    def m2p_loss_fn(m2p_net, tokens_arr):
        # 1. Extract hidden states (no grad through frozen base)
        layer_hs = mx.stop_gradient(extract_hidden_states(tokens_arr))

        # 2. Generate B-matrices from context (grad flows through m2p_net)
        b_by_key = m2p_net(layer_hs)

        # 3. Set lora_b in all layers to M2P-generated values
        #    MLX lazy eval: model(tokens) will reference these values in its graph
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = b_by_key[(li, "q_proj")]
            layer.self_attn.v_proj.lora_b = b_by_key[(li, "v_proj")]

        # 4. Run model forward (Fix #3: causal mask applied automatically)
        logits = model(tokens_arr)  # (1, T, V)

        # 5. NTP loss
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    gc.disable()
    losses = []
    for step in range(TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        loss, grads = loss_and_grad(m2p, tokens_arr)
        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        if (step + 1) % 100 == 0 or (step + 1) == TRAIN_STEPS:
            recent = sum(losses[-20:]) / len(losses[-20:])
            log(f"  Step {step+1}/{TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"  Final M2P loss: {final_loss:.4f}")

    # Save M2P weights
    m2p_params = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params.items()}
    np.savez(str(M2P_PATH), **m2p_save)
    log(f"  Saved M2P to {M2P_PATH}")
    log(f"  Phase 6 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")

    cleanup(m2p, optimizer, model, tokenizer)
    return {"m2p_final_loss": float(final_loss), "m2p_params": n_params}


# ---- Phase 7: Evaluate M2P adapter -------------------------------------------

def phase_eval_m2p(test_examples: list, model_dims: dict) -> dict:
    """Evaluate M2P-generated adapter on GSM8K test.

    For each test example:
    1. Extract hidden states from prompt (full causal forward)
    2. Run M2P to generate B-matrices
    3. Inject B-matrices into LoRALinear modules
    4. Generate answer via mlx_lm.generate (causal, max_tokens=384)

    Fix #1: B-matrices injected into LoRALinear.lora_b (weight-space LoRA).
    Fix #2: B-matrix shapes from model_dims.
    Fix #3: mlx_lm.generate -> model(tokens) -> causal mask automatic.
    Fix #4: MAX_GEN_TOKENS=384.
    """
    log("\n" + "=" * 70)
    log("[Phase 7] Evaluating M2P adapter accuracy")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Load saved lora_a (Fix #2 cross-phase: same A as SFT and M2P training)
    lora_a_dict = load_lora_a_matrices()
    apply_lora_to_model(model, lora_a_dict=lora_a_dict)
    model.eval()

    # Load M2P
    m2p_saved = np.load(str(M2P_PATH))
    m2p = M2PNetwork(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P, l_m2p=L_M2P,
        n_memory=N_MEMORY, rank=LORA_RANK,
        q_proj_out=q_proj_out, v_proj_out=v_proj_out,
    )
    weight_list = [(k, mx.array(m2p_saved[k])) for k in m2p_saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    log(f"  Loaded M2P from {M2P_PATH}")

    # Hidden state extraction (same as training, Fix #3)
    from mlx_lm.models.base import create_attention_mask

    def extract_hidden_states(tokens_arr: mx.array) -> mx.array:
        h = model.model.embed_tokens(tokens_arr)
        mask = create_attention_mask(h, None)
        layer_states = []
        for layer in model.model.layers:
            h = layer(h, mask=mask, cache=None)
            layer_states.append(mx.mean(h[0], axis=0))
        return mx.stack(layer_states, axis=0)

    correct = 0
    total = len(test_examples)
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # 1. Extract hidden states
        layer_hs = extract_hidden_states(tokens_arr)
        mx.eval(layer_hs)

        # 2. Generate B-matrices
        b_by_key = m2p(layer_hs)
        mx.eval(list(b_by_key.values()))

        # 3. Inject into LoRALinear modules (Fix #1)
        set_lora_b_matrices(model, b_by_key)
        mx.eval(model.parameters())

        # 4. Generate via mlx_lm.generate (Fix #3, Fix #4)
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            correct += 1

        del tokens_arr, layer_hs, b_by_key
        if (i + 1) % max(1, total // 4) == 0 or (i + 1) == total:
            log(f"  [M2P] {i+1}/{total}: acc={correct/(i+1):.3f}")
        if i == 0:
            log(f"  [DEBUG-M2P] Generated[:200]: {generated[:200]!r}")
            log(f"  [DEBUG-M2P] Gold: {gold!r}, Pred: {pred!r}")

    accuracy = correct / total if total > 0 else 0.0
    log(f"  M2P accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 7 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-eval")

    cleanup(m2p, model, tokenizer)
    return {"m2p_accuracy": accuracy, "m2p_correct": correct}


# ---- Main -------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P on Qwen3-0.6B + GSM8K v2 — All 6 Bugs Fixed")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN={N_TRAIN} | N_TEST={N_TEST} | STEPS={TRAIN_STEPS}")
    log(f"MAX_SEQ_LEN={MAX_SEQ_LEN} | MAX_GEN_TOKENS={MAX_GEN_TOKENS}")
    log(f"LORA_RANK={LORA_RANK} | LORA_SCALE={LORA_SCALE} | D_M2P={D_M2P}")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Load data
    train_examples, test_examples = phase_load_data()
    log_memory("after-data")

    # Phase 2: Base model accuracy + fail-fast (Fix #4, #6)
    base_results = phase_eval_base(test_examples)
    log_memory("after-base-eval")

    # Inspect model dims (Fix #2)
    log("\n[Inspecting model dims]")
    model_tmp, _tok_tmp = mlx_load(MODEL_ID)
    model_dims = get_model_dims(model_tmp)
    cleanup(model_tmp, _tok_tmp)
    log_memory("after-dim-check")

    # Phase 4: SFT LoRA training (Fix #1, #3, #5)
    sft_train_results = phase_sft_train(train_examples, model_dims)
    log_memory("after-sft-train")

    # Phase 5: Evaluate SFT adapter (Fix #1, #3, #4)
    sft_eval_results = phase_eval_sft(test_examples, model_dims)
    log_memory("after-sft-eval")

    # Phase 6: M2P training (Fix #1, #2, #3, #5)
    m2p_train_results = phase_m2p_train(train_examples, model_dims)
    log_memory("after-m2p-train")

    # Phase 7: Evaluate M2P (Fix #1, #2, #3, #4)
    m2p_eval_results = phase_eval_m2p(test_examples, model_dims)
    log_memory("after-m2p-eval")

    # Kill criteria assessment
    base_acc = base_results["base_accuracy"]
    sft_acc = sft_eval_results["sft_accuracy"]
    m2p_acc = m2p_eval_results["m2p_accuracy"]

    sft_improvement = sft_acc - base_acc
    m2p_improvement = m2p_acc - base_acc
    quality_ratio = (
        m2p_improvement / sft_improvement
        if abs(sft_improvement) > 1e-9
        else 0.0
    )

    k909_pass = base_acc > 0.0
    k910_pass = sft_improvement >= 0.05
    k911_pass = quality_ratio >= 0.70
    k912_kill = quality_ratio < 0.30

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "m2p_qwen06b_gsm8k_v2",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P,
            "l_m2p": L_M2P,
            "n_memory": N_MEMORY,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "train_steps": TRAIN_STEPS,
            "lr": LR,
            "max_seq_len": MAX_SEQ_LEN,
            "max_gen_tokens": MAX_GEN_TOKENS,
            **model_dims,
        },
        **base_results,
        **sft_train_results,
        **sft_eval_results,
        **m2p_train_results,
        **m2p_eval_results,
        "sft_improvement": round(sft_improvement, 4),
        "m2p_improvement": round(m2p_improvement, 4),
        "quality_ratio": round(quality_ratio, 4),
        "kill_criteria": {
            "K909_base_gt_0pct": "PASS" if k909_pass else "FAIL",
            "K910_sft_gain_ge_5pp": "PASS" if k910_pass else "FAIL",
            "K911_quality_ratio_ge_70pct": "PASS" if k911_pass else "FAIL",
            "K912_KILL_ratio_lt_30pct": "TRIGGERED" if k912_kill else "not_triggered",
            "base_accuracy": round(base_acc, 4),
            "sft_accuracy": round(sft_acc, 4),
            "m2p_accuracy": round(m2p_acc, 4),
            "sft_improvement_pp": round(sft_improvement * 100, 2),
            "quality_ratio": round(quality_ratio, 4),
        },
        "peak_memory_gb": round(peak_gb, 2),
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Base accuracy:   {base_acc:.4f}")
    log(f"  SFT accuracy:    {sft_acc:.4f} ({sft_improvement:+.4f} = {sft_improvement*100:+.1f}pp)")
    log(f"  M2P accuracy:    {m2p_acc:.4f} ({m2p_improvement:+.4f} = {m2p_improvement*100:+.1f}pp)")
    log(f"  Quality ratio:   {quality_ratio:.4f} ({quality_ratio*100:.1f}%)")
    log(f"  Peak memory:     {peak_gb:.1f} GB")
    log(f"  Total time:      {results['total_time_s']:.0f}s")
    log("")
    log(f"  K909 (base > 0%):          {'PASS' if k909_pass else 'FAIL'}")
    log(f"  K910 (sft_gain >= 5pp):    {'PASS' if k910_pass else 'FAIL'}")
    log(f"  K911 (ratio >= 70%):       {'PASS' if k911_pass else 'FAIL'}")
    log(f"  K912 KILL (ratio < 30%):   {'TRIGGERED' if k912_kill else 'not triggered'}")

    if IS_SMOKE:
        log("\n  [SMOKE TEST] Not statistically meaningful.")
        log("  Smoke success = no crash + text generated + answer regex finds at least 1 match.")


if __name__ == "__main__":
    main()
