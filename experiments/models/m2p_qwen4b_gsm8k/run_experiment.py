#!/usr/bin/env python3
"""M2P on Qwen3-4B + GSM8K: full-scale adapter generation on production model.

Scales the M2P (Map-to-Parameters) mechanism from Qwen3-0.6B (Finding #378)
to Qwen3-4B. Key question: does the hidden-state-to-adapter mapping transfer
to a larger model with different dimensions?

Kill criteria:
  K937: M2P quality >= 60% of SFT on GSM8K
        quality_ratio = (M2P_acc - base_acc) / (SFT_acc - base_acc) >= 0.60
  K938: Adapter generation < 100ms (one-time M2P forward per prompt)
  K939: KILL — M2P quality < 20% improvement over base (approach fails at 4B)

Qwen3-4B dimensions (verified from config.json):
  d_model=2560, n_layers=36, n_heads=32, n_kv_heads=8, head_dim=128
  q_proj_out=4096 (32×128), v_proj_out=1024 (8×128)

M2P architecture (bottleneck, D_M2P=1024):
  enc: 2560→2048→1024 (GELU)
  b_heads_q[36]: 1024→rank×4096
  b_heads_v[36]: 1024→rank×1024
  Total M2P params ≈ 760M (11.9% overhead on 4B)

References:
  Ha et al. (arXiv:1609.09106) — HyperNetworks
  SHINE (arXiv:2602.06358) — functional LoRA forward
  Finding #378 (exp_m2p_qwen06b_gsm8k_v4) — proven recipe on 0.6B

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
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config ------------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-4B-4bit"

# Qwen3-4B dimensions (from config.json)
D_MODEL = 2560
N_LAYERS = 36
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
Q_PROJ_OUT = N_HEADS * HEAD_DIM       # 4096
V_PROJ_OUT = N_KV_HEADS * HEAD_DIM   # 1024

# LoRA config
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P config
D_M2P = 1024        # bottleneck (Theorem 1: sufficient for d_intrinsic≈86)
OUTPUT_SCALE = 0.032  # SHINE sqrt(0.001) convention

# Training budget (keep within 2hr wall-clock)
N_TRAIN = 30 if IS_SMOKE else 500
N_TEST = 10 if IS_SMOKE else 100
SFT_TRAIN_STEPS = 10 if IS_SMOKE else 200
M2P_TRAIN_STEPS = 10 if IS_SMOKE else 300
LR = 5e-5
LR_WARMUP = 3 if IS_SMOKE else 30
MAX_SEQ_LEN = 64 if IS_SMOKE else 512
MAX_GEN_TOKENS = 32 if IS_SMOKE else 384
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
LORA_A_PATH = EXPERIMENT_DIR / "grassmannian_a_matrices.npz"
SFT_B_PATH = EXPERIMENT_DIR / "sft_b_matrices.npz"
M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

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
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"(?:the\s+)?answer\s+is\s+[-–]?\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, center - half), min(1.0, center + half))


# ---- Grassmannian A-matrix init ----------------------------------------------

def init_grassmannian_a_matrices() -> dict:
    """Initialize Grassmannian A-matrices via QR decomposition.

    Each layer gets an orthonormal A-matrix for q_proj and v_proj.
    Shape: A_q[li] ∈ R^{D_MODEL × LORA_RANK}, A_v[li] ∈ R^{D_MODEL × LORA_RANK}
    (LoRA convention: A has shape (input_dims, rank))
    """
    rng = np.random.default_rng(SEED)
    result = {}
    for li in range(N_LAYERS):
        for mod_name, in_dim in [("q_proj", D_MODEL), ("v_proj", D_MODEL)]:
            # QR gives orthonormal columns: A_q^T A_q = I_r
            M = rng.standard_normal((in_dim, LORA_RANK)).astype(np.float32)
            Q, _ = np.linalg.qr(M)
            result[(li, mod_name)] = mx.array(Q[:, :LORA_RANK]).astype(mx.bfloat16)
    return result


def save_lora_a_matrices(lora_a_dict: dict) -> None:
    save_dict = {}
    for (li, mod_name), A in lora_a_dict.items():
        save_dict[f"layer_{li}_{mod_name}_A"] = np.array(A.astype(mx.float32))
    np.savez(str(LORA_A_PATH), **save_dict)
    log(f"  Saved {len(save_dict)} A-matrices to {LORA_A_PATH}")


def load_lora_a_matrices() -> dict:
    saved = np.load(str(LORA_A_PATH))
    result = {}
    for key in saved.files:
        assert key.endswith("_A"), f"Unexpected key: {key}"
        body = key[:-2]
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} A-matrices from {LORA_A_PATH}")
    return result


# ---- LoRA structure application ----------------------------------------------

def apply_lora_to_model(model, lora_a_dict: dict) -> None:
    """Wrap q_proj/v_proj with LoRALinear (handles QuantizedLinear for 4-bit)."""
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    model.freeze()


# ---- Functional LoRA forward (proven design from v3/v4) ----------------------

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    """LoRA projection with B as a tensor argument.

    y = linear_module(x) + scale * (x @ A) @ B
    B is in the computation graph → gradients flow to M2P.
    """
    y = linear_module(x)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)
    return y + (scale * z).astype(x.dtype)


def functional_attention_forward(
    attn,
    x: mx.array,
    B_q: mx.array,
    B_v: mx.array,
    A_q: mx.array,
    A_v: mx.array,
    lora_scale: float,
    mask,
    cache=None,
) -> mx.array:
    """Functional attention with B_q, B_v as tensor arguments.

    Replicates Qwen3 Attention.__call__ with functional LoRA for q_proj/v_proj.
    The base q_proj and v_proj are accessed via attn.q_proj.linear (frozen base).
    """
    B_batch, L, _ = x.shape

    q = functional_lora_proj(x, attn.q_proj.linear, A_q, B_q, lora_scale)
    k = attn.k_proj(x)
    v = functional_lora_proj(x, attn.v_proj.linear, A_v, B_v, lora_scale)

    queries = attn.q_norm(q.reshape(B_batch, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    keys = attn.k_norm(k.reshape(B_batch, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = v.reshape(B_batch, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = attn.rope(queries)
        keys = attn.rope(keys)

    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=attn.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B_batch, L, -1)
    return attn.o_proj(output)


def model_forward_with_loras(
    model,
    tokens_arr: mx.array,
    B_q_layers: list,
    B_v_layers: list,
    A_q_layers: list,
    A_v_layers: list,
    lora_scale: float = LORA_SCALE,
) -> mx.array:
    """Full Qwen3-4B forward with functional LoRA. Returns logits."""
    qwen3_model = model.model
    h = qwen3_model.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)

    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn_out = functional_attention_forward(
            attn=layer.self_attn,
            x=normed,
            B_q=B_q_layers[li],
            B_v=B_v_layers[li],
            A_q=A_q_layers[li],
            A_v=A_v_layers[li],
            lora_scale=lora_scale,
            mask=mask,
            cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    h = qwen3_model.norm(h)
    if model.args.tie_word_embeddings:
        logits = qwen3_model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    return logits


def extract_hidden_states(
    model,
    tokens_arr: mx.array,
    A_q_layers: list,
    A_v_layers: list,
    B_q_zero: list,
    B_v_zero: list,
) -> mx.array:
    """Extract per-layer mean-pooled hidden states (stop_gradient)."""
    qwen3_model = model.model
    h = qwen3_model.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)

    layer_states = []
    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn_out = functional_attention_forward(
            attn=layer.self_attn,
            x=normed,
            B_q=B_q_zero[li],
            B_v=B_v_zero[li],
            A_q=A_q_layers[li],
            A_v=A_v_layers[li],
            lora_scale=0.0,  # base model only
            mask=mask,
            cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))

    return mx.stop_gradient(mx.stack(layer_states, axis=0))


# ---- M2P Architecture --------------------------------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: hidden states → LoRA B-matrices.

    Qwen3-4B adaptation of v4 design:
    - Bottleneck encoder: 2560 → 2048 → 1024
    - Per-layer B-heads: 1024 → rank × 4096 (q), 1024 → rank × 1024 (v)
    """

    def __init__(
        self,
        n_layers: int = N_LAYERS,
        d_model: int = D_MODEL,
        d_m2p: int = D_M2P,
        rank: int = LORA_RANK,
        q_proj_out: int = Q_PROJ_OUT,
        v_proj_out: int = V_PROJ_OUT,
        output_scale: float = OUTPUT_SCALE,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.output_scale = output_scale

        # Bottleneck encoder: d_model → 2*d_m2p → d_m2p
        self.enc_linear1 = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2 = nn.Linear(2 * d_m2p, d_m2p)

        # B-matrix generator heads
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out) for _ in range(n_layers)]

    def __call__(self, layer_hs: mx.array):
        """layer_hs: (n_layers, d_model) → B_q_layers, B_v_layers."""
        h = mx.mean(layer_hs, axis=0)          # (d_model,)
        h = nn.gelu(self.enc_linear1(h))       # (2*d_m2p,)
        z = self.enc_linear2(h)                # (d_m2p,)

        B_q_layers = []
        B_v_layers = []
        for li in range(self.n_layers):
            b_q = self.b_heads_q[li](z).reshape(self.rank, -1) * self.output_scale
            b_v = self.b_heads_v[li](z).reshape(self.rank, -1) * self.output_scale
            B_q_layers.append(b_q)
            B_v_layers.append(b_v)

        return B_q_layers, B_v_layers


# ---- Tokenization ------------------------------------------------------------

def tokenize_texts(tokenizer, examples: list) -> list:
    result = []
    for ex in examples:
        text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) >= 2:
                result.append(ids)
    return result


# ---- Phase 0: Load data ------------------------------------------------------

def phase_load_data():
    log("\n" + "=" * 70)
    log("[Phase 0] Loading GSM8K data")
    log("=" * 70)
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
    return train_examples, test_examples


# ---- Phase 1: Base eval ------------------------------------------------------

def phase_eval_base(test_examples: list) -> dict:
    """Evaluate Qwen3-4B-4bit without any adapter."""
    log("\n" + "=" * 70)
    log("[Phase 1] Base model eval (no adapter)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    log_memory("base-model-loaded")

    correct = 0
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if gold and pred and gold.strip() == pred.strip():
            correct += 1
        if (i + 1) % 20 == 0:
            log(f"  [{i+1}/{len(test_examples)}] acc={correct/(i+1):.3f}")

    acc = correct / len(test_examples)
    ci_lo, ci_hi = wilson_ci(correct, len(test_examples))
    log(f"  Base accuracy: {acc:.3f} ({correct}/{len(test_examples)}), "
        f"Wilson 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    log(f"  Phase 1 time: {time.time()-t0:.1f}s")

    cleanup(model, tokenizer)
    return {"base_accuracy": acc, "base_correct": correct, "base_n": len(test_examples)}


# ---- Phase 2: SFT LoRA training ----------------------------------------------

def phase_sft_train(train_examples: list) -> dict:
    """Train SFT LoRA adapter on GSM8K (provides K937 comparison baseline)."""
    log("\n" + "=" * 70)
    log(f"[Phase 2] SFT LoRA Training ({SFT_TRAIN_STEPS} steps)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Initialize Grassmannian A-matrices (first time) and apply LoRA
    lora_a_dict = init_grassmannian_a_matrices()
    apply_lora_to_model(model, lora_a_dict)
    save_lora_a_matrices(lora_a_dict)
    mx.eval(model.parameters())

    # Unfreeze only lora_b for SFT
    model.unfreeze(keys=["lora_b"], strict=False)
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable LoRA-B params: {trainable:,}")

    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences")

    rng = random.Random(SEED)
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(mdl, tokens_arr):
        logits = mdl(tokens_arr)
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    gc.disable()
    losses = []
    for step in range(SFT_TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        loss, grads = loss_and_grad(model, tokens_arr)
        optimizer.update(model, grads)
        del grads, tokens_arr
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, SFT_TRAIN_STEPS // 5) == 0:
            recent = sum(losses[-20:]) / min(len(losses[-20:]), 20)
            log(f"  Step {step+1}/{SFT_TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"  Final SFT loss: {final_loss:.4f}")

    # Save B-matrices
    save_dict = {}
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        save_dict[f"layer_{li}_q_proj_B"] = np.array(
            attn.q_proj.lora_b.astype(mx.float32)
        )
        save_dict[f"layer_{li}_v_proj_B"] = np.array(
            attn.v_proj.lora_b.astype(mx.float32)
        )
    np.savez(str(SFT_B_PATH), **save_dict)
    log(f"  Saved {len(save_dict)} SFT B-matrices to {SFT_B_PATH}")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-sft-train")

    cleanup(model, tokenizer, optimizer)
    return {"sft_final_loss": float(final_loss), "sft_initial_loss": float(losses[0])}


# ---- Phase 3: SFT eval -------------------------------------------------------

def phase_eval_sft(test_examples: list) -> dict:
    """Evaluate SFT adapter on GSM8K."""
    log("\n" + "=" * 70)
    log("[Phase 3] SFT adapter eval")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    apply_lora_to_model(model, lora_a_dict)

    # Load SFT B-matrices
    sft_b = np.load(str(SFT_B_PATH))
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        q_key = f"layer_{li}_q_proj_B"
        v_key = f"layer_{li}_v_proj_B"
        if q_key in sft_b:
            attn.q_proj.lora_b = mx.array(sft_b[q_key]).astype(mx.bfloat16)
            attn.v_proj.lora_b = mx.array(sft_b[v_key]).astype(mx.bfloat16)
    mx.eval(model.parameters())
    log("  Loaded SFT B-matrices")

    correct = 0
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if gold and pred and gold.strip() == pred.strip():
            correct += 1
        if (i + 1) % 20 == 0:
            log(f"  [{i+1}/{len(test_examples)}] acc={correct/(i+1):.3f}")

    acc = correct / len(test_examples)
    ci_lo, ci_hi = wilson_ci(correct, len(test_examples))
    log(f"  SFT accuracy: {acc:.3f} ({correct}/{len(test_examples)}), "
        f"Wilson 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    log(f"  Phase 3 time: {time.time()-t0:.1f}s")

    cleanup(model, tokenizer)
    return {"sft_accuracy": acc, "sft_correct": correct}


# ---- Phase 4: M2P training ---------------------------------------------------

def phase_m2p_train(train_examples: list) -> dict:
    """Train M2P hypernetwork on Qwen3-4B."""
    log("\n" + "=" * 70)
    log(f"[Phase 4] M2P Training ({M2P_TRAIN_STEPS} steps)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    apply_lora_to_model(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(N_LAYERS)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(N_LAYERS)]
    B_q_zero = [mx.zeros((LORA_RANK, Q_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]
    B_v_zero = [mx.zeros((LORA_RANK, V_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]

    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences")

    m2p = M2PNetwork()
    mx.eval(m2p.parameters())
    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  M2P params: {n_params:,}")
    log_memory("m2p-init")

    rng = random.Random(SEED + 1)

    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    optimizer = optim.Adam(learning_rate=LR)

    def m2p_loss_fn(m2p_net, tokens_arr):
        layer_hs = extract_hidden_states(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )
        B_q_layers, B_v_layers = m2p_net(layer_hs)
        logits = model_forward_with_loras(
            model, tokens_arr, B_q_layers, B_v_layers,
            A_q_layers, A_v_layers, LORA_SCALE,
        )
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # K937-analog: gradient smoke test
    log("\n  [K_grad] Gradient smoke test...")
    smoke_seq = rng.choice(tokenized)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)

    grad_norms = [float(mx.sum(g ** 2).item())
                  for _, g in tree_flatten(smoke_grads) if isinstance(g, mx.array)]
    grad_norm = math.sqrt(sum(grad_norms))
    initial_loss = float(smoke_loss.item())

    log(f"  grad_norm at step 0: {grad_norm:.4f}")
    log(f"  initial_loss: {initial_loss:.4f}")

    k_grad_pass = grad_norm > 0.0
    if not k_grad_pass:
        log("  KILL: zero gradients — Theorem 2 violated")
        results = {
            "experiment": "m2p_qwen4b_gsm8k",
            "model": MODEL_ID,
            "is_smoke": IS_SMOKE,
            "k937_pass": False,
            "kill_reason": "zero gradients at step 0",
            "grad_norm_step0": 0.0,
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        cleanup(m2p, model, tokenizer, optimizer)
        return results

    del smoke_tokens, smoke_loss, smoke_grads

    # Full M2P training
    log(f"\n  Training {M2P_TRAIN_STEPS} steps...")
    gc.disable()
    losses = []
    for step in range(M2P_TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        optimizer.learning_rate = lr_schedule(step)
        loss, grads = loss_and_grad(m2p, tokens_arr)
        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, M2P_TRAIN_STEPS // 5) == 0:
            recent = sum(losses[-20:]) / min(len(losses[-20:]), 20)
            log(f"  Step {step+1}/{M2P_TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"  Final M2P loss: {final_loss:.4f}")

    # Save M2P weights
    m2p_params = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params.items()}
    np.savez(str(M2P_PATH), **m2p_save)
    log(f"  Saved M2P to {M2P_PATH}")
    log(f"  Phase 4 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")

    cleanup(m2p, model, tokenizer, optimizer)
    return {
        "m2p_params": n_params,
        "m2p_initial_loss": initial_loss,
        "m2p_final_loss": float(final_loss),
        "grad_norm_step0": grad_norm,
        "k_grad_pass": k_grad_pass,
    }


# ---- Phase 5: M2P eval -------------------------------------------------------

def phase_eval_m2p(test_examples: list) -> dict:
    """Evaluate M2P-generated adapter on GSM8K + measure K938 (gen time)."""
    log("\n" + "=" * 70)
    log(f"[Phase 5] M2P adapter eval (n={len(test_examples)})")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    apply_lora_to_model(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(N_LAYERS)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(N_LAYERS)]
    B_q_zero = [mx.zeros((LORA_RANK, Q_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]
    B_v_zero = [mx.zeros((LORA_RANK, V_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]

    # Load M2P
    m2p = M2PNetwork()
    m2p_saved = np.load(str(M2P_PATH))
    weight_list = [(k, mx.array(m2p_saved[k])) for k in m2p_saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    log("  Loaded M2P weights")
    log_memory("m2p-loaded")

    # K938: measure adapter generation time (M2P forward)
    t_gen_start = time.perf_counter()
    # Use a dummy input to warm up and measure M2P forward time
    dummy_seq = [1, 2, 3, 4, 5, 6, 7, 8]
    dummy_arr = mx.array(dummy_seq)[None, :]
    for _ in range(3):  # warm up
        hs = extract_hidden_states(model, dummy_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero)
        mx.eval(hs)
        B_q, B_v = m2p(hs)
        mx.eval(*B_q, *B_v)
    # Timed run
    t_m2p_start = time.perf_counter()
    for _ in range(5):
        hs = extract_hidden_states(model, dummy_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero)
        mx.eval(hs)
        B_q, B_v = m2p(hs)
        mx.eval(*B_q, *B_v)
    t_m2p_ms = (time.perf_counter() - t_m2p_start) / 5 * 1000
    log(f"  K938: M2P adapter generation time = {t_m2p_ms:.1f}ms (5-run avg)")
    k938_pass = t_m2p_ms < 100.0
    del dummy_arr, hs, B_q, B_v

    # Full GSM8K eval with M2P adapter
    correct = 0
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # Extract hidden states
        layer_hs = extract_hidden_states(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )
        mx.eval(layer_hs)

        # Generate B-matrices
        B_q_layers, B_v_layers = m2p(layer_hs)
        mx.eval(*B_q_layers, *B_v_layers)

        # Inject into model for generation
        for li, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            attn.q_proj.lora_b = B_q_layers[li]
            attn.v_proj.lora_b = B_v_layers[li]
        mx.eval(model.parameters())

        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )

        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if gold and pred and gold.strip() == pred.strip():
            correct += 1

        if (i + 1) % 20 == 0:
            log(f"  [{i+1}/{len(test_examples)}] acc={correct/(i+1):.3f}")

    acc = correct / len(test_examples)
    ci_lo, ci_hi = wilson_ci(correct, len(test_examples))
    log(f"  M2P accuracy: {acc:.3f} ({correct}/{len(test_examples)}), "
        f"Wilson 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    log(f"  K938: {'PASS' if k938_pass else 'FAIL'} gen_time={t_m2p_ms:.1f}ms")
    log(f"  Phase 5 time: {time.time()-t0:.1f}s")

    cleanup(model, tokenizer, m2p)
    return {
        "m2p_accuracy": acc,
        "m2p_correct": correct,
        "m2p_n": len(test_examples),
        "m2p_ci_lo": ci_lo,
        "m2p_ci_hi": ci_hi,
        "k938_gen_time_ms": t_m2p_ms,
        "k938_pass": k938_pass,
    }


# ---- Main --------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P on Qwen3-4B + GSM8K")
    log(f"Model: {MODEL_ID}")
    log(f"Smoke: {IS_SMOKE} | M2P steps: {M2P_TRAIN_STEPS} | N_TEST: {N_TEST}")
    log("=" * 70)

    # Phases
    train_examples, test_examples = phase_load_data()
    base_results = phase_eval_base(test_examples)
    sft_train_results = phase_sft_train(train_examples)
    sft_eval_results = phase_eval_sft(test_examples)
    m2p_train_results = phase_m2p_train(train_examples)
    m2p_eval_results = phase_eval_m2p(test_examples)

    # Kill criteria
    base_acc = base_results["base_accuracy"]
    sft_acc = sft_eval_results["sft_accuracy"]
    m2p_acc = m2p_eval_results["m2p_accuracy"]

    denom = sft_acc - base_acc
    if abs(denom) > 1e-6:
        quality_ratio = (m2p_acc - base_acc) / denom
    else:
        quality_ratio = None

    k937_pass = quality_ratio is not None and quality_ratio >= 0.60
    k939_kill = quality_ratio is not None and quality_ratio < 0.20

    log("\n" + "=" * 70)
    log("[KILL CRITERIA RESULTS]")
    log("=" * 70)
    log(f"  base_acc = {base_acc:.3f}")
    log(f"  sft_acc  = {sft_acc:.3f}")
    log(f"  m2p_acc  = {m2p_acc:.3f}")
    log(f"  quality_ratio = {quality_ratio}")
    log(f"  K937 (quality >= 60% SFT): {'PASS' if k937_pass else 'FAIL'}")
    log(f"  K938 (gen < 100ms): {'PASS' if m2p_eval_results['k938_pass'] else 'FAIL'}")
    log(f"  K939 (KILL if quality < 20%): {'KILL' if k939_kill else 'OK'}")
    log(f"  grad_norm_step0 = {m2p_train_results.get('grad_norm_step0', 0):.4f}")

    total_time = time.time() - t_start
    log(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    results = {
        "experiment": "m2p_qwen4b_gsm8k",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "n_kv_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "q_proj_out": Q_PROJ_OUT,
            "v_proj_out": V_PROJ_OUT,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P,
            "output_scale": OUTPUT_SCALE,
            "sft_steps": SFT_TRAIN_STEPS,
            "m2p_steps": M2P_TRAIN_STEPS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
        },
        **base_results,
        **sft_train_results,
        **sft_eval_results,
        **m2p_train_results,
        **m2p_eval_results,
        "quality_ratio": quality_ratio,
        "k937_pass": k937_pass,
        "k939_kill": k939_kill,
        "all_criteria_pass": k937_pass and m2p_eval_results["k938_pass"] and not k939_kill,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()
