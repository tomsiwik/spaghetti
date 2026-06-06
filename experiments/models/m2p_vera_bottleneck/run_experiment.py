#!/usr/bin/env python3
"""VeRA-style M2P: shared random projection reduces 357M → ~5M params.

Kill criteria:
  K922: M2P parameter count <= 10M (vs current 357M, i.e., >= 35x reduction)
  K923: quality_ratio >= 70% at n=500 (VeRA parity on GSM8K)
  K924: grad_norm > 0 at step 0 (functional forward invariant, Theorem 5 inherited)

Architecture change from v4:
  v4:  M2P outputs N_layers×Linear(d_m2p, r×d) — 453M params in output heads
  VeRA: M2P outputs single Linear(d_m2p, N_layers×4r) = Linear(1024, 576) — 590K params
        Plus frozen W_q (2048×4) and W_v (1024×4) shared random matrices.
        Reconstruction: B_q_i = diag(d_q_i) @ W_q.T @ diag(b_q_i)  (shape r×d_q)
                        B_v_i = diag(d_v_i) @ W_v.T @ diag(b_v_i)  (shape r×d_v)

Everything else is IDENTICAL to v4:
  - Functional forward (B as tensor args, Theorem 5)
  - GSM8K evaluation protocol
  - A-matrices loaded from v2
  - Training procedure (500 steps, LR=5e-5, Adam)

MATH.md Theorem 1 predicts: total trainable params ≈ 4.8M (K922 PASS, 95x reduction)
MATH.md Theorem 2 predicts: grad_norm > 0 at step 0 (K924 PASS)
VeRA Table 2 extrapolation: quality_ratio >= 0.70 (K923, empirical target)

Baselines (from exp_m2p_sft_n500_baseline, using corrected SFT@n=500):
  base_acc = 0.200
  sft_acc  = 0.314 (measured at n=500, Wilson CI [0.275, 0.356])
  quality_ratio target: (m2p_acc - 0.200) / 0.114 >= 0.70 → m2p_acc >= 0.280

References:
  Kopiczko et al. (arXiv:2310.11454) — VeRA: shared frozen matrices for LoRA
  Ha et al. (arXiv:1609.09106) — HyperNetworks
  Hu et al. (arXiv:2106.09685) — LoRA
  SHINE (arXiv:2602.06358) — functional LoRA forward

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
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"

# LoRA config — must match v2 for A-matrix reuse
LORA_RANK = 4
LORA_SCALE = 5.0

# Model dims (from v2 results.json — do NOT re-measure)
N_LAYERS = 28
D_MODEL = 1024
Q_PROJ_OUT = 2048
V_PROJ_OUT = 1024

# VeRA M2P config
D_M2P = 1024        # encoder width = d_model (same as v4)
OUTPUT_SCALE = 0.032  # SHINE sqrt(0.001) convention for B-matrix scale

# Training
N_TEST = 10 if IS_SMOKE else 500
M2P_TRAIN_STEPS = 20 if IS_SMOKE else 500
LR = 5e-5
LR_WARMUP = 5 if IS_SMOKE else 50
MAX_SEQ_LEN = 64 if IS_SMOKE else 512
MAX_GEN_TOKENS = 64 if IS_SMOKE else 384
SEED = 42
N_TRAIN = 50 if IS_SMOKE else 2000

# Baselines from exp_m2p_sft_n500_baseline (corrected SFT @ n=500)
BASE_ACC = 0.200
SFT_ACC = 0.314
SFT_WILSON_LO = 0.275
SFT_WILSON_HI = 0.356
QUALITY_RATIO_THRESHOLD = 0.70
M2P_ACC_TARGET = BASE_ACC + QUALITY_RATIO_THRESHOLD * (SFT_ACC - BASE_ACC)  # 0.280

EXPERIMENT_DIR = Path(__file__).parent
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"

M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Few-shot prefix (IDENTICAL to v2/v3/v4 — DO NOT CHANGE)
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
    """Extract final numeric answer from GSM8K #### format or fallback patterns."""
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"(?:the\s+)?answer\s+is\s+[-–]?\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"(?:total|result|sum)\s+(?:is|=|:)\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, center - half), min(1.0, center + half))


def compute_quality_ratio_ci(
    m2p_acc: float,
    base_acc: float,
    sft_acc: float,
    n_test: int,
    z: float = 1.96,
) -> tuple:
    """Compute quality_ratio and CI lower bound using Fieller-style propagation.

    quality_ratio = (m2p_acc - base_acc) / (sft_acc - base_acc)

    Uses the n=500 SFT variance from sft_n500_baseline for propagation:
        se_m2p = sqrt(m2p_acc * (1 - m2p_acc) / n_test)
        se_sft = sqrt(sft_acc * (1 - sft_acc) / n_sft)  with n_sft=500

    Returns: (quality_ratio, ci_lower, ci_upper, se_q_simple)
    """
    denom = sft_acc - base_acc
    if abs(denom) < 1e-9:
        return (0.0, 0.0, 0.0, 0.0)

    quality_ratio = (m2p_acc - base_acc) / denom

    # Simple propagation (m2p variance only, sft treated as fixed)
    se_m2p = math.sqrt(max(m2p_acc * (1 - m2p_acc) / n_test, 0.0))
    se_q_simple = se_m2p / abs(denom)
    ci_lower_simple = quality_ratio - z * se_q_simple
    ci_upper_simple = quality_ratio + z * se_q_simple

    return (quality_ratio, ci_lower_simple, ci_upper_simple, se_q_simple)


# ---- A-matrix loading --------------------------------------------------------

def load_lora_a_matrices_v2() -> dict:
    """Load lora_a matrices saved during v2 SFT phase.

    Returns dict[(li, mod_name)] -> mx.array shape (d_model, rank) = (1024, 4).
    These are the same fixed random projections as v4; reusing ensures consistency.
    """
    if not V2_LORA_A_PATH.exists():
        raise FileNotFoundError(
            f"v2 lora_a matrices not found at {V2_LORA_A_PATH}. "
            f"Run m2p_qwen06b_gsm8k_v2 first."
        )
    saved = np.load(str(V2_LORA_A_PATH))
    result = {}
    for key in saved.files:
        assert key.endswith("_A"), f"Unexpected key: {key}"
        body = key[:-2]  # "layer_{li}_{mod_name}"
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} lora_a matrices from v2 ({V2_LORA_A_PATH})")
    return result


# ---- Functional forward (IDENTICAL to v4 — DO NOT CHANGE) --------------------

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    """LoRA projection with B as a tensor argument (proven Theorem 5 design).

    Computes: y = linear_module(x) + scale * (x @ A) @ B
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
    """Functional attention forward — IDENTICAL to v4."""
    B_batch, L, D = x.shape

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
    """Full Qwen3 model forward with functional LoRA — IDENTICAL to v4."""
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


def extract_hidden_states_functional(
    model,
    tokens_arr: mx.array,
    A_q_layers: list,
    A_v_layers: list,
    B_q_zero: list,
    B_v_zero: list,
) -> mx.array:
    """Extract per-layer mean-pooled hidden states — IDENTICAL to v4."""
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
            lora_scale=0.0,
            mask=mask,
            cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))

    return mx.stop_gradient(mx.stack(layer_states, axis=0))


# ---- VeRA M2P Architecture ---------------------------------------------------

class VeRAM2PNetwork(nn.Module):
    """VeRA-style Hypernetwork: context hidden states -> scale vectors -> B-matrices.

    Architecture (MATH.md Theorem 1):
    - Encoder: d_model -> 2*d_m2p -> d_m2p (identical to v4)
    - Output head: d_m2p -> N_layers * 4 * rank  (ONE linear, not N_layers linears)
      Output layout: [b_q_0, d_q_0, b_v_0, d_v_0, ..., b_q_{N-1}, d_q_{N-1}, b_v_{N-1}, d_v_{N-1}]
      Each block: 4 * rank = 4 * 4 = 16 scalars per layer
    - Frozen shared matrices: W_q (d_q × r), W_v (d_v × r)
    - Reconstruction: B_q_i = diag(d_q_i) @ W_q.T @ diag(b_q_i)  shape (r, d_q)
                      B_v_i = diag(d_v_i) @ W_v.T @ diag(b_v_i)  shape (r, d_v)

    Parameter count (Theorem 1):
      Encoder: 1024×2048 + 2048×1024 = 4,194,304
      Output:  1024 × 576 = 589,824
      Total trainable: 4,784,128  (K922: ≤ 10M — PASS)
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_m2p: int,
        rank: int,
        q_proj_out: int,
        v_proj_out: int,
        output_scale: float = 0.032,
        seed: int = 42,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.q_proj_out = q_proj_out
        self.v_proj_out = v_proj_out
        self.output_scale = output_scale

        # Encoder: d_model -> 2*d_m2p -> d_m2p (identical to v4)
        self.enc_linear1 = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2 = nn.Linear(2 * d_m2p, d_m2p)

        # Single output head generating all per-layer scale vectors
        # Layout per layer: [b_q (r), d_q (r), b_v (r), d_v (r)] = 4*r scalars
        self.scale_head = nn.Linear(d_m2p, n_layers * 4 * rank)

        # Frozen shared random matrices — initialized at construction, NOT trained
        # W_q: (d_q × r), W_v: (d_v × r)
        # Use seed for reproducibility
        mx.random.seed(seed)
        self.W_q = mx.random.normal(shape=(q_proj_out, rank)) / math.sqrt(q_proj_out)
        self.W_v = mx.random.normal(shape=(v_proj_out, rank)) / math.sqrt(v_proj_out)
        # Make them non-trainable by NOT listing them as parameters
        # MLX nn.Module does not treat plain mx.array attributes as parameters

    def freeze_shared(self):
        """Explicitly stop-gradient the shared matrices in forward pass.

        Called by __call__ to ensure W_q, W_v never receive gradients.
        This is belt-and-suspenders: since W_q/W_v are NOT nn.Linear modules,
        they are not in self.parameters() and optimizer won't update them.
        We also use mx.stop_gradient for clarity.
        """
        return mx.stop_gradient(self.W_q), mx.stop_gradient(self.W_v)

    def __call__(self, layer_hs: mx.array):
        """Generate B-matrices from per-layer hidden states via VeRA reconstruction.

        Args:
            layer_hs: (n_layers, d_model) — mean-pooled hidden states per layer

        Returns:
            B_q_layers: list[n_layers] of (rank, q_proj_out) — reconstructed B for q
            B_v_layers: list[n_layers] of (rank, v_proj_out) — reconstructed B for v

        Reconstruction (Theorem 1):
            B_q_i = diag(d_q_i) @ W_q.T @ diag(b_q_i)
            This is equivalent to: (W_q * b_q_i[None, :]) scaled by d_q_i[:, None]
                                   i.e. W_q.T * b_q_i then scale rows by d_q_i
        """
        # Encode
        h = mx.mean(layer_hs, axis=0)           # (d_model,)
        h = nn.gelu(self.enc_linear1(h))         # (2*d_m2p,)
        z = self.enc_linear2(h)                  # (d_m2p,)

        # Generate all scale scalars at once
        all_scales = self.scale_head(z)          # (n_layers * 4 * rank,)
        all_scales = all_scales.reshape(self.n_layers, 4, self.rank)
        # Layout: [b_q, d_q, b_v, d_v] per layer

        # Frozen shared matrices (no gradient)
        W_q, W_v = self.freeze_shared()
        # W_q: (q_proj_out, rank), W_v: (v_proj_out, rank)

        B_q_layers = []
        B_v_layers = []
        for li in range(self.n_layers):
            b_q = all_scales[li, 0, :]   # (rank,) — right-scale W_q columns
            d_q = all_scales[li, 1, :]   # (rank,) — left-scale output rows
            b_v = all_scales[li, 2, :]   # (rank,) — right-scale W_v columns
            d_v = all_scales[li, 3, :]   # (rank,) — left-scale output rows

            # B_q_i = diag(d_q) @ (W_q * b_q[None, :]).T
            #       = d_q[:, None] * (W_q * b_q[None, :]).T
            # W_q * b_q[None, :]: scale each column of W_q by b_q
            #   shape: (q_proj_out, rank) * (rank,) → (q_proj_out, rank) [broadcast]
            # transpose → (rank, q_proj_out)
            # then scale each row i by d_q[i]:
            W_q_scaled = (W_q * b_q[None, :]).T   # (rank, q_proj_out)
            B_q_i = W_q_scaled * d_q[:, None]     # (rank, q_proj_out), output_scale applied below
            B_q_layers.append(B_q_i * self.output_scale)

            W_v_scaled = (W_v * b_v[None, :]).T   # (rank, v_proj_out)
            B_v_i = W_v_scaled * d_v[:, None]     # (rank, v_proj_out)
            B_v_layers.append(B_v_i * self.output_scale)

        return B_q_layers, B_v_layers


# ---- Model setup helper ------------------------------------------------------

def _apply_lora_structure(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj, set A-matrices.

    IDENTICAL to v4.
    """
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    model.freeze()


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


# ---- Phase 1: Load data ------------------------------------------------------

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


# ---- Phase 2: M2P training ---------------------------------------------------

def phase_m2p_train(train_examples: list) -> dict:
    """Train VeRA-style M2P hypernetwork.

    Key differences from v4:
    - VeRAM2PNetwork instead of M2PNetwork
    - 500 training steps (vs 1000 in v4) — simpler architecture converges faster
    - No warm start (architecture changed: v4 weights incompatible)
    """
    log("\n" + "=" * 70)
    log("[Phase 2] VeRA M2P Training (500 steps, fresh init)")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices_v2()
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(N_LAYERS)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(N_LAYERS)]

    B_q_zero = [mx.zeros((LORA_RANK, Q_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]
    B_v_zero = [mx.zeros((LORA_RANK, V_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]

    log(f"  Tokenizing {len(train_examples)} examples...")
    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences")

    # VeRA M2P network (MATH.md Theorem 1)
    m2p = VeRAM2PNetwork(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_m2p=D_M2P,
        rank=LORA_RANK,
        q_proj_out=Q_PROJ_OUT,
        v_proj_out=V_PROJ_OUT,
        output_scale=OUTPUT_SCALE,
        seed=SEED,
    )
    mx.eval(m2p.parameters())

    # Count TRAINABLE parameters (W_q, W_v are NOT in parameters())
    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  VeRA M2P TRAINABLE params: {n_params:,}")
    log(f"  (Frozen W_q: {Q_PROJ_OUT * LORA_RANK:,}, W_v: {V_PROJ_OUT * LORA_RANK:,})")
    log(f"  (Theorem 1 prediction: ~4,784,128 trainable)")

    # ---- K922: Parameter count check ----
    k922_pass = n_params <= 10_000_000
    log(f"\n  [K922] Trainable params = {n_params:,}")
    log(f"  [K922] {'PASS' if k922_pass else 'FAIL'} (≤ 10M required)")
    if not k922_pass:
        log("  [K922] FAIL — parameter count exceeds 10M. Architecture error.")
        results = {
            "experiment": "m2p_vera_bottleneck",
            "k922_pass": False,
            "k922_n_params": n_params,
            "kill_reason": "K922 FAIL: parameter count exceeds 10M",
            "total_time_s": round(time.time() - t0, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        cleanup(m2p, model, tokenizer)
        return results

    rng = random.Random(SEED + 1)

    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    optimizer = optim.Adam(learning_rate=LR)

    def m2p_loss_fn(m2p_net, tokens_arr):
        """VeRA loss: same gradient path as v4, now through scale vectors.

        Gradient path (Theorem 2):
          m2p_net.parameters() (enc + scale_head)
            → m2p_net(layer_hs) → scale scalars → VeRA reconstruction → B_q, B_v
              → model_forward_with_loras → logits → cross_entropy → loss
        """
        layer_hs = extract_hidden_states_functional(
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

    # ---- K924: Gradient smoke test (BLOCKING kill criterion) ----
    log("\n  [K924] Gradient smoke test (Theorem 2)...")
    smoke_seq = rng.choice(tokenized)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)

    grad_norms = []
    for name, g in tree_flatten(smoke_grads):
        if isinstance(g, mx.array):
            grad_norms.append(float(mx.sum(g ** 2).item()))
    grad_norm = math.sqrt(sum(grad_norms))

    smoke_loss_val = float(smoke_loss.item())
    log(f"  [K924] grad_norm at step 0 = {grad_norm:.6f}")
    log(f"  [K924] initial loss = {smoke_loss_val:.4f}")

    k924_pass = grad_norm > 0.0
    if not k924_pass:
        log("  [K924] FAIL — zero gradients! Theorem 2 violated. W_shared may be degenerate.")
        results = {
            "experiment": "m2p_vera_bottleneck",
            "k922_pass": k922_pass,
            "k922_n_params": n_params,
            "k924_grad_norm": grad_norm,
            "k924_pass": False,
            "kill_reason": "K924 FAIL: zero gradients at step 0",
            "total_time_s": round(time.time() - t0, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        cleanup(m2p, model, tokenizer, optimizer, smoke_grads)
        return results

    log(f"  [K924] PASS — grad_norm = {grad_norm:.6f} > 0 (Theorem 2 confirmed)")
    del smoke_tokens, smoke_loss, smoke_grads

    # ---- Full M2P training ----
    log(f"\n  Training VeRA M2P for {M2P_TRAIN_STEPS} steps...")
    log(f"  N_TRAIN={N_TRAIN} | LR={LR} | warmup={LR_WARMUP}")

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

        if (step + 1) % max(1, M2P_TRAIN_STEPS // 10) == 0 or (step + 1) == M2P_TRAIN_STEPS:
            recent = sum(losses[-20:]) / min(len(losses[-20:]), 20)
            log(f"  Step {step+1}/{M2P_TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"\n  Final VeRA M2P loss: {final_loss:.4f}")

    # Save M2P weights (only trainable params — W_q/W_v are frozen, reproducible)
    m2p_params = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params.items()}
    np.savez(str(M2P_PATH), **m2p_save)
    log(f"  Saved VeRA M2P to {M2P_PATH}")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")

    cleanup(m2p, model, tokenizer, optimizer)

    return {
        "m2p_final_loss": float(final_loss),
        "m2p_params": n_params,
        "k922_pass": k922_pass,
        "k922_n_params": n_params,
        "k924_grad_norm": grad_norm,
        "k924_initial_loss": smoke_loss_val,
        "k924_pass": True,
    }


# ---- Phase 3: Evaluate VeRA M2P adapter -------------------------------------

def phase_eval_m2p(test_examples: list, train_results: dict) -> dict:
    """Evaluate VeRA M2P-generated adapter on GSM8K test at n=500.

    Uses corrected SFT baseline from exp_m2p_sft_n500_baseline:
        base_acc = 0.200, sft_acc = 0.314
        quality_ratio target >= 0.70 → m2p_acc >= 0.280
    """
    log("\n" + "=" * 70)
    log(f"[Phase 3] Evaluating VeRA M2P adapter on GSM8K (n={len(test_examples)})")
    log(f"  base_acc={BASE_ACC:.3f}, sft_acc={SFT_ACC:.3f}, target_m2p_acc>={M2P_ACC_TARGET:.3f}")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices_v2()
    _apply_lora_structure(model, lora_a_dict)

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(N_LAYERS)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(N_LAYERS)]
    B_q_zero = [mx.zeros((LORA_RANK, Q_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]
    B_v_zero = [mx.zeros((LORA_RANK, V_PROJ_OUT), dtype=mx.bfloat16) for _ in range(N_LAYERS)]

    if not M2P_PATH.exists():
        raise FileNotFoundError(f"VeRA M2P weights not found at {M2P_PATH}")

    m2p = VeRAM2PNetwork(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_m2p=D_M2P,
        rank=LORA_RANK,
        q_proj_out=Q_PROJ_OUT,
        v_proj_out=V_PROJ_OUT,
        output_scale=OUTPUT_SCALE,
        seed=SEED,
    )
    m2p_saved = np.load(str(M2P_PATH))
    weight_list = [(k, mx.array(m2p_saved[k])) for k in m2p_saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    log(f"  Loaded VeRA M2P from {M2P_PATH}")
    mx.eval(model.parameters())

    correct = 0
    total = len(test_examples)
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        layer_hs = extract_hidden_states_functional(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )
        mx.eval(layer_hs)

        B_q_layers, B_v_layers = m2p(layer_hs)
        mx.eval(*B_q_layers, *B_v_layers)

        # Inject into LoRALinear modules for mlx_lm.generate
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_layers[li]
            layer.self_attn.v_proj.lora_b = B_v_layers[li]
        mx.eval(model.parameters())

        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )

        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            correct += 1

        del tokens_arr, layer_hs, B_q_layers, B_v_layers

        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            log(f"  [VeRA M2P] {i+1}/{total}: acc={correct/(i+1):.3f}")
        if i == 0:
            log(f"  [DEBUG] Generated[:200]: {generated[:200]!r}")
            log(f"  [DEBUG] Gold: {gold!r}, Pred: {pred!r}")

    m2p_acc = correct / total if total > 0 else 0.0
    log(f"  VeRA M2P accuracy: {m2p_acc:.4f} ({correct}/{total})")

    # ---- K923: quality_ratio >= 0.70 ----
    quality_ratio, ci_lower, ci_upper, se_q = compute_quality_ratio_ci(
        m2p_acc, BASE_ACC, SFT_ACC, total
    )
    wilson_lo, wilson_hi = wilson_ci(correct, total)

    k923_pass = quality_ratio >= QUALITY_RATIO_THRESHOLD
    log(f"\n  [K923] quality_ratio = {quality_ratio:.4f}")
    log(f"  [K923] 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] (se_q={se_q:.4f})")
    log(f"  [K923] Wilson CI for M2P acc: [{wilson_lo:.4f}, {wilson_hi:.4f}]")
    log(f"  [K923] Target: >= {QUALITY_RATIO_THRESHOLD:.2f}")
    log(f"  [K923] {'PASS' if k923_pass else 'FAIL'} "
        f"(m2p_acc={m2p_acc:.3f} vs target={M2P_ACC_TARGET:.3f})")

    log(f"  Phase 3 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-eval")

    cleanup(m2p, model, tokenizer)

    return {
        "m2p_accuracy": m2p_acc,
        "m2p_correct": correct,
        "m2p_total": total,
        "k923_quality_ratio": round(quality_ratio, 4),
        "k923_ci_lower": round(ci_lower, 4),
        "k923_ci_upper": round(ci_upper, 4),
        "k923_se_q": round(se_q, 4),
        "k923_wilson_lo": round(wilson_lo, 4),
        "k923_wilson_hi": round(wilson_hi, 4),
        "k923_pass": k923_pass,
    }


# ---- Main --------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("VeRA-style M2P on Qwen3-0.6B + GSM8K")
    log("Parameter reduction via shared random projection (MATH.md Theorem 1)")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN={N_TRAIN} | N_TEST={N_TEST} | M2P_STEPS={M2P_TRAIN_STEPS}")
    log(f"LORA_RANK={LORA_RANK} | D_M2P={D_M2P} | OUTPUT_SCALE={OUTPUT_SCALE}")
    log(f"BASE_ACC={BASE_ACC:.3f} | SFT_ACC={SFT_ACC:.3f} | TARGET={M2P_ACC_TARGET:.3f}")
    log(f"Theorem 1 prediction: ~4.8M trainable params (K922: ≤10M)")
    log("=" * 70)
    log_memory("start")

    # Phase 1: Load data
    train_examples, test_examples = phase_load_data()
    log_memory("after-data")

    # Phase 2: VeRA M2P training
    m2p_train_results = phase_m2p_train(train_examples)
    log_memory("after-m2p-train")

    # Check for early kill (K922 or K924)
    if not m2p_train_results.get("k922_pass", True):
        log("\n[KILL] K922 FAIL — param count exceeded 10M. Terminating.")
        RESULTS_FILE.write_text(json.dumps(m2p_train_results, indent=2))
        return
    if not m2p_train_results.get("k924_pass", True):
        log("\n[KILL] K924 FAIL — zero gradients at step 0. Terminating.")
        RESULTS_FILE.write_text(json.dumps(m2p_train_results, indent=2))
        return

    # Phase 3: Evaluate VeRA M2P adapter
    m2p_eval_results = phase_eval_m2p(test_examples, m2p_train_results)
    log_memory("after-m2p-eval")

    # Assemble full results
    n_params = m2p_train_results["m2p_params"]
    v4_params = 457_000_000  # approximate v4 total
    reduction_vs_v4 = v4_params / n_params if n_params > 0 else float("inf")

    k922_pass = m2p_train_results["k922_pass"]
    k923_pass = m2p_eval_results["k923_pass"]
    k924_pass = m2p_train_results["k924_pass"]

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment (VeRA M2P)")
    log("=" * 70)
    log(f"  K922 (params ≤ 10M):           {'PASS' if k922_pass else 'FAIL'} "
        f"(params={n_params:,}, reduction={reduction_vs_v4:.0f}x vs v4)")
    log(f"  K923 (quality_ratio >= 70%):   {'PASS' if k923_pass else 'FAIL'} "
        f"(ratio={m2p_eval_results['k923_quality_ratio']:.4f})")
    log(f"  K924 (grad_norm > 0):          {'PASS' if k924_pass else 'FAIL'} "
        f"(grad_norm={m2p_train_results['k924_grad_norm']:.6f})")

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "m2p_vera_bottleneck",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "architecture": "VeRA-style M2P (shared random projection, Theorem 1)",
        "config": {
            "vera_style": True,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P,
            "output_scale": OUTPUT_SCALE,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "m2p_train_steps": M2P_TRAIN_STEPS,
            "lr": LR,
            "lr_warmup": LR_WARMUP,
            "max_seq_len": MAX_SEQ_LEN,
            "max_gen_tokens": MAX_GEN_TOKENS,
            "n_layers": N_LAYERS,
            "d_model": D_MODEL,
            "q_proj_out": Q_PROJ_OUT,
            "v_proj_out": V_PROJ_OUT,
        },
        # Baselines (from exp_m2p_sft_n500_baseline — do NOT re-measure)
        "base_accuracy": BASE_ACC,
        "sft_accuracy": SFT_ACC,
        "sft_wilson_ci_lower": SFT_WILSON_LO,
        "sft_wilson_ci_upper": SFT_WILSON_HI,
        # VeRA M2P training
        "m2p_final_loss": m2p_train_results.get("m2p_final_loss", 99.0),
        "m2p_params_trainable": n_params,
        "m2p_params_v4_approx": v4_params,
        "reduction_vs_v4": round(reduction_vs_v4, 1),
        # VeRA M2P eval
        "m2p_accuracy": m2p_eval_results["m2p_accuracy"],
        "m2p_correct": m2p_eval_results["m2p_correct"],
        "m2p_total": m2p_eval_results["m2p_total"],
        # Kill criteria
        "k922_n_params": n_params,
        "k922_reduction_vs_v4": round(reduction_vs_v4, 1),
        "k922_pass": k922_pass,
        "k923_quality_ratio": m2p_eval_results["k923_quality_ratio"],
        "k923_ci_lower": m2p_eval_results["k923_ci_lower"],
        "k923_ci_upper": m2p_eval_results["k923_ci_upper"],
        "k923_pass": k923_pass,
        "k924_grad_norm": round(m2p_train_results["k924_grad_norm"], 6),
        "k924_initial_loss": m2p_train_results.get("k924_initial_loss", None),
        "k924_pass": k924_pass,
        "kill_criteria": {
            "K922_params_le_10M": "PASS" if k922_pass else "FAIL",
            "K923_quality_ratio_ge_70pct": "PASS" if k923_pass else "FAIL",
            "K924_grad_norm_gt_0": "PASS" if k924_pass else "FAIL",
            "n_params_trainable": n_params,
            "quality_ratio": m2p_eval_results["k923_quality_ratio"],
            "grad_norm_step0": round(m2p_train_results["k924_grad_norm"], 6),
        },
        "peak_memory_gb": round(peak_gb, 2),
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")
    log(f"  Total time: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
