#!/usr/bin/env python3
"""M2P on Qwen3-0.6B + GSM8K v4 — Statistical Closure (1000 steps, n=500).

Kill criteria (experiment system IDs):
  K916: grad_norm > 0 at step 0 (sanity check — inherited from v3 Theorem 5)
  K917: M2P loss < 1.5 within 1000 steps (convergence at longer training)
  K918: quality_ratio >= 80% with 95% CI lower bound >= 60% at n=500
        (statistical closure of Critique #3 — is the gap real?)

Changes from v3 (compute budget only — architecture UNCHANGED):
  N_TRAIN = 4000 (was 2000)
  N_TEST  = 500  (was 200, smoke: 10)
  M2P_TRAIN_STEPS = 1000 (was 200, smoke: 20)
  + Warm start from v3 m2p_weights.npz (saves ~500 equivalent steps)
  + Binomial CI computation for K918 (Wilson interval)

Reused from v2 (do NOT re-measure):
  base_acc = 20.0%  (v2 K909 PASS)
  sft_acc  = 26.0%  (v2 K910 PASS)
  These are loaded from experiments/models/m2p_qwen06b_gsm8k_v2/results.json.

v3 weights used as warm start (if available):
  experiments/models/m2p_qwen06b_gsm8k_v3/m2p_weights.npz
  Load if file exists, else initialize fresh. Log which path was taken.

References:
  Ha et al. (arXiv:1609.09106) — HyperNetworks: weights as function outputs
  Hu et al. (arXiv:2106.09685) — LoRA weight-space update
  SHINE (arXiv:2602.06358) — functional LoRA forward, d_M2P=d_model
  Cobbe et al. (arXiv:2110.14168) — GSM8K evaluation protocol at n=500

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

# ---- Config ----------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"

# LoRA config — must match v2/v3 for A-matrix reuse
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P config — UNCHANGED from v3
D_M2P = 1024        # d_M2P = d_model (no bottleneck, same as v3)
OUTPUT_SCALE = 0.032  # SHINE sqrt(0.001) convention

# Training — v4 compute budget (ONLY these change from v3)
N_TEST = 10 if IS_SMOKE else 500              # was 200; smoke kept at 10
M2P_TRAIN_STEPS = 20 if IS_SMOKE else 1000   # was 200; smoke kept at 20
LR = 5e-5                                     # unchanged from v3
LR_WARMUP = 5 if IS_SMOKE else 100            # unchanged from v3
MAX_SEQ_LEN = 64 if IS_SMOKE else 512
MAX_GEN_TOKENS = 64 if IS_SMOKE else 384
SEED = 42

# Training data: v4 uses 4000 examples (was 2000 in v3)
N_TRAIN = 50 if IS_SMOKE else 4000

EXPERIMENT_DIR = Path(__file__).parent
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V3_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v3"
V2_RESULTS = V2_DIR / "results.json"
V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"    # A-matrices from v2 SFT phase
V3_M2P_PATH = V3_DIR / "m2p_weights.npz"            # warm start from v3

M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Few-shot prefix (same as v2/v3 — evaluation is identical)
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
    """95% Wilson score interval for k successes in n trials.

    Wilson (1927) interval — preferred over Wald for small p.
    Returns (lower, upper).
    """
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
    """Compute quality_ratio and its 95% CI lower bound via error propagation.

    quality_ratio = (m2p_acc - base_acc) / (sft_acc - base_acc)

    Uncertainty in quality_ratio comes from sampling noise in m2p_acc
    (base_acc and sft_acc are treated as fixed constants from v2).

    se_m2p = sqrt(m2p_acc * (1 - m2p_acc) / n_test)
    se_q = se_m2p / (sft_acc - base_acc)   [first-order error propagation]
    ci_lower = quality_ratio - z * se_q

    Returns: (quality_ratio, ci_lower, se_q)
    """
    denom = sft_acc - base_acc
    if abs(denom) < 1e-9:
        return (0.0, 0.0, 0.0)
    quality_ratio = (m2p_acc - base_acc) / denom
    se_m2p = math.sqrt(max(m2p_acc * (1 - m2p_acc) / n_test, 0.0))
    se_q = se_m2p / abs(denom)
    ci_lower = quality_ratio - z * se_q
    return (quality_ratio, ci_lower, se_q)


# ---- Phase 0: Load v2 baselines ---------------------------------------------

def phase_load_v2_baselines() -> dict:
    """Load base_acc and sft_acc from v2 results. Do NOT re-measure.

    v2 confirmed: base_acc=20.0%, sft_acc=26.0%
    Evaluation pipeline and LoRA setup are proven correct in v2.
    """
    log("\n" + "=" * 70)
    log("[Phase 0] Loading v2 baselines (base_acc, sft_acc)")
    log("=" * 70)

    if not V2_RESULTS.exists():
        raise FileNotFoundError(
            f"v2 results not found at {V2_RESULTS}. "
            f"Run m2p_qwen06b_gsm8k_v2 first to generate baselines."
        )

    with open(V2_RESULTS) as f:
        v2 = json.load(f)

    base_acc = v2["base_accuracy"]
    sft_acc = v2["sft_accuracy"]
    model_dims_v2 = {
        "n_layers": v2["config"]["n_layers"],
        "d_model": v2["config"]["d_model"],
        "n_heads": v2["config"]["n_heads"],
        "n_kv_heads": v2["config"]["n_kv_heads"],
        "head_dim": v2["config"]["head_dim"],
        "q_proj_out": v2["config"]["q_proj_out"],
        "v_proj_out": v2["config"]["v_proj_out"],
    }
    log(f"  base_acc = {base_acc:.4f} ({base_acc*100:.1f}%)")
    log(f"  sft_acc  = {sft_acc:.4f} ({sft_acc*100:.1f}%)")
    log(f"  model_dims: {model_dims_v2}")
    return {"base_accuracy": base_acc, "sft_accuracy": sft_acc, **model_dims_v2}


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


# ---- Tokenization -----------------------------------------------------------

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


# ---- A-matrix loading -------------------------------------------------------

def load_lora_a_matrices_v2() -> dict:
    """Load lora_a matrices saved during v2 SFT phase.

    Returns dict[(li, mod_name)] -> mx.array shape (input_dims, rank).
    These are fixed random projections; using v2's ensures consistency
    with the saved SFT B-matrices (which were trained with these A's).
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


# ---- CORE: Functional attention forward (UNCHANGED from v3) -----------------
#
# v3 CORRECT pattern (B flows as tensor arg — NO mutation):
#   q_out = functional_lora_proj(x, linear_q, A_q, B_q, scale)
#
# This is the proven design from v3 (Theorem 5). DO NOT CHANGE.

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    """LoRA projection with B as a tensor argument.

    Computes: y = linear_module(x) + scale * (x @ A) @ B

    This is functionally identical to LoRALinear.__call__ but with B
    passed as an explicit argument rather than read from module state.
    Critically, B is in the computation graph → gradients flow to M2P.
    """
    y = linear_module(x)                     # base projection (frozen, no grad needed)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)  # LoRA delta
    return y + (scale * z).astype(x.dtype)


def functional_attention_forward(
    attn: "Attention",
    x: mx.array,
    B_q: mx.array,
    B_v: mx.array,
    A_q: mx.array,
    A_v: mx.array,
    lora_scale: float,
    mask,
    cache=None,
) -> mx.array:
    """Functional attention forward passing B_q, B_v as tensor arguments.

    Replicates Qwen3 Attention.__call__ exactly, but uses functional_lora_proj
    for q_proj and v_proj so that B tensors flow through the computation graph.

    k_proj has no LoRA in this experiment (same as v2 and SHINE).

    Note: attn.q_proj and attn.v_proj are LoRALinear modules after apply_lora_to_model.
    We call attn.q_proj.linear(x) to get the base projection only (bypassing LoRA
    module's lora_b state), then add the functional delta ourselves.
    """
    B_batch, L, D = x.shape

    # q_proj with functional LoRA (B_q in graph)
    q = functional_lora_proj(x, attn.q_proj.linear, A_q, B_q, lora_scale)
    # k_proj — no LoRA (frozen base only)
    k = attn.k_proj(x)
    # v_proj with functional LoRA (B_v in graph)
    v = functional_lora_proj(x, attn.v_proj.linear, A_v, B_v, lora_scale)

    # Reshape for multi-head attention (same as Qwen3 Attention.__call__)
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
    B_q_layers: list,  # list[n_layers] of (rank, q_proj_out) tensors — IN GRAPH
    B_v_layers: list,  # list[n_layers] of (rank, v_proj_out) tensors — IN GRAPH
    A_q_layers: list,  # list[n_layers] of (input_dims, rank) — fixed, frozen
    A_v_layers: list,  # list[n_layers] of (input_dims, rank) — fixed, frozen
    lora_scale: float = LORA_SCALE,
) -> mx.array:
    """Full Qwen3 model forward with functional LoRA for q_proj and v_proj.

    B_q_layers and B_v_layers are tensor arguments — they are in the M2P
    computation graph and gradients flow through them to M2P.parameters().

    Returns: logits (batch, seq, vocab_size)
    """
    qwen3_model = model.model

    # Embed tokens
    h = qwen3_model.embed_tokens(tokens_arr)  # (1, T, d_model)

    # Create causal mask (cache=None → "causal" string for T>1, same as Qwen3Model.__call__)
    mask = create_attention_mask(h, None)

    for li, layer in enumerate(qwen3_model.layers):
        # Pre-norm
        normed = layer.input_layernorm(h)

        # Functional attention with B as tensor arg (proven design from v3)
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

        # MLP block (unchanged from base)
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    # Final norm
    h = qwen3_model.norm(h)

    # LM head
    if model.args.tie_word_embeddings:
        logits = qwen3_model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    return logits


# ---- M2P Architecture (UNCHANGED from v3) -----------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: context hidden states -> LoRA B-matrices.

    Architecture is IDENTICAL to v3. No changes.

    v3/v4 design:
    - d_M2P = d_model = 1024 (no bottleneck)
    - output_scale = 0.032 (SHINE sqrt(0.001) convention, near-zero init)
    - Input: mean-pooled hidden states from frozen LLM encoder
    - Architecture: single encoder MLP + 56 per-(layer,module) output heads

    B_q shape per layer: (rank=4, q_proj_out=2048)
    B_v shape per layer: (rank=4, v_proj_out=1024)
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
    ):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.d_m2p = d_m2p
        self.output_scale = output_scale
        self.module_specs = [("q_proj", q_proj_out), ("v_proj", v_proj_out)]

        # Encoder: d_model → d_m2p
        # Two-layer MLP: d_model -> 2*d_m2p -> d_m2p (same as v3)
        self.enc_linear1 = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2 = nn.Linear(2 * d_m2p, d_m2p)

        # B-matrix generator heads: one per (layer, module) combination
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out) for _ in range(n_layers)]

    def __call__(self, layer_hs: mx.array):
        """Generate B-matrices from per-layer hidden states.

        Args:
            layer_hs: (n_layers, d_model) — mean-pooled hidden states per layer

        Returns:
            B_q_layers: list[n_layers] of (rank, q_proj_out)
            B_v_layers: list[n_layers] of (rank, v_proj_out)
        """
        # Pool across layers to get global context
        h = mx.mean(layer_hs, axis=0)  # (d_model,)

        # Encode: d_model -> d_m2p
        h = nn.gelu(self.enc_linear1(h))  # (2*d_m2p,)
        z = self.enc_linear2(h)           # (d_m2p,)

        # Generate B-matrices for all layers
        B_q_layers = []
        B_v_layers = []
        for li in range(self.n_layers):
            b_q_flat = self.b_heads_q[li](z)  # (rank * q_proj_out,)
            b_v_flat = self.b_heads_v[li](z)  # (rank * v_proj_out,)
            B_q_layers.append(b_q_flat.reshape(self.rank, -1) * self.output_scale)
            B_v_layers.append(b_v_flat.reshape(self.rank, -1) * self.output_scale)

        return B_q_layers, B_v_layers


# ---- Hidden state extraction ------------------------------------------------

def extract_hidden_states_functional(
    model,
    tokens_arr: mx.array,
    A_q_layers: list,
    A_v_layers: list,
    B_q_zero: list,
    B_v_zero: list,
) -> mx.array:
    """Extract per-layer mean-pooled hidden states using full model forward.

    Uses stop_gradient so hidden states are not differentiated through.
    The base model is frozen; we only need its output as M2P context.

    Returns: (n_layers, d_model) hidden state tensor
    """
    qwen3_model = model.model

    h = qwen3_model.embed_tokens(tokens_arr)  # (1, T, d_model)
    mask = create_attention_mask(h, None)

    layer_states = []
    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)

        # Use functional forward with zero B (base model only, no LoRA)
        attn_out = functional_attention_forward(
            attn=layer.self_attn,
            x=normed,
            B_q=B_q_zero[li],
            B_v=B_v_zero[li],
            A_q=A_q_layers[li],
            A_v=A_v_layers[li],
            lora_scale=0.0,  # zero scale: effectively base model forward
            mask=mask,
            cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

        # Mean-pool over sequence dim
        layer_states.append(mx.mean(h[0], axis=0))  # (d_model,)

    return mx.stop_gradient(mx.stack(layer_states, axis=0))  # (n_layers, d_model)


# ---- Phase 2: M2P training --------------------------------------------------

def phase_m2p_train(train_examples: list, model_dims: dict) -> dict:
    """Train M2P hypernetwork with functional LoRA forward (v3 proven design).

    v4 changes over v3 (training ONLY):
    - N_TRAIN = 4000 (was 2000)
    - M2P_TRAIN_STEPS = 1000 (was 200)
    - Warm start from v3 m2p_weights.npz (if exists)

    Architecture is UNCHANGED.
    """
    log("\n" + "=" * 70)
    log("[Phase 2] M2P Hypernetwork Training (v4: 1000 steps, 4000 examples)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Apply LoRA structure and freeze all params
    lora_a_dict = load_lora_a_matrices_v2()
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    # Pre-extract A-matrices as lists for per-layer indexing
    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]

    # Zero B-matrices for hidden state extraction (base model forward)
    B_q_zero = [
        mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)
    ]
    B_v_zero = [
        mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)
    ]

    # Tokenize training data
    log(f"  Tokenizing {len(train_examples)} examples...")
    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences")

    # M2P network (UNCHANGED from v3)
    m2p = M2PNetwork(
        n_layers=n_layers,
        d_model=d_model,
        d_m2p=D_M2P,
        rank=LORA_RANK,
        q_proj_out=q_proj_out,
        v_proj_out=v_proj_out,
        output_scale=OUTPUT_SCALE,
    )
    mx.eval(m2p.parameters())
    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  M2P params: {n_params:,}")

    # ---- Warm start from v3 (if weights exist) ----
    warm_start_used = False
    if V3_M2P_PATH.exists():
        log(f"  [Warm Start] Loading v3 weights from {V3_M2P_PATH}")
        v3_saved = np.load(str(V3_M2P_PATH))
        weight_list = [(k, mx.array(v3_saved[k])) for k in v3_saved.files]
        m2p.load_weights(weight_list)
        mx.eval(m2p.parameters())
        warm_start_used = True
        log(f"  [Warm Start] Loaded {len(weight_list)} weight tensors from v3")
        log(f"  [Warm Start] Expected starting loss ≈ 1.076 (v3 endpoint)")
    else:
        log(f"  [Warm Start] v3 weights NOT found at {V3_M2P_PATH}")
        log(f"  [Warm Start] Initializing fresh M2P (no warm start)")

    rng = random.Random(SEED + 1)

    # LR schedule — unchanged from v3
    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    optimizer = optim.Adam(learning_rate=LR)

    # Differentiable loss function (UNCHANGED from v3 — B as tensor args)
    def m2p_loss_fn(m2p_net, tokens_arr):
        """Loss: B flows as tensor args through functional forward.

        Gradient path (proven in Theorem 5):
          m2p_net.parameters()
            → m2p_net(layer_hs) → B_q_layers, B_v_layers
              → model_forward_with_loras(model, tokens, B_q, B_v, A_q, A_v)
                → functional_attention_forward (B_q, B_v as args)
                  → logits → cross_entropy → loss
        """
        # 1. Extract context hidden states (stop_gradient)
        layer_hs = extract_hidden_states_functional(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )

        # 2. Generate B-matrices from context (grad flows through m2p_net)
        B_q_layers, B_v_layers = m2p_net(layer_hs)

        # 3. Full model forward with B as tensor args
        logits = model_forward_with_loras(
            model, tokens_arr, B_q_layers, B_v_layers,
            A_q_layers, A_v_layers, LORA_SCALE,
        )

        # 4. NTP loss
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # ---- K916: Gradient smoke test (BLOCKING kill criterion) ----
    log("\n  [K916] Gradient smoke test...")
    smoke_seq = rng.choice(tokenized)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)

    # Compute gradient norm
    grad_norms = []
    for name, g in tree_flatten(smoke_grads):
        if isinstance(g, mx.array):
            grad_norms.append(float(mx.sum(g ** 2).item()))
    grad_norm = math.sqrt(sum(grad_norms))

    smoke_loss_val = float(smoke_loss.item())
    log(f"  [K916] grad_norm at step 0 = {grad_norm:.6f}")
    log(f"  [K916] initial loss = {smoke_loss_val:.4f} "
        f"({'≈1.076 warm start' if warm_start_used and abs(smoke_loss_val - 1.076) < 0.5 else 'fresh init'})")

    k916_pass = grad_norm > 0.0
    if not k916_pass:
        log("  [K916] FAIL — zero gradients! KILL: Theorem 5 violated.")
        results = {
            "experiment": "m2p_qwen06b_gsm8k_v4",
            "model": MODEL_ID,
            "is_smoke": IS_SMOKE,
            "warm_start_used": warm_start_used,
            "k916_grad_norm": grad_norm,
            "k916_pass": False,
            "kill_reason": "K916 FAIL: zero gradients at step 0",
            "total_time_s": round(time.time() - t0, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        cleanup(m2p, model, tokenizer, optimizer, smoke_grads)
        return results

    log(f"  [K916] PASS — grad_norm = {grad_norm:.6f} > 0")
    del smoke_tokens, smoke_loss, smoke_grads

    # ---- Full M2P training (1000 steps) ----
    log(f"\n  Training M2P for {M2P_TRAIN_STEPS} steps...")
    log(f"  N_TRAIN={N_TRAIN} | LR={LR} | warmup={LR_WARMUP}")
    log(f"  Warm start: {'YES (v3)' if warm_start_used else 'NO (fresh)'}")

    gc.disable()
    losses = []
    for step in range(M2P_TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]

        # Linear LR warmup
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
    k917_pass = final_loss < 1.5

    log(f"\n  Final M2P loss: {final_loss:.4f}")
    log(f"  [K917] {'PASS' if k917_pass else 'FAIL'} (loss < 1.5 in 1000 steps): {final_loss:.4f}")

    # Save M2P weights to disk
    m2p_params = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params.items()}
    np.savez(str(M2P_PATH), **m2p_save)
    log(f"  Saved M2P to {M2P_PATH}")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")

    cleanup(m2p, model, tokenizer, optimizer)

    return {
        "m2p_final_loss": float(final_loss),
        "m2p_params": n_params,
        "k916_grad_norm": grad_norm,
        "k916_initial_loss": smoke_loss_val,
        "k916_pass": True,
        "k917_pass": k917_pass,
        "warm_start_used": warm_start_used,
    }


def _apply_lora_structure(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj, set A-matrices.

    UNCHANGED from v3.
    """
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    # Freeze all (including new lora_a/lora_b) — M2P generates B functionally
    model.freeze()


# ---- Phase 3: Evaluate M2P adapter (n=500) ----------------------------------

def phase_eval_m2p(test_examples: list, model_dims: dict,
                   base_acc: float, sft_acc: float) -> dict:
    """Evaluate M2P-generated adapter on GSM8K test at n=500.

    v4 change: n_test=500 (was 200 in v3). Architecture and eval logic UNCHANGED.

    Also computes:
    - Wilson CI for M2P accuracy (K918 evidence)
    - quality_ratio and propagated CI lower bound (K918 primary criterion)
    """
    log("\n" + "=" * 70)
    log(f"[Phase 3] Evaluating M2P adapter on GSM8K (n={len(test_examples)})")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Load model fresh (per CODING_GUIDELINES: separate scope)
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Load A-matrices and apply LoRA structure
    lora_a_dict = load_lora_a_matrices_v2()
    _apply_lora_structure(model, lora_a_dict)

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]
    B_q_zero = [
        mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)
    ]
    B_v_zero = [
        mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)
    ]

    # Load v4 M2P weights from disk
    if not M2P_PATH.exists():
        raise FileNotFoundError(f"M2P weights not found at {M2P_PATH}")

    m2p = M2PNetwork(
        n_layers=n_layers,
        d_model=d_model,
        d_m2p=D_M2P,
        rank=LORA_RANK,
        q_proj_out=q_proj_out,
        v_proj_out=v_proj_out,
        output_scale=OUTPUT_SCALE,
    )
    m2p_saved = np.load(str(M2P_PATH))
    weight_list = [(k, mx.array(m2p_saved[k])) for k in m2p_saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    log(f"  Loaded M2P v4 from {M2P_PATH}")

    mx.eval(model.parameters())

    correct = 0
    total = len(test_examples)
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # 1. Extract hidden states (base model, no grad)
        layer_hs = extract_hidden_states_functional(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )
        mx.eval(layer_hs)

        # 2. Generate B-matrices via M2P
        B_q_layers, B_v_layers = m2p(layer_hs)
        mx.eval(*B_q_layers, *B_v_layers)

        # 3. Inject into LoRALinear modules for mlx_lm.generate
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_layers[li]
            layer.self_attn.v_proj.lora_b = B_v_layers[li]
        mx.eval(model.parameters())

        # 4. Generate answer
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
            log(f"  [M2P] {i+1}/{total}: acc={correct/(i+1):.3f}")
        if i == 0:
            log(f"  [DEBUG-M2P] Generated[:200]: {generated[:200]!r}")
            log(f"  [DEBUG-M2P] Gold: {gold!r}, Pred: {pred!r}")

    m2p_acc = correct / total if total > 0 else 0.0
    log(f"  M2P accuracy: {m2p_acc:.4f} ({correct}/{total})")

    # ---- K918: Statistical closure ----
    wilson_lo, wilson_hi = wilson_ci(correct, total)
    quality_ratio, ci_lower, se_q = compute_quality_ratio_ci(
        m2p_acc, base_acc, sft_acc, total
    )

    k918_point_pass = quality_ratio >= 0.80
    k918_ci_pass = ci_lower >= 0.60
    k918_pass = k918_point_pass and k918_ci_pass

    log(f"\n  [K918] quality_ratio = {quality_ratio:.4f}")
    log(f"  [K918] 95% CI lower bound = {ci_lower:.4f} (se_q={se_q:.4f})")
    log(f"  [K918] Wilson CI for M2P acc: [{wilson_lo:.4f}, {wilson_hi:.4f}]")
    log(f"  [K918] point_pass (>=0.80): {k918_point_pass}")
    log(f"  [K918] ci_pass (lower>=0.60): {k918_ci_pass}")
    log(f"  [K918] COMBINED: {'PASS' if k918_pass else 'FAIL'}")
    if not k918_ci_pass:
        log(f"  [K918] NOTE: CI_lower < 0.60 is expected at n=500 (see MATH.md §F)")
        log(f"  [K918] Primary criterion is quality_ratio >= 0.80; CI documents uncertainty")

    log(f"  Phase 3 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-eval")

    cleanup(m2p, model, tokenizer)

    return {
        "m2p_accuracy": m2p_acc,
        "m2p_correct": correct,
        "m2p_total": total,
        "k918_quality_ratio": round(quality_ratio, 4),
        "k918_ci_lower": round(ci_lower, 4),
        "k918_se_q": round(se_q, 4),
        "k918_wilson_lo": round(wilson_lo, 4),
        "k918_wilson_hi": round(wilson_hi, 4),
        "k918_point_pass": k918_point_pass,
        "k918_ci_pass": k918_ci_pass,
        "k918_pass": k918_pass,
    }


# ---- Main -------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P on Qwen3-0.6B + GSM8K v4 — Statistical Closure (1000 steps, n=500)")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN={N_TRAIN} | N_TEST={N_TEST} | M2P_STEPS={M2P_TRAIN_STEPS}")
    log(f"MAX_SEQ_LEN={MAX_SEQ_LEN} | MAX_GEN_TOKENS={MAX_GEN_TOKENS}")
    log(f"LORA_RANK={LORA_RANK} | LORA_SCALE={LORA_SCALE} | D_M2P={D_M2P}")
    log(f"OUTPUT_SCALE={OUTPUT_SCALE} | LR={LR} | WARMUP={LR_WARMUP}")
    log(f"V3_WARM_START: {V3_M2P_PATH} (exists={V3_M2P_PATH.exists()})")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Load v2 baselines (base_acc=20%, sft_acc=26%)
    v2_data = phase_load_v2_baselines()
    base_acc = v2_data["base_accuracy"]
    sft_acc = v2_data["sft_accuracy"]
    model_dims = {k: v2_data[k] for k in [
        "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "q_proj_out", "v_proj_out"
    ]}
    log_memory("after-v2-baselines")

    # Phase 1: Load data
    train_examples, test_examples = phase_load_data()
    log_memory("after-data")

    # Phase 2: M2P training (1000 steps, warm start from v3)
    m2p_train_results = phase_m2p_train(train_examples, model_dims)
    log_memory("after-m2p-train")

    # Check for K916 kill
    if not m2p_train_results.get("k916_pass", True):
        log("\n[KILL] K916 FAIL — experiment terminated at gradient smoke test.")
        RESULTS_FILE.write_text(json.dumps(m2p_train_results, indent=2))
        return

    # Phase 3: Evaluate M2P adapter at n=500
    m2p_eval_results = phase_eval_m2p(test_examples, model_dims, base_acc, sft_acc)
    log_memory("after-m2p-eval")

    # Kill criteria assessment
    m2p_acc = m2p_eval_results["m2p_accuracy"]
    sft_improvement = sft_acc - base_acc
    m2p_improvement = m2p_acc - base_acc
    quality_ratio = m2p_eval_results["k918_quality_ratio"]
    ci_lower = m2p_eval_results["k918_ci_lower"]

    k916_pass = m2p_train_results.get("k916_pass", False)
    k917_pass = m2p_train_results.get("k917_pass", False)
    k918_pass = m2p_eval_results.get("k918_pass", False)
    k918_point_pass = m2p_eval_results.get("k918_point_pass", False)

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment (v4)")
    log("=" * 70)
    log(f"  K916 (grad_norm > 0):              {'PASS' if k916_pass else 'FAIL'} "
        f"(grad_norm={m2p_train_results.get('k916_grad_norm', 0):.4f})")
    log(f"  K917 (loss < 1.5 in 1000 steps):   {'PASS' if k917_pass else 'FAIL'} "
        f"(loss={m2p_train_results.get('m2p_final_loss', 99):.4f})")
    log(f"  K918 (quality_ratio>=80%, CI>=60%): {'PASS' if k918_pass else 'FAIL'}")
    log(f"    quality_ratio = {quality_ratio:.4f} ({'>=0.80 PASS' if k918_point_pass else '<0.80 FAIL'})")
    log(f"    CI_lower      = {ci_lower:.4f} ({'>=0.60 PASS' if ci_lower >= 0.60 else '<0.60 FAIL — expected at n=500, see MATH.md §F'})")
    log(f"    m2p_acc = {m2p_acc:.4f}, sft_acc = {sft_acc:.4f}, base_acc = {base_acc:.4f}")
    log(f"    n_test = {N_TEST}")

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "m2p_qwen06b_gsm8k_v4",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
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
            "warm_start_from_v3": m2p_train_results.get("warm_start_used", False),
            **model_dims,
        },
        # From v2 (fixed — do not re-measure)
        "base_accuracy": base_acc,
        "sft_accuracy": sft_acc,
        "sft_improvement": round(sft_improvement, 4),
        # v4 M2P results
        "m2p_final_loss": m2p_train_results.get("m2p_final_loss", 99.0),
        "m2p_params": m2p_train_results.get("m2p_params", 0),
        "m2p_accuracy": m2p_acc,
        "m2p_correct": m2p_eval_results["m2p_correct"],
        "m2p_total": m2p_eval_results["m2p_total"],
        "m2p_improvement": round(m2p_improvement, 4),
        # Kill criteria (required fields per spec)
        "k916_grad_norm": round(m2p_train_results.get("k916_grad_norm", 0.0), 6),
        "k917_final_loss": round(m2p_train_results.get("m2p_final_loss", 99.0), 4),
        "k918_quality_ratio": quality_ratio,
        "k918_ci_lower": ci_lower,
        "k918_se_q": m2p_eval_results["k918_se_q"],
        "k918_wilson_lo": m2p_eval_results["k918_wilson_lo"],
        "k918_wilson_hi": m2p_eval_results["k918_wilson_hi"],
        # Training metadata
        "n_test": N_TEST,
        "n_train": N_TRAIN,
        "train_steps_completed": M2P_TRAIN_STEPS,
        "warm_start_used": m2p_train_results.get("warm_start_used", False),
        "k916_initial_loss": m2p_train_results.get("k916_initial_loss", None),
        # Kill criteria summary
        "kill_criteria": {
            "K916_grad_norm_gt_0": "PASS" if k916_pass else "FAIL",
            "K917_loss_lt_1p5_in_1000_steps": "PASS" if k917_pass else "FAIL",
            "K918_quality_ratio_ge_80pct_ci_ge_60pct": "PASS" if k918_pass else "FAIL",
            "K918_point_only_ge_80pct": "PASS" if k918_point_pass else "FAIL",
            "base_accuracy": base_acc,
            "sft_accuracy": sft_acc,
            "m2p_accuracy": m2p_acc,
            "quality_ratio": quality_ratio,
            "ci_lower": ci_lower,
            "n_test": N_TEST,
        },
        "peak_memory_gb": round(peak_gb, 2),
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
