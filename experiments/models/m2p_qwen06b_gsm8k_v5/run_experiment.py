#!/usr/bin/env python3
"""M2P v5 on Qwen3-0.6B + GSM8K — SHINE-Aligned Base-as-Encoder.

Kill criteria:
  K1: grad_norm > 0 at step 0 (gradient flows through M2P transformer + heads)
  K2: quality_ratio >= 0.60 on GSM8K (matches v3 performance)
  K3: M2P transformer + positional params < 100M

Architecture fix over v3/v4:
  v3 BROKEN at 4B: mean-pool (L, d_model) → (d_model) destroys per-layer variation.
  v5 FIXED: frozen base model acts as encoder, memory tokens extracted per-layer.
  M2P transformer with alternating column/row attention contextualizes across layers.
  Each layer's B-head receives a DIFFERENT input (layer-specific, not global average).

References:
  SHINE (arXiv:2602.06358) — base model as encoder, alternating row/col attention
  Ha et al. (arXiv:1609.09106) — HyperNetworks: weights as function outputs
  Hu et al. (arXiv:2106.09685) — LoRA weight-space update
  Finding #376/#378 — functional LoRA gradient flow (v3 proven correct)
  Finding #363/#365 — M2P quality at L=36

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
from mlx.utils import tree_flatten, tree_map

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

# LoRA config — same as v3 for A-matrix compatibility
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P v5 config (SHINE-aligned)
N_MEM_TOKENS = 16        # M=16 (from Theorem 1: need >=12 for Qwen3-0.6B)
D_M2P = 1024             # = d_model (SHINE requirement for 0.6B)
N_M2P_HEADS = 4          # attention heads in M2P transformer
N_M2P_LAYERS = 4         # alternating col/row blocks (SHINE default)
OUTPUT_SCALE = 0.032     # SHINE sqrt(0.001) convention

# Training
N_TRAIN = 50 if IS_SMOKE else 2000
N_TEST = 10 if IS_SMOKE else 200
M2P_TRAIN_STEPS = 20 if IS_SMOKE else 300
LR = 5e-5
LR_WARMUP = 5 if IS_SMOKE else 50
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 64 if IS_SMOKE else 512
MAX_GEN_TOKENS = 64 if IS_SMOKE else 384
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
V3_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v3"
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V3_RESULTS = V3_DIR / "results.json"
V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"

M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
RESULTS_DIR = Path("/workspace/llm/results/m2p_qwen06b_gsm8k_v5")

# Few-shot prefix (identical to v3 — evaluation pipeline unchanged)
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


# ---- Phase 0: Load v3 baselines (base_acc, sft_acc) -------------------------

def phase_load_baselines() -> dict:
    """Load base_acc and sft_acc from v3 results. Do NOT re-measure."""
    log("\n" + "=" * 70)
    log("[Phase 0] Loading baselines (base_acc, sft_acc) from v3")
    log("=" * 70)

    # Try v3 first, fall back to v2
    results_path = V3_RESULTS if V3_RESULTS.exists() else V2_DIR / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Neither v3 nor v2 results found. Run v3 or v2 first to generate baselines."
        )

    with open(results_path) as f:
        prior = json.load(f)

    base_acc = prior["base_accuracy"]
    sft_acc = prior["sft_accuracy"]
    model_dims = {
        "n_layers": prior["config"]["n_layers"],
        "d_model": prior["config"]["d_model"],
        "n_heads": prior["config"]["n_heads"],
        "n_kv_heads": prior["config"]["n_kv_heads"],
        "head_dim": prior["config"]["head_dim"],
        "q_proj_out": prior["config"]["q_proj_out"],
        "v_proj_out": prior["config"]["v_proj_out"],
    }
    log(f"  base_acc = {base_acc:.4f} ({base_acc*100:.1f}%)")
    log(f"  sft_acc  = {sft_acc:.4f} ({sft_acc*100:.1f}%)")
    log(f"  model_dims: {model_dims}")
    log(f"  (loaded from {results_path})")
    return {"base_accuracy": base_acc, "sft_accuracy": sft_acc, **model_dims}


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


# ---- A-matrix loading --------------------------------------------------------

def load_lora_a_matrices_v2() -> dict:
    """Load lora_a matrices saved during v2 SFT phase.

    Returns dict[(li, mod_name)] -> mx.array shape (input_dims, rank).
    These are fixed random projections; reusing v2's ensures consistency.
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


# ---- Functional LoRA forward (verbatim from v3) ------------------------------
# v2 BROKEN: `layer.lora_b = m2p_output` (attribute mutation — severs gradient chain)
# v3/v5 CORRECT: B flows as tensor argument through custom functional forward

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    """LoRA projection with B as a tensor argument.

    Computes: y = linear_module(x) + scale * (x @ A) @ B

    B is passed as an explicit argument rather than read from module state.
    Critically, B is in the computation graph → gradients flow to M2P.

    Args:
        x: input (batch, seq, d_model) — bfloat16
        linear_module: the base frozen linear (LoRALinear wrapping QuantizedLinear)
        A: lora_a (input_dims, rank) — fixed random projection
        B: lora_b (rank, output_dims) — generated by M2P (tensor arg, in graph)
        scale: lora_scale (5.0)
    """
    y = linear_module(x)                     # base projection (frozen, no grad needed)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)  # LoRA delta
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
    """Functional attention forward passing B_q, B_v as tensor arguments.

    Replicates Qwen3 Attention.__call__ exactly, but uses functional_lora_proj
    for q_proj and v_proj so that B tensors flow through the computation graph.

    k_proj has no LoRA in this experiment (same as v3 and SHINE).

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

        # Functional attention with B as tensor arg
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


def _apply_lora_structure(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj, set A-matrices.

    Needed so that attn.q_proj.linear gives access to the base linear layer
    and attn.q_proj is a LoRALinear with the saved A-matrices.

    The B-matrices in these wrappers are NOT used during M2P training —
    the functional forward bypasses them entirely. They are only used
    during M2P eval (where we set lora_b for mlx_lm.generate to use).
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


# ---- v5: Memory causal mask --------------------------------------------------

def build_memory_causal_mask(M: int, T: int) -> mx.array:
    """Additive attention mask for [memory; input] sequence.

    Returns (1, 1, M+T, M+T) float tensor: 0.0 = attend, -inf = block.
    Layout:
      - Memory-to-memory:   0.0 (bidirectional — memory sees all memory)
      - Memory-to-input:    0.0 (memory sees all input)
      - Input-to-memory:   -inf (input CANNOT attend to memory tokens)
      - Input-to-input:   causal (0.0 lower-tri, -inf upper-tri)

    Note: "query attends to key" means mask[query_pos, key_pos].
    Row = query, Col = key.
    """
    S = M + T
    neg_inf = float("-inf")

    # Start with all-zero
    mask = mx.zeros((S, S))

    # Input-to-memory block: rows M..S, cols 0..M → -inf
    # Build as numpy, then convert (MLX lacks advanced boolean-mask write)
    mask_np = np.zeros((S, S), dtype=np.float32)

    # Input rows (M..S): block memory cols (0..M)
    mask_np[M:, :M] = neg_inf

    # Input rows: causal mask within input cols (M..S)
    for i in range(T):
        for j in range(i + 1, T):
            mask_np[M + i, M + j] = neg_inf

    mask = mx.array(mask_np).astype(mx.bfloat16)
    # Expand to (1, 1, S, S) for broadcast over batch and heads
    return mask[None, None, :, :]


# ---- v5: Extract memory hidden states from frozen base model -----------------

def extract_memory_hidden_states(
    model,
    tokens_arr: mx.array,    # (1, T)
    memory_embeddings: mx.array,  # (M, d_model) — learnable, part of M2P graph
) -> mx.array:
    """Run frozen base model with prepended memory tokens, extract per-layer memory states.

    CRITICAL: We iterate model.model.layers MANUALLY (not model.__call__) to
    extract per-layer hidden states at memory positions. This mirrors v3's
    extract_hidden_states_functional pattern but operates on [mem; input] concat.

    Memory tokens get positions 0..M-1; input tokens get M..M+T-1.
    This shifts input RoPE positions by M — acceptable (MATH.md note 4).

    For gradient flow: memory_embeddings are in the M2P graph. We do NOT
    apply stop_gradient here so that gradients flow back to memory_embeddings.
    The frozen base model layers themselves don't need gradients — MLX
    automatically skips frozen parameters. The gradient path is:
      memory_embeddings → h_init → frozen layers (no param grads) → memory_states
      → M2P transformer → B → functional forward → loss

    Returns: mx.array (L, M, d_model)
    """
    qwen3_model = model.model
    M = memory_embeddings.shape[0]
    B_batch, T = tokens_arr.shape

    # Embed input tokens
    tok_embs = qwen3_model.embed_tokens(tokens_arr)  # (1, T, d_model)

    # Prepend memory tokens: (1, M+T, d_model)
    # memory_embeddings is (M, d_model), expand batch dim
    mem_expanded = memory_embeddings[None, :, :]  # (1, M, d_model)
    h = mx.concatenate([mem_expanded, tok_embs], axis=1)  # (1, M+T, d_model)

    # Build additive mask for [memory; input] sequence
    mask = build_memory_causal_mask(M, T)  # (1, 1, M+T, M+T)

    memory_states = []
    for li, layer in enumerate(qwen3_model.layers):
        # Pre-norm
        normed = layer.input_layernorm(h)  # (1, M+T, d_model)

        # Base attention forward (no LoRA — we want BASE model encoding)
        # Use the layer's self_attn directly. q_proj/v_proj are LoRALinear here,
        # but we can call layer.self_attn(normed, mask=mask) which goes through
        # the standard LoRALinear path. Since lora_b is initialized to zeros and
        # the model is frozen, the LoRA delta is zero — effectively base model.
        # We call each proj directly to bypass rope offset complications:
        attn = layer.self_attn
        S = M + T

        q_full = attn.q_proj(normed)   # (1, M+T, n_heads * head_dim)
        k_full = attn.k_proj(normed)   # (1, M+T, n_kv_heads * head_dim)
        v_full = attn.v_proj(normed)   # (1, M+T, n_kv_heads * head_dim)

        queries = attn.q_norm(q_full.reshape(B_batch, S, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = attn.k_norm(k_full.reshape(B_batch, S, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = v_full.reshape(B_batch, S, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = attn.rope(queries)
        keys = attn.rope(keys)

        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_batch, S, -1)
        attn_out = attn.o_proj(attn_out)

        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

        # Extract memory token hidden states at this layer: (M, d_model)
        memory_states.append(h[0, :M, :])  # (M, d_model) — grad flows if needed

    # Stack: (L, M, d_model)
    return mx.stack(memory_states, axis=0)


# ---- v5: M2PBlock (alternating row/column attention) -------------------------

class M2PBlock(nn.Module):
    """Alternating row/column attention (SHINE §3.4).

    Column (even blocks, is_column=True):  attend across LAYERS for each memory token
      Input transposed to (M, L, d) → attention → transpose back to (L, M, d)
    Row (odd blocks, is_column=False): attend across MEMORY TOKENS for each layer
      Standard attention on (L, M, d)
    """

    def __init__(self, d: int, n_heads: int = 4, is_column: bool = True):
        super().__init__()
        self.is_column = is_column
        self.norm1 = nn.RMSNorm(d)
        self.attn = nn.MultiHeadAttention(d, n_heads, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.mlp_fc1 = nn.Linear(d, 4 * d, bias=False)
        self.mlp_fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (L, M, d)
        if self.is_column:
            # Transpose to (M, L, d) — attend across layers for each memory token
            x_t = x.transpose(1, 0, 2)  # (M, L, d)
            normed = self.norm1(x_t)
            x_t = x_t + self.attn(normed, normed, normed)
            normed2 = self.norm2(x_t)
            x_t = x_t + self.mlp_fc2(nn.gelu(self.mlp_fc1(normed2)))
            return x_t.transpose(1, 0, 2)  # (L, M, d)
        else:
            # Attend across memory tokens for each layer (x shape already (L, M, d))
            normed = self.norm1(x)
            x = x + self.attn(normed, normed, normed)
            normed2 = self.norm2(x)
            x = x + self.mlp_fc2(nn.gelu(self.mlp_fc1(normed2)))
            return x


# ---- v5: M2PNetworkV5 --------------------------------------------------------

class M2PNetworkV5(nn.Module):
    """SHINE-aligned M2P hypernetwork: base model hidden states → LoRA B-matrices.

    Architecture:
    1. memory_embeddings: (M, d_model) — learnable, Xavier init
    2. input_proj: Linear(d_model, d_m2p) if d_model != d_m2p, else None
    3. p_layer: (n_layers, 1, d_m2p) — layer positional embedding
    4. p_token: (1, n_mem_tokens, d_m2p) — token positional embedding
    5. blocks: N_M2P_LAYERS M2PBlocks (alternating column/row)
    6. final_norm: RMSNorm(d_m2p)
    7. b_heads_q: [Linear(d_m2p, rank*q_proj_out) for each layer]
    8. b_heads_v: [Linear(d_m2p, rank*v_proj_out) for each layer]

    Key vs v3: each layer's B-head receives the L-th row of the M2P transformer
    output (layer-specific), NOT the same global mean-pooled vector. This is
    the architectural fix for the 4B mean-pool bottleneck failure.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_m2p: int,
        n_mem_tokens: int,
        rank: int,
        q_proj_out: int,
        v_proj_out: int,
        n_m2p_layers: int = 4,
        n_heads: int = 4,
        output_scale: float = 0.032,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_mem_tokens = n_mem_tokens
        self.rank = rank
        self.output_scale = output_scale
        self.has_input_proj = (d_model != d_m2p)

        # Learnable memory embeddings — Xavier uniform init
        scale = math.sqrt(1.0 / d_model)
        mem_init = np.random.uniform(-scale, scale, (n_mem_tokens, d_model)).astype(np.float32)
        self.memory_embeddings = mx.array(mem_init).astype(mx.bfloat16)

        # Input projection (only if d_model != d_m2p)
        if self.has_input_proj:
            self.input_proj = nn.Linear(d_model, d_m2p, bias=False)
        else:
            self.input_proj = None

        # Positional embeddings (learnable, small init)
        self.p_layer = mx.zeros((n_layers, 1, d_m2p)).astype(mx.bfloat16)
        self.p_token = mx.zeros((1, n_mem_tokens, d_m2p)).astype(mx.bfloat16)

        # M2P transformer blocks: alternating column (even) / row (odd)
        self.blocks = [
            M2PBlock(d=d_m2p, n_heads=n_heads, is_column=(i % 2 == 0))
            for i in range(n_m2p_layers)
        ]

        # Final normalization
        self.final_norm = nn.RMSNorm(d_m2p)

        # Per-layer B-matrix heads
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out, bias=False) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out, bias=False) for _ in range(n_layers)]

    def __call__(self, memory_grid: mx.array):
        """Generate B-matrices from per-layer memory hidden states.

        Args:
            memory_grid: (L, M, d_model) — per-layer memory hidden states
                         extracted by extract_memory_hidden_states

        Returns:
            B_q_layers: list[n_layers] of (rank, q_proj_out)
            B_v_layers: list[n_layers] of (rank, v_proj_out)
        """
        # Project to d_m2p if needed
        if self.has_input_proj:
            # memory_grid: (L, M, d_model) → (L, M, d_m2p)
            L, M, d = memory_grid.shape
            flat = memory_grid.reshape(L * M, d)
            projected = self.input_proj(flat.astype(mx.bfloat16))
            x = projected.reshape(L, M, -1)
        else:
            x = memory_grid.astype(mx.bfloat16)  # (L, M, d_m2p)

        # Add positional embeddings (broadcast over L and M dims respectively)
        x = x + self.p_layer.astype(mx.bfloat16)   # (L, 1, d_m2p) broadcasts
        x = x + self.p_token.astype(mx.bfloat16)   # (1, M, d_m2p) broadcasts

        # M2P transformer blocks (alternating column/row)
        for block in self.blocks:
            x = block(x)  # (L, M, d_m2p)

        # Final normalization
        x = self.final_norm(x)  # (L, M, d_m2p)

        # Mean-pool across memory tokens: (L, d_m2p)
        z = mx.mean(x, axis=1)  # (L, d_m2p)

        # Per-layer heads generate B-matrices
        B_q_layers = []
        B_v_layers = []
        for li in range(self.n_layers):
            z_li = z[li]  # (d_m2p,) — layer-specific input
            b_q_flat = self.b_heads_q[li](z_li)  # (rank * q_proj_out,)
            b_v_flat = self.b_heads_v[li](z_li)  # (rank * v_proj_out,)
            B_q_layers.append(b_q_flat.reshape(self.rank, -1) * self.output_scale)
            B_v_layers.append(b_v_flat.reshape(self.rank, -1) * self.output_scale)

        return B_q_layers, B_v_layers


# ---- Phase 2: M2P training ---------------------------------------------------

def phase_m2p_train(train_examples: list, model_dims: dict) -> dict:
    """Train M2PNetworkV5 with SHINE-aligned base-as-encoder architecture.

    Key differences from v3:
    - M2PNetworkV5 contains learnable memory_embeddings (not standalone)
    - extract_memory_hidden_states prepends memory tokens to input
    - M2P transformer with alternating column/row attention (SHINE §3.4)
    - Each layer's B-head receives layer-specific representation

    Gradient path:
      m2p.memory_embeddings
        → extract_memory_hidden_states (through frozen base model)
          → memory_grid (L, M, d_model)
            → M2PNetworkV5.blocks → z (L, d_m2p)
              → b_heads_q/v → B_q/v_layers
                → model_forward_with_loras → logits → loss
    """
    log("\n" + "=" * 70)
    log("[Phase 2] M2P v5 Training (SHINE-Aligned Base-as-Encoder)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Apply LoRA structure and freeze (same as v3)
    lora_a_dict = load_lora_a_matrices_v2()
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]

    # Tokenize training data
    log(f"  Tokenizing {len(train_examples)} examples...")
    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences")

    # M2PNetworkV5
    m2p = M2PNetworkV5(
        n_layers=n_layers,
        d_model=d_model,
        d_m2p=D_M2P,
        n_mem_tokens=N_MEM_TOKENS,
        rank=LORA_RANK,
        q_proj_out=q_proj_out,
        v_proj_out=v_proj_out,
        n_m2p_layers=N_M2P_LAYERS,
        n_heads=N_M2P_HEADS,
        output_scale=OUTPUT_SCALE,
    )
    mx.eval(m2p.parameters())

    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  M2P total params: {n_params:,}")
    log(f"  N_MEM_TOKENS={N_MEM_TOKENS}, D_M2P={D_M2P}, N_M2P_LAYERS={N_M2P_LAYERS}")
    log(f"  output_scale={OUTPUT_SCALE}, LORA_RANK={LORA_RANK}")

    # Count M2P transformer + positional params (K3)
    m2p_transformer_params = sum(
        p.size for name, p in tree_flatten(m2p.parameters())
        if any(s in name for s in ["blocks", "p_layer", "p_token", "final_norm"])
    )
    log(f"  M2P transformer + positional params: {m2p_transformer_params:,} "
        f"({'PASS' if m2p_transformer_params < 100_000_000 else 'FAIL'} < 100M)")

    rng = random.Random(SEED + 1)

    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    optimizer = optim.Adam(learning_rate=LR)

    # Loss function: memory_embeddings flow through extraction → M2P transformer → B → loss
    def m2p_loss_fn(m2p_net, tokens_arr):
        """Loss function where memory_embeddings are in the computation graph.

        Note: We do NOT stop_gradient on memory_grid because memory_embeddings
        must be trainable. The frozen base model layers don't accumulate param
        gradients — only memory_embeddings get gradients through the base forward.
        """
        # 1. Extract memory hidden states (memory_embeddings in graph → trainable)
        memory_grid = extract_memory_hidden_states(
            model, tokens_arr, m2p_net.memory_embeddings
        )
        # memory_grid: (L, M, d_model) — grads flow back to memory_embeddings

        # 2. Generate B-matrices from memory grid (M2P transformer)
        B_q_layers, B_v_layers = m2p_net(memory_grid)

        # 3. Full model forward with B as tensor args (CORE FIX from v3 — no mutation)
        logits = model_forward_with_loras(
            model, tokens_arr, B_q_layers, B_v_layers,
            A_q_layers, A_v_layers, LORA_SCALE,
        )

        # 4. NTP loss
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # ---- K1: Gradient smoke test (BLOCKING kill criterion) ----
    log("\n  [K1] Gradient smoke test...")
    smoke_seq = rng.choice(tokenized)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)

    grad_norms = []
    for name, g in tree_flatten(smoke_grads):
        if isinstance(g, mx.array):
            grad_norms.append(float(mx.sum(g ** 2).item()))
    grad_norm = math.sqrt(sum(grad_norms))

    log(f"  [K1] grad_norm at step 0 = {grad_norm:.6f}")
    log(f"  [K1] smoke_loss at step 0 = {float(smoke_loss.item()):.4f}")

    k1_pass = grad_norm > 0.0
    if not k1_pass:
        log("  [K1] FAIL — zero gradients! KILL: gradient flow broken.")
        log("  Check: extract_memory_hidden_states, M2PNetworkV5, functional_lora_proj")
        results = {
            "experiment": "m2p_qwen06b_gsm8k_v5",
            "model": MODEL_ID,
            "is_smoke": IS_SMOKE,
            "k1_grad_norm": grad_norm,
            "k1_pass": False,
            "k3_pass": m2p_transformer_params < 100_000_000,
            "m2p_transformer_params": m2p_transformer_params,
            "m2p_params": n_params,
            "kill_reason": "K1 FAIL: zero gradients at step 0",
            "total_time_s": round(time.time() - t0, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        cleanup(m2p, model, tokenizer, optimizer, smoke_grads)
        return results

    log(f"  [K1] PASS — grad_norm = {grad_norm:.6f} > 0")
    del smoke_tokens, smoke_loss, smoke_grads

    # ---- Full M2P training ----
    log(f"\n  Training M2P for {M2P_TRAIN_STEPS} steps...")
    log(f"  LR={LR}, warmup={LR_WARMUP}, grad_clip={GRAD_CLIP}")

    gc.disable()
    losses = []
    for step in range(M2P_TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]

        optimizer.learning_rate = lr_schedule(step)

        loss, grads = loss_and_grad(m2p, tokens_arr)

        # Gradient clipping (preserve tree structure for optimizer)
        flat_grads = tree_flatten(grads)
        grad_norm_step = math.sqrt(sum(
            float(mx.sum(g ** 2).item())
            for _, g in flat_grads
            if isinstance(g, mx.array)
        ))
        if grad_norm_step > GRAD_CLIP:
            clip_factor = GRAD_CLIP / (grad_norm_step + 1e-8)
            grads = tree_map(lambda g: g * clip_factor if isinstance(g, mx.array) else g, grads)

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
    k2_converging = final_loss < 3.0  # softer threshold — v5 converges from higher init

    log(f"\n  Final M2P loss: {final_loss:.4f}")
    log(f"  [K2 prelim] loss < 3.0: {'PASS' if k2_converging else 'FAIL'} ({final_loss:.4f})")

    # Save M2P weights to disk
    m2p_params_flat = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params_flat.items()}
    np.savez(str(M2P_PATH), **m2p_save)
    log(f"  Saved M2P to {M2P_PATH}")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")

    cleanup(m2p, model, tokenizer, optimizer)

    return {
        "m2p_final_loss": float(final_loss),
        "m2p_params": n_params,
        "m2p_transformer_params": m2p_transformer_params,
        "k1_grad_norm": grad_norm,
        "k1_pass": True,
        "k3_pass": m2p_transformer_params < 100_000_000,
    }


# ---- Phase 3: Evaluate M2P adapter -------------------------------------------

def phase_eval_m2p(test_examples: list, model_dims: dict) -> dict:
    """Evaluate M2P v5 adapter on GSM8K test.

    Loads model and M2P weights fresh from disk.
    For each test example:
    1. Extract memory hidden states (base model forward with memory tokens prepended)
    2. Run M2PNetworkV5 to generate B_q_layers, B_v_layers
    3. Inject B-matrices into LoRALinear.lora_b modules
    4. Generate answer via mlx_lm.generate

    Note: for eval, we CAN use module-injection (same as v3 eval) because:
    - No gradient computation during eval
    - mlx_lm.generate needs the model's standard forward path
    """
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating M2P v5 adapter on GSM8K")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices_v2()
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    # Load M2P weights from disk
    if not M2P_PATH.exists():
        raise FileNotFoundError(f"M2P weights not found at {M2P_PATH}")

    m2p = M2PNetworkV5(
        n_layers=n_layers,
        d_model=d_model,
        d_m2p=D_M2P,
        n_mem_tokens=N_MEM_TOKENS,
        rank=LORA_RANK,
        q_proj_out=q_proj_out,
        v_proj_out=v_proj_out,
        n_m2p_layers=N_M2P_LAYERS,
        n_heads=N_M2P_HEADS,
        output_scale=OUTPUT_SCALE,
    )
    m2p_saved = np.load(str(M2P_PATH))
    weight_list = [(k, mx.array(m2p_saved[k])) for k in m2p_saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    log(f"  Loaded M2P from {M2P_PATH}")

    correct = 0
    total = len(test_examples)
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # 1. Extract memory hidden states (using trained memory_embeddings)
        memory_grid = extract_memory_hidden_states(
            model, tokens_arr, m2p.memory_embeddings
        )
        mx.eval(memory_grid)

        # 2. Generate B-matrices via M2P
        B_q_layers, B_v_layers = m2p(memory_grid)
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

        del tokens_arr, memory_grid, B_q_layers, B_v_layers

        if (i + 1) % max(1, total // 4) == 0 or (i + 1) == total:
            log(f"  [M2P] {i+1}/{total}: acc={correct/(i+1):.3f}")
        if i == 0:
            log(f"  [DEBUG-M2P] Generated[:200]: {generated[:200]!r}")
            log(f"  [DEBUG-M2P] Gold: {gold!r}, Pred: {pred!r}")

    accuracy = correct / total if total > 0 else 0.0
    log(f"  M2P accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 3 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-eval")

    cleanup(m2p, model, tokenizer)
    return {"m2p_accuracy": accuracy, "m2p_correct": correct}


# ---- Main --------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P v5 on Qwen3-0.6B + GSM8K — SHINE-Aligned Base-as-Encoder")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN={N_TRAIN} | N_TEST={N_TEST} | M2P_STEPS={M2P_TRAIN_STEPS}")
    log(f"MAX_SEQ_LEN={MAX_SEQ_LEN} | MAX_GEN_TOKENS={MAX_GEN_TOKENS}")
    log(f"LORA_RANK={LORA_RANK} | LORA_SCALE={LORA_SCALE} | D_M2P={D_M2P}")
    log(f"N_MEM_TOKENS={N_MEM_TOKENS} | N_M2P_LAYERS={N_M2P_LAYERS} | N_M2P_HEADS={N_M2P_HEADS}")
    log(f"OUTPUT_SCALE={OUTPUT_SCALE} | LR={LR} | WARMUP={LR_WARMUP} | GRAD_CLIP={GRAD_CLIP}")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Load baselines from v3 (base_acc, sft_acc)
    baselines = phase_load_baselines()
    base_acc = baselines["base_accuracy"]
    sft_acc = baselines["sft_accuracy"]
    model_dims = {k: baselines[k] for k in [
        "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "q_proj_out", "v_proj_out"
    ]}
    log_memory("after-baselines")

    # Phase 1: Load data
    train_examples, test_examples = phase_load_data()
    log_memory("after-data")

    # Phase 2: M2P v5 training
    m2p_train_results = phase_m2p_train(train_examples, model_dims)
    log_memory("after-m2p-train")

    # Check for K1 kill
    if not m2p_train_results.get("k1_pass", True):
        log("\n[KILL] K1 FAIL — experiment terminated at gradient smoke test.")
        results = {**m2p_train_results, "total_time_s": round(time.time() - t_start, 1)}
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # Phase 3: Evaluate M2P adapter
    m2p_eval_results = phase_eval_m2p(test_examples, model_dims)
    log_memory("after-m2p-eval")

    # Kill criteria assessment
    m2p_acc = m2p_eval_results["m2p_accuracy"]
    sft_improvement = sft_acc - base_acc
    m2p_improvement = m2p_acc - base_acc
    quality_ratio = (
        m2p_improvement / sft_improvement
        if abs(sft_improvement) > 1e-9
        else 0.0
    )

    k1_pass = m2p_train_results.get("k1_pass", False)
    k2_pass = quality_ratio >= 0.60
    k3_pass = m2p_train_results.get("k3_pass", False)
    m2p_transformer_params = m2p_train_results.get("m2p_transformer_params", 0)

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment")
    log("=" * 70)
    log(f"  K1 (grad_norm > 0):                  {'PASS' if k1_pass else 'FAIL'} "
        f"(grad_norm={m2p_train_results.get('k1_grad_norm', 0):.6f})")
    log(f"  K2 (quality_ratio >= 0.60):           {'PASS' if k2_pass else 'FAIL'} "
        f"(ratio={quality_ratio:.4f}, m2p={m2p_acc:.4f}, sft={sft_acc:.4f})")
    log(f"  K3 (M2P transformer params < 100M):   {'PASS' if k3_pass else 'FAIL'} "
        f"({m2p_transformer_params:,})")

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "m2p_qwen06b_gsm8k_v5",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P,
            "n_mem_tokens": N_MEM_TOKENS,
            "n_m2p_layers": N_M2P_LAYERS,
            "n_m2p_heads": N_M2P_HEADS,
            "output_scale": OUTPUT_SCALE,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "m2p_train_steps": M2P_TRAIN_STEPS,
            "lr": LR,
            "lr_warmup": LR_WARMUP,
            "grad_clip": GRAD_CLIP,
            "max_seq_len": MAX_SEQ_LEN,
            "max_gen_tokens": MAX_GEN_TOKENS,
            **model_dims,
        },
        "base_accuracy": base_acc,
        "sft_accuracy": sft_acc,
        "sft_improvement": round(sft_acc - base_acc, 4),
        "m2p_final_loss": m2p_train_results.get("m2p_final_loss", 99.0),
        "m2p_params": m2p_train_results.get("m2p_params", 0),
        "m2p_transformer_params": m2p_transformer_params,
        "m2p_accuracy": m2p_acc,
        "m2p_correct": m2p_eval_results["m2p_correct"],
        "m2p_improvement": round(m2p_improvement, 4),
        "quality_ratio": round(quality_ratio, 4),
        "k1_grad_norm": round(m2p_train_results.get("k1_grad_norm", 0.0), 6),
        "kill_criteria": {
            "K1_grad_norm_gt_0": "PASS" if k1_pass else "FAIL",
            "K2_quality_ratio_ge_60pct": "PASS" if k2_pass else "FAIL",
            "K3_m2p_transformer_lt_100M": "PASS" if k3_pass else "FAIL",
            "base_accuracy": base_acc,
            "sft_accuracy": sft_acc,
            "m2p_accuracy": m2p_acc,
            "quality_ratio": round(quality_ratio, 4),
            "m2p_transformer_params": m2p_transformer_params,
        },
        "peak_memory_gb": round(peak_gb, 2),
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Also save to /workspace/llm/results/ if it exists
    if RESULTS_DIR.parent.parent.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
        log(f"Results also saved to {RESULTS_DIR}/results.json")

    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
