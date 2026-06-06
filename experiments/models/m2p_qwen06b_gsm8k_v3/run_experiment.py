#!/usr/bin/env python3
"""M2P on Qwen3-0.6B + GSM8K v3 — Functional LoRA Forward + Gradient Smoke Test.

Kill criteria (experiment system IDs):
  K913: grad_norm > 0 at step 1 (smoke test — if zero, KILL immediately)
  K914: M2P loss decreases below 2.0 within 200 steps (convergence)
  K915: quality_ratio = M2P_acc / SFT_acc >= 70% (M2P useful on real NLP)

Root cause fixed:
  v2 BROKEN: `layer.lora_b = m2p_output` (attribute mutation — severs gradient chain)
  v3 CORRECT: B flows as tensor argument through custom functional forward

Reused from v2 (do NOT re-measure):
  base_acc = 20.0%  (v2 K909 PASS)
  sft_acc  = 26.0%  (v2 K910 PASS)
  These are loaded from experiments/models/m2p_qwen06b_gsm8k_v2/results.json.

v3 changes over v2:
  Fix #1 [BLOCKING]: Functional LoRA forward — B as tensor arg (not attribute mutation)
  Fix #2 [BLOCKING]: Gradient smoke test at step 1 (assert grad_norm > 0)
  Fix #3 [IMPORTANT]: d_M2P = d_model = 1024 (removes 8x bottleneck from v2)
  Fix #4 [IMPORTANT]: output_scale = 0.032 (SHINE sqrt(0.001) convention)
  Fix #5 [ADVISORY]:  LR=5e-5, warmup=100 steps

References:
  Ha et al. (arXiv:1609.09106) — HyperNetworks: weights as function outputs
  Hu et al. (arXiv:2106.09685) — LoRA weight-space update
  SHINE (arXiv:2602.06358) — functional LoRA forward, d_M2P=d_model

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

# LoRA config — must match v2 for A-matrix reuse
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P config v3 — d_M2P = d_model (Fix #3: removes 8x bottleneck)
D_M2P = 1024  # was 128 in v2
OUTPUT_SCALE = 0.032  # Fix #4: SHINE sqrt(0.001) convention, matches LoRA B=0 init

# Training
N_TEST = 10 if IS_SMOKE else 200
M2P_TRAIN_STEPS = 20 if IS_SMOKE else 200   # smoke: fast check; full: convergence test
LR = 5e-5                                    # Fix #5: was 1e-4 in v2
LR_WARMUP = 5 if IS_SMOKE else 100           # Fix #5: linear warmup
MAX_SEQ_LEN = 64 if IS_SMOKE else 512
MAX_GEN_TOKENS = 64 if IS_SMOKE else 384
SEED = 42

# Training data: reuse v2 tokenized data (2000 examples)
N_TRAIN = 50 if IS_SMOKE else 2000

EXPERIMENT_DIR = Path(__file__).parent
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V2_RESULTS = V2_DIR / "results.json"
V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"  # A-matrices from v2 SFT phase

M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Few-shot prefix (same as v2 — evaluation is identical)
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


# ---- CORE FIX: Functional attention forward ---------------------------------
#
# v2 BROKEN pattern (severs gradient chain):
#   layer.self_attn.q_proj.lora_b = B_q   <-- Python attribute assignment
#   logits = model(tokens)                 <-- lora_b not in m2p's graph
#
# v3 CORRECT pattern (B flows as tensor arg):
#   q_out = functional_lora_proj(x, linear_q, A_q, B_q, scale)
#   ...all downstream computation uses q_out...
#
# This makes the gradient chain continuous from loss → B_q → M2P.parameters().

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    """LoRA projection with B as a tensor argument.

    Computes: y = linear_module(x) + scale * (x @ A) @ B

    This is functionally identical to LoRALinear.__call__ but with B
    passed as an explicit argument rather than read from module state.
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

    This is the CORE FIX over v2. The structure mirrors Qwen3Model.__call__
    and TransformerBlock.__call__ exactly, replacing only the attention forward
    with our functional version.

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

        # Functional attention with B as tensor arg (CORE FIX)
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


# ---- M2P Architecture -------------------------------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: context hidden states -> LoRA B-matrices.

    v3 design (Fix #3, #4, #5):
    - d_M2P = d_model = 1024 (removes 8x bottleneck from v2's d_M2P=128)
    - output_scale = 0.032 (SHINE sqrt(0.001) convention, near-zero init)
    - Input: mean-pooled hidden states from frozen LLM encoder
    - Architecture: single encoder MLP + 56 per-(layer,module) output heads

    The output_scale is applied to all B-matrix outputs so that at initialization,
    M2P generates near-zero B-matrices. This matches the standard LoRA convention
    (B=0 at init) and prevents large initial perturbations of the frozen model.

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
        # Two-layer MLP: d_model -> 2*d_m2p -> d_m2p (same as SHINE bottleneck-free design)
        self.enc_linear1 = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2 = nn.Linear(2 * d_m2p, d_m2p)

        # B-matrix generator heads: one per (layer, module) combination
        # Output flat tensor reshaped to (rank, out_dims)
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

    Uses functional forward with zero B-matrices (no LoRA delta, just base).
    This is consistent: during training, context comes from the *base* model
    hidden states, not the LoRA-adapted ones. The M2P conditions on base
    representations and generates the adaptation.

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
    """Train M2P hypernetwork with functional LoRA forward (CORE FIX).

    Key difference from v2:
    - B tensors flow as arguments through functional_attention_forward
    - nn.value_and_grad(m2p, loss_fn) traces M2P→B→model→loss continuously
    - No module attribute mutation anywhere in the differentiable path

    Steps:
    1. Gradient smoke test at step 0 (K913): assert grad_norm > 0
    2. Full training for M2P_TRAIN_STEPS (K914): loss < 2.0 by step 200
    """
    log("\n" + "=" * 70)
    log("[Phase 2] M2P Hypernetwork Training (Functional Forward)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Apply LoRA structure and freeze all params
    # _apply_lora_structure wraps q_proj/v_proj and calls model.freeze()
    # M2P generates B via functional forward; no parameter mutation anywhere
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

    # M2P network (v3: d_M2P=1024, output_scale=0.032)
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
    log(f"  d_M2P={D_M2P}, output_scale={OUTPUT_SCALE}")
    log(f"  B_q shape per layer: ({LORA_RANK}, {q_proj_out})")
    log(f"  B_v shape per layer: ({LORA_RANK}, {v_proj_out})")

    rng = random.Random(SEED + 1)

    # Build optimizer with linear LR warmup schedule
    # MLX Adam accepts a schedule callable or fixed float
    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    optimizer = optim.Adam(learning_rate=LR)  # initial LR; we update manually per step

    # ---- CORE FIX: Differentiable loss function with B as tensor args ----
    def m2p_loss_fn(m2p_net, tokens_arr):
        """Loss function where B flows as tensor args — no attribute mutation.

        Gradient path:
          m2p_net.parameters()
            → m2p_net(layer_hs) → B_q_layers, B_v_layers  [functional]
              → model_forward_with_loras(model, tokens, B_q, B_v, A_q, A_v)
                → functional_attention_forward (B_q, B_v as args)
                  → logits → cross_entropy → loss
        """
        # 1. Extract context hidden states (stop_gradient — no grad through frozen model)
        layer_hs = extract_hidden_states_functional(
            model, tokens_arr, A_q_layers, A_v_layers, B_q_zero, B_v_zero
        )

        # 2. Generate B-matrices from context (grad flows through m2p_net)
        B_q_layers, B_v_layers = m2p_net(layer_hs)

        # 3. Full model forward with B as tensor args (CORE FIX — no mutation)
        logits = model_forward_with_loras(
            model, tokens_arr, B_q_layers, B_v_layers,
            A_q_layers, A_v_layers, LORA_SCALE,
        )

        # 4. NTP loss
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # ---- K913: Gradient smoke test (BLOCKING kill criterion) ----
    log("\n  [K913] Gradient smoke test...")
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

    log(f"  [K913] grad_norm at step 0 = {grad_norm:.6f}")
    log(f"  [K913] smoke_loss at step 0 = {float(smoke_loss.item()):.4f}")

    k913_pass = grad_norm > 0.0
    if not k913_pass:
        log("  [K913] FAIL — zero gradients! KILL: functional forward still broken.")
        log("  Check: functional_lora_proj, model_forward_with_loras, M2PNetwork")
        # Save kill result
        results = {
            "experiment": "m2p_qwen06b_gsm8k_v3",
            "model": MODEL_ID,
            "is_smoke": IS_SMOKE,
            "k913_grad_norm": grad_norm,
            "k913_pass": False,
            "kill_reason": "K913 FAIL: zero gradients at step 0",
            "total_time_s": round(time.time() - t0, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        cleanup(m2p, model, tokenizer, optimizer, smoke_grads)
        return results

    log(f"  [K913] PASS — grad_norm = {grad_norm:.6f} > 0")
    del smoke_tokens, smoke_loss, smoke_grads

    # ---- Full M2P training ----
    log(f"\n  Training M2P for {M2P_TRAIN_STEPS} steps...")
    log(f"  LR={LR}, warmup={LR_WARMUP}")

    gc.disable()
    losses = []
    for step in range(M2P_TRAIN_STEPS):
        seq = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]

        # Linear LR warmup (Fix #5)
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
    k914_pass = final_loss < 2.0

    log(f"\n  Final M2P loss: {final_loss:.4f}")
    log(f"  [K914] {'PASS' if k914_pass else 'FAIL'} (loss < 2.0): {final_loss:.4f}")

    # Save M2P weights to disk (per CODING_GUIDELINES: no large objects across phases)
    m2p_params = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params.items()}
    np.savez(str(M2P_PATH), **m2p_save)
    log(f"  Saved M2P to {M2P_PATH}")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")

    # Per CODING_GUIDELINES: cleanup everything before returning
    cleanup(m2p, model, tokenizer, optimizer)

    return {
        "m2p_final_loss": float(final_loss),
        "m2p_params": n_params,
        "k913_grad_norm": grad_norm,
        "k913_pass": True,
        "k914_pass": k914_pass,
    }


def _apply_lora_structure(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj, set A-matrices.

    Needed so that attn.q_proj.linear gives access to the base linear layer
    and attn.q_proj is a LoRALinear with the saved A-matrices.

    The B-matrices in these wrappers are NOT used during M2P training —
    the functional forward bypasses them entirely. They are only used
    during M2P eval (where we set lora_b for mlx_lm.generate to use).
    """
    # Apply LoRA wrappers FIRST, then freeze
    # (freeze before wrapping would miss the newly added lora_a/lora_b parameters)
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    # Freeze all (including new lora_a/lora_b) — M2P generates B functionally
    model.freeze()


# ---- Phase 3: Evaluate M2P adapter ------------------------------------------

def phase_eval_m2p(test_examples: list, model_dims: dict) -> dict:
    """Evaluate M2P-generated adapter on GSM8K test.

    Loads model, M2P weights, and A-matrices fresh from disk.
    Per CODING_GUIDELINES: separate function scope, no carry-over from training.

    For each test example:
    1. Extract hidden states from prompt (base model forward, stop_gradient)
    2. Run M2P to generate B_q_layers, B_v_layers
    3. Inject B-matrices into LoRALinear.lora_b modules
    4. Generate answer via mlx_lm.generate

    Note: for eval, we CAN use the standard module-injection approach
    (set lora_b = generated B) because:
    - No gradient computation during eval
    - mlx_lm.generate needs the model's standard forward path
    - The functional forward is only needed for training (gradient flow)
    """
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating M2P adapter on GSM8K")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Load model fresh
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    # Load A-matrices and apply LoRA structure (freeze is done inside _apply_lora_structure)
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

    # Load M2P weights from disk
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
    log(f"  Loaded M2P from {M2P_PATH}")

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


# ---- Main -------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P on Qwen3-0.6B + GSM8K v3 — Functional LoRA Forward")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN={N_TRAIN} | N_TEST={N_TEST} | M2P_STEPS={M2P_TRAIN_STEPS}")
    log(f"MAX_SEQ_LEN={MAX_SEQ_LEN} | MAX_GEN_TOKENS={MAX_GEN_TOKENS}")
    log(f"LORA_RANK={LORA_RANK} | LORA_SCALE={LORA_SCALE} | D_M2P={D_M2P}")
    log(f"OUTPUT_SCALE={OUTPUT_SCALE} | LR={LR} | WARMUP={LR_WARMUP}")
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

    # Phase 2: M2P training (functional forward — CORE FIX)
    m2p_train_results = phase_m2p_train(train_examples, model_dims)
    log_memory("after-m2p-train")

    # Check for K913 kill
    if not m2p_train_results.get("k913_pass", True):
        log("\n[KILL] K913 FAIL — experiment terminated at gradient smoke test.")
        RESULTS_FILE.write_text(json.dumps(m2p_train_results, indent=2))
        return

    # Phase 3: Evaluate M2P adapter (only if training completed)
    m2p_eval_results = phase_eval_m2p(test_examples, model_dims)
    log_memory("after-m2p-eval")

    # Kill criteria assessment
    m2p_acc = m2p_eval_results["m2p_accuracy"]
    sft_improvement = sft_acc - base_acc  # from v2: 0.06
    m2p_improvement = m2p_acc - base_acc
    quality_ratio = (
        m2p_improvement / sft_improvement
        if abs(sft_improvement) > 1e-9
        else 0.0
    )

    k913_pass = m2p_train_results.get("k913_pass", False)
    k914_pass = m2p_train_results.get("k914_pass", False)
    k915_pass = quality_ratio >= 0.70

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment")
    log("=" * 70)
    log(f"  K913 (grad_norm > 0):         {'PASS' if k913_pass else 'FAIL'} "
        f"(grad_norm={m2p_train_results.get('k913_grad_norm', 0):.4f})")
    log(f"  K914 (loss < 2.0 in 200 steps): {'PASS' if k914_pass else 'FAIL'} "
        f"(loss={m2p_train_results.get('m2p_final_loss', 99):.4f})")
    log(f"  K915 (quality_ratio >= 70%):    {'PASS' if k915_pass else 'FAIL'} "
        f"(ratio={quality_ratio:.4f}, m2p={m2p_acc:.4f}, sft={sft_acc:.4f})")

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "m2p_qwen06b_gsm8k_v3",
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
            **model_dims,
        },
        "base_accuracy": base_acc,
        "sft_accuracy": sft_acc,
        "sft_improvement": round(sft_acc - base_acc, 4),
        "m2p_final_loss": m2p_train_results.get("m2p_final_loss", 99.0),
        "m2p_params": m2p_train_results.get("m2p_params", 0),
        "m2p_accuracy": m2p_acc,
        "m2p_correct": m2p_eval_results["m2p_correct"],
        "m2p_improvement": round(m2p_improvement, 4),
        "quality_ratio": round(quality_ratio, 4),
        "k913_grad_norm": round(m2p_train_results.get("k913_grad_norm", 0.0), 6),
        "kill_criteria": {
            "K913_grad_norm_gt_0": "PASS" if k913_pass else "FAIL",
            "K914_loss_lt_2_in_200_steps": "PASS" if k914_pass else "FAIL",
            "K915_quality_ratio_ge_70pct": "PASS" if k915_pass else "FAIL",
            "base_accuracy": base_acc,
            "sft_accuracy": sft_acc,
            "m2p_accuracy": m2p_acc,
            "quality_ratio": round(quality_ratio, 4),
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
