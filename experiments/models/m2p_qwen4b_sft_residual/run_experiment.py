#!/usr/bin/env python3
"""M2P v6: SFT-Residual M2P on Qwen3-4B + GSM8K.

Kill criteria:
  K1: init_quality_ratio >= 0.80 (SFT B-matrices preserved at init)
  K2: quality_ratio >= 0.60 at n=500 after 1000 training steps
  K3: grad_norm > 0 at step 0

Architecture: v5 SHINE base-as-encoder + SFT residual connection in weight space.
  B_applied[li] = B_sft[li] + output_scale * head(z[li])
At init, head output is near-zero, so B_applied ≈ B_sft → quality ≈ SFT quality.

References:
  He et al. (2016) — Residual learning (ResNet)
  Aghajanyan et al. (2020, arXiv:2012.13255) — Intrinsic dimensionality
  SHINE (arXiv:2602.06358) — Base-as-encoder, alternating row/col attention
  Finding #378 — v4 warm-start achieves quality_ratio=1.433 at 0.6B
  Finding #401 — v5 SHINE achieves quality_ratio=0.833 at 0.6B
  Finding #402 — v5 SHINE fails at 4B (quality_ratio=-0.187)

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
MODEL_ID = "mlx-community/Qwen3-4B-4bit"

# LoRA config — same as prior experiments for A-matrix compatibility
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P v5 config (SHINE-aligned) — unchanged
N_MEM_TOKENS = 16
D_M2P = 1024             # d_model=2560 != d_m2p → input_proj used
N_M2P_HEADS = 4
N_M2P_LAYERS = 4
OUTPUT_SCALE = 0.032     # SHINE sqrt(0.001) convention — scales the RESIDUAL only

# Training — KEY CHANGE: 1000 steps (v5 used 300)
N_TRAIN = 50 if IS_SMOKE else 2000
N_TEST = 10 if IS_SMOKE else 500  # n=500 for statistical power
M2P_TRAIN_STEPS = 20 if IS_SMOKE else 1000
LR = 5e-5
LR_WARMUP = 5 if IS_SMOKE else 100
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 64 if IS_SMOKE else 512
MAX_GEN_TOKENS = 64 if IS_SMOKE else 384
SEED = 42

EXPERIMENT_DIR = Path(__file__).parent
V1_DIR = EXPERIMENT_DIR.parent / "m2p_qwen4b_gsm8k"
SFT_B_PATH = V1_DIR / "sft_b_matrices.npz"
LORA_A_PATH = V1_DIR / "grassmannian_a_matrices.npz"
V1_RESULTS = V1_DIR / "results.json"

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
    match = re.search(r"(?:total|result|sum)\s+(?:is|=|:)\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    return None


# ---- Phase 0: Load baselines from v1 ----------------------------------------

def phase_load_baselines() -> dict:
    log("\n" + "=" * 70)
    log("[Phase 0] Loading baselines from v1 (m2p_qwen4b_gsm8k)")
    log("=" * 70)

    if not V1_RESULTS.exists():
        raise FileNotFoundError(f"v1 results not found at {V1_RESULTS}")

    with open(V1_RESULTS) as f:
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
    return {"base_accuracy": base_acc, "sft_accuracy": sft_acc, **model_dims}


# ---- Phase 1: Load data ------------------------------------------------------

def phase_load_data():
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
    result = []
    for ex in examples:
        text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) >= 2:
                result.append(ids)
    return result


# ---- Load A-matrices and SFT B-matrices -------------------------------------

def load_lora_a_matrices() -> dict:
    if not LORA_A_PATH.exists():
        raise FileNotFoundError(f"A-matrices not found at {LORA_A_PATH}")
    saved = np.load(str(LORA_A_PATH))
    result = {}
    for key in saved.files:
        assert key.endswith("_A"), f"Unexpected key: {key}"
        body = key[:-2]
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} lora_a matrices from {LORA_A_PATH}")
    return result


def load_sft_b_matrices(n_layers: int) -> tuple:
    """Load SFT B-matrices from v1 experiment.

    Returns:
        B_sft_q: list[n_layers] of mx.array (rank, q_proj_out)
        B_sft_v: list[n_layers] of mx.array (rank, v_proj_out)
    """
    if not SFT_B_PATH.exists():
        raise FileNotFoundError(f"SFT B-matrices not found at {SFT_B_PATH}")
    saved = np.load(str(SFT_B_PATH))
    B_sft_q = []
    B_sft_v = []
    for li in range(n_layers):
        bq = mx.array(saved[f"layer_{li}_q_proj_B"]).astype(mx.bfloat16)
        bv = mx.array(saved[f"layer_{li}_v_proj_B"]).astype(mx.bfloat16)
        B_sft_q.append(bq)
        B_sft_v.append(bv)
    log(f"  Loaded {2*n_layers} SFT B-matrices from {SFT_B_PATH}")
    log(f"  B_sft_q[0] shape: {B_sft_q[0].shape}, norm: {float(mx.sum(B_sft_q[0]**2).sqrt().item()):.4f}")
    log(f"  B_sft_v[0] shape: {B_sft_v[0].shape}, norm: {float(mx.sum(B_sft_v[0]**2).sqrt().item()):.4f}")
    return B_sft_q, B_sft_v


# ---- Functional LoRA forward (verbatim from v5) ------------------------------

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    y = linear_module(x)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)
    return y + (scale * z).astype(x.dtype)


def functional_attention_forward(
    attn, x: mx.array, B_q: mx.array, B_v: mx.array,
    A_q: mx.array, A_v: mx.array, lora_scale: float, mask, cache=None,
) -> mx.array:
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
    model, tokens_arr: mx.array, B_q_layers: list, B_v_layers: list,
    A_q_layers: list, A_v_layers: list, lora_scale: float = LORA_SCALE,
) -> mx.array:
    qwen3_model = model.model
    h = qwen3_model.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)

    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn_out = functional_attention_forward(
            attn=layer.self_attn, x=normed,
            B_q=B_q_layers[li], B_v=B_v_layers[li],
            A_q=A_q_layers[li], A_v=A_v_layers[li],
            lora_scale=lora_scale, mask=mask, cache=None,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    h = qwen3_model.norm(h)
    if model.args.tie_word_embeddings:
        logits = qwen3_model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    return logits


def _apply_lora_structure(model, lora_a_dict: dict) -> None:
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    model.freeze()


# ---- Memory causal mask (from v5) -------------------------------------------

def build_memory_causal_mask(M: int, T: int) -> mx.array:
    S = M + T
    neg_inf = float("-inf")
    mask_np = np.zeros((S, S), dtype=np.float32)
    mask_np[M:, :M] = neg_inf
    for i in range(T):
        for j in range(i + 1, T):
            mask_np[M + i, M + j] = neg_inf
    mask = mx.array(mask_np).astype(mx.bfloat16)
    return mask[None, None, :, :]


# ---- Extract memory hidden states (from v5) ---------------------------------

def extract_memory_hidden_states(
    model, tokens_arr: mx.array, memory_embeddings: mx.array,
) -> mx.array:
    qwen3_model = model.model
    M = memory_embeddings.shape[0]
    B_batch, T = tokens_arr.shape

    tok_embs = qwen3_model.embed_tokens(tokens_arr)
    mem_expanded = memory_embeddings[None, :, :]
    h = mx.concatenate([mem_expanded, tok_embs], axis=1)

    mask = build_memory_causal_mask(M, T)

    memory_states = []
    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn = layer.self_attn
        S = M + T

        q_full = attn.q_proj(normed)
        k_full = attn.k_proj(normed)
        v_full = attn.v_proj(normed)

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
        memory_states.append(h[0, :M, :])

    return mx.stack(memory_states, axis=0)


# ---- M2PBlock (from v5) -----------------------------------------------------

class M2PBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, is_column: bool = True):
        super().__init__()
        self.is_column = is_column
        self.norm1 = nn.RMSNorm(d)
        self.attn = nn.MultiHeadAttention(d, n_heads, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.mlp_fc1 = nn.Linear(d, 4 * d, bias=False)
        self.mlp_fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_column:
            x_t = x.transpose(1, 0, 2)
            normed = self.norm1(x_t)
            x_t = x_t + self.attn(normed, normed, normed)
            normed2 = self.norm2(x_t)
            x_t = x_t + self.mlp_fc2(nn.gelu(self.mlp_fc1(normed2)))
            return x_t.transpose(1, 0, 2)
        else:
            normed = self.norm1(x)
            x = x + self.attn(normed, normed, normed)
            normed2 = self.norm2(x)
            x = x + self.mlp_fc2(nn.gelu(self.mlp_fc1(normed2)))
            return x


# ---- M2PNetworkV6: SFT-Residual M2P -----------------------------------------

class M2PNetworkV6(nn.Module):
    """SHINE-aligned M2P with SFT residual connection in weight space.

    Key change from v5: B_applied = B_sft + output_scale * head(z)
    At init, head output ≈ 0, so B_applied ≈ B_sft → quality ≈ SFT quality.
    Training refines the residual ΔB = output_scale * head(z).
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
        B_sft_q: list,   # SFT B-matrices (frozen, not trainable)
        B_sft_v: list,
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

        # Frozen SFT B-matrices (not in parameter graph)
        self.B_sft_q = B_sft_q
        self.B_sft_v = B_sft_v

        # Learnable memory embeddings
        scale = math.sqrt(1.0 / d_model)
        mem_init = np.random.uniform(-scale, scale, (n_mem_tokens, d_model)).astype(np.float32)
        self.memory_embeddings = mx.array(mem_init).astype(mx.bfloat16)

        if self.has_input_proj:
            self.input_proj = nn.Linear(d_model, d_m2p, bias=False)
        else:
            self.input_proj = None

        self.p_layer = mx.zeros((n_layers, 1, d_m2p)).astype(mx.bfloat16)
        self.p_token = mx.zeros((1, n_mem_tokens, d_m2p)).astype(mx.bfloat16)

        self.blocks = [
            M2PBlock(d=d_m2p, n_heads=n_heads, is_column=(i % 2 == 0))
            for i in range(n_m2p_layers)
        ]

        self.final_norm = nn.RMSNorm(d_m2p)

        # Per-layer RESIDUAL heads — ZERO init so B_applied = B_sft at step 0
        # (Random init amplifies ΔB ~ d_m2p * init_scale, overwhelming B_sft)
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out, bias=False) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out, bias=False) for _ in range(n_layers)]
        # Zero-init weights: ΔB = 0 at init → B_applied = B_sft exactly
        for head in self.b_heads_q + self.b_heads_v:
            head.weight = mx.zeros_like(head.weight)

    def __call__(self, memory_grid: mx.array):
        """Generate B-matrices: B_applied = B_sft + output_scale * head(z).

        Returns:
            B_q_layers: list[n_layers] of (rank, q_proj_out) — SFT + residual
            B_v_layers: list[n_layers] of (rank, v_proj_out) — SFT + residual
        """
        if self.has_input_proj:
            L, M, d = memory_grid.shape
            flat = memory_grid.reshape(L * M, d)
            projected = self.input_proj(flat.astype(mx.bfloat16))
            x = projected.reshape(L, M, -1)
        else:
            x = memory_grid.astype(mx.bfloat16)

        x = x + self.p_layer.astype(mx.bfloat16)
        x = x + self.p_token.astype(mx.bfloat16)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        z = mx.mean(x, axis=1)  # (L, d_m2p)

        B_q_layers = []
        B_v_layers = []
        for li in range(self.n_layers):
            z_li = z[li]
            # Residual: B_sft + small ΔB from head
            delta_q = self.b_heads_q[li](z_li).reshape(self.rank, -1) * self.output_scale
            delta_v = self.b_heads_v[li](z_li).reshape(self.rank, -1) * self.output_scale
            B_q_layers.append(self.B_sft_q[li] + delta_q.astype(self.B_sft_q[li].dtype))
            B_v_layers.append(self.B_sft_v[li] + delta_v.astype(self.B_sft_v[li].dtype))

        return B_q_layers, B_v_layers


# ---- Eval helper (shared by init eval and final eval) ------------------------

def evaluate_m2p_on_examples(
    test_examples: list, model_dims: dict, m2p, model, tokenizer,
    lora_a_dict: dict, A_q_layers: list, A_v_layers: list,
    max_examples: int = None,
) -> dict:
    """Evaluate M2P adapter on test examples. Returns accuracy."""
    n_layers = model_dims["n_layers"]
    examples = test_examples[:max_examples] if max_examples else test_examples

    correct = 0
    total = len(examples)
    for i, ex in enumerate(examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        memory_grid = extract_memory_hidden_states(
            model, tokens_arr, m2p.memory_embeddings
        )
        mx.eval(memory_grid)

        B_q_layers, B_v_layers = m2p(memory_grid)
        mx.eval(*B_q_layers, *B_v_layers)

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

        del tokens_arr, memory_grid, B_q_layers, B_v_layers

        if (i + 1) % max(1, total // 4) == 0 or (i + 1) == total:
            log(f"    [{i+1}/{total}] acc={correct/(i+1):.3f}")

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ---- Phase 2: M2P training ---------------------------------------------------

def phase_m2p_train(train_examples: list, test_examples: list, model_dims: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 2] M2P v6 Training (SFT-Residual + SHINE Base-as-Encoder)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Load model and apply LoRA structure
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]

    # Load SFT B-matrices (the residual base)
    B_sft_q, B_sft_v = load_sft_b_matrices(n_layers)

    # Tokenize training data
    log(f"  Tokenizing {len(train_examples)} examples...")
    tokenized = tokenize_texts(tokenizer, train_examples)
    log(f"  Tokenized: {len(tokenized)} sequences")

    # M2PNetworkV6 with SFT residual
    m2p = M2PNetworkV6(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
        n_mem_tokens=N_MEM_TOKENS, rank=LORA_RANK,
        q_proj_out=q_proj_out, v_proj_out=v_proj_out,
        B_sft_q=B_sft_q, B_sft_v=B_sft_v,
        n_m2p_layers=N_M2P_LAYERS, n_heads=N_M2P_HEADS,
        output_scale=OUTPUT_SCALE,
    )
    mx.eval(m2p.parameters())

    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  M2P total params: {n_params:,}")
    log(f"  OUTPUT_SCALE={OUTPUT_SCALE}, M2P_TRAIN_STEPS={M2P_TRAIN_STEPS}")

    # Count M2P transformer + positional params
    m2p_transformer_params = sum(
        p.size for name, p in tree_flatten(m2p.parameters())
        if any(s in name for s in ["blocks", "p_layer", "p_token", "final_norm"])
    )
    log(f"  M2P transformer + positional params: {m2p_transformer_params:,}")

    rng = random.Random(SEED + 1)

    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    optimizer = optim.Adam(learning_rate=LR)

    def m2p_loss_fn(m2p_net, tokens_arr):
        memory_grid = extract_memory_hidden_states(
            model, tokens_arr, m2p_net.memory_embeddings
        )
        B_q_layers, B_v_layers = m2p_net(memory_grid)
        logits = model_forward_with_loras(
            model, tokens_arr, B_q_layers, B_v_layers,
            A_q_layers, A_v_layers, LORA_SCALE,
        )
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # ---- K3: Gradient smoke test ----
    log("\n  [K3] Gradient smoke test...")
    smoke_seq = rng.choice(tokenized)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)

    grad_norms = []
    for name, g in tree_flatten(smoke_grads):
        if isinstance(g, mx.array):
            grad_norms.append(float(mx.sum(g ** 2).item()))
    grad_norm = math.sqrt(sum(grad_norms))

    log(f"  [K3] grad_norm at step 0 = {grad_norm:.6f}")
    log(f"  [K3] smoke_loss at step 0 = {float(smoke_loss.item()):.4f}")

    k3_pass = grad_norm > 0.0
    if not k3_pass:
        log("  [K3] FAIL — zero gradients! KILL.")
        results = {
            "experiment": "m2p_qwen4b_sft_residual",
            "model": MODEL_ID, "is_smoke": IS_SMOKE,
            "k3_grad_norm": grad_norm, "k3_pass": False,
            "kill_reason": "K3 FAIL: zero gradients",
            "total_time_s": round(time.time() - t0, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        cleanup(m2p, model, tokenizer, optimizer, smoke_grads)
        return results

    log(f"  [K3] PASS — grad_norm = {grad_norm:.6f} > 0")
    del smoke_tokens, smoke_loss, smoke_grads

    # ---- K1: Init quality measurement (SFT residual at step 0) ----
    log("\n  [K1] Measuring init quality (SFT residual, step 0)...")
    # Use test examples for unbiased measurement
    init_eval_n = 10 if IS_SMOKE else 100
    init_results = evaluate_m2p_on_examples(
        test_examples, model_dims, m2p, model, tokenizer,
        lora_a_dict, A_q_layers, A_v_layers, max_examples=init_eval_n,
    )
    init_accuracy = init_results["accuracy"]
    log(f"  [K1] Init accuracy = {init_accuracy:.4f} ({init_results['correct']}/{init_results['total']})")

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

        if (step + 1) % max(1, M2P_TRAIN_STEPS // 10) == 0:
            recent = sum(losses[-20:]) / min(len(losses[-20:]), 20)
            log(f"  Step {step+1}/{M2P_TRAIN_STEPS}: loss={recent:.4f} grad_norm={grad_norm_step:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"\n  Final M2P loss: {final_loss:.4f}")

    # Save M2P weights
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
        "k3_grad_norm": grad_norm,
        "k3_pass": True,
        "init_accuracy": init_accuracy,
        "init_eval_n": init_eval_n,
    }


# ---- Phase 3: Evaluate M2P adapter -------------------------------------------

def phase_eval_m2p(test_examples: list, model_dims: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating M2P v6 (SFT-Residual) on GSM8K")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    lora_a_dict = load_lora_a_matrices()
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    B_sft_q, B_sft_v = load_sft_b_matrices(n_layers)

    if not M2P_PATH.exists():
        raise FileNotFoundError(f"M2P weights not found at {M2P_PATH}")

    m2p = M2PNetworkV6(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
        n_mem_tokens=N_MEM_TOKENS, rank=LORA_RANK,
        q_proj_out=q_proj_out, v_proj_out=v_proj_out,
        B_sft_q=B_sft_q, B_sft_v=B_sft_v,
        n_m2p_layers=N_M2P_LAYERS, n_heads=N_M2P_HEADS,
        output_scale=OUTPUT_SCALE,
    )
    m2p_saved = np.load(str(M2P_PATH))
    weight_list = [(k, mx.array(m2p_saved[k])) for k in m2p_saved.files]
    m2p.load_weights(weight_list)
    m2p.eval()
    mx.eval(m2p.parameters())
    log(f"  Loaded M2P from {M2P_PATH}")

    A_q_layers = [lora_a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v_layers = [lora_a_dict[(li, "v_proj")] for li in range(n_layers)]

    eval_result = evaluate_m2p_on_examples(
        test_examples, model_dims, m2p, model, tokenizer,
        lora_a_dict, A_q_layers, A_v_layers,
    )

    accuracy = eval_result["accuracy"]
    log(f"  M2P accuracy: {accuracy:.4f} ({eval_result['correct']}/{eval_result['total']})")
    log(f"  Phase 3 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-eval")

    cleanup(m2p, model, tokenizer)
    return {"m2p_accuracy": accuracy, "m2p_correct": eval_result["correct"], "m2p_n": eval_result["total"]}


# ---- Main --------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P v6: SFT-Residual M2P on Qwen3-4B + GSM8K")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN={N_TRAIN} | N_TEST={N_TEST} | M2P_STEPS={M2P_TRAIN_STEPS}")
    log(f"OUTPUT_SCALE={OUTPUT_SCALE} | LR={LR} | WARMUP={LR_WARMUP}")
    log(f"Key: B_applied = B_sft + output_scale * head(z)")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Load baselines
    baselines = phase_load_baselines()
    base_acc = baselines["base_accuracy"]
    sft_acc = baselines["sft_accuracy"]
    model_dims = {k: baselines[k] for k in [
        "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "q_proj_out", "v_proj_out"
    ]}

    # Phase 1: Load data
    train_examples, test_examples = phase_load_data()

    # Phase 2: M2P training
    m2p_train_results = phase_m2p_train(train_examples, test_examples, model_dims)

    # Check for K3 kill
    if not m2p_train_results.get("k3_pass", True):
        log("\n[KILL] K3 FAIL — experiment terminated.")
        results = {**m2p_train_results, "total_time_s": round(time.time() - t_start, 1)}
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # Phase 3: Evaluate M2P adapter
    m2p_eval_results = phase_eval_m2p(test_examples, model_dims)

    # Kill criteria assessment
    m2p_acc = m2p_eval_results["m2p_accuracy"]
    sft_improvement = sft_acc - base_acc
    m2p_improvement = m2p_acc - base_acc
    quality_ratio = (
        m2p_improvement / sft_improvement
        if abs(sft_improvement) > 1e-9
        else 0.0
    )

    init_accuracy = m2p_train_results.get("init_accuracy", 0.0)
    init_improvement = init_accuracy - base_acc
    init_quality_ratio = (
        init_improvement / sft_improvement
        if abs(sft_improvement) > 1e-9
        else 0.0
    )

    k1_pass = init_quality_ratio >= 0.80
    k2_pass = quality_ratio >= 0.60
    k3_pass = m2p_train_results.get("k3_pass", False)

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment")
    log("=" * 70)
    log(f"  K1 (init_quality_ratio >= 0.80):     {'PASS' if k1_pass else 'FAIL'} "
        f"(init_qr={init_quality_ratio:.4f}, init_acc={init_accuracy:.4f})")
    log(f"  K2 (quality_ratio >= 0.60):           {'PASS' if k2_pass else 'FAIL'} "
        f"(qr={quality_ratio:.4f}, m2p={m2p_acc:.4f}, sft={sft_acc:.4f})")
    log(f"  K3 (grad_norm > 0):                   {'PASS' if k3_pass else 'FAIL'} "
        f"(grad_norm={m2p_train_results.get('k3_grad_norm', 0):.6f})")

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "m2p_qwen4b_sft_residual",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P, "n_mem_tokens": N_MEM_TOKENS,
            "n_m2p_layers": N_M2P_LAYERS, "n_m2p_heads": N_M2P_HEADS,
            "output_scale": OUTPUT_SCALE,
            "n_train": N_TRAIN, "n_test": N_TEST,
            "m2p_train_steps": M2P_TRAIN_STEPS,
            "lr": LR, "lr_warmup": LR_WARMUP, "grad_clip": GRAD_CLIP,
            "max_seq_len": MAX_SEQ_LEN, "max_gen_tokens": MAX_GEN_TOKENS,
            "sft_residual": True,
            **model_dims,
        },
        "base_accuracy": base_acc,
        "sft_accuracy": sft_acc,
        "sft_improvement": round(sft_acc - base_acc, 4),
        "init_accuracy": init_accuracy,
        "init_quality_ratio": round(init_quality_ratio, 4),
        "m2p_final_loss": m2p_train_results.get("m2p_final_loss", 99.0),
        "m2p_params": m2p_train_results.get("m2p_params", 0),
        "m2p_transformer_params": m2p_train_results.get("m2p_transformer_params", 0),
        "m2p_accuracy": m2p_acc,
        "m2p_correct": m2p_eval_results["m2p_correct"],
        "m2p_n": m2p_eval_results.get("m2p_n", N_TEST),
        "m2p_improvement": round(m2p_improvement, 4),
        "quality_ratio": round(quality_ratio, 4),
        "k3_grad_norm": round(m2p_train_results.get("k3_grad_norm", 0.0), 6),
        "kill_criteria": {
            "K1_init_quality_ratio_ge_80pct": "PASS" if k1_pass else "FAIL",
            "K2_quality_ratio_ge_60pct": "PASS" if k2_pass else "FAIL",
            "K3_grad_norm_gt_0": "PASS" if k3_pass else "FAIL",
            "base_accuracy": base_acc,
            "sft_accuracy": sft_acc,
            "init_accuracy": init_accuracy,
            "init_quality_ratio": round(init_quality_ratio, 4),
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
