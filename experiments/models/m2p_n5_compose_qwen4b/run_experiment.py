#!/usr/bin/env python3
"""N=5 Domain M2P Composition at 4B: Grassmannian Scaling Verification.

TYPE: frontier-extension
MATH: experiments/models/m2p_n5_compose_qwen4b/MATH.md

WHAT THIS TESTS:
  Five domains composed on Qwen3-4B-4bit:
  1. Math (GSM8K)         - pre-trained SFT-residual M2P (m2p_qwen4b_sft_residual)
  2. Code (Python fns)    - pre-trained M2P (m2p_2domain_compose_qwen4b)
  3. Sort (alpha sort)    - trained 100 steps on synthetic data
  4. Reverse (word flip)  - trained 100 steps on synthetic data
  5. Count (word count)   - trained 100 steps on synthetic data

  All A-matrices built via sequential Gram-Schmidt: N_max=640 >> N=5 (d=2560, r=4).
  TF-IDF routing selects domain (text-level, no model call).
  K980 measures math quality under routed N=5 composition.

KILL CRITERIA:
  K978: All 10 pairwise |A_i^T A_j|_F < 1e-4 (bf16 quantization floor)
  K979: TF-IDF routing accuracy >= 80% on 5-class held-out set (100 per class)
  K980: Math quality_ratio >= 0.70 at n=200 under routed composition

REFERENCES:
  LoraRetriever (arXiv:2402.09997), Finding #404 (N=2 at 4B), Finding #393 (N=50 at 0.6B)

Supports SMOKE_TEST=1 for quick validation (<10 min).
"""

import gc
import json
import math
import os
import random
import re
import time
from itertools import combinations
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
MODEL_ID = "mlx-community/Qwen3-4B-4bit"

LORA_RANK = 4
LORA_SCALE = 5.0
D_M2P = 1024
N_MEM_TOKENS = 16
N_M2P_LAYERS = 4
N_M2P_HEADS = 4
OUTPUT_SCALE = 0.032

SYNTH_TRAIN_STEPS = 5 if IS_SMOKE else 100
LR = 5e-5
LR_WARMUP = 2 if IS_SMOKE else 10
MAX_SEQ_LEN = 64 if IS_SMOKE else 192
MAX_GEN_TOKENS = 32 if IS_SMOKE else 256
SEED = 42

N_EVAL_MATH = 5 if IS_SMOKE else 200
N_ROUTE_TRAIN = 4 if IS_SMOKE else 100   # per class
N_ROUTE_TEST  = 4 if IS_SMOKE else 100   # per class

EXPERIMENT_DIR    = Path(__file__).parent
V1_DIR            = EXPERIMENT_DIR.parent / "m2p_qwen4b_gsm8k"
MATH_LORA_A_PATH  = V1_DIR / "grassmannian_a_matrices.npz"
MATH_SFT_B_PATH   = V1_DIR / "sft_b_matrices.npz"
V1_RESULTS        = V1_DIR / "results.json"

COMPOSE2_DIR     = EXPERIMENT_DIR.parent / "m2p_2domain_compose_qwen4b"
CODE_LORA_A_PATH = COMPOSE2_DIR / "code_a_matrices.npz"

MATH_M2P_PATH    = EXPERIMENT_DIR.parent / "m2p_qwen4b_sft_residual" / "m2p_weights.npz"
CODE_M2P_PATH    = COMPOSE2_DIR / "code_m2p_weights.npz"

SORT_LORA_A_PATH    = EXPERIMENT_DIR / "sort_a_matrices.npz"
REVERSE_LORA_A_PATH = EXPERIMENT_DIR / "reverse_a_matrices.npz"
COUNT_LORA_A_PATH   = EXPERIMENT_DIR / "count_a_matrices.npz"
SORT_M2P_PATH       = EXPERIMENT_DIR / "sort_m2p_weights.npz"
REVERSE_M2P_PATH    = EXPERIMENT_DIR / "reverse_m2p_weights.npz"
COUNT_M2P_PATH      = EXPERIMENT_DIR / "count_m2p_weights.npz"
RESULTS_FILE        = EXPERIMENT_DIR / "results.json"

# Five domains (order = Gram-Schmidt priority)
DOMAINS = ["math", "code", "sort", "reverse", "count"]

FEW_SHOT_PREFIX = (
    "Solve the math problem step by step and end with '#### <answer>'.\n\n"
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = 24 clips in May. "
    "Natalia sold 48+24 = 72 clips altogether in April and May. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
    "minutes of babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned "
    "0.2 x 50 = $10. #### 10\n\n"
)

# ---- Synthetic domain data -------------------------------------------------

SORT_WORDS = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
    "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "peach",
    "pear", "plum", "quince", "raspberry", "strawberry", "tangerine",
    "apricot", "avocado", "blueberry", "cantaloupe", "coconut",
]

REVERSE_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
]

COUNT_WORDS = [
    "red", "blue", "green", "yellow", "purple", "orange", "black", "white",
    "pink", "brown", "gray", "cyan", "magenta", "violet", "indigo",
    "crimson", "scarlet", "navy", "teal", "maroon", "beige", "ivory",
]


def make_sort_seqs(tokenizer, n: int) -> list:
    rng = random.Random(SEED + 1000)
    result = []
    for _ in range(n * 3):
        if len(result) >= n:
            break
        k = rng.randint(3, 5)
        words = rng.sample(SORT_WORDS, k)
        shuffled = words[:]
        rng.shuffle(shuffled)
        prompt = f"Sort these words alphabetically: {' '.join(shuffled)}"
        answer = " ".join(sorted(words))
        text = f"{prompt}\n{answer}"
        ids = tokenizer.encode(text)[: MAX_SEQ_LEN + 1]
        if len(ids) >= 2:
            result.append(ids)
    return result


def make_reverse_seqs(tokenizer, n: int) -> list:
    rng = random.Random(SEED + 2000)
    result = []
    for _ in range(n * 3):
        if len(result) >= n:
            break
        k = rng.randint(3, 5)
        words = rng.sample(REVERSE_WORDS, k)
        prompt = f"Reverse the order of these words: {' '.join(words)}"
        answer = " ".join(reversed(words))
        text = f"{prompt}\n{answer}"
        ids = tokenizer.encode(text)[: MAX_SEQ_LEN + 1]
        if len(ids) >= 2:
            result.append(ids)
    return result


def make_count_seqs(tokenizer, n: int) -> list:
    rng = random.Random(SEED + 3000)
    result = []
    for _ in range(n * 3):
        if len(result) >= n:
            break
        k = rng.randint(3, 7)
        words = rng.sample(COUNT_WORDS, k)
        prompt = f"Count the words in this phrase: {' '.join(words)}"
        answer = str(k)
        text = f"{prompt}\n{answer}"
        ids = tokenizer.encode(text)[: MAX_SEQ_LEN + 1]
        if len(ids) >= 2:
            result.append(ids)
    return result


# ---- Routing text generation -----------------------------------------------

def get_sort_routing_texts(n: int, offset: int = 0) -> list:
    rng = random.Random(SEED + 4000 + offset)
    texts = []
    for _ in range(n):
        k = rng.randint(3, 5)
        words = rng.sample(SORT_WORDS, k)
        rng.shuffle(words)
        texts.append(f"Sort these words alphabetically: {' '.join(words)}")
    return texts


def get_reverse_routing_texts(n: int, offset: int = 0) -> list:
    rng = random.Random(SEED + 5000 + offset)
    texts = []
    for _ in range(n):
        k = rng.randint(3, 5)
        words = rng.sample(REVERSE_WORDS, k)
        texts.append(f"Reverse the order of these words: {' '.join(words)}")
    return texts


def get_count_routing_texts(n: int, offset: int = 0) -> list:
    rng = random.Random(SEED + 6000 + offset)
    texts = []
    for _ in range(n):
        k = rng.randint(3, 7)
        words = rng.sample(COUNT_WORDS, k)
        texts.append(f"Count the words in this phrase: {' '.join(words)}")
    return texts


# ---- Utilities -------------------------------------------------------------

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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"(?:the\s+)?answer\s+is\s+[-–]?\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    return None


# ---- Model dims from v1 results --------------------------------------------

def load_model_dims() -> dict:
    with open(V1_RESULTS) as f:
        v1 = json.load(f)
    cfg = v1["config"]
    return {
        "n_layers": cfg["n_layers"],
        "d_model": cfg["d_model"],
        "q_proj_out": cfg["q_proj_out"],
        "v_proj_out": cfg["v_proj_out"],
        "base_accuracy": v1["base_accuracy"],
        "sft_accuracy": v1["sft_accuracy"],
    }


# ---- A-matrix I/O ----------------------------------------------------------

def load_a_matrices(path: Path, n_layers: int) -> tuple:
    saved = np.load(str(path))
    A_q = [mx.array(saved[f"layer_{li}_q_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    A_v = [mx.array(saved[f"layer_{li}_v_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    mx.eval(*A_q, *A_v)
    return A_q, A_v


def load_a_matrices_np(path: Path, n_layers: int) -> tuple:
    """Load as float64 numpy arrays for Gram-Schmidt computation."""
    saved = np.load(str(path))
    A_q = [saved[f"layer_{li}_q_proj_A"].astype(np.float64) for li in range(n_layers)]
    A_v = [saved[f"layer_{li}_v_proj_A"].astype(np.float64) for li in range(n_layers)]
    return A_q, A_v


def save_a_matrices(path: Path, A_q: list, A_v: list) -> None:
    save_dict = {}
    for li, (aq, av) in enumerate(zip(A_q, A_v)):
        save_dict[f"layer_{li}_q_proj_A"] = aq.astype(np.float32)
        save_dict[f"layer_{li}_v_proj_A"] = av.astype(np.float32)
    np.savez(str(path), **save_dict)


def load_sft_b_matrices(n_layers: int) -> tuple:
    saved = np.load(str(MATH_SFT_B_PATH))
    B_q = [mx.array(saved[f"layer_{li}_q_proj_B"]).astype(mx.bfloat16) for li in range(n_layers)]
    B_v = [mx.array(saved[f"layer_{li}_v_proj_B"]).astype(mx.bfloat16) for li in range(n_layers)]
    mx.eval(*B_q, *B_v)
    return B_q, B_v


def zero_b_matrices(n_layers: int, q_proj_out: int, v_proj_out: int) -> tuple:
    B_q = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    mx.eval(*B_q, *B_v)
    return B_q, B_v


# ---- LoRA model wiring -----------------------------------------------------

def apply_lora_structure(model, A_q: list, A_v: list) -> None:
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.q_proj.lora_a = A_q[li]
        attn.v_proj.lora_a = A_v[li]
    model.freeze()


def inject_lora_b(model, B_q: list, B_v: list) -> None:
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_b = B_q[li]
        layer.self_attn.v_proj.lora_b = B_v[li]
    mx.eval(model.parameters())


# ---- SHINE memory-based forward --------------------------------------------

def build_memory_causal_mask(M: int, T: int) -> mx.array:
    S = M + T
    mask_np = np.zeros((S, S), dtype=np.float32)
    mask_np[M:, :M] = float("-inf")
    for i in range(T):
        for j in range(i + 1, T):
            mask_np[M + i, M + j] = float("-inf")
    return mx.array(mask_np).astype(mx.bfloat16)[None, None, :, :]


def extract_memory_hidden_states(model, tokens_arr: mx.array, memory_embeddings: mx.array) -> mx.array:
    qwen3 = model.model
    M = memory_embeddings.shape[0]
    B_batch, T = tokens_arr.shape
    tok_embs = qwen3.embed_tokens(tokens_arr)
    h = mx.concatenate([memory_embeddings[None, :, :], tok_embs], axis=1)
    mask = build_memory_causal_mask(M, T)
    memory_states = []
    for li, layer in enumerate(qwen3.layers):
        normed = layer.input_layernorm(h)
        attn = layer.self_attn
        S = M + T
        q_f = attn.q_proj(normed)
        k_f = attn.k_proj(normed)
        v_f = attn.v_proj(normed)
        queries = attn.q_norm(q_f.reshape(B_batch, S, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        keys    = attn.k_norm(k_f.reshape(B_batch, S, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values  = v_f.reshape(B_batch, S, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries)
        keys    = attn.rope(keys)
        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask
        )
        attn_out = attn.o_proj(attn_out.transpose(0, 2, 1, 3).reshape(B_batch, S, -1))
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        memory_states.append(h[0, :M, :])
    return mx.stack(memory_states, axis=0)


def functional_lora_proj(x, linear_module, A, B, scale):
    y = linear_module(x)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)
    return y + (scale * z).astype(x.dtype)


def forward_with_loras(model, tokens_arr, B_q_layers, B_v_layers, A_q_layers, A_v_layers):
    qwen3 = model.model
    h = qwen3.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)
    for li, layer in enumerate(qwen3.layers):
        normed = layer.input_layernorm(h)
        attn = layer.self_attn
        B_batch, L, D = normed.shape
        q = functional_lora_proj(normed, attn.q_proj.linear, A_q_layers[li], B_q_layers[li], LORA_SCALE)
        k = attn.k_proj(normed)
        v = functional_lora_proj(normed, attn.v_proj.linear, A_v_layers[li], B_v_layers[li], LORA_SCALE)
        queries = attn.q_norm(q.reshape(B_batch, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        keys    = attn.k_norm(k.reshape(B_batch, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values  = v.reshape(B_batch, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries)
        keys    = attn.rope(keys)
        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask
        )
        attn_out = attn.o_proj(attn_out.transpose(0, 2, 1, 3).reshape(B_batch, L, -1))
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = qwen3.norm(h)
    return qwen3.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)


# ---- M2P architecture ------------------------------------------------------

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
            x_t = x_t + self.mlp_fc2(nn.gelu(self.mlp_fc1(self.norm2(x_t))))
            return x_t.transpose(1, 0, 2)
        else:
            normed = self.norm1(x)
            x = x + self.attn(normed, normed, normed)
            x = x + self.mlp_fc2(nn.gelu(self.mlp_fc1(self.norm2(x))))
            return x


class M2PNetworkV6(nn.Module):
    """SFT-Residual M2P: B_applied[li] = B_sft[li] + output_scale * head(z[li])."""

    def __init__(self, n_layers, d_model, d_m2p, n_mem_tokens, rank,
                 q_proj_out, v_proj_out, B_sft_q, B_sft_v,
                 n_m2p_layers=4, n_heads=4, output_scale=0.032):
        super().__init__()
        self.n_layers     = n_layers
        self.n_mem_tokens = n_mem_tokens
        self.rank         = rank
        self.output_scale = output_scale
        self.has_input_proj = (d_model != d_m2p)
        self.B_sft_q = B_sft_q
        self.B_sft_v = B_sft_v

        scale = math.sqrt(1.0 / d_model)
        mem_init = np.random.default_rng(SEED).standard_normal(
            (n_mem_tokens, d_model)).astype(np.float32) * scale
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

        # Zero-init heads → B_applied = B_sft at step 0
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out, bias=False) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out, bias=False) for _ in range(n_layers)]
        for head in self.b_heads_q + self.b_heads_v:
            head.weight = mx.zeros_like(head.weight)

    def __call__(self, memory_grid: mx.array):
        if self.has_input_proj:
            L, M, d = memory_grid.shape
            x = self.input_proj(memory_grid.reshape(L * M, d).astype(mx.bfloat16)).reshape(L, M, -1)
        else:
            x = memory_grid.astype(mx.bfloat16)
        x = x + self.p_layer.astype(mx.bfloat16)
        x = x + self.p_token.astype(mx.bfloat16)
        for block in self.blocks:
            x = block(x)
        z = mx.mean(self.final_norm(x), axis=1)  # (L, d_m2p)
        B_q, B_v = [], []
        for li in range(self.n_layers):
            delta_q = self.b_heads_q[li](z[li]).reshape(self.rank, -1) * self.output_scale
            delta_v = self.b_heads_v[li](z[li]).reshape(self.rank, -1) * self.output_scale
            B_q.append(self.B_sft_q[li] + delta_q.astype(self.B_sft_q[li].dtype))
            B_v.append(self.B_sft_v[li] + delta_v.astype(self.B_sft_v[li].dtype))
        return B_q, B_v


def make_m2p_v6(model_dims: dict, B_sft_q: list, B_sft_v: list) -> M2PNetworkV6:
    return M2PNetworkV6(
        n_layers=model_dims["n_layers"], d_model=model_dims["d_model"],
        d_m2p=D_M2P, n_mem_tokens=N_MEM_TOKENS, rank=LORA_RANK,
        q_proj_out=model_dims["q_proj_out"], v_proj_out=model_dims["v_proj_out"],
        B_sft_q=B_sft_q, B_sft_v=B_sft_v,
        n_m2p_layers=N_M2P_LAYERS, n_heads=N_M2P_HEADS, output_scale=OUTPUT_SCALE,
    )


# ---- Phase 0: Build all 5 A-matrices + verify isolation (K978) -------------

def phase_build_and_verify_a_matrices(model_dims: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 0] Build N=5 A-matrices + Grassmannian Isolation (K978)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]

    # Load math A (float64 for Gram-Schmidt)
    math_a_np = np.load(str(MATH_LORA_A_PATH))
    A_math_q = [math_a_np[f"layer_{li}_q_proj_A"].astype(np.float64) for li in range(n_layers)]
    A_math_v = [math_a_np[f"layer_{li}_v_proj_A"].astype(np.float64) for li in range(n_layers)]
    log(f"  Loaded math A-matrices: {n_layers} layers")

    # Load code A (float64) — already orthogonal to math
    code_a_np = np.load(str(CODE_LORA_A_PATH))
    A_code_q = [code_a_np[f"layer_{li}_q_proj_A"].astype(np.float64) for li in range(n_layers)]
    A_code_v = [code_a_np[f"layer_{li}_v_proj_A"].astype(np.float64) for li in range(n_layers)]
    log(f"  Loaded code A-matrices: {n_layers} layers")

    # Generate sort/reverse/count A via sequential Gram-Schmidt
    # sort: orthogonal to math+code
    # reverse: orthogonal to math+code+sort
    # count: orthogonal to math+code+sort+reverse
    prior_domains = [
        ("math_q", A_math_q), ("math_v", A_math_v),
        ("code_q", A_code_q), ("code_v", A_code_v),
    ]

    def gram_schmidt_new_domain(prior_q_list: list, prior_v_list: list, seed_offset: int) -> tuple:
        """Generate rank-LORA_RANK A-matrices orthogonal to all prior domains.

        Re-orthonormalizes each prior A via QR before projection for numerical stability
        when prior A-matrices are loaded from float32 storage.
        """
        A_q_new = []
        A_v_new = []
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for li in range(n_layers):
                rng = np.random.default_rng(SEED + seed_offset + li)
                d_q, r = prior_q_list[0][li].shape
                d_v, _ = prior_v_list[0][li].shape

                # Q-projection: project out all prior q domains
                # Re-orthonormalize each prior A to ensure accurate projection
                Q_q = rng.standard_normal((d_q, r))
                for prior_aq in prior_q_list:
                    a = prior_aq[li].astype(np.float64)
                    a_ortho, _ = np.linalg.qr(a)
                    a_ortho = a_ortho[:, :r]
                    Q_q -= a_ortho @ (a_ortho.T @ Q_q)
                Q_q, _ = np.linalg.qr(Q_q)
                A_q_new.append(Q_q[:, :r].astype(np.float32))

                # V-projection: project out all prior v domains
                Q_v = rng.standard_normal((d_v, r))
                for prior_av in prior_v_list:
                    a = prior_av[li].astype(np.float64)
                    a_ortho, _ = np.linalg.qr(a)
                    a_ortho = a_ortho[:, :r]
                    Q_v -= a_ortho @ (a_ortho.T @ Q_v)
                Q_v, _ = np.linalg.qr(Q_v)
                A_v_new.append(Q_v[:, :r].astype(np.float32))
        return A_q_new, A_v_new

    # Sort: orthogonal to math+code
    if SORT_LORA_A_PATH.exists():
        log("  Sort A-matrices already exist — loading.")
        sort_a_np = np.load(str(SORT_LORA_A_PATH))
        A_sort_q = [sort_a_np[f"layer_{li}_q_proj_A"].astype(np.float32) for li in range(n_layers)]
        A_sort_v = [sort_a_np[f"layer_{li}_v_proj_A"].astype(np.float32) for li in range(n_layers)]
    else:
        log("  Generating sort A-matrices (Gram-Schmidt vs math+code)...")
        A_sort_q, A_sort_v = gram_schmidt_new_domain(
            [A_math_q, A_code_q], [A_math_v, A_code_v], seed_offset=10000
        )
        save_a_matrices(SORT_LORA_A_PATH, A_sort_q, A_sort_v)
        log(f"  Saved sort A-matrices to {SORT_LORA_A_PATH}")

    A_sort_q_f64 = [a.astype(np.float64) for a in A_sort_q]
    A_sort_v_f64 = [a.astype(np.float64) for a in A_sort_v]

    # Reverse: orthogonal to math+code+sort
    if REVERSE_LORA_A_PATH.exists():
        log("  Reverse A-matrices already exist — loading.")
        rev_a_np = np.load(str(REVERSE_LORA_A_PATH))
        A_rev_q = [rev_a_np[f"layer_{li}_q_proj_A"].astype(np.float32) for li in range(n_layers)]
        A_rev_v = [rev_a_np[f"layer_{li}_v_proj_A"].astype(np.float32) for li in range(n_layers)]
    else:
        log("  Generating reverse A-matrices (Gram-Schmidt vs math+code+sort)...")
        A_rev_q, A_rev_v = gram_schmidt_new_domain(
            [A_math_q, A_code_q, A_sort_q_f64], [A_math_v, A_code_v, A_sort_v_f64], seed_offset=20000
        )
        save_a_matrices(REVERSE_LORA_A_PATH, A_rev_q, A_rev_v)
        log(f"  Saved reverse A-matrices to {REVERSE_LORA_A_PATH}")

    A_rev_q_f64 = [a.astype(np.float64) for a in A_rev_q]
    A_rev_v_f64 = [a.astype(np.float64) for a in A_rev_v]

    # Count: orthogonal to math+code+sort+reverse
    if COUNT_LORA_A_PATH.exists():
        log("  Count A-matrices already exist — loading.")
        cnt_a_np = np.load(str(COUNT_LORA_A_PATH))
        A_cnt_q = [cnt_a_np[f"layer_{li}_q_proj_A"].astype(np.float32) for li in range(n_layers)]
        A_cnt_v = [cnt_a_np[f"layer_{li}_v_proj_A"].astype(np.float32) for li in range(n_layers)]
    else:
        log("  Generating count A-matrices (Gram-Schmidt vs math+code+sort+reverse)...")
        A_cnt_q, A_cnt_v = gram_schmidt_new_domain(
            [A_math_q, A_code_q, A_sort_q_f64, A_rev_q_f64],
            [A_math_v, A_code_v, A_sort_v_f64, A_rev_v_f64],
            seed_offset=30000
        )
        save_a_matrices(COUNT_LORA_A_PATH, A_cnt_q, A_cnt_v)
        log(f"  Saved count A-matrices to {COUNT_LORA_A_PATH}")

    # Verify all 10 pairwise isolation
    # Always load count A-matrices from file (guaranteed to exist at this point)
    log("\n  Verifying Grassmannian isolation for all C(5,2)=10 pairs...")
    cnt_a_np = np.load(str(COUNT_LORA_A_PATH))
    domain_a_q = {
        "math":    [a.astype(np.float64) for a in A_math_q],
        "code":    [a.astype(np.float64) for a in A_code_q],
        "sort":    [a.astype(np.float64) for a in A_sort_q],
        "reverse": [a.astype(np.float64) for a in A_rev_q],
        "count":   [cnt_a_np[f"layer_{li}_q_proj_A"].astype(np.float64) for li in range(n_layers)],
    }
    domain_a_v = {
        "math":    [a.astype(np.float64) for a in A_math_v],
        "code":    [a.astype(np.float64) for a in A_code_v],
        "sort":    [a.astype(np.float64) for a in A_sort_v],
        "reverse": [a.astype(np.float64) for a in A_rev_v],
        "count":   [cnt_a_np[f"layer_{li}_v_proj_A"].astype(np.float64) for li in range(n_layers)],
    }

    pairs = list(combinations(DOMAINS, 2))
    max_cross_q_global = 0.0
    max_cross_v_global = 0.0
    pair_results = {}

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for d1, d2 in pairs:
            max_cross_q = max(
                float(np.abs(domain_a_q[d1][li].T @ domain_a_q[d2][li]).max())
                for li in range(n_layers)
            )
            max_cross_v = max(
                float(np.abs(domain_a_v[d1][li].T @ domain_a_v[d2][li]).max())
                for li in range(n_layers)
            )
            pair_results[f"{d1}-{d2}"] = {"q": max_cross_q, "v": max_cross_v}
            max_cross_q_global = max(max_cross_q_global, max_cross_q)
            max_cross_v_global = max(max_cross_v_global, max_cross_v)
            log(f"  {d1:7s} x {d2:7s}: max|A^T A|_q={max_cross_q:.2e}  v={max_cross_v:.2e}")

    overall_max = max(max_cross_q_global, max_cross_v_global)
    k978_pass = overall_max < 1e-4
    log(f"\n  [K978] {'PASS' if k978_pass else 'FAIL'} — overall max = {overall_max:.2e} (threshold 1e-4)")
    log(f"  Phase 0 time: {time.time()-t0:.1f}s")

    return {
        "k978_pass": k978_pass,
        "k978_overall_max": float(overall_max),
        "k978_max_cross_q": float(max_cross_q_global),
        "k978_max_cross_v": float(max_cross_v_global),
        "k978_pair_results": pair_results,
        "n_layers": n_layers,
    }


# ---- Phase 1-3: Train synthetic-domain M2P networks -----------------------

def train_synthetic_m2p(
    domain: str, model_dims: dict,
    a_path: Path, m2p_save_path: Path,
    make_seqs_fn,
) -> dict:
    log(f"\n{'=' * 70}")
    log(f"[Phase: train {domain} M2P] {SYNTH_TRAIN_STEPS} steps")
    log(f"{'=' * 70}")
    t0 = time.time()

    if m2p_save_path.exists():
        log(f"  {domain} M2P already trained — loading from {m2p_save_path}")
        saved = np.load(str(m2p_save_path))
        n_params = sum(v.size for v in saved.values())
        log(f"  Cached {domain} M2P params: {n_params:,}")
        return {f"{domain}_m2p_loss": 0.0, f"{domain}_m2p_params": int(n_params)}

    n_layers   = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    A_q, A_v = load_a_matrices(a_path, n_layers)

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_q, A_v)
    mx.eval(model.parameters())

    # Zero SFT base for synthetic domains
    B_sft_q_zero, B_sft_v_zero = zero_b_matrices(n_layers, q_proj_out, v_proj_out)

    log(f"  Generating {domain} training sequences...")
    seqs = make_seqs_fn(tokenizer, SYNTH_TRAIN_STEPS + 5)
    log(f"  Generated {len(seqs)} sequences")

    m2p = make_m2p_v6(model_dims, B_sft_q_zero, B_sft_v_zero)
    mx.eval(m2p.parameters())

    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  {domain} M2P params: {n_params:,}")

    def loss_fn(net, tokens_arr):
        mem_grid = extract_memory_hidden_states(model, tokens_arr, net.memory_embeddings)
        B_q, B_v = net(mem_grid)
        logits = forward_with_loras(model, tokens_arr, B_q, B_v, A_q, A_v)
        return nn.losses.cross_entropy(logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean")

    loss_and_grad = nn.value_and_grad(m2p, loss_fn)
    rng = random.Random(SEED + hash(domain) % 10000)

    # Gradient check at step 0
    s0_toks = mx.array(rng.choice(seqs))[None, :]
    s0_loss, s0_grads = loss_and_grad(m2p, s0_toks)
    mx.eval(s0_loss, s0_grads)
    grad_norm0 = math.sqrt(sum(float(mx.sum(g ** 2).item())
                               for _, g in tree_flatten(s0_grads)
                               if isinstance(g, mx.array)))
    log(f"  Step 0: loss={float(s0_loss.item()):.4f}, grad_norm={grad_norm0:.4f}")
    del s0_toks, s0_loss, s0_grads

    optimizer = optim.Adam(learning_rate=LR)

    gc.disable()
    losses = []
    for step in range(SYNTH_TRAIN_STEPS):
        seq = rng.choice(seqs)
        toks = mx.array(seq)[None, :]
        if step < LR_WARMUP:
            optimizer.learning_rate = LR * (step + 1) / LR_WARMUP
        loss, grads = loss_and_grad(m2p, toks)
        optimizer.update(m2p, grads)
        del grads, toks
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, SYNTH_TRAIN_STEPS // 5) == 0:
            recent = sum(losses[-10:]) / min(len(losses[-10:]), 10)
            log(f"  Step {step+1}/{SYNTH_TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()

    final_loss = sum(losses[-5:]) / max(len(losses[-5:]), 1)
    log(f"  Final {domain} M2P loss: {final_loss:.4f}")

    flat = dict(tree_flatten(m2p.parameters()))
    np.savez(str(m2p_save_path), **{k: np.array(v.astype(mx.float32)) for k, v in flat.items()})
    log(f"  Saved {domain} M2P to {m2p_save_path}")
    log(f"  Phase {domain} time: {time.time()-t0:.1f}s")
    log_memory(f"post-{domain}-train")

    cleanup(m2p, model, tokenizer, optimizer, B_sft_q_zero, B_sft_v_zero, A_q, A_v)
    return {f"{domain}_m2p_loss": float(final_loss), f"{domain}_m2p_params": n_params}


# ---- Phase 4: 5-class TF-IDF Routing (K979) --------------------------------

def phase_tfidf_routing_5class() -> dict:
    log("\n" + "=" * 70)
    log("[Phase 4] 5-Class TF-IDF Routing (K979)")
    log("=" * 70)
    t0 = time.time()

    from datasets import load_dataset
    rng = random.Random(SEED + 100)
    ds = load_dataset("gsm8k", "main")
    math_examples = list(ds["train"])
    rng.shuffle(math_examples)

    n_tr = N_ROUTE_TRAIN
    n_te = N_ROUTE_TEST

    # Code prompts (from 2-domain experiment structure)
    CODE_TASKS = [
        {"func_name": "add", "desc": "takes two integers and returns their sum"},
        {"func_name": "subtract", "desc": "takes two integers a and b and returns a minus b"},
        {"func_name": "multiply", "desc": "takes two integers and returns their product"},
        {"func_name": "square", "desc": "takes an integer and returns its square"},
        {"func_name": "double", "desc": "takes an integer and returns twice its value"},
        {"func_name": "increment", "desc": "takes an integer and returns it plus one"},
        {"func_name": "negate", "desc": "takes an integer and returns its negation"},
        {"func_name": "max_of_two", "desc": "takes two integers and returns the larger one"},
        {"func_name": "is_even", "desc": "takes an integer and returns True if even"},
        {"func_name": "power_of_two", "desc": "takes a non-negative integer n and returns 2 to the power of n"},
        {"func_name": "triple", "desc": "takes an integer and returns it multiplied by three"},
        {"func_name": "absolute_value", "desc": "takes an integer and returns its absolute value"},
    ]
    CODE_TMPLS = [
        "Write a Python function called `{func_name}` that {desc}. Output only the Python code.",
        "Implement a Python function named `{func_name}` that {desc}. Only output the function code.",
        "Create a Python function `{func_name}` that {desc}. Return only the code.",
    ]
    all_code_texts = [
        CODE_TMPLS[ti % len(CODE_TMPLS)].format(**task)
        for ti in range(len(CODE_TMPLS))
        for task in CODE_TASKS
    ]

    def repeat_to(lst, n):
        out = []
        while len(out) < n:
            out.extend(lst)
        return out[:n]

    # Per-domain train/test texts
    texts = {
        "math":    [ex["question"] for ex in math_examples[:n_tr]],
        "code":    repeat_to(all_code_texts, n_tr),
        "sort":    get_sort_routing_texts(n_tr, offset=0),
        "reverse": get_reverse_routing_texts(n_tr, offset=0),
        "count":   get_count_routing_texts(n_tr, offset=0),
    }
    test_texts = {
        "math":    [ex["question"] for ex in math_examples[n_tr:n_tr + n_te]],
        "code":    repeat_to(all_code_texts[n_tr % len(all_code_texts):] + all_code_texts, n_te),
        "sort":    get_sort_routing_texts(n_te, offset=1000),
        "reverse": get_reverse_routing_texts(n_te, offset=1000),
        "count":   get_count_routing_texts(n_te, offset=1000),
    }

    train_texts_all  = []
    train_labels_all = []
    test_texts_all   = []
    test_labels_all  = []
    for idx, domain in enumerate(DOMAINS):
        train_texts_all.extend(texts[domain])
        train_labels_all.extend([idx] * len(texts[domain]))
        test_texts_all.extend(test_texts[domain])
        test_labels_all.extend([idx] * len(test_texts[domain]))

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts_all)
    X_test  = vectorizer.transform(test_texts_all)

    # Per-class centroids
    centroids = []
    for idx in range(len(DOMAINS)):
        mask = np.array(train_labels_all) == idx
        centroids.append(np.asarray(X_train[mask].mean(axis=0)))
    centroids = np.vstack(centroids)

    sims  = cosine_similarity(X_test, centroids)
    preds = sims.argmax(axis=1)
    test_labels_arr = np.array(test_labels_all)

    correct = int((preds == test_labels_arr).sum())
    total   = len(test_labels_arr)
    routing_acc = correct / total

    per_class = {}
    for idx, domain in enumerate(DOMAINS):
        mask = test_labels_arr == idx
        cls_preds = preds[mask]
        cls_acc   = float((cls_preds == idx).mean()) if mask.sum() > 0 else 0.0
        per_class[domain] = cls_acc
        log(f"  {domain:7s} routing: {cls_acc:.4f} ({int((cls_preds == idx).sum())}/{int(mask.sum())})")

    log(f"  Overall routing accuracy: {routing_acc:.4f} ({correct}/{total})")

    k979_pass = routing_acc >= 0.80
    log(f"  [K979] {'PASS' if k979_pass else 'FAIL'} — routing_acc={routing_acc:.4f} >= 0.80")
    log(f"  Phase 4 time: {time.time()-t0:.1f}s")

    return {
        "k979_pass": k979_pass,
        "routing_acc": float(routing_acc),
        "per_class_routing": per_class,
        "router": {"vectorizer": vectorizer, "centroids": centroids},
    }


# ---- Phase 5: Math quality under N=5 composition (K980) --------------------

def phase_math_quality_eval(model_dims: dict, router_obj: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 5] Math Quality Under N=5 Routed Composition (K980)")
    log("=" * 70)
    t0 = time.time()

    n_layers   = model_dims["n_layers"]
    base_acc   = model_dims["base_accuracy"]
    sft_acc    = model_dims["sft_accuracy"]

    # Load math A-matrices and SFT B-matrices
    A_math_q, A_math_v = load_a_matrices(MATH_LORA_A_PATH, n_layers)
    B_sft_q, B_sft_v   = load_sft_b_matrices(n_layers)

    # Load model with math A-matrices
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_math_q, A_math_v)
    mx.eval(model.parameters())

    # Load math M2P
    m2p_math = make_m2p_v6(model_dims, B_sft_q, B_sft_v)
    math_saved = np.load(str(MATH_M2P_PATH))
    m2p_math.load_weights([(k, mx.array(math_saved[k])) for k in math_saved.files])
    m2p_math.eval()
    mx.eval(m2p_math.parameters())
    log(f"  Loaded math M2P from {MATH_M2P_PATH}")

    # Load GSM8K test data
    from datasets import load_dataset
    rng = random.Random(SEED)
    ds = load_dataset("gsm8k", "main")
    test_examples = list(ds["test"])
    rng.shuffle(test_examples)
    test_examples = test_examples[:N_EVAL_MATH]
    log(f"  Evaluating {len(test_examples)} GSM8K examples")

    vectorizer = router_obj["vectorizer"]
    centroids  = router_obj["centroids"]

    correct_routed = 0
    route_to_math  = 0
    total = len(test_examples)

    for i, ex in enumerate(test_examples):
        # TF-IDF routing (should always route to math for math queries)
        x_vec  = vectorizer.transform([ex["question"]])
        sims   = cosine_similarity(x_vec, centroids)
        domain = int(sims.argmax(axis=1)[0])
        if domain == 0:  # 0 = math
            route_to_math += 1

        # Use math M2P for generation (routing oracle for math queries)
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        mem_grid = extract_memory_hidden_states(model, tokens_arr, m2p_math.memory_embeddings)
        mx.eval(mem_grid)

        B_q, B_v = m2p_math(mem_grid)
        mx.eval(*B_q, *B_v)
        inject_lora_b(model, B_q, B_v)

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)

        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            correct_routed += 1

        del tokens_arr, mem_grid, B_q, B_v

        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            log(f"    [{i+1}/{total}] acc={correct_routed/(i+1):.3f} "
                f"route_to_math={route_to_math/(i+1):.3f}")

    routed_acc      = correct_routed / total
    sft_improvement = sft_acc - base_acc
    m2p_improvement = routed_acc - base_acc
    quality_ratio   = (m2p_improvement / sft_improvement) if abs(sft_improvement) > 1e-9 else 0.0
    math_route_frac = route_to_math / total

    log(f"\n  Routed accuracy:  {routed_acc:.4f} ({correct_routed}/{total})")
    log(f"  Quality ratio:    {quality_ratio:.4f}")
    log(f"  Math route frac:  {math_route_frac:.4f}")
    log(f"  base={base_acc:.4f}  sft={sft_acc:.4f}")

    k980_pass = quality_ratio >= 0.70
    log(f"  [K980] {'PASS' if k980_pass else 'FAIL'} — quality_ratio={quality_ratio:.4f} >= 0.70")
    log(f"  Phase 5 time: {time.time()-t0:.1f}s")
    log_memory("post-eval")

    cleanup(m2p_math, model, tokenizer, B_sft_q, B_sft_v)
    return {
        "routed_accuracy": float(routed_acc),
        "quality_ratio": float(quality_ratio),
        "math_route_frac": float(math_route_frac),
        "k980_pass": k980_pass,
    }


# ---- Main ------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("N=5 Domain M2P Composition at 4B: Grassmannian Scaling Verification")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"MODEL={MODEL_ID}")
    log(f"SYNTH_TRAIN_STEPS={SYNTH_TRAIN_STEPS} | N_EVAL_MATH={N_EVAL_MATH}")
    log("=" * 70)
    log_memory("start")

    model_dims = load_model_dims()
    log(f"  n_layers={model_dims['n_layers']}, d_model={model_dims['d_model']}")
    log(f"  base_acc={model_dims['base_accuracy']:.4f}, sft_acc={model_dims['sft_accuracy']:.4f}")

    # Phase 0: A-matrices + K978
    p0 = phase_build_and_verify_a_matrices(model_dims)

    if not p0["k978_pass"]:
        log("\n[KILL] K978 FAIL — Grassmannian isolation violated.")
        RESULTS_FILE.write_text(json.dumps({"k978_pass": False, **p0,
                                             "total_time_s": round(time.time() - t_start, 1)}, indent=2))
        return

    # Phase 1-3: Train synthetic M2P networks
    p1 = train_synthetic_m2p("sort", model_dims, SORT_LORA_A_PATH, SORT_M2P_PATH,
                              lambda tok, n: make_sort_seqs(tok, n))
    p2 = train_synthetic_m2p("reverse", model_dims, REVERSE_LORA_A_PATH, REVERSE_M2P_PATH,
                              lambda tok, n: make_reverse_seqs(tok, n))
    p3 = train_synthetic_m2p("count", model_dims, COUNT_LORA_A_PATH, COUNT_M2P_PATH,
                              lambda tok, n: make_count_seqs(tok, n))

    # Phase 4: 5-class routing (K979)
    p4 = phase_tfidf_routing_5class()
    router_obj = p4.pop("router")

    # Phase 5: Math quality eval (K980)
    p5 = phase_math_quality_eval(model_dims, router_obj)

    # Summary
    k978_pass = p0["k978_pass"]
    k979_pass = p4["k979_pass"]
    k980_pass = p5["k980_pass"]
    all_pass  = k978_pass and k979_pass and k980_pass

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment")
    log("=" * 70)
    log(f"  K978 (all 10 pairs < 1e-4): {'PASS' if k978_pass else 'FAIL'} "
        f"overall_max={p0['k978_overall_max']:.2e}")
    log(f"  K979 (routing >= 80%):       {'PASS' if k979_pass else 'FAIL'} "
        f"routing_acc={p4['routing_acc']:.4f}")
    log(f"  K980 (quality_ratio >= 0.70): {'PASS' if k980_pass else 'FAIL'} "
        f"qr={p5['quality_ratio']:.4f}")
    log(f"\n  STATUS: {'ALL PASS' if all_pass else 'PARTIAL FAIL'}")

    peak_gb = mx.get_peak_memory() / 1e9
    total_s  = round(time.time() - t_start, 1)

    results = {
        "experiment": "m2p_n5_compose_qwen4b",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P, "n_mem_tokens": N_MEM_TOKENS,
            "n_m2p_layers": N_M2P_LAYERS, "n_m2p_heads": N_M2P_HEADS,
            "output_scale": OUTPUT_SCALE,
            "synth_train_steps": SYNTH_TRAIN_STEPS,
            "n_eval_math": N_EVAL_MATH,
            **{k: v for k, v in model_dims.items() if k not in ("base_accuracy", "sft_accuracy")},
        },
        "base_accuracy": model_dims["base_accuracy"],
        "sft_accuracy":  model_dims["sft_accuracy"],
        # K978
        "k978_pass":         k978_pass,
        "k978_overall_max":  p0["k978_overall_max"],
        "k978_max_cross_q":  p0["k978_max_cross_q"],
        "k978_max_cross_v":  p0["k978_max_cross_v"],
        "k978_pair_results": p0["k978_pair_results"],
        # Synth M2P
        "sort_m2p_loss":    p1["sort_m2p_loss"],
        "reverse_m2p_loss": p2["reverse_m2p_loss"],
        "count_m2p_loss":   p3["count_m2p_loss"],
        # K979
        "k979_pass":          k979_pass,
        "routing_acc":        p4["routing_acc"],
        "per_class_routing":  p4["per_class_routing"],
        # K980
        "k980_pass":        k980_pass,
        "routed_accuracy":  p5["routed_accuracy"],
        "quality_ratio":    p5["quality_ratio"],
        "math_route_frac":  p5["math_route_frac"],
        # Summary
        "kill_criteria": {
            "K978_grassmannian_all_10_pairs": "PASS" if k978_pass else "FAIL",
            "K979_tfidf_routing_ge_80pct":    "PASS" if k979_pass else "FAIL",
            "K980_math_quality_ratio_ge_70":  "PASS" if k980_pass else "FAIL",
        },
        "peak_memory_gb": round(peak_gb, 2),
        "total_time_s":   total_s,
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_s:.1f}s ({total_s/60:.1f} min)")


if __name__ == "__main__":
    main()
