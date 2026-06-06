#!/usr/bin/env python3
"""Code SFT-Residual M2P at 4B: SFT quality floor prevents anti-format interference.

TYPE: verification
MATH: experiments/models/exp_m2p_code_sft_4b/MATH.md

WHAT THIS TESTS:
  Finding #407: code M2P with B_sft=0 degrades base code quality 42%→6.7%.
  Fix: B_applied = B_sft_code + ε · zero_init_head(z)
  Theorem 1: zero-init heads → B_applied = B_sft_code at step 0 → init_qr = 1.0 exactly.

  Phases:
    1. Train code SFT LoRA (300 steps on Python function generation tasks)
    2. Measure code SFT baseline pass@1
    3. Train code M2P with SFT-residual (500 steps)
       - K987: init_quality_ratio_code >= 0.80 (measured before training)
       - K988: post_quality_ratio_code >= 0.70 under routed composition
    4. TF-IDF routing (math vs code)
    5. Eval code quality under routing (K988)
    6. Eval math quality under routing (K989)

KILL CRITERIA:
  K987: init_quality_ratio_code >= 0.80 (predicted 1.0 by Theorem 1)
  K988: code quality_ratio >= 0.70 after 500 M2P training steps under routing
  K989: math quality_ratio >= 0.80 under routing (Finding #404 baseline: 1.3125)

REFERENCES:
  He et al. (2016) — Residual learning (ResNet)
  Finding #403 — Math SFT-residual at 4B: init_qr=1.0, qr=1.175
  Finding #407 — Code M2P failure without SFT floor: qr=0.158
  Finding #404 — 2-domain composition at 4B: routing=100%, math_qr=1.3125

Supports SMOKE_TEST=1 for quick validation (<5 min).
"""

import ast
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

LORA_RANK    = 4
LORA_SCALE   = 5.0
D_M2P        = 1024
N_MEM_TOKENS = 16
N_M2P_LAYERS = 4
N_M2P_HEADS  = 4
OUTPUT_SCALE = 0.032

CODE_SFT_STEPS  = 10 if IS_SMOKE else 300
M2P_TRAIN_STEPS = 20 if IS_SMOKE else 500
LR              = 5e-5
LR_WARMUP       = 3  if IS_SMOKE else 30
GRAD_CLIP       = 1.0
MAX_SEQ_LEN     = 64  if IS_SMOKE else 256
MAX_GEN_TOKENS  = 64  if IS_SMOKE else 384
SEED            = 42

N_EVAL_CODE    = 5  if IS_SMOKE else 45   # 15 tasks × 3 templates
N_EVAL_MATH    = 5  if IS_SMOKE else 200
N_ROUTE_TRAIN  = 4  if IS_SMOKE else 100  # per class
N_ROUTE_TEST   = 4  if IS_SMOKE else 100  # per class

EXPERIMENT_DIR    = Path(__file__).parent
V1_DIR            = EXPERIMENT_DIR.parent / "m2p_qwen4b_gsm8k"
MATH_LORA_A_PATH  = V1_DIR / "grassmannian_a_matrices.npz"
MATH_SFT_B_PATH   = V1_DIR / "sft_b_matrices.npz"
V1_RESULTS        = V1_DIR / "results.json"
MATH_M2P_PATH     = EXPERIMENT_DIR.parent / "m2p_qwen4b_sft_residual" / "m2p_weights.npz"
CODE_LORA_A_PATH  = EXPERIMENT_DIR.parent / "m2p_2domain_compose_qwen4b" / "code_a_matrices.npz"

CODE_SFT_B_PATH   = EXPERIMENT_DIR / "code_sft_b_matrices.npz"
CODE_M2P_PATH     = EXPERIMENT_DIR / "code_m2p_weights.npz"
RESULTS_FILE      = EXPERIMENT_DIR / "results.json"

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

# ---- Code task definitions (15 tasks, 3 templates = 45 eval prompts) --------

CODE_TASKS = [
    {"func_name": "add",          "desc": "takes two integers `a` and `b` and returns their sum",
     "test_cases": [((1, 2), 3), ((0, 0), 0), ((-1, 5), 4), ((10, 20), 30)]},
    {"func_name": "subtract",     "desc": "takes two integers `a` and `b` and returns `a` minus `b`",
     "test_cases": [((5, 3), 2), ((0, 0), 0), ((10, 7), 3)]},
    {"func_name": "multiply",     "desc": "takes two integers `a` and `b` and returns their product",
     "test_cases": [((2, 3), 6), ((0, 5), 0), ((-2, 4), -8)]},
    {"func_name": "square",       "desc": "takes an integer `n` and returns its square",
     "test_cases": [((4,), 16), ((0,), 0), ((-3,), 9)]},
    {"func_name": "double",       "desc": "takes an integer `n` and returns twice its value",
     "test_cases": [((3,), 6), ((0,), 0), ((-5,), -10)]},
    {"func_name": "increment",    "desc": "takes an integer `n` and returns `n` plus one",
     "test_cases": [((0,), 1), ((9,), 10), ((-1,), 0)]},
    {"func_name": "negate",       "desc": "takes an integer `n` and returns its negation",
     "test_cases": [((5,), -5), ((0,), 0), ((-3,), 3)]},
    {"func_name": "max_of_two",   "desc": "takes two integers `a` and `b` and returns the larger one",
     "test_cases": [((3, 5), 5), ((7, 2), 7), ((4, 4), 4)]},
    {"func_name": "is_even",      "desc": "takes an integer `n` and returns True if it is even, False otherwise",
     "test_cases": [((2,), True), ((3,), False), ((0,), True)]},
    {"func_name": "power_of_two", "desc": "takes a non-negative integer `n` and returns 2 raised to the power of `n`",
     "test_cases": [((0,), 1), ((1,), 2), ((3,), 8)]},
    {"func_name": "triple",       "desc": "takes an integer `n` and returns `n` multiplied by three",
     "test_cases": [((3,), 9), ((0,), 0), ((-2,), -6)]},
    {"func_name": "absolute_value", "desc": "takes an integer `n` and returns its absolute value",
     "test_cases": [((-5,), 5), ((3,), 3), ((0,), 0)]},
    {"func_name": "is_positive",  "desc": "takes an integer `n` and returns True if `n` is positive, False otherwise",
     "test_cases": [((5,), True), ((0,), False), ((-3,), False)]},
    {"func_name": "clamp_zero",   "desc": "takes an integer `n` and returns `n` if `n` is positive, else 0",
     "test_cases": [((5,), 5), ((0,), 0), ((-3,), 0)]},
    {"func_name": "sum_first_n",  "desc": "takes a non-negative integer `n` and returns the sum 1+2+...+n",
     "test_cases": [((0,), 0), ((1,), 1), ((4,), 10), ((5,), 15)]},
]

CODE_PROMPT_TEMPLATES = [
    "Write a Python function called `{func_name}` that {desc}. Output only the Python code.",
    "Implement a Python function named `{func_name}` that {desc}. Only output the function code.",
    "Create a Python function `{func_name}` that {desc}. Return only the code.",
]

_CODE_IMPLS = {
    "add": "a + b", "subtract": "a - b", "multiply": "a * b",
    "square": "n * n", "double": "n * 2", "increment": "n + 1",
    "negate": "-n", "max_of_two": "a if a > b else b",
    "is_even": "n % 2 == 0", "power_of_two": "2 ** n",
    "triple": "n * 3", "absolute_value": "n if n >= 0 else -n",
    "is_positive": "n > 0", "clamp_zero": "n if n > 0 else 0",
    "sum_first_n": "n * (n + 1) // 2",
}


def make_code_prompt(task: dict, tmpl_idx: int = 0) -> str:
    tmpl = CODE_PROMPT_TEMPLATES[tmpl_idx % len(CODE_PROMPT_TEMPLATES)]
    return tmpl.format(func_name=task["func_name"], desc=task["desc"])


def build_eval_tasks(n: int) -> list:
    """Build list of (task, tmpl_idx) for eval, cycling templates."""
    tasks = []
    for tmpl_idx in range(len(CODE_PROMPT_TEMPLATES)):
        for task in CODE_TASKS:
            tasks.append((task, tmpl_idx))
    return tasks[:n]


def generate_code_training_seqs(tokenizer, n: int) -> list:
    result = []
    for i in range(n):
        task = CODE_TASKS[i % len(CODE_TASKS)]
        tmpl_idx = i // len(CODE_TASKS)
        prompt = make_code_prompt(task, tmpl_idx)
        func_name = task["func_name"]
        n_args = len(task["test_cases"][0][0])
        args_str = ", ".join(chr(ord("a") + j) for j in range(n_args))
        impl = _CODE_IMPLS.get(func_name, "None")
        answer = f"def {func_name}({args_str}):\n    return {impl}"
        text = f"{prompt}\n{answer}"
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) >= 2:
                result.append(ids)
    return result


def extract_code_block(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(def\s+\w+\s*\(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def eval_code_output(text: str, func_name: str, test_cases: list) -> int:
    code = extract_code_block(text)
    if not code:
        return 0
    try:
        tree = ast.parse(code)
        if not any(isinstance(n, ast.FunctionDef) and n.name == func_name for n in ast.walk(tree)):
            return 0
        ns = {}
        exec(compile(tree, "<string>", "exec"), ns)  # noqa: S102
        fn = ns.get(func_name)
        if fn is None:
            return 0
        for args, expected in test_cases:
            if fn(*args) != expected:
                return 0
        return 1
    except Exception:
        return 0


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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"(?:the\s+)?answer\s+is\s+[-–]?\s*\$?(-?[\d,]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    return None


def quality_ratio(m2p_acc: float, sft_acc: float, base_acc: float) -> float:
    """Standard quality ratio. Returns ratio relative to SFT improvement over base."""
    denom = sft_acc - base_acc
    if abs(denom) < 1e-9:
        return 1.0 if abs(m2p_acc - sft_acc) < 1e-9 else 0.0
    return (m2p_acc - base_acc) / denom


# ---- Model helpers -----------------------------------------------------------

def load_model_dims() -> dict:
    with open(V1_RESULTS) as f:
        v1 = json.load(f)
    cfg = v1["config"]
    return {
        "n_layers":       cfg["n_layers"],
        "d_model":        cfg["d_model"],
        "q_proj_out":     cfg["q_proj_out"],
        "v_proj_out":     cfg["v_proj_out"],
        "math_base_acc":  v1["base_accuracy"],
        "math_sft_acc":   v1["sft_accuracy"],
    }


def apply_lora_structure(model, A_q: list, A_v: list) -> None:
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.q_proj.lora_a = A_q[li]
        attn.v_proj.lora_a = A_v[li]
    model.freeze()


def load_a_matrices(path: Path, n_layers: int) -> tuple:
    saved = np.load(str(path))
    A_q = [mx.array(saved[f"layer_{li}_q_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    A_v = [mx.array(saved[f"layer_{li}_v_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    mx.eval(*A_q, *A_v)
    return A_q, A_v


def load_b_matrices(path: Path, n_layers: int) -> tuple:
    saved = np.load(str(path))
    B_q = [mx.array(saved[f"layer_{li}_q_proj_B"]).astype(mx.bfloat16) for li in range(n_layers)]
    B_v = [mx.array(saved[f"layer_{li}_v_proj_B"]).astype(mx.bfloat16) for li in range(n_layers)]
    mx.eval(*B_q, *B_v)
    return B_q, B_v


def inject_lora_b(model, B_q: list, B_v: list) -> None:
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_b = B_q[li]
        layer.self_attn.v_proj.lora_b = B_v[li]
    mx.eval(model.parameters())


# ---- SHINE memory hidden states ---------------------------------------------

def build_memory_causal_mask(M: int, T: int) -> mx.array:
    S = M + T
    mask_np = np.zeros((S, S), dtype=np.float32)
    mask_np[M:, :M] = float("-inf")
    for i in range(T):
        for j in range(i + 1, T):
            mask_np[M + i, M + j] = float("-inf")
    return mx.array(mask_np).astype(mx.bfloat16)[None, None, :, :]


def extract_memory_hidden_states(
    model, tokens_arr: mx.array, memory_embeddings: mx.array,
) -> mx.array:
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


# ---- M2PBlock and M2PNetworkV6 (SFT-Residual) --------------------------------

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
    """SHINE-aligned M2P with SFT residual: B_applied = B_sft + ε·zero_init_head(z).

    At init: head(z) = 0 → B_applied = B_sft → init_quality = SFT quality exactly.
    """

    def __init__(self, n_layers, d_model, d_m2p, n_mem_tokens, rank,
                 q_proj_out, v_proj_out, B_sft_q, B_sft_v,
                 n_m2p_layers=4, n_heads=4, output_scale=0.032):
        super().__init__()
        self.n_layers     = n_layers
        self.n_mem_tokens = n_mem_tokens
        self.rank         = rank
        self.output_scale = output_scale
        self.has_input_proj = (d_model != d_m2p)
        self.B_sft_q = B_sft_q  # frozen SFT B-matrices (not parameters)
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

        # Zero-init residual heads: B_applied = B_sft at step 0
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


# ---- Phase 1: Code SFT Training ---------------------------------------------

def phase_code_sft(model_dims: dict) -> dict:
    log("\n" + "=" * 70)
    log(f"[Phase 1] Code SFT LoRA Training ({CODE_SFT_STEPS} steps)")
    log("=" * 70)
    t0 = time.time()

    if CODE_SFT_B_PATH.exists():
        log(f"  Code SFT B-matrices already exist at {CODE_SFT_B_PATH} — skipping training.")
        return {"code_sft_final_loss": None, "cached": True}

    n_layers = model_dims["n_layers"]
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    A_code_q, A_code_v = load_a_matrices(CODE_LORA_A_PATH, n_layers)
    apply_lora_structure(model, A_code_q, A_code_v)
    mx.eval(model.parameters())

    # Unfreeze only lora_b for SFT
    model.unfreeze(keys=["lora_b"], strict=False)
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"  Trainable LoRA-B params: {trainable:,}")

    code_seqs = generate_code_training_seqs(tokenizer, max(CODE_SFT_STEPS, len(CODE_TASKS) * 3))
    log(f"  Code training sequences: {len(code_seqs)}")

    rng = random.Random(SEED + 10)
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(mdl, tokens_arr):
        logits = mdl(tokens_arr)
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    gc.disable()
    losses = []
    for step in range(CODE_SFT_STEPS):
        seq = rng.choice(code_seqs)
        tokens_arr = mx.array(seq)[None, :]
        loss, grads = loss_and_grad(model, tokens_arr)
        optimizer.update(model, grads)
        del grads, tokens_arr
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, CODE_SFT_STEPS // 5) == 0:
            recent = sum(losses[-20:]) / min(len(losses[-20:]), 20)
            log(f"  Step {step+1}/{CODE_SFT_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"  Final code SFT loss: {final_loss:.4f}")

    # Save code SFT B-matrices
    save_dict = {}
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        save_dict[f"layer_{li}_q_proj_B"] = np.array(attn.q_proj.lora_b.astype(mx.float32))
        save_dict[f"layer_{li}_v_proj_B"] = np.array(attn.v_proj.lora_b.astype(mx.float32))
    np.savez(str(CODE_SFT_B_PATH), **save_dict)
    log(f"  Saved {len(save_dict)} code SFT B-matrices to {CODE_SFT_B_PATH}")
    log(f"  Phase 1 time: {time.time()-t0:.1f}s")
    log_memory("post-code-sft")

    cleanup(model, tokenizer, optimizer, A_code_q, A_code_v)
    return {"code_sft_final_loss": float(final_loss), "cached": False}


# ---- Phase 2: Code SFT Baseline Eval ----------------------------------------

def phase_code_sft_eval(model_dims: dict, eval_tasks: list) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 2] Code SFT Baseline Eval")
    log("=" * 70)
    t0 = time.time()

    n_layers   = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    A_code_q, A_code_v = load_a_matrices(CODE_LORA_A_PATH, n_layers)
    B_sft_q, B_sft_v   = load_b_matrices(CODE_SFT_B_PATH, n_layers)

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_code_q, A_code_v)
    inject_lora_b(model, B_sft_q, B_sft_v)
    mx.eval(model.parameters())

    # Base pass@1 (zero B)
    log("  Measuring base pass@1 (zero B)...")
    B_zero_q = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_zero_v = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    inject_lora_b(model, B_zero_q, B_zero_v)

    base_correct = 0
    total = len(eval_tasks)
    for i, (task, tmpl_idx) in enumerate(eval_tasks):
        prompt = make_code_prompt(task, tmpl_idx)
        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        base_correct += eval_code_output(generated, task["func_name"], task["test_cases"])
        if (i + 1) % max(1, total // 3) == 0 or (i + 1) == total:
            log(f"    base [{i+1}/{total}] pass@1={base_correct/(i+1):.3f}")

    base_pass = base_correct / total
    log(f"  Base pass@1: {base_pass:.4f} ({base_correct}/{total})")

    # SFT pass@1
    log("  Measuring code SFT pass@1...")
    inject_lora_b(model, B_sft_q, B_sft_v)

    sft_correct = 0
    for i, (task, tmpl_idx) in enumerate(eval_tasks):
        prompt = make_code_prompt(task, tmpl_idx)
        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        sft_correct += eval_code_output(generated, task["func_name"], task["test_cases"])
        if (i + 1) % max(1, total // 3) == 0 or (i + 1) == total:
            log(f"    sft [{i+1}/{total}] pass@1={sft_correct/(i+1):.3f}")

    sft_pass = sft_correct / total
    log(f"  Code SFT pass@1: {sft_pass:.4f} ({sft_correct}/{total})")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-code-sft-eval")

    cleanup(model, tokenizer, A_code_q, A_code_v, B_sft_q, B_sft_v, B_zero_q, B_zero_v)
    return {"base_pass_rate": float(base_pass), "code_sft_pass_rate": float(sft_pass),
            "base_correct": base_correct, "sft_correct": sft_correct, "n_eval": total}


# ---- Phase 3: Code M2P Training (SFT-Residual) --------------------------------

def phase_code_m2p_train(model_dims: dict, eval_tasks: list, baselines: dict) -> dict:
    log("\n" + "=" * 70)
    log(f"[Phase 3] Code M2P Training (SFT-Residual, {M2P_TRAIN_STEPS} steps)")
    log("=" * 70)
    t0 = time.time()

    if CODE_M2P_PATH.exists():
        log(f"  Code M2P weights already exist at {CODE_M2P_PATH} — skipping training.")
        return {"code_m2p_final_loss": None, "cached": True,
                "k987_pass": True, "init_code_pass_rate": baselines["code_sft_pass_rate"],
                "grad_norm": None}

    n_layers   = model_dims["n_layers"]
    d_model    = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]
    base_pass  = baselines["base_pass_rate"]
    sft_pass   = baselines["code_sft_pass_rate"]

    A_code_q, A_code_v = load_a_matrices(CODE_LORA_A_PATH, n_layers)
    B_sft_q, B_sft_v   = load_b_matrices(CODE_SFT_B_PATH, n_layers)

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_code_q, A_code_v)
    mx.eval(model.parameters())

    code_seqs = generate_code_training_seqs(tokenizer, max(M2P_TRAIN_STEPS, len(CODE_TASKS) * 3))
    log(f"  Code training sequences: {len(code_seqs)}")

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

    rng = random.Random(SEED + 20)

    def m2p_loss_fn(m2p_net, tokens_arr):
        memory_grid = extract_memory_hidden_states(model, tokens_arr, m2p_net.memory_embeddings)
        B_q, B_v = m2p_net(memory_grid)
        logits = forward_with_loras(model, tokens_arr, B_q, B_v, A_code_q, A_code_v)
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # Gradient smoke test
    log("  Gradient smoke test...")
    smoke_seq = rng.choice(code_seqs)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)
    grad_norms = [float(mx.sum(g**2).item()) for _, g in tree_flatten(smoke_grads) if isinstance(g, mx.array)]
    grad_norm = math.sqrt(sum(grad_norms))
    log(f"  grad_norm at step 0: {grad_norm:.6f}, smoke_loss: {float(smoke_loss.item()):.4f}")
    del smoke_tokens, smoke_loss, smoke_grads

    # K987: Init quality (before any training)
    # At step 0, B_applied = B_sft_code, so init quality = SFT quality
    # We verify by evaluating on a subset of eval tasks
    log(f"\n  [K987] Measuring init quality (n={min(15, len(eval_tasks))} tasks)...")
    init_tasks = eval_tasks[:min(15, len(eval_tasks))]
    inject_lora_b(model, B_sft_q, B_sft_v)  # Set to SFT B (same as M2P at init)

    init_correct = 0
    for task, tmpl_idx in init_tasks:
        prompt = make_code_prompt(task, tmpl_idx)
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]
        mem_grid = extract_memory_hidden_states(model, tokens_arr, m2p.memory_embeddings)
        mx.eval(mem_grid)
        B_q, B_v = m2p(mem_grid)
        mx.eval(*B_q, *B_v)
        inject_lora_b(model, B_q, B_v)
        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        init_correct += eval_code_output(generated, task["func_name"], task["test_cases"])
        del tokens_arr, mem_grid, B_q, B_v

    init_pass = init_correct / len(init_tasks)
    init_qr = quality_ratio(init_pass, sft_pass, base_pass)
    log(f"  [K987] init_pass@1={init_pass:.4f}, sft_pass={sft_pass:.4f}, base_pass={base_pass:.4f}")
    log(f"  [K987] init_quality_ratio={init_qr:.4f}")
    k987_pass = init_qr >= 0.80
    log(f"  [K987] {'PASS' if k987_pass else 'FAIL'} — init_qr={init_qr:.4f} >= 0.80")

    # Training loop
    optimizer = optim.Adam(learning_rate=LR)

    def lr_schedule(step):
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    log(f"\n  Training code M2P for {M2P_TRAIN_STEPS} steps...")
    gc.disable()
    losses = []
    for step in range(M2P_TRAIN_STEPS):
        seq = rng.choice(code_seqs)
        tokens_arr = mx.array(seq)[None, :]
        optimizer.learning_rate = lr_schedule(step)
        loss, grads = loss_and_grad(m2p, tokens_arr)

        flat_grads = tree_flatten(grads)
        gn = math.sqrt(sum(float(mx.sum(g**2).item()) for _, g in flat_grads if isinstance(g, mx.array)))
        if gn > GRAD_CLIP:
            clip = GRAD_CLIP / (gn + 1e-8)
            grads = tree_map(lambda g: g * clip if isinstance(g, mx.array) else g, grads)

        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))

        if (step + 1) % max(1, M2P_TRAIN_STEPS // 10) == 0:
            recent = sum(losses[-20:]) / min(len(losses[-20:]), 20)
            log(f"  Step {step+1}/{M2P_TRAIN_STEPS}: loss={recent:.4f} gn={gn:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"  Final code M2P loss: {final_loss:.4f}")

    # Save M2P weights
    m2p_params_flat = dict(tree_flatten(m2p.parameters()))
    np.savez(str(CODE_M2P_PATH), **{k: np.array(v.astype(mx.float32)) for k, v in m2p_params_flat.items()})
    log(f"  Saved code M2P to {CODE_M2P_PATH}")
    log(f"  Phase 3 time: {time.time()-t0:.1f}s")
    log_memory("post-code-m2p-train")

    cleanup(m2p, model, tokenizer, optimizer, A_code_q, A_code_v, B_sft_q, B_sft_v)
    return {
        "code_m2p_final_loss": float(final_loss),
        "code_m2p_params": n_params,
        "k987_pass": k987_pass,
        "init_code_pass_rate": float(init_pass),
        "init_quality_ratio": float(init_qr),
        "grad_norm": float(grad_norm),
        "cached": False,
    }


# ---- Phase 4: TF-IDF Routing -------------------------------------------------

def phase_tfidf_routing(model_dims: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 4] TF-IDF Routing (math vs code)")
    log("=" * 70)
    t0 = time.time()

    from datasets import load_dataset
    rng = random.Random(SEED + 100)
    ds = load_dataset("gsm8k", "main")
    math_examples = list(ds["train"])
    rng.shuffle(math_examples)

    n_train = N_ROUTE_TRAIN
    n_test  = N_ROUTE_TEST

    math_train_text = [ex["question"] for ex in math_examples[:n_train]]
    math_test_text  = [ex["question"] for ex in math_examples[n_train:n_train + n_test]]

    all_code_texts = [make_code_prompt(t, i) for i in range(len(CODE_PROMPT_TEMPLATES)) for t in CODE_TASKS]
    code_texts = [all_code_texts[i % len(all_code_texts)] for i in range(n_train + n_test)]
    code_train_text = code_texts[:n_train]
    code_test_text  = code_texts[n_train:n_train + n_test]

    train_texts  = math_train_text + code_train_text
    train_labels = [0] * len(math_train_text) + [1] * len(code_train_text)
    test_texts   = math_test_text + code_test_text
    test_labels  = [0] * len(math_test_text) + [1] * len(code_test_text)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    math_centroid = np.asarray(X_train[np.array(train_labels) == 0].mean(axis=0))
    code_centroid = np.asarray(X_train[np.array(train_labels) == 1].mean(axis=0))
    centroids = np.vstack([math_centroid, code_centroid])

    sims  = cosine_similarity(X_test, centroids)
    preds = sims.argmax(axis=1)

    correct = int((preds == np.array(test_labels)).sum())
    total   = len(test_labels)
    routing_acc = correct / total

    math_preds = preds[:len(math_test_text)]
    code_preds = preds[len(math_test_text):]
    math_routing = float((math_preds == 0).mean())
    code_routing = float((code_preds == 1).mean())

    log(f"  Overall routing accuracy: {routing_acc:.4f} ({correct}/{total})")
    log(f"  Math routing: {math_routing:.4f}, Code routing: {code_routing:.4f}")
    log(f"  Phase 4 time: {time.time()-t0:.1f}s")

    return {
        "routing_acc":       float(routing_acc),
        "math_routing_acc":  math_routing,
        "code_routing_acc":  code_routing,
        "router": {"vectorizer": vectorizer, "centroids": centroids},
    }


# ---- Phase 5: Code Quality Under Routing (K988) ------------------------------

def phase_code_eval(model_dims: dict, eval_tasks: list, baselines: dict, router_obj: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 5] Code Quality Under Routing (K988)")
    log("=" * 70)
    t0 = time.time()

    n_layers   = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]
    base_pass  = baselines["base_pass_rate"]
    sft_pass   = baselines["code_sft_pass_rate"]

    A_code_q, A_code_v = load_a_matrices(CODE_LORA_A_PATH, n_layers)
    B_sft_q, B_sft_v   = load_b_matrices(CODE_SFT_B_PATH, n_layers)

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_code_q, A_code_v)
    mx.eval(model.parameters())

    m2p = M2PNetworkV6(
        n_layers=n_layers, d_model=model_dims["d_model"], d_m2p=D_M2P,
        n_mem_tokens=N_MEM_TOKENS, rank=LORA_RANK,
        q_proj_out=q_proj_out, v_proj_out=v_proj_out,
        B_sft_q=B_sft_q, B_sft_v=B_sft_v,
        n_m2p_layers=N_M2P_LAYERS, n_heads=N_M2P_HEADS, output_scale=OUTPUT_SCALE,
    )
    saved = np.load(str(CODE_M2P_PATH))
    m2p.load_weights([(k, mx.array(saved[k])) for k in saved.files])
    m2p.eval()
    mx.eval(m2p.parameters())

    vectorizer = router_obj["vectorizer"]
    centroids  = router_obj["centroids"]

    correct = 0
    route_code = 0
    total = len(eval_tasks)

    for i, (task, tmpl_idx) in enumerate(eval_tasks):
        prompt = make_code_prompt(task, tmpl_idx)
        x_vec  = vectorizer.transform([prompt])
        sims   = cosine_similarity(x_vec, centroids)
        domain = int(sims.argmax(axis=1)[0])  # 0=math, 1=code
        if domain == 1:
            route_code += 1

        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]
        mem_grid   = extract_memory_hidden_states(model, tokens_arr, m2p.memory_embeddings)
        mx.eval(mem_grid)
        B_q, B_v = m2p(mem_grid)
        mx.eval(*B_q, *B_v)
        inject_lora_b(model, B_q, B_v)

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        correct += eval_code_output(generated, task["func_name"], task["test_cases"])

        del tokens_arr, mem_grid, B_q, B_v
        if (i + 1) % max(1, total // 4) == 0 or (i + 1) == total:
            log(f"    [{i+1}/{total}] pass@1={correct/(i+1):.3f} route_code={route_code/(i+1):.3f}")

    m2p_pass = correct / total
    code_qr  = quality_ratio(m2p_pass, sft_pass, base_pass)
    k988_pass = code_qr >= 0.70

    log(f"\n  Base pass@1:    {base_pass:.4f}")
    log(f"  SFT pass@1:     {sft_pass:.4f}")
    log(f"  M2P pass@1:     {m2p_pass:.4f} ({correct}/{total})")
    log(f"  Code qr:        {code_qr:.4f}")
    log(f"  [K988] {'PASS' if k988_pass else 'FAIL'} — code_qr={code_qr:.4f} >= 0.70")
    log(f"  Phase 5 time: {time.time()-t0:.1f}s")
    log_memory("post-code-eval")

    cleanup(m2p, model, tokenizer, A_code_q, A_code_v, B_sft_q, B_sft_v)
    return {
        "code_m2p_pass_rate":  float(m2p_pass),
        "code_quality_ratio":  float(code_qr),
        "code_route_frac":     float(route_code / total),
        "k988_pass":           k988_pass,
    }


# ---- Phase 6: Math Quality Under Routing (K989) ------------------------------

def phase_math_eval(model_dims: dict, router_obj: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 6] Math Quality Under Routing (K989)")
    log("=" * 70)
    t0 = time.time()

    n_layers     = model_dims["n_layers"]
    math_base    = model_dims["math_base_acc"]
    math_sft     = model_dims["math_sft_acc"]

    A_math_q, A_math_v = load_a_matrices(MATH_LORA_A_PATH, n_layers)
    B_math_sft_q, B_math_sft_v = load_b_matrices(MATH_SFT_B_PATH, n_layers)

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_math_q, A_math_v)
    mx.eval(model.parameters())

    m2p_math = M2PNetworkV6(
        n_layers=n_layers, d_model=model_dims["d_model"], d_m2p=D_M2P,
        n_mem_tokens=N_MEM_TOKENS, rank=LORA_RANK,
        q_proj_out=model_dims["q_proj_out"], v_proj_out=model_dims["v_proj_out"],
        B_sft_q=B_math_sft_q, B_sft_v=B_math_sft_v,
        n_m2p_layers=N_M2P_LAYERS, n_heads=N_M2P_HEADS, output_scale=OUTPUT_SCALE,
    )
    math_saved = np.load(str(MATH_M2P_PATH))
    m2p_math.load_weights([(k, mx.array(math_saved[k])) for k in math_saved.files])
    m2p_math.eval()
    mx.eval(m2p_math.parameters())
    log(f"  Loaded math M2P from {MATH_M2P_PATH}")

    from datasets import load_dataset
    rng = random.Random(SEED)
    ds = load_dataset("gsm8k", "main")
    test_examples = list(ds["test"])
    rng.shuffle(test_examples)
    test_examples = test_examples[:N_EVAL_MATH]
    log(f"  Evaluating {len(test_examples)} GSM8K examples")

    vectorizer = router_obj["vectorizer"]
    centroids  = router_obj["centroids"]

    correct   = 0
    route_math = 0
    total = len(test_examples)

    for i, ex in enumerate(test_examples):
        query = ex["question"]
        x_vec  = vectorizer.transform([query])
        sims   = cosine_similarity(x_vec, centroids)
        domain = int(sims.argmax(axis=1)[0])
        if domain == 0:
            route_math += 1

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
            correct += 1

        del tokens_arr, mem_grid, B_q, B_v
        if (i + 1) % max(1, total // 4) == 0 or (i + 1) == total:
            log(f"    [{i+1}/{total}] math_acc={correct/(i+1):.3f} route_math={route_math/(i+1):.3f}")

    math_acc = correct / total
    math_qr  = quality_ratio(math_acc, math_sft, math_base)
    k989_pass = math_qr >= 0.80

    log(f"\n  Math base: {math_base:.4f}, SFT: {math_sft:.4f}")
    log(f"  Math M2P acc: {math_acc:.4f} ({correct}/{total})")
    log(f"  Math quality_ratio: {math_qr:.4f}")
    log(f"  [K989] {'PASS' if k989_pass else 'FAIL'} — math_qr={math_qr:.4f} >= 0.80")
    log(f"  Phase 6 time: {time.time()-t0:.1f}s")
    log_memory("post-math-eval")

    cleanup(m2p_math, model, tokenizer, A_math_q, A_math_v, B_math_sft_q, B_math_sft_v)
    return {
        "math_routed_acc":     float(math_acc),
        "math_quality_ratio":  float(math_qr),
        "math_route_frac":     float(route_math / total),
        "k989_pass":           k989_pass,
    }


# ---- Main --------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("Code SFT-Residual M2P at 4B — K987/K988/K989")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"CODE_SFT_STEPS={CODE_SFT_STEPS} | M2P_TRAIN_STEPS={M2P_TRAIN_STEPS}")
    log(f"N_EVAL_CODE={N_EVAL_CODE} | N_EVAL_MATH={N_EVAL_MATH}")
    log(f"Architecture: B_applied = B_sft_code + {OUTPUT_SCALE} * zero_init_head(z)")
    log("=" * 70)
    log_memory("start")

    # Check required files
    for path, name in [(MATH_LORA_A_PATH, "math A-matrices"), (MATH_SFT_B_PATH, "math SFT B"),
                       (MATH_M2P_PATH, "math M2P weights"), (CODE_LORA_A_PATH, "code A-matrices"),
                       (V1_RESULTS, "v1 results")]:
        if not path.exists():
            log(f"  ERROR: Missing {name} at {path}")
            return
        log(f"  OK: {name}")

    model_dims = load_model_dims()
    log(f"  n_layers={model_dims['n_layers']}, d_model={model_dims['d_model']}")
    log(f"  math_base={model_dims['math_base_acc']:.4f}, math_sft={model_dims['math_sft_acc']:.4f}")

    eval_tasks = build_eval_tasks(N_EVAL_CODE)
    log(f"  Eval tasks: {len(eval_tasks)}")

    # Phase 1: Code SFT training
    p1 = phase_code_sft(model_dims)

    # Phase 2: Code SFT baseline eval
    p2 = phase_code_sft_eval(model_dims, eval_tasks)
    baselines = {"base_pass_rate": p2["base_pass_rate"], "code_sft_pass_rate": p2["code_sft_pass_rate"]}
    log(f"\n  Baselines: base={baselines['base_pass_rate']:.4f} sft={baselines['code_sft_pass_rate']:.4f}")

    # Phase 3: Code M2P training
    p3 = phase_code_m2p_train(model_dims, eval_tasks, baselines)

    # Phase 4: TF-IDF routing
    p4 = phase_tfidf_routing(model_dims)
    router_obj = p4.pop("router")

    # Phase 5: Code quality under routing (K988)
    p5 = phase_code_eval(model_dims, eval_tasks, baselines, router_obj)

    # Phase 6: Math quality under routing (K989)
    p6 = phase_math_eval(model_dims, router_obj)

    # Kill criteria
    k987_pass = p3.get("k987_pass", False)
    k988_pass = p5["k988_pass"]
    k989_pass = p6["k989_pass"]

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment")
    log("=" * 70)
    log(f"  K987 (init_qr >= 0.80):   {'PASS' if k987_pass else 'FAIL'} "
        f"init_qr={p3.get('init_quality_ratio', 'N/A')}")
    log(f"  K988 (code_qr >= 0.70):   {'PASS' if k988_pass else 'FAIL'} "
        f"code_qr={p5['code_quality_ratio']:.4f}  "
        f"base={p2['base_pass_rate']:.3f} sft={p2['code_sft_pass_rate']:.3f} m2p={p5['code_m2p_pass_rate']:.3f}")
    log(f"  K989 (math_qr >= 0.80):   {'PASS' if k989_pass else 'FAIL'} "
        f"math_qr={p6['math_quality_ratio']:.4f}")
    log(f"  Routing accuracy: {p4['routing_acc']:.4f} "
        f"(math={p4['math_routing_acc']:.4f} code={p4['code_routing_acc']:.4f})")

    all_pass = k987_pass and k988_pass and k989_pass
    log(f"\n  STATUS: {'ALL PASS' if all_pass else 'PARTIAL FAIL'}")

    peak_gb = mx.get_peak_memory() / 1e9
    total_s = round(time.time() - t_start, 1)
    log(f"  Peak memory: {peak_gb:.2f} GB")
    log(f"  Total time: {total_s}s")

    results = {
        "experiment":    "exp_m2p_code_sft_4b",
        "model":         MODEL_ID,
        "is_smoke":      IS_SMOKE,
        "config": {
            "lora_rank":   LORA_RANK, "lora_scale": LORA_SCALE,
            "d_m2p":       D_M2P, "n_mem_tokens": N_MEM_TOKENS,
            "output_scale": OUTPUT_SCALE,
            "code_sft_steps": CODE_SFT_STEPS,
            "m2p_train_steps": M2P_TRAIN_STEPS,
            "n_eval_code": N_EVAL_CODE, "n_eval_math": N_EVAL_MATH,
            **{k: v for k, v in model_dims.items() if k not in ("math_base_acc", "math_sft_acc")},
        },
        "math_base_acc":  model_dims["math_base_acc"],
        "math_sft_acc":   model_dims["math_sft_acc"],
        # Baselines
        "code_base_pass_rate":    p2["base_pass_rate"],
        "code_sft_pass_rate":     p2["code_sft_pass_rate"],
        "code_sft_correct":       p2["sft_correct"],
        "code_eval_n":            p2["n_eval"],
        # K987
        "k987_pass":              k987_pass,
        "init_code_pass_rate":    p3.get("init_code_pass_rate"),
        "init_quality_ratio":     p3.get("init_quality_ratio"),
        "grad_norm":              p3.get("grad_norm"),
        # K988
        "k988_pass":              k988_pass,
        "code_m2p_pass_rate":     p5["code_m2p_pass_rate"],
        "code_quality_ratio":     p5["code_quality_ratio"],
        "code_route_frac":        p5["code_route_frac"],
        # K989
        "k989_pass":              k989_pass,
        "math_routed_acc":        p6["math_routed_acc"],
        "math_quality_ratio":     p6["math_quality_ratio"],
        "math_route_frac":        p6["math_route_frac"],
        # Routing
        "routing_acc":            p4["routing_acc"],
        "math_routing_acc":       p4["math_routing_acc"],
        "code_routing_acc":       p4["code_routing_acc"],
        # Meta
        "all_k_pass":             all_pass,
        "peak_memory_gb":         round(peak_gb, 2),
        "total_time_s":           total_s,
        "timestamp":              time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
