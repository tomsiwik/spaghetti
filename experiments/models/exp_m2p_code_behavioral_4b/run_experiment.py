#!/usr/bin/env python3
"""Code M2P Behavioral Quality at 4B — Format Overfitting Check Under Composition.

TYPE: guided-exploration
MATH: experiments/models/exp_m2p_code_behavioral_4b/MATH.md

WHAT THIS TESTS:
  Loads pre-trained code M2P (from m2p_2domain_compose_qwen4b/code_m2p_weights.npz)
  and evaluates behavioral quality on Python function generation tasks.
  Finding #404 trained code M2P but only measured math quality (K977=1.3125).
  This experiment measures the missing code quality dimension.

  Also re-measures math quality under the same routing setup to confirm K985.

KILL CRITERIA:
  K984: code quality_ratio >= 0.50 under routed composition
        (code_pass@1_M2P / base_pass@1; K984 FAIL = format overfitting confirmed)
  K985: math quality_ratio >= 0.80 under routed composition
        (Finding #404: qr=1.3125; routing must preserve this)
  K986: TF-IDF routing >= 80% on math vs code inputs
        (Finding #404: 100%; should be identical)

REFERENCES:
  LoraRetriever (arXiv:2402.09997), Finding #395, Finding #403, Finding #404
  Aghajanyan et al. (arXiv:2012.13255) — intrinsic dimensionality
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

LORA_RANK    = 4
LORA_SCALE   = 5.0
D_M2P        = 1024
N_MEM_TOKENS = 16
N_M2P_LAYERS = 4
N_M2P_HEADS  = 4
OUTPUT_SCALE = 0.032

MAX_SEQ_LEN    = 64  if IS_SMOKE else 256
MAX_GEN_TOKENS = 128 if IS_SMOKE else 384  # Qwen3 thinks before answering; 128 min for smoke
SEED = 42

N_EVAL_MATH   = 5  if IS_SMOKE else 200   # GSM8K examples for math K985
N_EVAL_CODE   = 5  if IS_SMOKE else 50    # code tasks for K984
N_ROUTE_TRAIN = 4  if IS_SMOKE else 100   # per class for routing
N_ROUTE_TEST  = 4  if IS_SMOKE else 100   # per class for K986

EXPERIMENT_DIR    = Path(__file__).parent
V1_DIR            = EXPERIMENT_DIR.parent / "m2p_qwen4b_gsm8k"
MATH_LORA_A_PATH  = V1_DIR / "grassmannian_a_matrices.npz"
MATH_SFT_B_PATH   = V1_DIR / "sft_b_matrices.npz"
V1_RESULTS        = V1_DIR / "results.json"
MATH_M2P_PATH     = EXPERIMENT_DIR.parent / "m2p_qwen4b_sft_residual" / "m2p_weights.npz"

# Code M2P and A-matrices from Finding #404
CODE_SRC_DIR     = EXPERIMENT_DIR.parent / "m2p_2domain_compose_qwen4b"
CODE_LORA_A_PATH = CODE_SRC_DIR / "code_a_matrices.npz"
CODE_M2P_PATH    = CODE_SRC_DIR / "code_m2p_weights.npz"

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

# ---- Code task definitions (same as m2p_2domain_compose_qwen4b) -----------

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
    {"func_name": "is_positive",  "desc": "takes an integer `n` and returns True if it is positive, False otherwise",
     "test_cases": [((5,), True), ((0,), False), ((-3,), False)]},
    {"func_name": "clamp_zero",   "desc": "takes an integer `n` and returns `n` if positive, else 0",
     "test_cases": [((5,), 5), ((0,), 0), ((-3,), 0)]},
    {"func_name": "sum_first_n",  "desc": "takes a non-negative integer `n` and returns the sum 1+2+...+n",
     "test_cases": [((0,), 0), ((1,), 1), ((4,), 10), ((5,), 15)]},
]

CODE_PROMPT_TEMPLATES = [
    "Write a Python function called `{func_name}` that {desc}. Output only the Python code.",
    "Implement a Python function named `{func_name}` that {desc}. Only output the function code.",
    "Create a Python function `{func_name}` that {desc}. Return only the code.",
]


def make_code_prompt(task: dict, tmpl_idx: int = 0) -> str:
    tmpl = CODE_PROMPT_TEMPLATES[tmpl_idx % len(CODE_PROMPT_TEMPLATES)]
    return tmpl.format(func_name=task["func_name"], desc=task["desc"])


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


# ---- Model loading helpers -------------------------------------------------

def load_model_dims() -> dict:
    with open(V1_RESULTS) as f:
        v1 = json.load(f)
    cfg = v1["config"]
    return {
        "n_layers":    cfg["n_layers"],
        "d_model":     cfg["d_model"],
        "q_proj_out":  cfg["q_proj_out"],
        "v_proj_out":  cfg["v_proj_out"],
        "base_accuracy": v1["base_accuracy"],
        "sft_accuracy":  v1["sft_accuracy"],
    }


def load_a_matrices_from_file(path: Path, n_layers: int) -> tuple:
    saved = np.load(str(path))
    A_q = [mx.array(saved[f"layer_{li}_q_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    A_v = [mx.array(saved[f"layer_{li}_v_proj_A"]).astype(mx.bfloat16) for li in range(n_layers)]
    mx.eval(*A_q, *A_v)
    return A_q, A_v


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


# ---- SHINE memory hidden states --------------------------------------------

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


# ---- M2PNetworkV6: same architecture as m2p_2domain_compose_qwen4b ---------

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


# ---- Phase 1: TF-IDF Routing (K986) ----------------------------------------

def phase_tfidf_routing() -> dict:
    log("\n" + "=" * 70)
    log("[Phase 1] TF-IDF Routing (K986)")
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

    all_code_texts = []
    for tmpl_idx in range(len(CODE_PROMPT_TEMPLATES)):
        for task in CODE_TASKS:
            all_code_texts.append(make_code_prompt(task, tmpl_idx))

    code_texts = []
    i = 0
    while len(code_texts) < n_train + n_test:
        code_texts.append(all_code_texts[i % len(all_code_texts)])
        i += 1

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
    math_routing_acc = float((math_preds == 0).mean())
    code_routing_acc = float((code_preds == 1).mean())

    log(f"  Overall routing accuracy: {routing_acc:.4f} ({correct}/{total})")
    log(f"  Math routing accuracy: {math_routing_acc:.4f}")
    log(f"  Code routing accuracy: {code_routing_acc:.4f}")

    k986_pass = routing_acc >= 0.80
    log(f"  [K986] {'PASS' if k986_pass else 'FAIL'} — routing_acc={routing_acc:.4f} >= 0.80")
    log(f"  Phase 1 time: {time.time()-t0:.1f}s")

    return {
        "routing_acc":       float(routing_acc),
        "math_routing_acc":  math_routing_acc,
        "code_routing_acc":  code_routing_acc,
        "k986_pass":         k986_pass,
        "router": {"vectorizer": vectorizer, "centroids": centroids},
    }


# ---- Phase 2: Math quality under routing (K985) ----------------------------

def phase_math_eval(model_dims: dict, router_obj: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 2] Math Quality Under Routing (K985)")
    log("=" * 70)
    t0 = time.time()

    n_layers   = model_dims["n_layers"]
    base_acc   = model_dims["base_accuracy"]
    sft_acc    = model_dims["sft_accuracy"]

    A_math_q, A_math_v = load_a_matrices_from_file(MATH_LORA_A_PATH, n_layers)
    B_sft_q, B_sft_v   = load_sft_b_matrices(n_layers)

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_math_q, A_math_v)
    mx.eval(model.parameters())

    m2p_math = make_m2p_v6(model_dims, B_sft_q, B_sft_v)
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

    correct = 0
    route_math = 0
    total = len(test_examples)

    for i, ex in enumerate(test_examples):
        query_text = ex["question"]
        x_vec  = vectorizer.transform([query_text])
        sims   = cosine_similarity(x_vec, centroids)
        domain = int(sims.argmax(axis=1)[0])
        if domain == 0:
            route_math += 1

        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids  = tokenizer.encode(prompt)
        tokens_arr  = mx.array(prompt_ids)[None, :]

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
            log(f"    [{i+1}/{total}] math_acc={correct/(i+1):.3f} route_math_frac={route_math/(i+1):.3f}")

    routed_acc  = correct / total
    sft_impr    = sft_acc - base_acc
    m2p_impr    = routed_acc - base_acc
    qr          = (m2p_impr / sft_impr) if abs(sft_impr) > 1e-9 else 0.0

    log(f"\n  Math routed accuracy: {routed_acc:.4f} ({correct}/{total})")
    log(f"  Math quality_ratio:   {qr:.4f}")
    log(f"  base={base_acc:.4f} sft={sft_acc:.4f}")
    k985_pass = qr >= 0.80
    log(f"  [K985] {'PASS' if k985_pass else 'FAIL'} — qr={qr:.4f} >= 0.80")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-math-eval")

    cleanup(m2p_math, model, tokenizer, B_sft_q, B_sft_v, A_math_q, A_math_v)

    return {
        "math_routed_acc":  float(routed_acc),
        "math_correct":     correct,
        "math_n":           total,
        "math_quality_ratio": float(qr),
        "math_route_frac":  float(route_math / total),
        "k985_pass":        k985_pass,
    }


# ---- Phase 3: Code behavioral eval (K984) ----------------------------------

def phase_code_eval(model_dims: dict, router_obj: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 3] Code Behavioral Quality Under Routing (K984)")
    log("=" * 70)
    t0 = time.time()

    if not CODE_M2P_PATH.exists():
        raise FileNotFoundError(f"Code M2P not found at {CODE_M2P_PATH}. "
                                f"Run m2p_2domain_compose_qwen4b first.")
    if not CODE_LORA_A_PATH.exists():
        raise FileNotFoundError(f"Code A-matrices not found at {CODE_LORA_A_PATH}.")

    n_layers   = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    A_code_q, A_code_v = load_a_matrices_from_file(CODE_LORA_A_PATH, n_layers)
    B_sft_zero_q, B_sft_zero_v = zero_b_matrices(n_layers, q_proj_out, v_proj_out)

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_code_q, A_code_v)
    mx.eval(model.parameters())

    m2p_code = make_m2p_v6(model_dims, B_sft_zero_q, B_sft_zero_v)
    code_saved = np.load(str(CODE_M2P_PATH))
    m2p_code.load_weights([(k, mx.array(code_saved[k])) for k in code_saved.files])
    m2p_code.eval()
    mx.eval(m2p_code.parameters())
    log(f"  Loaded code M2P from {CODE_M2P_PATH}")

    vectorizer = router_obj["vectorizer"]
    centroids  = router_obj["centroids"]

    # Build eval tasks (up to N_EVAL_CODE)
    eval_tasks = []
    rng_tasks = random.Random(SEED + 999)
    tmpl_indices = list(range(len(CODE_PROMPT_TEMPLATES)))
    for tmpl_idx in tmpl_indices:
        for task in CODE_TASKS:
            eval_tasks.append((task, tmpl_idx))
    rng_tasks.shuffle(eval_tasks)
    eval_tasks = eval_tasks[:N_EVAL_CODE]
    log(f"  Evaluating {len(eval_tasks)} code tasks")

    # Phase 3a: Base model pass@1 (no adapter)
    log("\n  [Phase 3a] Base model pass@1 (no adapter)")
    model.freeze()  # already frozen; remove lora_b injection
    base_correct = 0
    for i, (task, tmpl_idx) in enumerate(eval_tasks):
        prompt = make_code_prompt(task, tmpl_idx)
        # Use zero B (no LoRA signal)
        B_q_zero, B_v_zero = zero_b_matrices(n_layers, q_proj_out, v_proj_out)
        inject_lora_b(model, B_q_zero, B_v_zero)

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        passed = eval_code_output(generated, task["func_name"], task["test_cases"])
        base_correct += passed
        if i == 0 and IS_SMOKE:  # diagnostic in smoke mode only
            clean_gen = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
            log(f"    [smoke diag] base gen (first 120 no-think): {clean_gen[:120]!r}")

        del B_q_zero, B_v_zero
        if (i + 1) % max(1, len(eval_tasks) // 3) == 0 or (i + 1) == len(eval_tasks):
            log(f"    base [{i+1}/{len(eval_tasks)}] pass@1={base_correct/(i+1):.3f}")

    base_pass_rate = base_correct / len(eval_tasks)
    log(f"  Base pass@1: {base_pass_rate:.4f} ({base_correct}/{len(eval_tasks)})")

    # Phase 3b: Code M2P pass@1 under routing
    log("\n  [Phase 3b] Code M2P pass@1 under routing")
    m2p_correct   = 0
    route_code    = 0
    format_overfit_count = 0   # count math prompts routed to code M2P

    # Also check format overfitting: apply code M2P to 10 math prompts
    from datasets import load_dataset
    ds_gsm = load_dataset("gsm8k", "main")
    math_prompts = [ex["question"] for ex in list(ds_gsm["test"])[:10]]

    for i, (task, tmpl_idx) in enumerate(eval_tasks):
        prompt    = make_code_prompt(task, tmpl_idx)
        x_vec     = vectorizer.transform([prompt])
        sims      = cosine_similarity(x_vec, centroids)
        domain    = int(sims.argmax(axis=1)[0])  # 0=math, 1=code
        if domain == 1:
            route_code += 1

        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        mem_grid = extract_memory_hidden_states(model, tokens_arr, m2p_code.memory_embeddings)
        mx.eval(mem_grid)
        B_q, B_v = m2p_code(mem_grid)
        mx.eval(*B_q, *B_v)
        inject_lora_b(model, B_q, B_v)

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        passed = eval_code_output(generated, task["func_name"], task["test_cases"])
        m2p_correct += passed

        del tokens_arr, mem_grid, B_q, B_v

        if (i + 1) % max(1, len(eval_tasks) // 3) == 0 or (i + 1) == len(eval_tasks):
            log(f"    m2p [{i+1}/{len(eval_tasks)}] pass@1={m2p_correct/(i+1):.3f} route_code={route_code/(i+1):.3f}")

    # Format overfitting check: apply code M2P to math prompts
    log("\n  [Phase 3c] Format overfitting check (code M2P on math prompts)")
    for mp in math_prompts[:5 if IS_SMOKE else 10]:
        prompt_ids = tokenizer.encode(mp)
        tokens_arr = mx.array(prompt_ids)[None, :]
        mem_grid = extract_memory_hidden_states(model, tokens_arr, m2p_code.memory_embeddings)
        mx.eval(mem_grid)
        B_q, B_v = m2p_code(mem_grid)
        mx.eval(*B_q, *B_v)
        inject_lora_b(model, B_q, B_v)
        generated = mlx_generate(model, tokenizer, prompt=mp, max_tokens=64, verbose=False)
        # Format overfitting = output starts with "def " or contains "```python"
        clean = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
        is_python = bool(re.search(r"^\s*(def |```python)", clean))
        if is_python:
            format_overfit_count += 1
        log(f"    math_prompt: first 60 chars of output: {clean[:60]!r} | python_fmt={is_python}")
        del tokens_arr, mem_grid, B_q, B_v

    # Compute quality_ratio
    m2p_pass_rate = m2p_correct / len(eval_tasks)
    if base_pass_rate > 0.0:
        code_qr = m2p_pass_rate / base_pass_rate
    else:
        code_qr = float(m2p_pass_rate > 0.0)  # any pass = improvement over 0% base

    log(f"\n  Base pass@1:  {base_pass_rate:.4f} ({base_correct}/{len(eval_tasks)})")
    log(f"  M2P pass@1:   {m2p_pass_rate:.4f} ({m2p_correct}/{len(eval_tasks)})")
    log(f"  Code quality_ratio: {code_qr:.4f}")
    log(f"  Format overfitting (math prompts→python fmt): {format_overfit_count}/10")
    log(f"  Code routing fraction: {route_code/len(eval_tasks):.4f}")

    k984_pass = code_qr >= 0.50
    log(f"  [K984] {'PASS' if k984_pass else 'FAIL'} — code_qr={code_qr:.4f} >= 0.50")
    if not k984_pass:
        log(f"  [K984 FAIL DIAGNOSIS] base={base_pass_rate:.3f} m2p={m2p_pass_rate:.3f} "
            f"fmt_overfit={format_overfit_count}/10")
    log(f"  Phase 3 time: {time.time()-t0:.1f}s")
    log_memory("post-code-eval")

    cleanup(m2p_code, model, tokenizer, B_sft_zero_q, B_sft_zero_v, A_code_q, A_code_v)

    return {
        "code_base_pass_rate":    float(base_pass_rate),
        "code_m2p_pass_rate":     float(m2p_pass_rate),
        "code_quality_ratio":     float(code_qr),
        "code_route_frac":        float(route_code / len(eval_tasks)),
        "format_overfit_count":   format_overfit_count,
        "k984_pass":              k984_pass,
    }


# ---- Main ------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("Code M2P Behavioral Quality at 4B — Format Overfitting Check")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"MODEL={MODEL_ID}")
    log(f"N_EVAL_MATH={N_EVAL_MATH} | N_EVAL_CODE={N_EVAL_CODE}")
    log("=" * 70)
    log_memory("start")

    model_dims = load_model_dims()
    log(f"  n_layers={model_dims['n_layers']}, d_model={model_dims['d_model']}")
    log(f"  base_acc={model_dims['base_accuracy']:.4f}, sft_acc={model_dims['sft_accuracy']:.4f}")

    # Verify required weight files exist
    for path, name in [(MATH_M2P_PATH, "math M2P"), (CODE_M2P_PATH, "code M2P"),
                       (MATH_LORA_A_PATH, "math A-matrices"), (CODE_LORA_A_PATH, "code A-matrices"),
                       (MATH_SFT_B_PATH, "math SFT B-matrices")]:
        if not path.exists():
            log(f"  ERROR: Missing {name} at {path}")
            return
        log(f"  OK: {name} at {path.name}")

    # Phase 1: TF-IDF routing (K986)
    p1 = phase_tfidf_routing()
    router_obj = p1.pop("router")

    # Phase 2: Math quality under routing (K985)
    p2 = phase_math_eval(model_dims, router_obj)

    # Phase 3: Code behavioral quality (K984)
    p3 = phase_code_eval(model_dims, router_obj)

    # ---- Final assessment ----
    k984_pass = p3["k984_pass"]
    k985_pass = p2["k985_pass"]
    k986_pass = p1["k986_pass"]

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment")
    log("=" * 70)
    log(f"  K984 (code_qr >= 0.50):  {'PASS' if k984_pass else 'FAIL'} "
        f"code_qr={p3['code_quality_ratio']:.4f} "
        f"base={p3['code_base_pass_rate']:.3f} m2p={p3['code_m2p_pass_rate']:.3f}")
    log(f"  K985 (math_qr >= 0.80):  {'PASS' if k985_pass else 'FAIL'} "
        f"qr={p2['math_quality_ratio']:.4f}")
    log(f"  K986 (routing >= 80%):   {'PASS' if k986_pass else 'FAIL'} "
        f"routing_acc={p1['routing_acc']:.4f}")
    log(f"  Format overfitting: {p3['format_overfit_count']}/10 math prompts → python format")

    all_pass = k984_pass and k985_pass and k986_pass
    log(f"\n  STATUS: {'ALL PASS' if all_pass else 'PARTIAL FAIL'}")

    peak_gb = mx.get_peak_memory() / 1e9
    total_s = round(time.time() - t_start, 1)
    log(f"  Peak memory: {peak_gb:.2f} GB")
    log(f"  Total time: {total_s}s")

    results = {
        "experiment":    "exp_m2p_code_behavioral_4b",
        "model":         MODEL_ID,
        "is_smoke":      IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P, "n_mem_tokens": N_MEM_TOKENS,
            "n_eval_math": N_EVAL_MATH, "n_eval_code": N_EVAL_CODE,
            **{k: v for k, v in model_dims.items() if k not in ("base_accuracy", "sft_accuracy")},
        },
        "base_accuracy": model_dims["base_accuracy"],
        "sft_accuracy":  model_dims["sft_accuracy"],
        # K984
        "k984_pass":              k984_pass,
        "code_quality_ratio":     p3["code_quality_ratio"],
        "code_base_pass_rate":    p3["code_base_pass_rate"],
        "code_m2p_pass_rate":     p3["code_m2p_pass_rate"],
        "format_overfit_count":   p3["format_overfit_count"],
        "code_route_frac":        p3["code_route_frac"],
        # K985
        "k985_pass":              k985_pass,
        "math_quality_ratio":     p2["math_quality_ratio"],
        "math_routed_acc":        p2["math_routed_acc"],
        "math_route_frac":        p2["math_route_frac"],
        # K986
        "k986_pass":              k986_pass,
        "routing_acc":            p1["routing_acc"],
        "math_routing_acc":       p1["math_routing_acc"],
        "code_routing_acc":       p1["code_routing_acc"],
        # Meta
        "all_k_pass":             all_pass,
        "peak_memory_gb":         round(peak_gb, 2),
        "total_time_s":           total_s,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
