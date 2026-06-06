#!/usr/bin/env python3
"""2-Domain M2P Composition on Qwen3-0.6B: Math + Code with TF-IDF routing.

TYPE: verification (Type 1)
MATH: experiments/models/m2p_2domain_compose_qwen06b/MATH.md

EXPERIMENT: exp_m2p_2domain_compose_qwen06b

WHAT THIS TESTS:
  Two independently-trained M2P adapters (math + Python code) composed on Qwen3-0.6B
  using TF-IDF routed selection (Theorem 2: routing invariant to adapter state).

  Math adapter: warm-start from v4 m2p_weights.npz, 300 additional steps.
  Code adapter: fresh M2P trained on 200 simple Python function prompts, 500 steps.

  Routed eval: TF-IDF selects one adapter, B-matrices computed per-example from that
  example's hidden states, injected into lora_b, then mlx_generate runs normally.

KILL CRITERIA:
  K954: Routed composition >= 80% of best-single on EACH domain
        quality_ratio = (composed_acc - base_acc) / (single_acc - base_acc)
  K955: TF-IDF routing accuracy >= 85% on math vs code inputs

References:
  Hu et al. (arXiv:2106.09685) — LoRA
  Ha et al. (arXiv:1609.09106) — HyperNetworks
  LoraRetriever (arXiv:2402.09997) — text-based routing invariant to model distribution
  Finding #354 — TF-IDF routing 95%; Finding #389 — 100% real NLP
  Finding #378 — M2P v4 quality_ratio=1.433; Finding #381 — composition grad_norm > 0

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

LORA_RANK = 4
LORA_SCALE = 5.0
D_M2P = 1024
OUTPUT_SCALE = 0.032

MATH_TRAIN_STEPS = 10 if IS_SMOKE else 300
CODE_TRAIN_STEPS = 20 if IS_SMOKE else 500
LR = 5e-5
LR_WARMUP = 3 if IS_SMOKE else 30
MAX_SEQ_LEN = 64 if IS_SMOKE else 256
MAX_GEN_TOKENS = 32 if IS_SMOKE else 128
SEED = 42

N_EVAL_MATH = 5 if IS_SMOKE else 100
N_EVAL_CODE = 5 if IS_SMOKE else 20     # 20 hardcoded code tasks for eval

N_ROUTE_TRAIN = 10 if IS_SMOKE else 200
N_ROUTE_TEST  = 5  if IS_SMOKE else 100

EXPERIMENT_DIR = Path(__file__).parent
V2_DIR  = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V4_DIR  = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v4"
V2_RESULTS  = V2_DIR / "results.json"
V4_M2P_PATH = V4_DIR / "m2p_weights.npz"

MATH_M2P_PATH     = EXPERIMENT_DIR / "math_m2p_weights.npz"
CODE_M2P_PATH     = EXPERIMENT_DIR / "code_m2p_weights.npz"
GRASSMANNIAN_PATH = EXPERIMENT_DIR / "grassmannian_a_matrices.npz"
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

# ---- Code tasks ------------------------------------------------------------
# Hardcoded simple Python function tasks with test cases. Safe to exec.

CODE_TASKS = [
    {"func_name": "add",           "desc": "takes two integers `a` and `b` and returns their sum",
     "test_cases": [((1, 2), 3), ((0, 0), 0), ((-1, 5), 4), ((10, 20), 30)]},
    {"func_name": "subtract",      "desc": "takes two integers `a` and `b` and returns `a` minus `b`",
     "test_cases": [((5, 3), 2), ((0, 0), 0), ((10, 7), 3), ((-1, -3), 2)]},
    {"func_name": "multiply",      "desc": "takes two integers `a` and `b` and returns their product",
     "test_cases": [((2, 3), 6), ((0, 5), 0), ((-2, 4), -8), ((7, 7), 49)]},
    {"func_name": "square",        "desc": "takes an integer `n` and returns its square",
     "test_cases": [((4,), 16), ((0,), 0), ((-3,), 9), ((10,), 100)]},
    {"func_name": "double",        "desc": "takes an integer `n` and returns twice its value",
     "test_cases": [((3,), 6), ((0,), 0), ((-5,), -10), ((100,), 200)]},
    {"func_name": "increment",     "desc": "takes an integer `n` and returns `n` plus one",
     "test_cases": [((0,), 1), ((9,), 10), ((-1,), 0), ((99,), 100)]},
    {"func_name": "decrement",     "desc": "takes an integer `n` and returns `n` minus one",
     "test_cases": [((1,), 0), ((10,), 9), ((0,), -1), ((100,), 99)]},
    {"func_name": "negate",        "desc": "takes an integer `n` and returns its negation",
     "test_cases": [((5,), -5), ((0,), 0), ((-3,), 3), ((100,), -100)]},
    {"func_name": "max_of_two",    "desc": "takes two integers `a` and `b` and returns the larger one",
     "test_cases": [((3, 5), 5), ((7, 2), 7), ((4, 4), 4), ((-1, 0), 0)]},
    {"func_name": "min_of_two",    "desc": "takes two integers `a` and `b` and returns the smaller one",
     "test_cases": [((3, 5), 3), ((7, 2), 2), ((4, 4), 4), ((-1, 0), -1)]},
    {"func_name": "absolute_value","desc": "takes an integer `n` and returns its absolute value",
     "test_cases": [((-5,), 5), ((3,), 3), ((0,), 0), ((-99,), 99)]},
    {"func_name": "is_even",       "desc": "takes an integer `n` and returns True if it is even, False otherwise",
     "test_cases": [((2,), True), ((3,), False), ((0,), True), ((7,), False)]},
    {"func_name": "is_positive",   "desc": "takes an integer `n` and returns True if it is positive, False otherwise",
     "test_cases": [((1,), True), ((0,), False), ((-3,), False), ((100,), True)]},
    {"func_name": "sum_three",     "desc": "takes three integers `a`, `b`, `c` and returns their sum",
     "test_cases": [((1, 2, 3), 6), ((0, 0, 0), 0), ((-1, 5, 2), 6), ((10, 20, 30), 60)]},
    {"func_name": "power_of_two",  "desc": "takes a non-negative integer `n` and returns 2 raised to the power of `n`",
     "test_cases": [((0,), 1), ((1,), 2), ((3,), 8), ((8,), 256)]},
    {"func_name": "ten_times",     "desc": "takes an integer `n` and returns `n` multiplied by 10",
     "test_cases": [((5,), 50), ((0,), 0), ((-3,), -30), ((12,), 120)]},
    {"func_name": "average_two",   "desc": "takes two numbers `a` and `b` and returns their average as a float",
     "test_cases": [((2, 4), 3.0), ((0, 0), 0.0), ((1, 3), 2.0), ((5, 5), 5.0)]},
    {"func_name": "clamp_to_100",  "desc": "takes an integer `n` and returns `n` if it is at most 100, else 100",
     "test_cases": [((50,), 50), ((100,), 100), ((150,), 100), ((-5,), -5)]},
    {"func_name": "is_zero",       "desc": "takes an integer `n` and returns True if `n` equals zero, False otherwise",
     "test_cases": [((0,), True), ((1,), False), ((-1,), False), ((100,), False)]},
    {"func_name": "triple",        "desc": "takes an integer `n` and returns `n` multiplied by three",
     "test_cases": [((3,), 9), ((0,), 0), ((-2,), -6), ((10,), 30)]},
]

PROMPT_TEMPLATES = [
    "Write a Python function called `{func_name}` that {desc}. Output only the Python code.",
    "Implement a Python function named `{func_name}` that {desc}. Only output the function code.",
    "Create a Python function `{func_name}` that {desc}. Return only the code.",
    "Define a Python function called `{func_name}` that {desc}. Output just the function.",
]

_IMPLS = {
    "add": "a + b", "subtract": "a - b", "multiply": "a * b",
    "square": "n * n", "double": "n * 2", "increment": "n + 1",
    "decrement": "n - 1", "negate": "-n",
    "max_of_two": "a if a > b else b", "min_of_two": "a if a < b else b",
    "absolute_value": "n if n >= 0 else -n",
    "is_even": "n % 2 == 0", "is_positive": "n > 0",
    "sum_three": "a + b + c", "power_of_two": "2 ** n",
    "ten_times": "n * 10", "average_two": "(a + b) / 2",
    "clamp_to_100": "n if n <= 100 else 100",
    "is_zero": "n == 0", "triple": "n * 3",
}


def make_code_prompt(task: dict, tmpl_idx: int = 0) -> str:
    tmpl = PROMPT_TEMPLATES[tmpl_idx % len(PROMPT_TEMPLATES)]
    return tmpl.format(func_name=task["func_name"], desc=task["desc"])


def generate_code_training_data(n: int, seed: int = SEED + 200) -> list:
    examples = []
    for i in range(n):
        task = CODE_TASKS[i % len(CODE_TASKS)]
        tmpl_idx = (i // len(CODE_TASKS)) % len(PROMPT_TEMPLATES)
        prompt = make_code_prompt(task, tmpl_idx=tmpl_idx)
        func_name = task["func_name"]
        args_str = ", ".join(chr(ord("a") + j) for j in range(len(task["test_cases"][0][0])))
        impl = _IMPLS.get(func_name, "None")
        target = f"def {func_name}({args_str}):\n    return {impl}"
        examples.append({"prompt": prompt, "answer": target})
    return examples


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
            result = fn(*args)
            if result != expected:
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


def tokenize_seq(tokenizer, prompt: str, answer: str) -> list:
    text = f"{prompt}\nAnswer: {answer}"
    ids = tokenizer.encode(text)
    if len(ids) >= 2:
        ids = ids[:MAX_SEQ_LEN + 1]
        if len(ids) >= 2:
            return ids
    return []


# ---- Grassmannian A-matrix pairs ------------------------------------------

def make_grassmannian_pair(d: int, r: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, 2 * r)).astype(np.float32)
    Q, _ = np.linalg.qr(X)
    return Q[:, :r], Q[:, r:2*r]


def build_or_load_grassmannian_pairs(n_layers: int, d_in: int, r: int) -> tuple:
    if GRASSMANNIAN_PATH.exists():
        log(f"  [Grassmannian] Loading from {GRASSMANNIAN_PATH}")
        saved = np.load(str(GRASSMANNIAN_PATH))
        A_math_q = [mx.array(saved[f"math_q_{li}"]).astype(mx.bfloat16) for li in range(n_layers)]
        A_code_q = [mx.array(saved[f"code_q_{li}"]).astype(mx.bfloat16) for li in range(n_layers)]
        A_math_v = [mx.array(saved[f"math_v_{li}"]).astype(mx.bfloat16) for li in range(n_layers)]
        A_code_v = [mx.array(saved[f"code_v_{li}"]).astype(mx.bfloat16) for li in range(n_layers)]
    else:
        log(f"  [Grassmannian] Generating {n_layers}-layer pairs (d={d_in}, r={r})")
        save_dict = {}
        A_math_q, A_code_q, A_math_v, A_code_v = [], [], [], []
        for li in range(n_layers):
            am_q, ac_q = make_grassmannian_pair(d_in, r, seed=SEED + li)
            am_v, ac_v = make_grassmannian_pair(d_in, r, seed=SEED + 1000 + li)
            A_math_q.append(mx.array(am_q).astype(mx.bfloat16))
            A_code_q.append(mx.array(ac_q).astype(mx.bfloat16))
            A_math_v.append(mx.array(am_v).astype(mx.bfloat16))
            A_code_v.append(mx.array(ac_v).astype(mx.bfloat16))
            save_dict.update({f"math_q_{li}": am_q, f"code_q_{li}": ac_q,
                               f"math_v_{li}": am_v, f"code_v_{li}": ac_v})
        np.savez(str(GRASSMANNIAN_PATH), **save_dict)
        cross = save_dict["math_q_0"].T @ save_dict["code_q_0"]
        log(f"  [Grassmannian] max|A_math^T A_code| layer0 = {np.abs(cross).max():.2e}")

    mx.eval(*A_math_q, *A_code_q, *A_math_v, *A_code_v)
    return A_math_q, A_code_q, A_math_v, A_code_v


# ---- LoRA structure --------------------------------------------------------

def apply_lora_structure(model, A_q_layers: list, A_v_layers: list) -> None:
    """Wrap q_proj/v_proj with LoRALinear and set A-matrices."""
    n_layers = len(A_q_layers)
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.q_proj.lora_a = A_q_layers[li]
        attn.v_proj.lora_a = A_v_layers[li]
    model.freeze()


def inject_lora_b(model, B_q_layers: list, B_v_layers: list) -> None:
    """Set lora_b on all layers (for mlx_generate compatibility)."""
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_b = B_q_layers[li]
        layer.self_attn.v_proj.lora_b = B_v_layers[li]
    mx.eval(model.parameters())


def inject_lora_a(model, A_q_layers: list, A_v_layers: list) -> None:
    """Update lora_a on all layers (when switching domains in routed eval)."""
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_a = A_q_layers[li]
        layer.self_attn.v_proj.lora_a = A_v_layers[li]
    mx.eval(model.parameters())


# ---- Functional forward (for training only) --------------------------------

def functional_lora_proj(x, linear_module, A, B, scale):
    y = linear_module(x)
    z = (x @ A.astype(x.dtype)) @ B.astype(x.dtype)
    return y + (scale * z).astype(x.dtype)


def functional_attn_forward(attn, x, B_q, B_v, A_q, A_v, lora_scale, mask, cache=None):
    B_batch, L, D = x.shape
    q = functional_lora_proj(x, attn.q_proj.linear, A_q, B_q, lora_scale)
    k = attn.k_proj(x)
    v = functional_lora_proj(x, attn.v_proj.linear, A_v, B_v, lora_scale)
    queries = attn.q_norm(q.reshape(B_batch, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    keys    = attn.k_norm(k.reshape(B_batch, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values  = v.reshape(B_batch, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
        keys    = attn.rope(keys,    offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = attn.rope(queries)
        keys    = attn.rope(keys)
    out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=attn.scale, mask=mask)
    return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B_batch, L, -1))


def forward_with_loras(model, tokens_arr, B_q_layers, B_v_layers, A_q_layers, A_v_layers):
    qwen3 = model.model
    h = qwen3.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)
    for li, layer in enumerate(qwen3.layers):
        normed   = layer.input_layernorm(h)
        attn_out = functional_attn_forward(
            attn=layer.self_attn, x=normed,
            B_q=B_q_layers[li], B_v=B_v_layers[li],
            A_q=A_q_layers[li], A_v=A_v_layers[li],
            lora_scale=LORA_SCALE, mask=mask,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = qwen3.norm(h)
    return qwen3.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)


def extract_hidden_states(model, tokens_arr, A_q_layers, A_v_layers, q_proj_out, v_proj_out):
    """Extract per-layer mean-pooled hidden states with zero LoRA (base model forward)."""
    n_layers = len(A_q_layers)
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    qwen3 = model.model
    h = qwen3.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)
    layer_states = []
    for li, layer in enumerate(qwen3.layers):
        normed   = layer.input_layernorm(h)
        attn_out = functional_attn_forward(
            attn=layer.self_attn, x=normed,
            B_q=B_q_zero[li], B_v=B_v_zero[li],
            A_q=A_q_layers[li], A_v=A_v_layers[li],
            lora_scale=0.0, mask=mask,
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layer_states.append(mx.mean(h[0], axis=0))
    return mx.stop_gradient(mx.stack(layer_states, axis=0))


# ---- M2P Network -----------------------------------------------------------

class M2PNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_m2p, rank, q_proj_out, v_proj_out, output_scale=0.032):
        super().__init__()
        self.n_layers     = n_layers
        self.rank         = rank
        self.output_scale = output_scale
        self.enc_linear1  = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2  = nn.Linear(2 * d_m2p, d_m2p)
        self.b_heads_q    = [nn.Linear(d_m2p, rank * q_proj_out) for _ in range(n_layers)]
        self.b_heads_v    = [nn.Linear(d_m2p, rank * v_proj_out) for _ in range(n_layers)]

    def __call__(self, layer_hs: mx.array):
        h = mx.mean(layer_hs, axis=0)
        h = nn.gelu(self.enc_linear1(h))
        z = self.enc_linear2(h)
        B_q = [self.b_heads_q[li](z).reshape(self.rank, -1) * self.output_scale
               for li in range(self.n_layers)]
        B_v = [self.b_heads_v[li](z).reshape(self.rank, -1) * self.output_scale
               for li in range(self.n_layers)]
        return B_q, B_v


def load_m2p_dims() -> dict:
    if not V2_RESULTS.exists():
        raise FileNotFoundError(f"v2 results not found at {V2_RESULTS}")
    with open(V2_RESULTS) as f:
        v2 = json.load(f)
    return {
        "n_layers":    v2["config"]["n_layers"],
        "d_model":     v2["config"]["d_model"],
        "q_proj_out":  v2["config"]["q_proj_out"],
        "v_proj_out":  v2["config"]["v_proj_out"],
    }


# ---- Training helper -------------------------------------------------------

def train_m2p_domain(
    domain: str,
    model_dims: dict,
    A_q_layers: list,
    A_v_layers: list,
    tokenized: list,
    n_steps: int,
    warm_start_path: Path | None = None,
    save_path: Path | None = None,
    seed_offset: int = 0,
) -> dict:
    n_layers   = model_dims["n_layers"]
    d_model    = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_q_layers, A_v_layers)
    mx.eval(model.parameters())

    m2p = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                     rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                     output_scale=OUTPUT_SCALE)
    mx.eval(m2p.parameters())
    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  [{domain}] M2P params: {n_params:,}")

    warm_start_used = False
    if warm_start_path and warm_start_path.exists():
        saved = np.load(str(warm_start_path))
        m2p.load_weights([(k, mx.array(saved[k])) for k in saved.files])
        mx.eval(m2p.parameters())
        warm_start_used = True
        log(f"  [{domain}] Warm-start from {warm_start_path}")

    def loss_fn(net, tokens_arr):
        hs = extract_hidden_states(model, tokens_arr, A_q_layers, A_v_layers, q_proj_out, v_proj_out)
        B_q, B_v = net(hs)
        logits = forward_with_loras(model, tokens_arr, B_q, B_v, A_q_layers, A_v_layers)
        return nn.losses.cross_entropy(logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean")

    loss_and_grad = nn.value_and_grad(m2p, loss_fn)

    rng_train = random.Random(SEED + seed_offset)
    smoke_toks = mx.array(rng_train.choice(tokenized))[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_toks)
    mx.eval(smoke_loss, smoke_grads)
    grad_norm = math.sqrt(sum(float(mx.sum(g ** 2).item())
                              for _, g in tree_flatten(smoke_grads) if isinstance(g, mx.array)))
    init_loss = float(smoke_loss.item())
    log(f"  [{domain}] step 0: grad_norm={grad_norm:.6f} loss={init_loss:.4f}")
    del smoke_toks, smoke_loss, smoke_grads

    optimizer = optim.Adam(learning_rate=LR)

    def lr_sched(step):
        return LR * min(1.0, (step + 1) / LR_WARMUP)

    log(f"  [{domain}] Training {n_steps} steps...")
    gc.disable()
    losses = []
    for step in range(n_steps):
        tokens_arr = mx.array(rng_train.choice(tokenized))[None, :]
        optimizer.learning_rate = lr_sched(step)
        loss, grads = loss_and_grad(m2p, tokens_arr)
        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, n_steps // 5) == 0:
            recent = sum(losses[-10:]) / min(len(losses[-10:]), 10)
            log(f"  [{domain}] step {step+1}/{n_steps}: loss={recent:.4f}")
    gc.enable()

    final_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1)
    log(f"  [{domain}] final loss: {final_loss:.4f}")

    if save_path:
        params = dict(tree_flatten(m2p.parameters()))
        np.savez(str(save_path), **{k: np.array(v.astype(mx.float32)) for k, v in params.items()})
        log(f"  [{domain}] Saved to {save_path}")

    cleanup(m2p, model, tokenizer, optimizer)
    return {
        f"{domain}_grad_norm_step0": grad_norm,
        f"{domain}_initial_loss": init_loss,
        f"{domain}_final_loss": float(final_loss),
        f"{domain}_warm_start": warm_start_used,
        f"{domain}_n_params": n_params,
    }


# ---- TF-IDF Router ---------------------------------------------------------

class TFIDFRouter:
    """Nearest-centroid TF-IDF router on raw input text (before any model forward)."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.labels: list[str] = []

    def _tok(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _tf(self, tokens: list[str]) -> dict[str, float]:
        freq: dict[str, float] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = max(len(tokens), 1)
        return {k: v / total for k, v in freq.items()}

    def fit(self, texts_by_label: dict[str, list[str]]) -> None:
        self.labels = list(texts_by_label.keys())
        all_docs = [t for texts in texts_by_label.values() for t in texts]
        word_set: set[str] = set()
        for doc in all_docs:
            word_set.update(self._tok(doc))
        self.vocab = {w: i for i, w in enumerate(sorted(word_set))}
        V, N = len(self.vocab), len(all_docs)

        dtm = np.zeros((N, V), dtype=np.float32)
        idx = 0
        for texts in texts_by_label.values():
            for text in texts:
                for w, v in self._tf(self._tok(text)).items():
                    if w in self.vocab:
                        dtm[idx, self.vocab[w]] = v
                idx += 1

        df = (dtm > 0).sum(axis=0) + 1
        self.idf = np.log((N + 1) / df) + 1
        tfidf = dtm * self.idf

        self.centroids = np.zeros((len(self.labels), V), dtype=np.float32)
        idx = 0
        for li, texts in enumerate(texts_by_label.values()):
            n = len(texts)
            self.centroids[li] = tfidf[idx:idx + n].mean(axis=0)
            idx += n
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8
        self.centroids /= norms

    def predict(self, text: str) -> str:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for w, v in self._tf(self._tok(text)).items():
            if w in self.vocab:
                vec[self.vocab[w]] = v
        vec *= self.idf
        norm = np.linalg.norm(vec) + 1e-8
        vec /= norm
        return self.labels[int(np.argmax(self.centroids @ vec))]

    def accuracy(self, texts: list[str], labels: list[str]) -> float:
        return sum(self.predict(t) == l for t, l in zip(texts, labels)) / max(len(texts), 1)


# ---- Phases ----------------------------------------------------------------

def phase1_train_math(model_dims: dict, A_math_q: list, A_math_v: list) -> dict:
    log("\n" + "=" * 70)
    log(f"[Phase 1] Train Math M2P ({MATH_TRAIN_STEPS} steps, warm-start from v4)")
    log("=" * 70)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    rng = random.Random(SEED)
    train_exs = list(ds["train"])
    rng.shuffle(train_exs)
    n_train = 50 if IS_SMOKE else 500
    train_exs = train_exs[:n_train]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    tokenized = [tokenize_seq(tokenizer, f"Question: {ex['question']}", ex["answer"])
                 for ex in train_exs]
    tokenized = [t for t in tokenized if t]
    cleanup(model)
    del tokenizer
    log(f"  Tokenized {len(tokenized)} math examples")

    result = train_m2p_domain(
        domain="math", model_dims=model_dims,
        A_q_layers=A_math_q, A_v_layers=A_math_v,
        tokenized=tokenized, n_steps=MATH_TRAIN_STEPS,
        warm_start_path=V4_M2P_PATH, save_path=MATH_M2P_PATH, seed_offset=10,
    )
    result["math_train_exs"] = train_exs
    return result


def phase2_train_code(model_dims: dict, A_code_q: list, A_code_v: list) -> dict:
    log("\n" + "=" * 70)
    log(f"[Phase 2] Train Code M2P ({CODE_TRAIN_STEPS} steps, fresh)")
    log("=" * 70)

    n_train = 50 if IS_SMOKE else 200
    code_exs = generate_code_training_data(n_train)
    log(f"  Generated {len(code_exs)} code examples")
    log(f"  Example: {code_exs[0]['prompt']!r} -> {code_exs[0]['answer']!r}")

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    tokenized = [tokenize_seq(tokenizer, ex["prompt"], ex["answer"]) for ex in code_exs]
    tokenized = [t for t in tokenized if t]
    cleanup(model)
    del tokenizer
    log(f"  Tokenized {len(tokenized)} code examples")

    result = train_m2p_domain(
        domain="code", model_dims=model_dims,
        A_q_layers=A_code_q, A_v_layers=A_code_v,
        tokenized=tokenized, n_steps=CODE_TRAIN_STEPS,
        warm_start_path=None, save_path=CODE_M2P_PATH, seed_offset=20,
    )
    return result


def phase3_routing(math_train_exs: list) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 3] TF-IDF Routing (math vs code)")
    log("=" * 70)

    rng = random.Random(SEED + 300)
    all_math = [f"Question: {ex['question']}" for ex in math_train_exs]
    rng.shuffle(all_math)
    math_train = all_math[:N_ROUTE_TRAIN]
    math_test  = all_math[N_ROUTE_TRAIN:N_ROUTE_TRAIN + N_ROUTE_TEST]
    if len(math_test) < N_ROUTE_TEST:
        math_test = all_math[-N_ROUTE_TEST:]

    all_code = [make_code_prompt(t, tmpl_idx=i % 4) for i, t in enumerate(CODE_TASKS * 20)]
    rng.shuffle(all_code)
    code_train = all_code[:N_ROUTE_TRAIN]
    code_test  = all_code[N_ROUTE_TRAIN:N_ROUTE_TRAIN + N_ROUTE_TEST]
    if len(code_test) < N_ROUTE_TEST:
        code_test = all_code[-N_ROUTE_TEST:]

    router = TFIDFRouter()
    router.fit({"math": math_train, "code": code_train})

    test_texts  = math_test + code_test
    test_labels = ["math"] * len(math_test) + ["code"] * len(code_test)
    acc = router.accuracy(test_texts, test_labels)
    log(f"  Routing accuracy: {acc:.3f}")

    for li, label in enumerate(router.labels):
        top_idx = np.argsort(router.centroids[li])[::-1][:5]
        inv_vocab = {v: k for k, v in router.vocab.items()}
        log(f"  Top terms [{label}]: {[inv_vocab[i] for i in top_idx]}")

    return {"routing_accuracy": float(acc), "router": router}


def phase4_evaluate(model_dims: dict, math_test_exs: list, router: TFIDFRouter,
                    A_math_q, A_code_q, A_math_v, A_code_v) -> dict:
    """Evaluate base, single-adapter, and composed+routed."""
    log("\n" + "=" * 70)
    log("[Phase 4] Evaluation: base + single + composed (routed)")
    log("=" * 70)

    n_layers   = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]
    d_model    = model_dims["d_model"]

    math_saved = np.load(str(MATH_M2P_PATH))
    code_saved = np.load(str(CODE_M2P_PATH))

    def load_m2p(saved) -> M2PNetwork:
        m2p = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                         rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                         output_scale=OUTPUT_SCALE)
        m2p.load_weights([(k, mx.array(saved[k])) for k in saved.files])
        mx.eval(m2p.parameters())
        return m2p

    code_eval_tasks = CODE_TASKS[:N_EVAL_CODE]
    results = {}

    # ---- Base model ----
    log("\n  [Base] evaluation...")
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_math_q, A_math_v)
    mx.eval(model.parameters())
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    inject_lora_b(model, B_q_zero, B_v_zero)

    math_base_correct = 0
    for ex in math_test_exs[:N_EVAL_MATH]:
        gen = mlx_generate(model, tokenizer,
                           prompt=FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:",
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        if extract_gsm8k_answer(gen) == extract_gsm8k_answer(ex["answer"]):
            math_base_correct += 1
        mx.clear_cache()
    math_base_acc = math_base_correct / max(N_EVAL_MATH, 1)
    log(f"  [Base] math acc: {math_base_acc:.3f}")

    # Base code: use A_math for hs extraction (zero B → base model)
    code_base_correct = 0
    for task in code_eval_tasks:
        gen = mlx_generate(model, tokenizer,
                           prompt=make_code_prompt(task, 0),
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        code_base_correct += eval_code_output(gen, task["func_name"], task["test_cases"])
        mx.clear_cache()
    code_base_acc = code_base_correct / max(len(code_eval_tasks), 1)
    log(f"  [Base] code acc: {code_base_acc:.3f}")
    results.update({"base_math_acc": math_base_acc, "base_code_acc": code_base_acc})
    cleanup(model, tokenizer, B_q_zero, B_v_zero)

    # ---- Math single ----
    log("\n  [Math single] evaluation...")
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_math_q, A_math_v)
    mx.eval(model.parameters())
    m2p_math = load_m2p(math_saved)
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    math_single_correct = 0
    for ex in math_test_exs[:N_EVAL_MATH]:
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        toks   = mx.array(tokenizer.encode(prompt))[None, :]
        hs = extract_hidden_states(model, toks, A_math_q, A_math_v, q_proj_out, v_proj_out)
        mx.eval(hs)
        B_q, B_v = m2p_math(hs)
        mx.eval(*B_q, *B_v)
        inject_lora_b(model, B_q, B_v)
        del toks, hs, B_q, B_v
        gen = mlx_generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        if extract_gsm8k_answer(gen) == extract_gsm8k_answer(ex["answer"]):
            math_single_correct += 1
        mx.clear_cache()

    math_single_acc = math_single_correct / max(N_EVAL_MATH, 1)
    log(f"  [Math single] math acc: {math_single_acc:.3f}")
    results["math_single_acc"] = math_single_acc
    log_memory("post-math-single")
    cleanup(model, tokenizer, m2p_math, B_q_zero, B_v_zero)

    # ---- Code single ----
    log("\n  [Code single] evaluation...")
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    apply_lora_structure(model, A_code_q, A_code_v)
    mx.eval(model.parameters())
    m2p_code = load_m2p(code_saved)
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    code_single_correct = 0
    for task in code_eval_tasks:
        prompt = make_code_prompt(task, 0)
        toks   = mx.array(tokenizer.encode(prompt))[None, :]
        hs = extract_hidden_states(model, toks, A_code_q, A_code_v, q_proj_out, v_proj_out)
        mx.eval(hs)
        B_q, B_v = m2p_code(hs)
        mx.eval(*B_q, *B_v)
        inject_lora_b(model, B_q, B_v)
        del toks, hs, B_q, B_v
        gen = mlx_generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        code_single_correct += eval_code_output(gen, task["func_name"], task["test_cases"])
        mx.clear_cache()

    code_single_acc = code_single_correct / max(len(code_eval_tasks), 1)
    log(f"  [Code single] code acc: {code_single_acc:.3f}")
    results["code_single_acc"] = code_single_acc
    log_memory("post-code-single")
    cleanup(model, tokenizer, m2p_code, B_q_zero, B_v_zero)

    # ---- Composed + routed ----
    log("\n  [Composed+Routed] evaluation...")
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    # Start with math A structure; will swap A-matrices per example
    apply_lora_structure(model, A_math_q, A_math_v)
    mx.eval(model.parameters())
    m2p_math2 = load_m2p(math_saved)
    m2p_code2 = load_m2p(code_saved)
    B_q_zero2 = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero2 = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    # Math under composition
    math_composed_correct = 0
    for ex in math_test_exs[:N_EVAL_MATH]:
        prompt   = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        route_in = f"Question: {ex['question']}"
        domain   = router.predict(route_in)

        A_q_use = A_math_q if domain == "math" else A_code_q
        A_v_use = A_math_v if domain == "math" else A_code_v
        m2p_use = m2p_math2 if domain == "math" else m2p_code2

        toks = mx.array(tokenizer.encode(prompt))[None, :]
        hs   = extract_hidden_states(model, toks, A_q_use, A_v_use, q_proj_out, v_proj_out)
        mx.eval(hs)
        B_q, B_v = m2p_use(hs)
        mx.eval(*B_q, *B_v)
        inject_lora_a(model, A_q_use, A_v_use)   # switch A-matrices for correct projection
        inject_lora_b(model, B_q, B_v)
        del toks, hs, B_q, B_v

        gen = mlx_generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        if extract_gsm8k_answer(gen) == extract_gsm8k_answer(ex["answer"]):
            math_composed_correct += 1
        mx.clear_cache()

    math_composed_acc = math_composed_correct / max(N_EVAL_MATH, 1)
    log(f"  [Composed] math acc: {math_composed_acc:.3f}")
    results["math_composed_acc"] = math_composed_acc

    # Code under composition
    code_composed_correct = 0
    for task in code_eval_tasks:
        prompt = make_code_prompt(task, 0)
        domain = router.predict(prompt)

        A_q_use = A_code_q if domain == "code" else A_math_q
        A_v_use = A_code_v if domain == "code" else A_math_v
        m2p_use = m2p_code2 if domain == "code" else m2p_math2

        toks = mx.array(tokenizer.encode(prompt))[None, :]
        hs   = extract_hidden_states(model, toks, A_q_use, A_v_use, q_proj_out, v_proj_out)
        mx.eval(hs)
        B_q, B_v = m2p_use(hs)
        mx.eval(*B_q, *B_v)
        inject_lora_a(model, A_q_use, A_v_use)
        inject_lora_b(model, B_q, B_v)
        del toks, hs, B_q, B_v

        gen = mlx_generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_GEN_TOKENS, verbose=False)
        code_composed_correct += eval_code_output(gen, task["func_name"], task["test_cases"])
        mx.clear_cache()

    code_composed_acc = code_composed_correct / max(len(code_eval_tasks), 1)
    log(f"  [Composed] code acc: {code_composed_acc:.3f}")
    results["code_composed_acc"] = code_composed_acc

    log_memory("post-composed-eval")
    cleanup(model, tokenizer, m2p_math2, m2p_code2, B_q_zero2)
    return results


# ---- Main ------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=" * 70)
    log("exp_m2p_2domain_compose_qwen06b — 2-Domain M2P: Math + Code")
    log(f"SMOKE_TEST: {IS_SMOKE}")
    log("=" * 70)

    random.seed(SEED)
    np.random.seed(SEED)

    model_dims = load_m2p_dims()
    log(f"  Model dims: {model_dims}")

    A_math_q, A_code_q, A_math_v, A_code_v = build_or_load_grassmannian_pairs(
        model_dims["n_layers"], model_dims["d_model"], LORA_RANK
    )

    # Phase 1: Math
    math_result = phase1_train_math(model_dims, A_math_q, A_math_v)
    math_train_exs = math_result.pop("math_train_exs")

    # Phase 2: Code
    code_result = phase2_train_code(model_dims, A_code_q, A_code_v)

    # Phase 3: Routing
    routing_result = phase3_routing(math_train_exs)
    router = routing_result.pop("router")
    routing_acc = routing_result["routing_accuracy"]

    # Load GSM8K test split for held-out math evaluation
    from datasets import load_dataset as _load_ds
    gsm8k_test = list(_load_ds("gsm8k", "main")["test"])
    rng_test = random.Random(SEED + 999)
    rng_test.shuffle(gsm8k_test)
    math_test_exs = gsm8k_test[:max(N_EVAL_MATH, 5)]

    # Phase 4: Evaluate
    eval_res = phase4_evaluate(
        model_dims,
        math_test_exs,
        router,
        A_math_q, A_code_q, A_math_v, A_code_v,
    )

    # Quality ratios
    def qr(composed, base, single):
        denom = single - base
        if abs(denom) < 1e-6:
            return 1.0 if abs(composed - base) < 1e-6 else 0.0
        return (composed - base) / denom

    qr_math = qr(eval_res["math_composed_acc"], eval_res["base_math_acc"], eval_res["math_single_acc"])
    qr_code = qr(eval_res["code_composed_acc"], eval_res["base_code_acc"], eval_res["code_single_acc"])

    k954_pass = qr_math >= 0.80 and qr_code >= 0.80
    k955_pass = routing_acc >= 0.85

    log("\n" + "=" * 70)
    log("[Kill Criteria]")
    log(f"  K954: qr_math={qr_math:.3f}, qr_code={qr_code:.3f} (≥0.80 each) "
        f"→ {'PASS' if k954_pass else 'FAIL'}")
    log(f"  K955: routing={routing_acc:.3f} (≥0.85) → {'PASS' if k955_pass else 'FAIL'}")
    log(f"  Theorem 5 (math): grad_norm={math_result.get('math_grad_norm_step0', 0):.6f}")
    log(f"  Theorem 5 (code): grad_norm={code_result.get('code_grad_norm_step0', 0):.6f}")
    log(f"  Total runtime: {time.time()-t0:.1f}s")

    results = {
        "experiment": "exp_m2p_2domain_compose_qwen06b",
        "smoke_test": IS_SMOKE,
        "total_time_s": time.time() - t0,
        "config": {**model_dims, "model": MODEL_ID, "lora_rank": LORA_RANK,
                   "d_m2p": D_M2P, "math_steps": MATH_TRAIN_STEPS, "code_steps": CODE_TRAIN_STEPS},
        "training": {**math_result, **code_result},
        "routing": routing_result,
        "eval": eval_res,
        "quality_ratios": {"qr_math": qr_math, "qr_code": qr_code},
        "kill_criteria": {"K954_pass": k954_pass, "K955_pass": k955_pass},
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Results saved to {RESULTS_FILE}")
    log("[Done]")
    return results


if __name__ == "__main__":
    main()
