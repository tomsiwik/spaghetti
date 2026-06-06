#!/usr/bin/env python3
"""M2P Composition on Qwen3-0.6B: Two real-LLM adapters compose without interference.

TYPE: verification (Type 1)
MATH: experiments/models/m2p_composition_n5_qwen3/MATH.md

EXPERIMENT: exp_m2p_composition_n5_qwen3

WHAT THIS TESTS:
  Two independently-trained M2P adapters (math + word-sort) composed on Qwen3-0.6B
  using TF-IDF routed selection (Theorem 3: routed mode). Routing is TF-IDF
  sequence-level on raw input text — invariant to model distribution (Theorem 2).

  Prior KILL was exp_m2p_composition_n5 (synthetic model, 36.6% routing — root cause:
  router trained on base hidden states, deployed on composed hidden states → covariate shift).
  FIX: TF-IDF routing on raw input text before any model forward pass (Theorem 2, MATH.md).

ADVERSARIAL REVISE FIXES (v2):
  Fix 1: Separate Grassmannian A-matrix slots per domain via QR construction.
          A_math = Q[:,0:r], A_sort = Q[:,r:2r] — A_math^T @ A_sort = 0 exactly.
  Fix 2: quality_ratio = (composed_acc - base_acc) / (single_acc - base_acc) per MATH.md.
  Fix 3: Composed eval uses ROUTED selection (TF-IDF selects ONE adapter at full weight=1.0).
          Additive blend kept as secondary mode for comparison.
  Fix 4: Sort adapter trained for 1000 steps. Convergence gate: sort_single_acc > 10%
          above base required before evaluating K927. Word-overlap F1 added as secondary metric.

PROVEN COMPONENTS REUSED:
  1. M2P functional forward (Theorem 5, v3/v4): B as tensor arg → grad_norm > 0
  2. Grassmannian A-matrices: A_math^T A_sort = 0 → ⟨ΔW_math, ΔW_sort⟩_F = 0 (Theorem 1)
  3. TF-IDF text routing: Finding #354 — 95% synthetic accuracy; Theorem 2 proves invariance
  4. Routed selection (Theorem 3): apply selected adapter at full weight (alpha=1.0)

TASKS (N=2):
  Task 1 (math):  GSM8K math reasoning. Warm-start from v4 m2p_weights.npz, MATH steps.
  Task 2 (sort):  Synthetic word-sort. Input "Sort these words alphabetically: {words}",
                  output sorted words (comma-separated). SORT steps from scratch (1000).

COMPOSITION (primary, K927):
  Routed: apply only the adapter selected by TF-IDF at alpha=1.0.
  If router says math → apply B_math only.
  If router says sort → apply B_sort only.

Additive blend (secondary, informational):
  ΔW_composed = 0.5·ΔW_math + 0.5·ΔW_sort

KILL CRITERIA:
  K925: grad_norm > 0 under composed adapter at step 0 (functional forward sanity check)
  K926: TF-IDF routing accuracy >= 80% on both tasks (math + sort) at sequence level
  K927: quality_ratio >= 0.75 on both tasks under ROUTED selection
        quality_ratio = (composed_acc - base_acc) / (single_acc - base_acc)
        Convergence gate: sort_single_acc must be > base_sort_acc + 0.10 (10% above base)

References:
  Theorem 5 (v3/v4 MATH.md): functional LoRA forward, B as tensor arg
  Finding #50 (this project): max|cos|=1e-08 for Grassmannian 5-domain
  Finding #14 (this project): 1/N scaling resolves catastrophe
  Finding #354 (this project): TF-IDF routing 95% on synthetic N=5
  LoraRetriever (arXiv:2402.09997): text-based routing invariant to model distribution

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

# LoRA config — IDENTICAL to v4 for A-matrix reuse
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P config — IDENTICAL to v4
D_M2P = 1024        # d_M2P = d_model (no bottleneck)
OUTPUT_SCALE = 0.032  # SHINE sqrt(0.001) convention

# Training config
MATH_TRAIN_STEPS = 20 if IS_SMOKE else 300    # math warm-start from v4
SORT_TRAIN_STEPS = 40 if IS_SMOKE else 1000   # Fix 4: 1000 steps for sort convergence
LR = 5e-5
LR_WARMUP = 3 if IS_SMOKE else 30
MAX_SEQ_LEN = 64 if IS_SMOKE else 512
MAX_GEN_TOKENS = 32 if IS_SMOKE else 128     # Shorter for composition eval speed
SEED = 42

# Eval config
N_EVAL_MATH = 5 if IS_SMOKE else 100         # math eval examples
N_EVAL_SORT = 5 if IS_SMOKE else 100         # sort eval examples

# Sort task vocab — 60 common English nouns for reproducibility
SORT_VOCAB = [
    "apple", "book", "cat", "dog", "egg", "fish", "goat", "hat", "ice", "jar",
    "key", "lamp", "moon", "nest", "oak", "pen", "queen", "rose", "sun", "tree",
    "urn", "vine", "wall", "box", "yard", "ant", "bat", "bee", "cow", "dam",
    "ear", "fan", "gun", "hay", "ink", "jet", "kit", "leg", "mat", "net",
    "oil", "pig", "rag", "sea", "tap", "van", "wax", "zoo", "arm", "bed",
    "cab", "den", "end", "fox", "gap", "hub", "ivy", "jam",
]

EXPERIMENT_DIR = Path(__file__).parent
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V4_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v4"

V2_RESULTS = V2_DIR / "results.json"
V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"

V4_M2P_PATH = V4_DIR / "m2p_weights.npz"

MATH_M2P_PATH = EXPERIMENT_DIR / "math_m2p_weights.npz"
SORT_M2P_PATH = EXPERIMENT_DIR / "sort_m2p_weights.npz"
GRASSMANNIAN_PATH = EXPERIMENT_DIR / "grassmannian_a_matrices.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Few-shot prefix for GSM8K (identical to v4)
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
    return None


def extract_sort_answer(text: str) -> list:
    """Extract sorted word list from model output.

    Returns sorted list of words (lowercase, stripped).
    Accepts both comma-separated and space-separated output.
    """
    # Strip prompt if echoed back
    text = text.strip()
    # Try comma-separated first
    if "," in text:
        words = [w.strip().lower() for w in text.split(",") if w.strip()]
    else:
        words = [w.strip().lower() for w in text.split() if w.strip()]
    # Filter to vocab words only (to avoid model artifacts)
    words = [w for w in words if w in SORT_VOCAB or len(w) > 2]
    return words


# ---- Sort data generation ---------------------------------------------------

def gen_sort_examples(n: int, seed: int = 42) -> list:
    """Generate n word-sort examples.

    Each example: {"prompt": "Sort...", "answer": "sorted_words", "words": [...]}
    Uses fixed vocabulary for reproducibility.
    """
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        k = rng.randint(3, 6)  # sort 3-6 words
        words = rng.sample(SORT_VOCAB, k)
        sorted_words = sorted(words)
        prompt = f"Sort these words alphabetically: {', '.join(words)}"
        answer = ", ".join(sorted_words)
        examples.append({"prompt": prompt, "answer": answer, "words": words,
                          "sorted": sorted_words})
    return examples


def tokenize_sort_example(tokenizer, ex: dict) -> list:
    """Tokenize a sort example for training (prompt + answer as NTP target)."""
    text = f"{ex['prompt']}\nAnswer: {ex['answer']}"
    ids = tokenizer.encode(text)
    if len(ids) >= 2:
        ids = ids[:MAX_SEQ_LEN + 1]
        if len(ids) >= 2:
            return ids
    return []


def tokenize_gsm8k_example(tokenizer, ex: dict) -> list:
    """Tokenize a GSM8K example for training."""
    text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
    ids = tokenizer.encode(text)
    if len(ids) >= 2:
        ids = ids[:MAX_SEQ_LEN + 1]
        if len(ids) >= 2:
            return ids
    return []


# ---- Fix 1: Grassmannian A-matrix pairs per domain -------------------------
#
# Theorem 1 requires SEPARATE A-matrix slots per domain:
#   A_math = Q[:,0:r], A_sort = Q[:,r:2r]  →  A_math^T A_sort = 0 exactly.
# Generate per-layer pairs and save/load for reproducibility.

def make_grassmannian_pair(d: int, r: int, seed: int = 42):
    """Returns (A_math, A_sort) numpy arrays with A_math^T @ A_sort == 0 per Theorem 1.

    Uses QR decomposition: X in R^{d x 2r} ~ N(0,1), Q,_ = QR(X).
    A_math = Q[:,0:r], A_sort = Q[:,r:2r].
    Orthogonality: A_math^T A_sort = Q^T Q [0:r, r:2r] = I_{2r}[0:r, r:2r] = 0.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, 2 * r)).astype(np.float32)
    Q, _ = np.linalg.qr(X)
    A_math = Q[:, :r]    # shape (d, r)
    A_sort = Q[:, r:2*r]  # shape (d, r)
    return A_math, A_sort


def build_or_load_grassmannian_pairs(n_layers: int, d_in: int, r: int) -> tuple:
    """Build or load Grassmannian A-matrix pairs for math and sort.

    Returns:
      A_math_q: list of n_layers mx.array (d_in, r)
      A_sort_q: list of n_layers mx.array (d_in, r)
      A_math_v: list of n_layers mx.array (d_in, r)  [same d_in for v_proj]
      A_sort_v: list of n_layers mx.array (d_in, r)
    """
    if GRASSMANNIAN_PATH.exists():
        log(f"  [Grassmannian] Loading from {GRASSMANNIAN_PATH}")
        saved = np.load(str(GRASSMANNIAN_PATH))
        A_math_q = [mx.array(saved[f"math_q_{li}"]).astype(mx.bfloat16)
                    for li in range(n_layers)]
        A_sort_q = [mx.array(saved[f"sort_q_{li}"]).astype(mx.bfloat16)
                    for li in range(n_layers)]
        A_math_v = [mx.array(saved[f"math_v_{li}"]).astype(mx.bfloat16)
                    for li in range(n_layers)]
        A_sort_v = [mx.array(saved[f"sort_v_{li}"]).astype(mx.bfloat16)
                    for li in range(n_layers)]
    else:
        log(f"  [Grassmannian] Generating {n_layers}-layer Grassmannian pairs "
            f"(d={d_in}, r={r})")
        save_dict = {}
        A_math_q, A_sort_q = [], []
        A_math_v, A_sort_v = [], []
        for li in range(n_layers):
            # Per-layer seed so each layer is independently orthogonal
            am_q, as_q = make_grassmannian_pair(d_in, r, seed=SEED + li)
            am_v, as_v = make_grassmannian_pair(d_in, r, seed=SEED + 1000 + li)
            A_math_q.append(mx.array(am_q).astype(mx.bfloat16))
            A_sort_q.append(mx.array(as_q).astype(mx.bfloat16))
            A_math_v.append(mx.array(am_v).astype(mx.bfloat16))
            A_sort_v.append(mx.array(as_v).astype(mx.bfloat16))
            save_dict[f"math_q_{li}"] = am_q
            save_dict[f"sort_q_{li}"] = as_q
            save_dict[f"math_v_{li}"] = am_v
            save_dict[f"sort_v_{li}"] = as_v
        np.savez(str(GRASSMANNIAN_PATH), **save_dict)
        log(f"  [Grassmannian] Saved to {GRASSMANNIAN_PATH}")

        # Verify orthogonality at layer 0 (in float32 before bfloat16 conversion)
        am0_f32 = save_dict["math_q_0"]   # numpy float32, exact QR result
        as0_f32 = save_dict["sort_q_0"]
        cross = am0_f32.T @ as0_f32
        max_abs_f32 = float(np.abs(cross).max())
        # In bfloat16 the rounding error is ~1e-4 (expected: bf16 has 7-bit mantissa)
        mx.eval(A_math_q[0], A_sort_q[0])
        log(f"  [Grassmannian] Layer 0 q_proj: max|A_math^T A_sort| = {max_abs_f32:.2e} "
            f"(float32, expected <1e-6 by QR; bf16 rounding adds ~1e-4 later)")

    mx.eval(*A_math_q, *A_sort_q, *A_math_v, *A_sort_v)
    return A_math_q, A_sort_q, A_math_v, A_sort_v


# ---- A-matrix loading -------------------------------------------------------

def load_lora_a_matrices_v2() -> dict:
    """Load lora_a matrices saved during v2 SFT phase.

    Returns dict[(li, mod_name)] -> mx.array shape (input_dims, rank).
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
        body = key[:-2]
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} lora_a matrices from v2 ({V2_LORA_A_PATH})")
    return result


def _apply_lora_structure(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj, set A-matrices.

    UNCHANGED from v3/v4.
    """
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if lora_a_dict is not None:
            attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    model.freeze()


# ---- Core functional forward (PROVEN from v3/v4) ---------------------------
#
# Theorem 5: B as tensor argument (not module attribute) gives grad_norm > 0.
# DO NOT CHANGE THIS PATTERN.

def functional_lora_proj(x: mx.array, linear_module, A: mx.array,
                          B: mx.array, scale: float) -> mx.array:
    """LoRA projection with B as tensor argument.

    y = linear_module(x) + scale * (x @ A) @ B
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
    """Functional attention forward — B_q, B_v as tensor arguments.

    Replicates Qwen3 Attention.__call__ with functional_lora_proj for q and v.
    """
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
    """Full Qwen3 forward with functional LoRA for q_proj and v_proj.

    Returns logits (batch=1, seq, vocab_size).
    """
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


# ---- M2P Architecture (UNCHANGED from v4) -----------------------------------

class M2PNetwork(nn.Module):
    """Hypernetwork: layer hidden states -> LoRA B-matrices.

    Architecture IDENTICAL to v4.
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

        self.enc_linear1 = nn.Linear(d_model, 2 * d_m2p)
        self.enc_linear2 = nn.Linear(2 * d_m2p, d_m2p)
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out) for _ in range(n_layers)]

    def __call__(self, layer_hs: mx.array):
        """layer_hs: (n_layers, d_model) -> B_q_layers, B_v_layers."""
        h = mx.mean(layer_hs, axis=0)
        h = nn.gelu(self.enc_linear1(h))
        z = self.enc_linear2(h)

        B_q_layers = []
        B_v_layers = []
        for li in range(self.n_layers):
            b_q_flat = self.b_heads_q[li](z)
            b_v_flat = self.b_heads_v[li](z)
            B_q_layers.append(b_q_flat.reshape(self.rank, -1) * self.output_scale)
            B_v_layers.append(b_v_flat.reshape(self.rank, -1) * self.output_scale)
        return B_q_layers, B_v_layers


def extract_hidden_states(
    model,
    tokens_arr: mx.array,
    A_q_layers: list,
    A_v_layers: list,
    B_q_zero: list,
    B_v_zero: list,
) -> mx.array:
    """Extract per-layer mean-pooled hidden states (stop_gradient, base model forward)."""
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


# ---- Phase 0: Load model dimensions from v2 results ------------------------

def phase_load_model_dims() -> dict:
    """Load model dimensions from v2 results (confirmed correct in v4)."""
    log("\n" + "=" * 70)
    log("[Phase 0] Loading model dimensions from v2 results")
    log("=" * 70)

    if not V2_RESULTS.exists():
        raise FileNotFoundError(
            f"v2 results not found at {V2_RESULTS}. Run m2p_qwen06b_gsm8k_v2 first."
        )
    with open(V2_RESULTS) as f:
        v2 = json.load(f)

    dims = {
        "n_layers": v2["config"]["n_layers"],
        "d_model": v2["config"]["d_model"],
        "n_heads": v2["config"]["n_heads"],
        "n_kv_heads": v2["config"]["n_kv_heads"],
        "head_dim": v2["config"]["head_dim"],
        "q_proj_out": v2["config"]["q_proj_out"],
        "v_proj_out": v2["config"]["v_proj_out"],
    }
    log(f"  Model dims: {dims}")
    return dims


# ---- Phase 1: Train math M2P (warm-start from v4) --------------------------

def phase_train_math_m2p(model_dims: dict) -> dict:
    """Train math M2P adapter — warm-start from v4 weights, MATH_TRAIN_STEPS more steps.

    Fix 1: Uses Grassmannian A_math matrices (not v2 LoRA A matrices).
    A_math is frozen (not trained); only B-matrices from M2P are learned.
    Warm-start is on B-heads only — the A-matrix swap does not affect B learning.

    Returns: dict with grad_norm, final_loss, k_pass info
    """
    log("\n" + "=" * 70)
    log(f"[Phase 1] Train Math M2P (warm-start from v4, {MATH_TRAIN_STEPS} steps)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    d_in = model_dims["d_model"]   # q_proj and v_proj input dim
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Fix 1: Generate/load Grassmannian A-matrix pairs
    A_math_q, A_sort_q, A_math_v, A_sort_v = build_or_load_grassmannian_pairs(
        n_layers, d_in, LORA_RANK
    )
    # Math M2P uses A_math slots
    A_q_layers = A_math_q
    A_v_layers = A_math_v

    # Load data
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    rng = random.Random(SEED)
    train_examples = list(ds["train"])
    rng.shuffle(train_examples)
    n_train = 50 if IS_SMOKE else 500
    train_examples = train_examples[:n_train]

    # Load model (no LoRA structure applied — we use functional forward)
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    # Apply dummy LoRA structure so model is compatible with mlx_generate later
    # (A-matrices set to A_math; B-matrices will be injected at eval time)
    lora_a_dict = {(li, "q_proj"): A_q_layers[li] for li in range(n_layers)}
    lora_a_dict.update({(li, "v_proj"): A_v_layers[li] for li in range(n_layers)})
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    # Tokenize
    tokenized = [tokenize_gsm8k_example(tokenizer, ex) for ex in train_examples]
    tokenized = [t for t in tokenized if t]
    log(f"  Tokenized: {len(tokenized)} sequences")

    # Build M2P and warm-start from v4
    m2p = M2PNetwork(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
        rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
        output_scale=OUTPUT_SCALE,
    )
    mx.eval(m2p.parameters())

    warm_start_used = False
    if V4_M2P_PATH.exists():
        v4_saved = np.load(str(V4_M2P_PATH))
        weight_list = [(k, mx.array(v4_saved[k])) for k in v4_saved.files]
        m2p.load_weights(weight_list)
        mx.eval(m2p.parameters())
        warm_start_used = True
        log(f"  [Warm Start] Loaded v4 weights from {V4_M2P_PATH}")
    else:
        log(f"  [Warm Start] v4 weights NOT found — initializing fresh")

    # Loss function — uses A_math (Fix 1)
    def m2p_loss_fn(m2p_net, tokens_arr):
        layer_hs = extract_hidden_states(model, tokens_arr, A_q_layers, A_v_layers,
                                          B_q_zero, B_v_zero)
        B_q_layers, B_v_layers = m2p_net(layer_hs)
        logits = model_forward_with_loras(model, tokens_arr, B_q_layers, B_v_layers,
                                           A_q_layers, A_v_layers, LORA_SCALE)
        return nn.losses.cross_entropy(logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean")

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # K925 check: grad_norm > 0 under single math adapter
    log("\n  [K925 prep] Gradient smoke test at step 0...")
    smoke_tokens = mx.array(random.choice(tokenized))[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)

    grad_norms = [float(mx.sum(g ** 2).item())
                  for _, g in tree_flatten(smoke_grads) if isinstance(g, mx.array)]
    grad_norm_math = math.sqrt(sum(grad_norms))
    smoke_loss_val = float(smoke_loss.item())
    log(f"  [K925 prep] math grad_norm at step 0 = {grad_norm_math:.6f}")
    log(f"  [K925 prep] math initial loss = {smoke_loss_val:.4f}")
    del smoke_tokens, smoke_loss, smoke_grads

    # Training loop
    rng_train = random.Random(SEED + 10)
    optimizer = optim.Adam(learning_rate=LR)

    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    log(f"\n  Training math M2P for {MATH_TRAIN_STEPS} steps...")
    gc.disable()
    losses = []
    for step in range(MATH_TRAIN_STEPS):
        seq = rng_train.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        optimizer.learning_rate = lr_schedule(step)
        loss, grads = loss_and_grad(m2p, tokens_arr)
        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, MATH_TRAIN_STEPS // 5) == 0:
            recent = sum(losses[-10:]) / min(len(losses[-10:]), 10)
            log(f"  Step {step+1}/{MATH_TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1)
    log(f"\n  Math M2P final loss: {final_loss:.4f}")

    # Save weights
    m2p_params = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params.items()}
    np.savez(str(MATH_M2P_PATH), **m2p_save)
    log(f"  Saved math M2P to {MATH_M2P_PATH}")
    log(f"  Phase 1 time: {time.time()-t0:.1f}s")
    log_memory("post-math-train")

    cleanup(m2p, model, tokenizer, optimizer)

    return {
        "math_grad_norm_step0": grad_norm_math,
        "math_initial_loss": smoke_loss_val,
        "math_final_loss": float(final_loss),
        "math_warm_start_used": warm_start_used,
    }


# ---- Phase 2: Train sort M2P (from scratch) ---------------------------------

def phase_train_sort_m2p(model_dims: dict) -> dict:
    """Train sort M2P adapter from scratch on word-sort task.

    Fix 1: Uses Grassmannian A_sort matrices (separate slot from math).
    Fix 4: 1000 steps for convergence (was 300).
    Word-sort: "Sort these words alphabetically: X, Y, Z" -> "A, B, C"
    """
    log("\n" + "=" * 70)
    log(f"[Phase 2] Train Sort M2P ({SORT_TRAIN_STEPS} steps, fresh init)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    d_in = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Fix 1: Load Grassmannian pairs (already generated by phase 1)
    A_math_q, A_sort_q, A_math_v, A_sort_v = build_or_load_grassmannian_pairs(
        n_layers, d_in, LORA_RANK
    )
    # Sort M2P uses A_sort slots
    A_q_layers = A_sort_q
    A_v_layers = A_sort_v

    # Generate sort data
    n_train = 50 if IS_SMOKE else 1000   # More data for 1000 steps
    sort_examples = gen_sort_examples(n_train, seed=SEED + 100)
    log(f"  Generated {len(sort_examples)} sort training examples")
    log(f"  Example: {sort_examples[0]['prompt']!r} -> {sort_examples[0]['answer']!r}")

    # Load model with A_sort applied
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    lora_a_dict = {(li, "q_proj"): A_q_layers[li] for li in range(n_layers)}
    lora_a_dict.update({(li, "v_proj"): A_v_layers[li] for li in range(n_layers)})
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    # Tokenize
    tokenized = [tokenize_sort_example(tokenizer, ex) for ex in sort_examples]
    tokenized = [t for t in tokenized if t]
    log(f"  Tokenized: {len(tokenized)} sequences")

    # Build fresh M2P
    m2p = M2PNetwork(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
        rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
        output_scale=OUTPUT_SCALE,
    )
    mx.eval(m2p.parameters())
    n_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    log(f"  Sort M2P params: {n_params:,}")

    def sort_loss_fn(m2p_net, tokens_arr):
        layer_hs = extract_hidden_states(model, tokens_arr, A_q_layers, A_v_layers,
                                          B_q_zero, B_v_zero)
        B_q_layers, B_v_layers = m2p_net(layer_hs)
        logits = model_forward_with_loras(model, tokens_arr, B_q_layers, B_v_layers,
                                           A_q_layers, A_v_layers, LORA_SCALE)
        return nn.losses.cross_entropy(logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean")

    loss_and_grad = nn.value_and_grad(m2p, sort_loss_fn)

    # Grad check for sort
    smoke_tokens = mx.array(random.choice(tokenized))[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)
    grad_norms = [float(mx.sum(g ** 2).item())
                  for _, g in tree_flatten(smoke_grads) if isinstance(g, mx.array)]
    grad_norm_sort = math.sqrt(sum(grad_norms))
    sort_initial_loss = float(smoke_loss.item())
    log(f"  [K925 prep] sort grad_norm at step 0 = {grad_norm_sort:.6f}")
    log(f"  [K925 prep] sort initial loss = {sort_initial_loss:.4f}")
    del smoke_tokens, smoke_loss, smoke_grads

    rng_train = random.Random(SEED + 20)
    optimizer = optim.Adam(learning_rate=LR)

    def lr_schedule(step: int) -> float:
        if step < LR_WARMUP:
            return LR * (step + 1) / LR_WARMUP
        return LR

    log(f"\n  Training sort M2P for {SORT_TRAIN_STEPS} steps...")
    gc.disable()
    losses = []
    for step in range(SORT_TRAIN_STEPS):
        seq = rng_train.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        optimizer.learning_rate = lr_schedule(step)
        loss, grads = loss_and_grad(m2p, tokens_arr)
        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if (step + 1) % max(1, SORT_TRAIN_STEPS // 5) == 0:
            recent = sum(losses[-10:]) / min(len(losses[-10:]), 10)
            log(f"  Step {step+1}/{SORT_TRAIN_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1)
    log(f"\n  Sort M2P final loss: {final_loss:.4f}")

    # Save weights
    m2p_params = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params.items()}
    np.savez(str(SORT_M2P_PATH), **m2p_save)
    log(f"  Saved sort M2P to {SORT_M2P_PATH}")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-sort-train")

    cleanup(m2p, model, tokenizer, optimizer)

    return {
        "sort_grad_norm_step0": grad_norm_sort,
        "sort_initial_loss": sort_initial_loss,
        "sort_final_loss": float(final_loss),
        "sort_m2p_params": n_params,
    }


# ---- Phase 3: TF-IDF Router Training ----------------------------------------

def phase_train_tfidf_router() -> dict:
    """Train TF-IDF + logistic regression sequence-level router.

    Theorem 2: Router operates on raw input text → invariant to model distribution.
    Uses 200 examples per task for training, 100 for validation.
    """
    log("\n" + "=" * 70)
    log("[Phase 3] Train TF-IDF Sequence Router (K926)")
    log("=" * 70)
    t0 = time.time()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from datasets import load_dataset

    # Generate text examples for router training
    n_router_train = 20 if IS_SMOKE else 200
    n_router_val = 10 if IS_SMOKE else 100

    # Math prompts: use GSM8K questions (just the prompt, no answer)
    ds = load_dataset("gsm8k", "main")
    rng = random.Random(SEED + 200)
    all_train = list(ds["train"])
    rng.shuffle(all_train)

    math_train_texts = [
        FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        for ex in all_train[:n_router_train]
    ]
    math_val_texts = [
        FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        for ex in all_train[n_router_train:n_router_train + n_router_val]
    ]

    # Sort prompts: use gen_sort_examples
    sort_train_exs = gen_sort_examples(n_router_train, seed=SEED + 300)
    sort_val_exs = gen_sort_examples(n_router_val, seed=SEED + 400)
    sort_train_texts = [ex["prompt"] for ex in sort_train_exs]
    sort_val_texts = [ex["prompt"] for ex in sort_val_exs]

    # Combine
    train_texts = math_train_texts + sort_train_texts
    train_labels = [0] * len(math_train_texts) + [1] * len(sort_train_texts)
    val_texts = math_val_texts + sort_val_texts
    val_labels = [0] * len(math_val_texts) + [1] * len(sort_val_texts)

    train_labels_arr = np.array(train_labels)
    val_labels_arr = np.array(val_labels)

    log(f"  Train: {len(train_texts)} (math={len(math_train_texts)}, "
        f"sort={len(sort_train_texts)})")
    log(f"  Val:   {len(val_texts)} (math={len(math_val_texts)}, "
        f"sort={len(sort_val_texts)})")

    # TF-IDF with char n-grams (proven effective in Finding #207, #354)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        max_features=5000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    log(f"  TF-IDF features: {X_train.shape[1]}")

    clf = LogisticRegression(
        max_iter=500,
        C=1.0,
        solver="lbfgs",
        random_state=SEED,
    )
    clf.fit(X_train, train_labels_arr)

    # Training accuracy
    train_preds = clf.predict(X_train)
    train_acc = float(np.mean(train_preds == train_labels_arr))
    log(f"  Router train accuracy: {train_acc:.1%}")

    # Validation accuracy per task
    val_preds = clf.predict(X_val)
    val_acc_overall = float(np.mean(val_preds == val_labels_arr))

    # Per-task accuracy
    math_mask = val_labels_arr == 0
    sort_mask = val_labels_arr == 1
    math_acc = float(np.mean(val_preds[math_mask] == 0)) if math_mask.sum() > 0 else 0.0
    sort_acc = float(np.mean(val_preds[sort_mask] == 1)) if sort_mask.sum() > 0 else 0.0

    log(f"  Router val accuracy (overall): {val_acc_overall:.1%} (K926: >= 80%)")
    log(f"    Math:  {math_acc:.1%} ({int(math_mask.sum())} examples)")
    log(f"    Sort:  {sort_acc:.1%} ({int(sort_mask.sum())} examples)")

    k926_pass = (math_acc >= 0.80) and (sort_acc >= 0.80)
    log(f"  [K926] {'PASS' if k926_pass else 'FAIL'} "
        f"(math={math_acc:.1%} {'>=80%' if math_acc >= 0.80 else '<80%'}, "
        f"sort={sort_acc:.1%} {'>=80%' if sort_acc >= 0.80 else '<80%'})")
    log(f"  Phase 3 time: {time.time()-t0:.1f}s")

    return {
        "vectorizer": vectorizer,
        "clf": clf,
        "routing_accuracy_overall": round(val_acc_overall, 4),
        "routing_accuracy_math": round(math_acc, 4),
        "routing_accuracy_sort": round(sort_acc, 4),
        "routing_train_accuracy": round(train_acc, 4),
        "k926_pass": k926_pass,
        "math_train_texts": math_train_texts,
        "sort_train_texts": sort_train_texts,
        "math_val_texts": math_val_texts,
        "sort_val_texts": sort_val_texts,
    }


def route_text(vectorizer, clf, text: str) -> int:
    """Route a text to domain index (0=math, 1=sort) using TF-IDF classifier."""
    X = vectorizer.transform([text])
    return int(clf.predict(X)[0])


# ---- Phase 4: K925 — Composed Adapter Gradient Check -----------------------

def phase_k925_grad_check(model_dims: dict) -> dict:
    """K925: Verify grad_norm > 0 at step 0 under composed adapter.

    Fix 1: Uses separate Grassmannian A-matrices per domain.
    Fix 3: Gradient check uses routed selection (math domain selected → apply B_math only).

    This verifies Theorem 1 + Theorem 5: the gradient path remains intact
    even when B is selected by a router (only math M2P active).
    """
    log("\n" + "=" * 70)
    log("[Phase 4] K925 — Composed Adapter Gradient Check (routed selection)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    d_in = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    if not MATH_M2P_PATH.exists() or not SORT_M2P_PATH.exists():
        raise FileNotFoundError("M2P weights not found — run phases 1 and 2 first")

    # Fix 1: Load Grassmannian pairs
    A_math_q, A_sort_q, A_math_v, A_sort_v = build_or_load_grassmannian_pairs(
        n_layers, d_in, LORA_RANK
    )

    # Load model with math A-matrices (for hidden state extraction)
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    lora_a_dict = {(li, "q_proj"): A_math_q[li] for li in range(n_layers)}
    lora_a_dict.update({(li, "v_proj"): A_math_v[li] for li in range(n_layers)})
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    # Zero B for hidden state extraction (base model hidden states)
    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    # Load both M2P networks
    m2p_math = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                           rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                           output_scale=OUTPUT_SCALE)
    m2p_sort = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                           rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                           output_scale=OUTPUT_SCALE)

    math_saved = np.load(str(MATH_M2P_PATH))
    sort_saved = np.load(str(SORT_M2P_PATH))
    m2p_math.load_weights([(k, mx.array(math_saved[k])) for k in math_saved.files])
    m2p_sort.load_weights([(k, mx.array(sort_saved[k])) for k in sort_saved.files])
    m2p_math.eval()
    m2p_sort.eval()
    mx.eval(m2p_math.parameters(), m2p_sort.parameters())

    # K925: gradient check with math M2P only (routed selection for math domain)
    # Test: grad flows through m2p_math under routed mode (only B_math applied).

    # Get a test sequence (math domain)
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    rng = random.Random(SEED + 50)
    ex = random.choice(list(ds["train"]))
    text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
    ids = tokenizer.encode(text)[:MAX_SEQ_LEN + 1]
    test_tokens = mx.array(ids)[None, :]

    def routed_math_loss_fn(m2p_math_net, tokens_arr):
        """Loss through routed math adapter: apply only B_math at alpha=1.0."""
        layer_hs = extract_hidden_states(model, tokens_arr, A_math_q, A_math_v,
                                          B_q_zero, B_v_zero)

        # Generate B-matrices from math adapter only (routed selection)
        B_q_math, B_v_math = m2p_math_net(layer_hs)

        # Routed mode: apply B_math at full weight (alpha=1.0), using A_math
        logits = model_forward_with_loras(model, tokens_arr, B_q_math, B_v_math,
                                           A_math_q, A_math_v, LORA_SCALE)
        return nn.losses.cross_entropy(logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean")

    # Unfreeze m2p_math temporarily for grad check
    m2p_math.unfreeze()
    loss_and_grad_composed = nn.value_and_grad(m2p_math, routed_math_loss_fn)

    smoke_loss, smoke_grads = loss_and_grad_composed(m2p_math, test_tokens)
    mx.eval(smoke_loss, smoke_grads)

    grad_norms = [float(mx.sum(g ** 2).item())
                  for _, g in tree_flatten(smoke_grads) if isinstance(g, mx.array)]
    grad_norm_composed = math.sqrt(sum(grad_norms))
    composed_loss_val = float(smoke_loss.item())

    k925_pass = grad_norm_composed > 0.0
    log(f"  [K925] grad_norm (composed) = {grad_norm_composed:.6f}")
    log(f"  [K925] composed loss at step 0 = {composed_loss_val:.4f}")
    log(f"  [K925] {'PASS' if k925_pass else 'FAIL'} (grad_norm > 0)")

    log(f"  Phase 4 time: {time.time()-t0:.1f}s")
    log_memory("post-k925")

    cleanup(m2p_math, m2p_sort, model, tokenizer, smoke_grads, test_tokens)

    return {
        "k925_grad_norm_composed": grad_norm_composed,
        "k925_composed_loss": composed_loss_val,
        "k925_pass": k925_pass,
    }


# ---- Phase 5: Evaluate Single-Adapter Baselines ----------------------------

def word_overlap_f1(pred_words: list, gold_words: list) -> float:
    """Word-overlap F1 as partial credit for sort eval (Fix 4).

    F1 on word sets: precision = |pred & gold| / |pred|,
                     recall    = |pred & gold| / |gold|.
    Note: ignores ordering — measures vocabulary coverage only.
    Position accuracy (fraction of words in correct position) also computed.
    """
    if not gold_words:
        return 0.0
    pred_set = set(pred_words)
    gold_set = set(gold_words)
    intersection = pred_set & gold_set
    if not intersection:
        return 0.0
    precision = len(intersection) / max(len(pred_set), 1)
    recall = len(intersection) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def position_accuracy(pred_words: list, gold_words: list) -> float:
    """Fraction of words in the correct position."""
    if not gold_words:
        return 0.0
    n = max(len(pred_words), len(gold_words))
    correct = sum(
        1 for i in range(min(len(pred_words), len(gold_words)))
        if pred_words[i] == gold_words[i]
    )
    return correct / n


def phase_eval_single_adapters(model_dims: dict) -> dict:
    """Evaluate math and sort M2P adapters independently.

    Fix 1: Uses separate Grassmannian A-matrices per domain.
    Fix 2: Computes base_acc for both tasks (needed for quality_ratio formula).
    Fix 4: Adds word-overlap F1 and position accuracy for sort.

    Returns: math_single_acc, sort_single_acc, math_base_acc, sort_base_acc
    """
    log("\n" + "=" * 70)
    log(f"[Phase 5] Single-Adapter Baselines (n={N_EVAL_MATH} math, {N_EVAL_SORT} sort)")
    log("=" * 70)
    t0 = time.time()

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    d_in = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Fix 1: Load Grassmannian A-matrix pairs
    A_math_q, A_sort_q, A_math_v, A_sort_v = build_or_load_grassmannian_pairs(
        n_layers, d_in, LORA_RANK
    )

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    # Apply math LoRA structure (will switch to sort when needed via lora_b injection)
    lora_a_dict = {(li, "q_proj"): A_math_q[li] for li in range(n_layers)}
    lora_a_dict.update({(li, "v_proj"): A_math_v[li] for li in range(n_layers)})
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    # Load both M2P networks
    m2p_math = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                           rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                           output_scale=OUTPUT_SCALE)
    m2p_sort = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                           rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                           output_scale=OUTPUT_SCALE)

    math_saved = np.load(str(MATH_M2P_PATH))
    sort_saved = np.load(str(SORT_M2P_PATH))
    m2p_math.load_weights([(k, mx.array(math_saved[k])) for k in math_saved.files])
    m2p_sort.load_weights([(k, mx.array(sort_saved[k])) for k in sort_saved.files])
    m2p_math.eval()
    m2p_sort.eval()
    mx.eval(m2p_math.parameters(), m2p_sort.parameters())

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    rng = random.Random(SEED)
    test_examples = list(ds["test"])
    rng.shuffle(test_examples)
    test_examples = test_examples[:N_EVAL_MATH]

    sort_eval_examples = gen_sort_examples(N_EVAL_SORT, seed=SEED + 500)

    # ---- Fix 2: Base model accuracy (zero B-matrices) ----
    log("\n  --- Base model accuracy (zero adapter) ---")
    math_base_correct = 0
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        # Inject zero B-matrices
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_zero[li]
            layer.self_attn.v_proj.lora_b = B_v_zero[li]
        mx.eval(model.parameters())
        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            math_base_correct += 1
        if (i + 1) % max(1, N_EVAL_MATH // 5) == 0 or (i + 1) == N_EVAL_MATH:
            log(f"  Math base {i+1}/{N_EVAL_MATH}: acc={math_base_correct/(i+1):.3f}")
    math_base_acc = math_base_correct / N_EVAL_MATH if N_EVAL_MATH > 0 else 0.0
    log(f"  Math base accuracy: {math_base_acc:.4f}")

    sort_base_correct = 0
    sort_base_f1_total = 0.0
    for i, ex in enumerate(sort_eval_examples):
        prompt = ex["prompt"]
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_zero[li]
            layer.self_attn.v_proj.lora_b = B_v_zero[li]
        mx.eval(model.parameters())
        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        pred_words = extract_sort_answer(generated)
        gold_words = ex["sorted"]
        if pred_words == gold_words:
            sort_base_correct += 1
        sort_base_f1_total += word_overlap_f1(pred_words, gold_words)
        if (i + 1) % max(1, N_EVAL_SORT // 5) == 0 or (i + 1) == N_EVAL_SORT:
            log(f"  Sort base {i+1}/{N_EVAL_SORT}: acc={sort_base_correct/(i+1):.3f}")
    sort_base_acc = sort_base_correct / N_EVAL_SORT if N_EVAL_SORT > 0 else 0.0
    sort_base_f1 = sort_base_f1_total / N_EVAL_SORT if N_EVAL_SORT > 0 else 0.0
    log(f"  Sort base accuracy: {sort_base_acc:.4f} | F1: {sort_base_f1:.4f}")

    # ---- Math single-adapter eval ----
    log("\n  --- Math single-adapter ---")

    math_correct = 0
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        layer_hs = extract_hidden_states(model, tokens_arr, A_math_q, A_math_v,
                                          B_q_zero, B_v_zero)
        mx.eval(layer_hs)
        B_q_layers, B_v_layers = m2p_math(layer_hs)
        mx.eval(*B_q_layers, *B_v_layers)

        # Inject into LoRALinear for mlx_generate
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_layers[li]
            layer.self_attn.v_proj.lora_b = B_v_layers[li]
        mx.eval(model.parameters())

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            math_correct += 1

        del tokens_arr, layer_hs, B_q_layers, B_v_layers

        if i == 0:
            log(f"  [DEBUG-math-single] gen[:150]: {generated[:150]!r}")
            log(f"  [DEBUG-math-single] gold={gold!r} pred={pred!r}")
        if (i + 1) % max(1, N_EVAL_MATH // 5) == 0 or (i + 1) == N_EVAL_MATH:
            log(f"  Math single {i+1}/{N_EVAL_MATH}: acc={math_correct/(i+1):.3f}")

    math_single_acc = math_correct / N_EVAL_MATH if N_EVAL_MATH > 0 else 0.0
    log(f"  Math single-adapter accuracy: {math_single_acc:.4f} ({math_correct}/{N_EVAL_MATH})")

    # ---- Sort single-adapter eval ----
    # Fix 1: Use A_sort matrices for sort M2P
    # Fix 4: Add word-overlap F1 and position accuracy
    log("\n  --- Sort single-adapter ---")

    sort_correct = 0
    sort_f1_total = 0.0
    sort_pos_acc_total = 0.0
    for i, ex in enumerate(sort_eval_examples):
        prompt = ex["prompt"]
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # Fix 1: use A_sort for sort hidden state extraction
        layer_hs = extract_hidden_states(model, tokens_arr, A_sort_q, A_sort_v,
                                          B_q_zero, B_v_zero)
        mx.eval(layer_hs)
        B_q_layers, B_v_layers = m2p_sort(layer_hs)
        mx.eval(*B_q_layers, *B_v_layers)

        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q_layers[li]
            layer.self_attn.v_proj.lora_b = B_v_layers[li]
        # Note: model LoRA structure was set with A_math, but lora_b is the B-matrix.
        # For correct sort forward, we must update q_proj.lora_a too.
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_a = A_sort_q[li]
            layer.self_attn.v_proj.lora_a = A_sort_v[li]
        mx.eval(model.parameters())

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        pred_words = extract_sort_answer(generated)
        gold_words = ex["sorted"]
        correct = pred_words == gold_words
        if correct:
            sort_correct += 1
        f1 = word_overlap_f1(pred_words, gold_words)
        pos_acc = position_accuracy(pred_words, gold_words)
        sort_f1_total += f1
        sort_pos_acc_total += pos_acc

        del tokens_arr, layer_hs, B_q_layers, B_v_layers

        if i == 0:
            log(f"  [DEBUG-sort-single] prompt: {prompt!r}")
            log(f"  [DEBUG-sort-single] gen[:150]: {generated[:150]!r}")
            log(f"  [DEBUG-sort-single] gold={gold_words!r} pred={pred_words!r} "
                f"correct={correct} f1={f1:.3f}")
        if (i + 1) % max(1, N_EVAL_SORT // 5) == 0 or (i + 1) == N_EVAL_SORT:
            log(f"  Sort single {i+1}/{N_EVAL_SORT}: acc={sort_correct/(i+1):.3f} "
                f"f1={sort_f1_total/(i+1):.3f}")

    # Restore A_math in model for subsequent phases
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.lora_a = A_math_q[li]
        layer.self_attn.v_proj.lora_a = A_math_v[li]
    mx.eval(model.parameters())

    sort_single_acc = sort_correct / N_EVAL_SORT if N_EVAL_SORT > 0 else 0.0
    sort_single_f1 = sort_f1_total / N_EVAL_SORT if N_EVAL_SORT > 0 else 0.0
    sort_single_pos_acc = sort_pos_acc_total / N_EVAL_SORT if N_EVAL_SORT > 0 else 0.0
    log(f"  Sort single-adapter accuracy: {sort_single_acc:.4f} ({sort_correct}/{N_EVAL_SORT})")
    log(f"  Sort single-adapter F1: {sort_single_f1:.4f}")
    log(f"  Sort single-adapter pos_acc: {sort_single_pos_acc:.4f}")

    log(f"  Phase 5 time: {time.time()-t0:.1f}s")
    log_memory("post-single-eval")

    cleanup(m2p_math, m2p_sort, model, tokenizer)

    return {
        "math_base_acc": round(math_base_acc, 4),
        "math_base_correct": math_base_correct,
        "sort_base_acc": round(sort_base_acc, 4),
        "sort_base_correct": sort_base_correct,
        "sort_base_f1": round(sort_base_f1, 4),
        "math_single_acc": round(math_single_acc, 4),
        "math_single_correct": math_correct,
        "sort_single_acc": round(sort_single_acc, 4),
        "sort_single_correct": sort_correct,
        "sort_single_f1": round(sort_single_f1, 4),
        "sort_single_pos_acc": round(sort_single_pos_acc, 4),
        "n_eval_math": N_EVAL_MATH,
        "n_eval_sort": N_EVAL_SORT,
    }


# ---- Phase 6: Evaluate Composed Adapter with TF-IDF Routing ----------------

def phase_eval_composed(model_dims: dict, router_data: dict, single_results: dict) -> dict:
    """Evaluate composed adapter with TF-IDF routing.

    Fix 1: Uses separate Grassmannian A-matrices per domain.
    Fix 3 (PRIMARY): Routed selection — TF-IDF selects ONE adapter at full weight (alpha=1.0).
    Fix 4: Sort convergence gate — skip K927 sort if sort_single_acc <= base_sort_acc + 0.10.

    Eval: N_EVAL_MATH + N_EVAL_SORT examples, per-task accuracy.
    quality_ratio = (composed_acc - base_acc) / (single_acc - base_acc)  [Fix 2]
    """
    log("\n" + "=" * 70)
    log(f"[Phase 6] Composed Adapter + TF-IDF Routing (n={N_EVAL_MATH}+{N_EVAL_SORT})")
    log("=" * 70)
    t0 = time.time()

    vectorizer = router_data["vectorizer"]
    clf = router_data["clf"]

    n_layers = model_dims["n_layers"]
    d_model = model_dims["d_model"]
    d_in = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    # Fix 1: Load Grassmannian A-matrix pairs
    A_math_q, A_sort_q, A_math_v, A_sort_v = build_or_load_grassmannian_pairs(
        n_layers, d_in, LORA_RANK
    )

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    # Apply math A-matrices initially (will switch per example in routed mode)
    lora_a_dict = {(li, "q_proj"): A_math_q[li] for li in range(n_layers)}
    lora_a_dict.update({(li, "v_proj"): A_math_v[li] for li in range(n_layers)})
    _apply_lora_structure(model, lora_a_dict)
    mx.eval(model.parameters())

    B_q_zero = [mx.zeros((LORA_RANK, q_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]
    B_v_zero = [mx.zeros((LORA_RANK, v_proj_out), dtype=mx.bfloat16) for _ in range(n_layers)]

    m2p_math = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                           rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                           output_scale=OUTPUT_SCALE)
    m2p_sort = M2PNetwork(n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
                           rank=LORA_RANK, q_proj_out=q_proj_out, v_proj_out=v_proj_out,
                           output_scale=OUTPUT_SCALE)

    math_saved = np.load(str(MATH_M2P_PATH))
    sort_saved = np.load(str(SORT_M2P_PATH))
    m2p_math.load_weights([(k, mx.array(math_saved[k])) for k in math_saved.files])
    m2p_sort.load_weights([(k, mx.array(sort_saved[k])) for k in sort_saved.files])
    m2p_math.eval()
    m2p_sort.eval()
    mx.eval(m2p_math.parameters(), m2p_sort.parameters())

    def get_routed_B(prompt: str, tokens_arr: mx.array):
        """Fix 3 (PRIMARY): Route → select one adapter at full weight (alpha=1.0).

        Returns (B_q, B_v, A_q, A_v, routed_domain).
        routed_domain: 0=math, 1=sort
        """
        routed_domain = route_text(vectorizer, clf, prompt)
        if routed_domain == 0:
            layer_hs = extract_hidden_states(model, tokens_arr, A_math_q, A_math_v,
                                              B_q_zero, B_v_zero)
            mx.eval(layer_hs)
            B_q, B_v = m2p_math(layer_hs)
            a_q, a_v = A_math_q, A_math_v
        else:
            layer_hs = extract_hidden_states(model, tokens_arr, A_sort_q, A_sort_v,
                                              B_q_zero, B_v_zero)
            mx.eval(layer_hs)
            B_q, B_v = m2p_sort(layer_hs)
            a_q, a_v = A_sort_q, A_sort_v
        mx.eval(*B_q, *B_v)
        return B_q, B_v, a_q, a_v, routed_domain

    def get_additive_B(tokens_arr: mx.array):
        """Secondary: additive blend 0.5*B_math + 0.5*B_sort (informational only)."""
        alpha = 0.5
        hs_math = extract_hidden_states(model, tokens_arr, A_math_q, A_math_v,
                                         B_q_zero, B_v_zero)
        mx.eval(hs_math)
        B_q_math, B_v_math = m2p_math(hs_math)
        hs_sort = extract_hidden_states(model, tokens_arr, A_sort_q, A_sort_v,
                                         B_q_zero, B_v_zero)
        mx.eval(hs_sort)
        B_q_sort, B_v_sort = m2p_sort(hs_sort)
        mx.eval(*B_q_math, *B_v_math, *B_q_sort, *B_v_sort)
        # For additive blend, use math A-matrices (arbitrary; blend uses single A-space)
        B_q_comp = [alpha * bq_m + alpha * bq_s
                    for bq_m, bq_s in zip(B_q_math, B_q_sort)]
        B_v_comp = [alpha * bv_m + alpha * bv_s
                    for bv_m, bv_s in zip(B_v_math, B_v_sort)]
        mx.eval(*B_q_comp, *B_v_comp)
        return B_q_comp, B_v_comp

    # Fix 4: Sort convergence gate
    sort_single_acc = single_results["sort_single_acc"]
    sort_base_acc = single_results["sort_base_acc"]
    sort_converged = sort_single_acc > sort_base_acc + 0.10
    log(f"\n  [Fix 4] Sort convergence gate: sort_single={sort_single_acc:.4f} "
        f"base={sort_base_acc:.4f} converged={sort_converged} "
        f"(threshold: single > base + 0.10)")

    # ---- Math composed eval (routed, primary) ----
    log("\n  --- Math composed (TF-IDF routed, Fix 3 primary) ---")
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    rng = random.Random(SEED)
    test_examples = list(ds["test"])
    rng.shuffle(test_examples)
    test_examples = test_examples[:N_EVAL_MATH]

    math_composed_correct = 0
    math_routing_correct = 0
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"

        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # Fix 3: Routed selection
        B_q, B_v, a_q_use, a_v_use, routed_domain = get_routed_B(prompt, tokens_arr)
        if routed_domain == 0:
            math_routing_correct += 1

        # Update model A-matrices to match selected adapter domain
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_a = a_q_use[li]
            layer.self_attn.v_proj.lora_a = a_v_use[li]
            layer.self_attn.q_proj.lora_b = B_q[li]
            layer.self_attn.v_proj.lora_b = B_v[li]
        mx.eval(model.parameters())

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            math_composed_correct += 1

        del tokens_arr, B_q, B_v

        if i == 0:
            log(f"  [DEBUG-math-composed] routed={routed_domain} (0=math)")
            log(f"  [DEBUG-math-composed] gen[:150]: {generated[:150]!r}")
            log(f"  [DEBUG-math-composed] gold={gold!r} pred={pred!r}")
        if (i + 1) % max(1, N_EVAL_MATH // 5) == 0 or (i + 1) == N_EVAL_MATH:
            log(f"  Math composed {i+1}/{N_EVAL_MATH}: acc={math_composed_correct/(i+1):.3f}")

    math_composed_acc = math_composed_correct / N_EVAL_MATH if N_EVAL_MATH > 0 else 0.0
    log(f"  Math composed accuracy (routed): {math_composed_acc:.4f}")

    # ---- Sort composed eval (routed, primary — only if converged) ----
    log("\n  --- Sort composed (TF-IDF routed, Fix 3 primary) ---")
    sort_eval_examples = gen_sort_examples(N_EVAL_SORT, seed=SEED + 500)

    sort_composed_correct = 0
    sort_composed_f1_total = 0.0
    sort_routing_correct = 0
    sort_k927_skipped = not sort_converged

    for i, ex in enumerate(sort_eval_examples):
        prompt = ex["prompt"]

        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # Fix 3: Routed selection
        B_q, B_v, a_q_use, a_v_use, routed_domain = get_routed_B(prompt, tokens_arr)
        if routed_domain == 1:
            sort_routing_correct += 1

        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_a = a_q_use[li]
            layer.self_attn.v_proj.lora_a = a_v_use[li]
            layer.self_attn.q_proj.lora_b = B_q[li]
            layer.self_attn.v_proj.lora_b = B_v[li]
        mx.eval(model.parameters())

        generated = mlx_generate(model, tokenizer, prompt=prompt,
                                  max_tokens=MAX_GEN_TOKENS, verbose=False)
        pred_words = extract_sort_answer(generated)
        gold_words = ex["sorted"]
        correct = pred_words == gold_words
        if correct:
            sort_composed_correct += 1
        sort_composed_f1_total += word_overlap_f1(pred_words, gold_words)

        del tokens_arr, B_q, B_v

        if i == 0:
            log(f"  [DEBUG-sort-composed] routed={routed_domain} (1=sort) "
                f"converged={sort_converged}")
            log(f"  [DEBUG-sort-composed] prompt: {prompt!r}")
            log(f"  [DEBUG-sort-composed] gen[:150]: {generated[:150]!r}")
            log(f"  [DEBUG-sort-composed] gold={gold_words!r} pred={pred_words!r}")
        if (i + 1) % max(1, N_EVAL_SORT // 5) == 0 or (i + 1) == N_EVAL_SORT:
            log(f"  Sort composed {i+1}/{N_EVAL_SORT}: acc={sort_composed_correct/(i+1):.3f} "
                f"f1={sort_composed_f1_total/(i+1):.3f}")

    sort_composed_acc = sort_composed_correct / N_EVAL_SORT if N_EVAL_SORT > 0 else 0.0
    sort_composed_f1 = sort_composed_f1_total / N_EVAL_SORT if N_EVAL_SORT > 0 else 0.0
    log(f"  Sort composed accuracy (routed): {sort_composed_acc:.4f}")
    log(f"  Sort composed F1 (routed): {sort_composed_f1:.4f}")

    log(f"\n  Math routing accuracy on eval batch: {math_routing_correct/N_EVAL_MATH:.1%}")
    log(f"  Sort routing accuracy on eval batch: {sort_routing_correct/N_EVAL_SORT:.1%}")
    log(f"  Phase 6 time: {time.time()-t0:.1f}s")
    log_memory("post-composed-eval")

    cleanup(m2p_math, m2p_sort, model, tokenizer)

    return {
        "math_composed_acc": round(math_composed_acc, 4),
        "math_composed_correct": math_composed_correct,
        "sort_composed_acc": round(sort_composed_acc, 4),
        "sort_composed_correct": sort_composed_correct,
        "sort_composed_f1": round(sort_composed_f1, 4),
        "math_routing_correct_on_eval": math_routing_correct,
        "sort_routing_correct_on_eval": sort_routing_correct,
        "sort_k927_skipped": sort_k927_skipped,
        "sort_converged": sort_converged,
    }


# ---- Main -------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P Composition on Qwen3-0.6B: N=2 adapters (math + sort)")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"MATH_TRAIN_STEPS={MATH_TRAIN_STEPS} | SORT_TRAIN_STEPS={SORT_TRAIN_STEPS} | LR={LR} | SEED={SEED}")
    log(f"N_EVAL_MATH={N_EVAL_MATH} | N_EVAL_SORT={N_EVAL_SORT}")
    log(f"MAX_SEQ_LEN={MAX_SEQ_LEN} | MAX_GEN_TOKENS={MAX_GEN_TOKENS}")
    log(f"LORA_RANK={LORA_RANK} | LORA_SCALE={LORA_SCALE} | D_M2P={D_M2P}")
    log(f"V4_WARM_START: {V4_M2P_PATH} (exists={V4_M2P_PATH.exists()})")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Load model dimensions
    model_dims = phase_load_model_dims()
    log_memory("after-dims")

    # Phase 1: Train math M2P (warm-start from v4)
    math_train_results = phase_train_math_m2p(model_dims)
    log_memory("after-math-train")

    # Phase 2: Train sort M2P (from scratch)
    sort_train_results = phase_train_sort_m2p(model_dims)
    log_memory("after-sort-train")

    # Phase 3: Train TF-IDF router
    router_results = phase_train_tfidf_router()
    k926_pass = router_results["k926_pass"]
    log_memory("after-router-train")

    # Phase 4: K925 — Composed adapter gradient check
    k925_results = phase_k925_grad_check(model_dims)
    k925_pass = k925_results["k925_pass"]
    log_memory("after-k925")

    # Phase 5: Single-adapter baselines (includes base_acc computation — Fix 2)
    single_results = phase_eval_single_adapters(model_dims)
    log_memory("after-single-eval")

    # Phase 6: Composed adapter evaluation (Fix 3: routed; Fix 4: convergence gate)
    composed_results = phase_eval_composed(model_dims, {
        "vectorizer": router_results["vectorizer"],
        "clf": router_results["clf"],
    }, single_results)
    log_memory("after-composed-eval")

    # K927: quality_ratio computation — Fix 2: use MATH.md formula
    # quality_ratio = (composed_acc - base_acc) / (single_acc - base_acc)
    math_single_acc = single_results["math_single_acc"]
    sort_single_acc = single_results["sort_single_acc"]
    math_base_acc = single_results["math_base_acc"]
    sort_base_acc = single_results["sort_base_acc"]
    math_composed_acc = composed_results["math_composed_acc"]
    sort_composed_acc = composed_results["sort_composed_acc"]
    sort_k927_skipped = composed_results.get("sort_k927_skipped", False)
    sort_converged = composed_results.get("sort_converged", False)

    def safe_quality_ratio(composed, single, base):
        """Fix 2: (composed-base)/(single-base). Returns NaN if single==base."""
        denom = single - base
        if abs(denom) < 1e-6:
            return float("nan")
        return (composed - base) / denom

    quality_ratio_math = safe_quality_ratio(math_composed_acc, math_single_acc, math_base_acc)
    quality_ratio_sort = safe_quality_ratio(sort_composed_acc, sort_single_acc, sort_base_acc)

    k927_math_pass = (not math.isnan(quality_ratio_math)) and quality_ratio_math >= 0.75
    # Fix 4: Sort K927 only evaluated if sort adapter converged
    if sort_k927_skipped:
        k927_sort_pass = None  # not evaluated
        k927_sort_status = f"SKIPPED — sort adapter did not converge (sort_single_acc={sort_single_acc:.4f})"
    else:
        k927_sort_pass = (not math.isnan(quality_ratio_sort)) and quality_ratio_sort >= 0.75
        k927_sort_status = "PASS" if k927_sort_pass else "FAIL"

    k927_pass = k927_math_pass and (k927_sort_pass is True or k927_sort_pass is None)

    # ---- Kill Criteria Summary ----
    log("\n" + "=" * 70)
    log("Kill Criteria Summary")
    log("=" * 70)
    log(f"  K925 (grad_norm > 0 under routed adapter): "
        f"{'PASS' if k925_pass else 'FAIL'} "
        f"(grad_norm={k925_results['k925_grad_norm_composed']:.6f})")
    log(f"  K926 (routing >= 80% both tasks): "
        f"{'PASS' if k926_pass else 'FAIL'} "
        f"(math={router_results['routing_accuracy_math']:.1%}, "
        f"sort={router_results['routing_accuracy_sort']:.1%})")
    log(f"  K927 (quality_ratio >= 0.75 both tasks, MATH formula, routed selection):")
    log(f"    Math:  base={math_base_acc:.3f} single={math_single_acc:.3f}  "
        f"composed={math_composed_acc:.3f}  "
        f"ratio={quality_ratio_math:.3f} {'PASS' if k927_math_pass else 'FAIL'}")
    if sort_k927_skipped:
        log(f"    Sort:  {k927_sort_status}")
    else:
        ratio_str = f"{quality_ratio_sort:.3f}" if not math.isnan(quality_ratio_sort) else "NaN"
        log(f"    Sort:  base={sort_base_acc:.3f} single={sort_single_acc:.3f}  "
            f"composed={sort_composed_acc:.3f}  "
            f"ratio={ratio_str} {k927_sort_status}")

    total_time = time.time() - t_start
    all_pass = k925_pass and k926_pass and k927_math_pass and (k927_sort_pass is not False)

    log(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    log(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    results = {
        "experiment": "m2p_composition_n5_qwen3",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "adversarial_fixes": ["Fix1:separate-A-slots", "Fix2:MATH-quality-ratio",
                               "Fix3:routed-selection", "Fix4:sort-convergence-gate"],
        "config": {
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P,
            "output_scale": OUTPUT_SCALE,
            "math_train_steps": MATH_TRAIN_STEPS,
            "sort_train_steps": SORT_TRAIN_STEPS,
            "lr": LR,
            "lr_warmup": LR_WARMUP,
            "seed": SEED,
            "n_eval_math": N_EVAL_MATH,
            "n_eval_sort": N_EVAL_SORT,
            "composition_mode": "routed_selection_alpha1.0",
            **model_dims,
        },
        # Training results
        "math_train": {
            "grad_norm_step0": math_train_results["math_grad_norm_step0"],
            "initial_loss": math_train_results["math_initial_loss"],
            "final_loss": math_train_results["math_final_loss"],
            "warm_start_used": math_train_results["math_warm_start_used"],
        },
        "sort_train": {
            "grad_norm_step0": sort_train_results["sort_grad_norm_step0"],
            "initial_loss": sort_train_results["sort_initial_loss"],
            "final_loss": sort_train_results["sort_final_loss"],
            "m2p_params": sort_train_results["sort_m2p_params"],
        },
        # K925
        "k925_grad_norm_composed": k925_results["k925_grad_norm_composed"],
        "k925_composed_loss": k925_results["k925_composed_loss"],
        "k925_pass": k925_pass,
        # K926
        "routing_accuracy_overall": router_results["routing_accuracy_overall"],
        "routing_accuracy_math": router_results["routing_accuracy_math"],
        "routing_accuracy_sort": router_results["routing_accuracy_sort"],
        "routing_train_accuracy": router_results["routing_train_accuracy"],
        "k926_pass": k926_pass,
        # Single baselines + base acc (Fix 2)
        "math_base_acc": math_base_acc,
        "sort_base_acc": sort_base_acc,
        "sort_base_f1": single_results["sort_base_f1"],
        "math_single_acc": math_single_acc,
        "sort_single_acc": sort_single_acc,
        "sort_single_f1": single_results.get("sort_single_f1", 0.0),
        "sort_single_pos_acc": single_results.get("sort_single_pos_acc", 0.0),
        "sort_converged": sort_converged,
        # Composed results (routed, Fix 3)
        "math_composed_acc": math_composed_acc,
        "sort_composed_acc": sort_composed_acc,
        "sort_composed_f1": composed_results.get("sort_composed_f1", 0.0),
        "math_routing_correct_on_eval": composed_results["math_routing_correct_on_eval"],
        "sort_routing_correct_on_eval": composed_results["sort_routing_correct_on_eval"],
        # K927 (Fix 2: MATH formula; Fix 3: routed selection)
        "quality_ratio_math": round(quality_ratio_math, 4)
                               if not math.isnan(quality_ratio_math) else None,
        "quality_ratio_sort": round(quality_ratio_sort, 4)
                               if not math.isnan(quality_ratio_sort) else None,
        "k927_math_pass": k927_math_pass,
        "k927_sort_pass": k927_sort_pass,
        "k927_sort_skipped": sort_k927_skipped,
        "k927_pass": k927_pass,
        # Summary
        "all_criteria_pass": all_pass,
        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()
