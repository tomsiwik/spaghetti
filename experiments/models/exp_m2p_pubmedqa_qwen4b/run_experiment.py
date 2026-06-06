#!/usr/bin/env python3
"""M2P v6: SFT-Residual M2P on Qwen3-4B + PubMedQA (Medical Domain).

Generalization test: same SFT-residual architecture as Finding #403 (math),
applied to PubMedQA biomedical yes/no/maybe question answering.

Kill criteria:
  K1137: init_quality_ratio >= 0.80 (predicted 1.00 by Theorem 1)
  K1138: quality_ratio >= 0.60 after 1000 M2P training steps
  K1139: base_accuracy < SFT_accuracy (domain weakness verified)

Architecture: identical to exp_m2p_qwen4b_sft_residual (M2PNetworkV6).
  B_applied[li] = B_sft_med[li] + output_scale * head(z[li])
At init, head output is zero → B_applied = B_sft_med → quality = SFT quality.

Key difference from math experiment:
  - Dataset: PubMedQA (yes/no/maybe biomedical QA) instead of GSM8K
  - A-matrices: new Grassmannian basis (seed=1, orthogonal to math seed=0)
  - SFT B-matrices: trained fresh on PubMedQA in Phase 1
  - Answer format: single token yes/no/maybe (MAX_GEN_TOKENS=16)

References:
  Finding #403 — SFT-residual M2P on GSM8K, quality_ratio=1.175 at 4B
  Finding #408 — A-matrix conflict (strong-base domains) → use B=0
  Finding #404 — Grassmannian isolation: |A^T A|=1.38e-05 across domains
  He et al. (2016) — Residual learning (ResNet)
  PubMedQA: Jin et al. (2019, arXiv:1909.06146)

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

# LoRA config
LORA_RANK = 4
LORA_SCALE = 5.0

# M2P v6 config (identical to Finding #403)
N_MEM_TOKENS = 16
D_M2P = 1024             # d_model=2560 != d_m2p → input_proj used
N_M2P_HEADS = 4
N_M2P_LAYERS = 4
OUTPUT_SCALE = 0.032     # SHINE sqrt(0.001) — scales the RESIDUAL only

# Grassmannian seed for MEDICAL domain (orthogonal to math seed=0)
MEDICAL_A_SEED = 1

# Training
N_TRAIN = 50 if IS_SMOKE else 500
N_TEST  = 10 if IS_SMOKE else 500
SFT_STEPS = 10 if IS_SMOKE else 300
M2P_TRAIN_STEPS = 20 if IS_SMOKE else 1000
LR = 5e-5
LR_WARMUP = 5 if IS_SMOKE else 100
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 64 if IS_SMOKE else 384   # contexts can be long — truncate
MAX_GEN_TOKENS = 16                      # yes/no/maybe only needs a few tokens
SEED = 42

# Paths
EXPERIMENT_DIR = Path(__file__).parent
MATH_A_DIR  = Path(__file__).parent.parent / "m2p_qwen4b_gsm8k"  # math A-matrices (required for Gram-Schmidt)
MEDICAL_A_PATH = EXPERIMENT_DIR / "medical_a_matrices.npz"
MEDICAL_SFT_B_PATH = EXPERIMENT_DIR / "medical_sft_b_matrices.npz"
M2P_PATH = EXPERIMENT_DIR / "m2p_weights.npz"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# PubMedQA few-shot prompt
FEW_SHOT_PREFIX = (
    "Answer the biomedical question with exactly one word: yes, no, or maybe.\n\n"
    "Abstract: Statins are prescribed to reduce cardiovascular risk. "
    "A meta-analysis of 27 RCTs (n=175,000) showed statin therapy reduced "
    "major cardiovascular events by 25% (RR=0.75, 95% CI 0.70-0.81, p<0.0001).\n"
    "Question: Do statins reduce cardiovascular events?\n"
    "Answer: yes\n\n"
    "Abstract: Zinc supplementation has been proposed to prevent the common cold. "
    "Three RCTs showed inconsistent results: two showed modest benefit, one showed "
    "no effect. Meta-analysis was not possible due to methodological heterogeneity.\n"
    "Question: Is zinc supplementation effective for cold prevention?\n"
    "Answer: maybe\n\n"
)


# ---- Utilities ---------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache  = mx.get_cache_memory()  / 1e9
    peak   = mx.get_peak_memory()   / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def extract_pubmedqa_answer(text: str):
    """Extract yes/no/maybe from model output."""
    text = text.strip().lower()
    # First word check
    for ans in ("yes", "no", "maybe"):
        if text.startswith(ans):
            return ans
    # Search in first 30 chars
    snippet = text[:30]
    for ans in ("yes", "no", "maybe"):
        if ans in snippet:
            return ans
    return None


# ---- Grassmannian A-matrix generation (Gram-Schmidt against math A) ----------

def generate_medical_a_matrices_gs(n_layers: int, rank: int) -> tuple:
    """Generate medical A-matrices via Gram-Schmidt orthogonalization against math A.

    Identical approach to Finding #404 (exp_m2p_2domain_compose_qwen4b):
      Q_med = random_normal - A_math @ (A_math^T @ random_normal)  # remove math components
      Q_med, _ = QR(Q_med)                                         # orthonormalize residual
    Result: A_math^T @ A_med = 0 exactly (up to fp64 precision → ~1e-15).
    After bf16 storage: max element ~ bf16 floor ≈ 1e-4.

    Returns:
        a_med_dict: {(li, mod): mx.array (d_model, rank)}
        a_math_raw: {(li, mod): np.array (d_model, rank)} for verification
    """
    math_a_path = MATH_A_DIR / "grassmannian_a_matrices.npz"
    if not math_a_path.exists():
        raise FileNotFoundError(
            f"Math A-matrices required for Gram-Schmidt: {math_a_path}\n"
            "Run m2p_qwen4b_gsm8k experiment first."
        )

    math_saved = np.load(str(math_a_path))
    a_med_dict = {}
    a_math_raw = {}
    max_cross   = 0.0

    for li in range(n_layers):
        for mod in ("q_proj", "v_proj"):
            A_math = math_saved[f"layer_{li}_{mod}_A"].astype(np.float64)  # (d_model, rank)
            d, r   = A_math.shape
            rng    = np.random.default_rng(MEDICAL_A_SEED * 10000 + li * 100 + (0 if mod == "q_proj" else 1))

            # Gram-Schmidt: project out math components
            Q = rng.standard_normal((d, r))
            Q -= A_math @ (A_math.T @ Q)   # Remove math A subspace
            Q, _ = np.linalg.qr(Q)         # Orthonormalize residual

            A_med = Q[:, :r].astype(np.float32)

            # Verify isolation in fp64
            cross = float(np.abs(A_math.T @ A_med.astype(np.float64)).max())
            max_cross = max(max_cross, cross)

            a_med_dict[(li, mod)] = mx.array(A_med).astype(mx.bfloat16)
            a_math_raw[(li, mod)] = A_math

    log(f"  Gram-Schmidt isolation (fp64): max|A_math^T A_med| = {max_cross:.2e} "
        f"({'PASS' if max_cross < 1e-4 else 'FAIL'} < 1e-4)")
    return a_med_dict, a_math_raw, max_cross


def load_or_generate_a_matrices(n_layers: int, rank: int) -> tuple:
    """Returns (a_med_dict, isolation_max_bf16)."""
    if MEDICAL_A_PATH.exists():
        log(f"  Loading medical A-matrices from {MEDICAL_A_PATH}")
        saved  = np.load(str(MEDICAL_A_PATH))
        result = {}
        for key in saved.files:
            # format: "layer_{li}_{mod}_A" (same as math convention)
            parts = key.split("_")
            li    = int(parts[1])
            mod   = "_".join(parts[2:-1])
            result[(li, mod)] = mx.array(saved[key]).astype(mx.bfloat16)
        log(f"  Loaded {len(result)} A-matrices")
        isolation_max = _check_isolation_fp32(n_layers)
        log(f"  fp32 isolation (loaded): max = {isolation_max:.2e} "
            f"(fp64 authoritative would be ~1.5e-5 if freshly generated)")
        return result, isolation_max
    else:
        log(f"  Generating medical A-matrices (Gram-Schmidt vs math, seed={MEDICAL_A_SEED})...")
        a_dict, _, isolation_fp64 = generate_medical_a_matrices_gs(n_layers, rank)
        # Save as float32 (precision sufficient for inference)
        save_dict = {f"layer_{li}_{mod}_A": np.array(v.astype(mx.float32))
                     for (li, mod), v in a_dict.items()}
        np.savez(str(MEDICAL_A_PATH), **save_dict)
        log(f"  Saved {len(a_dict)} medical A-matrices to {MEDICAL_A_PATH}")
        log(f"  fp64 isolation: max = {isolation_fp64:.2e} (authoritative, Finding #404 method)")
        return a_dict, isolation_fp64


def _check_isolation_fp32(n_layers: int) -> float:
    """Measure max|A_math^T A_med| element-wise in float32 from saved files.
    Reads float32 values directly (not through bf16 dict) to avoid quantization artifacts.
    Threshold 1e-4 matches Finding #404 (K975) — Gram-Schmidt gives ~1.5e-5 in fp32.
    """
    math_a_path = MATH_A_DIR / "grassmannian_a_matrices.npz"
    if not math_a_path.exists() or not MEDICAL_A_PATH.exists():
        return float("nan")
    math_saved = np.load(str(math_a_path))
    med_saved  = np.load(str(MEDICAL_A_PATH))
    max_cross  = 0.0
    for li in range(n_layers):
        for mod in ("q_proj", "v_proj"):
            A_math = math_saved[f"layer_{li}_{mod}_A"].astype(np.float32)
            A_med  = med_saved[f"layer_{li}_{mod}_A"].astype(np.float32)
            cross  = float(np.abs(A_math.T @ A_med).max())
            max_cross = max(max_cross, cross)
    return max_cross


# ---- Data loading ------------------------------------------------------------

def load_pubmedqa_data():
    """Load PubMedQA pqa_labeled (1000 examples) and split train/eval."""
    log("\n" + "=" * 70)
    log("[Data] Loading PubMedQA pqa_labeled")
    log("=" * 70)
    t0 = time.time()

    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)

    # pqa_labeled has only a 'train' split with 1000 examples
    all_examples = list(ds["train"])
    log(f"  Total PubMedQA pqa_labeled: {len(all_examples)} examples")

    rng = random.Random(SEED)
    rng.shuffle(all_examples)

    train_examples = all_examples[:N_TRAIN]
    test_examples  = all_examples[-(N_TEST):]  # hold-out from end (no overlap)

    # Distribution check
    for split_name, split_data in [("train", train_examples), ("test", test_examples)]:
        counts = {}
        for ex in split_data:
            label = ex.get("final_decision", "?")
            counts[label] = counts.get(label, 0) + 1
        log(f"  {split_name}: {len(split_data)} examples | {counts}")

    log(f"  Data loaded in {time.time()-t0:.1f}s")
    return train_examples, test_examples


def format_pubmedqa_prompt(ex: dict, include_answer: bool = False) -> str:
    """Format a PubMedQA example into a model prompt."""
    contexts = ex.get("context", {}).get("contexts", [])
    if not contexts:
        contexts = [ex.get("long_answer", "")]
    abstract = " ".join(contexts)
    # Truncate abstract to ~300 chars to fit in MAX_SEQ_LEN
    if len(abstract) > 400:
        abstract = abstract[:400] + "..."
    question = ex["question"]

    prompt = FEW_SHOT_PREFIX + f"Abstract: {abstract}\nQuestion: {question}\nAnswer:"
    if include_answer:
        prompt += f" {ex['final_decision']}"
    return prompt


def tokenize_train_examples(tokenizer, examples: list) -> list:
    """Tokenize training examples (prompt + answer) for SFT/M2P training."""
    result = []
    for ex in examples:
        text = format_pubmedqa_prompt(ex, include_answer=True)
        ids = tokenizer.encode(text)
        if len(ids) >= 2:
            ids = ids[:MAX_SEQ_LEN + 1]
            if len(ids) >= 2:
                result.append(ids)
    return result


# ---- Functional LoRA forward (identical to reference) -----------------------

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
    keys    = attn.k_norm(k.reshape(B_batch, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values  = v.reshape(B_batch, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
        keys    = attn.rope(keys,    offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = attn.rope(queries)
        keys    = attn.rope(keys)

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
    h    = qwen3_model.embed_tokens(tokens_arr)
    mask = create_attention_mask(h, None)

    for li, layer in enumerate(qwen3_model.layers):
        normed   = layer.input_layernorm(h)
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


def _apply_lora_structure(model, a_dict: dict) -> None:
    """Wrap attention proj layers with LoRALinear and inject A-matrices."""
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        if a_dict is not None:
            attn.q_proj.lora_a = a_dict[(li, "q_proj")]
            attn.v_proj.lora_a = a_dict[(li, "v_proj")]
    model.freeze()


# ---- Memory extraction (identical to reference) -----------------------------

def build_memory_causal_mask(M: int, T: int) -> mx.array:
    S = M + T
    neg_inf   = float("-inf")
    mask_np   = np.zeros((S, S), dtype=np.float32)
    mask_np[M:, :M] = neg_inf
    for i in range(T):
        for j in range(i + 1, T):
            mask_np[M + i, M + j] = neg_inf
    return mx.array(mask_np).astype(mx.bfloat16)[None, None, :, :]


def extract_memory_hidden_states(
    model, tokens_arr: mx.array, memory_embeddings: mx.array,
) -> mx.array:
    qwen3_model = model.model
    M = memory_embeddings.shape[0]
    B_batch, T = tokens_arr.shape

    tok_embs = qwen3_model.embed_tokens(tokens_arr)
    h = mx.concatenate([memory_embeddings[None, :, :], tok_embs], axis=1)
    mask = build_memory_causal_mask(M, T)
    memory_states = []

    for li, layer in enumerate(qwen3_model.layers):
        normed = layer.input_layernorm(h)
        attn   = layer.self_attn
        S      = M + T

        q_full = attn.q_proj(normed)
        k_full = attn.k_proj(normed)
        v_full = attn.v_proj(normed)

        queries = attn.q_norm(q_full.reshape(B_batch, S, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        keys    = attn.k_norm(k_full.reshape(B_batch, S, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values  = v_full.reshape(B_batch, S, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = attn.rope(queries)
        keys    = attn.rope(keys)

        attn_out = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attn.scale, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_batch, S, -1)
        attn_out = attn.o_proj(attn_out)

        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        memory_states.append(h[0, :M, :])

    return mx.stack(memory_states, axis=0)


# ---- M2PBlock (identical to reference) ----------------------------------------

class M2PBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, is_column: bool = True):
        super().__init__()
        self.is_column = is_column
        self.norm1 = nn.RMSNorm(d)
        self.attn  = nn.MultiHeadAttention(d, n_heads, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.mlp_fc1 = nn.Linear(d, 4 * d, bias=False)
        self.mlp_fc2 = nn.Linear(4 * d, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_column:
            x_t   = x.transpose(1, 0, 2)
            normed = self.norm1(x_t)
            x_t   = x_t + self.attn(normed, normed, normed)
            normed = self.norm2(x_t)
            x_t   = x_t + self.mlp_fc2(nn.gelu(self.mlp_fc1(normed)))
            return x_t.transpose(1, 0, 2)
        else:
            normed = self.norm1(x)
            x      = x + self.attn(normed, normed, normed)
            normed = self.norm2(x)
            return  x + self.mlp_fc2(nn.gelu(self.mlp_fc1(normed)))


# ---- M2PNetworkV6 (identical architecture to Finding #403) -------------------

class M2PNetworkV6(nn.Module):
    """SFT-residual M2P: B_applied = B_sft + output_scale * head(z).
    Zero-init heads → B_applied = B_sft at step 0 → init_quality_ratio = 1.0.
    """

    def __init__(
        self, n_layers: int, d_model: int, d_m2p: int, n_mem_tokens: int,
        rank: int, q_proj_out: int, v_proj_out: int,
        B_sft_q: list, B_sft_v: list,
        n_m2p_layers: int = 4, n_heads: int = 4, output_scale: float = 0.032,
    ):
        super().__init__()
        self.n_layers      = n_layers
        self.n_mem_tokens  = n_mem_tokens
        self.rank          = rank
        self.output_scale  = output_scale
        self.has_input_proj = (d_model != d_m2p)

        # Frozen SFT B-matrices (not in parameter graph)
        self.B_sft_q = B_sft_q
        self.B_sft_v = B_sft_v

        scale    = math.sqrt(1.0 / d_model)
        mem_init = np.random.uniform(-scale, scale, (n_mem_tokens, d_model)).astype(np.float32)
        self.memory_embeddings = mx.array(mem_init).astype(mx.bfloat16)

        self.input_proj = nn.Linear(d_model, d_m2p, bias=False) if self.has_input_proj else None
        self.p_layer = mx.zeros((n_layers, 1, d_m2p)).astype(mx.bfloat16)
        self.p_token = mx.zeros((1, n_mem_tokens, d_m2p)).astype(mx.bfloat16)

        self.blocks     = [M2PBlock(d=d_m2p, n_heads=n_heads, is_column=(i % 2 == 0))
                           for i in range(n_m2p_layers)]
        self.final_norm = nn.RMSNorm(d_m2p)

        # Per-layer RESIDUAL heads — ZERO init so B_applied = B_sft at step 0
        self.b_heads_q = [nn.Linear(d_m2p, rank * q_proj_out, bias=False) for _ in range(n_layers)]
        self.b_heads_v = [nn.Linear(d_m2p, rank * v_proj_out, bias=False) for _ in range(n_layers)]
        for head in self.b_heads_q + self.b_heads_v:
            head.weight = mx.zeros_like(head.weight)

    def __call__(self, memory_grid: mx.array):
        if self.has_input_proj:
            L, M, d  = memory_grid.shape
            flat      = memory_grid.reshape(L * M, d)
            projected = self.input_proj(flat.astype(mx.bfloat16))
            x         = projected.reshape(L, M, -1)
        else:
            x = memory_grid.astype(mx.bfloat16)

        x = x + self.p_layer.astype(mx.bfloat16)
        x = x + self.p_token.astype(mx.bfloat16)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        z = mx.mean(x, axis=1)  # (L, d_m2p)

        B_q_layers, B_v_layers = [], []
        for li in range(self.n_layers):
            z_li    = z[li]
            delta_q = self.b_heads_q[li](z_li).reshape(self.rank, -1) * self.output_scale
            delta_v = self.b_heads_v[li](z_li).reshape(self.rank, -1) * self.output_scale
            B_q_layers.append(self.B_sft_q[li] + delta_q.astype(self.B_sft_q[li].dtype))
            B_v_layers.append(self.B_sft_v[li] + delta_v.astype(self.B_sft_v[li].dtype))
        return B_q_layers, B_v_layers


# ---- BMatrices: trainable B-matrices for SFT --------------------------------

class BMatrices(nn.Module):
    """Trainable B-matrices for SFT phase (LoRA = A * B)."""
    def __init__(self, n_layers: int, rank: int, q_proj_out: int, v_proj_out: int):
        super().__init__()
        # Zero-init: ΔW = A*B = 0 at start → SFT starts from base model
        self.B_q = [mx.zeros((rank, q_proj_out)).astype(mx.bfloat16) for _ in range(n_layers)]
        self.B_v = [mx.zeros((rank, v_proj_out)).astype(mx.bfloat16) for _ in range(n_layers)]


# ---- Eval helpers -----------------------------------------------------------

def evaluate_base_on_pubmedqa(test_examples: list, model, tokenizer) -> dict:
    """Evaluate base model (no LoRA) on PubMedQA."""
    correct = 0
    total   = len(test_examples)
    for i, ex in enumerate(test_examples):
        prompt = format_pubmedqa_prompt(ex, include_answer=False)
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = ex["final_decision"].strip().lower()
        pred = extract_pubmedqa_answer(generated)
        if pred is not None and pred == gold:
            correct += 1
        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            log(f"    [{i+1}/{total}] base_acc={correct/(i+1):.3f}")
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_lora_on_pubmedqa(
    test_examples: list, model, tokenizer,
    B_q: list, B_v: list, A_q: list, A_v: list,
    label: str = "adapter", max_examples: int = None,
) -> dict:
    """Evaluate model with given B-matrices on PubMedQA."""
    examples = test_examples[:max_examples] if max_examples else test_examples
    correct  = 0
    total    = len(examples)
    for i, ex in enumerate(examples):
        prompt     = format_pubmedqa_prompt(ex, include_answer=False)
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        # Inject B-matrices into LoRA layers
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.lora_b = B_q[li]
            layer.self_attn.v_proj.lora_b = B_v[li]
        mx.eval(model.parameters())

        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = ex["final_decision"].strip().lower()
        pred = extract_pubmedqa_answer(generated)
        if pred is not None and pred == gold:
            correct += 1
        del tokens_arr
        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            log(f"    [{i+1}/{total}] {label}_acc={correct/(i+1):.3f}")

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_m2p_on_pubmedqa(
    test_examples: list, model, tokenizer, m2p, A_q: list, A_v: list,
    max_examples: int = None,
) -> dict:
    """Evaluate M2P-generated B-matrices on PubMedQA."""
    examples = test_examples[:max_examples] if max_examples else test_examples
    correct  = 0
    total    = len(examples)
    for i, ex in enumerate(examples):
        prompt     = format_pubmedqa_prompt(ex, include_answer=False)
        prompt_ids = tokenizer.encode(prompt)
        tokens_arr = mx.array(prompt_ids)[None, :]

        memory_grid = extract_memory_hidden_states(model, tokens_arr, m2p.memory_embeddings)
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
        gold = ex["final_decision"].strip().lower()
        pred = extract_pubmedqa_answer(generated)
        if pred is not None and pred == gold:
            correct += 1

        del tokens_arr, memory_grid, B_q_layers, B_v_layers
        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            log(f"    [{i+1}/{total}] m2p_acc={correct/(i+1):.3f}")

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ---- Phase 0: Inspect model dimensions ---------------------------------------

def phase_inspect_model() -> dict:
    log("\n" + "=" * 70)
    log("[Phase 0] Inspecting Qwen3-4B model dimensions")
    log("=" * 70)
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())

    attn  = model.model.layers[0].self_attn
    d_model   = model.args.hidden_size
    n_layers  = len(model.model.layers)
    n_heads   = attn.n_heads
    n_kv_heads = attn.n_kv_heads
    head_dim  = getattr(attn, 'head_dim', d_model // n_heads)
    q_proj_out = attn.q_proj.weight.shape[0]  # (out, in)
    v_proj_out = attn.v_proj.weight.shape[0]

    dims = {
        "n_layers": n_layers, "d_model": d_model,
        "n_heads": n_heads, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
        "q_proj_out": q_proj_out, "v_proj_out": v_proj_out,
    }
    log(f"  {dims}")
    cleanup(model, tokenizer)
    return dims


# ---- Phase 1: Base model evaluation ------------------------------------------

def phase_base_eval(test_examples: list, model_dims: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 1] Base model evaluation on PubMedQA")
    log("=" * 70)
    t0 = time.time()

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    model.freeze()

    eval_n = 20 if IS_SMOKE else 200  # quick base eval (full eval done in Phase 4)
    result = evaluate_base_on_pubmedqa(test_examples[:eval_n], model, tokenizer)
    log(f"  Base accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    log(f"  Phase 1 time: {time.time()-t0:.1f}s")
    log_memory("post-base-eval")
    cleanup(model, tokenizer)
    return result


# ---- Phase 2: SFT training ---------------------------------------------------

def phase_sft_train(train_examples: list, test_examples: list, model_dims: dict,
                    a_dict: dict) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 2] SFT training on PubMedQA (medical LoRA B-matrices)")
    log("=" * 70)
    t0 = time.time()

    if MEDICAL_SFT_B_PATH.exists() and not IS_SMOKE:
        log(f"  SFT B-matrices already exist at {MEDICAL_SFT_B_PATH} — skipping training")
        saved = np.load(str(MEDICAL_SFT_B_PATH))
        n_layers = model_dims["n_layers"]
        B_sft_q  = [mx.array(saved[f"layer_{li}_q_proj_B"]).astype(mx.bfloat16)
                    for li in range(n_layers)]
        B_sft_v  = [mx.array(saved[f"layer_{li}_v_proj_B"]).astype(mx.bfloat16)
                    for li in range(n_layers)]
        # Quick SFT eval on subset
        model, tokenizer = mlx_load(MODEL_ID)
        mx.eval(model.parameters())
        _apply_lora_structure(model, a_dict)
        mx.eval(model.parameters())
        A_q = [a_dict[(li, "q_proj")] for li in range(n_layers)]
        A_v = [a_dict[(li, "v_proj")] for li in range(n_layers)]
        sft_result = evaluate_lora_on_pubmedqa(
            test_examples, model, tokenizer, B_sft_q, B_sft_v, A_q, A_v,
            label="sft", max_examples=100,
        )
        log(f"  Cached SFT accuracy: {sft_result['accuracy']:.4f}")
        cleanup(model, tokenizer)
        return {"sft_accuracy": sft_result["accuracy"], "sft_B_q": B_sft_q, "sft_B_v": B_sft_v}

    n_layers   = model_dims["n_layers"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    _apply_lora_structure(model, a_dict)
    mx.eval(model.parameters())

    A_q = [a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v = [a_dict[(li, "v_proj")] for li in range(n_layers)]

    tokenized = tokenize_train_examples(tokenizer, train_examples)
    log(f"  Tokenized {len(tokenized)} examples")

    # Trainable B-matrices (zero-init: ΔW=0 at start)
    b_mats = BMatrices(n_layers, LORA_RANK, q_proj_out, v_proj_out)
    mx.eval(b_mats.parameters())

    rng       = random.Random(SEED + 10)
    optimizer = optim.Adam(learning_rate=LR)

    def sft_loss_fn(b_mats_inner, tokens_arr):
        logits = model_forward_with_loras(
            model, tokens_arr,
            b_mats_inner.B_q, b_mats_inner.B_v,
            A_q, A_v, LORA_SCALE,
        )
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(b_mats, sft_loss_fn)

    log(f"  SFT training: {SFT_STEPS} steps, LR={LR}")
    gc.disable()
    losses = []
    for step in range(SFT_STEPS):
        seq       = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]

        loss, grads = loss_and_grad(b_mats, tokens_arr)

        flat_g     = tree_flatten(grads)
        grad_norm  = math.sqrt(sum(float(mx.sum(g**2).item())
                                   for _, g in flat_g if isinstance(g, mx.array)))
        if grad_norm > GRAD_CLIP:
            factor = GRAD_CLIP / (grad_norm + 1e-8)
            grads  = tree_map(lambda g: g * factor if isinstance(g, mx.array) else g, grads)

        optimizer.update(b_mats, grads)
        del grads, tokens_arr
        mx.eval(b_mats.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))

        if (step + 1) % max(1, SFT_STEPS // 5) == 0:
            recent = sum(losses[-10:]) / min(len(losses[-10:]), 10)
            log(f"  SFT step {step+1}/{SFT_STEPS}: loss={recent:.4f}")
    gc.enable()
    gc.collect()

    # Save SFT B-matrices
    save_dict = {}
    for li in range(n_layers):
        save_dict[f"layer_{li}_q_proj_B"] = np.array(b_mats.B_q[li].astype(mx.float32))
        save_dict[f"layer_{li}_v_proj_B"] = np.array(b_mats.B_v[li].astype(mx.float32))
    np.savez(str(MEDICAL_SFT_B_PATH), **save_dict)
    log(f"  Saved SFT B-matrices to {MEDICAL_SFT_B_PATH}")

    # Quick SFT accuracy
    sft_result = evaluate_lora_on_pubmedqa(
        test_examples, model, tokenizer,
        list(b_mats.B_q), list(b_mats.B_v),
        A_q, A_v, label="sft",
        max_examples=50 if IS_SMOKE else 150,
    )
    B_sft_q = list(b_mats.B_q)
    B_sft_v = list(b_mats.B_v)

    log(f"  SFT accuracy: {sft_result['accuracy']:.4f} ({sft_result['correct']}/{sft_result['total']})")
    log(f"  Phase 2 time: {time.time()-t0:.1f}s")
    log_memory("post-sft-train")
    cleanup(b_mats, model, tokenizer, optimizer)
    return {"sft_accuracy": sft_result["accuracy"], "sft_B_q": B_sft_q, "sft_B_v": B_sft_v}


# ---- Phase 3: M2P training ---------------------------------------------------

def phase_m2p_train(
    train_examples: list, test_examples: list, model_dims: dict,
    a_dict: dict, B_sft_q: list, B_sft_v: list,
) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 3] M2P v6 Training (SFT-Residual, medical domain)")
    log("=" * 70)
    t0 = time.time()

    n_layers   = model_dims["n_layers"]
    d_model    = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    _apply_lora_structure(model, a_dict)
    mx.eval(model.parameters())

    A_q = [a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v = [a_dict[(li, "v_proj")] for li in range(n_layers)]

    tokenized = tokenize_train_examples(tokenizer, train_examples)
    log(f"  Tokenized {len(tokenized)} examples for M2P training")

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
    log(f"  OUTPUT_SCALE={OUTPUT_SCALE} | M2P_STEPS={M2P_TRAIN_STEPS}")

    rng = random.Random(SEED + 2)

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
            A_q, A_v, LORA_SCALE,
        )
        return nn.losses.cross_entropy(
            logits[0, :-1, :], tokens_arr[0, 1:], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(m2p, m2p_loss_fn)

    # ---- K1: Init quality (SFT residual at step 0) ----
    log("\n  [K1] Measuring init quality (zero-init heads → B_applied = B_sft)...")
    init_n   = 10 if IS_SMOKE else 100
    init_res = evaluate_m2p_on_pubmedqa(
        test_examples, model, tokenizer, m2p, A_q, A_v, max_examples=init_n,
    )
    init_accuracy = init_res["accuracy"]
    log(f"  [K1] Init M2P accuracy = {init_accuracy:.4f} ({init_res['correct']}/{init_res['total']})")

    # ---- Grad smoke test (K not a kill, just verification) ----
    log("\n  Gradient smoke test...")
    smoke_seq    = rng.choice(tokenized)
    smoke_tokens = mx.array(smoke_seq)[None, :]
    smoke_loss, smoke_grads = loss_and_grad(m2p, smoke_tokens)
    mx.eval(smoke_loss, smoke_grads)
    grad_norm_smoke = math.sqrt(sum(
        float(mx.sum(g**2).item())
        for _, g in tree_flatten(smoke_grads)
        if isinstance(g, mx.array)
    ))
    log(f"  Grad norm at step 0: {grad_norm_smoke:.6f} (smoke loss: {float(smoke_loss.item()):.4f})")
    del smoke_tokens, smoke_loss, smoke_grads

    # ---- Full M2P training ----
    log(f"\n  Training M2P for {M2P_TRAIN_STEPS} steps...")
    gc.disable()
    losses = []
    for step in range(M2P_TRAIN_STEPS):
        seq        = rng.choice(tokenized)
        tokens_arr = mx.array(seq)[None, :]
        optimizer.learning_rate = lr_schedule(step)

        loss, grads = loss_and_grad(m2p, tokens_arr)

        flat_g    = tree_flatten(grads)
        grad_norm = math.sqrt(sum(
            float(mx.sum(g**2).item()) for _, g in flat_g if isinstance(g, mx.array)
        ))
        if grad_norm > GRAD_CLIP:
            factor = GRAD_CLIP / (grad_norm + 1e-8)
            grads  = tree_map(lambda g: g * factor if isinstance(g, mx.array) else g, grads)

        optimizer.update(m2p, grads)
        del grads, tokens_arr
        mx.eval(m2p.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))

        if (step + 1) % max(1, M2P_TRAIN_STEPS // 10) == 0:
            recent = sum(losses[-20:]) / min(len(losses[-20:]), 20)
            log(f"  Step {step+1}/{M2P_TRAIN_STEPS}: loss={recent:.4f} grad_norm={grad_norm:.4f}")
    gc.enable()
    gc.collect()

    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    log(f"\n  Final M2P loss: {final_loss:.4f}")

    # Save M2P weights
    m2p_params_flat = dict(tree_flatten(m2p.parameters()))
    m2p_save = {k: np.array(v.astype(mx.float32)) for k, v in m2p_params_flat.items()}
    np.savez(str(M2P_PATH), **m2p_save)
    log(f"  Saved M2P to {M2P_PATH}")
    log(f"  Phase 3 time: {time.time()-t0:.1f}s")
    log_memory("post-m2p-train")
    cleanup(m2p, model, tokenizer, optimizer)

    return {
        "m2p_final_loss": float(final_loss),
        "m2p_params": n_params,
        "init_accuracy": init_accuracy,
        "grad_norm_step0": grad_norm_smoke,
    }


# ---- Phase 4: Evaluate M2P --------------------------------------------------

def phase_eval_m2p(test_examples: list, model_dims: dict, a_dict: dict,
                   B_sft_q: list, B_sft_v: list) -> dict:
    log("\n" + "=" * 70)
    log("[Phase 4] Evaluating M2P (medical domain) + Full base/SFT re-eval")
    log("=" * 70)
    t0 = time.time()

    n_layers   = model_dims["n_layers"]
    d_model    = model_dims["d_model"]
    q_proj_out = model_dims["q_proj_out"]
    v_proj_out = model_dims["v_proj_out"]

    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    _apply_lora_structure(model, a_dict)
    mx.eval(model.parameters())

    A_q = [a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v = [a_dict[(li, "v_proj")] for li in range(n_layers)]

    # Load saved M2P
    m2p = M2PNetworkV6(
        n_layers=n_layers, d_model=d_model, d_m2p=D_M2P,
        n_mem_tokens=N_MEM_TOKENS, rank=LORA_RANK,
        q_proj_out=q_proj_out, v_proj_out=v_proj_out,
        B_sft_q=B_sft_q, B_sft_v=B_sft_v,
        n_m2p_layers=N_M2P_LAYERS, n_heads=N_M2P_HEADS,
        output_scale=OUTPUT_SCALE,
    )
    m2p_saved = np.load(str(M2P_PATH))
    m2p.load_weights([(k, mx.array(m2p_saved[k])) for k in m2p_saved.files])
    m2p.eval()
    mx.eval(m2p.parameters())
    log(f"  Loaded M2P from {M2P_PATH}")

    # Full M2P evaluation (n=500 or N_TEST)
    eval_n = min(N_TEST, len(test_examples))
    log(f"\n  Evaluating M2P on {eval_n} PubMedQA examples...")
    m2p_res = evaluate_m2p_on_pubmedqa(
        test_examples, model, tokenizer, m2p, A_q, A_v, max_examples=eval_n,
    )

    # Also evaluate base and SFT B-matrices for fair comparison
    log(f"\n  Re-evaluating base model on {min(eval_n, 200)} examples...")
    model_frozen = model
    model_frozen.freeze()

    # For base eval, zero-out B-matrices
    zero_B_q = [mx.zeros_like(b) for b in B_sft_q]
    zero_B_v = [mx.zeros_like(b) for b in B_sft_v]
    base_res = evaluate_lora_on_pubmedqa(
        test_examples, model_frozen, tokenizer,
        zero_B_q, zero_B_v, A_q, A_v,
        label="base(zero_B)", max_examples=min(eval_n, 200),
    )

    log(f"\n  Re-evaluating SFT on {min(eval_n, 200)} examples...")
    sft_res = evaluate_lora_on_pubmedqa(
        test_examples, model_frozen, tokenizer,
        B_sft_q, B_sft_v, A_q, A_v,
        label="sft", max_examples=min(eval_n, 200),
    )

    log(f"\n  Base accuracy (zero B): {base_res['accuracy']:.4f}")
    log(f"  SFT  accuracy:          {sft_res['accuracy']:.4f}")
    log(f"  M2P  accuracy:          {m2p_res['accuracy']:.4f}")
    log(f"  Phase 4 time: {time.time()-t0:.1f}s")
    log_memory("post-eval")
    cleanup(m2p, model, tokenizer)

    return {
        "base_accuracy": base_res["accuracy"],
        "sft_accuracy":  sft_res["accuracy"],
        "m2p_accuracy":  m2p_res["accuracy"],
        "m2p_correct":   m2p_res["correct"],
        "m2p_n":         m2p_res["total"],
    }


# ---- Main -------------------------------------------------------------------

def main():
    t_start = time.time()
    log("=" * 70)
    log("M2P v6: SFT-Residual M2P on Qwen3-4B + PubMedQA (Medical Domain)")
    log("Generalization test — Finding #403 architecture, new domain")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"N_TRAIN={N_TRAIN} | N_TEST={N_TEST}")
    log(f"SFT_STEPS={SFT_STEPS} | M2P_STEPS={M2P_TRAIN_STEPS}")
    log(f"MEDICAL_A_SEED={MEDICAL_A_SEED} | OUTPUT_SCALE={OUTPUT_SCALE}")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Inspect model
    model_dims = phase_inspect_model()
    n_layers   = model_dims["n_layers"]
    d_model    = model_dims["d_model"]

    # Generate / load medical A-matrices (Gram-Schmidt vs math → exact isolation)
    a_dict, isolation_max = load_or_generate_a_matrices(n_layers, LORA_RANK)
    A_q = [a_dict[(li, "q_proj")] for li in range(n_layers)]
    A_v = [a_dict[(li, "v_proj")] for li in range(n_layers)]

    isolation_pass = (not math.isnan(isolation_max)) and (isolation_max < 1e-4)
    log(f"\n  [Isolation] max|A_math^T A_med| (fp32) = {isolation_max:.2e} "
        f"({'PASS' if isolation_pass else 'FAIL'} < 1e-4)")

    # Load data
    train_examples, test_examples = load_pubmedqa_data()

    # Phase 1: Base model eval
    base_phase1 = phase_base_eval(test_examples, model_dims)
    base_acc_quick = base_phase1["accuracy"]

    # Phase 2: SFT training
    sft_result = phase_sft_train(train_examples, test_examples, model_dims, a_dict)
    sft_acc    = sft_result["sft_accuracy"]
    B_sft_q    = sft_result["sft_B_q"]
    B_sft_v    = sft_result["sft_B_v"]

    # Phase 3: M2P training
    m2p_train  = phase_m2p_train(
        train_examples, test_examples, model_dims,
        a_dict, B_sft_q, B_sft_v,
    )
    init_accuracy = m2p_train["init_accuracy"]

    # Phase 4: Full evaluation
    eval_result = phase_eval_m2p(test_examples, model_dims, a_dict, B_sft_q, B_sft_v)

    # ---- Kill Criteria Assessment ----
    base_acc = eval_result["base_accuracy"]
    sft_acc_final = eval_result["sft_accuracy"]
    m2p_acc  = eval_result["m2p_accuracy"]

    sft_improvement = sft_acc_final - base_acc
    m2p_improvement = m2p_acc  - base_acc
    init_improvement = init_accuracy - base_acc

    quality_ratio      = m2p_improvement / sft_improvement if abs(sft_improvement) > 1e-9 else 0.0
    init_quality_ratio = init_improvement / sft_improvement if abs(sft_improvement) > 1e-9 else 0.0

    k1_pass       = init_quality_ratio >= 0.80
    k2_pass       = quality_ratio >= 0.60
    k3_pass       = base_acc < sft_acc_final  # domain weakness verified
    isolation_pass = (not math.isnan(isolation_max)) and (isolation_max < 1e-4)

    log("\n" + "=" * 70)
    log("Kill Criteria Assessment")
    log("=" * 70)
    log(f"  K1137 (init_qr >= 0.80):         {'PASS' if k1_pass else 'FAIL'} "
        f"(init_qr={init_quality_ratio:.4f}, init_acc={init_accuracy:.4f})")
    log(f"  K1138 (quality_ratio >= 0.60):   {'PASS' if k2_pass else 'FAIL'} "
        f"(qr={quality_ratio:.4f}, m2p={m2p_acc:.4f}, sft={sft_acc_final:.4f})")
    log(f"  K1139 (base < SFT):              {'PASS' if k3_pass else 'FAIL'} "
        f"(base={base_acc:.4f}, sft={sft_acc_final:.4f})")
    log(f"  Isolation (< 1e-4):  {'PASS' if isolation_pass else 'FAIL'} "
        f"(max={isolation_max:.2e})")

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "exp_m2p_pubmedqa_qwen4b",
        "model": MODEL_ID,
        "dataset": "qiaojin/PubMedQA/pqa_labeled",
        "is_smoke": IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
            "d_m2p": D_M2P, "n_mem_tokens": N_MEM_TOKENS,
            "n_m2p_layers": N_M2P_LAYERS, "n_m2p_heads": N_M2P_HEADS,
            "output_scale": OUTPUT_SCALE, "medical_a_seed": MEDICAL_A_SEED,
            "n_train": N_TRAIN, "n_test": N_TEST,
            "sft_steps": SFT_STEPS, "m2p_steps": M2P_TRAIN_STEPS,
            "lr": LR, "lr_warmup": LR_WARMUP, "grad_clip": GRAD_CLIP,
            "max_seq_len": MAX_SEQ_LEN, "max_gen_tokens": MAX_GEN_TOKENS,
            **model_dims,
        },
        "base_accuracy":       round(base_acc,          4),
        "sft_accuracy":        round(sft_acc_final,     4),
        "sft_improvement":     round(sft_improvement,   4),
        "init_accuracy":       round(init_accuracy,     4),
        "init_quality_ratio":  round(init_quality_ratio, 4),
        "m2p_accuracy":        round(m2p_acc,           4),
        "m2p_correct":         eval_result["m2p_correct"],
        "m2p_n":               eval_result["m2p_n"],
        "m2p_improvement":     round(m2p_improvement,   4),
        "quality_ratio":       round(quality_ratio,     4),
        "m2p_final_loss":      round(m2p_train["m2p_final_loss"], 4),
        "m2p_params":          m2p_train["m2p_params"],
        "grad_norm_step0":     round(m2p_train["grad_norm_step0"], 6),
        "grassmannian_isolation_max": round(isolation_max, 8) if isolation_max is not None else None,
        "kill_criteria": {
            "K1137_init_qr_ge_80pct":     "PASS" if k1_pass else "FAIL",
            "K1138_quality_ratio_ge_60pct": "PASS" if k2_pass else "FAIL",
            "K1139_base_lt_sft":          "PASS" if k3_pass else "FAIL",
            "base_accuracy":  round(base_acc,          4),
            "sft_accuracy":   round(sft_acc_final,     4),
            "m2p_accuracy":   round(m2p_acc,           4),
            "quality_ratio":  round(quality_ratio,     4),
            "init_quality_ratio": round(init_quality_ratio, 4),
        },
        "peak_memory_gb":  round(peak_gb, 2),
        "total_time_s":    round(time.time() - t_start, 1),
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']:.1f}s | Peak memory: {peak_gb:.2f}GB")


if __name__ == "__main__":
    main()
