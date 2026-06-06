#!/usr/bin/env python3
"""SFT n=500 Baseline Measurement — Statistical Closure for M2P-vs-SFT.

Kill criteria:
  K919: SFT accuracy at n=500 measured with Wilson 95% CI (unconditional measurement)
  K920: Two-proportion z-test M2P(28.6%, n=500) vs SFT(n=500) produces p-value
        (PASS if p<0.05, INCONCLUSIVE if p>=0.05)
  K921: quality_ratio CI lower bound recomputed with Fieller delta method
        (propagates SFT baseline uncertainty; expected to show CI_lower < 0.773)

This is a MEASUREMENT experiment — no training occurs.

Reuses (do NOT re-measure):
  - M2P accuracy: 28.6% at n=500 (from v4 results.json) — FIXED
  - Base accuracy: 20.0% (from v2 K909 PASS) — FIXED

Loads:
  - experiments/models/m2p_qwen06b_gsm8k_v2/lora_a_matrices.npz (fixed random A-matrices)
  - experiments/models/m2p_qwen06b_gsm8k_v2/sft_b_matrices.npz (trained B-matrices)

Statistical formulas from MATH.md:
  Wilson CI: [hat_p + z^2/(2n) ± z*sqrt(hat_p*(1-hat_p)/n + z^2/(4n^2))] / (1+z^2/n)
  Two-proportion z: z = (p1-p2) / sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
  Fieller delta: Var(ratio) ~ Var(M2P)/(SFT-base)^2 + Var(SFT)*(M2P-base)^2/(SFT-base)^4

References:
  Wilson (1927) — score interval for binomial proportion
  Newcombe (1998) — two-proportion z-test
  Casella & Berger (2002, §5.5.4) — delta method for ratio
  Cobbe et al. (arXiv:2110.14168) — GSM8K evaluation protocol

Supports SMOKE_TEST=1: N_TEST=10 (smoke) vs N_TEST=500 (full).
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
from mlx.utils import tree_flatten

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.tuner.lora import LoRALinear

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# ---- Config -------------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"

# LoRA config — MUST match v2 for correct adapter loading
LORA_RANK = 4
LORA_SCALE = 5.0

# Evaluation
N_TEST = 10 if IS_SMOKE else 500
MAX_GEN_TOKENS = 64 if IS_SMOKE else 384
MAX_SEQ_LEN = 512
SEED = 42  # same seed as v2/v3/v4 for identical test set

# Fixed results from prior experiments (DO NOT REMEASURE)
M2P_ACCURACY = 0.286    # v4 result, n=500 (143/500)
M2P_N = 500
BASE_ACCURACY = 0.200   # v2 K909 PASS (40/200)

# Paths
EXPERIMENT_DIR = Path(__file__).parent
V2_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v2"
V4_DIR = EXPERIMENT_DIR.parent / "m2p_qwen06b_gsm8k_v4"

V2_LORA_A_PATH = V2_DIR / "lora_a_matrices.npz"
V2_SFT_B_PATH = V2_DIR / "sft_b_matrices.npz"
V4_RESULTS_PATH = V4_DIR / "results.json"

RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Few-shot prefix — MUST be identical to v2/v3/v4 for fair comparison
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


# ---- Utilities ----------------------------------------------------------------

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
    """Extract final numeric answer from GSM8K #### format or fallback patterns.

    Identical to v2/v3/v4 for consistent evaluation.
    """
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


# ---- Statistical functions (from MATH.md) ------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson (1927) 95% score interval for k successes in n trials.

    Returns (lower, upper).
    Formula: [hat_p + z^2/(2n) ± z*sqrt(hat_p*(1-hat_p)/n + z^2/(4n^2))] / (1+z^2/n)
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple:
    """Two-proportion z-test (Newcombe 1998) for H0: p1 = p2.

    Returns (z_stat, p_value_two_tailed).
    Uses pooled proportion under H0.
    """
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se < 1e-12:
        return (0.0, 1.0)
    z_stat = (p1 - p2) / se
    # Two-tailed p-value from standard normal
    # Using complementary error function: p = 2 * (1 - Phi(|z|))
    # scipy not available in all envs — use math.erfc
    p_value = math.erfc(abs(z_stat) / math.sqrt(2))
    return (z_stat, p_value)


def fieller_quality_ratio_ci(
    m2p_acc: float, m2p_n: int,
    sft_acc: float, sft_n: int,
    base_acc: float,
    z: float = 1.96,
) -> tuple:
    """Fieller/delta method 95% CI for quality_ratio.

    quality_ratio = (m2p_acc - base_acc) / (sft_acc - base_acc)

    Both M2P and SFT sampling variances are propagated (unlike v4 which
    treated SFT as a known constant).

    Var(quality_ratio) ≈ Var(m2p_acc)/(sft_acc-base)^2
                       + Var(sft_acc)*(m2p_acc-base)^2/(sft_acc-base)^4

    From Casella & Berger (2002, §5.5.4) delta method.
    Returns (quality_ratio, ci_lower, ci_upper, se_total, term1, term2).
    """
    denom = sft_acc - base_acc
    if abs(denom) < 1e-9:
        return (float("nan"),) * 6
    numer = m2p_acc - base_acc
    quality_ratio = numer / denom

    var_m2p = m2p_acc * (1 - m2p_acc) / m2p_n
    var_sft = sft_acc * (1 - sft_acc) / sft_n

    term1 = var_m2p / (denom ** 2)
    term2 = var_sft * (numer ** 2) / (denom ** 4)
    var_total = term1 + term2
    se_total = math.sqrt(max(var_total, 0.0))

    ci_lower = quality_ratio - z * se_total
    ci_upper = quality_ratio + z * se_total
    return (quality_ratio, ci_lower, ci_upper, se_total, term1, term2)


def v4_quality_ratio_ci(
    m2p_acc: float, m2p_n: int,
    sft_acc: float,
    base_acc: float,
    z: float = 1.96,
) -> tuple:
    """v4 method: treats SFT as fixed constant (reproduces v4 CI for comparison).

    Returns (quality_ratio, ci_lower, ci_upper, se).
    """
    denom = sft_acc - base_acc
    if abs(denom) < 1e-9:
        return (float("nan"),) * 4
    numer = m2p_acc - base_acc
    quality_ratio = numer / denom
    var_m2p = m2p_acc * (1 - m2p_acc) / m2p_n
    se = math.sqrt(var_m2p) / abs(denom)
    ci_lower = quality_ratio - z * se
    ci_upper = quality_ratio + z * se
    return (quality_ratio, ci_lower, ci_upper, se)


# ---- Phase 0: Validate prerequisites -----------------------------------------

def phase_validate_prerequisites() -> dict:
    """Check that all required files exist and load fixed M2P result from v4."""
    log("\n" + "=" * 70)
    log("[Phase 0] Validating prerequisites")
    log("=" * 70)

    for path, desc in [
        (V2_LORA_A_PATH, "v2 lora_a matrices"),
        (V2_SFT_B_PATH, "v2 SFT B-matrices"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path} ({desc}). "
                f"Run m2p_qwen06b_gsm8k_v2 first."
            )
        log(f"  FOUND: {path}")

    # Load fixed M2P result from v4
    m2p_accuracy = M2P_ACCURACY
    m2p_n = M2P_N
    base_accuracy = BASE_ACCURACY

    if V4_RESULTS_PATH.exists():
        with open(V4_RESULTS_PATH) as f:
            v4 = json.load(f)
        m2p_accuracy = v4.get("m2p_accuracy", M2P_ACCURACY)
        m2p_n = v4.get("m2p_total", M2P_N)
        log(f"  Loaded v4 results: M2P accuracy={m2p_accuracy:.4f}, n={m2p_n}")
    else:
        log(f"  WARNING: v4 results not found at {V4_RESULTS_PATH}")
        log(f"  Using hardcoded values: M2P={m2p_accuracy:.4f}, n={m2p_n}")

    log(f"  Fixed values: M2P={m2p_accuracy:.4f} (n={m2p_n}), base={base_accuracy:.4f}")
    return {
        "m2p_accuracy": m2p_accuracy,
        "m2p_n": m2p_n,
        "base_accuracy": base_accuracy,
    }


# ---- Phase 1: Load data -------------------------------------------------------

def phase_load_data() -> list:
    """Load GSM8K test examples (same SEED as v2/v3/v4 for identical test set)."""
    log("\n" + "=" * 70)
    log("[Phase 1] Loading GSM8K test data")
    log("=" * 70)
    t0 = time.time()

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")

    rng = random.Random(SEED)
    test_examples = list(ds["test"])
    rng.shuffle(test_examples)
    test_examples = test_examples[:N_TEST]

    log(f"  Test: {len(test_examples)} examples (SEED={SEED})")
    log(f"  Data loaded in {time.time()-t0:.1f}s")
    return test_examples


# ---- Phase 2: Load LoRA matrices from v2 -------------------------------------

def load_lora_a_matrices_v2() -> dict:
    """Load v2 lora_a matrices. Returns dict[(li, mod_name)] -> mx.array.

    These are the SAME A-matrices used during v2 SFT training. Using identical
    A-matrices is critical: the SFT B-matrices were trained with these specific A's.
    """
    saved = np.load(str(V2_LORA_A_PATH))
    result = {}
    for key in saved.files:
        assert key.endswith("_A"), f"Unexpected key: {key}"
        body = key[:-2]
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} lora_a matrices from v2")
    return result


def load_sft_b_matrices_v2() -> dict:
    """Load v2 trained SFT B-matrices. Returns dict[(li, mod_name)] -> mx.array."""
    saved = np.load(str(V2_SFT_B_PATH))
    result = {}
    for key in saved.files:
        assert key.endswith("_B"), f"Unexpected key: {key}"
        body = key[:-2]
        parts = body.split("_", 2)
        li = int(parts[1])
        mod_name = parts[2]
        result[(li, mod_name)] = mx.array(saved[key]).astype(mx.bfloat16)
    log(f"  Loaded {len(result)} sft_b matrices from v2")
    return result


# ---- Phase 3: Apply LoRA and evaluate SFT ------------------------------------

def apply_lora_to_model(model, lora_a_dict: dict) -> None:
    """Apply LoRALinear wrappers to q_proj and v_proj in all layers.

    Wraps each projection with LoRALinear.from_base and injects the saved
    A-matrices from v2 SFT training.

    LORA_RANK=4, LORA_SCALE=5.0 must match v2 exactly.
    """
    model.freeze()
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = LoRALinear.from_base(attn.q_proj, r=LORA_RANK, scale=LORA_SCALE)
        attn.v_proj = LoRALinear.from_base(attn.v_proj, r=LORA_RANK, scale=LORA_SCALE)
        # Inject saved A-matrices (critical: must match what SFT B-matrices were trained with)
        attn.q_proj.lora_a = lora_a_dict[(li, "q_proj")]
        attn.v_proj.lora_a = lora_a_dict[(li, "v_proj")]
    # Unfreeze lora_b for injection (though we won't train)
    model.unfreeze(keys=["lora_b"])


def set_lora_b_matrices(model, b_by_key: dict) -> None:
    """Inject B-matrices into LoRA-wrapped model."""
    for li, layer in enumerate(model.model.layers):
        for mod_name in ["q_proj", "v_proj"]:
            key = (li, mod_name)
            if key in b_by_key:
                lora_module = getattr(layer.self_attn, mod_name)
                lora_module.lora_b = b_by_key[key]


def phase_eval_sft(test_examples: list) -> dict:
    """Load SFT adapter from v2 and evaluate on n=500 GSM8K test examples.

    Evaluation is identical to v2's Phase 5 (phase_eval_sft):
    - Same few-shot prefix
    - Same max_gen_tokens
    - Same answer extraction
    - Same LoRA config (rank=4, scale=5.0)

    This is a pure measurement — no training.
    """
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating SFT adapter (v2 weights) on n=500 test examples")
    log("=" * 70)
    t0 = time.time()

    # Load model
    model, tokenizer = mlx_load(MODEL_ID)
    mx.eval(model.parameters())
    log_memory("post-model-load")

    # Load v2 LoRA weights
    lora_a_dict = load_lora_a_matrices_v2()
    sft_b_dict = load_sft_b_matrices_v2()

    # Apply LoRA structure + inject saved A-matrices
    apply_lora_to_model(model, lora_a_dict)
    # Inject trained B-matrices
    set_lora_b_matrices(model, sft_b_dict)
    mx.eval(model.parameters())
    log(f"  LoRA applied: rank={LORA_RANK}, scale={LORA_SCALE}")
    log_memory("post-lora-inject")

    # Evaluate
    correct = 0
    total = len(test_examples)
    log(f"  Evaluating {total} examples...")
    for i, ex in enumerate(test_examples):
        prompt = FEW_SHOT_PREFIX + f"Question: {ex['question']}\nAnswer:"
        generated = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_GEN_TOKENS, verbose=False,
        )
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generated)
        if pred is not None and gold is not None and pred == gold:
            correct += 1

        # Progress logging
        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            log(f"  [SFT] {i+1}/{total}: acc={correct/(i+1):.3f} ({correct}/{i+1})")

        # Debug first example
        if i == 0:
            log(f"  [DEBUG] Prompt (first 120 chars): {prompt[:120]!r}")
            log(f"  [DEBUG] Generated (first 200 chars): {generated[:200]!r}")
            log(f"  [DEBUG] Gold: {gold!r}, Pred: {pred!r}")

        # Periodic cache cleanup (every 100 examples)
        if (i + 1) % 100 == 0:
            gc.collect()
            mx.clear_cache()

    accuracy = correct / total if total > 0 else 0.0
    elapsed = time.time() - t0
    log(f"  SFT accuracy: {accuracy:.4f} ({correct}/{total})")
    log(f"  Phase 3 time: {elapsed:.1f}s")
    log_memory("post-sft-eval")

    cleanup(model, tokenizer, lora_a_dict, sft_b_dict)

    return {
        "sft_accuracy": accuracy,
        "sft_correct": correct,
        "sft_total": total,
        "eval_time_s": round(elapsed, 1),
    }


# ---- Phase 4: Statistical analysis -------------------------------------------

def phase_statistical_analysis(sft_results: dict, fixed: dict) -> dict:
    """Compute all statistical quantities from MATH.md.

    Inputs:
      sft_results: from phase_eval_sft (sft_accuracy, sft_correct, sft_total)
      fixed: from phase_validate_prerequisites (m2p_accuracy, m2p_n, base_accuracy)

    Computes:
      1. Wilson 95% CI for SFT accuracy (K919)
      2. Two-proportion z-test M2P vs SFT (K920)
      3. Fieller/delta quality_ratio CI (K921)
      4. v4-method quality_ratio CI (for comparison, shows bias)
    """
    log("\n" + "=" * 70)
    log("[Phase 4] Statistical Analysis")
    log("=" * 70)

    sft_acc = sft_results["sft_accuracy"]
    sft_correct = sft_results["sft_correct"]
    sft_total = sft_results["sft_total"]
    m2p_acc = fixed["m2p_accuracy"]
    m2p_n = fixed["m2p_n"]
    base_acc = fixed["base_accuracy"]
    m2p_correct = round(m2p_acc * m2p_n)  # 143 for 28.6% * 500

    # ---- 1. Wilson CI for SFT (K919) ----
    log("\n--- K919: Wilson CI for SFT accuracy ---")
    sft_wilson_lo, sft_wilson_hi = wilson_ci(sft_correct, sft_total)
    log(f"  SFT accuracy:  {sft_acc:.4f} ({sft_correct}/{sft_total})")
    log(f"  Wilson 95% CI: [{sft_wilson_lo:.4f}, {sft_wilson_hi:.4f}]")

    # Wilson CI for M2P (for reference)
    m2p_wilson_lo, m2p_wilson_hi = wilson_ci(m2p_correct, m2p_n)
    log(f"  M2P accuracy:  {m2p_acc:.4f} ({m2p_correct}/{m2p_n}) [fixed from v4]")
    log(f"  M2P Wilson CI: [{m2p_wilson_lo:.4f}, {m2p_wilson_hi:.4f}]")

    k919_pass = True  # Unconditional: measurement was made

    # ---- 2. Two-proportion z-test (K920) ----
    log("\n--- K920: Two-proportion z-test M2P vs SFT ---")
    z_stat, p_value = two_proportion_ztest(m2p_correct, m2p_n, sft_correct, sft_total)
    log(f"  M2P: {m2p_acc:.4f} (n={m2p_n}), SFT: {sft_acc:.4f} (n={sft_total})")
    log(f"  z-stat: {z_stat:.4f}")
    log(f"  p-value (two-tailed): {p_value:.4f}")
    if p_value < 0.05:
        log("  K920: PASS — gap is statistically significant (p < 0.05)")
        k920_pass = True
        k920_label = "significant"
    else:
        log("  K920: INCONCLUSIVE — gap is NOT significant (p >= 0.05)")
        k920_pass = False
        k920_label = "not_significant"

    # ---- 3. Fieller quality_ratio CI (K921) ----
    log("\n--- K921: Fieller/delta quality_ratio CI ---")
    q_ratio, q_ci_lo, q_ci_hi, q_se, term1, term2 = fieller_quality_ratio_ci(
        m2p_acc=m2p_acc, m2p_n=m2p_n,
        sft_acc=sft_acc, sft_n=sft_total,
        base_acc=base_acc,
    )
    log(f"  quality_ratio = ({m2p_acc:.4f} - {base_acc:.4f}) / ({sft_acc:.4f} - {base_acc:.4f})")
    log(f"               = {m2p_acc - base_acc:.4f} / {sft_acc - base_acc:.4f} = {q_ratio:.4f}")
    log(f"  Variance Term1 (M2P noise): {term1:.6f}")
    log(f"  Variance Term2 (SFT noise): {term2:.6f}")
    log(f"  Total se: {q_se:.4f}")
    log(f"  Fieller 95% CI: [{q_ci_lo:.4f}, {q_ci_hi:.4f}]")
    log(f"  K921: CI computed (K921 is unconditional: measurement made)")

    # Compare with v4 method (SFT treated as fixed)
    log("\n--- Comparison: v4 method (SFT as fixed constant) ---")
    q_v4, q_ci_lo_v4, q_ci_hi_v4, q_se_v4 = v4_quality_ratio_ci(
        m2p_acc=m2p_acc, m2p_n=m2p_n,
        sft_acc=sft_acc,
        base_acc=base_acc,
    )
    log(f"  v4 quality_ratio: {q_v4:.4f} (same ratio, different CI)")
    log(f"  v4 se (M2P only): {q_se_v4:.4f}")
    log(f"  v4 95% CI:        [{q_ci_lo_v4:.4f}, {q_ci_hi_v4:.4f}]")
    log(f"  v4 CI_lower was:  0.7732 (with SFT=0.260 fixed, same as reported)")
    log(f"  Bias estimate:    {q_ci_lo_v4 - q_ci_lo:.4f} (v4 CI_lower minus Fieller CI_lower)")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  K919: SFT measured at n=500 — PASS (SFT={sft_acc:.4f}, CI=[{sft_wilson_lo:.4f},{sft_wilson_hi:.4f}])")
    log(f"  K920: z={z_stat:.4f}, p={p_value:.4f} — {k920_label.upper()}")
    log(f"  K921: Fieller CI_lower={q_ci_lo:.4f} (vs v4 optimistic CI_lower≈0.773)")

    return {
        # K919: Wilson CI for SFT
        "sft_wilson_ci_lower": round(sft_wilson_lo, 6),
        "sft_wilson_ci_upper": round(sft_wilson_hi, 6),
        "k919_sft_measured": True,
        # K920: Two-proportion z-test
        "z_stat_m2p_vs_sft": round(z_stat, 6),
        "p_value_m2p_vs_sft": round(p_value, 6),
        "k920_pass": k920_pass,
        "k920_label": k920_label,
        # K921: Fieller quality_ratio CI
        "quality_ratio": round(q_ratio, 6),
        "quality_ratio_ci_lower": round(q_ci_lo, 6),
        "quality_ratio_ci_upper": round(q_ci_hi, 6),
        "quality_ratio_se_fieller": round(q_se, 6),
        "quality_ratio_term1_m2p_variance": round(term1, 6),
        "quality_ratio_term2_sft_variance": round(term2, 6),
        "k921_ci_computed": True,
        # v4 method for comparison
        "quality_ratio_ci_lower_v4_method": round(q_ci_lo_v4, 6),
        "quality_ratio_ci_upper_v4_method": round(q_ci_hi_v4, 6),
        "quality_ratio_ci_lower_v4_reported": 0.7732,
        "bias_ci_lower": round(q_ci_lo_v4 - q_ci_lo, 6),
        # M2P Wilson CI
        "m2p_wilson_ci_lower": round(m2p_wilson_lo, 6),
        "m2p_wilson_ci_upper": round(m2p_wilson_hi, 6),
    }


# ---- Main --------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=" * 70)
    log("SFT n=500 Baseline Measurement — Statistical Closure for M2P-vs-SFT")
    log(f"SMOKE_TEST={IS_SMOKE}, N_TEST={N_TEST}")
    log("=" * 70)
    log_memory("start")

    # Phase 0: Validate prerequisites and load fixed M2P result
    fixed = phase_validate_prerequisites()
    mx.reset_peak_memory()

    # Phase 1: Load test data
    test_examples = phase_load_data()

    # Phase 3: Evaluate SFT adapter from v2 on n=500 test examples
    sft_results = phase_eval_sft(test_examples)
    peak_after_eval = mx.get_peak_memory() / 1e9
    mx.reset_peak_memory()

    # Phase 4: Statistical analysis
    stats = phase_statistical_analysis(sft_results, fixed)

    total_time = round(time.time() - t0, 1)
    log(f"\n  Total time: {total_time}s")
    log_memory("end")

    # Assemble results
    results = {
        "experiment": "m2p_sft_n500_baseline",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "config": {
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "n_test": N_TEST,
            "seed": SEED,
            "max_gen_tokens": MAX_GEN_TOKENS,
        },
        # SFT measurement (K919)
        "sft_accuracy": sft_results["sft_accuracy"],
        "sft_correct": sft_results["sft_correct"],
        "sft_total": sft_results["sft_total"],
        "sft_wilson_ci_lower": stats["sft_wilson_ci_lower"],
        "sft_wilson_ci_upper": stats["sft_wilson_ci_upper"],
        # Fixed M2P result (from v4, not re-measured)
        "m2p_accuracy": fixed["m2p_accuracy"],
        "m2p_n": fixed["m2p_n"],
        "m2p_wilson_ci_lower": stats["m2p_wilson_ci_lower"],
        "m2p_wilson_ci_upper": stats["m2p_wilson_ci_upper"],
        # Fixed base result (from v2, not re-measured)
        "base_accuracy": fixed["base_accuracy"],
        # Quality ratio — Fieller method (K921)
        "quality_ratio": stats["quality_ratio"],
        "quality_ratio_ci_lower": stats["quality_ratio_ci_lower"],
        "quality_ratio_ci_upper": stats["quality_ratio_ci_upper"],
        "quality_ratio_se_fieller": stats["quality_ratio_se_fieller"],
        "quality_ratio_term1_m2p_variance": stats["quality_ratio_term1_m2p_variance"],
        "quality_ratio_term2_sft_variance": stats["quality_ratio_term2_sft_variance"],
        # v4 method CI for bias comparison
        "quality_ratio_ci_lower_v4_method": stats["quality_ratio_ci_lower_v4_method"],
        "quality_ratio_ci_upper_v4_method": stats["quality_ratio_ci_upper_v4_method"],
        "quality_ratio_ci_lower_v4_reported": stats["quality_ratio_ci_lower_v4_reported"],
        "bias_ci_lower_v4_vs_fieller": stats["bias_ci_lower"],
        # Two-proportion z-test (K920)
        "z_stat_m2p_vs_sft": stats["z_stat_m2p_vs_sft"],
        "p_value_m2p_vs_sft": stats["p_value_m2p_vs_sft"],
        # Kill criteria status
        "k919_sft_measured": stats["k919_sft_measured"],
        "k920_pass": stats["k920_pass"],
        "k920_label": stats["k920_label"],
        "k921_ci_computed": stats["k921_ci_computed"],
        # Runtime
        "peak_memory_gb": round(peak_after_eval, 2),
        "total_time_s": total_time,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")

    # Final kill criteria summary
    log("\n" + "=" * 70)
    log("KILL CRITERIA FINAL ASSESSMENT")
    log("=" * 70)
    log(f"  K919 (SFT n=500 measured): {'PASS' if results['k919_sft_measured'] else 'FAIL'}")
    log(f"         SFT={results['sft_accuracy']:.4f}, Wilson CI=[{results['sft_wilson_ci_lower']:.4f}, {results['sft_wilson_ci_upper']:.4f}]")
    log(f"  K920 (two-prop z-test M2P vs SFT): {'PASS' if results['k920_pass'] else 'INCONCLUSIVE'}")
    log(f"         z={results['z_stat_m2p_vs_sft']:.4f}, p={results['p_value_m2p_vs_sft']:.4f}")
    log(f"  K921 (Fieller CI computed): {'PASS' if results['k921_ci_computed'] else 'FAIL'}")
    log(f"         quality_ratio CI_lower={results['quality_ratio_ci_lower']:.4f}")
    log(f"         (v4 reported optimistic CI_lower=0.773; bias={results['bias_ci_lower_v4_vs_fieller']:.4f})")
    log("=" * 70)

    return results


if __name__ == "__main__":
    main()
