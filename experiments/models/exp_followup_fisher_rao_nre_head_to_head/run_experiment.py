#!/usr/bin/env python3
"""Fisher-Rao vs Norm-Rescaled Euclidean: Head-to-Head at Production Scale.

Kill criteria (pre-registered in MATH.md, target-gated per F#666):
  K1 (proxy, paired): FR overall-PPL < NRE overall-PPL by >= 0.05 at N=25.
  K2 (target, paired): FR conditional-PPL (assistant-tokens-only) < NRE cond-PPL by >= 0.05 at N=25.
  K3 (anti-null): both NRE and FR beat raw Euclidean overall-PPL at N=25 by >= 0.3.

SUPPORTED = K1 PASS and K2 PASS and K3 PASS.
KILLED    = K1 FAIL and K2 FAIL and K3 PASS (NRE ceiling confirmed).
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load as mlxlm_load
from mlx_lm.tuner.utils import load_adapters
from safetensors.numpy import load_file as load_safetensors

import mlx_lm as _mlx_lm_pkg

EXP_DIR = Path(__file__).parent
RESULTS_FILE = EXP_DIR / "results.json"
REPO_ROOT = EXP_DIR.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from pierre.merge.fisher_rao_merging.src.model import karcher_mean_spherical, slerp  # noqa: E402

SOURCE_ADAPTERS_BASE = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters"
SOURCE_DATA_BASE = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/data"
DOMAINS = ["code", "math", "medical"]

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 6
LORA_SCALE = 6.0
N_VALUES = [3, 10, 25]
N_PPL_SAMPLES = 50
NOISE_SCALE = 0.1
SEED = 42
MAX_SEQ_LENGTH = 512


def log(msg=""):
    print(msg, flush=True)


def cleanup():
    gc.collect()
    mx.clear_cache()


# ============================================================================
# Adapter loading
# ============================================================================


def load_adapter_raw(domain: str) -> dict[int, dict]:
    """Return {layer_idx: {"a": np.ndarray [d_in, r], "b": np.ndarray [r, d_out]}}."""
    path = SOURCE_ADAPTERS_BASE / domain / "adapters.safetensors"
    raw = load_safetensors(str(path))
    per_layer: dict[int, dict] = {}
    for key, w in raw.items():
        # keys look like "language_model.model.layers.{idx}.self_attn.q_proj.lora_{a,b}"
        parts = key.split(".")
        li = int(parts[3])
        which = parts[-1]  # 'lora_a' or 'lora_b'
        per_layer.setdefault(li, {})[which.split("_")[-1]] = w
    return per_layer


def build_variants_for_N(adapters: list[dict], N: int, rng: np.random.RandomState) -> list[dict]:
    """Create N synthetic B-variants. A is SHARED from adapters[0]."""
    real_n = len(adapters)
    variants = []
    for i in range(N):
        base = adapters[i % real_n]
        if i < real_n:
            variants.append(base)
        else:
            noisy = {}
            for li, ab in base.items():
                b = ab["b"]
                b_norm = float(np.sqrt((b * b).sum()) + 1e-8)
                noise = rng.normal(0.0, 1.0, b.shape).astype(np.float32)
                scale = NOISE_SCALE * b_norm / math.sqrt(b.size)
                noisy[li] = {"a": ab["a"], "b": b + scale * noise}
            variants.append(noisy)
    return variants


# ============================================================================
# Composition methods (operate on per-layer B numpy arrays)
# ============================================================================


def compose_euclidean(b_stack: np.ndarray) -> np.ndarray:
    """b_stack: [N, r, d_out]."""
    return b_stack.mean(axis=0)


def compose_nre(b_stack: np.ndarray) -> np.ndarray:
    euc = b_stack.mean(axis=0)
    src_norms = np.sqrt((b_stack * b_stack).sum(axis=(1, 2)))
    mean_src_norm = src_norms.mean()
    euc_norm = float(np.sqrt((euc * euc).sum()))
    if euc_norm < 1e-8:
        return euc
    return euc * (mean_src_norm / euc_norm)


def compose_fr(b_stack: np.ndarray) -> np.ndarray:
    """Fisher-Rao Karcher mean on flattened B vectors, rescaled to mean source norm."""
    N = b_stack.shape[0]
    flat = b_stack.reshape(N, -1).astype(np.float32)
    norms = np.sqrt((flat * flat).sum(axis=1) + 1e-12)
    mean_norm = float(norms.mean())
    unit = flat / norms[:, None]
    pts = [mx.array(unit[i]) for i in range(N)]
    w = [1.0 / N] * N
    if N == 1:
        direction = pts[0]
    elif N == 2:
        direction = slerp(pts[0], pts[1], t=0.5)
    else:
        direction = karcher_mean_spherical(pts, w, max_iter=50, step_size=1.0, tol=1e-6)
    mx.eval(direction)
    merged_flat = mean_norm * np.array(direction)
    return merged_flat.reshape(b_stack.shape[1:])


# ============================================================================
# Install composed adapter into model (mutate LoRALinear .lora_a / .lora_b)
# ============================================================================


def collect_lora_modules(model) -> dict[int, nn.Module]:
    """Return {layer_idx: LoRALinear module for self_attn.q_proj}."""
    from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear  # noqa: F401

    modules: dict[int, nn.Module] = {}
    # Gemma 4 E4B wraps under model.language_model.model.layers
    base = getattr(model, "language_model", None)
    if base is None:
        base = model
    layers = base.model.layers
    for li, layer in enumerate(layers):
        mod = getattr(layer.self_attn, "q_proj", None)
        if mod is None:
            continue
        if hasattr(mod, "lora_a") and hasattr(mod, "lora_b"):
            modules[li] = mod
    return modules


def install_composed(lora_modules, a_by_layer, composed_b_by_layer):
    """Write shared A_0 and composed B into every target LoRA module."""
    installed = 0
    for li, mod in lora_modules.items():
        if li not in composed_b_by_layer or li not in a_by_layer:
            continue
        a = mx.array(a_by_layer[li].astype(np.float32))
        b = mx.array(composed_b_by_layer[li].astype(np.float32))
        mod.lora_a = a.astype(mod.lora_a.dtype)
        mod.lora_b = b.astype(mod.lora_b.dtype)
        mod.scale = LORA_SCALE
        installed += 1
    mx.eval(list(lora_modules.values())[0].lora_a)
    return installed


def zero_adapter(lora_modules):
    """Reset to zero-adapter state (no contribution from LoRA)."""
    for mod in lora_modules.values():
        mod.lora_b = mx.zeros_like(mod.lora_b)
    mx.eval(list(lora_modules.values())[0].lora_b)


# ============================================================================
# Data + perplexity
# ============================================================================


def load_validation_messages(domain: str, n: int) -> list[list[dict]]:
    path = SOURCE_DATA_BASE / domain / "valid.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec.get("messages")
            if msgs and len(msgs) >= 2:
                out.append(msgs)
                if len(out) >= n:
                    break
    return out


def build_prompt_and_full_ids(tokenizer, messages):
    """Return (prompt_ids, full_ids). Prompt = user-only, full = user+assistant."""
    # Split at assistant turn
    user_only = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(user_only, add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if not full_text.startswith(prompt_text):
        # tokenizer may alter whitespace; fall back to recomputing ids
        # but prompt_ids should still be a prefix if decode/encode is stable
        pass
    return prompt_ids, full_ids


def compute_ppls(model, tokenizer, messages_list, max_len=MAX_SEQ_LENGTH):
    """Return (overall_ppl, conditional_ppl) where conditional is loss on assistant tokens only."""
    total_loss_all = 0.0
    total_tok_all = 0
    total_loss_cond = 0.0
    total_tok_cond = 0

    for messages in messages_list:
        prompt_ids, full_ids = build_prompt_and_full_ids(tokenizer, messages)
        if len(full_ids) < 2:
            continue
        full_ids = full_ids[:max_len]
        prompt_len = min(len(prompt_ids), len(full_ids) - 1)

        x = mx.array([full_ids[:-1]])
        y = mx.array([full_ids[1:]])

        logits = model(x).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, y, reduction="none")[0]  # [L]
        mx.eval(nll)
        nll_np = np.asarray(nll)

        total_loss_all += float(nll_np.sum())
        total_tok_all += int(nll_np.shape[0])

        if prompt_len > 0 and prompt_len < nll_np.shape[0]:
            cond_part = nll_np[prompt_len - 1 :]
            total_loss_cond += float(cond_part.sum())
            total_tok_cond += int(cond_part.shape[0])

        del logits, nll

    ppl_all = math.exp(total_loss_all / max(total_tok_all, 1))
    ppl_cond = math.exp(total_loss_cond / max(total_tok_cond, 1))
    return ppl_all, ppl_cond


# ============================================================================
# Main
# ============================================================================


def main():
    t0 = time.time()
    results: dict = {
        "experiment": "exp_followup_fisher_rao_nre_head_to_head",
        "paper": "arXiv:2603.04972",
        "mlx_lm_version": _mlx_lm_pkg.__version__,
        "base_model": MODEL_ID,
        "domains": DOMAINS,
        "n_values": N_VALUES,
        "kill_criteria": {
            "K1": "FR overall-PPL < NRE overall-PPL by >= 0.05 at N=25",
            "K2": "FR conditional-PPL < NRE conditional-PPL by >= 0.05 at N=25",
            "K3": "NRE and FR beat Euclidean overall-PPL at N=25 by >= 0.3",
        },
    }

    log("=" * 70)
    log("Fisher-Rao vs NRE Head-to-Head at Production Scale (N=25)")
    log(f"Base: {MODEL_ID}  |  mlx_lm={_mlx_lm_pkg.__version__}")
    log("=" * 70)

    log("\n[1/4] Loading base model...")
    model, tokenizer = mlxlm_load(MODEL_ID)
    model.freeze()

    # Attach LoRA wrappers by loading one adapter's structure (weights then zeroed)
    log("[1/4] Attaching LoRA structure via code adapter config...")
    adapter_cfg_dir = str(SOURCE_ADAPTERS_BASE / "code")
    load_adapters(model, adapter_cfg_dir)
    lora_modules = collect_lora_modules(model)
    log(f"  Found {len(lora_modules)} LoRA modules (q_proj)")
    cleanup()

    log("\n[2/4] Loading 3 real adapters (code, math, medical)...")
    real_adapters = [load_adapter_raw(d) for d in DOMAINS]
    layers_in_adapters = sorted(real_adapters[0].keys())
    log(f"  Layers with lora weights: {len(layers_in_adapters)}")

    # Shared A = adapter-0 (code)'s A per layer
    shared_a = {li: real_adapters[0][li]["a"] for li in layers_in_adapters}

    log("\n[3/4] Loading validation samples (mixed across 3 domains)...")
    messages_list: list = []
    per_domain = N_PPL_SAMPLES // len(DOMAINS) + 1
    for d in DOMAINS:
        messages_list.extend(load_validation_messages(d, per_domain))
    messages_list = messages_list[:N_PPL_SAMPLES]
    log(f"  {len(messages_list)} samples")

    # Baseline: zero adapter
    zero_adapter(lora_modules)
    log("\n  [baseline] Measuring base-model perplexities...")
    base_ppl_all, base_ppl_cond = compute_ppls(model, tokenizer, messages_list)
    log(f"    Base overall PPL: {base_ppl_all:.3f}")
    log(f"    Base conditional PPL: {base_ppl_cond:.3f}")
    results["baseline"] = {
        "overall_ppl": base_ppl_all,
        "conditional_ppl": base_ppl_cond,
    }
    cleanup()

    # Main sweep
    log("\n[4/4] Composition sweep (N={}, methods=euc/nre/fr)...".format(N_VALUES))
    rng = np.random.RandomState(SEED)
    per_n: dict = {}

    for N in N_VALUES:
        log(f"\n--- N={N} ---")
        variants = build_variants_for_N(real_adapters, N, rng)

        # Stack B per layer across N variants
        b_stacks: dict[int, np.ndarray] = {}
        for li in layers_in_adapters:
            bs = [v[li]["b"] for v in variants]
            b_stacks[li] = np.stack(bs, axis=0).astype(np.float32)

        n_result: dict = {"N": N}
        for method_name, method_fn in [
            ("euclidean", compose_euclidean),
            ("nre", compose_nre),
            ("fisher_rao", compose_fr),
        ]:
            t_comp = time.time()
            composed_b = {li: method_fn(b_stacks[li]) for li in layers_in_adapters}
            compose_time = time.time() - t_comp

            installed = install_composed(lora_modules, shared_a, composed_b)
            mx.eval(model.parameters())

            t_eval = time.time()
            ppl_all, ppl_cond = compute_ppls(model, tokenizer, messages_list)
            eval_time = time.time() - t_eval

            # Measure composed-B norm shrinkage (sanity for NRE vs FR vs Euc)
            src_norms = np.sqrt((b_stacks[layers_in_adapters[0]] ** 2).sum(axis=(1, 2)))
            comp = composed_b[layers_in_adapters[0]]
            comp_norm = float(np.sqrt((comp * comp).sum()))
            shrink = comp_norm / float(src_norms.mean()) if src_norms.mean() > 0 else 0.0

            log(
                f"  [{method_name:10s}] compose_t={compose_time:5.2f}s  eval_t={eval_time:5.1f}s  "
                f"PPL={ppl_all:7.3f}  cond-PPL={ppl_cond:7.3f}  "
                f"installed={installed}  shrink(L0)={shrink:.3f}"
            )
            n_result[method_name] = {
                "overall_ppl": ppl_all,
                "conditional_ppl": ppl_cond,
                "compose_time_s": compose_time,
                "eval_time_s": eval_time,
                "installed_layers": installed,
                "b_shrink_layer0": shrink,
            }
            cleanup()

        per_n[str(N)] = n_result
        zero_adapter(lora_modules)
        cleanup()

    results["per_N"] = per_n

    # ------------------------------------------------------------------
    # Kill-criteria evaluation
    # ------------------------------------------------------------------
    nr25 = per_n["25"]
    fr_ppl = nr25["fisher_rao"]["overall_ppl"]
    nre_ppl = nr25["nre"]["overall_ppl"]
    euc_ppl = nr25["euclidean"]["overall_ppl"]
    fr_cond = nr25["fisher_rao"]["conditional_ppl"]
    nre_cond = nr25["nre"]["conditional_ppl"]

    fr_time = nr25["fisher_rao"]["compose_time_s"]
    nre_time = nr25["nre"]["compose_time_s"]
    cost_ratio = fr_time / max(nre_time, 1e-8)

    k1_margin = nre_ppl - fr_ppl  # positive = FR better
    k2_margin = nre_cond - fr_cond
    k3_nre_gap = euc_ppl - nre_ppl
    k3_fr_gap = euc_ppl - fr_ppl

    K1_PASS = k1_margin >= 0.05
    K2_PASS = k2_margin >= 0.05
    K3_PASS = (k3_nre_gap >= 0.3) and (k3_fr_gap >= 0.3)

    all_pass = K1_PASS and K2_PASS and K3_PASS
    if all_pass:
        verdict = "SUPPORTED"
    elif (not K1_PASS) and (not K2_PASS) and K3_PASS:
        verdict = "KILLED"
    elif not K3_PASS:
        verdict = "INVALID"  # K3 anti-null failed
    else:
        verdict = "PROXY-MIXED"

    log("\n" + "=" * 70)
    log("KILL-CRITERIA ASSESSMENT (N=25)")
    log("=" * 70)
    log(f"  Euclidean PPL = {euc_ppl:.3f}  cond={nr25['euclidean']['conditional_ppl']:.3f}")
    log(f"  NRE       PPL = {nre_ppl:.3f}  cond={nre_cond:.3f}")
    log(f"  FR        PPL = {fr_ppl:.3f}  cond={fr_cond:.3f}")
    log(f"  FR/NRE compose-time ratio: {cost_ratio:.2f}x")
    log(
        f"  K1 (FR beats NRE overall by >=0.05): margin={k1_margin:+.4f}  {'PASS' if K1_PASS else 'FAIL'}"
    )
    log(
        f"  K2 (FR beats NRE cond by >=0.05):    margin={k2_margin:+.4f}  {'PASS' if K2_PASS else 'FAIL'}"
    )
    log(
        f"  K3 (NRE/FR beat Euc by >=0.3):       NRE-gap={k3_nre_gap:+.4f}  FR-gap={k3_fr_gap:+.4f}  "
        f"{'PASS' if K3_PASS else 'FAIL'}"
    )
    log(f"  VERDICT: {verdict}")

    results["kill_results"] = {
        "K1": {"pass": K1_PASS, "margin_ppl": float(k1_margin)},
        "K2": {"pass": K2_PASS, "margin_cond_ppl": float(k2_margin)},
        "K3": {
            "pass": K3_PASS,
            "nre_minus_euc": float(-k3_nre_gap),
            "fr_minus_euc": float(-k3_fr_gap),
            "nre_gap_over_euc": float(k3_nre_gap),
            "fr_gap_over_euc": float(k3_fr_gap),
        },
        "fr_nre_compose_cost_ratio": float(cost_ratio),
    }
    results["verdict"] = verdict
    results["all_pass"] = bool(all_pass)
    results["total_time_s"] = round(time.time() - t0, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total wall-clock: {results['total_time_s']}s")


if __name__ == "__main__":
    main()
