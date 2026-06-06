#!/usr/bin/env python3
"""exp_g4_zs_base_transfer_4bit_fp16

Zero-shot precision transfer: 4-bit trained adapter → 8-bit base.
Measures adapter benefit retention under precision change.

Phased execution:
  Phase 1: Load 4-bit model → eval base PPL + adapter PPL per domain → clear
  Phase 2: Load 8-bit model → eval base PPL + adapter PPL per domain → clear
  Phase 3: Compute transfer ratios, K1/K2, write results.json
"""
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters, remove_lora_layers

# --- Config ---
MODEL_4BIT = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_8BIT = "mlx-community/gemma-4-e4b-it-8bit"
ADAPTERS_DIR = Path("experiments/models/exp_p1_t2_single_domain_training/adapters")
DATA_DIR = Path("experiments/models/exp_p1_t2_single_domain_training/data")
DOMAINS = ["code", "math", "medical"]
MAX_SEQ_LEN = 512
SMOKE = os.environ.get("SMOKE_TEST", "").strip() in ("1", "true", "yes")
N_EVAL = 10 if SMOKE else None

EXP_DIR = Path("experiments/models/exp_g4_zs_base_transfer_4bit_fp16")


def load_data(domain):
    path = DATA_DIR / domain / "valid.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if N_EVAL:
        samples = samples[:N_EVAL]
    return samples


def _build_prompt_messages(messages):
    """Messages with last assistant content cleared — gives prompt boundary."""
    return messages[:-1] + [{"role": messages[-1]["role"], "content": ""}]


def _tokenize_with_mask(messages, tokenizer):
    """Return (token_ids, prompt_len) where prompt_len marks assistant start."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_msgs = _build_prompt_messages(messages)
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False)
    full_ids = tokenizer.encode(full_text)[: MAX_SEQ_LEN + 1]
    prompt_ids = tokenizer.encode(prompt_text)
    return full_ids, min(len(prompt_ids), len(full_ids))


def per_sample_ppl(model, tokenizer, messages_list):
    """Per-sample PPL on assistant-only tokens."""
    ppls = []
    for messages in messages_list:
        tokens, prompt_len = _tokenize_with_mask(messages, tokenizer)
        if len(tokens) < prompt_len + 2:
            ppls.append(float("inf"))
            continue

        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        logits = model(x)
        per_tok_ce = nn.losses.cross_entropy(logits, y, reduction="none")

        # mask: 1 for assistant tokens (shifted targets)
        T = per_tok_ce.shape[1]
        assistant_start = max(0, prompt_len - 1)
        mask_vals = [0.0] * assistant_start + [1.0] * (T - assistant_start)
        mask = mx.array(mask_vals)[None, :]

        masked_loss = (per_tok_ce * mask).sum()
        n_tok = mask.sum()
        mx.eval(masked_loss, n_tok)

        n = n_tok.item()
        if n < 1:
            ppls.append(float("inf"))
        else:
            avg = masked_loss.item() / n
            ppls.append(math.exp(min(avg, 100)))

        del logits, per_tok_ce, mask, masked_loss, n_tok, x, y

    return ppls


def _median(xs):
    finite = [x for x in xs if math.isfinite(x) and x > 0]
    if not finite:
        return float("inf")
    finite.sort()
    return finite[len(finite) // 2]


def _frac_finite(xs):
    return sum(1 for x in xs if math.isfinite(x) and x > 0) / max(len(xs), 1)


def eval_model(model_id):
    """Load model, eval base + adapter PPLs for all domains."""
    print(f"\n{'=' * 60}")
    print(f"Loading {model_id} ...")
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())
    print("Loaded.")

    # --- Base PPL (no adapter) ---
    print("  Base PPL ...")
    base = {}
    for dom in DOMAINS:
        msgs = [s["messages"] for s in load_data(dom)]
        ppls = per_sample_ppl(model, tokenizer, msgs)
        base[dom] = ppls
        print(f"    {dom}: median={_median(ppls):.4f}  finite={_frac_finite(ppls)*100:.0f}%")

    # --- Adapter PPL per domain ---
    adapter = {}
    for dom in DOMAINS:
        print(f"  Adapter PPL [{dom}] ...")
        load_adapters(model, str(ADAPTERS_DIR / dom))
        mx.eval(model.parameters())
        msgs = [s["messages"] for s in load_data(dom)]
        ppls = per_sample_ppl(model, tokenizer, msgs)
        adapter[dom] = ppls
        print(f"    {dom}: median={_median(ppls):.4f}  finite={_frac_finite(ppls)*100:.0f}%")
        remove_lora_layers(model)
        mx.eval(model.parameters())

    del model, tokenizer
    mx.clear_cache()
    gc.collect()
    return base, adapter


def main():
    t0 = time.time()
    print(f"SMOKE={SMOKE}, N_EVAL={N_EVAL or 'all'}")

    # Phase 1: 4-bit
    base_4, adap_4 = eval_model(MODEL_4BIT)

    # Phase 2: 8-bit
    base_8, adap_8 = eval_model(MODEL_8BIT)

    # Phase 3: Metrics
    print(f"\n{'=' * 60}")
    print("TRANSFER ANALYSIS")
    print(f"{'=' * 60}")

    domain_metrics = {}
    for dom in DOMAINS:
        b4, a4 = _median(base_4[dom]), _median(adap_4[dom])
        b8, a8 = _median(base_8[dom]), _median(adap_8[dom])
        g4 = (b4 - a4) / b4 if b4 > 0 else 0.0
        g8 = (b8 - a8) / b8 if b8 > 0 else 0.0
        R = g8 / g4 if g4 > 0 else 0.0
        domain_metrics[dom] = dict(
            base_ppl_4bit=b4, adapter_ppl_4bit=a4, gain_4bit=g4,
            base_ppl_8bit=b8, adapter_ppl_8bit=a8, gain_8bit=g8,
            transfer_ratio=R,
        )
        print(f"\n{dom}:")
        print(f"  4-bit base={b4:.4f} adapter={a4:.4f} gain={g4:.4f}")
        print(f"  8-bit base={b8:.4f} adapter={a8:.4f} gain={g8:.4f}")
        print(f"  R = {R:.4f}")

    R_vals = [domain_metrics[d]["transfer_ratio"] for d in DOMAINS]
    median_R = sorted(R_vals)[1]  # 3 domains → index 1
    min_R = min(R_vals)

    # --- K1 ---
    adapter_helps = all(domain_metrics[d]["gain_4bit"] > 0 for d in DOMAINS)
    all_ppl_lists = (
        [base_4[d] for d in DOMAINS] + [adap_4[d] for d in DOMAINS]
        + [base_8[d] for d in DOMAINS] + [adap_8[d] for d in DOMAINS]
    )
    finite_ok = all(_frac_finite(p) >= 0.95 for p in all_ppl_lists)
    k1 = adapter_helps and finite_ok

    # --- K2 ---
    k2_median = median_R >= 0.95
    k2_floor = min_R >= 0.85
    k2 = k2_median and k2_floor

    # Verdict
    if not k1:
        verdict = "INCONCLUSIVE"
    elif k2:
        verdict = "SUPPORTED"
    else:
        verdict = "KILLED"

    print(f"\nK1 structural: {'PASS' if k1 else 'FAIL'}"
          f"  (helps={adapter_helps}, finite={finite_ok})")
    print(f"K2 transfer:   {'PASS' if k2 else 'FAIL'}"
          f"  (median_R={median_R:.4f}>={0.95}:{k2_median}, min_R={min_R:.4f}>={0.85}:{k2_floor})")
    print(f"\nVERDICT: {verdict}")

    results = {
        "experiment_id": "exp_g4_zs_base_transfer_4bit_fp16",
        "verdict": verdict,
        "all_pass": k1 and k2,
        "is_smoke": SMOKE,
        "mlx_lm_version": mlx_lm.__version__,
        "n_eval_per_domain": N_EVAL or len(load_data(DOMAINS[0])),
        "domains": domain_metrics,
        "k1_pass": k1,
        "k2_pass": k2,
        "median_transfer_ratio": median_R,
        "min_transfer_ratio": min_R,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = EXP_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
