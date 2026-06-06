#!/usr/bin/env python3
"""
exp_memento_gemma4_replication_impl — Phase A (executable) + Phase B/C/D (deferred).

MEMENTO 2-stage SFT + block-mask attention on Gemma 4 E4B 4-bit MLX.
Grounded: arxiv:2604.09852 (Kontonis et al., MSFT+UT+OpenAI, 2026-04-10).
Parent: exp_memento_gemma4_replication (PROVISIONAL design-only, F#682-family novel-mechanism cluster).

Phase A in this iteration:
  - Load Gemma 4 E4B 4-bit via mlx_lm.utils.load.
  - Inspect tokenizer (vocab=262144), embed_tokens layer (QuantizedEmbedding), tie_word_embeddings.
  - Mutate tokenizer to vocab+4 (add 4 memento boundary tokens in-memory).
  - Resolve new token IDs.
  - Document quantized-embedding resize plan in results.json (no actual resize executed —
    that is Phase A.v2 follow-up; quantized resize on 4-bit MLX is itself a research subtask).

Phases B/C/D raise NotImplementedError per researcher-hat single-iteration cap.

KC binding: all 4 (K#1829-#1832) require Phase B+C+D and are 'untested' here.
Verdict: PROVISIONAL with is_smoke=true.

Skills invoked: /mlx-dev (documented in MATH.md §0).
"""

from __future__ import annotations

import gc
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
DATASET = "microsoft/OpenMementos"

MEMENTO_TOKENS = [
    "<|block_start|>",
    "<|block_end|>",
    "<|summary_start|>",
    "<|summary_end|>",
]

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Phase B/C/D constants pre-registered for the follow-up iteration.
SFT_STEPS = 2000          # per stage; 2 stages = 4000 total
BATCH_SIZE = 1            # batch=1 + grad-accumulation; OOM-safe on M5 Pro 48GB
SEQLEN = 4096             # paper recommends; if OOM use 2048 (narrower-context finding)
LR = 2e-5
N_GSM8K = 200             # K1/K2 evaluation N
N_MMLU = 200              # K2 secondary
N_K3_ABLATION = 50        # K3 channel ablation N
N_LONG_CTX = 50           # K4 throughput prompts


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — tokenizer extension + embed inspection (executable)
# ─────────────────────────────────────────────────────────────────────────────


def phase_a_inspect_and_extend() -> dict[str, Any]:
    """Execute Phase A: load model, extend tokenizer in-memory, inspect embed layer.

    Returns a dict suitable for `results.json["phase_a_results"]`.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load

    t0 = time.perf_counter()
    out: dict[str, Any] = {"status": "pending"}

    print(f"[A.1] Loading {BASE_MODEL}", flush=True)
    model, tokenizer = mlx_load(BASE_MODEL)
    out["load_seconds"] = time.perf_counter() - t0

    # A.2 baseline tokenizer inspection
    hf_tok = tokenizer._tokenizer  # underlying HF PreTrainedTokenizerFast
    baseline_vocab = len(hf_tok)
    out["baseline_vocab"] = baseline_vocab
    out["baseline_special_tokens_count"] = len(hf_tok.all_special_tokens)
    print(f"[A.2] baseline_vocab={baseline_vocab} special={len(hf_tok.all_special_tokens)}", flush=True)

    # A.3 add 4 memento boundary tokens
    n_added = hf_tok.add_special_tokens({"additional_special_tokens": MEMENTO_TOKENS})
    new_vocab = len(hf_tok)
    out["new_vocab"] = new_vocab
    out["n_added_tokens"] = n_added
    out["expected_added"] = len(MEMENTO_TOKENS)
    print(f"[A.3] add_special_tokens added={n_added} new_vocab={new_vocab}", flush=True)

    # A.4 resolve new token IDs
    token_ids: dict[str, int] = {}
    for tok in MEMENTO_TOKENS:
        ids = hf_tok.encode(tok, add_special_tokens=False)
        token_ids[tok] = ids[0] if len(ids) == 1 else -1
    out["memento_token_ids"] = token_ids
    out["all_new_ids_above_baseline"] = all(
        tid >= baseline_vocab for tid in token_ids.values()
    )
    print(f"[A.4] memento_token_ids={token_ids}", flush=True)

    # A.5 inspect embed layer
    # Gemma 4 E4B (mlx-lm 0.31): model.language_model.model.embed_tokens
    try:
        lm = model.language_model
        text_model = lm.model
        embed = text_model.embed_tokens
        embed_type = type(embed).__name__
        # nn.QuantizedEmbedding has .weight + .scales + .biases (packed 4-bit)
        embed_attrs = sorted([a for a in dir(embed) if not a.startswith("_")])
        # Try to extract shape via .weight attribute (packed) or num_embeddings if exposed
        embed_shape = None
        if hasattr(embed, "weight"):
            try:
                w = embed.weight
                embed_shape = list(w.shape)
            except Exception as e:
                embed_shape = f"weight-access-error: {e}"
        # vocab_size from config
        cfg_vocab = getattr(lm.args, "vocab_size", None)
        out["embed_type"] = embed_type
        out["embed_shape_packed"] = embed_shape  # may be packed dim, e.g. (vocab, hidden//8) for 4-bit
        out["embed_attrs_public"] = embed_attrs[:20]  # cap
        out["config_vocab_size"] = cfg_vocab
        print(f"[A.5] embed_type={embed_type} embed_shape={embed_shape} cfg_vocab={cfg_vocab}", flush=True)
    except Exception as e:
        out["embed_inspect_error"] = repr(e)
        out["embed_inspect_traceback"] = traceback.format_exc()
        print(f"[A.5] embed inspect error: {e}", flush=True)

    # A.6 tie_word_embeddings
    try:
        text_inner = lm.language_model if hasattr(lm, "language_model") else lm.model.__class__
        # gemma4_text.Model has self.tie_word_embeddings attribute
        # Walk to find it
        tie = None
        candidate = lm
        for attr in ("tie_word_embeddings",):
            if hasattr(candidate, attr):
                tie = getattr(candidate, attr)
                break
        if tie is None and hasattr(lm, "model"):
            for attr in ("tie_word_embeddings",):
                if hasattr(lm.model, attr):
                    tie = getattr(lm.model, attr)
                    break
        # Whole-model search for tie_word_embeddings flag
        # Per gemma4_text.py L222 self.tie_word_embeddings is on the Model wrapping language_model
        if tie is None:
            # try lm itself or its children
            for name in dir(lm):
                if name.startswith("_"):
                    continue
                child = getattr(lm, name, None)
                if hasattr(child, "tie_word_embeddings"):
                    tie = getattr(child, "tie_word_embeddings")
                    break
        out["tie_word_embeddings"] = tie
        out["lm_head_separate"] = (
            hasattr(lm, "lm_head") and not (tie is True)
        )
        print(f"[A.6] tie_word_embeddings={tie} lm_head_separate={out['lm_head_separate']}", flush=True)
    except Exception as e:
        out["tie_inspect_error"] = repr(e)

    # A.7 resize plan (documentation only — no actual resize executed)
    out["resize_plan"] = {
        "deferred_to": "Phase A.v2 in follow-up iteration (quantized-resize subtask)",
        "approach": [
            "1. Read packed embed: w_packed, scales, biases = embed.weight, embed.scales, embed.biases",
            "2. Dequantize: w_fp = mx.dequantize(w_packed, scales, biases, group_size=embed.group_size, bits=embed.bits)",
            "3. Compute mean_row = w_fp.mean(axis=0, keepdims=True) (1, hidden)",
            "4. Build new_w_fp = mx.concatenate([w_fp, mx.broadcast_to(mean_row, (4, hidden))], axis=0)",
            "5. Create new_embed = nn.Embedding(new_vocab, hidden); new_embed.weight = new_w_fp",
            "6. (Optional) re-quantize: nn.quantize(new_embed, group_size=..., bits=4)",
            "7. Replace text_model.embed_tokens = new_embed; mx.eval(model.parameters())",
            "8. lm_head reuses via embed_tokens.as_linear (tie=True path), so no separate lm_head resize.",
        ],
        "risks": [
            "Quantized re-quantization may shift logits subtly; alternative is to leave new embed layer unquantized (mixed precision).",
            "Dequantize+resize+requantize is itself a research validation step — ablate against unquantized-resize baseline.",
            "MLX nn.Embedding.weight assignment must respect mx.array contract (in-place mutation of model state).",
        ],
        "blocked_by": [
            "Need /fast-mlx skill for dequantize/quantize idiomatic patterns.",
            "Need a forward-pass smoke after resize to confirm logits are sane on existing tokens (regression check).",
        ],
    }

    out["status"] = "ok"
    out["wall_clock_seconds"] = time.perf_counter() - t0

    # Cleanup before returning
    del model, tokenizer
    gc.collect()
    mx.clear_cache()

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — 2-stage SFT (deferred)
# ─────────────────────────────────────────────────────────────────────────────


def phase_b_sft_stage1(model, tokenizer, dataloader, steps=SFT_STEPS):
    """Stage-1 standard next-token CE on OpenMementos.

    See MATH.md §6 for full pipeline. Uses nn.value_and_grad(model, loss_fn);
    mx.eval at each step boundary; mx.clear_cache() between stage-1/stage-2.
    """
    raise NotImplementedError(
        "phase_b_sft_stage1: deferred to Phase B implementation iteration "
        "(invoke /fast-mlx; full-parameter SFT, no LoRA substitution per antipattern-t)"
    )


def phase_b_sft_stage2(model, tokenizer, dataloader, steps=SFT_STEPS):
    """Stage-2 attend-only-to-mementos SFT.

    Mask all block-content KV after <|summary_end|> during forward; loss only
    on response tokens past the summary boundary.
    """
    raise NotImplementedError(
        "phase_b_sft_stage2: deferred to Phase B implementation iteration"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase C — custom MLX inference loop with block-mask attention (deferred)
# ─────────────────────────────────────────────────────────────────────────────


class BlockMaskState:
    """Tracks per-token block boundaries from emitted memento tokens.

    Used during Phase C generation to drive selective KV eviction and attention
    mask surgery. See MATH.md §6 Phase C for spec.
    """
    def __init__(self):
        raise NotImplementedError(
            "BlockMaskState: deferred to Phase C implementation iteration "
            "(invoke /fast-mlx for mx.fast.scaled_dot_product_attention(mask=...))"
        )


def phase_c_block_mask_generate(model, tokenizer, prompt: str, max_tokens: int = 1024):
    """Generation loop with block-mask attention + selective KV eviction.

    Per-token: detect emitted <|block_start|>/<|block_end|>/<|summary_end|> via
    BlockMaskState; mutate attention mask; free evicted block KV from cache.
    """
    raise NotImplementedError(
        "phase_c_block_mask_generate: deferred to Phase C implementation iteration"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase D — KC eval harness (deferred)
# ─────────────────────────────────────────────────────────────────────────────


def phase_d_eval_kc1_kv_reduction(model_memento, model_base, prompts):
    """K#1829: peak KV cache reduction ratio on GSM8K-Hard n>=200."""
    raise NotImplementedError("phase_d_eval_kc1: deferred")


def phase_d_eval_kc2_accuracy(model_memento, model_base):
    """K#1830: GSM8K-Hard drop < 5pp AND MMLU drop < 3pp at n>=200."""
    raise NotImplementedError("phase_d_eval_kc2: deferred")


def phase_d_eval_kc3_channel_ablation(model_memento):
    """K#1831: KV-channel ablation drop >= 10pp (replicates paper's 15pp at our scale)."""
    raise NotImplementedError("phase_d_eval_kc3: deferred")


def phase_d_eval_kc4_throughput(model_memento, model_base, long_prompts):
    """K#1832: end-to-end throughput >= 1.3x base on long-context prompts."""
    raise NotImplementedError("phase_d_eval_kc4: deferred")


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Results:
    is_smoke: bool
    verdict: str
    all_pass: bool
    kc: dict
    measurements: dict
    runtime_seconds: float
    phase_a_results: dict
    phases_deferred: list = field(default_factory=lambda: ["B", "C", "D"])


def main():
    t0 = time.perf_counter()

    # Run Phase A (executable). Catch errors — Phase A failure is plumbing, not mechanism.
    try:
        phase_a = phase_a_inspect_and_extend()
    except Exception as e:
        phase_a = {
            "status": "error",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
        print(f"[Phase A FAILED] {e}", flush=True)

    kc_untested = {
        "K1_kv_reduction": {
            "kc_id": 1829,
            "type": "proxy",
            "paired_with": "K2",
            "threshold": "ratio <= 0.5 (>=2x reduction) on GSM8K-Hard n>=200",
            "value": None,
            "pass": "untested",
            "reason": "Phase B+C+D deferred (researcher-hat cap)",
        },
        "K2_gsm8k_drop": {
            "kc_id": 1830,
            "type": "target",
            "pair_of": "K1",
            "threshold": "GSM8K-Hard drop < 5pp",
            "value": None,
            "pass": "untested",
            "reason": "Phase B+C+D deferred",
        },
        "K2_mmlu_drop": {
            "kc_id": 1830,
            "type": "target",
            "pair_of": "K1",
            "threshold": "MMLU drop < 3pp",
            "value": None,
            "pass": "untested",
            "reason": "Phase B+C+D deferred",
        },
        "K3_kv_channel_ablation": {
            "kc_id": 1831,
            "type": "target_replication",
            "threshold": "acc_gap >= 10pp (paper 15pp at 8B)",
            "value": None,
            "pass": "untested",
            "reason": "Phase B+C+D deferred",
        },
        "K4_throughput": {
            "kc_id": 1832,
            "type": "target_serving",
            "threshold": ">= 1.3x base tok/s on long prompts",
            "value": None,
            "pass": "untested",
            "reason": "Phase B+C+D deferred",
        },
    }

    r = Results(
        is_smoke=True,  # Phase A inspect-only is smoke
        verdict="PROVISIONAL",
        all_pass=False,
        kc=kc_untested,
        measurements={
            "base_model": BASE_MODEL,
            "dataset": DATASET,
            "phase_a_status": phase_a.get("status", "unknown"),
            "n_memento_tokens": len(MEMENTO_TOKENS),
            "phases_executed": ["A_inspect"],
            "phases_deferred": ["A_resize", "B_sft", "C_inference", "D_eval"],
            "design_rationale": (
                "Novel-mechanism IMPL Phase A inspect+extend (executable) + "
                "Phase B/C/D deferred per researcher-hat single-iteration cap "
                "(mem-antipattern-novel-mechanism-single-iteration-scope). "
                "Marginal contribution over parent design-only PROVISIONAL: "
                "actual model load + tokenizer mutation + embed inspection."
            ),
        },
        runtime_seconds=time.perf_counter() - t0,
        phase_a_results=phase_a,
    )

    RESULTS_FILE.write_text(json.dumps(asdict(r), indent=2, default=str))
    print(
        f"verdict={r.verdict} is_smoke={r.is_smoke} "
        f"phase_a_status={r.measurements['phase_a_status']} "
        f"runtime={r.runtime_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
