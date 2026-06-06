#!/usr/bin/env python3
"""exp_rdt_loop_kv_cache_impl — empirical companion to exp_rdt_loop_kv_cache.

Runs the K1837 (bit-exact cached/uncached) and (in full mode only) K1838
(speedup) verifications scope-deferred from parent. Smoke mode focuses on
K1837 with n=2 prompts × T=3 to land PROVISIONAL with a real measurement
under researcher-hat budget. K1838/K1986/K1987 explicitly deferred to
exp_rdt_loop_kv_cache_full (P=2 macro).

Skills /mlx-dev + /fast-mlx invoked before writing this file
(MATH.md §0 + PLAN.md guardrail 1012).

Env knobs:
- SMOKE_TEST (default "1") — "0" to attempt full K1837 (n=20 × T sweep) +
  K1838 (5x speedup, n=20, M=64) + K1986 (greedy-token agreement) +
  K1987 (n=200 unlock budget).
- N_PROMPTS_SMOKE (default 2)
- T_SWEEP_SMOKE (default "3")
- N_PROMPTS_FULL (default 20)
- T_SWEEP_FULL (default "1,2,3,6")
- M_TOKENS_FULL (default 64)
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import KVCache, RotatingKVCache

SEED = 42
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LOOP_START = 12
LOOP_END = 21      # exclusive => 9 layers
N_LOOP = LOOP_END - LOOP_START
N_LOOPS = 6        # number of loop iterations the LoRA bank supports (T_max)
LORA_RANK = 16
LORA_ALPHA = 2.0
HIDDEN = 2560
LOGIT_TOL = 1e-3
SPEEDUP_THRESH = 5.0
PARITY_THRESH = 0.99

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"
N_PROMPTS_SMOKE = int(os.environ.get("N_PROMPTS_SMOKE", 2))
T_SWEEP_SMOKE = [int(x) for x in os.environ.get("T_SWEEP_SMOKE", "3").split(",")]
N_PROMPTS_FULL = int(os.environ.get("N_PROMPTS_FULL", 20))
T_SWEEP_FULL = [int(x) for x in os.environ.get("T_SWEEP_FULL", "1,2,3,6").split(",")]
M_TOKENS_FULL = int(os.environ.get("M_TOKENS_FULL", 64))

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"

# Hand-coded smoke prompts (avoid GSM8K data dep for speed). All ≤ 50 tokens.
SMOKE_PROMPTS = [
    "What is 12 plus 7?",
    "If a train travels 60 km in 2 hours, what is its average speed?",
]


# ─────────────────────────────────────────────────────────────────
# LoopLoRA wiring (parent-inherited; avoid copy-paste-scaffolding antipattern
# by keeping minimal ports — only the classes used here).
# ─────────────────────────────────────────────────────────────────

class LTIInjection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_A = mx.zeros((dim,))
        self.log_dt = mx.zeros((1,))
        self.B = mx.full((dim,), 0.1)

    def get_A(self) -> mx.array:
        s = mx.clip(self.log_dt + self.log_A, -20.0, 20.0)
        return mx.exp(-mx.exp(s))

    def __call__(self, h: mx.array, e: mx.array, tfm_out: mx.array) -> mx.array:
        return self.get_A() * h + self.B * e + tfm_out


class LoRADelta(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, A_init: mx.array) -> None:
        super().__init__()
        self._A_fixed = A_init
        self.B = mx.zeros((out_dim, r))
        self.scale = float(alpha / r)

    def __call__(self, x: mx.array) -> mx.array:
        z = x @ self._A_fixed.T
        return self.scale * (z @ self.B.T)


def partition_qr_lora_A(in_dim: int, n: int, r: int, key) -> list[mx.array]:
    W = mx.random.normal(shape=(in_dim, n * r), key=key)
    Q, _ = mx.linalg.qr(W, stream=mx.cpu)
    Q = Q.astype(mx.float32)
    return [Q[:, (i * r):(i + 1) * r].T for i in range(n)]


class LoopLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, deltas: list[LoRADelta], loop_idx_ref: list[int]) -> None:
        super().__init__()
        self._base = base
        self.deltas = deltas
        self._loop_idx = loop_idx_ref

    def __call__(self, x: mx.array) -> mx.array:
        return self._base(x) + self.deltas[self._loop_idx[0]](x)


def wire_loop_lora(model, n_loops: int, r: int, alpha: float, key):
    layers = model.language_model.model.layers
    loop_idx_ref = [0]
    bank: dict = {}
    for ell in range(LOOP_START, LOOP_END):
        attn = layers[ell].self_attn
        v_in = HIDDEN
        v_out = attn.n_kv_heads * attn.head_dim
        o_in = attn.n_heads * attn.head_dim
        o_out = HIDDEN

        key_v, key_o, key = mx.random.split(key, 3)
        As_v = partition_qr_lora_A(v_in, n_loops, r, key_v)
        As_o = partition_qr_lora_A(o_in, n_loops, r, key_o)

        v_deltas = [LoRADelta(v_in, v_out, r, alpha, As_v[t]) for t in range(n_loops)]
        o_deltas = [LoRADelta(o_in, o_out, r, alpha, As_o[t]) for t in range(n_loops)]

        attn.v_proj = LoopLoRALinear(attn.v_proj, v_deltas, loop_idx_ref)
        attn.o_proj = LoopLoRALinear(attn.o_proj, o_deltas, loop_idx_ref)

        bank[ell] = {"v": v_deltas, "o": o_deltas}
    return bank, loop_idx_ref


# ─────────────────────────────────────────────────────────────────
# Patched __call__ unifying uncached + cached recurrent-depth forward.
# See MATH.md §1.2 for design rationale.
# ─────────────────────────────────────────────────────────────────

def install_patch(model, lti_bank, loop_idx_ref, T_ref):
    from mlx_lm.models.gemma4_text import Gemma4TextModel

    def patched(self, inputs=None, cache=None, input_embeddings=None, per_layer_inputs=None):
        L = len(self.layers)
        T_now = T_ref[0]
        expected_len = LOOP_START + T_now * N_LOOP + (L - LOOP_END)

        if input_embeddings is None:
            input_embeddings = self.embed_tokens(inputs)
        h = input_embeddings * self.embed_scale

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self._get_per_layer_inputs(inputs, input_embeddings)
            per_layer_inputs = self._project_per_layer_inputs(h, per_layer_inputs)
        if per_layer_inputs is not None:
            per_layer_inputs = [per_layer_inputs[:, :, i, :] for i, _ in enumerate(self.layers)]
        else:
            per_layer_inputs = [None] * L

        if cache is None:
            cache = [None] * expected_len
        elif len(cache) < expected_len:
            cache = cache + [None] * (expected_len - len(cache))

        # Length-L mask-cache alias — MATH.md §1.3.
        mask_cache = []
        for i in range(L):
            if i < LOOP_START:
                mask_cache.append(cache[i])
            elif i < LOOP_END:
                mask_cache.append(cache[LOOP_START + (T_now - 1) * N_LOOP + (i - LOOP_START)])
            else:
                mask_cache.append(cache[LOOP_START + T_now * N_LOOP + (i - LOOP_END)])
        masks = self._make_masks(h, mask_cache)
        intermediates = [(None, None)] * L

        idx = 0
        while idx < L:
            if idx == LOOP_START:
                h_block_entry = h
                for t in range(T_now):
                    loop_idx_ref[0] = t
                    h_loop = h
                    for j in range(LOOP_START, LOOP_END):
                        c_idx = LOOP_START + t * N_LOOP + (j - LOOP_START)
                        h_loop, _, _ = self.layers[j](
                            h_loop, masks[j], cache[c_idx],
                            per_layer_input=per_layer_inputs[j],
                            shared_kv=None, offset=None,
                        )
                    if t < T_now - 1:
                        h = lti_bank[t](h_block_entry, h_block_entry, h_loop)
                    else:
                        h = h_loop
                idx = LOOP_END
                continue
            c_idx = idx if idx < LOOP_START else LOOP_START + T_now * N_LOOP + (idx - LOOP_END)
            kvs, offset = intermediates[self.previous_kvs[idx]]
            h, kvs, offset = self.layers[idx](
                h, masks[idx], cache[c_idx],
                per_layer_input=per_layer_inputs[idx],
                shared_kv=kvs, offset=offset,
            )
            intermediates[idx] = (kvs, offset)
            idx += 1
        return self.norm(h)

    Gemma4TextModel.__call__ = patched


# ─────────────────────────────────────────────────────────────────
# Cache-list construction (length 33 + 9T) with proper KV / sliding mix.
# ─────────────────────────────────────────────────────────────────

def make_recurrent_cache(model, T_now: int) -> list:
    inner = model.language_model.model
    L = len(inner.layers)
    sliding = inner.window_size
    out: list = []

    # Prefix [0, LOOP_START)
    for i in range(LOOP_START):
        if inner.previous_kvs[i] != i:
            out.append(None)
            continue
        if inner.layers[i].layer_type == "full_attention":
            out.append(KVCache())
        else:
            out.append(RotatingKVCache(max_size=sliding, keep=0))

    # Loop region: T copies of (LOOP_START..LOOP_END)
    for _t in range(T_now):
        for j in range(LOOP_START, LOOP_END):
            if inner.previous_kvs[j] != j:
                out.append(None)
                continue
            if inner.layers[j].layer_type == "full_attention":
                out.append(KVCache())
            else:
                out.append(RotatingKVCache(max_size=sliding, keep=0))

    # Suffix [LOOP_END, L)
    for i in range(LOOP_END, L):
        if inner.previous_kvs[i] != i:
            out.append(None)
            continue
        if inner.layers[i].layer_type == "full_attention":
            out.append(KVCache())
        else:
            out.append(RotatingKVCache(max_size=sliding, keep=0))

    expected = LOOP_START + T_now * N_LOOP + (L - LOOP_END)
    assert len(out) == expected, f"cache length {len(out)} != expected {expected}"
    return out


# ─────────────────────────────────────────────────────────────────
# K1837 verification: bit-exact cached vs uncached on prompt logits.
# ─────────────────────────────────────────────────────────────────

def verify_bit_exact(model, tokenizer, prompts: list[str], t_sweep: list[int],
                     loop_idx_ref, T_ref) -> dict:
    pairs: list[dict] = []
    for T in t_sweep:
        T_ref[0] = T
        for prompt in prompts:
            toks = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            x = mx.array([toks], dtype=mx.int32)

            # Uncached: cache=None branch.
            loop_idx_ref[0] = 0
            logits_unc = model(x)
            mx.eval(logits_unc)
            logits_unc_last = logits_unc[0, -1, :].astype(mx.float32)
            mx.eval(logits_unc_last)
            del logits_unc
            mx.clear_cache()

            # Cached: fresh KVCache list of length 33 + 9T.
            cache = make_recurrent_cache(model, T)
            loop_idx_ref[0] = 0
            # Pass cache through model.language_model.model.__call__ via the
            # outer Model (the lm_head wrapper). The Model.__call__ in mlx_lm
            # accepts cache; pass it through.
            logits_c = model(x, cache=cache)
            mx.eval(logits_c)
            logits_c_last = logits_c[0, -1, :].astype(mx.float32)
            mx.eval(logits_c_last)
            del logits_c

            diff = mx.abs(logits_unc_last - logits_c_last)
            max_diff = float(mx.max(diff).item())
            mean_diff = float(mx.mean(diff).item())

            pairs.append({
                "T": T,
                "prompt_chars": len(prompt),
                "n_tokens": int(x.shape[1]),
                "max_abs_logit_diff": max_diff,
                "mean_abs_logit_diff": mean_diff,
            })
            print(f"  T={T} prompt[:20]={prompt[:20]!r} ntok={x.shape[1]} "
                  f"max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}", flush=True)
            del cache
            mx.clear_cache()
            gc.collect()

    max_over_pairs = max(p["max_abs_logit_diff"] for p in pairs) if pairs else float("inf")
    n_pairs_under = sum(1 for p in pairs if p["max_abs_logit_diff"] < LOGIT_TOL)
    return {
        "pairs": pairs,
        "max_abs_logit_diff_over_all": max_over_pairs,
        "n_pairs": len(pairs),
        "n_pairs_under_tol": n_pairs_under,
        "tol": LOGIT_TOL,
        "all_under_tol": n_pairs_under == len(pairs),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.time()
    mx.random.seed(SEED)
    mx.set_memory_limit(46 * 1024 * 1024 * 1024)
    mx.set_cache_limit(2 * 1024 * 1024 * 1024)

    n_prompts = N_PROMPTS_SMOKE if SMOKE_TEST else N_PROMPTS_FULL
    t_sweep = T_SWEEP_SMOKE if SMOKE_TEST else T_SWEEP_FULL

    print(f"[phase0] SMOKE_TEST={SMOKE_TEST} n_prompts={n_prompts} t_sweep={t_sweep}", flush=True)
    print(f"[phase1] loading {MODEL_ID}...", flush=True)
    model, tokenizer = load(MODEL_ID)
    model.freeze()

    print("[phase2] wiring loop-LoRA on layers 12..20 (v+o)...", flush=True)
    key = mx.random.key(SEED)
    bank, loop_idx_ref = wire_loop_lora(model, N_LOOPS, LORA_RANK, LORA_ALPHA, key)
    lti_bank = [LTIInjection(HIDDEN) for _ in range(N_LOOPS)]
    T_ref = [3]
    install_patch(model, lti_bank, loop_idx_ref, T_ref)

    a_ok = all(
        isinstance(model.language_model.model.layers[ell].self_attn.v_proj, LoopLoRALinear)
        and isinstance(model.language_model.model.layers[ell].self_attn.o_proj, LoopLoRALinear)
        for ell in range(LOOP_START, LOOP_END)
    )
    print(f"[phase2] wiring_ok={a_ok}", flush=True)

    # Use only n_prompts smoke prompts for smoke; full mode would load GSM8K.
    if SMOKE_TEST:
        prompts = SMOKE_PROMPTS[:n_prompts]
    else:
        # Full mode: load GSM8K-valid (parent's path). Deferred to _full
        # but if SMOKE_TEST=0 is invoked here we make a best-effort.
        gsm8k_valid = (Path(__file__).resolve().parents[1]
                       / "exp_p1_t2_single_domain_training/data/math/valid.jsonl")
        prompts = []
        if gsm8k_valid.exists():
            with gsm8k_valid.open() as f:
                for line in f:
                    if len(prompts) >= n_prompts:
                        break
                    ex = json.loads(line)
                    user_msg = next((m for m in ex["messages"] if m["role"] == "user"), None)
                    if user_msg:
                        prompts.append(user_msg["content"])
        if not prompts:
            prompts = SMOKE_PROMPTS * (n_prompts // len(SMOKE_PROMPTS) + 1)
            prompts = prompts[:n_prompts]

    print(f"[phase3] K1837 bit-exact verification on n={len(prompts)} × T={t_sweep}...", flush=True)
    k1837 = verify_bit_exact(model, tokenizer, prompts, t_sweep, loop_idx_ref, T_ref)

    # Compute K1837 result (proxy + target_pair_complete check).
    k1837_pass = bool(k1837["all_under_tol"])
    if SMOKE_TEST:
        k1837_status = "pass_smoke" if k1837_pass else "fail_smoke"
    else:
        # Full mode: pass requires all 80 pairs under tol.
        k1837_status = "pass" if (k1837_pass and k1837["n_pairs"] >= 80) else (
            "fail" if not k1837_pass else "under_powered"
        )

    # K1838 / K1986 / K1987 — deferred for smoke.
    deferral_reason_smoke = (
        f"smoke-mode (SMOKE_TEST=1) measures K1837 only at n={n_prompts} × T={t_sweep}; "
        "deferred to exp_rdt_loop_kv_cache_full (P=2 macro) which runs the full "
        "n=20 × T={1,2,3,6} K1837 + n=20 × M=64 K1838 + n=20 K1986 (greedy parity) "
        "+ n=200 × M=128 K1987 (unlock budget) on a 3-4h pueue task."
    )

    # Verdict per PLAN §1010:
    # - smoke: ceiling PROVISIONAL (never SUPPORTED in smoke per #4)
    # - full: SUPPORTED requires k1837 + k1838 + k1986 + k1987 all PASS
    if SMOKE_TEST:
        verdict = "PROVISIONAL"
        all_pass = False
    else:
        # Full path not implemented in this iter — would require K1838+K1986+K1987
        # measurement code beyond scope. Mark PROVISIONAL until full follow-on lands.
        verdict = "PROVISIONAL"
        all_pass = False

    elapsed = time.time() - t0

    out = {
        "experiment_id": "exp_rdt_loop_kv_cache_impl",
        "is_smoke": bool(SMOKE_TEST),
        "verdict": verdict,
        "all_pass": all_pass,
        "preemptive": False,
        "executed": True,
        "elapsed_sec": round(elapsed, 2),
        "mlx_version": "0.31.1",
        "mlx_lm_version": "0.31.2",
        "seed": SEED,
        "config": {
            "model": MODEL_ID,
            "loop_layers": [LOOP_START, LOOP_END - 1],
            "n_loops_supported": N_LOOPS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "n_prompts": len(prompts),
            "t_sweep": t_sweep,
            "logit_tol": LOGIT_TOL,
            "speedup_threshold": SPEEDUP_THRESH,
            "parity_threshold": PARITY_THRESH,
            "wiring_ok": a_ok,
        },
        "kill_criteria": {
            "K1837": {
                "id": 1837,
                "desc": ("Cached vs uncached recurrent-depth forward agree to "
                         "max_abs_logit_diff < 1e-3 in fp16 across n=20 × T∈{1,2,3,6}=80 pairs."),
                "result": k1837_status,
                "max_abs_logit_diff_over_all_pairs": k1837["max_abs_logit_diff_over_all"],
                "n_pairs_measured": k1837["n_pairs"],
                "n_pairs_under_tol": k1837["n_pairs_under_tol"],
                "tol": LOGIT_TOL,
                "pairs": k1837["pairs"],
                "smoke_subset": ("n=2 × T=3 = 2 pairs" if SMOKE_TEST else "full n=20 × T={1,2,3,6}"),
            },
            "K1838": {
                "id": 1838,
                "desc": "Cached T=3 gen ≥ 5× faster than uncached on n=20 × M=64.",
                "result": "not_measured",
                "reason": deferral_reason_smoke if SMOKE_TEST else (
                    "K1838 generation harness not implemented in this _impl iter; "
                    "full follow-on exp_rdt_loop_kv_cache_full inherits."
                ),
            },
            "K1986": {
                "id": 1986,
                "desc": "Greedy-token agreement ≥ 99% cached vs uncached gen on n=20 × T=3 × M=64.",
                "result": "not_measured",
                "reason": deferral_reason_smoke if SMOKE_TEST else (
                    "K1986 behavioral parity harness deferred to full follow-on."
                ),
            },
            "K1987": {
                "id": 1987,
                "desc": "Cached T=3 n=200 GSM8K-Hard M=128 completes within 2h wall-clock budget.",
                "result": "not_measured",
                "reason": deferral_reason_smoke if SMOKE_TEST else (
                    "K1987 budget-unlock harness deferred to full follow-on."
                ),
            },
        },
        "antipatterns_flagged": [],
        "notes": (
            "SMOKE measurement of K1837 (bit-exact cached/uncached) on n=2 hand-coded "
            "prompts × T=3 — landed real measurement under researcher-hat budget. "
            "K1838/K1986/K1987 explicitly deferred to exp_rdt_loop_kv_cache_full (P=2 "
            "macro). PLAN §1010 #4: smoke completes as PROVISIONAL, never SUPPORTED. "
            "F#666 target-gating: K1837 (proxy) is paired with K1986 (target) and K1838 "
            "(proxy) with K1987 (target); both target KCs deferred to full → verdict "
            "ceiling PROVISIONAL even at K1837 PASS. Mathematical guarantee in parent "
            "MATH §4 transfers verbatim via unified §1.2 patched call (this MATH.md)."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"\n=== SUMMARY (smoke={SMOKE_TEST}) ===\n"
        f"K1837: {k1837_status} max_diff={k1837['max_abs_logit_diff_over_all']:.6e} "
        f"({k1837['n_pairs_under_tol']}/{k1837['n_pairs']} under {LOGIT_TOL})\n"
        f"K1838: not_measured (deferred to _full)\n"
        f"K1986: not_measured (deferred to _full)\n"
        f"K1987: not_measured (deferred to _full)\n"
        f"verdict={verdict} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
