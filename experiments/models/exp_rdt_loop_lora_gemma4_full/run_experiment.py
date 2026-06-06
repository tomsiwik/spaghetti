#!/usr/bin/env python3
"""exp_rdt_loop_lora_gemma4_full — full-scale follow-up to smoke.

Replaces the surrogate zero-pad forward with a real monkey-patched Gemma 4
forward: layers 12..20 loop T times with loop-indexed LoRA on v_proj+o_proj
and LTI injection at the block entry. Trains on real GSM8K CE loss.

KCs measured (pre-registered in MATH.md):
- K-FULL-A: real block integration via LoopLoRALinear wrapper (structural)
- K-FULL-B: both v_proj and o_proj B-matrix grads non-zero on first batch
- K-FULL-C: rho(A_d)<1 over N_STEPS real-loss steps; log_A and log_dt move
- K1740/K1742: GSM8K acc at T=3 and T in {1..6} on reduced n (PROVISIONAL if n<200)
- K1741: MMLU — not measured; deferred follow-up.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load

# Knobs — defensible choices, see MATH.md Theorem 4
SEED = 42
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LOOP_START = 12
LOOP_END = 21  # exclusive -> 9 layers
N_LOOPS = 6
LORA_RANK = 16
LORA_ALPHA = 2.0
HIDDEN = 2560
N_STEPS = int(os.environ.get("N_STEPS", 200))
BATCH = 1
MAX_SEQ = 256
LR = 5e-4
N_EVAL_T3 = int(os.environ.get("N_EVAL_T3", 50))   # K1740 target: n>=200; scoped down
N_EVAL_PER_T = int(os.environ.get("N_EVAL_PER_T", 10))  # K1742 per-T sweep
EVAL_T_VALUES = [1, 2, 3, 4, 5, 6]
MAX_EVAL_TOKENS = 512

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"
GSM8K_TRAIN = Path(__file__).resolve().parents[1] / "exp_p1_t2_single_domain_training/data/math/train.jsonl"
GSM8K_VALID = Path(__file__).resolve().parents[1] / "exp_p1_t2_single_domain_training/data/math/valid.jsonl"


# ─────────────────────────────────────────────
# LoRA delta and LTI (F#667 port)
# ─────────────────────────────────────────────

class LTIInjection(nn.Module):
    """h' = A_d * h + B * e + tfm_out; A_d = exp(-exp(clamp(log_dt+log_A)))."""

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
    """Frozen Grassmannian A, trainable B (init zero). Output: scale * (x A^T) B^T."""

    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, A_init: mx.array) -> None:
        super().__init__()
        # Non-parameter storage so A_init is excluded from optimizer updates.
        self._A_fixed = A_init
        self.B = mx.zeros((out_dim, r))
        self.scale = float(alpha / r)

    def __call__(self, x: mx.array) -> mx.array:
        z = x @ self._A_fixed.T
        return self.scale * (z @ self.B.T)


def partition_qr_lora_A(in_dim: int, n: int, r: int, key) -> list[mx.array]:
    """Grassmannian A-family via partition-QR (F#562)."""
    W = mx.random.normal(shape=(in_dim, n * r), key=key)
    Q, _ = mx.linalg.qr(W, stream=mx.cpu)
    Q = Q.astype(mx.float32)
    return [Q[:, (i * r):(i + 1) * r].T for i in range(n)]


# ─────────────────────────────────────────────
# LoopLoRALinear monkey-patch wrapper
# ─────────────────────────────────────────────

class LoopLoRALinear(nn.Module):
    """Wraps a base nn.Linear and adds loop-indexed LoRA delta on top.

    Shape-preserving: base(x) + deltas[loop_idx_ref[0]](x).
    """

    def __init__(self, base: nn.Linear, deltas: list[LoRADelta], loop_idx_ref: list[int]) -> None:
        super().__init__()
        self._base = base          # frozen; underscore to hide from freeze/unfreeze walks
        self.deltas = deltas       # list -> MLX module tree
        self._loop_idx = loop_idx_ref

    def __call__(self, x: mx.array) -> mx.array:
        base_y = self._base(x)
        delta_y = self.deltas[self._loop_idx[0]](x)
        return base_y + delta_y


# ─────────────────────────────────────────────
# Looped forward patch for Gemma4TextModel
# ─────────────────────────────────────────────

def install_loop_patch(model, lti_bank: list[LTIInjection], loop_idx_ref: list[int], T_ref: list[int]):
    """Monkey-patch Gemma4TextModel.__call__ at the class level.

    Python's `obj(...)` resolves `__call__` via `type(obj)`, not the instance;
    patching `self.__call__` is silently ignored. We patch the class's __call__
    once; since only one Gemma4TextModel instance is loaded per process, this
    is safe. The closure captures lti_bank, loop_idx_ref, T_ref so the patch
    knows how to drive the loop.

    Semantics:
      - layers 0..LOOP_START-1: run once (standard).
      - layers LOOP_START..LOOP_END-1: run T_ref[0] times, with loop_idx_ref
        mutated before each block pass, and LTI-injection applied at block
        entry each iter (except the last).
      - layers LOOP_END..end: run once (standard).
    """
    from mlx_lm.models.gemma4_text import Gemma4TextModel

    def patched_call(self, inputs=None, cache=None, input_embeddings=None, per_layer_inputs=None):
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
            per_layer_inputs = [None] * len(self.layers)

        if cache is None:
            cache = [None] * len(self.layers)
        else:
            cache = cache + [None] * (len(self.layers) - len(cache))

        masks = self._make_masks(h, cache)
        intermediates = [(None, None)] * len(self.layers)

        idx = 0
        while idx < len(self.layers):
            if idx == LOOP_START:
                h_block_entry = h
                for t in range(T_ref[0]):
                    loop_idx_ref[0] = t
                    h_loop = h
                    for j in range(LOOP_START, LOOP_END):
                        # For the looped block, skip previous_kvs sharing (training,
                        # cache=None); each iter's k/v is regenerated.
                        h_loop, _, _ = self.layers[j](
                            h_loop,
                            masks[j],
                            cache[j],
                            per_layer_input=per_layer_inputs[j],
                            shared_kv=None,
                            offset=None,
                        )
                        intermediates[j] = (None, None)
                    if t < T_ref[0] - 1:
                        h = lti_bank[t](h_block_entry, h_block_entry, h_loop)
                    else:
                        h = h_loop
                idx = LOOP_END
                continue
            # Standard path.
            layer = self.layers[idx]
            kvs, offset = intermediates[self.previous_kvs[idx]]
            h, kvs, offset = layer(
                h, masks[idx], cache[idx],
                per_layer_input=per_layer_inputs[idx],
                shared_kv=kvs,
                offset=offset,
            )
            intermediates[idx] = (kvs, offset)
            idx += 1

        return self.norm(h)

    Gemma4TextModel.__call__ = patched_call


# ─────────────────────────────────────────────
# Wire LoopLoRA on v_proj and o_proj for layers 12..20
# ─────────────────────────────────────────────

def wire_loop_lora(model, n_loops: int, r: int, alpha: float, key) -> tuple[dict, list[int]]:
    """Replace v_proj and o_proj on layers LOOP_START..LOOP_END-1 with LoopLoRALinear."""
    # Gemma 4 E4B 4-bit is wrapped as multimodal: Model -> language_model (Model) -> model (Gemma4TextModel)
    layers = model.language_model.model.layers
    loop_idx_ref = [0]
    bank: dict = {}
    for ell in range(LOOP_START, LOOP_END):
        attn = layers[ell].self_attn
        # Derive dims from the layer's attention module (handles sliding vs full_attention
        # asymmetry: L17 is full_attention with head_dim=512, others are sliding with 256).
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


# ─────────────────────────────────────────────
# K1743 measurement (reused from smoke — structural, fast)
# ─────────────────────────────────────────────

def measure_init_orthogonality(bank: dict) -> dict:
    per: dict = {}
    for ell, projs in bank.items():
        for pname, deltas in projs.items():
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    Ai = deltas[i]._A_fixed
                    Aj = deltas[j]._A_fixed
                    Ain = Ai / (mx.linalg.norm(Ai, axis=1, keepdims=True) + 1e-12)
                    Ajn = Aj / (mx.linalg.norm(Aj, axis=1, keepdims=True) + 1e-12)
                    mc = mx.max(mx.abs(Ain @ Ajn.T)).item()
                    k = f"L{ell}_{pname}"
                    per[k] = max(per.get(k, 0.0), mc)
    return per


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_gsm8k_train(tokenizer, max_examples: int, seqlen: int) -> list[mx.array]:
    """Tokenize GSM8K train.jsonl with Gemma 4 chat template, truncate to seqlen."""
    samples = []
    with GSM8K_TRAIN.open() as f:
        for line in f:
            if len(samples) >= max_examples:
                break
            ex = json.loads(line)
            messages = ex["messages"]
            toks = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            if len(toks) > seqlen:
                toks = toks[:seqlen]
            samples.append(toks)
    return samples


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train_and_measure(
    model,
    bank: dict,
    lti_bank: list[LTIInjection],
    loop_idx_ref: list[int],
    T_ref: list[int],
    train_samples: list[list[int]],
    n_steps: int,
):
    """Runs n_steps of real GSM8K CE loss training, T_ref[0] set to 3.
    Records rho at each step; records log_A/log_dt pre/post for K-FULL-C.
    Returns (rho_per_step, grad_v_max_first, grad_o_max_first, log_A_delta, log_dt_delta).
    """
    T_ref[0] = 3
    loop_idx_ref[0] = 0

    # Collect trainable module tree.
    class TrainBundle(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lti = lti_bank
            self.lora_v = [[bank[ell]["v"][t] for ell in range(LOOP_START, LOOP_END)]
                           for t in range(N_LOOPS)]
            self.lora_o = [[bank[ell]["o"][t] for ell in range(LOOP_START, LOOP_END)]
                           for t in range(N_LOOPS)]

    bundle = TrainBundle()

    # Snapshot log_A and log_dt for movement check.
    log_A_init = [mx.array(lti.log_A) for lti in lti_bank]
    log_dt_init = [mx.array(lti.log_dt) for lti in lti_bank]

    def ce_loss_fn(_bundle, input_ids):
        # input_ids: (B, L)
        # Teacher-forcing next-token CE on [1:] given [:-1].
        logits = model(input_ids[:, :-1])  # (B, L-1, vocab)
        targets = input_ids[:, 1:]
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    opt = optim.AdamW(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(bundle, ce_loss_fn)

    rho_per_step = []
    grad_v_max_first = None
    grad_o_max_first = None

    for step in range(n_steps):
        # pick sample (cycling)
        toks = train_samples[step % len(train_samples)]
        if len(toks) < 8:
            continue
        x = mx.array([toks], dtype=mx.int32)  # (1, L)
        loss, grads = loss_and_grad(bundle, x)

        # Extract max|grad| on v and o LoRA B for first step (K-FULL-B).
        if step == 0:
            vmax = 0.0
            omax = 0.0
            # grads mirrors bundle tree: grads["lora_v"][t][ell_idx]["B"], etc.
            gv = grads.get("lora_v", [])
            go = grads.get("lora_o", [])
            for t in range(len(gv)):
                for ell_idx in range(len(gv[t])):
                    gB = gv[t][ell_idx].get("B")
                    if gB is not None:
                        vmax = max(vmax, mx.max(mx.abs(gB)).item())
            for t in range(len(go)):
                for ell_idx in range(len(go[t])):
                    gB = go[t][ell_idx].get("B")
                    if gB is not None:
                        omax = max(omax, mx.max(mx.abs(gB)).item())
            grad_v_max_first = vmax
            grad_o_max_first = omax

        opt.update(bundle, grads)
        mx.eval(bundle.parameters(), opt.state, loss)

        rho = max(mx.max(mx.abs(lti.get_A())).item() for lti in lti_bank)
        rho_per_step.append(rho)

        if step % 20 == 0:
            print(f"  step {step} loss={loss.item():.4f} rho={rho:.6f}", flush=True)

    # Final movement check.
    dlog_A = max(
        mx.max(mx.abs(lti.log_A - log_A_init[i])).item()
        for i, lti in enumerate(lti_bank)
    )
    dlog_dt = max(
        mx.max(mx.abs(lti.log_dt - log_dt_init[i])).item()
        for i, lti in enumerate(lti_bank)
    )

    return rho_per_step, grad_v_max_first, grad_o_max_first, dlog_A, dlog_dt


# ─────────────────────────────────────────────
# GSM8K eval (non-cached forward, teacher-forced answer extraction via greedy gen)
# ─────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"\b\d+\.?\d*\b", text.replace(",", ""))
    return nums[-1] if nums else None


def gsm8k_greedy(
    model,
    tokenizer,
    loop_idx_ref: list[int],
    T_ref: list[int],
    T: int,
    n_eval: int,
    max_tokens: int = MAX_EVAL_TOKENS,
) -> tuple[float, int, int]:
    """Greedy generation evaluation at the given loop count T.

    Uses a minimal uncached loop (slow but correct) because the patched
    forward replays layers 12..20 multiple times per forward, which the
    stock mlx_lm cache-based generate() cannot handle without care.
    """
    T_ref[0] = T
    correct = 0
    total = 0
    with GSM8K_VALID.open() as f:
        for line in f:
            if total >= n_eval:
                break
            ex = json.loads(line)
            messages = ex["messages"]
            user_msg = [m for m in messages if m["role"] == "user"][0]
            gt_ans = extract_gsm8k_answer(messages[-1]["content"] or "")
            if gt_ans is None:
                continue
            prompt_toks = tokenizer.apply_chat_template(
                [user_msg], tokenize=True, add_generation_prompt=True
            )
            prompt = mx.array([prompt_toks], dtype=mx.int32)

            # Greedy no-cache gen: regenerate the full context each step.
            # Slow but safe under the recurrent-depth patch.
            gen = prompt
            for _ in range(max_tokens):
                loop_idx_ref[0] = 0
                logits = model(gen)
                nxt = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(nxt)
                nxt_id = int(nxt.item())
                gen = mx.concatenate([gen, nxt[:, None]], axis=1)
                if nxt_id == tokenizer.eos_token_id:
                    break
            full = tokenizer.decode(gen[0].tolist()[len(prompt_toks):])
            pred = extract_gsm8k_answer(full)
            if pred is not None and pred == gt_ans:
                correct += 1
            total += 1
    return (100.0 * correct / max(total, 1)), correct, total


# ─────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    mx.random.seed(SEED)

    print("[phase1] loading base Gemma 4 E4B 4-bit...", flush=True)
    model, tokenizer = load(MODEL_ID)
    model.freeze()

    print("[phase1] wiring loop-LoRA on layers 12..20 v_proj+o_proj...", flush=True)
    key = mx.random.key(SEED)
    bank, loop_idx_ref = wire_loop_lora(model, N_LOOPS, LORA_RANK, LORA_ALPHA, key)

    lti_bank = [LTIInjection(2560) for _ in range(N_LOOPS)]
    T_ref = [3]

    # Install the patched forward.
    install_loop_patch(model, lti_bank, loop_idx_ref, T_ref)

    # ── K-FULL-A: structural check ──
    a_ok = True
    for ell in range(LOOP_START, LOOP_END):
        attn = model.language_model.model.layers[ell].self_attn
        if not isinstance(attn.v_proj, LoopLoRALinear):
            a_ok = False
            break
        if not isinstance(attn.o_proj, LoopLoRALinear):
            a_ok = False
            break

    # K1743 — init orthogonality (carry-over structural check, free)
    cos_map = measure_init_orthogonality(bank)
    max_cos = max(cos_map.values()) if cos_map else 0.0

    # ── Tokenize training data ──
    print(f"[phase2] tokenizing GSM8K train (max {N_STEPS} samples, seqlen {MAX_SEQ})...", flush=True)
    train_samples = load_gsm8k_train(tokenizer, N_STEPS, MAX_SEQ)
    print(f"  got {len(train_samples)} samples", flush=True)

    # ── K-FULL-B + K-FULL-C: train and record dynamics ──
    print(f"[phase3] training {N_STEPS} steps on real GSM8K CE loss (T=3)...", flush=True)
    rhos, gv_max, go_max, dlogA, dlogdt = train_and_measure(
        model, bank, lti_bank, loop_idx_ref, T_ref, train_samples, N_STEPS
    )
    max_rho = max(rhos) if rhos else 1.0
    b_v_ok = gv_max is not None and gv_max > 1e-6
    b_o_ok = go_max is not None and go_max > 1e-6
    c_rho_ok = max_rho < 1.0
    c_move_ok = dlogA > 1e-4 and dlogdt > 1e-4

    # ── K1740 / K1742: GSM8K eval at T=3 and T-sweep ──
    eval_results: dict = {}
    base_acc = None
    acc_t3 = None
    acc_by_t = {}
    if N_EVAL_T3 <= 0:
        print("[phase4-6] eval skipped (N_EVAL_T3=0); K1740/K1741/K1742 remain not_measured", flush=True)
    try:
        if N_EVAL_T3 <= 0:
            raise RuntimeError("eval_skipped")
        print(f"[phase4] eval base Gemma 4 on GSM8K-valid (T=1 fixed, n={N_EVAL_T3})...", flush=True)
        # baseline == T=1 with ZERO B (which LoRA B is init=0) — but it has trained!
        # For an honest base, temporarily zero LoRA B's and set T=1.
        saved_Bs_v = {}
        saved_Bs_o = {}
        for ell, projs in bank.items():
            for pname, deltas in projs.items():
                for t, d in enumerate(deltas):
                    key = (ell, pname, t)
                    saved_Bs_v[key] = d.B
                    d.B = mx.zeros_like(d.B)
        # Also zero LTI log_A to keep ρ=exp(-1)=0.368 effectively neutralizing loop.
        saved_logA = [mx.array(lti.log_A) for lti in lti_bank]
        saved_logdt = [mx.array(lti.log_dt) for lti in lti_bank]
        # For base, set T=1 (effectively single pass through the block, LoRA=0).
        acc_t1_base, c1, n1 = gsm8k_greedy(model, tokenizer, loop_idx_ref, T_ref, T=1, n_eval=N_EVAL_T3)
        base_acc = acc_t1_base
        eval_results["base_T1_acc"] = {"pct": acc_t1_base, "correct": c1, "n": n1}
        # Restore trained params.
        for ell, projs in bank.items():
            for pname, deltas in projs.items():
                for t, d in enumerate(deltas):
                    d.B = saved_Bs_v[(ell, pname, t)]
        for i, lti in enumerate(lti_bank):
            lti.log_A = saved_logA[i]
            lti.log_dt = saved_logdt[i]

        print(f"[phase5] eval trained loop-LoRA T=3 (n={N_EVAL_T3})...", flush=True)
        acc_t3, c3, n3 = gsm8k_greedy(model, tokenizer, loop_idx_ref, T_ref, T=3, n_eval=N_EVAL_T3)
        eval_results["loop_T3_acc"] = {"pct": acc_t3, "correct": c3, "n": n3}

        print(f"[phase6] T-sweep at n={N_EVAL_PER_T} per T ∈ {EVAL_T_VALUES}...", flush=True)
        for T in EVAL_T_VALUES:
            accT, cT, nT = gsm8k_greedy(
                model, tokenizer, loop_idx_ref, T_ref, T=T, n_eval=N_EVAL_PER_T
            )
            acc_by_t[T] = {"pct": accT, "correct": cT, "n": nT}
            print(f"  T={T}: {accT:.1f}% ({cT}/{nT})", flush=True)
    except Exception as e:  # honest failure capture
        eval_results["error"] = f"{type(e).__name__}: {e}"
        print(f"[eval-error] {eval_results['error']}", flush=True)

    # K1740: +5pp vs base at n>=200 full eval
    k1740_status = "not_measured"
    if base_acc is not None and acc_t3 is not None:
        if N_EVAL_T3 >= 200:
            k1740_status = "pass" if (acc_t3 - base_acc) >= 5.0 else "fail"
        else:
            k1740_status = "under_powered"  # measured but below n=200; PROVISIONAL
    # K1741: MMLU — deferred follow-up
    k1741_status = "not_measured"
    # K1742: R^2 fit on T∈{1..6}
    k1742_status = "not_measured"
    r2 = None
    if acc_by_t and len(acc_by_t) == 6:
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            Ts = np.array(list(acc_by_t.keys()), dtype=float)
            ys = np.array([acc_by_t[T]["pct"] for T in acc_by_t], dtype=float)

            def sat_exp(t, y_inf, y0, tau):
                return y_inf - (y_inf - y0) * np.exp(-t / max(tau, 1e-6))

            popt, _ = curve_fit(
                sat_exp, Ts, ys, p0=[max(ys), min(ys), 2.0], maxfev=5000
            )
            yhat = sat_exp(Ts, *popt)
            ss_res = float(np.sum((ys - yhat) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            if N_EVAL_PER_T >= 30:
                k1742_status = "pass" if r2 > 0.90 else "fail"
            else:
                k1742_status = "under_powered"
        except Exception as e:
            eval_results["r2_error"] = f"{type(e).__name__}: {e}"
    # ── Verdict ──
    struct_dynamic_pass = a_ok and b_v_ok and b_o_ok and c_rho_ok and c_move_ok
    target_pass = (k1740_status == "pass") and (k1742_status == "pass") and (k1741_status == "pass")
    target_unmeasured = any(s in ("not_measured", "under_powered") for s in [k1740_status, k1741_status, k1742_status])

    if struct_dynamic_pass and target_pass:
        verdict = "SUPPORTED"
    elif struct_dynamic_pass and target_unmeasured:
        verdict = "PROVISIONAL"  # F#673: structural pass, target not_measured
    else:
        verdict = "KILLED"

    all_pass = struct_dynamic_pass and target_pass

    out = {
        "experiment_id": "exp_rdt_loop_lora_gemma4_full",
        "is_smoke": False,
        "verdict": verdict,
        "all_pass": bool(all_pass),
        "preemptive": False,
        "executed": True,
        "elapsed_sec": round(time.time() - t0, 2),
        "mlx_version": "0.31.1",
        "mlx_lm_version": "0.31.2",
        "seed": SEED,
        "config": {
            "model": MODEL_ID,
            "loop_layers": [LOOP_START, LOOP_END - 1],
            "n_loops": N_LOOPS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "n_steps": N_STEPS,
            "n_eval_t3": N_EVAL_T3,
            "n_eval_per_t": N_EVAL_PER_T,
            "t_sweep": EVAL_T_VALUES,
            "max_seq": MAX_SEQ,
            "max_eval_tokens": MAX_EVAL_TOKENS,
        },
        "kill_criteria": {
            "K1743": {
                "desc": "max |cos(A_t_i, A_t_j)| < 0.1 at init across projections/layers",
                "max_abs_cos": max_cos,
                "threshold": 0.1,
                "result": "pass" if max_cos < 0.1 else "fail",
            },
            "K-FULL-A": {
                "desc": "Real block integration: v_proj and o_proj are LoopLoRALinear on layers 12..20",
                "result": "pass" if a_ok else "fail",
            },
            "K-FULL-B": {
                "desc": "max|dL/dB_v| > 1e-6 AND max|dL/dB_o| > 1e-6 on first batch",
                "grad_v_max": gv_max,
                "grad_o_max": go_max,
                "result": "pass" if (b_v_ok and b_o_ok) else "fail",
            },
            "K-FULL-C": {
                "desc": f"max rho(A_d) < 1 over {N_STEPS} real-loss steps AND |dlog_A|,|dlog_dt| > 1e-4",
                "max_rho_over_steps": max_rho,
                "rho_first_step": rhos[0] if rhos else None,
                "rho_last_step": rhos[-1] if rhos else None,
                "dlog_A_max": dlogA,
                "dlog_dt_max": dlogdt,
                "result": "pass" if (c_rho_ok and c_move_ok) else "fail",
            },
            "K1740": {
                "desc": ">=+5pp GSM8K-Hard at T=3 vs base, n>=200",
                "base_acc": base_acc,
                "loop_T3_acc": acc_t3,
                "delta_pp": (None if (base_acc is None or acc_t3 is None) else round(acc_t3 - base_acc, 2)),
                "n_eval": N_EVAL_T3,
                "threshold_pp": 5.0,
                "result": k1740_status,
                "reason": (
                    "eval n < 200; PROVISIONAL per F#673"
                    if k1740_status == "under_powered"
                    else ("not evaluated" if k1740_status == "not_measured" else "measured")
                ),
            },
            "K1741": {
                "desc": "|ΔMMLU| <= 1pp vs base",
                "result": k1741_status,
                "reason": "MMLU (57 subjects) deferred to follow-up exp_rdt_loop_mmlu_eval per MATH.md §Theorem 4",
            },
            "K1742": {
                "desc": "saturating-exp fit R^2 > 0.90 on T in {1..6}",
                "acc_by_t": acc_by_t,
                "r_squared": r2,
                "n_per_t": N_EVAL_PER_T,
                "result": k1742_status,
                "reason": (
                    "n_per_t < 30; PROVISIONAL" if k1742_status == "under_powered"
                    else ("not evaluated" if k1742_status == "not_measured" else "measured")
                ),
            },
        },
        "eval_results": eval_results,
        "antipatterns_flagged": [],
        "notes": (
            "Real monkey-patch of Gemma4TextModel.__call__ for layers 12..20 recurrent "
            f"block. No surrogate. Training used real GSM8K CE loss on {N_STEPS} samples. "
            "Target KCs (K1740/K1741/K1742) measured at reduced n per MATH §Theorem 4; "
            "PROVISIONAL verdict if structural+dynamical pass with target under-powered "
            "or not measured (F#673)."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"K-FULL-A={a_ok} K-FULL-B v_max={gv_max} o_max={go_max} "
        f"K-FULL-C rho_max={max_rho:.4f} dlogA={dlogA:.6f} dlogdt={dlogdt:.6f}"
    )
    print(f"verdict={verdict} all_pass={all_pass} elapsed={out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
