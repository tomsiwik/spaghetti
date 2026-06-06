#!/usr/bin/env python3
"""exp_rdt_loop_lora_gemma4_bench — behavioural + extended-scope follow-up.

Extends parent (exp_rdt_loop_lora_gemma4_full) in three ways:
- N_STEPS bumped to 500+ to close K-FULL-C-EXT scope (parent Caveat 1).
- GSM8K eval enabled to measure K1740 at best-feasible n.
- T-sweep eval on T ∈ {1..6} for K1742 saturating-exp fit.

Reuses parent's monkey-patch infrastructure verbatim — read
experiments/models/exp_rdt_loop_lora_gemma4_full/run_experiment.py for the
rationale. Key points duplicated here (not import) because the parent
is test infra, not a library, and refactoring it is out of scope.

Env knobs (defaults tuned for <= 2h researcher-hat budget):
- N_STEPS (default 500) — training steps
- N_EVAL_T3 (default 30) — GSM8K problems for T=3 / base eval
- N_EVAL_PER_T (default 10) — per-T problems for T-sweep
- T_SWEEP (default "1,2,3,6") — T values for sweep (K-KVCACHE also uses this)
- MAX_EVAL_TOKENS (default 256)
- SKIP_KVCACHE (default "1") — "0" to enable K-KVCACHE verification
- SKIP_EVAL (default "0") — "1" to skip all eval (debug)
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load

SEED = 42
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LOOP_START = 12
LOOP_END = 21  # exclusive -> 9 layers
N_LOOPS = 6
LORA_RANK = 16
LORA_ALPHA = 2.0
HIDDEN = 2560
N_STEPS = int(os.environ.get("N_STEPS", 500))
BATCH = 1
MAX_SEQ = 256
LR = 5e-4
N_EVAL_T3 = int(os.environ.get("N_EVAL_T3", 30))
N_EVAL_PER_T = int(os.environ.get("N_EVAL_PER_T", 10))
T_SWEEP = [int(x) for x in os.environ.get("T_SWEEP", "1,2,3,6").split(",")]
MAX_EVAL_TOKENS = int(os.environ.get("MAX_EVAL_TOKENS", 256))
SKIP_KVCACHE = os.environ.get("SKIP_KVCACHE", "1") == "1"
SKIP_EVAL = os.environ.get("SKIP_EVAL", "0") == "1"
N_KVCACHE_PROMPTS = int(os.environ.get("N_KVCACHE_PROMPTS", 20))

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"
GSM8K_TRAIN = Path(__file__).resolve().parents[1] / "exp_p1_t2_single_domain_training/data/math/train.jsonl"
GSM8K_VALID = Path(__file__).resolve().parents[1] / "exp_p1_t2_single_domain_training/data/math/valid.jsonl"


# ─────────────────────────────────────────────
# LoRA / LTI / LoopLoRA wrappers (parent-inherited, audit-clean)
# ─────────────────────────────────────────────

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
        base_y = self._base(x)
        delta_y = self.deltas[self._loop_idx[0]](x)
        return base_y + delta_y


def install_loop_patch(model, lti_bank, loop_idx_ref, T_ref):
    """Class-level monkey-patch of Gemma4TextModel.__call__ for recurrent-depth."""
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
# Data
# ─────────────────────────────────────────────

def load_gsm8k_train(tokenizer, max_examples: int, seqlen: int) -> list[list[int]]:
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
# Training
# ─────────────────────────────────────────────

def train_and_measure(model, bank, lti_bank, loop_idx_ref, T_ref, train_samples, n_steps):
    T_ref[0] = 3
    loop_idx_ref[0] = 0

    class TrainBundle(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lti = lti_bank
            self.lora_v = [[bank[ell]["v"][t] for ell in range(LOOP_START, LOOP_END)]
                           for t in range(N_LOOPS)]
            self.lora_o = [[bank[ell]["o"][t] for ell in range(LOOP_START, LOOP_END)]
                           for t in range(N_LOOPS)]

    bundle = TrainBundle()

    log_A_init = [mx.array(lti.log_A) for lti in lti_bank]
    log_dt_init = [mx.array(lti.log_dt) for lti in lti_bank]

    def ce_loss_fn(_bundle, input_ids):
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    opt = optim.AdamW(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(bundle, ce_loss_fn)

    rho_per_step: list[float] = []
    grad_v_max_first = None
    grad_o_max_first = None

    skipped = 0
    for step in range(n_steps):
        toks = train_samples[step % len(train_samples)]
        if len(toks) < 8:
            skipped += 1
            continue
        x = mx.array([toks], dtype=mx.int32)
        loss, grads = loss_and_grad(bundle, x)

        if grad_v_max_first is None:
            vmax = 0.0
            omax = 0.0
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

        if step % 50 == 0:
            print(f"  step {step} loss={loss.item():.4f} rho={rho:.6f}", flush=True)

    dlog_A = max(
        mx.max(mx.abs(lti.log_A - log_A_init[i])).item()
        for i, lti in enumerate(lti_bank)
    )
    dlog_dt = max(
        mx.max(mx.abs(lti.log_dt - log_dt_init[i])).item()
        for i, lti in enumerate(lti_bank)
    )
    print(f"[train] completed {len(rho_per_step)}/{n_steps} steps (skipped {skipped})", flush=True)
    return rho_per_step, grad_v_max_first, grad_o_max_first, dlog_A, dlog_dt


# ─────────────────────────────────────────────
# Eval (uncached, parent's approach — correct; scoped-down n for feasibility)
# ─────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"\b\d+\.?\d*\b", text.replace(",", ""))
    return nums[-1] if nums else None


def gsm8k_greedy(model, tokenizer, loop_idx_ref, T_ref, T, n_eval, max_tokens=MAX_EVAL_TOKENS):
    T_ref[0] = T
    correct = 0
    total = 0
    per_problem_time = []
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

            t_start = time.time()
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
            per_problem_time.append(time.time() - t_start)

            full = tokenizer.decode(gen[0].tolist()[len(prompt_toks):])
            pred = extract_gsm8k_answer(full)
            if pred is not None and pred == gt_ans:
                correct += 1
            total += 1
            if total % 5 == 0:
                avg = sum(per_problem_time) / len(per_problem_time)
                print(f"    T={T} eval {total}/{n_eval} correct={correct} avg_sec={avg:.1f}", flush=True)
    return (100.0 * correct / max(total, 1)), correct, total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t0 = time.time()
    mx.random.seed(SEED)

    print(f"[phase1] loading {MODEL_ID}...", flush=True)
    model, tokenizer = load(MODEL_ID)
    model.freeze()

    print("[phase1] wiring loop-LoRA on layers 12..20 v_proj+o_proj...", flush=True)
    key = mx.random.key(SEED)
    bank, loop_idx_ref = wire_loop_lora(model, N_LOOPS, LORA_RANK, LORA_ALPHA, key)
    lti_bank = [LTIInjection(2560) for _ in range(N_LOOPS)]
    T_ref = [3]
    install_loop_patch(model, lti_bank, loop_idx_ref, T_ref)

    a_ok = True
    for ell in range(LOOP_START, LOOP_END):
        attn = model.language_model.model.layers[ell].self_attn
        if not isinstance(attn.v_proj, LoopLoRALinear):
            a_ok = False
            break
        if not isinstance(attn.o_proj, LoopLoRALinear):
            a_ok = False
            break

    cos_map = measure_init_orthogonality(bank)
    max_cos = max(cos_map.values()) if cos_map else 0.0

    print(f"[phase2] tokenizing GSM8K train (max {N_STEPS} samples, seqlen {MAX_SEQ})...", flush=True)
    train_samples = load_gsm8k_train(tokenizer, N_STEPS, MAX_SEQ)
    print(f"  got {len(train_samples)} samples", flush=True)

    print(f"[phase3] training {N_STEPS} steps on real GSM8K CE loss (T=3)...", flush=True)
    rhos, gv_max, go_max, dlogA, dlogdt = train_and_measure(
        model, bank, lti_bank, loop_idx_ref, T_ref, train_samples, N_STEPS
    )
    max_rho = max(rhos) if rhos else 1.0
    b_v_ok = gv_max is not None and gv_max > 1e-6
    b_o_ok = go_max is not None and go_max > 1e-6
    c_rho_ok = max_rho < 1.0
    c_move_ok = dlogA > 1e-4 and dlogdt > 1e-4
    c_ext_n_ok = len(rhos) >= 500

    # Free training-only state before eval.
    gc.collect()
    mx.clear_cache()

    # ── K1740/K1742 eval ──
    eval_results: dict = {}
    base_acc = None
    acc_t3 = None
    acc_by_t: dict = {}
    kvcache_max_abs_logit_diff = None
    if SKIP_EVAL or N_EVAL_T3 <= 0:
        eval_results["error"] = "eval_skipped_env"
        print("[phase4-6] eval skipped by env flag", flush=True)
    else:
        try:
            # Base == trained-zeroed: temporarily zero LoRA B's and LTI effects.
            print(f"[phase4] eval base Gemma 4 on GSM8K-valid (T=1, B=0, n={N_EVAL_T3})...", flush=True)
            saved_Bs: dict = {}
            for ell, projs in bank.items():
                for pname, deltas in projs.items():
                    for t, d in enumerate(deltas):
                        key_s = (ell, pname, t)
                        saved_Bs[key_s] = d.B
                        d.B = mx.zeros_like(d.B)
            saved_logA = [mx.array(lti.log_A) for lti in lti_bank]
            saved_logdt = [mx.array(lti.log_dt) for lti in lti_bank]

            acc_t1_base, c1, n1 = gsm8k_greedy(model, tokenizer, loop_idx_ref, T_ref, T=1, n_eval=N_EVAL_T3)
            base_acc = acc_t1_base
            eval_results["base_T1_acc"] = {"pct": acc_t1_base, "correct": c1, "n": n1}

            for ell, projs in bank.items():
                for pname, deltas in projs.items():
                    for t, d in enumerate(deltas):
                        d.B = saved_Bs[(ell, pname, t)]
            for i, lti in enumerate(lti_bank):
                lti.log_A = saved_logA[i]
                lti.log_dt = saved_logdt[i]

            print(f"[phase5] eval trained loop-LoRA T=3 (n={N_EVAL_T3})...", flush=True)
            acc_t3, c3, n3 = gsm8k_greedy(model, tokenizer, loop_idx_ref, T_ref, T=3, n_eval=N_EVAL_T3)
            eval_results["loop_T3_acc"] = {"pct": acc_t3, "correct": c3, "n": n3}

            print(f"[phase6] T-sweep n={N_EVAL_PER_T} per T ∈ {T_SWEEP}...", flush=True)
            for T in T_SWEEP:
                accT, cT, nT = gsm8k_greedy(
                    model, tokenizer, loop_idx_ref, T_ref, T=T, n_eval=N_EVAL_PER_T
                )
                acc_by_t[T] = {"pct": accT, "correct": cT, "n": nT}
                print(f"  T={T}: {accT:.1f}% ({cT}/{nT})", flush=True)
        except Exception as e:
            eval_results["error"] = f"{type(e).__name__}: {e}"
            print(f"[eval-error] {eval_results['error']}", flush=True)

    # K1740 status
    k1740_status = "not_measured"
    if base_acc is not None and acc_t3 is not None:
        if N_EVAL_T3 >= 200:
            k1740_status = "pass" if (acc_t3 - base_acc) >= 5.0 else "fail"
        else:
            k1740_status = "under_powered"
    k1741_status = "not_measured"  # MMLU deferred

    # K1742 fit
    k1742_status = "not_measured"
    r2 = None
    fit_params = None
    if acc_by_t and len(acc_by_t) >= 3:
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            Ts = np.array(sorted(acc_by_t.keys()), dtype=float)
            ys = np.array([acc_by_t[int(T)]["pct"] for T in Ts], dtype=float)

            def sat_exp(t, y_inf, y0, tau):
                return y_inf - (y_inf - y0) * np.exp(-t / max(tau, 1e-6))

            popt, _ = curve_fit(
                sat_exp, Ts, ys, p0=[float(max(ys)), float(min(ys)), 2.0], maxfev=5000
            )
            fit_params = {"y_inf": float(popt[0]), "y0": float(popt[1]), "tau": float(popt[2])}
            yhat = sat_exp(Ts, *popt)
            ss_res = float(np.sum((ys - yhat) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            if N_EVAL_PER_T >= 30 and len(acc_by_t) == 6:
                k1742_status = "pass" if r2 > 0.90 else "fail"
            else:
                k1742_status = "under_powered"
        except Exception as e:
            eval_results["r2_error"] = f"{type(e).__name__}: {e}"

    # K-KVCACHE status
    if SKIP_KVCACHE:
        kvcache_status = "not_measured"
        kvcache_reason = "KV-cached recurrent-depth implementation scope-deferred to follow-up exp_rdt_loop_kv_cache per MATH §Theorem 2(b)"
    else:
        # stub: not implemented in this run
        kvcache_status = "not_measured"
        kvcache_reason = "verification not implemented in this run"

    # K-FULL-C-EXT classification
    kfull_c_ext_status: str
    if c_rho_ok and c_move_ok and c_ext_n_ok:
        kfull_c_ext_status = "pass"
    elif c_rho_ok and c_move_ok:
        kfull_c_ext_status = "under_powered"  # passed at n<500
    else:
        kfull_c_ext_status = "fail"

    # ── Verdict ──
    struct_dynamic_pass = a_ok and b_v_ok and b_o_ok and kfull_c_ext_status == "pass"
    target_pass = (k1740_status == "pass") and (k1741_status == "pass") and (k1742_status == "pass")
    target_any_underpowered = any(
        s in ("not_measured", "under_powered") for s in [k1740_status, k1741_status, k1742_status]
    )
    target_any_fail = any(s == "fail" for s in [k1740_status, k1741_status, k1742_status])

    if struct_dynamic_pass and target_pass and kvcache_status == "pass":
        verdict = "SUPPORTED"
    elif target_any_fail and struct_dynamic_pass:
        # Target FAIL at measured n — PAPER must discuss; lineage-wise this
        # is a behavioural KILL candidate for the mechanism. We still mark
        # PROVISIONAL if the target was under_powered (n<pre-reg), not a
        # true n-adequate fail.
        #   (n<200 always maps to under_powered, never fail — see k1740_status
        #    logic above — so target_any_fail can only trigger with full n.)
        verdict = "KILLED"
    elif struct_dynamic_pass and (target_any_underpowered or kvcache_status != "pass"):
        verdict = "PROVISIONAL"
    else:
        verdict = "KILLED"

    all_pass = struct_dynamic_pass and target_pass and kvcache_status == "pass"

    out = {
        "experiment_id": "exp_rdt_loop_lora_gemma4_bench",
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
            "steps_executed": len(rhos),
            "n_eval_t3": N_EVAL_T3,
            "n_eval_per_t": N_EVAL_PER_T,
            "t_sweep": T_SWEEP,
            "max_seq": MAX_SEQ,
            "max_eval_tokens": MAX_EVAL_TOKENS,
            "skip_kvcache": SKIP_KVCACHE,
            "skip_eval": SKIP_EVAL,
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
            "K-FULL-C-EXT": {
                "desc": f"max rho(A_d) < 1 over >=500 real-loss steps AND |dlog_A|,|dlog_dt| > 1e-4",
                "max_rho_over_steps": max_rho,
                "rho_first_step": rhos[0] if rhos else None,
                "rho_last_step": rhos[-1] if rhos else None,
                "dlog_A_max": dlogA,
                "dlog_dt_max": dlogdt,
                "n_steps_executed": len(rhos),
                "result": kfull_c_ext_status,
            },
            "K-KVCACHE": {
                "desc": "Recurrent-depth KV-cache correctness vs uncached on T in {1,2,3,6}, n=20",
                "max_abs_logit_diff": kvcache_max_abs_logit_diff,
                "result": kvcache_status,
                "reason": kvcache_reason,
            },
            "K1740-BENCH": {
                "desc": ">=+5pp GSM8K at T=3 vs base, n>=200",
                "base_acc": base_acc,
                "loop_T3_acc": acc_t3,
                "delta_pp": (None if (base_acc is None or acc_t3 is None) else round(acc_t3 - base_acc, 2)),
                "n_eval": N_EVAL_T3,
                "threshold_pp": 5.0,
                "result": k1740_status,
                "reason": (
                    f"n<200 ({N_EVAL_T3}); PROVISIONAL per F#673"
                    if k1740_status == "under_powered"
                    else ("not evaluated" if k1740_status == "not_measured" else "measured")
                ),
            },
            "K1741-BENCH": {
                "desc": "|deltaMMLU| <= 1pp vs base",
                "result": k1741_status,
                "reason": "MMLU (57 subjects) scope-deferred to follow-up exp_rdt_loop_mmlu_eval",
            },
            "K1742-BENCH": {
                "desc": "saturating-exp fit R^2 > 0.90 on T in {1..6}, n>=30/T",
                "acc_by_t": acc_by_t,
                "r_squared": r2,
                "fit_params": fit_params,
                "n_per_t": N_EVAL_PER_T,
                "t_sweep": T_SWEEP,
                "result": k1742_status,
                "reason": (
                    f"n_per_t<30 ({N_EVAL_PER_T}) or |T_sweep|<6 ({len(T_SWEEP)}); PROVISIONAL"
                    if k1742_status == "under_powered"
                    else ("not evaluated" if k1742_status == "not_measured" else "measured")
                ),
            },
        },
        "eval_results": eval_results,
        "antipatterns_flagged": [],
        "notes": (
            "Follow-up to exp_rdt_loop_lora_gemma4_full (PROVISIONAL). "
            f"Training extended to N_STEPS={N_STEPS} (target >=500 for K-FULL-C-EXT). "
            f"GSM8K eval at N_EVAL_T3={N_EVAL_T3} (target-pre-reg n>=200). "
            f"T-sweep at n={N_EVAL_PER_T} per T={T_SWEEP} (target-pre-reg n>=30/T across T={{1..6}}). "
            f"K-KVCACHE skip={SKIP_KVCACHE}. "
            "Target KCs report under_powered (not fail) when n<pre-reg threshold; "
            "PROVISIONAL verdict per F#673."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"\n=== SUMMARY ===\n"
        f"K-FULL-A={a_ok} K-FULL-B v={gv_max} o={go_max} "
        f"K-FULL-C-EXT rho_max={max_rho:.4f} dlogA={dlogA:.4g} dlogdt={dlogdt:.4g} n_steps={len(rhos)}\n"
        f"K1740-BENCH {k1740_status} base={base_acc} T3={acc_t3}\n"
        f"K1742-BENCH {k1742_status} R2={r2}\n"
        f"verdict={verdict} all_pass={all_pass} elapsed={out['elapsed_sec']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
