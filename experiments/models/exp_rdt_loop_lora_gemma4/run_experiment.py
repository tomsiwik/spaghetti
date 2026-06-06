#!/usr/bin/env python3
"""exp_rdt_loop_lora_gemma4 — smoke-mode build.

Wires loop-indexed LoRA + LTI on frozen Gemma 4 E4B (layers 12–20).
Tests K1743 (partition-QR orthogonality at init) and K1744 (ρ(A)<1 across
50 Adam steps on synthetic loss). K1740/K1741/K1742 are full-scale and
deferred by design — see MATH.md §Theorem 3.

Pre-registered kill criteria (MATH.md):
  K1743: max |cos(A_t_i, A_t_j)| < 0.1 at init, across projections/layers.
  K1744: max ρ(A_d) < 1 during 50 training steps.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load

SEED = 42
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LOOP_START = 12
LOOP_END = 21  # exclusive → layers 12..20 (9 layers)
N_LOOPS = 6
LORA_RANK = 16
LORA_ALPHA = 2.0
N_STEPS = 50
BATCH = 2
SEQLEN = 32
HIDDEN = 2560
LR = 1e-3

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"


class LTIInjection(nn.Module):
    """Element-wise LTI: h' = A_d⊙h + B⊙e + transformer_out. F#667 port."""

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


def partition_qr_lora_A(in_dim: int, n: int, r: int, key) -> list[mx.array]:
    """Return [A_1,...,A_n] with A_i A_j^T = 0 (i≠j) by partition-QR on R^{in_dim × nr}."""
    W = mx.random.normal(shape=(in_dim, n * r), key=key)
    # MLX linalg.qr is CPU-only; route explicitly.
    Q, _ = mx.linalg.qr(W, stream=mx.cpu)
    Q = Q.astype(mx.float32)
    return [Q[:, (i * r):(i + 1) * r].T for i in range(n)]


class LoRADelta(nn.Module):
    """Additive LoRA delta. A fixed (Grassmannian per F#562); only B trains."""

    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, A_init: mx.array) -> None:
        super().__init__()
        self._A_fixed = A_init  # (r, in_dim) — stored under non-parameter name
        self.B = mx.zeros((out_dim, r))
        self.scale = float(alpha / r)

    @property
    def A(self) -> mx.array:
        return self._A_fixed

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., in_dim); returns (..., out_dim)
        z = x @ self._A_fixed.T  # (..., r)
        y = z @ self.B.T         # (..., out_dim)
        return self.scale * y


def build_loop_lora_bank(model, n_loops: int, r: int, alpha: float, key) -> dict:
    """Build per-loop LoRA deltas on v_proj and o_proj for layers 12..20."""
    bank: dict = {}
    layers = model.language_model.layers
    for ell in range(LOOP_START, LOOP_END):
        attn = layers[ell].self_attn
        # Unquantized logical dims:
        # v_proj: 2560 → num_kv_heads*head_dim = 512
        # o_proj: num_heads*head_dim = 2048 → 2560
        v_in, v_out = 2560, 512
        o_in, o_out = 2048, 2560
        key_v, key_o, key = mx.random.split(key, 3)
        As_v = partition_qr_lora_A(v_in, n_loops, r, key_v)
        As_o = partition_qr_lora_A(o_in, n_loops, r, key_o)
        bank[ell] = {
            "v": [LoRADelta(v_in, v_out, r, alpha, As_v[t]) for t in range(n_loops)],
            "o": [LoRADelta(o_in, o_out, r, alpha, As_o[t]) for t in range(n_loops)],
        }
    return bank


def measure_init_orthogonality(bank: dict, r: int) -> dict:
    """K1743: max |cos| across loop pairs within each projection/layer."""
    max_cos_per_proj: dict = {}
    for ell, projs in bank.items():
        for pname, deltas in projs.items():
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    Ai = deltas[i].A  # (r, in)
                    Aj = deltas[j].A
                    # row-wise cos then max over r×r pairs
                    Ain = Ai / (mx.linalg.norm(Ai, axis=1, keepdims=True) + 1e-12)
                    Ajn = Aj / (mx.linalg.norm(Aj, axis=1, keepdims=True) + 1e-12)
                    C = Ain @ Ajn.T  # (r, r)
                    mc = mx.max(mx.abs(C)).item()
                    key = f"L{ell}_{pname}"
                    max_cos_per_proj[key] = max(max_cos_per_proj.get(key, 0.0), mc)
    return max_cos_per_proj


def build_lti_bank(n_loops: int, hidden: int) -> list[LTIInjection]:
    return [LTIInjection(hidden) for _ in range(n_loops)]


def measure_rho_all(lti_bank: list[LTIInjection]) -> float:
    rhos = [mx.max(mx.abs(lti.get_A())).item() for lti in lti_bank]
    return max(rhos)


def train_smoke(lti_bank: list[LTIInjection], bank: dict, n_steps: int):
    """Synthetic reconstruction loss over a fake loop-block forward.
    Trains LTI + LoRA deltas so we exercise both sets of grads simultaneously.
    """
    key = mx.random.key(SEED)
    rho_max_per_step = []

    # Collect trainable modules as a container nn.Module so value_and_grad sees them.
    class TrainBundle(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lti = lti_bank
            # Flatten LoRA deltas; layout: per-loop-t lists of (v,o) per layer.
            self.lora_v = [[bank[ell]["v"][t] for ell in range(LOOP_START, LOOP_END)]
                           for t in range(N_LOOPS)]
            self.lora_o = [[bank[ell]["o"][t] for ell in range(LOOP_START, LOOP_END)]
                           for t in range(N_LOOPS)]

    bundle = TrainBundle()

    # Synthetic target: fixed random tensor.
    target = mx.random.normal(shape=(BATCH, SEQLEN, HIDDEN), key=mx.random.split(key, 1)[0])

    def loss_fn(m, h0):
        h = h0
        loss = mx.array(0.0)
        # Simulate T=3 loop iterations using a tiny surrogate of the block:
        # h_new = LTI(h, h0, lora_sum(h)).
        # lora_sum exercises all per-loop LoRA A/B across layers 12..20 for v_proj only
        # (o_proj shape mismatch with hidden — smoke-surrogate uses v_proj slice only).
        for t in range(3):
            # Aggregate LoRA-v delta across the 9 layers for loop t:
            # map R^{2560} → R^{2560} via slice-zero pad of the (B@A)x path's first 2560 dims.
            # Use v_proj A but pad B to 2560 out-dim for the surrogate.
            tfm_out = mx.zeros_like(h)
            for ell_idx in range(LOOP_END - LOOP_START):
                lora_v = m.lora_v[t][ell_idx]
                # shape check: A (r, 2560), B (512, r). For surrogate, map h (*,2560) →(*,512)
                # then zero-pad to 2560 so we stay in residual space.
                d512 = lora_v(h)  # (B, S, 512)
                tfm_out = tfm_out.at[..., :512].add(d512)
            h = m.lti[t](h, h0, tfm_out)
        return mx.mean((h - target) ** 2)

    opt = optim.AdamW(learning_rate=LR)
    loss_and_grad = nn.value_and_grad(bundle, loss_fn)

    for step in range(n_steps):
        h0 = mx.random.normal(shape=(BATCH, SEQLEN, HIDDEN), key=mx.random.split(key, step + 2)[0])
        loss, grads = loss_and_grad(bundle, h0)
        opt.update(bundle, grads)
        mx.eval(bundle.parameters(), opt.state, loss)
        rho = measure_rho_all(lti_bank)
        rho_max_per_step.append(rho)
    return rho_max_per_step


def phase1_load_and_wire():
    model, tok = load(MODEL_ID)
    # Freeze base explicitly.
    model.freeze()
    key = mx.random.key(SEED)
    bank = build_loop_lora_bank(model, N_LOOPS, LORA_RANK, LORA_ALPHA, key)
    lti_bank = build_lti_bank(N_LOOPS, HIDDEN)
    return model, bank, lti_bank


def main():
    t0 = time.time()
    mx.random.seed(SEED)

    # Phase 1: load model, wire adapters.
    model, bank, lti_bank = phase1_load_and_wire()
    mx.eval(*[d.A for ell in bank for d in bank[ell]["v"] + bank[ell]["o"]])

    # K1743 — measure at init.
    cos_map = measure_init_orthogonality(bank, LORA_RANK)
    max_cos = max(cos_map.values())
    k1743_pass = max_cos < 0.1

    # K1744 — 50 Adam steps on synthetic loss with LTI + LoRA in composition.
    # Drop the base model to free memory for training; we only need the adapter bank + LTI.
    del model
    gc.collect()
    mx.clear_cache()

    rhos = train_smoke(lti_bank, bank, N_STEPS)
    max_rho = max(rhos)
    k1744_pass = max_rho < 1.0

    # K1740/K1741/K1742: not measured (full-scale deferred).
    all_pass = k1743_pass and k1744_pass

    out = {
        "experiment_id": "exp_rdt_loop_lora_gemma4",
        "is_smoke": True,
        "verdict": "PROVISIONAL",
        "all_pass": bool(all_pass),
        "preemptive": False,
        "executed": True,
        "elapsed_sec": round(time.time() - t0, 2),
        "mlx_version": "0.31.1",
        "mlx_lm_version": "0.31.2",
        "seed": SEED,
        "config": {
            "model": MODEL_ID, "loop_layers": [LOOP_START, LOOP_END - 1],
            "n_loops": N_LOOPS, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "n_steps": N_STEPS, "batch": BATCH, "seqlen": SEQLEN, "hidden": HIDDEN,
        },
        "kill_criteria": {
            "K1743": {
                "desc": "max |cos(A_t_i, A_t_j)| < 0.1 at init across projections/layers",
                "max_abs_cos": max_cos,
                "per_proj_max": cos_map,
                "threshold": 0.1,
                "result": "pass" if k1743_pass else "fail",
            },
            "K1744": {
                "desc": "max rho(A_d) < 1 across 50 Adam training steps",
                "max_rho_over_steps": max_rho,
                "rho_first_step": rhos[0],
                "rho_last_step": rhos[-1],
                "threshold": 1.0,
                "result": "pass" if k1744_pass else "fail",
            },
            "K1740": {
                "desc": ">=+5pp GSM8K-Hard at T=3 vs base",
                "result": "not_measured",
                "reason": "full-scale; smoke mode. Follow-up required.",
            },
            "K1741": {
                "desc": "|ΔMMLU| <= 1pp vs base",
                "result": "not_measured",
                "reason": "full-scale; smoke mode. Follow-up required.",
            },
            "K1742": {
                "desc": "saturating-exp fit R^2>0.90 on T in {1..6}",
                "result": "not_measured",
                "reason": "full-scale; smoke mode. Follow-up required.",
            },
        },
        "antipatterns_flagged": [],
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"K1743 max|cos|={max_cos:.3e} (pass={k1743_pass})")
    print(f"K1744 max ρ(A)={max_rho:.6f} (pass={k1744_pass})")
    print(f"verdict=PROVISIONAL is_smoke=True elapsed={out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
