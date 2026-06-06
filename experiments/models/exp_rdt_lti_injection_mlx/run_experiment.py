#!/usr/bin/env python3
"""exp_rdt_lti_injection_mlx — LTI injection primitive in MLX.

Tests (pre-registered in MATH.md):
  K1737: ρ(A_discrete) < 1 across 1000 training steps
  K1738: MLX ↔ PyTorch cos > 0.9999 on 100 random inputs
  K1739: NaN-free at {log_A, log_dt} ∈ [-20, 20] extremes
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
import torch.nn as tnn

SEED = 42
DIM = 128
BATCH = 4
SEQLEN = 16
N_STEPS = 1000
N_PARITY = 100

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"


class LTIInjectionMLX(nn.Module):
    """MLX port of OpenMythos LTIInjection. Stable by construction."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_A = mx.zeros((dim,))
        self.log_dt = mx.zeros((1,))
        self.B = mx.full((dim,), 0.1)

    def get_A(self) -> mx.array:
        s = mx.clip(self.log_dt + self.log_A, -20.0, 20.0)
        return mx.exp(-mx.exp(s))

    def __call__(self, h: mx.array, e: mx.array, transformer_out: mx.array) -> mx.array:
        A = self.get_A()
        return A * h + self.B * e + transformer_out


class LTIInjectionTorch(tnn.Module):
    """Reference PyTorch impl mirroring OpenMythos main.py:643-686."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_A = tnn.Parameter(torch.zeros(dim))
        self.log_dt = tnn.Parameter(torch.zeros(1))
        self.B = tnn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ------------------------------------------------------------------- K1737 ---
def test_k1737_training_trajectory() -> dict:
    """1000 Adam steps; record ρ(A_d) every step. Pass iff no step violates < 1."""
    mx.random.seed(SEED)
    model = LTIInjectionMLX(DIM)
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(m, h, e, t_out, target):
        y = m(h, e, t_out)
        return ((y - target) ** 2).mean()

    grad_fn = nn.value_and_grad(model, loss_fn)

    rhos = []
    log_A_max_abs = []
    log_dt_history = []
    for step in range(N_STEPS):
        h = mx.random.normal((BATCH, SEQLEN, DIM))
        e = mx.random.normal((BATCH, SEQLEN, DIM))
        t_out = mx.random.normal((BATCH, SEQLEN, DIM))
        target = mx.random.normal((BATCH, SEQLEN, DIM))

        loss, grads = grad_fn(model, h, e, t_out, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        A = model.get_A()
        rho = float(mx.max(mx.abs(A)).item())
        rhos.append(rho)
        log_A_max_abs.append(float(mx.max(mx.abs(model.log_A)).item()))
        log_dt_history.append(float(model.log_dt.item()))

    max_rho = max(rhos)
    n_violations = sum(1 for r in rhos if r >= 1.0)
    return {
        "steps": N_STEPS,
        "max_rho": max_rho,
        "min_rho": min(rhos),
        "n_violations_rho_ge_1": n_violations,
        "final_max_abs_log_A": log_A_max_abs[-1],
        "final_log_dt": log_dt_history[-1],
        "max_abs_log_A_trajectory": max(log_A_max_abs),
        "max_abs_log_dt_trajectory": max(abs(x) for x in log_dt_history),
        "pass": n_violations == 0 and max_rho < 1.0,
    }


# ------------------------------------------------------------------- K1738 ---
def test_k1738_torch_parity() -> dict:
    """100 random triples (h, e, transformer_out); shared init; cos > 0.9999."""
    rng = np.random.default_rng(SEED)

    mlx_model = LTIInjectionMLX(DIM)
    torch_model = LTIInjectionTorch(DIM)

    # Align weights: start from zeros init which both use by default.
    # Perturb slightly to avoid degenerate (all-ones A) comparison.
    pert_log_A = rng.standard_normal(DIM).astype(np.float32) * 0.5
    pert_log_dt = rng.standard_normal(1).astype(np.float32) * 0.5
    pert_B = (rng.standard_normal(DIM).astype(np.float32) * 0.1).astype(np.float32)

    mlx_model.log_A = mx.array(pert_log_A)
    mlx_model.log_dt = mx.array(pert_log_dt)
    mlx_model.B = mx.array(pert_B)

    with torch.no_grad():
        torch_model.log_A.copy_(torch.from_numpy(pert_log_A))
        torch_model.log_dt.copy_(torch.from_numpy(pert_log_dt))
        torch_model.B.copy_(torch.from_numpy(pert_B))

    cos_sims = []
    abs_max_diffs = []
    for i in range(N_PARITY):
        h_np = rng.standard_normal((BATCH, SEQLEN, DIM)).astype(np.float32)
        e_np = rng.standard_normal((BATCH, SEQLEN, DIM)).astype(np.float32)
        t_np = rng.standard_normal((BATCH, SEQLEN, DIM)).astype(np.float32)

        mx_out = mlx_model(mx.array(h_np), mx.array(e_np), mx.array(t_np))
        mx.eval(mx_out)
        mx_np = np.array(mx_out)

        with torch.no_grad():
            t_out = torch_model(
                torch.from_numpy(h_np),
                torch.from_numpy(e_np),
                torch.from_numpy(t_np),
            ).numpy()

        a = mx_np.reshape(-1)
        b = t_out.reshape(-1)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos_sims.append(cos)
        abs_max_diffs.append(float(np.max(np.abs(mx_np - t_out))))

    min_cos = min(cos_sims)
    max_abs_diff = max(abs_max_diffs)
    return {
        "n_samples": N_PARITY,
        "min_cos": min_cos,
        "mean_cos": float(np.mean(cos_sims)),
        "max_abs_diff": max_abs_diff,
        "pass": min_cos > 0.9999,
    }


# ------------------------------------------------------------------- K1739 ---
def test_k1739_nan_freedom() -> dict:
    """4 corners of (log_A, log_dt) ∈ {-20, +20}^2. Forward + backward NaN-free."""
    corners = [(-20.0, -20.0), (-20.0, 20.0), (20.0, -20.0), (20.0, 20.0)]
    results = []
    all_nan_free = True
    for la_val, ldt_val in corners:
        model = LTIInjectionMLX(DIM)
        model.log_A = mx.full((DIM,), la_val)
        model.log_dt = mx.full((1,), ldt_val)
        model.B = mx.full((DIM,), 0.1)

        mx.random.seed(SEED)
        h = mx.random.normal((BATCH, SEQLEN, DIM))
        e = mx.random.normal((BATCH, SEQLEN, DIM))
        t_out = mx.random.normal((BATCH, SEQLEN, DIM))

        def loss_fn(m):
            return m(h, e, t_out).sum()

        grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = grad_fn(model)
        mx.eval(loss, grads)

        out = model(h, e, t_out)
        mx.eval(out)

        out_np = np.array(out)
        out_nan = bool(np.isnan(out_np).any() or np.isinf(out_np).any())

        grad_nan = False
        for name, g in grads.items():
            g_np = np.array(g)
            if np.isnan(g_np).any() or np.isinf(g_np).any():
                grad_nan = True

        loss_val = float(loss.item())
        loss_nan = not np.isfinite(loss_val)

        A = model.get_A()
        rho = float(mx.max(mx.abs(A)).item())

        corner_ok = not (out_nan or grad_nan or loss_nan)
        all_nan_free = all_nan_free and corner_ok
        results.append({
            "log_A": la_val,
            "log_dt": ldt_val,
            "rho": rho,
            "out_nan_or_inf": out_nan,
            "grad_nan_or_inf": grad_nan,
            "loss_nan_or_inf": loss_nan,
            "pass": corner_ok,
        })
    return {"corners": results, "pass": all_nan_free}


# ------------------------------------------------------------------- main ---
def main() -> int:
    t0 = time.time()
    print("[K1737] Spectral-radius trajectory (1000 training steps)…", flush=True)
    k1737 = test_k1737_training_trajectory()
    print(f"  K1737 max_rho = {k1737['max_rho']:.9f}  violations = {k1737['n_violations_rho_ge_1']}  pass = {k1737['pass']}")

    print("[K1738] MLX ↔ PyTorch parity (100 random triples)…", flush=True)
    k1738 = test_k1738_torch_parity()
    print(f"  K1738 min_cos = {k1738['min_cos']:.9f}  pass = {k1738['pass']}")

    print("[K1739] NaN-freedom at 4 extremes…", flush=True)
    k1739 = test_k1739_nan_freedom()
    for c in k1739["corners"]:
        print(f"  corner (log_A={c['log_A']}, log_dt={c['log_dt']}) rho={c['rho']:.6e}  pass={c['pass']}")
    print(f"  K1739 all_nan_free = {k1739['pass']}")

    all_pass = bool(k1737["pass"] and k1738["pass"] and k1739["pass"])
    verdict = "SUPPORTED" if all_pass else "KILLED"

    results = {
        "experiment": "exp_rdt_lti_injection_mlx",
        "seed": SEED,
        "dim": DIM,
        "batch": BATCH,
        "seqlen": SEQLEN,
        "n_steps": N_STEPS,
        "n_parity_samples": N_PARITY,
        "platform": "mlx-float32-gpu",
        "torch_version": torch.__version__,
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        "elapsed_sec": round(time.time() - t0, 2),
        "kill_criteria": {
            "K1737": k1737,
            "K1738": k1738,
            "K1739": k1739,
        },
        "all_pass": all_pass,
        "verdict": verdict,
        "is_smoke": False,
        "preemptive": False,
        "executed": True,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[done] verdict = {verdict}  all_pass = {all_pass}  elapsed = {results['elapsed_sec']}s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
