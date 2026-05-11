#!/usr/bin/env python3
"""
Joint-Stiefel multi-adapter training: K=7 adapters with joint B orthogonality.

Trains K=7 PoLAR adapters with joint Stiefel constraint B_all B_all^T = I_{Kr}
across all layers. Tests whether the mathematical guarantee of zero cross-
contribution translates to behavioral non-interference.

Phases:
  A: Independent single-adapter training (3 main domains) for K1 baselines
  B: Joint training of K=7 adapters with round-robin + joint Stiefel retraction
  C: Evaluation: per-adapter accuracy, orthogonality, cross-contribution, composition

Kill criteria (pre-registered in MATH.md):
  K1: Joint-trained accuracy >= independent - 5pp per adapter (3 main domains)
  K2: ||B_all B_all^T - I_Kr||_F <= 1e-3 averaged over layers
  K3: Cross-contribution perturbation <= 1% per adapter (NLL-based)
  K4: Composed accuracy >= TIES-B baseline (71.3%) on 3-bench average
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.polar_train import (
    PoLARLinear, _get_layers, inject_polar_adapters,
    tokenize_record, collate, loss_fn, _grad_clip,
    retract_all, eval_gsm8k, eval_humaneval, eval_medqa, cleanup,
    SEED, RANK, SCALE, LR, GRAD_CLIP, BATCH_SIZE,
)

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
K = 7
N_TRAIN = 50 if IS_SMOKE else 500
N_STEPS = 20 if IS_SMOKE else 300
N_EVAL = 5 if IS_SMOKE else 50
RETRACT_A_EVERY = 20

DOMAIN_NAMES = ["math", "code", "medical", "finance", "legal", "biology", "physics"]
MAIN_DOMAINS = ["math", "code", "medical"]
MMLU_SUBJECTS = {
    "finance": "high_school_macroeconomics",
    "legal": "professional_law",
    "biology": "anatomy",
    "physics": "conceptual_physics",
}


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_domain_records(domain: str, n_train: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    if domain == "math":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n_train, len(ds))))
        return [{"messages": [
            {"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"},
            {"role": "assistant", "content": ex["answer"]},
        ]} for ex in ds]

    elif domain == "code":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n_train, len(ds))))
        return [{"messages": [
            {"role": "user", "content": ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")},
            {"role": "assistant", "content": ex["output"]},
        ]} for ex in ds if ex.get("output")]

    elif domain == "medical":
        ds = load_dataset("openlifescienceai/medmcqa", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n_train * 2, len(ds))))
        letters = "ABCD"
        records = []
        for ex in ds:
            if ex.get("cop") is None or not (0 <= ex["cop"] <= 3):
                continue
            choices = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
            q = ex["question"] + "\n" + "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
            records.append({"messages": [
                {"role": "user", "content": f"Answer with just the letter.\n\n{q}"},
                {"role": "assistant", "content": f"({letters[ex['cop']]})"},
            ]})
            if len(records) >= n_train:
                break
        return records

    else:
        subject = MMLU_SUBJECTS[domain]
        try:
            ds = load_dataset("cais/mmlu", subject, split="auxiliary_train")
        except Exception:
            ds = load_dataset("cais/mmlu", subject, split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n_train, len(ds))))
        letters = "ABCD"
        return [{"messages": [
            {"role": "user", "content": ex["question"] + "\n" + "\n".join(
                f"({l}) {c}" for l, c in zip(letters, ex["choices"])) + "\n\nAnswer with just the letter."},
            {"role": "assistant", "content": f"({letters[ex['answer']]})"},
        ]} for ex in ds]


def load_all_domain_data(n_train: int, seed: int) -> dict[str, list[dict]]:
    print("Loading training data for 7 domains...", flush=True)
    data = {}
    for domain in DOMAIN_NAMES:
        data[domain] = load_domain_records(domain, n_train, seed)
        print(f"  {domain}: {len(data[domain])} records", flush=True)
    return data


# ─────────────────────────────────────────────
# JointPoLARLinear module
# ─────────────────────────────────────────────

class JointPoLARLinear(nn.Module):
    def __init__(self, base_linear: nn.Module, num_adapters: int,
                 rank: int, scale: float, seed: int = SEED):
        super().__init__()
        self.base = base_linear
        self.K = num_adapters
        self.rank = rank
        self.scale = scale
        self.active_k = 0
        self._compose_mode = False
        self._composed_delta = None

        if hasattr(base_linear, "group_size"):
            d_out = base_linear.weight.shape[0]
            d_in = base_linear.scales.shape[1] * base_linear.group_size
        else:
            d_in = base_linear.weight.shape[1]
            d_out = base_linear.weight.shape[0]
        self.d_in, self.d_out = d_in, d_out

        rng = np.random.default_rng(seed)
        for k in range(num_adapters):
            rand_mat = rng.standard_normal((d_in, rank)).astype(np.float32)
            Q, _ = np.linalg.qr(rand_mat)
            setattr(self, f'a_{k}', mx.array(Q))
            setattr(self, f'b_{k}', mx.zeros((rank, d_out)))

    def __call__(self, x):
        if self._compose_mode and self._composed_delta is not None:
            return self.base(x) + self.scale * (x @ self._composed_delta)
        a = getattr(self, f'a_{self.active_k}')
        b = getattr(self, f'b_{self.active_k}')
        return self.base(x) + self.scale * ((x @ a) @ b)


def inject_joint_adapters(model, num_adapters: int, rank: int = RANK,
                          scale: float = SCALE, seed: int = SEED) -> list[JointPoLARLinear]:
    modules = []
    for layer in _get_layers(model):
        wrapped = JointPoLARLinear(layer.self_attn.q_proj, num_adapters=num_adapters,
                                   rank=rank, scale=scale, seed=seed)
        layer.self_attn.q_proj = wrapped
        modules.append(wrapped)
    return modules


def unwrap_adapters(model):
    for layer in _get_layers(model):
        qp = layer.self_attn.q_proj
        if hasattr(qp, 'base'):
            layer.self_attn.q_proj = qp.base


# ─────────────────────────────────────────────
# Joint Stiefel retraction
# ─────────────────────────────────────────────

def joint_retract_b(modules: list[JointPoLARLinear]) -> float:
    num_k = modules[0].K
    r = modules[0].rank
    Kr = num_k * r
    I_Kr = np.eye(Kr)
    max_dist = 0.0

    for m in modules:
        B_list = []
        for k in range(num_k):
            B_list.append(np.array(getattr(m, f'b_{k}')).astype(np.float64))
        B_all = np.vstack(B_list)

        norm = np.sqrt(np.sum(B_all ** 2))
        if not np.isfinite(norm) or norm < 1e-12:
            max_dist = max(max_dist, float(np.sqrt(Kr)))
            continue

        U, _, Vh = np.linalg.svd(B_all, full_matrices=False)
        B_ret = U @ Vh

        dist = float(np.sqrt(np.sum((B_ret @ B_ret.T - I_Kr) ** 2)))
        max_dist = max(max_dist, dist)

        for k in range(num_k):
            setattr(m, f'b_{k}', mx.array(B_ret[k * r:(k + 1) * r].astype(np.float32)))

    return max_dist


def retract_a_independent(modules: list[JointPoLARLinear]) -> float:
    r = modules[0].rank
    num_k = modules[0].K
    I_r = np.eye(r)
    max_dist = 0.0
    for m in modules:
        for k in range(num_k):
            A_np = np.array(getattr(m, f'a_{k}')).astype(np.float64)
            if not np.all(np.isfinite(A_np)) or np.sum(A_np ** 2) < 1e-12:
                continue
            W, _, Vh = np.linalg.svd(A_np, full_matrices=False)
            A_ret = W @ Vh
            setattr(m, f'a_{k}', mx.array(A_ret.astype(np.float32)))
            dist = float(np.sqrt(np.sum((A_ret.T @ A_ret - I_r) ** 2)))
            max_dist = max(max_dist, dist)
    return max_dist


def measure_joint_orthogonality(modules: list[JointPoLARLinear]) -> dict:
    num_k = modules[0].K
    r = modules[0].rank
    I_Kr = np.eye(num_k * r)
    dists = []
    for m in modules:
        B_list = [np.array(getattr(m, f'b_{k}')).astype(np.float64) for k in range(num_k)]
        B_all = np.vstack(B_list)
        gram = B_all @ B_all.T
        dist = float(np.sqrt(np.sum((gram - I_Kr) ** 2)))
        dists.append(dist)
    return {"avg": float(np.mean(dists)), "max": float(np.max(dists)), "per_layer": dists}


# ─────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────

def eval_mmlu(model, tokenizer, subject: str, n_eval: int = 50, seed: int = SEED) -> float:
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("cais/mmlu", subject, split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_eval, len(ds))))
    letters = "ABCD"
    correct = 0
    for ex in ds:
        choices_str = "\n".join(f"({l}) {c}" for l, c in zip(letters, ex["choices"]))
        msgs = [{"role": "user", "content": f"{ex['question']}\n{choices_str}\n\nAnswer with just the letter."}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)
        gt = letters[ex["answer"]]
        pred = response.strip().upper()
        pred_letter = next((l for l in "ABCD" if pred.startswith(l)), None)
        if not pred_letter:
            match = re.search(r"\b([ABCD])\b", pred)
            pred_letter = match.group(1) if match else None
        if pred_letter == gt:
            correct += 1
    return correct / len(ds) * 100


def eval_domain(model, tokenizer, domain: str, n_eval: int) -> float:
    if domain == "math":
        return eval_gsm8k(model, tokenizer, n_eval=n_eval, seed=SEED)
    elif domain == "code":
        return eval_humaneval(model, tokenizer, n_eval=n_eval)
    elif domain == "medical":
        return eval_medqa(model, tokenizer, n_eval=n_eval, seed=SEED)
    else:
        return eval_mmlu(model, tokenizer, MMLU_SUBJECTS[domain], n_eval=n_eval, seed=SEED)


# ─────────────────────────────────────────────
# Phase A: Independent training (baselines)
# ─────────────────────────────────────────────

def retract_b_only_single(modules: list[PoLARLinear]) -> float:
    I_r = np.eye(RANK)
    max_dist = 0.0
    for m in modules:
        B_np = np.array(m.lora_b).astype(np.float64)
        if not np.all(np.isfinite(B_np)) or np.sum(B_np ** 2) < 1e-12:
            continue
        W, _, Vh = np.linalg.svd(B_np, full_matrices=False)
        B_ret = W @ Vh
        m.lora_b = mx.array(B_ret.astype(np.float32))
        dist = float(np.sqrt(np.sum((B_ret @ B_ret.T - I_r) ** 2)))
        max_dist = max(max_dist, dist)
    return max_dist


def phase_a_independent(model, tokenizer, all_data, n_steps, n_eval) -> dict[str, float]:
    results = {}
    for domain in MAIN_DOMAINS:
        print(f"\n=== Phase A: Independent training — {domain} ===", flush=True)
        modules = inject_polar_adapters(model, rank=RANK, scale=SCALE, seed=SEED)
        mx.eval(model.parameters())

        optimizer = optim.Adam(learning_rate=LR)
        rng = np.random.default_rng(SEED)
        tokenized = [tokenize_record(tokenizer, r) for r in all_data[domain]]
        n_data = len(tokenized)
        grad_fn = nn.value_and_grad(model, loss_fn)

        for step in range(n_steps):
            idx = rng.choice(n_data, size=min(BATCH_SIZE, n_data), replace=(n_data < BATCH_SIZE))
            batch = [tokenized[j] for j in idx]
            inputs, labels_batch = collate(batch)

            loss, grads = grad_fn(model, inputs, labels_batch)
            grads = _grad_clip(grads, GRAD_CLIP)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            retract_b_only_single(modules)
            if (step + 1) % RETRACT_A_EVERY == 0:
                retract_all(modules)
            mx.eval(model.parameters())

            if step % 50 == 0:
                print(f"  [{domain} step {step}/{n_steps}] loss={float(loss.item()):.4f}", flush=True)

        retract_all(modules)
        mx.eval(model.parameters())

        print(f"  Evaluating {domain}...", flush=True)
        score = eval_domain(model, tokenizer, domain, n_eval)
        print(f"  {domain}: {score:.1f}%", flush=True)
        results[domain] = score

        unwrap_adapters(model)
        mx.eval(model.parameters())
        del optimizer, grad_fn
        gc.collect()
        mx.clear_cache()

    return results


# ─────────────────────────────────────────────
# Phase B: Joint training
# ─────────────────────────────────────────────

def phase_b_joint(model, tokenizer, all_data, n_steps_per_adapter, n_eval):
    print(f"\n=== Phase B: Joint training K={K} ===", flush=True)
    modules = inject_joint_adapters(model, num_adapters=K, rank=RANK, scale=SCALE, seed=SEED)
    mx.eval(model.parameters())

    total_steps = K * n_steps_per_adapter
    optimizer = optim.Adam(learning_rate=LR)
    rng = np.random.default_rng(SEED)
    grad_fn = nn.value_and_grad(model, loss_fn)

    tokenized = {}
    for dom_idx, domain in enumerate(DOMAIN_NAMES):
        tokenized[dom_idx] = [tokenize_record(tokenizer, r) for r in all_data[domain]]

    step_times = []
    losses_per_domain = {d: [] for d in DOMAIN_NAMES}

    for step in range(total_steps):
        k = step % K
        domain = DOMAIN_NAMES[k]
        for m in modules:
            m.active_k = k

        domain_data = tokenized[k]
        n_data = len(domain_data)
        idx = rng.choice(n_data, size=min(BATCH_SIZE, n_data), replace=(n_data < BATCH_SIZE))
        batch = [domain_data[j] for j in idx]
        inputs, labels_batch = collate(batch)

        t0 = time.perf_counter()
        loss, grads = grad_fn(model, inputs, labels_batch)
        grads = _grad_clip(grads, GRAD_CLIP)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        joint_retract_b(modules)
        if (step + 1) % RETRACT_A_EVERY == 0:
            retract_a_independent(modules)
        mx.eval(model.parameters())

        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            print(f"  [step {step}] DIVERGED loss={loss_val}", flush=True)
            losses_per_domain[domain].append(loss_val)
            break
        losses_per_domain[domain].append(loss_val)

        if step % (K * 50) == 0:
            orth = measure_joint_orthogonality(modules)
            print(f"  [step {step}/{total_steps}] domain={domain} loss={loss_val:.4f} "
                  f"orth_avg={orth['avg']:.2e}", flush=True)

    joint_retract_b(modules)
    retract_a_independent(modules)
    mx.eval(model.parameters())

    orth_final = measure_joint_orthogonality(modules)
    print(f"\nFinal orthogonality: avg={orth_final['avg']:.2e} max={orth_final['max']:.2e}", flush=True)

    # Eval each adapter individually
    print("\n--- Per-adapter evaluation ---", flush=True)
    joint_scores = {}
    for k, domain in enumerate(DOMAIN_NAMES):
        for m in modules:
            m.active_k = k
            m._compose_mode = False
        print(f"  Evaluating joint {domain}...", flush=True)
        score = eval_domain(model, tokenizer, domain, n_eval)
        print(f"  {domain}: {score:.1f}%", flush=True)
        joint_scores[domain] = score

    # Cross-contribution test (K3): NLL perturbation
    print("\n--- Cross-contribution test (K3) ---", flush=True)
    cross_perturbations = {}
    for k, domain in enumerate(DOMAIN_NAMES):
        test_records = all_data[domain][:min(10, len(all_data[domain]))]
        test_tokenized = [tokenize_record(tokenizer, r) for r in test_records]
        test_inputs, test_labels = collate(test_tokenized)

        for m in modules:
            m.active_k = k
            m._compose_mode = False
        nll_single = float(loss_fn(model, test_inputs, test_labels).item())

        for m in modules:
            delta = None
            for j in range(K):
                a_j = getattr(m, f'a_{j}')
                b_j = getattr(m, f'b_{j}')
                d = a_j @ b_j
                delta = d if delta is None else delta + d
            m._composed_delta = delta / K
            m._compose_mode = True
        mx.eval(*[m._composed_delta for m in modules])
        nll_composed = float(loss_fn(model, test_inputs, test_labels).item())

        for m in modules:
            m._compose_mode = False
            m._composed_delta = None

        perturbation = abs(nll_composed - nll_single) / max(abs(nll_single), 1e-8)
        cross_perturbations[domain] = {
            "nll_single": nll_single,
            "nll_composed": nll_composed,
            "perturbation_pct": perturbation * 100,
        }
        print(f"  {domain}: NLL single={nll_single:.4f} composed={nll_composed:.4f} "
              f"perturbation={perturbation * 100:.2f}%", flush=True)

    # Composition eval (K4): 3-bench average
    print("\n--- Composition eval (K4: 3-bench) ---", flush=True)
    for m in modules:
        delta = None
        for j in range(K):
            a_j = getattr(m, f'a_{j}')
            b_j = getattr(m, f'b_{j}')
            d = a_j @ b_j
            delta = d if delta is None else delta + d
        m._composed_delta = delta / K
        m._compose_mode = True
    mx.eval(*[m._composed_delta for m in modules])

    composed_gsm8k = eval_gsm8k(model, tokenizer, n_eval=n_eval, seed=SEED)
    print(f"  Composed GSM8K: {composed_gsm8k:.1f}%", flush=True)
    composed_humaneval = eval_humaneval(model, tokenizer, n_eval=n_eval)
    print(f"  Composed HumanEval: {composed_humaneval:.1f}%", flush=True)
    composed_medqa = eval_medqa(model, tokenizer, n_eval=n_eval, seed=SEED)
    print(f"  Composed MedQA: {composed_medqa:.1f}%", flush=True)
    composed_avg = (composed_gsm8k + composed_humaneval + composed_medqa) / 3
    print(f"  3-bench avg: {composed_avg:.1f}%", flush=True)

    for m in modules:
        m._compose_mode = False
        m._composed_delta = None

    avg_step = float(np.mean(step_times[10:])) if len(step_times) > 10 else float(np.mean(step_times))

    return {
        "joint_scores": joint_scores,
        "orthogonality": orth_final,
        "cross_perturbations": cross_perturbations,
        "composition": {
            "gsm8k": composed_gsm8k,
            "humaneval": composed_humaneval,
            "medqa": composed_medqa,
            "avg": composed_avg,
        },
        "avg_step_time_s": avg_step,
        "losses_final": {d: l[-1] if l else float("nan") for d, l in losses_per_domain.items()},
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def load_model():
    from mlx_lm import load
    print(f"Loading {MODEL_ID}...", flush=True)
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    return model, tokenizer


def main():
    t_start = time.time()
    print("=== Joint-Stiefel multi-adapter training (K=7) ===", flush=True)
    print(f"SMOKE={IS_SMOKE}, K={K}, N_TRAIN={N_TRAIN}, N_STEPS={N_STEPS}, N_EVAL={N_EVAL}", flush=True)

    all_data = load_all_domain_data(N_TRAIN, SEED)
    model, tokenizer = load_model()

    # Phase A
    print("\n" + "=" * 60, flush=True)
    print("PHASE A: Independent training baselines (3 main domains)", flush=True)
    print("=" * 60, flush=True)
    independent_scores = phase_a_independent(model, tokenizer, all_data, N_STEPS, N_EVAL)

    # Phase B
    print("\n" + "=" * 60, flush=True)
    print("PHASE B: Joint Stiefel training (K=7)", flush=True)
    print("=" * 60, flush=True)
    joint_results = phase_b_joint(model, tokenizer, all_data, N_STEPS, N_EVAL)

    cleanup(model, tokenizer)

    # Kill criteria
    print("\n" + "=" * 60, flush=True)
    print("KILL CRITERIA EVALUATION", flush=True)
    print("=" * 60, flush=True)

    # K1: Convergence
    k1_details = {}
    k1_all_pass = True
    for domain in MAIN_DOMAINS:
        delta = joint_results["joint_scores"][domain] - independent_scores[domain]
        passed = delta >= -5.0
        k1_details[domain] = {
            "joint": joint_results["joint_scores"][domain],
            "independent": independent_scores[domain],
            "delta_pp": delta, "pass": passed,
        }
        if not passed:
            k1_all_pass = False
        print(f"K1 {domain}: joint={joint_results['joint_scores'][domain]:.1f}% "
              f"independent={independent_scores[domain]:.1f}% "
              f"Δ={delta:+.1f}pp → {'PASS' if passed else 'FAIL'}", flush=True)

    # K2: Orthogonality
    k2_avg = joint_results["orthogonality"]["avg"]
    k2_pass = k2_avg <= 1e-3
    print(f"K2 ORTHOGONALITY: avg={k2_avg:.2e} (threshold=1e-3) → {'PASS' if k2_pass else 'FAIL'}",
          flush=True)

    # K3: Cross-contribution
    k3_all_pass = True
    for domain in DOMAIN_NAMES:
        pct = joint_results["cross_perturbations"][domain]["perturbation_pct"]
        passed = pct <= 1.0
        if not passed:
            k3_all_pass = False
        print(f"K3 {domain}: {pct:.2f}% → {'PASS' if passed else 'FAIL'}", flush=True)

    # K4: Composed accuracy
    composed_avg = joint_results["composition"]["avg"]
    k4_pass = composed_avg >= 71.3
    print(f"K4 COMPOSED: avg={composed_avg:.1f}% (threshold=71.3%) → {'PASS' if k4_pass else 'FAIL'}",
          flush=True)

    all_pass = k1_all_pass and k2_pass and k3_all_pass and k4_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"
    elapsed = time.time() - t_start
    print(f"\n=== VERDICT: {verdict} (elapsed: {elapsed / 60:.1f} min) ===", flush=True)

    results = {
        "verdict": verdict,
        "all_pass": all_pass,
        "config": {
            "model": MODEL_ID,
            "K": K, "rank": RANK, "scale": SCALE,
            "n_train_per_domain": N_TRAIN,
            "n_steps_per_adapter": N_STEPS,
            "n_eval": N_EVAL,
            "domains": DOMAIN_NAMES,
            "main_domains": MAIN_DOMAINS,
            "is_smoke": IS_SMOKE,
            "seed": SEED,
        },
        "independent_baselines": independent_scores,
        "joint_scores": joint_results["joint_scores"],
        "orthogonality": {
            "avg": joint_results["orthogonality"]["avg"],
            "max": joint_results["orthogonality"]["max"],
        },
        "cross_perturbations": joint_results["cross_perturbations"],
        "composition": joint_results["composition"],
        "kill_criteria": {
            "K1_convergence": {"pass": k1_all_pass, "per_domain": k1_details},
            "K2_orthogonality": {"pass": k2_pass, "avg_frobenius": k2_avg,
                                 "max_frobenius": joint_results["orthogonality"]["max"],
                                 "threshold": 1e-3},
            "K3_cross_contribution": {"pass": k3_all_pass,
                                      "per_domain": joint_results["cross_perturbations"]},
            "K4_composed_accuracy": {"pass": k4_pass, "composed_avg": composed_avg,
                                     "threshold": 71.3},
        },
        "timing": {
            "avg_step_time_s": joint_results["avg_step_time_s"],
            "total_elapsed_min": elapsed / 60,
        },
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
