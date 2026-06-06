#!/usr/bin/env python3
"""exp_followup_answer_conditioned_ppl

Tests whether answer-only PPL correctly routes mixed-domain queries to their
domain expert (top-1 rank), where full-sequence PPL is expected to fail.

Builds directly on `answer_conditioned_scoring` (r=0.81 answer-PPL/accuracy)
and `ppl_vs_task_performance` (r=0.08 full-PPL/accuracy, KILLED).

See MATH.md for the pre-registered K1567 kill criterion:
    answer-only top-1 >= 0.85 AND full-seq top-1 < 0.85.

CPU-only. Reuses training/forward/PPL primitives from predecessor.

Usage:
    uv run python -m micro.models.exp_followup_answer_conditioned_ppl.run_experiment
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as onp

from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import (
    CharTokenizer,
    DOMAIN_GENERATORS,
    DOMAIN_DELIMITERS,
    compute_batched_per_token_losses,
    init_model,
    train_expert,
    train_model,
)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


def per_query_ppls(params, data_strings, data_encoded, delimiter, pad_id):
    """Per-query full-sequence PPL and answer-only PPL.

    Returns
    -------
    full_ppls : np.ndarray shape (N,)
    ans_ppls  : np.ndarray shape (N,)
    """
    token_data = compute_batched_per_token_losses(params, data_encoded, pad_id)
    full_ppls = []
    ans_ppls = []
    for i, (losses, mask) in enumerate(token_data):
        if len(losses) == 0:
            full_ppls.append(float("inf"))
            ans_ppls.append(float("inf"))
            continue

        total_loss = float(onp.sum(losses * mask))
        total_tokens = float(onp.sum(mask))
        full_ppls.append(
            math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        )

        s = data_strings[i]
        delim_pos = s.rfind(delimiter)
        if delim_pos < 0:
            ans_ppls.append(full_ppls[-1])
            continue

        a_loss = 0.0
        a_tokens = 0
        for t in range(delim_pos, len(losses)):
            if mask[t] > 0:
                a_loss += float(losses[t])
                a_tokens += 1
        ans_ppls.append(
            math.exp(a_loss / a_tokens) if a_tokens > 0 else float("inf")
        )

    return onp.array(full_ppls), onp.array(ans_ppls)


def run(seed=42, n_queries_per_domain=200, n_train=2000, n_train_expert=500,
        base_epochs=30, expert_epochs=40, d=32, H=2, L=2, max_T=48):
    """Run the full experiment, returning the results dict."""
    onp.random.seed(seed)
    random.seed(seed)
    t_start = time.time()

    tokenizer = CharTokenizer()
    V = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    domains = list(DOMAIN_GENERATORS.keys())
    D = len(domains)

    print(f"\n{'='*70}")
    print(f"  exp_followup_answer_conditioned_ppl | seed={seed}")
    print(f"  d={d}, H={H}, L={L}, V={V}, N_q={n_queries_per_domain}, D={D}")
    print(f"{'='*70}\n")

    # 1. Generate training data (same as predecessor)
    print("[1] Generating training data...")
    all_train_enc = []
    domain_train_enc = {}
    for domain in domains:
        gen = DOMAIN_GENERATORS[domain]
        strs = gen(n_train, random.Random(seed + hash(domain) % 1000))
        enc = [tokenizer.encode(s) for s in strs]
        domain_train_enc[domain] = enc
        all_train_enc.extend(enc)

    # 2. Train base
    print("[2] Training base model on all 5 domains...")
    params = init_model(V, d, H, L, max_T, seed)
    t0 = time.time()
    params = train_model(
        params, all_train_enc, pad_id,
        epochs=base_epochs, lr=0.001, batch_size=32, verbose=False,
    )
    base_params = {k: (v.copy() if k != "_config" else v) for k, v in params.items()}
    print(f"    base training: {time.time()-t0:.1f}s")

    # 3. Train 5 experts
    print("[3] Training 5 domain experts...")
    expert_params_by_domain = {}
    for domain in domains:
        t0 = time.time()
        delta = train_expert(
            base_params, domain_train_enc[domain], pad_id,
            epochs=expert_epochs, lr=0.001, batch_size=32, verbose=False,
        )
        # Apply delta to base to get expert params
        expert_theta = {
            k: (base_params[k].copy() if k != "_config" else base_params[k])
            for k in base_params
        }
        for k, v in delta.items():
            expert_theta[k] = expert_theta[k] + v
        expert_params_by_domain[domain] = expert_theta
        print(f"    expert '{domain}' trained in {time.time()-t0:.1f}s")

    # 4. Build held-out mixed-domain query set
    #    Uses a fresh RNG offset so queries are disjoint from training.
    print(f"[4] Generating {n_queries_per_domain} held-out queries per domain...")
    held_strs = {}
    held_encs = {}
    for domain in domains:
        gen = DOMAIN_GENERATORS[domain]
        held_rng = random.Random(seed + 999999 + hash(domain) % 10000)
        strs = gen(n_queries_per_domain, held_rng)
        enc = [tokenizer.encode(s) for s in strs]
        held_strs[domain] = strs
        held_encs[domain] = enc

    # 5. Routing eval
    #    For each (true_domain i, expert j): per-query full-PPL and ans-PPL.
    #    Tensor shape: [D_true, D_expert, N_q] each for full and ans.
    print("[5] Computing per-(query, expert) PPLs...")
    t0 = time.time()
    full_tensor = onp.zeros((D, D, n_queries_per_domain), dtype=onp.float64)
    ans_tensor = onp.zeros((D, D, n_queries_per_domain), dtype=onp.float64)

    for i, dom_true in enumerate(domains):
        strs = held_strs[dom_true]
        encs = held_encs[dom_true]
        delim = DOMAIN_DELIMITERS[dom_true]
        for j, dom_exp in enumerate(domains):
            theta_j = expert_params_by_domain[dom_exp]
            full_vec, ans_vec = per_query_ppls(theta_j, strs, encs, delim, pad_id)
            full_tensor[i, j, :] = full_vec
            ans_tensor[i, j, :] = ans_vec
    print(f"    PPL eval: {time.time()-t0:.1f}s")

    # 6. Per-query argmin (routing decision) under each metric
    # predicted_full[i, q] = argmin over j
    predicted_full = onp.argmin(full_tensor, axis=1)  # shape (D, N_q)
    predicted_ans = onp.argmin(ans_tensor, axis=1)

    # Top-1 accuracy
    correct_full_mat = (predicted_full == onp.arange(D)[:, None])  # (D, N_q)
    correct_ans_mat = (predicted_ans == onp.arange(D)[:, None])

    top1_full = float(correct_full_mat.mean())
    top1_ans = float(correct_ans_mat.mean())

    # Per-domain breakdown
    per_domain_top1_full = {dom: float(correct_full_mat[i].mean())
                            for i, dom in enumerate(domains)}
    per_domain_top1_ans = {dom: float(correct_ans_mat[i].mean())
                           for i, dom in enumerate(domains)}

    # Confusion matrices (5x5): rows = true domain, cols = predicted domain
    confusion_full = onp.zeros((D, D), dtype=onp.int64)
    confusion_ans = onp.zeros((D, D), dtype=onp.int64)
    for i in range(D):
        for q in range(n_queries_per_domain):
            confusion_full[i, int(predicted_full[i, q])] += 1
            confusion_ans[i, int(predicted_ans[i, q])] += 1

    # 7. Assess K1567
    K_THRESHOLD = 0.85
    pass_ans = top1_ans >= K_THRESHOLD
    pass_full_failed = top1_full < K_THRESHOLD
    k1567_pass = bool(pass_ans and pass_full_failed)

    verdict = "SUPPORTED" if k1567_pass else "KILLED"
    all_pass = k1567_pass

    runtime_s = time.time() - t_start

    # 8. Print summary
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  Top-1 (full-seq PPL routing):    {top1_full:.4f}")
    print(f"  Top-1 (answer-only PPL routing): {top1_ans:.4f}")
    print()
    print("  Per-domain Top-1 (answer-only):")
    for dom in domains:
        print(f"    {dom:12s}: {per_domain_top1_ans[dom]:.4f}")
    print()
    print("  Per-domain Top-1 (full-seq):")
    for dom in domains:
        print(f"    {dom:12s}: {per_domain_top1_full[dom]:.4f}")
    print()
    print(f"  K1567 pass_ans   (top1_ans  >= 0.85): {pass_ans}")
    print(f"  K1567 pass_full_failed (top1_full < 0.85): {pass_full_failed}")
    print(f"  K1567.pass = pass_ans AND pass_full_failed: {k1567_pass}")
    print(f"  Verdict: {verdict}")
    print(f"  Runtime: {runtime_s:.1f}s")
    print(f"{'='*70}\n")

    results = {
        "seed": seed,
        "n_queries_per_domain": n_queries_per_domain,
        "n_domains": D,
        "domains": domains,
        "top1_full": top1_full,
        "top1_ans": top1_ans,
        "per_domain_top1_full": per_domain_top1_full,
        "per_domain_top1_ans": per_domain_top1_ans,
        "confusion_full": confusion_full.tolist(),
        "confusion_ans": confusion_ans.tolist(),
        "kill_criteria": {
            "K1567": {
                "description": "answer-only top-1 >= 0.85 AND full-seq top-1 < 0.85",
                "top1_ans": top1_ans,
                "top1_full": top1_full,
                "pass_ans": bool(pass_ans),
                "pass_full_failed": bool(pass_full_failed),
                "pass": k1567_pass,
            }
        },
        "verdict": verdict,
        "all_pass": bool(all_pass),
        "is_smoke": False,
        "runtime_s": runtime_s,
    }
    return results


def main():
    results = run(seed=42, n_queries_per_domain=200)
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
