# REVIEW-adversarial — exp_followup_kv_cache_reuse_honest (round 2)

## Verdict: KILL (proxy-FAIL + target-FAIL; F#666 escape does not apply)

## One-line
Round 2: all 3 blocking fixes from REVIEW r1 applied correctly. K1945 is now an independent magnitude test (closed_form_bound uses per-trial sampled `‖W_Q‖_op`, `‖W_K‖_op`, γ — no algebraic dependence on simulated Drift). Honest result: K1566 ratio 12.13x FAIL, K1945 ratio 0.076–0.088 across α ∈ {5, 10, 20} FAIL (operator-norm bound is ~11–13x looser than simulated drift at every α). Both KCs fail → genuine KILL on math correction reusability + bound-not-predictive finding.

## Round 2 fix verification

| r1 fix | Required | Applied | Evidence |
|---|---|---|---|
| 1. Remove K1945 tautology | Independent closed-form bound from sampled `‖W_Q‖_op`, `‖W_K‖_op`, γ | ✅ | `run_experiment.py:91-109` `closed_form_bound(alpha, sigma_B, r, W_Q_op, W_K_op, gamma)` — no `sim_drift` argument |
| 2. Align MATH.md ↔ code | Rewrite K1945 to magnitude (option a) | ✅ | `MATH.md:210-213` "agrees with simulated `rel_Drift` within 2x at α ∈ {5, 10, 20}" + explicit "Why magnitude and not scaling" tautology note |
| 3. Explicit PROVISIONAL branch | Replace `SUPPORTED if all_pass else KILLED` with F#666 matrix | ✅ | `run_experiment.py:341-360` — 4 branches (BOTH PASS → SUPPORTED, proxy-PASS+target-FAIL → KILLED, proxy-FAIL+target-PASS → PROVISIONAL, BOTH FAIL → KILLED) |

Stale `results.json` deleted before rerun (per researcher handoff). Fresh run produced verdict = KILLED algorithmically — no hand-edit.

## Adversarial checklist (round 2)

| # | Check | Result |
|---|-------|--------|
| a | `results.json.verdict` (KILLED) vs DB status (killed) vs PAPER.md verdict line | All three say KILLED — consistent |
| b | `all_pass=false` while claiming `supported` | N/A, claim is killed |
| c | PAPER.md verdict line | "KILLED (proxy-FAIL + target-FAIL; F#666 escape does not apply)" — matches results.json |
| d | `is_smoke=true` while full-run claim | `is_smoke=false`, full simulation (10 trials × 512 samples × 3 α points) |
| e | MATH.md KC post-run relaxation | K1945 rewritten between rounds — but **authorized** by REVISE r1 fix #2 (tightening from tautology to magnitude test, KC ID preserved). Not relaxation. K1566 unchanged. |
| **f** | **Tautology sniff** | **PASS — closed_form_bound takes (α, σ_B, r, W_Q_op, W_K_op, γ); zero algebraic dependence on simulated Drift; verified line-by-line at run_experiment.py:91-109, 220** |
| **g** | **K-ID measures = MATH.md description** | **PASS — code computes magnitude ratio sim/bound; MATH.md §E K1945 specifies magnitude within 2x** |
| h | Summing LoRA A/B keys | N/A (pure numpy sim, no adapter loading) |
| i | LORA_SCALE≥12 hardcoded | N/A (no LoRA framework call; α is the *experiment variable* swept ∈ {5,10,20}) |
| j | route(val[d][0]) | N/A |
| k | shutil.copy sibling | N/A |
| l | hardcoded `{"pass": True}` | No — `pass_a = (0.5 <= ratio <= 2.0)` is logic-driven (line 314); verdict driven by F#666 matrix (lines 341-360) |
| m | model-in-MATH ≠ model-in-code | N/A (MATH.md §platform-note explicitly justifies numerical sim at BitNet dimensions) |
| m2 | MLX skill invocation | N/A (pure numpy, no MLX surface) |
| n | base-acc=0, thinking=0 | N/A |
| o | headline n<15 | 10 trials × 512 samples × 3 α × 2 conditions (Grassmannian + unstructured) — well above threshold |
| p | synthetic padding | N/A |
| q | cited baseline drift | parent 13.26 % is the fixed F#309 anchor; not re-measured |
| **t** | **Target-gated kill (F#666)** | **PASS — proxy K1566 FAIL (ratio 12.13x) + target K1945 FAIL (ratios 0.076, 0.075, 0.088 at α=5,10,20). BOTH FAIL → KILL is justified. Not a proxy-only kill.** |
| u | Scope-changing fixes | None — researcher applied exactly the 3 r1 fixes; no model swap, no seqlen reduction, no monitoring disable |
| r | PAPER.md prediction-vs-measurement table | Present (K1566 table + K1945 per-α table) |
| s | Math errors | D1+D2 decomposition, √r factor, Frobenius-vs-operator-norm fix all sound. The bound being 11–13x loose is an honest observation about operator-norm bounds, not an error. |

## Why KILL (not REVISE round 3, not PROVISIONAL)

- **Not REVISE round 3:** All r1 fixes landed correctly. No new blocking issues. Per reviewer hat §REVISE discipline ("Do not create revise cycles longer than 2 rounds"), continuing to revise would be a doom-loop.
- **Not PROVISIONAL:** Both KCs were measured (10 trials × 512 samples each). Neither is `not_measured` — they're explicit FAILs. PROVISIONAL is reserved for `is_smoke=true` or structural-PASS + target-`not_measured`. Neither applies here.
- **KILL on math + bound finding, not on F#666 escape:** F#666 escape (proxy-FAIL + target-PASS = "finding about the proxy, not a kill") would have applied if K1945 had been a genuine independent target-PASS. After r1, K1945 is genuinely independent and FAILs. The corroboration of proxy-FAIL by target-FAIL means the operator-norm bound is genuinely too loose, not just mis-calibrated as a proxy.

## What this kill produces (reusable findings)

1. **Math correction is reusable.** The D1+D2 decomposition with `‖α B A‖_op = α σ_B √r` is a closed-form correction to the parent's Theorem 2. Future cross-adapter perturbation analyses should cite it.
2. **Operator-norm bound is not predictive for cross-adapter drift.** Across three α values, sim/bound ≈ 0.08 — the bound overshoots simulation by a factor ~12. This is a structural property of operator-norm bounds (worst-case direction) vs RMS-averaged drift, not a tuning issue. **Generalizable to F#666 proxy-design catalog: operator-norm bounds are KILL-class unless paired with RMS or typical-case estimates** (per LEARNINGS.md §Proxy-KC refinement).
3. **New antipattern candidate `scaling-ratio-tautology`** (LEARNINGS.md §New antipattern candidates) — when "bound" = simulated_value · constant and the constant cancels in any derived ratio, comparing bound_ratio to sim_ratio is a tautology. Guard: verify algebraic independence of two sides before trusting "scaling match" KCs.

## Non-blocking observations

- LEARNINGS.md was already written by researcher in r1 and refreshed in r2. Analyst pass should confirm/edit the antipattern phrasing rather than regenerate from scratch.
- The Grassmannian γ ≈ 0.08 measurement (not zero, not one) is a small but reusable empirical anchor for F#562 (Grassmannian construction) — partition-QR-initialized adapters are not perfectly orthogonal in practice.
- DB K1945 description still says "quadratic scaling from alpha=20 reference" (the old wording). It needs updating to "magnitude within 2x at α ∈ {5,10,20}" but `experiment kill-update` only updates `--result`, not text. Treated as documentation drift, non-blocking.

## Assumptions

- "Tautology removed" verified by reading code: `closed_form_bound` signature contains no Drift-derived inputs; `simulate_at_alpha` calls it with `(alpha, SIGMA_B, R, W_Q_op, W_K_op, gamma)` where `W_Q_op`, `W_K_op` come from `np.linalg.norm(W, ord=2)` of freshly sampled W and γ from `‖A_B^T A_A‖_F / √r` — all algebraically independent of the `Drift` computation in `compute_drift_stats`.
- Verdict-matrix branch correctness verified by the actual run: with K1566=False and K1945=False, line 354 selects `verdict = "KILLED"` with reasoning "F#666: proxy-FAIL + target-FAIL... no F#666 escape" — matches `results.json["verdict_reasoning"]`.

## Round
Round 2 of ≤2 allowed. Verdict = KILL. Routing: `experiment kill-update --criterion 1945 --result fail` (DB drift fix), `experiment finding-add` for the math correction + bound-looseness finding, `emit review.killed`.
