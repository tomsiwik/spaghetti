# REVIEW-adversarial.md — exp_followup_cayley_riemannian_adam

**Verdict: KILL** (target-gated F#666 satisfied; both proxy K1559 and target K_target FAIL)
**Reviewer iteration:** 2026-04-24
**DB state:** already `killed` w/ evidence; this review documents the adversarial pass and adds the finding.

---

## Adversarial Checklist

| Item | Result |
|---|---|
| (a) results.json verdict ↔ DB status | `KILLED` ↔ `killed` ✓ |
| (b) all_pass ↔ claim | `false`, killed ✓ |
| (c) PAPER.md verdict line | `KILLED` (no PROVISIONAL/PARTIAL/INCONCLUSIVE) ✓ |
| (d) is_smoke vs full-run claim | `is_smoke=false`, train_steps=150, n=30 ✓ |
| (e) Post-run KC manipulation | MATH.md untracked; KC IDs in code/MATH.md/DB align (K1559 in DB matches K1559 in MATH/code) ✓ |
| (f) Tautology | No (final loss + MMLU are real measurements) ✓ |
| (g) Code measures what MATH says | `k1559 = hra_riem_conv <= lora_conv` matches "ratio ≤ 1.0" ✓ |
| (h) Buggy composition | N=1, no composition ✓ |
| (i) LORA_SCALE ≥12 | `scale=float(r)=16` for single-adapter LoRA baseline — alpha=r convention; not the F#328/#330 composition antipattern. **Non-blocking** (caveat below). |
| (j) Tautological routing | None (no router) ✓ |
| (k) shutil.copy adapter | None; weights via mx.savez ✓ |
| (l) Hardcoded pass=True | None; all KCs derived from measurements ✓ |
| (m) Proxy substitution | Target = loaded = `mlx-community/gemma-4-e4b-it-4bit` ✓ |
| (m2) Skill invocation | MLX-dev patterns evident: `mx.eval`, `mx.clear_cache`, `nn.value_and_grad`, phased execution, CPU-only `mx.linalg.qr`, `mx.set_memory_limit` ✓ |
| (n) Base = 0% w/ thinking | Base MMLU=13.3% (not zero), no thinking truncation ✓ |
| (o) Headline n | n=30 ≥15 ✓ |
| (p) Synthetic padding | None ✓ |
| (q) Baseline drift | LoRA measured in same run; HRA_euc reproduces parent F#416 (PASS) ✓ |
| (t) **Target-gated kill (F#666)** | K1559 FAIL (proxy: both DNF) **AND** K_target FAIL (Δ=−16.7pp << −3pp threshold) → KILL safe ✓ |
| (u) Scope-changing fixes | None — full pre-registered protocol ran ✓ |
| (r) Prediction-vs-measurement table | Present in PAPER.md §"Prediction vs Measurement" ✓ |
| (s) Math soundness | Theorems 1–3 cite Wen-Yin (1208.4298), Edelman-Arias-Smith 1998, Bécigneul-Ganea (1810.00760); QR retraction first-order equivalent to Cayley (Absil 2008 §4.1.1) ✓ |

## Mechanistic verdict

PAPER.md's signal-bottleneck explanation is the load-bearing finding: at r=16, d=2560, the canonical-metric tangent projection `Ξ = G − ½(GVᵀ + VGᵀ)V` discards ~½ the per-element gradient energy because the normal component dominates when r ≪ d. Strict-Stiefel HRA at LoRA-tuned LR effectively trains at LR/10. K_stiefel PASS (8.5e-7 Frobenius) confirms the retraction itself is exact — the Riemannian *machinery* works; the *biological* hypothesis (Stiefel drift caused parent K1013) is falsified.

## Caveats (non-blocking)

1. **n=30 MMLU 95% CI ≈ ±17pp.** Δ=−16.7pp is at the edge of significance (two-prop z-test p≈0.14). Researcher acknowledges in PAPER.md §Assumptions; KILL stands on joint evidence (proxy + target + mechanism).
2. **LoRA scale=16 (alpha=r convention).** Not the composition antipattern (F#328/#330 target multi-adapter sums); for single-adapter baseline this is the standard convention and the LoRA reaches 43.3% MMLU as expected.
3. **K_stiefel only measured for HRA_riem.** HRA_euc Stiefel-drift unmeasured (instrumentation gap noted in PAPER.md). Does not affect KILL — HRA_euc and HRA_riem both fail vs LoRA at this scale; the structural drift question is orthogonal.

## Assumptions (reviewer pick-and-proceed)

- F#666 governs: proxy-FAIL + target-FAIL = KILL. Both fail decisively here.
- The "n=30 MMLU underpowered" critique would matter if K_target were the *sole* failing KC; it is not — K1559 (proxy) also FAILs and the mechanistic story (signal-bottleneck arithmetic) is independently derivable. KILL is robust.
- DB already has the KILLED status and one evidence entry; this review adds the finding (no `experiment complete` re-run needed).

## Routing
- Add finding (Cayley/Riemannian-Adam-on-Stiefel does not rescue HRA at r ≪ d on Gemma 4 E4B; signal-bottleneck mechanism).
- Emit `review.killed` → Analyst writes LEARNINGS.md with literature anchoring.
