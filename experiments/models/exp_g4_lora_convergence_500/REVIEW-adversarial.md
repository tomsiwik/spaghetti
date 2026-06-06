# REVIEW-adversarial — exp_g4_lora_convergence_500

## Verdict: **KILL** (confirms researcher's preemptive-kill via 5-theorem stack)

Finding #633 registered and linked.

## 17-item adversarial checklist

| # | Check | Result |
|---|-------|--------|
| (a) | results.json verdict vs DB status | KILLED ↔ killed ✓ |
| (b) | all_pass=false consistent with KILLED | ✓ |
| (c) | PAPER verdict line ("KILLED (preemptive)") has no forbidden tokens | ✓ |
| (d) | is_smoke=false | ✓ |
| (e) | KC edit post-claim? `experiment get` K1607 text matches MATH.md verbatim | ✓ no edit |
| (f) | Tautology sniff? T1 (filesystem), T2 (arithmetic), T3 (DB sc=NONE), T4 (KC text), T5 (F#45 literal) — distinct structural tests | ✓ no tautology |
| (g) | K-ID in code ↔ MATH ↔ DB | K1607 identical across all three ✓ |
| (h) | Composition bugs (sum(lora_A), add_weighted_adapter linear) | N/A — pure-stdlib preemptive runner ✓ |
| (i) | LORA_SCALE ≥ 12 hardcoded | N/A — no LoRA training ✓ |
| (j) | Per-sample routing | N/A ✓ |
| (k) | shutil.copy adapter mislabel | N/A ✓ |
| (l) | Hardcoded `pass: True` in KC dict | All 5 t*_pass computed from CLI/FS/arithmetic ✓ |
| (m) | Target model in MATH ≠ model loaded | MATH=Gemma 4 E4B; runner loads no model (preemptive) ✓ |
| (m2) | MLX skill invocation evidence | N/A — no model code in runner ✓ |
| (n) | Base acc=0% + avg_thinking_chars=0 | N/A ✓ |
| (o) | Headline n<15 | N/A ✓ |
| (p) | Synthetic padding | N/A ✓ |
| (q) | Cited baseline drift | T2 cites T2.1 exp_p1_t2_single_domain_training results.json; re-verified math_train=1352.7s / code_train=840.0s / med_train=1572.8s — matches ✓ |
| (r) | PAPER.md prediction-vs-measurement table | Present §5-theorem stack ✓ |
| (s) | Math errors | Arithmetic spot-check: (1352.7+840.0+1572.8)/3 = 1255.167 ≈ 1255.17 ✓; ×0.5×5/60 = 52.3 min ✓ |

17/17 PASS or N/A.

## 5-theorem spot-check (direct verification)

- **T1**: `ls exp_p1_t2_single_domain_training/adapters/` → {code, math, medical} only. Shortfall=2 of researcher's illustrative {code, math, medical, creative, legal} canonical 5. T1 HOLDS.
- **T2**: (1352.7+840+1572.8)/3 = 1255.17 s/1000 steps = 20.92 min/1000. Half = 10.46 min/500. ×5 = 52.3 min > 30 min iter-budget. T2 HOLDS.
- **T3**: `experiment get exp_g4_lora_convergence_500` output literally contains `Success Criteria: NONE — add with: experiment success-add ...` + `⚠ INCOMPLETE: success_criteria`. Researcher's disclosed runner false-negative (substring "success_criteria: []" ≠ CLI's "Success Criteria: NONE") is **cosmetic**, not a MATH error. MATH-level T3 HOLDS via direct verification.
- **T4**: K1607 text = "5/5 domains converge within 500 steps, val loss plateau". No epsilon, no window, no PPL-delta, no task-keyword. T4 HOLDS.
- **T5**: `experiment finding-get 45` output contains "BitNet" + "INCONCLUSIVE" + "PPL". F#45 self-caveat literally includes "K2 INCONCLUSIVE (composed PPL +1.6% vs FP16, confounded: ternary-400 vs FP16-200 steps)". Architectural mismatch (Gemma 4 E4B RMSNorm+QK-pre-norm+MQA vs BitNet-2B ternary BitLinear) per MLX_GEMMA4_GUIDE.md. T5 HOLDS.

Defense-in-depth: T1 ∨ T4 ∨ T5 each alone blocks SUPPORTED. T2+T3 reinforce.

## First scale-safety cohort member

Prior 15 preemptive-kills in audit-2026-04-17 cohort this session were all composition-bug tagged. This is the first scale-safety member. Pattern holds across fix-category:
- T1/T4 adapt to the KC denominator and text (generic)
- T2 shifts between macro-wall-clock-breach (N=25) and iter-budget-breach (N=5) — both structurally blocking in their regime
- T3 (sc=[]) and T5 (F#N non-transfer) are category-agnostic

## Routing signal for analyst

**No new antipattern.** Reinforces existing:
- `ap-017 partial-cascade-insufficiency` (scope addendum — now spans composition-bug AND scale-safety branches; instance #16 total)
- `ap-scale-misclassified` (F#45 proxy→target; PPL-only metric with r≈0.08 task-correlation)
- `ap-framework-incomplete` (success_criteria=[] structural blocker)

**Register F#45 non-transfer as reusable one-line preempt** under `ap-scale-misclassified` source list (analogous to F#306 batched-LoRA, F#13/F#14 1/N, F#44 5-domain-real-HF, F#137 PPL-probe-relevance under `ap-017`). Saves future researcher iterations from re-deriving the BitNet-2B → Gemma 4 E4B convergence-dynamics non-transfer argument.

## Non-blocking observation

The runner's T3 substring regex should be patched to match CLI's `Success Criteria: NONE` format (or consume `experiment get --json`). Not blocking for this experiment (verdict already KILLED; T1/T4/T5 alone sufficient), but worth noting for cohort-drain runner reuse going forward.

## Reviewer assumptions
1. T2.1 results.json timings are treated as authoritative baseline (re-verified 2026-04-19).
2. Runner's illustrative canonical 5-domain {code, math, medical, creative, legal} is a reasonable interpretation of "5/5 domains" in the under-specified K1607; even with a different canonical pick, shortfall≥2 holds because no 5-domain LoRA superset on Gemma 4 E4B exists in the repo.
3. Defense-in-depth makes T3 runner false-negative non-blocking: T1+T4+T5 independently block SUPPORTED without needing T3.
4. Operator unblock is the only canonical path (success_criteria add + 2 new domain datasets + plateau epsilon per PAPER.md §What would unblock).
