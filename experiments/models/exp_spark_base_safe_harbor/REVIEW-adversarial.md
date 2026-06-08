# REVIEW (adversarial) — exp_spark_base_safe_harbor

**Verdict: KILL** (kill id 2292 fails on a real run; premise refuted by real evidence).

## Gate
`verify-experiment.sh exp_spark_base_safe_harbor` → exit 0 (REAL, is_smoke!=true, model-backed).

## No-mock / no-fabrication checks — ALL PASS
- Real MLX run: loads `mlx-community/gemma-4-e4b-it-4bit` 3× (base + comp + base-for-gate), real `datasets` HumanEval/GSM8K, subprocess code execution. No numpy/random stand-in.
- Adapter is REAL F#627 math LoRA: 84 keys = 42 q_proj.lora_a + 42 lora_b, shape (2560,6) rank-6, float32. Matches MATH.md (q_proj, r=6, scale=6.0, 42 layers). `attach_composed_lora` asserts count==42. NOT a sibling copy, NOT a placeholder.
- Gate logic real: lockstep base+composed decode, discrete top-k(8) Jaccard, J<tau→base argmax else composed argmax. Composition `(x@A)@B`*scale (N=1 single adapter, legit — not buggy independent A/B summation). LORA_SCALE=6.0 ≤ 8 (safe).
- Model in MATH.md == model loaded (no proxy substitution). enable_thinking=True (avoids F#530 base-acc-zero). n=40 (>15).

## Consistency — ALL PASS
- results.json verdict=KILLED == all_pass=false == PAPER §5 == proposed status killed.
- is_smoke:false → killed status allowed.
- ≥1 target-metric KC: both K1 (HumanEval pass@1) and K2 (GSM8K exact) are behavioral task accuracy, not proxy (F#666 OK).
- No KC modified after run: whole dir untracked in git; MATH.md (11:08) predates results.json (15:05); total_time_s=14151 (~3.9h) consistent.

## KILLED vs PROVISIONAL adjudication
F#627 signature (+22pp GSM / −12pp HE) did NOT reproduce: measured drop_fixed=−5.0pp (HE rose), lift_fixed=−7.5pp (GSM fell). Both denominators ≤0 → script returns None for IR/RET → K1 fail, K2 fail. This is a CORRECT, internally-sound computation on REAL evidence, not an artifact: the correct adapter at the correct scale was loaded and measured. Pre-registered conjunctive KC fails on a completed real run → KILL per GUIDE §3. PROVISIONAL is reserved for smoke/awaiting-proof; this is a finished n=40 real run. PAPER correctly refuses to rescue via KC-swap (forbidden).

## Non-blocking notes
- Genuinely interesting sub-observation: gate at tau≤0.5 net-improves GSM8K over FIXED (0.85 vs 0.65) at ~2% conflict, HE held ~0.50. Directionally consistent with the localized-interference mechanism. Warrants a v2 with a re-derived target — recorded, does NOT rescue this verdict.

**Route: KILL. --k 2292:fail.**
