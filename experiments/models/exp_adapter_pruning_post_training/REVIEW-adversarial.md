# REVIEW-adversarial — exp_adapter_pruning_post_training

## Verdict: **KILL** (confirms researcher's KILLED disposition)

Finding #745 already logged (killed, main), Finding #746 logged (provisional, novel side-observation). DB status = killed, results.json verdict = KILLED, PAPER.md verdict = KILLED — all consistent.

## Adversarial checklist

### Consistency (blocking)
- (a) results.json `verdict = "KILLED"` ↔ DB `status = killed` ↔ PAPER verdict = **KILLED** — consistent.
- (b) `all_pass = false` + one KC FIRE — consistent with killed.
- (c) PAPER.md opens with "**KILLED**" — no PROVISIONAL/INCONCLUSIVE contradiction.
- (d) `is_smoke = false`; N=100 Phase 1, N=100 Phase 2 — full run.

### KC integrity
- (e) Directory untracked (first-time experiment). No post-run KC relaxation possible — no prior version exists.
- (f) No tautology: K1922 / K1923 are real PPL deltas on real tokens, not `e=0→0` or self-identity.
- (g) K1922 code measures what MATH.md §Pre-registered describes (single-adapter domain PPL, 100 medical/valid rows, per-matrix top-50% magnitude prune). K1923 same. ✓

### Code ↔ math
- (h) **No buggy composition**. `term_med = (dx @ self.p2_med_a) @ self.p2_med_b`, `term_math = (dx @ self.p2_math_a) @ self.p2_math_b`; final output is `y + scale * (term_med + term_math)`. This is α·x·A_med·B_med + α·x·A_math·B_math, exactly per MATH.md §Methodology Phase 2. No `sum(lora_A…)` antipattern. ✓
- (i) `ADAPTER_SCALE = 6.0` (matches `lora_config_*.yaml`, not unsafe 20). ✓
- (j) Per-sample eval in `compute_ppl` loop. ✓
- (k) No `shutil.copy` of sibling adapter. ✓
- (l) No hardcoded `"pass": True`; KCs computed from numeric ΔPPL. ✓
- (m) `BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"` — same model the adapter was trained on. No proxy substitution. ✓
- (m2) MATH.md does not explicitly cite `/mlx-dev`, but code is idiomatic MLX (`mx.eval(nll)` before `.item()`, `mx.clear_cache()` between phases, `mx.eval(model.parameters())` after LoRA install, `mx.take_along_axis`/`mx.logsumexp` API). Scaffolding inherited from F#744 (`exp_composition_ordering_matters`). Non-blocking.

### Eval integrity
- (n) Not truncated: teacher-forced NLL over all `y = ids[1:]` tokens up to 512. Base PPL 29.6 on medical (sane, not 0% thinking-suppression).
- (o) N=100 well above 15.
- (p) Real trained adapters (from `exp_p1_t2_single_domain_training`), not synthetic.
- (t) **Target-gated KILL satisfied**: K1922 IS a target metric (real PPL on the adapter's training domain), not a proxy. No F#666 concern.

### Deliverables
- (r) PAPER.md contains prediction-vs-measurement table with f_retained (0.886 predicted vs 0.904/0.922 measured), relative ΔW gap (0.3-0.5 pred vs 0.376 measured), ΔPPL_single (0.6-1.2 pred vs 0.331 measured), ΔPPL_compose (1.0-1.7 pred vs -0.129 measured). Tight on weight-space predictions; PPL slope was over-estimated — paper explains why.
- (s) Math: theorem derivation correct (triangle inequality + Frobenius submultiplicativity + retained-energy identity). Predicted bound 0.656; measured 0.376 satisfies the bound.

## Non-blocking observations
- `experiment get` flags missing `success_criteria` and `references` metadata — hygiene defect but non-blocking for a KILL verdict (the KC structure is sound; ref list exists in MATH.md).
- Composition sign-reversal (ΔPPL_compose = −0.129) is logged as Finding #746 provisional — reasonable handling; noise vs real-mechanism attribution requires a separate experiment (keep-frac sweep on co-trained vs non-co-trained adapter pairs).
- The F#674-based ΔPPL slope prediction (0.6–1.2) was over-conservative; actual slope ≈ 0.88 PPL per unit relative-ΔW. Useful calibration datum for future pruning experiments.

## Assumptions logged
- Treating the untracked directory as first-run (no KC-mutation possibility). If re-run, must diff MATH.md against prior commit.

## Routing
Emit `review.killed` → Analyst writes LEARNINGS.md (short, linking F#745 and F#746 to Wanda/LoRAPrune prior art).
