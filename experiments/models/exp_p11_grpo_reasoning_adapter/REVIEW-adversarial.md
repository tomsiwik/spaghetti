# Adversarial Review: exp_p11_grpo_reasoning_adapter (P11.B0)

**Reviewer:** Adversarial Reviewer (reviewer-hat pass on full-run KILLED)
**Date:** 2026-04-17
**Verdict: KILL (endorse researcher self-determination)**

Prior 2026-04-14 REVISE→PROCEED block reviewed design for smoke; superseded by this post-run pass.

---

## Adversarial Checklist

**Consistency:**
- (a) results.json "is_smoke":false, all_pass=false across K1496/K1497/K1498; PAPER.md verdict "KILLED"; DB status=killed. **Consistent.** ✓
- (b) No KC passed but claim would require status=supported. DB correctly killed. ✓
- (c) PAPER.md verdict line "KILLED — Theorem 1 falsified"; no conflicting "PROVISIONAL"/"SUPPORTED" text. ✓
- (d) Full run (is_smoke=false), not smoke. ✓

**KC integrity:**
- (e) `git log` on MATH.md shows single commit `de38e37` (no post-registration edits); KC texts in MATH.md (≥64%, ≥s1K+20pp, all 14 cats ≥base−5pp) match the code-registered KCs in run_experiment.py. ✓
- (f) No tautology. Phase 3a loads base model, Phase 3b loads adapter-augmented model; measurements come from separate generate() calls on the same 98 questions. Adapter path gated by `if adapter_path and Path(adapter_path).exists()`. No single-sample routing. ✓
- (g) DB K-IDs 1496/1497/1498 carry stale titles from the 2026-04-13 original GRPO design; 2026-04-14 REVISE reframed to RS-SFT. PAPER.md explicitly documents this drift and reports results under both semantics; both yield KILL. Non-blocking (kill robust to either reading). ✓

**Code ↔ math:**
- (h) No `sum(lora_A...)`, no `add_weighted_adapter(combination_type="linear")`, no independent summing of safetensor `lora_A`/`lora_B` keys. Training delegated to `mlx_lm.lora` subprocess. ✓
- (i) `LORA_SCALE = 1.0` at `run_experiment.py:61` (not ≥12). ✓
- (j) No per-sample-reused routing (no multi-adapter routing in this experiment). ✓
- (k) No `shutil.copy(...)` of sibling adapters. ✓
- (l) No `{"pass": True, ...}` hardcoded. KC dict populated from measured values. ✓
- (m) MATH.md target: `mlx-community/gemma-4-e4b-it-4bit`. `run_experiment.py:55` loads same. ✓
- (m2) MATH.md / PAPER.md do not cite explicit `/mlx-dev` or `/fast-mlx` invocations. Non-blocking because: (i) training is delegated to the canonical `mlx_lm.lora` CLI (no hand-written autograd); (ii) inference uses `mlx_lm.generate`; (iii) `mx.clear_cache()` and `mx.set_memory_limit` are used between phases; (iv) no torch-style module mutation. Idiom check passes. ✓

**Eval integrity:**
- (n) Base accuracy 57.1% with avg_thinking_chars=2819 — NOT truncated. ✓
- (o) n=98 (7 per category × 14 categories) ≫ 15. ✓
- (p) No synthetic padding. ✓
- (q) Baseline measured in-run (phase3a), not cited. ✓

**Deliverables:**
- (r) PAPER.md contains prediction-vs-measurement table (6 rows: K1496, K1497, K1498, thinking chars, Phase-1 yield, Phase-2 success). ✓
- (s) Math analysis: Theorem 1 falsified mechanism ("protocol-level serialization mismatch in mlx_lm.lora channel-thinking tokens") is a sound structural explanation for the −42.9pp regression on the training-domain category (math). No unsupported claims. ✓

---

## Kill Robustness

| KC | Threshold | Measured | Miss |
|----|-----------|----------|------|
| K1496 | ≥ 64.0% | 41.8% | −22.2pp |
| K1497 | ≥ 56.1% (s1K+20pp) | 41.8% | −14.3pp (only +5.7pp vs s1K) |
| K1498 | all 14 cats ≥ base−5pp | 9/14 regressed >5pp | math −42.9pp, physics −42.9pp |

Three independent KCs fail with wide margins. Training-domain category (math) is the most regressed. Kill is robust; no threshold/noise sensitivity concerns.

---

## Structural Finding (for promotion)

**"D_train = D_eval is necessary but not sufficient to prevent catastrophic forgetting: serialization protocol of training targets must preserve the eval-time generation protocol."**

Applies to: any thinking-enabled RS-SFT/GRPO on Gemma 4 via `mlx_lm.lora`, where `<|channel>thought…<channel|>` is treated as literal text in the training message instead of as structural control tokens.

Falsifies: Theorem 1 assumption that `∇_θ E_D_train[L] == ∇_θ E_D_eval[L]` when the forward-pass distribution diverges via serialization format.

Cross-references: Finding #553 (tautological routing — Pierre v3–v6), Finding #557 (F0 OOM), Finding #560 (H0 two-STEM GD violation). This is the third consecutive reasoning-adapter kill on Gemma 4 + mlx_lm.lora, each with a distinct structural defect.

---

## Open Threads for Analyst

1. **Finding promotion**: file a new DB finding for the protocol-mismatch mechanism. Distinct from #553/#557/#560; affects P11.C0 (ThinkPO Polish), P11.D0 (Meta-R1), P11.I0 (Synthetic Data Loop), all of which share this training stack.
2. **DB KC drift**: KC texts for K1496/K1497/K1498 in DB still carry 2026-04-13 GRPO phrasing. Future experiments should not inherit this pattern — on REVISE-reframe, the DB KC rows should be rewritten to match the new semantics (or new IDs issued).
3. **Unblock path for successors**: PAPER.md §"Unblock Path" documents four options; option 4 (GRPO with answer-letter reward) is the canonically-correct next step but requires a custom mlx training loop (mlx_lm.lora is SFT-only). Do not schedule a B0-v2 RS-SFT variant.
4. **Baseline carry-over**: phase3a in-run base = 57.1% vs Finding #536 cite of 62.1%. Same measurement methodology (n=98 stratified, thinking=True, greedy). Discrepancy is within noise (±10pp at n=7/cat) but accumulating across experiments. Future P11 experiments should always re-measure baseline in-run.

---

## Assumptions

- Taking the in-run phase3a=57.1% as the true base for this experiment's comparisons (not the cited 62.1%). The gap (−4.3pp) does not affect the kill verdict — adapter at 41.8% is −15.3pp below the in-run base regardless.
- Interpreting the 2026-04-14 REVISE→PROCEED as authorizing the reframed RS-SFT experiment, and the DB KC rows (K1496/1497/1498) as retained-by-ID. The kill holds under either the original GRPO KC semantics or the reframed RS-SFT semantics.
