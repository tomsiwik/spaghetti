# PAPER.md — P11.Z1: Injection Decoding for Extended Thinking

## Verdict: KILLED

- **K1532 FAIL**: PS+injection = 56.0%, far below 65% target.
- **K1533 PASS (fragile)**: injection = +1.0pp over base, exactly at threshold and within statistical noise (±4.3pp at N=100).
- **K1534 PASS**: 0% degenerate loops.

Overall the hypothesis that zero-training injection decoding + Plan-and-Solve lifts Gemma 4 E4B MMLU-Pro past 65% is falsified. Injection alone produces an improvement within sampling noise; PS prompting actively harms accuracy.

---

## Summary

Testing "Well, Keep Thinking" adaptive injection decoding (arXiv:2503.10167) and Plan-and-Solve (PS) prompting (arXiv:2205.01068) as zero-training accuracy improvements on Gemma 4 E4B 4-bit MMLU-Pro with thinking enabled. Four conditions × 100 questions across math, biology, physics, chemistry.

---

## Prediction vs Measurement Table

| Condition | Predicted | Measured (N=100) | Δ vs prediction |
|-----------|-----------|------------------|-----------------|
| Base + thinking | 62.1% ± 2pp | 75.0% | +12.9pp (category selection bias — STEM-heavy) |
| + Plan-and-Solve | 62–65% | 57.0% | −8pp; PS HURTS |
| + Wait injection | 63–66% | 76.0% | within noise of base |
| + PS + injection | 65–67% | 56.0% | −9pp to −11pp; PS damage dominates |
| Degenerate loop rate | < 5% | 0.0% | PASS |
| Injection trigger rate | ~30–50% | 36% (injection_only), 30% (ps_injection) | in range |

Mean thinking chars: base 3489, ps 3549, inj 3570, ps+inj 3558 — all well above the 1500 threshold, so injection fired only when a sample dipped below.

Per-category accuracy (base vs injection):
- math 0.92 → 0.96 (+4pp)
- biology 0.76 → 0.76 (0)
- physics 0.72 → 0.76 (+4pp)
- chemistry 0.60 → 0.56 (−4pp)

Per-category gains are within noise (±10pp at N=25/cat) and inconsistent in sign.

---

## Kill Criteria Results

| Kill | Criterion | Measurement | Result |
|------|-----------|-------------|--------|
| K1532 | PS+injection ≥ 65% MMLU-Pro | 56.0% | **FAIL** |
| K1533 | injection ≥ base + 1pp | +1.0pp (exactly threshold) | **PASS (fragile)** |
| K1534 | degenerate rate < 5% | 0.0% | **PASS** |

K1533 passes numerically but is statistically inconclusive — 1pp at N=100 is well inside the ±4.3pp sampling std. A replicate would likely flip the sign.

---

## Why This Killed

1. **Gemma 4 E4B is not under-thinking.** Mean thinking = 3489c at baseline is >2× the 1500 threshold and >6× the 500 threshold predicted in the original proof. The failure mode injection decoding was designed to repair (premature thinking termination) is rare in this model. The ~36% injection rate triggered only on tail questions where the model happened to stop earlier; those questions get modest help, but most questions don't need extending.

2. **PS prompting conflicts with "Do not explain".** The question prompt ends with "Answer with ONLY the letter... Do not explain." The PS prefix ("Let's first plan what steps are needed...") asks for the opposite. Gemma 4 resolves the conflict by degrading: accuracy drops 18pp with PS alone. This effect is category-dependent (chemistry 60%→32% vs biology 76%→80%) but net negative. The PAPER review-round-1 flagged this as a non-blocking risk; the data confirms the risk materialized.

3. **Saturation bound (Theorem 2) is the binding constraint.** Even if injection fired on every question, the theoretical maximum gain is $\epsilon \leq p_{\text{sat}} - p_{\text{base}}$. With base already at 75% on this category mix, and Gemma's public thinking ceiling ~69.4% on full MMLU-Pro, there is no information-theoretic room for injection to add much.

---

## Confounds and Caveats

- **Category bias**: base=75.0% vs finding #536's 62.1% is due to STEM-only sampling (math/bio/phys/chem). This is not a reproduction failure of #536; it is a different slice. The comparison is internally valid (all four conditions share the same questions), but the absolute numbers don't transfer.
- **Threshold choice post-smoke**: MIN_THINKING_CHARS was raised 500→1500 between smoke and full run (documented in REVIEW round 1). This is not a KC edit — K1533 still measures "injection vs no injection" — but the threshold change did alter what "injection" means operationally.
- **N=100 per condition; std ~4.3pp**. The +1pp for injection and the −1pp for PS+injection vs PS are both inside this band.

---

## Assumptions

1. **Theorem 1 monotonicity**: accepted as approximately true but untested directly; mean thinking > predicted θ for nearly all samples.
2. **PS_PREFIX vs "Do not explain" conflict** remains the cleanest explanation for PS harm; replicated across three of four categories.
3. **Injection correctness**: the injected prefix correctly reconstructs `formatted_prompt + thinking_start + partial + "Wait..."`; 0% degenerate-loop rate is consistent with the 2-attempt cap. Some injection attempts returned 0-char new thinking (model immediately closed `<channel|>`), which is a neutral null-event, not a loop.

---

## Implications

- Injection decoding (arXiv:2503.10167, s1 arXiv:2501.12599) provides no meaningful benefit on Gemma 4 E4B 4-bit MMLU-Pro. The paper's mechanism requires under-thinking; Gemma 4 over-thinks.
- Plan-and-Solve prompting is **incompatible** with terse letter-only answer prompts on this model. Any future use of PS must reformat the answer instruction.
- For accuracy improvements on this model, the useful levers are training-based (adapter fine-tune, reasoning SFT) or prompt reformulation, not inference-time injection.

---

## Next Steps

- Mark experiment KILLED on K1532. K1533 nominal PASS is noise; do not claim "supported" for a +1pp single-run delta.
- Do not unblock `exp_p11_full_pipeline_v2` on this branch. If full-pipeline requires injection decoding, it must be redesigned to acknowledge that Gemma 4 rarely under-thinks.
- A candidate follow-up worth writing up only if motivated by evidence: **budget compression**, not budget forcing — reducing mean thinking from 3489 → 1500 while preserving accuracy. Inverse problem, inverse math. Not generated here; would require its own MATH.md grounded in a paper.
