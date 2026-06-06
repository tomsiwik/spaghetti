# REVIEW — P4.D0: Domain + Format Adapter Simultaneous Composition

## Verdict: PROCEED (as KILLED)

## Strengths

1. **Honest theorem with prescient caveat.** Theorem 2 explicitly flags the functional
   dependency issue before the experiment runs. The experiment then confirms the caveat
   dominates. This is proof-first research done right.

2. **Scaled composition experiment is excellent.** The α sweep (0.0→0.1→0.25→0.5→1.0)
   localizes the collapse threshold to 12-24% o_proj perturbation and reveals the
   sub-collapse attenuation effect. This is the kind of quantitative impossibility
   evidence the project needs.

3. **Prediction-vs-measurement table is complete.** All three kill criteria have clear
   predictions and measurements. All fail catastrophically — no ambiguity.

4. **Impossibility structure is sound.** The q→attn→o functional chain argument is
   correct. Cross-projection composition creates compound perturbation, not additive.

## Issues (non-blocking)

1. **v_proj learning nothing (all lora_b=0) is a secondary finding worth tracking.**
   The P4.C1 SOAP adapter learned its entire effect through o_proj alone. This means
   v_proj was wasted capacity — future format adapters could use o_proj-only (halving
   parameters). Consider whether this deserves its own finding note.

2. **Cross-Jacobian condition is qualitative.** The condition
   ||∂output/∂q_weights × ∂output/∂o_weights|| ≈ 0 is stated but not evaluated.
   For a killed experiment this is acceptable — the empirical α sweep is more useful
   than a bound that would require Hessian estimation.

3. **Solo adapter baselines look weak.** SOAP solo=60%, Legal-brief solo=80% — these
   were reported differently in MATH.md (citing P4.C1 as +70pp and +90pp). This
   discrepancy may reflect different evaluation prompts or the 4-bit quantized model
   vs P4.C1's setup. Not blocking since the composition result is unambiguous collapse.

## Assessment

The impossibility structure is the real contribution: parameter disjointness ≠ functional
independence. The attention mechanism's q→o chain means cross-projection adapters can
NEVER be composed post-hoc at these perturbation magnitudes. The three safe composition
paths identified (small perturbations, same-projection, co-training) are all actionable.

No blocking issues. Proceed to finding registration and LEARNINGS.md.
