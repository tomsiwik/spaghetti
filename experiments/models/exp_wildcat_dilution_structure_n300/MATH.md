# MATH — exp_wildcat_dilution_structure_n300

## Question
F#882 (n=100, provisional) found mask dilution of the dense thinking delta (v/o_proj, ckpt 0001000)
composed with the math q_proj adapter gives EM(B random mask f=0.4335) − EM(C dense α=√f) = exactly
5.0pp — knife-edge between "structural" and "pure norm". At n=300 with ALL arms at ONE matched
Frobenius norm, WHICH structure carries the gap: coordinates, sparsity-per-se, or the small-|dW| subspace?

## Setup
Let dW = Σ-per-(layer,proj) dense delta of the thinking adapter, ‖dW‖_F = 3.161 (measured, F#882).
Target norm N* = √f · ‖dW‖_F = 0.65841 × 3.161 ≈ 2.081 (the expected norm of a random f=0.4335 mask,
since E[‖M⊙dW‖²_F] = f‖dW‖²_F for a uniform mask independent of dW).

Arms, all rescaled by a single global scalar to ‖·‖_F = N* exactly:
- **B** random mask f=0.4335, seeds {0,1,2} (norm ≈ N* by construction, rescale residual <0.1%)
- **C** dense α = 0.65841 (norm = N* by construction)
- **D′** keep-largest-|dW| mask at f, rescaled ×(N*/‖D‖_F) ≈ ×(2.081/3.080) ≈ 0.676
- **E′** keep-smallest-|dW| mask at f, rescaled ×(N*/‖E‖_F) ≈ ×(2.081/0.431) ≈ 4.83
- **P** keep-largest mask with WITHIN-ROW permutation of mask bits (identical per-row density as D′,
  destroys column coordinates, preserves sparsity pattern statistics), rescaled to N*

Same fixed 300 GSM8K test items (first 300, deterministic), no-thinking, greedy, max 1024 tokens,
math q_proj scale 6.0, thinking scale 1.0, composition Σᵢ(BᵢAᵢ) on disjoint projections.

## Theorem (norm-deconfounding identity)
If dilution benefit were a pure function of ‖Δ‖_F (the "α-knob" null), then all five arms — having
identical Frobenius norm — must have equal expected EM up to sampling noise. Any pairwise gap ≥ 3pp
with paired McNemar p < 0.05 on identical items refutes the norm-only null and localizes the benefit
to the structure that differs between those arms.

## Predictions (pre-registered, before run)
1. **Primary**: pooled mean EM(B) − EM(C) ≈ +5pp (point estimate from F#882: 0.84 vs 0.79),
   predicted ≥ 3pp with pooled paired McNemar p < 0.05 (900 B-vs-C pairs).
2. **Secondary**: EM(E′) ≥ mean EM(B) + 3pp — F#882 showed EM monotone DECREASING in delta norm and
   E (keep-smallest, unrescaled, norm 0.431) = 0.92 was the best arm; if small-weight structure (not
   just low norm) is the carrier, E′ at the SAME norm 2.081 should still beat B.
3. **Topology read-out**: |EM(P) − EM(D′)| ≤ 3pp ⇒ sparsity-per-se (topology irrelevant);
   EM(D′) − EM(P) ≥ 5pp ⇒ specific coordinates matter (lottery-ticket mask design rung opens).

## Refutation thresholds (kill 2336)
- **KILLED** if pooled mean EM(B) − EM(C) < 3pp (= 0.03 in EM units) on the same 300 items:
  dilution is pure norm reduction; mask arc dies, the α knob feeds the simplex bet.
- **Secondary kill** (flag, recorded independently): EM(E′) ≤ EM(C) ⇒ the E=0.92 anomaly in F#882
  was a norm artifact, not small-weight structure.
- **SUPPORTED** if gap ≥ 5pp AND pooled McNemar p < 0.05. Gap in [3pp, 5pp) or p ≥ 0.05 ⇒ provisional.

## Runtime guard
7 conditions × 300 generations ≈ 2100 generations. After the first condition, if projected total
> 2h, drop B to 2 seeds (s0, s1) per spec.
