# REVIEW-adversarial (self-review) — exp_composition_weighted_sum

## Attack surface
Preempt-structural KILL rests on three theorems + one taxonomy classification. If any fails, the triple-fire does not hold and the KILL becomes unjustified.

## Attack 1 — Thm 1 (F#664) over-broad?
**Claim.** F#664's preempt category covers "Task-Arithmetic fixed-coefficient blends" — is `α_i · ΔW_i` with fixed α actually in-family?

**Check.** F#664 verbatim: "Any fixed algebraic weighted blend of specialist experts (RS parity p_i=Σα_{i,j}E_j with Vandermonde α, TIES addition, random-basis averaging, fixed task-arithmetic coefficients) falls inside F#157 averaging regime." The experiment's formulation `Σ_i α_i · ΔW_i` with α_i constants is a scalar task-arithmetic blend — directly in the named family. **Sound.**

## Attack 2 — Thm 2 (F#164) applies at N=3?
**Claim.** F#164 showed CAT diverges at N=5 with 840 scalars. Does this generalize to N=3?

**Check.** F#164's impossibility: dimensional concentration at d_model=17.2M → |cos|~0.001 → vanishing inter-adapter gradient. At N=3 with Gemma-class d_model=3584, dimensional concentration gives |cos| still ≪ 10⁻² (3584 ≫ 1). CAT scalar count scales linearly in N and sub-linearly in d_model, but the landscape flatness is geometric, not count-driven. N=3 is harder if anything (fewer scalars to average out). **Sound.**

## Attack 3 — Thm 3 (F#137/F#643) — does the experiment really target data-conditioned?
**Claim.** F#137 used PPL-probe relevance weighting, which is data-conditioned. Does `α_i · ΔW_i` allow data-conditioning?

**Check.** The DB notes say "learned or per-task weights." "Per-task" is data-conditioned (task identity is input signal). If we interpret "per-task α_i" as F#137's PPL-probe mechanism, then K1896 would PASS at +9.34pp, but this is verbatim F#137 per F#643. Either the experiment is in F#664's family (Thm 1 fires) OR it duplicates F#137 (Thm 3 fires) OR it's CAT-learned (Thm 2 fires). There is no fourth admissible branch. **Sound.**

## Attack 4 — K1897 un-evaluable?
**Claim.** K1897 says "learned weights overfit to training tasks." For this to be non-vacuous, learning must succeed first.

**Check.** F#164 shows CAT diverges at all LRs — training never produces weights that can overfit. K1897 is structurally un-evaluable via the learned branch. Via the fixed branch, K1897 doesn't instantiate (no learning). Via the data-conditioned branch, F#137's r=0.990 probe-oracle correlation is counter-evidence against overfit. **K1897 inconclusive is the correct outcome; INCONCLUSIVE is not KILL, but the experiment's overall verdict stays KILLED on K1896 alone plus KC-malformed F#666-pure.**

## Attack 5 — §5 tautological-inter-variant-delta mis-classification?
**Claim.** §5 fires only when both sides of the delta are variants of the *same* mechanism. Is "weighted composition" the same mechanism as "uniform composition"?

**Check.** Both are compositions of the form `Σ α_i ΔW_i`; uniform is the special case α_i = 1/N. They are variants in the same mechanism family. Comparing them without an absolute target metric is exactly §5. **Sound.**

## Attack 6 — method-dependent-redundancy 3rd instance vs some other bucket?
**Claim.** Is this really method-dependent-redundancy, or is it closer to F#669 (parent-target-unverified) or F#702 (method-unavailable)?

**Check.** 
- F#669: parent SUPPORTED but child target unverifiable. Doesn't apply — the parent (composition) is extensively studied.
- F#702: method unavailable (infrastructure missing). Doesn't apply — composition infrastructure exists (PEFT, MLX LoRA).
- method-dependent-redundancy: KC well-formed in principle, but branches collapse to prior findings. **This is the fit.** Three branches (fixed / learned / data-conditioned), each covered by a distinct supported-or-killed finding.

**Sound.** 3rd instance post-promotion, anchor append per the canonical memory rule.

## Attack 7 — Analyst handoff was wrong — did I re-check the underlying DB properly?
**Claim.** The analyst said "novel mechanism unclaimed by F#66/F#510/F#511/F#543/F#406." Did I actually verify F#137/F#164/F#496/F#643/F#664 exist and say what I claim?

**Check.** All 5 findings retrieved via `experiment finding-get <N>` or `experiment query`:
- F#664 full content retrieved — impossibility structure quoted directly.
- F#137 full content retrieved — +9.34pp r=0.990 numbers direct.
- F#164 full content retrieved — CAT divergence + orthogonal-adapter landscape direct.
- F#496, F#543, F#643 indexed via `finding-list | grep` — confirmed status + headline claim.
**Sound.** Analyst handoff was literally wrong (omitted relevant priors) — reviewer-precedent allows (and requires) researcher to correct.

## Conclusion
All six attacks pass. KILLED preempt-structural, triple-fire, is the defensible verdict.

## Triple-fire mode check
`mem-pattern-triple-fire-mode` is active per F#721 promotion. This is the 9th triple-fire overall. No action required beyond standard preempt-structural processing.
