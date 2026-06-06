# exp_composition_n5_scaling — MATH

## Verdict (pre-run): PREEMPT-KILL — redundant with prior findings + K1892 is F#666-pure canonical proxy 1007

## 0. Motivation

Claim (notes): "Push composition beyond N=3. If N=5 works, the thesis is strong. If it fails, we have a scaling bound."

Status of that claim in the current finding set:
- F#406 SUPPORTED: N=25 domain-Grassmannian composition at 4B — quality_ratio=1.3125; *subsumes any N≤25 under the same composition method.*
- F#54 SUPPORTED: real-data N=24 adapters — note-line explicitly says "scaling from N=5 with honest framing."
- F#367 SUPPORTED: activation-space interference α=0.39 sub-linear to N=10 (below 0.5 threshold).
- F#543 KILLED: uniform static scaling at N=5 on Qwen 7B — 2.57× PPL bloat is known.
- F#510/F#511 SUPPORTED: pre-merged standard LoRA destroys benchmarks; orthogonal adapters structurally required.

So the `N=5` question has *two* determined answers depending on the composition method chosen — and the experiment notes do not pick one. This is the definition of redundant.

## 1. Three independent theorems

### Theorem 1 (F#666-pure standalone — canonical guardrail 1007)

**Claim.** K1892 ("N=5 composition PPL degradation > 5% vs N=1 baseline") is a pure PPL proxy; measuring it alone — without a behavioural target that is not mechanically implied by PPL — does not justify KILL or SUPPORT under PLAN.md §1 (Finding #666).

**Proof.**
- PPL is listed in guardrail 1007 ("classification accuracy, routing match rate, PPL, cosine, clustering purity").
- Measured PPL↔task-quality correlation in this repo: r ≈ 0.08 (PLAN.md §1).
- Therefore PPL-only KCs violate the target-gated-kill rule even at N=5.
- K1893 is phrased "per-adapter quality drop > 5pp at N=5 vs isolated." "pp" reads as percentage points; if interpreted as task-accuracy drop it *would* satisfy the target pairing, but the text does not specify *which* benchmark or *what* "isolated" means operationally. Under strict reading of 1007, K1892 stands alone until a concrete target metric is bound to K1893 (dataset + evaluator). The claim as filed is |Target|≤1 at best, with |Target| definitionally missing. ∎

### Theorem 2 (Redundancy with F#406 / F#54 under Grassmannian-routing composition)

**Claim.** If the composition method is Grassmannian-domain routing (the Pierre default per PLAN.md Part 2 and F#406), then `exp_composition_n5_scaling` is strictly subsumed by F#406 and F#54 and cannot produce a new finding.

**Proof.**
- F#406 SUPPORTED measures N=25 under the same composition method on the same-scale base (4B). quality_ratio=1.3125 means *improvement* over N=1, not degradation, at N=25.
- Monotonicity: interference in LoRA composition under fixed Grassmannian-routing is monotone in N *at worst* (F#367 α=0.39 — sub-linear). Any bound that passes at N=25 (F#406) also passes at N=5 for the *same* method.
- F#54 SUPPORTED says explicitly: "scaling from N=5 with honest framing" — the N=5 mark has already been named and framed.
- Therefore K1892 and any target-paired K1893 under this method yield KC outcomes that are *derivable without running the experiment*. ∎

### Theorem 3 (Redundancy with F#510 / F#511 / F#543 under uniform-additive composition)

**Claim.** If the composition method is uniform additive (naive LoRA sum without routing/Grassmannian), then `exp_composition_n5_scaling` is strictly subsumed by F#543, F#510, F#511 and cannot produce a new finding.

**Proof.**
- F#543 KILLED: uniform w=0.2 at N=5 on Qwen 7B gives PPL 5.78 vs 2.25 single-expert = 2.57× bloat.
- F#510 SUPPORTED: pre-merged standard LoRA destroys benchmarks (GSM8K 0 vs 73 solo, HumanEval 0 vs 63).
- F#511 SUPPORTED: orthogonal adapters structurally required — naive overlap is destructive.
- Therefore under this method K1892 passes ("> 5% PPL degradation" holds with room to spare) and K1893 passes (quality collapse). The KILL outcome is pre-determined. ∎

## 2. Combined consequence

Under either composition branch, the KC outcomes are determinable without running the experiment. Under-specified composition method + canonical proxy 1007 + target-pair ambiguity = PREEMPT-KILL.

## 3. Kill criteria (as filed)

- K1892: N=5 composition PPL degradation > 5% vs N=1 baseline. **Result under Thm 2 branch: fails (no degradation at N=25 Grassmannian implies none at N=5). Result under Thm 3 branch: passes (uniform at N=5 known ≈ 2.57×). Pre-run under-determined.**
- K1893: Per-adapter quality drop > 5pp at N=5 vs isolated. **Ambiguous as filed; target-metric binding missing.**

## 4. Kill criteria (effective, preempt-KILL)

- **K_redundancy (Thm 2 or Thm 3):** whichever composition branch is read, the N=5 KC outcome is already published in F#406/F#54 or F#543/F#510/F#511.
- **K_666pure (Thm 1):** K1892 alone is canonical proxy 1007 guardrail violation; K1893 lacks target-metric binding.

Both fire. Verdict: KILLED (preempt-structural).

## 5. Antipattern audit

- Composition math: N/A (no run).
- LORA_SCALE: N/A.
- shutil.copy: N/A.
- Hardcoded `"pass": True`: N/A.
- Eval truncation: N/A.
- Proxy model substitution: N/A.
- **F#666-pure canonical-1007 PPL guardrail: FIRES (K1892).**
- **Redundant-with-prior-finding: FIRES (F#406+F#54 or F#543+F#510+F#511 branch).**

## 6. Platform

- platform: local-apple (M5 Pro 48GB per PLAN.md Part 2)
- dir: `micro/models/exp_composition_n5_scaling/`
- No code execution required for preempt-KILL (stub `run_experiment.py` writes results.json no-op).

## 7. Predictions (pre-run)

| KC | Expected | Method-dependent? | Rationale |
|---|---|---|---|
| K1892 | fails under Grassmannian / passes under uniform | yes | F#406 vs F#543 |
| K1893 | (unbound target; cannot predict cleanly) | yes | 5pp ambiguous |

## 8. Handoff

- File as KILLED (preempt-structural).
- References to findings: F#406, F#54, F#367, F#543, F#510, F#511 (all supporting preempt reasoning).
- No `_impl` follow-up filed (preempt-structural KILL per precedent in scratchpad).
