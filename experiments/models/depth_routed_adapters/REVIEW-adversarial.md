# Peer Review: Depth-Routed Adapters

## NotebookLM Findings

Skipped (manual deep review sufficient for a KILLED experiment with clear negative signal and strong corroboration from two prior experiments).

## Mathematical Soundness

### What holds

1. **Theorem 1 (Layer Specialization Under Grassmannian Orthogonality)** is a plausible existence argument but not a theorem. The "proof sketch" assumes that per-domain feature spaces decay at different rates across layers, then asserts the routing weights will learn to track this. This is a hope, not a derivation. The statement "the effective rank of B_i^l at layer l decreases in layers where domain-relevant features have already been collapsed" is an unverified empirical claim dressed in mathematical notation. **Verdict: The math is internally consistent but the claim is unfalsifiable at this scale -- there is no measurement of feature decay rates to confirm or deny the mechanism.**

2. **The depth routing forward pass equation (Section: Combined Forward Pass)** is correct. The normalization by N_LAYERS (line 494 in code: `v * (depth_weight * N_LAYERS)`) ensures that uniform routing (alpha_{i,l} = 1/N for all i,l) recovers exactly the token-only routing behavior. This is important: it means the depth router CAN do nothing, and indeed it converges to doing nothing (or close to it). The normalization is sound.

3. **Entropy ratio as K1 metric** is well-chosen. Entropy ratio H/H_max in [0,1] correctly measures specialization, and the 0.95 threshold is reasonable (requiring at least 5% reduction from maximum entropy).

4. **Complexity analysis** is thorough and accurate. The 57% parameter overhead (16,672 additional params) is correctly computed. FLOPs analysis is reasonable.

### What does not hold

1. **MATH.md describes AttnRes pseudo-queries with depth attention over prior hidden states (Section: Axis 2)** but the actual code implements something much simpler. The code (lines 363-384) uses static learned pseudo-queries `w_l` and expert embeddings `r_i` with a simple dot product `w_l^T r_i / sqrt(d_e)`. There is NO depth attention, NO hidden state conditioning, NO pseudo-query computation from prior layers. The MATH.md equation:

   ```
   q_l = softmax((phi_l h_{l-1})^T H_{<l} / sqrt(d_r)) H_{<l}^T
   ```

   is never implemented. The actual mechanism is:

   ```
   alpha_{i,l} = softmax(w_l^T r_i / sqrt(d_e))
   ```

   **This is a critical mismatch.** The MATH.md describes an input-dependent, context-aware depth routing mechanism. The code implements a static, input-independent weight matrix that is the same for every input. This means the depth router cannot condition on the actual hidden states -- it learns a single fixed set of per-layer weights regardless of what token sequence is being processed.

   This mismatch is partially acknowledged in PAPER.md Section 2.2 ("Depth Router (Novel)") which describes the simpler mechanism but not in MATH.md which promises the full AttnRes pseudo-query mechanism.

2. **The "gradient-free perturbation search" (lines 851-878)** is a poor optimization method for this problem. With 40 iterations, sigma=0.3, and parameters in R^{(4*32 + 5*32) = 288}, the search explores approximately 40 random directions in a 288-dimensional space. The probability of finding a useful direction by chance is negligible. The sigma decay schedule (0.7x every 15 steps) reduces to 0.21 after 30 iterations and 0.147 after 45, meaning later perturbations are too small to escape the uniform basin.

   For seed 314, the search finds EXACTLY uniform weights (0.200 everywhere to 9 decimal places), meaning zero of the 40 perturbations improved over uniform. For seeds 42 and 137, tiny deviations from uniform are found (improvement 0.05% and 0.19%) -- these are within noise of the 3-batch evaluation used during search (line 865: `max_batches=3`).

3. **Theorem 2 (Gumbel-Sigmoid vs Softmax)** is mathematically correct but irrelevant. The code uses neither Gumbel-Sigmoid nor any token-level gating. The token router is a standard softmax classifier with argmax selection (line 414: `top1 = mx.argmax(probs, axis=-1)`). There is no Gumbel noise, no temperature annealing, no differentiable gating. This section of MATH.md describes a mechanism not present in the experiment.

## Novelty Assessment

### Prior art that renders this experiment partially redundant

1. **exp_pointer_routing_no_merge** (already in FINDINGS.md, line 275): This experiment independently tested per-layer adapter routing and found that `same_adapter_fraction=1.0` -- the optimal strategy is to use the SAME adapter at ALL layers. Cross-layer variation was exactly 0.0. Hash routing and MLP routing both performed WORSE than uniform. The conclusion: "per-layer adapter mixing actively hurts -- adapters calibrated for same-adapter residual stream at all preceding layers." This is the same conclusion as the current experiment but was already known.

2. **exp_attnres_depth_composition** (FINDINGS.md, line 273): Already showed entropy ratio 0.775 (non-uniform depth weights exist in principle) but composition improvement is only 0.39% at L=4, "indistinguishable from noise." The current experiment adds depth routing on top but cannot escape the same L=4 limitation.

### Delta over existing work

The delta is small: this experiment tests a different depth routing mechanism (learned pseudo-queries + expert embeddings vs AttnRes residual reweighting) and gets a similar or worse result. The novel element -- input-independent per-layer routing weights -- was not actually tested because the mechanism described in MATH.md was not implemented.

## Experimental Design

### Does this test what it claims?

**Partially.** The experiment tests whether static per-layer adapter weights improve composition over uniform weights. This is a valid and interesting question, but it is NOT what the MATH.md claims to test (context-dependent depth routing via AttnRes pseudo-queries).

### Critical confound: The optimization method is broken

The most important experimental flaw is that the depth router training uses gradient-free random search with 40 iterations in 288 dimensions. This is not an adequate optimization procedure. The experiment cannot distinguish between:

(a) Depth routing weights genuinely have no useful non-uniform solution
(b) The optimizer failed to find a non-uniform solution that exists

For seed 314, the search found zero improvements in 40 tries, yielding perfectly uniform weights. This is consistent with either (a) or (b), but given the tiny search budget, (b) cannot be ruled out.

**However:** The pointer_routing_no_merge experiment used gradient-based MLP optimization and ALSO found that the optimal strategy collapses to uniform per-layer routing. This corroborating evidence from a different methodology strengthens the case that (a) is the correct explanation.

### Adequate controls

The controls are appropriate:
- Oracle (best possible) correctly implemented
- Token-only routing correctly implemented
- Random routing and uniform 1/N provide meaningful baselines
- Three seeds provide basic reproducibility

### The "mixed domain blowup" is informative

The mixed domain catastrophically degrades under depth routing (1.837x PPL at seed 42, 4.959x at seed 314). The PAPER.md explanation is reasonable: mixed has 2.8x norm gradient across layers, so non-uniform scaling amplifies the imbalance. This is a genuine finding about the interaction between adapter norm gradients and depth scaling.

### The gamma metric is correctly computed

Geometric mean PPL across domains (gamma) is the right metric for composition quality, matching the convention in prior experiments.

## Is the Kill Verdict Correct?

**Yes, the kill is justified.** Both kill criteria fail clearly:

- **K1 (entropy ratio < 0.95):** 0.992 mean, with seed 314 at exactly 1.000. The depth weights are uniform. FAIL is correct.
- **K2 (improvement >= 2%):** -18.3% (degradation, not improvement). FAIL is correct and decisive.

The kill criteria are fair and were pre-registered. The thresholds are reasonable. The negative result is consistent across all 3 seeds.

**However, the kill should be qualified:** The experiment does not rule out input-dependent depth routing (the mechanism described in MATH.md but not implemented). It rules out static per-layer weights, which is a weaker claim. The PAPER.md correctly notes this ("Do not pursue depth routing when token-level routing achieves oracle performance") but the stronger claim would be: "Static per-layer routing is provably unnecessary when token routing is perfect. Input-dependent depth routing remains untested."

## Are the Conclusions Warranted?

### Warranted conclusions

1. "Token-only routing already achieves oracle performance (0.0% gap)" -- confirmed by data, warranted.
2. "When oracle gap is zero, depth routing has no room to improve" -- logically sound and empirically confirmed.
3. "L=4 is too shallow for depth-axis effects" -- consistent with attnres_depth_composition and pointer_routing_no_merge, warranted as a pattern across three experiments.

### Overreaching conclusions

1. "Per-layer adapter modulation is a dead end" -- too strong. Three experiments (attnres_depth_composition, pointer_routing_no_merge, this one) all test at L=4 with trivially separable domains. The dead end is specific to (a) L=4 and (b) perfect token routing. Neither condition holds at macro scale.

2. The PAPER.md recommendation "Do not pursue depth routing when token-level routing achieves oracle performance" is tautological. Oracle performance means the routing problem is already solved; any additional mechanism is by definition unnecessary. The interesting question -- does depth routing help when token routing is imperfect? -- is not answered.

## Macro-Scale Risks (advisory)

1. **The zero oracle gap is a micro-scale artifact.** Five character-level domains with fully separable vocabularies make routing trivial. At macro scale with overlapping domains (e.g., medical-legal, math-code), the oracle gap will be nonzero and depth routing may have room to contribute.

2. **The broken optimization method would need to be replaced.** If depth routing is revisited at macro scale, gradient-based optimization through the per-layer routing weights is mandatory. The gradient-free search used here is inadequate.

3. **The MATH.md mechanism (context-dependent pseudo-queries from hidden state history) was never tested.** If the idea is revisited, the actual AttnRes pseudo-query computation should be implemented, not the static approximation.

4. **The pointer_routing_no_merge finding (adapters calibrated for same-adapter residual stream) is the deeper objection.** Even with better optimization and more layers, adapters trained expecting uniform contribution across all layers may not benefit from post-hoc per-layer reweighting. Depth routing would need to be part of the adapter training loop, not a post-hoc addition.

## Verdict

**PROCEED** (kill confirmed, negative result is informative)

The kill is correct. Both K1 and K2 fail decisively across 3 seeds. The experiment design has one significant flaw (gradient-free optimization in 288 dimensions with 40 iterations) and one significant mismatch between MATH.md and implementation (static weights vs. context-dependent pseudo-queries). However, these flaws do not invalidate the kill because:

1. The oracle gap is exactly 0.0% -- no routing mechanism can improve on perfection
2. The corroborating negative result from pointer_routing_no_merge (using gradient-based optimization) supports the same conclusion
3. The attnres_depth_composition experiment (testing the actual mechanism described in MATH.md) already showed only 0.39% improvement at L=4

The negative result is genuinely informative: **at L=4 with perfect token routing, depth routing is unnecessary.** This should be recorded with the caveat that the interesting case (imperfect token routing at macro scale with overlapping domains) remains untested.

### Specific issues to record in FINDINGS.md caveats

1. MATH.md describes AttnRes pseudo-queries with hidden-state conditioning; code implements static learned weights. The mechanism tested is weaker than the mechanism described.
2. Gradient-free optimization (40 iterations, 288 dimensions) is inadequate. Corroborated by pointer_routing_no_merge using gradient-based methods with same outcome.
3. Gumbel-Sigmoid mechanism described in MATH.md is not implemented in code (standard argmax routing used).
4. Zero oracle gap makes the experiment uninformative about whether depth routing helps when token routing is imperfect.
5. Three experiments now converge on the same conclusion: per-layer adapter routing does not help at L=4 (attnres_depth_composition, pointer_routing_no_merge, depth_routed_adapters).

---

## Audit-Rerun Closure Review (2026-04-18)

Reviewer pass on `experiment.done exp_depth_routed_adapters: KILLED (audit-rerun closure)`.

**State on review.** `git diff --stat` shows the closure is append-only to PAPER.md (+125 lines); MATH.md, results.json, and run_experiment.py unchanged since the 2026-03-28 run. No KC-swap possible. DB state `killed` with evidence rows on 2026-03-28 and 2026-04-18. K#528/K#529 numeric IDs consistent across DB ↔ MATH.md ↔ PAPER.md ↔ results.json.

**Adversarial checklist (a)–(s):** all PASS under closure.
- (a)–(d): verdict/all_pass/verdict-line consistent (killed); no smoke flag relevant.
- (e): KCs unchanged from original run (git log confirms).
- (f): closure theorems use the failure direction (no improvement possible) — not a favorable-direction tautology. γ_M ≥ γ_oracle uses the Grassmannian-interference hypothesis from MATH.md Thm 1; measured γ_oracle = γ_token = 1.012 (§3.2). Safe.
- (g): K IDs 528/529 consistent across DB, MATH.md, PAPER.md, results.json.
- (h)–(m2): closure, no new code; skill invocation N/A.
- (n)–(q): no thinking-channel, 3-seed × 5-domain coverage adequate, no synthetic padding.
- (r): PAPER.md has prediction-vs-measurement tables (§3.1, §3.2) plus per-domain breakdown (§3.3).
- (s): Theorems C1/C2/C3 soundness — C1 valid under Grassmannian interference and finite training variance; C2 cites pointer_routing_no_merge L=30 gradient-based result (`same_adapter_fraction=1.0`, `cross_layer_variation=0.0`) as corroborating evidence that more optimization power converges to uniform; C3 mechanism-level train-test distribution mismatch is consistent with the mixed-domain norm-gradient (2.82× L3/L0) → 1.837 PPL blowup at seed 42.

**Antipattern self-check (from closure §Antipattern self-check):**
- ap-017 N/A (real trained adapters, distinct per-domain individual_ppl)
- ap-020 N/A (upstream attnres_depth_composition is SUPPORTED)
- ap-003 N/A (default scale, not the inflation antipattern)
- ap-no-knowledge-gap, ap-convex-hull-projection-tautology: both N/A (documented)
- Candidate new: ap-oracle-ceiling-blocks-headroom (first instance in this experiment). Promote if a second surfaces.

**Verdict (closure):** PROCEED (kill confirmed). The original kill is correct; the closure addendum demonstrates that both K1 and K2 are structurally unreachable under the code-bug fix (C1 oracle ceiling; C2 optimization-class invariance via pointer_routing_no_merge; C3 train-test distribution mismatch). Researcher's decision not to rerun is supported by three independent structural arguments. No DB change needed (researcher already logged `--status killed --k 528:fail --k 529:fail` with 2026-04-18 evidence).

**Open threads for analyst:**
- Promote `ap-oracle-ceiling-blocks-headroom` to the antipattern list if a second instance appears. Distinct from ap-017 (stub adapters), ap-020 (upstream killed), ap-003 (scale inflation), ap-no-knowledge-gap (adapter-training capacity), ap-convex-hull-projection-tautology (projection onto training span). This one is about composition-mechanism layered on oracle-matching baseline.
- Candidate closure-rule finding: "Any post-hoc composition reweighting layered on an oracle-matching token router has zero headroom; K2-style improvement criteria are structurally unreachable at test time." Distinct from F#503 (empirical pointer_routing_no_merge); this is the general closure rule for oracle-ceiling composition experiments.

**Backlog state:** P=1 open remaining: `exp_p1_t5_user_local_training` (macro scale, last P=1). This experiment was P=2; now closed. Nothing stuck in `experiment list --status active`.
