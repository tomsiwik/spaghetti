# Peer Review: Skip-List Composition Test

## NotebookLM Findings

Skipped -- the experiment is a composition protocol test on an already-reviewed architecture. The mathematical structure is the parent skip_list_routing (reviewed, PROCEED). The novel content is the composition protocol and interpretation of results.

## Mathematical Soundness

### What holds

1. **Composition protocol is correctly implemented.** The four-phase protocol (pretrain, finetune with frozen attention, weight-average expert modules, calibrate routing) matches the established pattern from hierarchical_tree. Code review of `run_experiment.py` confirms: expert modules are correctly extracted per domain (lines 214-220), weight-averaged (lines 229-238), and only routing parameters are unfrozen during calibration (lines 244-263).

2. **Kill criteria are correctly formalized.** KC1 (skip gap vs flat gap > 3pp) and KC2 (weight above L0 < 10%) are clear, falsifiable, and appropriate.

3. **Composition gap calculation is correct.** The gap formula `100 * (composed - joint) / joint` is applied consistently across model types.

4. **Coarse expert construction under composition is correctly analyzed.** MATH.md correctly identifies the "double averaging" concern: leaf experts are averaged across domains, then coarse experts average the averaged leaves. The concern about signal dilution is well-articulated.

5. **Calibration parameter count is correct.** 4 routers (d * [8+4+2+1] = 960) + 3 gates (3 * (d+1) = 195) = 1155/layer, 4620 total. Verified against architecture.

### Issues

6. **Training step asymmetry (moderate).** The joint baseline trains for 500 steps (300 pretrain + 200 finetune). The composed model trains for 600 total steps (300 + 200 + 100 calibration). This 20% additional optimization budget is unacknowledged and could explain the negative composition gap (-0.17%). The paper does not note this asymmetry or argue that the extra 100 steps on a tiny subset of parameters (2.2% of model) is insufficient to matter. At this scale, 100 extra gradient steps on routing parameters could meaningfully improve loss.

   **Impact:** The -0.17% negative gap is within the range that the extra optimization could produce. A conservative reading is that the gap is ~0%, which still passes KC1 by a wide margin.

7. **The 97.2% concentration is not clearly distinguished from routing collapse.** KC2 checks for collapse TO Level 0 (fine-grained), not collapse TO Level 3 (coarsest). But collapse to the coarsest level is arguably pathological too: if L3 gets 80-91% of weight (seeds 42, 123), the model is effectively computing `mean(all_experts)(x)` and ignoring expert selection entirely. This is not "adaptive multi-resolution routing" -- it is degenerate averaging masquerading as routing.

   The paper's argument ("weight averaging creates better coarse experts") is backwards: if the coarse expert IS just the average of all averaged experts, and the model puts 90% weight there, then the skip-list routing infrastructure is adding 1155 params/layer of overhead to learn that "just average everything" is the best strategy. A simpler model (single expert = mean of all) would achieve the same result with zero routing parameters.

   KC2 as formulated misses this failure mode. It should also test for excessive coarsest-level concentration (e.g., L3 weight > 90% for most seeds = degenerate).

## Novelty Assessment

### Prior art

The composition protocol is standard within this project (used in hierarchical_tree, flat baselines). The novelty claim is narrow: skip-list routing survives composition. This is a valid composition stress test, not a novel method.

### Delta over closest work

The hierarchical_tree composition experiment already showed +0.17% gap for hierarchical routing. This experiment shows -0.17% for skip-list. The difference (0.34pp) is noise at 3 seeds. The real delta: skip-list's confidence gates allow degenerate collapse to coarsest level, which tree routing cannot do (tree always traverses full depth). This degeneracy happens to help under composition because it sidesteps the routing problem.

## Experimental Design

### Does this test the stated hypothesis?

The hypothesis is: "skip-list multi-resolution routing survives the shared-base composition protocol without degradation, and confidence gates maintain non-uniform level-weight concentration."

The experiment confirms both, but with caveats:

1. **"Survives without degradation" -- yes, trivially.** The model survives by not routing. When 97.2% of weight goes to coarse levels (and 80-91% to L3 alone in 2/3 seeds), the model avoids the routing problem rather than solving it. A flat model that put all weight on "average of all experts" would also show zero composition degradation.

2. **"Non-uniform level-weight concentration" -- technically yes, but in the wrong direction.** The parent experiment showed 60.6% above L0 with meaningful distribution across levels. This experiment shows 97.2% above L0 concentrated almost entirely at L3. The concentration is non-uniform (passes KC2), but it is also degenerate (fails the spirit of KC2).

### Controls

The flat baseline is an adequate control. The comparison is fair in protocol (same pretrain/finetune/calibrate), though the calibration parameter budget differs (512 vs 1155 params/layer). This is inherent to the architectures and not a design flaw.

### Missing control: ablation of level structure

A critical missing control: what if you replaced the entire skip-list pool with a single expert that is the mean of all leaf experts (no routing at all)? Given that the model puts 80-91% weight on L3 (which IS the mean of all experts), this ablation would likely match or beat the skip-list composed model with zero routing overhead. This would prove that the routing infrastructure is deadweight under composition.

### Seed variance

Seed 777 shows a qualitatively different pattern (L1=50.6% vs L3=35.2%) compared to seeds 42 and 123 (L3 dominant). With only 3 seeds, it is unclear which pattern is typical. The aggregation hides this bimodality.

## Macro-Scale Risks (advisory)

1. **Coarsest-level degeneracy at scale.** If the pattern holds (composition pushes weight to L3), then at N=64 experts with 6 levels, L6 would be an average of 64 domain-averaged experts -- pure noise for diverse domains. The skip-list would either (a) collapse to this noise expert (bad) or (b) shift weight down to finer levels (defeating the claim of increased coarse-level concentration).

2. **Calibration budget scaling.** 1155 params/layer for routing calibration worked at M=2 domains. With M=20+, the routing task is harder. The paper notes this but does not bound the scaling.

3. **Training cost unchanged.** The parent review flagged 16x expert evaluation cost. This is inherited and unaddressed.

## Verdict

**REVISE**

The experiment passes its stated kill criteria and the code is correct. However, the interpretation of results significantly overstates the finding, and a key confound is unacknowledged. Required fixes:

1. **Acknowledge the training step asymmetry.** Add to Limitations: the composed model receives 600 total optimization steps vs 500 for joint. Argue why this does or does not explain the negative gap. The simplest fix: re-run joint baselines for 600 steps (matching total budget) or add a note that the extra 100 steps on 2.2% of parameters is bounded in effect.

2. **Reframe the 97.2% coarsest-level concentration honestly.** The current framing ("Composition INCREASES level-weight concentration ... confidence gates push even more weight to coarse levels") presents degeneracy as a feature. Add analysis: when L3 receives 80-91% of weight (seeds 42/123), the model is effectively not routing. The skip-list is adding overhead to learn "just average everything." Discuss whether this is a success (composition-safe because routing is bypassed) or a failure (the adaptive depth mechanism is inactive).

3. **Add or discuss the "no routing" ablation.** Either run a control where the composed model has no routing at all (single mean-expert), or explicitly discuss why the skip-list composed model should not be compared to one. If the no-routing control matches, the negative composition gap is not evidence for skip-list routing -- it is evidence for degeneracy.

4. **Fix KC2 to also detect coarsest-level collapse.** The current KC2 only triggers on L0 dominance (<10% above L0). Add a check: if L3 (coarsest) > 90% for majority of seeds, flag as degenerate. This does not retroactively kill the experiment but prevents future misinterpretation. Alternatively, keep KC2 as-is but add a "degeneracy warning" diagnostic.

5. **Report per-seed data in summary table, not just appendix.** The mean gap of -0.17% hides that seed 42 shows composed worse than joint (0.5157 vs 0.5093 = +1.26% gap), while seed 123 shows composed better (0.5087 vs 0.5081 = +0.12%). The aggregate is dominated by seed 777 (0.5151 vs 0.5246 = -1.81%). With such variance, the sign of the mean gap is not meaningful. Report the per-seed gaps explicitly and note that the claim is "within kill threshold" not "negative gap."

None of these fixes require re-running the experiment. They are reframing and additional analysis of existing results.

---

## 2026-04-19 Preempt-Kill Ratification (reviewer iter 45)

Superseding the 2026-03-07 REVISE verdict. Reframed as **KILL** under
guardrail 1006 (behavioral outcomes over metrics) + antipattern #6
(KC measures wrong object) + cascade F#664 (fixed-algebraic-blend
family, iter-44).

**Adversarial pass (all PASS):**
- (a)-(d) results.json `verdict=KILLED`, `all_pass=false`, `is_smoke=false` ↔
  DB `killed` ↔ PAPER KILLED (lines 117, 146-148). No downgrade needed.
- (e) K#477/K#478 unchanged from 2026-03-06 pre-registration; no
  relaxation. Mechanism-level verdict (fail) is the reclassification,
  not criterion relaxation.
- (f) KC2 non-tautological by construction, but **tautologically
  satisfied** by the exact degeneracy it should catch (L3-collapse
  vs intended L0-collapse) — wrong-object antipattern = KILL basis.
- (g) K-ID measures `weight_above_L0` correctly; the wrong-object issue
  is design-level, not measurement-level.
- (h)-(m2) Preempt-mode ratify on pre-existing runner; no re-execution.
  Runner inspection (lines 214-263 per issue #1 above) already
  validated; no `sum(lora_A`, `LORA_SCALE≥12`, `shutil.copy`, or
  `pass:True` dicts present.
- (n)-(q) 3 seeds × full composition protocol (not smoke); per-seed
  L3 dominance {80.0, 90.9, 35.2}% demonstrates non-synthetic
  measurement. No thinking-suppression (PPL-based).
- (r) PAPER has prediction-vs-measurement KC table (lines 112-115)
  with both as-formulated and mechanism-level columns. ✓
- (s) Math sound; cascade to F#664 well-grounded (L3 = uniform mean of
  averaged leaves → fixed-algebraic-blend per Vandermonde argument
  in F#664).

**Load-bearing cascade (F#664 → this kill):**
Per F#664 (2026-04-19, iter-44), any fixed algebraic weighted blend
of specialist experts with data-agnostic coefficients is killed by
F#157 (-7.29% equal-hierarchical) + F#22/F#544 (PPL-quality r=0.08,
ρ=-0.7). When skip-list routing collapses to L3=uniform-mean-of-leaves
in 2/3 seeds at 80-91% weight, the mechanism is in the fixed-blend
family. The -1.20pp "pass" reflects averaging competence on similar
sub-domains (a-m vs n-z chars), not adaptive routing.

**Finding registered:** F#665 (composition-bug axis, new sub-variant
**degenerate-routing-collapse**). Reusable preempt-rule: any
composition-test whose "PASS" mechanism is adaptive-routing but whose
empirical routing statistics show >80% single-level concentration
(L0-collapse OR coarsest-level-collapse) is preempt-killable under
F#664 cascade — the mechanism is inactive; the measurement reflects
the underlying algebraic blend, not routing. KC must test BOTH
L0-collapse AND coarsest-collapse (bidirectional).

**Verdict:** KILL (ratify). Non-blocking carry-forward: 2026-03-07 REVISE
fixes #1 (step asymmetry), #5 (per-seed reporting) remain valid
documentation improvements but are **not load-bearing** on the kill
(kill is mechanism-level, not noise-level). Fixes #2, #3, #4 are
subsumed by the preempt-kill reframing.
