# Peer Review: MLP-Only Per-Token Routing (RE-REVIEW)

## Experiment Type
Guided exploration (Type 2)

**Framework:** MoE architecture principle (shared attention + routed FFN).
**Unknown:** Whether MLP-only LoRA per-token adaptation provides sufficient domain
signal on this specific ternary base + adapters.

Type 2 requirements: MATH.md states the proven framework (MoE shared-attention +
routed-FFN, citing Mixtral and Switch Transformer) and identifies the unknown
(magnitude of MLP-only domain signal). The experiment narrows the unknown (MLP
contributes ~6x more per-token signal than attention). PASS.

## Hack Detector
- Fix count: 1 (MLP-only restriction). Not a hack stack.
- Is MATH.md a proof or a description? A correct proof sketch (Proposition with
  QED) of MLP token-independence -- a real property, but one that the experiment
  never actually tests in its proven form (single-pass mixed-adapter). This is
  acknowledged post-revision.
- Metric used as evidence: PPL delta from oracle per-token selection. Appropriate
  for measuring adapter component signal strength. Not sufficient for testing
  contamination elimination.
- Kill criteria source: K790 derived from the proof's prediction that contamination
  elimination improves PPL. K791 derived from the proof's ambition. K792 retired
  as vacuous (correct). Kill criteria are reasonable for the Type 2 exploration
  even though the contamination elimination mechanism itself was not tested.

## Self-Test Audit

1. **One-sentence impossibility property:** "MLP token-independence: MLP(x_t) depends
   only on x_t, so per-token MLP routing cannot contaminate other tokens'
   representations." Correctly states one property. PASS.

2. **Cited theorems:** Vaswani et al. (2017) attention mechanism definition, Mixtral
   (Jiang et al., 2024, arxiv 2401.04088), Switch Transformer (Fedus et al., 2022,
   arxiv 2101.03961). Real references. Mixtral/Switch are architectural designs
   validated at scale, not theorems per se, but this is appropriate for a Type 2
   guided exploration that operates within an established architectural principle
   rather than proving a new one. PASS.

3. **Predicted numbers:** K790 < 4.815, K791 < 4.042, K792 diff > 0.01. Specific
   and falsifiable. K792 is now correctly retired as vacuous in PAPER.md. The
   predictions target the experimental methodology that was actually run (multi-pass
   oracle selection), not the proof's mechanism (single-pass mixed-adapter). This
   gap is acknowledged. PASS (with caveat noted).

4. **Falsification condition:** "The proof is wrong if MLP operations have cross-token
   dependencies." Targets the proof correctly. The proof is trivially true for
   standard SiLU MLP (no serious risk of falsification), which is why this
   experiment is a Type 2 exploration (the proof is safe; the unknown is the
   magnitude of signal). PASS.

5. **Hyperparameter count:** 0. Correct. PASS.

6. **Hack check:** Correctly identifies this as an architectural reframe, not a
   stacked fix. PASS.

## Revision Fix Verification

All 6 required fixes from the first review have been applied:

| # | Required Fix | Applied? | Quality |
|---|-------------|----------|---------|
| 1 | Experiment-proof gap at top of PAPER.md | YES | Clear 3-paragraph disclaimer at top of PAPER.md and MATH.md. States exactly what was proved vs what was tested. Excellent. |
| 2 | Reframe primary finding to MLP 6x signal | YES | Finding #1 in PAPER.md explicitly labeled "(PRIMARY FINDING)" and reframed as "empirical observation about adapter component decomposition, not a test of the contamination elimination thesis." Correct. |
| 3 | K792 retired as vacuous | YES | PAPER.md explains why any multi-pass oracle scheme passes K792. results.json shows K792_pass = "RETIRED" with note. Correct. |
| 4 | Finding #305 null reclassified | YES | Both MATH.md and PAPER.md call it a "methodological artifact" and note contamination is "untested -- circumvented, not confirmed." Correct. |
| 5 | Variance/significance analysis added | YES | Paired t-test table: t(9)=4.69, p<0.001 (percentage); t(9)=3.00, p<0.02 (raw PPL). Per-pair range and SD reported. **Independently verified: arithmetic is correct.** |
| 6 | Status downgraded to PROVISIONAL | YES | PAPER.md line 34, results.json line 443 both show PROVISIONAL. Correct. |

## Mathematical Soundness

### What holds

The Proposition in MATH.md Section C is **mathematically correct**: if attention uses
base weights for all tokens and MLP is applied per-token independently, then
per-token MLP routing cannot produce cross-attention contamination. The proof
sketch (steps 1-3 plus QED) is valid. The multi-layer propagation discussion
(lines 106-117) honestly acknowledges that MLP-adapted residuals flow through
subsequent base attention and correctly argues this matches MoE design.

### What the proof does not cover

The proof guarantees structural isolation. It does not predict the magnitude of
MLP-only signal vs full-module signal. This is correctly identified as the Type 2
unknown and is what the experiment explores.

### Residual inconsistency (minor)

MATH.md Section D still contains the K792 prediction (line 159:
"|MLP-only per-token PPL - per-sequence PPL| > 0.01") without noting its retirement.
Since MATH.md was written before the experiment ran, this is the pre-registered
prediction, and the retirement is properly documented in PAPER.md and results.json.
This is acceptable -- the prediction-vs-measurement table in PAPER.md is the
authoritative record. **Not blocking.**

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table (lines 16-23).

| Prediction | Measured | Match? | Assessment |
|-----------|----------|--------|------------|
| K790: MLP-only < 4.815 | 4.656 | YES (3.3%) | Genuine improvement, statistically significant (p<0.001) |
| K791: MLP-only < 4.042 | 4.656 | NO (+15.2%) | Honest failure. Segment isolation dominates. |
| K792: diff > 0.01 | 0.159 | RETIRED | Correctly retired as vacuous |
| Control: full vs seq | diff = 0.315 | Contradicts F#305 | Properly attributed to methodological difference |

The table is honest and complete. K791 FAIL is not hidden. K792 retirement is
well-justified. The control check (full-module per-token 4.500 beats per-sequence
4.815 in multi-pass) is correctly identified as showing that multi-pass eliminates
contamination for ALL strategies, not just MLP-only.

**Key insight the paper correctly surfaces:** Full-module per-token (4.500) beats
MLP-only (4.656) in multi-pass, directly contradicting the contamination motivation
for MLP-only routing. In the multi-pass regime where contamination cannot occur,
more adapter signal (full-module) is strictly better. The paper correctly concludes
that MLP-only is only advantageous in single-pass where contamination actually occurs.

## Statistical Verification

**Independently verified the reported statistics:**

Percentage improvements across 10 pairs: [3.39, 2.15, 0.90, 2.04, 1.57, 3.60,
6.43, 0.77, 4.91, 6.94]. Mean = 3.27%, SD = 2.21%, SE = 0.698%.
t(9) = 3.27/0.698 = 4.69. CONFIRMED.

Raw PPL diffs across 10 pairs: [0.100, 0.060, 0.053, 0.083, 0.053, 0.269, 0.336,
0.042, 0.240, 0.680]. Mean = 0.191, SD = 0.202, SE = 0.064.
t(9) = 0.191/0.064 = 3.00. CONFIRMED.

The statistics are correct and the effect is genuinely non-zero.

## Novelty Assessment

The MoE principle (shared attention + routed FFN) is well-established. Applying it
to LoRA adapter composition is a reasonable but not novel architectural idea.

The genuine novel contribution is the empirical decomposition: MLP adapters carry
~6x more per-token domain signal than attention adapters (3.3% vs 0.5%). This is
consistent with Finding #304's perturbation energy split (MLP ~69%, attention ~31%)
but provides a complementary measurement from a different angle (per-token NLL
selection vs perturbation energy). Having two independent measurements converge on
the same conclusion (MLP >> attention for domain signal) strengthens the finding.

## Macro-Scale Risks (advisory)

1. **O(N) forward passes do not scale.** Multi-pass oracle selection requires N
   forward passes. Production needs single-pass mixed-adapter MLP routing -- the
   exact scenario the proof covers but the experiment does not test. The experiment
   that the proof calls for (single-pass mixed-adapter comparison) is explicitly
   listed as needed future work (PAPER.md lines 203-206). Not blocking for micro.

2. **15% gap to segment isolation persists.** K791 FAIL shows that cross-domain
   context degradation (not adapter contamination) is the dominant effect. This
   is a fundamental limitation of full-sequence per-token routing at any scale.

3. **Adapters trained full-module, applied MLP-only.** Post-hoc ablation may not
   represent purpose-trained MLP-only adapters. Finding #308 suggests post-hoc
   outperforms purpose-trained, but this is from a different experiment.

## Remaining Issues

### Issue A: Code-level methodology mismatch (acknowledged, not blocking)

The `compute_mlp_only_per_token_ppl` function (lines 260-329) applies the full
adapter then zeros attention lora_b. While mathematically correct (zero lora_b
means zero LoRA contribution), the lora_a matrices for attention are still loaded
with non-zero values. These are multiplied by zero lora_b so they have no effect,
but this is worth noting for anyone reading the code expecting a "clean" MLP-only
forward pass. Not a bug -- just a readability note.

### Issue B: Attention-only near-null interpretation (minor ambiguity)

PAPER.md line 105 says the attention-only result (0.5% improvement) is "consistent
with our contamination theory (attention LoRA's benefit is partially cancelled by
cross-adapter K/V mixing)." But the multi-pass methodology eliminates contamination
for attention-only too (each pass uses one domain's attention adapter for all
tokens). So the small attention signal cannot be attributed to contamination -- it
simply means attention adapters carry less domain-specific per-token information
than MLP adapters. PAPER.md Finding #2 (lines 73-83) correctly identifies this for
the full-module case but the attention-only interpretation on line 105 still leans
on the contamination framing. **Minor -- the correct interpretation is stated
elsewhere in the same document.**

### Issue C: per_seq_best is also oracle (semantic note)

"per_seq_best" selects the best single adapter across the entire sequence via oracle
NLL comparison (5 forward passes, pick the one with lowest total NLL). Both
per_seq_best and per_token_* strategies run 5 forward passes. The difference is
granularity of selection (sequence-level vs token-level). This is correctly described
in the code comments but worth making explicit in the methodology section. **Not
blocking.**

## Verdict

**PROCEED**

### Justification

All 6 required fixes from the first review have been correctly applied. The
experiment-proof gap is clearly acknowledged at the top of both MATH.md and PAPER.md.
The primary finding has been reframed to what was actually measured (MLP 6x signal
contribution, empirical). K792 is retired. Finding #305's null is reclassified as a
methodological artifact. Statistics are correct and independently verified. Status
is appropriately PROVISIONAL.

The remaining issues (B and C above) are minor interpretive ambiguities that do not
affect the validity of the finding. The paper is now honest about what it proves
(MLP token-independence), what it measures (multi-pass oracle per-token adapter
signal decomposition), and what it does not test (single-pass mixed-adapter
contamination elimination).

**The finding "MLP adapters contribute ~6x more per-token signal than attention
adapters" is a genuine, statistically significant (p<0.001) empirical observation
that converges with Finding #304's perturbation energy split. PROVISIONAL status
is appropriate for a Type 2 guided exploration that narrowed the unknown (MLP vs
attention signal magnitude) without testing the proof's mechanism (single-pass
contamination elimination).**

### What should come next

The natural follow-up is the single-pass mixed-adapter experiment that MATH.md's
proof actually covers: apply domain A's MLP adapter to tokens 0-127 and domain B's
MLP adapter to tokens 128-255 in ONE forward pass. Compare against multi-pass oracle
MLP-only. If MLP token-independence holds (which it must, by the proof), these should
match. This would upgrade the finding from PROVISIONAL to SUPPORTED.

---

## Audit-Rerun Addendum (2026-04-18, reviewer)

**Prior verdict PROCEED/PROVISIONAL is superseded by KILLED (audit-rerun closure).**

Tags: `audit-2026-04-17-rerun, tautological-routing`.

### Closure verification

PAPER.md §"Audit-Rerun Closure" proves no fix to the tautological-routing bug
rescues K791. I verified:

1. **Oracle interpretation correct.** `compute_mlp_only_per_token_ppl` runs 5
   forward passes (one per adapter) and selects per-token adapter by minimum NLL
   at the *true* label. This is the Bayes-optimal per-token adapter assignment
   with full label knowledge → genuine lower bound on PPL(R) for any router R.
2. **Arithmetic.** Oracle PPL = 4.656; K791 threshold = 4.042. Gap 15.2% >> 0.
   Any R: PPL(R) ≥ 4.656 > 4.042, so K791 unreachable. ∎
3. **Tautology direction correct.** "Router = evaluation criterion" *inflates*
   the apparent gain (router peeks at labels). Fixing the bug removes label
   leakage → PPL weakly increases → K791 gap cannot shrink.
4. **K790 preserved under fix.** Oracle NLL selection trivially beats
   per-sequence single-adapter selection (4.656 < 4.815 = per-seq).
5. **K792 retirement correct.** Any multi-pass oracle scheme satisfies K792 by
   construction; criterion is vacuous.

### Closure-rule promotion

This is the **second instance** of the oracle-ceiling antipattern family:
- 1st: `exp_depth_routed_adapters` → `ap-oracle-ceiling-blocks-headroom` (K2:
  gamma_oracle == gamma_token → 0% headroom).
- 2nd: `exp_mlp_only_per_token_routing` → **`oracle-upper-bound-blocks-kill-threshold`**:
  when a kill criterion fails under an oracle upper bound on the same data,
  no routing-mechanism fix can salvage it.

Promote the candidate rule to a full closure-rule; future audit reruns should
check this before proposing a rerun.

### Adversarial checklist (delta from prior review)

- (a) results.json verdict "PROVISIONAL" vs current "killed" — expected
  audit-rerun pattern; PAPER.md closure addendum is authoritative, results.json
  is frozen raw-run output. Not blocking.
- (f) Tautology sniff test: researcher **self-identified** the tautological
  router (router ≡ per-token NLL minimiser = evaluation criterion). Caught and
  documented correctly.
- (g) K-ID measurement vs MATH.md: K791 measures multi-pass oracle, while
  MATH.md's proof describes single-pass mixed-adapter. Gap acknowledged in
  PAPER.md §"CRITICAL: Experiment-Proof Gap" — closure does not rely on
  closing this gap.
- Code ↔ math (h–m): unaffected by audit rerun; prior review passed these.

Preserved genuine finding: MLP adapters contribute ~6x more per-token signal
than attention (3.3% vs 0.5%, t(9)=4.69, p<0.001) — orthogonal to K791,
documented in LEARNINGS.md. Do not let the kill erase this.

### Verdict (audit-rerun)

**KILL** — closure robust to the tautological-routing bug fix; K791
structurally unreachable under every implementation on this base + adapters.

Route: `review.killed` → Analyst (LEARNINGS.md already present from prior
Analyst pass; no rewrite required).
