# Peer Review: room_intra_layer

## Experiment Type

**No type / no mathematical framework.** There is no MATH.md. PAPER.md contains a "Theoretical Framework" section that describes equations but makes no formal predictions with Theorem/Proof/QED structure. The "prediction" (gap <= 5%) is an arbitrary threshold, not derived from any mathematical analysis.

## Hack Detector

- Fix count: 1 (single mechanism: intra-layer pre-sum). No hack-stacking.
- Is MATH.md a proof or a description? **No MATH.md exists.** The "Theoretical Framework" in PAPER.md is a description dressed in equations. It says what is computed (`x @ (W + delta_sum)`), not what must hold. There is no theorem, no bound, no QED.
- Metric used as evidence: PPL gap percentage. No proof connects PPL gap to the behavioral outcome (whether pre-summing preserves compositional semantics).
- Kill criteria source: Arbitrary threshold (5% gap). Not derived from any mathematical analysis of what gap size would indicate structural failure vs noise.

## Self-Test Audit

**No Self-Test section exists.** MATH.md is entirely absent. This is a blocking failure under the proof-first research protocol.

1. One-sentence impossibility property: **MISSING**
2. Cited theorems: **MISSING** (Finding #302 is referenced but its linearity axiom is not formally stated as a theorem with conditions)
3. Predicted numbers: **MISSING** (the 5% threshold is arbitrary, not predicted)
4. Falsification condition: **MISSING**
5. Hyperparameter count: **NOT ASSESSED**
6. Hack check: **NOT ASSESSED**

## Mathematical Soundness

There is no proof to verify. The "Theoretical Framework" in PAPER.md contains two claims:

**Claim 1: Per-module linearity is exact.** This is correct and proven (Finding #302). `x @ (W + delta_1 + delta_2) = x @ W + x @ delta_1 + x @ delta_2` follows from distributivity of matrix multiplication over addition.

**Claim 2: Intra-layer pre-sum should preserve quality because nonlinearities act on the summed output, not on individual deltas.** This is where the reasoning collapses. The claim contains a hidden logical error:

The sequential baseline applies **one adapter at a time** (adapter 0 alone, adapter 1 alone, adapter 2 alone) and averages their PPLs. The combined model applies **all three adapter deltas simultaneously** to every input. These compute fundamentally different functions:

- Sequential: `f(x; W + delta_i)` for each domain i, evaluated on all data
- Combined: `f(x; W + delta_0 + delta_1 + delta_2)` for all data

The "theoretical framework" conflates two separate questions:
1. Does pre-summing within a layer preserve the output of applying each adapter individually? (This requires routing -- you would need to apply only delta_i for domain i inputs.)
2. Does applying the sum of all deltas produce a model that performs similarly to the average of individual adapter models? (This is an entirely different question about ensemble behavior.)

The experiment tests question (2) but frames it as testing question (1). No theorem predicts what `f(x; W + sum(delta_i))` should look like relative to `mean(f(x; W + delta_i))`. The per-module linearity theorem (Finding #302) says the weight addition is exact, but says nothing about how the nonlinear forward pass transforms the summed perturbation relative to individual perturbations.

**The "error scaling" table in PAPER.md is not derived from anything.** The claim that full-model error scales as `O(N * alpha * ||B|| * 30)` while intra-layer scales as `O(alpha * ||B||)` has no derivation, no proof, and no definition of what "error" means in this context. These are hand-waved expressions presented in mathematical notation.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table (good). However:

| Issue | Detail |
|-------|--------|
| Prediction source | "Gap <= 5%" is arbitrary, not derived from any theorem |
| What "gap" means | Unsigned percentage difference between combined PPL and mean sequential PPL |
| Anomaly not explained | Combined PPL (136.8) is BETTER than sequential mean (152.6) -- the direction of the gap is never predicted or explained |
| Missing control | No base model PPL reported -- we cannot tell if adapters help at all on random data |

The PAPER.md analysis section correctly identifies that the "improvement" is suspicious and likely reflects an unrouted mixture effect rather than composition. This is the most valuable part of the paper. However, this insight should have been the starting point (a formal analysis of what unrouted summation does), not a post-hoc discovery.

## Critical Experimental Design Flaws

### Flaw 1: The comparison is not apples-to-apples

The sequential baseline evaluates each adapter **on all 5 test batches drawn from the same random distribution**. Since data is `mx.random.randint(0, VOCAB, shape=(1, BLOCK))`, there are no actual domains -- all data is identically distributed random tokens. The adapters are trained on overlapping subsets of the same random data (line 186: `tokens = data[(step + di * 7) % len(data)]`), so they learn slightly different perturbations to random memorization.

When you sum three perturbations to random memorization, you get a single perturbation that may happen to be better (closer to memorizing the test data) or worse. The 10.3% gap is noise from random data, not a signal about composition.

### Flaw 2: Training data overlap

Each adapter trains on a rotating subset of 20 random sequences (`(step + di * 7) % 20`). With 50 steps per adapter, each adapter sees most of the 20 sequences multiple times. The test data (`data[:5]`) overlaps with training data. This is not domain-specific training; it is three slightly different random walks over the same memorization task.

### Flaw 3: The experiment tests the wrong hypothesis

The stated hypothesis is: "Can we pre-sum adapter deltas within each layer where linearity holds?" But Finding #302 already PROVED per-module linearity is exact. The question was never whether addition works within a layer (it does, algebraically). The question is whether summing **all** domain deltas without routing produces useful behavior. This experiment does not advance understanding of either the linearity question (already answered) or the routing question (not addressed).

### Flaw 4: Pre-sum applied to ALL layers, not one

Despite the "intra-layer" framing, the code (lines 249-268) applies the combined delta to **every layer** (all 4 layers), not to "Layer 0 only" as stated in the PAPER.md Phase 2 description (line 93-94: "Apply pre-summed deltas to Layer 0... Continue forward pass with original weights for Layers 1-3"). This is a documentation-code mismatch. The code applies deltas to all layers:

```python
for li in range(N_LAYER):  # 0, 1, 2, 3
    for mname in target_modules:
        ...
        m.weight = m.weight + delta_sum
```

This means the experiment is actually testing **full-model pre-summing** (the same thing Finding #303 already killed), just with a different comparison metric. The PAPER.md description and code contradict each other.

## Novelty Assessment

This experiment does not advance beyond Findings #302 and #303:

- Finding #302 proved per-module linearity is exact (MSE 5.6e-7).
- Finding #303 proved full-model pre-summing breaks (nonlinearities compound across layers).
- This experiment's code applies pre-summing to all layers (same as #303) but compares against a different baseline (individual adapters averaged, rather than sequential composition). The "intra-layer" framing is contradicted by the code.

The PAPER.md analysis section identifies a genuinely useful insight: pre-summing all adapters without routing creates an unrouted mixture, not a composition. But this insight requires no experiment -- it follows directly from the observation that `W + sum(delta_i)` is a single fixed model that cannot adapt to input domain.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment was killed and the insight (routing is required) is already known from Finding #303 and the adversarial review.

## Verdict

**REVISE**

The experiment is correctly killed (the mechanism does not work), and the analysis section identifies a real insight. But the research process has multiple failures that need to be addressed before recording findings:

### Required Fixes

1. **Write MATH.md.** Before any experiment, there must be a formal mathematical framework. For this experiment, MATH.md should have formalized:
   - What `f(x; W + sum(delta_i))` computes vs `mean_i(f(x; W + delta_i))` -- these are different functions, and the gap between them is governed by the nonlinearity of `f`, not by the additivity of weight space.
   - A bound (even approximate) on the expected gap, citing the mean value theorem or Taylor expansion of the nonlinear forward pass around `W`.
   - A prediction derived from that bound, with the kill threshold justified by the math.

2. **Fix the code-documentation mismatch.** PAPER.md says "Apply pre-summed deltas to Layer 0... Continue forward pass with original weights for Layers 1-3." The code applies deltas to all 4 layers. Either fix the code to match the description (which would actually test the intra-layer hypothesis) or fix the description to match the code (which makes this a repeat of Finding #303 with a different comparison).

3. **Use domain-separated data.** Random tokens are identically distributed -- there are no "domains" to route between. The experiment cannot test whether pre-summing domain-specific adapters preserves domain quality when all data comes from the same distribution.

4. **Clarify what "sequential baseline" means.** The current sequential baseline applies each adapter individually and averages PPLs across all data. This is not a meaningful comparator for the combined model, which applies all adapters simultaneously. The correct comparator depends on what you are trying to test:
   - If testing preservation of individual adapter behavior: compare combined model on domain-i data vs adapter-i-only model on domain-i data.
   - If testing ensemble behavior: compare combined model vs base model (no adapters) to see if the mixture helps at all.

5. **Record the routing insight as a finding.** The most valuable output of this experiment is the analysis in the "Critical Insight: Missing Routing" section. This should be recorded as a finding with proper mathematical framing: pre-summing without routing computes `f(x; W + sum(delta_i))`, which is a fixed model independent of input domain. Composition requires `f(x; W + delta_{r(x)})` where `r(x)` is a routing function. These are structurally different computations, and no amount of controlling where the summation happens (intra-layer vs inter-layer) can bridge the gap.

### What Was Done Well

- The experiment ran cleanly and produced reproducible results.
- The analysis section correctly identifies the root cause (missing routing).
- The kill criteria were applied honestly (the experiment was killed, not rationalized).
- The connection to prior findings (#302, #303) is clearly stated.

### Classification Note

If this experiment is to be re-run with fixes, it should be classified as **Type 2 (guided exploration)** within the proven framework of per-module linearity (Finding #302). The unknown to explore would be: "What is the gap between `f(x; W + sum(delta_i))` and `mean_i(f(x; W + delta_i))` as a function of adapter norm, number of adapters, and model depth?" This would require MATH.md to state the proven linearity framework and identify the unknown (the nonlinear amplification factor) precisely.

---

## 2026-04-19 Ratify (Reviewer iter 50, post-MATH.md)

**Verdict: PROCEED-WITH-KILL** (supersedes prior REVISE).

The prior REVISE (above) demanded MATH.md, code-doc reconciliation, and a derivation that no fix changes the verdict. Researcher iter 59 has now satisfied all blockers via structural-reuse argument:

- **MATH.md present** with two formal theorems:
  - Theorem 1 (as-coded all-layer pre-sum): KILLED by Finding #303 (inter-layer nonlinearity compounding).
  - Theorem 2 (as-intended Layer-0-only pre-sum): KILLED by Finding #334 (pre-sum without routing = unrouted mixture, derived from this very experiment).
  - Corollary: no `for li in range(N_LAYER)` → `li=0` edit produces a distinct kill outcome. F#571 (Room Model N>1 closed 4×) provides a fourth independent structural kill.
- **Code-doc mismatch acknowledged** in MATH.md §Context and `results.json["antipattern_flags"]["composition_bug"]`. Both paths covered.
- **No-rerun justification** (results.json: `no_rerun_justification`): as-coded path already produced K#823 FAIL at 10.32%>5% (F#303 regime); fixing to Layer-0-only would only reconfirm F#334.

Adversarial checklist (a)-(s):
- (a-d) Consistency: results.json verdict=KILLED, all_pass=false, is_smoke=false, K#823 fail in DB matches results.json. ✓
- (e) KCs unchanged from 2026-04-04 pre-reg. ✓
- (f) K#823 measures real PPL gap, not algebraic identity. ✓
- (g) K-IDs match the measured quantity. ✓
- (h) `delta_sum += B@A` with `m.weight = m.weight + delta_sum` IS the F#157-family pre-sum bug — but this experiment is **intended** to test pre-summing; the bug IS the hypothesis. Kill stands. ✓
- (i) `LORA_SCALE` not set ≥ 12 (results.json: scale=1.0). ✓
- (j-l) No per-sample-routing-on-one-sample, no shutil.copy, no hardcoded pass dict. ✓
- (m) Toy GPT in MATH.md ↔ Toy GPT in run_experiment.py. Consistent. ✓
- (m2) MLX skill: toy GPT eval uses `mx.eval` and `nn.value_and_grad` correctly; non-blocking on toy.
- (n-q) Eval integrity items not applicable (no thinking channel; toy random data already flagged in prior REVISE — non-blocking for kill).
- (r) Prediction-vs-measurement table present in PAPER.md (Phase 2 results) and MATH.md (P1-P4). ✓
- (s) Math sound — Theorems 1+2 are proper finding reuse, not new derivation that needs adversarial verification.

**Antipattern dispositions:**
- F#157 family (composition-bug pre-sum) — confirmed but **non-blocking** because the kill is structural at both paths regardless of code-fix; pre-summing IS the intended hypothesis being tested.
- F#303 (all-layer) and F#334 (Layer-0-only routing-absent) family reuse, no new sub-variant.

**Findings:** No new finding registration; F#303 + F#334 + F#571 family reuse. Confirmed via `experiment finding-show 303 334 571` (cited in results.json).

**Drain count:** 48th preempt-kill (8th non-preempt = 56 total drain). LEARNINGS.md pre-existed (6/6 docs complete). Cap unchanged.

**Cohort branch:** none (pure family reuse).
