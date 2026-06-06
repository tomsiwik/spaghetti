# Peer Review: Softmax Router Scaling

## NotebookLM Findings

Skipped -- reviewing directly from source materials as the critical issues are code-level, not conceptual.

## Mathematical Soundness

### What holds

1. The softmax vs. binary head argument is correct. N independent binary heads face 1:(N-1) class imbalance; a single multi-class head with cross-entropy does not. The gradient analysis in MATH.md Section 2 is standard and accurate.

2. The parameter count analysis is correct: 330K (softmax) vs. 1.97M (24 binary heads).

3. The gamma metric is computed correctly: avg(adapted_ppl) / avg(base_ppl), consistent with the prior experiment.

4. Oracle PPLs match exactly between this experiment and exp_more_adapters_is_better at N=24, confirming the same adapters/data are used. This is good experimental hygiene.

### What does not hold

1. **MATH.md makes no mathematical claim about interchangeability.** The paper's central finding is "adapters are functionally interchangeable" but MATH.md contains zero derivation of why this should be true. The Grassmannian orthogonality (mean |cos|=0.0238) is cited but never connected to PPL equivalence through any formal argument. This is a hand-wave dressed as an explanation. Low cosine similarity between A-matrices means low interference in the *A-projected subspace*, but says nothing about whether different B-matrices produce equivalent output when applied to out-of-domain text. The claim requires showing that for text x from domain i, applying adapter j gives similar loss -- this is an empirical statement about the B-matrices and the data distribution, not a consequence of A-matrix orthogonality.

2. **The "Grassmannian skeleton ensures adapter interference is near-zero" does not logically imply "any adapter gives same PPL."** These are different claims. Low interference means activating adapter j does not degrade the contribution of adapter i (relevant for multi-adapter composition). Interchangeability means adapter j alone performs as well as adapter i alone on domain i's data. Orthogonality of A-matrices is necessary for the first, irrelevant for the second.

## Novelty Assessment

The softmax router itself is not novel -- the paper correctly cites MoLoRA (arXiv 2603.15965). The claimed novelty is the finding that routing accuracy does not matter for PPL. This finding, if valid, would be genuinely interesting. However, it has a much simpler explanation than "Grassmannian interchangeability" (see Experimental Design below).

## Experimental Design

### Critical Issue 1: Routing decision uses mean of ALL val hidden states

Lines 689-695 of run_experiment.py:
```python
val_h = all_hidden[domain_name]["val"]
h_mean = mx.mean(val_h, axis=0, keepdims=True)
logits = router(h_mean)
```

The routing decision for each domain is made on the **mean of all 50 validation samples' hidden states**. This is a single centroid vector per domain. The router sees one input and picks one adapter for the entire domain's PPL evaluation.

This means the "routing accuracy" that determines PPL (12/24 correct) is the router's accuracy on 24 centroid vectors, not the per-sample accuracy (39.75% on 1200 samples). The paper conflates these two numbers in its discussion. The 39.75% figure measures something different from what actually determines the PPL numbers.

**This design choice is defensible** for per-sequence routing (you route the whole domain at once), but it means the 39.75% accuracy figure is irrelevant noise. The actual routing accuracy that matters is 12/24 = 50%, which would PASS K1.

### Critical Issue 2: The "interchangeability" claim has a simpler explanation

Look at which domains get misrouted and their oracle gaps:

| Domain | Selected | Oracle Gap |
|--------|----------|------------|
| science | agriculture | +0.03% |
| history | agriculture | -1.10% |
| philosophy | agriculture | -0.01% |
| creative_writing | agriculture | +0.03% |
| environmental | agriculture | +0.92% |
| politics | agriculture | +0.64% |
| economics | agriculture | +1.16% |

Seven out of twelve misrouted domains go to agriculture. The max oracle gap is 1.16% (economics).

The simpler explanation: **these adapters are all producing near-zero effective perturbation on out-of-domain text.** The LoRA contribution is `(x @ A_i) @ B_i * (scale/k)`. When text x is from a domain that the adapter was not trained on, the B-matrix may produce outputs that are small or cancel out across the sequence. The result would look identical to any other adapter's near-zero contribution. The experiment does not measure or report:
- The norm of `(x @ A_i) @ B_i` for in-domain vs. out-of-domain text
- Whether the LoRA contribution is actually being applied (could be vanishingly small)
- Whether the STE quantization `b_q = clip(round(b_scaled), -1, 1) * alpha` is zeroing out most of the B-matrix

**Without measuring the actual LoRA activation magnitude for in-domain vs. out-of-domain text, the "interchangeability" claim is unfalsifiable from this data alone.** The PPL numbers are consistent with "all adapters contribute almost nothing on out-of-domain text" rather than "all adapters contribute equally."

### Critical Issue 3: Agriculture dominance in misrouting

7/12 misrouted domains go to agriculture at N=24. The router validation metrics (lines 2237-2251 of results.json) show agriculture itself only gets 44% accuracy -- its own hidden states scatter across history, philosophy, economics, etc. This is a bidirectional confusion cluster, not one adapter "attracting" others.

However, the centroid-based routing decision routes to agriculture because agriculture's centroid happens to be near the center of this confused cluster in hidden-state space. This is a centroid-averaging artifact: when the router is confused among {agriculture, history, philosophy, politics, economics, environmental, science, creative_writing}, averaging 50 scattered hidden states per domain produces centroids that all point toward the same ambiguous region, and agriculture wins the argmax.

This is not evidence that "adapters are interchangeable." It is evidence that **these 8 domains have overlapping hidden-state representations in this base model**, and the centroid-based routing degenerates to always picking the same adapter for all of them.

### Critical Issue 4: Validation is 100% single-domain

Each domain's PPL is computed on 100% same-domain validation text. This is appropriate for measuring domain-specific quality, but it means we never test the router on mixed-domain or ambiguous inputs -- the actual use case where routing matters.

### What the experiment DOES show validly

1. The softmax router eliminates the binary head fallback problem (0% vs 46% fallback). This is correct and valuable.
2. Routed gamma beats uniform gamma at all N. This is correct: even a mediocre router that picks one adapter outperforms activating all N simultaneously.
3. The oracle gap at N=24 is small (max 1.2%). This is real but its explanation is ambiguous (see above).

### Training data adequacy

40 train samples per domain, 960 total. At N=24, the train accuracy is only 75.3% and final loss is 1.0037 (random would be ln(24)=3.18, so the router has learned something but is far from converged). The 500-step budget with batch size 32 means only ~16 passes through the data. This is likely undertrained, which contributes to the accuracy degradation. More training steps or a larger hidden dimension might improve accuracy, but this was not ablated.

## Macro-Scale Risks (advisory)

1. **The "interchangeability" finding almost certainly will not hold at scale.** With real task-specific metrics (code correctness, medical accuracy), selecting the wrong adapter will matter. The experiment's own Limitations section acknowledges this.

2. **Centroid-based routing will not work for per-sequence routing in production.** Each input must be routed individually, not as a batch centroid. The per-sample accuracy of 39.75% is the number that matters for deployment, not the 12/24 centroid accuracy.

3. **The confused cluster of 8 domains** (science, history, philosophy, creative_writing, agriculture, environmental, politics, economics) all have moderate base PPL (12-21) and moderate oracle PPL (7-15). At scale with better-separated data, this cluster may dissolve. Or it may persist if the base model genuinely cannot distinguish these domains in hidden-state space.

## Verdict

**REVISE**

The experiment has valid and useful findings (softmax eliminates binary head collapse, routed beats uniform) but the extraordinary "adapters are interchangeable" claim is not adequately supported. Specific fixes:

1. **Measure LoRA activation magnitudes.** For each domain's val text, compute `||xA_i B_i||` for both the correct adapter i and the selected adapter j. If these are both near-zero for out-of-domain text, the explanation is "adapters contribute nothing on out-of-domain text" not "adapters are interchangeable." If they are both large and similar, that is genuine interchangeability. This is 20 lines of code and would resolve the ambiguity entirely.

2. **Report the centroid-based accuracy (12/24 = 50%) separately from per-sample accuracy (39.75%).** The centroid accuracy is what determines the PPL numbers. The paper currently uses the per-sample accuracy to claim K1 FAIL, but the actual PPL evaluation uses centroid routing. If the correct accuracy metric is 50%, K1 might PASS.

3. **Ablate training steps.** The router at N=24 has final loss 1.0 and 75% train accuracy. Try 2000 steps or learning rate warmup. If per-sample accuracy improves to 60%+, the "routing doesn't matter" conclusion changes to "we just undertrained the router."

4. **Remove or qualify the Grassmannian interchangeability explanation.** A-matrix orthogonality does not imply B-matrix interchangeability. Either derive the connection formally or present it as a hypothesis rather than an established root cause.

5. **Add a random-adapter baseline.** If truly any adapter is interchangeable, then selecting a random adapter should give the same gamma as the softmax router. This is the cheapest possible falsification test and it is missing.

## Closing Ratification (2026-04-19, DB-completion ratify)

Researcher iter 55 marked the experiment killed (K#540 fail / K#541 pass) after the artifacts had pre-existed since 2026-03-28 with no DB-status update. All five REVISE fixes from the original review were addressed inline in PAPER.md "Phase 4: REVISE Fixes":

1. LoRA activation magnitudes measured (in-domain 28,470 vs OOD 26,406, ratio 1.08×) — disproves "adapters do nothing" hypothesis.
2. Centroid (45.8%) vs per-sample (40.2%) accuracy reported separately; both still < 50% threshold, K1 fails.
3. Random-routing baseline (gamma 0.697) vs softmax (0.625) = 11.6% gap — softmax adds genuine value despite mediocre classification accuracy.
4. Grassmannian "interchangeability" claim qualified to "semantic-cluster routing" — empirically grounded by activation+random-baseline data.
5. Training-step ablation deferred (acknowledged limitation).

Adversarial checklist: results.json verdict KILLED ↔ DB status killed ↔ PAPER "KILLED (K1 FAIL)" all consistent. KCs (#540/#541) pre-registered 2026-03-06; no relaxation. Non-tautological (K1 measures classification accuracy, K2 measures gamma — distinct quantities). LORA_SCALE=20.0 (antipattern (i)) is non-blocking for the kill since Phase 4 activation magnitudes refute scale-collapse explanation. No shutil.copy / no sum-LoRA bug / no thinking-suppression / N=24 domains × 50 val samples sufficient.

**Verdict: PROCEED-WITH-KILL** (ratify existing 2026-03-28 REVISE → addressed → kill stands on K1 fail).

Mechanistic finding worth preserving: softmax router achieves gamma_top1 = gamma_oracle (0.0% gap) via semantic-cluster routing — within-cluster misclassification is quality-benign. KCs tied to proxy metrics (classification accuracy) can produce "kill" verdict while the target metric (gamma vs oracle) is fully achieved through a different mechanism than the KC assumed. Reusable rule for future routing experiments: pair classification-accuracy KCs with target-metric oracle-gap KCs to avoid mechanism/measurement mismatch.
