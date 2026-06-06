# Mixed-Domain Per-Token Routing: Mathematical Foundations

## Type: Guided Exploration

**Proven framework:** Per-adapter routing heads achieve 100% domain classification on
single-domain sequences (Finding #58). Per-sequence softmax routing matches oracle
at N=24 (VISION.md). Composition landscape is convex (Finding #41).

**Unknown parameter:** On mixed-domain sequences where different segments belong to
different domains, what fraction of the oracle gap can segment-level routing capture,
and does it exceed the 5% threshold over per-sequence routing?

## Step A: Diagnose the Disease

The prior experiment (exp_mixed_domain_sequences) was KILLED with K1=+0.28% (threshold 5%).
Three failure modes compounded:

1. **Cross-attention contamination** (architectural): Running a full forward pass through
   a mixed sequence with adapter A means tokens from domain B still influence domain A
   tokens via self-attention. This is inherent to any transformer-based per-token routing
   with full-sequence forward passes.

2. **Router collapse** (representational): The single multi-class MLP (2560->64->5)
   collapsed to a 2-class detector (code/math vs prose). Medical, legal, creative are
   indistinguishable in BitNet-2B-4T's hidden space.

3. **Post-hoc evaluation methodology** (evaluative): Per-token routing required running
   the full sequence once per unique expert set, scoring only the assigned tokens.
   This systematically penalizes per-token routing because "wrong-adapter" tokens
   contaminate "right-adapter" tokens via attention.

**The disease is not the router -- it's the evaluation/composition methodology.**
Even with a perfect router (oracle = 97% on python+math), per-token routing was
-6.4% WORSE than per-sequence. The architecture cannot support token-level adapter
switching within a single forward pass.

## Step B: Reframe the Question

**Wrong question:** "How do we make per-token routing work on mixed-domain sequences?"

**Right question:** "What is the maximum PPL improvement achievable by segment-level
adapter selection on mixed sequences, given that within-segment composition must be
consistent to avoid cross-attention contamination?"

The answer shifts from per-TOKEN routing to per-SEGMENT routing: detect domain
boundaries, then evaluate each segment independently with its optimal adapter.

## Step C: Derive From Existing Math

### Theorem (Segment-Level PPL Decomposition)

For an autoregressive LM with sequence x = [x_1, ..., x_T] and boundary at position b,
the sequence-level negative log-likelihood decomposes as:

$$\text{NLL}(x) = \sum_{t=1}^{T} -\log p(x_t | x_{<t})$$

For segment-level routing with adapter alpha_A on positions [1..b] and alpha_B on
positions [b+1..T], the NLL with segment-isolated evaluation is:

$$\text{NLL}_{seg}(x) = \sum_{t=1}^{b} -\log p_{A}(x_t | x_{<t}) + \sum_{t=b+1}^{T} -\log p_{B}(x_t | x_{<t})$$

where p_A and p_B denote the model with adapter A and B respectively.

**Key distinction from prior experiment:** Each segment is evaluated as an independent
subsequence (or with correct-adapter context for the preceding segment). This eliminates
cross-attention contamination by construction.

### Prior Result: Convex Composition Landscape (Finding #41)

From Finding #41: the composition weight landscape is smooth and convex. Mixing beats
selection by 3.5-5.2%. Uniform 1/N is only 0.7% from optimal. This means per-sequence
routing with top-2 already captures most of the composition value for SINGLE-domain
sequences.

### Prior Result: Oracle Gap (exp_mixed_domain_sequences)

The oracle achieved PPL 7.03 vs per-sequence 8.61 = 18.4% improvement. This proves
the value of correct per-segment routing exists. But the prior experiment's per-token
routing captured only 0.28% of this gap.

### Bound on Segment-Level Advantage

Let PPL_seq(A,B) be the per-sequence PPL where the router selects based on the
mean-pooled hidden state. Let PPL_seg(A,B) be the segment-level PPL with oracle
segmentation.

The per-sequence router sees h_mean = (1/T) * sum(h_t). For balanced segments (b=T/2),
h_mean is the midpoint between domain A and domain B centroids. The router's selection
depends on which adapter's scoring region this midpoint falls in.

**Case 1: Well-separated domains (code vs medical).** The midpoint may fall in neither
domain's high-confidence region, causing the router to select a suboptimal adapter or
hedge with top-2. Segment-level routing selects the correct adapter for each half.
Expected advantage: proportional to domain separability.

**Case 2: Near domains (medical vs legal).** The midpoint is already close to both
domains. Per-sequence routing nearly matches segment-level because the adapter
selected for the whole sequence is roughly appropriate for both halves. Expected
advantage: near zero.

### Predicted Improvement Function

Let sep(A,B) = ||mu_A - mu_B|| / (sigma_A + sigma_B) be the domain separability.

For segment-level vs per-sequence improvement:

Delta_PPL(A,B) ~ c * sep(A,B)^2 / (1 + sep(A,B)^2)

where c is a constant depending on adapter strength. This is a saturating function:
- At sep=0 (identical domains): Delta=0 (no advantage)
- At sep->infinity (perfectly separated): Delta->c (full oracle advantage)
- At sep~1 (moderate separation): Delta ~ c/2 (half the oracle advantage)

From prior data:
- python+math (high separability): oracle 15.0% better than per-seq
- medical+legal (low separability): oracle 25.6% better than per-seq (but per-seq
  is WORSE than uniform here, suggesting the router actively hurts)

## Step D: Proof of Guarantee

**Theorem 1 (Segment Isolation Eliminates Cross-Attention Contamination).**

Let x = [A || B] be a mixed sequence. If we evaluate segment A as an independent
subsequence x_A = [A] with adapter alpha_A, then no token in A attends to any
token in B. Therefore:

$$p_{A}(x_t | x_{<t}) \text{ for } t \leq b$$

depends ONLY on the domain-A tokens and adapter alpha_A. The cross-attention
contamination identified in the prior experiment (where the python+math pair had
97% routing accuracy but -6.4% PPL) is structurally impossible under segment isolation.

*Proof.* Self-attention in a causal transformer computes, for position t:

Attn(Q_t, K_{<=t}, V_{<=t}) = softmax(Q_t K_{<=t}^T / sqrt(d_k)) V_{<=t}

If we restrict the context to positions [1..b], then K_{<=t} and V_{<=t} contain
only domain-A tokens for all t <= b. No domain-B information enters the computation.
QED.

**Corollary.** Segment-isolated evaluation is an UPPER BOUND on what any per-token
routing can achieve with full-sequence forward passes, because segment isolation
removes the cross-attention contamination that degrades per-token routing.

**Theorem 2 (Segment-Level Routing Recovers Oracle Value When Router Is Correct).**

If the segment boundary is known and the per-adapter routing heads correctly classify
each segment's domain, then segment-level routing achieves oracle-equivalent PPL.

*Proof.* Oracle routing assigns adapter alpha_A to positions [1..b] and alpha_B to
positions [b+1..T], evaluated independently. Segment-level routing with correct
classification does exactly the same thing. QED.

Note: This is trivially true by construction. The interesting question is whether
the router can correctly classify the segments, which is the guided-exploration
unknown.

**Theorem 3 (Per-Adapter Binary Heads Generalize to Segment Classification).**

Per-adapter binary routing heads achieve 100% accuracy on single-domain sequences
(Finding #58, proven). A segment from domain A extracted as an independent
subsequence IS a single-domain sequence. Therefore, per-adapter binary heads
applied to isolated segments maintain 100% accuracy.

*Proof.* The per-adapter heads were trained on single-domain hidden states and
classify domain membership via binary sigmoid gates. An isolated segment
[A_1, ..., A_b] produces hidden states identical to a length-b single-domain
sequence from domain A (the hidden states depend only on the input tokens and
model parameters, not on what follows). By the proven 100% accuracy on
single-domain inputs, the heads correctly classify the segment. QED.

**Caveat:** This assumes segment isolation. With full-sequence context (preceding
segment from a different domain), hidden states at the boundary may be contaminated.
The experiment measures this.

## Step D: Predictions (Quantitative)

From the oracle results of the prior experiment and the theoretical framework:

| Prediction | Source | Quantitative |
|-----------|--------|-------------|
| P1: Segment-level routing matches oracle PPL | Theorem 2 | PPL_seg / PPL_oracle in [0.95, 1.05] |
| P2: Segment-level > per-sequence by >= 5% | Oracle gap is 18.4%, segment captures ~50% | >= 5% improvement |
| P3: High-sep pairs (code+X) show largest gains | Separability function | code+medical > medical+legal |
| P4: Segment classification accuracy >= 80% | Theorem 3 + per-adapter heads | >= 80% per segment |
| P5: Cross-attention contamination effect | Theorem 1 | Isolated eval > full-sequence eval |

**Kill criteria mapping:**
- K772: P2 predicts >= 5%. If < 5% -> FAIL (segment-level routing doesn't help enough)
- K773: P4 predicts >= 80% for segment classification. If < 40% -> FAIL
- K774: We use the same domain data as prior experiments (proven available). PASS by construction.

## Step E: Assumptions & Breaking Conditions

1. **Segment boundaries are known or detectable.** We test with oracle boundaries first
   (they're known because we construct the mixed sequences). If oracle-boundary segment
   routing doesn't help, boundary detection is irrelevant. If it does help, boundary
   detection becomes the next research question.

2. **Segments are long enough to form coherent context.** Short segments (< 32 tokens)
   may not provide enough context for the adapter to help. We use 128-token segments
   (half of 256 max sequence length).

3. **Per-adapter binary heads transfer to segment classification.** The heads were
   trained on full single-domain sequences. Segments are shorter subsequences.
   If the heads fail on short segments, Theorem 3 breaks.

4. **Domain adapters provide meaningful specialization.** The adapters must actually
   reduce PPL on their target domain. This is proven (Finding #58: top-2 beats
   uniform by 13.9%).

## Step F: Worked Example

d=2560, N=5, T=256, boundary at b=128.

Sequence: [128 medical tokens | 128 code tokens]

**Per-sequence routing (baseline):**
- Mean-pool hidden states across all 256 positions
- Router selects top-2 (e.g., medical + creative if prose dominates)
- Apply composed adapter to ENTIRE sequence
- Both halves get same suboptimal adapter mix
- Expected PPL: ~8.6 (from prior experiment average)

**Segment-level routing (this experiment):**
- Split sequence at position 128
- Segment 1: evaluate [medical tokens] independently with medical adapter
  - Expected PPL on medical segment: ~5.3 (from prior oracle data)
- Segment 2: evaluate [code tokens] independently with code adapter
  - Expected PPL on code segment: ~3.5 (from prior oracle data)
- Combined: weighted average by token count = (5.3 * 128 + 3.5 * 128) / 256 = 4.4
- vs per-sequence: 8.6 -> 4.4 = 48.8% improvement

This is an extreme example (medical+code has high separability). The average across
all pairs should be more modest.

## Step G: Complexity & Architecture Connection

- Segment detection: O(T * N) for N binary head evaluations across T positions
  (trivial: 0.58% overhead proven)
- Segment-level evaluation: 2 forward passes per mixed sequence (one per segment)
  vs 1 for per-sequence routing. 2x compute cost.
- Memory: same as single forward pass (segments evaluated sequentially)
- Production integration: detect domain shift via sliding window on binary head
  scores, split at detected boundary, route each segment independently.

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Segment isolation eliminates cross-attention contamination by construction --
   tokens in segment A cannot attend to tokens in segment B when evaluated
   independently.

2. **Which existing theorem(s) does the proof build on?**
   Causal attention mask properties (standard transformer definition), prior
   Finding #58 (per-adapter heads 100% accuracy), Finding #41 (convex landscape).

3. **What specific numbers does the proof predict?**
   P1: PPL_seg/PPL_oracle in [0.95, 1.05]. P2: >= 5% improvement over per-sequence.
   P4: >= 80% segment classification accuracy.

4. **What would FALSIFY the proof?**
   If segment-isolated evaluation with correct adapter assignment (oracle) does NOT
   improve over per-sequence routing, then the oracle gap from the prior experiment
   was an evaluation artifact, not real value. This would falsify the premise.

5. **How many hyperparameters does this approach add?**
   Count: 1 (segment boundary detection threshold). This can be derived from the
   binary head confidence scores (Otsu thresholding, proven in entropy gating).

6. **Hack check:** No stacking of fixes. One mechanism (segment isolation) addresses
   one disease (cross-attention contamination). The router architecture is reused
   from proven components (per-adapter binary heads).
