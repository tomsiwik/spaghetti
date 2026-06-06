# MATH.md: M2P Depth Sweep — Transformer Depth as Generation Quality Bottleneck

**Experiment type:** Guided exploration (Type 2)
**Prior kill:** exp_m2p_bottleneck_width (Finding #355/#356) — JL distortion ≠ generation quality.
  Width d_M2P is a closed direction. This experiment targets the remaining bottleneck.

---

## A. Failure Mode Identification

**Disease:** The M2P generation quality ceiling at L=2 (M2P_LAYERS=2) is ~95–97% of SFT.
This ceiling persists across all d_M2P widths {64, 128, 256} (Finding #355). The width
direction is closed (JL impossibility proven and verified). The next structural candidate
is architecture depth.

**Precise failure mode:** An M2P with L=2 transformer layers may lack the function
approximation capacity to faithfully map base model hidden states to domain-specific
B-matrix weights. If the mapping h → B is more complex than a depth-2 transformer can
express, the M2P will regress to a blurred approximation with a bounded quality ceiling.

**Root cause diagnosis (not symptoms):** The B-matrix generation function
  f: H^(T × d_model) → R^(N_LAYERS × N_MODULES × LORA_RANK × d_out)
is a sequence-to-weight mapping. Universal approximation theory (Yun et al., 2020)
guarantees that any continuous sequence-to-sequence function on a compact domain is
approximable by a sufficiently deep transformer. If L=2 is insufficient for this
specific mapping, L=4 should reduce the approximation gap.

**Is this the root cause or a symptom?** The depth hypothesis is the next structural
candidate after the width hypothesis was killed. However, this is a guided exploration
(Type 2): we know depth MATTERS by universal approximation theory, but we do not yet
know whether L=2 already achieves the representational ceiling for THIS specific
mapping. The experiment narrows this unknown.

**Closed direction:** Do NOT apply JL lemma here. LEARNINGS.md (exp_m2p_bottleneck_width)
establishes that projection dimension is not the bottleneck. Any argument of the form
"wider → better quality" has been empirically and theoretically refuted.

---

## B. Prior Mathematical Foundations

### B.1 Transformer Universal Approximation (Yun et al., 2020)

**Theorem (Yun et al., 2020, arXiv:1912.10077, Theorem 2):**
For any continuous permutation equivariant sequence-to-sequence function
g: R^{n×d} → R^{n×d} with compact support, and any ε > 0, there exists a transformer
T with fixed width and sufficient depth L = O(n² × d / ε) such that:

    sup_{x ∈ K} ‖T(x) - g(x)‖ < ε

where K is the compact support domain.

*Reference:* C. Yun, S. Bhojanapalli, A. S. Rawat, S. Reddi, S. Kumar,
"Are Transformers Universal Approximators of Sequence-to-Sequence Functions?"
ICLR 2020, arXiv:1912.10077.

**Key conditions for applicability:**
1. The target function g must be continuous on a compact domain
2. The transformer must have sufficient depth L
3. Width can be fixed (the paper uses width proportional to d)

**Corollary:** For fixed width, as ε → 0 (exact approximation), depth L → ∞.
This implies: for any finite approximation budget ε, there exists a minimum depth L*
such that no transformer with fewer layers can achieve ε-approximation.

### B.2 Depth-Width Tradeoff for Transformers

**Theorem (Perez et al., 2021; formalized in depth-width tradeoff literature):**
Transformer depth and width can compensate for each other up to a point, but
depth provides qualitatively different capabilities:
- Shallow networks (L=1): can only compute one round of attention + MLP
- Deeper networks (L>1): can compose attention patterns iteratively

For sequence-to-weight generation specifically, each layer refines the hidden
representation. A depth-L transformer can make L sequential "passes" over the memory
tokens, iteratively sharpening the B-matrix prediction.

### B.3 The M2P Mapping Function

At micro scale, the M2P maps:

    Input: hidden states h ∈ R^{1 × T × d_model}  (T=48, d_model=256)
    After pooling: h_enc ∈ R^{N_MEMORY × d_M2P}  (N_MEMORY=32, d_M2P=64)
    Output: B_matrices ∈ R^{N_LAYERS × N_MODULES × LORA_RANK × d_out}

The output space has dimension:
    N_LAYERS=2 × N_MODULES=5 × LORA_RANK=4 × mean(d_out) ≈ 2 × 5 × 4 × 512 = 20,480

This is a complex high-dimensional output from a compressed representation.
The mapping is NOT trivially simple:
- It requires learning WHICH domain the hidden states belong to (classification)
- Given domain identity, it must RETRIEVE the correct B-matrix directions
- These two steps (classification + retrieval) naturally compose into L≥2 layers

**Prior evidence (Finding #345 — Algebraic Proof of M2P Centroid Trap):**
Multi-domain M2P training causes B-matrix centroid collapse. This was proven
algebraically: the M2P learns a weighted average of all B-matrices. Per-domain
M2P training (Finding #351) achieves 93.3% quality with L=2. Fresh training
(exp_m2p_bottleneck_width) achieves 95–97% with L=2.

The question is whether L=4 breaks through the 95–97% ceiling.

---

## C. Proof of Guarantee (Depth Exploration Theorem)

This is a **Type 2 guided exploration** — the mathematical framework (universal
approximation) is proven, but the unknown is which depth L is sufficient for THIS
specific mapping. We cannot prove a specific quality bound without knowing the
Lipschitz constant of the target function. What we CAN prove is:

**Theorem 1 (Depth Necessity for M2P).** Let f_L denote the M2P with L transformer
layers. Under the conditions of Yun et al. (2020) Theorem 2, there exists a depth
L* such that:

    For all L ≥ L*, ‖f_L(h) - B*‖_F < ε
    For all L < L*, ‖f_L(h) - B*‖_F ≥ ε'  for some ε' > 0

where B* is the optimal B-matrix and ε, ε' > 0 are approximation budgets.

*Proof sketch.*
The mapping h → B* is a continuous function on a compact domain (hidden states live
in a bounded hypercube K ⊂ R^{T×d_model}; B-matrices are bounded by SFT training).
By Yun et al. (2020) Theorem 2, any such function is approximable by a transformer
with sufficient depth. The existence of L* follows from the compactness of K and
the continuity of f.

*What the proof does NOT guarantee:*
- The specific value of L* for the M2P mapping (unknown, measured by experiment)
- That L=4 is sufficient (might need L >> 4 if the mapping is highly complex)
- That depth strictly dominates width at this scale (interaction effects possible)

QED (existence only — depth L* exists but is experimentally unknown).

**Framing as guided exploration:** This experiment narrows the unknown L*. The
proven framework says L* exists and depth L < L* has a bounded quality ceiling.
The experiment probes whether L=2 < L* (depth is still the bottleneck) or
L=2 ≥ L* (something else limits quality at 95–97%).

---

## D. Quantitative Predictions (Derived from Theorem 1)

**Note:** These predictions are EXPLORATORY (Type 2), not VERIFICATION (Type 1).
The theorem guarantees only that L* exists, not its value. Predictions are based
on the prior observation that L=2 saturates at ~95–97%.

| L | Predicted behavior | Kill criterion | Derived from |
|---|-------------------|----------------|--------------|
| 1 | quality < quality(L=2) — insufficient depth | Reference | Universal approx (depth L=1 < L*) |
| 2 | ~95–97% of SFT (confirmed by Finding #355) | Reference | Finding #355 direct measurement |
| 4 | If K873/K874 PASS: depth L=2 < L*; If FAIL: L=2 ≥ L* or other bottleneck | K873, K874 | Theorem 1 existence |

**Three possible outcomes:**

Outcome A (K873 PASS + K874 PASS): L=4 achieves ≥97%, strictly better than L=2.
  Interpretation: L* is between 2 and 4. Depth IS the bottleneck.
  Action: Try L=8 to find when quality saturates.

Outcome B (K873 PASS + K874 FAIL): L=4 beats L=2 by >2pp but stays below 97%.
  Interpretation: Depth helps but L=4 < L*. More depth needed.
  Action: Try L=8 or L=16.

Outcome C (K873 FAIL + K874 FAIL): L=4 shows no improvement over L=2 (<2pp).
  Interpretation: L=2 ≥ L* — depth is NOT the bottleneck. Something else caps quality.
  Action: Investigate training steps, curriculum, or target B-matrix complexity.

**Kill criterion K875 (plateau, FAIL case):** Triggered by Outcome C.
  Measured by: |quality(L=4) - quality(L=2)| < 2pp (absolute, median across valid domains).

**Parity domain exclusion (from LEARNINGS.md):**
All quality statistics exclude domains where (base_loss - sft_loss) < 0.05.
This guard catches the parity domain artifact (near-zero quality denominator).

---

## E. Assumptions & Breaking Conditions

**Assumption 1:** The M2P target function (hidden states → B-matrices) is continuous
on a compact domain. This is standard: neural network activations are bounded and
continuous; B-matrices are bounded by SFT training loss.
BREAKING: If B-matrices have discontinuous dependence on hidden states (e.g., sharp
domain boundaries), the Yun 2020 theorem requires adjustment. Unlikely at micro scale.

**Assumption 2:** L=2 is a meaningful starting point — not already at the quality
ceiling imposed by factors other than depth (training budget, optimizer, data).
EVIDENCE: Finding #355 shows quality flat at 95–97% across all widths at L=2.
This is consistent with L=2 being a depth ceiling, not a width ceiling.
BREAKING: If L=1 achieves quality close to L=2 (within 2pp), depth is not the bottleneck
even at L=2. L=1 is included in the sweep to test this.

**Assumption 3:** 500 training steps is sufficient to reveal the depth signal.
Deeper models may converge slower. If L=4 trains to a lower quality than L=2 due
to optimization difficulty (not capacity), the experiment would misidentify depth
as a bottleneck in the wrong direction.
MITIGATION: Use identical optimizer settings (Adam, lr=1e-3) across all L.
BREAKING: If learning curves show L=4 still descending at step 500, training budget
(not architecture) is the bottleneck. Report final loss for all L to check.

**Assumption 4:** The 95–97% ceiling observed at L=2 is reproducible with fresh
training. EVIDENCE: Finding #355 measured this consistently across d_M2P in {64,128,256}.
FRESH TRAINING REQUIRED: per LEARNINGS.md — must retrain base model, SFT adapters,
and M2P from scratch to avoid the 2.9pp reuse artifact.

**Kill criterion derivation:**
- K873 (depth strictly helps): If L=4 quality > L=2 quality + 2pp, depth is the bottleneck.
  The 2pp threshold is derived from the known within-run training variance (~1–2pp across
  domains in Finding #355). A 2pp gap is at the boundary of detectability.
- K874 (quality ceiling): If L=4 quality ≥ 97%, depth is sufficient to reach the ceiling.
  The 97% threshold is the same as the killed K870 from exp_m2p_bottleneck_width.
- K875 (plateau FAIL): If |quality(L=4) - quality(L=2)| < 2pp, depth is NOT the bottleneck.
  This is the KILL case — experiment is dead if L=4 cannot improve over L=2 by 2pp.

**Interaction with training variance:**
At micro scale, single-run quality estimates have ±1–2pp variance (observed in Finding #355).
To reliably detect a 2pp signal, the experiment should show consistent improvement
ACROSS MULTIPLE DOMAINS (not just one). "Depth strictly helps" requires median quality
across valid domains (excluding parity) to improve by ≥2pp.

---

## F. Worked Example (d=64, L=1 vs L=2 vs L=4)

**Architecture parameter counts at d_M2P=64, N_MEMORY=32:**

For each M2P layer (M2PBlock):
  - M2PAttention: 4 × (64×64) = 16,384 params
  - M2PMLP: 64×256 + 256×64 = 32,768 params
  - RMSNorm × 2: 2 × 64 = 128 params
  - Total per block: ~49,280 params

Shared components:
  - input_proj: 256×64 = 16,384 params
  - memory_tokens: 32×64 = 2,048 params
  - pos_embed: 32×64 = 2,048 params
  - norm_f: 64 params
  - out_heads: 64 × (2×4×256 + 2×4×256 + 2×4×256 + 2×4×256 + 2×4×1024)
             = 64 × (2048 + 2048 + 2048 + 2048 + 8192)
             = 64 × 16,384 = 1,048,576 params

Total M2P parameters:
  L=1: 16,384 + 2,048 + 2,048 + 64 + 49,280 + 1,048,576 ≈ 1,118,400
  L=2: L=1 + 49,280 ≈ 1,167,680  (matches m2p_bottleneck_width measurement)
  L=4: L=2 + 2 × 49,280 ≈ 1,266,240

**Compute cost:**
Each M2PBlock forward pass: O(N_MEMORY² × d_M2P) = O(32² × 64) = O(65,536) ops.
L=4 adds 2× more M2PBlocks vs L=2: ~10% more compute for M2P (negligible at this scale).

**Expected runtime:** Given exp_m2p_bottleneck_width ran in 68.9s for 3 d_M2P values
(500 steps per domain per width, 5 domains), this experiment sweeps 3 depths (L=1,2,4)
at 5 domains each. Estimated runtime: ~70–100s.

---

## G. Complexity & Architecture Connection

**Parameter count scaling with L:**
    M2P total params ≈ 1,118,000 + L × 49,280  (for d_M2P=64)

At L=1: ~1.12M, L=2: ~1.17M, L=4: ~1.27M. All three are tiny — depth increase is <15%.
This is intentional: we isolate the depth signal, not confound with parameter count.

**Production context:** Production MoE hypernetworks (e.g., HyperDreamer, HyperNetworks
for NLP) typically use L=2–6 layers. The M2P at L=2 is at the shallow end of this range.
Empirical results from production systems suggest L=2–4 is sufficient for most hypernetwork
tasks, but B-matrix generation is more complex than typical hypernetwork outputs.

**Interaction with d_M2P:**
- Width was proven closed (Finding #355). Depth sweep is orthogonal.
- d_M2P=64 is fixed throughout. This is the "proven sufficient" value from Finding #355.
- The depth signal should be cleanly measurable at d_M2P=64.

**Architecture reference:** https://sebastianraschka.com/llm-architecture-gallery/
Standard transformer depth recommendations: GPT-2 uses L=12–48; small models like
TinyGPT use L=2–6. Our toy base model uses L=2 (N_LAYERS=2). The M2P depth should
ideally match or exceed the base model depth for faithful imitation.

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode impossible?**
By Yun et al. (2020) Theorem 2, a transformer with sufficient depth L ≥ L* is a
universal approximator of the M2P mapping. If L < L*, the approximation gap is
bounded AWAY from zero, making quality < ceiling unavoidable.

**2. Which existing theorem(s) does the proof build on?**
Yun et al. (2020), "Are Transformers Universal Approximators of Sequence-to-Sequence
Functions?", ICLR 2020, arXiv:1912.10077, Theorem 2. The theorem states: any
continuous permutation equivariant function is approximable by a fixed-width
sufficiently deep transformer.

**3. What specific numbers does the proof predict?**
The proof predicts only the EXISTENCE of L*. It does not predict:
- The value of L* for the M2P mapping
- Whether L=4 is sufficient
This is why the experiment is Type 2 (guided exploration) not Type 1 (verification).
Known reference point: quality at L=2 ≈ 95–97% (Finding #355 baseline).

**4. What would FALSIFY the proof (not just the experiment)?**
The proof (Yun 2020 existence) cannot be falsified by this experiment. What CAN be
falsified: "Depth is the bottleneck for M2P quality at micro scale." This is falsified
if quality(L=4) ≤ quality(L=2) + 2pp (K875 PASS = experiment FAIL).

**5. How many hyperparameters does this approach add?**
Count: 0. The sweep values {1, 2, 4} are chosen to bracket the known L=2 baseline
and extend one step above. No new loss terms, regularizers, or architectural components.

**6. Hack check: Am I adding fix #N to an existing stack?**
No. This is a pure ablation of ONE architectural parameter (M2P_LAYERS). The only
change from exp_m2p_bottleneck_width is M2P_LAYERS ∈ {1,2,4} instead of d_M2P ∈ {64,128,256}.
No new mechanisms, losses, or tricks. One variable, one question.
