# MATH: Scaling Domain Adapters to N=25

## 1. Mechanism Definition

### Grassmannian A-Matrix Skeleton at N=25

We pre-compute N=25 orthonormal frames A_i in R^{d x r} via Alternating Projection
on the Grassmannian Gr(r, d), where d=2560 (BitNet-2B hidden dim) and r=16 (LoRA rank).

**Key capacity check:** N*r = 25 * 16 = 400 << d = 2560.

Since Nr < d, perfect orthogonality is achievable: A_i^T A_j = 0 for all i != j.
The AP algorithm converges to QR-orthogonal frames in 1 iteration (the Welch bound
is vacuous when Nr <= d). This is our regime: we have abundant capacity.

Contrast with the concern in the delegation: "N_max = d^2/r^2 = 25,600" assumes
d=128, r=8 (micro-model scale). At d=2560, r=16: N_max = 2560/16 = 160 frames
with guaranteed zero pairwise coherence (since Nr <= d requires N <= d/r = 160).
We are at N=25, well within this bound.

### Per-Expert LoRA Training

Each expert i computes:
  y_i = W_base @ x + scale * (x @ A_i) @ ternary(B_i)

Where ternary(B) applies STE quantization:
  alpha = mean(|B|)
  B_q = clip(round(B / alpha), -1, 1) * alpha
  B_ste = B + stop_gradient(B_q - B)

B_i in R^{r x d_out} is the only trainable parameter per expert.
A_i in R^{d_in x r} is frozen from the Grassmannian skeleton.

### Correct Multi-Expert Composition

For K active experts (selected by routing), the composed output is:
  y = W_base @ x + (scale / K) * sum_{i in S}[(x @ A_i) @ ternary(B_i)]

where S is the set of K selected experts. This is the per-expert A_i @ B_i
formulation that was validated in the 5-domain experiment (avg -26.3% PPL vs base).

The BROKEN formulation (single A_0 + averaged B) loses the subspace separation
guarantee and was shown to produce 3.3x worse results.

### Gumbel-Sigmoid Routing

Each of N=25 routing heads is a binary classifier:
  h_i(x) = W2_i @ relu(W1_i @ x)  where W1 in R^{d x 32}, W2 in R^{32 x 1}

During inference, the top-K experts by sigmoid(h_i(x)) are selected.
With Gumbel noise during training:
  gate_i = sigma(h_i(x) + epsilon), epsilon ~ Gumbel(0, 1)

This is independent per-expert (no softmax competition), which allows
constructive multi-activation per MoLoRA findings (arXiv 2603.15965).

## 2. Why It Works

**Orthogonality guarantee:** When A_i^T A_j = 0 (guaranteed for N <= d/r):
  ||Delta_W_i^T Delta_W_j|| = ||(scale * B_i @ A_i^T)(scale * A_j @ B_j^T)||
                             = scale^2 * ||B_i @ (A_i^T A_j) @ B_j^T||
                             = 0

This means expert weight updates are provably non-interfering regardless of
B-matrix correlation. The 5-domain experiment confirmed: B-matrix cos 0.0298,
but delta-W cos 0.0017 (17x filter from Grassmannian).

**Scaling property:** The 1/K normalization ensures composed output magnitude
matches individual expert magnitude. Without it, PPL explodes to trillions
(proven in macro/composition_weight_normalization/).

## 3. What Breaks It

**Orthogonality failure:** Only when N > d/r = 160. At N=25, this is not a risk.
The Welch bound forces max |cos(A_i, A_j)| >= sqrt((N-d/r)/((d/r)(N-1))) but
since N < d/r, the argument under the square root is negative, so the bound is
vacuous (it simply does not apply). There is no lower bound on coherence in this
regime, and perfect orthogonality is achievable.

**Semantic composition gap (arXiv 2510.03262):** Weight-space orthogonality does
not guarantee data-space orthogonality. Our OSRM experiment confirmed: 100% pairs
fail OSRM (<0.1). However, composition WORKS empirically (4/5 pairs). The mechanism
is constructive transfer under 1/N scaling, not strict non-interference.

**Routing difficulty at N=25:** With 25 domains, the routing heads must discriminate
among more categories. The 5-domain experiment achieved 99.9% accuracy, but that
was trivially separable domains. At N=25 with potentially overlapping domains
(e.g., science vs. engineering, economics vs. finance), accuracy may drop.
Kill threshold: > 70% routing accuracy (S2).

**Memory:** BitNet-2B-4T unpacked to bf16 is ~4GB. Each adapter training adds
optimizer state (~40MB for rank-16 on 210 projections). Sequential training
with cleanup between adapters should peak at ~6GB active. 25 sequential trainings
= 25 * ~65s = ~27 min. Total with eval: ~45-60 min.

## 4. Complexity Analysis

**Training per adapter:**
- Forward: O(seq_len * d * r) per projection, 7 projections * 30 layers = 210 LoRA layers
- Trainable params: 210 * 16 * d_out = ~10.9M (confirmed from 5-domain exp)
- Time: ~65s on M5 Pro (confirmed)
- Total training: 25 * 65s = ~27 min

**Grassmannian AP:**
- N=25, r=16, d=2560: Gram matrix is 400x400, eigendecomposition O(400^3) = trivial
- Per projection type: 1 AP call (same dims across layers)
- 7 projection types, but d_in varies: {2560, 6912}
- Total: 2 AP calls (one per unique d_in)

**Composition evaluation (all-N uniform, as tested):**
- N expert forward passes per token (all 24 experts)
- 24x cost vs base; routed top-2 would be only 2x cost vs base

**Memory peak:**
- Model: ~4GB (bf16 unpacked BitNet-2B)
- LoRA state: ~40MB per adapter during training
- Skeleton: 25 * 210 * (d * r * 4 bytes) ~ 25 * 210 * 163KB = 858MB on disk
  But only 1 adapter's A matrices loaded at a time during training: 210 * 163KB = 34MB
- Total peak during training: ~4.1GB. Safe for 48GB.

## 5. Worked Example (d=64, r=4, N=4)

Generate 4 frames on Gr(4, 64): N*r = 16 < 64 = d.
QR orthogonalization gives A_1, A_2, A_3, A_4 with A_i in R^{64x4}, A_i^T A_j = 0.

Train B_1 on medical data: B_1 in R^{4x64}, values in {-alpha, 0, +alpha}.
Train B_2 on code data: B_2 in R^{4x64}.

Compose top-2 (medical + code):
  y = W@x + (scale/2) * [(x@A_1)@B_q1 + (x@A_2)@B_q2]

Since A_1^T A_2 = 0, the cross terms vanish: the medical adapter literally cannot
interfere with the code adapter in weight space.

## 6. Connection to Architecture

This experiment validates the full BitNet-SOLE pipeline at N=25:
- Grassmannian skeleton (Track C) provides the orthogonality guarantee
- STE ternary B-matrices (proven at 5-domain) enable 10x compression
- Gumbel-sigmoid routing (Track B) provides per-token expert selection
- Correct multi-A composition (Track B) prevents the single-A composition bug

The N=25 result directly enables the competitive benchmark: "any model is a
sparse composition of experts" needs enough domains to be credible.

## 7. Prior Art

- arXiv 2510.03262 (OSRM): Weight-space orth != semantic orth, but composition
  works via constructive transfer. We monitor both metrics.
- arXiv 2602.21222 (Task-Aware LoRA Composition): Linear merging at scale.
  Our per-expert A_i@B_i is more principled than single-A averaging.
- arXiv 2508.11985 (Naive LoRA Summation): Orthogonality enables additive
  composition. Our Grassmannian provides stronger guarantees than random init.
- arXiv 2603.03535 (Ensembling vs Merging vs Routing): Routing > merging at scale.
  Validates our Gumbel routing approach over static composition.
