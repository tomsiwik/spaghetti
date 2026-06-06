# C0.1: Port P0 Grassmannian + TF-IDF Composition to Gemma 4 E4B

## Motivation

P0 proved composable domain experts work on Qwen3-4B: Grassmannian A-matrices
guarantee zero parameter-space interference (Finding #341, cos=1.7e-16), TF-IDF
exclusive routing achieves 100% accuracy at N=5 (Finding #404), and quality_ratio
= 1.3125 is invariant from N=2 to N=25 (Findings #404-406).

P1 moved to Gemma 4 E4B and trained excellent single-domain adapters (+82pp math,
+46pp code, +22pp medical — Finding #421). But the composition infrastructure was
silently dropped: P1 uses plain `mlx_lm.lora` with random A-init, no Gram-Schmidt,
no TF-IDF routing, no composition mechanism.

Finding #425 tested simultaneous activation of all 5 adapters WITHOUT routing or
Grassmannian isolation, and declared "routing is structurally required." This is
an implementation error, not a finding — P0 never activated all adapters simultaneously.

This experiment ports the proven P0 composition pipeline to Gemma 4.

## Type: Verification

The math is already proven (Findings #3, #126, #341, #404-406). This experiment
verifies that the same guarantees hold on Gemma 4 E4B with rank-6 LoRA adapters.

## Theorem 1: Grassmannian Orthogonality via Sequential Gram-Schmidt

**Statement.** Given N LoRA adapters with A-matrices A_1, ..., A_N in R^{d x r},
sequential Gram-Schmidt orthogonalization produces A'_1, ..., A'_N such that:

  (A'_i)^T A'_j = 0   for all i != j   (to machine precision in float64)

**Proof.** By induction on k.
- Base: A'_1 = QR(A_1), columns orthonormal by construction.
- Step: Given orthogonal {A'_1, ..., A'_{k-1}}, construct A'_k:
  1. Initialize Q_k = random(d, r)
  2. For each prior A'_i, project out: Q_k <- Q_k - A'_i (A'_i^T Q_k)
  3. Re-orthonormalize: A'_k = QR(Q_k)[:, :r]

  After step 2, Q_k lies in the orthogonal complement of span(A'_1, ..., A'_{k-1}).
  QR in step 3 produces orthonormal columns within this complement.
  Therefore (A'_k)^T A'_i = 0 for all i < k.  QED.

**Capacity bound.** At d=2560, r=6: N_max = floor(d/r) = 426 orthogonal adapters.
We need N=5, using 5*6/2560 = 1.17% of capacity. Ample margin.

**Signal retention.** The projection in step 2 removes the component of A_k that
overlaps with prior adapters. Signal retention = ||A'_k||_F / ||A_k||_F.
At N=5 with random initialization, expected overlap ~ r*N/d = 6*5/2560 = 0.012 (1.2%).
Signal retention should be > 98%.

**Note:** We orthogonalize the A-matrices extracted from already-trained adapters.
The B-matrices are NOT modified. Parameter-space orthogonality of delta_W = B*A
holds for ANY B when A-matrices are orthogonal (proven in VISION.md, Finding #341).

## Theorem 2: TF-IDF Routing is Model-Independent

**Statement.** TF-IDF + cosine-similarity routing operates on raw input text only.
Its accuracy is invariant to:
- The base model (Qwen3 vs Gemma 4)
- The adapter state (loaded or not)
- The composition method (exclusive or simultaneous)

**Proof.** The routing function f: text -> domain_id is:
  1. x = TfidfVectorizer.transform(text)   (bag-of-words, no model)
  2. sims = cosine_similarity(x, centroids) (centroids from training text)
  3. domain = argmax(sims)

No step involves the model. Therefore f is model-independent.  QED.

**Prediction.** At N=5 with domains {math, code, medical, legal, finance}:
- Math: "Solve the following math problem" — highly distinctive vocabulary
- Code: "Write a function", variable names, syntax keywords
- Medical: clinical terms, disease names, drug names
- Legal: "court", "statute", "jurisdiction", legal terminology
- Finance: "portfolio", "interest rate", "market", financial terms

These domains have near-disjoint vocabularies. Predicted routing accuracy >= 95%.

## Theorem 3: Exclusive Routing Eliminates Activation-Space Interference

**Statement.** Under exclusive routing (exactly 1 adapter active per query):

  h_out = W_base * x + B_j * A'_j * x

where j = route(text). The activation-space interference is exactly zero because
no other adapter contributes to the output.

**Proof.** Trivial — only one term in the sum.  QED.

**Contrast with Finding #425:** T3.1 tested simultaneous activation:
  h_out = W_base * x + SUM_{i=1}^{5} B_i * A_i * x
This has N-1=4 interference terms and collapsed math from 82% to 8%.
Exclusive routing eliminates all interference terms by construction.

## Kill Criteria

| K | Criterion | Prediction | Grounding |
|---|-----------|------------|-----------|
| KC01 | TF-IDF routing accuracy >= 95% | >= 97% (disjoint vocabularies) | Finding #354, Theorem 2 |
| KC02 | max\|A'_i^T A'_j\|_F < 1e-4 (all 10 pairs, 42 layers) | < 1e-10 (float64) | Theorem 1, Finding #417 |
| KC03 | Math quality_ratio >= 0.90 under routed composition | >= 0.95 (exclusive routing = zero interference) | Theorem 3, Finding #404 |
| KC04 | No domain below 70% of solo accuracy | All >= 90% of solo | Theorem 3 |

## Implementation Phases

**Phase 1:** Extract A/B matrices from 5 safetensors adapters. Apply Gram-Schmidt
to A-matrices in float64. Verify KC02.

**Phase 2:** Build TF-IDF 5-class router from training data (100 examples/domain).
Evaluate on 500 held-out. Verify KC01.

**Phase 3:** Manual LoRA injection: load Gemma 4, for each test query route to
domain j, apply delta_W = scale * B_j @ A'_j to q_proj, generate response.
Evaluate GSM8K accuracy. Verify KC03, KC04.
