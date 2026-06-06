# MATH.md — SHINE M2P Port: Guided Exploration (Revision 3)

## Experiment Type

**Guided Exploration (Type 2) — no formal theorem claimed.**

The proven mathematical framework is SHINE's Memory-to-Parameter (M2P) Transformer
(arXiv:2602.06358). The SHINE paper proves that an M2P transformer taking per-layer
hidden-state memory as input can generate input-dependent LoRA adapter weights.

This experiment is a **porting exercise**: we implement the M2P architecture in MLX
and probe whether the port produces structurally meaningful outputs.

**The unknown being probed:** Do M2P outputs lie in a distinct statistical regime
compared to random projections, or does the architecture simply act as a random
projection of its input?

No new formal theorem is claimed. Predictions are derived from **random matrix theory**
applied to the initialization regime of a transformer with our specific dimensions.

---

## A. Disease Diagnosis

The degenerate behavior we risk: **the M2P transformer produces outputs that are
statistically indistinguishable from random noise of the same shape and norm**.

If this happens, the port compiles but does not provide any structure beyond what a
random linear projection would give. For the SHINE use case, this means the generated
adapter weights have no relation to the input memory state — the architecture is
functioning as a noise generator, not a structured adapter generator.

Is this a real risk? Yes. Phase 7 in the previous run (revision 2) showed M2P outputs
were NOT statistically distinguishable from random noise at n=10 pairs. The revised
experiment increases to n=30 pairs to have adequate power for a formal statistical test.

A secondary risk: **the positional embeddings are zero-initialized**, making the
architecture blind to layer/token position at the start of any test phase that runs
before training. Revision 3 fixes this with Xavier normal initialization.

---

## B. Reframe: The Right Question

Wrong: "Does cos(M2P(m1), M2P(m2)) < 1.0?"
(Any non-constant function satisfies this. Not a meaningful test.)

Wrong: "Does cos(M2P(m1), M2P(m2)) < 0.99?"
(Any non-degenerate function satisfies this. Provides no discrimination power.)

Right: "Are M2P outputs statistically distinguishable from random projections of
the same dimensionality at p < 0.05?"

This is the PRIMARY question because it directly tests whether the architecture
imposes structure beyond what a random initialization would produce. If M2P outputs
are indistinguishable from random, the port has no demonstrated value as a structured
adapter generator.

---

## C. Prior Mathematical Foundations

**C.1 Random Matrix Theory — Expected Cosine Similarity at Initialization**

For two independent standard normal vectors u, v in R^n:

  E[cos(u, v)] = 0   (by symmetry: E[<u, v>] = 0)
  Var[cos(u, v)] = 1/n   (by CLT argument on inner product of unit normals)
  std[cos(u, v)] ≈ 1/√n

For our M2P architecture with L=4, M=8, H=64:
  Output shape: (L, M, H) = (4, 8, 64)
  Flattened dimension: n = L * M * H = 4 * 8 * 64 = 2048

  **Random matrix theory prediction at initialization:**
  E[cos] ≈ 0,   std ≈ 1/√2048 ≈ 0.022

A randomly initialized transformer acts approximately as a random projection of
its input at initialization (each layer applies a learned-but-random linear map
plus nonlinearity). The output cosine similarity for two independent inputs should
be near 0 with standard deviation ≈ 0.022 per pair.

**Reference:** Johnson & Lindenstrauss (1984); Vershynin, "High-Dimensional Probability"
(2018), Chapter 3; Karol Gregor & Yann LeCun, "Learning Fast Approximations of Sparse
Coding" (ICML 2010) for random initialization regimes.

**C.2 Two-Sample t-Test for Distinguishability**

Given two samples X = {x_1, ..., x_n} and Y = {y_1, ..., y_n}, the two-sample t-test
checks H0: E[X] = E[Y] with test statistic:

  t = (x̄ - ȳ) / sqrt(s_X²/n + s_Y²/n)

where s_X², s_Y² are sample variances. Under H0, t ~ t(2n-2) approximately.

For p < 0.05 at n=30 pairs each, we need |t| > 2.00 (two-tailed, df=58).

**Power calculation:** To detect a mean shift of δ with std σ at n=30, power ≈ 0.80
when δ > 2σ/√30. For the random baseline (σ ≈ 0.022), this means detectable shift
≈ 0.008. For M2P (σ ≈ 0.17 from previous run), we need shift > 0.06 to detect.

**C.3 Positional Embeddings and Xavier Initialization**

SHINE §3.4 Equation 5 specifies:
  memory_states = memory_states + P_layer + P_token

where:
  P_layer ∈ R^{L×1×H}: learned layer-position embedding, broadcast across M
  P_token ∈ R^{1×M×H}: learned token-position embedding, broadcast across L

Without positional embeddings (zero init = no-op at initialization), the M2P
transformer has permutation-equivariant attention in the layer dimension (Vaswani
et al. 2017): it cannot distinguish layer i from layer j.

**Xavier normal initialization** for positional embeddings:
  scale = sqrt(2 / (1 + H))
  P ~ N(0, scale²)

This provides non-trivial positional signal from the start, consistent with how
learned positional embeddings are initialized in production (e.g., BERT, GPT-2).

**C.4 Input Sensitivity — Phase 6**

For a transformer at initialization, input sensitivity is measured as:

  cos(M2P(m1), M2P(m2)) for m1, m2 ~ N(0, I_{LMH})

At initialization, the transformer applies random linear maps + nonlinearities.
By random matrix theory (C.1), the expected output cosine similarity ≈ 0 with
std ≈ 0.022. After training on random targets (Phase 5), the weights adapt to
the training data — measuring input sensitivity both before and after training
reveals whether training changes the input-conditional behavior.

---

## D. Quantitative Predictions

| # | Prediction | Source | Primary/Secondary |
|---|-----------|--------|-------------------|
| P1 | At initialization: mean M2P cosine similarity in [-0.1, 0.1] | Random matrix theory (C.1): E[cos] = 0 | Secondary (sanity check) — **NOT VALID for M2P; see note below** |
| P2 | At initialization: std of M2P cosines ≈ 0.022 for n=2048 | Random matrix theory: std ≈ 1/√n | Secondary (characterization) |
| **P3** | **After training on random targets: M2P cosines are statistically distinguishable from random baseline at p < 0.05 AND \|diff in means\| > 0.05** | **PRIMARY KILL CRITERION (K827)** | **PRIMARY** |
| P4 | Training changes input sensitivity (Phase 6b differs from Phase 6a) | Training updates weights; positional embeddings become task-adapted | Secondary (observation) |

**⚠ P1 Validity Note — Shared Positional Embeddings Violate RMT Independence Assumption:**

P1 derives from RMT (C.1), which requires the two output vectors being compared to arise from *independent* random processes. This assumption is violated in M2P:

SHINE §3.4 Eq. 5 adds shared P_layer ∈ R^{L×1×H} and P_token ∈ R^{1×M×H} to every memory input:
  memory_input = m + P_layer + P_token

Both outputs M2P(m1) and M2P(m2) receive the same additive positional offset before the transformer. The inner product ⟨M2P(m1), M2P(m2)⟩ accumulates a systematic positive contribution from the shared positional component. The expected cosine is therefore:
  E[cos(M2P(m1), M2P(m2))] > 0   (not = 0 as RMT predicts for independent vectors)

**Experimental confirmation:** Phase 6a measured mean=0.0818, which is ~19.5σ above zero (SE = std/√n = 0.0230/√30 = 0.0042; deviation = 0.0818/0.0042 ≈ 19.5). The std prediction (P2: 0.022) holds because std is unaffected by the shared mean shift — it still measures input-conditional variance. P1 should be struck from future M2P experiments that use shared positional embeddings.

**P3 is the primary scientific question.** If M2P outputs are NOT statistically
distinguishable from random noise after training, the architecture has no demonstrated
advantage over random projection as a structured adapter generator. This would be an
honest negative result — the port works mechanically but does not demonstrate structured
behavior on toy tasks.

If M2P outputs ARE statistically distinguishable from random noise, the architecture
imposes some detectable structure. The nature of that structure (shared direction,
structured variance, etc.) becomes the next question. Note that the K827 result
reflects *combined* structure from positional embeddings and training — future work
should ablate positional embeddings to separate these contributions.

---

## E. Assumptions and Breaking Conditions

**Assumption A1:** Xavier-initialized positional embeddings provide non-trivial
positional signal from the start.
- Breaking: if the attention mechanism is too shallow to use positional signal (L=4
  is small). Mitigation: the experiment measures pre-training sensitivity, so we
  observe directly whether positional signal matters.

**Assumption A2:** n=30 pairs provides adequate statistical power.
- Breaking: if M2P std is much higher than 0.17 (prior run), the test may be
  underpowered. Mitigation: we also report effect size (|diff in means|).
- The effect size criterion (>0.05) provides a practical significance guard even
  when p < 0.05 by chance with high variance.

**Assumption A3:** Training on random targets for 100 steps is sufficient to show
whether the architecture imposes structure.
- Breaking: 100 steps may be too few for structure to emerge. Mitigation: we
  measure both pre- and post-training sensitivity, so if both are near 0, the
  conclusion is that neither initialization nor short training creates structure.

**Connection to kill criterion K827:**
- If p >= 0.05 OR |diff| <= 0.05: K827 FAIL (M2P not distinguishable from random)
- If p < 0.05 AND |diff| > 0.05: K827 PASS (M2P shows detectable structure)

---

## F. Worked Example (n = L*M*H = 4*8*4 = 128, minimal config)

Config: L=2, M=2, H=4, so n = 16.

Random matrix theory prediction at n=16:
  E[cos] = 0
  std ≈ 1/√16 = 0.25

Two pairs drawn i.i.d. from N(0, I_16):
  pair 1: cos ≈ 0.18  (within ±0.25 of 0, consistent with theory)
  pair 2: cos ≈ -0.12 (within ±0.25 of 0, consistent with theory)

For full config (n=2048):
  std ≈ 1/√2048 ≈ 0.022
  Expected range for 30 pairs: most values in [-0.066, +0.066] (3-sigma)

M2P output cosines should track this theory at initialization. After training,
if M2P learns to map inputs to a biased direction, the mean will shift away from 0.

---

## G. Complexity and Architecture Connection

**FLOPs for one M2P forward pass:**
- Each M2PBlock: standard self-attention on sequence length S, dim H
  - QKV projection: 3 * S * H^2 FLOPs
  - Attention: S^2 * H FLOPs
  - MLP: 2 * S * (4H) * H FLOPs
  - Column attention: S = L, MLP over M sequences
  - Row attention: S = M, MLP over L sequences

For L=4, M=8, H=64, N_blocks=4:
  Column block: 2*(3*4*64^2 + 4^2*64 + 2*4*(4*64)*64) ≈ 134K FLOPs
  Row block: 2*(3*8*64^2 + 8^2*64 + 2*8*(4*64)*64) ≈ 270K FLOPs
  Total: ~4 blocks * ~200K FLOPs ≈ 800K FLOPs

**Memory: ~197K parameters at float32 = ~0.79 MB**

This is negligible on Apple Silicon. The experiment runs locally on device.

**Connection to production architecture:**
In production SHINE (§3.4), the M2P transformer processes real hidden states
from a deployed LLM (L = number of LLM layers, H = LLM hidden dimension).
For Qwen-0.5B: L=28 layers, H=1024 → M2P processes (28, M, 1024) inputs.
Xavier-initialized positional embeddings become essential at L=28 to distinguish
early vs late layers (which have very different representational roles).

---

## H. Self-Test (MANDATORY)

**1. What is the ONE property being tested?**
Answer: Whether M2P outputs are statistically distinguishable from random noise
(two-sample t-test, p < 0.05, effect size > 0.05). This is the primary question.
No impossibility theorem is claimed — this is guided exploration without formal proof.

**2. Which existing theorem(s) does the prediction build on?**
Random matrix theory (Johnson-Lindenstrauss 1984; Vershynin 2018 Ch. 3):
for independent standard normal vectors in R^n, E[cos] = 0, std ≈ 1/√n.
This gives the null distribution for the t-test baseline.

**3. What specific numbers does the framework predict?**
- At initialization: mean M2P cosine ≈ 0 (in [-0.1, 0.1])
- At initialization: std ≈ 0.022 (for n=2048)
- Random baseline: mean ≈ 0, std ≈ 0.022 (theoretical, will be measured)
- Primary K827: p < 0.05 AND |diff| > 0.05 after training

**4. What would FALSIFY the prediction?**
- At initialization: P1 (mean in [-0.1, 0.1]) is NOT a valid prediction for M2P
  because shared positional embeddings (P_layer + P_token) create systematic positive
  cosine bias that violates the RMT independence assumption. Measured mean=0.0818
  (~19.5σ from zero) confirms this. P1 is stricken.
- P2 (std ≈ 0.022) remains valid: it measures input-conditional variance, unaffected
  by the shared positional mean shift. Measured std=0.0230, consistent with prediction.
- After training: K827 FAIL (p >= 0.05 OR |diff| <= 0.05) = M2P outputs not
  distinguishable from random = the port does not demonstrate structured behavior beyond
  the positional embedding bias.
- This is an honest falsifiable claim, not a tautology.

**5. How many hyperparameters does this approach add?**
Count: 0. The experiment tests the SHINE architecture as designed. L, M, H, n_pairs
are configuration choices, not method hyperparameters.

**6. Hack check: Am I adding fix #N to an existing stack?**
No. The revision removes Theorem 1 (tautology), adds a statistically grounded test,
and fixes positional embedding initialization. These are corrections, not additions.
The architecture is SHINE M2P, unchanged.
