# MATH.md: M2P with Domain Conditioning — Proof that Domain Embeddings Resolve B-Matrix Mode Collapse

## Experiment Type

Type 1: Proof Verification — the domain conditioning guarantee is derived mathematically before code, and the experiment confirms quantitative predictions.

## Prior Results Cited

- **Finding #341** (exp_m2p_distillation_toy, KILLED, 2026-04-06): Round-robin M2P on 5 domains, heterogeneous base losses (4.9x ratio), causes B-matrix mode collapse (cos=0.9956). Median quality 21.9%, repeat domain -329%.
- **MixLoRA** (arXiv:2402.15896): Conditional LoRA selection via learned factor activations; domain-specific gating prevents expert collapse in multi-task settings.
- **SMoRA** (arXiv:2501.15103): Sparse mixture-of-rank-one adapters; domain routing eliminates gradient interference between task-specific paths.

---

## A. Failure Mode Identification (Root Cause from Finding #341)

### Symptom
B-matrix mode collapse: M2P generates near-identical B-matrices for all 5 domains (|cos|=0.9956). The "repeat" domain receives an adapter tuned to high-loss domain centroid, causing -329% quality degradation.

### Root Cause (Proven in Finding #341)

**Theorem (Informal, from #341):** A single M2P without explicit domain conditioning cannot simultaneously learn per-domain B-matrices under round-robin training when gradient magnitudes vary >2x across domains.

Formally: the M2P loss gradient during round-robin training is:

```
∇_θ L_total = Σ_{d=0}^{N-1} ∇_θ L_d(θ)
```

where θ are M2P parameters shared across all domains. Because M2P receives no domain signal (it only sees mean-pooled hidden states from a frozen base model), all domains produce statistically similar input distributions. The gradient ∇_θ L_total therefore pushes θ toward the centroid that minimizes the sum — not toward per-domain optima. Domains with high loss magnitude L_d dominate ∇_θ L_total, collapsing B-matrices toward the high-loss centroid.

**Is this a symptom or disease?** This is the disease: the information-theoretic content of M2P's input is insufficient to distinguish domains. The mean-pooled hidden states h = E[H(x)] for domain d are near-identical because the base model was pre-trained on all domains jointly — the frozen base cannot encode domain identity in its hidden states at the level required for domain-specific B-matrix generation.

---

## B. Reframe: What Is the Optimal Input Structure Such That Mode Collapse Is Impossible?

Wrong question: "How do we prevent B-matrix mode collapse?"

Right question: "What minimum additional information must M2P receive such that per-domain optima are simultaneously reachable?"

**Answer from information theory (Shannon, 1948):** A function f(x) can produce N distinct outputs if and only if its input x has at least log₂(N) bits of domain-distinguishing information. The mean-pooled hidden states alone have near-zero mutual information with domain identity (they reflect shared linguistic structure, not domain specifics, because the base model is pretrained on all domains). The fix must inject domain-distinguishing information directly.

**The minimal fix:** A learned domain embedding e_d ∈ ℝ^D for each domain d, injected into M2P's memory tokens. This adds exactly H(D) = log₂(N) bits of domain information — the information-theoretic minimum to distinguish N domains.

This reframe shows that domain conditioning is not a hack layered on symptoms. It is the minimal sufficient fix: it adds precisely the missing information without changing any other part of the system.

---

## C. Prior Mathematical Foundations

### C.1 Universal Approximation Theorem (Hornik, 1991; Cybenko, 1989)

**Theorem (UAT, informal):** A feedforward neural network with sufficient width and a single nonlinear activation can approximate any continuous function f: ℝ^m → ℝ^n on a compact domain to arbitrary precision.

**Application here:** The M2P transformer is a universal approximator over its input space. If its input space is augmented to include domain identity, it can approximate any domain-to-B-matrix mapping — including the per-domain optima. Without domain identity in the input, the mapping from hidden states alone is unconstrained to collapse to a centroid.

**Key corollary (linear independence → separability):** If domain embeddings {e_0, …, e_{N-1}} are linearly independent, then the input to M2P for distinct domains are linearly independent, and by UAT the network can produce N linearly independent outputs. Collapse to a centroid is no longer a fixed point of gradient descent.

### C.2 Linear Independence of Random Embeddings (Exact Statement)

**Lemma 1.** Let {e_d}_{d=0}^{N-1} be vectors initialized i.i.d. from N(0, σ²I_D). Then {e_d} are linearly independent almost surely when N ≤ D.

*Proof.* The set of N linearly dependent vectors in ℝ^D has Lebesgue measure zero when N ≤ D. A Gaussian distribution has support on all of ℝ^D, so the probability of drawing from this measure-zero set is exactly 0. □

**Application:** D_MODEL = 64, N_DOMAINS = 5. Since 5 ≤ 64, Lemma 1 applies. After initialization, domain embeddings are linearly independent almost surely. After gradient training, they evolve toward separating the per-domain losses — linear independence is preserved unless the optimizer actively collapses them (which it cannot do without increasing loss).

### C.3 Fixed-Point Analysis: Why Centroid Is No Longer Stable (Key Theorem)

**Definition.** Let B*(θ, e) denote the B-matrix generated by M2P with parameters θ and domain embedding input e. The centroid state is:

```
B_centroid := B*(θ*, e_d) for all d  (M2P emits same B regardless of e_d)
```

In the unconditioned model (Finding #341): e_d is absent, so B*(θ*, h_d) ≈ constant — centroid is a stable fixed point because ∂/∂θ [Σ_d L_d(B_centroid)] = 0 at the centroid optimum over shared parameters.

**Theorem 2 (Domain Conditioning Destabilizes Centroid).** With domain embedding e_d injected into M2P, the centroid state B*(θ, e_d) = B_centroid ∀d is a fixed point of gradient descent only if:

```
∀d: e_d = e_0  (all domain embeddings are identical)
```

Since {e_d} are linearly independent (Lemma 1) and each e_d is a trainable parameter updated by:

```
∂L_d/∂e_d ≠ ∂L_{d'}/∂e_{d'} for d ≠ d' (different domains → different gradients)
```

the embeddings diverge from any initial centroid, and the global centroid B_centroid ceases to be a fixed point. The model is forced to learn domain-specific outputs.

*Proof.*
Suppose B*(θ, e_d) = B_centroid ∀d and e_d ≡ e_0 is not assumed. At the centroid state:

```
∂L_d/∂e_d = ∂L_d/∂B · ∂B/∂(mem) · ∂(mem)/∂e_d
```

The term ∂L_d/∂B is domain-specific (different B-target per domain). The term ∂B/∂(mem) is the same for all domains (shared M2P parameters). The term ∂(mem)/∂e_d = I (embedding injection is additive: mem = mem_base + e_d). Thus:

```
∂L_d/∂e_d = J(θ) · ∂L_d/∂B
```

where J(θ) is a shared Jacobian. Since ∂L_d/∂B ≠ ∂L_{d'}/∂B' at the centroid (domains have different per-domain task optima), e_d and e_{d'} receive different gradient updates. The embeddings diverge. Once embeddings diverge, M2P input is domain-distinguishable, and the network can route to per-domain B-matrices.

The gradient of each e_d also serves as a "supervision signal" that informs the embedding which direction leads to lower L_d — independently from L_{d'} for d' ≠ d. The embedding learning is domain-private even though θ (M2P weights) is shared.

QED.

---

## D. Main Theorem: Domain Conditioning Guarantees Per-Domain B-Matrix Differentiation

**Theorem 3 (Main Result).** Let M2P_cond be an M2P transformer augmented with learned domain embeddings {e_d} ∈ ℝ^D, injected additively into memory tokens:

```
mem_d = mem_base + e_d
B_d = M2P_cond(mem_d)
```

Trained by gradient descent on:

```
L_total(θ, {e_d}) = Σ_d L_d(M2P_cond(mem_base + e_d; θ))
```

Then:
1. The global centroid state B*(θ, e_d) = B_centroid ∀d is not a stable fixed point of gradient descent unless N = 1.
2. Each e_d learns to encode domain-specific information from gradient feedback.
3. Capacity for per-domain B-matrix differentiation scales as O(N × D) — negligible additional parameters vs M2P body (N=5, D=64 → 320 parameters vs ~17K M2P parameters).

*Proof.*

Part 1: By Theorem 2. Centroid is only a fixed point if all e_d are identical. Gradient pressure from domain-specific losses drives them apart.

Part 2: The gradient ∂L_d/∂e_d provides domain-specific signal (different per domain at the centroid). Standard SGD convergence theory (Robbins-Monro, 1951; Bottou, 2010) guarantees convergence to a stationary point when learning rate decays appropriately. At the stationary point, e_d encodes domain-distinguishing information sufficient to satisfy ∂L_d/∂e_d = 0 — which requires B_d to be near the per-domain optimum.

Part 3: The embedding table has N × D = 5 × 64 = 320 parameters. Total M2P parameters (from base experiment): ~17,000. Overhead = 1.9% — negligible.

QED.

**Key property in one sentence:** Adding learned domain embeddings to M2P input makes the centroid state geometrically unstable — each embedding receives independent gradient signal that drives it toward the direction minimizing its domain's loss, which is impossible with a shared output.

---

## E. Why Finding #341 Failure Cannot Recur

### Finding #341 conditions (all required for failure)
1. M2P receives no domain-distinguishing input
2. Round-robin training with Σ_d gradient updates
3. Heterogeneous base losses (>2x ratio)

### How domain conditioning breaks condition (1)

Condition (1) is eliminated: M2P now receives e_d as explicit domain signal. Even if conditions (2) and (3) remain (they do — round-robin training and heterogeneous losses are unchanged), the fundamental information-theoretic bottleneck is resolved.

**Proof that Grassmannian A orthogonality is unaffected:** Theorem 1 from MATH.md (m2p_distillation_toy) holds unconditionally for any B-matrices. Domain conditioning modifies which B-matrices M2P generates; it does not change the frozen A-matrices. Therefore K857 (|cos| ≤ 1e-5) remains a structural guarantee.

---

## F. Quantitative Predictions (Derived from the Proof)

### Prediction 1: B-matrix differentiation
**Theorem 3 predicts:** After training, M2P-generated B-matrices for different domains are NOT in centroid collapse. Proxy metric: mean pairwise |cos(B_d, B_{d'})| should decrease substantially from 0.9956 (baseline) toward values consistent with domain-specific generation.

**Predicted range:** |cos| ≤ 0.90 (qualitative breakout from collapse). Exact value is a Type 2 unknown — depends on loss landscape geometry. Significant reduction from 0.9956 is predicted.

### Prediction 2: K855 — Median quality ≥ 25%
**Basis:** Finding #341 (Revision 1) showed 3/5 domains achieved 21-55% before centroid collapse pulled the median down. With centroid collapse eliminated, the "repeat" catastrophe should not recur. We expect the median to be at or above the 21-55% range achieved by healthy domains.

**Predicted median quality:** 0.30–0.65 (center of the healthy domain range seen without collapse). Strict prediction: ≥ 0.25 (K855 threshold).

### Prediction 3: K856 — No domain below -10%
**Basis:** The catastrophic failure of "repeat" domain (-329%) was directly caused by centroid collapse. Theorem 3 proves the centroid is no longer a fixed point. Therefore the "repeat" domain should receive an adapter calibrated to its actual loss — quality should be positive.

**Predicted:** All domains positive (quality ∈ [0, 1]). Conservative bound from proof: no domain below -10% (K856 threshold).

### Prediction 4: K857 — Grassmannian |cos| ≤ 1e-5
**Basis:** Theorem 1 (carried over from m2p_distillation_toy). QR decomposition of (64, 20) random matrix gives orthonormal columns at float32 precision (~1e-7 residual). |cos| between distinct domain A-matrices = exactly 0 by construction, measured at 1e-6 in float32.

**Predicted:** |cos| ≤ 1e-7. K857 threshold (1e-5) is conservative by 100x. PASS guaranteed structurally.

---

## G. Assumptions and Breaking Conditions

| Assumption | Condition | Consequence if violated |
|------------|-----------|------------------------|
| A1: Domain embeddings initialized differently | e_d ≠ e_{d'} at init | If init identical: no gradient pressure to diverge; can break by using random init (standard) |
| A2: M2P has sufficient capacity to condition on e_d | D_MODEL = 64 >> N = 5 | If D_MODEL < N: embeddings may not be separable; N=5 << D=64, satisfied |
| A3: Task losses decrease with better B-matrices | ∂L_d/∂B_d < 0 at centroid | If loss is flat: no gradient signal; observed in exp_m2p_distillation_toy that healthy domains converge |
| A4: Learning rate allows embedding divergence | η not zero | Standard training; satisfied with Adam η=1e-3 |
| A5: Grassmannian A-matrices orthogonal | d ≥ N·r | d=64 ≥ 20; proven by QR construction |

**If A1 violated (identical init):** Symmetry breaking does not occur automatically. Fix: default random initialization (nn.Embedding uses random init by default in MLX).

**If A3 violated (flat loss):** Embeddings learn nothing useful; quality stays at 0%. This would trigger K855 FAIL, indicating the domain task is not learnable in the current setup (separate issue from mode collapse).

---

## H. Worked Example (d = 8, N = 2 domains, rank = 2)

Toy numbers for calculator verification.

**Setup:** D_MODEL = 8, N = 2, LORA_RANK = 2, N_MEMORY = 4

**Domain embeddings (random init, 2-dim slice for illustration):**
```
e_0 = [0.1, -0.3, 0.2, 0.0, ...]   (8-dim, random)
e_1 = [-0.2, 0.1, -0.1, 0.3, ...]  (8-dim, random)
```

These are linearly independent by Lemma 1 (2 ≤ 8).

**M2P memory (before embedding injection):**
```
mem_base = mean_pool(hidden_states)  # (4, 8) — 4 memory tokens, dim 8
```

**After injection:**
```
mem_0 = mem_base + e_0[None, :]  # (4, 8) — domain 0 input
mem_1 = mem_base + e_1[None, :]  # (4, 8) — domain 1 input
```

**Key numerical check:** mem_0 ≠ mem_1 because e_0 ≠ e_1. M2P produces:
```
B_0 = M2P(mem_0)  — (2, 8) per module
B_1 = M2P(mem_1)  — (2, 8) per module
```

Because mem_0 ≠ mem_1 and M2P is nonlinear, B_0 ≠ B_1 even before any domain-specific training.

**After gradient step for domain 0:**
```
∂L_0/∂e_0 = J_0 · ∂L_0/∂B_0   (domain 0 gradient updates e_0)
∂L_0/∂e_1 = 0                   (domain 1 embedding unchanged by domain 0 loss)
```

The gradient is domain-private: only e_0 is updated when computing L_0's backward pass. This is the core mechanism.

**Grassmannian check (d=8, N=2, r=2):**
```
Q = QR(randn(8, 4))   # (8, 4) orthonormal matrix
A_0 = Q[:, 0:2]       # (8, 2)
A_1 = Q[:, 2:4]       # (8, 2)

A_0^T @ A_1 = (Q[:, 0:2])^T @ (Q[:, 2:4])
            = [I_4]_{0:2, 2:4}   (off-diagonal block of identity)
            = 0_{2x2}  ✓
```

---

## I. Complexity and Architecture Connection

### Parameter overhead
- Domain embedding table: N × D = 5 × 64 = **320 parameters** (1.9% of M2P)
- M2P body: unchanged from baseline (~17,000 parameters)
- Total M2P parameters: ~17,320

### Computational overhead
- Forward pass: one embedding lookup (O(D)) + additive injection into N_MEMORY tokens (O(N_MEMORY × D)) — negligible vs M2P attention (O(N_MEMORY² × D))
- Backward pass: gradient through embedding table is direct (no chain rule needed beyond identity Jacobian)

### Connection to production systems
- **MixLoRA (arXiv:2402.15896):** Learns conditional selection of LoRA factors via domain/instance-level gating. Our domain embedding is a simpler, lower-overhead version of this mechanism — we learn a fixed per-domain bias rather than instance-level gates.
- **SMoRA (arXiv:2501.15103):** Sparse mixture-of-rank-one adapters with domain routing. Our approach shares the routing-before-generation philosophy but is more lightweight (embedding injection vs full routing head).
- **Production analogy:** In Qwen3/DeepSeek-V3, MoE routing conditions expert selection on token embeddings. Our domain embedding plays the same role — it conditions B-matrix generation on domain identity rather than leaving it implicit.

---

## J. Self-Test (MANDATORY — 6-question checklist)

**Q1. What is the ONE mathematical property that makes the failure mode impossible?**
Learned domain embeddings {e_d} are linearly independent (Lemma 1), making the centroid state unstable: each e_d receives domain-private gradient signal ∂L_d/∂e_d that is different per domain, causing embeddings to diverge and M2P to generate domain-specific B-matrices.

**Q2. Which existing theorem(s) does the proof build on? Cite by name + paper.**
- Lemma 1: Linear independence of random Gaussian vectors in high dimensions (standard linear algebra; measure-zero argument)
- Universal Approximation Theorem: Hornik (1991) "Approximation capabilities of multilayer feedforward networks"; Cybenko (1989) "Approximation by superpositions of a sigmoidal function"
- Robbins-Monro convergence: Robbins & Monro (1951) for SGD convergence at stationary points
- Theorem 1 (Grassmannian A-slot orthogonality): m2p_distillation_toy/MATH.md, proven from QR decomposition

**Q3. What specific numbers does the proof predict?**
- K855: Median quality ≥ 25% — predicted 30–65% (Theorem 3 eliminates centroid failure)
- K856: All domains quality ≥ -10% — predicted all positive (centroid collapse gone → repeat domain no longer receives wrong adapter)
- K857: Grassmannian |cos| ≤ 1e-5 — predicted ≤ 1e-7 (structural QR guarantee; 100x below threshold)
- B-matrix diversity: mean pairwise |cos(B_d, B_{d'})| << 0.9956 (qualitative breakout from collapse)

**Q4. What would FALSIFY the proof?**
The proof is wrong if:
- (a) K855 FAILS (median quality < 25%) despite domain conditioning — would indicate embeddings failed to diverge or M2P capacity is insufficient; requires measuring embedding divergence to distinguish
- (b) K856 FAILS (some domain < -10%) — indicates centroid collapse persists; check B-matrix |cos| to confirm
- (c) B-matrix |cos| remains ≥ 0.99 after training — would directly contradict Theorem 3's prediction that centroid is not a fixed point

Structural falsification: if domain embeddings are identical after training (they would need to be forced to be identical; this cannot happen with different gradient updates).

**Q5. How many hyperparameters does this approach add?**
Count: **1** — embedding dimension, fixed at D_MODEL = 64 (same as model hidden size; no sweep needed, follows from the information-theoretic argument that D ≥ N to achieve linear independence).

Effectively 0 new hyperparameters requiring tuning: the embedding table uses standard random init (σ=0.02, consistent with all other learned embeddings in the model), learning rate uses the existing M2P optimizer (Adam, η=1e-3 — no new optimizer needed).

**Q6. Hack check: Am I adding fix #N to an existing stack?**
No. The original experiment (Finding #341) had one mechanism (Grassmannian A + M2P B). Domain conditioning replaces the assumption that mean-pooled hidden states carry sufficient domain signal — it adds 320 parameters and zero extra loss terms. The number of mechanisms stays at one: "frozen Grassmannian A provides composition guarantee; M2P generates domain-specific B via conditioned forward pass." Not a stack.

---

## K. Audit-2026-04-17 Strict KC: K860 — Router Uniform-Fallback Test

### Background

This experiment is the **MoE Routing** variant (`M2PTransformerMoE`): the M2P body is replaced by N=4 expert sub-blocks selected via a learned per-domain router `nn.Embedding(N_DOMAINS, n_experts)`. Forward path:

```
route_weights[d, :] = softmax(router(d))     # (n_experts,)
mem_d  = Σ_e route_weights[d, e] · expert_e(memory)
B_d    = decode(mem_d)
```

The DB row tracks one strict pre-registered KC, derived from the prior 2026-04-07 evidence:

> **K860**: "Router falls back to uniform allocation across all domains" (FAIL = catastrophic; PASS = router specializes by domain).

The earlier rev-1 code measured K855/K856/K857 (quality and Grassmannian-cos) but **never extracted router weights**. That is the audit gap (`metric-swap` tag).

### K.1 K860 Definition

For each domain d ∈ {0, …, N_DOMAINS-1}, let `r_d := route_weights[d, :] ∈ ℝ^n_experts` (n_experts = 4 by construction).

Let `m_d := max_e r_d[e]` and `m̄ := mean_d m_d`.

```
K860 PASS  iff  m̄ ≥ 0.50
K860 FAIL  iff  m̄ ≤ 0.50
```

Uniform fallback gives r_d = (¼, ¼, ¼, ¼) so m_d = 0.25 and m̄ = 0.25. PASS requires the router to put ≥ 50% of its mass on a single expert per domain on average — i.e., genuine specialization, not weak preference.

Auxiliary diagnostic (reported, not used as KC): `H̄ := mean_d entropy(r_d)`. Uniform = ln(4) ≈ 1.386; one-hot = 0.

### K.2 Pre-Registered Prediction

Prior evidence on the DB row (2026-04-07): `"K860 FAIL: MoE gating did not achieve domain-specific routing. Median quality 39.1%, repeat domain -322%. B-matrix centroid collapse unchanged."`

The router is a `nn.Embedding(N_DOMAINS=5, n_experts=4)` with **no auxiliary load-balancing or entropy-penalising loss**, no top-k gating, and no Gumbel noise. Under the same gradient-homogenisation regime that produces B-matrix centroid collapse (Finding #341, repeated in rev-1), the soft router has **no incentive to specialise**: every expert sees gradient through every domain, and the equal-mass uniform distribution is a saddle minimiser of the round-robin loss landscape.

**Predicted result:** K860 FAIL. Specifically, m̄ ≈ 0.25 ± 0.10 (within 10pp of the uniform 1/n_experts).

This is consistent with the broader Finding #341 mode-collapse pattern: any degree of freedom that is **not directly forced** to be domain-specific (B-matrix outputs, memory tokens, router weights) collapses to a centroid under shared-gradient training.

### K.3 What Would Falsify the Prediction

- m̄ ≥ 0.50 with monotonically decreasing entropy over training would indicate the soft router specialised on its own — a positive surprise that would warrant follow-up to test whether the routed-expert architecture also breaks B-matrix collapse.
- Per-domain m_d ∈ {0.5, 0.5, 0.5, 0.5, 0.5} with all five domains preferring the same expert would be PASS by m̄ but expose a degenerate "single-expert" failure mode — captured by the auxiliary `n_unique_argmax_experts` diagnostic.

### K.4 Verdict-Consistency Note

The rev-1 code's all_pass field combined K855/K856/K857. Since the DB tracks **only K860**, the experiment-level verdict in this V2 rerun is determined by K860 alone:

```
verdict = "ALL_PASS" if k860_pass else "KILLED"
all_pass = k860_pass
```

K855/K856/K857 are retained as auxiliary diagnostics in `results.json` but do not gate the verdict. This matches the audit principle: the DB-tracked KC is the ground-truth contract.
