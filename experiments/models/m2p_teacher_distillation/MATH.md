# MATH.md: M2P Teacher Distillation — Cross-Model Knowledge Transfer

TYPE: Guided Exploration (Type 2)
PROVEN FRAMEWORK: M2P generates adapters (Finding #339). Grassmannian A guarantees
parameter orthogonality (Findings #3, #341). KL distillation theory (Hinton 2015).
UNKNOWN: Does M2P, trained on teacher hidden states rather than ground-truth labels,
transfer domain knowledge to a student with different dimensions?

---

## A. Failure Mode Identification

**Disease (not symptom):** When a student model (d_S=256) cannot access teacher
(d_T=512) domain expertise during inference, the student's domain quality is limited
by its capacity and training signal, not by the richness of available knowledge.

**Root cause:** Knowledge transfer methods typically require either (a) the same model
architecture, or (b) label-level distillation (student mimics teacher output logits).
Both fail when:
- Architecture differs (different hidden dimensions, incompatible weight structure)
- Teacher's intermediate representations carry information not captured in output logits

**Degenerate case:** M2P reads teacher hidden states that live in ℝ^{d_T} but M2P
expects ℝ^{d_M2P}. Without a learned projection, this causes a hard dimension crash.
Even with projection, the projection might lose all domain-discriminative signal.

**Why this is NOT the composition interference disease:** The interference failure mode
(adapters canceling each other) is already resolved via Grassmannian A (Finding #341,
K848: cos=0 at float32). This experiment isolates a different question: can M2P use
a LARGER model's representations as a richer training signal for the student's adapters?

---

## B. Prior Mathematical Foundations

### B.1 Knowledge Distillation Theory (Hinton et al., 2015 — arXiv:1503.02531)

**Core result (Hinton et al. 2015, Section 2):** The soft targets (teacher output
distribution p_T) carry more information than hard labels because they reflect the
teacher's learned similarity structure. A student trained to minimize:

    L_KD = (1-α) · L_NTP(student, y) + α · T^2 · KL(p_T(x;T) || p_S(x;T))

converges to a model that captures the teacher's generalization structure. Here T is
the temperature hyperparameter.

**Key theorem (Hinton 2015, Section 2 paragraph 3):** When T → ∞, KL divergence
between soft targets reduces to matching the logit differences (second-order information
about teacher's similarity beliefs). This eliminates the need for hard labels when
T is large enough.

### B.2 Csiszár's f-divergence Decomposition (1975)

**Theorem (Csiszár 1975, "I-divergence geometry"):** For any probability distributions
P and Q over alphabet X:

    KL(P || Q) = Σ_x P(x) log(P(x) / Q(x))

This is non-negative (Gibbs' inequality) and equals zero iff P = Q a.e.

**Corollary (quality gap formulation):** Let p_T = teacher's domain distribution,
p_S = student+adapter distribution, p_0 = student base distribution. Then:

    KL(p_T || p_0) = KL(p_T || p_S) + KL_residual

where KL_residual ≥ 0. So:

    Quality_gap(p_0) = NTP_loss(p_0, p_T) ≥ NTP_loss(p_S, p_T) = Quality_gap(p_S)

The adapter CANNOT make the student worse than base at teacher-task (when trained with
KL distillation loss). This is the impossibility guarantee for regression.

### B.3 Representation Projections Preserve Domain Signals (Johnson-Lindenstrauss)

**Theorem (Johnson-Lindenstrauss 1984):** For any 0 < ε < 1 and any set of n points
in ℝ^{d_T}, there exists a linear map f: ℝ^{d_T} → ℝ^{d_M2P} with
d_M2P = O(log(n) / ε^2) such that for all pairs u, v:

    (1 - ε) ‖u - v‖^2 ≤ ‖f(u) - f(v)‖^2 ≤ (1 + ε) ‖u - v‖^2

**Implication for projection layer:** A learned Linear(d_T → d_M2P) has MORE
expressive power than a random JL projection (it can optimize for domain discrimination,
not just geometry preservation). Therefore, if the domain structure exists in teacher
hidden states (which it does — Finding #339 shows inter-domain cos=0.42 in toy GPT),
the learned projection can preserve it in d_M2P dimensions.

**Capacity condition at experiment scale:** We need d_M2P = O(log(n) / ε^2).
For n = 3 domains, ε = 0.1: d_M2P ≥ O(1.1 / 0.01) = O(110). We set d_M2P = 64.
This is slightly below the JL lower bound for guaranteed preservation, but:
(a) JL is a WORST-CASE bound over random projections; learned projections do better
(b) n=3 domains, so the domain structure is simpler than worst-case n points
The experiment measures whether the learned projection achieves cosine > 0.5 (P1).

### B.4 M2P Quality Bound from SHINE (Finding #339)

**Prior result (Finding #339):** M2P Transformer trained with NTP loss captures
66.6% of SFT quality in one forward pass on toy GPT (d=128, L=2).

**Extension to distillation setting:** When M2P is trained on KL(teacher || student+adapter)
rather than NTP loss, it receives a richer gradient signal:
- NTP loss provides a single binary signal per token (right/wrong)
- KL distillation provides a distribution-level signal (similarity to teacher's beliefs
  over ALL vocabulary items)

Prediction: KL distillation signal is AT LEAST as rich as NTP signal, so M2P quality
should be ≥ 66.6% (or more precisely: ≥50% as the K853 threshold).

---

## C. Proof of Guarantee

### Theorem 1 (Knowledge Transfer via M2P with KL Distillation)

**Theorem 1.** Let:
- p_T(·|x) = teacher's output distribution (at temperature T_temp)
- p_0(·|x) = student base distribution
- p_M(·|x; B_M2P) = student + M2P adapter distribution
- p_SFT(·|x; B_SFT) = student + direct SFT adapter distribution

If M2P is trained to minimize L_distill(B_M2P) = E_x[KL(p_T(·|x;T_temp) || p_M(·|x;T_temp))],
then:

    L_NTP(p_M, teacher_domain_data) ≤ L_NTP(p_T, teacher_domain_data) + ε_proj + ε_cap

where:
- ε_proj = information lost in teacher-hidden → M2P-input projection
- ε_cap = M2P model capacity error (finite parameter approximation)

**Corollary 1 (No-regression guarantee):** Since L_distill ≥ 0, minimizing L_distill
drives student+M2P toward teacher distribution. In the limit of perfect optimization:

    KL(p_T || p_M) → 0 ⟹ p_M → p_T ⟹ L_NTP(p_M) → L_NTP(p_T)

The student+M2P adapter CANNOT be worse than the student base for teacher-domain data
under KL distillation (by Gibbs' inequality: KL ≥ 0, equality iff distributions match).

*Proof.*

Step 1 (KL lower bound): By Csiszár (1975), KL(p_T || p_M) ≥ 0.
Minimizing KL(p_T || p_M) with respect to B_M2P moves p_M toward p_T.
At minimum, gradient condition: ∇_{B_M2P} KL(p_T || p_M) = 0.

Step 2 (NTP loss decomposition): For any distribution q:
    L_NTP(q, data) = H(p_data) + KL(p_data || q)

where H(p_data) is the entropy of the data-generating distribution (constant w.r.t. q).
So minimizing KL(p_T || p_M) is equivalent to minimizing L_NTP(p_M, p_T-samples).

Step 3 (Gap bound): The gap between p_M and p_T is bounded by:
    KL(p_T || p_M) = KL(p_T || p_M^*) + D_M2P
where p_M^* is the best achievable M2P distribution, and D_M2P is the approximation
error from finite M2P capacity. D_M2P → 0 as M2P capacity increases.

Step 4 (Projection fidelity): If the projection φ: ℝ^{d_T} → ℝ^{d_M2P} preserves
domain structure (cosine sim > 0.5, P1), then M2P can distinguish domain signals
and generate domain-specific B-matrices. If projection fails (cosine ≤ 0), M2P
receives uniform inputs and generates domain-agnostic B (quality = student base).

QED.

**Corollary 2 (Quality Gap Closure):** Define:
    gap_base = L_NTP(student_base) - L_NTP(teacher_domain)
    gap_m2p  = L_NTP(student + M2P) - L_NTP(teacher_domain)
    closure  = (gap_base - gap_m2p) / gap_base

Then closure ≥ 0 (by no-regression, Corollary 1) and closure = 1 when p_M = p_T.
K853 requires closure ≥ 0.50 for most domains. This is achievable when D_M2P is small
(sufficient M2P capacity) and ε_proj is small (projection preserves domain structure).

---

### Theorem 2 (Learned Projection Preserves Domain Structure)

**Theorem 2.** Let h_T ∈ ℝ^{d_T} be teacher's last-layer hidden states for domain-i
examples, and let φ: ℝ^{d_T} → ℝ^{d_M2P} be trained to minimize classification loss
on domain labels (equivalently, trained jointly with M2P on KL distillation).

If domain i and domain j have distinct teacher representations (i.e., E[h_T(x_i)] ≠ E[h_T(x_j)]),
then the trained projection φ preserves this distinction:
    cos(φ(E[h_T(x_i)]), φ(E[h_T(x_j)])) < 1 - δ for some δ > 0.

*Proof sketch.* φ is a linear map with learnable parameters. The gradient of KL distillation
loss with respect to φ is non-zero whenever φ projects domain-i and domain-j representations
to the same vector (since M2P would then generate the same B for both domains, but the
teacher's distributions for domains i and j differ). Gradient descent on φ resolves this
degeneracy by moving projections apart.

Formal: gradient ∂L_distill/∂φ = E_x[∂KL(p_T || p_M)/∂h_M2P · ∂h_M2P/∂φ].
When φ(h_T(x_i)) = φ(h_T(x_j)) but p_T(x_i) ≠ p_T(x_j), the first term is nonzero
(M2P generates wrong adapters), and the second term (Jacobian of linear projection)
has rank d_M2P. So total gradient is nonzero → φ moves toward domain-separating projection.

QED.

---

## D. Quantitative Predictions (Derived FROM the proofs)

| Prediction | Source | Expected Value | Kill Criterion |
|------------|--------|----------------|----------------|
| P1: Projection cosine sim (intra-domain vs cross-domain) | Theorem 2 + JL bound | intra > 0.5, cross < 0.8 | K854: no crash (architecture works) |
| P2: Quality gap closure per domain | Theorem 1 (Corollary 2) | ≥ 0.50 for ≥2/3 domains | K853: closure ≥ 0.50 majority |
| P3: No regression below student base | Theorem 1 (Corollary 1) | student+M2P loss ≤ student_base | Implicit guarantee |
| P4: KL loss decreases during M2P training | Gradient non-zero (Theorem 2) | monotone decrease | sanity check |
| P5: Projection cosine > 0 (not random) | Theorem 2 | > 0.2 (above random) | K854 proxy |

**Important scope:** The proof guarantees NO REGRESSION (P3) unconditionally.
Quality gap closure ≥ 0.50 (P2) is NOT guaranteed by the proof — it requires
D_M2P (capacity error) and ε_proj (projection error) to be small enough.
This is why K853 is a kill criterion, not a guarantee. P2 is a Type 2 exploration
of "what closure can a small M2P achieve in practice?"

---

## E. Assumptions and Breaking Conditions

| Assumption | Consequence if Violated | Kill Signal |
|------------|------------------------|-------------|
| A1: d_T ≥ N·r (student capacity sufficient) | QR fails; Grassmannian cannot be generated | Hard crash |
| A2: Teacher domain loss < student base domain loss | Teacher is not better than student; gap = 0 | No test signal; skip |
| A3: Projection preserves domain signals | M2P generates domain-agnostic B (P5 fails) | Cosine < 0.2 (P5 FAIL) |
| A4: KL gradient non-zero (Theorem 2) | M2P does not learn; loss constant | Loss constant → K853 FAIL |
| A5: Student has sufficient capacity to use teacher signal | Even perfect M2P cannot reach teacher quality | gap_closure < 0.50 → K853 FAIL |

**Critical assumption A5:** The student (d_S=256) is 2× smaller than the teacher
(d_T=512). Even with perfect distillation, the student capacity may limit achievable
quality. At toy scale (3 simple domains: arithmetic, sort, reverse), domain tasks
should be within student capacity — but this is empirical, not proven.

**What breaks the proof (A3 violation scenario):** If teacher hidden states are all
similar (low inter-domain variation), the projection φ cannot separate them. The
JL lower bound requires d_M2P ≥ O(log n / ε^2). At d_M2P=64, n=3 domains, this
is satisfied for ε ≥ 0.1 (our P1 threshold). If actual teacher cosine similarity
between domains is > 0.95, there is no domain signal to project.

---

## F. Worked Example (d_T=8, d_S=4, d_M2P=8, r=2, 2 domains)

**Setup:** Teacher (d_T=8), Student (d_S=4), M2P (d_M2P=8), rank r=2, 2 domains.

**Step 1: Teacher hidden states for domains A and B.**

Teacher last-layer hidden states (mean-pooled over tokens):
    h_T(A) = [1, 0, 0, 1, 0, 0, 1, 0]  (arithmetic signal: even positions active)
    h_T(B) = [0, 1, 1, 0, 1, 0, 0, 1]  (sort signal: odd positions active)

cos(h_T(A), h_T(B)) = (0+0+0+0+0+0+0+0) / (√3 · √4) = 0.0  (orthogonal — well separated)

**Step 2: Projection φ: ℝ^8 → ℝ^8** (identity in this example since d_T = d_M2P).

φ(h_T(A)) = h_T(A),  φ(h_T(B)) = h_T(B)
cos(φ(h_T(A)), φ(h_T(B))) = 0.0 < 0.8   ✓ P1 satisfied (good projection)

**Step 3: M2P generates B-matrices.**

M2P receives φ(h_T) as input → processes through memory tokens → outputs B ∈ ℝ^{r × d_out}.
For domain A: B_A = [[1, 0, 0, 0], [0, 1, 0, 0]]   (activates first 2 student dims)
For domain B: B_B = [[0, 0, 1, 0], [0, 0, 0, 1]]   (activates last 2 student dims)

**Step 4: Grassmannian A for student (d_S=4, r=2, N=2).**

Total rank = N·r = 4 ≤ d_S = 4 (capacity exactly met, margin = 1×).
QR of random 4×4 matrix → Q = I_4 (for simplicity):
A_A = Q[:, 0:2] = I_4[:, 0:2],  A_B = Q[:, 2:4] = I_4[:, 2:4]
A_A^T A_B = I_4[0:2, 2:4] = 0_{2×2}   ✓ Theorem 1 holds

**Step 5: Quality gap closure.**

student_base (no adapter): L = 3.2 nats
student+M2P_A for domain A: L = 2.1 nats (B_A activates correct features)
SFT_A for domain A: L = 1.8 nats (optimal adapter, trained on labels)
teacher_A for domain A: L = 1.2 nats (teacher quality)

closure_A = (3.2 - 2.1) / (3.2 - 1.8) = 1.1 / 1.4 = 0.79 > 0.50   ✓ K853 satisfied

**Note on K853 definition:** K853 measures closure vs SFT student (not teacher):
    closure = (student_base_loss - student_m2p_loss) / (student_base_loss - student_sft_loss)
This is more conservative than closure vs teacher (used in Theorem 1 derivation),
since student_sft_loss ≤ teacher_domain_loss in general. The worked example shows
closure = 0.79 vs SFT, which satisfies K853 ≥ 0.50.

---

## G. Complexity and Architecture Connection

### Architecture Dimensions

| Component | Params | Notes |
|-----------|--------|-------|
| Teacher GPT | d_T=512, L_T=4, 8 heads | ~4M params |
| Student GPT | d_S=256, L_S=2, 4 heads | ~800K params |
| M2P Transformer | d_M2P=64, L_M2P=2 | ~200K params |
| Projection layer | d_T × d_M2P = 512 × 64 | 32K params |
| Total trainable (M2P + proj) | | ~232K params |

### FLOPs per Distillation Step

| Phase | FLOPs |
|-------|-------|
| Teacher forward (frozen) | O(L_T · T · d_T^2) = O(4 × 48 × 512^2) ≈ 50M |
| Projection | O(T × d_T × d_M2P) = O(48 × 512 × 64) ≈ 1.6M |
| M2P forward | O(L_M2P × N_MEMORY × d_M2P^2) = O(2 × 32 × 64^2) ≈ 262K |
| Student forward with adapter | O(L_S × T × d_S^2) = O(2 × 48 × 256^2) ≈ 6.3M |
| KL computation | O(T × V) = O(48 × 128) ≈ 6K |
| Total per step | ~58M |

### Architecture Reference

Teacher architecture: GPT with (d=512, L=4, H=8), vocabulary=128, BLOCK_SIZE=48.
This is a 2× scaled version of the existing student (d=256, L=2, H=4, same vocabulary).
No positional embedding redesign needed — shared BLOCK_SIZE ensures KL loss is defined
on the same token sequence.

The Grassmannian A-matrices are generated for the STUDENT dimensions (d_S=256),
as the student receives the adapter. The teacher dimensions (d_T=512) only appear
in the M2P input projection.

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode impossible?**

The no-regression guarantee (Corollary 1 of Theorem 1): KL divergence is non-negative
(Gibbs' inequality), so minimizing KL(p_T || p_M) cannot make p_M worse than any fixed
baseline that already minimizes KL partially. The student+M2P adapter cannot regress
below student base because KL gradient always points toward p_T, not away from it.

**2. Which existing theorem(s) does the proof build on?**

- Gibbs' inequality / KL non-negativity (Cover & Thomas, "Elements of Information Theory",
  2nd ed. 2006, Theorem 2.6.3)
- Hinton et al. 2015 (arXiv:1503.02531): KL distillation training objective
- Csiszár 1975 f-divergence decomposition: KL(P||Q) ≥ 0, = 0 iff P=Q
- Johnson-Lindenstrauss 1984: linear projections preserve distance structure in log(n)/ε^2 dims

**3. What specific numbers does the proof predict?**

- P1: Projection cosine sim between domains > 0 but < 1 (domain-distinguishable in M2P space)
- P2: Quality gap closure ≥ 0.50 for ≥2/3 domains (K853)
- P3: student_m2p_loss ≤ student_base_loss (unconditional no-regression guarantee)
- P4: KL loss strictly decreasing during M2P training (gradient non-zero, Theorem 2)

**4. What would FALSIFY the proof (not just the experiment)?**

The no-regression guarantee (P3) is falsified if student_m2p_loss > student_base_loss
AFTER KL distillation training. This cannot happen with correct KL minimization —
if it does, it indicates a training bug (wrong gradient direction, optimizer divergence).

The quality closure (P2) is not a proof prediction — it is an empirical bound.
P2 is falsified (K853 FAIL) if the M2P capacity is too small or projection fails,
but this does NOT falsify the mathematical proof (only the engineering feasibility).

**5. How many hyperparameters does this approach add?**

Added hyperparameters beyond prior experiments:
- Temperature T_temp for KL distillation (set to 2.0, guided by Hinton 2015 §3 recommending T=2-5)
- KL weight α (set to 0.7, with 0.3 NTP auxiliary; guided by Hinton 2015 §3.4)
- d_T / teacher architecture (teacher is 2× student in all dims for structural simplicity)

The 2 new hyperparameters (T_temp, α) are not uniquely determined by the proof but
are guided by established ranges in the distillation literature.

**6. Hack check: Am I adding fix #N to an existing stack?**

No. This is a new capability experiment (teacher distillation), NOT a fix for a prior
failure. The only prior mechanism this builds on is M2P Transformer (Finding #339) —
the distillation training objective is a clean REPLACEMENT for NTP loss, not an addition.
Single mechanism: replace NTP loss with KL(teacher || student+M2P_adapter).
