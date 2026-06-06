# LEARNINGS: exp_m2p_cross_domain_graph

## Core Finding

Cross-domain M2P transfer is broadly effective (8/10 pairs reduce target loss >5%), with Option A (cross-prediction: student M2P reads teacher hidden states to generate its own B matrix) consistently outperforming residual and combined approaches. However, dissolve-recrystallize at multi-adapter scale fails catastrophically for near-optimal domains due to 25x effective scale amplification (10 adapters × PROMOTE_SCALE=5 / LORA_SCALE=2).

---

## Why This Happened

### Option A wins over Option B

Option A generates a new B matrix for the student domain from teacher-domain hidden states. Option B transfers the teacher's B matrix residual directly. Option A is superior because:

- **Distribution alignment**: the student's M2P encoder maps teacher hidden states into the student's null-space context. The generated B matrix is native to the student adapter's geometry.
- **Avoids feature-space bridging**: Option B tries to transfer parameters across two different feature spaces, which fails when their spectral profiles diverge. This matches Task Arithmetic (Ilharco et al., 2212.04089) — composing task vectors works when they live in the same vector space; residual transfer breaks when domains are non-isomorphic.
- **Empirical split**: arithmetic and sort (structured domains) benefit most from cross-prediction. The student generates B matrices that capture structural regularities from the teacher, not just teacher-specific parameters.

### Dissolve-Recrystallize: success and catastrophe

The repeat domain improves 103.3% of SFT via dissolve — the domain has genuine headroom (base loss 1.37, SFT 0.77) and gains from cross-domain enrichment.

Parity collapses from loss 0.59 to 3.73 (6.3x regression). Root cause: **scale amplification**.
- Each adapter was trained at LORA_SCALE=2
- Dissolve promotes at PROMOTE_SCALE=5: single-adapter amplification = 5/2 = 2.5x (validated by Finding #333)
- 10 adapters dissolved simultaneously: effective amplification = 2.5x × 10 = **25x total**
- Finding #333 validated single-adapter promotion at scale=5 only — the N-adapter case was never tested
- Parity base loss 0.58 is near-trivial (the pattern is learnable from positional structure alone); any perturbation of this magnitude destroys the near-optimal encoding

This matches TIES Merging (Yadav et al., 2306.01708): sign conflicts and magnitude interference compound with adapter count. At N=10 with per-adapter amplification, TIES predicts catastrophic interference for domains that are already near-optimal.

### Conjecture 2 refuted: enrichment monotonicity is circular

Conjecture 2 assumed "additional signal always lowers loss" but this is false when the base is near-optimal. The proof sketch assumed A2 (base has headroom) while trying to prove A2 — circular reasoning. This is the same A2 violation that killed Finding #352 (KL distillation). The requirement is universal: **only enrich domains where base_loss >> SFT_loss**.

---

## Confirming Evidence (Literature)

- **Task Arithmetic** (Ilharco et al., arXiv:2212.04089): adding task vectors in weight space is geometrically valid when vectors share a common base. Cross-domain M2P is the continuous-space analog: generating new adapter parameters conditioned on cross-domain signal preserves the shared null-space geometry. 8/10 useful pairs confirms the geometry is broadly compatible.

- **LoRAHub** (Huang et al., arXiv:2307.13269): demonstrated that composing LoRA adapters from *diverse* tasks generalizes to unseen tasks. Their success with unrelated tasks (code→QA) mirrors our 8/10 result. Their failure mode (distributional mismatch in the hidden-state space) is what Option A avoids by generating a native B matrix.

- **DARE** (Yu et al., arXiv:2311.03099): weight sparsification before merging reduces interference. The dissolve-recrystallize failure is the complementary case: merging WITHOUT sparsification amplifies interference 25x. DARE's approach (drop weights below threshold before promotion) would reduce effective amplification from 25x to ~2-3x.

- **Grassmannian capacity bounds** (Theorem 1, this experiment): max|cos| = 1.02e-8 across 1050 slot pairs. Slot orthogonality holds regardless of N. The failure is not in the slot geometry — it is entirely in the scale amplification of the B-matrix weights.

---

## Contradicting Evidence

- **AdaLoRA** (Zhang et al., arXiv:2303.10512) assigns adaptive rank per adapter based on singular value importance. If ranks were adaptive, lower-importance domains (parity, near-optimal) would receive lower effective rank and thus lower amplification during promotion. Fixed-rank promotion is a known risk when domain complexity varies.

- **Born Again Networks** (Furlanello et al., arXiv:1805.04770): same-capacity iterative distillation CAN improve over the original, which contradicts the idea that near-optimal domains cannot benefit from dissolve. However, BAN uses iterative retraining with fresh initialization — not weight-space promotion. The mechanism is fundamentally different.

---

## Alternative Approaches (with paper references)

1. **N-normalized promotion scale**: EFFECTIVE_SCALE = PROMOTE_SCALE / sqrt(N). For N=10: 5/3.16 = 1.58x per adapter, 15.8x total. Better but still 4x beyond single-adapter validation. Requires new experiment to find safe N.
   - *Motivation*: DARE (2311.03099) Section 4.2 derives N-dependent scaling for safe merging.

2. **Domain headroom guard before dissolve**: skip promotion when base_loss < SFT_loss + epsilon (e.g., 0.1). This is already partially described in PAPER.md but not implemented in code. 
   - *Motivation*: directly addresses Conjecture 2's A2 violation.

3. **Sparse promotion (DARE-style)**: before dissolving N adapters, zero weights below a threshold to reduce effective amplification.
   - *Motivation*: DARE (2311.03099) shows 91.7% task accuracy preserved with 90% weights dropped before merging.

4. **Bidirectional transfer measurement**: current experiment measures only 10/20 directed pairs. Bidirectional pairs may show asymmetry in which direction benefits — relevant for routing decisions.
   - *Motivation*: LoRAHub (2307.13269) section 3.3 shows task distance is asymmetric.

5. **Fix routing bottleneck first** (Finding #351): 36.6% routing accuracy means cross-domain transfer quality (91.5% per-adapter) is masked by wrong adapter assignment. The composition pipeline loses ~60% of the individual adapter gain before this is fixed.

---

## Implications for Next Experiments

1. **Priority: routing is the bottleneck** (Finding #351: 36.6% routing accuracy). Cross-domain quality (91.5% median) is irrelevant until routing accuracy exceeds 80%. The Analyst recommends no further cross-domain graph experiments until routing is solved.

2. **Dissolve is valid but requires scale normalization**: the mechanism works (8/10 pairs, repeat +36.3%), but the scale formula for multi-adapter dissolve is unvalidated. Any follow-up must derive a safe N-dependent scale from DARE's analysis before running.

3. **Option A is the confirmed architecture for cross-domain M2P**: student generates its own B matrix from teacher hidden states. This should be the default for all future M2P cross-domain work.

4. **Near-optimal domain detection is necessary infrastructure**: parity's base loss (0.58) was identified post-hoc. A domain headroom metric (base_loss/SFT_loss ratio) should be computed before any dissolve operation and used as a gate.

5. **Directional asymmetry remains an open question**: we measured 10 unidirectional pairs (A→B). The 10 reverse pairs (B→A) are unmeasured. If routing is fixed, the reverse pairs should be tested to determine if source domain selection requires directional routing logic.

---

## Recommended Follow-Up

**No new cross-domain experiments until routing is fixed (Finding #351).**

If routing accuracy reaches 80%+, the recommended next step is:
- **Safe multi-adapter dissolve**: derive N-dependent scale from DARE (2311.03099) Theorem 1, then validate on toy scale at N=2, 5, 10 before applying to the 5-domain real model.
- **Bidirectional transfer**: measure the 10 reverse pairs to determine if Option A quality is directionally symmetric.

Both require routing working first. Current priority remains the routing problem.

---

## New References Added

- arXiv:2212.04089 — Task Arithmetic (Ilharco et al., 2022): weight-space task vector arithmetic
- arXiv:2307.13269 — LoRAHub (Huang et al., 2023): composing LoRA adapters for cross-task generalization  
- arXiv:2311.03099 — DARE (Yu et al., 2023): weight sparsification before merging
- arXiv:2306.01708 — TIES Merging (Yadav et al., 2023): sign conflict resolution in model merging
- arXiv:2303.10512 — AdaLoRA (Zhang et al., 2023): adaptive rank allocation for LoRA
