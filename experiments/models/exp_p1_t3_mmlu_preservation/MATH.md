# MATH.md: T3.2 — MMLU Preserved Under N=5 Composition

## Context

T3.1 (exp_p1_t3_pairwise_interference, KILLED) showed that **simultaneous activation** of N=5
adapters causes catastrophic degradation: math 82→8%, code 66→8%. The impossibility structure
was O(N-1) additive noise in activation space (Finding #425). **Routing** is structurally required.

T3.2 verifies the complementary claim: when routing selects a **single adapter**, does MMLU
preservation hold? And does Gemma 4's QK-normalization make this scale-invariant (i.e., no manual
calibration needed)?

The adapters are: rank=6, scale=6, q_proj only, all 42 layers (math/code/medical/legal/finance).

---

## Theorem 1: QK-Normalization Bounds Query Perturbation (Scale Invariance)

**Setup.** Let x ∈ ℝ^{d_model} be a hidden state. Gemma 4 applies per-head RMSNorm (QK-norm)
to queries after q_proj (ARCHITECTURE_P1.md §4.1):

```
Q_raw(s) = W_q x + s · B A x          ∈ ℝ^{n_heads × h_q}
Q_norm(s) = RMSNorm_head(Q_raw(s))    per-head normalization
```

where A ∈ ℝ^{d_model × r}, B ∈ ℝ^{r × h_q·n_heads}, s > 0 is the LoRA scale.

**Theorem 1.** For any scale s > 0 and hidden state x ≠ 0:

```
||Q_norm(s)||_RMS = sqrt(h_q)    (independent of s)
```

**Proof.**
```
Q_norm(s)_i = Q_raw(s)_i / sqrt( mean_j Q_raw(s)_j² )
```
Therefore:
```
mean_i Q_norm(s)_i² = mean_i Q_raw(s)_i² / mean_j Q_raw(s)_j² = 1
⟹ ||Q_norm(s)||_RMS = sqrt(h_q)  ∀ s > 0
```
**QED.**

**Corollary (Scale Invariance of Attention).** The attention weight:
```
A_{ij} = softmax( Q_norm(s)_i · K_norm_j / sqrt(h_q) )  ∈ [0,1]
```
Since ||Q_norm(s)||₂ = sqrt(h_q) · sqrt(h_q) = h_q for all s, and ||K_norm||₂ = h_q similarly,
the dot product Q_norm · K_norm ∈ [-h_q, h_q] regardless of s. The softmax output is always
in [0,1]^{n_tokens}.

**Consequence:** Changing s only **rotates** Q_norm, never changes its norm. There is no
scale catastrophe on q_proj adapters on Gemma 4 — the query magnitude is clamped by QK-norm.

This contrasts with Qwen3-4B (no QK-norm): Q_raw(s) = W_q x + s·BAx → ∞ as s → ∞, causing
the Davis-Kahan bound to become vacuous (Finding #330: -42pp at scale=20 on Qwen3).

---

## Theorem 2: MMLU Degradation Bound for OOD Inputs

**Setup.** Let T_adapt be the adapter's training distribution and T_MMLU ⊥ T_adapt be OOD
evaluation queries (neutral MMLU: geography, philosophy, world_religions, sociology, world_history).

**Theorem 2.** For a q_proj LoRA with rank r trained on T_adapt, evaluated on x ~ T_MMLU:

```
E_{x ~ T_MMLU}[ ||ΔQ_raw||₂ ] ≤ s · σ_max(B) · ||Ax||₂
```

For OOD input x: by JL-Lemma, if A learned domain-specific projections from T_adapt, then
for x ∼ T_MMLU with ||x|| ~ 1:
```
E[ ||Ax||₂ ] ≈ sqrt(r/d_model)  (concentration of random-like projections)
```

For rank=6, d_model=2560:
```
sqrt(r/d_model) = sqrt(6/2560) ≈ 0.048
```

The adapter fires at ~5% amplitude on OOD queries. Since Q_norm clamps magnitude regardless,
the directional perturbation is small, and MMLU degradation is expected to be near-zero.

**Prediction.**

| Adapter | Training Domain | OOD from Neutral MMLU? | Expected Δ |
|---------|----------------|------------------------|-----------|
| Math    | GSM8K (arithmetic word problems) | YES | ≤ 1pp |
| Code    | HumanEval (Python code) | YES | ≤ 1pp |
| Medical | MedMCQA (medical MCQ) | YES | ≤ 2pp |
| Legal   | MMLU legal subjects | PARTIAL (same format) | ≤ 2pp |
| Finance | MMLU economics subjects | PARTIAL (same format) | ≤ 2pp |

Legal and finance adapters were trained on MMLU-format MCQ, so their A matrices may have
learned MCQ-style representations that activate weakly on neutral MMLU — hence 2pp allowance.

---

## Kill Criteria Derivation

**K1053** — MMLU(base + 5 adapters via routing) ≥ MMLU(base) − 1pp

Routing (PLE-M2P) selects exactly ONE adapter per query. If each adapter individually
preserves MMLU (K1055), then routing trivially preserves MMLU — the composed system
never activates more than one adapter simultaneously.

K1053 is therefore a **consequence of K1055**: it reduces to "the worst single adapter
degrades MMLU by at most 1pp."

Prediction: PASS if K1055 passes.

**K1054** — MMLU preserved with V-norm (QK-norm on q_proj) at any scale (no manual calibration)

By Theorem 1, ||Q_norm(s)||_RMS = sqrt(h_q) for ALL s > 0. Therefore MMLU degradation
at scale=12 and scale=18 should not exceed MMLU degradation at scale=6.

Prediction: PASS. At most ±2pp variance between scales (noise from n=25 sampling).

**K1055** — Each adapter individually: MMLU(base + adapter_i) ≥ MMLU(base) − 1pp

By Theorem 2: OOD adapter activation ~5% amplitude. QK-norm bounds query magnitude.
Base model knowledge lives in W_q, which is unchanged by LoRA addition.

Prediction: PASS for math/code. PASS with margin for medical/legal/finance.

---

## Evaluation Design

**MMLU subjects (neutral — no overlap with any adapter training domain):**
- high_school_geography (199 questions)
- world_religions (169 questions)
- philosophy (311 questions)
- high_school_world_history (237 questions)
- sociology (201 questions)

n=25 per subject = 125 total questions (same seed=42 throughout).

**Phase 1:** Base model MMLU on 125 neutral questions.
**Phase 2:** Each of 5 adapters at scale=6 on same 125 questions (K1055).
**Phase 3:** Math adapter at scale=6, 12, 18 on 75 questions (3 subjects × 25) (K1054).
**Phase 4:** Kill criteria computation.

---

## References

- Davis, C. & Kahan, W.M. (1970). "The rotation of eigenvectors by a perturbation." SIAM 7(1).
- ARCHITECTURE_P1.md §4.1: Gemma 4 QK-normalization per head.
- Finding #425: Routing is structurally required; simultaneous activation = O(N) noise.
- Finding #330: Scale=5 safe zone on Qwen3-4B; QK-norm extends to all scales on Gemma 4.
- exp_p1_t0_vnorm_scale_safety PAPER.md: V-norm/QK-norm theorem (Theorem 1) verified.
- Hu, E. et al. (2022). "LoRA." arXiv:2106.09685.
