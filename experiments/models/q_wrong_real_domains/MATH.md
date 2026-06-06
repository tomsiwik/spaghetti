# MATH.md — Q_wrong Measurement: Cross-Domain Interference from M2P Adapter

## Experiment Type: Guided Exploration (Epsilon-Map Calibration, Blind Spot #2)

## Background

The epsilon-map models adapter interference with two quantities:
- **Q_right**: quality when the correct adapter is applied to its trained domain
- **Q_wrong**: quality when the wrong adapter is applied to an out-of-distribution domain

Prior experiments measured Q_right = 1.433 (point est.) for the GSM8K M2P adapter on GSM8K (Finding #378/379). Q_wrong has never been measured on Qwen3-0.6B with real-LLM M2P weights. This experiment fills that blind spot.

## Theorem 1: M2P Interference Bound

**Setup:**  
Let M : ℝ^{L×d} → {B_q^l, B_v^l}_{l=1}^L be the trained M2P hypernetwork.  
Let A_q^l, A_v^l ∈ ℝ^{d×r} be the fixed Grassmannian A-matrices (trained on D_math = GSM8K).  
Let x ∈ D_eval be a prompt from evaluation domain D_eval ≠ D_train.

The effective adapted weight for layer l, module q is:
  W_eff^l = W_base + A_q^l · M_q^l(h(x)) · scale

where h(x) ∈ ℝ^{L×d} are per-layer mean-pooled hidden states extracted via the base LLM.

**Theorem (Interference Bound):**  
Let D_eval be structurally orthogonal to D_train in the sense that:
  E_{x ~ D_eval}[h(x)] ≠ E_{x ~ D_train}[h(x)]

Then the M2P output M(h(x)) for x ∈ D_eval is determined by the M2P network's generalization outside its training distribution. Two structural outcomes are possible:

**Case A (OOD collapse):** M2P maps wrong-domain hidden states to near-zero B-matrices
  → ΔW ≈ 0, Q_wrong ≈ 0 (neutral, base model dominates)

**Case B (spurious transfer):** M2P maps wrong-domain hidden states to non-zero B-matrices
  → Q_wrong ≠ 0 in some direction (positive or negative)

**Distinguishing prediction** (from the structure of the M2P architecture):
The M2P encoder is a 2-layer MLP trained exclusively on GSM8K representations. For structurally different domains (sort, reverse, parity), the encoder's output z = Enc(mean(h(x))) falls in a region the B-heads were never trained to map usefully. Three sub-cases:
  1. If z ≈ 0: B-heads produce small outputs → neutral adapter (Q_wrong ≈ 0)
  2. If z is a generic non-zero vector: B-heads produce non-specific B-matrices → random perturbation
  3. If z activates shared structure: potentially non-trivial Q_wrong

**Prediction P1** (from concentration): For n=50 examples with simple synthetic tasks (sort/reverse/parity), where the base model accuracy ≥ 60%, the adapter perturbation is unlikely to dominate the strong base signal. |Q_wrong| < 1.0 expected.

**Prediction P2** (from finding #368): The toy-scale result showed cross-domain transfer = 97.78% (sort→cipher). If that pattern holds at LLM scale, we expect Q_wrong > 0 (helpful, not harmful). But finding #371 killed this: cipher was structurally similar to sort at toy scale. Real LLM domains may differ.

**This experiment is EMPIRICAL**: the math cannot predict the sign of Q_wrong without knowing M2P's OOD generalization behavior. The measurement itself is the finding.

## Theorem 2: Q_wrong Metric Definition

**Definition:** For domain pair (D_adapter, D_eval) where D_adapter ≠ D_eval:

  Q_wrong = (acc_adapted(D_eval) - acc_base(D_eval)) / max(acc_base(D_eval), 0.01)

where:
- acc_base(D_eval) = accuracy of base model on D_eval with NO adapter
- acc_adapted(D_eval) = accuracy when M2P (trained on D_adapter) is applied, conditioned on D_eval inputs

**Interpretation:**
- Q_wrong > 0: adapter is transfer-positive (wrong adapter helps)
- Q_wrong = 0: adapter is neutral (wrong adapter doesn't change accuracy)
- Q_wrong < 0: adapter is harmful (wrong adapter hurts accuracy)

**Routing urgency implication (Finding #354):**
- If Q_wrong ≈ 0: routing importance is LOW (any adapter is fine)
- If Q_wrong < -0.2: routing importance is HIGH (must use correct adapter)
- If Q_wrong > 0.2: routing is beneficial even when wrong (system is robust)

## Kill Criteria

K944 (UNCONDITIONAL): Q_wrong measured for at least 3 domain pairs.
  - Domain pairs: (GSM8K adapter → sort), (GSM8K adapter → reverse), (GSM8K adapter → parity)
  - PASS if all 3 measured without error
  - KILL if fewer than 3 complete successfully

## Experimental Design

**Domains tested:**
1. **Sort words** (50 examples, 5 smoke): "Sort these words alphabetically: W1, W2, W3\nAnswer: W_sorted"
2. **Reverse words** (50 examples, 5 smoke): "Reverse the order of these words: W1 W2 W3\nAnswer: W3 W2 W1"
3. **Count evens** (50 examples, 5 smoke): "How many even numbers are in: [N1, N2, N3]\nAnswer: K"

**Adapter:** v4 M2P weights (micro/models/m2p_qwen06b_gsm8k_v4/m2p_weights.npz)
**A-matrices:** v2 SFT A-matrices (micro/models/m2p_qwen06b_gsm8k_v2/lora_a_matrices.npz)

**Evaluation:**
- Base accuracy: mlx_generate with no adapter (zero B-matrices)
- Adapted accuracy: M2P conditioned on each prompt → B-matrices injected → mlx_generate
- n=50 per domain, few-shot prefix (2 examples per domain)

## Citations

- Hu et al. (2106.09685) — LoRA: intrinsic rank of fine-tuning updates
- Ha et al. (1609.09106) — HyperNetworks: weights as function of context
- Aghajanyan et al. (2012.13255) — Intrinsic dimensionality of fine-tuning
- Finding #378/379: Q_right = 1.433 point est. (GSM8K M2P v4/n500)
- Finding #384: Per-user M2P adapters, behavioral differentiation confirmed
- Finding #385 (killed): Square GQA matrices near-degenerate under 4-bit quantization

## Prediction-vs-Measurement Table (to be filled in PAPER.md)

| Metric | Prediction | Measurement |
|--------|-----------|-------------|
| Q_wrong (sort) | \|Q\| < 1.0 | TBD |
| Q_wrong (reverse) | \|Q\| < 1.0 | TBD |
| Q_wrong (count_even) | \|Q\| < 1.0 | TBD |
| K944 (3 pairs measured) | PASS | TBD |
