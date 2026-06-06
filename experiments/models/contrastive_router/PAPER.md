# Contrastive Routing Keys: Experiment Report

## Status: KILLED

All three kill thresholds from MATH.md Section 10 exceeded. The contrastive
routing key mechanism fails at micro scale (d=64, a-m vs n-z character names).

---

## 1. Hypothesis

InfoNCE-trained contrastive routing keys K_i can replace the softmax router
calibration in capsule MoE composition, achieving >85% domain routing accuracy
with fewer samples (~50/domain) and steps (~50) than the 100-step softmax
calibration protocol.

## 2. Setup

- **Architecture:** ContrastiveRouterGPT — CapsuleMoEGPT with contrastive
  keys (K_i in R^{d x d_key}) replacing the linear router per group.
- **Config:** d=64, G=4/domain, D=2 domains, N=8 composed groups, d_key=8,
  top_k=4, 219K total params (8% overhead vs capsule_moe).
- **Protocol:** Pretrain shared base (300 steps all data) → fine-tune capsule
  groups per domain (300 steps each, attention frozen) → compose → calibrate
  routing keys with InfoNCE on ~128 labeled hidden states/domain, 50 steps.
- **Baselines:** Joint training (gold), softmax router calibration (100 steps),
  linear probe (same data/steps as contrastive).
- **Seeds:** 42, 123, 7 (3 seeds).

## 3. Results

### 3.1 Composition Quality (val loss, mean ± std over 3 seeds)

| Method              | a-m    | n-z    | avg    | vs joint  |
|---------------------|--------|--------|--------|-----------|
| Joint training      | 0.531  | 0.509  | 0.520  | baseline  |
| Softmax calibrated  | 0.530  | 0.511  | 0.521  | +0.2%     |
| **Contrastive keys**| 1.262  | 1.247  | 1.255  | **+141%** |

Contrastive keys catastrophically degrade composition quality. The composed
model with contrastive routing performs 2.4x worse than joint training on
average. The softmax router baseline remains within 0.2% of joint.

### 3.2 Routing Accuracy (held-out, mean over 3 seeds)

| Method              | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Avg   |
|---------------------|---------|---------|---------|---------|-------|
| Contrastive keys    | 53.3%   | 52.4%   | 55.1%   | 52.3%   | 53.3% |
| Linear probe        | 55.4%   | 58.5%   | 63.5%   | 61.8%   | 59.8% |
| Random baseline     |         |         |         |         | 50.0% |

Contrastive keys achieve 53.3% routing accuracy — barely above random (50%)
and well below the 70% kill threshold. The linear probe outperforms contrastive
keys (59.8% vs 53.3%), meaning the contrastive loss adds nothing over a
simple classifier.

### 3.3 Tau Sweep (seed=42, held-out accuracy)

| tau  | Train acc | Held-out acc | Val loss |
|------|-----------|--------------|----------|
| 0.05 | 55.0%     | 53.1%        | 1.21     |
| 0.10 | 55.7%     | 53.2%        | 2.45     |
| 0.50 | 57.3%     | 53.6%        | 1.59     |
| 1.00 | 58.6%     | 54.0%        | 1.83     |

No temperature setting helps. Higher tau (softer distribution) gives marginally
better train accuracy but no held-out improvement — consistent with overfitting
to noise in the small training sample.

## 4. Kill Threshold Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| Routing accuracy | >85% (kill <70%) | 53.3% | **KILLED** |
| Composition quality | <5% worse (kill >10%) | +141% worse | **KILLED** |
| vs linear probe | Must beat | 53.3% < 59.8% | **KILLED** |
| Sample/step efficiency | <100 each | 128 samples, 50 steps | OK |

## 5. Root Cause Analysis

### 5.1 Domain Indistinguishability at Micro Scale

The fundamental failure mode is **Assumption 6 from MATH.md**: at d=64 with
a-m vs n-z character-level name tokenization, the hidden state representations
are NOT domain-discriminative.

Evidence:
- Even the linear probe only achieves 60% accuracy (vs 50% random).
- The discriminative signal is weak because both domains share the same
  character vocabulary (a-z) and similar name length/structure distributions.
- The domain split (first character a-m vs n-z) is a coarse lexicographic
  criterion that doesn't create strong distributional differences in the
  character-level hidden state space.

### 5.2 Why the Softmax Router Works

The softmax router calibration (+0.2% vs joint) works because it doesn't need
domain discrimination. It optimizes reconstruction loss directly — routing
tokens to whichever groups minimize prediction error, regardless of domain
identity. This is an implicit, task-aligned routing signal.

The contrastive keys attempt explicit domain discrimination via the InfoNCE
objective, but at micro scale there's no domain signal to learn. The keys
converge to near-random projections.

### 5.3 Score Magnitude Issues

With random keys at initialization, routing scores ||x @ K_i||^2 are O(1-10)
across all groups. After 50 steps of InfoNCE training, score gaps between
domains remain tiny (<0.5), meaning the softmax over domain scores is nearly
uniform — routing is essentially random.

## 6. What This Means for VISION.md

The contrastive routing key mechanism is **not validated at micro scale**. This
does NOT necessarily kill the approach at macro scale, because:

1. **Stronger domain signal at scale.** With larger models (d=256+) and
   meaningful domains (Python vs JavaScript code, not a-m vs n-z names), hidden
   states will carry much stronger domain-discriminative information.

2. **More parameters to learn.** At d_key=8 with d=64, the key matrices are
   extremely constrained. At d=512 with d_key=32, the capacity ratio improves.

3. **Richer token vocabularies.** BPE tokenization creates domain-specific
   tokens (Python keywords vs JavaScript keywords) that are trivially separable.

However, the micro experiment reveals a real risk: **if domains share
representation structure (as they do for similar tasks), contrastive keys may
underperform reconstruction-based routing even at scale.**

## 7. Recommendations

1. **Do NOT proceed with contrastive keys at micro scale.** The mechanism
   cannot work when domains are indistinguishable in hidden space.

2. **Re-evaluate at macro scale with distinct domains.** Test on
   Python vs JavaScript LoRA adapters where domain signal is strong.

3. **Consider hybrid routing.** Use contrastive keys for coarse domain
   selection (where domains are distinguishable) and softmax routing for
   fine-grained group selection within a domain.

4. **Skip to VISION.md Exp 2 (sparse routing)** which doesn't depend on
   domain discrimination — it tests whether top-1 group selection matches
   top-2 quality, using the existing softmax router.

## 8. Reproducibility

```bash
# Run full experiment (3 seeds, ~5 min)
python -m micro.models.contrastive_router.run_experiment

# Run unit tests
python -m micro.models.contrastive_router.test_contrastive_router
```

All code at `micro/models/contrastive_router/`. Parent model: `capsule_moe`.
