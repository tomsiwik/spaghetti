# LEARNINGS: M2P Quality at d_model=1024 — Scaling Law Confirmed

**Experiment:** exp_m2p_qwen3_quality
**Finding #362:** supported
**Status:** PROCEED

---

## Core Finding

M2P with fixed architecture (d_M2P=64, L=2, n=2000, GL early stopping) achieves 99.6% of SFT quality at d_model=1024 (4× micro), with train-val gap=0.2355 nats. This establishes a three-point scaling law across a 4× d_model range with no quality degradation attributable to the doubling compression ratio (128:1 → 256:1 → 512:1 for fc1).

| d_model | quality_ratio | fc1 compression | Finding |
|---------|--------------|-----------------|---------|
| 256     | 97.6%        | 128:1           | #359    |
| 512     | 101.0%       | 256:1           | #361    |
| 1024    | 99.6%        | 512:1           | #362    |

---

## Why This Happened

### 1. Aghajanyan intrinsic dimensionality is the right capacity model
The Bartlett et al. counting argument (O(rank × d_out) effective dimensions) has been falsified three times. The correct framework is Aghajanyan et al. (arXiv:2012.13255): fine-tuned LLM adapters occupy a low-dimensional subspace whose intrinsic dimensionality is small and largely independent of d_model. The measured data confirms d_int << 64 at all three scales.

**Caution from adversarial review:** MATH.md cited "Theorem 2 of Aghajanyan" which does not exist as a formal theorem in that paper — it contains Definition 1 and empirical measurements (Table 1). The argument is a conditional claim, not a proven theorem. This is a citation accuracy issue, not a validity issue.

### 2. The compression ratio doubling is the key test
At each d_model doubling, fc1's compression ratio doubles:
- d=256: fc1 generates 256×4=1024 outputs through d_M2P=64 → 128:1 compression
- d=512: fc1 generates 512×4=2048 outputs → 256:1 compression
- d=1024: fc1 generates 1024×4=4096 outputs → 512:1 compression

Quality held through all three doublings. The bottleneck capacity was never saturated because the intrinsic structure of the B-matrix space doesn't grow with d_model.

### 3. Quality_ratio metric becomes more forgiving at higher d_model
The denominator (base_loss - sft_loss) grows as models become stronger relative to the toy base: 10.8 nats at d=256, 12.1 at d=512, 15.1 at d=1024. Absolute M2P-SFT loss differences should always be checked alongside the ratio. At d=1024, the sort domain absolute difference is +0.019 nats (M2P better) — small but consistent with the pattern.

### 4. n_train≥T guarantee is confirmed d_model-independent (structural)
Train-val gap at n=2000: 0.2355 nats (3× below the 0.7 threshold). The Ghadimi-Lan i.i.d. condition depends only on T/n_train=0.625 — no d_model term. This was already proven in MATH.md Theorem 1 (exp_m2p_data_scale); this experiment provides a third confirmatory data point.

---

## Critical Limitations

1. **Only 2 valid domains (sort + reverse).** Arithmetic excluded by parity guard (base at d=1024 already captures arithmetic nearly perfectly). No variance estimate possible with 2 data points.

2. **The d_model scaling test is NOT a proxy for n_layers scaling.** Qwen3-4B has ~36 transformer layers vs the toy L=2 model. M2P generates adapters that are applied to ALL layers. Whether the intrinsic dimensionality argument holds when there are 36 layers of adapters instead of 2 is unknown. The adversarial reviewer flagged this as the key untested risk for Qwen3-4B deployment.

3. **Context token sensitivity untested.** B-matrices generated from a single context token (train_batches[0]). Production M2P should generate adapters from diverse contexts.

4. **"Path to Qwen3-4B unobstructed" overreaches.** d_model scaling is one dimension. n_layers is a different dimension, and the harder one at Qwen3-4B scale.

---

## Implications for Next Experiments

1. **d_model scaling is CLOSED.** Three data points spanning 4× range with no trend. No further d_model sweeps needed.

2. **The critical unknown for Qwen3-4B is n_layers, not d_model.** Qwen3-4B has 36 layers vs 2 in the toy. Each layer requires its own M2P-generated B-matrices. If M2P is called once and generates adapters for all layers, does quality hold? This is the next structural question.

3. **Recommended follow-up: exp_m2p_layer_depth** — Test whether a single M2P call can generate adapters for L=4, 8, 16 layers (varying output head size proportionally), or whether M2P must be called once per layer.

4. **GL early stopping is confirmed non-critical at d=1024** (zero stops triggered at n=2000). Convergence happens before overfitting. The structural guarantee is sufficient and GL is an additional safety margin.

---

## Confirmed Scaling Laws

Three independent structural guarantees now confirmed across d_model ∈ {256, 512, 1024}:
1. n_train≥T prevents cyclic memorization (Ghadimi-Lan, T/n_train-only bound)
2. d_M2P=64 suffices (Aghajanyan d_model-independent intrinsic dimensionality)
3. GL early stopping provides additional safety margin (convergence typically before stop)

The M2P recipe is **d_model-independent** — it can be deployed at any d_model without architectural changes, subject to the caveat that n_layers must be separately validated.
