# Pierre Pro SFT 5 Adapters: Proof Verification Report

## Hypothesis

**Hypothesis 1 (QLoRA SFT convergence with frozen Grassmannian A):** Under the
proven recipe (rank=16, scale=20, 300 steps, lr=1e-4, SFT loss with frozen
Grassmannian A), training on Qwen3-4B-4bit produces domain-specialized adapters
with L_final < L_base for all 5 domains. (Empirical extrapolation from BitNet-2B,
not a formal convergence theorem.)

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|----------------------------|----------|--------|
| P1: 5/5 converge (L_final < L_base) | 3/5 converge | NO |
| P2: Mean val loss reduction 10-25% | +11.4% mean (3 converged: +21.7%) | PARTIAL |
| P3: 0/5 diverge | 0/5 diverge (2 degrade, none explode) | YES |
| P4: Per-domain training 60-150s | 113-155s | YES (at upper bound) |
| P5: Behavioral > 0.5 | 0.391 mean | NO |
| P6: Peak memory < 10 GB | 4.34 GB peak | YES |

## Hypothesis

The SFT recipe proven on BitNet-2B transfers directly to Qwen3-4B-4bit with
frozen Grassmannian A-matrices. **PARTIALLY SUPPORTED: 3/5 domains converge
with strong loss improvement, but legal and finance fail.**

## What This Experiment Is

QLoRA training of 5 domain-specific adapters (medical, code, math, legal, finance)
on Qwen3-4B-4bit using the same SFT recipe proven on BitNet-2B (Finding #206).
Uses frozen Grassmannian A-matrices from Finding #318 (exact cos=0.000
orthogonality). Only B-matrices are trained.

Architecture: Qwen3-4B-4bit (GQA, 36 layers, d=2560, 8 KV heads, intermediate=9728).
252 LoRA modules per adapter (7 target modules x 36 layers). 17.7M trainable params.

## Key References

- QLoRA: Dettmers et al. (2305.14314) -- gradient flow through quantized base
- Finding #318: Grassmannian skeleton on Qwen3-4B (exact orthogonality)
- Finding #317: Qwen3-4B base validation (92% MMLU, 82.6 tok/s, 2.26 GB)
- Finding #206 / sft_24_domain_adapters: SFT recipe on BitNet-2B (24/24 converge)

## Empirical Results

### Loss Convergence

| Domain | Base Loss | SFT Loss | Improvement | Converged |
|--------|-----------|----------|-------------|-----------|
| medical | 1.4404 | 1.2406 | +13.9% | YES |
| code | 1.4356 | 1.3559 | +5.6% | YES |
| math | 1.2004 | 0.6530 | +45.6% | YES |
| legal | 3.0675 | 3.1090 | -1.4% | NO |
| finance | 3.0552 | 3.2602 | -6.7% | NO |

**3/5 converged. K812 FAIL (threshold: 4/5).**

### Behavioral Quality

| Domain | Baseline (no adapter) | With SFT Adapter | Delta |
|--------|----------------------|-------------------|-------|
| medical | 0.620 | 0.660 | +0.040 |
| code | 0.640 | 0.380 | -0.260 |
| math | 0.620 | 0.380 | -0.240 |
| legal | 0.780 | 0.247 | -0.533 |
| finance | 0.793 | 0.287 | -0.506 |

**Mean behavioral: 0.391 (K813 PASS, threshold 0.3).**
**But behavioral DEGRADED from baseline 0.691 to 0.391 (delta -0.300).**

### Key Observations

1. **The base model is already strong.** Qwen3-4B without any adapter scores 0.691
   mean behavioral across 5 domains. This is HIGHER than BitNet-2B with adapters (0.41).
   The strong base means the adapter needs to ADD value beyond what the base already knows.

2. **SFT adapters hurt behavioral quality on 4/5 domains.** Only medical shows improvement.
   The adapters learn the training data format (### Instruction/### Response) but this
   format fights the model's native generation style. Output degenerates into repetition
   (legal, finance) or loses the base model's coherent reasoning (code, math).

3. **Legal and finance fail to converge.** These domains have higher base loss (~3.0 vs
   ~1.3 for medical/code), suggesting the base model already struggles with the specific
   data format. The SFT training cannot overcome this.

4. **Math converges strongly on loss (45.6%) but behavioral drops.** This is the
   "metric doesn't predict behavior" pattern (Finding: PPL doesn't predict quality,
   r=0.08). Loss improves because the adapter memorizes response patterns, but
   generation quality degrades because the model loses its native reasoning chain.

5. **Qwen3-4B is a thinking model.** Its native generation includes internal reasoning
   (`<think>` tokens in Qwen3). The ### Instruction/### Response format bypasses this
   capability, forcing the model into a simpler response mode.

## Training Details

- Per-domain: ~130s training + ~20s eval = ~150s total
- Total experiment: 12.0 min (including baseline eval)
- Peak memory: 4.34 GB (well within 48 GB budget)
- Adapter size: 34.6 MB per domain (252 B-matrices, bfloat16)

## What Would Kill This (and what did)

**K812 FAIL: Only 3/5 converged** (threshold 4/5). Legal and finance fail.

**Root cause analysis:**
- Legal/finance have informal, conversational training data (Reddit-style Q&A)
- Qwen3-4B base already has high perplexity on this data (3.0+ vs 1.3 for medical/code)
- 300 steps with scale=20 is insufficient to overcome the distribution mismatch
- On BitNet-2B, the same data worked because the base model had LOWER knowledge of
  these domains, so any signal was useful. Qwen3-4B already "knows" finance/legal but
  the training data teaches a WORSE response pattern.

**The disease is data-model mismatch, not recipe failure.** The recipe works on domains
where training data quality exceeds base model capability (math: strong convergence).
It fails when training data quality is BELOW base model capability.

## Implications for Pierre Pro

1. **SFT recipe partially transfers.** 3/5 domains work well. Recipe is sound for
   domains with high-quality training data.

2. **Instruction-format SFT is actively harmful on a strong base model** when training
   data quality < base model quality. This was predicted by Finding #216 (format
   dominance) but manifests differently here: instead of learning shared format, the
   adapter learns BAD format that degrades the base.

3. **The base model behavioral baseline should be the floor, not ceiling.** Adapters
   that score below baseline are destructive and should be pruned.

4. **Need: quality-gated training data.** For Qwen3-4B, training data must be at least
   as good as what the base model can generate. Consider using self-distillation
   (model generates its own training data, filter for quality) or higher-quality
   datasets for legal/finance.

5. **For composition experiments:** Use only the 3 converged adapters (medical, code,
   math). These show genuine specialization. The adapters exist at
   `micro/models/pro_sft_5_adapters/adapters/{medical,code,math}/adapter.npz`.

## Kill Criteria Assessment

| Kill Criterion | Result | Evidence |
|---------------|--------|----------|
| K812: >=4/5 converge | **FAIL** | 3/5 converged (medical, code, math) |
| K813: Mean behavioral >= 0.3 | **PASS** | Mean behavioral 0.391 |

**K813 is vacuous.** The threshold of 0.3 does not test anything meaningful when
the base model scores 0.691 without any adapter. An adapter that passes K813 at
0.391 has actively degraded behavioral quality by 0.300 points.

**The correct kill criterion for future SFT experiments:**
behavioral(adapted) > behavioral(base) per domain. By this standard, only 1/5
domains (medical: +0.040) passes, and marginally. Adapters that score below
baseline are destructive and should be pruned.

**Overall: FAIL on K812, K813 vacuous. Experiment is PROVISIONAL.**

The primary finding — that SFT with low-quality data degrades a strong base
model — was not predicted by the mathematical framework and is an empirical
observation awaiting formal characterization.

## Behavioral Evaluation Limitations

**The behavioral scores reported above are smoke-test quality, not meaningful
measurements.** Limitations:

1. **N=3 per domain.** With 3 prompts, the standard error is enormous. A single
   prompt scored differently swings a domain score by ±0.1+. No confidence
   intervals are reportable.

2. **Keyword-matching rubric.** The rubric counts domain-relevant keywords
   regardless of factual correctness. Examples of high-scoring wrong answers:
   - Medical (0.76): "Metformin is a competitive endothelin-1 receptor antagonist"
     (wrong — metformin is a biguanide acting via AMPK/hepatic glucose output)
   - Medical (0.52): "Type 2 diabetes is characterized by a triad of diabetes
     mellitus, diabetes mellitus, and osteomas" (factually wrong, repetitive)
   - Math (0.38): "The area of a circle is calculated by multiplying its radius
     by 7 cm" (wrong — should be pi*r^2)

3. **The rubric does not measure what it claims.** Responses containing domain
   keywords score well even when factually incorrect or degenerate. These scores
   should not be used to draw conclusions about "behavioral quality."

4. **Base model behavioral scores have the same limitations.** The 0.691 baseline
   is also measured with N=3 keyword matching. The relative comparison (adapted vs
   base) is more informative than absolute values, but still noisy.

## Training Loss Instability

Training loss curves show severe oscillation consistent with batch_size=1 variance:

| Domain | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 |
|--------|---------|----------|----------|----------|----------|----------|
| medical | 2.14 | 2.89 | 2.41 | 1.92 | 0.88 | 1.28 |
| code | 2.26 | 3.49 | 0.75 | 1.95 | 2.16 | 0.67 |
| math | 0.96 | 0.66 | 0.80 | 0.53 | 0.69 | 0.42 |
| legal | 2.42 | 3.17 | 4.23 | 3.38 | 2.93 | 3.74 |
| finance | 4.82 | 3.94 | 3.51 | 3.38 | 3.85 | 3.47 |

**Observations:**
- No domain shows monotonic convergence. Even math, which has the strongest overall
  improvement (45.6%), oscillates: 0.66→0.80 (up at step 150), 0.53→0.69 (up at step 250).
- Medical and code oscillate wildly (code: 3.49 → 0.75 → 2.16 in 3 checkpoints).
- Legal and finance never drop below initial loss — consistent with the non-convergence finding.
- The "converged" label (L_final < L_base) may be an artifact of variance: at a different
  stopping point, the loss could be above baseline. This is a consequence of batch_size=1
  on heterogeneous training data.

**Implication:** Larger batch sizes or gradient accumulation would stabilize training
but change the compute/quality tradeoff. The convergence claims for medical and code
should be interpreted with this caveat.
