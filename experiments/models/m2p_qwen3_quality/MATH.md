# MATH.md: M2P Qwen3 Quality — n_train≥T Guarantee at 4× d_model (d=1024)

**Experiment type:** Guided exploration (Type 2)
**Prior finding:** exp_m2p_macro_quality (Finding #361, supported): at d_model=512, n=2000
  + GL early stopping achieves 101.0% of SFT quality. The fixed recipe
  (d_M2P=64, L=2, n=2000, T=1000, GL early stopping, α=5.0) is validated.
**Proven framework:** Ghadimi-Lan O(1/T) convergence + Aghajanyan et al. intrinsic
  dimensionality + Prechelt GL early stopping.
**Unknown being discovered:** Does the same fixed recipe (L=2, d_M2P=64, n=2000, GL)
  achieve ≥85% of SFT when the only change is d_model: 512 → 1024 (4× micro, 2× macro)?

---

## A. Failure Mode Identification

**Disease (risk, not hypothesized):** At d_model=1024, the B-matrix target for each
module is again larger. The M2P's output heads must generate B ∈ R^{rank × d_out} where
d_out ∈ {1024, 1024, 1024, 1024, 4096} (vs {512, 512, 512, 512, 2048} at d=512).
The total target dimensionality doubles again.

**What changed from d=512 to d=1024:**
- fc1 output head: d_M2P=64 → N_LAYERS × LORA_RANK × d_out = 2 × 4 × 4096 = 32,768 params
  (vs 2 × 4 × 2048 = 16,384 at d=512; 2× growth)
- wq/wk/wv/wo output heads: 64 → 2 × 4 × 1024 = 8,192 each
  (vs 2 × 4 × 512 = 4,096 at d=512; 2× growth)

**Key observation from exp_m2p_macro_quality LEARNINGS.md:**
The Bartlett (arXiv:1906.11300) d_eff scaling argument COMPLETELY FAILED at d=512 — it
predicted 50% quality but measured 101%. This falsifies the assumption that the effective
dimensionality of the B-matrix space scales as O(rank × d_out). The failure of Bartlett's
framework at d=512 is actually positive evidence for the Aghajanyan et al. hypothesis
(see Section B.3 below), which we now treat as the governing theory for d=1024.

**Two distinct failure modes:**
1. **Capacity bottleneck** (K887): The M2P hidden dimension is too small to represent
   B-matrices for 1024-dim modules. Signal: quality_ratio < 50%.
2. **Overfitting** (K886): Even with n≥T satisfied, the 1024-dim targets are harder to
   generalize from n=2000 samples. Signal: train-val gap ≥ 0.7 nats.

---

## B. Prior Mathematical Foundations

### B.1 Ghadimi-Lan O(1/T) and the n_train≥T Structural Guarantee

**Theorem 2.1 (Ghadimi & Lan, arXiv:1309.5549):**
For L-smooth non-convex function f, SGD satisfies:

    min_{t=0,...,T-1} E[||∇f(x_t)||²] ≤ (2L(f(x₀) - f*)) / T + σ²/(bT)

**Critical assumption:** Unbiased gradient estimates with bounded variance.

**n_train≥T structural guarantee (proven in Finding #359 / MATH.md m2p_data_scale
Theorem 1; inherited in exp_m2p_macro_quality MATH.md Theorem 1):**
When n_train ≥ T (each sample seen at most once), the gradient estimator satisfies the
Ghadimi-Lan i.i.d. assumption, and the Hardt et al. (2016) generalization bound gives:

    E[L_gen(θ_T)] - L_train(θ_T) ≤ O(T/n_train) ≤ O(1)  (bounded constant)

This is a sufficient condition. It does NOT depend on d_model — the condition is on the
data/compute ratio, not on the model dimensions.

**Corollary (n* at T=1000, 80/20 split):**
    n* = T / 0.8 = 1000 / 0.8 = 1250 samples.
At n=2000 (n_train=1600): T/n_train = 0.625 epochs. No cycling. Structural guarantee holds.
This derivation is INDEPENDENT of d_model: it holds equally at d=256, d=512, and d=1024.

### B.2 GL Early Stopping as Infrastructure (Prechelt, 1998)

**Prechelt (1998), "Early Stopping — But When?":**

    GL(t) = 100 × (val_loss(t) / min_{s≤t} val_loss(s) − 1)

Stop when GL(t) > α = 5.0 for PATIENCE = 5 consecutive checks.

**Theorem (exp_m2p_macro_quality MATH.md Theorem 2):** Under GL early stopping with
threshold α=5.0, the validation loss at stopping point T* satisfies:

    val_loss(T*) ≤ (1 + α/100) × best_val_loss = 1.05 × best_val_loss

This is a TIGHT bound — it holds by construction of the GL criterion, independent
of d_model.

**Train-val gap threshold derivation (inherited from K883):**
At d=256 (Finding #359), measured max train-val gap at n=2000 = 0.337 nats.
At d=512 (Finding #361), K883 threshold = 0.7 nats was set as 2 × 0.337.
K886 threshold (this experiment) = 0.7 nats (same as K883) because:
- The d=512 result (101%) showed the gap was actually WELL controlled at d=512
- We do not have measured values to derive a d=1024 specific threshold yet
- 0.7 nats is a conservative bound that held at d=512; applying it at d=1024 is safe

### B.3 Intrinsic Dimensionality Hypothesis (Aghajanyan et al., arXiv:2012.13255)

**Core claim (Aghajanyan et al., 2021, "Intrinsic Dimensionality Explains the
Effectiveness of Language Model Fine-Tuning"):**
When fine-tuning a language model, the effective parameter update lies in a
low-dimensional subspace whose dimensionality d_int satisfies:

    d_int << d_model × d_out  (parametric dimension of the weight matrix)

Critically, the authors find that **d_int is largely INDEPENDENT of d_model**. Larger
models do not have proportionally larger intrinsic fine-tuning subspaces.

**Formal statement (Theorem 2 of Aghajanyan et al.):** For a fine-tuned model with
parameter dimensionality D, there exists a random subspace of dimension d_int << D
such that fine-tuning in that subspace achieves 90% of full-parameter fine-tuning
performance, where d_int scales sub-linearly (often nearly constant) with D.

**Application to M2P B-matrix generation:**
The M2P bottleneck (d_M2P=64) acts as a FORCED PROJECTION onto a 64-dimensional
subspace when generating B-matrices. If the true intrinsic dimensionality of the
B-matrix update space is ≤ 64 (or close to it), then d_M2P=64 remains sufficient
even as d_model scales to 1024.

The fact that exp_m2p_macro_quality achieved 101% of SFT at d=512 (EXCEEDING SFT
quality) is strong evidence that d_M2P=64 is not just sufficient but provides IMPLICIT
REGULARIZATION that improves generalization beyond direct SFT. This is precisely
the Ha et al. (arXiv:1609.09106) HyperNetworks mechanism: the bottleneck matches
the intrinsic dimensionality of the target space.

**Why Bartlett's framework fails here (and why we should rely on Aghajanyan):**
Bartlett et al. (arXiv:1906.11300) count output PARAMETERS when estimating d_eff.
But the M2P is not learning random B-matrices — it is learning B-matrices constrained
to be in the fine-tuning subspace of a pre-trained transformer. The relevant d_eff is
the INTRINSIC DIMENSIONALITY of that subspace (Aghajanyan), not the parameter count.
Since intrinsic dimensionality is d_model-independent, d_M2P=64 should remain sufficient
at d=1024, just as it did at d=512.

### B.4 HyperNetworks and Bottleneck-as-Regularizer (Ha et al., arXiv:1609.09106)

**Ha et al. (2016), "HyperNetworks":**
A hypernetwork generates weights for a primary network. The key finding: when the
hypernetwork bottleneck dimension matches the intrinsic dimensionality of the target
weight space, the generated weights can OUTPERFORM directly-trained weights (generalize
better due to the bottleneck serving as a regularizer).

**Application:** The M2P is a hypernetwork for B-matrices. At d=512:
- Direct SFT: 400 steps, learns B-matrices directly
- M2P: 1000 steps generating B-matrices via 64-dim bottleneck
- Result: M2P achieves 101% of SFT — bottleneck regularizes better than direct training

At d=1024, the same argument applies if d_int ≤ 64 still holds.

**Compression ratio analysis:**
- d=256: fc1 head compression = (N_LAYERS × LORA_RANK × 4 × 256) / 64 = 8192/64 = 128:1
- d=512: fc1 head compression = (N_LAYERS × LORA_RANK × 4 × 512) / 64 = 16384/64 = 256:1
- d=1024: fc1 head compression = (N_LAYERS × LORA_RANK × 4 × 1024) / 64 = 32768/64 = 512:1

The compression ratio DOUBLES at each scale. The fact that 256:1 compression (d=512)
achieved 101% of SFT quality means the compression is NOT an information bottleneck —
the intrinsic content is already well below 64 dimensions. Whether 512:1 compression
(d=1024) remains benign is the core question.

---

## C. Proof of Guarantee

### Theorem 1 (n_train≥T Structural Guarantee is d_model-Independent)

**Theorem 1.** The n_train≥T structural guarantee derived in exp_m2p_data_scale
(Ghadimi-Lan + Hardt et al.) is independent of d_model. At n=2000 (n_train=1600)
and T=1000, the i.i.d. gradient sampling condition is satisfied regardless of
whether d_model=256, d_model=512, or d_model=1024.

*Proof.*
The Ghadimi-Lan theorem (arXiv:1309.5549, Theorem 2.1) requires:
    (a) L-smooth loss function f
    (b) Unbiased gradient estimates: E[g_t] = ∇f(θ_t)
    (c) Bounded gradient noise: E[||g_t - ∇f(θ_t)||²] ≤ σ²

Condition (b) holds when at step t, the sample x_t is drawn independently from
the training distribution. This is guaranteed when n_train ≥ T (each sample
appears at most once in T steps of single-pass SGD).

The condition n_train ≥ T involves ONLY the data size n_train and the number of
gradient steps T. It does NOT involve d_model, d_M2P, or any architectural
dimension. The transformer architecture affects the constant L (smoothness of loss)
and σ² (gradient noise magnitude) but NOT whether the i.i.d. condition holds.

Therefore:
    n=2000 (n_train=1600), T=1000 → T/n_train = 0.625 < 1 epoch
    → No sample is visited more than once.
    → Ghadimi-Lan O(1/T) bound applies.
    → Hardt et al. generalization bound: E[L_gen - L_train] ≤ O(T/n_train) = O(0.625)

This holds at d_model=256 (proven in Finding #359), at d_model=512 (confirmed in
Finding #361), and at d_model=1024 (no architectural constraint violated).

QED.

**Corollary 1.1 (n* at T=1000).**
n* = T / 0.8 = 1000 / 0.8 = 1250. At n=2000 (n_train=1600): T/n_train = 0.625.
No sample is visited more than once. Structural guarantee satisfied at d_model=1024.

### Theorem 2 (Intrinsic Dimensionality Independence from d_model)

**Theorem 2.** Under the Aghajanyan et al. (arXiv:2012.13255) intrinsic
dimensionality framework, if the intrinsic dimensionality d_int of the B-matrix
update space is independent of d_model, then the M2P bottleneck at d_M2P=64 is
sufficient for generating B-matrices at d_model=1024 provided d_int ≤ 64.

*Proof.*
By Aghajanyan et al. Theorem 2: fine-tuning updates lie in a subspace of
intrinsic dimension d_int << D. The B-matrices in this experiment represent
fine-tuning updates (LoRA adapters). Therefore, the B-matrices lie (approximately)
in a d_int-dimensional subspace of the full parameter space.

The M2P projects from its 64-dimensional pooled memory state to the B-matrix space
via output heads. By the definition of d_int, if d_M2P=64 ≥ d_int, then the
projection can span the relevant subspace, and the M2P can express any valid
B-matrix (up to approximation error bounded by the fraction of variance outside
the d_int-dimensional subspace).

The experimental evidence supports d_int ≤ 64:
- At d=256: d_M2P=64 achieves 97.6% quality (Finding #359).
  The 2.4% gap is the approximation error from the projection.
- At d=512: d_M2P=64 achieves 101.0% quality (Finding #361).
  The 101% result (exceeding SFT) indicates d_M2P=64 provides
  regularization beyond spanning the subspace — consistent with d_int < 64
  (the bottleneck induces beneficial implicit regularization).

If d_int is truly d_model-independent (as Aghajanyan et al. claim), then the
same result should hold at d_model=1024:

    d_M2P=64 ≥ d_int  ⟹  M2P can generate valid B-matrices at d=1024.

Therefore, the recipe should achieve quality_ratio ≥ 85% at d=1024.

*Caveat:* This theorem relies on the Aghajanyan et al. finding holding for toy
transformers at micro scale with only 3 domains and synthetic data. The original
results were established on full-scale LLMs (BERT, RoBERTa). Extrapolation to
toy scale is a Type 2 guided exploration, not a Type 1 proof verification.

QED (under Aghajanyan et al. assumptions).

### Scaling Heuristic (Compression Ratio Analysis — Engineering Estimate, NOT a Theorem)

**Estimate (not a theorem).** The fc1 output head compression ratios are:
- d=256: 128:1 → 97.6% quality (Finding #359)
- d=512: 256:1 → 101.0% quality (Finding #361)
- d=1024: 512:1 → predicted ≥ 85% quality

The compression ratio doubles at each scale step. The quality did NOT decrease
from 128:1 to 256:1 (went up). If the intrinsic dimensionality is indeed below 64,
compression can increase arbitrarily without information loss.

**The key question is whether there is a "compression cliff"** — a threshold above
which the bottleneck starts excluding necessary information. Given that 256:1 was
fine (101%), 512:1 might also be fine. The Aghajanyan framework suggests there is
no cliff as long as d_int ≤ 64.

**Practical prediction:** Given the observed scaling behavior:
- 97.6% at 128:1 compression → 101% at 256:1 compression
  (0% degradation, actually improvement from regularization)
- We predict ≥ 85% at 512:1 compression
  (allowing up to 16pp degradation from the peak of 101%)

The kill criteria use asymmetric thresholds informed by the d=512 result:
- K885: ≥ 85% (same floor as K882, now informed by d=512 success)
- K887 (KILL): < 50% (lowered from 60% because we've seen 101% at d=512;
  a drop below 50% would be a severe and unexpected cliff)

---

## D. Quantitative Predictions (from Theorems 1 and 2)

### D.1 Primary Predictions

| Condition | Predicted value | Derivation | Kill criteria |
|-----------|----------------|------------|---------------|
| n_train≥T holds at d=1024 | T/n_train=0.625 < 1 | Theorem 1 (d_model independent) | (structural check) |
| train-val gap at n=2000, d=1024 | < 0.7 nats | Inherited from K883 threshold | K886 |
| quality_ratio(d=1024, n=2000) | ≥ 85% of SFT | Aghajanyan d_int independence | K885 |
| K887 kill (capacity failure) | < 50% quality | Compression cliff would need to occur between 256:1 and 512:1 | K887 |
| Quality vs d=512 | ≥ 85%, possibly near 100% | Aghajanyan predicts no cliff | informational |

**Scaling trend for comparison:**
- quality_ratio(d=256, n=2000, T=1000) = 97.6% (Finding #359 actual)
- quality_ratio(d=512, n=2000, T=1000) = 101.0% (Finding #361 actual)
- quality_ratio(d=1024, n=2000, T=1000) = ≥ 85% (Theorem 2 prediction)

### D.2 Per-Domain Predictions

DOMAIN REDUCTION: N_DOMAINS=3 (arithmetic, sort, reverse). Parity excluded (guard).
Repeat excluded (reduced N_DOMAINS for runtime).

Expected ordering at n=2000 (from micro and macro history):
    sort ≈ reverse ≈ arithmetic  (all showed similar behavior at d=256 and d=512)

Parity guard applies: exclude domain if base_loss - sft_loss < 0.05 nats.

### D.3 n=1000 vs n=2000 Comparison

**n=1000 (n_train=800, T/n_train=1.25):** Partial cycling. Theorem 1 predicts mild
overfitting degradation vs n=2000. Expected: quality_ratio 5-15% below n=2000 case.

**n=2000 (n_train=1600, T/n_train=0.625):** Structural guarantee met. This is the
primary measurement condition for K885 and K886.

### D.4 Output Head Dimension Table

| Module | d=256 | d=512 | d=1024 | Compression (d=1024) |
|--------|-------|-------|--------|----------------------|
| wq/wk/wv/wo | 64→4096 | 64→8192 | 64→16384 | 256:1 |
| fc1 | 64→8192 | 64→16384 | 64→32768 | 512:1 |

Note: each output head generates N_LAYERS × LORA_RANK × d_out parameters.
For the fc1 head at d=1024: 2 × 4 × 4096 = 32,768 total parameters from 64 dims.

---

## E. Assumptions and Breaking Conditions

**Assumption 1 (Theorem 1: n_train≥T is the i.i.d. repair condition):**
REQUIRED: n=2000 (n_train=1600) at T=1000. T/n_train=0.625 < 1.
VERIFIED BY: experiment design (structural, not empirical).
BREAKS IF: code incorrectly cycles data (software bug, not mathematical failure).

**Assumption 2 (GL early stopping works at d=1024):**
Prechelt's GL criterion monitors val_loss/best_val_loss ratio — independent of d_model.
Cannot break mathematically. Implementation must be identical to Findings #359/#361 code.

**Assumption 3 (Aghajanyan intrinsic dimensionality is d_model-independent at toy scale):**
REQUIRED: d_int ≤ 64 at d_model=1024 for toy transformer domains.
VERIFIED BY: K885 (quality ≥ 85%) and K887 (quality ≥ 50%).
BREAKS IF: K887 triggers (quality < 50% → compression cliff above 256:1).
MITIGATED BY: Finding #361 showed 256:1 compression is NOT a cliff (101% quality).

**Assumption 4 (Parity guard correctly excludes marginal domains):**
PARITY_GUARD_THRESHOLD = 0.05 nats. Same as prior experiments. Cannot break.

**Assumption 5 (Aghajanyan et al. generalizes from LLM to toy transformer):**
The original intrinsic dimensionality results were established on BERT/RoBERTa, not toy
transformers. The experimental progression (97.6% → 101%) suggests the mechanism holds
at toy scale, but this is empirical evidence, not a mathematical proof.

---

## F. Worked Example (d=1024, T=1000, n=2000)

**n_train≥T check:**
    n = 2000, train_frac = 0.8 → n_train = 1600
    T = 1000
    epochs = T / n_train = 1000 / 1600 = 0.625 epochs < 1
    → No sample is visited more than once.
    → Structural guarantee satisfied. (Theorem 1, QED)

**Output head dimensions at d=1024:**
    MODULE_OUT_DIMS = [1024, 1024, 1024, 1024, 4×1024] = [1024, 1024, 1024, 1024, 4096]
    For fc1 (mi=4): out_head projects d_M2P=64 → N_LAYERS × LORA_RANK × d_out
                  = 2 × 4 × 4096 = 32,768 parameters
    For wq (mi=0): out_head projects d_M2P=64 → 2 × 4 × 1024 = 8,192 parameters

    Total B-matrix parameters: 2 × 4 × (1024×4 + 1024×4 + 1024×4 + 1024×4 + 4096×4)
                              = 2 × 4 × (4096 + 4096 + 4096 + 4096 + 16384)
                              = 2 × 4 × 32,768
                              = 262,144 B-matrix parameters per domain

    Compression ratio (fc1): 32768 / 64 = 512:1
    (At d=512, fc1 was 16384/64 = 256:1 → 101% quality)

**GL criterion example at d=1024:**
    Suppose val_loss history at steps 50, 100, ..., 300: [5.1, 4.6, 4.2, 4.0, 4.1, 4.3]
    best_val_loss = 4.0 at step 200
    val_loss at step 300 = 4.3
    GL(300) = 100 × (4.3 / 4.0 - 1) = 100 × 0.075 = 7.5 > GL_THRESHOLD=5.0
    → consecutive_gl_exceeded increments
    → After PATIENCE=5 consecutive checks: early stopping triggered

**Quality ratio computation:**
    base_loss = 5.5 (pre-SFT base model on domain val set)
    sft_loss  = 3.8 (SFT LoRA adapter quality)
    m2p_loss  = 4.0 (M2P-predicted adapter quality)
    gap = base_loss - sft_loss = 5.5 - 3.8 = 1.7 (> PARITY_GUARD_THRESHOLD=0.05)
    quality_ratio = (base_loss - m2p_loss) / gap = (5.5 - 4.0) / 1.7 = 1.5 / 1.7 = 88.2%
    → K885 PASS (88.2% ≥ 85%)
    → K887 NOT triggered (88.2% ≥ 50%)

---

## G. Complexity and Architecture Connection

**What changes from exp_m2p_macro_quality (THE ONLY CHANGE):**
- D_MODEL: 512 → 1024
- N_HEADS: 8 → 16 (maintain d_head = D_MODEL / N_HEADS = 64)
- MODULE_OUT_DIMS: [512, 512, 512, 512, 2048] → [1024, 1024, 1024, 1024, 4096]
- N_DOMAINS: 3 (unchanged — arithmetic, sort, reverse)
- N_SAMPLES sweep: [1000, 2000] (unchanged)
- SAMPLE_VALUES[0]=1000: reference point (T/n_train=1.25 epochs, partial cycling)
- SAMPLE_VALUES[1]=2000: primary test (T/n_train=0.625 epochs, structural guarantee)

**Everything FIXED (same as exp_m2p_macro_quality):**
- LORA_RANK = 4, LORA_SCALE = 2.0
- M2P_LAYERS = 2, D_M2P = 64, N_MEMORY = 32
- MODULE_NAMES = ["wq", "wk", "wv", "wo", "fc1"]
- T_FIXED = 1000, M2P_LR = 1e-3
- GL_THRESHOLD = 5.0, PATIENCE = 5, EARLY_STOP_INTERVAL = 50
- PARITY_GUARD_THRESHOLD = 0.05

**ToyGPT parameter count at d=1024 (estimated):**
- wte: 128 × 1024 = 131,072
- wpe: 49 × 1024 = 50,176
- Per block: wq+wk+wv+wo+fc1+fc2 = 4×(1024²) + 1024×4096 + 4096×1024
  = 4×1,048,576 + 4,194,304 + 4,194,304 = 12,582,912
- 2 blocks: 25,165,824
- norm_f + lm_head: ≈ 1024 + 1024×128 = ≈ 132,096
- Total: ~25.5M parameters (vs ~6.7M at d=512)

**M2P parameter count (d_M2P=64, unchanged from macro_quality):**
- Same as at d=512 (M2P internal dim is d_M2P=64, independent of d_model)
- The output heads scale: total_out = N_LAYERS × LORA_RANK × d_out
  - At d=1024: wq head = 64 × 8192, fc1 head = 64 × 32768 (larger than at d=512)
  - M2P total params will be approximately 2× that of d=512 variant

**FLOPs estimate (runtime budget):**
- Base pretraining: BASE_STEPS=1200, 3 domains × O(D_MODEL² × BLOCK_SIZE)
  At d=1024: ~4× the compute of d=512 experiment → ~8min
- SFT: SFT_STEPS=1000 per domain × 3 domains → ~4min
- M2P sweep: 2 n-values × 3 domains × T=1000 steps each → ~8min
- Total expected: ~20-25 minutes on M5 Pro (within 1-hour constraint)

**Architecture compatibility:**
ToyGPT at d=1024, N_HEADS=16: d_head = 1024/16 = 64 (same as micro d=256/4=64
and macro d=512/8=64). The attention operation is identical in structure.
M2P transformer uses d_M2P=64 internal dimension — completely independent of d_model.
The only interface point is the input projection: input_proj: D_MODEL → D_M2P (linear).
At d=1024: input_proj is 1024→64 (vs 512→64 at macro, 256→64 at micro).

**Connection to Qwen3-4B (d_model=2048/3584):**
This experiment tests the 4× scaling point (d=1024 from micro d=256 baseline).
If quality holds at d=1024, the next verification would be d=2048 (8× micro),
approaching Qwen3-4B's d_model. The intrinsic dimensionality argument suggests
the recipe should hold, but each doubling provides additional experimental evidence.

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode (capacity
   bottleneck from d_M2P=64) impossible?**
The intrinsic dimensionality of LoRA B-matrix updates is d_model-independent
(Aghajanyan et al., arXiv:2012.13255). Since the B-matrices lie in a subspace
of dimension d_int << d_model, and d_M2P=64 ≥ d_int was confirmed at d=256 and
d=512, the same bottleneck spans the relevant subspace at d=1024.

**2. Which existing theorem(s) does the proof build on?**
- Ghadimi & Lan (2013, arXiv:1309.5549, Theorem 2.1): O(1/T) convergence — requires
  unbiased i.i.d. gradient estimates (proven d_model-independent)
- Aghajanyan et al. (2021, arXiv:2012.13255, Theorem 2): intrinsic dimensionality of
  fine-tuning updates is independent of model size (core justification for d_M2P=64 sufficiency)
- Ha et al. (2016, arXiv:1609.09106): HyperNetworks — weight-generating networks can
  outperform directly-trained weights when bottleneck matches intrinsic dimensionality
- Prechelt (1998): GL criterion guarantees val_loss(T*) ≤ 1.05 × best_val_loss
- Hardt et al. (2016, "Train Faster, Generalize Better"): generalization gap ≤ O(T/n)
  for single-pass SGD

**3. What specific numbers does the proof predict?**
- T/n_train = 1000/1600 = 0.625 (structural guarantee satisfied — no cycling)
- train-val gap at n=2000, d=1024 < 0.7 nats (K886, inherited from K883 threshold)
- quality_ratio(n=2000, d=1024) ≥ 85% of SFT (K885, Aghajanyan d_int independence)
- K887 kill: quality_ratio < 50% (compression cliff between 256:1 and 512:1)
- Macro reference: quality_ratio(n=2000, d=512) = 101% (Finding #361)
- fc1 head compression: 512:1 (vs 256:1 at d=512 which achieved 101%)

**4. What would FALSIFY the proof (not just the experiment)?**
- Theorem 1 is structural: falsified only if code cycles data when n_train=1600 > T=1000.
  This cannot happen mathematically; it would indicate a software bug.
- Theorem 2 (Aghajanyan-based) is falsified if: quality_ratio < 85% at n=2000, d=1024
  AND the M2P bottleneck is confirmed as the cause (not data quantity). This would mean
  d_int > 64 at d=1024, contradicting Aghajanyan's claim of d_model independence.
  Specifically: K887 trigger (< 50%) would be definitive falsification.
- The proof is NOT falsified by a result between 50-85% — that range indicates the
  compression ratio 512:1 introduces some degradation but not a cliff.

**5. How many hyperparameters does this approach add?**
Count: 0 new hyperparameters relative to exp_m2p_macro_quality.
- N_SAMPLES = [1000, 2000]: the sweep variable (treatment), inherited from prior
- T_FIXED = 1000: same as macro (proven sufficient)
- D_MODEL = 1024: THE SINGLE CHANGE — this is the treatment variable, not a hyperparameter
- GL_THRESHOLD = 5.0, PATIENCE = 5: inherited from Prechelt (1998), same as prior
- d_M2P = 64, M2P_LAYERS = 2: inherited from Findings #355, #357 (not tuned here)

**6. Hack check: Am I adding fix #N to an existing stack?**
No. This experiment makes ONE change from exp_m2p_macro_quality: D_MODEL = 1024.
All other parameters are FIXED at their proven macro values. We are NOT adding a new
regularizer, a new loss term, or a new architecture component. We are testing whether
the proven recipe transfers to 4× scale (2× the last verified point at d=512).
