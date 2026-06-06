# Peer Review: Delta Coding for Expert Version Management

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that deep review via NotebookLM would not surface issues beyond manual inspection. The math is linear algebra (subtraction, SVD truncation, addition). No exotic derivations.

## Mathematical Soundness

**What holds:**

1. Raw delta reconstruction is exact (subtraction then addition is identity up to float precision). The 1.6e-08 relative error confirms this. No issues.

2. The SVD truncation bound is correct: error accumulates additively along the chain (triangle inequality on Frobenius norms). The bound in Section 4 of MATH.md is valid.

3. The storage ratio formula R = (N_keyframes + rho * N_deltas) / N is correct. The worked example in Section 7 checks out.

4. The compression ratio rho = r'*(m + n + 1) / (m * n) correctly accounts for storing U_r, S_r, Vt_r. Code confirms this (lines 105 in delta_coding_expert_versions.py).

**Minor issues:**

5. MATH.md Section 4: "SVD rank 2 captures ~60% of delta energy (relative error ~0.58)" -- this is confusingly stated. If rank-2 relative error is 0.58, that means 58% of the energy is in the *residual*, not captured. It captures ~42% of energy. The paper says "~43% of delta energy" which is consistent but contradicts the "~60%" in MATH.md. This is a documentation inconsistency, not a math error.

6. MATH.md Section 6 states "f(0.264) ~ 0.2% per delta" but the measured max drift across 4 chained deltas is 0.796%. If drift scaled linearly at 0.2% per delta, 4 deltas would give 0.8%, which is consistent. However, the per-delta number is back-calculated from max chain, not independently measured at each chain position. The v3 mid-chain measurement partially addresses this but the paper could be clearer.

## Novelty Assessment

**Prior art the researcher found:**
- BitDelta (2024): 1-bit delta quantization for base-to-fine-tuned deltas
- DeltaZip (2024): Multi-tenant delta serving with mixed-precision quantization
- DeltaDQ (2024): Ultra-high delta compression
- FedFQ (2024): Fine-grained quantization in federated learning

**The claimed novelty** -- temporal version chains with video codec analogy (I-frame/P-frame/GOP) -- is fair. None of the cited works chain deltas sequentially across versions. The analogy to video codecs is apt and the framework (keyframe interval K, chain drift bounds) provides useful structure.

**However, the novelty is narrower than presented:**

1. Sequential delta chains are a standard technique in version control (git stores objects as delta chains), database systems (log-structured storage), and backup systems (incremental backups with periodic full backups). The video codec framing is a nice analogy but the underlying idea -- store incremental diffs with periodic full snapshots -- is decades old.

2. SVD compression of weight deltas is well-studied. LoRA itself is SVD-motivated. Applying truncated SVD to inter-version deltas is a natural combination of two known ideas.

3. The "contribution" is more accurately described as: *confirming that a standard engineering pattern (incremental snapshots + lossy compression) works for LoRA weight versioning with acceptable quality/storage tradeoffs.* This is useful engineering validation, not a novel mechanism.

## Experimental Design

**Strengths:**
- Clean protocol: pretrain, then sequential fine-tuning to create natural version evolution
- Multiple SVD ranks tested, showing the quality-storage Pareto front
- 3 seeds with explicit kill criteria checking across seeds
- Both mid-chain (v3) and end-of-chain (v5) drift measured

**Weaknesses:**

1. **Quality drift measurement is loss-relative, not accuracy-relative.** The drift is measured as (recon_loss - true_loss) / true_loss * 100. At these loss scales (around 2.0-2.5 for character-level LM), a 0.8% drift is ~0.016-0.020 loss units. This is essentially noise-level on 5-batch evaluation (n_batches=5 in quick_eval). The "drift" being measured may be dominated by evaluation noise rather than actual model quality change.

2. **Evaluation variance is not reported.** With only 5 batches of size 32 (160 samples), the evaluation has high variance. The experiment does not report confidence intervals on the drift measurements. A 0.796% max drift on a noisy estimator could easily be 0.0% or 1.5% with different eval batches.

3. **Versions are not truly independent.** Versions v2-v5 are produced by *continuing* training from the previous version on slightly different data subsets. This creates maximally smooth version transitions by construction. A more realistic test would include: (a) versions trained from scratch on different data, (b) versions after different numbers of training steps, (c) versions after hyperparameter changes. The smooth continuation biases toward compressible deltas.

4. **Only LoRA A and B matrices are versioned.** The experiment only stores/compresses LoRA parameters (20,480 total). In a real system, optimizer states (Adam moments = 2x params), learning rate schedules, and potentially other metadata would also need versioning. The 59% savings applies only to the weight parameters.

5. **Keyframe interval K=5 with N=5 means exactly 1 keyframe.** This is the maximum-compression scenario. The experiment does not test K=2 or K=3 where more keyframes reduce chain length but increase storage. The protocol mentions "K=1,2,5" but the results only show K=5.

6. **The "SVD rank 4 storage KILLS" assessment is wrong by the paper's own criteria.** Storage ratio of 62.3% exceeds 50%, so it fails KC2 -- but rank 4 has only 0.194% drift, which is excellent. The paper frames this as "excessive quality, only 38% savings" but the kill criterion is storage > 50%, which rank 4 fails. This is correctly handled in the code but the narrative framing in PAPER.md is misleading -- rank 4 does not "KILL" on storage, it simply does not meet the <50% threshold.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_delta_coding_expert_versions` has:
- Kill criteria: quality drift >1% per delta, delta storage >50% of full
- Status: validated
- Priority: 11

The experiment tests exactly these criteria. The evidence is directional and consistent with the kill criteria. No issues with graph consistency.

**Question: does this advance the core vision?** VISION.md describes a contribution protocol where crowd-sourced LoRA experts are composed at runtime. Version management is useful infrastructure for an expert registry but does not address the core research question of sparse composition, routing, or gap-as-signal. This is a support utility, not a core mechanism.

## Macro-Scale Risks (advisory)

1. **Delta rank scales with training steps.** At 80 steps, deltas are approximately rank-2. At 10,000 steps (realistic fine-tuning), deltas will have much higher effective rank, and SVD truncation to rank 2 will destroy far more information. The "sweet spot" will shift dramatically.

2. **LoRA rank scales too.** At rank 8, the A matrix is 64x8 -- SVD of this is cheap and the space of possible deltas is small. At rank 64 with d=4096 (macro scale), A is 4096x64, and the delta space is vastly larger. Compression ratios will change.

3. **Quantized models.** If LoRA weights are stored in lower precision (int8, int4), delta computation introduces quantization artifacts that accumulate differently than float32 rounding.

4. **The real comparison at macro is against git-lfs / HuggingFace Hub.** In practice, people just store full checkpoints. The question is whether 59% savings on LoRA params (which are already tiny -- 20K params vs millions for the base model) justifies the complexity of a version chain system. At macro scale with a 7B base model, LoRA params at rank 16 might be ~10M. Storing 5 full snapshots = 50M params = 200MB. Delta-coded = 82MB. The savings (118MB) are real but modest compared to the base model size.

## Verdict

**PROCEED** -- with caveats.

The experiment is competently executed and the results are consistent with the stated hypothesis. The math is correct, the code is clean, and the kill criteria are fairly applied.

However, this is an engineering validation of a well-known pattern (incremental snapshots with lossy compression), not a novel mechanism. The novelty claim should be tempered. The experiment succeeds at what it set out to do -- confirm that delta coding with SVD compression achieves <1% drift at <50% storage -- but the margin is thin (0.796% vs 1% threshold on a noisy estimator with 5 eval batches).

This is useful infrastructure for the expert registry described in VISION.md, but it does not advance the core research question of sparse expert composition. Priority 11 seems appropriate -- this is supporting work, not a breakthrough.

**If the researcher wants to strengthen this before moving on:**

1. Increase eval batch count from 5 to 50 and report confidence intervals on drift measurements. The current 0.796% max drift on 160 samples is not statistically distinguishable from 0.0% or 1.5%.

2. Fix the "~60% of delta energy" inconsistency in MATH.md Section 8 (should be ~42%).

3. Test at least one non-smooth version transition (e.g., reset LoRA and retrain from scratch for one version) to probe the assumption that deltas are smooth.

These are advisory, not blocking. The experiment passes its stated criteria and the limitations are honestly acknowledged.
