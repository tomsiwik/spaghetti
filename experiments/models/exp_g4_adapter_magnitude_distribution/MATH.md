# MATH.md — exp_g4_adapter_magnitude_distribution

## Title
Adapter weight magnitude distribution: Gaussian (no exploitable structure) or structured?

## Type
Verification — profiling experiment testing whether trained LoRA adapters on Gemma 4 E4B develop exploitable structure (sparse, clustered, sub-low-rank, non-Gaussian) beyond the baseline rank-r linear factorization, OR remain close to Gaussian (the null structure implied by Kaiming LoRA-A init + SGD).

## Failure mode being addressed
Magnitude-based compression / magnitude-as-importance heuristics (e.g., top-k pruning, weighted-merge) assume adapter weights carry additional exploitable structure beyond their rank constraint. If that structure is absent (weights remain Gaussian after training, magnitude does not predict behavioral importance), the entire class of "magnitude-based adapter compression" methods is not applicable to trained Gemma 4 LoRA adapters — saving us future dead-end experiments.

## References
- LoRA (Hu et al. 2021, arxiv:2106.09685): baseline LoRA init `A ~ Kaiming Gaussian(fan_in)`, `B = 0`. Under SGD with small learning rates, A and B remain close to their initialization distribution (plus learned signal).
- Wanda (Sun et al. 2023, arxiv:2306.11695): pruning by `|w · σ(activation)|` outperforms pure `|w|` on LLMs — evidence that magnitude alone is insufficient.
- Finding #500: null-space projection magnitude cannot predict adapter quality (AUC = 0.43, below chance).
- Finding #526: pre-merge composition failure is direction-dependent, not magnitude-dependent.
- Finding #350: preservation loss does not teach M2P scale differentiation; all adapters same magnitude (CV = 0.9%).
- Finding #666: proxy-only KCs must be paired with target-metric KCs; KILL requires both to fail.

## Theorem (weak, constructive)

**Theorem.** Let `A_l ∈ R^(d×r)` and `B_l ∈ R^(r×d')` be the trained LoRA factors at layer `l` of a Gemma-4-E4B q_proj adapter, with `A_l(0) ~ Kaiming(d)`, `B_l(0) = 0`, trained by Adam with learning rate `η = 1e-4` for `T = 1000` iters on a domain corpus.

Define:
- `μ̂_A, σ̂_A` := empirical mean, std of `vec(A_l)`; `μ̂_B, σ̂_B` := same for `B_l`.
- `H_0^A`: `vec(A_l)` is drawn from `N(μ̂_A, σ̂_A²)` (Gaussian null).
- `r_eff^A := min{k : Σ_{i≤k} s_i² / Σ s_j² > 0.99}` where `s_i` are the singular values of `A_l`; i.e. effective rank at 99% variance.
- For each per-layer weight `w ∈ A_l ∪ B_l`, let `I(w) := (∂L_domain / ∂w)²` be the Fisher-proxy importance computed on the training distribution.

**Prediction set (derived from priors):**

1. **P1 — Gaussianity (null).** By init + small-step Adam regularization toward zero, `vec(A_l)` is approximately Gaussian. Concretely: Shapiro-Wilk `p > 0.01` for `A_l` on `≥ 80%` of layers; skewness `|γ_1| < 0.5` and excess kurtosis `|γ_2| < 1.0`.

2. **P2 — No sub-low-rank structure.** `A_l` is already rank-6 by construction; no further compressibility. Prediction: `r_eff^A ≥ 5` on `≥ 80%` of layers (i.e., all 6 directions carry meaningful variance, no single dominant axis).

3. **P3 — No sparsity structure.** Fraction of `|w| < 1e-4` within each matrix is `< 10%` (Kaiming init populates every entry; Adam with no L1 keeps it that way).

4. **P4 — Magnitude ≠ Fisher importance.** Cross-weight Pearson correlation between `|w|` and `I(w)` satisfies `|r| < 0.3` — magnitude tells you almost nothing about behavioral sensitivity (F#500, F#526 generalize).

5. **P5 — Magnitude ≠ behavioral ablation impact.** Zeroing the top-20% magnitude weights of each layer (per layer) produces a PPL delta within a factor of 2 of zeroing a random 20% of weights at the same sparsity level. I.e., `|ΔPPL_top - ΔPPL_rand| / max(ΔPPL_top, ΔPPL_rand) < 0.5` averaged across domains.

**Proof (constructive).** Each prediction is a standard consequence of the training dynamics:
- P1 follows because Adam at `η = 1e-4 ⋅ T = 1000` with no L1 regularization drifts the Kaiming Gaussian init by small signed steps; the result stays close to Gaussian in distribution (classical perturbation argument).
- P2 is forced by the rank constraint `A_l ∈ R^(d×6)`: the effective rank at the 99% variance level cannot exceed 6; it also cannot be << 6 unless SGD collapsed directions, which is suppressed by adaptive per-coordinate step sizes in Adam.
- P3 follows because Kaiming init variance `2/d = 2/2560 ≈ 7.8e-4` puts std `≈ 2.8e-2`, so the 1e-4 tail is at `z ≈ 3.6e-3` of the std — roughly `0.3%` under Gaussian (far less than 10%).
- P4 follows from the rank-r factorization: the scaling of `|w|` is arbitrary under the rescaling symmetry `A → cA, B → B/c`, which does not affect `W = BA` and therefore does not affect `I(w)` in a monotone way. This IS F#526's direction-vs-magnitude distinction applied to single weights.
- P5 generalizes P4 behaviorally: since magnitude doesn't track importance, top-magnitude and random-magnitude pruning at matched sparsity should have comparable behavioral effect. QED.

## Pre-registered Kill Criteria (F#666-compliant)

| ID | Kind | Condition (firing = kill) |
|---|---|---|
| K1917 | proxy | Adapter weights ARE Gaussian: Shapiro-Wilk `p > 0.01` on `≥ 80%` of (lora_a OR lora_b) matrices across the 3 domains, AND `|skew|<0.5` AND `|excess-kurtosis|<1.0` on `≥ 80%`. |
| K1971 | target (paired with K1917) | Compression is NOT behaviorally exploitable: `|ΔPPL(top-k mag prune) - ΔPPL(random-k prune)| / max(...) < 0.5` at 20% sparsity on held-out domain corpus (averaged across domains). |
| K1918 | proxy | Magnitude does NOT predict importance: cross-weight Pearson `|r|(|w|, I(w))` `< 0.3` averaged across layers and domains. |
| K1972 | target (paired with K1918) | Behavioral ablation matches P5: `|ΔPPL_top - ΔPPL_rand| / max(ΔPPL_top, ΔPPL_rand) < 0.5` on held-out corpus (reuses K1971 measurement path). |

**Kill rule (F#666):**
- K1 (structure) killed ⇔ K1917 AND K1971 both fire.
- K2 (magnitude=importance) killed ⇔ K1918 AND K1972 both fire.

Predicted outcome: ALL FOUR KCs fire → experiment KILLS the "adapter weights have exploitable structure" and "magnitude predicts importance" hypotheses together.

This is a KILL-expected verification: the hypothesis is predicted to die, and this is a cleanup experiment for future roadmap (removes magnitude-based compression from the candidate list).

## Data and model
- **Base model**: `mlx-community/gemma-4-e4b-it-4bit` (per PLAN.md Part 2, pinned dev base).
- **Adapters**: 3 trained Gemma 4 E4B LoRA adapters (`q_proj`, `r=6`, scale=6.0, 1000 iters, Adam η=1e-4) from `exp_p1_t2_single_domain_training`:
  - `micro/models/exp_p1_t2_single_domain_training/adapters/medical/adapters.safetensors`
  - `.../math/adapters.safetensors`
  - `.../code/adapters.safetensors`
- **Eval corpus**: first 100 samples of each domain's training JSONL (the eval set used in `data/medical/valid.jsonl` etc.), for PPL delta measurement. 100 is a rough-order target-metric proxy — the question is "do top-k and random-k differ by >2x" which is a large effect, easy to measure with modest N.

Note: q_proj is the **training target** of these existing adapters; v_proj+o_proj per F#627 is the optimal target but requires retraining. Since the hypothesis is about the distribution of learned adapter weights (not specific to which module), q_proj adapters are valid evidence — and the result generalizes structurally.

## Methodology

### K1917 — Normality test per matrix
For each of `3 adapters × 42 layers × 2 matrices = 252` weight tensors:
- Shapiro-Wilk test on the flattened tensor (subsample to n=5000 if matrix is larger; `scipy.stats.shapiro` hard limit).
- Skewness (`scipy.stats.skew`) and excess kurtosis (`scipy.stats.kurtosis`).
- SVD of each `A_l` for effective rank at 99% variance (P2 sanity).
- Sparsity fraction: `mean(|w| < 1e-4)` per matrix.

K1917 fires iff ≥ 80% of matrices pass Shapiro (p>0.01) AND |skew|<0.5 AND |kurtosis|<1.0.

### K1918 — Magnitude vs Fisher importance correlation
For each adapter × domain:
- Load Gemma 4 E4B base + single adapter via `mlx_lm`.
- Take first 32 training samples from the adapter's domain, construct a single training batch (max 512 tokens per sample).
- Compute `loss = -logp(batch)` and `grads = mx.grad(loss)` wrt adapter parameters via `nn.value_and_grad`.
- Fisher-proxy importance per weight: `I(w) = g(w)²`.
- Per-layer Pearson `r(|w|, I(w))` over all weights in that layer.
- Average `|r|` across 42 layers × 3 domains = 126 per-layer correlations.

K1918 fires iff mean `|r| < 0.3`.

### K1971 / K1972 — Behavioral ablation (single shared measurement path)
Per domain, per adapter:
- Compute baseline PPL on 100 held-out domain samples (no ablation).
- Magnitude-prune mask `M_top`: zero the top-20% `|w|` per matrix (per-layer top-k, not global).
- Random-prune mask `M_rand`: zero a random 20% per matrix (seeded, deterministic, matched count).
- Apply mask by in-place zeroing the lora tensors (structural ablation), measure PPL on the same 100 samples.
- Record `ΔPPL_top := PPL(M_top) - PPL(baseline)`, `ΔPPL_rand := PPL(M_rand) - PPL(baseline)`.
- Ratio `R := |ΔPPL_top - ΔPPL_rand| / max(ΔPPL_top, ΔPPL_rand)`. Average R across 3 domains.

K1971 fires iff mean `R < 0.5` (compression is no better than random).
K1972 is bound to the same measurement (P5 is literally this test) — fires iff mean `R < 0.5`.

Note: K1971 and K1972 share one measurement path by construction — they test the same behavioral question from two framings (structure unlocks compression vs. magnitude predicts importance). This is deliberate: the behavioral target is *"does magnitude carry exploitable signal for adapter ablation?"* and the answer is the same test.

## Assumptions and caveats
- **q_proj adapters, not v_proj+o_proj**: the existing pre-trained adapters use q_proj. F#627 established v_proj+o_proj as optimal target; however, the question being tested here (distribution of trained LoRA weights) is not tied to a specific projection — Kaiming Gaussian init + Adam is the same regardless of module. Caveat logged for generalization.
- **Fisher proxy, not true Fisher**: `g(w)²` on a single batch is a noisy estimate of the expected Hessian diagonal. With 32 samples this is a proxy-of-a-proxy for behavioral importance; the paired target-metric KC (K1972) closes this gap.
- **Small N eval (100)**: PPL is measured over 100 samples per domain; effect size needs to be >2x to cross the KC threshold, well within statistical power.

## Antipattern scan (pre-flight checklist)
- **Composition math bug**: N/A — no composition, pure single-adapter analysis.
- **Unsafe LORA_SCALE**: adapters loaded at their trained scale (6.0 per config). Not inflated.
- **Tautological routing / single-sample proxy**: N/A — no routing.
- **`shutil.copy` as new adapter**: N/A — no new adapter creation.
- **Hardcoded `"pass": True`**: results.json populated from measured values.
- **Proxy-model substitution**: base model is `mlx-community/gemma-4-e4b-it-4bit` matching MATH.md; not substituting to smaller.
- **KC measures wrong object**: K1917 measures weight distribution (the claim). K1918 measures magnitude-importance correlation (the claim). K1971/K1972 measure behavioral ablation (the claim). All aligned.
- **Smoke reported as full**: N/A — full profiling, not smoke.
- **KC-swap-after-failure**: pre-registered before the first run; no post-hoc modifications.

## Runtime budget
Expected wall-clock: ~45-75 minutes total.
- K1917 (pure numpy, no model): ~30 seconds.
- K1918 (1 backward pass per domain on small batch): 3 × 2 min = 6 min.
- K1971/K1972 (baseline + 2 ablations × 3 domains × 100 samples): 3 × (3 × 100 × ~0.6s) ≈ 15 min with batching, conservatively 30 min.

Upper timeout: 2h.

## mlx-lm version
Cited in `run_experiment.py` at runtime via `mlx_lm.__version__`.
