# MATH.md — exp_composition_ordering_matters

## Title
Composition ordering: does adapter sum order affect output quality in IEEE-754 floating-point?

## Type
Verification — testing whether the non-associativity of floating-point summation produces any practically detectable difference when summing N=3 trained LoRA adapter deltas on Gemma 4 E4B in different permutation orders.

## Failure mode being addressed
If composition ordering mattered in a non-negligible way, then any deployment that computes `y = W_q @ x + Σ ΔW_i @ x` or any caller that merges `W_base + Σ ΔW_i` would produce order-dependent results. Users permuting adapter lists would see different quality. Every experiment that computed composed deltas without fixing the order would be silently non-reproducible. A confirmed invariance **below a well-posed bound** forecloses an entire class of "reorder adapters to get a better answer" bug hunts and enables reproducible composition in Pierre.

## References
- Higham, N. J. — *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM, 2002), §4 "Summation": IEEE-754 floating-point summation error is bounded by `|Σ̂ - Σ_true| ≤ (n-1) · u · Σ |x_i|` where `u` is unit roundoff. For FP32, `u ≈ 1.2e-7`.
- Ilharco et al. 2022 (arxiv:2212.04089): task arithmetic / Room Model: `W_comp = W_base + Σ ΔW_i`. Paper treats the sum as mathematically associative; no ordering claims made.
- Finding #7 (conclusive): prune-then-compose order invariant at 0.012% PPL gap (170× margin on a related but distinct ordering question — pruning vs. merging order, not summation order).
- Finding #14 (supported): 1/N scaling resolves composition catastrophe; this experiment is downstream of that stability regime.
- Finding #526: composition is direction-dependent, not magnitude-dependent — consistent with ordering being about direction of accumulation, which FP summation preserves to `O(u)`.
- Finding #666 (target-gated kill): every proxy KC must be paired with a target-metric KC. Both must fire for KILL.

## Theorem (constructive)

**Setup.** Let `{(A_i, B_i)}_{i=1}^{N}` be `N=3` trained rank-`r=6` LoRA adapter pairs on Gemma 4 E4B (medical, math, code from `exp_p1_t2_single_domain_training`, q_proj target, scale `s=6.0`). For a given layer `ℓ`, define the per-adapter delta:

```
ΔW_i^ℓ := (s / r) · B_i^ℓ @ A_i^ℓ ∈ R^(d_out × d_in)
```

(The `s/r` factor matches the mlx-lm LoRALinear forward: `y = W x + (s/r) B A x`; see `mlx_lm/tuner/lora.py`.)

Let `π ∈ S_N` be a permutation. Define the order-π composed delta:

```
S_π^ℓ := ΔW_{π(1)}^ℓ  ⊕  ΔW_{π(2)}^ℓ  ⊕  ΔW_{π(3)}^ℓ
```

where `⊕` is left-associative IEEE-754 FP32 addition evaluated in the order written.

**Theorem (ordering invariance).** For any two permutations `π, σ ∈ S_N`, the element-wise max absolute difference between composed deltas is bounded:

```
max_ℓ max_{(i,j)} |S_π^ℓ[i,j] - S_σ^ℓ[i,j]|  ≤  2 (N-1) u · max_ℓ max_i Σ_j |ΔW_i^ℓ[i,j]|
```

where `u = 2^(-23) ≈ 1.19e-7` is the FP32 unit roundoff. The factor `2(N-1)` upper-bounds the worst-case over the two summation orders. For `N=3`, this is `4u ≈ 4.8e-7` per row.

**Operator-norm corollary.** By the inequality `||M||_op ≤ ||M||_F ≤ sqrt(min(d_in,d_out)) · ||M||_F-row-max`:

```
||S_π - S_σ||_op  ≤  4u · sqrt(min(d_in, d_out)) · ||ΔW_typ||_∞
```

For Gemma 4 E4B `(d_in, d_out) ∈ {(2560, 2048), (2560, 4096)}` and empirical `||ΔW||_∞ ~ 1e-2` (typical LoRA trained on small LR):

```
||S_π - S_σ||_op  ≤  4 · 1.2e-7 · sqrt(2048) · 1e-2  ≈  2.2e-7
```

**PPL corollary.** For any input token sequence `x` with base hidden state `h` of norm `||h|| ≲ 10`, the perturbation of the pre-softmax logit vector at the q_proj path is bounded by:

```
||Δlogits|| ≤ ||S_π - S_σ||_op · ||h||  ≲  2.2e-6
```

The per-token NLL delta is bounded by `||Δlogits||_∞` (softmax Lipschitz ≤ 1 on logits differences). Thus `ΔPPL/PPL ≲ exp(2.2e-6) - 1 ≈ 2.2e-6` — **at least five orders of magnitude below the 1pp (0.01 relative) kill threshold**.

**QED.** The invariance holds deterministically by the FP32 summation error bound, not empirically.

## Pre-registered Kill Criteria (F#666-compliant)

| ID | Kind | Condition (firing = kill) |
|---|---|---|
| K1928 | proxy | **Weight-space max pairwise relative Frobenius gap** across all 6 permutations of 3 adapters exceeds `1e-5` on **any** layer: `max_ℓ max_{π,σ} ‖S_π^ℓ - S_σ^ℓ‖_F / ‖S_π^ℓ‖_F > 1e-5`. |
| K1975 | target (paired with K1928) | **Behavioral PPL gap**: max pairwise per-permutation PPL delta across 6 permutations on held-out corpus exceeds **1.0 percentage point relative** (i.e. `max_{π,σ} \|PPL_π - PPL_σ\| / PPL_baseline > 0.01`). |

**F#666 kill rule:** K1 (ordering matters) killed ⇔ K1928 **AND** K1975 both fire.
- Both pass (invariance holds) → SUPPORTED.
- K1928 fires but K1975 does not → finding about the proxy (FP noise exists but doesn't propagate) — issues a provisional note, not a kill.
- K1928 passes but K1975 fires → target fired on tautological proxy; dig into why (likely base-model noise, not ordering).

**Predicted outcome (from theorem):** Both KCs pass by an enormous margin. Weight-space relative gap `< 1e-6` (below `u` × N); PPL gap `< 1e-4` (rounded down from ~2e-6 to account for measurement noise over 100 eval samples). Experiment is expected to SUPPORT ordering invariance as a cleanup result.

## Data and model
- **Adapters**: 3 trained q_proj r=6 scale=6.0 LoRA adapters from `micro/models/exp_p1_t2_single_domain_training/adapters/{medical,math,code}/adapters.safetensors`.
- **Base model** (for K1975 PPL only): `mlx-community/gemma-4-e4b-it-4bit` per PLAN.md Part 2.
- **Eval corpus**: 100 held-out rows — 33 medical + 33 math + 34 code — from `exp_p1_t2_single_domain_training/data/*/valid.jsonl`.

## Methodology

### Phase 1 — Weight-space ordering (K1928)

For each of 42 Gemma 4 E4B layers × 6 permutations of `(medical, math, code)`:

1. Load `A_i ∈ R^(d_in × r)`, `B_i ∈ R^(r × d_out)` from safetensors, cast to `numpy.float32`.
2. Compute per-adapter delta `ΔW_i = (s/r) · B_i.T @ A_i.T` (orientation note: shape will match W's `(d_out, d_in)`; we use the same order as the mlx-lm forward pass).
3. For each permutation π of `(0, 1, 2)`, compute `S_π^ℓ = ΔW_{π(0)}^ℓ + ΔW_{π(1)}^ℓ + ΔW_{π(2)}^ℓ` using `numpy.float32` left-associative addition (the `+` operator in numpy).
4. Pairwise over all `C(6, 2) = 15` permutation pairs (but note S_π only depends on `π` up to the addition tree, so there are at most 2 distinct left-fold orders per unordered triple; we still compute all 15 to be safe):
   - `‖S_π - S_σ‖_F / ‖S_π‖_F`
   - `‖S_π - S_σ‖_op / ‖S_π‖_op` (largest singular value via `np.linalg.svd`)
5. Aggregate per-layer max across all pairs; report max across all layers.

**K1928 fires** iff the global max relative Frobenius gap across layers > `1e-5`.

Note: By theorem, the gap should be `~1e-7` or smaller. A result > `1e-5` would indicate either a bug in the sum or unexpected numerical pathology (e.g., catastrophic cancellation on heavily anti-aligned deltas).

### Phase 2 — Behavioral PPL (K1975)

1. Load `mlx-community/gemma-4-e4b-it-4bit` **once**.
2. Identify the q_proj LoRALinear modules. (42 layers × 1 module = 42 target modules.)
3. For each of 6 permutations π:
   - Construct **per-layer rank-`N·r = 18` LoRA** via direct sum:
     - `A_concat^ℓ = concat([A_{π(0)}, A_{π(1)}, A_{π(2)}], axis=rank_dim)` → `(d_in, 18)` numpy FP32.
     - `B_concat^ℓ = concat([B_{π(0)}, B_{π(1)}, B_{π(2)}], axis=rank_dim)` → `(18, d_out)` numpy FP32.
     - Mathematically: `(s/r) · B_concat @ A_concat = (s/r) · Σ B_i A_i`, and **the permutation order fixes the order in which rank columns are accumulated in the FP32 GEMM inner sum**, which is the behavioral analog of "summation order" at the forward pass.
   - The rank-18 concatenated LoRA cannot be inserted into the rank-6 `LoRALinear` module as-is. Instead we **override the LoRALinear module's `forward`** to compute `y_base + x @ A_concat @ B_concat * (s/r)` for the specific ordering π.
   - A simpler equivalent: replace the LoRALinear modules with a `PermOrderLoRA(N=3, rank=6, order=π)` module that stores the 3 `(A_i, B_i)` pairs and computes `y_base + Σ_{i in order π} (x @ A_i) @ B_i * (s/r)`. This exactly reproduces the forward-pass summation order.
   - Measure PPL on the 100 held-out eval rows (mixed domain).
4. `K1975` fires iff `max_{π, σ} |PPL_π - PPL_σ| / PPL_avg > 0.01` (1pp relative).

**Note on side-path forward equivalence to weight-space sum.** The forward-pass sum `Σ (xA_i) B_i` is mathematically equal to `x · (Σ B_i A_i)` — but FP32 GEMM may accumulate in a different order than a naive `Σ` over the pairs. Both answer the same question ("does the order we present the adapters change the answer?"); the forward-pass version is closer to deployment reality.

### Validation of target ≥ proxy discriminating power (F#666 sanity)
By theorem, K1975 target can only fire if K1928 proxy fires (operator norm bound). The reverse is not true — FP roundoff could produce a K1928 weight-space gap that does not propagate to measurable PPL. This matches F#666's "proxy FAIL + target PASS = finding about the proxy" semantics.

## Assumptions and caveats
- **q_proj adapters, not v_proj+o_proj**: same caveat as exp_g4_adapter_magnitude_distribution. The hypothesis concerns FP summation order, which is module-agnostic. F#627 optimal target is irrelevant here.
- **FP32 arithmetic throughout**: numpy default is FP64, but safetensors stored as FP32 and we cast. MLX side-path runs at its native dtype (likely BF16 on Metal); we compare PPL in whatever dtype MLX chose. If MLX internal dtype is BF16 not FP32, the unit roundoff is larger (`u ≈ 2^(-7) ≈ 7.8e-3`), but this affects **all 6 permutations equally** and the pairwise gap is still bounded tightly.
- **N=3 adapters**: larger N would slightly increase the error-bound constant (`(N-1)u`), but at N=10 we'd still be at `10 · 1.2e-7 ≈ 1.2e-6`, still below K1928 threshold.
- **Scale = 6.0**: inflates deltas by 6× vs scale=1.0. This 6× factor is absorbed into `‖ΔW‖_∞` and cancels in relative-gap comparisons.
- **Eval only on q_proj-adapted path**: since q_proj is the only module with LoRA, the full-model PPL reflects only the q_proj changes. No cross-module interaction to worry about.

## Antipattern scan (pre-flight checklist)
- **Composition math bug**: N/A — this experiment IS about composition, and we compute `S = ΣB_iA_i` per-layer per-permutation using the corrected formula (not the `(ΣB_i)(ΣA_i)` bug from v1). Cross-check: shape `(d_out, d_in)` is computed from `(d_out × r) @ (r × d_in)`, which is `B @ A` with correct orientation.
- **Unsafe LORA_SCALE=20**: N/A — we use the adapter's own trained scale `s = 6.0` per adapter_config.json. Not inflated.
- **Tautological routing**: N/A — no routing. All 3 adapters applied to all samples.
- **`shutil.copy` as new adapter**: N/A — we compute permutation sums directly, not copy.
- **Hardcoded `"pass": True`**: results.json populated from measured values only.
- **Proxy-model substitution**: base model is `mlx-community/gemma-4-e4b-it-4bit` per PLAN.md; MATH.md agrees.
- **KC measures wrong object**: K1928 measures the Frobenius gap of the composed delta — directly the ordering-hypothesis object. K1975 measures PPL across permutations — the behavioral downstream. Aligned.
- **Smoke reported as full**: N/A — full profiling; `is_smoke: False`.
- **KC-swap-after-failure**: pre-registered here, git-tracked before first run.
- **Eval-template truncation / base=0%**: PPL on held-out data; no eval template. No truncation risk.
- **Single-sample proxy / per-sample routing**: eval over 100 samples aggregated.

## Runtime budget
- Phase 1 (pure numpy, 42 layers × 6 perms × SVDs): ~30s–2min.
- Phase 2 (MLX, 1 model load + 6 PPL passes × 100 samples): ~15-25 min.
- **Upper timeout**: 1h (should be comfortable).

## mlx-lm version
Cited in `run_experiment.py` at runtime via `mlx_lm.__version__`.
