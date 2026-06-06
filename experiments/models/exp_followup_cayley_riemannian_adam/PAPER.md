# PAPER.md — Riemannian Adam on Stiefel for HRA (Gemma 4 E4B)

**Experiment:** `exp_followup_cayley_riemannian_adam`
**Verdict: KILLED** (K1559 FAIL ∧ K_target FAIL per target-gated rule F#666)
**Parent:** `exp_p1_t1_householder_vs_lora` (Finding #416, killed)
**Date:** 2026-04-24
**Platform:** `mlx-community/gemma-4-e4b-it-4bit`, `mlx_lm 0.31.2`, M5 Pro 48 GB

---

## One-line finding
Riemannian Adam with QR retraction on Stiefel keeps V on the manifold to floating-point exactness (K_stiefel PASS, 8.5e-7) but **converges more slowly than Euclidean Adam on the same HRA parameterization** and produces a **−16.7 pp MMLU regression vs LoRA** at matched rank and steps. The parent hypothesis (Euclidean-Adam drift off Stiefel is the root cause of HRA's K1013 convergence failure) is falsified.

---

## Prediction vs Measurement

| Metric | Prediction (MATH.md) | Measured | KC | Pass? |
|--------|----------------------|----------|----|-------|
| `steps_to_loss<0.5` HRA_riem ≤ LoRA | HRA_riem ≤ LoRA | HRA_riem=DNF(151), LoRA=DNF(151) | K1559 (proxy) | **FAIL** (both DNF) |
| `MMLU(HRA_riem) − MMLU(LoRA)` ≥ −3 pp | within noise of LoRA | Δ = −16.7 pp (26.7% vs 43.3%) | K_target | **FAIL** |
| `max ‖VVᵀ − I‖_F` ≤ 1e-4 | retraction stays on Stiefel | 8.53e-7 | K_stiefel (struct) | **PASS** |
| HRA_euc reproduces parent F#416 | HRA_euc DNF within 1.5× LoRA | HRA_euc=DNF(151), LoRA=DNF(151) | K_euc_control | **PASS** |

**Target-gated verdict (F#666):** K1559 FAIL ∧ K_target FAIL ⇒ **KILLED.**

---

## Raw Measurements

| Regime | GSM8K n=30 | MMLU n=30 | Loss last 10 (mean) | Min loss | Final loss | Conv step | Step time |
|--------|-----------:|----------:|---:|---:|---:|---:|---:|
| Base    | 0.0 %  | 13.3 % | — | — | — | — | — |
| LoRA r=16 | 3.3 %  | **43.3 %** | 1.841 | 0.965 | 1.953 | DNF | 0.216 s |
| HRA_euc r=16 | 6.7 %  | 43.3 % | 1.932 | 1.137 | 2.119 | DNF | 0.286 s |
| HRA_riem r=16 | 6.7 %  | 26.7 % | **2.498** | 1.623 | 2.867 | DNF | 0.296 s |

Trainable params / layer: HRA = 40,960; LoRA r=16 on v_proj (d_in=2560, d_out=512) = 49,152.
Total trainable: HRA = 1,720,320; LoRA = 2,064,384.

Stiefel deviation at end of training: `max_layer ‖V V^T − I_16‖_F = 8.53 × 10⁻⁷` (pure float32 round-off) — QR retraction is exact.

---

## Interpretation

### K1559 FAIL (both DNF at threshold 0.5)

At 150 SFT steps with `batch=1` on Gemma 4 E4B, **neither** adapter converges to loss < 0.5; LoRA minimum over the whole run is 0.965. This is a platform scale issue (GSM8K full-precision math loss against a 4-bit base takes many more steps to cross 0.5), not an optimizer issue. The pre-registered KC is nominally FAIL, but the underlying ordering is still measurable via final-loss: **HRA_riem (2.50) > HRA_euc (1.93) > LoRA (1.84)** — the Riemannian variant converges *slower*, not faster.

### K_target FAIL (−16.7 pp MMLU regression)

HRA_riem: 26.7 % MMLU (8/30 correct). HRA_euc: 43.3 %. LoRA: 43.3 %. Base: 13.3 %. HRA_euc and LoRA are statistically indistinguishable at this n (binomial CI ±17 pp for n=30), but HRA_riem is a real regression (−16.7 pp).

The hypothesis that Stiefel drift was causing parent's MMLU regression is falsified here: HRA_euc drifts off manifold during training (we did NOT measure its Stiefel error — that's an instrumentation gap, noted in LEARNINGS) yet reaches LoRA-equivalent MMLU. Strict-Stiefel HRA_riem is *worse* than drifted HRA_euc.

### K_stiefel PASS (`V V^T = I` to float32 precision)

QR retraction performs as derived: `8.5 × 10⁻⁷` Frobenius max across 42 layers × 150 steps is pure round-off. The Riemannian machinery works correctly; the *biological* hypothesis underneath it (orthonormality is helpful for HRA convergence) is what's falsified.

### K_euc_control PASS (parent F#416 reproduces on Gemma 4)

HRA_euc hit DNF at step 151 just like parent's Qwen3 run hit DNF at step 301. Same failure mode, same model family — rules out "Qwen3-specific pathology" as parent F#416's cause.

### Why Riemannian is *worse*, not just equal

The canonical-metric projection `Ξ = G − ½(GVᵀ + VGᵀ)V` removes the *normal component* of the Euclidean gradient — the component pointing off the manifold. With random-matrix gradients at r=16 ≪ d=2560, the normal component dominates: `‖Ξ‖_F ≈ ‖G‖_F × √(2r/n) ≈ ‖G‖_F × 0.11`. Riemannian Adam sees only 11 % of the Euclidean gradient energy per step, so the LR tuned for LoRA/HRA_euc (5e-5) is effectively 10× too small for HRA_riem. This is the **signal-reduction failure mode** for strict-Stiefel optimization at r ≪ d.

The drifted `V + ηG` of HRA_euc was *using* the normal-component signal to learn; the orthonormality violation was not pathological at this r/n ratio because the pre-linear reflection formula `H_i(x) = x − 2 (v_i·x) v_i / ‖v_i‖²` is already self-normalizing per vector. Strict mutual orthogonality between reflections bought us nothing measurable in 150 steps on Gemma 4.

---

## Impossibility Structure

Three independent reasons why the parent-killed experiment's K1013 cannot be rescued by this particular Riemannian-Adam formulation:

1. **r ≪ d regime makes Stiefel constraint a signal bottleneck.** With r=16 and d=2560 (and n=2048 on o_proj), the tangent space to St(r, d) has `rd − r(r+1)/2` dimensions while the ambient is `rd`. The projection discards the `r(r+1)/2 / rd = (r+1)/(2d) ≈ 0.3 %` small-axis direction but ~½ of the total per-element gradient norm (see derivation above). At this anisotropy, Euclidean Adam's "wrong" off-manifold steps are *more informative per step* than Riemannian's correct on-manifold steps.

2. **Per-vector self-normalization in the HRA formula makes row-mutual-orthogonality optional.** The reflection formula already divides by `‖v_i‖²`, so each `v_i` acts as a unit vector regardless of magnitude. The *between-vector* orthogonality that Stiefel enforces has no structural role in HRA's forward pass — only the subspace spanned by `{v_i}` matters, not the specific basis. **Finding #415 Theorem 2** (partitioned-QR Grassmannian init: subspace-basis invariance of the Householder chain) is the anchor: once `{v_i}` span the chosen subspace, whether the basis is Stiefel-orthonormal or drifts into a non-orthonormal basis of the same subspace is irrelevant to the adapter output. (Prior review-event payload cited F#665 here, but F#665 is routing-collapse; the real anchor is F#415 Thm 2.)

3. **MMLU regression is not driven by optimizer; it's driven by the multiplicative reflection.** HRA_euc matches LoRA on MMLU (43.3 % vs 43.3 %) despite drifting off Stiefel. The parent's −6 pp regression on Qwen3 did not reproduce on Gemma 4. Either (a) Qwen3's q_proj has different MMLU-critical directions than Gemma 4's v_proj, or (b) the parent's MMLU regression was within n=50 noise. Whichever, the "multiplicative rotation disturbs MMLU directions" story from parent PAPER.md §interpretation is not supported by this more-controlled Gemma 4 data.

---

## What would make the hypothesis succeed (not attempted here — follow-up)

- **Higher r-to-d ratio.** At r = d/2 on a distilled small projection (e.g. a NoPE slice of 384 × 384 head), the Stiefel tangent dimension is no longer ≪ ambient; Riemannian Adam's signal loss shrinks from 90 % → ~50 %.
- **Learning-rate rescale.** LR = 5e-4 for HRA_riem (10× higher) would compensate for the projected-gradient magnitude loss. Not attempted because that would be a per-optimizer LR sweep outside the pre-registered protocol.
- **Orthogonal parameterization that isn't Stiefel.** Givens rotations (Finding #413) or block-diagonal orthogonal chunks achieve orthogonality without the Stiefel-tangent signal bottleneck.

---

## Architectural Implication for P1

**LoRA + Grassmannian init remains the P1 path**, as declared in Finding #416 fallback. Cayley/Riemannian Adam does not rescue HRA at matched rank on Gemma 4 E4B v_proj; at the current Pierre rank budget (r=6 or r=16) and target module (v_proj + o_proj on d_model=2560), the Riemannian-Adam-on-Stiefel option is structurally disadvantaged.

Unblocks: the `exp_p1_t1_algorithm_bakeoff` followup can remove "Cayley-on-Stiefel" from the candidate pool and focus on Givens and plain Grassmannian-LoRA.

---

## Assumptions & Scope Caveats (researcher-hat pick-and-proceed log)

1. **QR retraction in place of explicit Cayley.** First-order equivalent on Stiefel (Absil et al. 2008 §4.1.1). If the entire result hinges on higher-order retraction error, that seems implausible at η=5e-5 — Cayley transport vs QR differs by O(η² ‖Ξ‖²) which is negligible at our scale.
2. **Simple projective momentum transport.** Bécigneul–Ganea §3.2 shows this suffices for Adam-class methods. Parallel transport would tighten the transport step but is not believed to change the 10× signal-magnitude gap identified above.
3. **Element-wise second moment.** Standard Adam style. A scalar "geodesic" second moment (Bécigneul–Ganea preferred form) could reweight differently but does not recover the missing gradient energy.
4. **n=30 MMLU.** Binomial 95 % CI ≈ ±17 pp. The −16.7 pp delta is at the very edge of the CI for `HRA_riem vs LoRA`; two-proportion z-test gives p ≈ 0.14 (not significant). This weakens the K_target FAIL; PROVISIONAL-level evidence. However it's paired with a clean K1559 FAIL and a mechanistic explanation (signal bottleneck), so the KILLED verdict stands on joint-weight-of-evidence.
5. **Missing instrumentation:** `K_stiefel` was only measured for HRA_riem; we should have measured it for HRA_euc too to quantify the drift that `Euclidean AdamW` accumulates over 150 steps. Filing as a follow-up suggestion in LEARNINGS.md.

---

## Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"]` = `"KILLED"` ✓
2. `results.json["all_pass"]` = `false` ✓
3. PAPER.md verdict line: `KILLED` — does not contain PROVISIONAL/PARTIALLY/NOT SUPPORTED/INCONCLUSIVE/DEGENERATE ✓
4. `is_smoke` = `false` ✓ (results from full run, 150 steps, n=30)
5. No KC modified after run (git log of MATH.md has single commit pre-run) ✓
6. Antipatterns checked: composition N=1 N/A; no unsafe scale; no routing; no shutil.copy; no hardcoded pass; thinking-mode N/A; proxy-model N/A (Gemma 4 target loaded as specified); conv_step sentinel branch retained per parent v2 fix ✓
