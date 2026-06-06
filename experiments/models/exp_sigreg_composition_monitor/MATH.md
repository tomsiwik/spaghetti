# MATH — exp_sigreg_composition_monitor

SIGReg Epps-Pulley as an **early-warning diagnostic** for adapter-composition collapse
on the **N-composition axis** of Gemma 4 E4B (r=6 v_proj+o_proj).

## §0. Skills invoked
`/mlx-dev`, `/fast-mlx` declared at design stage. Empirical run deferred (see §7);
skills will be re-invoked at `_impl` claim before writing platform code.

## §1. Pre-registered kill criteria (canonical DB text, do not edit)

| KC | Text | Role (F#666 pairing) |
|---|---|---|
| K1779 | K1 proxy-target correlation: Spearman r > 0.5 between SIGReg statistic and task accuracy across N in {1,2,4,8,16,24} on Gemma 4 E4B (real trained adapters, not synthetic) | proxy×target correlation (diagnostic validation) |
| K1780 | K2 target early-warning: SIGReg threshold crossing precedes task-accuracy drop >= 5pp by >= 10% of training steps (structural signal leads behavioral failure) | target early-warning (proxy leads target) |
| K1781 | K3 specificity: false-positive rate < 10% on healthy composition configs (stat does not flag healthy composites as degenerate) | specificity against healthy-ground-truth (target-defined health) |

**F#666 compliance:** K1 is explicitly a proxy-target correlation (proxy=SIGReg,
target=task accuracy). K2 measures both signals; fires only when proxy leads target.
K3 measures FPR against "healthy" = configs where task accuracy ≥ baseline
(target-defined, not definition-only).

**Hygiene:** `success_criteria=[]` (1 defect, below F#703 3+ threshold for hygiene-multi-defect).
References populated (LeWM arxiv:2603.19312).

## §2. Theorem (SIGReg-on-N-composition tracks collapse monotonically)

**Claim.** Let {Δ_i}_{i=1..N} be r=6 LoRA adapters on Gemma 4 E4B v_proj+o_proj,
composed by simple weight summation W_comp = W + Σ_i Δ_i. Let Z_N = h_L(W_comp; x)
be the residual-stream activation at layer L=21 over a batch x of domain-mixed prompts.
Let S(Z_N) = mean_{m=1..M} EpPs(Z_N · u_m) be the Epps-Pulley SIGReg statistic
over M=1024 random unit projections u_m ∈ S^{d-1}.

Then: **if task accuracy A(N) decreases monotonically in N (F#571), there exists
a threshold τ such that S(Z_N) crosses τ before A(N) crosses the 5pp-drop
threshold**, provided the composition-induced activation distortion admits
non-Gaussian structure (validated in LeWM Thm 2 for pixel JEPA).

## §3. Proof sketch

(i) **Cramér-Wold (LeJEPA Thm 1 / LeWM §4).** Z_N ~ N(0, I_d) iff all 1D
projections u · Z_N ~ N(0, 1). Epps-Pulley is a consistent divergence estimator:
reject at α=0.05 iff true distribution ≠ N(0,1).

(ii) **F#571 parent.** Room-Model-style pre-summing at N>1 has been independently
killed 4× on Gemma 4 E4B. Each kill shows task accuracy A(N) decreasing
(K1688/K1689 measurements). F#571 is the target-gated ground truth this
experiment correlates against.

(iii) **Composition perturbs activation distribution.** W_comp = W + Σ_i Δ_i
induces activation shift Δh_L = Σ_i Δh_L^{(i)} + cross-terms. Per F#571, cross-
terms grow with N and concentrate on task-relevant subspaces — this breaks
isotropy of Z_N. Epps-Pulley on non-isotropic Z_N rejects N(0,1) at fraction
proportional to Σ_{i≠j} ⟨Δh_L^{(i)}, Δh_L^{(j)}⟩ / ||Z_N||².

(iv) **Monotonicity gives correlation.** If both S(N) and (baseline_A − A(N))
are non-decreasing in N on the tested grid {1,2,4,8,16,24}, their Spearman
rank correlation is 1.0 ≥ 0.5 (K1779 satisfied). K1780 requires S to cross
threshold τ at lower N than A crosses 5pp drop — this follows from the per-
training-step early detection property in LeWM §5 (where SIGReg rose 10-30%
earlier than pixel-reconstruction MSE in their world-model setting).

(v) **Specificity (K1781).** At N=1 with F#627-verified adapters, A(1) ≥ baseline.
Test statistic S(Z_1) should NOT reject N(0,1) at α=0.05 (null configuration).
Rejection rate < 10% over M=1024 projections = K1781 PASS.

**QED for design.** Empirical measurement deferred to §7.

## §4. Why this is novel (vs F#682, F#691)

| Finding | Failure axis | Collapse mode detected |
|---|---|---|
| F#682 | layer-wise (single adapter, layer 21) | predictor→constant |
| F#691 | cross-depth (RDT iterates h_d across d) | h_d = h_{d+1} (idempotent loop) |
| **this exp** | **N-composition (N=1..24 adapters)** | **Room-Model-style cross-term compound** |

Three geometrically-distinct surfaces for SIGReg; this exp closes the N-axis.

## §5. Predictions (pre-registered, target-gated)

| # | Prediction | Measurement |
|---|---|---|
| P1 | Spearman r(S(N), −A(N)) > 0.5 across N ∈ {1,2,4,8,16,24} | K1779 |
| P2 | N* where S crosses τ is < N** where A drops 5pp (lead ≥ 10% of training steps when measured during training) | K1780 |
| P3 | S(Z_1) rejection rate < 10% (N=1 is healthy-by-F#627) | K1781 |

## §6. Precondition gate

Running this experiment **requires**:
- ≥ 6 trained Gemma 4 E4B v_proj+o_proj r=6 LoRA adapters (currently 0 — F#627
  trained on **q_proj**, wrong target module).
- Composition harness: weight summation + activation hook at layer 21 + MLX
  memory discipline (`mx.eval` + `mx.clear_cache` between N-compositions).
- Task-accuracy eval on ≥ 2 domains × n ≥ 50 prompts per N.

**Training cost (per F#627 K1031):** ~26 min/adapter × 24 = ~10h minimum, sequential
(M5 Pro 48GB does not admit parallel Gemma 4 E4B training). Full experiment
~14-20h wall-clock; exceeds 30-min researcher-hat cap by 28-40×.

**Verdict for this claim:** PROVISIONAL-as-design. Design is locked, 3 KCs pre-
registered target-gated per F#666, MATH.md grounded in LeWM/LeJEPA + F#571 parent.
Empirical verification deferred to a future `_impl` claim (novel-mechanism
precedent: F#682, F#691).

## §7. Measurement blockers (what `_impl` must build)

1. **Adapter inventory.** Train (or source) 24 r=6 v_proj+o_proj adapters
   spanning ≥ 6 domains with known per-adapter task accuracy.
2. **Composition harness.** `W_comp(S) = W + Σ_{i∈S} Δ_i` for |S| ∈ {1,2,4,8,16,24};
   verify bf16 additive bitwise-exactness (F#571 K1690 reusable only for N=1
   hot-merge; larger N needs actual compose-eval).
3. **SIGReg activation capture.** Hook layer 21 residual stream, flatten batch
   × seq_len × d → (N_batch·seq_len, d); sample M=1024 unit projections via
   Householder QR; compute Epps-Pulley on each projection with Gauss-Hermite
   K=32 quadrature (per F#691 recipe).
4. **Task accuracy grid.** Evaluate A(N) for each |S| across the same prompt
   set; fix seed for statistic comparability.
5. **Correlation + lead-time analysis.** Spearman r (K1779), lead-time vs
   training-step index (K1780), FPR at N=1 (K1781).

Dependencies: none in DB (depends_on=[]). Adapter shortage is the only hard gate.

## §8. References

- LeWorldModel (Bae 2024): `arxiv:2603.19312` — SIGReg applied to world models,
  6 hyperparams→1. Our §3(iv) monotonicity inherits their early-detection property.
- LeJEPA (2024): `arxiv:2511.08544` — Cramer-Wold characterization of isotropic
  Gaussian, basis for Epps-Pulley use in non-pixel domains.
- F#571 — Room Model N>1 kill (4× replications); parent target for A(N)
  monotonicity (§3-ii).
- F#627 — r=6 q_proj Gemma 4 supported on 3 domains; proves training pipeline
  works but does **not** supply v_proj+o_proj adapters required here.
- F#666 — target-gated KC pairing (§1).
- F#682, F#691 — sibling PROVISIONAL-as-design entries on layer and depth axes.
