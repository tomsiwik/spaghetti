---
name: LEARNINGS — Cayley/Riemannian Adam on Stiefel for HRA (Gemma 4 E4B)
experiment: exp_followup_cayley_riemannian_adam
status: KILLED
date: 2026-04-24
parent_finding: F#416
child_finding: F#692
---

# LEARNINGS.md — Riemannian Adam on Stiefel for HRA (Gemma 4 E4B)

**Experiment:** `exp_followup_cayley_riemannian_adam`
**Status: KILLED** — 2026-04-24 (F#692)
**Parent:** `exp_p1_t1_householder_vs_lora` (F#416, killed)

---

## Core learning

Strict-Stiefel Riemannian Adam (QR retraction; first-order equivalent to Cayley per
Absil 2008 §4.1.1) does not rescue HRA convergence at `r ≪ d` — it makes it *worse*.
The canonical-metric projection [Edelman–Arias–Smith 1998] removes the normal component
of the Euclidean gradient, which at r=16, d=2560 is ≳90 % of the gradient energy.
Euclidean Adam's "wrong" off-manifold steps carry more useful signal per iteration
than Riemannian's "right" on-manifold steps at this rank.

## Why the hypothesis failed (impossibility structure)

1. **Signal bottleneck at `r ≪ d`.** For the canonical Stiefel metric
   [Edelman–Arias–Smith 1998 §2.2], `‖Π_T(G)‖_F / ‖G‖_F ≲ √(r/d)` for isotropic G
   with `r ≪ d`. At `(r, d) = (16, 2560)` this is ≈ 0.08 — strict Stiefel retains only
   8 % of the per-step Euclidean gradient. LR=5e-5 tuned for LoRA is effectively
   LR=4e-6 for Riemannian-projected HRA. Bécigneul–Ganea [2019, Thm 4.1] guarantees
   `O(1/√T)` convergence on geodesically-convex objectives, but that asymptotic bound
   does not save you from a 10× per-step magnitude gap at finite step budgets.
   The optimizer is **under-powered by construction**, not by tuning.

2. **HRA reflection is self-normalising per vector, so Stiefel is optional.**
   `H_i(x) = x − 2 (v_i·x) v_i / ‖v_i‖²` divides by `‖v_i‖²` explicitly; the
   forward pass is invariant to the magnitude of each `v_i`. Only the *span* of
   `{v_i}` enters the output — i.e. the **Grassmannian** point, not its **Stiefel**
   representative. Partitioned-QR Grassmannian init (F#415, Theorem 2) already
   guarantees the span is correct at step 0; enforcing mutual orthonormality
   *between* `v_i`s during training adds no mechanical role in the forward pass.
   The Stiefel constraint is tighter than HRA needs.

3. **HRA_euc matches LoRA on Gemma 4 E4B MMLU (both 43.3 %).** Parent F#416's
   −6 pp MMLU regression on Qwen3-4B did **not** reproduce on Gemma 4 E4B — it was
   within n=50 sampling noise. The "multiplicative rotation disturbs MMLU directions"
   narrative from F#416 §interpretation is **not supported** by this more-controlled
   Gemma 4 data. Take-home: the parent kill was correctly diagnosed as a convergence
   failure (K1013) but incorrectly extended into an MMLU-structure story.

## Confirmed (structural)

- **QR retraction on Stiefel is numerically exact on 4-bit Gemma 4 E4B:**
  `max ‖VVᵀ − I_16‖_F = 8.5 × 10⁻⁷` over 42 layers × 150 steps.
  `mx.linalg.qr` with `stream=mx.cpu` is a production-grade Stiefel retraction in MLX.
  Cayley (Wen–Yin 2013) vs QR (standard Absil 2008) differ by `O(τ² ‖Ξ‖²)` on
  St(r,d) at step size τ=η, which is negligible at our LR.

- **Parent F#416 K1013 reproduces on Gemma 4 E4B** (HRA_euc DNF at 150 steps, same as
  LoRA-on-v_proj DNF at 150 steps). The parent failure is NOT Qwen3-specific —
  K_euc_control PASS rules out the "proxy-model pathology" alternative explanation.

## Implications for siblings

- **`exp_p1_t1_algorithm_bakeoff`** (sibling, open): remove "Cayley/Riemannian Adam
  on Stiefel" from the candidate pool at the current Pierre rank budget (r = 6 or
  r = 16 on d = 2560). Prefer Givens rotations [Finding #413] or plain
  LoRA + partitioned-QR Grassmannian init [F#415]; neither triggers the signal-
  bottleneck mode because Givens parameterises a compact orthogonal subgroup and
  Euclidean-LoRA never projects.

- **`exp_followup_adapter_orthogonality_audit`** (sibling, still open): must measure
  `‖VVᵀ − I‖_F` drift for *both* Euclidean and Riemannian variants in the same run.
  This experiment only measured Riemannian (which is exact by construction); we
  need the Euclidean-drift baseline to quantify whether the accumulated drift is
  (a) small enough to be irrelevant (our interpretation, consistent with HRA_euc
  matching LoRA MMLU here), or (b) large enough that the comparison is unfair.

- **Any future Stiefel-based adapter work** must derive `r*` from the signal-
  bottleneck inequality `r/d ≥ threshold` *before* committing to a Riemannian
  optimiser. Typical hidden sizes (d = 2048-4096) imply r ≥ 500 for Riemannian-Adam
  parity with Euclidean at the same LR — that's outside the LoRA rank regime entirely.
  Alternative: rescale LR by `√(d/r)` per-optimiser, but that breaks the
  apples-to-apples hyperparameter comparison.

## Proposed antipattern (for review)

**"Riemannian optimiser at `r ≪ d`."** Projecting to the Stiefel tangent at low
rank discards ~(1 − r/d) of the gradient norm, silently starving the optimiser.
Symptom: retraction fidelity perfect (`VVᵀ = I` to float32), but training-loss
plateau is strictly *worse* than Euclidean baseline. Fix: (a) scale LR by `√(d/r)`,
(b) raise the rank-to-dim ratio until `r ~ d/2` (e.g. NoPE head-slice), or
(c) pick a parameterisation where the Stiefel constraint has a mechanical role
(full-rank OFT, not rank-r HRA).

## Instrumentation gap (file as follow-up)

- Did not measure `‖VVᵀ − I‖_F` drift for HRA_euc during training. Would clarify
  whether "Euclidean drift" is meaningfully off-manifold or stays within float32
  epsilon by serendipity. Cheap to add to `train_hra_euc` (one Frobenius norm
  per step). Belongs in `exp_followup_adapter_orthogonality_audit`.

## Contribution to framework

- **Validates the target-gated kill rule [F#666].** The proxy metric (K1559,
  convergence-step ratio) went DNF — but *so did LoRA's*, making the proxy
  uninformative at this step budget. The target metric (K_target, MMLU accuracy)
  is what actually killed: a −16.7 pp regression on the downstream task. Without
  F#666's rule "proxy FAIL + target FAIL = KILL", this might have been logged
  as INCONCLUSIVE (proxy mis-calibrated at 150 steps). Instead it is a clean KILL
  with a mechanistic explanation.

- **Validates `mx.linalg.qr(stream=mx.cpu)` as a reliable Stiefel retraction on
  Gemma 4 E4B 4-bit.** Reusable building block for any future manifold-constrained
  adapter work in MLX.

- **Separates F#416's correctly-diagnosed convergence claim (K1013) from its
  incorrectly-extended MMLU claim (K1012).** The convergence story stands and
  reproduces; the "multiplicative rotation corrupts MMLU" story does not
  generalise across base models. Future adapter-vs-LoRA experiments should
  treat these as independent claims.

---

## References

### Literature (anchor papers)

- **Wen & Yin (2013).** "A feasible method for optimization with orthogonality
  constraints." arXiv:1208.4298. *Provides the low-rank Cayley retraction
  `V⁺ᵀ = Vᵀ − τ U (I + (τ/2) Yᵀ U)⁻¹ Yᵀ Vᵀ`* with `2r × 2r` inverse per step.
  Used in MATH.md Theorem 1; QR retraction is first-order equivalent at our LR.

- **Bécigneul & Ganea (2019).** "Riemannian adaptive optimization methods."
  arXiv:1810.00760, ICLR 2019. *Riemannian Adam with tangent-transport momentum;
  Thm 4.1 `O(1/√T)` convergence on g-convex objectives.* Referenced in MATH.md
  Theorem 3; §3.2's simple projective momentum transport is what our
  implementation uses. Their convergence rate is asymptotic — this experiment
  shows the *finite-budget per-step signal loss* can dominate at `r ≪ d`
  regardless of the asymptotic guarantee.

- **Edelman, Arias & Smith (1998).** "The geometry of algorithms with orthogonality
  constraints." SIAM J. Matrix Anal. Appl. 20(2):303–353. *Canonical-metric tangent
  projection `grad L = G − ½ V(Vᵀ G + Gᵀ V)` on St(r,d) (Theorem 2 in MATH.md).*
  Used throughout PAPER.md §Interpretation — this is the exact projection whose
  signal loss scales as `√(r/d)`.

- **Absil, Mahony & Sepulchre (2008).** *Optimization Algorithms on Matrix
  Manifolds*, Princeton University Press §4.1.1. *First-order equivalence between
  QR and Cayley retractions on Stiefel.* Justifies swapping Cayley for
  `mx.linalg.qr` in MLX (Cayley requires a dense `d × d` matrix inverse; QR is
  thin and CPU-streamable).

- **Yuan et al. (2024).** "HRA: Householder Reflection Adapters." arXiv:2405.17484.
  *Defines the HRA reflection formula `H_i(x) = x − 2(v_i·x)v_i/‖v_i‖²` used here.*
  Its per-vector self-normalisation is what makes Stiefel constraint optional for
  HRA (impossibility-structure point #2).

### Findings (repo anchors)

- **F#415 — Householder chain orthogonality algebraically exact.** Partitioned-QR
  Grassmannian init Theorem 2: orthogonal subspace init implies
  `⟨range(H₂−I), ker((H₁−I)^⊥)⟩ = 0`. Shows that once `{v_i}` span a chosen
  subspace, basis choice within that subspace does not affect the output — Stiefel
  is a tighter invariant than HRA needs (impossibility point #2).

- **F#416 — HRA vs LoRA at equal rank (parent kill).** K1012/K1013 FAIL on
  Qwen3-4B. This experiment's K_euc_control confirms K1013 reproduces on Gemma 4
  (convergence failure is not Qwen3-specific) but K1012 does NOT reproduce
  (MMLU regression was within n=50 noise). Refines F#416's impossibility structure:
  "Euclidean Adam causes slow convergence" ✓; "multiplicative reflections disturb
  base MMLU representations" ✗.

- **F#666 — Target-gated kill rule.** Requires BOTH proxy and target KC to fail
  before KILL; BOTH to pass before SUPPORTED. Applied here: K1559 FAIL (proxy,
  step-ratio DNF) ∧ K_target FAIL (MMLU −16.7 pp) ⇒ KILLED. Without this rule,
  the K1559-only reading would be INCONCLUSIVE because LoRA also DNFed.

- **F#692 — THIS experiment's child finding.** "Cayley/Riemannian Adam on Stiefel
  does not rescue HRA at `r ≪ d` on Gemma 4 E4B." Records the signal-bottleneck
  impossibility structure and the −16.7 pp K_target FAIL.

### Grassmannian-vs-Stiefel-optionality note

The review event payload cited `F#665` for "Grassmannian-vs-Stiefel orthogonality
optionality." F#665 in the DB is actually about degenerate-routing-collapse
(`exp_skip_list_composition_test`) — unrelated. The real anchor for the
"HRA only needs the Grassmannian span, not the Stiefel representative" claim is
**F#415 Theorem 2** (subspace-basis invariance). PAPER.md §Impossibility
Structure #2 (line 79) explicitly notes the F#665→F#415 correction inline;
no further document edit needed.
