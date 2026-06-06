# MATH.md: Hard Top-1 Gumbel MoE Routing for Absolute Gradient Isolation

## Experiment Type

Type 1 (Proof verification) — pre-registered quantitative predictions under Theorem 1.
Single DB kill criterion: **K861** "Median M2P quality drops < 25% of SFT"
(PASS iff median_quality_ratio ≥ 0.25).

## Prior Results Cited

- **Finding #341** (exp_m2p_distillation_toy, KILLED): Shared-centroid collapse of
  M2P-generated B-matrices under round-robin training (|cos|=0.9956, repeat=-329%).
- **Finding #342** (exp_m2p_domain_conditioned, KILLED): Additive domain-embedding
  injection reduced |cos| only 0.0171 (target ≥0.0956) → J(θ) low-rank → additive
  signal is **inert** through attention compression.
- **Finding #574** (exp_m2p_moe_routing V2, 2026-04-18): **Soft** MoE without aux
  load-balance loss collapses (m̄_route=0.343 ≈ uniform 0.25) — uniform is a
  saddle minimiser of the round-robin loss for soft routing.
- **Gumbel-Softmax / Concrete** (Jang et al., arXiv:1611.01144, 2016): Hard top-1
  via Straight-Through Estimator gives zero gradient flow to non-selected experts.
- **Switch Transformer** (Fedus et al., arXiv:2101.03961): Hard top-1 routing with
  auxiliary load-balance loss prevents expert collapse at scale; without that loss,
  expert collapse is the default failure mode.

## A. Why Hard Top-1 Is Mathematically Distinct from Soft MoE (Finding #574)

Under **soft** MoE, memory = Σ_e w_e · expert_e(mem) with w_e = softmax(logits).
Every expert receives gradient ∝ w_e; a saddle at w_e ≈ 1/N_e is stable because
∂L/∂w_e is symmetric across experts when the task distribution is uniform
round-robin (proven in Finding #574).

Under **hard top-1 via STE**, forward uses a one-hot mask `e*(d) = argmax_e logits(d)`.
The non-selected experts receive ZERO activation in the forward pass.
Backward gradient is injected through `soft - stop_gradient(hard - soft)`:
only the selected expert's parameters receive a **gradient of meaningful magnitude**
on each step (the non-selected experts receive only router-logit gradient via
softmax, not expert-parameter gradient).

**Theorem 1 (Gradient Isolation under STE).** Let f_e(·) be expert e with
parameters θ_e. The per-step loss for domain d is
L_d = ℓ(Σ_e r_e(d) · f_e(mem_d))
where r_e(d) is the STE-composed routing weight. Then
∂L_d/∂θ_e = r_hard_e(d) · ∂ℓ/∂f_e(mem_d)·∂f_e(mem_d)/∂θ_e.
Because r_hard_e(d) ∈ {0, 1} is one-hot on e*(d), only θ_{e*(d)} receives a
non-zero gradient at step d.

*Proof.* The forward quantity is y = Σ_e (sg(hard_e − soft_e) + soft_e) f_e(·) =
Σ_e hard_e · f_e(·). Chain rule: ∂y/∂θ_e = hard_e · ∂f_e/∂θ_e. Zero for e ≠ e*(d). □

## B. Predicted Failure Mode: Router Collapse

Theorem 1 guarantees **parameter-space isolation** per step. But across steps the
router itself is trained (receives softmax gradient from the task loss). Without
an auxiliary load-balance loss, the router can collapse so that multiple domains
route to the same expert — one specialist expert + (N_e − 1) dead experts.

**Lemma 1 (Router-Collapse as Stable Equilibrium).** If at step 0 the router
happens to argmax to expert e* for both domains d₁ and d₂ with SFT losses
ℓ_SFT(d₁) < ℓ_SFT(d₂), gradient descent on θ_{e*} drives it toward the domain-d₁
optimum (higher gradient magnitude). Once converged there, domain d₂ receives
no improvement; its argmax remains e* because the router gradient is dominated
by d₁'s loss (which is now lower on e*). No other expert is ever activated.

*Proof sketch.* At step t > 0, ∂L_{d₁}/∂logit_{e*} = softmax-gradient >0,
and by ordering ℓ_SFT(d₁) < ℓ_SFT(d₂) after a few steps, L_{d₁}(θ_{e*}^{(t)}) <
L_{d₂}(θ_{e*}^{(t)}), so more loss reduction is attainable for d₁, reinforcing
e*. □

## C. Pre-Registered Predictions (Bound the Experiment Before Running)

Locked before run. No KC edits after data.

### C.1 DB KC #861 — Median quality
- **Predicted range:** 0.15 ≤ median ≤ 0.40 (lower than additive conditioning's
  0.473 because Finding #341/342 median was dominated by 4 healthy domains; hard
  routing commits early and cannot borrow representations across domains).
- **Pass condition (K861):** median_quality_ratio ≥ 0.25.
- **Expected verdict:** borderline — likely FAIL or marginal PASS. Finding
  #574 evidence (27.8% on an earlier hard-MoE run) is consistent with this band.

### C.2 Diagnostic D1 — Router expert uniqueness
- **Predicted:** n_unique_argmax_experts ∈ {1, 2, 3} out of 5.
- **Basis:** Lemma 1 + no aux load-balance loss (verified in Finding #574 for
  soft MoE at 3/4; hard top-1 expected to collapse further because no gradient
  spreads to dead experts).
- **Implication if 4-5 unique:** Lemma 1 is wrong or Gumbel noise prevented
  collapse; would be a surprising positive outcome worth resurrecting.

### C.3 Diagnostic D2 — B-matrix |cos|
- **Predicted:** mean |cos| ∈ [0.80, 0.97] — better than additive (0.9785) ONLY
  if multiple experts specialise (D1 shows ≥ 3 unique). If 1-2 unique, expect
  |cos| ≥ 0.98 (effectively one centroid + tiny perturbation from Gumbel).

### C.4 Structural guarantee — Grassmannian A
- **Predicted:** max |cos| ≤ 1e-5. QR orthogonality is unchanged by MoE routing
  (does not touch A-matrices). Type-1 PASS guaranteed.

### C.5 Per-domain min quality
- **Predicted:** worst domain ≤ -2.0 (i.e., catastrophic on at least one domain).
  Expected worst = "repeat" (lowest SFT loss → smallest gradient signal) or
  whichever domain lost router arbitration.

## D. Why Hard Top-1 Does NOT Rescue the M2P Design Even If K861 Passes

If D1 shows router collapse (< 5 unique experts), K861 can pass by accident:
surviving experts become competent on 2-3 domains, pulling median above 0.25
while the displaced domains fail catastrophically (D5). A K861 PASS under
router collapse would be an **implementation-level pass with zero architectural
validation** — the experiment must then be re-labeled as a D1 KILL (routing
mechanism failed even if aggregate quality metric passes).

This is the **metric-swap failure mode** (audit-2026-04-17 tag): reading K861
alone without D1/D5 would mis-interpret router collapse as success. All three
must be reported together.

## E. Code ↔ Math Crosswalk

| MATH concept | run_experiment.py |
|---|---|
| Gumbel noise (training only) | lines 216–225 (gated by `self.training_mode`) |
| STE one-hot forward, soft backward | `route_weights = sg(hard - soft) + soft` |
| Per-expert forward | `expert_outputs = [...]; memory = Σ route_weights[i] * expert_outputs[i]` |
| K861 measurement | `phase_evaluate_m2p` with `m2p.training_mode = False` |
| D1 router uniqueness | `phase_router_check` → `n_unique_experts` |
| D2 B-matrix |cos| | `phase_composition_check` → `m2p_b_cos_mean` |
| Grassmannian A (D4) | `phase_grassmannian` → `grassmannian_cos_max` |

## F. Assumptions

1. **Round-robin schedule** (steps cycle d=0,1,2,3,4). If shuffled, router
   collapse kinetics change; no impact on Lemma 1 asymptotic.
2. **M2P_PRETRAIN_STEPS=500, N_DOMAINS=5** → 100 steps per domain. Budget
   identical to siblings; this is NOT a training-length failure mode.
3. **Seed=42** (mx + numpy). Results reproducible under deterministic
   `training_mode=False` eval.

## G. Falsification Conditions

- K861 FAIL (median < 0.25) AND D1 shows 4-5 unique experts → Theorem 1
  isolation works but gradient budget per expert insufficient. Possible
  resurrection: longer training or larger experts.
- K861 FAIL AND D1 shows 1-2 unique experts → Lemma 1 confirmed; architectural
  fix required (aux load-balance loss or softmax-with-temperature schedule).
- K861 PASS but D1 < 5 unique → metric-swap false-positive; re-label KILLED.
- Structural D4 FAIL (Grassmannian |cos| > 1e-5) → bug in QR construction (not
  expected; unchanged from sibling).
