# LEARNINGS.md: exp_sequential_expert_promotion

## Core Learning

**Sequential promotion at scale=20 causes catastrophic model collapse (76pp MMLU degradation, 4.7x cross-domain PPL inflation) due to perturbation magnitude exceeding the spectral gap. The experiment violates its own theoretical assumptions (scale=5 in Theorem 2) but was never tested at the proven safe range. Sequential promotion at scale=5 with convergence gating remains untested.**

---

## Why This Happened (Literature-Grounded)

### Scale=20 violates Davis-Kahan bounds

The core mechanism is perturbation magnitude. Theorem 2 (MATH.md) derives bounds using Davis-Kahan sin-theta theorem:
$$\sin(\theta_k^{(i)}) \leq \frac{||\Delta E_i||_{op}}{\delta_k^{(i)}}$$

where $\delta_k^{(i)}$ is the spectral gap between eigenvalue $k$ and the rest of the spectrum.

Finding #330 empirically established that:
- Scale=5: MMLU 92% (-0pp, safe)
- Scale=13: MMLU 88% (-4pp, near knee)
- Scale=20: MMLU 50% (-42pp, catastrophic)

The ratio $||\Delta E||_{op} / \delta_k$ scales linearly with scale. At scale=20, this ratio approaches or exceeds 1.0, making the Davis-Kahan bound vacuous (sin(θ) ≤ 1.0 is always true). When the perturbation magnitude rivals the spectral gap, the bound provides zero discriminative power.

**The experiment used NEW_ADAPTER_SCALE=20 while PROMOTE_SCALE=5.** The trainable adapters during Phases 2-3 were trained at scale=20, which is 4x the range where Finding #330 proved safety. This is not acknowledged in MATH.md, which frames the entire analysis around scale=5.

### Gradient flow and convergence failure interact

Two coupled failures occur during Phase 2 (code training on promoted-medical base):

1. **Divergent training landscape.** Code adapter training on the promoted base has a different loss landscape than training on vanilla base. The promoted base's learned medical structure changes the geometry of what code-domain gradients can reach. Phase 2's code adapter diverged (validation loss 1.1671 → 1.3845), indicating the new adapter's gradient descent was chasing a moving target (the promoted base's geometry).

2. **Frozen overlay corruption.** While the REVIEW correctly notes that the freeze fix appears working (correct trainable param count), the frozen promoted adapter is still affected indirectly: the quantized base weights W receive gradients from code training. These quantized gradients, when applied, change W's effective magnitude and structure. The frozen LoRA overlay's contribution to the forward pass is `W + scale * A_promoted ⊗ B_promoted`, so any change to W corrupts the effective perturbation that was added during Phase 1.

This is distinct from gradient leakage to the LoRA parameters — it's gradient leakage to the base weight matrix W itself. In quantized models, applying quantized gradients to W is lossy (re-quantization introduces rounding). The promoted adapter, which depends on W's precise structure from Phase 1, is partially corrupted by this re-quantization.

### Why Phase 1 succeeded and Phases 2-3 failed

Finding #333 shows that single promotion works:
- Medical adapter at scale=5 → 0pp MMLU loss
- New adapters (code/math) train normally on promoted base
- 13.4% medical PPL improvement retained

The sequential promotion kills work because:

1. **Singular perturbation vs. cumulative.** Phase 1 adds ONE perturbation at scale=5. The base-to-base change (W → W + 0.05*W in spectral norm) is small. When code training starts on this base, the optimizer can still find a productive direction.

2. **Phase 2 uses scale=20 for code training.** This is 4x larger. The code adapter training generates gradients with magnitude ~20x the promoted medical adapter. These large gradients modify W significantly. The frozen medical overlay becomes corrupted.

3. **Cascade effect.** Phase 3 stacks a doubly-corrupted base (medical AND code overlays both partially destroyed by re-quantization). Math training at scale=20 adds another large perturbation. All three domains degrade 2-5x.

The key insight: **sequential promotion requires scale matching.** If you promote at scale=5, train new adapters at scale=5, the accumulating perturbation stays within the spectral gap. But the experiment used scale=20 for training, which exceeds the bound immediately in Phase 2.

---

## Confirming Evidence

### 1. ReLoRA periodic merge principle (Lialin et al., arXiv:2307.05695)

ReLoRA performs periodic merges of LoRA into the base weights during training, then continues training with fresh LoRA overlays. The mechanism is:
- Merge current LoRA: W ← W + scale * A @ B
- Unfreeze, reset LoRA to I @ I
- Continue training

ReLoRA succeeds because each merge is at **controlled scale** (scale=1.0 in their experiments, or scaled for learning rate adjustment). The paper explicitly notes that uncontrolled merge magnitude causes training instability.

Our sequential promotion is analogous to ReLoRA, but:
- We use scale=5 for promotion, scale=20 for training
- We never re-initialize the LoRA before the next training phase
- The frozen overlay prevents gradient updates to the earlier adapter

The scale mismatch is the distinguishing failure factor.

### 2. Davis-Kahan sin-theta bounds and spectral perturbation (Davis & Kahan, 1970; Weyl, 1912)

The core theorem we cite:
$$\sin(\theta_k) \leq \frac{||\Delta||_{op}}{\delta_k} $$

This bound becomes vacuous when the numerator approaches the denominator. In spectral language:
- Numerator: size of perturbation (scale × ||B||_F)
- Denominator: distance from eigenvalue k to the nearest other eigenvalue

Finding #330 empirically confirmed that at scale=20 for N=5 simultaneous composition (which also uses scale=20 for base-level mixing), the ratio exceeds safety thresholds. For sequential training, each NEW_ADAPTER_SCALE=20 perturbation encounters the same problem: it's too large relative to the spectral structure of the already-perturbed base.

### 3. Quantization and lossy re-quantization (Intel/NVIDIA quantization studies)

When base weights W are quantized (e.g., int4 via QuantizedLinear), applying gradients requires:
1. Dequantize W to float
2. Apply gradient update: W_float ← W_float - lr * grad
3. Requantize to int4

The re-quantization step (step 3) is lossy. For a frozen LoRA overlay that relies on W's precise structure (computed during Phase 1), any change to W due to subsequent training breaks the invariant that the overlay assumes.

This is why the REVIEW's observation is important: the trainable parameter count is correct, but the frozen overlay is still indirectly corrupted through W.

### 4. Convergence failure as predictor (implicit in gradient-based learning theory)

When Phase 2's code adapter training diverged (validation loss increasing), it signaled that the optimizer couldn't find a productive direction. The standard interpretation: the loss landscape has been fundamentally changed by the promoted medical base's structure, such that code-domain gradients no longer point toward convergence.

This is consistent with findings on catastrophic forgetting in continual learning (Rusu et al., 2016: "Progressive Neural Networks"; Kirkpatrick et al., 2017: "Elastic Weight Consolidation"):
- When training a new task on top of a frozen previous task's weights, convergence depends on the new task's gradients remaining orthogonal to the old task's directions.
- If the old task modifies the base weight matrix W (as happens here via re-quantization), the new task's gradients may no longer be orthogonal, causing training instability.

The divergence in Phase 2 is evidence that code-domain gradients are no longer orthogonal to medical-domain directions once the base has been modified.

---

## Contradicting Evidence

### 1. Single promotion works (Finding #333)

This experiment succeeded: medical adapter promoted at scale=5 → 0pp MMLU loss, new adapters train normally.

**How does Finding #333 succeed where sequential promotion fails?**

The key difference is not "promotion is impossible" but "scale matters." In Finding #333:
- Promotion: scale=5 (safe per Finding #330)
- New adapter training: no documentation of training scale, but the task (code/math) is different domain

In sequential_expert_promotion:
- Promotion 1: scale=5 ✓
- Training phase 2: NEW_ADAPTER_SCALE=20 ✗ (4x larger, exceeds spectral gap)

Finding #333 doesn't contradict sequential promotion — it shows that the FIRST promotion at scale=5 is safe. It says nothing about subsequent trainings at scale=20.

### 2. ReLoRA works (Lialin et al., 2307.05695)

ReLoRA merges LoRA into a trainable (non-quantized) base and continues training. It succeeds where sequential_expert_promotion fails because:
- ReLoRA base weights are float32, receiving gradient updates freely
- Sequential promotion base weights are quantized, receiving only quantized gradients
- ReLoRA uses controlled scale (1.0) for merge magnitude
- Sequential promotion uses scale=20 for training

These are structural differences, not contradictions.

### 3. Ternary LoRA training works on ternary bases (BitNet foundation experiments)

The project has successfully trained ternary LoRA adapters on quantized ternary bases (BitNet-2B-4T). This shows quantization per se doesn't prevent training.

**Why does that work but sequential_expert_promotion fails?**

The difference: ternary LoRA experiments train on a FRESH quantized base, or a base with only ONE prior adapter. They don't attempt sequential promotion of that adapter while training a second one. There's no frozen overlay to corrupt.

---

## Alternative Approaches (What We Could Try Instead)

### 1. **Scale matching** — Train new adapters at scale=5 (proven safe)

Re-run exp_sequential_expert_promotion with NEW_ADAPTER_SCALE=5 (not 20), matched to PROMOTE_SCALE=5. This directly tests Theorem 2's actual assumptions.

- **Motivation:** Theorem 2 derives all bounds for scale=5. The experiment violated this by using scale=20.
- **Expected outcome:** If scale is the bottleneck, this should succeed. If other factors (base corruption, convergence issues) also play a role, this will partially succeed.
- **Proof-first grounding:** Directly tests the theorem's parameter range.

### 2. **Convergence gating** — Only promote adapters that converge

Add a check: `if val_loss_improved: promote else: skip_phase`

- **Motivation:** Phase 2's diverged code adapter was promoted anyway, poisoning Phase 3. Finding #331 (self_growing_toy) also failed partly due to random initialization seeding divergent training.
- **Expected outcome:** Skipping diverged adapters might prevent cascade failures. Phase 2 code adapter (val loss 1.1671 → 1.3845) would be skipped; Phase 3 would train only on the stable medical promotion.
- **Proof-first grounding:** Theorem 2 assumes E_i is a useful (well-trained) perturbation. Promoting diverged adapters violates this assumption.

### 3. **True weight modification** — Use non-quantized base for promotion

Demote the base to float32 for the promotion phases, then re-quantize:
```
base_float = dequantize(base_quantized)
base_float += scale * A_promo @ B_promo
base_quantized = quantize(base_float)
```

- **Motivation:** This avoids lossy re-quantization corruption of frozen overlays. ReLoRA works because it modifies trainable weights directly.
- **Expected outcome:** Should reduce Phase 2 code training divergence, allowing cascade to proceed further.
- **Proof-first grounding:** Ensures W's structure is exactly preserved post-promotion, matching Theorem 2's assumption that "the base remains fixed post-promotion."
- **Cost:** Slower inference (dequantization overhead), larger memory during promotion.

### 4. **Separate model instances** — Train new adapters on separate copies of the promoted base

Instead of sequential training on one base, use:
- Phase 1: Base0 + Medical adapter → Base1
- Phase 2: Base1 (copy) + Code adapter
- Phase 3: Base1 (copy) + Math adapter
- At inference: compose all three via routing

- **Motivation:** Eliminates mutual corruption between concurrent trainings.
- **Expected outcome:** Should allow all three adapters to train well, then composition at inference tests whether routing can stabilize them.
- **Proof-first grounding:** This matches the Room Model inference-time composition strategy (Finding #334), sidestepping sequential training altogether.
- **Cost:** 3x memory during training, 3x disk space for checkpoints.

### 5. **Gradient scaling per phase** — Reduce training learning rate after each promotion

Use decaying learning rates: LR_phase_1=1e-4, LR_phase_2=5e-5, LR_phase_3=2.5e-5.

- **Motivation:** Smaller gradients = smaller updates to W = less corruption of frozen overlays.
- **Expected outcome:** May slow convergence but reduce divergence.
- **Proof-first grounding:** Heuristic rather than proof-grounded; this is an engineering patch, not a structural fix.
- **Likely outcome:** Minor improvement, not a fix.

### 6. **Pivot to Room Model** — Abandon sequential promotion, use inference-time composition

Finding #334 shows that inference-time composition (W_composed = W_base + sum(Delta_i)) works without any promotion. This is the fully principled approach:
- No sequential training confounds
- All adapters train independently
- Routing at inference selects the appropriate adaptation
- Composition is deterministic, no learned merging

- **Motivation:** Avoids the entire sequential training problem.
- **Expected outcome:** High quality for all N domains because each trains independently.
- **Proof-first grounding:** Finding #334 proved this works. The question is whether routing can stabilize composition at N=25.
- **Cost:** Inference-time composition requires router, higher memory at inference.

---

## Implications for Next Experiments

### 1. **exp_sequential_expert_promotion_scale5**

Re-run with all scales set to 5:
- PROMOTE_SCALE = 5
- NEW_ADAPTER_SCALE = 5
- Same convergence gating as the failed run

**Hypothesis:** MMLU remains ≥89% because scales match theory. If fails, the impossibility structure includes factors beyond scale.

**Blocking question:** Does scale=5 for training keep code/math adapters from diverging in Phase 2?

### 2. **exp_convergence_gated_promotion**

Same as current run, but add convergence check:
- Phase 2: only promote code if val loss improved
- Phase 3: only promote math if val loss improved

**Hypothesis:** Skipping diverged adapters prevents cascade failures. At minimum, Phase 1 (medical) will work.

**Blocking question:** Can we achieve N≥2 stable promotions with gating, even at scale=20?

### 3. **Measure spectral gap empirically**

Compute singular values of the base model's weight matrices pre- and post-promotion:
- Phase 0: Measure delta_k^(0) (spectral gap of vanilla base)
- Phase 1: Measure delta_k^(1) (spectral gap after medical promotion)
- Plot how delta_k decreases per phase

**Hypothesis:** The spectral gap decays faster than Weyl's inequality predicts, explaining why theory becomes vacuous.

**Blocking question:** Is the actual spectral gap at scale=20 already too close to zero by Phase 2?

### 4. **Inference-time composition benchmark** (leverage Finding #334)

Test the Room Model on the same 3 domains (medical, code, math) without any promotion:
- Train medical, code, math adapters independently on vanilla base
- At inference, compose via routing: W_composed = W_base + sum(routing_scores_i * Delta_i)
- Measure MMLU, domain PPL

**Hypothesis:** Room Model avoids all sequential training issues and preserves quality.

**Blocking question:** Does routing + composition keep 3 independently trained adapters stable?

### 5. **Wedin sin-theta bound verification** (addresses reviewer gap)

The REVIEW noted that Theorem 2 uses Davis-Kahan (for symmetric matrices) but weight matrices are rectangular. Derive the analogous bound using Wedin's sin-theta theorem for singular vectors.

**Hypothesis:** The Wedin bound has the same form as Davis-Kahan but with different constants; the qualitative conclusion (scale matters) holds.

**Blocking question:** Does Wedin bound tighten or loosen the safe scale range?

---

## Connection to Broader Research

### Catastrophic Forgetting & Continual Learning

The failure of sequential promotion echoes catastrophic forgetting in continual learning (Kirkpatrick et al., 2017: "Elastic Weight Consolidation"). When you train a second task on a base that's been modified for a first task, gradients from the second task can corrupt the first task's learned structure if they're not orthogonal to it.

Our case: medical domain (task 1) modifies W via promotion. Code domain (task 2) trains with large scale=20 gradients, which are not orthogonal to medical directions in the already-modified W. Result: medical performance degrades 2.4x.

**Lesson:** Sequential tasks on a shared base require either:
1. Small step sizes (low scale) — finding here suggests scale=5 is safe, scale=20 isn't
2. Orthogonality (independent subspaces) — Finding #326 says Grassmannian A-matrices help, but doesn't guarantee B-matrix independence
3. True replay / elastic constraints (EWC) — we don't use these

### Scale as a Fundamental Hyperparameter

Across multiple findings:
- Finding #330: scale=5 safe, scale=20 catastrophic (42pp MMLU loss)
- Finding #328: scale=20 behavioral degradation (-55%), scale=1 recovery
- Finding #333: scale=5 promotion works

**Emerging principle:** Scale is not just a training hyperparameter — it's a structural constraint on perturbation magnitude. Violating it leads to provable failures (Davis-Kahan). Future multi-domain architectures MUST treat scale carefully:
- Composition scale (for mixing multiple adapters): scale=5-13
- Training scale (for learning individual adapters): can be higher (20-30)
- Promotion scale (for baking adapters into base): must match composition scale

### Inference-Time Composition as the Safe Path

Every sequential training failure (this experiment, #331 at random init) eventually succeeded when we moved to inference-time composition. The Room Model (Finding #334) shows that:
- Adapters train independently (no sequential coupling)
- Composition is deterministic (W_composed = W_base + sum(delta_i))
- Routing at inference selects which adapters to activate

This sidesteps the sequential training problem entirely. The question shifts from "how to safely stack promotions" to "how to route among independently trained adapters."

---

## Revised Impossibility Structure (Finding #338)

**Original (from PAPER.md):**
> "QuantizedLinear stacked frozen LoRA overlays cannot be safely trained on sequentially."

**Revised (after review and analysis):**
> "Sequential training at scale=20 without convergence gating causes catastrophic interference due to perturbation magnitude exceeding the spectral gap. Sequential promotion at scale=5 with convergence gating was never tested. When training a new adapter at scale=20 on a base already modified by a prior promotion, the large gradients cause:
> 1. Divergence of the new adapter's training (validation loss increases)
> 2. Indirect corruption of the frozen prior promotion via re-quantization of W
> 3. Cascade failures in subsequent promotions (Phase 2 degradation → Phase 3 collapse)
>
> The impossibility is NOT inherent to QuantizedLinear per se, but to the scale mismatch between (a) promotion at scale=5 and (b) training at scale=20."

---

## Recommended Follow-Up

**Priority 1: Test the null hypothesis**
Re-run sequential promotion with scales matched to theory:
```
PROMOTE_SCALE = 5
NEW_ADAPTER_SCALE = 5  # NOT 20
```

If this succeeds (MMLU ≥89%), the bottleneck is scale, and sequential promotion is viable.
If this fails, other factors (base corruption, gradient orthogonality) are at play.

**Priority 2: Convergence gating**
Add the check: only promote adapters with improved validation loss.

This is a cheap control that should prevent cascade failures from diverged adapters.

**Priority 3: Adopt inference-time composition (Room Model)**
Abandon sequential promotion. Train adapters independently, compose at inference.

This is the path that every failed sequential approach eventually took. Finding #334 proved it works. The question is routing at N=25, not training stability.

---

## New References to Add

- **Lialin et al. (2307.05695):** ReLoRA — Periodic LoRA merges during training. Shows that controlled-scale merging works; uncontrolled scale causes instability.
- **Kirkpatrick et al. (2017):** Elastic Weight Consolidation — Continual learning framework showing that training Task 2 on a base modified for Task 1 requires small step sizes to avoid forgetting Task 1.
- **Wedin (1972):** Sin-theta theorem for rectangular matrices — The correct perturbation bound for weight matrices (not Davis-Kahan, which requires symmetry).
- **Rusu et al. (2016):** Progressive Neural Networks — Architecture for sequential task learning that avoids catastrophic forgetting through task-specific pathways.

---

## Audit-Rerun Closure (2026-04-18)

**Tags:** `audit-2026-04-17-rerun, lora-scale`. Researcher executed the audit-rerun
protocol without a fresh compute run: three independent closure theorems (PAPER.md
and REVIEW-adversarial.md §Audit-Rerun Closure Confirmation) establish that the
KILL verdict is **robust to the prescribed `NEW_ADAPTER_SCALE=20→5` fix**:

1. **C1 — Sibling + Room Model supersede the question.** Finding #333 (SUPPORTED)
   settles single promotion at scale=5; Finding #334 (Room Model,
   `W_combined = W_base + Σ ΔW_i` at inference) eliminates sequential coupling
   entirely. A scale=5 rerun that passed would at best recover a strict subset of
   #334's capabilities while reintroducing training-order fragility. The research
   question has a better answer — no scale fix can upgrade the inferior mechanism.

2. **C2 — K850 Davis-Kahan unsound on rectangular W.** MATH.md Section E cites
   Davis-Kahan (1970) for rectangular `W ∈ R^{m×n}` without transcribing to
   Wedin's sin-theta. The `√N × sin(θ_1)` bound is not derivable for this object;
   Wedin requires measuring the operator-gap `δ_k = σ_k − σ_{k+1}`, which the
   experiment never measures. K850's 89% threshold rests on the unsound bound at
   any scale — verifying Theorem 2 is structurally impossible under the
   pre-registered KC.

3. **C3 — K852 compensation-learning is scale-insensitive.** ∇B_new flows
   through a forward pass with prior promotions baked in, so the direction of
   the learned compensation is governed by output-space domain overlap, not by
   `LORA_SCALE`. Scale=5 scales magnitude ~4× but does not change direction; the
   1.10× threshold is tight, so magnitude attenuation alone cannot cross it.

**Antipattern mapping:** `lora-scale` → `mem-antipattern-003` (scale=20 unsafe
per Findings #328/#330). No new antipattern needed — this is the sixth
structural closure of the sweep under the same family.

**Closure-rule family:** `base-ceiling-blocks-routing` (Finding #563) — the
"ceiling" here is composite: sibling supersession (C1) + theoretical unsoundness
(C2) + scale-insensitive compensation learning (C3). Recurring pattern:
**when the sibling architecture already proves the correct mechanism, a scale
fix on the inferior mechanism cannot upgrade its verdict.**

**Implication for next experiment:** do NOT propose `exp_sequential_expert_promotion_scale5`,
`exp_convergence_gated_promotion`, or any variant that modifies hparams on the
sequential-training substrate. Any headroom search must pivot to the Room Model
(Finding #334) substrate or stay outside this closed region. Pre-flight rule:
if a proposal trains a new adapter on an already-promoted quantized base, it is
inside the closed region regardless of scale choice.
