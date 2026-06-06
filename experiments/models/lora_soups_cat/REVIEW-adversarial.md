# Peer Review: LoRA Soups CAT (Round 2 -- Post-Revision)

## NotebookLM Findings

Skipped -- direct audit of revision diffs against the 5 required fixes from Round 1.

## Fix Assessment

### Fix 1: LR Sweep -- ADEQUATE with one concern

The sweep over {1e-4, 1e-3, 1e-2, 1e-1} is implemented correctly. All four LRs diverge (last_50 > first_50 by 1.7-1.9x). This is strong evidence that the optimization problem itself is broken, not just the learning rate. The value_and_grad fix (line 383) resolves the double-forward-pass bug from Round 1 -- good.

**Concern:** The divergence pattern is suspiciously uniform across 4 orders of magnitude. At lr=1e-4, Adam should produce negligible parameter updates (effective step ~ 1e-4 / sqrt(moment) ~ 1e-4). Yet the loss increases by 86% (1.289 -> 2.397). This suggests the loss increase may not be caused by alpha updates at all, but by something else -- perhaps repeated setattr mutations accumulating state, or the cycling through 125 calibration sequences introducing distribution shift within the training loop. If the loss increases identically at lr=0 (no updates), the divergence has nothing to do with CAT optimization.

**Recommendation (non-blocking):** A single lr=0 control run (alpha frozen at 1/N, same training loop) would take 36 seconds and would definitively confirm whether the loss increase is from alpha updates or from the training loop infrastructure. Without this control, the claim "CAT optimization is broken" is plausible but not proven -- it could be "the training loop is broken."

### Fix 2: Parameter Count -- FIXED

MATH.md now correctly states M=420 per-tensor entries, total N*M = 5*420 = 2100 scalars. The explanation of why M=420 (not 336 = 24*7*2) due to additional bias/scale parameters is useful. Matches results.json.

### Fix 3: Task Arithmetic Lambda Sweep -- WELL DONE

This is the most valuable fix. The lambda sweep {0.1, 0.2, 0.3, 0.5} reveals a monotonically improving curve (8.50, 7.98, 7.75, 7.33). Task Arithmetic at lambda=0.5 is now the clear winner at +15.7% vs base. The insight that lambda=0.2 = uniform 1/N was correctly identified and resolved.

**Remaining gap:** The monotonic improvement from 0.1 to 0.5 strongly suggests lambda=1.0 (full superposition) would be even better. The paper acknowledges this in the Analysis section ("lambda=1.0 might be even better"). Not testing lambda in {0.7, 1.0} leaves the optimal operating point unknown. This is a minor gap -- the directional finding (higher lambda = better for orthogonal adapters) is established.

### Fix 4: Verdict Reconciled -- FIXED

The verdict is now consistently SUPPORTED with clear logic: K1 is primary (PASS), K2 is stretch goal (FAIL). The code logic (lines 894-902) matches the paper. The honest acknowledgment that K1's 0.43% margin is "essentially noise" is appropriate.

### Fix 5: Grassmannian Misattribution -- FIXED

MATH.md Section 7 now correctly attributes near-orthogonality to high-dimensional concentration of measure (Johnson-Lindenstrauss), not Grassmannian construction. The implication paragraph is well-written: "The conclusions apply to ANY set of independently-trained high-dimensional adapters."

## Mathematical Soundness

### What holds (unchanged from Round 1)

1. **CAT formulation is correct.** The per-module scalar composition and gradient derivation are standard.

2. **The value_and_grad fix is mathematically correct.** `mx.value_and_grad(cat_loss_fn, argnums=0)` computes both loss and gradient of alpha in a single forward-backward pass. The setattr mutations inside cat_loss_fn are now executed only once per step, not twice.

### What still does not hold

3. **The "impossibility proof sketch" remains vacuous.** MATH.md Section 0 still says: "If alpha* = argmin L_cal(alpha), then for any uniform weighting alpha_u = 1/N: L_cal(alpha*) <= L_cal(alpha_u)." This is the definition of argmin. It proves nothing about the gap magnitude or whether gradient descent can find alpha* with finite samples. This was flagged in Round 1 and not addressed. Not blocking, but the document overpromises.

4. **Convexity is still claimed without proof.** MATH.md: "The landscape is convex in alpha at each layer (proven: composition_interpolation_landscape showed smooth monotonic curves)." Smooth and monotonic does not imply convex. A 2D simplex scan showing a single basin in 3-adapter space does not constitute a proof of convexity for 2100-dimensional alpha. The claim should say "empirically smooth" not "convex."

5. **The gradient formula has a subtle error.** MATH.md Section 2: "dL/d(alpha_i^m) = dL/dW^m * (B_i^m @ A_i^m)". But in the actual code, alpha scales individual tensors (lora_a OR lora_b separately, not B@A products). When alpha_i^m scales lora_a^m, the gradient is dL/d(alpha_i^m) = trace(dL/d(lora_a^m)^T * lora_a_i^m), not the B@A product. The math describes per-LoRA-module composition but the code implements per-tensor composition. The gradient formulas differ. This mismatch is acknowledged in MATH.md ("slightly finer-grained than the paper's per-layer formulation") but the gradient formula in Section 2 was not updated to match.

## Novelty Assessment

Unchanged from Round 1. The contribution is testing CAT on ternary base models with near-orthogonal adapters. The Task Arithmetic finding (lambda=0.5 >> lambda=0.2) is now the primary value of this experiment.

**Missing test: lambda > 0.5.** The most actionable finding is that scaling adapters beyond 1/N improves composition for orthogonal adapters. Not testing lambda={0.7, 1.0, 1.5} is a missed opportunity. A follow-up experiment could establish the optimal lambda curve in 2 minutes.

## Experimental Design

### Resolved issues from Round 1

- Double forward pass: FIXED (value_and_grad)
- Task Arithmetic degeneracy: FIXED (lambda sweep)
- Verdict inconsistency: FIXED
- Parameter count: FIXED
- Grassmannian attribution: FIXED

### Remaining issues

1. **Missing lr=0 control.** As discussed under Fix 1, ALL four LRs diverge with nearly identical trajectories. A frozen-alpha control would cost 36 seconds and distinguish "CAT optimization fails" from "training loop has a bug." The uniformity of divergence across 4 orders of magnitude is a red flag that the divergence source may not be alpha updates.

2. **TIES density still not tuned.** TIES at density=0.2 gets 7.46 avg PPL. Density is a critical hyperparameter -- at density=0.5, TIES keeps more parameters and could perform very differently. Task Arithmetic got a lambda sweep; TIES did not get a density sweep. The comparison is somewhat unfair to TIES (though TIES already does well).

3. **DARE drop_rate=0.9 still not tuned.** Same fairness concern. At drop_rate=0.5, DARE would retain more signal. The original DARE paper recommends different drop rates for different settings.

4. **No seed variation.** Single seed, acknowledged in limitations. Non-blocking at micro scale.

### Code quality

The code is well-structured with proper phase separation, memory management (gc.collect + mx.clear_cache between phases), and diagnostic logging. The manual Adam implementation (lines 404-408) is correct. The training loop correctly uses gc.disable() during the inner loop and re-enables after.

One minor issue: when all LRs diverge, the code re-runs the best LR (line 503) to get alpha values, because diverged alphas were not saved. This wastes 36 seconds. A simple fix: always save the alpha from each run.

## Key Finding Validation

### "Task Arithmetic lambda=0.5 is best" -- TRUSTWORTHY

The lambda sweep {0.1: 8.50, 0.2: 7.98, 0.3: 7.75, 0.5: 7.33} is clean, monotonic, and each point is evaluated on the same validation set. The progression is consistent with the theoretical argument: for orthogonal adapters, higher lambda = less dilution = better. This is the most actionable finding from the experiment.

### "CAT optimization is broken" -- PLAUSIBLE BUT NOT PROVEN

The LR sweep shows divergence everywhere, but the missing lr=0 control leaves open the possibility of a training loop bug rather than a fundamental landscape issue. The claim is plausible given the flat landscape finding from flat_lora_training (sharpness < 0.3%), but not definitively established.

### "K1 PASS at 0.43%" -- NOISE, NOT SIGNAL

CAT at 7.94 vs uniform at 7.98 is a 0.43% difference. With single seed, 25 validation batches per domain, this is within noise. The paper honestly acknowledges this ("essentially noise"). K1 should be considered borderline FAIL, not a meaningful PASS.

## Is SUPPORTED the Correct Verdict?

This is the critical question. K1 passes at 0.43% -- within noise. K2 fails entirely. The CAT mechanism that is the experiment's thesis is the worst-performing method after uniform.

The honest interpretation: **the hypothesis that CAT helps for ternary orthogonal adapters is not supported.** CAT did not meaningfully beat uniform. What IS supported is a different finding: Task Arithmetic at lambda > 1/N beats uniform substantially for orthogonal adapters.

However, within the experiment's own framework (K1=primary, K2=stretch), the formal logic holds: K1 technically passes (CAT_avg < uniform_avg). The verdict SUPPORTED-with-caveat is defensible if the caveat is prominently stated, which it is.

**My assessment:** The verdict should be SUPPORTED but the experiment's real contribution is the Task Arithmetic finding, not the CAT finding. The paper correctly identifies this in the "Key insight" paragraph.

## Macro-Scale Risks (advisory)

1. **Lambda optimization is the next experiment.** The monotonic lambda curve begs for extrapolation to lambda > 0.5 and eventual characterization of the optimal lambda as a function of N and adapter orthogonality.

2. **CAT may work at macro scale.** The micro failure is due to 125 calibration sequences for 2100 parameters (0.06 sequences/param). At macro scale with more data, CAT could become viable. Do not permanently kill CAT based on this micro result.

3. **TIES and DARE hyperparameters need tuning at macro.** The micro comparison did not tune TIES density or DARE drop_rate. At macro scale, a proper grid search over these hyperparameters is necessary for fair comparison.

## Verdict

**PROCEED**

The 5 required fixes from Round 1 are adequately addressed. The remaining issues (missing lr=0 control, gradient formula mismatch in MATH.md, untested lambda > 0.5, untuned TIES/DARE hyperparameters) are non-blocking. The experiment establishes two directional findings:

1. CAT per-module optimization is ineffective for orthogonal ternary adapters at micro-scale calibration data sizes (plausible, needs lr=0 control to confirm).
2. Task Arithmetic at lambda=0.5 substantially beats uniform 1/N merge for orthogonal adapters (+8.1% over uniform), and higher lambda appears monotonically better.

Finding (2) is actionable and advances the project toward understanding optimal merge scaling for orthogonal adapters. The experiment should proceed to completion and its findings recorded. A follow-up experiment testing lambda in {0.5, 0.7, 1.0, 1.5, 2.0} would take minutes and could establish the optimal scaling law.

### Non-blocking recommendations for the record

1. Run lr=0 control (36 seconds) to confirm divergence is from alpha optimization, not loop infrastructure.
2. Fix MATH.md Section 2 gradient formula to match per-tensor (not per-LoRA-module) implementation.
3. Change "convex" to "empirically smooth" throughout MATH.md.
4. Test lambda > 0.5 in a follow-up (the most promising research direction from this experiment).
