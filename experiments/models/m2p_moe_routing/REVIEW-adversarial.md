# Peer Review: M2P MoE Routing

## V2 Rerun Review (audit-2026-04-17, metric-swap) — 2026-04-18

### Verdict: KILL

### Adversarial checklist (a)–(s)

| Check | Result |
|---|---|
| (a) results.json `verdict`=KILLED vs DB status `killed` | consistent ✓ |
| (b) `all_pass=false` with DB=killed | consistent ✓ |
| (c) PAPER.md verdict line ("KILLED. Closing on K860") — no PROVISIONAL/etc. | ✓ |
| (d) `is_smoke=false` + `ran=true` — full run, not smoke | ✓ |
| (e) MATH.md git diff: §K (K860 spec) appended. KC #860 was already registered on the DB row (2026-04-07 evidence) — §K formalises, does not add or relax | ✓ no retroactive KC change |
| (f) Tautology sniff: K860 measures `mean_d max_e softmax(router(d))[e]` — a real learned parameter, not identity | ✓ |
| (g) K860 in code (`mean_max_route_weight ≥ 0.50`) matches MATH.md §K.1 definition and DB KC title | ✓ |
| (h) Composition form: `expert_outputs[e]` computed per-expert then `Σ w_e · expert_outputs[e]` (soft-MoE), B-matrices decoded from mixed memory — no `sum(lora_A)` cross-product | ✓ |
| (i) `scale=2.0` (lines 460, 480) — well below the 12/20 unsafe range | ✓ |
| (j) Per-sample domain_id passed through training/eval loops; no `route(val[d][0])` shortcut | ✓ |
| (k) No `shutil.copy` of a sibling adapter | ✓ |
| (l) No hardcoded `{"pass": True, ...}` in KC dict | ✓ |
| (m) MATH.md target (toy GPT d=64 N=4) matches model actually loaded in code | ✓ no proxy substitution |
| (m2) Skill evidence: code uses `mx.eval`, `mx.clear_cache`, `mx.reset_peak_memory`, `nn.value_and_grad`, phased execution, `mx.set_memory_limit`/`set_cache_limit`. Idiomatic MLX; no torch-style module mutation | ✓ |
| (n) Not a thinking-mode MCQ eval; N/A | N/A |
| (o) n=5 domains on K860 (mean over domains) | small but matches MATH.md §K.1 |
| (p) No synthetic padding | ✓ |
| (q) Baseline cos 0.9956 cited from Finding #341 (not drift-prone; measured this run 0.9774) | ✓ |
| (r) PAPER.md V2 prediction-vs-measurement table present | ✓ |
| (s) Math sound. Predicted m̄ ≈ 0.25 ± 0.10; measured 0.3432 (edge of predicted band, matches FAIL side) | ✓ |

### Why this kill is load-bearing

Pre-registered prediction (MATH.md §K.2 — committed before the run per git diff) stated K860 FAIL with m̄ ≈ 0.25 ± 0.10. Measured m̄ = 0.3432 — inside the predicted band. Auxiliary diagnostics reinforce the kill: H̄/ln(4) = 0.966 (96.6% of uniform entropy), 3/4 unique argmax experts (expert_3 ignored), B-matrix |cos| = 0.9774 (unchanged from baseline 0.9956). Soft routing without auxiliary load-balance/entropy loss collapses under round-robin shared-gradient training — the same mode-collapse pattern as Finding #341, just expressed in the router degree of freedom.

### Propagation

- **Finding #574** (see below) — soft-router collapse generalises the Finding #341 mode-collapse pattern to a new degree of freedom.
- Sibling `exp_m2p_hard_moe` (P2, open) directly addresses permanent-learning #1 (hard top-k Gumbel routing for gradient isolation). Do not auto-spawn — analyst gates.
- `exp_m2p_teacher_distillation` / `exp_m2p_tfidf_routing_n5` inherit "routing as an open problem" — downstream constraint.

### Permanently-learned rules (confirmed, propagate via analyst memory)

1. Any "free" DOF under shared-gradient training collapses to centroid (B-matrix → embeddings → router). Fix must **force** domain identity, not "provide available signal".
2. Soft routing without aux load-balance / entropy-penalising loss = saddle-minimiser uniform.
3. metric-swap audit rule: always extract the DB-tracked KC in code, not just the script's internal heuristics.

### Assumptions (for audit)

- Interpreting `all_pass = k860` (MATH.md §K.4 explicitly promotes K860 to sole gate; DB row confirms it tracks only KC #860). K855/K856/K857 retained as auxiliary diagnostics only — this is correct.

---

## V1 Review (Domain Conditioned, retained for context)

> Reviews the earlier "domain conditioning" experiment (additive embedding injection), not the MoE routing variant the DB tracks. Retained because the failure mechanism (gradient homogenisation under round-robin training) is the same.

## Experiment Type

**Type 1 (Verification)** -- MATH.md contains Theorem 2 and Theorem 3 with Proof/QED blocks. The experiment was designed to verify Theorem 3's prediction that domain embeddings destabilize the B-matrix centroid.

## Hack Detector

- **Fix count:** 1 mechanism (additive domain embedding injection). Not a hack stack. Clean single intervention.
- **Is MATH.md a proof or a description?** Mixed. Theorem 2 has a genuine proof sketch with a QED. Theorem 3's "proof" is mostly a forward reference to Theorem 2 plus an appeal to SGD convergence (Robbins-Monro). The core argument structure is sound but contains a critical hidden assumption (see below).
- **Metric used as evidence:** Quality ratio (base - m2p)/(base - sft). This is a reasonable proxy for adapter quality but is not derived from the proof. The proof predicts B-matrix diversity (cos < 0.90), which IS directly measured. Good.
- **Kill criteria source:** K855/K856 are derived from the proof's predictions about centroid destabilization. K857 is a structural carry-forward from the prior experiment. Kill criteria are well-grounded.

## Self-Test Audit

1. **One-sentence impossibility property:** "Learned domain embeddings are linearly independent, making the centroid state unstable." This is one property. PASS.

2. **Cited theorems:** UAT (Hornik 1991, Cybenko 1989), Robbins-Monro (1951), Lemma 1 (linear independence of Gaussian vectors). All real. However, the UAT citation is cosmetic -- it says "M2P *can* approximate any mapping" but does not guarantee it *will* under gradient descent with 500 steps of training. The UAT is an existence result, not a convergence guarantee. Robbins-Monro requires decaying learning rate; Adam with fixed lr=1e-3 does not satisfy these conditions. PARTIAL PASS -- citations are real but conditions are not met.

3. **Predicted numbers:** K855 >= 25% (predicted 30-65%), K856 >= -10% (predicted all positive), K857 <= 1e-5 (predicted <= 1e-7), B-matrix cos <= 0.90. Specific and falsifiable. PASS.

4. **Falsification condition:** "Proof is wrong if B-matrix |cos| remains >= 0.99 after training." This correctly targets the proof's central prediction. PASS.

5. **Hyperparameter count:** Claims 0 new hyperparameters (embedding dim = D_MODEL, standard init). Fair -- the embedding dimension is forced by the architecture. PASS.

6. **Hack check:** Claims this is not a stacking fix but the minimal sufficient fix from information theory. Reasonable -- one mechanism, one new component, zero new loss terms. PASS.

## Mathematical Soundness

### Step-by-step verification

**Lemma 1 (Linear independence of random embeddings):** Correct. N=5 vectors in R^64 drawn from continuous distribution are linearly independent a.s. Standard result. No issues.

**Theorem 2 (Centroid destabilization):**

The proof structure is:
1. At the centroid state, dL_d/de_d = J(theta) * dL_d/dB
2. Since dL_d/dB differs across domains d, the gradients on e_d differ
3. Therefore embeddings diverge, breaking the centroid

**Critical gap in Theorem 2:** The proof shows that *gradient directions* on e_d differ across domains. This is necessary but not sufficient for centroid destabilization. The proof implicitly assumes:

(a) **J(theta) has sufficient rank.** If M2P's Jacobian with respect to memory perturbations is low-rank (e.g., attention concentrates on few tokens), then even with different dL_d/dB, the effective gradient dL_d/de_d could project onto a low-dimensional subspace, allowing the centroid to persist as an approximate attractor. PAPER.md correctly identifies this as the failure mode (Section "Root Cause Analysis", Observation 3), but MATH.md does not acknowledge it as an assumption.

(b) **Gradient magnitude is sufficient relative to centroid attraction.** Even if gradients differ in direction, the centroid minimum of Sigma_d L_d is a basin of attraction. The proof needs to show that embedding-divergence gradients are larger than the centroid-attraction gradients. This is not established.

(c) **The claim that "embedding injection is additive: d(mem)/d(e_d) = I" (line 99 of MATH.md) is misleading.** While the immediate injection is additive (mem = mem_base + e_d), the effective Jacobian dB/de_d passes through multiple transformer layers with attention and nonlinearities. The identity Jacobian at the injection point does NOT mean the end-to-end Jacobian is well-conditioned. The proof elides this crucial detail.

**Theorem 3 (Main result):**

Part 1 follows from Theorem 2, inheriting its gaps.

Part 2 appeals to Robbins-Monro convergence, but the conditions are not met: (i) Adam uses fixed learning rate, not decaying; (ii) the loss landscape is non-convex; (iii) 500 steps may be far from convergence. The appeal to "standard SGD convergence theory" is hand-waving in this context.

Part 3 (parameter overhead) is arithmetic and correct, except the claimed "17K M2P params" is wrong -- actual is 102,976 params (verified from results.json and code). The domain embedding overhead is 0.31%, not 1.9%. This is a factual error that does not affect the proof logic.

### Verdict on mathematical soundness

The proof establishes a necessary condition for centroid destabilization (different gradient directions on embeddings) but not a sufficient one. The gap is exactly what the experiment exposed: gradients differ but are too weak relative to the centroid attraction, because the M2P attention bottleneck makes J(theta) effectively low-rank. The proof would need a bound on the minimum eigenvalue of J(theta) to be complete.

This is a **real gap**, not a cosmetic one. The experiment was correctly designed to test the prediction, and the prediction failed, which means the gap was load-bearing.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. This is well done.

| Prediction | Predicted | Measured | Status |
|-----------|-----------|----------|--------|
| K855: median quality >= 25% | 30-65% | 47.3% | PASS |
| K856: no domain < -10% | all positive | -303.7% | **FAIL** |
| K857: Grassmannian cos | <= 1e-7 | 0.0 | PASS |
| B-matrix cos | <= 0.90 | 0.9785 | **FAIL** |

The critical prediction -- B-matrix centroid destabilization (cos < 0.90) -- failed. Measured 0.9785 vs predicted <= 0.90. This is a 0.0171 reduction from baseline (0.9956), where the proof predicted a reduction of at least 0.0956. The prediction failed by a factor of ~5.6x.

The "repeat" domain failure (-303.7%) is essentially unchanged from baseline (-329%). The domain conditioning had negligible effect on the catastrophic failure mode it was designed to eliminate.

**Honesty assessment:** PAPER.md is excellent in its honesty. It clearly states "Theorem 3 KILLED", identifies the root cause (attention bottleneck making J(theta) low-rank), and does not attempt to rescue the result. The root cause analysis correctly identifies a structural weakness in the proof's assumptions. This is model scientific writing.

## NotebookLM Findings

Skipped -- the MATH.md and PAPER.md are sufficiently clear for manual review. The mathematical gaps and experimental results are unambiguous.

## Novelty Assessment

**Prior art:** The experiment correctly cites MixLoRA (arXiv:2402.15896) and SMoRA (arXiv:2501.15103) as related domain conditioning approaches. The key difference is that those approaches use gating mechanisms (multiplicative conditioning or explicit routing heads), not simple additive injection. The experiment tests the weakest possible form of domain conditioning, which is a reasonable first step.

**Omitted prior art:** The multi-task learning literature on gradient conflict resolution (e.g., PCGrad, GradNorm, CAGrad) is relevant but not cited. These methods address the same root cause (gradient competition across tasks) with different mechanisms (gradient projection, loss weighting). The LEARNINGS.md from the prior experiment (Finding #341) mentions loss normalization as Strategy 4 but the current experiment did not try it.

**Delta over existing work:** The contribution is testing whether the simplest possible domain conditioning (additive embedding injection) is sufficient for M2P-style hypernetworks. The answer is no. This is a useful negative result.

## Experimental Design Critique

### Confounds and bugs

1. **Training steps per domain:** 500 total steps, round-robin over 5 domains = 100 steps per domain. With the "repeat" domain having base loss 1.1 and SFT loss 0.5 (small gap = 0.6), the gradient signal for "repeat" is inherently weak. However, this is identical to the baseline, so the comparison is fair.

2. **No embedding divergence measurement:** The experiment measures B-matrix cosine but does NOT measure whether the domain embeddings actually diverged during training. PAPER.md identifies this as a needed diagnostic (Section "Needed to Test Theorem 3 Correctly", point 1) but does not include it. This is a missed diagnostic that would have distinguished between "embeddings diverged but M2P ignored them" vs "embeddings did not diverge."

3. **B-matrix cosine is measured on a single training example per domain** (line 548: `context_tokens = domain_data[name]["train"][0]`). Using a single example could produce noisy estimates. However, given that the result (0.9785) is so far from the prediction (< 0.90), this is unlikely to change the conclusion.

4. **The quality ratio formula has a guard:** `if (base_losses[name] - sft_losses[name]) > 0.01 else 0`. For the "repeat" domain, base_loss - sft_loss = 1.1061 - 0.5107 = 0.5954, well above the guard. No issue.

### Controls

The experiment uses the same base model, same data, same training schedule as the baseline. The only change is the domain embedding injection. This is a clean controlled experiment. Good.

### Could measurements be explained without the proof being correct?

Yes. The K855 PASS (median 47.3%) could simply reflect that 4/5 domains have sufficient loss gap for M2P to learn partial adapters even with mode collapse. The baseline had 3 domains at 10-56% quality with mode-collapsed B-matrices. The small improvement (21.9% to 47.3% median) could be noise or a minor regularization effect from the embedding, not evidence of centroid destabilization. The B-matrix cosine (0.9785) confirms mode collapse persists.

## Macro-Scale Risks (advisory)

1. **Additive injection is architecturally insufficient.** At any scale, if the hypernetwork's attention pattern compresses memory token information, additive embedding injection will be dominated. Multiplicative gating or explicit routing heads (as in MixLoRA/SMoRA) would be needed.

2. **The M2P architecture itself may be the bottleneck.** The paper correctly notes that attention concentration over memory tokens can create an information bottleneck regardless of input signal strength. This is a structural concern that scales up.

3. **100 training steps per domain is extremely low.** At macro scale with more data, the embedding gradients might have time to overcome the centroid attraction. The failure could be partly a training budget issue, though the B-matrix cosine (0.9785 vs 0.9956) suggests the mechanism is fundamentally weak, not just undertrained.

## Verdict

**KILL**

### Justification

The experiment is well-designed, honestly reported, and produces a clear negative result. The kill is correct. Specific issues:

1. **Theorem 3's proof has a load-bearing gap** (assumption about J(theta) rank) that the experiment exposed. The proof establishes necessary but not sufficient conditions for centroid destabilization. This is not a minor technicality -- it is exactly the gap that caused the prediction to fail.

2. **The central prediction failed quantitatively.** B-matrix cos = 0.9785 vs predicted <= 0.90. The centroid was NOT destabilized. The "repeat" domain failure persists at -303.7%, essentially unchanged from baseline -329%.

3. **The factual error in parameter count** (claimed 17K, actual 103K) does not affect the kill decision but indicates the MATH.md was not verified against the implementation.

4. **Missing diagnostic:** Embedding divergence was not measured, leaving a gap in understanding whether the embeddings diverged but were ignored vs. did not diverge. This should be measured in any follow-up.

5. **PAPER.md's root cause analysis is correct and high quality.** The identification of the attention bottleneck (J(theta) low-rank) as the violated assumption is the right diagnosis. This directly informs what the next experiment should fix.

### Recommendations for follow-up (if pursued)

1. **Measure embedding divergence** (||e_d - e_{d'}|| after training) to distinguish "embeddings stuck" from "embeddings diverged but ignored."
2. **Try multiplicative conditioning** (gating: mem = mem_base * sigmoid(W @ e_d)) which forces the M2P to attend to domain signal.
3. **Try loss normalization** (Strategy 4 from LEARNINGS.md) which addresses the root cause of gradient imbalance without architectural changes.
4. **Fix the parameter count** in MATH.md: actual M2P body is ~103K parameters, not 17K.
5. **Add a bound on J(theta) eigenvalues** to the proof, or acknowledge this as an assumption explicitly.

### On the research arc

Two consecutive M2P experiments (m2p_distillation_toy, m2p_domain_conditioned) have now been killed by the same failure mode: B-matrix mode collapse. The second experiment correctly identified the root cause from the first and proposed a minimal fix -- but the fix was insufficient. Before a third attempt, the question should be: **is M2P-style hypernetwork generation of B-matrices the right approach at all, or should per-domain B-matrices be trained independently (Strategy 1 from LEARNINGS.md)?** Independent training eliminates the gradient interference problem by construction, at the cost of losing the "generate adapter from context" capability.
