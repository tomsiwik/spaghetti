# Peer Review: M2P Composition at N=5 with Grassmannian Guarantee

## V2 Rerun — 2026-04-18 (addendum)

**Trigger:** V2 surgical fix for v1 Issue 1 (BLOCKING — router train/test mismatch).

**Scope of v2 code delta (git diff):**
- `run_experiment.py:663-675`: pre-compute routing ONCE from base-model last-layer hidden state via stop-gradient prefix pass; reuse `routing_weights` at every block.
- `PAPER.md`: prepended V2 section; v1 abstract retained.
- **MATH.md: unchanged** — no post-hoc KC modification.

**Adversarial checklist (a)–(s), v2 run:**
- (a) verdict consistency: DB=killed, results.json all_pass=false, PAPER.md "Verdict (v2): KILLED." ✓
- (b) K852 fails → not claiming `supported` ✓
- (c) PAPER.md verdict line matches DB status ✓
- (d) `smoke_test=false`, 738 tokens, honest full run ✓
- (e) MATH.md untouched in git diff → no KC relaxation ✓
- (f) tautology: K851 (-23.3pp measured against 10pp threshold) and K852 (41.2% vs 50%) are real FP measurements, not identities ✓
- (g) K851/K852 in code measure exactly what MATH.md and DB describe ✓
- (h) composition form (`run_experiment.py:692`): `LORA_SCALE * (x_in @ A) @ B` per-adapter, then routing-weighted sum. NO `sum(lora_A)` bug ✓
- (i) `LORA_SCALE=2.0` ≪ 12 ✓
- (j) per-token, per-sample routing: `base_last_hidden` shape (B,T,D), `routing_weights` (B,T,N_DOMAINS); no `route(val[d][0])` broadcast ✓
- (k) no `shutil.copy` of sibling adapter ✓
- (l) no hardcoded `{"pass": True}` in KC dict ✓
- (m) MATH.md target = ToyGPT d=256 r=4 N=5 = code ✓
- (m2) MLX skill evidence: `mx.stop_gradient`, `mx.eval`, `mx.softmax`, `base.get_hidden_states` used idiomatically; fix respects lazy evaluation (explicit `mx.eval(base_last_hidden)` before reusing across layers) ✓
- (n) `base_gen_loss=10.0087` is a real loss, not truncation artifact ✓
- (o) n=738 tokens ≫ 15 ✓
- (r) prediction-vs-measurement table present in PAPER.md V2 section with v1/v2 columns ✓
- (s) Theorem 1 exact (|cos|=1e-8), Theorem 2 falsified honestly, Theorem 3 never tested (routing precondition unmet) — all stated ✓

**Fix efficacy:**
- v1 routing 36.6% → v2 41.2% (+4.6pp). Real signal, not noise (mean PPL ratio also improved 3.32 → 2.91).
- Fix was correct — Issue 1 WAS a real bug — but not sufficient to rescue KC852.
- Residual gap attributable to secondary v1 Issues 2 & 3 (domain vocabulary overlap `abcdefgh` shared by sort/reverse/repeat; 64-dim MLP router trained 300 steps on 738 tokens).

**v2 verdict: KILL (confirmed).**

Root-cause refinement over v1: KC852 fails for **structural** reasons (signal-level domain overlap + router underparametrisation), not merely for the incidental train/test code bug. This is a stronger kill than v1 — the v2 rerun eliminates the leading alternative explanation.

### Routing signal for analyst
1. The three permanently-learned rules in PAPER.md §"Permanently learned, propagate to siblings" should be encoded as pattern memories (route from base hidden states; disjoint token alphabets; <50% routing is signal-level, not optimization-level).
2. v2 closes the "did we just have a code bug?" question — the Grassmannian composition stack at this scale is routing-limited, not parameter-limited. Downstream `exp_m2p_teacher_distillation` and `exp_m2p_tfidf_routing_n5` inherit this constraint; treat routing architecture (not LoRA composition) as the next open problem.
3. No new mem-antipattern — the reviewer's own v1 Issue 1 was handled correctly by the researcher. Process working.

---

## V1 Review (original, retained for context)

## Experiment Type
Verification (Type 1)

## Hack Detector
- Fix count: 2 (Grassmannian A-slots + independent per-domain M2P training). Not flagged --- both are genuine components, not patches.
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 is a genuine proof with QED. Theorem 2 is a proof sketch that cites an empirical finding as a premise. Theorem 3 is a description dressed in equations --- it states a bound but never proves tightness or derives the 1.20 threshold rigorously.
- Metric used as evidence: Cross-entropy loss and PPL ratio. Loss is a reasonable proxy for next-token prediction quality. Routing accuracy (classification accuracy) directly measures the behavioral outcome.
- Kill criteria source: K851 (10pp degradation) is loosely derived from Theorem 3. K852 (50% routing) is derived from above-random baseline heuristic (2.5x random), not from the proof.

## Self-Test Audit

1. **One-sentence impossibility property:** PASS. "QR decomposition produces orthonormal columns, making parameter-space interference geometrically impossible." Genuinely one property.

2. **Cited theorems --- are they real? Do conditions apply?**
   - QR/Gram-Schmidt (Golub & Van Loan Ch. 5): Real, conditions apply (d >= Nr satisfied).
   - Frobenius trace identity (Horn & Johnson): Real, correctly applied.
   - arXiv:2508.11985 Theorem 1 (Naive LoRA Summation): **Caution.** The paper's theorem requires orthogonal *row spaces* of the A-matrices, which is exactly what QR provides here. Conditions met.
   - Cover 1965: Real theorem, but applied loosely. Cover's theorem is about the capacity of a single-layer perceptron for random dichotomies, not a direct guarantee that any linearly separable representation stays separable after model modification. **Flag: the citation is slightly misleading in context.**
   - Finding #310 (98.3% linear separability): This is an internal empirical finding, not a theorem. Using it as a premise for Theorem 2 makes Theorem 2 empirically grounded, not mathematically proven. MATH.md acknowledges this.

3. **Predicted numbers:** PASS. Four specific falsifiable predictions (P1-P4) with numerical thresholds.

4. **Falsification condition:** PASS. "A_i^T A_j != 0 despite QR" would falsify Theorem 1. The routing non-transfer falsifies Theorem 2 (as happened). However, the self-test does not identify the most important falsification for Theorem 3: there is no way to test whether the 1.20 bound is tight vs vacuous since it depends on the routing accuracy assumption that was itself falsified.

5. **Hyperparameter count:** 4 acknowledged (r=4, scale=2.0, M2P hidden=64, M2P steps=500). Plus router hidden dim (64), router training steps (300), router LR (1e-3), number of router training examples per domain (30). Total is closer to 8. **Flag: 4 additional router hyperparameters are not mentioned.** The router is not a tuning-free component --- it has its own design space that significantly affected the outcome.

6. **Hack check:** PASS. This is genuinely a fresh approach (independent per-domain training) rather than fix #N on the killed multi-domain stack.

## Mathematical Soundness

### Theorem 1 (Parameter Orthogonality under Grassmannian A): CORRECT

The proof is clean, each step follows from standard linear algebra, and the QED is legitimate.

One minor note: the proof correctly handles the case where B-matrices are arbitrary, but MATH.md Section B.2 states the trace identity "equals zero if and only if A_j^T A_i = 0." The "if and only if" direction is wrong --- trace(A_i B_i^T B_j A_j^T) can be zero even when A_i^T A_j != 0 (e.g., if B_i^T B_j = 0 or if B cancels the non-zero A^T A product). The correct statement is: A_j^T A_i = 0 is SUFFICIENT (not necessary) for the Frobenius inner product to vanish. This does not affect Theorem 1's correctness since Theorem 1 only uses the sufficient direction.

### Theorem 2 (Routing Correctness): STRUCTURALLY FLAWED

Theorem 2 has a critical logical gap. It states: "If hidden state representations are linearly separable (Finding #310: 98.3%), then a linear router achieves routing accuracy >= 98.3%."

Problems:

1. **Finding #310 measured separability on the BASE MODEL's hidden states, not on the COMPOSED model's hidden states.** The composed_forward function routes using `router(x)` where `x` is the hidden state DURING composition --- meaning x is being modified by routing-weighted LoRA corrections at every layer. The hidden states the router sees are NOT the same as those Finding #310 measured. This is acknowledged in PAPER.md but NOT in the theorem statement.

2. **The router in composed_forward operates on the CURRENT layer's hidden state (line 676: `router_logits = router(x)`), while the router was TRAINED on the base model's LAST layer hidden state (line 1031: `last_hidden = hidden_states_list[-1]`).** This is a train/test distribution mismatch. During router training, the router sees clean base-model final-layer hidden states. During composed inference, the router sees intermediate hidden states that have been perturbed by LoRA corrections from previous layers. This mismatch alone could explain the routing accuracy drop from the expected 80-98% to the measured 36.6%.

3. **The router is recomputed at every layer in composed_forward (line 676), but trained only once on last-layer states.** The router must make correct decisions at layer 0 using layer-0 hidden states, even though it was trained exclusively on layer-1 (last) hidden states. At layer 0, the hidden state is just embedding + positional encoding --- which may have much weaker domain separability than the final hidden state.

This is not a minor implementation detail --- it is a fundamental conceptual error in Theorem 2. The theorem's premise (linear separability) was measured in a different distribution than where the router operates. The "proof sketch" hand-waves this by assuming convergence of the softmax router to the linear classifier, but the classifier and the router see different inputs.

**Verdict: Theorem 2 is falsified not by "insufficient router capacity" (as PAPER.md suggests) but by a train/test distribution mismatch in the experiment design.**

### Theorem 3 (Composition Quality Bound): DESCRIPTION, NOT PROOF

Theorem 3 states an upper bound on quality degradation:

    Delta_quality <= Sum_{j!=i} r_j(x_i) * ||Delta_j|| * ||x_i||

This is dimensionally correct and follows from the triangle inequality and Cauchy-Schwarz. However:

1. **The bound is never tightened or evaluated numerically.** The prediction "PPL ratio <= 1.20 if routing >= 80%" is not derived from this bound --- it is a hand-wavy assertion. How does an activation-space norm bound translate to a 20% PPL increase? No derivation is given. The relationship between ||Delta_j|| * ||x_i|| and PPL is highly nonlinear and depends on the softmax temperature, the vocabulary size, and the base model's confidence distribution. The 1.20 number appears to be pulled from intuition, not mathematics.

2. **The bound assumes the router weights are the ONLY source of error.** But the composed_forward function applies soft routing (weighted sum of all 5 adapters), not hard routing. Even with 80% routing accuracy, the soft weights distribute probability mass across all 5 adapters, and the bound does not account for how softmax temperature affects this distribution.

3. **"QED" is written after a prediction paragraph, not after a proof.** This is a description dressed in a QED marker.

### Verification Code vs. Theorem: MISMATCH IN ORTHOGONALITY CHECK

The verification function `verify_grassmannian_orthogonality` (line 286-311) computes cosine similarity between FLATTENED A-matrices: it flattens A_i from (d, r) to (d*r,) and computes the cosine of the flattened vectors. This measures something different from what Theorem 1 proves.

Theorem 1 proves A_i^T A_j = 0_{r x r} (a matrix of zeros). The code checks cos(flatten(A_i), flatten(A_j)), which is the inner product of the vectorized matrices divided by their norms. Since A_i has orthonormal columns (from QR), ||A_i||_F = sqrt(r). The inner product of the flattened vectors is trace(A_i^T A_j), which equals sum of all entries of A_i^T A_j. If A_i^T A_j = 0, then trace = 0, so the cosine is 0. Thus the code is a WEAKER check (it verifies the sum of the entries of A_i^T A_j is zero, not that each entry is zero individually). In this case both pass because QR guarantees A_i^T A_j = 0 entirely, but the code would also pass if some positive and negative entries of A_i^T A_j happened to cancel. Not a bug here, but a less rigorous check than claimed.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Results:

| Prediction | Expected | Measured | Status |
|---|---|---|---|
| P1: Grassmannian cos_max | 0.000000 | 1e-08 | PASS |
| P2: Composition PPL ratio | <= 1.20 | 3.32 | FAIL |
| P3: Routing accuracy | >= 80% | 36.6% | FAIL |
| P4: General quality degradation | <5pp | -14.4pp (improvement) | PASS |

The table is present and honest. Two of four predictions failed, and the experiment was correctly killed.

## NotebookLM Findings

NotebookLM review was not executed (authentication not available in this session). The review below was conducted manually with equivalent rigor.

## Novelty Assessment

**Prior art found:**
- PHATGOOSE (post-hoc gating of independently trained experts with zero-shot routing) directly addresses the same problem of routing across independently trained LoRA adapters. PHATGOOSE avoids the routing-training-on-wrong-distribution problem by computing gating scores from the adapter parameters themselves rather than from a separately trained router. This is cited in VISION.md but not engaged with in MATH.md.
- MoLoRA (arXiv:2603.15965) does per-token LoRA routing, which is exactly what this experiment attempts. Its routing mechanism and training procedure would have been worth examining before designing the router.
- CLONE (arXiv:2506.02847) provides a tested MoE router design for dynamic LoRA composition on edge hardware.

**Delta over existing work:** Theorem 1 (Grassmannian parameter orthogonality) is a clean result but is essentially the well-known fact that QR-partitioned subspaces are orthogonal, applied to LoRA. The novelty is in combining this with M2P-generated B-matrices, but this combination was not meaningfully tested because routing failed.

## Critical Issues (Ordered by Severity)

### Issue 1: Router Train/Test Distribution Mismatch (BLOCKING)

The router is trained on base model last-layer hidden states but deployed on intermediate hidden states perturbed by LoRA corrections at every layer of composed_forward. This is the most likely cause of the routing failure and is NOT identified in PAPER.md's root cause analysis. PAPER.md instead blames "router capacity" and "domain signal strength," which are secondary to the distribution mismatch.

This means the experiment does not actually test Theorem 2. It tests a DIFFERENT proposition: "Can a router trained on base-model hidden states transfer to composition-time hidden states?" --- which is a much harder claim that Theorem 2 does not address.

### Issue 2: Domain Overlap Makes Routing Fundamentally Hard

Looking at the data generation code:
- Sort: `"abcdefgh" + ">" + sorted`
- Reverse: `"abcdefgh" + ">" + reversed`
- Repeat: `"abcdefgh" + "*" + digit + "=" + repeated`

Sort, reverse, and repeat share the SAME character set ("abcdefgh"). Parity uses "01" which overlaps with arithmetic's digits. At the per-TOKEN level (which is what the router sees), a single character "a" could belong to sort, reverse, or repeat. The router must learn sequence-level patterns from individual token representations --- this requires attention or memory, not a 2-layer MLP.

The DOMAIN_TRIGGER_CHARS comment in the code (lines 86-94) actually documents this problem: sort, reverse, and repeat all map to the same trigger character set.

### Issue 3: Theorem 3's 1.20 Prediction Is Not Derived

The 1.20 PPL ratio threshold is stated as a prediction from Theorem 3 but is never mathematically derived. It is conditional on routing accuracy >= 80%, which itself failed. But even if routing had succeeded, we cannot verify whether 1.20 was the correct prediction from the bound because the bound was never instantiated with numerical values.

## Macro-Scale Risks (advisory)

1. **Routing at scale:** At d=3584 (Qwen3-4B) with N=200+ adapters, the router must classify among 200+ domains per token. The linear separability finding (#310) was at N=5. The Grassmannian guarantee scales (capacity d/r), but routing complexity scales with N.

2. **Activation interference:** The B-matrix cosine max of 0.291 at N=5 will likely increase with N. No theorem bounds this. At N=200, the probability of destructive activation-space interference grows substantially.

3. **Composition speed:** The composed_forward loop iterates over all N domains for every token at every layer. At N=200 this is 200x the LoRA overhead per layer, defeating the purpose of sparse composition. A hard-routing (top-1) strategy would fix this.

## Verdict

**KILL (confirmed)**

The experiment was correctly killed. The kill is justified and the team's analysis is mostly sound, but the root cause identification is incomplete. Specific issues:

1. **Theorem 1 is verified.** Grassmannian A-matrices produce exact parameter-space orthogonality. This is a solid, proven result.

2. **Theorem 2 was not tested.** The experiment tested a DIFFERENT claim than what Theorem 2 states, due to the train/test distribution mismatch in the router (trained on base-model last-layer hidden states, deployed on composed intermediate hidden states). Before concluding that "routing is the bottleneck," the next experiment should fix this mismatch by either: (a) training the router on the same distribution it will see at inference (composition-time hidden states), or (b) using a separate prefix pass to extract base-model hidden states for routing before applying composition.

3. **Theorem 3's prediction (PPL <= 1.20) was never rigorously derived** and cannot be verified or falsified independently of routing accuracy.

4. **Domain design made routing unnecessarily hard.** Three of five domains share the same character set at the token level. A fairer test would use domains with truly distinct token distributions.

### If Revisiting This Direction

- Fix the router distribution mismatch (Issue 1) --- this is the cheapest falsification test
- Use domains with distinct vocabularies (Issue 2) --- isolate routing signal quality
- Derive Theorem 3's bound numerically with actual adapter norms (Issue 3)
- Consider PHATGOOSE-style zero-shot routing from adapter parameters instead of a trained router
