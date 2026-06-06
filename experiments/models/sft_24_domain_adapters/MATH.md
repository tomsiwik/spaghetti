# SFT 24 Domain Adapters: Mathematical Foundation

## Experiment Type: Guided Exploration

**Proven framework:** Grassmannian orthogonality (Finding #54) + SFT convergence at N=5
(Finding #206). **Unknown:** Does the SFT recipe generalize from 5 to 24 domains? The
math predicts it should (independent training, identical spectral properties), but the
24 new domains may have different gradient-subspace alignment characteristics.

---

## Step A: Failure Mode Identification

**Potential failure:** SFT adapters may fail to improve PPL over base on some or all
domains. Two failure modes:

1. **Non-convergence:** Training loss does not decrease below initial. This would mean
   the adapter has insufficient capacity or the learning rate is wrong.

2. **Format dominance (Finding #216):** SFT adapters learn shared instruction format
   (cos=0.97 between domains) rather than domain-specific knowledge. This was observed
   at scale=2.0, 200 steps. We mitigate with scale=20.0, 300 steps (proven recipe from
   Finding #206).

**Root cause of format dominance:** SFT zeros instruction-token gradients. For short
training (200 steps, scale=2.0), the gradient signal is dominated by the instruction
format, which is shared across all domains. The LIMA hypothesis (Zhou et al., 2305.11206)
explains this: most of what SFT teaches is format, with domain knowledge as a secondary
signal. At scale=20.0 with 300 steps, the adapter has more capacity and more iterations
to capture the domain-specific gradient component.

---

## Step B: The Right Question

**Wrong:** "Does SFT beat NTP on PPL?"
(Finding #262/264 showed NTP is better on PPL for math/reasoning, SFT is better on code
and behavioral quality. PPL is the wrong metric for comparing training recipes.)

**Right:** "Does SFT training with Grassmannian A produce adapters that (a) converge on
all 24 domains, and (b) maintain sufficient PPL improvement over base to be useful for
composition?"

The behavioral advantage of SFT (judge scores, instruction following) is established at
N=5 (Finding #187). This experiment extends convergence to N=24.

---

## Step C: Prior Mathematical Foundations

**Theorem (SFT gradient decomposition, Finding #206):** For input sequence x = [x_inst, x_resp],
the SFT loss function L_SFT = -sum_{t in resp} log p(x_t | x_{<t}) produces gradients:

  dL_SFT/dW = sum_{t in resp} dL_t/dW

where L_t = -log p(x_t | x_{<t}). The instruction tokens contribute to the conditioning
context but receive zero loss, so their gradients are zero by the chain rule applied to
the mask.

**Consequence:** All gradient updates concentrate on response-token prediction. This has
two effects:
1. **Positive:** No gradient noise from predicting instruction tokens (which are templated
   and provide weak signal). This improves convergence efficiency per step.
2. **Negative:** Adapter parameters that only affect instruction-token logits receive no
   gradient and remain at initialization. This reduces effective adapter capacity for
   instruction understanding.

**Grassmannian orthogonality (Finding #54):** The Grassmannian AP skeleton generates N
orthonormal frames A_1, ..., A_N in R^{d x r} such that A_i^T A_j approx 0 for i != j.
The composition interference bound is:

  ||DW_i^T DW_j|| <= ||B_i||_F * ||A_i^T A_j||_F * ||B_j||_F / r^2

When A_i^T A_j approx 0, interference approaches zero regardless of B-matrix correlation.
Empirically: mean |cos(DW_i, DW_j)| = 0.024 at N=24 (Finding #54), 6.7x below the
orthogonal-frame capacity limit.

---

## Step D: Proof of Convergence Guarantee

**Proposition 1 (SFT convergence on BitNet-2B with TernaryLoRA).**
Under the following conditions:
- Base model BitNet-2B-4T unpacked to nn.Linear (bf16 weights)
- TernaryLoRA with rank r=16, scale alpha=20, frozen Grassmannian A
- Adam optimizer with lr=1e-4
- SFT loss with response-only masking
- At least 100 training samples with mean response length >= 20 tokens

then after 300 training steps, the validation loss L_final satisfies L_final < L_base
(the base model validation loss without adapter).

*Argument.* This is not a formal convergence theorem but an empirical scaling argument.
At N=5 (Finding #206), all 5 domains converged with the identical recipe:
- Math: 32% reduction, Code: 24%, Medical: 23%, Legal: 9%, Finance: 6%
- The weakest convergence (finance, 6%) still passed.

At N=24, the same recipe applies to 24 independent training runs. Each domain has
400 training samples (matching the N=5 setup) and uses an independent Grassmannian A
matrix. Since training runs are independent (no shared state between domains), the
convergence behavior of each domain depends only on:
1. The data distribution of that domain
2. The A-matrix assigned to that domain

By the Grassmannian construction, all A-matrices have identical spectral properties
(they are orthonormal frames), so the training dynamics differ only in the data.

*Note:* This is an empirical scaling argument, not a formal convergence proof. The
argument assumes all 24 domains have sufficient gradient-subspace alignment with their
respective A-matrices, which is the hypothesis being tested.

**Prediction 1:** All 24 domains converge (L_final < L_base). Based on N=5 results
where the weakest domain still showed 6% reduction, we predict at least 22/24 domains
show measurable improvement (allowing for 2 potential edge cases in the 19 new domains).

**Prediction 2:** Training time per adapter is ~60-75 seconds (matching Finding #206:
~2 min/adapter at 200 steps, scaling to 300 steps ~ 3 min, but with faster M5 Pro).
Total time: 24 * 3 min = ~72 min.

---

## Step D (continued): Quantitative Predictions

| Prediction | Source | Value |
|------------|--------|-------|
| P1: Domains with L_final < L_base | Extension of Finding #206 (5/5) | >= 22/24 |
| P2: Mean val loss reduction | Finding #206 (mean ~19% at 5 domains) | 10-25% |
| P3: No divergence (K752) | Finding #206 (0/5 diverged) | 0/24 diverge |
| P4: Total training time | 24 * ~3 min | ~72 min (< 4 hrs) |
| P5: Weakest domain improvement | Finding #206 (finance: 6%) | >= 3% |

**Kill criteria (derived from predictions):**
- K750: P1 predicts >= 22/24 improve. Threshold: 15/24 (conservative, accounts for
  19 new domains that may include edge cases). FAIL if < 15.
- K752: P3 predicts 0 divergence. FAIL if any domain diverges.

---

## Step E: Assumptions & Breaking Conditions

1. **Data format assumption:** All 24 domains have `### Response:\n` markers for
   instruction masking. VERIFIED: all 24 domains confirmed to have markers.

2. **Data quality assumption:** Training data is sufficiently diverse within each
   domain to provide a gradient signal. If a domain's data is all near-identical
   (e.g., a slice from a single template), the adapter may overfit to format rather
   than learn domain content.

3. **Scale assumption:** lora_scale=20.0 was tuned at N=5. At N=24, the same scale
   should work because training is independent per domain. However, the optimal
   scale may differ for domains with very different text statistics.

4. **Capacity assumption:** rank=16 provides sufficient capacity for all domains.
   Finding #206 showed this works for math, code, medical, legal, finance. New
   domains (music, sports, agriculture, etc.) may have different capacity needs.

**Breaking conditions:**
- If assumption 2 fails: some domains won't improve. K750 has margin (15 threshold vs
  24 domains) to absorb this.
- If assumption 4 fails: specific domains will show weak improvement. Not catastrophic
  for composition (Grassmannian A still ensures orthogonality).

---

## Step F: Worked Example (not applicable)

This experiment is a direct extension of a proven recipe to more domains. No new
mathematical mechanism is being tested. The worked example from Finding #206's MATH.md
(SFT gradient decomposition) applies directly.

---

## Step G: Complexity & Architecture Connection

**Per-domain training:**
- Model load + unpack: ~10s (BitNet-2B-4T, 3GB)
- LoRA setup: 30 layers * 7 modules = 210 TernaryLoRA modules, 10.9M trainable params
- Training: 300 steps * ~0.2s/step = ~60s
- Eval: 25 val samples * ~0.1s = ~2.5s
- Total per domain: ~75s

**Total:** 24 * 75s = ~30 min (optimistic) to ~72 min (conservative)

**Memory:** Peak ~17GB (proven at N=24 in Finding #54). Well within 48GB M5 Pro.

**Output:** 24 adapter files (B-matrices only), each ~42KB (210 matrices * r=16 * out_f * 2 bytes).

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **No impossibility property claimed. This is a guided-exploration experiment
   operating within the proven Grassmannian orthogonality framework (Finding #54).
   We test whether the SFT recipe transfers from 5 to 24 domains. Non-zero gradient
   is necessary but not sufficient for convergence — it does not prevent oscillation,
   saddle points, or capacity saturation.**

2. Which existing theorem(s) does the exploration build on?
   **Grassmannian frame orthogonality (Finding #54) — ensures inter-adapter interference
   is bounded. Finding #206 — empirical convergence at N=5 with identical recipe. LIMA
   (Zhou et al., 2305.11206) — data efficiency hypothesis for SFT. Note: these are
   empirical findings and hypotheses, not formal convergence theorems.**

3. What specific numbers does the proof predict?
   **P1: >= 22/24 domains improve. P2: 10-25% mean val loss reduction. P3: 0/24
   diverge. P4: ~72 min total. P5: weakest domain >= 3% improvement.**

4. What would FALSIFY the proof?
   **If > 9 domains fail to improve (K750 FAIL), the recipe does not generalize
   from N=5 to N=24. If any domain diverges, the training setup is unstable.**

5. How many hyperparameters does this approach add?
   **0 new. All hyperparameters (rank=16, scale=20, lr=1e-4, steps=300) are inherited
   from the proven N=5 recipe (Finding #206).**

6. Hack check: Am I adding fix #N?
   **No. This is a direct replication of a proven recipe at larger scale.**
