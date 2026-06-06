# REVIEW-adversarial.md — T5.4: User Adapter Privacy

## Verdict: KILL

Two of three kill criteria failed per MATH.md's own definitions. The behavioral
isolation theorem (K1110) is sound and validated. The MIA bound theorem (K1111)
and the Grassmannian isolation theorem (K1112) made false predictions.

---

## Kill Criterion Analysis

### K1110 PASS — Behavioral Isolation (Theorem 1 Verified)
**Sound.** Exclusive routing guarantees this by construction. 0/5 User-B sign-offs
confirmed. User A compliance=60% confirms the adapter is active. No issues.

### K1111 KILL — MIA Delta = 40pp > 20pp Kill Condition

**Math flaw in Theorem 2:** The proof claimed "rank-4 LoRA learns uniform style injection
because gradient = style_token_gradient (same for all q_i by linearity of sign-off)."

**This is incorrect.** The gradient IS uniform at a given weight configuration, but the
weight trajectory during training is NOT uniform — it accumulates per-batch signal from
specific training examples. With n=40 training examples, the adapter overfits slightly
to the training distribution, producing 100% compliance on training examples vs 60% on
held-out examples. This is expected learning behavior, not a proof failure — the proof
failed to account for training dynamics.

**Root cause:** The non-member questions (boiling point, rainbow, hurricanes) were held
out but are from the SAME general science distribution as member questions (photosynthesis,
vaccines, DNA). The adapter generalizes to 60% on non-members, meaning K1111 is testing
whether the adapter generalizes PERFECTLY (delta=0pp), which is impossible by design.

**Structural impossibility:** You cannot simultaneously maximize training compliance AND
have zero MIA distinguishability when train/test come from the same distribution. The
proof should have used OOD non-members or DP training.

### K1112 KILL — max|cos(Y_A_a, Y_B_a)| = 0.6219 > 0.50 Kill Condition

**Wrong measurement in Theorem 3:** The proof analyzed lora_a matrices (input extractors)
and predicted they'd be uncorrelated because different styles produce different gradients.

**This is incorrect.** lora_a (the "down-projection") extracts features from h. When both
users receive general science questions, BOTH lora_a matrices learn to respond to the same
"science question" input features in h. The style lives in lora_b (the output direction).
Measuring cos(Y_A_a, Y_B_a) tests input-side representation overlap, NOT style-specific
subspace isolation.

**Structural impossibility:** lora_a isolation requires input distribution diversity between
users. When users share input distributions (both ask science questions), their lora_a
matrices are guaranteed to correlate. The correct isolation test for privacy is lora_b.

---

## Non-Blocking Observations

1. **K1112 is still partially informative:** max|cos|=0.6219 for lora_a of two adapters
   trained on the same input distribution is expected, not alarming. It does NOT mean the
   adapters will produce correlated OUTPUTS — that depends on lora_b. A follow-up measuring
   cos(Y_A_b, Y_B_b) would be the correct privacy test.

2. **K1111 signal is partially informative:** 60% non-member compliance shows the adapter
   does NOT strongly memorize training queries at the content level. A truly content-memorizing
   adapter would show 0% non-member compliance. The 60% result is evidence of style generalization,
   not per-example memorization — but the threshold design (20pp) was too strict for same-domain test.

3. **Training crash:** Task 21 exited with code 1 because all_pass=False. The experiment ran
   to completion and wrote results.json — the exit code is intentional (non-zero = KC failure).

---

## Impossibility Structure (for Next Experiment)

**K1111 impossibility:** MIA resistance via "uniform style injection" requires OOD test
questions OR differential privacy. Same-distribution train/test makes zero-delta impossible.
**Fix:** Redesign K1111 with semantically distant non-member questions (medical vs science).

**K1112 impossibility:** lora_a subspace isolation is structurally impossible when two users
share input distributions. The correct test is lora_b cosine — output directions should be
orthogonal even when input directions are not.
**Fix:** Measure cos(Y_A_b, Y_B_b) in the follow-up (predicted: much lower than 0.62).

---

## Routing: KILL → emit review.killed → Analyst writes LEARNINGS.md
