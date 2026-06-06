# REVIEW-adversarial.md — T2.4: PLE Injection vs Weight Modification

**Verdict: KILLED**

---

## Critical Issues

### 1. K1040 fails catastrophically — not just marginally

QR_frozen = −5.89, QR_full = −4.58. Both conditions make the model WORSE than the base.
This is not a parameter tuning problem. It's a structural failure: random projection
injection corrupts hidden state distributions at all 28 layers simultaneously.

**Finding:** Random-projection PLE injection cannot match LoRA quality. The JL bound
preserves projected-space distances but does NOT bound the quality of the perturbed
residual stream. This is not a missing theorem — it's a gap in Theorem 1's scope.

### 2. MATH.md's Theorem 1 scope was too narrow

The theorem proves JL preservation in the PROJECTED space. It says nothing about
whether adding a random-projection-derived term to h preserves downstream layer function.
These are different claims. The correct theorem should have been:

**What MATH.md needed:** "Does h + Δh(e_l) preserve the model's computational graph
quality when W_gate/W_proj are random?"

Answer: No, because Δh ~ O(sqrt(d)) for random W_proj, matching the scale of h itself.

### 3. PLE-full has 21× more params than LoRA yet fails

This rules out a capacity explanation. The failure is architectural, not capacity-limited.
More parameters in the wrong place (random projections) doesn't help — it makes it worse
(QR_full = −4.58 vs QR_frozen = −5.89; more params → slightly less bad, but still KILL).

---

## Non-Blocking Observations

- K1042 PASS confirms gradients flow correctly. The mechanism works — it just starts
  from a catastrophic initialization and can't recover in 300 steps.
- K1043 PASS (0.182ms M2P) is a durable result: the M2P architecture is fast regardless
  of whether PLE injection works.
- K1041 PASS (1.35× overhead) is also durable.

---

## Impossibility Structure

**What makes this failure impossible to fix without redesign:**

PLE with random projections fails because the injection Δh = RMSNorm(W_proj v) has
expected norm O(sqrt(d)) for random W_proj. This is O(1) relative to ||h||, so the
model receives a perturbed hidden state where the noise is signal-scale at every layer.
No amount of e_l tuning can zero out the damage — the damage is structural (in W_proj).

**The fix:** Use W_gate/W_proj from pretrained MLP layers (not random initialization).
Gemma 4 already has these. This is the Gemma 4 native PLE test — only train e_l.

---

## Verdict

**KILLED.** K1040 fails by a factor of 6× (QR = −4.58 vs threshold 0.85).

The impossibility structure is proven: random-projection PLE injection cannot match LoRA.
Future experiments should test Gemma 4 native PLE (pretrained W_gate/W_proj from MLP,
train e_l only) — this is what T0.5 already validated on Qwen3-0.6B (Finding #416).

**LEARNINGS.md to be written by Analyst.**
