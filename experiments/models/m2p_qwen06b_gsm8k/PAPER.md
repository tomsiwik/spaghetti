# PAPER.md: M2P on Qwen3-0.6B + GSM8K

**Status: KILLED** — K908 (K_KILL) triggered.

**Experiment type:** frontier-extension (Type 3)
**Question:** Does M2P generate useful adapters from real language on a real model?

---

## Prediction vs. Measurement

| Metric | Predicted (MATH.md) | Measured | Gap |
|--------|---------------------|----------|-----|
| Base accuracy (GSM8K, n=200) | baseline | 0.0% | — |
| SFT accuracy gain | +5 to +15pp | +0.0pp | -5 to -15pp |
| M2P quality ratio | 70–90% | 0.0% (undefined) | collapsed |
| M2P final loss | — | 6.507 | very high |
| SFT final loss | — | 0.0038 | deceptively low |
| MMLU degradation | < 1pp | N/A (not measured) | — |
| Total run time | — | 922s | — |

The quality_ratio metric is numerically undefined (0/0): SFT produced zero improvement
over base, so quality_ratio = 0.0 / 0.0 defaults to 0.0. Neither SFT nor M2P produced
a measurable accuracy signal.

---

## Kill Criteria Results

| ID | Criterion | Result | Value |
|----|-----------|--------|-------|
| K906 | M2P quality_ratio >= 70% of SFT accuracy gain | FAIL | 0.0 |
| K907 | MMLU degradation <= -3pp | N/A | not measured |
| K908 | M2P quality_ratio < 30% (KILL: hypernetwork toy-only) | TRIGGERED | 0.0 |

K908 triggered. Experiment is killed.

---

## Root Cause Analysis

There are four compounding failure modes, ordered by severity.

### 1. Architectural mismatch: LoRA applied to the wrong operation

The custom `forward_with_lora` function (run_experiment.py lines 246–284) applies the
LoRA correction to the post-layernorm hidden state `h_norm` BEFORE the attention
projection, then adds the result directly to the attention output:

```
delta = LORA_SCALE * (h_norm @ A) @ B   # shape: (1, T, D_MODEL)
h = h + attn_out + lora_add
```

Standard LoRA (Hu et al. 2021) applies the delta INSIDE the weight matrix operation:
`W_out = W + AB`, meaning the correction participates in the query/key/value
computation. This implementation bypasses q_proj and v_proj entirely — the module
list `["q_proj", "v_proj"]` names are used only to generate different A-matrix seeds,
not to actually hook into those projections. The "LoRA" here is a free-standing
additive layer on the hidden state, not a weight-space modification of q/v projections.

This is structurally different from what the MATH.md assumes. The proven M2P
B-matrix generation targets the subspace of q/v weight updates; this implementation
computes something else entirely.

### 2. Max generation length too short for GSM8K

`max_gen_tokens=128` is insufficient for GSM8K. GSM8K chain-of-thought solutions
routinely span 200–400 tokens. The greedy decode loop (lines 463–476) stops at 128
tokens, so the "#### answer" terminator is often never reached. The answer extraction
regex finds no match and returns None; None != gold always, so correct=0 for every
example.

This explains why base_accuracy=0.0 despite Qwen3-0.6B-4bit having non-zero baseline
GSM8K performance in the literature.

### 3. SFT loss does not reflect generation quality

SFT final loss of 0.0038 looks good but is misleading. Training uses next-token
prediction on inputs truncated to 256 tokens. For GSM8K examples where the question +
chain-of-thought exceeds 256 tokens, the "#### answer" suffix is truncated away. The
model trains to predict the chain-of-thought but never sees the answer token. So loss
minimization on truncated inputs does not improve answer accuracy.

### 4. Insufficient training steps for real NLP

300 steps on 2000 examples is adequate for toy synthetic domains (vocab=128, d=64)
but insufficient for real NLP. Aghajanyan et al. (2012.13255, Table 1) measured
d_int = 100–1000 for NLP tasks; the MATH.md acknowledged d_M2P=64 may be tight.
With a broken SFT signal (failure mode 3), M2P has nothing to imitate; it trains
on the same truncated-input NTP objective and converges to a high loss of 6.507.

---

## Impossibility Structure

The quality_ratio metric collapses when SFT_improvement = 0. This is not a boundary
case — it is guaranteed whenever any of the following hold:

1. The generation length is too short to ever produce the answer token (base=0,
   sft=0, m2p=0 are all jointly zero, not just comparably bad).
2. The training target (NTP on truncated text) does not contain the answer suffix.
3. The adapter operation is structurally wrong (hidden-state additive vs. weight-space).

With base_accuracy=0.0, sft_accuracy=0.0, and m2p_accuracy=0.0 all equal, the
experiment cannot distinguish between "M2P matches SFT" and "nothing works." The
kill criterion K908 triggers by default in the 0/0 case (code treats undefined as 0.0).

The M2P at d_M2P=64 producing m2p_final_loss=6.507 (vs. SFT final_loss=0.0038) also
confirms M2P could not replicate the SFT signal — but this is a secondary failure.
Even perfect M2P imitation of SFT would produce accuracy=0.0 given failure modes 1-3.

---

## What Carries Forward

The failure is experimental-design specific, not a refutation of M2P on real models.

**What is unresolved:**
- d_M2P=64 was never tested against a working SFT ceiling. The capacity question
  (does M2P need d_M2P > 64 for real NLP?) remains open.
- Adversarial critique #3 (no real NLP result) remains unresolved.
- The 0.6B model scale is untested with a correctly-implemented LoRA.

**Minimum fixes required for a valid retry:**
1. Use standard LoRA: hook into q_proj/v_proj weight matrices directly. The mlx_lm
   LoRA fine-tuning path already supports this; use it instead of the custom
   forward_with_lora.
2. Increase max_gen_tokens to at least 256, preferably 512.
3. Increase train_steps to at least 1000 (SFT) before measuring the M2P ceiling.
4. Use max_seq_len >= 512 to avoid truncating answer suffixes during training.

**What the kill does NOT invalidate:**
- M2P composition and routing math from prior findings (synthetic domains).
- The Grassmannian A-matrix design (orthonormality holds regardless of how B is used).
- Pierre architecture feasibility (SFT-only adapters on real models, findings #319, #332).

---

## Config at Time of Run

| Parameter | Value |
|-----------|-------|
| Model | mlx-community/Qwen3-0.6B-4bit |
| lora_rank | 4 |
| lora_scale | 5.0 |
| d_m2p | 64 |
| l_m2p | 2 |
| n_memory | 32 |
| train_steps | 300 |
| n_train | 2000 |
| n_test | 200 |
| max_seq_len | 256 |
| max_gen_tokens | 128 |
| m2p_params | 15,055,104 |
