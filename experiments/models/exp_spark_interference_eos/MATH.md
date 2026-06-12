# MATH — Off-domain LoRA delta crossing on-domain delta is a free EOS

Experiment: `exp_spark_interference_eos`
Base: `mlx-community/gemma-4-e4b-it-4bit` (frozen, 4-bit), `mlx-lm == 0.31.2`.
Adapters: r=6 q_proj LoRA, math(GSM8K) and medical(PubMedQA) from the F#627 recipe
(`experiments/models/exp_composition_residual_analysis/adapter_{math,medical}.safetensors`,
84 keys = 42 layers × {lora_a (2560,6), lora_b (6,2048)}).

References:
- F#862: interference between summed LoRA deltas concentrates in **late decode positions** (the
  repetitive/"yapping" tail), not the early content-bearing tokens.
- F#627: the r=6 math & medical q_proj adapters this run reuses.
- F#866: the F#627 math adapter is net-slightly-negative on GSM8K accuracy at scale 6.0
  (OFF 0.72 → ON 0.64 in that probe). This is an **accuracy** observation; the present hypothesis is
  about **termination timing**, not accuracy lift. Its consequence for us is the *baseline definition*
  in §5: the early-stop must be compared like-for-like against **math-adapter-on greedy-to-EOS**, never
  against base, so the −2pp kill clause compares the same adapter configuration with vs without the
  early-stop rule.

---

## 1. Setup: two delta streams from one frozen base

For a code/math prompt the on-domain adapter is **math**; the off-domain (wrong) adapter is **medical**.
At decode step `t`, with the frozen base producing next-token logits `ℓ_base(t) ∈ ℝ^V`, define the two
adapter-perturbed logit streams (each a full, real forward pass with that adapter's q_proj delta active):
```
ℓ_on(t)  = logits(base + math_adapter,    context_t)
ℓ_off(t) = logits(base + medical_adapter, context_t)
```
and the per-step delta magnitudes
```
on_δ(t)  = ‖ℓ_on(t)  − ℓ_base(t)‖₂
off_δ(t) = ‖ℓ_off(t) − ℓ_base(t)‖₂.
```
The **generation stream is the math (on-domain) stream**: token `x_t = argmax ℓ_on(t)` (greedy, temp=0).
The base and medical streams are read-only probes computed on the *same* context `x_{1..t-1}` (the math
stream's own committed tokens), each with its own KV cache. No training, no composition of the two
adapters into one forward — they are three independent forwards sharing the committed token sequence.

## 2. The claim: off_δ overtakes on_δ at content exhaustion

**Hypothesis.** While the math stream is emitting *useful* content (the reasoning chain and the final
numeric answer), the on-domain adapter is "engaged": it is actively steering logits toward the
math-relevant continuation, so `on_δ(t)` is large. The medical adapter, mismatched to the content, is
comparatively inert there, so `off_δ(t) < on_δ(t)`. Once the answer is delivered and generation drifts
into the repetitive/formatting tail (F#862's late-position interference regime), the math adapter has
nothing left to steer — `on_δ` decays — while the medical adapter's mismatch perturbation does **not**
decay (its bias is content-agnostic), so `off_δ` rises and **crosses** `on_δ`.

**Early-stop rule (the free EOS):** stop the instant `off_δ(t) > on_δ(t)` (first crossing), i.e. emit a
termination at the first decode step where the wrong adapter perturbs the base more than the right one.

This costs **one extra forward per step** (the medical probe; the base probe is shared) and **zero
training** — a length controller read off a free-ish forward pass.

### Why this is a termination signal, not damage to remove
All 14 prior interference sparks treat the wrong-adapter delta as damage to mask/route/rotate/subtract.
Here we **read its takeover as a behavioral content-exhaustion signal**: the crossover time is the
event, not the magnitude. The prediction is about *when* the crossover happens relative to the answer
span, which §5 KC-3 makes falsifiable.

## 3. Predicted numbers

n = 50 GSM8K `test[0:50]`, greedy temp=0, `enable_thinking=True`, MAX_NEW_TOKENS headroom.

Per problem we run the math stream to natural EOS (the **baseline**) and record:
- `T_eos` = tokens to natural `<end_of_turn>`/EOS,
- `T_cross` = first step where `off_δ > on_δ` (the early-stop point; `T_eos` if never),
- exact-match of the early-stopped text vs the gold `#### N`,
- exact-match of the full greedy-to-EOS text (math-adapter-on) vs gold.

Predicted (hypothesis true):
- **Median token savings** `median_t (1 − T_cross/T_eos) ≥ 0.15` — the tail trimmed is a real fraction.
- **Accuracy preserved:** early-stop exact-match `≥` (math-on greedy-to-EOS exact-match) `− 2pp`.
- **Crossover lands after the answer:** in `≥ 80%` of *correct* cases (cases the to-EOS math baseline got
  right) the final answer number appears in the text **at or before** `T_cross` (the crossover does not
  amputate the answer).

## 4. Falsification — pre-registered numeric thresholds (DB kill id 2304)

**KILL if ANY of:**
1. `exact_match(early_stop) < exact_match(math_on_greedy_to_EOS) − 2pp`.
   — Baseline is **math-adapter-on greedy-to-EOS** (same adapter config, F#866), NOT base. Like-for-like.
2. `median(1 − T_cross/T_eos) < 0.15` (median token savings under 15%).
3. The crossover precedes the final answer number in `> 20%` of correct cases (i.e. in >20% of cases the
   math-on-to-EOS baseline got right, `T_cross` occurs strictly **before** the position where the gold
   numeric answer first appears in the committed math-stream tokens).

**SUPPORTED** = none of the three clauses fires:
accuracy within 2pp of the math-on-to-EOS baseline AND median savings ≥15% AND crossover-after-answer in
≥80% of correct cases.

Numeric refutation threshold (single line): the experiment is killed unless
`Δacc ≥ −0.02` AND `median_savings ≥ 0.15` AND `early_crossover_rate ≤ 0.20`.

## 5. Baseline definition (explicit, per F#866)

The accuracy comparison is **math-adapter-on, greedy decode all the way to natural EOS** vs the **same
math-adapter-on stream, early-stopped at the off_δ>on_δ crossover**. Both arms use identical adapter
config (math on, scale 6.0), identical prompts, identical greedy argmax — the ONLY difference is WHERE
generation halts. We do not compare to base-model accuracy; F#866 shows the math adapter is itself
net-slightly-negative on GSM8K accuracy, so a base comparison would conflate adapter-accuracy cost with
the termination question. The hypothesis is solely: does stopping at the crossover lose ≤2pp vs running
to EOS, while trimming ≥15% of tokens, without amputating answers.

## 6. Implementation invariants (enforced in code)
- Three independent forwards (base, math, medical), each its own `make_prompt_cache`; the committed
  token at each step is `argmax ℓ_on` (the math stream drives generation). Composition is NOT used —
  these are three separate single-adapter forwards, never `(ΣB)(ΣA)` and never a two-adapter sum.
- Adapter q_proj wrapper attaches via **subclass nn.Module + setattr** on `layer.self_attn.q_proj`;
  never override `__call__` on an instance (F#831).
- `LORA_SCALE = 6.0 ≤ 8`. Per-sample by construction (same adapters on every prompt).
- `enable_thinking=True`, MAX_NEW_TOKENS headroom (no thinking-truncation).
- Delta norm is the L2 norm of the **full vocab logit difference** at the step, on the SAME context.
- `is_smoke: false`. Phased: build base/math/medical streams per problem, `del`+`gc`+`clear_cache`.
