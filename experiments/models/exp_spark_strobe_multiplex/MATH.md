# MATH — exp_spark_strobe_multiplex

## Claim
Content-BLIND round-robin **time-multiplexing** of N domain adapters across decode
steps beats a **magnitude-matched** static 1/N composition
`ΔW = (1/N) Σ_i (B_i @ A_i)` on a mixed-domain eval, because destructive
interference is a *simultaneity* artifact (deltas clash inside the same matmul),
not a *weighting / total-magnitude* one. The gating baseline is deliberately
magnitude-matched to STROBE so the only thing that differs is simultaneity.

## Setup
- Base: frozen `mlx-community/gemma-4-e4b-it-4bit` (42 layers, q_proj only adapted).
- 3 real trained LoRA adapters, rank r=6, scale s=6.0, on `self_attn.q_proj` of every
  layer: math (gsm8k), python (code), medical (medqa). Each adapter i is a pair
  (A_i ∈ R^{2560×6}, B_i ∈ R^{6×2048}); its delta is ΔW_i = s · A_i B_i.
- Eval: 51 mixed-domain prompts = 17 gsm8k + 17 HumanEval + 17 MedQA, greedy decode.

## Conditions on the SAME prompts
1. **STATIC_NORM** (magnitude-matched 1/N composition — the GATING baseline):
   for layer ℓ, q-proj output = W_ℓ x + (s/N) · Σ_{i=1}^{3} (x A_{i,ℓ}) B_{i,ℓ}.
   All three deltas coexist in one matmul at every step, but each runs at scale s/N
   so the per-step residual budget equals STROBE's (one adapter at full s).
2. **STROBE** (content-blind clock, exactly one adapter per decode step):
   let k(t) = t mod 3 where t = decode-step index (a single global clock shared by
   all 42 layers, advanced once per generated token by the logits-processor hook,
   independent of the token's content). q-proj output = W_ℓ x + s · (x A_{k(t),ℓ}) B_{k(t),ℓ}.
   No two deltas ever coexist in a forward pass.
3. **STATIC_RAW** (raw sum, CONTEXT only — not gating):
   q-proj output = W_ℓ x + s · Σ_{i=1}^{3} (x A_{i,ℓ}) B_{i,ℓ}. This is ~N× over-driven
   relative to STROBE; reported only to show the magnitude confound the review flagged.

STROBE vs STATIC_NORM is the clean, magnitude-matched test: both inject the same
total residual budget per step, so a STROBE win isolates *simultaneity*, not
*magnitude*.

## Theorem (interference bound — why STROBE can win at matched magnitude)
For a single decode step, the magnitude-matched static residual injected into q_proj is
  δ_static_norm = (s/N) Σ_i (x A_i) B_i,
with ‖δ_static_norm‖² = (s/N)² [ Σ_i ‖(xA_i)B_i‖² + 2 Σ_{i<j} ⟨(xA_i)B_i,(xA_j)B_j⟩ ].
STROBE injects δ_strobe = s (x A_{k(t)}) B_{k(t)}, a SINGLE on-clock term at full scale.
The two share comparable total residual budget (E‖δ‖ matched across steps), so the
ONLY structural difference is the cross terms 2(s/N)² Σ_{i<j}⟨·,·⟩ present in
STATIC_NORM and absent in STROBE. If adapters are trained on disjoint domains those
cross terms are sign-indefinite and uncorrelated with the on-domain signal — on a
math token the python/medical deltas add a zero-mean off-axis perturbation the
softmax-attention readout cannot cancel within the step. STROBE removes ALL cross
terms within every step, trading intra-step contamination for inter-step time-sharing
that attention integrates across the KV cache. The hypothesis: the recovered
intra-step SNR outweighs the 2/3 of steps a given domain's adapter is "off-clock" —
and since magnitude is now held fixed, a STROBE win can only be the simultaneity term.

## Pre-registered prediction
acc_aggregate(STROBE) ≥ acc_aggregate(STATIC_NORM) + 4.0 pp  (aggregate over all 51
items), where STATIC_NORM is the magnitude-matched (s/N)Σ baseline.

## Refutation threshold (kill criterion K2299, pre-registered)
If acc_aggregate(STROBE) − acc_aggregate(STATIC_NORM) < +4.0 pp  →  **killed**.
(I.e. blind round-robin token-level strobing does NOT beat the magnitude-matched
(1/N)Σ static merge by ≥+4pp. If the previously-reported win against raw-sum STATIC
collapses against STATIC_NORM, it was a magnitude effect, not simultaneity → killed.)

## Reference baselines (context, not gating)
Per-domain single-adapter accuracy (math-adapter on gsm8k slice, etc.) is recorded to
bound the ceiling; STROBE cannot exceed the best single adapter on its own domain slice.
The registry lists the math adapter at ~82% gsm8k (F#421).
