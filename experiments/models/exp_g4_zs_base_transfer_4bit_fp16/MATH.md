# exp_g4_zs_base_transfer_4bit_fp16 — Zero-shot precision transfer at Gemma 4 scale

## Scope and honest reframe

DB title: *"Gemma 4 adapters trained on 4-bit base transfer losslessly to fp16 base."*

The strict form requires **bf16/fp16** Gemma 4 E4B as the eval base
(`mlx-community/gemma-4-e4b-it-bf16`, ~22 GB on disk). This is not cached, and
downloading inside a single researcher iteration is not a defensible use of the
40-tool-call budget.

Proxy used here: **4-bit (training base) → 8-bit (eval base)**, both
`mlx-community/gemma-4-e4b-it-{4bit,8bit}` (cached). 8-bit is a strict precision
*increase* over 4-bit and is the closest available rung on the precision ladder.
Theoretically:

- If transfer 4→8 **fails** (quality_ratio < 0.95 on adapter benefit), 4→bf16 cannot
  succeed (higher precision shift, more dequantization drift between train and eval
  weight realizations). The hypothesis is rejected.
- If transfer 4→8 **passes**, 4→bf16 is *not yet verified* but the mechanism is
  consistent with the F#97 micro-scale finding extending to Gemma 4. We file a
  follow-up `exp_g4_zs_base_transfer_4bit_bf16` to download bf16 and complete
  the rung.

The verdict here therefore covers the **4→8 step only**; the title's full claim
remains scoped down. PAPER.md repeats this in its verdict line.

## Motivation

- **Finding #97** (conclusive, 2026-03-11): "Zero-shot base transfer works."
  rank-16: 4.2% loss; rank-32: 0.3% loss (3 seeds). **Caveats logged in F#97
  itself**: "Micro scale only (d=64, r=8). SVD perturbation is a controlled
  decomposition; real base model updates may produce different perturbation
  patterns." This experiment extends the test to a real LLM at Gemma 4 E4B
  scale (d_hidden=2560, r=6) and replaces the SVD perturbation with the real
  4→8-bit dequantization difference.
- **Finding #100** (killed, 2026-03-11): "0% expert failure in ZS transfer" —
  killed because the threshold itself was too lenient (2× PPL). Here we track a
  ratio of *adapter benefit*, not absolute PPL, which controls for base-model
  PPL shifts that are not adapter-attributable.

## Type: frontier-extension

Proven framework: F#97 (controlled-perturbation ZS transfer holds at micro
scale). Extension: real LLM, real precision change, larger d.

## Theorem (adapter-benefit transfer under precision change)

**Setting.** Let `W_4` and `W_8` denote the 4-bit and 8-bit dequantized weight
realizations of the same Gemma 4 E4B reference model. The adapter `ΔW = α·BA`
is trained against `W_4` (frozen) by SFT minimizing NLL on a domain corpus.

**Quantities.**

- `PPL(W, ø, S)`: token-level perplexity of base `W` on samples `S` (no adapter).
- `PPL(W, ΔW, S)`: PPL with the adapter delta added to attention output.
- `gain(W, ΔW, S) = (PPL(W, ø, S) − PPL(W, ΔW, S)) / PPL(W, ø, S)`: relative
  PPL improvement attributable to the adapter on base `W`. Positive if the
  adapter helps.

**Definition (transfer ratio).**

```
R = gain(W_8, ΔW, S) / gain(W_4, ΔW, S)
```

`R = 1` means the adapter benefit transfers exactly. `R ∈ [0.95, 1.05]` means
≤5% of the benefit is lost in transfer, which matches the DB-registered KC
threshold "quality_ratio ≥ 0.95" interpreted in the natural ratio direction.

**Prediction.** From F#97 the adapter delta lies in a low-rank (r=6) subspace
that is approximately preserved under bounded weight perturbation. The
4→8-bit dequantization perturbation `||W_8 − W_4||_F` is bounded by the
quantization step, which is **smaller** than the SVD perturbation of F#97
(removing a rank component). We therefore predict transfer is at least as
good as F#97 micro: `R ≥ 0.95` per domain.

## Assumptions logged

1. We use the existing `exp_p1_t2_single_domain_training` adapters (r=6,
   `self_attn.q_proj`, scale=6) as `ΔW`. They are not the v_proj+o_proj
   target recommended by F#627 — this is inherited from the source experiment
   and does not affect the transfer claim (the claim is over precision change
   for *the trained adapter as it exists*, not the optimal adapter).
2. We use the existing `data/{code,math,medical}/valid.jsonl` as `S`. PPL is
   measured only on the **assistant continuation tokens** (prompt is masked),
   matching standard SFT evaluation.
3. 4-bit and 8-bit MLX models share the same tokenizer and architecture; only
   the weight quantization differs. Verified by config inspection at runtime
   (asserts on layer count, hidden size).
4. PPL is a metric proxy. Per project guidance (PLAN.md §"Behavioral outcomes
   over metrics"), we add a behavioral target KC (K2 below) so SUPPORTED
   requires both proxy and behavioral PASS, per F#666.

## Pre-registered kill criteria (target-gated per F#666)

**K1 (proxy / structural).** All four PPL sweeps complete: `PPL(W_4, ø)`,
`PPL(W_4, ΔW)`, `PPL(W_8, ø)`, `PPL(W_8, ΔW)` are finite and positive on
≥95% of evaluation samples for each of the three domains (code, math,
medical). Adapter helps on the training base: `gain(W_4, ΔW) > 0` for all
three domains. PASS iff both hold.

**K2 (target / behavioral).** Median per-domain transfer ratio
`R = gain(W_8, ΔW) / gain(W_4, ΔW) ≥ 0.95`. Equivalently, the 8-bit base
retains ≥95% of the per-token NLL improvement the adapter delivers on the
4-bit base, **measured per-sample** so within-domain variance is captured.
PASS iff median R across the three domains is ≥0.95 AND each individual
domain R is ≥0.85 (no single domain catastrophically fails). The per-domain
floor at 0.85 prevents a single domain dragging from ratio collapse to be
masked by averaging.

Per F#666:

- K1 PASS + K2 PASS → **SUPPORTED** at 4→8 transfer; mechanism extends from
  F#97 micro to Gemma 4 E4B precision change. File follow-up for 4→bf16 step.
- K1 PASS + K2 FAIL → **KILLED**: precision change destroys adapter benefit
  in expectation. F#97 micro-scale result does not extend.
- K1 FAIL → **INCONCLUSIVE**: numerical instability or degenerate adapter
  output; cannot decide.
- K2 has both a median floor (proxy of distributional shift) and per-domain
  floor (target-style robustness check); both must hold for PASS.

## What this does NOT claim

- Does not directly verify 4→bf16 transfer (only 4→8). The 8→bf16 follow-up
  rung is filed in the experiment DB if K2 passes.
- Does not test transfer to a *different* base model (e.g. 26B). That is
  `exp_model_knowledge_gap_26b_base`'s scope.
- Does not measure downstream task accuracy (GSM8K / HumanEval / MedQA). PPL
  is the only metric here; the behavioral angle is captured by per-sample
  ratio dispersion via the per-domain floor in K2.

## References

- Finding #97 — Zero-shot base transfer works (micro scale, conclusive).
- Finding #666 — Target-gated kill rule.
- Finding #627 — Gemma 4 LoRA target modules.
- arxiv:2106.09685 — LoRA (Hu et al, 2021).
- mlx-lm version: pinned at runtime in results.json["mlx_lm_version"].

## Antipattern scan (pre-flight)

- composition math bug: N/A (single-adapter eval, no composition).
- LORA_SCALE: existing adapter trained at scale=6 (safe per F#328/#330 default).
- shutil.copy as new adapter: no — same adapter weights loaded twice via
  `model.load_weights(adapter_path)` API on two different bases.
- hardcoded `pass: True`: KCs computed from real measurements; verdict derived
  from K1 AND K2 booleans.
- eval truncation: PPL only, no generation, no `max_tokens` truncation risk.
- proxy-model substitution: 8-bit substitutes for bf16 at the *test base* only,
  documented above. Adapter trained on 4-bit per the DB title — not substituted.
- KC measures wrong object: K2 measures the relative ratio that the title's
  "transfer losslessly" claim resolves to.
- N=smoke reported as full: `is_smoke` set strictly from `SMOKE_TEST` env;
  results.json carries `is_smoke` and `n_eval_per_domain`.
