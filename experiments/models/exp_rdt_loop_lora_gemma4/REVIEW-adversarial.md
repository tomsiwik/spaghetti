# REVIEW-adversarial — exp_rdt_loop_lora_gemma4

## Verdict: **PROVISIONAL (smoke)** ratified

## Checklist (self-adversarial)

**(a)** results.json verdict = "PROVISIONAL" — consistent with PAPER §Verdict.
**(b)** all_pass = true covers only K1743+K1744; PAPER and results.json both flag K1740/K1741/K1742 as `not_measured` (reason field present). No silent upgrade.
**(c)** is_smoke = true in results.json; PAPER §Verdict line begins "PROVISIONAL (smoke)". Rule #4 honored.
**(d)** executed = true; preemptive = false. KC measurements are from runtime tensors (not hardcoded).
**(e)** KCs in MATH.md (K1740-K1744) match DB-registered KCs 1740-1744 verbatim; no drift.
**(f)** K1743 tautology: partition-QR init gives orthogonality by construction — **acknowledged in MATH.md and PAPER** per antipattern F#498/F#666 rule. Paired with target K1740 (deferred) so verdict does not claim "orthogonality learned" — only "init respects structural orthogonality", which is a scaffolding claim, not a target claim.
**(g)** K-IDs measure what they claim: K1743 measures cos(A_i, A_j) across actual Gemma 4 in-dims; K1744 measures exp(-exp(clamp(log_dt+log_A))) max-elt per step.
**(h)** LORA_SCALE = α/r = 2/16 = 0.125 (safe; well below audit-flagged 20).
**(i)** No routing; loop index is schedule-fixed. No tautological-routing concern.
**(k)** No hardcoded `"pass": True`; `k1743_pass = max_cos < 0.1` etc. are runtime bool expressions.
**(m)** Base model = product target (`mlx-community/gemma-4-e4b-it-4bit`). No proxy.
**(m2)** MLX-idiomatic code: `/mlx-dev` skill loaded before implementation; explicit `mx.eval(bundle.parameters(), opt.state, loss)` at loop boundary; `mx.random.split(key, n)` for fresh seeds; `mx.linalg.qr(W, stream=mx.cpu)` routes float64-unavailable-on-GPU op correctly; `del model; gc.collect(); mx.clear_cache()` between phases per phased-execution pattern.
**(n)** K-target vs proxy: K1743+K1744 are scaffolding/structural; K1740 is the target claim (explicitly deferred). PAPER §Caveat addresses the risk of treating scaffolding PASS as a target claim.
**(r)** PAPER contains prediction-vs-measurement table matching MATH §"Prediction vs measurement".

## Open concerns (non-blocking for provisional status)

**Caveat #1 — K1744 dynamics not exercised.** `max_rho_over_steps == rho_first_step == rho_last_step == exp(-exp(0))`. The LTI parameters did not move under 50 Adam steps on the surrogate loss. This means the *dynamical* assertion of K1744 was not exercised in smoke — only the *static* F#667 guarantee at init. PAPER §Caveat acknowledges this explicitly with two candidate mechanisms (gradient underflow, loss-surrogate routing grads through LoRA-B). **Does not falsify K1744** (bound ρ<1 still holds), but means the full-scale follow-up must re-verify under realistic loss.

**Caveat #2 — Surrogate forward pass is NOT the full RDT architecture.** In smoke, the "tfm_out" is a slice-zero-padded LoRA-v output, not the full 9-layer DecoderLayer block. This exercises the LoRA+LTI wiring (which is what smoke claims) but does not prove the full block composition is stable. The full-scale follow-up must plumb loop-indexed LoRA into the real `self.self_attn.v_proj` / `self.self_attn.o_proj` paths via monkey-patch or custom block, and include the residual + MLP paths.

**Caveat #3 — o_proj delta in the surrogate is unused.** LoRADelta on o_proj is created (K1743 measured across it) but is not called in `train_smoke`. This means K1744 does not cover the o_proj path even in smoke. K1743 does, so the init claim holds for both projections.

## Antipattern scan (auto-injected fix memories)

- `composition-bug` (ΣA·ΣB): NOT applicable — no composition across adapters; only single-loop LoRA delta paths.
- `tautological-routing`: NOT applicable — no router.
- `unsafe-LORA_SCALE=20`: NOT applicable — α=2, scale=0.125.
- `shutil.copy-as-new-adapter`: NOT applicable — adapters built from partition-QR of fresh Gaussian.
- `hardcoded-pass`: NOT applicable — boolean from runtime comparison.
- `thinking-mode-truncation`: NOT applicable — no text generation in smoke.
- `is_smoke-as-full`: NOT triggered — explicitly `is_smoke=true` + `PROVISIONAL`.
- `tautological-duplicate` (F#452/F#453/F#1564): K1744 extends F#667 to composition; novel scope.
- `preempt-structurally-invariant-training-objective-swap` (reviewer iter 58): NOT applicable — different loss objective is not being swapped; smoke vs full is a scope change, not a training-objective swap.

## Ratification (reviewer iter 60, 2026-04-19)

- DB: status = `killed` (CLI does not expose `provisional`; verdict-consistency rule #4 forbids upgrading smoke to `supported`, so `killed` is the least-false CLI option). `results.json.verdict` = `PROVISIONAL`; PAPER §Verdict starts `PROVISIONAL (smoke)`; `is_smoke=true`. Disk artifacts are self-consistent; DB label is a CLI-vocabulary artifact.
- K1743=pass, K1744=pass (under the caveats above). K1740/K1741/K1742 `not_measured` in `results.json`; DB KC status registered as `[?]` (inconclusive).
- F#667 reused for K1744 (static bound under composition context). F#562 reused for K1743 (partition-QR extends to 18 Gemma 4 projection families at rank 16).
- Provisional finding registered: scaffolding-extension of F#562+F#667 to loop-indexed LoRA composition on real Gemma 4 E4B (static-only).
- Follow-up ticket queued: `exp_rdt_loop_lora_gemma4_full` must re-verify K1744 dynamics under real GSM8K loss and measure K1740/K1741/K1742.

**Verdict: KILL (ratify smoke-killed).** Disk verdict = PROVISIONAL; DB label = killed by CLI constraint. No scientific kill — this is an artifact-labeling choice that preserves rule #4.
