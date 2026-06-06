# PAPER: NRE composition beats naive 1/N sum by ≥3pp on GSM8K at N=5 on Gemma 4

**Verdict: KILLED (precondition-probe, KC #1579 unmeasurable on current platform state)**

## Claim (pre-registered in MATH.md)

KC #1579: On Gemma 4 E4B 4-bit at N=5, NRE composition of
`{ΔW_i}_{i=1..5}` outperforms the 1/N uniform sum by ≥ 3pp on GSM8K
(N=200, 8-shot CoT, deterministic).

## Motivation

Finding #275 (supported): norm-rescaled Euclidean (NRE) matches the
Fisher-Rao Karcher mean on adapter composition on Qwen3-4B-4bit.
Finding #330 (supported): scale=5 is the safe composition operating
point; √N attenuation from 1/N averaging shrinks the composed delta's
Frobenius norm to ~0.45 of per-adapter training scale, putting
composed-1/N deltas below the safe scale threshold. Claim: the same
geometric argument transfers to Gemma 4 E4B with a predicted ≥3pp
GSM8K lift.

## Method (pre-registered)

Before the measurement can run, three preconditions must hold:

- **P1:** five Gemma 4 domain adapter safetensors exist on disk (math,
  code, medical, finance, legal) with non-zero bytes.
- **P2:** upstream `exp_p1_t2_single_domain_training` has verdict
  `supported` or `provisional` (not `KILLED`), i.e. the adapters are
  not no-ops.
- **P3:** Gemma 4 base GSM8K at `max_tokens ≥ 512` exceeds 20%
  (guarding against the truncation format artifact documented in the
  audit-2026-04-17 reconstruction note on the upstream experiment).

If all three hold, the runner merges each `ΔW_i`, measures GSM8K
accuracy, computes `ΔW_unif = (1/5) Σ ΔW_i` and the NRE variant
`ΔW_nre = ((Σ c_i)/5) · Σ ΔW_i / ‖Σ ΔW_i‖_F`, merges each, and
reports the accuracy delta.

If any FAIL, the experiment is KILLED on the probe — not marked
"inconclusive" or "deferred" — and the blocker is logged.

## Prediction vs Measurement

| Quantity                          | Predicted | Measured | Result          |
| --------------------------------- | --------: | -------: | --------------- |
| P1 adapter safetensors present    |       5/5 |      0/5 | **FAIL**        |
| P2 upstream training supported    |    passed |   KILLED | **FAIL**        |
| P3 base GSM8K @ max_tokens ≥ 512  |    ≥ 20 % |    0.0 % | **FAIL**        |
| K1579: acc(NRE) − acc(1/N) on GSM8K | ≥ 3 pp | unmeasurable | **UNMEASURABLE** |

## Conclusion

K1579 cannot be measured on the current platform state. All three
pre-registered preconditions FAIL. The probe KILLs the experiment
honestly: every per-domain Gemma 4 adapter referenced by this
experiment is either absent from disk (math/code/medical — safetensors
missing per the upstream `exp_p1_t2_single_domain_training`
reconstruction note) or never trained at all (finance, legal). The
format artifact (base GSM8K=0% from max_tokens=256) compounds the
failure — even if adapters existed, the 1/N baseline would be
unreliable.

This is the 8th precondition-probe KILL in the `audit-2026-04-17`
cohort. All 8 point to the same upstream:
`exp_p1_t2_single_domain_training` must be rerun at LORA_SCALE=5
(Finding #586 scale-safety bound) with disjoint corpora and
max_tokens ≥ 512 before any Gemma 4 N≥5 composition experiment can
produce a measurable result.

## Antipattern audit (auto-injected memory check)

- Composition math bug: N/A — no composition ran.
- Hardcoded `"pass": True`: not present; verdict/all_pass computed
  from probe results.
- `shutil.copy` as new adapter: not used.
- Tautological routing: no routing in this experiment.
- Eval-template truncation producing base=0%: explicitly surfaced in
  P3 rather than silently accepted (the 0.0% is treated as FAIL, not
  as a measurement).
- Proxy-model-substituted-for-target: target is Gemma 4; probe
  confirms Gemma 4 adapters are absent rather than substituting
  Qwen3-4B results from Finding #275.
- KC measures wrong object: KC locked in MATH.md, not edited after
  probe ran.
- N=smoke reported as full: `is_smoke=false` and verdict=KILLED, not
  `supported`.

## Assumptions

- The audit-2026-04-17 cohort standing rule (no heavy retraining —
  ~4h MLX — inside a single researcher hat iteration) applies. Under
  this rule, the pre-registered probe is the honest outcome.
- KC #1579 is locked; relaxation requires a v2 experiment with a new
  DB entry.
- Finance and legal are pre-registered as the two additional disjoint
  domains the recovery path calls for; any v2 may substitute other
  disjoint domains without invalidating the probe.

## Recovery path (for a v2 experiment, not this one)

1. Rerun `exp_p1_t2_single_domain_training` at LORA_SCALE=5 with
   disjoint math/code/medical corpora and max_tokens ≥ 512.
2. Train two additional Gemma 4 adapters on disjoint finance and
   legal corpora.
3. Re-run this probe; P1/P2/P3 now PASS.
4. Execute the measurement branch; write v2 PAPER.md with the actual
   acc(NRE) − acc(1/N) delta.
