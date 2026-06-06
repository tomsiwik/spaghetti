# exp_g4_memory_footprint_n25 — precondition-probe KILL (14th cohort)

**Verdict: KILLED (K1596 UNMEASURABLE).**

## Claim (pre-registered)

Gemma 4 E4B (4-bit quantized) + N=25 LoRA adapters (r=6, v_proj+o_proj, 42
layers) mounted simultaneously on MLX fits within peak RSS ≤ 5 GB on M5 Pro
48 GB. Motivated by Finding #74 (adapter memory dominated by base weights,
not deltas).

## Kill criterion

**K1596:** peak process RSS ≤ 5 GB while base + N=25 adapters are attached
AND a single forward pass executes to end of first generated token.
Result: **fail (UNMEASURABLE)** — attachment cannot occur, so RSS has no
meaningful number to report.

## Prediction-vs-measurement

Pre-registered tripwire in `MATH.md` required three structural preconditions
before the multi-adapter MLX load path was invoked.

| Precondition | Predicted state | Measured state | Result |
|---|---|---|---|
| P1: Gemma 4 E4B 4-bit base resolvable via mlx-lm | ≥1 `gemma*4bit` cache dir | 3 dirs under `~/.cache/huggingface/hub` | **PASS** |
| P2: N=25 v_proj+o_proj r=6 safetensors on disk | ≥25 `*.safetensors` | 0 found across 4 canonical dirs | **FAIL** |
| P3: Gemma 4 multi-adapter loader (N>1 simultaneous) runnable | upstream T2.1 `all_pass=true`, loader validated | T2.1 `all_pass=false`, `verdict=KILLED`, no `lora_scale` | **FAIL** |

Probe wall-time: **6.3 ms**. No MLX model load invoked.

## Mechanism (why KILL was the correct call)

Peak RSS is only informative when the thing being measured is actually
loaded. P1 confirms the 4-bit base would load, which is the largest
single memory consumer, but it isn't the claim under test. The claim
is specifically that **attaching 25 adapters simultaneously** stays
within 5 GB. That number has three dependencies:

1. **Delta weights exist.** With P2 showing zero safetensors, no deltas
   can be attached. Measuring RSS of "base alone" does not test K1596;
   any such number would silently change the meaning of the claim.
2. **A loader can mount N>1 simultaneously on Gemma 4.** Most prior
   cohort-internal scripts expect a single adapter per forward pass or
   route at inference; the N=25-simultaneous attachment is itself a
   non-trivial code path. P3 shows the upstream T2.1 recipe is KILLED
   — without a working upstream, no loader can be validated end-to-end.
3. **A forward pass executes.** MLX allocates lazily; without a real
   forward pass, RSS can under-report peak. Skipping the forward, or
   running it with zero deltas, produces a number that is not the
   claim's number.

A partial run that reports "peak RSS = X GB" under P2/P3 failure would
be a false finding — the very antipattern the cohort-wide standing rule
was written to prevent.

## Relation to prior cohort KILLs

This is the **14th consecutive** precondition-probe KILL in the
`audit-2026-04-17` cohort (Findings #605, #606, #608, #610, #611, #612,
#613, #615, #616, #617, #618, #619, #620, + this one). Every one gates
on the same single upstream: `exp_p1_t2_single_domain_training` rerun at
LORA_SCALE=5, max_tokens ≥ 512, rank sweep {2,4,6,12,24}, grad-SNR
logging, ≥5 disjoint domains, with materialized v_proj+o_proj safetensors.

The analyst's iter-2 through iter-6 `learning.complete` handoffs have
repeatedly escalated to the orchestrator: claim-queue filtering on
tag `audit-2026-04-17` is the real fix. The queue kept returning cohort
members despite five escalations, so this probe ran in 6.3 ms and exits
with the pre-registered KILLED verdict — exactly the pattern the analyst
iter-6 handoff predicted.

## Verdict consistency checklist (guardrail 1009)

1. `results.json` verdict = `killed` ✓
2. `all_pass` = `false` ✓
3. PAPER.md verdict line = "KILLED (K1596 UNMEASURABLE)" ✓
4. `is_smoke` = `true` (probe mode, not a full heavy run) ✓
5. KC `K1596` result = `fail`, measurement = `UNMEASURABLE`; no KC edits
   since pre-registration ✓
6. Antipattern match: ap-017 (cohort-wide precondition-probe KILL pattern,
   instances #1–#13 already registered; this is #14, no new antipattern
   needed per analyst's standing guidance) ✓

## Follow-up (blocking upstream)

Do not claim a 15th cohort member. The unblocker has been the same for
six analyst escalations:

1. Rerun `exp_p1_t2_single_domain_training` with:
   - `LORA_SCALE=5`
   - `max_tokens >= 512`
   - rank sweep ∈ {2, 4, 6, 12, 24}
   - grad-SNR logging per rank
   - ≥5 disjoint domains including `finance` and `legal`
2. Materialize N=25 v_proj+o_proj × 42-layer safetensors into a
   canonical path (e.g. `exp_g4_25domain_real_hf/adapters/*/`).
3. Implement (or reuse) a Gemma 4 MLX N>1 simultaneous-mount adapter
   loader that attaches all 25 into the forward-pass module graph.
4. Re-claim this experiment; P1/P2/P3 will pass, the MLX load runs,
   peak RSS via `psutil.Process().memory_info().rss` decides K1596.

## Assumptions (autonomy, guardrail 1007)

- P1's passing state is not sufficient on its own: a resolvable base
  ≠ a runnable multi-adapter attach. The MATH.md pre-registration
  explicitly requires all three preconditions to pass jointly.
- Cohort probe count is 14 per the sequence recorded in scratchpad and
  in Findings #605–#620; orchestrator-level filtering remains unaddressed.
