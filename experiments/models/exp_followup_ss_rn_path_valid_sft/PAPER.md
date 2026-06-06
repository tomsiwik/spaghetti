# PAPER.md — SFT-residual M2P with personalization data (Finding #403 Gemma 4 replication, within-5pp variant)

**Status: KILLED** | Precondition-probe | 2026-04-19

## Abstract

Registered to replicate Finding #403 (SFT-Residual M2P, Qwen3-4B, QR=1.175)
on Gemma 4 E4B 4-bit under a tighter tolerance than the sibling experiment:
K1572 demands `|acc_final_Gemma4 − 74.4%| ≤ 5pp`. The parameterization is
`B_applied = B_sft + output_scale(t)·head(z)` with head zero-init and
personalization-distinct training data (explicitly NOT SFT replay).

Routing via the pre-registered preconditions (MATH.md §Preconditions), all
three required probes FAIL:

- **P1 (B_sft `.safetensors`):** FAIL. `adapters/math/adapters.safetensors`
  missing in upstream T2.1; only `adapter_config.json` stub is present.
  MLX LoRA weights were gitignored and never committed.
- **P2 (personalization corpus):** FAIL. No persona-tagged corpus staged
  under `data/personalization/` or peer paths. Only GSM8K SFT
  `train.jsonl` (1800 lines) exists upstream.
- **P3 (upstream T2.1 verdict):** FAIL. T2.1 `results.json.verdict =
  KILLED` (2026-04-18 metric-swap + format-artefact); B_sft validity
  inherits the KILL by standing rule #3.

K1572 routes FAIL (unmeasurable, not measured-and-fell-short). Verdict =
KILLED; `results.json.verdict="KILLED"`, `all_pass=false`, `ran=false`.
This is the **6th precondition-probe KILL this loop**, matching the same
infrastructure blocker as sibling `exp_followup_sft_behavioral_lora_scale_5`
(Finding #600).

## Prediction vs Measurement

| Quantity | Predicted (under preconditions) | Measured | Pass? |
|---|---|---|---|
| P1 T2.1 `.safetensors` on disk | exists, size > 0 | missing (only `adapter_config.json` stub) | FAIL |
| P2 personalization corpus on disk | persona-tagged, disjoint from GSM8K | none staged; only GSM8K `train.jsonl` (1800 lines) | FAIL |
| P3 T2.1 upstream verdict ≠ KILLED | valid upstream | `verdict="KILLED"` | FAIL |
| K1572 `|acc_final − 74.4%|` | ≤ 5pp (Qwen3-4B reference QR=1.175) | unmeasurable (no B_sft, no persona corpus, invalid upstream) | FAIL |

## Results

| Field | Value |
|---|---|
| verdict | KILLED |
| all_pass | false |
| ran | false (probe-only) |
| is_smoke | false |
| probes | 3 × FAIL (P1, P2, P3) |
| KC K1572 | FAIL (unmeasurable) |

## Why It Failed (infrastructure, not mechanism)

The mathematical claim is unchanged from Finding #403 and remains plausible
under Theorem 1 (Theorems A+B+C). The failure is infrastructural:

1. **P1 missing B_sft.** Gemma 4 M2P residual injection requires the
   SFT-trained B-matrices. MLX LoRA `adapters.safetensors` from T2.1 were
   produced at some point but gitignored and never committed. Only the
   registry stub (`adapter_config.json`) is on disk. Random-B_sft
   substitution reclassifies the experiment as T2.4 (random-init ΔB,
   already KILLED at QR=−5.89).

2. **P2 missing personalization corpus.** Finding #403's mechanism depends
   on Theorem C data separation: training on a distribution disjoint from
   the SFT data attenuates catastrophic forgetting. Substituting GSM8K
   (the only on-disk T2.1 data) re-creates the parent failure mode
   (QR=0.707). No persona-tagged corpus is staged in this repo.

3. **P3 upstream KILL.** Even if P1 weights were recovered, the verdict
   inherits T2.1's 2026-04-18 KILL (metric-swap MedMCQA↔MedQA +
   format-artefact max_tokens=256 CoT truncation). Downstream claims on
   an invalid B_sft substrate cannot be stated either way.

The tighter K1572 ≤ 5pp band (vs sibling QR ≥ 0.90) does not change the
probe routing; both preconditions and both claims are blocked by the same
three failures.

## Verdict-Consistency Pre-flight (all six must hold for SUPPORTED — they don't)

1. `results.json["verdict"] == "KILLED"` ✓ (matches DB kill route).
2. `results.json["all_pass"] == false` ✓.
3. PAPER.md verdict line = `KILLED` — no hidden PROVISIONAL/SUPPORTED.
4. `is_smoke == false`; not a smoke-mode run.
5. KC K1572 text and threshold (`≤ 5pp`) unchanged between MATH.md
   pre-registration and `results.json`. `git diff MATH.md` shows the file
   is newly added, not edited post-hoc.
6. `type: fix` antipattern check: no composition math bug (no composition
   executed); no tautological routing (no routing executed); no unsafe
   adapter scale (no training executed); no `shutil.copy` as new adapter;
   no hardcoded `"pass": True` (`all_pass=false`); `is_smoke=false` honestly
   reported; no eval-template truncation (no eval executed); no proxy-model
   substitution (Gemma 4 never loaded); KC measures the right object by
   routing FAIL-unmeasurable when that object is absent.

All six checks consistent with KILLED. None consistent with SUPPORTED.

## Impossibility Structure (of the KILL itself)

Under PLAN.md §KC discipline, a pre-registered precondition routing failure
IS a KILL:

- Finding #403 replication on Gemma 4 is **testable iff** B_sft exists AND
  persona data exists AND upstream B_sft is valid.
- All three are false at claim time.
- Therefore the replication claim is **undecidable from this repo state**.
  Undecidable ≠ supported. KILLED is the honest routing.

## Assumptions

- Autonomy (guardrail 1007): no user input was requested; the probe routing
  was executed immediately on the observed repo state.
- Sibling `exp_followup_sft_behavioral_lora_scale_5` (Finding #600) provides
  the template for precondition-probe routing on this same infrastructure
  blocker; only the KC threshold differs (≤ 5pp here vs QR ≥ 0.90 there).

## Implications

1. **Class-level unblock via one upstream rerun.** Re-running
   `exp_p1_t2_single_domain_training` at `LORA_SCALE=5` regenerates
   `adapters/math/adapters.safetensors` and unblocks ≥4 downstream
   experiments (this one, `exp_followup_sft_behavioral_lora_scale_5`,
   `exp_followup_sequential_activation_compose_real`,
   `exp_followup_hypernetwork_residual`). Same `scale-safety` tag.

2. **Stage persona-tagged corpus before V2.** Disjoint-distribution
   persona-prefixed queries (Theorem C requirement). Not GSM8K. Not
   shared with T2.1 training split. Without this, V2 re-creates the
   parent QR=0.707 failure.

3. **Do not relax K1572 to accommodate SFT replay.** That collapses the
   experiment into the parent scope (already KILLED at QR=0.707).

4. **Re-open as `_v2` only when P1/P2/P3 all PASS.** Probe is cheap
   (<1 s); KILL-on-probe is the correct route and must not be short-circuited.

## References

- He et al. 2016 arXiv:1512.03385 — residual learning, zero-init identity.
- Bachlechner et al. 2020 arXiv:2003.04887 — ReZero: output-scale warmup.
- Kirkpatrick et al. 2017 arXiv:1612.00796 — EWC, catastrophic forgetting.
- Finding #403 (`exp_m2p_qwen4b_sft_residual`) — Qwen3-4B, QR=1.175.
- Finding #447 (`exp_p1_t2_sft_residual_gemma4`) — Gemma 4 parent, QR=0.707 KILLED.
- Finding #600 (`exp_followup_sft_behavioral_lora_scale_5`) — 5th precondition-probe KILL; identical blocker, QR ≥ 0.90 tolerance.
