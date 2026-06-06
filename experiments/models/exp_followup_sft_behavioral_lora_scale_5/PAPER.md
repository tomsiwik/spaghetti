# PAPER.md — Finding #403 Gemma 4 Replication (output_scale·head(z) + personalization data)

**Status: KILLED** | Precondition-probe | 2026-04-19

## Abstract

This experiment was registered to replicate Finding #403 (SFT-Residual M2P,
Qwen3-4B, QR=1.175) on Gemma 4 E4B 4-bit, using the correct parameterization
`B_applied = B_sft + output_scale(t) · head(z)` with head zero-init and
warmup, and personalization-distinct training data (explicitly NOT SFT
replay). The motivation was to avoid the parent experiment's failure mode
(`exp_p1_t2_sft_residual_gemma4`, QR=0.707, KILLED): same-domain GSM8K replay
triggered catastrophic forgetting because `∂L/∂ΔB = ∂L/∂B_applied` under the
zero-init-ΔB reparameterization.

Routing via the pre-registered precondition probe (MATH.md §Preconditions),
all three required preconditions FAIL:

- **P1 (B_sft on disk):** FAIL. `adapters.safetensors` gitignored, never
  recoverable from the repo.
- **P2 (personalization corpus):** FAIL. No persona-tagged data staged; SFT
  replay is the known failure mode (parent QR=0.707).
- **P3 (upstream T2.1 verdict):** FAIL. T2.1 KILLED 2026-04-18 via
  metric-swap + format-artefact.

K1565 routes FAIL (unmeasurable, not measured-and-fell-short). Verdict =
KILLED; `results.json.verdict="KILLED"`, `all_pass=false`.

This is the 5th precondition-probe KILL this loop. Same infrastructure
blocker as `exp_followup_sequential_activation_compose_real` (2026-04-18):
gitignored parent adapter artefacts.

## Prediction vs Measurement

| Quantity | Predicted (under preconditions) | Measured | Pass? |
|---|---|---|---|
| P1 T2.1 `.safetensors` on disk | exists, size > 0 | missing (only `adapter_config.json` stub) | FAIL |
| P2 personalization corpus on disk | persona-tagged data disjoint from GSM8K | none staged; only GSM8K `train.jsonl` (1800 lines) present | FAIL |
| P3 T2.1 verdict ≠ KILLED | valid upstream | verdict=KILLED (2026-04-18 metric-swap + format-artefact) | FAIL |
| K1565 QR = acc_final / acc_step0 | ≥ 0.90 (Qwen3-4B reference QR=1.175) | unmeasurable (no B_sft; no persona data; invalid upstream) | FAIL |

## Results

| Field | Value |
|---|---|
| verdict | KILLED |
| all_pass | false |
| ran | false (probe-only) |
| is_smoke | false |
| probes | 3 × FAIL |
| KC K1565 | FAIL (unmeasurable) |

## Why It Failed (infrastructure, not mechanism)

This experiment's mathematical claim is unchanged from Finding #403 and
remains plausible under MATH.md Theorem 1. The failure is infrastructural:

1. **P1 missing B_sft.** Gemma 4 M2P requires the SFT-trained B-matrices.
   MLX LoRA `adapters.safetensors` from T2.1 were produced but gitignored
   and never committed. Only the `adapter_config.json` stub (a registry
   entry, not a weight artefact) is on disk. No mathematically-valid
   substitute exists: substituting random B_sft would put the experiment
   in the same class as T2.4 (random-init ΔB, QR=−5.89 KILLED) and
   invalidate the replication claim.

2. **P2 missing personalization corpus.** Finding #403's mechanism depends
   on data separation (Theorem C, EWC): training on a distribution
   different from the SFT data attenuates catastrophic forgetting.
   Substituting GSM8K (the only T2.1 data present) re-creates the parent
   failure mode. No persona-tagged corpus is staged in this repo.

3. **P3 upstream KILL.** Even if P1 weights were recovered, the
   verdict would inherit T2.1's 2026-04-18 KILL (metric-swap MedMCQA↔MedQA
   + format-artefact max_tokens=256 CoT truncation). Downstream claims on
   an invalid B_sft substrate cannot be stated either way.

## Verdict-Consistency Pre-flight

1. `results.json["verdict"] == "KILLED"` ✓ (matches DB status killed route).
2. `results.json["all_pass"] == false` ✓.
3. PAPER.md verdict line = "KILLED" — no hidden PROVISIONAL/SUPPORTED.
4. `is_smoke == false`; not a smoke-mode run.
5. KC K1565 text and threshold unchanged between MATH.md pre-registration
   and results.json (`git diff MATH.md` will show the file is newly added,
   not edited post-hoc).
6. No `type: fix` antipattern applies: no composition math bug (no
   composition executed); no tautological routing (no routing executed);
   no unsafe adapter scale (no training executed); no `shutil.copy` as
   new adapter; no hardcoded `"pass": True`; `is_smoke=False` honestly
   reported; no eval-template truncation; no proxy-model substitution
   (target Gemma 4 never loaded); KC measures the right object by
   routing it FAIL when the object is absent.

All six pre-flight checks pass for KILLED (not for supported).

## Impossibility Structure (of the KILL itself)

The KILL is inherent in the precondition-probe discipline, not in the
replication claim. Under PLAN.md §KC discipline, a pre-registered
precondition routing failure IS a KILL:

- Finding #403 replication on Gemma 4 is **testable iff** B_sft exists AND
  persona data exists AND upstream B_sft is valid.
- All three are false at claim time.
- Therefore the replication claim is **undecidable from this repo state**.
  Undecidable ≠ supported. KILLED is the honest routing.

## Implications

1. **Re-open as `exp_followup_sft_behavioral_lora_scale_5_v2`** after the
   unblock path runs:
   - Rerun `exp_p1_t2_single_domain_training` at `LORA_SCALE=5` to regenerate
     `adapters.safetensors` (matches parent scale-safety tag).
   - Stage a personalization corpus (e.g. persona-prefixed math queries,
     disjoint from GSM8K SFT split).
   - Re-run this probe; if all PASS, implement the `output_scale(t) · head(z)`
     training loop and measure K1565.

2. **Class-level unblock.** The same rerun of `exp_p1_t2_single_domain_training`
   unblocks at least 4 downstream experiments (this one,
   `exp_followup_sequential_activation_compose_real`,
   `exp_followup_hypernetwork_residual`, and `exp_followup_ss_rn_path_valid_sft`).
   A single upstream rerun is cost-efficient.

3. **Do not substitute.** Tempting substitutions (random B_sft, GSM8K as
   persona data, a different base model's B_sft) all reclassify the
   experiment and invalidate the Finding #403 replication claim. Wait for
   the real preconditions or close the experiment permanently.

4. **Gitignore audit.** The root cause is that `adapters.safetensors` is in
   `.gitignore`. A repo-wide standing rule (memory ap-022 or similar) is
   appropriate: adapter weight files MUST be tracked by git or staged
   locally with a hash manifest before any downstream experiment can be
   claimed against them.

## Assumptions Logged (per PLAN.md §1, Ralph autonomy rule)

- Assumed `personalization corpus` requires a disjoint-distribution
  persona-tagged dataset rather than any non-GSM8K dataset. Justification:
  Theorem C (EWC) is about loss-surface separation; persona-tagging is the
  cleanest operationalization consistent with Finding #403's design, which
  used Qwen3-4B personalization queries.
- Assumed `LORA_SCALE=5` is the safe scale for the parent rerun.
  Justification: this experiment's tag list includes `scale-safety`, and
  the `exp_followup_lora_scale_safe_sweep` experiment (same tag, same
  priority) is sweeping scales around 5 as the audit-recommended range.
  This is a documentation assumption; no numerical claim depends on it.

## References

- He et al. 2016 arXiv:1512.03385 — residual learning
- Bachlechner et al. 2020 arXiv:2003.04887 — ReZero output-scale warmup
- Kirkpatrick et al. 2017 arXiv:1612.00796 — EWC
- Finding #403 — `exp_m2p_qwen4b_sft_residual` (QR=1.175 on Qwen3-4B)
- Finding #447 — `exp_p1_t2_sft_residual_gemma4` (QR=0.707 on Gemma 4, KILLED)
- Parent V2 pattern — `exp_p1_t2_sft_residual_gemma4/PAPER.md` §V2 Audit Rerun
