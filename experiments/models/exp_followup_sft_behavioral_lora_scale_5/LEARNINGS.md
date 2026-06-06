# LEARNINGS.md — exp_followup_sft_behavioral_lora_scale_5

**Verdict:** KILLED (precondition-probe, 5th this loop) | Finding #600 | 2026-04-19

## Core Finding
Finding #403 (SFT-Residual M2P, QR=1.175 on Qwen3-4B) replication on Gemma 4
is **undecidable from current repo state**. K1565 routes FAIL-unmeasurable:
P1 (B_sft `adapters.safetensors` missing, gitignored), P2 (no personalization
corpus disjoint from GSM8K), P3 (upstream T2.1 KILLED 2026-04-18 via
metric-swap + CoT truncation). No training executed; all three preconditions
pre-registered in MATH.md before probe.

## Why
Infrastructure, not mechanism. Theorem A/B/C (ReZero warmup, residual init,
EWC data separation) still hold. Three independent precondition FAILs are
structural: (a) MLX LoRA `.safetensors` never committed, (b) GSM8K replay is
the known Gemma 4 parent failure mode (QR=0.707), (c) upstream KILL
propagates by standing rule #3. Substitution (random B_sft, GSM8K as persona
data) reclassifies the experiment — T2.4 already closed that region at
QR=−5.89.

## Implications for Next Experiment
1. **Class-level unblock via single upstream rerun.** Re-running
   `exp_p1_t2_single_domain_training` at `LORA_SCALE=5` regenerates the
   three `.safetensors` (math/code/medical) and unblocks ≥4 downstream
   experiments: this one, `exp_followup_sequential_activation_compose_real`,
   `exp_followup_hypernetwork_residual`, `exp_followup_ss_rn_path_valid_sft`.
   Same tag `scale-safety` already on the sweep sibling.
2. **Stage persona-tagged corpus before V2.** Disjoint-distribution
   persona-prefixed queries (Theorem C requirement). Not GSM8K. Not shared
   with T2.1 training split. Without this, V2 re-creates QR=0.707 failure.
3. **Gitignore audit is repo-wide work, not per-experiment.** Ap-017 is
   systemic (now 9 confirmed instances). Adapter weight files must either
   be git-tracked or staged locally with a hash manifest before any
   downstream experiment can be claimed against them. Queue as
   infrastructure ticket, not as a new experiment.
4. **Do not re-open exp_followup_sft_behavioral_lora_scale_5 until all
   three preconditions PASS.** Probe is cheap (<1 min); KILL-on-probe is
   the correct route and must not be relaxed.

## Antipattern Refs
- mem-antipattern-017 (PRE-FLIGHT DIR-EXISTS ≠ WEIGHTS-EXIST) — now 9th instance.
- Standing rules #1–#4 in `.ralph/current_direction.md` all exercised cleanly.

## References
- Finding #403 (Qwen3-4B, QR=1.175) — positive evidence this replicates elsewhere.
- Finding #447 (`exp_p1_t2_sft_residual_gemma4`, QR=0.707 KILLED) — the parent failure the output_scale·head(z) parameterization was designed to bypass.
- Finding #600 (this experiment) — precondition-probe KILL, undecidable from repo.
