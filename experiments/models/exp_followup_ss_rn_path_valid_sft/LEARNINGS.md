# LEARNINGS.md — exp_followup_ss_rn_path_valid_sft

**Verdict:** KILLED (precondition-probe, 6th this loop) | 2026-04-19

## Core Finding
Finding #403 (SFT-Residual M2P, QR=1.175 on Qwen3-4B) replication on Gemma 4
under the tighter K1572 tolerance (|acc_final − 74.4%| ≤ 5pp) is
**undecidable from current repo state** — same blocker as Finding #600,
different KC band. K1572 routes FAIL-unmeasurable: P1 (T2.1
`adapters.safetensors` missing, gitignored), P2 (no personalization corpus
disjoint from GSM8K), P3 (upstream T2.1 KILLED 2026-04-18 metric-swap +
CoT truncation). No training executed; all three preconditions pre-registered
in MATH.md before the probe ran.

## Why
Infrastructure, not mechanism. Theorems A/B/C (residual zero-init, ReZero
warmup, EWC data separation) still hold. The three independent precondition
FAILs are structural: (a) MLX LoRA `.safetensors` never committed,
(b) GSM8K replay is the known Gemma 4 parent failure mode (QR=0.707),
(c) upstream KILL propagates by standing rule #3. Substitution (random
B_sft, GSM8K as persona data) reclassifies the experiment — T2.4 already
closed that region at QR=−5.89.

## Duplicate-scope observation
This experiment and `exp_followup_sft_behavioral_lora_scale_5` (Finding #600)
target the same replication claim with different KC bands (≤ 5pp here, QR ≥
0.90 there). Both share the same three preconditions and the same
infrastructure blocker. After the class-level unblock (T2.1 rerun + persona
corpus staged), **only one of the two should be re-opened** — the other
becomes redundant. Recommend the tighter-band version (this one) be the
survivor, since QR ≥ 0.90 is a corollary of `|Δ| ≤ 5pp`.

## Implications for Next Experiment
1. **Class-level unblock via single upstream rerun.** Re-running
   `exp_p1_t2_single_domain_training` at `LORA_SCALE=5` regenerates the
   three `.safetensors` (math/code/medical) and unblocks ≥4 downstream
   experiments. Same tag `scale-safety` already on the sweep sibling.
2. **Stage persona-tagged corpus before V2.** Disjoint-distribution
   persona-prefixed queries (Theorem C). Not GSM8K.
3. **Gitignore audit is systemic.** ap-017 now exercised on ≥10 confirmed
   instances. Weight files must be git-tracked or staged locally with
   a hash manifest before any downstream experiment can be claimed.
4. **Do not re-open this experiment until P1∧P2∧P3.** Probe is <1 s;
   KILL-on-probe is the correct route.

## Antipattern Refs
- mem-antipattern-017 (PRE-FLIGHT DIR-EXISTS ≠ WEIGHTS-EXIST) — now 10th instance.
- Standing rules #1–#4 in `.ralph/current_direction.md` exercised cleanly.

## References
- Finding #403 (Qwen3-4B, QR=1.175) — positive evidence this replicates elsewhere.
- Finding #447 (`exp_p1_t2_sft_residual_gemma4`, QR=0.707 KILLED) — parent failure this parameterization was designed to bypass.
- Finding #600 (`exp_followup_sft_behavioral_lora_scale_5`) — 5th precondition-probe KILL, identical blocker.
