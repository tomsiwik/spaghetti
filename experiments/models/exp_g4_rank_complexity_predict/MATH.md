# MATH: Domain-complexity → optimal LoRA rank on Gemma 4 E4B 4-bit

## Theorem (pre-registered, KC #1629 + paired target KC)

Let `{D_1, ..., D_5}` be five disjoint Gemma 4 E4B 4-bit LoRA training
corpora (math, code, medical, finance, legal). For each domain `D_d`
define:

- `c(D_d)` — a pre-registered scalar *domain-complexity* metric computed
  from the corpus alone (no training needed). Candidates come from
  Finding #153's ranked list (Shannon entropy of token distribution,
  mean type–token ratio, Gavish–Donoho-calibrated TF-IDF spectral
  effective-rank). The predictor is fixed before any rank-sweep data is
  inspected; the experiment reports the best of the three *by
  construction*, not by post-hoc selection.
- `r*(D_d)` — the minimal rank on the sweep `r ∈ {2, 4, 6, 8, 12}` at
  which the fine-tuned adapter reaches ≥ 95% of its own `r=12`
  behavioral score on the domain's held-out benchmark (GSM8K / HumanEval
  / MedMCQA / FinQA-dev / CaseHOLD-dev, all at `max_tokens ≥ 512`,
  `enable_thinking=True`).

**KC #1629 (PROXY — predictor rank correlation):** Spearman
`ρ ( c, r* ) ≥ 0.85` across the 5 domains, using the pre-registered
complexity metric.

**KC #1629-T (TARGET — behavioral-quality check, paired per F#666):**
Routing each domain to its predicted rank `r̂(D_d) := snap(r*-estimate
from c(D_d), sweep)` yields a mean held-out behavioral score that is
within `2.0pp` of the per-domain `r=12` oracle. This is the real claim:
the predictor is useful only if acting on it preserves downstream
quality.

The pairing is mandatory by PLAN.md §1 / Finding #666: a Spearman
threshold on a rank label is a proxy for the behavioral utility of
using the predictor; killing on the proxy alone would violate the
target-gated kill rule.

## Why (prior-theorem citation)

Finding #153 (`exp_adaptive_rank_selection` v2, supported, micro synthetic,
d ∈ {64, 128, 256}): `energy_rank_99` achieved Spearman `ρ = 0.942–0.987`
with `r*` across 9 synthetic conditions × 15 domains; `compound_r95_t2.0`
(Finding #154) hit `95.0%` mean within-2x across 12 conditions. These
are synthetic-spectrum results — the open question is whether a
predictor computed on *corpus text* (not a known ground-truth Δ
spectrum) transfers to a real 4-bit Gemma 4 training regime where the
quantization-noise floor shifts both `c(D)` and `r*`.

Mechanism (Eckart–Young–Mirsky + Gavish–Donoho 2014, `arxiv:1305.5870`):
a LoRA adapter approximates a residual `ΔW` whose task-carrying
singular directions survive the optimization-noise floor. Domain
complexity — operationalised as the effective-rank of a spectral proxy
computed on the corpus — *should* correlate monotonically with the
minimum rank at which the trained adapter captures the task-surviving
directions, because the width of the residual spectrum is bounded below
by the domain's intrinsic variability.

## Failure mode this makes impossible

A constant-rank heuristic (always r=6, always r=12) over-provisions
compute on narrow domains (r* ≪ 6) and under-provisions on wide
domains (r* ≫ 6). Finding #153's synthetic result says the
over/under-provisioning is predictable to within a factor of 2 from
spectral statistics alone; a real-Gemma-4 correlation of `ρ ≥ 0.85`
would make the over/under-provisioning *statistically impossible* on
the target platform. A failure (ρ < 0.85) tells us the synthetic
spectrum is not the right proxy for a 4-bit base — and the target KC
either corroborates or contradicts that proxy verdict per F#666.

## Platform

- Base: `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B 4-bit)
- Adapter: LoRA on `v_proj+o_proj`, `LORA_SCALE=5` (Finding #586
  scale-safety bound); audit-2026-04-17 cohort standard.
- Rank sweep: `r ∈ {2, 4, 6, 8, 12}` per domain — 25 trainings total.
- Domains: math, code, medical, finance, legal (disjoint corpora).
- Eval harness: GSM8K (math), HumanEval (code), MedMCQA (medical),
  FinQA-dev (finance), CaseHOLD-dev (legal); all at `max_tokens ≥ 512`
  with `enable_thinking=True` (Finding #530 / #627; shorter budgets
  truncate Gemma 4 CoT and produce the base-=-0% format artifact).
- mlx-lm version pinned to the version used for the upstream training
  regeneration (API drift between 0.21 and 0.31 changes LoRA optimizer
  state layout).

## Pre-registered preconditions (UNMEASURABLE routing)

Per the `audit-2026-04-17` cohort standing rule (heavy retraining —
25 adapters × ≥30min ≈ 12+ h MLX — is out of scope for a researcher
iteration), KC #1629 and KC #1629-T are measurable *only if* three
upstream artifacts exist on disk. If any precondition FAILs, the
experiment is **KILLED on UNMEASURABLE** — not relaxed, not deferred,
not inconclusive. This mirrors the pattern registered by sibling
`exp_g4_snr_rank_predictor`.

- **P1 — five-domain rank-sweep adapters.** For each `d ∈ {math, code,
  medical, finance, legal}` and each `r ∈ {2, 4, 6, 8, 12}`, an
  `adapter_model.safetensors` or `adapters.safetensors` with non-zero
  bytes exists at `micro/models/exp_p1_t2_single_domain_training/
  adapters/{d}_r{r}/` (or matches r=6 at `adapters/{d}/` for the central
  sweep point). Minimum: 25 non-zero safetensors (5 domains × 5 ranks).

- **P2 — per-domain complexity metric source corpus.** The five-domain
  corpora exist as `.jsonl` files under
  `micro/models/exp_p1_t2_single_domain_training/data/{d}/train.jsonl`
  with ≥ 1k training rows each. The complexity metric `c(D_d)` is
  computed directly from these files and does not depend on training
  — this precondition is about data availability, not training cost.

- **P3 — rank-ablation behavioral baseline.** The upstream training
  experiment `exp_p1_t2_single_domain_training` has `verdict ∈
  {supported, proven, provisional}` and `all_pass: true` with a
  plausible base score (`base_gsm8k_pct > 20.0` — i.e. not the
  max-tokens-truncation format artifact). Without a credible r=12
  oracle per domain, `r*` is undefined and KC #1629 has no ground
  truth.

## Kill criteria (locked)

- **K1629 PASS:** P1 ∧ P2 ∧ P3 hold AND Spearman `ρ(c, r*) ≥ 0.85`.
- **K1629 FAIL:** P1 ∧ P2 ∧ P3 hold AND `ρ < 0.85` — the predictor
  does not transfer from synthetic d≤256 to real 4-bit Gemma 4.
- **K1629 UNMEASURABLE → KILLED:** any of P1/P2/P3 FAIL — the
  measurement cannot be evaluated; experiment is KILLED on the probe,
  blocker recorded, upstream rebuild routed as the fix.
- **K1629-T PASS:** K1629 PASS AND mean behavioral score at predicted
  rank is within 2.0pp of r=12 oracle across 5 domains.
- **K1629-T FAIL:** K1629 PASS AND mean behavioral gap > 2.0pp — the
  proxy passes but behaviorally the predictor is not useful; report as
  a finding about the proxy (F#666-class).
- **K1629-T UNMEASURABLE:** inherits K1629's measurability.

Killing routes: `proxy FAIL ∧ target FAIL` ⇒ kill.
`proxy PASS ∧ target FAIL` ⇒ kill on target, tag proxy tautological.
`proxy FAIL ∧ target PASS` ⇒ not a kill, finding about the proxy.
`UNMEASURABLE` ⇒ kill on precondition blocker.

## Predicted numbers (registered pre-run)

If preconditions hold:
- Null (constant-r=6) Spearman `ρ ≈ 0` by construction.
- Token-entropy `c(D)` Spearman `ρ ≈ 0.6` (weak; captures corpus length
  heterogeneity but not spectral structure).
- TF-IDF effective-rank `c(D)` Spearman `ρ ≈ 0.85–0.95` (candidate
  pre-registered winner, matches F#153 synthetic performance).
- Target gap at predicted rank ≈ `0.5–3.0 pp` — K1629-T borderline.

Honest outcome space — if P1/P2/P3 fail, the experiment is KILLED on
UNMEASURABLE without opinion-forming on `ρ`.

## Assumptions

- `audit-2026-04-17` cohort standing rule applies: 25-adapter retraining
  is out of scope for a single researcher iteration.
- KC #1629 and KC #1629-T are locked at pre-registration; any
  relaxation (within-4x, threshold 0.80, fewer domains) invalidates the
  probe and requires a v2 experiment.
- `mlx-lm` version used for training must match the eventual upstream
  rebuild — otherwise the gradient-SNR / rank-sweep statistics drift.
- This experiment is the 10th+ downstream probe in the cohort (sibling
  `exp_g4_snr_rank_predictor` LEARNINGS warned Ralph against claiming
  a 10th). Completing this honestly (UNMEASURABLE kill, upstream
  rebuild flagged) serves as a reinforcement of the cohort-saturation
  lesson; the LEARNINGS file for this experiment will escalate to a
  memory-level warning.
