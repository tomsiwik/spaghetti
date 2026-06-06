# PAPER.md — exp_hedgehog_data_augmentation_prompt_rephrase

**Verdict: PROVISIONAL (design-only; KCs K1877/K1878 untested — implementation deferred to `exp_hedgehog_data_augmentation_prompt_rephrase_impl`)**

## Claim

Rephrasing each Hedgehog training prompt 5× at temperature 1.2 — all other
Hedgehog components held identical (politeness axis reuse from F#683, 26B Gemma 4
teacher with π_Polite in context, rank-8 LoRA on `(v_proj, o_proj)`, cos-sim
per-layer loss, 800 steps, same seed) — produces a student adapter whose (a)
downstream politeness-axis behavioral quality exceeds the non-augmented adapter's
quality by > 3 pp (K1877 target), and (b) per-layer cos-sim variance across the
last 200 training batches in the augmented arm is > 0.10 (K1878 proxy — training-
stability signal). The ablation is a DATA-AUGMENTATION-ABLATION — the NEW 5th
sub-type in the Hedgehog-ablation super-family (cousin of loss-variant, layer-
selection, hyperparameter ablations). It tests whether prompt-rephrasing
augmentation is a net-positive regularizer or a net-negative noise injector on
cos-sim distillation.

## Scope (this iteration)

This iteration executes **design-only** — lifting the sibling-Hedgehog-axis
PROVISIONAL precedent (F#683, F#684, F#696, F#697, F#717, F#718 — axis-extension
sub-type; F#719 — loss-variant-ablation sub-type 1st-instance). The scaffold in
`run_experiment.py` loads `mlx.core`, logs memory, writes `results.json`, and
raises `NotImplementedError` in seven phases that require the ~12–15 h two-
adapter pipeline (Phase 0 politeness-axis corpus reuse from F#683; Phase 0.5
rephrase generation + semantic-equivalence QA gate; Phase A teacher attention
capture on both corpora; Phase B_base non-augmented cos-loss student training;
Phase B_aug augmented cos-loss student training with per-step cos-sim logging;
Phase C K1878 cos-sim variance over last 200 steps; Phase D K1877 blind-paired
behavioral-quality judge).

A dedicated `_impl` follow-up (`exp_hedgehog_data_augmentation_prompt_rephrase_impl`,
P=3) is filed inline this iteration per `mem-antipattern-impl-follow-up-delegation`
remedy. K-IDs K1877/K1878 inherit verbatim into the `_impl` row.

## Prediction vs measurement

| KC | Prediction | Kill condition (KILL if TRUE) | Measurement (this iter) |
|---|---|---|---|
| K1877 target behavioral-quality Δ | `Δ = (augmented) − (non-augmented) ∈ [−1.0, +5.0] pp`; mean +2.0 pp (Δ < +3 pp expected → K1877 FAIL expected per §3.2 collapse-to-equivalence hypothesis) | `Δ > +3 pp` strictly (augmented beats non-augmented by > 3 pp) | not measured (Phase B_base + Phase B_aug not implemented) |
| K1878 proxy cos-sim variance | `variance(aug cos-sim last 200 steps) ∈ [0.06, 0.18]`; mean 0.11 (variance > 0.10 expected → K1878 PASS expected — broader prompt diversity increases batch-to-batch variance) | variance > 0.10 strictly | not measured (Phase B_aug training log not produced) |

Both KCs locked pre-run; no post-hoc relaxation. Verdict is PROVISIONAL because
nothing was measured — design fidelity only. Rephrase hyperparameters locked:
5× depth, temperature 1.2, 26B Gemma 4 rephraser (no π_Polite context),
semantic-similarity floor 0.7, drift-rate fail-abort at 20%.

## Why K1877 predicts Δ < +3 pp (most-likely outcome: PROVISIONAL target-FAIL)

Per MATH.md §3.2 effective-training-set analysis: non-augmented arm sees 200
unique pairs over 4 epochs; augmented arm sees 1000 rephrased pairs over 0.8
epochs. Broader coverage, fewer repeats per pair — canonical regularization
regime (Wei 2024). BUT: at temperature 1.2 on Gemma 4 (over-regularized teacher),
rephrases partially collapse to semantically-equivalent stylistic variants. On
the politeness axis specifically, Finding #469 (this repo) saw +1–3 pp gain on
*bias-axis* tasks (similar linguistic-register) but NOT on code-axis. Expected
effect size is in the ambiguous range (+0.5 to +4 pp); mean prediction +2.0 pp
falls below the +3 pp JND.

If K1877 AND K1878 both PASS: augmentation adds measurable quality AND
measurable variance cost — "regularization-via-noise" regime. Adopt with
variance-aware LR schedule.

If K1877 FAIL AND K1878 PASS (mean prediction): augmentation adds variance
without quality — skip it.

If K1877 FAIL AND K1878 FAIL: 5× rephrase collapsed to redundancy; no signal,
no cost variance. Skip augmentation.

If K1877 PASS AND K1878 FAIL: quality gain without variance — free lunch.
Adopt as default.

## Why K1878 is NOT §5-tautological (intra-variant threshold, NOT inter-variant delta)

K1878 is a SINGLE-VARIANT absolute threshold: "variance(augmented cos-sim over
last 200 steps) > 0.10". It does NOT compare augmented-arm cos-sim to non-
augmented-arm cos-sim. Per `mem-antipattern-tautological-inter-variant-delta`
§5, the anti-pattern fires on inter-variant DELTAS without base anchoring;
intra-variant absolute thresholds are orthogonal to §5.

K1877 is inter-variant (augmented − non-augmented) BUT grounded to external
ground truth (F#683 rubric judge scores on held-out prompts). Per §5 definition,
claims grounded to external ground truth are NOT tautological. §5 does NOT fire
on K1877.

## Why this is NOT F#666-pure-standalone preempt-KILLable

K1877 is a TARGET metric (behavioral quality — named in rule 1007). K1878 is a
proxy (training stability). Per F#666, every proxy KC must be paired with a
target KC; here K1878 is paired with K1877. The pairing IS the design. Unlike
F#720 (MSE loss variant — both KCs pure PPL/cos-sim proxy → F#666-pure KILL) or
F#722 (temperature sweep — both KCs pure cos-sim delta → F#666-pure KILL), this
experiment satisfies F#666 target-gating.

## Scope-preservation explicit rejections (antipattern-t)

The following "silent downscales" are explicitly out of scope in `_impl`:

- **Axis mismatch between arms.** Both arms MUST train on politeness (F#683
  reuse). Varying axis across arms breaks the A/B.
- **Hyperparameter mismatch between arms.** Rank, scale, targets, steps, optimizer,
  seed, batch all locked identical across arms. Only training-data presentation
  (1× vs 5× rephrased) differs.
- **Rephrase temperature drift.** Locked at 1.2. Running at 1.0 or 1.5 without
  filing a separate follow-up would silently test a different hypothesis.
- **Rephrase depth drift.** Locked at 5×. Running 3× or 7× would test a
  different hypothesis (depth sweep belongs in a separate follow-up).
- **Rephrase model substitution.** Locked at 26B Gemma 4 teacher (A2). Using
  E4B or a third model class adds a confound.
- **π_Polite context during rephrasing.** Rephrasing is axis-neutral (no
  π_Polite); using π_Polite during rephrasing would convert this to a "synthetic
  polite prompts" experiment, a different hypothesis.
- **Teacher proxy.** Substituting E4B for 26B teacher erases the teacher-with-
  context gap the distillation depends on (F#683).
- **F#683 rubric drift.** K1877 judge uses F#683 rubric verbatim. Changing the
  rubric decouples this ablation from the sibling baseline.
- **N_STEPS asymmetry.** Both arms run 800 steps (augmented arm sees fewer
  epochs by design). Running aug at 4000 steps because "we have 5× more data"
  would test training-time, not augmentation.
- **Drift-gate relaxation.** If too many rephrases drop below semantic floor,
  the fix is to lower rephrase temperature (file follow-up) or raise the floor,
  NOT to let low-quality rephrases into training.

## Measurement blockers (to resolve in `_impl`)

1. **Phase 0 politeness corpus reuse** — requires F#683 `_impl` to have landed
   (corpus + held-out eval slice on disk). F#683 is still PROVISIONAL — this
   experiment has a **transitive blocker on F#683 _impl**.
2. **Phase 0.5 rephrase generation** — 26B Gemma 4 teacher residency (~14 GB),
   generate(temperature=1.2) loop over 200 prompts × 5 rephrases = 1000 calls,
   plus semantic-equivalence QA gate (sentence-transformer or model-based judge)
   with drift-rate accounting.
3. **Phase A teacher attention capture** — 26B Gemma 4 + π_Polite in context,
   capture `attn_output` per-layer for 42 layers on BOTH corpora (original +
   augmented). Peak-memory load-bearing on 48 GB (F#673); sequential-phase
   eviction required.
4. **Phase B_base custom MLX training** — per-layer attention-output hooks,
   `nn.value_and_grad + AdamW`, `mx.eval + mx.clear_cache` between batches. Not
   available via `mlx_lm.lora` CLI.
5. **Phase B_aug custom MLX training** — same training loop as B_base PLUS
   per-step cos-sim logging to disk for K1878 variance computation.
6. **Phase C K1878** — variance aggregation over logged cos-sim time series,
   last 200 batches.
7. **Phase D K1877** — blind-paired 50-prompt politeness-axis auto-judge using
   F#683 rubric. Order-swap 50/50 to control for position bias. Pinned judge.

Shared blockers:
- **26B Gemma 4 teacher model not cached** (~14 GB) — common to 8+ Hedgehog-
  framework `_impl` dependents; standalone prereq task candidate per F#718
  analyst guidance, re-affirmed at F#719/F#721/F#722.
- **F#683 politeness `_impl` has not landed** — transitive blocker for this
  ablation (reuses F#683 corpus + rubric + held-out set). If F#683 `_impl`
  stalls indefinitely, this ablation should be re-scoped to whichever Hedgehog
  axis `_impl` lands first.

## Assumptions (from MATH.md §8, restated for paper context)

A1 Politeness axis has the most mature teacher-capture pipeline among siblings
   (F#683); axis-locking is the variance-reduction choice for the ablation.
A2 Rephrasing model = teacher (26B Gemma 4), no π_Polite context — axis-neutral
   rephrase avoiding third-model-class confound.
A3 5× rephrase depth is Wei 2024 median; 3× / 7× are follow-up sweeps.
A4 Blind-paired judge on 50 held-out pairs detects Δ ≥ +3 pp at α=0.05
   (F#683 power).
A5 Single-iteration cap (30 min / 40 tool calls) — ~12–15 h two-arm pipeline
   explicitly out of scope this iteration.
A6 LORA_SCALE = 6.0 ≤ 8 per F#328/F#330.
A7 Only K1877 + K1878 pre-registered. No cross-axis, cross-rank, rephrase-
   temperature-sweep, or rephrase-depth-sweep KCs — follow-ups, NOT retro KCs.
A8 Drift gate locked at ≥ 0.7 semantic similarity; > 20% drift → fail-abort.
A9 F#702 hygiene-patch applied (platform = local-apple, success_criteria populated;
   references field INCOMPLETE matching F#702 precedent — global ref library CLI
   limitation). `mem-impossibility-f666pure-saturation-implies-f702-unavailable`
   does NOT fire (K1877 is a target KC, not F#666-pure).
A10 **Data-augmentation-ablation is a NEW 5th sub-type** within Hedgehog-ablation
    super-family. Super-family ledger now 5 sub-types / 12 total instances:
    axis-extension (6 — closed after F#718); loss-variant-ablation (2 — F#719
    PROVISIONAL + F#720 KILL); layer-selection-ablation (1 — F#721 KILL);
    hyperparameter-ablation (1 — F#722 KILL); **data-augmentation-ablation (1
    — this, PROVISIONAL)**. KC-design bifurcation (paired-target → PROVISIONAL;
    pure-proxy → KILL) axis-invariant across super-family; this filing is
    paired-target-anchored → PROVISIONAL.
A11 **8th Hedgehog-framework PROVISIONAL** in the hard-defer pile. Pile was
    7 designs / 0 measurements before; this makes 8 designs / still 0
    measurements. Pile growth explicitly tracked in analyst guidance; 26B
    teacher cache remains standalone-prereq-task candidate blocking 8+
    dependents.
A12 Transitive blocker on F#683 `_impl` acknowledged.

## Sibling position — Hedgehog-ablation super-family

This is the **8th Hedgehog-framework PROVISIONAL** and **1st data-augmentation-
ablation sub-type** (NEW):

| # | Finding | Sub-type | Classification | Status |
|---|---|---|---|---|
| 1 | F#683 | axis-extension | politeness | PROVISIONAL |
| 2 | F#684 | axis-extension | procedural refactor | PROVISIONAL |
| 3 | F#696 | axis-extension | JS domain | PROVISIONAL |
| 4 | F#697 | axis-extension | Python domain | PROVISIONAL |
| 5 | F#717 | axis-extension | Rust domain | PROVISIONAL |
| 6 | F#718 | axis-extension | SQL domain (closed domain sub-family) | PROVISIONAL |
| 7 | F#719 | loss-variant-ablation | KL-divergence vs cos-sim | PROVISIONAL |
| — | F#720 | loss-variant-ablation | MSE vs cos-sim | KILLED (F#666-pure triple-fire) |
| — | F#721 | layer-selection-ablation | top-6 layers | KILLED (F#666-pure triple-fire) |
| — | F#722 | hyperparameter-ablation | teacher temperature sweep | KILLED (F#666-pure triple-fire, 5th) |
| 8 | **this** | **data-augmentation-ablation** | **5× prompt rephrase (temp 1.2)** | **PROVISIONAL (design-only)** |

Classification: **data-augmentation-ablation** is NEW 5th sub-type in the
super-family. Super-family now 5 sub-types / 12 total instances (8 PROVISIONAL,
4 KILLED). KC-design bifurcation pattern confirmed axis-invariant: paired-
target-anchored → PROVISIONAL; pure-proxy → F#666-pure preempt-KILL.

## References

- Moudgil et al., Hedgehog attention distillation, arxiv:2604.14191 §3.1 eq. 6
  (cos-sim loss definition).
- Wei et al. (2024) prompt-rephrasing for SFT — +1–3 pp on GSM8K / HumanEval.
- Alpaca / Self-Instruct (Stanford, 2023) — canonical LLM-self-rephrase at
  temperature 1.0.
- Finding #469 (this repo): diversity-via-rephrase +1.3× effective training on
  political-bias axis; NO gain on code-axis. Effect is axis-dependent.
- Pierre F#627 (v_proj+o_proj LoRA sufficiency); F#614/F#536 (thinking-mode
  load-bearing); F#328/F#330 (LORA_SCALE ≤ 8); F#673 (mx.clear_cache between
  phases, MLX audit 2026-04-17).
- F#666 target-gating convention; F#702 hygiene-patch PROVISIONAL;
  F#683/F#684/F#696/F#697/F#717/F#718 axis-extension precedents; F#719
  loss-variant-ablation 1st-instance precedent.
- `mem-antipattern-tautological-inter-variant-delta` §5 — K1878 NOT §5 (intra-
  variant threshold); K1877 NOT §5 (grounded to external rubric judge).
- `mem-impossibility-f666pure-saturation-implies-f702-unavailable` — inapplicable
  here (K1877 target present).
- `mem-pattern-triple-fire-hierarchy-axis-invariant` — inapplicable here
  (this is PROVISIONAL, not preempt-KILL).

## Handoff

- Status: PROVISIONAL.
- `_impl` follow-up: `exp_hedgehog_data_augmentation_prompt_rephrase_impl`
  filed inline at P=3, KCs K1877/K1878 inherited verbatim.
- Hygiene-patch applied: platform set to `local-apple`, success_criteria #TBD
  populated before `experiment complete`. References field INCOMPLETE (F#702
  precedent: global ref library CLI-linking limitation).
- Transitive blocker: F#683 `_impl` must land before this ablation's `_impl`
  can run (corpus + rubric reuse). Documented in analyst handoff.
- Sub-family position: 5th sub-type (NEW data-augmentation-ablation), 8th
  PROVISIONAL in hard-defer pile.
