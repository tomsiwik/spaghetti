# REVIEW (adversarial) — exp_spark_adapter_as_head_compass

VERDICT: KILLED (sealed). kill-2310 fired on both clauses; refutation is real.

## Integrity / mock checks — all pass
- `verify-experiment.sh` exits 0; `is_smoke:false`; real model `gemma-4-e4b-it-4bit`, real adapter
  `data/adapters/math/adapters.safetensors`, n_eval=80 (>=60 quorum), 8256s runtime. No mock/hardcode.
- MATH.md model == loaded model. Kill threshold (+4pp both clauses) is in MATH.md and matches the code
  verdict logic (run_experiment.py L346-350). Experiment is untracked (single unstaged state) — no
  post-hoc goalpost move possible/observed.

## Point 1 — read-only constraint held (CONFIRMED)
`AmplifyQProj.__call__ = self.base(x) * self._mask`: pure coordinate rescale, no additive B@A.
`build_layer_masks` writes only scalar gamma or 1.0. `assert_no_delta_in_mask` asserts every entry in
{1.0, gamma} and ran on every arm. Delta path is a disjoint class (DeltaQProj), used only in the labeled
refuting arm. The compass/random arms never inject Delta.

## Point 2 — compass=random tie is real, not a bug (CONFIRMED)
Compass and random head sets are COMPLETELY DISJOINT (0 overlap, verified from results.json), different
masks, yet both hit 50.0 EM at best gamma. Compass even dips to 48.75 at gamma=1.2 while random holds 50.0.
The adapter's per-head B-energy ranking carries no selection signal beyond generic small-gamma head
up-weighting. Genuine tie.

## Point 3 — cross-experiment discrepancy (DIAGNOSED, neither experiment invalidated)
base EM 46.25 (here) vs 81.25 (exp_spark_intra_rank_phase_gate) is a pure HARNESS difference, same model
+ same adapter:
- phase_gate: apply_chat_template(enable_thinking=True), prompt instructs "End with '#### '", max_tokens
  1024, float parser w/ strip_thinking + multi-pattern.
- compass: default template (thinking OFF), weak "Answer:" prompt that never asks for '####', max_tokens
  800, string-== parser whose last-number fallback keeps trailing '.' (e.g. "72." != "72") — depresses
  absolute EM but applied identically to ALL arms, so within-experiment margins are unbiased.

Toxicity-narrative reconciliation: the adapter EFFECT SIGN FLIPS with harness.
- phase_gate (thinking): base 81.25 -> uniform-math applied 71.25 = -10pp  => F#866/873 toxicity holds.
- compass (no-thinking): base 46.25 -> delta-applied 70.0 = +23.75pp.
The math adapter imposes a structured CoT->'####' format. When the base substrate is already near ceiling
(thinking harness), the adapter's lower-quality forced structure hurts (toxicity). When the base is
crippled (no-thinking, weak prompt, noisy parser -> 46.25), that imposed structure helps. Substrate
toxicity is REAL but harness-relative: it manifests only when the base is near its own ceiling. This does
NOT undermine either experiment — F#866/873 stands within its measurement harness; the compass kill rests
on within-harness relative margins immune to the absolute-EM offset.

## Why the kill is sound
- Clause (a): compass-base = +3.75pp < +4.0.
- Clause (b): compass-random = +0.00pp < +4.0 (identical EM, disjoint heads -> no compass signal).
- Clause 2: delta_is_best_arm=true (delta 70.0 > base 46.25, > compass 50.0). Value lives in APPLYING
  Delta, not in reading it as a read-only selector — the exact hypothesis under test, refuted.
results.json verdict, all_pass=false, and PAPER.md verdict line all agree. No tautology (3 disjoint arms,
matched count/gamma, pre-registered numeric threshold).
