# PAPER.md — exp_g4_quantization_aware_adapter_training

**Verdict:** KILLED (preempt-structural)
**Form:** Triple-fire structural KILL — F#666-pure-standalone + F#502/F#646 schema + self-documented predicate-not-met
**Finding registration:** SKIPPED per F#769 closing-note (ledger-explosion antipattern at Nth instance of established cohort)

## §1 Prediction-vs-measurement table

| KC | Predicted (DB-frozen) | Measured | Status | Reason |
|---|---|---|---|---|
| K1920: QAT adapter PPL within 0.05 of full-precision adapter | (proxy) PPL gap < 0.05 | untested | preempt-KILL | proxy-only, no paired target-metric KC per F#666 |
| K1921: QAT training time > 2x standard LoRA training time | (engineering) wall-clock ratio > 2× | untested | preempt-KILL | meta-engineering metric, not a kill-criterion |

Neither KC is a target-metric. There is **no proxy/target pair**. See MATH.md §2.1.

## §2 Why preempt-KILL, not measure

### §2.1 Three independent blockers (any one sufficient)

1. **F#666-pure-standalone**: K1920 is a PPL proxy with no paired behavioral
   target; K1921 is a wall-clock budget gate, not a metric. Per guardrail
   1007, KILL requires both proxy and target to fail; SUPPORT requires both
   to pass. With no target KC, the experiment can produce **neither verdict
   honestly** — only an inconclusive proxy-only report. Canonical prior
   instances: F#700, F#705, F#706, F#707, F#722.
2. **F#502/F#646 schema-incomplete**: `experiment get` shows
   `Success Criteria: NONE` and DB-flagged `⚠ INCOMPLETE`. Per PLAN.md §1
   verdict-consistency, `success_criteria=[]` blocks any `supported`
   verdict regardless of measurement. This is a recognized cohort
   (F#629/F#655/F#769 etc.).
3. **Self-documented predicate-not-met**: the prior researcher's release
   note (DB notes field, frozen 2026-04-25) explicitly conditions
   re-attempt on (a) lock-in of a specific QAT-LoRA paper reference and
   (b) derivation of STE composition with `mlx.QuantizedLinear` (forward
   replacement, since the dequant op has no native grad path in MLX 0.31).
   Both predicates remain `references=[]` and unspecified. Re-attempting
   without resolving them re-creates the >2h budget burn the prior
   researcher already documented.

### §2.2 Category note

This is not a category error like the PROD-deliverable cohort (which mis-
files product specs as experiments). The underlying scientific question
("does QAT preserve adapter behavior?") is a legitimate research
experiment — **it is mis-specified**, not mis-categorized. The KCs need to
be rewritten with a paired target-metric and the references/STE-mechanism
predicates resolved before re-attempt at P≤2.

## §3 Verdict-consistency pre-flight

| Check | Result |
|---|---|
| (1) `results.json["verdict"]` not `"KILLED"` (required for `supported`) | FAILS — verdict IS `"KILLED"`, intentional |
| (2) `results.json["all_pass"] == True` | FAILS — `False`, intentional |
| (3) PAPER.md verdict line free of PROVISIONAL/etc | OK — verdict is `KILLED` |
| (4) `is_smoke == False` | OK — `False`, no smoke run |
| (5) No KC modified between MATH.md and now | OK — KCs reproduced byte-for-byte from DB (K1920/K1921) |
| (6) Type:fix antipattern memories checked | OK — see MATH.md §5 |

Pre-flight is correctly set up to fail the `supported` gate. Verdict is
**KILLED** and routing to reviewer for `experiment update --status killed`.

## §4 Doom-loop self-check

- Prior researcher iteration: KILLED (PROD F#765 super-family no-parent sub-form).
- This iteration: KILLED (F#666-pure-standalone + F#502/F#646 + predicate-not-met).
- Two consecutive KILLs **but on structurally distinct mechanisms** —
  PROD-deliverable-cascade in the prior iteration is a category error;
  F#666-pure-standalone in this iteration is a KC-pairing violation. Not
  A→B→A→B alternation, no loop signal.
- `python3 .ralph/tools/doom_loop.py` exit=0.

## §5 Assumptions

- **A1**: The DB representation of K1920/K1921 is complete; reading the
  notes field confirms but does not contradict the impossibility argument.
  The KC text itself encodes the F#666 violation (PPL gap is a proxy;
  training-time ratio is engineering).
- **A2**: F#769 closing-note's ledger-explosion guidance applies even to
  triple-fire situations where each blocker is independently a closed
  cohort. Filing one finding per blocker × per experiment would explode the
  ledger; reviewer should reuse cohort evidence rather than create per-
  instance findings.
- **A3**: Prior researcher's notes field is authoritative for the
  predicate-not-met blocker — those predicates were set as preconditions
  for re-attempt, not as guidance the next researcher may override. The
  prior researcher had context this researcher does not (>2h budget burn
  observation).

## §6 Suggested follow-ups (content-level, not workflow-required)

- **Re-spec the KC pair**: rewrite as
  - K_proxy: PPL gap on held-out validation set < 0.05 (current K1920)
  - K_target: behavioral task accuracy gap < 1pp on at least one of
    {MMLU-Pro, HumanEval, GSM8K} (NEW)
  with the F#666 paired-KC requirement explicit.
- **Lock the citation**: select either LoftQ (arxiv:2402.10193) or
  arxiv:2310.08659 as the governing equation source; record in DB
  `references` array.
- **Lock the STE-MLX mechanism**: invoke `/mlx-dev` and write a one-page
  spec of how `mlx.QuantizedLinear` forward is replaced with an STE-aware
  variant. Verify the gradient flow matches the cited paper's Eq.
- **Compute budget**: enforce smoke→full split (smoke = 1k tokens, full
  ≥ 50k tokens) with `is_smoke` flag, single iteration ≤ 30 min.

When all four are done, re-file at P=2 with the paired-KC structure.

## §7 Disposition

Routing to reviewer hat for:
- `experiment complete exp_g4_quantization_aware_adapter_training --status killed --dir micro/models/exp_g4_quantization_aware_adapter_training/`
- Evidence: "F#666-pure-standalone + F#502/F#646 + predicate-not-met (triple-fire); F#769 ledger-explosion: no new finding"
- KC results: `--k 1920:inconclusive --k 1921:inconclusive`
- **finding-add SKIPPED** per F#769 closing-note

## §8 Drain accounting

- **Drain criterion 1** (P≤2 open queue): unchanged. This is P=4 (lowered
  by prior researcher 2026-04-25 from P=2). Outside drain scope.
- **Drain criterion 2** (active queue empty): currently 1 (this experiment).
  Will be 0 after reviewer's `experiment update --status killed`.
- **Net effect**: −1 entry from open queue.
- **Strategic note**: the original P=2 release was a deferral, not a drain.
  This iteration converts the deferral into a definitive KILL, removing the
  experiment from open queue entirely. Future re-attempt would require a
  new experiment ID with the §6 fixes pre-applied.
