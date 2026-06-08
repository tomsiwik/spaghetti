# REVIEW-adversarial.md — exp_spark_entropy_gated_lora (id 2291)

**Reviewer verdict: KILLED (confirmed). No blocking issues.**

## No-mock / fabrication
PASS. Real MLX load of `gemma-4-e4b-it-4bit`, real 4.99MB math adapter, real
HumanEval+GSM8K via `datasets`, real subprocess test execution. 4084s wall time
for 3 phases x 80 evals. Per-problem preds, gates, ntok all vary. `is_smoke=false`.
The adapter measurably changes the forward pass: fixed (gate=1.0) diverges from
base on 4 HumanEval tasks and shifts GSM8K 0.725->0.650. Mechanism is real.

## Consistency / KC integrity
PASS. results.verdict=KILLED, all_pass=false, PAPER verdict=KILLED, is_smoke=false
(killed allowed). K1/K2 in results.json match MATH.md id 2291 verbatim and were
not reformulated. N=1 adapter -> B@A per token, no (SB)(SA) composition bug.
LORA_SCALE=6.0<=8. Gate is per-step from the frozen base softmax (not one-sample).
Model in MATH == model in code == model in results. >=1 target-metric KC (both are
behavioral: HumanEval pass@1, GSM8K exact-match) — Finding #666 satisfied.

## Adjudication: is "premise-not-reproduced" a legit KILL?
YES. Three independent reasons:
1. MATH.md §5 PRE-REGISTERED the degenerate guard: undefined interference_reduction
   when drop_fixed<=0 and undefined retention when lift_fixed<=0 both fail by design.
   The kill fires exactly as the proof said it would. This is not post-hoc.
2. Per GUIDE §3 kill discipline: K1 and K2 both FAIL on real data; status is `killed`,
   not "criterion reformulated." The researcher did not edit the KC to rescue it.
3. The hypothesis as stated is conditional on F#827's pattern. On the only in-repo
   adapter (q_proj r6/scale6) that pattern is absent (drop_fixed=-5pp, lift_fixed=-7.5pp),
   so the dissociation is untestable on this artifact. The correct epistemic move is
   to kill THIS experiment and, if desired, spin a v2 with an adapter that actually
   reproduces F#827 — not to mark this provisional and leave it active.

The caveat is honestly recorded: this kills the experiment-as-configured, NOT the
entropy-gating idea in general. That belongs in the finding caveat, which it is.

## Non-blocking flags
- n=40 per slice (>=15) — adequate, but small; CIs wide. Noted, not blocking.
- Base GSM8K 72.5% with thinking chars present — not a truncated-eval (flag o N/A).
- F#827's -12/+22pp did not reproduce here; future work needs the matching adapter recipe.

**Route: PROCEED-to-KILL. K1 FAIL, K2 FAIL.**
