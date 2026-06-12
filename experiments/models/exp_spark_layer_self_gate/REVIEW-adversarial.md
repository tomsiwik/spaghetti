# REVIEW (adversarial) — exp_spark_layer_self_gate

**Verdict: PROVISIONAL** (downgrade from preliminary SUPPORTED). Not a mock; not a kill; the
mechanism claim is UNPROVEN due to a missing control. verify-experiment.sh exits 0.

## No-mock / integrity (all PASS)
- is_smoke=false; results.json present; verify-experiment.sh exit 0.
- Both adapters real & loaded from exp_p1_t2_single_domain_training/adapters; math & code lora_a (2560,6) lora_b (6,2048).
- pass@1 from REAL HumanEval unit-test subprocess execution (run_humaneval_test).
- gamma computed from a genuine per-prompt free forward (compute_gamma probe pass fills ctrl.gamma per layer);
  top-k from those floats — NO hardcoded layer indices.
- Mask applied via setattr(layer.self_attn,"q_proj",wrapper) — real submodule replacement, not __call__-on-instance.
- results.json verdict/all_pass and PAPER verdict line agree; kill_criteria.id=2296, anchored to in-run C (0.02), not F#827.
- MATH.md untracked (new exp) ⇒ no post-run threshold edit possible; thresholds in MATH match results.json.
- All headline numbers reproduce exactly from details (k6 17/50=0.34 ... k36 6/50=0.12; A.44 B.34 C.02).

## DIMENSION CHECK (PASS)
base q_proj 2560→2048; adapter delta (x@a)@b →2048. q_codebase=y+zc and out=q_codebase+delta_m both 2048-space;
gamma=cos(delta_m,q_codebase) both 2048. No 2048-vs-4096 mismatch, no degenerate space.

## CONFOUND #1 — degraded ceiling B<A (REAL, must be recorded)
B(code-solo)=0.34 < A(base)=0.44. The code adapter HURTS its own task by 10pp. best_D "recovers to" 0.34 =
a SUB-BASE ceiling. "Full ceiling recovery" is true only against B; against the honest baseline A it is a net
-10pp. GSM8K shows the same: A=0.78 > B=0.52. The scale-6 adapters net-degrade both tasks vs base.
PAPER.md does disclose A>B (lines 31, 81) — credit for honesty — but the verdict line frames "recovers ... to the
code-solo ceiling" without flagging that the ceiling is itself below base. Honest framing: recovery is to a
degraded ceiling, not to base ability.

## CONFOUND #2 — SELECTION vs AMOUNT (THE KILLER; mechanism UNPROVEN)
Recovery is MONOTONE in fewer math layers (k6 .34 > k12 .24 > k18 .18 ~ k24 .22 > k30 .12 ~ k36 .12) and
best_D=0.34 EQUALS B (code-solo, ZERO math) EXACTLY. This is fully consistent with the null "the gamma SELECTION
is irrelevant; what helps is simply using almost no math adapter." The frame-break REQUIRES top-k-by-gamma to
beat a RANDOM choice of k=6 layers (and ideally bottom-k / anti-gamma). 
**No random-layer, anti-gamma, or bottom-k control was run** (confirmed: zero such code/branch in run_experiment.py,
MATH.md, PAPER.md). Without it the localization/self-reporting mechanism is indistinguishable from "less math = better."
This is the F#858 decohere-vs-destroy confound in new clothes. The PAPER even concedes "the gamma sign alone is
not the clean oracle" — leaving gamma RANKING as the only claimed signal, which is exactly what is uncontrolled.

## CONFOUND #3 — gamma-rank signal beyond "fewer is better" (NO EVIDENCE)
~25/42 layers positive-gamma every prompt (min23/mean25.2/max27), but k=6 wins and k>6 monotonically degrades.
There is NO evidence the gamma ordering past the top few carries signal: the monotone-in-k curve is equally
explained by amount. mean_gamma ~ +0.011 (essentially zero) across all k — the cosine is tiny; the ranking past
the leaders is plausibly noise.

## Required controls before SUPPORTED (hand back, ≤3)
1. **Random-k=6 control:** math in 6 RANDOM q_proj layers (seed-fixed, several seeds), per-prompt or fixed.
   SUPPORTED needs top-k-by-gamma > random-k by a meaningful margin at n=50.
2. **Anti-gamma / bottom-k=6 control:** math in the 6 MOST-destructive (lowest gamma) layers. If bottom-6 ~ top-6,
   gamma sign/rank is dead.
3. Report recovery against BASE (A), not only B; relabel "ceiling" as the degraded code-solo ceiling.

## Caveats the analyst MUST record
(a) NO random/anti-gamma layer control exists — SELECTION is NOT separated from AMOUNT; the localization mechanism
    is UNPROVEN. best_D=B exactly is the smoking gun for the "amount" null.
(b) B(0.34) < A(0.44): degraded ceiling; the adapter hurts its own task; "full recovery" is to a sub-base ceiling.
(c) gamma is near-zero (mean ~+0.011) and ~25/42 layers positive while k=6 wins ⇒ no shown rank signal past the top few.

Kill-id 2296: did NOT fire on its literal terms (recovery +32.0pp ≥ +8; floor best_D=0.34 ≥ B−6pp=0.28). But the
criterion is insufficient — it never required beating a random/anti-gamma control, so passing it does not establish
layer-localization. PROVISIONAL pending controls 1–2.
