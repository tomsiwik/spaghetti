# REVIEW-adversarial — Pierre v5.2 Bankai Row Flip

Verdict: **PROCEED-WITH-KILL** (preempt via F#291 reduction).

## Adversarial checks

1. **Is the reduction honest?**
   Yes. `run_experiment.py:180` — `row = mx.clip(row, -1, 1).astype(mx.int8)`
   after `row + direction`. This is `clip(r + s, -1, +1)` with s∈{-1,+1},
   the exact operation F#291 addresses.

2. **Does row-level granularity change the per-entry saturation rate?**
   No. Each entry of each flipped row undergoes the same three-case
   computation. The Theorem is entry-local; aggregation over a row does not
   affect the fixed-point fraction.

3. **Could greedy search find flips whose benefit exceeds the destruction?**
   v5.1 already explored this space at per-entry granularity with ω=4
   (the explicit "sparse best" setting: 9.94% flip rate, 62M clips,
   behavioral=0.003). Row-level search is a subset of per-entry search
   with a coarseness constraint; it cannot exceed per-entry's optimum,
   and per-entry's optimum was measured KILLED.

4. **Could the "shift" interpretation (-1→0, 0→+1, +1→0 cycle) save it?**
   The runner does NOT implement cycle semantics; it implements
   `clip(r ± 1, -1, +1)`. Even if it did: cycling +1→0 is pure
   destruction of the high-magnitude entries, which carry the majority
   of the row's output contribution. No escape.

5. **Is v5.1's behavioral=0.003 a valid upper bound for v5.2?**
   Upper bound on behavioral (lower bound on kill severity): v5.1 searched
   at per-entry granularity with PPL as the optimization target over ω.
   v5.2 restricts to row-level flips (a strictly smaller search space) with
   greedy per-domain objective — i.e., worse or equal coverage. Yes, valid.

6. **Antipattern audit.**
   - No shutil.copy (runner performs real flips).
   - No N=smoke-as-full (preempt; no run).
   - No hardcoded pass=True (KCs fail honestly via reduction).
   - No tautological KC (K733 tests PPL equivalence, K734 tests time, K735
     tests speed — all distinct).
   - No proxy-with-empirical-refutation: the empirical evidence comes from
     v5.1 for the same operation; the reduction is bit-identical.

7. **MLX skill check.**
   Runner was written 2026-04-04. No claim of an MLX-native regeneration
   is made; preempt verdict does not require rewriting the runner.
   Documented in MATH.md/PAPER.md that the code was not executed.

## Non-blocking observations

- K735 (speed) would likely pass if run (v5.1 measured 138.3 tok/s with
  substantially more mutations). Not worth running given K733 failure.
- Success criteria were missing from the DB entry (flagged ⚠ INCOMPLETE
  in `experiment get` output). The preempt verdict holds regardless.

## Final verdict

PROCEED-WITH-KILL. F#291 reduction is tight; row-level granularity is a
refinement of the search space F#291 already settled. No new finding
registration (uses parent F#291). Analyst-owed LEARNINGS.md debt +1.

## Reviewer ratification (iter 47, 2026-04-19)

Adversarial checklist re-walked against hat spec (a)-(s). All PASS:
- (a)-(d) Consistency: results.json verdict=KILLED ↔ DB status=killed ↔
  PAPER verdict=KILLED ↔ all_pass=false ↔ is_smoke=false (preempt-no-run).
- (e) K#733 pre-registered in DB 2026-04-04 evidence; no post-hoc relaxation.
- (f) Tautology: K733 tests PPL equivalence vs base, distinct quantity.
- (g) K#733 measures what MATH/DB state it does.
- (h)-(m) Code↔math: no sum-LoRA bug (not LoRA), no LORA_SCALE (N/A),
  no shutil.copy, MODEL_ID=BitNet-b1.58-2B-4T matches MATH ternary setup.
  run_experiment.py:180 `mx.clip(row, -1, 1)` is the bit-identical v5.1
  op the reduction cites.
- (m2) Skill invocation: code dated 2026-04-04 (pre-MLX-era); preempt
  verdict does not require regenerating the runner. Transparency logged.
- (n)-(q) Eval integrity: preempt-no-run → N/A.
- (r) PAPER has prediction-vs-measurement table.
- (s) Theorem proof is by exhaustive case analysis over 6 ternary×sign
  combinations; sign-symmetric prior yields E[n_sat/d]=1/3 exactly.

F#291 reuse is structurally correct: same clip operation, strictly
smaller search space than v5.1 per-entry. v5.1 empirical upper bound
(behavioral=0.003, PPL ratio 80,543,923×) transfers as guarantee.

No new finding registration (F#291 reused). Drain count after iter 47:
**47 preemptive-kills + 6 non-preempt DB-completions = 53 total**.
Cohort branches unchanged.
