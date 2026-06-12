# REVIEW-adversarial — exp_spark_intra_rank_phase_gate

## Verdict: KILLED (sealed)

Refutation is real. All adversarial checks pass.

### Mock / real
- is_smoke:false; model in MATH.md == loaded (gemma-4-e4b-it-4bit); adapter is a real 5MB
  safetensors, not a copied stand-in. verify-experiment.sh exits 0. Wall 7990s (real run).

### Consistency
- results.json verdict KILLED, all_pass False, clause_A_pass False, clause_B_pass False,
  PAPER.md verdict line all agree.

### Integrity
- Experiment dir is a single untracked add (MATH.md + results.json together); no post-run
  goalpost move. Code reads the 5pp/strict thresholds exactly as MATH.md pre-registers.

### Requested confirmations
(a) Magnitude-match REAL and NON-TRIVIAL: 12,294,828 checks, 0 violations, max_rel_err
    1.9e-6 << tol 1e-3, hard-asserted at line 374. Renorm equalizes L2 only, not direction:
    per-item preds still differ 21-39/80 across distinct-subspace arms -> arms NOT collapsed.
(b) Head/tail SVD split REAL: tail-only vs head-only differ 36/80; schedule vs swap differ
    28/80. NOTABLE: head-only-always vs swap = 0/80 (byte-identical) — the answer-emit phase
    is so short (boundary fires late) that swap collapses onto static head-only. This is not a
    bug; it directly substantiates clause B (timing irrelevant — a static rank-half == best
    timed arm) and shows the headline +5pp is pure rank-truncation by the swap CONTROL.
(c) Boundary detector fires on GENERATED tokens (string-match of cumulative decoded text,
    never gold): varied positions min 41 / max 971 / medians 118-442; 8 honest "none" items
    where model never emitted '#### '. Not oracle.
(d) Both kill clauses correctly evaluated vs pre-registered text:
    - Clause A: best_schedule(0.7625) - uniform(0.7125) = 4.999e-2 < 0.05 strict -> FAIL;
      best is the swap CONTROL, the hypothesis `schedule` is -2.5pp BELOW uniform.
    - Clause B: max(head 0.7625, tail 0.6000) = 0.7625 >= uniform 0.7125 -> FAIL (static match).
    Either clause kills; both fired.
- Underpower guard (F#866): uniform-math -10pp below base; best arm still -5pp below base.

Evidence is behavioral (GSM8K EM, n=80, real greedy decode), not a proxy. KILL is honest.
