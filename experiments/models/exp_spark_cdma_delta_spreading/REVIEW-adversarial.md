# REVIEW (adversarial) — exp_spark_cdma_delta_spreading

Reviewer: fresh-context Reviewer hat. Claim is POSITIVE → maximum scrutiny.
verify-experiment.sh: **EXIT 0** ("REAL result ok, is_smoke!=true, verdict present, model-backed").

## Verdict: PROVISIONAL (downgraded from preliminary SUPPORTED)

The result is REAL (not a mock) and the kill K2295 genuinely did not fire — but the headline
CDMA claim is UNPROVEN because the decisive confound was never measured.

## No-mock checklist — ALL PASS
- is_smoke=false. Real HumanEval unit-test execution: real task_ids (HumanEval/0..49), varied
  ntok (B 189–1024 / D 426–1024 / C all 1024 = real truncation under interference), real pass
  counts A=22 B=37 C=9 D=34. Not stubbed, not constant.
- Both adapters real & loaded (F#627 r=6 q_proj, 84 keys), 42 layers wrapped (assert count==42).
- P orthogonal: ‖PᵀP−I‖ ≈ 5e-15 (float64 QR), seed=1337, FIXED via RotBox (never learned),
  built at the TRUE delta widths p_dim_built=[2048,4096].
- Wrapper via subclass nn.Module + setattr on q_proj — NOT __call__-on-instance bypass (F#831).
- Kill-id 2295 text in results.json == MATH.md == run_experiment.py. Thresholds NOT moved after
  the run (git diff of MATH.md shows no change to the 8pp/6pp/2295 lines).

## THE KEY CONFOUND — NOT BROKEN, and that blocks SUPPORTED
The run evaluates ONLY HumanEval (code) pass@1 across all four conditions. GSM8K / MATH accuracy
under condition D was **never measured** (results.json contains no math/gsm8k field; conditions are
code-only). Therefore the experiment CANNOT distinguish:
  (i) "decoherence recovers code WHILE the math adapter stays functional" (the CDMA claim), vs
  (ii) "Pᵀ scatters the math delta into a pseudo-random subspace, NULLIFYING the math contribution,
       so D collapses to code-solo B."
D=0.68 vs B=0.74, Jaccard(B,D)=0.73 (D passes 4 that B fails, B passes 7 that D fails) is fully
consistent with (ii): P mostly deletes the interferer. The CDMA / multiple-access framing —
multiple adapters coexisting and BOTH remaining usable — has zero supporting evidence here.

## Fragility
Ceiling clause met EXACTLY at the boundary: D = B − 6pp = 0.68, margin 0.00. A single flipped
HumanEval problem (34→33 of 50 = 0.66) fires K2295. The positive verdict rests on one problem at n=50.

## Required caveat the analyst MUST record
Honest evidenced claim, scoped down: "A fixed orthogonal rotation of the off-domain (math)
delta-output removes its code-interference (C 0.18 → D 0.68, recovering 89% of the B−C gap),
**possibly by destroying the math adapter rather than decohering it**." The CDMA/decoherence
mechanism (both adapters survive) is NOT proven: math-under-D was not measured. Ceiling margin is
0.00 (one problem from killing). n=50, one adapter pair, one seed — untested beyond.

## Route
PROVISIONAL. To reach SUPPORTED, re-run adding GSM8K pass@1 for math-solo and for condition D
(rotated math delta on a MATH prompt). If math-under-D ≈ math-solo → CDMA confirmed → SUPPORTED.
If math-under-D ≈ base (rotation destroyed it) → claim is "interference deletion," not CDMA, and
the paper must be rewritten to drop the multiple-access framing.
