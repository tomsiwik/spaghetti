# LEARNINGS — exp_p8_vproj_domain_behavioral

## Audit Re-classification (2026-04-18)
**Status: KILLED** (supersedes original SUPPORTED claim.)

Tagged `audit-2026-04-17-rerun` + `tautological-routing`. K1315 pre-registers
"5-adapter Grassmannian composition retains ≥80% of solo quality" but
`run_experiment.py::phase_composition_test` measures sequential hot-swap
serving (each adapter loaded independently at T=0.0). `comp_rate == solo_rate`
by construction, retention=1.00 is a mechanical artifact — antipattern #6
(KC measures wrong object). Re-classified KC tally: K1312 FAIL (math 0.55 < 0.60),
K1313 FAIL (code 0.50 < 0.60), K1314 PASS (medical 0.70), K1315 FAIL on
pre-reg KC → 1/4 pass → KILLED. MATH.md git-clean since pre-reg commit
78538d2 — KC unchanged, measurement is what diverges. results.json was
reconstructed from PAPER.md tables without re-running code because the
antipattern is structural, not transient.

## Core (Behavioral) Finding — preserved, not a credited KC pass
v_proj+o_proj adapters strictly dominate q_proj adapters on behavioral
vocabulary improvement across all 5 domains (math 30→55, code 20→50,
medical 60→70, legal 20→35; finance 50% no q_proj baseline). The
directional claim "output-path targets behavior, query-path targets
attention" is supported by the data. It is *not* credited as K1312/K1313
passing because the absolute 60% thresholds were pre-registered, and those
specific thresholds were not met.

## Why It Works
Output-path adapters (v_proj+o_proj) directly modify the token vocabulary distribution
at generation time. Query-path adapters (q_proj) only change attention patterns — what
the model looks at — which is insufficient for generation tasks. This is mechanistically
confirmed by vocabulary shift: mean +21% to +59% across all domains.

## Measured vs Predicted
Predictions (70-80% math, 65-75% code) overestimated measured values (55%, 50%).
Post-hoc explanation: ceiling effect from base model (Gemma 4 E4B-IT) already being
strong at math/code, plus only 80 training examples (8-10 unique, cycled).
Medical passes (70%) where base model is weaker — consistent with the ceiling story
but not prospectively predicted, so treat as hypothesis not confirmed mechanism.

## Caveats
1. K1315 composition trivially satisfied — sequential serving guarantees 100% retention
   by construction; actual adapter weight merging was not tested.
2. Legal at 35% is the weakest domain — improved vs q_proj (20→35pp) but notably
   below finance (50%) and code (50%) despite rich legal vocabulary.
3. Ceiling effect explanation is post-hoc and unfalsifiable from this experiment alone.

## Implications for Next Experiment
To push math/code past 60% threshold, need: (a) more diverse training data (>100
unique examples), OR (b) longer training (500+ iters), OR (c) rank-32 adapters.
A stronger design would pre-measure base model competence per domain and use it to
predict which domains will hit ceiling effects — turning the post-hoc explanation into
a prospective prediction.

## V2 Requirements (exp_p8_vproj_vs_qproj_v2)
- Drop or reformulate K1315 to an **actual** parameter-merge composition test:
  build ΔW = Σ B_i A_i^T, merge into weights, single forward per query.
  Grassmannian-orthogonal A via QR on random Gaussian seeds per domain.
  KC on per-layer activation cross-talk max |cos(A_vi·x, A_vj·x)| ≤ 0.30.
  Do NOT route-then-serve (that's sequential, not composition).
- Pre-measure base vocabulary baseline per domain (N=20 same prompts used in
  eval) and pre-register per-domain thresholds as `base + Δ` rather than
  flat 60%. Avoids the post-hoc ceiling-effect excuse.
- Train ≥100 unique examples per domain (not 8–10 cycled × 200 iters).
- Keep v_proj+o_proj as projection target — it's the validated direction.

## Cross-referenced audit pattern
Same class of bug as `exp_p6_lingering_adapter_online` (MMLU KC text, trivia
measurement) and `exp_p7_null_space_adapter_quality` (MMLU KC text, 5-prose
PPL measurement). All three: pre-registered KC names a concept (MMLU,
Grassmannian composition) that the code does not measure. Mitigation is
infrastructure-level: reviewer must read both KC text AND measurement code
before accepting a pass.
