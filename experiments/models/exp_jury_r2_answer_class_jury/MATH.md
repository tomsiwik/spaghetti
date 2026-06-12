# exp_jury_r2_answer_class_jury — jury-weighted SC in answer-class space

## Disease, not symptom
R1 (exp_bet_jury_r1_verifier_gain, F#877/K2316) killed adapter-verifier **selection**: AUC 0.821 but
BoN(8) = 0.785 < SC(8) = 0.820. The disease was not weak signal — it was the *argmax over chains*: one
miscalibrated high score on a wrong chain discards 7 votes. Selector-class R1s are closed (F#879/K2332).

## Theorem (mechanism)
Let question q have chains c_1..c_8 with parseable answers, partitioned into answer-equivalence classes
A (equal final answer). Juror j ∈ {math, python, medical} (frozen LoRA r=6 q_proj, scale 6, on frozen
gemma-4-e4b-it-4bit) assigns each chain a judge score s_j(c) = logP(Yes) − logP(No). Per question, scores
are standardized (zero mean, unit std over the 8 chains — no free temperature parameter) and softmaxed:
p_j(c). Class mass m_j(A) = Σ_{c∈A} p_j(c) > 0. Jury vote: argmax_A Σ_j log m_j(A).

Claim: because no juror ever *selects* a chain — each only reshapes the class prior multiplicatively over
SC's vote structure — a single miscalibrated score moves at most one softmax weight, not the whole
decision. If juror errors are decorrelated (different training domains on the shared base), log-linear
pooling suppresses idiosyncratic misrankings, so the calibration failure that killed BoN is structurally
unreachable.

## Predictions (pre-registered, before run)
- SC(8) on the regenerated chains ≈ 0.820 (identical seed schedule, same mlx_lm 0.31.2 → expect exact
  reproduction; validated against cached R1 preds).
- **Jury accuracy: predicted 0.855** (between SC 0.820 and pass@8 ceiling 0.935; AUC 0.821 converted to
  vote margins is worth ~half the BoN headroom that miscalibration destroyed).
- Mandatory control: math-only weighted SC predicted ≈ 0.825–0.835 (signal exists but single-juror
  miscalibration leaks through) — must NOT clear the +2pp gate, isolating decorrelation as the cause.
- Juror error decorrelation: mean pairwise Cohen kappa on misranking indicators < math-juror split-half
  self-kappa.

## Refutation thresholds (numeric, pre-registered — K2333)
Let acc_sc = SC(8) accuracy recomputed on the regenerated chains (the shared-budget baseline).
- **KILLED** if jury_acc < acc_sc + 0.02 (i.e. < 0.84 at acc_sc = 0.82), OR
- **KILLED** if mean pairwise juror kappa ≥ math-juror bootstrap self-kappa (shared frozen base
  correlates failures → jury is one verifier in a trenchcoat; R2-as-R1 dead).
- **SUPPORTED** only if jury_acc ≥ max(acc_sc, best single-juror weighted SC) + 0.02 AND the overlap
  condition passes AND the math-only control does NOT itself clear acc_sc + 0.02 (if the control also
  clears it, verdict is PROVISIONAL: weighting, not decorrelation, would be the cause).
- Otherwise PROVISIONAL.

### Overlap metric (fixed before run)
Misranking indicator per (juror, question, chain-half): split the 8 chains into halves H_A = {0..3},
H_B = {4..7}; e_j^H(q) = 1 iff juror j's top-scored chain within half H has a wrong answer. Restrict to
questions where both halves contain ≥1 correct and ≥1 wrong parseable chain (mixed questions).
Pairwise overlap(j,k) = mean of kappa(e_j^A, e_k^B) and kappa(e_j^B, e_k^A) over questions.
Self-overlap = same construction with j = k = math (its split-half consistency), bootstrap mean over
1000 question resamples. Identical construction for pair and self → directly comparable.

## Why this is not goalpost-moving on R1
R1's kill was about chain *selection* at equal budget. Here generation budget is again identical to SC
(the same 8 chains, regenerated bit-exact); jurors add prefill-only cost (3 × 1600 ≈ 4800 judge passes,
reported separately). The vote structure of SC is retained; only the per-class mass is modulated.

## Grounding
- BoN-MAV (ref #728): multi-aspect verifiers help BoN — but still select; we remove selection entirely.
- R1 measured artifacts reused: greedy 0.705, SC(8) 0.820, BoN(8) 0.785, verifier AUC 0.8214,
  pass@8 0.935 (experiments/models/exp_bet_jury_r1_verifier_gain/results.json).
