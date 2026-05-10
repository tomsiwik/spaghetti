# REVIEW-adversarial — exp_pierre_lora_hub_learned_scalars

## Verdict: PROCEED

## Adversarial Checklist

| Check | Result | Notes |
|-------|--------|-------|
| (a) verdict consistency | OK | results.json="SUPPORTED", DB="supported", PAPER="SUPPORTED" |
| (b) all_pass matches | OK | K1∧K2→SUPPORTED per verdict logic; K3 FAIL is non-blocking |
| (c) PAPER verdict | OK | Matches DB |
| (d) smoke guard | N/A | N=50/bench, not smoke |
| (e) KC post-hoc edit | OK | KCs in MATH.md match results.json field names and thresholds |
| (f) tautological KC | OK | All KCs compare distinct methods or measure wall-clock/interpretability |
| (g) code↔math alignment | OK | `compose_weighted_mean_b` implements Σ w_i B_i as described |
| (h) Σ A_i / Σ B_i bug | N/A | Shared-A, only B summed — by design |
| (i) LORA_SCALE | OK | scale=6.0 |
| (j) single-sample routing | N/A | Global merge, not routing |
| (k) shutil.copy fake | OK | No shutil |
| (l) hardcoded pass | OK | KCs computed from measured values |
| (m) model mismatch | OK | gemma-4-e4b-it-4bit in MATH.md and code |
| (n) base=0% | OK | single_best=62%, Fisher-Rao=64.7% |
| (o) n<15 | OK | N=50/bench |
| (p) target-metric KC | OK | K1/K2 measure task accuracy averages |

## Non-blocking Observations

1. **576 vs 40 evals**: scipy DE with `popsize=5` on K=7 creates population of 35, not 5. 8 generations × ~72 evals/gen (with sobol init overhead) ≈ 576. MATH.md's "40 evals" estimate was wrong, but K3 FAIL is correctly reported and the overshoot is acknowledged in PAPER.md caveats.

2. **MedQA regression**: Learned scalars (52%) < Fisher-Rao (58%). The optimizer sacrificed MedQA to boost HumanEval (82% vs 68%). Acknowledged implicitly by per-benchmark table but worth explicit mention in LEARNINGS.

3. **Validation panel overfitting risk**: 9 prompts is very small. The 88.9% validation score vs 67.3% final eval suggests the optimizer memorized the panel to some degree. Not blocking since final eval uses separate data.

## Conclusion

Clean experiment. Well-framed as architectural ceiling test. KCs are meaningful, code implements what MATH.md describes, measurements are internally consistent. The K3 budget failure is honestly reported and doesn't affect the scientific conclusion (scalar reweighting helps but doesn't close the full gap). PROCEED to LEARNINGS.
