# PAPER.md — exp_adapter_orthogonality_audit

## Verdict: KILLED (preempt, F#666-pure KC-structural — 2nd drain-window instance)

This experiment was preempt-killed before any MLX code was written. The kill is **structural**: the pre-registered kill-criterion set K = {K1857 pairwise cosine, K1858 effective rank} consists entirely of proxy geometric metrics with no paired target-metric KC. Under F#666 (guardrail 1007), proxy-alone verdicts are tautological — KILL requires proxy+target BOTH to fail, SUPPORTED requires BOTH to pass. No empirical outcome can produce a valid verdict.

**Distinguishing feature:** no parent dependency (`depends_on: []`). This is NOT an F#669-family preempt. It is an F#666-pure structural violation, **the 2nd instance** of this sub-case (after F#700 on `exp_g4_per_layer_cos_baseline`). The 2nd-instance threshold triggers antipattern memory promotion this iteration.

## Prediction vs measurement

| KC    | Prediction                                                                                       | Kind  | Measurement  | Verdict   |
| ----- | ------------------------------------------------------------------------------------------------ | ----- | ------------ | --------- |
| K1857 | Pairwise cosine between any two adapter weight matrices > 0.15 (not orthogonal)                 | proxy | not measured | untested  |
| K1858 | Effective rank of N-adapter stack < N * rank/2 (subspace overlap > 50%)                         | proxy | not measured | untested  |

Rows are "not measured" because measurement would yield a proxy-only verdict, which F#666 explicitly forbids. Running the measurement and marking K1857/K1858 PASS/FAIL in isolation would constitute antipattern-t (producing a structurally invalid verdict). Inventing a target-metric KC after the run would be antipattern-u (post-claim KC mutation).

## Secondary structural defects

Per `experiment get exp_adapter_orthogonality_audit`:

1. `success_criteria: []` — no SUPPORTED-condition declared; independent defect that also blocks SUPPORTED verdict.
2. `references: []` — violates guardrail 1002 (cite arxiv or prior finding). Notes reference F#562 informally; F#42, F#571 also relevant but not registered.
3. `platform: null` — unset; MATH.md §0 discipline violated.

**Exact same 4-defect structural shape as F#700** (F#666-violating KC + empty success_criteria + empty references + null platform). Two instances of identical structure ⇒ antipattern promotion triggered.

## Semantic corroboration (why this audit cannot yield actionable information)

Even if the pre-reg were repaired to add a target-metric KC, the experiment's value is near-zero on the current target platform:

- **F#571 already settled N>1 behavioral composition** as failing on Gemma 4 E4B (cross-term compound under LayerNorm nonlinearity). An orthogonality audit would either re-confirm cosine near-zero (F#42/F#562 pattern) while composition still fails behaviorally, OR find cosine > 0.15 and be consistent with the known behavioral failure — neither is novel.
- **F#42 caveat** explicitly flags the gap: raw-(A,B)-concatenation cosine does not imply effective-delta-W orthogonality — two adapters with orthogonal (A,B) can produce correlated `W = B·A` perturbations.

Higher-value redesign: an orthogonality-**intervention** experiment (does Riemannian-constrained training preserve behavioral composition success at N>1?), not a proxy audit.

## Assumptions

- No attempt was made to substitute a runnable proxy-only verdict. Measuring cosine and effective-rank and marking K1857/K1858 pass/fail in isolation would produce a tautological outcome per F#666.
- No attempt was made to invent a target-metric KC post-claim (antipattern-u).
- Adapters present on disk (`exp_p1_t2_single_domain_training/adapters/{code,math,medical}/adapters.safetensors` and ~10 others) were **not loaded**.
- The 2nd-instance threshold for antipattern promotion was validated structurally (identical 4-defect shape to F#700), not merely by count.

## Pattern taxonomy (drain-window context)

| Sub-case                                      | Parent status       | KC-structure        | Finding             | Count |
| --------------------------------------------- | ------------------- | ------------------- | ------------------- | ----- |
| F#669 classic (parent-unverified, F#666-ok)   | PROVISIONAL         | target-gated        | F#669/F#687/F#699   | 3     |
| F#669 + F#666 compound                        | PROVISIONAL         | proxy-only          | F#698               | 1     |
| **F#666-pure standalone**                     | **none**            | **proxy-only**      | **F#700 + this**    | **2** |
| (runnable, F#666-compliant)                   | none / SUPPORTED    | target-gated        | regular KILL/SUPPORT | N/A   |

Row 3 reaches 2 instances — promotion trigger.

Drain-wide pattern count after this iteration:
- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **2 F#666-pure standalone preempt-KILLs (F#700, this iteration)**
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)

## Promoted antipatterns (this iteration)

### AP-F666-pure-standalone
**Pattern:** Experiment with `depends_on: []` but KC set is proxy-only (no target-metric pair).
**Detection:** At claim, scan `kill_criteria`; if every KC is a proxy (cosine, variance, rank, purity, routing-match, PPL, classification-accuracy), preempt-KILL before running.
**Reason:** F#666 (guardrail 1007) makes proxy-only verdicts structurally invalid.
**Confirmed instances:** F#700, this experiment.

### AP-prereg-hygiene-multi-defect
**Pattern:** Pre-reg simultaneously has: empty `success_criteria`, empty `references`, null `platform`, AND F#666-violating KC set.
**Detection:** CLI's own `⚠ INCOMPLETE` warning names the empty fields; if 3+ hygiene fields + F#666 violation coincide, the pre-reg is structurally unsalvageable.
**Reason:** Single defects are patchable; 4-defect pre-regs indicate a design process that hasn't engaged with F#666 / F#42 / F#571 prior art. Clean re-register is strictly cheaper than patch.
**Confirmed instances:** F#700, this experiment.

## Unblock path

Re-claim requires **KC-augmentation** (pre-registration modification before re-claim):

1. Add a target-metric KC pairing K1857/K1858 to a behavioral outcome. Candidate: "N=3 composition (code+math+medical adapters from `exp_p1_t2_single_domain_training`) via mean-summation retains ≥90% single-adapter task accuracy on each domain's held-out eval set."
2. Add references: F#42, F#562, F#571; optionally arxiv 2504.10957 (Zhong et al. task-arithmetic conditions).
3. Set `platform=mlx`.
4. Populate `success_criteria` (mirror of KC pass condition).

Alternative (recommended): **close this pre-reg as structurally-malformed**, do not resurrect. Re-register as an orthogonality-intervention experiment (Riemannian-constrained training for composition preservation) rather than a proxy audit.

## Related

- **F#666** (guardrail 1007) — target-gated KILL discipline; the rule this experiment's KC set violates.
- **F#700** — 1st F#666-pure standalone instance. This is 2nd.
- **F#698** — F#666 compound (parent-unverified + proxy-only). Orthogonal to this case.
- **F#669 / F#687 / F#699** — preempt-KILL family on parent-unverified. Not applicable (no parent).
- **F#42** (2026-03-21, conclusive) — Cosine orthogonality plateaus at convergence (BitNet). Caveat flags raw-(A,B) cosine ↛ effective delta-W orthogonality.
- **F#562** (2026-04-18, supported) — Partition-QR structural orthogonality at Gemma 4 native dims (pre-training construction only).
- **F#571** (2026-04-18, killed-macro) — N>1 composition behaviorally fails on Gemma 4 E4B.
- **Guardrail 1002** — cite paper/finding; empty `references` violates this.

## Follow-up filed

None — preempt-structural KILL does not spawn an `_impl` companion (per F#687/F#698/F#699/F#700 + reviewer.md §5). Unblock is pre-registration-external (edit DB entry to add target KC), not implementation-external.
