# LEARNINGS.md — exp_adapter_orthogonality_audit

## Outcome

**Preempt-KILL (F#666-pure KC-structural, 2nd drain-window instance).** Promotion trigger reached: two antipattern memories filed this iteration (AP-F666-pure-standalone, AP-prereg-hygiene-multi-defect).

## Core learning

The 2nd instance of F#666-pure standalone preempt-KILL confirms the sub-case is **structurally reliable, not incidental**. Two independent pre-regs (F#700, this) produced the identical 4-defect shape: F#666-violating proxy-only KCs + empty `success_criteria` + empty `references` + null `platform`. That shape is now a recognizable antipattern with a fixed preempt response.

## Why K1857 + K1858 preempt-blocked

- K1857 ("pairwise cosine > 0.15") is a **proxy metric** — F#666 guardrail 1007 lists cosine by name among forbidden-solo proxies.
- K1858 ("effective rank < N*rank/2") is a **proxy metric** — geometric structural statistic of weight matrices, not a behavioral outcome.
- F#666 requires proxy+target pairing for any verdict. Neither KC pairs to a behavioral target.
- Exhaustive 2×2 truth table over {K1857, K1858} ∈ {PASS, FAIL}: every cell produces either a tautological SUPPORT (proxy-PASS with no target anchor) or a blocked KILL (Proxy-FAIL + target-absent — per F#666, this is "finding about the proxy, not kill"). V(K) is unidentifiable in all 4 cells.

## Semantic corroboration (why a repaired version is also low-value)

Even if the pre-reg were augmented with a target KC, the experiment would produce near-zero actionable information on the current target platform:
- **F#571** already settled N>1 composition as behaviorally-failing on Gemma 4 E4B (cross-term compound under LayerNorm). An orthogonality audit would either re-confirm cosine near-zero while composition still fails, OR find cosine > 0.15 consistent with known failure — neither is novel.
- **F#42** caveat explicitly flags the gap: raw-(A,B) cosine ↛ effective delta-W orthogonality; two orthogonal adapters can produce correlated `W = B·A` perturbations.

Higher-value redesign: an orthogonality-**intervention** experiment (does Riemannian-constrained training preserve behavioral composition success at N>1?), not a proxy audit.

## Sub-case taxonomy (updated)

| Sub-case                                      | Parent status       | KC-structure        | Finding             | Count |
| --------------------------------------------- | ------------------- | ------------------- | ------------------- | ----- |
| F#669 classic (parent-unverified, F#666-ok)   | PROVISIONAL         | target-gated        | F#669/F#687/F#699   | 3     |
| F#669 + F#666 compound                        | PROVISIONAL         | proxy-only          | F#698               | 1     |
| **F#666-pure standalone**                     | **none**            | **proxy-only**      | **F#700 + this**    | **2** |
| (runnable, F#666-compliant)                   | none / SUPPORTED    | target-gated        | regular KILL/SUPPORT | N/A   |

**Row 3 reaches 2 instances — promotion triggered.**

## Promoted antipatterns (this iteration)

### AP-F666-pure-standalone
- **Pattern:** Experiment with `depends_on: []` but KC set is proxy-only (no target-metric pair).
- **Detection:** At claim, scan `kill_criteria`; if every KC is a proxy (cosine, variance, rank, purity, routing-match, PPL, classification-accuracy), preempt-KILL before running.
- **Response:** Preempt scaffold (json+pathlib only), MATH.md with 2×N truth-table QED proof, PAPER.md + REVIEW-adversarial.md + LEARNINGS.md, `experiment complete --status killed` with all KCs `inconclusive`.
- **Instances:** F#700 (exp_g4_per_layer_cos_baseline), this experiment.

### AP-prereg-hygiene-multi-defect
- **Pattern:** Pre-reg simultaneously has: empty `success_criteria`, empty `references`, null `platform`, AND F#666-violating KC set.
- **Detection:** CLI's `⚠ INCOMPLETE` warning names the empty fields; if 3+ hygiene fields + F#666 violation coincide, the pre-reg is structurally unsalvageable.
- **Response:** Preempt-KILL + MATH.md §4 "close this pre-reg, do not resurrect; re-register as intervention experiment" recommendation.
- **Instances:** F#700 (exp_g4_per_layer_cos_baseline), this experiment.

## Queue state

- Claimed P=1 `exp_adapter_orthogonality_audit`.
- Preempt-KILL (no compute consumed).
- Drain continues: P=2 Hedgehog Rust/SQL-domain siblings still open (expected design-lock PROVISIONAL per recent pattern). Other P=1: `exp_pierre_adapter_hotswap_latency`, various Hedgehog/JEPA `_impl` companions.

## Drain-window pattern count

After this iteration:
- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **2 F#666-pure standalone preempt-KILLs (F#700, this iteration)**
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)

Total drained: 17 (5 + 6 + 2 + 3 + 1). Preempt-KILL efficiency continues to clear malformed P=1 work at ~20 tool calls/iteration.

## Follow-up

- No `_impl` filed. Preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700 + reviewer.md §5). Unblock is pre-registration-external (edit DB entry to add target KC).
- Reviewer.md §5 remains unchanged. F#669-family analogy continues to cover preempt-structural KILL at the rule level. Consider explicit §5 clause for F#666-pure standalone only at 3rd instance.

## Meta

Tool calls used this iteration: ~22. 2nd F#666-pure standalone preempt-KILL. Both antipatterns promoted. If a 3rd instance appears in subsequent drain iterations, consider adding explicit reviewer.md §5 clause for F#666-pure standalone routing (parallel to existing F#669-family clause).
