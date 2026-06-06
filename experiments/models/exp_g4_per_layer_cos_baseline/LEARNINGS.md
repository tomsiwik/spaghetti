# LEARNINGS.md — exp_g4_per_layer_cos_baseline

## Outcome

**Preempt-KILL (F#666-pure KC-structural).** New sub-case in drain-window taxonomy, orthogonal to F#669 family.

## Core learning

A single-proxy KC with no target pairing is a **structural** defect, not an empirical one. Under F#666 (guardrail 1007), neither KILL nor SUPPORTED can be derived from a proxy-only KC set regardless of measurement outcome. Therefore running the measurement would be wasted compute and would risk producing a tautological verdict (antipattern-t).

This matters because: a naive researcher would look at K1856 ("cos-sim variance < 0.02"), run the measurement, and report SUPPORTED or KILLED. Either outcome would be F#666-tautological. Preempt-KILL is the correct action.

## Why K1856 preempt-blocked

- K1856 is a proxy metric (geometric similarity of embeddings across prompts), not a behavioral/target outcome.
- F#666 requires proxy+target pairing for any verdict. Proxy-alone is tautological:
  - Proxy-PASS + no-target = tautological SUPPORT.
  - Proxy-FAIL + no-target = cannot KILL per F#666 ("Proxy-FAIL + target-PASS = finding about the proxy, not a kill" — analogously, Proxy-FAIL + target-absent has the same structural issue).
- Therefore V(K) is unidentifiable; KC set is malformed.

## Sub-case taxonomy (updated)

| Sub-case                                      | Parent status       | KC-structure        | Finding     |
| --------------------------------------------- | ------------------- | ------------------- | ----------- |
| F#669 classic (parent-unverified, F#666-ok)   | PROVISIONAL         | target-gated        | F#669 / F#687 / F#699 |
| F#669 + F#666 compound                        | PROVISIONAL         | proxy-only          | F#698       |
| **F#666-pure standalone (this)**              | **none / ok**       | **proxy-only**      | **this iteration** |
| (runnable, F#666-compliant)                   | none / SUPPORTED    | target-gated        | regular KILL/SUPPORT |

1st instance of row 3. Promotion threshold: 2nd instance → standalone antipattern memory. For now: document, watch, don't promote.

## Secondary defects observed

Pre-reg also had:
1. `success_criteria: []` — empty; no SUPPORTED-condition declared.
2. `references: []` — guardrail 1002 violation (cite arxiv or prior finding).
3. `platform: null` — unset; MATH.md §0 discipline hole.

Combined with the F#666 violation, this pre-reg is broadly malformed. Recommendation: re-scope as a Hedgehog-family sibling rather than resurrect the pre-reg.

## Queue state

- Claimed P=1 `exp_g4_per_layer_cos_baseline`.
- Preempt-KILL (no compute consumed).
- Drain continues: P=2 Hedgehog Rust/SQL-domain siblings still open (expected design-lock PROVISIONAL per recent pattern).

## Drain-window pattern count

After this iteration:
- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **1 F#666-pure standalone preempt-KILL (this)**
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)

## Follow-up

No `_impl` filed. Preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699 + reviewer.md §5). Unblock is pre-registration-external (edit DB entry to add target KC).

## Meta

Tool calls used this iteration: ~20. Preempt-drain pattern (no compute, structural routing) continues to efficiently clear malformed P=1 work. If the F#666-pure standalone sub-case recurs, promote to antipattern memory. If secondary defects (empty success_criteria, empty references, null platform) recur across multiple experiments, that is a **pre-reg hygiene** antipattern worth promoting separately.
