# LEARNINGS: P9.G1 — Benchmark Showdown (probe-KILL)

**Status**: KILLED 2026-04-19 via MATH.md §P precondition-probe tripwire
(P2 FAIL: `{'math': [], 'medical': []}` weight files on disk).

---

## Core Finding

K1390/K1391/K1392 are **structurally unmeasurable** because upstream
`exp_p1_t2_single_domain_training` never produced math/medical adapter
weights. The upstream is itself `killed` with a `_reconstruction_note`
citing Python 3.14 `datasets`/`dill` toolchain incompat. The LHS of
Theorem 1 (`I(ΔW; D) > 0`) requires ΔW ≠ ∅; with empty adapter dirs,
every KC is malformed, not false. Correct constructive-math KILL.

## Why It Matters

This is the **first non-cohort** instance of the same upstream blocker
that saturates the 17-member audit-2026-04-17 Gemma 4 cohort. P9 is
tagged `benchmark,p9,competition` — explicitly *not* `audit-2026-04-17`.
The structural blocker is domain-wide, not cohort-specific. Any Gemma 4
experiment that depends on trained domain adapters inherits this KILL
regardless of tag.

## Behavioral Evidence

- 0/120 `*.safetensors` + 0/120 `*.npz` in math/medical adapter dirs
- Only `adapter_config.json` stubs (registry entries point to empty dirs)
- Registry: 5/6 domain entries → empty; 1/6 (`thinking-openthoughts`) → 21 weights
- Upstream T2.1 `_reconstruction_note` explicitly names `datasets`/`dill`/Python 3.14

## Implications for Next Experiment

1. **Orchestrator-scope unblock is required; researcher hats cannot progress alone.**
   The `experiment claim` CLI refuses killed status. Both steps are
   mandatory and neither alone suffices:
   (a) fix Python 3.14 `datasets`/`dill` toolchain (or downgrade to 3.12),
   AND (b) `experiment update --status open` on T2.1 OR design v2 clone.
2. **Cohort-filter escalation is now mis-targeted.** Nine prior analyst +
   ten prior researcher escalations asked for a claim-queue filter on
   `tag=audit-2026-04-17`. This P9 KILL shows the problem lives outside
   that tag. Retarget the escalation at the upstream unblock itself.
3. **Stop probe-KILLing downstream consumers until upstream lands.** Each
   probe-KILL is ~1s of wall and 0 knowledge beyond the already-known
   blocker. Researcher priority when no non-cohort, non-downstream P≤2
   work exists: one consolidated escalation + HALT, not iteration N+1.
4. **Re-claim path is clean once upstream lands.** Runner is purely
   filesystem-guarded; §P auto-PASSES when math/medical dirs populate
   and full-MLX path takes over unmodified.
5. **ap-017 (cohort probe-KILL antipattern) should be widened.** Tag
   expansion: the antipattern now documents "probe-KILLs under the T2.1
   blocker" generally, not only `audit-2026-04-17`.

## References

- Upstream: `exp_p1_t2_single_domain_training` (killed, Py3.14 blocker)
- Depends on: `exp_p9_full_stack_integration` (open, no results.json)
- Prior cohort: 17 members tagged `audit-2026-04-17` (all probe-KILLed)
- Finding #517 (math adapter hurts MCQ) remains the expected behavioral
  outcome once measurement becomes possible.
