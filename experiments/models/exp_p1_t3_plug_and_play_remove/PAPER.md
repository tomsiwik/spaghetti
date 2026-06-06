# T3.7: Plug-and-Play Hot-Remove — Prediction vs Measurement

## Status: KILLED (V2 audit-rerun 2026-04-18)

V1 "supported" (2026-04-10) is retroactively invalid. V2 probe flips to KILLED
for three independent structural reasons plus a missing-artefact precondition.
No KC relaxation: V1 KC thresholds are preserved byte-for-byte in MATH.md.

## V2 Prediction vs Measurement Table

| Kill | Theorem | Prediction | V2 Measurement | V2 Result |
|------|---------|-----------|----------------|-----------|
| K1070: bit-exact remaining outputs | Theorem 1 | 0 token diffs across N×4 domains | Structurally unmeasurable (hardcoded `REAL_ADAPTER_PATHS[domain]` routing forces "no change" by dict semantics, not Theorem 1). 0/5 upstream `.safetensors` on disk. | **FAIL** (cannot-measure, not measured-and-fell-short) |
| K1071: freed slot reusable | Theorem 2 | new adapter > 4% on new domain | Adapter copy forgery: `geography` and `history` are both `shutil.copy(finance)`. V1's 100% measures finance-weights on MCQ letter format, not a new adapter. Finance `.safetensors` also absent. | **FAIL** (mem-antipattern-009) |
| K1072: p99 remove latency < 10ms | Theorem 3 | ~0.005ms | Measures `del d[k]` (O(1) dict deletion, hash-table semantics), not weight unload. V1 0.0009ms p99 could not fail under any implementation. | **FAIL** (mem-antipattern-011 specialisation) |

## V1 Numbers (reference only, unverifiable now)

| Kill | V1 Measurement | V1 Verdict |
|------|----------------|------------|
| K1070 | 0/40 token diffs | PASS |
| K1071 | history = 100% vs base = 4% | PASS |
| K1072 | p99 = 0.000922ms (10,800× margin) | PASS |

V1 runtime: 116s. V2 probe runtime: 0.001s (filesystem + micro-dict benchmark only,
no model load).

## Kill Causes

### C1 — Tautological routing (mem-antipattern-002)

V1 code hardcodes `REAL_ADAPTER_PATHS[domain] → fixed path`. Each domain
is evaluated only against its matched adapter. Removing `geography` changes
the dict key `geography`; it never affects the dict paths returned for
`math`, `medical`, `legal`, `finance` because Python dict lookup is a
constant function of the requested key, not of registry state. K1070
"remaining outputs bit-exact after removing geography" is then algebraically
forced, not Theorem 1 evidence. A genuine test requires simultaneous N≥2
activation or per-sample routing — not `dict[domain]`.

### C2 — Adapter copy forgery (mem-antipattern-009)

V1 mints both adapters in the test via `shutil.copy(finance_adapter_dir, ...)`:

    create_adapter_copy(T26/adapters/finance, adapter_geography/)
    create_adapter_copy(T26/adapters/finance, adapter_history/)

So `weights(geography) = weights(history) = weights(finance)` byte-for-byte.
K1071's `history = 100%` on `high_school_european_history` is finance-weights
answering MCQ letter-space format, not a novel adapter. The `> 4% base`
threshold rules out "had no effect"; a byte-copy of a trained adapter
trivially clears it.

### C3 — Dict mutation vs weight unload (mem-antipattern-011 specialisation)

Theorem 3's "hot-remove latency" should be the cost of releasing adapter
*weights* — GPU memory free, mmap close, model reference drop. V1 measured
`del d[k]` on a Python dict (`dict.__delitem__` is O(1) amortised by
hash-table construction). A 10 ms threshold on a ~1 µs operation has no
discriminating power. V1's 10,800× margin is on the wrong benchmark.

### C4 — Upstream artefact precondition (downstream of T2.1 + T2.6 audit)

All five upstream adapter `.safetensors` files are absent from disk:

    math        — micro/models/exp_p1_t2_single_domain_training/adapters/math/       [config only]
    code        — micro/models/exp_p1_t2_single_domain_training/adapters/code/       [config only]
    medical     — micro/models/exp_p1_t2_single_domain_training/adapters/medical/    [config only]
    legal       — micro/models/exp_p1_t2_multi_domain_5/adapters/legal/              [config only]
    finance     — micro/models/exp_p1_t2_multi_domain_5/adapters/finance/            [config only]

T2.1 status=KILLED (2026-04-18, metric-swap + format-artefact). T2.6 weights
lost. Even a V3 that fixed C1–C3 cannot run without weights.

## Theorem Correctness Note

Theorems 1–3 in MATH.md are mathematically correct *as statements*:

- Theorem 1: under exclusive routing with r(x) a function of registry state,
  removing k ≠ j genuinely leaves f_j invariant.
- Theorem 2: a freed dict key can hold a new value; a new adapter occupying
  that label is functionally independent of the removed one.
- Theorem 3: weight unload is O(1) in the sense that it does not depend on N.

The V1 sin is not the theorems but the *operationalisation*:
- K1070 measured `dict[j]` not `f_j(x)` under changing registry.
- K1071 measured finance-weights-under-two-labels not two distinct adapters.
- K1072 measured `del d[k]` not unload(weights).

## Permanently learned (class-level standing rules from 7 precondition-probe kills in 24 h)

1. **Precondition-probe before macro claim** (mem-antipattern-002 + 006).
   Every macro-scale claim must first verify artefact presence (`.safetensors`
   on disk, not `adapter_config.json` alone).

2. **Registry ≠ artefacts.** Dir existence is not file existence. Grep for
   `.safetensors` size, not directory listings.

3. **Downstream P1 macros inherit upstream audit flags.** If an upstream is
   KILLED or its artefacts are lost, the downstream inherits precondition
   failure even if its own code is correct.

4. **`code-bug` tag may be a decoy.** A V1 failure can be a mathematical
   property of the test design (gradient identity, oracle lookup) — fixing
   "code bugs" won't resurrect it; only rewriting the operationalisation will.

5. **Composition requires genuine routing** — not `ADAPTER_PATHS[domain]`.
   Any composition / interference / remove-invariance test whose routing is
   a constant function of domain is a tautology.

6. **Hot-add/hot-remove latency ≠ dict mutation.** Time the weight I/O or
   weight unload (the Theorem 3 object), not `dict.__setitem__` /
   `dict.__delitem__`.

7. **Adapter mint must be training, not `shutil.copy`** (mem-antipattern-009).
   A new adapter minted by copying an existing trained one is the same
   weights under a different label — any quality claim is a lie by identity.

## Routing signal for next hat

- Reviewer: 7th precondition-probe kill in 24 h. Class-level standing.
  No new mem-antipattern required — mem-antipattern-002, 009, 011 all apply.
  Adversarial checklist: verify results.json KILLED, no V2 KC relaxation,
  no V1 leakage as supported, `.safetensors` actually absent (independent re-check).
- Downstream of T3.6 (already KILLED). Unblocks further T4/T5 only once
  T2.1 is rebuilt AND the V1 design flaws are fixed in a V3 that implements
  genuine routing, trains real novel adapters, and benches weight unload.
- Do not auto-spawn V3 until T2.1 is rebuilt with MedQA USMLE 5-choice,
  max_tokens ≥ 512, persisted `.safetensors`, and `adapters/code/` created.
