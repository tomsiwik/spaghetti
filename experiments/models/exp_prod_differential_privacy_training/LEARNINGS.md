# LEARNINGS: `exp_prod_differential_privacy_training` (KILLED_PREEMPTIVE, ap-017)

## Cohort-level reusable insight
ap-017 (5-theorem stack) now has a **6th F#502 schema-incomplete
instance** and — for the first time in the drain — a **4-theorem block**
(T1 ∧ T2 ∧ T3 ∧ T5 all independently fire). Pattern after this iter:

| F#502 # | Target                                      | Theorems firing  |
| ------- | ------------------------------------------- | ---------------- |
| 1       | `exp_g4_tfidf_routing_no_alias_composition` | T1, T3, T5       |
| 2       | `exp_g4_flywheel_real_users`                | T1, T3, T5       |
| 3       | `exp_prod_adapter_loader_portability`       | T1, T3, T5       |
| 4       | `exp_prod_adapter_registry_host`            | T1, T3, T5       |
| 5       | `exp_prod_version_resolution`               | T1, T3, T5       |
| **6 (this iter)** | `exp_prod_differential_privacy_training` | **T1, T2, T3, T5** |

The 6×-stable heuristic for analyst formalization post-cap-raise:
**DB literal `success_criteria: []` + `⚠ INCOMPLETE` tag ≡ preemptible
target under ap-017 unless the author can point to an out-of-DB spec.**

## Target-specific insight
`exp_prod_differential_privacy_training` demands a DP-SGD optimizer on
MLX + RDP accountant + per-sample gradient primitive + non-DP
comparator baseline. None of these exist in:
1. The repo (`opacus|make_private|RDPAccountant|noise_multiplier` = 0
   non-comment code hits after self-match exclusion).
2. `pyproject.toml` (no DP library declared in any extra).
3. The open-source MLX ecosystem as of 2026-01 (no MLX port of Opacus,
   jax-privacy is JAX-only, TensorFlow-Privacy is TF-only).
4. The declared source `exp_p1_t5_user_local_training`'s MATH.md
   (0 DP-vocabulary hits: `privacy | differential | epsilon | dp-sgd
   | gaussian noise | clip grad`).

The source's Theorem 1 Step 2 convergence bound is a *standard* SGD
bound that does not hold under DP-SGD noise injection. Bassily et al.
2014 (arxiv:1405.7085) gives the correct excess-risk term
`O(√(log(1/δ)) / (ε·√n))` — a separate theorem the source never
derived.

## New sub-axis registration (analyst to formalize)
**ap-017 (s3) platform-library-absent-from-target-ecosystem.** Distinct
from iter 35–36's (s) `hardware-topology-unavailable` (physical/DNS
infra) and iter 37's (s2) `software-infrastructure-unbuilt` (in-repo
software gap, fixable by internal build-out). (s3) is the case where
the required library *exists in the ML ecosystem* (Opacus for PyTorch,
jax-privacy for JAX) but is *absent from the target platform's
ecosystem* (MLX has no equivalent). This makes the gap irreducible by
`pip install` *or* in-repo build-out short of a full library port.

Decision tree for analyst:
```
Is the required library present in-repo or as a pyproject dep?
├── Yes → no ap-017 (s*) axis applies; continue standard 5-theorem check
└── No → Does it exist in the target platform's open-source ecosystem?
        ├── Yes → ap-017 (s2) software-infrastructure-unbuilt
        └── No → Does it exist in ANY ML ecosystem (PyTorch/JAX/TF)?
                ├── Yes → ap-017 (s3) platform-library-absent-from-target-ecosystem ← this iter
                └── No → The feature is unbuilt globally; P≥3 by default
```

## Reusable T1 probe pattern (refined from iter 37)
```python
def probe_library_presence(patterns, *, drop_comments=True, drop_self=True):
    # Refinements learned from this iter:
    # 1. drop_self (exclude the probe runner's own file from grep hits;
    #    iter 37's runner did not and this iter's runner initially
    #    self-matched 10 dp_primitive hits);
    # 2. drop_comments (docstring / bullet-list / math-notation matches
    #    like `sigma_noise` as a variable name don't count as evidence
    #    of library usage);
    # 3. drop_data (jsonl / cassettes / data dirs are not code usage);
    # 4. drop_skills (agent skill docs are not in-repo evidence).
    hits = grep_filtered(patterns)
    if drop_self:
        hits = [l for l in hits if not l.startswith(str(Path(__file__)))]
    if drop_comments:
        hits = [l for l in hits
                if not l.split(":", 2)[-1].lstrip().startswith("#")]
    return hits
```

This refinement should be lifted into a shared `preempt_common.py`
helper when the analyst cap raises; iter 37's runner had the same bug
but fewer false positives because the semver probe strings
(`VersionRange`, `packaging.version`) are rarer than DP-SGD primitives
in scientific docstrings.

## Why T2 first blocks on its own here
The 3-seed K1666 reproducibility requirement is the **first macro-level
KC in the audit drain** where wall-time cost compounds *multiplicatively*
with a library-absence cost:
- Non-DP T2.1 baseline: 22 min → manageable on 120-min ceiling.
- DP-SGD at 10× Opacus floor: 220 min → already over ceiling.
- K1666 × 3 seeds: 660 min → 5.5× over.
- K1665 non-DP comparator pair × 3: +66 min → **726 min, 6.05× over.**

This makes ap-017 T2 a *first-class blocker* when:
- The target KC has an N-seed reproducibility subclause.
- The target's per-trial cost has a multiplicative library overhead
  (DP-SGD, formal verification, multi-party computation, homomorphic
  encryption — any sub-field where per-sample semantics forces
  loop-level overhead on top of a standard training run).

## Integration with HALT §C analyst cap
With the cap at 50/50 (per scratchpad iter-35 coordinator log), this
iter adds the **12th LEARNINGS debt entry**. When the operator lifts
the cap, this iter's LEARNINGS.md should be the seed for:
1. A cohort-level paper on "6× F#502 schema-completeness heuristic"
   (first 6-point pattern in the drain).
2. An ap-017 axis family diagram with (s), (s2), (s3) as sub-nodes.
3. A `preempt_common.py` helper with the self-match + comment +
   data-dir + skill-dir exclusions baked in.

## Non-goals for LEARNINGS.md (analyst-owned)
The following are *explicitly* deferred to analyst iter 32+ when the
cap lifts:
- Deciding whether (s3) should be promoted to a top-level ap-017 axis
  (parallel to composition-bug) or remain a sub-axis of composition-bug.
- Recommending a PLAN.md Part 2 amendment (operator decision).
- Formalizing F#X (success-failure-path-non-transfer) from iter 37 into
  a registered Finding (analyst-owned).
- Deciding whether the 6× F#502 stability warrants auto-killing future
  `success_criteria: []` targets on DB intake, or whether a human
  sign-off should remain in the loop.
