# PAPER.md — exp_method_composition_k_saturation

## Verdict

**PROVISIONAL (smoke — aborted at Phase 1 teacher gate).**
`is_smoke=true`; `all_pass=false`; `results.json["verdict"]="ABORTED"`.

Per guardrail 1009 a smoke-scale run cannot be promoted to `supported`
or `killed`.  This run did not reach Phase 3 (the saturation sweep),
so none of `K1730/K1731/K1732` were measured.  The substantive
pre-training result — **Gemma-4-E4B-it-4bit under thinking mode does
NOT reliably emit explicit method signatures at teacher-generation
time** — is reported as the primary smoke finding with a v2 plan
below.

## Summary

Goal: measure the LoRA-composition k-saturation curve predicted by
Skill-Mix (Arora & Goyal 2023, `arxiv:2310.17567`) on
Gemma-4-E4B-it-4bit.  5 method adapters (rank 8, `v_proj+o_proj`,
scale 4.0, top-16 layers) with textually-disjoint signatures
(`restate / numbered / verify / principle / tldr`) would be trained,
rank-stack-composed for `k ∈ {1..5}`, and evaluated on held-out
MMLU-Pro.

## Predictions vs measurements

| Quantity                                        | Predicted                          | Measured (smoke)                                                                      |
| ----------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| Teacher signature rate (gate ≥ 70 %)            | ≥ 70 % for each of 5 methods       | **20–40 %** for **all 5 methods** (restate 20, numbered 40, verify 40, principle 20, tldr 40) |
| # method-adapters trained                       | 5                                  | **0** (all skipped: teacher gate fail)                                                |
| `M_solo` (mean solo method-match rate)          | ≥ 0.70                             | **unmeasured** (no trained adapters)                                                  |
| `μ(k=2)` survival (K1730: ≥ 0.95 · M_solo)      | ≥ 0.95 · M_solo                    | **unmeasured**                                                                        |
| `μ(k=5)` saturation (K1731: ≤ 0.80 · M_solo)    | ≤ 0.80 · M_solo                    | **unmeasured**                                                                        |
| Monotonic curve (K1732)                         | μ non-increasing, no +0.05 spikes  | **unmeasured**                                                                        |

## KC evaluation (smoke — all unmeasured)

- **K1730** (k=2 ≥ 0.95 × M_solo): **UNMEASURED.**  Phase 3 never
  executed (no adapters available).
- **K1731** (k=5 ≤ 0.80 × M_solo, saturation): **UNMEASURED.**
- **K1732** (monotonic, no +0.05 spike): **UNMEASURED.**

Because all three KCs are unmeasured, the experiment is `ABORTED`
in `results.json["verdict"]` and cannot be completed as `supported`
or `killed`.  The appropriate status is `provisional` (smoke +
aborted precondition).

## Smoke outcome — teacher-signature gate failure

All five teacher-prompt designs failed the 70 % signature gate.
Raw rates on n=5 per method:

| Method    | Example prompt cue                                | Regex                                    | n | hits | rate |
| --------- | ------------------------------------------------- | ---------------------------------------- | - | ---- | ---- |
| restate   | "Begin with: Problem restated: …"                 | `(?m)^\s*Problem restated:\s`            | 5 | 1    | 20 % |
| numbered  | "Step 1, Step 2, Step 3, then Answer: X"          | `\bStep\s+2\b`                           | 5 | 2    | 40 % |
| verify    | "After reasoning, add: Check: … then Answer: X"   | `(?mi)^\s*(Verification\|Check):\s`      | 5 | 2    | 40 % |
| principle | "Begin with: Principle: …"                        | `(?mi)^\s*(Principle\|Rule):\s`          | 5 | 1    | 20 % |
| tldr      | "Finish with two lines: Answer: X / TL;DR: …"     | `(?mi)\bTL;?DR:\s`                       | 5 | 2    | 40 % |

### Failure-mode diagnosis

1. **Thinking-channel dominance.**  Gemma-4-E4B-it-4bit under
   `enable_thinking=True` emits `<|channel>thought...<channel|>`
   preamble that absorbs most of the instruction-following budget.
   After `strip_thinking`, the visible answer is typically a
   minimal `Answer: X` line without the requested method-specific
   scaffolding.  The model satisfies the `Answer: X` contract but
   drops the method prefix/suffix.
2. **Regex-too-strict (partial).**  Some responses may have had
   near-signatures (e.g., "Let me restate …" instead of the exact
   "Problem restated: …" prefix, or "Step 1 … Step 3" that skips
   the literal "Step 2" marker).  The regex definitions in MATH.md
   are anchored-strict on purpose (for disjointness during
   composition) but this raises the teacher miss rate.
3. **4-bit quantization drift.**  Relative to the full-precision
   base used in Arora et al.'s Skill-Mix work, a 4-bit quant base
   may have lower instruction-following fidelity for format-level
   directives.  Parent experiment `exp_method_vs_domain_adapter`
   hit the same class of issue (teacher sig rate 40 %, below gate).
4. **Single seed at smoke.**  n=5 per method gives ±20 pp noise
   in the gate measurement.  A 40 % observation on n=5 is
   consistent with a true 60 % population rate — still below gate
   but not catastrophically so.

**Category (2) — too-strict regex — is the most actionable root
cause**: relaxing the regex disjointness is what the parent
experiment's v2 plan pre-registered (Issue 1, "revised signature
definition").

## V2 pre-registered plan (required before macro rerun)

To produce a meaningful k-saturation measurement we need a teacher
pipeline that hits ≥ 70 % signature rate on each of the 5 methods.
Fixes, all to be applied **before** a rerun is scheduled:

1. **Relax regexes to multi-surface detection.**  Each signature
   is a **disjunction** of 2-3 surface patterns rather than a
   single anchored pattern.  E.g. `restate` becomes
   `(?i)(problem restated|restated:|let me restate|rephrasing)` —
   still textually disjoint from other methods but tolerant of
   natural-language variants.

2. **Disable thinking during teacher generation.**  Set
   `enable_thinking=False` on the teacher side so the full output
   budget goes to the visible answer (and the method signature
   gets attention).  Student data at train time already has thinking
   stripped — this change only affects teacher output shape.

3. **Increase teacher n to n=20 per method.**  At n=5 the 20 %
   observation has a 95 % CI of ≈ [3, 56] %; at n=20 the gate
   measurement is reliable.

4. **Few-shot calibration prompts.**  Prepend 2 hand-written
   demonstrations to the teacher system prompt, each showing the
   exact method signature in place.  Orca-2 and Skill-Mix both
   used few-shot prompting of the teacher; this repo's experiments
   have been zero-shot.

Only once all 5 methods gate-pass at ≥ 70 % on n ≥ 20 should the
full macro k-sweep be scheduled.  Estimated full-run cost:

- Teacher regen n=20 × 5 methods × ~15 s = ~25 min
- 5 adapters × 200 steps × ~0.5 s = ~8 min
- 10 evals (5 solo + 5 composed) × 30 qs × ~10 s = ~25 min
- **Total ~60 min** — within a single macro run budget.

## Antipattern self-scan

- ✓ No composition math bug — the rank-stack sum
  `Δ = Σ B_i @ A_i = B_cat @ A_cat` (cat along rank axis) is
  algebraically exact (see MATH.md).
- ✓ No tautological routing — all `k` subsets use the deterministic
  first-k order, no test-time adapter selection.
- ✓ No unsafe LORA_SCALE — 4.0 is within the safe band (< 8).
- ✓ No `shutil.copy` as new adapter — composed adapter is
  computed from loaded weights, not copied.
- ✓ No hardcoded `"pass": True` — all KC booleans are
  derived from measurements via comparison against thresholds.
- ✓ No KC modified between MATH.md (git history) and this run —
  KCs locked pre-first-run.
- ✓ No eval-template truncation causing base 0 % — unmeasured
  at smoke (no eval reached), so no base-0 artefact to flag.
- ✓ No smoke-as-full — `is_smoke=true` explicitly tagged in
  `results.json` and this PAPER.md verdict is `PROVISIONAL`.
- ✓ No proxy-model — same `mlx-community/gemma-4-e4b-it-4bit`
  is the teacher, trainee, and eval target per PLAN.md Part 2.

## Assumptions (per guardrail 1007)

- **Defensible call on Phase-3 skip.**  The code skips all k-sweep
  evaluation when zero methods gate-pass.  An alternative would be
  to force training of all adapters regardless of gate and let the
  composition evaluation reveal whether at-smoke method signals
  carry through.  I chose the gate-skip because training on
  teacher traces that don't contain the target signature is
  near-zero signal (an SFT loss on method-less text can't induce
  a method); the remainder of the sweep would return meaningless
  curves.  The v2 plan above targets the teacher step, which is
  the actual bottleneck.

- **Rank-stack composition choice.**  I compose via
  `A_combined = concat_rank A_i`, `B_combined = concat_rank B_i`.
  This is algebraically identical to Σ (B_i @ A_i) and has the
  advantage that a single standard LoRA-rank-k*r adapter can be
  loaded via mlx-lm's unmodified `load(..., adapter_path=...)`.
  Alternative would be hook-level runtime composition; the
  rank-stack route produces a persistable artefact per condition
  (`adapters/composed_k{k}/`) that is reproducible and inspectable.

## References

- Arora & Goyal (2023). "Skill-Mix: A Flexible and Expandable
  Family of Evaluations for AI Models."  `arxiv:2310.17567`
- Ilharco et al. (2023). "Editing models with task arithmetic."
  `arxiv:2212.04089`
- Mitra et al. (2023). "Orca 2: Teaching Small Language Models How
  to Reason." `arxiv:2311.11045`
- Sibling: `exp_method_vs_domain_adapter` (PROVISIONAL).  This
  experiment's teacher-gate failure is the same class of issue as
  the sibling's "teacher signature rate 40 % < 70 %"
  (PAPER.md Issue 1).  Both require the v2 regex-relaxation +
  few-shot calibration fix.

## mlx-lm version

```
mlx 0.31.1
mlx-lm 0.31.2
```
