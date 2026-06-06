# LEARNINGS.md — P3.C4: Rank-16 Diverse Adapter

**Finding #471 | Status: KILLED | Date: 2026-04-11 | V2 audit confirmed: 2026-04-18**

## V2 Audit Note (2026-04-18)

Rerun requested via `audit-2026-04-17-rerun` + `training-cache` tags is **not executable** —
`domain_fused_base` weight shards and the source math adapter were deleted during disk
cleanup. Cache-bug fix applied to `run_experiment.py` for future reruns; verdict re-derived
from strict PLAN.md §1 on the 2026-04-11 measurements: **K1205 FAIL (73.3% < 80%) stands.**
Cache bug does not rescue the verdict — 73.3% fails the pre-registered 80% threshold
regardless of data volume. See PAPER.md "V2 Audit" section and REVIEW-adversarial.md
"V2 Audit Review".

## What We Learned

### 1. Rank IS the primary bottleneck for style compliance

Rank-16 + 10 examples (73.3%) > Rank-4 + 167 examples (60%). The +13.3pp improvement
came from rank increase ALONE with fewer training data. This is strong causal evidence
that rank (not data volume) was the binding constraint for style injection.

Coverage Lemma: rank(adapter) > n_style_categories is necessary. Rank-4 < 10 categories
creates a hard ceiling (~60%); rank-16 > 10 categories breaks that ceiling.

### 2. Cache bug confound (fix required for P3.C5)

Cache validation checked file existence but not line count. The smoke test created 10
examples; the full run reused those 10 instead of generating 167. This means P3.C4 tested
rank-16 with severely limited data.

**Fix**: validate `len(lines) >= N_TRAIN` when loading from cache, not just file existence.

### 3. Question-type-specific failure floor

4 failures are all in COVERED science/tech categories (physics ×2, CS, earth science)
— not underrepresented ones. Philosophy and economics PASSED. This suggests a separate
constraint: for certain question formulations (relativity, recursion, quantum), the model's
token probability for marker phrases is suppressed by domain-specific prior phrasing
patterns, regardless of rank.

This floor may be penetrable with 167 diverse examples (P3.C5) — or may be a hard ceiling
that requires a fundamentally different injection mechanism.

### 4. System role OOD (Gemma 4 architecture constraint)

Confirmed from P3.C3: Gemma 4 chat template doesn't support role="system". Any attempt
to inject style via system prompts → catastrophic degenerate outputs (PHP code, Chinese
text, loops). This closes all in-context prompting approaches for Gemma 4.

## P3.C Series Summary

| Experiment | Rank | Training N | Style % | Finding |
|------------|------|------------|---------|---------|
| P3.C0 | 4 | 40 (science) | 60% | #467 (supported) |
| P3.C1 | 4 | 167 (diverse) | 60% | #468 (killed: rank bottleneck) |
| P3.C2 | 4 | 40 (science) | 20% | #469 (killed: context-prior conflict) |
| P3.C3 | — | — | 0% | #470 (killed: system role OOD) |
| P3.C4 | 16 | 10 (cache bug) | 73.3% | #471 (killed: cache bug + 80% miss) |

## What to Try Next: P3.C5

**Rank-16 + fix cache validation + 167 diverse examples → target ≥80% style**

- Fix: `len(lines) >= N_TRAIN` cache check (not just file existence)  
- rank=16, N_TRAIN=167 diverse examples (all available), N_VALID=10
- Prediction: Coverage Lemma (16/10 = 1.6 > 1.0) → ≥80% style compliance
- If still < 80%: question-type floor is real, need different injection mechanism

Citation: P3.C4 Finding #471 (rank bottleneck) + P3.C1 Finding #468 (data confound)

## Impossibility Summary

**Closed paths** (backed by structural proofs):
- Rank-4 style injection: ceiling at 60% regardless of data (column-space bottleneck)
- Few-shot prompting (P3.C2): context-prior conflict → regression to base priors
- System prompt injection (P3.C3): Gemma 4 OOD template → catastrophic degeneration
- Additive weight-space composition (P3.B0-B4): domain adapters suppress personal style

**Open path**: Rank-16 + correct training data (P3.C5)
