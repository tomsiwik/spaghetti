# P5.C1: Per-Adapter Reasoning Strategy — Results

## Summary

**Status: PROVISIONAL (2/3 pass, but mechanism is wrong)**

The strategy selection framework (Theorem 1, Theorem 2) works: choosing the optimal
strategy per domain improves accuracy by +12pp and reduces tokens by 37.9%. However,
the optimal strategy is uniformly "Direct" across ALL five domains — there is no
per-domain differentiation. The improvement comes from avoiding CoT (which hurts
code/finance accuracy), not from matching domains to specialized strategies.

## Prediction vs Measurement

| Metric | Prediction (MATH.md) | Measured | Match? |
|---|---|---|---|
| K1279: Accuracy ≥ +5pp | PASS if δ̄ ≥ 6pp | **+12.0pp** (88% → 100%) | **PASS** (wrong mechanism) |
| K1280: Token reduction ≥ 30% | PASS if Direct ~60-70% shorter | **37.9%** (502 → 311 tokens) | **PASS** |
| K1281: Routing accuracy ≥ 80% | Trivially pass (TF-IDF 95%) | **64.0%** (16/25) | **FAIL** |
| Per-domain strategy differentiation | δ̄ > 0, strategies differ | **δ̄ = 12pp but ALL domains → Direct** | **WRONG** |
| Hypothesized strategy matches | math=CoT, code=PAL, legal=structured | **0/5 match** (all=Direct) | **WRONG** |

## Strategy × Domain Accuracy Matrix

| Domain | CoT | Direct | PAL | Structured | Best |
|---|---|---|---|---|---|
| Math | 100% | **100%** | 100% | 100% | Direct (fewest tokens) |
| Code | 60% | **100%** | 100% | 80% | Direct |
| Legal | 100% | **100%** | 80% | 100% | Direct |
| Medical | 100% | **100%** | 100% | 100% | Direct (fewest tokens) |
| Finance | 80% | **100%** | 100% | 100% | Direct |

## Strategy × Domain Token Count Matrix

| Domain | CoT | Direct | PAL | Structured |
|---|---|---|---|---|
| Math | 459 | **173** | 238 | 512 |
| Code | 512 | **397** | 488 | 512 |
| Legal | 512 | **330** | 512 | 512 |
| Medical | 512 | **332** | 512 | 512 |
| Finance | 512 | **324** | 466 | 512 |

## Analysis

### 1. Direct Answering Dominates

The most striking finding: "Direct" achieves 100% accuracy on all domains while
generating 38% fewer tokens. Gemma 4 E4B-IT is capable enough for these QA tasks
that step-by-step reasoning (CoT) doesn't help and sometimes hurts:

- **Code + CoT = 60%**: CoT generates reasoning about the code instead of writing
  the actual function, hitting max_tokens before producing the expected `def` pattern.
- **Finance + CoT = 80%**: CoT sometimes elaborates on concepts instead of computing
  the numerical answer.
- **Legal + PAL = 80%**: PAL generates Python code for legal questions, which is
  nonsensical for knowledge-retrieval tasks.

### 2. CoT Harms Token Efficiency

CoT and Structured strategies consistently hit the 512 max_tokens ceiling, generating
verbose reasoning that often doesn't improve accuracy. Direct answers average 312 tokens
— 38% reduction. This matches Theorem 2's prediction that concise strategies save tokens.

### 3. No Strategy Differentiation Exists

The core hypothesis was wrong: different domains do NOT prefer different strategies
at this task complexity level. The per-domain delta vector is:

    δ = [0, 40, 0, 0, 20] pp

Only code (+40pp) and finance (+20pp) show any strategy differentiation, and in both
cases it's because CoT HURTS, not because an alternative is specifically better.
Math, legal, and medical show 100% accuracy across Direct/CoT/Structured.

### 4. TF-IDF Routing Is Too Simplistic for QA

K1281 failed (64% vs 80% threshold) because the TF-IDF corpus uses domain keywords
but the test questions are generic QA questions without heavy domain vocabulary.
For example, "If 3x + 7 = 22, what is x?" doesn't contain many math-specific keywords.
Finding #196's 95% accuracy was on domain-specific text passages, not QA questions.

### 5. Why Hypotheses Were Wrong

All five domain→strategy mappings were wrong (0/5). The hypotheses assumed:
- Math needs step-by-step (CoT) → but these are simple arithmetic, not complex reasoning
- Code needs PAL → but PAL is redundant when the task IS to write code
- Legal needs structured → but structured adds overhead without helping accuracy

The error was assuming task complexity high enough to differentiate strategies.
On harder tasks (multi-step proofs, complex debugging, nuanced legal analysis),
strategy differentiation may emerge.

## Behavioral Assessment

Direct answering produces correct, concise responses across all domains. CoT produces
unnecessarily verbose responses that sometimes fail to reach the answer within the
token budget. No behavioral pathology was observed (no degeneration, repetition loops,
or format contamination as in P5.C0).

The behavioral finding: **"less is more"** — for this model on these tasks, the
simplest strategy (Direct) produces the best behavioral outcomes.

## Impossibility Structure

Strategy differentiation requires tasks of sufficient complexity that different
reasoning formats provide structurally different paths to the answer. When the model
can answer correctly with direct recall (as with these QA tasks), all strategies
converge to the same answer — differentiation is impossible.

**What would make it work**: Harder tasks where direct answering fails but specialized
reasoning helps (e.g., multi-step math proofs where CoT is structurally necessary,
or debugging tasks where PAL can actually execute code).

## Connection to P5.C0

P5.C0 (Standing Committee Adapter) was killed because composing reasoning + domain
adapters on different modules caused Q×V interference. P5.C1 avoids composition
entirely (one strategy per query). The finding here is complementary: even without
composition issues, per-domain strategy routing doesn't help at this task complexity.

## Theorem Validation

- **Theorem 1 (Strategy Selection Bound)**: Validated. The bound predicts
  E[Δacc] ≥ 0.95 × 12 - 0.05 × 40 = 9.4pp. Measured: +12pp.
  But this overstates the finding — the improvement is from uniformly
  switching to Direct, not from per-domain routing.

- **Theorem 2 (Token Reduction)**: Validated. Predicted reduction when optimal
  strategy ≠ CoT: 38% measured vs 30% threshold. Direct averaging 312 tokens
  vs CoT averaging 502 tokens.

## Caveats

1. Simple QA tasks — insufficient complexity to differentiate strategies
2. Keyword-matching evaluation — not a true accuracy measure
3. max_tokens=512 ceiling — CoT/PAL may perform better with higher budget
4. No adapter coupling — tested strategy prompts only, not adapter + strategy
5. Small model (E4B 4-bit) — larger models may show different strategy preferences

## Total Runtime

550.4 seconds (100 generations at ~5.5s average, including model load).
