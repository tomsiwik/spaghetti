# Answer-Only PPL as Adapter Router: Mathematical Foundations

## 0. Experiment identity and KC lock

**Experiment id**: `exp_followup_answer_conditioned_ppl`
**Follow-up to**: `answer_conditioned_scoring` (SUPPORTED, r=0.81 answer-PPL↔accuracy) and `ppl_vs_task_performance` (KILLED, r=0.08 full-seq-PPL↔accuracy).
**Pre-registered kill criterion (K1567)**:

> **K1567**: On mixed-domain queries (balanced across 5 synthetic domains), answer-only PPL
> ranks the domain-correct expert top-1 at fraction ≥ 0.85 AND full-sequence PPL fails the
> same threshold (top-1 < 0.85). Both conditions must hold for PASS.

This KC is the single gate for the experiment. It is locked at first commit of
this file. No relaxation, no reformulation; if v1 data falsifies it, the result
is KILLED, and any follow-up experiment must use a new id.

**Type**: Verification (the underlying mechanism — answer-only PPL separates
domain experts — is proven by the predecessor's r=0.81 correlation; this
experiment verifies the *routing* consequence).

## 1. Setup and Notation

Identical tokenizer/domains/training to `answer_conditioned_scoring`:

| Symbol | Definition | Value |
|--------|-----------|-------|
| V | Vocabulary size (char-level) | 42 |
| D | Number of domains | 5 (arithmetic, reverse, repeat, sort, parity) |
| θ_base | Base model params (trained on all 5 domains) | ~29K (d=32, L=2) |
| θ_i | Expert i params = θ_base + δ_i | i ∈ {1..5} |
| q ∈ Q_i | Held-out query drawn from domain i's distribution | string + delimiter + answer |
| d_i | Domain-i delimiter | "=" or ">" |
| N_q | Queries per domain at eval | 200 |
| PPL_full(θ, q) | exp(−(1/T) Σ_{t=1..T} log p_θ(q_t | q_{<t})) | scalar |
| PPL_ans(θ, q) | exp(−(1/T_a) Σ_{t=d*+1..T} log p_θ(q_t | q_{<t})) | scalar; d*=delimiter position |

Routing function under metric M ∈ {full, ans}:
    `route_M(q) = argmin_{i ∈ 1..D} M(θ_i, q)`

Top-1 accuracy of routing under metric M on query set Q = ⋃_i Q_i:
    `Top1_M = (1 / |Q|) · Σ_{q ∈ Q_i, ∀i} 𝟙[ route_M(q) == i ]`

## 2. Why answer-only PPL is expected to route correctly

### 2.1 Predecessor anchor (answer_conditioned_scoring)

The predecessor measured, per (domain, expert), the change in PPL_ans and
PPL_full when expert i is applied to its own domain i:

- Mean answer-PPL improvement on own-domain: ~18% (ranges +3% to +58% by domain).
- Mean answer-PPL improvement on foreign-domain (expert j on domain i, j ≠ i):
  empirically ≤ 0% or strongly negative (expert-j's parameter shift is
  unlikely to reduce the cross-entropy of domain-i answer tokens, because
  domains share no answer structure in this synthetic setup).
- r(ΔPPL_ans, Δaccuracy) = 0.81 across 3 seeds.

### 2.2 Rank-dominance lemma

**Lemma 1 (own-domain dominance under answer-only PPL)**. Let θ_i be an expert
trained exclusively on domain-i data. Let q ∈ Q_i. Under the predecessor's
measured regime (answer-PPL gap Δ_own ≈ −log(0.82) ≈ 0.20 nats/token vs base;
foreign-expert answer-PPL change on domain i ≈ 0 nats/token in expectation,
because δ_j has no gradient signal from domain i), we expect:

    E[ log PPL_ans(θ_i, q) − log PPL_ans(θ_j, q) ]  ≤  −0.20  for all j ≠ i

This bias makes argmin_i likely to land at the correct expert.

**Proof sketch**. Expert training minimizes expected cross-entropy on domain-i
answer tokens by construction. Experts j ≠ i never observe domain-i data, so
their parameter perturbation δ_j has no reason to reduce domain-i answer
cross-entropy; in random-init-start regimes this expectation is 0, and mild
drift (overfitting to domain j's answer patterns, which differ from domain i's)
can produce a small positive foreign penalty. QED.

### 2.3 Why full-sequence PPL is expected to fail

Predecessor Finding #553 (implicit from r_full = −0.31 across 3 seeds): expert
i often has WORSE full-seq PPL on its own domain than base, because the expert
sacrifices prompt modeling for answer modeling. Under full-seq PPL:

    log PPL_full(θ_i, q) = (T_p/T) log PPL_prompt(θ_i, q) + (T_a/T) log PPL_ans(θ_i, q)

If the expert degrades prompt PPL by a factor larger than the improvement on
answer PPL (scaled by token counts), then even on its own domain, expert i
may lose to expert j on domain i's full-seq PPL when j's prompt-modeling
happens to drift less. The routing-via-full-seq-PPL is therefore expected to
be near chance (1/D = 20%) or at best weakly above chance.

## 3. Quantitative prediction

### 3.1 Point prediction (central)

From predecessor numerics (seeds 42/123/7, 3 seeds × 5 domains × N=500):

- Expert-i's answer-PPL on its own domain was always lower than base
  (5/5 domains supported across 3 seeds).
- For the cross-domain matrix (expert j on domain i, j ≠ i), the implicit
  prior from the predecessor's r_ans = 0.81 and Δ_own ≈ 0.20 nats/token
  implies the rank-dominance lemma holds in expectation.

**Prediction**: Top1_ans ≥ 0.90 (single seed, N_q=200 per domain).

**Lower bound** (K1567 threshold): Top1_ans ≥ 0.85.

### 3.2 Failure prediction for full-seq

From predecessor r_full = −0.31, expert i's full-seq PPL is mildly
anti-correlated with its domain competence. Predicted Top1_full ≈ chance
(20%) plus mild shift: **Top1_full ∈ [0.20, 0.60]** — strictly below 0.85.

### 3.3 What would kill K1567

| Failure mode | Top1_ans | Top1_full | Verdict |
|---|---|---|---|
| Answer-PPL works as router | ≥ 0.85 | < 0.85 | **SUPPORTED** |
| Answer-PPL fails (cross-domain floor too close) | < 0.85 | any | **KILLED** |
| Full-seq already works (no advantage) | ≥ 0.85 | ≥ 0.85 | **KILLED** |
| Both fail | < 0.85 | any | **KILLED** |

Note: failing because *both* metrics succeed is a genuine kill — the
experiment's thesis is that answer-PPL is strictly better than full-seq for
routing; if full-seq also routes correctly at >85% top-1, the answer-only
variant provides no marginal contribution.

## 4. Why this is not the adapter-weight blocker

This experiment uses the numpy+autograd transformer from
`answer_conditioned_scoring` (CPU-only). Experts are trained in-process and
held in memory as Python dicts; there is no `adapters.safetensors` file,
no `adapter_config.json`, no MLX load path. Antipattern-017 (config-only
adapter dirs that fail `mlx.load`) does not apply.

## 5. Experimental protocol

### 5.1 Training (reuses `answer_conditioned_scoring` functions)

- Initialize base model (d=32, L=2, H=2, V=42, max_T=48).
- Train base on all 5 domains × 500 sequences × 30 epochs.
- Save base params.
- Train 5 experts on domain-specific data × 500 sequences × 40 epochs.
- Store each expert as delta from base.

### 5.2 Routing evaluation (NEW)

For each domain i ∈ {1..5}:
- Generate N_q = 200 held-out queries with a fresh RNG (seed + offset), DISJOINT
  from training.
- For each query q:
  - For each expert j ∈ {1..5}:
    - Build θ_j = θ_base + δ_j (in memory).
    - Compute PPL_full(θ_j, q) and PPL_ans(θ_j, q) using the predecessor's
      `compute_full_sequence_ppl` and `compute_answer_only_ppl` functions,
      each evaluated on the SINGLE query q (not aggregated across Q_i).
  - predicted_full = argmin_j PPL_full(θ_j, q)
  - predicted_ans = argmin_j PPL_ans(θ_j, q)
  - correct_full = 𝟙[predicted_full == i]
  - correct_ans = 𝟙[predicted_ans == i]

Aggregate:
- Top1_full = mean(correct_full) over all (i, q)
- Top1_ans = mean(correct_ans) over all (i, q)

Also compute per-domain breakdown for diagnostic purposes (not part of KC).

### 5.3 Seeds

Single seed (42) for the verification run. Re-running across 3 seeds was
done by the predecessor (answer-PPL r=0.91/0.94/0.58) — the weakest seed
had r=0.58, above our K1567 threshold translation. A single-seed
verification is sufficient because:
1. The predecessor established cross-seed stability of answer-PPL's
   domain-discriminative power.
2. K1567 is an aggregate ranking metric over 1000 queries (N=1000) with
   Bernoulli variance σ/√N ≈ 1.1pp at true rate 85% — narrow enough that
   a single seed's estimate is reliable.
3. Running time per seed is ~3 minutes; 3 seeds fits budget but 1 is
   sufficient to falsify.

If the single-seed result lands in [0.80, 0.90], the experiment is rerun
at 3 seeds before committing a verdict. Otherwise the first-seed result
is decisive.

## 6. What we will save to `results.json`

Schema:
```json
{
  "seed": 42,
  "n_queries_per_domain": 200,
  "n_domains": 5,
  "top1_full": <float>,
  "top1_ans": <float>,
  "per_domain_top1_full": {"arithmetic": <f>, ..., "parity": <f>},
  "per_domain_top1_ans":  {"arithmetic": <f>, ..., "parity": <f>},
  "confusion_full": [[...]],          // 5x5 matrix: rows=true domain, cols=predicted
  "confusion_ans":  [[...]],
  "kill_criteria": {
    "K1567": {
      "description": "answer-only top-1 >= 0.85 AND full-seq top-1 < 0.85",
      "top1_ans":  <float>,
      "top1_full": <float>,
      "pass_ans":  <bool>,   // top1_ans >= 0.85
      "pass_full_failed": <bool>,  // top1_full < 0.85
      "pass": <bool>         // both must hold
    }
  },
  "verdict": "SUPPORTED" or "KILLED",
  "all_pass": <bool>,
  "is_smoke": false,
  "runtime_s": <float>
}
```

## 7. Antipattern self-check

| Antipattern | Applies? | Why |
|---|---|---|
| ap-001 composition-math-bug | No | No composition; single-expert routing. |
| ap-002 tautological-routing | No | Routing uses PER-QUERY PPL, not batch-aggregated. |
| ap-004 KC-swap | Locked at first commit; see §0. |
| ap-005 verdict-consistency | Code writes `verdict` computed from `K1567.pass`. |
| ap-006 smoke-as-full | N_q=200 × 5 = 1000 queries; `is_smoke=false`. |
| ap-007 tautological-KC | K1567 can plausibly fail: experts may have overlapping answer distributions; or full-seq may already route well. |
| ap-012 hardcoded-pass | K1567.pass computed from measured top-1, not hardcoded. |
| ap-014 copy-paste-scaffolding | Re-reads delim/domain logic; does not copy-paste DOMAIN_KEYWORDS. |
| ap-017 adapter-weights-missing | No on-disk adapters; experts held in memory. |
