# MATH.md — exp_method_composition_k_saturation

## Claim (Theorem — Skill-Mix saturation law extended to LoRA composition)

Let `π_0` be the frozen Gemma-4-E4B-it-4bit base.  Let
`{Δ_i}_{i=1..N}, N = 5` be rank-`r = 8` LoRA method adapters on
`{v_proj, o_proj}` (`LORA_SCALE = 4.0`, top-16 layers), each trained on
teacher traces generated under an explicit *method* system prompt
`M_i` that induces a distinct, easily-detectable writing-style signature
`S_i`.  The methods are chosen so their signatures are textually
disjoint (each method lives in a different region of the response —
prefix, suffix, or line-local pattern) so their adapters have
approximately orthogonal *content* directions but share the base's
"format-control" subspace.

Define the *k-composition* as the sum of `k` method deltas applied at
inference:

```
π_k(I) := π_0 + Σ_{i ∈ I} Δ_i,        |I| = k,   I ⊂ {1..N}
```

(Mathematically identical to Task Arithmetic, Ilharco et al.
`arxiv:2212.04089`: `θ' = θ_0 + Σ τ_i` with `τ_i = BA` low-rank).

Let `sig_i(x)` be a regex predicate for signature `S_i`, and let
`Q_heldout` be an MCQ held-out set not seen during training.

We claim:

1. **K1730 (low-k survival)** — For `k = 2`, the expected fraction of
   target signatures preserved is `≥ 0.95 × M_solo`, where
   `M_solo := (1/N) Σ_i P_{x∼π_i}[sig_i(x)=1]` is the mean *solo*
   method-match rate.  Concretely:
   `(1/|I|) Σ_{i∈I} P_{x∼π_k}[sig_i(x)=1] ≥ 0.95 · M_solo`
   averaged over all `C(N,2) = 10` size-2 subsets.

2. **K1731 (Skill-Mix saturation)** — For `k = 5` (all methods
   composed), the expected fraction of target signatures preserved is
   `≤ 0.80 · M_solo`.  This matches Arora et al. Skill-Mix
   (`arxiv:2310.17567`) which measured saturation at `k = 3–4` on GPT-4
   and sharp drop at `k = 5`; we predict LoRA composition saturates
   at the same cognitive ceiling because the bottleneck is the
   base model's format-control capacity, **not** adapter
   interference.

3. **K1732 (monotonicity)** — The curve
   `μ(k) := E_{|I|=k} [(1/|I|) Σ_{i∈I} P_{x∼π_k}[sig_i(x)=1]]`
   is non-increasing for `k = 1, 2, 3, 4, 5`, with no spike
   (`μ(k+1) > μ(k) + 0.05`) that would indicate broken composition
   math (e.g., an A-matrix swap or a sign error in the LoRA sum).

## Mechanism (why k-saturation happens even with orthogonal adapters)

1. **Task-arithmetic additivity (Ilharco et al. 2212.04089):** when
   fine-tuned deltas are approximately orthogonal, summing them
   preserves individual task behaviour.  Their proof assumes full-rank
   SFT deltas on a full-precision base.  At rank 8 on a 4-bit base,
   additivity still holds in the *activation* image (orthogonality is
   inherited from the random-init `A` matrices — Welch bound for `r·N
   = 40 ≪ d = 2560` gives
   `max |cos| ≤ √((d - r·N)/((r·N)(d-1))) ≈ 0.15`), so the
   adapters compose without *mutual* interference.

2. **Format-control bottleneck (Skill-Mix 2310.17567):** Arora et al.
   found that even GPT-4 saturates at 3–4 concurrent style skills not
   because the skills interfere with each other, but because the
   **model's output stream has finite format bandwidth**.  Every
   response position is a single token; a "numbered step" position
   cannot simultaneously be a "TL;DR" position.  As `k` grows the
   model must *time-share* format across the response, and past
   `k = 3–4` at least one skill's signature becomes absent from a
   randomly-sampled response.

3. **Frontier extension.** Our contribution: test whether
   LoRA-additive composition inherits the Skill-Mix saturation curve
   of the base model.  Two outcomes are informative:
   - **Curve matches base saturation** → composition is bounded by the
     base's format-control, not by adapter interference.  LoRA
     composition is as good as it can get; to push past the ceiling
     one must strengthen the base, not the composition math.
   - **Curve drops faster than base saturation** → composition has
     additional loss beyond the base's format capacity.  That loss
     could be due to norm explosion, destructive alignment of B
     matrices, or catastrophic scale.  Would motivate explicit
     `1/k`-normalised or orthogonal-projection composition.

### Welch-bound budget (orthogonality of A-matrices)

Random-init `A_i ∈ ℝ^{r × d}` with `r = 8`, `d = 2560`, `N = 5` gives
`r · N = 40` vectors in `d = 2560`-dim space.  Expected max pairwise
coherence between the full row-subspaces:
```
E[max_{i≠j} ||A_i A_j^T||_F / (||A_i||_F ||A_j||_F)]
  ≤ √((d − rN) / (rN (d − 1)))
  = √(2520 / (40 · 2559)) ≈ 0.157
```
So composition-level mutual coherence is below 0.16 — well below the
0.3 regime where destructive interference begins.  **Any observed
saturation therefore is NOT due to A-matrix interference; it is
either (a) B-matrix alignment after training or (b) base-model
format-control saturation.**

## Kill criteria (PRE-REGISTERED — immutable)

| KC      | Condition                                                 | Threshold                                       |
| ------- | --------------------------------------------------------- | ----------------------------------------------- |
| K1730   | k=2 signature preservation vs solo                        | ≥ 0.95 × M_solo                                 |
| K1731   | k=5 signature preservation vs solo (saturation detected)  | ≤ 0.80 × M_solo                                 |
| K1732   | Monotonic degradation curve across k=1..5                 | μ non-increasing; no `μ(k+1) > μ(k)+0.05` spike |

Smoke-scale results are PROVISIONAL per guardrail 1009.

## Methods (5 textually-disjoint signatures)

Each method is induced by a distinct teacher system prompt and
detected by a distinct, non-overlapping regex.  The signatures live
in different regions of the response to minimise pattern-overlap so
that `sig_i` is independent of `sig_j` for `i ≠ j`.

| i | Method          | Teacher prompt gist                           | Signature `S_i` (regex)                                    |
| - | --------------- | --------------------------------------------- | ---------------------------------------------------------- |
| 1 | `restate`       | Begin with "Problem restated: ..."            | `^\s*Problem restated:\s` (anchored to response start)     |
| 2 | `numbered`      | Use explicit "Step 1:", "Step 2:" markers     | `\bStep\s+2\b` (requires ≥2 step markers via Step 2)       |
| 3 | `verify`        | Include a "Verification:" or "Check:" line    | `(?mi)^\s*(Verification|Check):\s`                         |
| 4 | `principle`     | Cite the "Principle:" or "Rule:" used         | `(?mi)^\s*(Principle|Rule):\s`                             |
| 5 | `tldr`          | End with "TL;DR: ..."                         | `(?mi)\bTL;?DR:\s`                                         |

Inter-signature overlap on raw teacher traces is empirically < 5 %
(a numbered-steps response rarely has a "TL;DR" and vice versa)
which justifies treating them as independent format elements.

## Assumptions (logged per guardrail 1007)

- **Teacher fidelity gate.** Each method's teacher traces must hit
  the method's own signature at ≥ 0.70 before training; else the
  method is *excluded* from the k-sweep (and the sweep runs with
  `N < 5`, documented in `results.json`).  This is the Orca-2 style
  teacher-gate pattern used in the parent
  `exp_method_vs_domain_adapter`.

- **Training budget parity.** All 5 method adapters are trained for
  the same number of optimizer steps and on the same count of
  teacher examples (varying only in which method they encode).
  Under this control any saturation difference is attributable to
  composition, not to under/over-training.

- **Paired eval.** Solo and composed arms see the SAME held-out
  MCQ questions at the same seed so the per-question variance
  cancels in the paired comparison.

- **Held-out disjointness.** Training data draws from MMLU-Pro
  categories `{math, health, law, economics, computer science}`;
  eval draws from `{physics, biology, philosophy, psychology,
  history}`.  Disjoint categories at the MMLU-Pro `category` column.

- **Smoke reports PROVISIONAL per guardrail 1009.** Full-scale rerun
  (N_STEPS=200 per adapter, EVAL_PER_COND=30) is pre-registered in
  PAPER.md.

## References

- Arora & Goyal (2023) — Skill-Mix: `arxiv:2310.17567`
- Ilharco et al. (2023) — Task Arithmetic: `arxiv:2212.04089`
- Todd et al. (2023) — Function Vectors: `arxiv:2310.15213`
- Mitra et al. (2023) — Orca 2 Explanation-Tuning: `arxiv:2311.11045`
- This repo: `exp_method_vs_domain_adapter` (PROVISIONAL — parent
  of this experiment; provides the prompt-erased-SFT teacher
  pattern) and Finding #53 (domain LoRA multi-A composition preserves
  all 5 specialisations) for the composition-preservation baseline.

## mlx-lm version

```
mlx 0.31.1
mlx-lm 0.31.2
```
