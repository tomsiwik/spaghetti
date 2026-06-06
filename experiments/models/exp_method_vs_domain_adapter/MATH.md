# MATH.md — exp_method_vs_domain_adapter

## Claim (Theorem)

Let `π_0` be the frozen Gemma-4-E4B-it-4bit base and let `π_θ^M = π_0 + Δ_r(θ)`
be an adaptation with a rank-`r` LoRA update (`r = 16`, keys
`{v_proj, o_proj}`, `LORA_SCALE ≤ 8`) trained via knowledge-distilled SFT on a
mixture `D_M = ⋃_{d ∈ D_train} D_d` where each `D_d` contains `(q, r(q))` with
`r(q)` a "decompose-into-subgoals" teacher trace emitted by `π_0` under an
explicit method system prompt.

We claim that the **method** (format + subgoal decomposition) is encoded by a
low-rank subspace of `Δ_r`, and that training over **diverse** domains (≥ 5
MMLU-Pro categories) preserves only the method-direction while averaging out
domain-content directions. Formally:

1. **K1718 (multi-domain → generalises)** — `π_θ^M` evaluated on 5
   *held-out* domains `D_eval ∩ D_train = ∅` beats `π_0` on **≥ 3 / 5**
   held-out categories by accuracy.
2. **K1719 (single-domain → domain-leaks)** — `π_θ^S`, trained with the same
   hyperparameters but only `D_train = {math}` for the same number of
   optimizer steps and the same total token budget, beats `π_0` on
   **≤ 1 / 5** held-out categories — confirming that one-domain training
   encodes the joint (method + math) direction, not method alone.
3. **K1720 (behavioural signature)** — the multi-domain adapter's responses
   on held-out queries exhibit the subgoal decomposition signature
   (≥ 2 distinct "Step N" markers or numbered enumeration) in **≥ 70 %** of
   outputs, and this rate is ≥ 20 pp higher than `π_0`'s rate on the same
   prompts.

## Mechanism (why it works)

1. **Task arithmetic (Ilharco et al., `arxiv:2212.04089`)** established that
   fine-tuned deltas in weight space are approximately additive for
   *single-task* training at full rank. Their proof is not low-rank, and
   does not cover *method* adapters (decomposition style) as distinct from
   *content* adapters.
2. **Function vectors (Todd et al., `arxiv:2310.15213`)** showed that
   high-level behavioural *functions* are encoded in activation space by a
   single vector that generalises across examples. However, they measure in
   activation space, not in weight-space LoRA parameters.
3. **Orca 2 Explanation-Tuning (Mitra et al., `arxiv:2311.11045`)**
   demonstrated that training on teacher rationale traces with **prompt
   erasure** induces the student to emit the same reasoning style without
   the system prompt. Orca 2 used full-rank tuning on a 7 B model.
4. **This experiment's frontier extension**: if (2) and (3) hold at
   rank 16 in weight space, then:
   - the method subspace `S_method ⊂ ℝ^(d × d)` has dimension much lower
     than `d`, because the method is a **format** (a few tokens of
     structure) rather than new knowledge;
   - training over a diverse mixture concentrates gradient mass on
     `S_method` (present in every example) while domain-content directions
     average to ≈ 0 (orthogonal across domains — same regularisation idea
     as in pre-merge LoRA composition, Finding #13);
   - therefore the multi-domain adapter transfers, but the single-domain
     adapter overfits to the joint (method ⊕ domain-content) direction.

### Decomposition of the LoRA delta

Under standard LoRA parameterisation `Δ = BA` with `A ∈ ℝ^{r × d}`,
`B ∈ ℝ^{d × r}`, the gradient for a single `(q, r(q))` example is

```
∂L / ∂(BA) = ∂L / ∂(h W_out) · (h^T · W_in)       [sketch]
           ≈ (method direction e_M) + (domain-content direction e_d(q))
```

With `|D_train|` domains and balanced sampling, the empirical mean of the
gradient is

```
𝔼_q[ ∂L / ∂Δ ]  ≈  e_M  +  (1/|D_train|) Σ_d e_d
```

The second term contracts linearly in `|D_train|` if `{e_d}` are
approximately orthogonal (reasonable for MMLU-Pro categories — Finding
#3). At `|D_train| = 5`, the domain-contamination term is ≈ 0.45 the
method norm (by Welch bound, `cos_max ≤ √((n-r)/(r(n-1))) ≈ 0.45` for
`n = 5`, `r = 1`). Rank 16 gives ample room to store `e_M` even if the
true method subspace has dim ≥ 2.

At `|D_train| = 1`, the domain-contamination term is ≥ the method term
and the adapter encodes `e_M + e_math` — which hurts transfer to any
`d' ≠ math` where `⟨e_d', e_math⟩ ≈ 0`.

## Assumptions (logged per guardrail 1007)

- **Training budget parity.** Multi-domain and single-domain adapters are
  trained for the **same number of optimizer steps** `N_STEPS` and the
  **same total sample count** — the difference is only the domain
  composition of the sample set. Under this control, any transfer gap is
  attributable to domain diversity, not budget.
- **Teacher fidelity.** Teacher responses used for training are generated
  by the same `π_0` under the method system prompt. We do **not** validate
  teacher correctness; we validate that the **method signature** is
  present in ≥ 70 % of teacher traces before training begins.
- **Held-out disjointness.** Training categories (`math, computer science,
  health, law, economics`) and held-out categories (`physics, biology,
  philosophy, psychology, history`) are disjoint at the MMLU-Pro
  `category` level. MMLU-Pro does not share questions across categories,
  so question-level leakage is zero.
- **Method signature definition.** "Subgoal decomposition present" means
  the response (after stripping any thinking channel) contains ≥ 2 of:
  `Step \d`, a numbered enumeration `^\d+\.`, a bullet list of length
  ≥ 2, or explicit connectives `First`, `Second`, `Next`, `Finally`. This
  is registered in code as `count_subgoal_markers(resp) ≥ 2`.
- **Domain mapping.** MMLU-Pro does not have a `code` category; we use
  `computer science`. It does not have a `medical` or `finance` category;
  we use `health` and `economics`. The title's "math, code, medical,
  legal, finance" was aspirational — the implementation uses the
  MMLU-Pro-canonical names and this is recorded here as the
  authoritative list.
- **LoRA scale bound.** `LORA_SCALE = 4.0` (safe, per antipattern
  `mem-antipattern-003` bounding scale ≤ 8). Held constant across both
  adapters.

## Kill criteria (pre-registered — locked at commit time)

| ID   | Criterion                                                                                                  | Pass threshold |
| ---- | ---------------------------------------------------------------------------------------------------------- | -------------- |
| K1718 | Multi-domain adapter accuracy `>` base accuracy on each held-out domain                                    | `≥ 3 / 5`      |
| K1719 | Single-domain (math-only) adapter accuracy `>` base accuracy on each held-out domain                       | `≤ 1 / 5`      |
| K1720 | Fraction of multi-domain adapter responses containing ≥ 2 subgoal markers on held-out prompts              | `≥ 0.70`       |
|       | AND signature rate for multi-domain adapter minus signature rate for base                                  | `≥ +20 pp`     |

All three must PASS for the hypothesis to be `supported`. Any single FAIL
kills the claim at the pre-registered verdict level.

## Predictions (quantitative)

Under the mechanism above, we predict, at full scale
(`N_STEPS = 300`, `eval_per_cat = 15`):

| Quantity                                         | Base (π_0)   | Multi-domain (π_θ^M) | Single-domain math (π_θ^S) |
| ------------------------------------------------ | ------------ | -------------------- | -------------------------- |
| Mean held-out accuracy (5 cats, 15 q each)       | 48–55 %      | **53–65 %**          | 46–55 %                    |
| # held-out cats with acc > base                  | 0 (self)     | **≥ 3**              | ≤ 1                        |
| Subgoal-signature rate on held-out responses     | 20–40 %      | **≥ 70 %**           | 40–60 %                    |

In smoke mode (`N_STEPS = 40`, `eval_per_cat = 3`) the absolute numbers
are statistically under-powered; the verdict is `provisional` regardless
and a full rerun is required to assign `supported` / `killed`.

## Failure modes this design rules out

- **Composition bug (`mem-antipattern-001`)** — not applicable: there is
  no adapter composition in this experiment, only individual adapters
  compared to base.
- **Tautological routing (`mem-antipattern-002`)** — not applicable: no
  routing. Each eval is a direct single-adapter-vs-base comparison.
- **Unsafe adapter scale (`mem-antipattern-003`)** — scale fixed at 4.0
  (≤ 8 bound). Logged in `results.json`.
- **Thinking-mode truncation (`mem-antipattern-008`)** — `enable_thinking
  = True` everywhere; `max_tokens = 2048` at generation; thinking regex
  (`strip_thinking`) is the copy from `exp_score_kl_constrained_mcq`
  which matched Gemma-4's native format.
- **KC swap after failure** — KCs locked above; any post-hoc relaxation
  invalidates the run.
- **Hardcoded pass (`mem-antipattern-...`)** — `all_pass` is derived
  from the three booleans; `pass` per-KC is a measured comparison
  against the pre-registered thresholds.
- **Proxy model substitution** — the target is Gemma-4-E4B-it-4bit and
  all three arms use the same weights. No Qwen proxy.
- **Smoke-as-full** — `SMOKE_TEST=1` by default; `verdict = PROVISIONAL`
  and `is_smoke = True` are written to `results.json`; the
  `experiment complete --status provisional` path applies on smoke.
- **Tautological KC** — K1718 and K1719 depend on disjoint training
  sets and are measured on held-out categories. K1720 is a behavioural
  claim not a metric tautology.

## References

- Ilharco et al., *Editing Models with Task Arithmetic*, `arxiv:2212.04089`.
- Todd et al., *Function Vectors in Large Language Models*, `arxiv:2310.15213`.
- Mitra et al., *Orca 2: Teaching Small Language Models How to Reason*,
  `arxiv:2311.11045`.
- Internal: Finding #3 (orthogonality), #13 (pre-merge composition), and
  `mem-antipattern-{001-008}` (memory-injected guardrails).
- Code: `mlx-lm ≥ 0.27`; `mlx` unified memory; phased execution pattern
  per `/mlx-dev`.
