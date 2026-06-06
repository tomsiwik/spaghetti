# MATH.md — exp_followup_grassmannian_native_macro

## Theorem (pre-registration, KC #1550)
Let `A_i ∈ R^{r×d}` be the row-spans of `N` Grassmannian-AP-initialized LoRA
adapters trained on Gemma 4 E4B (d=3072) or Qwen3-4B (d=2048). In the
**over-packed regime** `Nr > d`, the worst-case pairwise principal cosine

```
max_{i≠j} |cos(A_i, A_j)|  ≤  100 · sqrt(r/d)
```

should hold if the Grassmannian-AP initialization survives training. If
this bound fails on real trained adapters, the orthogonality claim from
the under-packed regime (`Nr ≤ d`, prior result in `killed_19.md`
`exp_bitnet_grassmannian_init`) does not generalize to the over-packed
regime — which is the regime Pierre actually targets at N=25, r=6.

## Cited prior results
- Finding #310-sibling under-packed Grassmannian orthogonality (cited in
  `killed_19.md`).
- Welch bound (`sqrt((N-r)/(r(N-1)))` lower bound on max |cos|) —
  asymptotically tight for `N` frames in `R^r`.
- Finding #586 LORA_SCALE safety bound (real adapters trained at safe
  scale only).

## Preconditions (measurement gate — tripwire)
The measurement of KC #1550 requires:

- **P1** — Real trained Gemma 4 E4B **or** Qwen3-4B LoRA adapter
  safetensors at `Nr > d` (e.g., for r=6 and Gemma 4 d=3072, `N > 512`;
  for Qwen3-4B d=2048, `N > 341`). Even a *smaller* over-packed set
  (e.g., N=20, r=6 vs. Qwen3-4B d=2048 requires `N > 341` — infeasible on
  M5 Pro; realistically, artificial over-pack via block-projection).
- **P2** — Those adapters must be initialized via Grassmannian AP
  (P_avg packing) **before** training — random-Gaussian adapters do not
  test the claim.
- **P3** — The upstream `exp_p1_t2_single_domain_training` that trains
  the adapters must be `status=supported` with real safetensors on disk,
  not `status=killed`.

If **any** of P1/P2/P3 FAIL, K1550 is **UNMEASURABLE** and the
experiment completes as `killed` per the audit-2026-04-17 cohort
standing rule (probe-KILL pattern, sibling cohort ≥15 prior instances).

## Kill criterion (tripwire restatement)
- **K1550 (tripwire)** — `All(P1, P2, P3) == True`. Fails if any
  precondition probe fails; no measurement branch is attempted because
  the upstream artifacts do not exist.

## Assumptions
- `mlx-lm` version ≥ 0.31 for Gemma 4 support (irrelevant under
  tripwire; no model load performed).
- Probe is a pure file-existence + upstream DB check. No MLX model
  load, no training, no cosine computation. Wall < 10s expected.

## Predicted outcome
Given the 15+ prior cohort instances with the same upstream blocker
(Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618/#619/
#620/#621/#622), the prediction is **probe-KILL** with `all_pass=false`
and `verdict=killed`. A researcher can resurrect this experiment only
after the upstream `exp_p1_t2_single_domain_training` rerun is
`status=supported` with real trained safetensors on disk.

## Failure-mode table
| Failure mode | Prevented by |
|---|---|
| Tautological orthogonality claim (N=1 or `Nr ≤ d`) | Preregistering `Nr > d` regime in KC |
| Random-Gaussian baseline slipping in as "Grassmannian" | P2 requires AP-init before training |
| Re-using proxy model when target is Gemma 4/Qwen3-4B | P1 requires matching target model |
| Silently upgrading to supported despite missing artifacts | Tripwire semantics force `killed` when P* FAIL |
