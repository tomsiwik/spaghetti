# PAPER — exp_spark_base_safe_harbor

**Hypothesis:** Off-domain LoRA interference is concentrated on rare "conflict tokens"
where the composed top-k next-token SET disagrees with the frozen base top-k set
(discrete Jaccard overlap). A zero-parameter per-token gate — emit the base token when
`Jaccard(top-k_composed, top-k_base) < tau`, else the composed token — should recover
≥60% of the off-domain HumanEval drop while retaining ≥80% of the on-domain GSM8K lift.

**Setup (REAL, no mocks):** frozen `mlx-community/gemma-4-e4b-it-4bit` (4-bit) + F#627
solo **math** LoRA (`data/adapters/math/adapters.safetensors`, `self_attn.q_proj`, rank 6,
scale 6.0 ≤ 8). 42 `ComposedLoRALinear` wrappers via `setattr` (F#831-safe). Greedy,
`enable_thinking=True`, top-k = 8 for the Jaccard set. n = 40 HumanEval / 40 GSM8K.
mlx-lm 0.31.2. Wall time 14,151 s (~3.9 h). `results.json` `is_smoke: false`.

---

## 1. Base vs FIXED (always-composed) — premise check

| Policy | HumanEval pass@1 | GSM8K exact-match |
|---|---|---|
| base (no adapter) | 0.450 (18/40) | 0.725 (29/40) |
| fixed (math adapter always on) | 0.500 (20/40) | 0.650 (26/40) |
| **delta (fixed − base)** | **+5.0 pp** | **−7.5 pp** |

`drop_fixed = base − fixed = −5.0 pp` (HumanEval **rose** under the adapter).
`lift_fixed = fixed − base = −7.5 pp` (GSM8K **fell** under the adapter).

**The F#627 premise (+22 pp GSM8K / −12 pp HumanEval) did NOT reproduce on this
n=40 / thinking-mode / greedy harness.** At this scale the math adapter mildly *helped*
code and mildly *hurt* math — the opposite sign of the interference the gate was designed
to repair. With no interference drop to recover (`drop_fixed ≤ 0`) and no positive lift to
retain (`lift_fixed ≤ 0`), both `interference_reduction` and `retention` are **undefined**
(the script returns `None` when the denominator is ≤ 0).

## 2. Tau sweep (gated policy) — full results

| tau | HumanEval | GSM8K | conflict_rate (HE / GSM) | interference_reduction | retention |
|---|---|---|---|---|---|
| 0.20 | 0.500 | **0.850** | 0.017 / 0.063 | undefined | undefined |
| 0.35 | 0.525 | 0.775 | 0.154 / 0.314 | undefined | undefined |
| 0.50 | 0.525 | 0.775 | 0.356 / 0.540 | undefined | undefined |
| 0.65 | 0.475 | 0.775 | 0.637 / 0.794 | undefined | undefined |
| 0.80 | 0.425 | 0.700 | 0.921 / 0.970 | undefined | undefined |

Reference rows: base HE 0.450 / GSM 0.725; fixed HE 0.500 / GSM 0.650.
Selected tau = 0.20 (selection rule: max `interference_reduction` s.t. `retention ≥ 0.80`;
no row feasible since both metrics are undefined → falls back to first row).

## 3. Kill-criterion evaluation (pre-registered, kill id 2292)

| KC | Metric | Target | Measured @ tau=0.20 | Result |
|---|---|---|---|---|
| K1 | HumanEval pass@1 — interference_reduction | ≥ 0.60 | undefined (`drop_fixed = −5.0 pp ≤ 0`) | **FAIL** |
| K2 | GSM8K exact-match — retention | ≥ 0.80 | undefined (`lift_fixed = −7.5 pp ≤ 0`) | **FAIL** |

`all_pass = false`. Both target-metric KCs fail because the premise they presuppose
(a measurable off-domain drop and a measurable on-domain lift from the FIXED adapter)
did not materialize. Per F#666 these are genuine target-metric KCs, so their joint
failure is a valid KILL of kill id 2292 as specified.

## 4. Notable sub-observation (does NOT rescue the verdict)

The discrete top-k base-fallback gate at **tau ≤ 0.5 raised GSM8K well above the FIXED
policy while holding HumanEval at ~0.50**:

- tau=0.20: GSM8K **0.850** vs fixed 0.650 (+20 pp) and base 0.725 (+12.5 pp), HE 0.500 = fixed.
- tau=0.35/0.50: GSM8K 0.775 (+12.5 pp over fixed), HE 0.525 (≥ both base and fixed).

So the gate behaved as a *net-beneficial* per-token policy here — it claws back the
on-domain damage the always-on adapter caused *and* keeps the small code gain, at very low
conflict rates (1.7% of HE steps, 6.3% of GSM steps at tau=0.20). The mechanism direction
(interference is localized on low-Jaccard steps; reverting them to base helps) is **directionally
consistent** with the hypothesis. But the pre-registered kill criterion was framed around
recovering a *drop* and retaining a *lift* that did not occur, so this improvement cannot
satisfy K1/K2 and is recorded as an observation, not a pass. Re-deriving a target around
"net GSM8K+HE improvement over FIXED" would require a v2 experiment (KC-swap-after-failure
is forbidden, GUIDE §3).

## 5. Verdict

**KILLED.** At the selected tau (0.20) both pre-registered target KCs FAIL: `interference_reduction`
(K1, ≥0.60) and `retention` (K2, ≥0.80) are undefined because the FIXED math adapter produced
neither the off-domain HumanEval drop nor the on-domain GSM8K lift the hypothesis presupposed
(measured drop_fixed = −5.0 pp, lift_fixed = −7.5 pp — both wrong sign). The discrete-Jaccard
base-fallback gate is real and well-behaved (tau≤0.5 net-improves GSM8K over FIXED at ~2% conflict
rate), but the experiment as pre-registered is killed on its own kill id 2292.

Real run, `is_smoke: false`. No premise rescue, no KC modification.
