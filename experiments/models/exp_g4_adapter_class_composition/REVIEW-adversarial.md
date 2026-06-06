# REVIEW-adversarial — exp_g4_adapter_class_composition

**Verdict: PROVISIONAL** (proxy mechanism PASS; DB-title behavioral target unmeasured)

## Adversarial checklist

| Check | Result | Note |
|---|---|---|
| (a) results.verdict vs DB status | OK | results=SUPPORTED, claim=supported (but see (t)) |
| (b) all_pass vs claim | OK | all_pass=true |
| (c) PAPER verdict line | CAVEATED | PAPER says "SUPPORTED (measurement-proxy scope; see §Limitations)" — explicit scope qualifier |
| (d) is_smoke vs full-run claim | OK | is_smoke=false; n=100 measurements is the full proxy design |
| (e) MATH.md KC drift | OK | All MATH/results files untracked (first commit); KCs in code match MATH.md K1 (≥0.95) and K2 (gap>1e-4) verbatim |
| (f) Tautology sniff | PARTIAL | dev_M = 2/3 by analytic construction (uniform 1/N gates → exactly \|1−1/N\|). dev_L = 0 by definition. Only dev_D is a non-trivial measurement. PAPER §Detail discloses this honestly. K2 is half-tautological but the D-side is real signal. |
| (g) K-ID code↔math match | OK | K1=success_rate, K2=class ordering — both match |
| (h) Composition math bug | OK | `sum_delta = deltas[0] + deltas[1] + deltas[2]` is correct LoRA additive composition; no add_weighted_adapter / per-key sum bug |
| (i) Unsafe LORA_SCALE | OK | scale=6.0 (read from adapter_config.json of source experiment) |
| (j) Per-sample routing | n/a | No routing in this experiment |
| (k) shutil.copy of sibling adapter | OK | None |
| (l) Hardcoded `pass: True` | OK | None |
| (m) Model in MATH = model loaded | OK | mlx-community/gemma-4-e4b-it-4bit in both |
| (m2) Skill invocation evidence | LIGHT | Code uses `mx.eval`, `mx.clear_cache`, `mx.dequantize`, `mlx_lm.load` — all idiomatic MLX. PAPER does not name `/mlx-dev` but the code shape suggests MLX awareness. Not blocking. |
| (n) Thinking-channel truncation | n/a | No generation eval |
| (o) Headline n | OK | n=100 (10 layers × 10 probes) |
| (p) Synthetic padding | n/a | All 3 adapters are real trained LoRAs |
| (q) Cited baseline drift | n/a | No external baseline cited |
| (r) Prediction-vs-measurement table | OK | PAPER §"Prediction vs measurement" present |
| **(t) Target-gated kill (F#666)** | **PROVISIONAL** | K2 is labeled "target" but measures a *geometric proxy* (composition deviation), NOT a behavioral target metric (task accuracy, behavioral quality, oracle-gap). The DB title's behavioral target — *3pp MMLU-Pro margin at N=5* — is `not_measured` (no DoRA/MoLoRA training, no MMLU eval). Per F#666, "structural-KC PASS with target-KC `not_measured` → PROVISIONAL". |
| (u) Scope-changing fixes | DISCLOSED, NOT SILENT | Researcher reframed scope from "MMLU 3pp at N=5" to "composition-geometry proxy at N=3" *upfront* (MATH.md §Scope, PAPER.md §Scope reframe). This is a deliberate pre-design scope choice driven by feasibility (no MLX DoRA/MoLoRA implementations, single-iteration budget). Not a silent mid-run swap. Not the antipattern. But the gap between DB title and what was measured is what triggers (t). |

## Key issue: proxy vs behavioral target (F#666)

The DB title asks: *does class A beat class B by ≥3pp on MMLU-Pro?*
This experiment measured: *does composition deviation order as L < D < M on Gemma 4 q_proj geometry?*

Both are valid questions. The geometric ordering result is real and useful — it
extends F#82's mechanism to Gemma 4 E4B 4-bit at d=2560, r=6. But a 9% geometric
deviation at q_proj does not bound MMLU-Pro task accuracy by F#82 itself; F#82
correlated geometric class with task quality at micro scale, not at Gemma 4 scale.

So: the *mechanism* is supported at new scale (proxy PASS). The *behavioral
prediction* (3pp gap) is unmeasured. Marking the DB row `supported` would
overclaim the behavioral result. Marking it `killed` would falsely reject a real
finding. PROVISIONAL is the honest verdict.

## Assumptions

- Treating the DB title's "≥3pp MMLU-Pro at N=5" as the *behavioral* target
  (per F#666 examples: task accuracy / behavioral quality / oracle-gap).
- Treating dev_D / dev_M as proxy measurements (geometric, not behavioral).
- The pseudo-DoRA design (`m_d = ||W_0||_c` frozen at init) is a conservative
  lower bound on trained-DoRA deviation — the researcher's note in §Assumptions
  is mathematically defensible.

## Action

1. Status → `provisional` (two-step workaround per reviewer hat instructions; the
   `complete` CLI rejects `provisional`).
2. File follow-up macro experiment `exp_g4_adapter_class_composition_full` with
   inherited target KCs (MMLU-Pro 3pp at N=5) plus a new caveat KC: report
   trained-DoRA `m_d` drift to verify the lower-bound assumption.
3. Emit `review.proceed` prefixed `PROVISIONAL:`.

## Non-blocking notes

- (f) tautology of dev_M=2/3 is real but disclosed; future composition-class
  experiments should include a non-analytic class-B KC (e.g. trained MoLoRA with
  learned gates) to make the comparison non-trivial.
- (m2) PAPER could explicitly note `/mlx-dev` invocation; not blocking because
  the code is idiomatic.
