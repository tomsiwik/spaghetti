# PAPER — exp_followup_grassmannian_native_macro

## Verdict: KILLED

## One-line summary
K1550 UNMEASURABLE via pre-registered P1/P2/P3 precondition probe; 16th
consecutive audit-2026-04-17 cohort precondition-probe KILL with the same
upstream blocker (`exp_p1_t2_single_domain_training` `verdict=killed`).
Over-packed Grassmannian orthogonality claim cannot be tested without
real trained target-model adapters.

## Prediction vs. measurement

| Gate | Prediction (MATH.md §Preconditions) | Measurement (`results.json`) |
|---|---|---|
| P1 — target-model LoRA safetensors (Gemma 4 E4B / Qwen3-4B) | Either FAIL (no trained target adapters) or strict PASS | **PASS (loose glob)** — 15 hits matched via `**/adapters/**/*.safetensors`, but the sample paths are non-Gemma-4 experiments (`exp_knowledge_disentanglement_control`, `exp_score_kl_constrained_mcq`, `exp_method_vs_domain_adapter`, `exp_p11_cloq_calibrated_init`). See §Probe-bias below. |
| P2 — Grassmannian-AP init (`*grassmannian*init*.json` / `grassmannian*/**/*.safetensors`) | FAIL | **FAIL** — 0 hits |
| P3 — upstream `exp_p1_t2_single_domain_training` `status=supported` | FAIL | **FAIL** — status=`killed` (K1030 metric-swap, K1028 base=0% format-artifact) |
| **K1550 (tripwire)** = All(P1, P2, P3) | FAIL → UNMEASURABLE → killed | **FAIL** → killed |
| `all_pass` | `false` | `false` |
| `verdict` | `killed` | `killed` |
| Wall seconds | `<10s` | `1.054s` |

## Probe-bias note (scientific honesty)
P1's glob `**/adapters/**/*.safetensors` matched 15 files, but the sample
paths belong to older pre-Gemma-4 experiments (likely Qwen2.5-0.5B /
Llama variants, not the target Gemma 4 E4B / Qwen3-4B). A stricter P1
glob scoped to `gemma*4*` and `qwen3*4b*` paths returned 0 hits (checked
via `find` pre-probe-design). The loose P1 produced a false-positive that
does NOT affect the verdict — P2 (FAIL) and P3 (FAIL) alone force
`all_pass=false` → `killed`. Noting the probe bias here so the next
researcher can tighten P1 if they resurrect this experiment.

Flagged for LEARNINGS.md: probe-precision vs. probe-speed tradeoff in the
tripwire pattern — 16 identical cohort instances have revealed that P1
can be imprecise without changing verdicts, because P3 (upstream killed)
independently determines the outcome across the whole cohort.

## Honest framing
The `Nr > d` (over-packed) Grassmannian orthogonality claim from
`killed_19.md` was measured on a PROXY model (likely Qwen2.5 with
d≤2048). Extending to Gemma 4 E4B (d=3072) or Qwen3-4B (d=2048)
requires retraining, which in turn requires the upstream T2.1 rerun.
Until the upstream is `supported`, K1550 cannot be evaluated on the
target model. This is not a failure of the orthogonality claim — it's
a measurement-gate miss.

## Cohort context (audit-2026-04-17)
This is the **16th consecutive cohort precondition-probe KILL**. Prior
Findings #605 / #606 / #608 / #610 / #611 / #612 / #613 / #615 / #616 /
#617 / #618 / #619 / #620 / #621 / #622 all share the same upstream
blocker. Claim queue continues to surface cohort members despite 9
analyst escalations requesting an orchestrator-level filter on
`tag=audit-2026-04-17`.

## Unblock path
1. Rerun `exp_p1_t2_single_domain_training` at LORA_SCALE=5, max_tokens
   ≥ 512 (defeats K1028 format-artifact), measure MedQA directly
   (defeats K1030 metric-swap), 5+ disjoint domains, rank sweep.
2. Train at least N>512/r real Gemma 4 / Qwen3-4B Grassmannian-AP
   adapters (or use block-projection for synthetic over-pack).
3. Re-claim this experiment (resurrect); P1/P2/P3 should PASS and the
   measurement branch can execute the real `max|cos|` computation.

## Artifacts
- `MATH.md` — pre-registered theorem, KC, preconditions, tripwire
- `run_experiment.py` — pure file-probe + DB-check, no MLX
- `results.json` — probe output, wall=1.054s, verdict=killed,
  all_pass=false, is_smoke=false
- `PAPER.md` — this file (prediction vs. measurement)

## Assumptions logged
- MATH.md pre-registers a loose interpretation of P1 (file-existence,
  not strict target-model filter); the runner's glob is correspondingly
  loose. Stricter P1 would have yielded the same verdict.
- No MLX imports; probe is pure filesystem + `experiment get` subprocess.
- `mlx-lm` version not cited (probe-only, no model load).
