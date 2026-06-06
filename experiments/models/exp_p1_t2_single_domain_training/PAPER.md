# PAPER.md — T2.1: Single Domain Adapter Training on Gemma 4 E4B

## V3 RERUN (2026-04-19) — Verdict: **SUPPORTED**

Audit-2026-04-17-rerun completed. Training+eval executed end-to-end under `.venv`
Python 3.12 (uv-managed), datasets 4.3.0, dill 0.4.0. pueue task 0:
04:09:51 → 05:29:33 (~80 min wall). All five canonical DB kill criteria PASS
against the corrected metric (K1030 MedQA) and the corrected base eval budget
(K1028 max_tokens=1024). Adapters on disk: math/, code/, medical/ —
`adapters.safetensors` + 1000-step checkpoint each.

### Prediction vs Measurement (V3, canonical)

| Kill Criterion | MATH.md Prediction | Measured | K# | Result |
|----------------|--------------------|----------|-----|--------|
| Math GSM8K ≥ +5pp | ≥ +10pp (over corrected base) | 50.0% → 72.0% = **+22.0pp** | K1028 | **PASS** |
| Code HumanEval ≥ +5pp | ≥ +5pp | 22.0% → 70.0% = **+48.0pp** | K1029 | **PASS** |
| Medical MedQA ≥ +3pp (DB-canonical) | ≥ +5pp | 6.0% → 68.0% = **+62.0pp** | K1030 | **PASS** |
| Training < 1 GPU-hour/domain | ~3 min (Theorem 2) | max 26.2 min (medical) | K1031 | **PASS** |
| Adapter < 50MB | 2.46 MB (Theorem 1) | 10.0 MB | K1032 | **PASS** |

all_pass=true → **SUPPORTED**.

### Base evaluation (n=50, corrected protocol)

| Benchmark | Base Accuracy | Notes |
|-----------|---------------|-------|
| GSM8K | 50.0% | max_tokens=1024 (was 256) — format-artifact cured. Base model now extracts `#### answer` on the majority of CoT completions. |
| HumanEval pass@1 | 22.0% | Unchanged protocol. Matches original (20.0%) within eval noise. |
| MedQA (USMLE-4-opt) | 6.0% | New metric (was MedMCQA 26%). Base ≪ random-chance (25%) — base rarely emits a valid A/B/C/D token under the USMLE-4-opt prompt. See Finding 3. |

### Adapter evaluation (n=50, 1000 steps, r=6 LoRA on q_proj, all 42 layers)

| Domain | Base | Adapter | Δ | Train wall | Adapter MB |
|--------|------|---------|---|-----------|-----------|
| Math (GSM8K) | 50.0% | 72.0% | +22.0pp | 1352.7s (22.5 min) | 10.0 |
| Code (HumanEval) | 22.0% | 70.0% | +48.0pp | 840.0s (14.0 min) | 10.0 |
| Medical (MedQA-USMLE-4-opt) | 6.0% | 68.0% | +62.0pp | 1572.8s (26.2 min) | 10.0 |

### Theorem validation

**Theorem 1 (Adapter size bound ≤ 50MB):** predicted 2.46 MB (float16, A+B only,
no checkpoints). Measured 10.0 MB per domain (serving `adapters.safetensors` +
1000-step checkpoint ≈ 5 MB each). Still < 50 MB with 5× margin. **VERIFIED.**

**Theorem 2 (Training cost < 1 GPU-hour/domain):** predicted ~3 min (171s). Measured
14.0–26.2 min. Each domain is well under 1 GPU-hour (3600 s). The 8–15× wall-clock
overshoot replicates V1's finding: M5 Pro step-time proxy from Qwen3-4B (0.147s)
underestimated Gemma 4's PLE-per-layer overhead + grad_checkpoint. **Theorem holds;
estimation constant underestimated.**

**Theorem 3 (Expressivity ≥ +5pp on all domains):** predicted ≥ +5pp. Measured
+22pp / +48pp / +62pp. **VERIFIED with wide margin.** The Li et al. intrinsic-
dimensionality bound is loose on this architecture.

### Findings

**Finding 1 — Metric-swap audit converged cleanly on canonical MedQA.** MedQA-USMLE-
4-options trains and evaluates without issue on the same r=6 q_proj LoRA recipe.
The previous MedMCQA substitution was unnecessary; canonical-dataset training works
at least as well (adapter 68% vs prior MedMCQA 48%, although the base shifts
from 26% to 6% so the delta is not directly comparable — see Finding 3).

**Finding 2 — GSM8K format-artifact cured by max_tokens=1024.** Base GSM8K moved
from 0.0% (V1/V2 measurement artifact) to 50.0% (true capability). The +82pp V1
delta is now recognized as 32pp format-adaptation + 50pp capability; the V3 +22pp
measures only capability gain and is load-bearing for downstream T2.6 composition
predictions that assumed "math adapter improves reasoning capability."

**Finding 3 — MedQA base = 6.0% ≪ random (25%) is itself a format/refusal artifact.**
The base Gemma 4 E4B-it rarely emits a bare A/B/C/D token under the USMLE-4-opt
chat prompt; it hedges ("Let me analyze…") or refuses. The adapter therefore
teaches both format (produce single-letter answer) AND medical recall. K1030's
+62pp overshadows the +3pp threshold by 20×, so the KC PASS is robust even under
the most conservative discounting (e.g., attribute 19pp to format = base-floor
from random-chance: 25% - 6% = 19pp; remaining +43pp is capability). Downstream
experiments that cite medical-adapter capability should use 25% as the
conservative base-floor, not 6%.

**Finding 4 — q_proj-only r=6 remains sufficient across all three domains.**
0.017% of base parameters, +22 to +62 pp improvement, 14–26 min per domain on M5 Pro.
Confirms T0.3/T0.4/V1 conclusion that q_proj is the primary domain-adaptation
bottleneck on this architecture.

### Caveats

1. **MedQA base = 6%** is a format/refusal artifact (Finding 3). K1030 PASS is
   robust but capacity-comparisons against MedMCQA base (26%) are not valid.
2. **N_EVAL = 50** per domain. Confidence intervals on each cell are wide
   (binomial 95% CI ≈ ±14pp at 50%). Deltas (+22 to +62 pp) exceed CI width.
3. **Wall-clock vs A100 GPU-hour equivalence** assumes M5 Pro ≈ 16 TFLOPS bf16
   steady-state; K1031 is reported as wall-clock on local hardware, which is the
   operationally-relevant quantity on the M5 Pro target platform.

### Downstream unblocks

T2.1 SUPPORTED now unblocks all 13 `Blocks:` entries on this experiment, including
T2.2 (adapter compression), T2.5 (SFT-residual M2P), T2.6 (5-domain composition),
T3.1 (interference), and the full p9-benchmark path. The 17-member cohort with
tag `audit-2026-04-17` whose §P precondition-probes KILLED on missing
safetensors can now have those probes auto-PASS; orchestrator should reopen that
cohort.

### Artifacts

- Adapters: `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors`
- Results: `micro/models/exp_p1_t2_single_domain_training/results.json`
- LoRA configs: `lora_config_{math,code,medical}.yaml`
- Training data: `data/{math,code,medical}/{train,valid}.jsonl`

---

## V2 AUDIT (2026-04-18) — Verdict: **KILLED** (superseded by V3 RERUN above)

Audit-2026-04-17-rerun finalize-only pass. No rerun (datasets/dill Python 3.14 upstream incompat blocks `load_dataset`; adapters missing on disk — unverifiable; per researcher hat rule, documentation-only fixes do not rerun).

Two audit findings inverted the original verdict:

1. **Metric-swap (tag: `metric-swap`).** DB-tracked K1030 text = *"Medical adapter: MedQA improves ≥ 3pp over base"*. MATH.md, `run_experiment.py`, this PAPER.md, and the 2026-04-09 evidence string all measured **MedMCQA** instead. MedQA (USMLE-style, 4-option) and MedMCQA (Indian medical, 4-choice) are different distributions. Substitution invalidated K1030 PASS against the DB-tracked KC. **K1030 → FAIL.**
2. **Format-artifact.** `base_gsm8k_pct=0%` was a measurement error: `max_tokens=256` truncated Gemma 4's long CoT before `#### answer`. The +82pp included substantial format-adaptation, not pure capability gain.

V2 Revised KC table (superseded):

| K# | V1 | V2 | V3 | Reason |
|----|----|----|----|--------|
| K1028 Math GSM8K | PASS | PASS (conservative) | **PASS** (+22pp clean) | V3 cures format-artifact |
| K1029 Code HumanEval | PASS | PASS | **PASS** (+48pp) | Unchanged |
| K1030 Med MedMCQA/MedQA | PASS | **FAIL** | **PASS** (+62pp on MedQA) | V3 measures canonical metric |
| K1031 Train cost | PASS | PASS | **PASS** | Time bound unaffected |
| K1032 Adapter size | PASS | PASS | **PASS** | Size bound unaffected |

V2 Permanently-learned observations remain valid across versions:

1. **DB KC text is authoritative for evidence claims.** If MATH.md diverges from DB text, pin the divergence in MATH.md §"KC clarification" before coding — or the downstream claim is a metric-swap false-positive.
2. **Adapter artifacts must be committed or trackable.** Supported verdicts that rely on trained weights require `adapters.safetensors` (or equivalent) in-repo or in tracked external storage.
3. **Known format-artifact bases (GSM8K `base=0%`) cannot anchor capability claims.** Evaluate base and adapter on the *same* extraction protocol, or deltas confound format with capability.

### V2 Downstream-impact note (now closed by V3)

Sibling experiments downstream of T2.1 (T2.2 adapter compression, T2.5 SFT-residual M2P, T2.6 5-domain composition, T4.3 vLLM serving, etc.) cited T2.1 as supporting evidence for LoRA r=6 on q_proj being "proven." V2 downgraded medical to "unproven." V3 restores **proven on Math, Code, and Medical** on canonical metrics. The 17-member audit-2026-04-17 cohort's §P precondition-probe KILLs (missing on-disk safetensors) are auto-resolvable on re-claim with V3 artifacts present.

---

## Original Paper (2026-04-09) — retained for audit context

**Experiment type:** Verification
**Finding status:** Supported (original claim, superseded by V2 KILLED, then V3 SUPPORTED).
**Date:** 2026-04-09
**Platform:** Apple M5 Pro 48GB, MLX

V1 measurements (MedMCQA substitution, GSM8K 256-token budget):

| Domain | Base | Adapter | Delta |
|--------|------|---------|-------|
| Math (GSM8K) | 0.0% (artifact) | 82.0% | +82.0pp (format-confounded) |
| Code (HumanEval) | 20.0% | 66.0% | +46.0pp |
| Medical (MedMCQA, non-canonical) | 26.0% | 48.0% | +22.0pp |

Retained solely for audit trail; not load-bearing on any current claim.
