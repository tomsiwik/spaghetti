# MATH.md — exp_model_quantization_composition_stability (KILLED_PREEMPTIVE)

## 1. Hypothesis (as declared by target)
W4A16-quantized Gemma 4 E4B base under N=5 LoRA composition matches
the bf16 reference within 1.5 pp MMLU-Pro, with per-domain contribution
ranking preserved across the two precisions.

KC (pre-registered, locked by claim):
- K1713 — MMLU-Pro under N=5 composition: W4A16 within 1.5 pp of bf16
  reference.
- K1714 — Per-domain behavioral delta preserved: each adapter's
  contribution ranking matches bf16.

Target declared parent: `exp_p1_t3_n25_composition` (single-parent,
current DB `Status: killed`; K1059 PASS, **K1060 FAIL** [0/5 adapter
.safetensors on disk], **K1061 FAIL** [MMLU regression under
composition]).

## 2. Preempt theorem (defense-in-depth, 5-of-5 independent blocks)

**Theorem (preempt).** The empirical run is **impossible** or
**guaranteed-to-fail** iff at least **one** of the five blocks holds.
We show **four** hold independently (T1 ∧ T2 ∧ T3 ∧ T5-K-single) plus
**one** reinforces (T4). Any single block suffices. T5 is the **T5-K
single-parent-KILLED variant**, 3rd instance in the drain (after iter
36 `exp_model_loader_portability` and iter 45
`exp_model_multi_seed_room_model`).

### T1 — Artifact-absence block
Required artifacts for a legitimate W4A16 × N=5 composition × MMLU-Pro
comparison against a bf16 reference on Gemma 4 E4B:

1. **5 Gemma 4 E4B LoRA adapter `.safetensors` on disk** (the five
   N=5 composition inputs: math / code / medical / legal / finance).
   Parent `exp_p1_t3_n25_composition` V2 audit on 2026-04-18 recorded
   **0/5 present** (all `safetensors_exists: false`, size 0). The
   five adapter artifacts are the literal inputs to composition — in
   their absence neither bf16 nor W4A16 composition can even begin.
2. **W4A16-quantized Gemma 4 E4B base checkpoint** bound to the
   composition path: weights at 4-bit (group-64 affine, bf16
   activations per F#555) plus the composition routine's delta
   injection operating correctly on the quantized `W_q/W_k/W_v/W_o`.
3. **N=5 composition routine passing parent K1060 and K1061**
   (0/5 below-base, MMLU ≥ base − 2 pp). Parent is KILLED on both.
4. **MMLU-Pro eval harness bound to the composed model** for
   apples-to-apples scoring at both precisions (same prompt template,
   same decode policy, same N-Q sample, same composition weights).
5. **bf16 reference composition score** as the anchor value K1713
   measures "within 1.5 pp" of. The anchor does not exist as a
   passing measurement — parent composition regressed MMLU
   (K1061 FAIL); there is no non-regressed bf16 reference to anchor
   against.

Block fires if shortfall ≥ 3 of 5.

- (1) Five Gemma 4 E4B domain adapters on disk: **absent** —
  parent results.json logs `safetensors_exists: false` for all five
  declared paths (math, code, medical, legal, finance).
- (2) W4A16 Gemma 4 E4B base bound to composition path: **partial** —
  micro-scale F#555 establishes W4A16 base is near-lossless on
  base-only MMLU-Pro, but no repo code binds it to an N=5 composition
  pipeline.
- (3) N=5 composition routine passing K1060 ∧ K1061: **absent** —
  parent KILLED on both.
- (4) MMLU-Pro composed-model harness: **partial / unbound** — eval
  fragments exist but no harness binds them to a W4A16-composed
  Gemma 4 E4B.
- (5) bf16 reference anchor: **absent as a passing measurement** —
  parent's bf16 composition regressed MMLU (K1061 FAIL).

Shortfall is 3 – 5 of 5 (A9 honesty: grep is over-inclusive in a
large repo; the runner reports the literal count and does NOT
inflate). The verdict is over-determined by T2 ∨ T3 ∨ T5-K without
T1, so T1 marginal shortfall is not load-bearing.

### T2 — Cost-bound block
Conservative W4A16 × N=5 composition × MMLU-Pro protocol on M5 Pro
48 GB, MLX, Gemma 4 E4B:

- Gemma 4 E4B base cold-load × 2 (bf16 reference run + W4A16 run,
  serial; cannot coexist in 48 GB unified memory): 2 × 15 min = 30 min.
- N=5 Gemma 4 E4B adapter training from scratch (parent shows the
  five adapters are missing on disk): 5 × 15 min = 75 min at 900 s
  each under the cost model used in parent runs.
- W4A16 quantize of base: 10 min one-shot.
- bf16 N=5 composition + MMLU-Pro eval at 1 000 Q × 5 s/Q: 5 000 s =
  83.3 min.
- W4A16 N=5 composition + MMLU-Pro eval at 1 000 Q × 5 s/Q: 5 000 s =
  83.3 min.
- Per-domain behavioral delta analysis (K1714): 10 min.

Conservative total:
  `30·60 + 75·60 + 10·60 + 5000 + 5000 + 10·60 = 1800 + 4500 + 600 +
  5000 + 5000 + 600 = 17 500 s ≈ 291.7 min`
vs **120 min ceiling**. Block fires by > 2.4 ×.

Floor (smoke: skip adapter train [but then K1713 is meaningless — no
adapters to compose]; reuse both base loads; 100 Q × 1 s × 2
precisions = 200 s): `1800 + 600 + 200 = 2 600 s ≈ 43.3 min` under
ceiling, **but** K1713 degenerates: at 100 Q the MMLU-Pro score
half-width at 95 % CI is ≈ ± 10 pp. A 1.5 pp threshold (K1713) is
strictly inside noise. Floor is scientifically incoherent — it cannot
even in principle falsify K1713.

**T2 blocks.**

### T3 — Schema-incomplete block
DB record (verbatim from `experiment get
exp_model_quantization_composition_stability`):
  `success_criteria: [] # MISSING`
  `⚠ INCOMPLETE: missing success_criteria`
  `references: []` (empty).

Empty `success_criteria`, empty `references`, INCOMPLETE flag
simultaneously. F#502/F#646 antipattern: **14th occurrence** in this
drain (iter 45 was 13th). Stable, earned heuristic.
**T3 blocks.**

### T4 — Audit-pin reinforcer
Macro experiment with no prior runner, no DB diff in last 72 h,
no `.audit` directory. Pin-ratio measured post-run; reinforce-only.
**T4 reinforces (does not block alone).**

### T5 — Source-scope breach block (T5-K single-parent KILLED)
Target declares one parent in `depends_on`:
- `exp_p1_t3_n25_composition` — current DB `Status: killed`
  (K1059 PASS Grassmannian cos; **K1060 FAIL** 0/5 adapter
  .safetensors present on disk; **K1061 FAIL** MMLU regression).

The declared parent is KILLED. This is the **T5-K variant**
(parent-KILLED), single-parent form. 3rd single-parent T5-K in the
drain.

Transitive-kill breach dimensions (pre-reg ≥ 1 required for T5-K;
we count ≥ 3 for defense-in-depth):

  (A) **Adapter-artifact breach (K1060).** Parent's V2 audit recorded
      0/5 Gemma 4 E4B domain adapter `.safetensors` present on
      disk. N=5 composition has no inputs; W4A16 quantization of a
      compositionless model is not a composition experiment.
  (B) **MMLU-anchor breach (K1061).** Parent measured MMLU
      **regression** under bf16 N=5 composition. K1713 anchors
      "within 1.5 pp of bf16 composition" — if the bf16 anchor is a
      *regressed* score, a W4A16 run "within 1.5 pp" of it is within
      1.5 pp of a failure, not a success. The claim is ill-posed.
  (C) **Parameter-orthogonality / behavioral-orthogonality gap
      (K1059).** Parent's one PASS was K1059 Grassmannian
      orthogonality (max|cos| = 2.165 e − 8). This is a
      *parameter-space* claim. W4A16 quantization perturbs the
      Frobenius norms of `W_q/W_k/W_v/W_o` (per F#555, base-only gap
      is 1.79 pp on MMLU-Pro+thinking). Parameter orthogonality at
      bf16 does not imply parameter orthogonality after 4-bit
      affine quantization — the quantization grid is not invariant
      under the orthogonal complement structure. K1059's bf16 PASS
      does not transfer.
  (D) **Tautological-routing inheritance (F#645 / F#502).** Parent's
      V2 audit flagged `REAL_ADAPTER_PATHS[domain]` hardcoded
      adapter-to-domain pairing — the "composition" path was
      routing-by-label, not genuine composition. K1714
      ("per-domain behavioral delta preserved") assumes a genuine
      per-domain contribution signal from composition; under the
      parent's tautological-routing design, the per-domain signal
      is identity-mapped to the adapter, not extracted from the
      composed model's behavior. K1714 cannot fire on tautological
      composition.
  (E) **KC-target coupling breach.** K1713 measures MMLU-Pro
      **under N=5 composition** and K1714 measures per-domain
      contribution **under N=5 composition**. The composed object
      is the parent-KILLED routine. A quantization-stability claim
      about a killed routine is ill-posed: the routine's central
      claim (no-regression composition) already failed at bf16;
      changing precision does not recover a failed method.

Count = **5/5 breaches**. All five are **transitive-kill** breaches
under the T5-K variant (parent-killed). **T5 (T5-K single-parent)
blocks** with wide margin.

**Theorem conclusion.** Verdict is **4-of-5 independent blocks**
(T1 ∧ T2 ∧ T3 ∧ T5-K-single) plus **1 reinforcing** (T4). Any single
block suffices. Target is unrunnable on `local-apple` / MLX / 48 GB
M5 Pro within a 120 min budget without operator action (resurrect
the KILLED parent composition routine, rebuild the five lost
`.safetensors` adapters, bind them to a W4A16-composed MMLU-Pro
harness, register success criteria and references, and — on top of
all that — empirically flip the parent-KILLED K1061 so the bf16
anchor is not itself a regression).

## 3. Predictions (pre-registered)

| ID | Prediction | Measurement |
|----|------------|-------------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts (5 adapters on disk; W4A16 base bound to compose; K1060∧K1061-passing compose; composed MMLU-Pro harness; non-regressed bf16 anchor) | code grep under `pierre/`, `macro/`, `composer/`, `micro/models/` + on-disk `.safetensors` inventory under `micro/models/**/adapters/` |
| P2 | T2 timing ≥ 120 min (conservative 291.7 min; floor 43.3 min but scientifically incoherent at 1.5 pp K1713 threshold under 100 Q noise) | arithmetic on 2-precision × MMLU-Pro × 5-adapter-train protocol |
| P3 | T3 DB has `success_criteria: [] # MISSING` + `⚠ INCOMPLETE` marker + empty references | DB probe via `experiment get` |
| P4 | T4 pin_ratio in `.audit/` = 0 (dir absent); reinforce-only | `.audit` listing |
| P5 | T5-K single-parent: parent `Status: killed`; breach count ≥ 3 of 5 transitive-kill dimensions (adapter-artifact / MMLU-anchor / parameter-behavioral-orth-gap / tautological-routing / KC-target coupling) | DB probe for parent status + parent `results.json` / `PAPER.md` / `MATH.md` read for K1060, K1061, K1059, tautological-routing audit |

## 4. Assumptions / caveats (A-series)
- **A1.** "Present in repo" = grep-reachable in `*.py` under `pierre/`,
  `macro/`, `composer/`, `micro/models/` (excluding this runner).
  Adapter-artifact probe reads on-disk `.safetensors` files under
  `micro/models/**/adapters/` with size > 1 KB; a Gemma 4 E4B
  LoRA `W_q/W_k/W_v/W_o` rank-r adapter should be ≫ 1 KB.
- **A2.** The five required Gemma 4 E4B domain adapters are
  math / code / medical / legal / finance per parent's
  `REAL_ADAPTER_PATHS`. The probe counts distinct existing
  `.safetensors` under those paths; any count < 5 is shortfall.
- **A3.** T1(2) W4A16 binding probe requires literal cooccur of
  `w4a16|W4A16|4bit|4[-_ ]bit|group[_-]?64|affine.*4` with one of
  `compose|compos|merge|add_adapter|W_combined` in the same file.
- **A4.** T2 uses conservative 5 s/sample for MMLU-Pro with
  generated-answer scoring on M5 Pro. Published MMLU-Pro eval
  times on comparable Apple Silicon are 3 – 7 s/Q. Not sensitive:
  floor variant is scientifically incoherent (1.5 pp threshold
  inside ± 10 pp CI at 100 Q).
- **A5.** T3 reads literal DB pretty-print. `success_criteria: [] #
  MISSING` and `⚠ INCOMPLETE: missing success_criteria` are the
  operator-facing "missing" signals.
- **A6.** Parent `exp_p1_t3_n25_composition` is KILLED per live DB
  with K1059 PASS, K1060 FAIL, K1061 FAIL. This is the T5-K variant
  (parent-KILLED), single-parent form. 3rd in drain.
- **A7.** F#555 verified W4A16 Gemma 4 E4B base-only MMLU-Pro gap at
  1.79 pp vs W8A16 (micro scale, base-only, not under composition).
  F#555 is a **base-only** result; extending it to N=5 composition
  is the empirical claim that this experiment would test — and
  T5-K breach (A) – (E) show the experimental premise is ill-posed
  because the bf16 anchor is itself regressed.
- **A8.** Runner is pure stdlib + `experiment get` shell-out. Zero
  MLX, zero model load, zero HTTP bind. Target ≤ 4 s wall.
- **A9.** T1 grep may false-positive on unrelated files that mention
  `w4a16` or `compose`. Runner reports shortfall literally and does
  NOT inflate — verdict over-determined by T2 ∨ T3 ∨ T5-K without
  T1.
- **A10.** F-axis placement: (s4) T5-K single-parent-KILLED lineage,
  3rd instance. Siblings:
  • iter 36 `exp_model_loader_portability` (F#652 sub-axis)
  • iter 45 `exp_model_multi_seed_room_model` (F#660)
  Possible novel sub-axis: (s4-q1) **quantization-on-killed-composition**
  — the target attempts to verify a precision-stability claim about
  a composition routine whose underlying no-regression claim
  already failed at bf16. Analyst owns sibling-vs-child placement
  under F#651 / F#652 / F#660 when 50/50 cap lifts.
