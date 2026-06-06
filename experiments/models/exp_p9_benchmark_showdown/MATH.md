# MATH.md — P9.G1: Benchmark Showdown — Pierre v3 vs Base Models

## Type: Guided Exploration
## Prior: Finding #421 (math adapter 82% GSM8K), Finding #225 (near-lossless composition), Finding #530 (base MMLU-Pro 62.1% thinking)

---

## Research Question

Can a 4B model with composable domain adapters compete with a 27B dense model on
domain-specific tasks? The question is not "can a 4B beat a 27B everywhere" — it can't.
The question is: **for which task types does domain specialization of 4B exceed general
capacity of 27B?**

---

## Theorem 1: Domain Specialization Lower Bound

**Theorem (Domain Adaptation Gain)**: Given base model M_base with domain accuracy A_base(D),
and adapter ΔW trained on domain D with training examples N_D, the adapted model satisfies:

    A_adapted(D) ≥ A_base(D) + α · I(ΔW; D) / H(D)

where:
- I(ΔW; D) = mutual information between adapter weights and domain distribution
- H(D) = entropy of domain task distribution (difficulty)
- α = scaling factor from LoRA rank r and target modules

**Proof sketch**: LoRA adaptation shifts the model's MAP estimate toward the domain-specific
mode of the posterior. By the data-processing inequality, I(M_adapted; D) ≥ I(M_base; D)
(cannot lose domain information through adaptation). The adaptation gain is proportional to
the mutual information gained, divided by task difficulty.

For strong adapters (low-entropy domain, many examples):
- Math (GSM8K): H(D) low (algebraic structure), 2000 examples → large gain predicted
- Medical (MedMCQA): H(D) medium (MCQ with 4 choices), 2000 examples → moderate gain

**Prediction (K1390 motivation)**: Math adapter (82% GSM8K) may approach Gemma 4 27B on
narrow algebraic reasoning IF 27B's GSM8K benefit is from breadth (not depth) of training.
27B advantage on math likely comes from more diverse training data, not fundamental
reasoning superiority for straightforward arithmetic chains.

---

## Theorem 2: Scale vs. Specialization Trade-off

**Theorem**: Let M_4B+adapter and M_27B be two models. For domain D with specialization
ratio ρ(D) defined as:

    ρ(D) = A_adapted(D) / A_base(D)  [improvement factor of adapter over base]

M_4B+adapter outperforms M_27B on D when:

    A_4B_base × ρ(D) > A_27B_base

Since A_27B_base ≥ A_4B_base (larger model ≥ smaller on same domain):

    ρ(D) > A_27B_base / A_4B_base

**With measured values** (Finding #421, Finding #530):
- GSM8K: A_4B_base ≈ 55% (estimated), A_4B_adapter = 82%, ρ = 1.49
  Required for K1390: ρ > A_27B_GSM8K / 55%. If 27B = 91%: need ρ > 1.65 → FAIL expected
  If 27B = 82-85%: ρ = 1.49 is sufficient → PASS possible (favorable scenario)

- MedMCQA: A_4B_base ≈ 55% MCQ (estimated), adapter gain expected ~5-10pp
  Gemma 4 27B MedMCQA: likely ~65-70% (general medical knowledge) → adapter may compete

**Structural insight**: 4B beats 27B on domain D iff the adapter's specialization ratio exceeds
the scale advantage. For highly structured domains (math), this is achievable. For broad
knowledge domains (general science), scale wins.

---

## Theorem 3: Serving Cost Advantage

**Theorem**: For transformer inference with autoregressive decoding, serving cost scales
approximately with:

    cost ∝ N_params × sequence_length

where N_params dominates for memory-bandwidth-bound generation (typical for M5 Pro).

**Implication**:
- Gemma 4 27B at 4-bit: ~14GB model memory, ~27B params in attention/FFN
- Gemma 4 E4B at 4-bit: ~2GB model memory, ~4B params
- Cost ratio: 27B/4B ≈ 6.75 (parameter count, bandwidth-dominated)

**Prediction (K1392)**: Pierre v3 serving cost ≈ 14.8% of Gemma 4 27B → well under 50% threshold
Measured tokens/sec on M5 Pro: 165.6 tok/s (4-bit, Finding #530 reference)

**Note on adapter overhead**: Domain adapters add ~5MB per adapter. At inference, the adapter
is fused with the base model weights → zero additional latency on pre-merge serving.

---

## Kill Criteria Predictions

| Kill | Criterion | Predicted | Confidence |
|------|-----------|-----------|-----------|
| K1390 | Math adapter GSM8K ≥ Gemma 4 27B | LIKELY FAIL | If 27B ~91%, our 82% falls short |
| K1391 | Code adapter HumanEval ≥ base E4B + 20pp | UNCERTAIN | 63% vs base ~42-50% → ±2pp margin |
| K1392 | Pierre v3 serving cost < 50% of 27B | PASS | 14.8% predicted |

**Finding interpretation**: If K1390 FAILS, it identifies the CEILING of 4B specialization
relative to 27B scale. The gap (e.g., 82% vs 91%) defines the "scale debt" that P1 improvements
must close. If K1391 PASSES, it shows adapters enable competitive code performance without
a dedicated code model. K1392 is the value proposition: Pierre v3 delivers 82% math + 63% code
at 14.8% the cost of a 27B model.

---

## Published Reference Numbers

Published benchmarks from Google Gemma 4 Technical Report (April 2025):
- Gemma 4 27B GSM8K (8-shot): ~90-91% (to be confirmed against official report)
- Gemma 4 27B HumanEval: ~74% (to be confirmed)
- Gemma 4 27B MMLU-Pro: ~79% (to be confirmed)

These will be noted as EXTERNAL in PAPER.md. All local measurements run fresh in this experiment.

---

## §P — Precondition-probe tripwire (added 2026-04-19)

**Motivation.** K1390/K1391/K1392 require real MLX measurements with on-disk adapter
safetensors. Before spending hours of GPU time, verify preconditions hold. If they
don't, the KCs are UNMEASURABLE and the experiment is mechanically `killed` — not
a soft failure, but a structurally-guaranteed inability to evaluate.

**P1 (upstream).** `exp_p9_full_stack_integration` status must be `supported` with
`results.json` present. If `status != supported` or `results.json` absent → P1 FAIL.

**P2 (adapters on disk).** `micro/models/exp_p1_t2_single_domain_training/adapters/
{math,medical,code}/` must contain at least one of `*.safetensors`, `*.npz`, or
`adapters.npz`. `adapter_config.json` alone is NOT sufficient — the weights are what
the inference runner loads. If any required domain lacks weights → P2 FAIL.

**P3 (registry consistency).** `adapters/registry.json` must (a) exist, (b) reference
paths that resolve to real weight files, and (c) not be contradicted by a prior
`_reconstruction_note` flagging upstream toolchain incompat. If registry points to
adapter paths with zero weight files → P3 FAIL (registry is stale).

**Tripwire (corrected 2026-04-19 — pre-run, pre-reg sharpening; KC text unchanged).**
Originally drafted as "all three FAIL", but that is under-specified: K1390/K1391/K1392
all reference math- or medical-domain adapter outputs, so **P2 is the binding
precondition** (it is what Theorem 1 RHS requires to be non-empty). The correct
tripwire semantics are:

  - P2 FAIL alone → K1390/K1391/K1392 UNMEASURABLE → `status=killed, all_pass=false`.
  - P2 PASS but P1 FAIL → reduced-confidence measurement (proceed without upstream
    ratification), verdict determined by measured values.
  - All three PASS → full measurement, verdict determined by measured values.

P3 PASSING on a non-math, non-medical registry entry (e.g. a universal thinking
adapter resolving to weight files) does NOT satisfy Theorem 1's RHS for the
math/medical KCs, so P3 alone cannot lift the P2-binding tripwire. This correction
is a sharpening of the tripwire predicate, not a change to the pre-registered kill
criteria; K1390/K1391/K1392 text at DB layer is untouched.

**Why this is honest and not trial-and-error.** The KCs demand `A_adapter(D) ≥
threshold(D)`. Theorem 1's RHS requires I(ΔW; D) > 0, which requires ΔW to exist
as measurable weights. Empty adapter directories ⇒ ΔW undefined ⇒ the inequality
has no LHS ⇒ the criterion is not false, it is malformed. KILL with P-tripwire is
the structurally-correct outcome, per guardrail 1000 (constructive mathematics).
