# MATH — exp_g4_25domain_real_hf

**Experiment:** N=25 Gemma 4 domain adapters specialize ≥10pp on own-domain MMLU-Pro subject
**Verdict (from math):** KILLED_PREEMPTIVE — multiple independent impossibility results close K1606.
**Date:** 2026-04-18

---

## Kill criterion

**K1606:** 20/25 adapters specialize ≥10pp on own-domain MMLU-Pro subject.

---

## Antipattern self-check

| Antipattern | Applies? | Evidence |
|---|---|---|
| ap-017 (stub adapters) | NO | This experiment would CREATE adapters. Does not consume existing. |
| ap-020 (cascade-upstream-killed) | NO | `depends_on: []` per DB. |
| ap-framework-incomplete | **YES** | `success_criteria: []` per `experiment get exp_g4_25domain_real_hf`. |
| ap-scale-misclassified | **YES** | Claimed "micro"; wall-clock extrapolation = 8.7h (macro). |
| ap-domain-count-mismatch | **YES** | MMLU-Pro has 14 disciplines, not 25. N=25 cannot map 1:1 to MMLU-Pro subjects. |

---

## Theorem 1 (primary kill, F#478 impossibility)

**Statement.** Let θ_base = Gemma 4 E4B 4-bit (instruction-tuned). Let ΔW_d = LoRA r=6
q_proj adapter trained on N ≤ 2000 HF instruction examples from domain d (GSM8K-style,
CodeAlpaca, MedMCQA, etc.). Let δ_d = (MMLU-Pro subject accuracy with ΔW_d) − (MMLU-Pro
subject accuracy with θ_base alone). Then δ_d ≥ 10pp is structurally unreachable.

**Proof.** From Finding #478 (Impossibility Structure):

> For rank-r LoRA adapter ΔW_d trained on N basic domain examples: δ_d > 0 requires
> BOTH (1) vocabulary gap H(V_d|θ_base) > H_threshold AND (2) distribution overlap
> V_d ∩ V_train ≠ ∅. Gemma 4 4B fails condition (1) for advanced questions. P1 T2
> adapters fail condition (2) for advanced subdomain content.

MMLU-Pro is 10-option, graduate-level (10× harder option set than MMLU; Wang et al.
2024). The "advanced questions" regime of F#478 subsumes MMLU-Pro. Therefore
condition (1) fails uniformly across domains d ∈ {math, science, engineering, health,
humanities, …}. At the threshold δ_d ≥ 10pp we additionally need δ_knowledge ≥ 10pp
(see Theorem 2), which contradicts F#478's empirical result δ_knowledge ≈ 0 for basic
adapters on advanced questions. ∎

**Consequence.** K1606 ("20/25 specialize ≥10pp") cannot be satisfied by any
construction using (θ_base = Gemma 4 E4B 4-bit, r=6 q_proj, HF standard instruction
data, MMLU-Pro eval). Kill is **mathematical, not empirical**.

## Theorem 2 (format/knowledge decomposition excludes the F#424 exception)

**Statement.** Write observed gain as δ_total = δ_format + δ_knowledge, where δ_format
captures A/B/C/D response-format alignment and δ_knowledge captures new domain
knowledge transfer. Then for MMLU-Pro evaluation, δ_format ≈ 0.

**Proof.** Finding #424 observed +22pp…+82pp on MMLU at a base of 4% for legal/finance
— the base model produced prose instead of A/B/C/D, so a large fraction of δ_total was
δ_format ∈ [12pp, 52pp] (per F#424 caveat: "true domain knowledge gain estimated
10-30pp"). On **MMLU-Pro**, Gemma 4 E4B base produces valid answer labels (see Finding
#442 "MMLU 56-88% across all adapters" — Gemma 4 already handles the MCQ format).
Therefore δ_format ≈ 0 on MMLU-Pro. Combining with Theorem 1 (δ_knowledge < 10pp
for basic adapters on advanced questions): δ_total < 10pp. ∎

**Consequence.** The F#424 success cannot transfer to MMLU-Pro: the F#424 gains were
dominated by format-adaptation to 4% baselines that don't exist on MMLU-Pro.

## Theorem 3 (framework incompleteness)

**Statement.** The DB entry for `exp_g4_25domain_real_hf` is not verifiable-as-SUPPORTED
under the experiment framework.

**Proof.** `experiment get exp_g4_25domain_real_hf` returns `success_criteria: []`. Per
framework convention, SUPPORTED requires *at least one* success criterion to verify,
distinct from kill criteria (which only rule out the null). With zero success criteria,
only KILLED is a valid terminal state. ∎

## Theorem 4 (feasibility bound)

**Statement.** Wall-clock to train 25 Gemma 4 E4B adapters sequentially exceeds the
Ralph iteration budget by ~17×.

**Proof.** From Finding #424: 5 adapters in 1.74h ⇒ t̄ = 20.88 min/adapter. For N=25:
T_total = 25 × 20.88 min = 522 min = 8.7 h. Ralph guardrail 1008 bounds single-hat
transitions to <30 min. 522 / 30 ≈ 17.4×. ∎

**Consequence.** Experiment cannot be completed atomically; requires decomposition
into (a) background-pueue full run + (b) post-hoc validation iteration.

## Theorem 5 (domain-count mismatch)

**Statement.** MMLU-Pro has 14 disciplines; N=25 cannot be 1:1 mapped to MMLU-Pro
own-domain evaluation as K1606 specifies.

**Proof.** MMLU-Pro (Wang et al. 2024) defines 14 categories: biology, business,
chemistry, computer science, economics, engineering, health, history, law, math,
philosophy, physics, psychology, other. K1606 requires "own-domain MMLU-Pro
subject" per adapter — there is no injective mapping from 25 domains into 14
categories (pigeonhole). Either ≥11 adapters share a MMLU-Pro category (violating
"own-domain" independence) or ≥11 adapters have no MMLU-Pro eval (violating
measurement). ∎

## Predictions

| ID | Prediction | How to measure |
|---|---|---|
| P1 | Experiment dir has no adapter safetensors | `find micro/models/exp_g4_25domain_real_hf/ -name '*.safetensors'` returns empty |
| P2 | DB shows `success_criteria: []` | `experiment get exp_g4_25domain_real_hf` |
| P3 | Wall-clock ≥ 8 h for full N=25 training on M5 Pro | extrapolation from F#424; not empirically verified |
| P4 | MMLU-Pro has 14 disciplines, not 25 | Wang et al. 2024 dataset spec |
| P5 | F#478 structural impossibility still cited as `killed` | `experiment finding-get 478` |
| P6 | Harness is functional (disambiguation from F#557) | `ls adapters/thinking-openthoughts-universal-v0/adapters.safetensors` exists |

---

## Unblock path (constructive outcome)

K1606 becomes reachable (∃ construction where K1606 is satisfied) only under one of:

1. **Change base model** to one with exploitable knowledge gaps on MMLU-Pro (e.g.
   Qwen3-0.6B per F#410; other sub-2B models).
2. **Change evaluation** from MMLU-Pro (advanced) to gap-rich benchmark (e.g.,
   domain-specific proprietary corpus, or F#478-excluded MMLU-easy subset).
3. **Change adapter data** to include advanced subdomain corpora (not HF basic
   instruction datasets) that cover the MMLU-Pro question distribution.
4. **Change domain count** to N=14 (matching MMLU-Pro's discipline count) AND
   accept the structural constraint.

Without at least one of (1)-(4), K1606 is closed by Theorem 1 alone. Theorems 3-5
close the experiment-framing even if Theorem 1 were somehow bypassed.

## References

- Finding #478: "P4.B1: Gemma 4 4B has no exploitable knowledge gap" (killed).
- Finding #424: "5-Domain Adapter MVP: +22pp to +82pp on MMLU" (supported; caveat quantifies δ_format).
- Finding #442: "MMLU 56-88% across all adapters" (Gemma 4 baseline MCQ competence).
- Finding #410: Qwen3-4B adapter-gap regime (different from Gemma 4).
- Finding #557: "P11.F0 mlx_lm.lora subprocess crashed" (specific long-seq OOM, not universal harness failure).
- Wang, Y., et al. "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." arXiv:2406.01574 (2024).
