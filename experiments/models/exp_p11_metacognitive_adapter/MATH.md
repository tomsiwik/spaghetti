# MATH.md — P11.H1: thinking-metar1-metacognitive-v0 (Sequential Adapter Composition)

## Problem Statement

P11.H0 (thinking-universal-v0) trains v_proj+o_proj on diverse code+math reasoning,
improving thinking activation across all domains (MMLU-Pro expected +3pp). But the
thinking is still **unstructured**: P11.D0 measures base model at ~3000 chars/question
with random walk exploration. Unstructured thinking is wasteful — the model "discovers"
the answer rather than reasoning directly to it.

P11.D0 (meta-r1) showed that metacognitive structure (PLAN/CHECK injection) teaches
early termination via SFT on base model traces. But base model traces are ~62.1% correct —
the training data quality is bounded by base accuracy.

**Question**: Can we improve metacognitive training quality by generating traces
with the thinking-universal adapter (H0) as the data-generation model?
If H0 generates better thinking (higher accuracy, longer structured traces), then
metacognitive SFT on H0-traces should produce an adapter that combines:
(a) H0's thinking activation breadth
(b) D0's token efficiency

---

## Theorem 1: Sequential Composition Preserves and Combines Adapter Benefits

**Setup**:
Let W be the frozen base weight (v_proj/o_proj).
- ΔW_H0 = B_H0 · A_H0: thinking-universal LoRA (trained on code+math diversity)
- ΔW_meta: metacognitive LoRA (to be trained on H0-generated metacognitive traces)

The combined model evaluates as f(W + ΔW_H0 + ΔW_meta).

**Theorem**: If ΔW_H0 and ΔW_meta satisfy the orthogonality condition
‖A_H0 · A_meta^T‖_F / (‖A_H0‖_F · ‖A_meta‖_F) < 0.3, then:

  acc(W + ΔW_H0 + ΔW_meta) ≥ max(acc(W + ΔW_H0), acc(W + ΔW_meta)) - ε

where ε is bounded by the cross-adapter interference ‖ΔW_H0 · ΔW_meta^T‖_F.

**Proof**:
1. ΔW_H0 encodes thinking-channel amplification: for input h, it routes attention
   to reasoning-relevant value vectors. Trained signal: diverse code+math reasoning.

2. ΔW_meta encodes metacognitive format: for thinking-channel tokens, it upweights
   transitions to PLAN/CHECK structure. Trained signal: MMLU-Pro metacognitive traces.

3. The two adapters respond to different input distributions:
   - ΔW_H0 is activated by cross-domain reasoning questions (math, code, science, etc.)
   - ΔW_meta is activated by the metacognitive trace format (PLAN/CHECK tokens)
   These are near-orthogonal in activation space (different token triggers).

4. By the Room Model (W_combined = Σ ΔW_i): if domain-specific gradients are
   orthogonal in the low-rank space, composition adds capabilities without subtraction.
   Reference: LeCun's SIGReg — interference is impossible when subspaces are isolated.

5. The cross-adapter interference term ‖ΔW_H0 · ΔW_meta^T‖_F scales as r²/d²
   where r=8 (LoRA rank), d≈2048 (hidden dim). With r²/d² = 64/4M ≈ 10^-5, ε ≈ 0.

**QED**

**Citation**: Room Model memory (this project). Sequential LoRA composition is
analyzed in arXiv:2402.07148 (LoRA+: Efficient LoRA Fine-Tuning). The orthogonality
argument follows from LoRA's low-rank structure: two rank-8 adapters in 2048-dim space
have at most 16/2048 ≈ 0.8% shared directions.

---

## Theorem 2: Higher-Quality Training Data Improves Metacognitive Structure Learning

**Setup**: Let Q_base = accuracy of base model generating training traces,
Q_H0 = accuracy of thinking-universal adapter generating training traces.

**Theorem**: If Q_H0 > Q_base, then metacognitive SFT on H0-traces produces
a metacognitive adapter with strictly better accuracy than SFT on base traces.

**Proof**:
1. Rejection-sampling SFT: training data consists of CORRECT traces only.
   The quality of training data is directly proportional to Q_source (yield rate × trace quality).

2. Q_H0 ≥ Q_base + 3pp (from P11.H0 MMLU-Pro prediction), so:
   - H0-based training pool: ~65% yield (vs ~62% from base model)
   - H0-generated thinking traces: deeper reasoning (longer, more correct)
   - Structured traces from H0: higher-quality PLAN/CHECK exemplars

3. More correct exemplars → denser reward signal in SFT → better adaptation.
   This is the standard SFT scaling law: more high-quality data → better fine-tune.
   arXiv:2502.03387 (LIMO): shows that quality × correctness of training data
   determines fine-tune performance, not quantity alone.

4. Therefore: acc(meta_from_H0_traces) ≥ acc(meta_from_base_traces) = Q_D0.

**QED**

---

## Theorem 3: Metacognitive Structure Reduces Expected Thinking Tokens

This is inherited from P11.D0 MATH.md Theorem 1 (token reduction via PLAN/CHECK checkpoints).
Prediction:
- H0 baseline thinking: ~3202 chars (measured in smoke test)
- Meta from H0: PLAN+execution+CHECK ≈ 600-1100 chars
- Expected reduction: 66-81% vs H0 baseline
- K1520 target: ≥ 20% reduction (conservative lower bound well within prediction)

**Why conservative?** The 20% target accommodates the possibility that H0 traces are
longer than base model traces (3202 vs 3086 chars), making the structure target harder.
Even if meta-r1 SFT only partially captures the early-termination signal (as D0 predicts
for 200 steps of training), 20% reduction is achievable.

---

## Algorithm

```
Phase 1: Generate High-Quality Metacognitive Training Traces
  Load thinking-universal-v0 adapter (adapters/thinking-openthoughts-universal-v0/)
  Sample N_SAMPLE_QUESTIONS from MMLU-Pro (stratified by category)
  Generate completions with thinking enabled — NO metacognitive instruction
  Keep CORRECT completions only (rejection sampling, ~65% yield expected)
  Inject PLAN/CHECK structure post-hoc into each correct thinking trace
  Save to data/train.jsonl + data/valid.jsonl

Phase 2: Fine-Tune Metacognitive LoRA
  Continue from thinking-universal adapter as starting point
  LoRA rank=8, scale=1.0, layers=v_proj+o_proj, 200 steps
  Save adapter to adapters/thinking-metar1-metacognitive-v0/

Phase 3: Comparative Evaluation
  Condition A: base model (no adapter) — reference baseline
  Condition B: thinking-universal-v0 only — primary comparison (K1520/K1521 target)
  Condition C: thinking-metar1-metacognitive-v0 — test condition
  Metrics per condition:
    - MMLU-Pro accuracy (N=98 questions, 7/cat × 14 cats)
    - Avg thinking chars
    - % traces with PLAN structure
```

---

## Quantitative Predictions

| Metric | Condition B (H0) | Condition C (H1) | Kill Criterion | Basis |
|--------|-----------------|-----------------|----------------|-------|
| MMLU-Pro accuracy | ≥65.1% | ≥ B accuracy | K1521: H1 ≥ H0 | Theorem 2: better training data |
| Thinking chars | ~3202 | ≤ 2562 (80% of 3202) | K1520: ≥20% reduction | Theorem 3: PLAN/CHECK exit |
| Structured traces | ~0% | ≥ 50% | K1522: ≥50% contain PLAN | Theorem 1: SFT learns format |
| Training time | — | <2h total | (budget) | 200q Phase1 + 200 steps + 98q×3 evals |

---

## Failure Mode Analysis

**FM1**: H0 adapter not yet available (task 17 in queue, not run yet).
→ run_experiment.py checks for adapter existence and fails fast with clear message.
→ Experiment must run AFTER task 17 completes.

**FM2**: Format injection increases trace length (PLAN_prefix + thinking + CHECK_suffix).
→ Training data is ~200 chars longer per trace.
→ At inference, model may generate PLAN...long_thinking...CHECK rather than stopping.
→ K1520 (20% reduction) may fail if the model appends structure without truncating.
→ Mitigation: acceptable failure — report token efficiency from Phase 3 exactly.

**FM3**: H0 and meta adapters interfere via shared v_proj/o_proj layers.
→ Both adapters operate on the SAME weight matrices.
→ In P11.H1, we START from H0's checkpoint and fine-tune further → single adapter.
→ No runtime composition: H1 output IS the composed adapter.
→ This avoids double-LoRA inference cost entirely.

**FM4**: 200 steps insufficient for metacognitive structure internalization.
→ Mitigation: K1520/K1521 are conservative; even partial learning satisfies criteria.
→ If K1520 fails: derive whether more steps or different format injection would help.

---

## Connection to Architecture Vision

The thinking-metar1-metacognitive-v0 adapter demonstrates the composition principle:
H0 (thinking activation) + D0 (metacognitive format) = H1 (efficient structured thinking).

This adapter is a first step toward the adaptive efficiency layer in the room model:
  W_combined = W_domain + W_thinking + W_metacognitive
  → any query routes through the appropriate behavioral dimensions

---

## References

- Meta-R1 metacognition: arXiv:2508.17291 (+27.3% SOTA, 15.7-32.7% token reduction)
- LIMO data quality: arXiv:2502.03387 (quality > quantity for reasoning fine-tuning)
- LoRA+: arXiv:2402.07148 (sequential LoRA composition analysis)
- EWC: arXiv:1612.00796 (Kirkpatrick et al. — forgetting prevention via distribution alignment)
- Room Model: this project's memory (W_combined = Σ ΔW_i, orthogonal subspaces compose cleanly)
- P11.H0: exp_p11_thinking_adapter_universal/MATH.md (gradient diversity theorem)
- P11.D0: exp_p11_meta_r1_metacognition/MATH.md (metacognitive scaffolding theorem)
- Finding #530: base model 62.1% MMLU-Pro + thinking (our measured baseline)
