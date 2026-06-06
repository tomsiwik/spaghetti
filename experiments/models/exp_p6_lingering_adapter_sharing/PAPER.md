# PAPER: P6.B0 — Adapter Sharing Flywheel

## Experiment Type
Verification — testing whether individually-proven components (online LoRA, crystallization,
promotion) compose into a working flywheel.

## Setup
- Model: Gemma 4 e4b-it-4bit (4B params, quantized)
- LoRA: rank-4, 8 layers, q_proj + o_proj (327,680 trainable params A+B)
- 10 users, each knows 6/10 ZephyrFlow facts (sliding window → uniform coverage c=6)
- 20 training turns per user (3 QA pairs × 6 facts = 18, padded to 20)
- Crystallization: average both A and B matrices across all 10 users
- Evaluation: 10 keyword-matching test questions (one per fact)

## Prediction vs Measurement

| Metric | Predicted | Measured | Match |
|--------|-----------|----------|-------|
| Base accuracy | 0% | 0% | YES |
| Best individual | 50-70% | 50% (User 3) | YES |
| Mean individual | ~40% | 32% | CLOSE |
| Crystal accuracy | 70-100% | 50% | MISS |
| Crystal vs best individual | +5pp | 0pp | MISS |
| Crystal-init new user | 70-90% | 20% | MISS |
| Control user (zero-init) | 50-60% | 30% | MISS |
| New user vs control | +20pp | -10pp | MISS |
| Total time | 5-7 min | 3.3 min | YES |

## Kill Criteria Results

| ID | Criterion | Result | Value |
|----|-----------|--------|-------|
| K1291 | Crystal >= best individual + 5pp | **FAIL** | margin = 0pp (50% vs 50%) |
| K1292 | Crystal-init user >= control + 3pp | **FAIL** | margin = -10pp (20% vs 30%) |
| K1293 | Total time < 10 min | **PASS** | 3.3 min |

## Key Findings

### 1. Crystal achieves coverage union (the mechanism works)
The crystal adapter correctly answers facts from 5 DIFFERENT users' training sets:
- "ZephyrFlow" — learned by Users 0,5-9 (NOT User 3 who was best individual)
- "FastAPI" — learned by Users 0-2
- "ClickHouse" — learned by Users 1-4
- "90 days" — learned by Users 2-7
- "ruff" — learned by Users 4-9

This confirms Theorem 2: crystallization creates a coverage union from partial-knowledge users.

### 2. Signal attenuation brings facts to threshold boundary
Several facts are ALMOST correct but miss exact keywords:
- Python "3.11" instead of "3.12" — close but below keyword threshold
- "256 bytes" instead of "256KB" — unit wrong, number close
- "zf_" instead of "zf:" — separator character wrong
- "AWS" instead of "Fly.io" — completely wrong (hallucination dominates)

The 0.6× signal attenuation from averaging 10 users brings many facts to the
decision boundary. Small perturbations flip between correct and incorrect keywords.
This is the threshold effect: generation has a hard boundary, and attenuated signal
hovers near it.

### 3. Crystal-init training causes catastrophic forgetting
The most important finding. When a new user initializes their LoRA from crystal
values and trains on their 6/10 facts:
- Crystal alone: 50% (5/10 facts)
- After 20 turns of training: 20% (2/10 facts)
- Shows pathological repetition: "ZephyrFlowFlowFlow..."

The gradient updates from the new user's training data (6 facts) perturb the
crystal's encoding of ALL 10 facts. Facts 6-9 (not in the new user's training
set) are catastrophically forgotten. Additionally, the model develops repetition
pathology (also observed in P6.A0 LEARNINGS).

**Implication:** The flywheel CANNOT use crystal as training initialization. The
crystal must be used as a FROZEN adapter (or promoted into base weights before
quantization). User adapters must be SEPARATE from the crystal.

### 4. Crystal matches but does not exceed best individual
Crystal (50%) = best individual User 3 (50%). The crystal covers MORE facts but
at LOWER per-fact signal. The threshold effect makes these equivalent in keyword
accuracy. With softer evaluation (e.g., semantic similarity), the crystal would
likely outperform because it encodes partial signal for near-miss facts.

## Architectural Implication
The flywheel architecture should be:
1. Crystal adapter = frozen domain knowledge layer
2. User adapter = separate, trainable personal knowledge layer
3. Output = base + crystal contribution + user contribution
4. Promotion (baking crystal into base) happens BEFORE quantization

This requires multi-adapter composition at inference time — already proven in
our architecture (Finding #225, N=25 composition at near-lossless quality).

## References
- P6.A0 (Finding #490): Online LoRA baseline, 60% accuracy with all 10 facts
- T6.2 (Finding #451): B-matrix crystallization, cos=0.9806
- T6.3: Promotion exact by construction (cos=0.99999988)
- Model Soup (arXiv:2203.05482): weight averaging
- Task Arithmetic (arXiv:2212.04089): linear composition
- Finding #225: N=25 adapter composition at near-lossless quality
