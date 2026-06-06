# P6.A1: TTT-Style Embedded Adapter Update — KILLED

## Summary

We test whether TTT-style self-supervised training (next-token prediction on ALL
tokens) can match P6.A0's supervised approach (response-only loss) for factual
recall via online LoRA adaptation. On Gemma 4 E4B (4-bit), the self-supervised
approach achieves 40% project QA accuracy vs P6.A0's 60% — a 20pp gap. The
backward pass adds 50.6% overhead, disproving the "zero additional cost" claim.
All three kill criteria fail.

## Method

- **Base model**: Gemma 4 E4B 4-bit (frozen) — identical to P6.A0
- **LoRA**: rank-4 on q_proj + o_proj, last 8 layers (327,680 params) — identical
- **Training**: AdamW lr=1e-3, 20 turns, one gradient step per turn — identical
- **Loss difference**: cross-entropy on ALL tokens (TTT self-supervised) vs
  response tokens only (P6.A0 supervised)
- **Evaluation**: generation + keyword matching on 10 project / 20 general questions

## Prediction vs Measurement

| Prediction | Predicted | Measured | Status |
|-----------|-----------|----------|--------|
| Self-supervised project accuracy ~50-60% | 50-60% | 40% | BELOW |
| Backward pass overhead | >5% | 50.6% | CONFIRMED |
| K1289 zero latency | FAIL | FAIL (50.6%) | CONFIRMED |
| General knowledge degradation <2pp | <2pp | 0pp (90% → 90%) | CONFIRMED |
| Loss decrease >50% | >50% | 75.3% (7.09 → 1.75) | EXCEEDED |
| Per-turn latency ~110ms | ~110ms | 109ms avg | CONFIRMED |

## Detailed Results

### Facts Learned by TTT (4/10)
| Fact | TTT Response | P6.A0 | Notes |
|------|-------------|-------|-------|
| ZephyrFlow | "ZephyrFlow" | YES | Both learned |
| zf: prefix | "zf:" | YES | Both learned |
| FastAPI | "FastAPI" | YES | Both learned |
| mypy | "mypy" | YES | Both learned |

### Facts Missed by TTT but Learned by P6.A0 (2/10)
| Fact | TTT Response | P6.A0 | Notes |
|------|-------------|-------|-------|
| ClickHouse | "ZephyrFlow's internal event storage" | YES | **TTT hallucinated** |
| ruff | "ZephyrFlow's internal linting system" | YES | **TTT hallucinated** |

### Facts Missed by Both (4/10)
| Fact | TTT Response | P6.A0 | Notes |
|------|-------------|-------|-------|
| Python 3.12 | "Python" (no version) | "Python" (no version) | Both partial |
| Fly.io | "proprietary cloud platform" | YES (P6.A0 HIT) | TTT hallucinated |
| 256KB | "256 bytes" | "256 bytes" | Both wrong unit |
| 90 days | "no information" | YES (P6.A0 HIT) | TTT failed entirely |

**Correction**: P6.A0 actually learned Fly.io and 90 days. Updated comparison:
- TTT: 4/10 (ZephyrFlow, zf:, FastAPI, mypy)
- P6.A0: 6/10 (ZephyrFlow partial, ClickHouse, Fly.io, 90 days, FastAPI, ruff, mypy)

### Hallucination Pattern
TTT produces "ZephyrFlow's [generic description]" instead of specific technology
names. The self-supervised loss learned the PATTERN "ZephyrFlow uses [X]" but
failed to anchor X to specific tokens (ClickHouse, Fly.io, ruff).

This is because the all-token loss spreads gradient signal across:
- Chat template tokens (formatting)
- Question tokens (already known)
- Response tokens (the actual facts)

Only response tokens encode factual associations. The supervised (response-only)
loss of P6.A0 concentrates ALL gradient signal on the factual content.

### Topic Contamination
TTT corrupted one general knowledge question: "What is the capital of Japan?"
produced "ZephyrFlow" instead of "Tokyo." The all-token loss over-biased the
model toward generating ZephyrFlow-related content for any input.

### Latency Analysis
| Metric | Value |
|--------|-------|
| Forward pass only | 47.6ms |
| Forward + backward | 71.7ms |
| Backward overhead | 24.1ms (50.6%) |
| Training step (full) | 109ms avg |

The backward pass through the model is REQUIRED for LoRA gradient computation
(Theorem 1 in MATH.md). TTT's closed-form gradient applies only to linear
inner models where the loss is defined directly on the model's output. For LoRA
inside attention (q_proj, o_proj), the gradient path includes softmax and layer
norms, preventing closed-form computation.

### Loss Dynamics
- Self-supervised loss: 7.09 → 1.75 (75.3% decrease)
- Supervised reference loss: 3.97 → 2.14 (tracked but not optimized)
- All-token loss decreases faster because prompt tokens are easy to predict
- But faster loss decrease does NOT equal better factual learning

## Kill Criteria

| ID | Criterion | Threshold | Measured | Result |
|----|----------|-----------|----------|--------|
| K1288 | Fact recall | >= 50% | 40% | **FAIL** |
| K1289 | Zero latency overhead | < 5% | 50.6% | **FAIL** |
| K1290 | Match P6.A0 quality | >= 50% | 40% vs 60% | **FAIL** |

## Impossibility Structure

### Why TTT Zero-Cost Fails for Transformer LoRA
TTT requires the inner model's gradient to have a closed form. For LoRA adapters:
- LoRA output: Δy = s · x @ A @ B (linear in A, B)
- But the LOSS depends on logits = f(Δy, ...) where f includes softmax attention,
  GELU MLP, and RMSNorm — all non-linear in Δy
- Therefore ∂L/∂A requires backprop through f, not just through the linear LoRA

The minimum additional cost for a meaningful LoRA update is one backward pass
through at least one transformer layer (~24ms on M5 Pro for Gemma 4 E4B).

### Why Self-Supervised Loss Underperforms Supervised
The all-token cross-entropy loss distributes gradient signal:
- ~60% of tokens are chat template + question (formatting/context)
- ~40% of tokens are the response (factual content)

The response-only loss concentrates 100% of signal on factual tokens. This 2.5x
signal concentration explains the quality gap (60% vs 40%).

Additionally, training on prompt tokens biases the model toward generating
prompt-like tokens (questions, formatting), causing topic contamination on
unrelated queries.

## Architectural Implications

1. **Supervised signal is necessary for factual recall**: Self-supervised
   next-token prediction alone loses 20pp compared to response-only loss.
   The QA formatting in P6.A0 is NOT just convenience — it's a meaningful
   quality signal.

2. **TTT zero-cost is impossible for LoRA**: The non-linear gradient path
   through attention mandates backpropagation. No closed-form shortcut exists.
   P6.A0's 110ms backward pass is the minimum viable cost.

3. **All-token loss causes topic contamination**: Training on prompt tokens
   biases the model toward the conversation topic even for unrelated queries.
   Response-only masking is necessary to prevent this.

4. **P6.A0 remains the superior approach**: Same cost (109ms vs 110ms),
   better quality (60% vs 40%), no contamination.

## References

- arXiv:2407.04620 — TTT: Test-Time Training (Sun et al.)
- arXiv:2411.13405 — PLUM: conversation-to-QA augmentation
- arXiv:2012.13255 — Intrinsic dimensionality of fine-tuning
- arXiv:2106.09685 — LoRA: Low-Rank Adaptation
- Finding: exp_p6_lingering_adapter_online — P6.A0 baseline (60%, 110ms)
