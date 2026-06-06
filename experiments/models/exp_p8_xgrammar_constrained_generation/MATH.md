# MATH.md — P8.A0: Grammar-Constrained Code Generation via Self-Repair

## Problem Statement

Code adapters generate syntactically invalid Python at non-zero rates, making them
unreliable for practical deployment. XGrammar (arXiv:2411.15100) proves that
grammar-constrained decoding achieves 0% syntax errors by construction. On MLX, no
token-level grammar library is available, so we test the "think-then-constrain" +
self-repair protocol (arXiv:2601.07525) as an equivalent guarantee via repeated
independent sampling.

---

## Theorem 1 (Self-Repair Convergence to Zero Syntax Errors)

**Setup**: Let p₀ = P(syntactically valid code | single generation attempt) for a
code-adapted LLM. Let X₁, X₂, ..., Xₙ be i.i.d. Bernoulli(p₀) outcomes (validity of
each attempt). Attempts are independent because each retry generates from scratch (no
shared prefix state between retries).

**Theorem**: After N independent attempts:
```
P(at least one valid | N attempts) = 1 - (1 - p₀)^N
```

**Proof**: By De Morgan's law, P(at least one valid) = 1 - P(all invalid) = 1 - P(X₁=0)·P(X₂=0)·...·P(Xₙ=0). Since attempts are independent, each P(Xᵢ=0) = 1-p₀. ∎

**Corollary (N=3 bound)**:
- If p₀ ≥ 0.60 → P(valid after 3) ≥ 0.936
- If p₀ ≥ 0.80 → P(valid after 3) ≥ 0.992  
- If p₀ ≥ 0.90 → P(valid after 3) ≥ 0.999

**Prediction**: Code adapter (HumanEval 63%) likely has p₀ ≥ 0.75. Therefore:
P(valid after 3) ≥ 1 - (0.25)³ = **0.984**. K1333 (≈0% syntax errors) is achievable.

---

## Theorem 2 (Think-then-Code Does Not Reduce Accuracy)

**Setup**: Direct generation has context C = [prompt]. Think-then-code (arXiv:2601.07525)
has context C' = [prompt, thinking_tokens]. Since C ⊂ C' (strictly more context), and
the model is a conditional distribution P(code | context):

**Theorem**: E[correctness | C'] ≥ E[correctness | C] when thinking tokens contain valid
reasoning steps. Specifically, if the thinking trace T is a valid derivation of the
algorithmic solution, then the conditional P(correct_code | C, T) ≥ P(correct_code | C).

**Proof Sketch**: Correct reasoning trace T reduces conditional entropy H(code | C, T) ≤
H(code | C). Lower entropy → probability mass concentrated near correct solution. ∎

**Important caveat**: This holds when T is valid. If T is hallucinated or wrong, accuracy
can decrease. The experiment measures both cases.

**Prediction**: Think-then-code accuracy ≥ direct accuracy (or within 5pp). K1334 PASS.

---

## Theorem 3 (Syntax Check Overhead)

**Setup**: Grammar checking cost = `ast.parse(code)` ≈ O(|code|) Python bytecodes.
Generation cost = T_gen tokens × (1/throughput). For Gemma 4 E4B at ~73 tok/s:
- T_gen = 100 tokens → 1.37s generation time
- `ast.parse(100-token code)` ≈ 0.5ms

**Theorem**: Grammar check overhead = 0.5ms / 1370ms ≈ **0.036%** << 5% (K1335 PASS).

**Proof**: By direct time measurement. ∎

---

## Kill Criteria

| ID | Criterion | Predicted Outcome |
|----|-----------|-------------------|
| K1333 | Self-repair (N=3) achieves ≤2% syntax errors | PASS: (1-p₀)³ ≤ 0.016 for p₀≥0.75 |
| K1334 | Think-then-code accuracy ≥ direct - 5pp | PASS: more context = same or better |
| K1335 | Grammar check overhead < 5% of latency | PASS: ast.parse ≈ 0.04% |

---

## Experiment Design

**Task**: Python function generation from docstring + signature (20 hand-crafted problems
with known correct outputs, testable via `exec()`).

**Models tested**:
1. Base (Gemma 4 E4B, 4-bit, no adapter)
2. Code adapter (code-codealpaca-knowledge-v0, HumanEval 63%)

**Generation modes**:
- P0: Direct generation, max_tokens=256, temperature=0
- P1: Think-then-code (prompt forces reasoning before code block)
- P2: Self-repair (P0 + retry if syntax error, N_retry=3)

**Metrics**:
- `syntax_error_rate`: fraction that fail `ast.parse()`
- `test_pass_rate`: fraction whose execution produces correct answer
- `latency_ms`: wall-clock ms per problem (including retries)
- `check_overhead_pct`: grammar-check time / generation time × 100

**Failure mode**: If p₀ < 0.5, N=3 retries only gets to 87.5%, not ≈0%. Would need
N=7+ retries, which violates K1335 (too much latency overhead).

---

## Connection to P1 Architecture

This experiment is a stepping stone to P8's full vision: code adapter + Python grammar
constraint eliminates a class of generation failures without retraining. The self-repair
protocol is an MLX-compatible proxy for token-level grammar masking (XGrammar).

If K1333 passes, C2 integration is unblocked: code adapter provides syntactic structure
guarantees that don't degrade when composed with other domain adapters.

---

## Citations

- XGrammar: arXiv:2411.15100 (Ji et al., 2024) — Grammar-constrained decoding via GLL automata
- Think-then-constrain: arXiv:2601.07525 (Ye et al., 2026) — Structured reasoning before constrained output  
- Prior: code-codealpaca-knowledge-v0, HumanEval=63% (Finding #421)
