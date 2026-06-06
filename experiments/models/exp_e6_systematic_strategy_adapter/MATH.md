# E6: First Strategy Adapter — Systematic Decomposition via Hedgehog Distillation

## §1 Hypothesis

A LoRA adapter trained via Hedgehog per-layer cosine-similarity distillation can transfer the "systematic decomposition" reasoning strategy from a prompted teacher to an unprompted student, improving task accuracy across multiple domains.

**One-sentence claim:** Hedgehog attention-matching distills strategy-level behavioral patterns (decompose → solve parts → combine) into adapter weights, and this strategy generalizes across domains because decomposition is domain-agnostic.

## §2 Prior Work and Grounding

1. **Hedgehog distillation** (Attention to Mamba, arxiv:2310.01101): Per-layer cosine similarity between teacher and student attention outputs is a viable distillation objective. Proven to work for behavioral transfer in this codebase (F#783: politeness adapter, cos=0.99 on heldout).

2. **E1 Finding #801**: Mean-difference activation extraction fails for strategies because format signal dominates. Implication: strategy adapters must be TRAINED via distillation, not extracted from activations. E6 follows this prescription exactly.

3. **Strategy prompting literature** (Wei et al., arxiv:2201.11903 — Chain of Thought): Prompting models to decompose problems improves accuracy on reasoning tasks. The hypothesis here is that this improvement can be distilled into weights.

4. **Finding #497** (exp_p5_per_adapter_reasoning_strategy): Direct prompting dominates all 5 domains on Gemma 4 E4B-IT at QA complexity. This establishes the teacher baseline — the prompted model already benefits from decomposition, so there IS signal to distill.

## §3 Mechanism (Atomic Level)

### Teacher setup
The teacher is Gemma 4 E4B with system prompt S_decomp:
> "For every problem: (1) identify the sub-problems, (2) solve each sub-problem independently showing your work, (3) verify each sub-result, (4) combine into a final answer."

This prompt activates attention patterns that route through decomposition circuits. The teacher processes input x with attention outputs A^T_l(x; S_decomp) at each layer l.

### Student setup
The student is the same model with LoRA adapters on v_proj + o_proj (F#627), scale=6.0. No system prompt. The adapter modifies attention outputs: A^S_l(x; θ_lora).

### Training objective
Per-layer cosine similarity loss (Hedgehog):

$$L = \frac{1}{L} \sum_{l=1}^{L} \left(1 - \frac{1}{T} \sum_{t=1}^{T} \frac{A^T_l[t] \cdot A^S_l[t]}{||A^T_l[t]|| \cdot ||A^S_l[t]||}\right)$$

where T = sequence length (aligned region), L = number of layers.

### Why this should work
1. The politeness adapter (F#783) achieved cos=0.99 via the same mechanism — Hedgehog distillation reliably transfers attention patterns.
2. Systematic decomposition is a behavioral mode encoded in attention routing (which tokens attend to which), not in specific token predictions. Attention matching captures this directly.
3. Domain-agnostic strategies (decompose, verify, combine) should transfer across domains because the attention routing pattern is the same regardless of content domain.

### What could fail (derive failure conditions)
1. **Strategy signal too weak relative to format signal** — same failure mode as E1 but mitigated by distillation (we train toward teacher attention, not extract a difference vector). Still, if the decomposition system prompt changes attention minimally compared to no-system-prompt, the adapter learns mostly identity.
2. **Overfitting to training domain distribution** — if training prompts are all math-like, the adapter may learn "math attention patterns" not "decomposition attention patterns". Mitigated by cross-domain training data.
3. **Accuracy improvement requires generation, not just attention matching** — the adapter matches attention patterns but generation is autoregressive. Attention matching on input processing may not transfer to generation quality. This is the key risk.

## §4 Predictions

1. Training loss converges: final cos-sim loss < 0.05 (corresponding to mean cos > 0.95)
2. Structural cos-sim on heldout: mean per-layer cos > 0.85 (matching politeness result)
3. GSM8K accuracy: adapter improves over base by >3pp (target domain for decomposition)
4. Cross-domain transfer: at least 2 of 3 eval domains show improvement ≥3pp
5. MMLU drop < 3pp (non-interference with general knowledge)

## §5 Kill Criteria (Pre-Registered)

### K_struct (structural, proxy): Mean per-layer cos < 0.85 on heldout
- Paired with K2028 per F#666
- If K_struct FAIL + K2028 PASS → finding about the proxy
- If K_struct PASS + K2028 FAIL → tautological proxy, kill on target

### K2028 (target): Systematic adapter does not improve ANY domain by >3pp over base on behavioral eval
- Domains: GSM8K (math), ARC-Challenge (science reasoning), MMLU-STEM subset
- "Behavioral eval" = task accuracy (correct answer extraction)
- KILL if all three domains show <3pp improvement

### K2029 (target): Adapter is domain-specific: helps only the training domain, not 2+ out-of-domain
- Training uses cross-domain prompts, but eval is on held-out domain benchmarks
- KILL if fewer than 2 domains show ≥3pp improvement
- This tests the core hypothesis: decomposition is domain-agnostic

### MMLU non-interference: MMLU full subset drop > 5pp
- Safety check: adapter doesn't break general knowledge
- Not a kill criterion alone, but flagged as a blocker

## §6 Experiment Design

### Training data
Cross-domain neutral prompts from UltraChat (filtered for length 20-600 chars, no politeness markers). N_TRAIN=200 (full) / 24 (smoke).

### Training hyperparameters
- Model: mlx-community/gemma-4-e4b-it-4bit
- Adapter: LoRA r=8, v_proj+o_proj, scale=6.0
- Optimizer: AdamW, lr=1e-4, weight_decay=0.01
- Steps: 800 (full) / 30 (smoke)
- Sequence length: 512 (full) / 256 (smoke)
- Batch size: 1

### Evaluation
1. **K_struct**: Cos-sim on N=50 heldout prompts (same as politeness protocol)
2. **K2028/K2029**: GSM8K (N=100), ARC-Challenge (N=100), MMLU-STEM (N=100)
3. **Non-interference**: MMLU general (N=100)

### Teacher system prompt
```
For every problem: (1) identify the sub-problems, (2) solve each sub-problem independently showing your work, (3) verify each sub-result, (4) combine into a final answer.
```

## §7 Verdict Matrix (F#666 Compliant)

| K_struct | K2028 | K2029 | Verdict |
|----------|-------|-------|---------|
| PASS | PASS | PASS | SUPPORTED |
| PASS | FAIL | - | KILLED (tautological proxy) |
| FAIL | PASS | PASS | Finding about proxy |
| FAIL | FAIL | - | KILLED |
| * | PASS | FAIL | KILLED (domain-specific, not strategy) |

## §8 Smoke Gate

Before full submission, smoke run must pass:
- A1: Phase B loss converges (first/last ratio ≥ 2.0)
- A2: Cos-sim ≥ 0.85 on smoke heldout
- A3: GSM8K base accuracy ≥ 20% (harness sanity)
- A4: Non-degenerate predictions (≥2 distinct answer letters)

## §9 Platform

- Hardware: Apple M5 Pro 48GB
- Framework: MLX + mlx-lm
- Skills invoked: /mlx-dev, /fast-mlx
- Expected runtime: ~45min (training ~25min + eval ~20min)
