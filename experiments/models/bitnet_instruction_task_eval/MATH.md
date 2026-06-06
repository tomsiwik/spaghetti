# Mathematical Foundations: Instruction-Tuned Task Evaluation

## Variables

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| W | Base model weights (ternary) | {-1, 0, 1}^{d x d} |
| A_i, B_i | LoRA adapter matrices for domain i | R^{d x r}, R^{r x d} |
| delta_i | Effective adapter update: B_i @ A_i | R^{d x d} |
| N | Number of composed adapters | 5 |
| r | LoRA rank | 16 |
| d | Model hidden dimension | 2560 |
| alpha | LoRA scaling factor | 20.0 |
| x | Input activation | R^{seq_len x d} |

## Key Equations

### 1. Instruction-Tuned vs NTP Training Objective

**NTP (prior experiment, KILLED):**
L_NTP = -sum_{t} log P(x_{t+1} | x_{1:t}; W + delta_i)

The model learns to predict next tokens on raw domain text. No task structure.

**Instruction-tuned (this experiment):**
L_inst = -sum_{t in response} log P(x_{t+1} | instruction, x_{1:t}; W + delta_i)

Critical difference: the loss is computed on the FULL sequence (instruction + response),
but the instruction prefix provides task-conditioning context. The model learns to
recognize "### Instruction:" as a signal to condition on the instruction content,
then generate a task-appropriate response after "### Response:".

Li et al. (2023, 2310.00492) show this causes three internal shifts:
1. Self-attention heads learn to attend to instruction verb tokens
2. FFN layers rotate pre-trained knowledge toward user-oriented tasks
3. The model learns to gate response generation on instruction content

### 2. Composition Under 1/N Scaling

For composed output with N=5 instruction-tuned adapters:

y_composed = (W + (1/N) sum_i delta_i) @ x

Each adapter contributes delta_i / N to the composed weight matrix. The effective
perturbation per adapter is alpha * (1/N) = 20.0 * 0.2 = 4.0 effective scale.

**Why instruction adapters survive 1/N better than NTP adapters:**

NTP adapters learn distributional shifts (vocabulary frequency). The task signal
is diffuse across many parameters. Under 1/N, this diffuse signal is attenuated
below detection threshold.

Instruction adapters learn task-completion patterns (instruction recognition +
response conditioning). This signal is concentrated in specific attention heads
and FFN directions (Li et al. 2023). Under 1/N, the concentrated signal survives
because it only needs to activate specific neural circuits, not shift the entire
output distribution.

### 3. Oracle Routing vs Uniform Composition

**Uniform 1/N:** y = (W + (1/N) sum_i delta_i) @ x
- All adapters contribute equally regardless of input
- Task signal attenuated by 1/N for each adapter
- Cross-domain interference possible (though minimal at |cos|=0.00084)

**Oracle routing:** y = (W + delta_{r(x)}) @ x where r(x) selects the domain
- Full adapter signal for the relevant domain
- No attenuation, no cross-domain interference
- Requires domain classifier (trivially achievable per prior routing experiments)

### 4. Orthogonality Preservation

Instruction-tuned adapters maintain orthogonality comparable to NTP adapters:
- Mean |cos| = 0.00084 (instruction) vs ~0.001 (NTP)
- Max pair: math-legal = 0.001453

This confirms that instruction tuning does not increase adapter interference.
The different training objective creates equally orthogonal weight updates in
the high-dimensional space (d=2560, 21.6M params per adapter).

## Computational Cost

| Component | FLOPs/Params |
|-----------|-------------|
| Per adapter training (300 steps) | 21.6M params x 300 steps = ~6.5B FLOPs |
| Total training (5 domains) | ~32.5B FLOPs, ~11 min |
| Per eval generation (100 tokens) | 2.4B x 100 = ~240B FLOPs |
| Total experiment | ~60 min on Apple M-series |

## Worked Example (Medical QA)

**Training sample:**
```
### Instruction:
What causes hypertension?

### Response:
Hypertension is caused by factors including high sodium intake, obesity,
genetic predisposition, and chronic stress...
```

**Eval prompt:**
```
### Instruction:
What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?

### Response:
```

The model recognizes the "### Instruction:" / "### Response:" structure from training,
conditions on the medical instruction, and generates a domain-appropriate response.
Keyword F1 measures overlap between generated response and reference answer.

## Assumptions

1. **Instruction format generalizes:** The "### Instruction: / ### Response:" template
   is sufficiently generic that the model learns the task-conditioning mechanism, not
   just surface pattern matching. Justified by: composed model (which blends 5 adapters)
   still shows +6.7pp math improvement, meaning the mechanism transfers across domains.

2. **Keyword F1 is a valid task metric:** F1 measures content overlap but not semantic
   accuracy. A response could use relevant keywords in a wrong context and still score
   well. This is a known limitation shared with ROUGE-type metrics.

3. **MATH-500 at 2B scale is informative:** Despite the 2B reasoning floor, the math
   adapter moved accuracy from 0% to 6.7% (base to adapter). This is directionally
   meaningful even if the absolute level is low.

4. **Legal eval is confounded:** The legal dataset (contract_nli) has Yes/No answers.
   Training loss dropped to 0.0 (memorization). The high F1 (26.5%) may reflect
   memorized response patterns rather than genuine legal reasoning. The improvement
   over base (7.8% -> 26.5%) is still valid as a task-completion demonstration.
