# MATH.md — P11.Z0: Plan-and-Solve Prompting on MMLU-Pro

## Background

Wang et al. (2023) arXiv:2305.04091 identify two failure modes in zero-shot chain-of-thought (CoT) reasoning:
1. **Step-skip error**: the model skips intermediate steps, jumping from problem statement to answer
2. **Semantic misunderstanding**: the model misidentifies what the problem is asking

Their fix: prepend an explicit planning instruction before the solve instruction.

## Theorem

**Theorem (Plan-and-Solve Decomposition)**: Let Q be an MMLU-Pro question with `n` required reasoning steps. Let `P_direct` denote the prompt "Answer with ONLY the letter" and `P_PS` denote the Plan-and-Solve prompt. If the model's response token sequence has expected length `E[|response|]` and reasoning completeness `c ∈ [0,1]`, then:

Under `P_direct`:
- Expected completeness: `c_direct = min(1, β · E[|response|])` where β captures step density
- Short-response pressure (max_tokens=16) truncates most intermediate steps

Under `P_PS` (thinking enabled):
- The planning instruction explicitly activates step enumeration in the thinking trace
- `c_PS ≥ c_direct` because planning reduces premature commitment to an answer

**Formal claim**: For multi-step MCQ with n ≥ 2 reasoning steps, the probability of correct answer satisfies:

```
P(correct | P_PS, thinking=True) ≥ P(correct | P_direct, thinking=True)
```

The inequality is strict when the model's thinking budget is sufficient to enumerate sub-tasks.

**Why**: 
- Wang et al. show PS outperforms zero-shot-CoT on GSM8K (+4.5pp), AQuA-RAT (+3.1pp), MATH (+1.3pp)
- MMLU-Pro is harder than MMLU (avg 10 options vs 4), requiring more reasoning steps
- With thinking enabled (Gemma 4 2M context), the planning instruction seeds the reasoning trace, not the output

## Predictions

| Metric | Prediction | Source |
|--------|-----------|--------|
| P_direct + thinking MMLU-Pro | ~62.1% | Finding #530 |
| P_PS + thinking MMLU-Pro | ≥ 64.1% (+2pp) | K1529 |
| P_PS+ ≥ P_PS | True (self-check adds +0.5–2pp) | Wang et al. Table 4 |
| Token count increase | < 2× vs P_direct | K1531 |

## Prompts Under Test

**P0 (baseline, direct-answer)**:
```
The following is a multiple choice question. Answer with ONLY the letter of the correct option (A through {max}). Do not explain.

Question: {Q}
Options: {opts}
Answer:
```

**P1 (Plan-and-Solve)**:
```
The following is a multiple choice question. Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step by step to get the correct answer.

Question: {Q}
Options: {opts}
Answer:
```

**P2 (PS+ with self-check)**:
```
The following is a multiple choice question. Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan, paying attention to not miss any calculations, and double-check the answer.

Question: {Q}
Options: {opts}
Answer:
```

## Kill Criteria

- **K1529 PASS**: Best prompt + thinking ≥ 64% MMLU-Pro (≥ 2pp over 62.1% baseline)
- **K1530 PASS**: PS+ ≥ PS accuracy (self-check instruction adds value)  
- **K1531 PASS**: Best prompt output token count ≤ 2× direct-answer count

## Failure Mode

If PS prompt KILLS: the planning instruction increases context without triggering better reasoning — the model ignores the planning prefix and answers directly. This would indicate the model's thinking mechanism already handles step decomposition implicitly.

## Connection to Architecture

Plan-and-Solve is a **zero-cost** improvement: no training, no adapter, no memory overhead. If it yields ≥ 2pp MMLU-Pro gain, it becomes the default prompt for all benchmark evaluations in P11, including the reasoning SFT experiments (s1K, LIMO). This directly improves the behavioral baseline before any training.

## QED

The theorem makes a directional prediction: PS ≥ direct. The experiment verifies whether the improvement is ≥ 2pp on Gemma 4 E4B with thinking enabled. Wang et al.'s results on smaller models provide a lower bound; the thinking mechanism should amplify the gain.
