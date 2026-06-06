# E4: Activation Arithmetic Composition

## Type
Guided exploration — ActAdd framework proven for sentiment/toxicity steering (Turner et al. 2308.10248); unknown whether it generalizes to reasoning strategies on Gemma 4.

## Failure Mode
Contrastive activation steering produces no behavioral change (strategy signal too weak relative to residual stream norm), or steering degrades rather than improves reasoning accuracy.

## Prior Math
- **ActAdd** (Turner et al., arxiv:2308.10248): Adding activation difference vectors v = act(prompt+) − act(prompt−) at layer L steers generation behavior. Proven for sentiment, sycophancy on GPT-2/LLaMA.
- **RepE** (Zou et al., arxiv:2310.01405): Representation engineering — reading and controlling LLM internals via linear probes on hidden states. Demonstrates that high-level concepts (honesty, fairness) have linear representations.
- **Finding #801** (E1 killed): Mean-difference extraction captures format signal not strategy content (cos=0.99 across strategies). Root cause: neutral baseline lets format signal dominate. Fix: contrastive extraction (strategy_A − strategy_B) cancels shared format signal.
- **Finding #804** (E6 killed): Hedgehog attention-matching distills input processing, not generation behavior. Strategy effects manifest during generation, not input encoding.

## Hypothesis
Contrastive activation steering (ActAdd) at inference time can steer Gemma 4 E4B toward step-by-step decomposition, producing measurable GSM8K accuracy improvement over direct prompting.

## Mechanism (Atomic Level)

### Contrastive Vector Extraction
Given N problems P_1..P_N, two system prompts S_step ("Solve this step by step, breaking into sub-problems") and S_direct ("Give the answer directly and concisely"):

```
v_strategy(L) = (1/N) Σ_i [h_L(S_step, P_i) − h_L(S_direct, P_i)]
```

where h_L(S, P) is the hidden state at layer L after processing system prompt S with problem P, taken at the last token position.

The contrastive subtraction cancels:
- Format signal (instruction-following mode): shared between S_step and S_direct
- Problem-specific encoding: shared across both conditions for same P_i

What remains: the direction in activation space that differentiates "decompose" from "answer directly."

### Injection
At inference time on new problem Q with neutral prompt:
```
h_L'(Q) = h_L(Q) + α · v_strategy(L) / ‖v_strategy(L)‖
```

where α is a scalar injection strength. Normalizing prevents magnitude-dependent layer sensitivity.

### Why This Might Work (and What Breaks It)
**Works if**: strategy information is linearly encoded at some layer L, with sufficient signal-to-noise after contrastive cancellation. ActAdd demonstrates this for simpler behaviors.

**Breaks if**:
1. Strategy signal is distributed across layers (no single L captures it) → test with per-layer sweep
2. Strategy is nonlinear in activation space → contrastive subtraction recovers only linear component
3. α range where steering works without degrading coherence is empty → sweep α

### Prediction
- **Layer sweep**: best injection layer will be in middle layers (L ∈ [14, 28] of 42), where representations are abstract but not yet committed to specific tokens (consistent with ActAdd findings on GPT-2 middle layers).
- **Magnitude**: inter-strategy cosine similarity of contrastive vectors < 0.5 (E1 fix: contrastive cancels format signal).
- **Behavioral**: GSM8K accuracy with optimal (layer, α) will exceed baseline by >2pp, confirming strategy information is linearly steerable.

## Kill Criteria (Pre-Registered)

### K_struct (proxy): Inter-strategy discrimination
Contrastive vectors for different strategies (decomposition vs. analogy) have cosine similarity < 0.7.
If cos ≥ 0.7, contrastive extraction still conflates strategies (E1 failure persists despite contrastive fix).

### K2024 (target): Behavioral steering effect
ActAdd with optimal (layer, α) produces >2pp GSM8K accuracy change vs base model.
"Change" not "improvement" — even degradation proves steering works; zero effect means the vector carries no behavioral information.

### K2025 (target): Hybrid composition benefit
Weight-space domain (LoRA) + activation-space strategy (ActAdd) outperforms pure weight-space by >2pp on GSM8K.
Only testable if K2024 shows steering effect. If K2024 fails, K2025 auto-fails.

## Smoke Gate
- A1: Base GSM8K accuracy ≥ 15% (model loads correctly)
- A2: Contrastive vector extraction completes without OOM
- A3: At least one (layer, α) combination produces non-identical output vs base

## mlx-lm version
Targeting mlx-lm ≥ 0.31; API: `mlx_lm.load()`, `mlx_lm.generate()`.
