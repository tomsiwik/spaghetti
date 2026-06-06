# Synthesized Findings & Follow-up: LZ Dictionary MoE

## Final Verdict: Invalid
The original experiment is structurally and theoretically invalid. 
1. **Vacuous Composition (MoE Bypass):** The dictionary weights (`alpha`) are initialized uniformly and lack any sparsity-inducing regularization. Consequently, the normalized entropy of ~0.999 indicates a complete collapse into a shared dense trunk, bypassing the MoE routing entirely.
2. **Weak Baseline:** The Standard MoE baseline (596K params) underperforms a smaller Dense GPT baseline (202K params), making any relative performance gains meaningless.
3. **Flawed Mathematics:** The theoretical rank analysis incorrectly commutes the sum and ReLU operations, treating non-linear capacity as linear.
4. **Redundant Computation:** The dictionary forward pass is recomputed independently for every expert, an O(N) efficiency bug that defeats the purpose of a shared dictionary.

## Follow-up Experiment Design

### Hypothesis
If we enforce strict sparsity on the dictionary composition weights (e.g., using Sparsemax or Top-K routing over the dictionary entries) and fix the redundant computation, then the Dictionary MoE architecture can achieve a better perplexity than a strictly parameter-matched Dense GPT baseline, proving genuine MoE routing and composition.

### Math Sketch
Instead of a uniform softmax over `alpha`:
$$ \alpha_i = \text{Sparsemax}(\text{logits}_i) $$
or
$$ \alpha_i = \text{TopK}(\text{logits}_i, k) $$
This ensures $\sum \alpha_{i,j} = 1$ but with most entries exactly zero. The expert output becomes a sparse combination of dictionary entries:
$$ \text{expert}_i(x) = \sum_{j \in \text{active}} \alpha_{i,j} \cdot \text{ReLU}(W^{\text{down}}_j x) W^{\text{up}}_j $$
This preserves the non-linearity appropriately and prevents the "shared trunk" collapse.

### Kill Criteria
The follow-up experiment will be killed (deemed a failure) if ANY of the following occur:
1. **Fails to beat Dense GPT:** The model's validation loss is not strictly better than a Dense GPT model with the exact same number of active and total parameters.
2. **Entropy Collapse:** The normalized entropy of the dictionary usage remains > 0.90, indicating a failure of the sparsity mechanism to induce specialized dictionary usage.
3. **O(N) Recomputation:** The dictionary outputs are not computed once and cached/shared across experts during the forward pass.