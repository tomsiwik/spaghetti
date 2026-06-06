# Code Audit: LZ Dictionary MoE

## 1. Vacuous Composition (Uniform Alpha Collapse)
In `lz_dictionary_moe.py`, `DictionaryExpert` defines its composition weights as:
```python
self.alpha_logits = mx.random.normal((n_dict,)) * 0.01
# ...
alpha = mx.softmax(self.alpha_logits)
```
Because `alpha_logits` are initialized near zero, `alpha` is initialized to a nearly uniform distribution ($\approx 1/D$). The model includes no sparsity-inducing regularization (like Gumbel-Softmax, L1 penalty, or sparsemax) to force experts to select specific dictionary entries. Consequently, the alphas never diverge meaningfully from uniform during training. The diagnostic `normalized_entropy ~ 0.999` mathematically proves that every expert is just uniformly averaging the entire dictionary.

## 2. MoE Bypass via Shared Sub-network
Because all experts learn the exact same uniform weighting over the dictionary, the dictionary portion of the expert becomes a constant shared trunk:
```
expert_i(x) ≈ shared_dict(x) + delta_i(x)
```
When the MoE layer computes the routed output:
```
output = sum(w_i * expert_i(x))
       = sum(w_i * (shared_dict(x) + delta_i(x)))
       = shared_dict(x) * sum(w_i) + sum(w_i * delta_i(x))
```
Since `sum(w_i) = 1` for routed tokens, the entire dictionary block (which contains the vast majority of the layer's parameters) completely bypasses the MoE routing mechanism. The model is actually functioning as a Dense GPT with a tiny MoE residual (`delta_i`). This is not a Dictionary MoE; it's a dense model with a tiny rank-16 MoE grafted on.

## 3. O(N) Redundant Dictionary Computation
In `DictionaryMoELayer.__call__`:
```python
for i, expert in enumerate(self.experts):
    w = masked_probs[..., i:i+1]
    out = out + w * expert(x, self.dictionary)
```
Each `expert` computes the full dictionary forward pass over all tokens `x` independently:
```python
for j in range(self.n_dict):
    out = out + alpha[j] * dictionary[j](x)
```
This means `dictionary[j](x)` is recomputed from scratch $N$ times per layer (where $N$ is the number of experts). While this is an efficiency bug rather than a correctness bug, it highlights that the implementation fails to exploit the promised "shared codebook" efficiency of a dictionary-based architecture.

## 4. Weak Baseline Comparison
In `run_experiment.py`, the experiment is designed to compare against a Standard MoE baseline. However, as noted in the findings, the standard MoE baseline is weaker than the `Dense GPT` baseline. Measuring performance degradation relative to a sub-standard baseline creates a false sense of success (or a falsely permissible kill threshold of ">3% worse than independent experts"), ignoring that the entire architecture is underperforming a simpler dense model.
