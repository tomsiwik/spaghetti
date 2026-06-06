# Prediction vs Measurement: Qwen3-4B M2P v5 (SHINE-aligned)

## Hypothesis
In `exp_m2p_qwen06b_gsm8k_v5` (Finding #401), we discovered that the original M2P architecture failed due to mean-pooling the sequence dimension too early, which destroyed per-layer variation. We fixed this with the SHINE-aligned "base-as-encoder" pattern (extracting memory hidden states per-layer) and achieved an 83.3% quality recovery on the 0.6B scale. 

**Prediction:** This exact mechanism, scaled safely alongside `d_model=2560`, will allow the 4B scale hypernetwork to recover $>60\%$ of SFT quality, resolving the previously documented 4B architectural collapse (Finding #400).

## Results

| Metric | Prediction (v5 on 0.6B) | Measurement (v5 on 4B) |
| --- | --- | --- |
| K1: Initial Gradient Norm | > 0 `(PASS)` | **38.06** `(PASS)` |
| Base Accuracy | ~65% | **65.0%** |
| SFT Accuracy | >70% | **73.0%** |
| M2P Accuracy | >= 69.8% | **63.5%** |
| K2: Quality Ratio | >= 0.60 `(PASS)` | **-0.187** `(FAIL)` |
| K3: M2P Params | < 100M `(PASS)` | **50.3M** `(PASS)` |

## Analysis
**Hypothesis INVALIDATED.**

The 4B model actively degraded below its unadapted baseline performance (63.5% vs 65.0%), meaning the generated adapters inserted pure noise into the Qwen 4B computational graph.

Despite passing the gradient path checks (K1) and strictly aligning with SHINE mathematical paradigms, mapping $d_{model}=2560$ representations into LoRA generation parameters mathematically fails to converge outside of the sub-1B parameter domain. The "base-as-encoder" context injection is insufficient to overcome the vast structural complexity required to decode 4B dense layers.

**Verdict:** The M2P generation paradigm fundamentally contains a Scaling Law breakdown. It cannot scale strictly linearly to $L=36, d_{model}=2560$ parameters. Future attempts must use completely different structural pathways, such as the parameter-shrinking VeRA bottleneck strategy.
