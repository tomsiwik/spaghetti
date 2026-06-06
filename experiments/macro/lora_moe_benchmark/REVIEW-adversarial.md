# Peer Review: 5-Domain LoRA MoE Benchmark

## Mathematical Soundness

### The "+7.22% vs joint" framing is misleading

The PAPER.md conclusion states: "LoRA MoE with learned router achieves +7.22% vs joint training." The "+" prefix universally means "better than." In this experiment, +7.22% means MoE loss is 7.22% **higher** (worse) than joint training. The conclusion should read "-7.22% vs joint" or "+7.22% degradation vs joint" or simply "7.22% worse than joint."

The FINDINGS.md gets this right: "MoE matches individual experts and TIES, but doesn't close the gap to joint training." The PAPER.md obscures this.

**Seed-level verification:**

| Seed | MoE vs Joint (%) | Consistent? |
|------|-------------------|-------------|
| 42   | +5.73% (worse)    | Yes         |
| 43   | +8.40% (worse)    | Yes         |
| 44   | +7.54% (worse)    | Yes         |

Mean: +7.22% worse. Standard deviation: 1.34pp. All three seeds agree: MoE is consistently worse than joint training. The math checks out; the presentation does not.

### Loss averaging across domains is misleading when domains have wildly different scales

Medical losses are ~3.5-3.8 across all methods. Math losses are ~0.17-0.34. Averaging these treats a 0.01 improvement on math (where loss is 0.17) the same as a 0.01 improvement on medical (where loss is 3.5). The medical domain dominates the average and hides meaningful differences in other domains.

A proper comparison would use either:
1. Per-domain relative improvement, then average those percentages
2. Loss normalization (divide each domain by its base model loss)

## The Medical Domain Is Broken

This is the most critical finding in the data. Across all 3 seeds:

| Seed | Base Medical | Expert Medical | Joint Medical | MoE Medical |
|------|-------------|----------------|---------------|-------------|
| 42   | 3.456       | 3.779          | 3.466         | 3.772       |
| 43   | 3.370       | 3.833          | 3.345         | 3.817       |
| 44   | 3.410       | 3.868          | 3.384         | 3.846       |

**The medical LoRA expert is worse than the base model in every seed.** Fine-tuning on PubMedQA `long_answer` field for 300 steps actually increases loss by 0.3-0.5 points. This means:

1. The "expert" for medical has learned nothing useful -- it has negative transfer
2. The MoE inherits this broken expert and routes tokens to it
3. The 5-domain average is polluted by a domain where training fundamentally failed

**Root cause:** PubMedQA `long_answer` entries are typically short explanatory paragraphs. Training a causal LM on these short, heterogeneous snippets for 300 steps likely produces a worse model than the pretrained base which already saw medical text during pretraining. The domain-specific signal is too weak relative to the forgetting induced by fine-tuning.

**Impact on conclusions:** If we exclude medical, the MoE vs joint gap likely changes substantially. The experiment should report 4-domain results alongside 5-domain results, or fix the medical training.

## Experimental Design Flaws

### Flaw 1: The "legal" domain is GSM8K answer text

Line 51 of the script:
```python
"legal": {"dataset": "openai/gsm8k", "name": "main", "field": "answer", "split": "train"},
```

This is GSM8K math word problem answers, not legal text. The "math" domain uses GSM8K questions. So the experiment has two domains from the same dataset (questions vs answers), which means:
- These two "domains" share vocabulary, structure, and topic
- Expert orthogonality between "legal" and "math" is artificially low
- The router has an easy time distinguishing question format from answer format, but this tests format discrimination, not domain routing

This invalidates the claim of testing "5 diverse domains." The experiment actually has: 2 code domains (synthetic), 1 failed medical domain, and 2 halves of the same math dataset.

### Flaw 2: Python and JS used synthetic fallback data

FINDINGS.md acknowledges this: "Python/JS used synthetic fallback (bigcode/the-stack-smol is gated)." The synthetic templates are trivial:

```python
"python": "def {f}(x):\n    return x + {n}\n"
"javascript": "function {f}(x) {{ return x * {n}; }}\n"
```

These are single-line function templates with random integers. Training a 0.5B model on these does not test code domain expertise. The low expert losses on Python (0.43) and JS (0.38) likely reflect that these templates are trivially memorizable, not that the experts learned useful code representations.

### Flaw 3: 300 expert steps vs 600 joint steps is unfair, but in the wrong direction

Joint training gets 600 steps cycling through 5 domains = 120 steps per domain. Individual experts get 300 steps on one domain. So experts see **2.5x more domain-specific data** than joint training sees per domain. Despite this advantage, MoE still loses to joint by 7.22%.

This actually makes the result **worse** for the MoE thesis: even with 2.5x more domain-specific training per expert, composition cannot match joint training. A fair comparison would give both approaches equal per-domain exposure.

### Flaw 4: The router is a domain classifier, not a token-level router

Lines 376-380 reveal the router training:
```python
domain_idx = domains.index(domain)
target_expert = torch.full((BATCH_SIZE,), domain_idx, dtype=torch.long, device=DEVICE)
router_loss = F.cross_entropy(gate_avg, target_expert)
```

The router is trained as a **batch-level domain classifier** using cross-entropy against domain labels. It learns to map "this batch came from domain X, so activate expert X." This is fundamentally different from a token-level MoE router that learns which expert produces the best output per token.

Implications:
1. The router cannot route mixed-domain inputs (e.g., code with medical comments)
2. At evaluation time, each domain's validation data is homogeneous, so the classifier approach "works" -- but this tests domain ID, not expert selection quality
3. The comparison to Mixtral/DeepSeek-style MoE is invalid since those systems route per-token based on learned representations, not domain labels

### Flaw 5: Router training uses detached expert logits

Lines 357-369 show that expert logits are computed with `torch.no_grad()` and gate weights are detached:
```python
mixed_logits = mixed_logits + w.detach() * elogits
```

Then the actual loss is a classification loss (Flaw 4). This means the router never receives gradient signal about which expert actually produces better predictions. A proper differentiable MoE would backpropagate through the gated expert outputs.

### Flaw 6: 200 router steps is undertested

200 training steps for the router is asserted to be sufficient, but no learning curve or convergence analysis is provided. Given that the router is a simple classifier on a 5-class problem with clear domain boundaries (synthetic code vs math text), 200 steps is probably enough for *this* easy task. But this says nothing about router convergence for real, overlapping domains.

## Latency Analysis

The 3x latency overhead (monolithic 32ms vs MoE 133ms) comes from the implementation doing **sequential** `set_lora_state()` calls for each expert (lines 446-459). Each call modifies model weights in-place and runs a full forward pass. This is purely an implementation artifact -- a proper MoE implementation would:

1. Pre-compute base model hidden states once
2. Apply LoRA deltas as additive low-rank operations (r=16, so this is cheap)
3. Use batched matrix operations for top-k experts simultaneously

VISION.md claims "base + 0.06% overhead" at k=2, r=16, d=896. The 317% measured here contradicts this theoretical claim. The gap between theory (0.06%) and measurement (317%) spans three orders of magnitude and is entirely attributable to the naive implementation, but the PAPER.md reports the naive measurement as the result.

## Novelty Assessment

This experiment tests a standard LoRA MoE setup. The architecture (frozen base + LoRA experts + learned router) is essentially:
- **LoRAHub** (Huang et al., 2023): learned composition of LoRA adapters
- **MoE-Adapters4CL** (Yu et al., 2024): MoE over adapters for continual learning
- **BTX** (Meta, 2024): branch-train-mix with MoE routing

The delta over prior work would be the crowd-compute contribution protocol (train experts independently, compose later). But this experiment doesn't test that -- it trains all experts from the same machine, same codebase, same hyperparameters.

## Hypothesis Graph Consistency

This experiment maps to `exp_gap_signal_macro` in HYPOTHESES.yml, which has kill criteria:
1. "gap-calibration correlation r^2 < 0.3 at d=896" -- not tested here
2. "5-domain composition >5% worse than joint training after calibration" -- **TRIGGERED: 7.22% worse**
3. "calibration requires >500 steps" -- 200 steps used, so this passes

Kill criterion 2 is triggered. The MoE is 7.22% worse than joint, exceeding the 5% threshold. If this experiment is mapped to that hypothesis node, the node should be marked as killed or the criterion should be updated with justification.

## Macro-Scale Risks (advisory)

1. **The medical domain failure will recur** with any domain where the base model is already strong. Qwen2.5-0.5B likely saw medical text in pretraining; a short LoRA fine-tune on noisy data can only hurt.

2. **Synthetic data for 2 of 5 domains** means the experiment provides no evidence about real-world code composition. Until run with actual code data, the code expert results are meaningless.

3. **The domain-classifier router** will not scale to N=100 experts with overlapping domains. Real deployment needs token-level routing that discovers specialization, not batch-level domain labels.

4. **The sequential expert evaluation** in the latency benchmark will scale linearly with k. At k=4 with 100 experts, this implementation would be 12x slower than monolithic.

## Verdict

**REVISE** -- the experiment has sound infrastructure but multiple design flaws that invalidate its conclusions.

### Required Fixes

1. **Fix the medical domain.** Either (a) increase training steps to 1000+ with a learning rate warmup, (b) use a medical dataset with longer, more coherent documents (e.g., MIMIC discharge summaries, medical textbooks), or (c) drop it and acknowledge 4 domains.

2. **Replace the fake "legal" domain.** Use an actual legal dataset (e.g., legal opinions, contracts, statutes). Two domains from GSM8K is methodologically invalid for testing multi-domain composition.

3. **Use real code data or acknowledge synthetic.** If bigcode/the-stack-smol is gated, use an alternative (e.g., CodeParrot, The Pile's GitHub subset, or manually downloaded Python/JS files). The PAPER.md must prominently disclose if synthetic data was used.

4. **Fix the PAPER.md framing.** "+7.22% vs joint" must be rewritten as "7.22% degradation vs joint training." The conclusion should honestly state that MoE does not close the gap to joint training.

5. **Equalize training compute.** Give joint training 5x300=1500 steps (300 per domain) or give experts 120 steps each (matching joint's per-domain budget). Currently the comparison is apples-to-oranges.

6. **Report per-domain results in PAPER.md.** The aggregate average hides the medical catastrophe and the GSM8K domain overlap. Add a per-domain table.

7. **Add a "MoE without medical" row.** Show the 4-domain comparison to isolate the impact of the broken domain.

8. **Reconcile with HYPOTHESES.yml kill criteria.** Kill criterion 2 of `exp_gap_signal_macro` (">5% worse than joint") is triggered. Either update the hypothesis node status or explain why 7.22% is acceptable.

9. **Note the latency measurement is implementation-bound, not architectural.** Add a theoretical latency calculation (base forward + k * 2 * r * d FLOPs for LoRA application) alongside the measured number. The 317% overhead is fixable; report both.

### Non-blocking Recommendations

- Train the router end-to-end through expert logits (REINFORCE or straight-through estimator) rather than as a domain classifier. The current approach tests domain ID, not routing quality.
- Add router accuracy metrics: what fraction of tokens get routed to the "correct" domain expert?
- Run with at least 5 seeds for publication-quality results.
