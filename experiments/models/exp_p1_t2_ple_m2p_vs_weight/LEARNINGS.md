# LEARNINGS.md — T2.4: PLE Injection vs Weight Modification

**Status: KILLED** | Finding #446

## Core Finding

Random-projection PLE injection cannot match LoRA quality: QR_frozen=−5.89, QR_full=−4.58 (threshold 0.85). Both conditions made the model worse than the uninstructed base after 300 steps.

## Why

Random W_proj creates an injection Δh = RMSNorm(W_proj v) with expected norm O(sqrt(d)), matching the scale of the hidden state h itself. This corrupts all 28 layers simultaneously — the model receives signal-scale noise at every layer. No e_l tuning can fix this; the damage is structural in W_proj. Theorem 1 (JL-lemma) proved projection distances are preserved but did NOT bound the quality of the perturbed residual stream.

Ref: Li et al. (arXiv:1804.08838), Aghajanyan et al. (arXiv:2012.13255)

## Durable Results

- M2P generation latency: 0.182ms (PASS, architecture-level result)
- PLE forward overhead: 1.35× LoRA (PASS, structural bound)
- K1042: Gradients flow (70.8% loss decrease), confirming the mechanism works but starts from catastrophic init

## Impossibility Structure

Random W_gate/W_proj → O(sqrt(d)) injection → SNR = O(1) at every layer → O(d) steps needed to align W_proj to task subspace. 300 steps is ≪ O(d). This is structural, not a budget problem.

## Implications for Next Experiment

Use Gemma 4's NATIVE MLP as pretrained projection: freeze W_gate/W_proj from Gemma 4's own MLP layers, train only e_l. T0.5 already validated this pattern on Qwen3-0.6B (Finding #416: 81.7% loss reduction with 128 trainable params). This experiment should achieve QR ≥ 0.85.
