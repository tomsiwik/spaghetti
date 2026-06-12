## Core finding

Intra-adapter phase-timing — splitting a single LoRA's SVD into head (high-σ) and tail (low-σ) rank halves and scheduling which half fires per decode phase — is dead. The predicted schedule arm scored −2.5pp below uniform-math; the only arm matching the "win" (swap, 0.7625) is byte-identical to static head-only because the answer-emit phase barely fires before context ends.

## Why

Static rank-truncation (head-only-always) already reaches the maximum observed benefit: keeping top-σ directions and discarding low-σ tail recovered the gain over uniform without any timing logic. Because head-only-always and swap are 0/80 different, the phase boundary contributes nothing — temporal scheduling collapses onto static subspace selection. This is a rank-truncation effect, not a phase-gate effect.

## Implication for the next experiment

Do not propose any intra-adapter temporal routing scheme on a single adapter's SVD spectrum; the mechanism is inert. The positive side-result is that static rank-truncation of a self-sabotaging adapter (F#866: uniform −10pp vs base) claws back ~5pp — follow-up should target a healthier adapter or lower scale (F#627 math at scale 6.0 is a poor substrate) if rank-truncation is the hypothesis. Phase-timing across two structurally distinct adapters (cf. F#872) is a separate question and remains open.
