# Delta Compression for Model Weights

## Related Work (discovered via NotebookLM research)

### BitDelta (2024)
- **Paper**: https://arxiv.org/abs/2402.10193
- **Key insight**: Quantize weight deltas (fine-tuned - base) to 1 bit.
  10x+ memory reduction with minimal quality loss.
- **Relevance**: Complementary to our SVD compression. BitDelta compresses
  base-to-fine-tuned deltas; our work compresses version-to-version deltas.
  Could potentially combine: LoRA (spatial compression) -> BitDelta (bit-level)
  -> SVD (temporal/version compression).

### DeltaZip (2024)
- **Paper**: https://arxiv.org/abs/2312.05215
- **Key insight**: System for serving multiple fine-tuned LLMs by compressing
  deltas up to 10x. Uses mixed-precision quantization on weight differences.
- **Relevance**: Production-oriented delta compression. Validates that model
  weight deltas are highly compressible in practice. Our video codec analogy
  extends this to VERSION CHAINS (their work is single-shot base->fine-tuned).

### FedFQ (2024)
- **Paper**: https://arxiv.org/abs/2408.08977
- **Key insight**: Fine-grained quantization of model updates in federated learning.
- **Relevance**: Validates delta compression in federated/distributed settings.

### DeltaDQ (2024)
- **Paper**: https://arxiv.org/abs/2410.08666
- **Key insight**: Ultra-high delta compression for fine-tuned models.
- **Relevance**: Pushes compression ratios further than BitDelta.

## Our Novelty

None of these works apply the video codec I-frame/P-frame/GOP framework to
model weight versioning. Our contribution is the TEMPORAL extension:
- Prior art: compress delta(base, fine-tuned) -- one-shot
- Our work: compress delta(version_t, version_{t+1}) -- sequential chain

The video codec analogy (keyframe interval K, chain drift, GOP structure)
is novel and provides a principled framework for version management.
