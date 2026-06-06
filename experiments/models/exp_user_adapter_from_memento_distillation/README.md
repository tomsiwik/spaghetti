# exp_user_adapter_from_memento_distillation

## Headline experiment
The user's product vision: **two-tier personalization.**
- Minute-scale: memento buffer (no training, serving-side) — from `exp_memento_cross_session_persistence`
- Day-scale: per-user LoRA adapter distilled from the accumulated memento buffer — this experiment

## Dependencies
- `exp_memento_gemma4_replication` (gate)
- `exp_memento_cross_session_persistence` (provides the memento buffer pipeline)

## Papers
- [arXiv:2604.09852](https://arxiv.org/abs/2604.09852) (MEMENTO) — mementos as training data source
- [arXiv:2604.14191](https://arxiv.org/abs/2604.14191) (Hedgehog recipe from Attention-to-Mamba) — reuse per-layer cos-sim loss to distill the "memento-rehydrated model" into the "memento-free model + user-adapter"

## Mechanism
1. Collect 50 sessions of mementos per user.
2. Build teacher = memento-rehydrated Gemma 4 E4B (session-aware context available).
3. Build student = memento-free Gemma 4 E4B + rank-6 LoRA.
4. Distill with per-layer cos-sim loss (Hedgehog) on user-typical prompts.
5. At inference, compose user-adapter with behavior/domain adapters.

## Privacy-by-construction (K5)
User-adapter weights should NOT allow exact reconstruction of training mementos. Many-to-one compression is structural privacy. Verified by white-box reconstruction attempt: given adapter weights, can an attacker recover any specific memento? If reconstruction error > threshold, weights are privacy-preserving.

## Reference code
- `microsoft/memento` — memento generation
- `HazyResearch/lolcats` — Hedgehog reference
- `pierre/core/compose.py` — existing runtime composition primitive

## Quick start
```bash
# After both MEMENTO parents are supported:
experiment claim <worker-id> --id exp_user_adapter_from_memento_distillation
experiment run exp_user_adapter_from_memento_distillation
```
