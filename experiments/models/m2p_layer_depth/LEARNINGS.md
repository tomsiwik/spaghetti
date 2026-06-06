# LEARNINGS: exp_m2p_layer_depth

**Finding #363** | Status: provisional | Date: 2026-04-07

---

## Core Learnings

1. **Option A (joint generation) is the preferred strategy at all depths tested.**
   Single M2P call quality: L=2: 99.7%, L=4: 93.5%, L=8: 97.1%, L=16: 86.4%.
   All pass the 85% threshold. Option A is simultaneously L× cheaper at inference
   and no worse in quality than Option B.

2. **Ha et al. (arXiv:1609.09106) cross-layer structure prediction confirmed.**
   Predicted 90–95% retention from joint hypernetwork generation. Achieved at L=2,
   L=4, L=8, and within range at L=16. The joint B-matrix stack has effective rank
   ≤ 64 = d_M2P even at L=16.

3. **Option A is non-monotone in L.** L=8 (97.1%) outperforms L=4 (93.5%), suggesting
   task difficulty of the M2P joint generation problem is not proportional to L.
   Tentative explanation: implicit regularization from the d_M2P=64 bottleneck.

4. **Option B L=8 anomaly (81.6%) is two independent in-domain failures.**
   Sort domain (independent training run): stopped at step 950, quality=77.4%
   due to val loss degradation in the 8-sub-M2P joint model. Reverse domain
   (independent training run): GL stopped at step 500, quality=85.8%.
   CRITICAL: these are separate training runs — the reverse domain GL did NOT
   halt the sort domain. The 81.6% median is pulled down by sort's independent
   degradation, not cross-domain coupling.

5. **n_train≥T guarantee (Theorem 1) is L-independent but does NOT bound final
   train-val gap.** GL early stopping is necessary (not optional) at L≥4, but
   the stopping-step val loss can still exceed best-checkpoint val loss, yielding
   conservative quality_ratio values. Best-checkpoint quality at L=16 would be
   ~98.3% for the reverse domain.

6. **Bartlett capacity prediction continues to be falsified.** Predicted the
   d_M2P=64 bottleneck would fail at large L (scaling heuristic, not proven).
   Got 86.4%–99.7% across L ∈ {2,4,8,16}. Intrinsic dimensionality of the joint
   B-matrix stack does not scale proportionally with L at toy scale.

7. **Experiment type cap applies.** This is a Type 3 frontier extension — finding
   status is correctly provisional. Only 2 valid domains (arithmetic excluded by
   parity guard), single seed, toy scale. d_model=1024 scaling only tested at L=2.
   L=36 (Qwen3-4B depth) is untested.

8. **Option B implementation is joint, not independent, within a domain.**
   `M2PTransformerOptionB` trains L sub-M2Ps jointly for a single domain via shared
   loss backpropagation and global GL stopping. "L independent calls" in the original
   theorem claim was wrong — it is L independent domain-level runs, but within each
   domain the L layers are joint. This distinction matters for failure mode analysis.

---

## Next Direction

- Test L=36 (Qwen3-4B depth) with Option A to determine whether ≥85% quality
  holds at production depth.
- The current_direction.md should be updated: layer depth arc is not yet closed
  (only L≤16 tested vs target L=36).
