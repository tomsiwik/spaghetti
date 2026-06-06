# RunPod 10-Hour Task Queue

RTX A5000 (24 GB), $0.16/hr on-demand. Budget: ~$1.60 for 10 hours.
All scripts are PyTorch. HF_HOME=/workspace/hf_cache. Base: Qwen2.5-0.5B.

Run sequentially. Each task writes results to its own directory.

---

## Task 1: Gap-as-Signal Bridge (d=256, 20 seeds) — ~2 hours
**Priority: CRITICAL. This is the breakthrough.**

Already started locally. Re-run on RunPod for speed (20 seeds).

```bash
python tools/runpod_exec.py run macro/gap_as_signal_bridge/experiment.py --timeout 7200
```

- d=256, n_layer=6, n_head=8, ~2M params
- 7 cosine levels (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9) × 20 seeds
- Measure: gap magnitude, calibration speed, r²
- **Must prove**: r² > 0.3 at d=256 (micro was r²=0.74)
- Output: `macro/gap_as_signal_bridge/results_bridge.json`

## Task 2: Gap-as-Signal Phase 2 (real LoRA on Qwen2.5-0.5B) — ~3 hours
**Priority: CRITICAL. The paper result.**

```bash
python tools/runpod_exec.py run macro/gap_signal_lora.py --timeout 10800
```

Script must:
1. Load Qwen2.5-0.5B (d=896, 24 layers)
2. Fine-tune 5 rank-16 LoRA adapters on different domains:
   - Domain 1: Python code (subset of The Stack)
   - Domain 2: JavaScript code
   - Domain 3: Medical text (PubMed abstracts)
   - Domain 4: Legal text (EuroParl or similar)
   - Domain 5: Math (GSM8K training set)
3. Measure pairwise orthogonality (cos similarity of LoRA deltas)
4. For each pair: compose, measure gap magnitude, calibrate router, record steps
5. Benchmark: vs joint training, vs simple average, vs TIES, vs DARE
6. **Must prove**: composed < 5% worse than joint, gap predicts calibration speed
- Output: `macro/gap_signal_lora/results.json`, `macro/gap_signal_lora/PAPER.md`

## Task 3: SwiGLU Gate Pruning at Macro Scale — ~1 hour
**Priority: HIGH. Macro transfer was the known failure point.**

```bash
python tools/runpod_exec.py run macro/swiglu_pruning_macro.py --timeout 3600
```

Script must:
1. Load Qwen2.5-0.5B (uses SwiGLU: gate * up projection)
2. Fine-tune 3 LoRA adapters on different domains
3. Compose them
4. Profile |gate_output * up_output| per neuron
5. Prune at tau=0.05, 0.01, 0.001
6. Measure quality before/after pruning
7. Compare with random pruning baseline
8. **Must prove**: >10% prunable at <3% quality loss (micro was 66.5%)
- Note: Previous macro attempt failed (low activation = specialist, not dead).
  This time we use the SwiGLU gate-product signal, not activation magnitude.
- Output: `macro/swiglu_pruning_macro/results.json`

## Task 4: LoRA Orthogonality at Real Scale — ~1 hour
**Priority: HIGH. Validates the scaling math N_max ∝ d²/r².**

```bash
python tools/runpod_exec.py run macro/ortho_scaling.py --timeout 3600
```

Script must:
1. Load Qwen2.5-0.5B
2. Fine-tune 10 rank-16 LoRA adapters on different domains/subsets
3. Compute all pairwise cosine similarities of LoRA deltas (per-layer and global)
4. Verify: mean cosine ≈ r/√D (≈ 0.004 for r=16, D=13M per SwiGLU layer)
5. Test at r=4, r=8, r=16, r=32: does cos scale as predicted?
6. **Must prove**: random LoRA deltas are naturally near-orthogonal at d=896
- Output: `macro/ortho_scaling/results.json`

## Task 5: Consistent Hash Routing at N=20 — ~1 hour
**Priority: MEDIUM. Protocol scalability proof.**

```bash
python tools/runpod_exec.py run macro/hash_routing_scale.py --timeout 3600
```

Script must:
1. Load Qwen2.5-0.5B + 20 LoRA adapters (can use random init for routing test)
2. Implement consistent hash ring with 150 virtual nodes per expert
3. Add expert #21: measure displacement (should be ~4.8% = 1/21)
4. Measure quality: hash routing vs softmax routing
5. Measure: add latency, no-recal quality, 50-step recal quality
6. **Must prove**: <5% degradation without recalibration at N=20
- Output: `macro/hash_routing_scale/results.json`

## Task 6: Pre-Composition Pruning Pipeline at Macro — ~1 hour
**Priority: MEDIUM. Validates the contribution protocol.**

```bash
python tools/runpod_exec.py run macro/prune_compose_macro.py --timeout 3600
```

Script must:
1. Fine-tune 3 LoRA adapters on Qwen2.5-0.5B
2. Pipeline A: compose → prune → calibrate (baseline)
3. Pipeline B: prune each independently → compose → calibrate (protocol)
4. Compare final quality
5. **Must prove**: Pipeline B within 2% of Pipeline A (micro was +0.01%)
- Output: `macro/prune_compose_macro/results.json`

## Task 7: L2 Norm Stability at Macro — ~30 min
**Priority: MEDIUM. Safety check for hybrid attention.**

```bash
python tools/runpod_exec.py run macro/l2_norm_macro.py --timeout 1800
```

Script must:
1. Check if Qwen2.5-0.5B already uses L2-normalized QK (likely yes)
2. Compose 3 LoRA adapters
3. Run 20+ seeds, check for catastrophic failures (>10% degradation)
4. Compare with/without L2 norm if applicable
5. **Must prove**: 0% catastrophic failure rate (micro was 0/25 with L2)
- Output: `macro/l2_norm_macro/results.json`

---

## Execution Order & Time Budget

| # | Task | Est. Time | Cumulative |
|---|------|-----------|------------|
| 1 | Gap-as-Signal Bridge (d=256) | 2h | 2h |
| 2 | Gap-as-Signal LoRA (Qwen, 5 domains) | 3h | 5h |
| 3 | SwiGLU Gate Pruning Macro | 1h | 6h |
| 4 | LoRA Orthogonality Scaling | 1h | 7h |
| 5 | Consistent Hash N=20 | 1h | 8h |
| 6 | Prune-Compose Pipeline | 1h | 9h |
| 7 | L2 Norm Stability | 0.5h | 9.5h |
| | **Buffer** | **0.5h** | **10h** |

## If Tasks Finish Early

Bonus tasks in priority order:
- **Delta coding**: test LoRA v1→v2 delta storage at macro
- **N=100 expert library**: load 100 random LoRA adapters from NVMe, measure routing latency
- **LZ dictionary macro**: shared sub-expert dictionary across 10 LoRA adapters
