# MATH.md — Micro-Skill Detector Pool

## Hypothesis

A pool of N independent ~4K-parameter detectors (one per skill), each L1-resident
and implemented in NEON C, can replace the TF-IDF + Ridge router with comparable
accuracy at ~100× lower latency, while exposing per-skill confidence scores
that current centroid routing cannot produce.

## Theoretical grounding

### Generalization of L1-reflex
exp_pierre_l1_reflex_classifier proves a single 4KB classifier in L1 runs in
<10µs. N parallel detectors across 12 P-cores: at 50 detectors and 1µs each,
single-threaded sweep is 50µs; parallelized to 12 cores → ~5µs wall clock.

### Why detectors > centroids
TF-IDF+Ridge (F#431) scores prompts by lexical overlap with skill centroids.
At N=25 skills it reaches 86.1% but degrades on topically-adjacent skills
(economics ↔ statistics, history ↔ archaeology). Detectors learn pattern-level
features (phrase structure, code blocks, numerical density), not just word bags.

### Calibration via score distribution
TF-IDF gives one cosine score per skill — uncalibrated, often peaked near 1.0
on training distribution. Detector pool gives independent sigmoids: top-1 score
in [0,1], top-2 score in [0,1], gap is meaningful confidence signal.
F#171 routing survey identifies calibration as the missing piece in current
Pierre routing; this experiment tests whether per-skill detectors fix it.

## Predictions

1. **K1**: Pool routing accuracy ≥80% on N=10 skills. F#431 reached 96.6% at
   N=5, 86.1% at N=25. Detector pool target: 80% at N=10 — accepts some accuracy
   loss for the 100× latency improvement.

2. **K2**: Total memory ≤256KB. 50 detectors × 5KB each = 250KB. Fits L1 (128KB)
   + L2 (32MB on M5 Pro, plenty of headroom). Per-detector budget: 4K params
   fp32 + 1KB activations.

3. **K3**: Pool inference ≤100µs total. Single-detector inference 1-2µs (per
   talos benchmark). Sequential 50 detectors: ~75µs. Parallel via OpenMP/pthread
   across 12 cores: ~6µs. Conservative budget 100µs absorbs scheduling overhead.

4. **K4**: Calibration ρ ≥ 0.4. Spearman correlation between (top-1 score - top-2
   score) and routing correctness on held-out. F#188 showed energy-gap routing
   collapses without calibration; properly trained detectors should show
   meaningful gap when confident.

## Implementation plan

### Architecture
Each detector: same architecture as exp_pierre_l1_reflex_classifier (16-dim, 1-head,
2-layer microGPT) but trained as N=1 binary classifier per skill.

### Training data per detector
- **Positive class**: skill-specific prompts (50-200 per skill from beehive)
- **Negative class**: hard negatives (other skills' prompts) + easy negatives
  (random unrelated text)
- Cross-skill negative-mining: prompts that the SISTER detector for an
  adjacent skill scores highest on but actually belong to a different skill
- Total per detector: ~500 train + 100 val

### Pool inference structure
```c
// pool_infer.c — pseudocode
parallel_for (i = 0; i < N_DETECTORS; i++) {
    scores[i] = detector_forward(detectors[i], tokens);
}
top1 = argmax(scores);
top2 = argmax_excluding(scores, top1);
confidence = scores[top1] - scores[top2];
```
Each detector_forward is ~1µs single-threaded NEON; parallel_for is OpenMP across
P-cores. Total ~6-100µs depending on parallelism.

### Files (planned)
- `prepare_data.py` — split beehive by skill, generate per-detector train/val
- `train_pool.py` — train N detectors in MLX (parallel within-batch); export
- `pool_export.py` — pack N detector weights into single binary
- `pool_infer.c` — NEON C inference, OpenMP parallelization
- `bench.c` — N=10 vs N=25 vs N=50 latency + accuracy sweep
- `Makefile` — clang -O3 -march=native -ffast-math -fopenmp
- `run_experiment.py` — orchestration, KC computation

## Risks

1. **Per-skill data scarcity**: beehive has ~50 trajectories per skill on average
   — sufficient for ~4K param detectors, may overfit. Cross-skill augmentation
   recommended.

2. **Calibration vs accuracy trade-off**: training each detector with margin loss
   improves K4 (calibration) at some cost to K1 (raw accuracy).

3. **Cold-start for new skills**: when adding skill 51, you train a new detector
   AND need to add its negatives to existing detectors' training. Bootstrap cost
   per added skill: ~1 hour Python + recompile.

## Pre-registered KCs

K2130: Pool routing accuracy ≥80% on N=10 skill classification
K2131: Pool size ≤256KB total
K2132: Pool inference latency ≤100µs (parallel)
K2133: Calibration ρ ≥0.4 (Spearman of confidence-gap vs correctness)

## References
- exp_pierre_l1_reflex_classifier (binary version of this — train one, validate
  pattern, then scale to N)
- F#431 — TF-IDF baseline 96.6%
- F#171 — routing survey, calibration is missing
- F#188 — energy gap collapses without calibration
- ../talos-vs-macbook — NEON kernel reference
