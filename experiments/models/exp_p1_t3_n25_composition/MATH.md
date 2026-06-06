# MATH.md — T3.4: N=25 Domain Composition on Gemma 4 (Grassmannian Stress Test)

## V2 Audit Section (audit-2026-04-17-rerun, tautological-routing) — 2026-04-18

**V1 verdict retroactively invalid for TWO independent reasons.**

### Failure 1: Adapters missing on disk (precondition probe)

V1 PAPER.md claims rely on 5 real adapters at:
  - `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/`
  - `micro/models/exp_p1_t2_multi_domain_5/adapters/{legal,finance}/`

Direct filesystem verification (2026-04-18): every directory contains only
`adapter_config.json` stubs. Zero `adapters.safetensors` weights. All five
behavioral numbers in PAPER.md §"Phase 2" and §"Phase 3" are therefore
**unverifiable** at this audit horizon.

Upstream status:
  - exp_p1_t2_single_domain_training (T2.1) = **KILLED 2026-04-18**
    (metric-swap: DB-tracked K1030 = MedQA, measured MedMCQA;
    format-artefact: max_tokens=256 truncated Gemma 4 CoT).
  - exp_p1_t2_multi_domain_5 (T2.6) = supported but weights lost.

5th instance of class-level precondition-probe pattern this loop
(peer_comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed,
sft_residual_gemma4, this). Rule is standing.

### Failure 2: Tautological routing by design (pre-existing, audit-flagged)

run_experiment.py v1 hardcodes `REAL_ADAPTER_PATHS[domain]` for Phase 2/3:
each adapter is loaded exclusively for its matched domain's test set.
`eval_gsm8k(REAL_ADAPTER_PATHS["math"], ...)` etc. That is **not composition** —
it is single-adapter eval on a per-domain subset.

True composition under N=25 requires one of:
  (a) **Simultaneous activation** — all 25 adapters add into the forward pass,
      test whether any domain degrades (this is what T3.1 tested and killed
      for N=5; routing was posed as the fix).
  (b) **Per-sample routing** — a router R: input → domain decides which
      adapter fires, test set includes inputs from multiple domains mixed,
      accuracy measures routing quality × adapter quality jointly.

Neither happens in V1. `mem-antipattern-002` (tautological routing) applies:
headline accuracy is single-adapter accuracy by construction. The Theorem 3
"Zero Interference Under Exclusive Routing" proof is valid mathematically
but V1 doesn't *exercise* it — the routing function is `R(x) = ground_truth_domain(x)`.

### V2 structural invariants

- K1059 (Grassmannian orthogonality) is **pure math** on random numpy
  matrices, independent of adapter weights. V1 measurement 2.165e-8 is
  reproducible and genuine. V2 reruns this phase.
- K1060 (0/25 degraded under composition) is **structurally unmeasurable**
  without (i) adapter weights and (ii) either simultaneous activation or
  per-sample routing. Both preconditions fail.
- K1061 (MMLU preservation under composition) is **structurally
  unmeasurable** for the same reason; additionally, V1 measured
  per-domain-adapter-on-neutral-subject, which conflates MCQ-format-transfer
  with composition behavior.
- K1062 (25 adapters < 1 GB) is theoretical: per-layer size
  42 × (2560 × 6 + 6 × 2048) × 4 B (float32) = 4.6 MB per adapter.
  25 × 4.6 MB = 115 MB < 1 GB by formula. V2 reports this as
  theoretical-only; moot without real adapter weights.

### V2 Kill Criteria routing

| KC | Outcome | Reason |
|---|---|---|
| K1059 | PASS (genuine) | QR orthogonality is pure numpy math, reproducible |
| K1060 | FAIL | Adapters missing on disk + V1 design is tautological routing |
| K1061 | FAIL | Adapters missing on disk + V1 design conflates format transfer with composition |
| K1062 | PASS (theoretical, moot) | Formula holds but no real adapters to measure |

**Verdict: KILLED.** `all_pass=false`. No thresholds were changed — V2 keeps the
exact thresholds from V1 MATH.md and routes KCs honestly based on what is
measurable.

### Unblocker for V3

V3 of this experiment requires:
  1. T2.1 rebuild with MedQA USMLE 5-choice (DB KC #1030), max_tokens ≥ 512,
     adapter .safetensors persisted to disk, `adapters/code/` directory
     created.
  2. T2.6 adapters rebuilt or recovered from backup.
  3. run_experiment.py rewritten to exercise genuine composition:
     - Phase 2 option (a): load all 25 adapters simultaneously and measure
       per-domain accuracy under simultaneous activation.
     - Phase 2 option (b): implement a real router (e.g. TF-IDF +
       hidden-state ridge from T4.1) and measure cross-domain mixed
       test set accuracy (routing quality × adapter quality).
  4. Drop `REAL_ADAPTER_PATHS[domain]` hardcoded map from phase 2/3 —
     routing must be decided from input features, not domain labels.

Until those unblock, V3 cannot be claimed. Researcher MUST NOT auto-spawn.

---

## Problem Statement

T3.1 proved that simultaneous activation of N=5 LoRA adapters destroys math/code accuracy
(82→8%, 66→8%) via O(N) activation-space interference. T3.2/T3.3 confirmed that routing is
load-bearing and scale=6 is the safe operating point. This experiment asks:

**Can N=25 domains be composed with zero interference on Gemma 4 E4B?**

The structural guarantee requires two components:
1. **Grassmannian A-matrices**: mutual orthogonality eliminates weight-space interference
2. **Exclusive routing**: only one adapter fires per query, so activation-space interference = 0 regardless of N

## Architecture Context

Gemma 4 E4B (mlx-community/gemma-4-e4b-it-4bit):
- Hidden size: 2560
- q_proj: (2560, 2048)
- LoRA rank r = 6, scale = 6.0 (scale=6 safe from T3.2 Finding #426)
- N_layers = 42 (q_proj on all layers)

## Theorem 1: Grassmannian Mutual Orthogonality

**Theorem**: Given N ≤ ⌊d/r⌋ domains, there exist A-matrices {A_i ∈ ℝ^{d×r}}_{i=1}^N such that
A_i^T A_j = δ_{ij} I_r (exact, up to floating-point precision).

**Proof (Construction)**:
1. Sample W ~ N(0, 1)^{d × rN}
2. Compute Q ∈ ℝ^{d × rN} via thin QR: QR(W) = [Q, R], so Q^T Q = I_{rN}
3. Define A_i = Q[:, ir:(i+1)r] for i = 0, ..., N−1
4. Then (A_i^T A_j)_{kl} = e_{ir+k}^T (Q^T Q) e_{jr+l} = e_{ir+k}^T I_{rN} e_{jr+l} = δ_{ij} δ_{kl}
5. Therefore A_i^T A_j = δ_{ij} I_r □

**Frobenius cosine**: cos_F(A_i, A_j) = ||A_i^T A_j||_F / (||A_i||_F · ||A_j||_F)
- For i = j: A_i^T A_i = I_r → ||I_r||_F = √r; cos = 1
- For i ≠ j: A_i^T A_j = 0 → cos = 0 exactly

In float32 arithmetic: numerical error ≈ ε_float32 × √d ≈ 1.2×10^{-7} × 50.6 ≈ 6×10^{-6},
normalized by (√r)^2 = r = 6, gives max|cos| ≲ 10^{-6} << 1×10^{-5}.

**Prediction for K1059**: max |cos_F(A_i, A_j)| < 1×10^{-5} for all C(25,2)=300 pairs. ✓

## Theorem 2: Capacity Bound

**Theorem**: For d = 2560, r = 6: at most N_max = ⌊2560/6⌋ = 426 mutually orthogonal rank-r
subspaces exist in ℝ^d. N = 25 occupies 25/426 = 5.9% of capacity.

**Proof**: The Gram matrix of {A_i^T A_j} has rank ≤ d/r = 426, so at most ⌊d/r⌋ orthogonal
r-dimensional subspaces fit. 25 × 6 = 150 ≤ 2560 = d, so the QR construction succeeds. □

**Consequence**: N = 25 uses 14.3% of available capacity (150/1050 dimensions from q_proj A-side).
No capacity pressure at this scale.

## Theorem 3: Zero Interference Under Exclusive Routing

**Theorem**: Let R: X → {1,...,N} be a deterministic routing function assigning each input
to exactly one domain. Then for input x from domain k with R(x) = k:

    output(x) = x @ A_k @ B_k (single adapter, zero cross-domain interference)

**Proof**: Under exclusive routing, the adapted output at inference is:
    f_composed(x) = Σ_{i=1}^N 1[R(x)=i] · (x @ A_i @ B_i)
                  = x @ A_{R(x)} @ B_{R(x)}    (exactly one indicator = 1)
                  = x @ A_k @ B_k              (since R(x) = k)

Cross-domain terms: Σ_{i≠k} 1[R(x)=i] · (x @ A_i @ B_i) = 0 (no indicator fires). □

**Consequence**: Activation-space interference = 0 for correctly routed inputs, regardless of N.
This is the structural fix for T3.1's catastrophic failure (math 82→8% under simultaneous activation).

## Corollary: N=25 Composition is Interference-Free

Combining Theorems 1–3:
- Orthogonal A-matrices → weight-space cosines ≈ 0 (Theorem 1)
- Exclusive routing → activation-space interference = 0 (Theorem 3)
- 25 << 426 capacity bound → no rank-deficiency risk (Theorem 2)

Therefore: N=25 domain composition is mathematically interference-free.

## Quantitative Predictions

| Kill Criterion | Prediction | Mechanism |
|---|---|---|
| K1059: max\|cos\| < 1e-5 | **PASS**: ~1e-7 | QR construction, float32 precision |
| K1060: 0/25 degraded below base | **PASS**: 0/25 | Theorem 3 (exclusive routing); 20 synthetic B=0 domains trivially at base |
| K1061: MMLU >= base - 2pp | **PASS**: all >40% | Adapters enable MCQ format; T3.2 showed 62-77% on neutral MMLU |
| K1062: 25 adapters < 1GB | **PASS**: ~117MB | 25 × (2560+2048) × 6 × 2B × 42 layers ≈ 117MB |

## Size Calculation (K1062)

Per domain, per layer:
- A-matrix: (2560, 6) × 2 bytes (bf16) = 30,720 B
- B-matrix: (6, 2048) × 2 bytes (bf16) = 24,576 B
- Total per layer: 55,296 B ≈ 54 KB

Over 42 layers:
- Per domain: 55,296 × 42 = 2,322,432 B ≈ 2.2 MB

For 25 domains:
- Total: 2.2 MB × 25 = **55 MB** (uncompressed, bf16) << 1 GB ✓

Existing real adapters (float32): 15 MB each × 5 = 75 MB → still << 1 GB.

## Connection to Pierre P1 Architecture

This experiment validates the **Room Model** premise (VISION.md):
- W_combined = Σ_k δ_{k,R(q)} ΔW_k  (only matched adapter adds to combined weight)
- Routing IS the matmul: no separate router overhead
- 25 domains fit in 117 MB: viable for on-device deployment on M5 Pro 48GB

Grassmannian initialization is the structural prerequisite for interference-free scaling.

## References

- **HRA (arxiv 2405.17484)**: Householder Reflection Adaptation — structural orthogonality via product of r Householder reflections; same O(rd) parameter budget as LoRA
- **Finding #406**: N=25 PASS on Qwen3-4B with Gram-Schmidt A-matrices; max_cos=1.38e-5 (Finding #406 used K981: < 1e-4; this experiment tightens to < 1e-5 using float32 computation)
- **Finding #427**: Gemma 4 q_proj activation power law alpha=0.15; routing confirmed load-bearing
- **T3.1 (exp_p1_t3_pairwise_interference)**: Killed — simultaneous N=5 activation causes 82→8% collapse; routing is structural fix
- **T3.2 (exp_p1_t3_mmlu_preservation)**: Killed — scale≥12 degrades; scale=6 is safe; adapters give 62-77% on neutral MMLU
