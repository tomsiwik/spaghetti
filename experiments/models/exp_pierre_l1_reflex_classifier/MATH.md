# MATH.md — Pierre L1-Reflex Classifier

## Hypothesis

A microGPT-scale (~4K param, ~16KB fp32) binary classifier, implemented in C with
NEON intrinsics and resident in L1 D-cache (128KB on M5 Pro), can decide in <10µs
whether an incoming query needs Pierre's adapter stack or can be handled by the
base model alone. Properly gated, this saves ~190ms of avoidable inference latency
on the ~30% of mixed-traffic queries that don't need Pierre.

## Theoretical grounding

### microGPT scale (Karpathy 2024 / talos-vs-macbook)
A 16-dim, 1-head, 2-layer transformer with 27-token vocab has ~4,192 parameters.
At fp32 = 16KB. Forward pass = ~4,000 multiply-accumulates per token. At single-core
NEON throughput (~16 GFLOPS scalar; much more with FMA pipelines), arithmetic
takes <<1µs. The bottleneck is dispatch overhead.

NumPy / MLX / Python all add per-call overhead in the µs-to-tens-of-µs range,
so they can't beat a Cyclone V FPGA on this workload (which dispatches in tens
of nanoseconds). Hand-tuned C with NEON beats the FPGA by ~70× because the M5
P-core has higher arithmetic throughput.

**Citation:** talos-vs-macbook (../talos-vs-macbook/README.md). 4K-param model,
~71× FPGA throughput in NEON C, ~28× perf-per-watt.

### Why this is "talos-shaped" not "Pierre-shaped"
Pierre's main forward pass is Gemma 4 E4B (~4B params, 2GB at 4-bit). L1-cache
optimization is irrelevant — the workload is bandwidth-bound on the unified memory,
not dispatch-bound. The talos insight only applies to **gating decisions** that
sit BEFORE the main inference. A reflex classifier is exactly that.

### Failure mode prevented
Pierre invocation latency is ~190ms p95 (measured exp_pierre_phase1_e2e_viability:
K2105 PASS at 187ms). For trivial queries ("hi", "2+2", "what time is it") the
adapter stack adds zero value but still pays the full latency cost. Reflex
classifier prevents this by routing such queries to base-only inference.

## Predictions

1. **K1**: Binary classifier accuracy ≥75% on (needs_pierre vs base_suffices).
   Threshold from prior text-classification baselines on similar size models.

2. **K2**: Inference latency ≤10µs single-core NEON. Math:
   - 4,000 MACs at ~16 GFLOPS scalar fp32 → 250ns arithmetic
   - Memory load + RMSNorm + sample = ~10× arithmetic = ~2.5µs
   - Conservative budget: 10µs. talos-vs-macbook hit ~280ns/token in NEON C
     for microGPT, so 10µs is safe with overhead.

3. **K3**: Total memory ≤64KB.
   - Weights: 16KB fp32 (4K params)
   - Tokenizer table: 1KB (256 chars × 4 bytes)
   - Activations: 16-dim × 16 layers × 4 bytes = 1KB
   - KV cache: same ≈ 1KB
   - Total ~20KB. 64KB budget is loose.

4. **K4**: P95 first-token latency drops from 190ms (full Pierre) to 80ms
   when reflex routes to base. Math: base-only forward ≈ 70ms (Gemma 4 E4B
   measured); reflex adds <1ms; total ~71ms; p95 buffer to 80ms.

5. **K5**: False-negative rate ≤10%. Skipping Pierre when needed is a quality
   regression; over-invoking Pierre is just compute waste. Asymmetric KC.

## Implementation plan

### Architecture
```
prompt → byte-level tokenization → 16-dim embedding lookup
       → 2× transformer block (16-dim, 1-head, RMSNorm, GELU)
       → mean-pool over tokens
       → linear → sigmoid → P(needs_pierre)
```

### Files (planned)
- `train.py` — Python/MLX training of weights, exports as packed binary
- `model.h`/`model.c` — C implementation with NEON intrinsics (modeled on
  ../talos-vs-macbook/bench_c.c)
- `weights.bin` — packed fp32 weights, ~16KB
- `bench.c` — single-binary benchmark: load weights, run N forwards, report p95
- `run_experiment.py` — Python harness that calls bench.c via subprocess,
  collects accuracy + latency, writes results.json

### Training data
- **Positive class (needs Pierre)**: beehive prompts (2069 from Turso snapshot),
  GSM8K test prompts (multi-step), HumanEval prompts (multi-step code), MedQA
  (multi-step reasoning). Total: ~3000 prompts.
- **Negative class (base suffices)**:
  - LMSys-Chat-1M short single-turn conversational
  - Trivial Q&A: "what's 2+2", greetings, single-fact lookups
  - 1-2 sentence simple-fact prompts
  - Total target: ~3000 prompts.
- Held-out 20% per class for K1 measurement.

### NEON kernel structure (per talos pattern)
Each (R, 16) · (16,) matmul: load 16 input floats into 4× float32x4_t registers,
loop over R rows, vfmaq_f32 + vaddvq_f32 horizontal reduce. (16, 64)·(64,)
fully unrolled. RMSNorm via vaddvq_f32. Sample via xorshift32 — but actually
we don't sample, we sigmoid-classify, so no sampling needed.

### Build
```
clang -O3 -march=native -ffast-math -o bench bench.c model.c
```

## Risks and unknowns

1. **Boundary cases**: queries that AMBIGUOUSLY need Pierre (some sub-questions
   trivial, others complex). Current binary framing is too coarse — may need
   3-class ('definitely Pierre', 'maybe Pierre', 'definitely base').

2. **Distribution shift**: training data from beehive + benchmarks may not
   represent real product traffic. Empirical calibration on real usage is
   downstream work.

3. **Async invocation**: in production, reflex runs sync before main forward
   begins. If reflex itself takes 5ms (10× our budget), we lose half the win.
   Profiling discipline matters.

## Pre-registered KCs

K2125: Reflex classifier ≥75% accuracy on Pierre-needed-vs-base-suffices binary
K2126: Classifier inference latency ≤10µs single-core NEON
K2127: Total memory ≤64KB (fits L1 with headroom)
K2128: End-to-end Pierre latency p95 ≤80ms when reflex routes to base
K2129: False-negative rate ≤10%

## References

- ../talos-vs-macbook/README.md — methodology + microGPT pattern
- ../talos-vs-macbook/bench_c.c — NEON kernel reference
- F#431 — TF-IDF routing 96.6% at N=5 on Gemma 4 (current routing baseline,
  ~1ms latency; reflex layer is upstream of this)
- F#766 — Hot-swap <1ms (production serving viable; reflex is even cheaper)
