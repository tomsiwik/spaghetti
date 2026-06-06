# PAPER — Pierre Adapter Hot-Swap Latency on Gemma 4 E4B (IMPL)

**Verdict: SUPPORTED**
**Parent: `exp_pierre_adapter_hotswap_latency` (F#702, provisional design-lock)**

## Summary

Two theorems inherited from the parent design-lock were measured on Gemma 4 E4B
4-bit via `mlx-community/gemma-4-e4b-it-4bit` with `self_attn.v_proj` +
`self_attn.o_proj` LoRA targets (F#627), rank 6, alpha 8.0, N=5 synthesized
adapters. Both pass:

| KC | Metric | Predicted | Measured | Threshold | Result |
|----|--------|-----------|----------|-----------|--------|
| K1953 | attach median over 20 runs (ms) | [0.4, 0.9] | **0.970** | ≤ 100 | **PASS** |
| K1954 | same-adapter detach/re-attach glitch-count | 0 | **0** | ≤ 1 | **PASS** |

K1953 lands one tick above the mid-point prediction envelope (0.97 vs 0.4–0.9)
but two orders of magnitude below the target threshold. K1954 matches the
predicted zero-glitch result exactly across all four swap positions.

## Setup

- Model: `mlx-community/gemma-4-e4b-it-4bit` (4-bit, 42 `DecoderLayer`).
- Targets: `self_attn.v_proj`, `self_attn.o_proj` (F#627).
- Rank 6, α = 8.0.
- N=5 adapters. A via partitioned QR of a `(in_features, N·r)` Gaussian ⇒
  N mutually-orthogonal `(in_features, 6)` blocks per `(layer, key)` (F#562).
  B via `N(0, 0.01²)` Gaussian.
- Bench: 3 warmup + 20 measured cycles (one attach-only time + detach, repeating).
- Determinism: 16 greedy-decoded tokens from a 20-token prompt; swap happens
  before decode step `k ∈ {1,2,4,8}` (same adapter detach + re-attach).

## Per-layer dimension heterogeneity (new observation, not in parent)

Gemma 4 E4B has two distinct attention-head groups:

- Layers {0..40 minus {5,11,17,23,29,35}}: `v_proj (2560 → 512)`,
  `o_proj (2048 → 2560)` — 35 layers (small-KV / sliding window).
- Layers {5, 11, 17, 23, 29, 35, 41}: `v_proj (2560 → 1024)`,
  `o_proj (4096 → 2560)` — 7 layers (wide-KV / global attention).

A single-layer-0 dim probe (as used by the parent's Theorem 1 upper-bound
calculation) would mis-shape 7 of 42 layers. This impl infers per-layer dims
before adapter synthesis; adapter B-matrix shapes match the layer they attach
to. The parent's `n_layers = 34` cited in §3 is also stale — the deployed
4-bit checkpoint has 42 layers.

## K1953 details

```
attach_times_ms  median=0.970  mean=0.977  std=0.021  min=0.956  max=1.040
```

Distribution is tight (std/mean ≈ 2.2%); no outliers. The measured median
(0.97 ms) is slightly above the Theorem 1 mid-point window (0.4–0.9 ms);
the margin is absorbed by:

1. Gemma 4 has 42 layers (not 34 as the parent cited) → scale mid-point by
   42/34 = 1.24 ⇒ 0.50–1.12 ms, and 0.97 ms lands squarely in that updated window.
2. Each adapter population attaches to 84 modules (42 × 2 targets), producing
   84 `RuntimeLoRA` constructors, 84 sub-module `tree_unflatten` updates, and
   one `mx.eval(model.parameters())`. The Python-overhead regime is confirmed.

The K1953 100 ms threshold is 103× the measured latency; swap is free on
the TTFT budget (Finding #388 puts the E4B forward at ≫ 10 ms per step; attach
is < 10% of a single decode step).

## K1954 details

Baseline `T_0` and swap-perturbed `T_swap(k)` are **bitwise-identical across
all 16 tokens for every k ∈ {1,2,4,8}**. 0 glitches total. Theorem 2
(MLX lazy eval produces a fresh computation graph per `__call__`; same-
adapter detach/re-attach is a module-identity change but not a
parameter-content change) is confirmed on Gemma 4 E4B text decode with KV cache.

The decoded prompt-continuation is degenerate (`'<|"|>...'`) because the
synthesized B-matrices are `N(0, 0.01²)` random noise — they perturb logits
but do not carry a trained task. This is **expected** and irrelevant to
K1954: the test is *equivalence under same-adapter detach/re-attach*, not
quality. The identical degenerate sequence across 5 runs (T_0 + 4 swap
variants) is the required signal.

## Assumptions & deviations from parent

- `n_layers = 42`, not 34 (parent §3 stale). Updated in IMPL MATH.md §2.
- Per-layer dim inference required (parent Theorem 1's uniform-T,
  uniform-dims assumption was an approximation for the asymptotic bound;
  the bound survives with the corrected constant).
- `model.layers`, not `model.model.layers`. Parent assumed qwen3/llama layout;
  Gemma 4 has a different wrapper. IMPL ships local `attach_adapter` and
  `detach_adapters` that honor the actual path. `pierre.pierre` is untouched
  (still correct for its Qwen3-target deployment).
- Targets restricted to v_proj + o_proj (parent §8 scope; parent §3 cited
  T=7 for its upper-bound worked example, but F#627 is 2-target).

## Verdict-consistency pre-flight (guardrail 1010)

1. `results.json["verdict"] == "SUPPORTED"` ✓
2. `results.json["all_pass"] == True` ✓
3. PAPER verdict line: SUPPORTED (no PROVISIONAL / INCONCLUSIVE / etc.) ✓
4. `is_smoke == False` ✓
5. KC git-diff: K1953/K1954 text verbatim from pre-reg, thresholds unchanged ✓
6. Antipattern memories checked: composition ✗, LORA_SCALE=8 ≤ 8 ✓,
   shutil.copy ✗, hardcoded pass ✗, eval-template truncation ✗,
   proxy-model ✗ (E4B not E2B). Clean. ✓

All six checks pass. `--status supported` is safe.

## Findings to register

- Both parent Theorems 1 and 2 transfer from Qwen3-0.6B to Gemma 4 E4B.
- Attach cost on a 42-layer × 2-target 4-bit Gemma 4 E4B is **~1 ms**; the
  2-order-of-magnitude margin below the 100 ms product threshold means
  adapter hot-swap is not a latency concern for Pierre on this base.
- Same-adapter detach/re-attach is **bitwise-exact** on the next 16 decoded
  tokens; the swap primitive is safe to call mid-generation.
- Gemma 4 E4B has **per-layer dim heterogeneity** (35 narrow + 7 wide
  attention groups). Any future adapter-synthesis code for Gemma 4 must
  infer dims per layer, not from layer 0.

## References

- Parent: `micro/models/exp_pierre_adapter_hotswap_latency/` (Theorems 1, 2).
- Prior art: `micro/models/adapter_hotswap_latency/` (Qwen3-0.6B,
  `t_inject_only = 0.260 ± 0.017 ms`).
- F#388, F#627, F#562, F#275, F#666, F#702.
- Hu et al. arxiv:2106.09685.
