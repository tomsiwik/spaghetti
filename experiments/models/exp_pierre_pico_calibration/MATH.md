# MATH.md — Pico calibration on stacked B (Pierre arch)

## Hypothesis

Pico (arxiv 2604.16826, "Crowded in B-Space", April 2026) diagnoses that
adapter merge interference is concentrated in the output-side matrix B
because multiple adapters over-rely on a small number of shared output
directions. Pico's fix: SVD the stacked B's to find these shared directions,
dampen them per-direction by a closed-form factor, then merge as usual.

Pierre's shared-A architecture **intensifies** this exact problem — all
inter-adapter variation lives in B, so over-sharing in B is the dominant
failure mode by construction. Pico is the cleanest spec match of any method
surveyed (see `notebooklm_briefing.md`: "since your storage uses shared-A,
the burden of task specificity falls entirely on the B matrices").

> **Does Pico calibration as a pre-stage to Fisher-Rao close the +6.7pp gap
> measured in `exp_pierre_dare_b_vs_fisher_rao` between Fisher-Rao (64.7%)
> and full-delta DARE (71.3%)?**

## Algorithm (verbatim from paper)

For each (layer, module) key:

```
B_all = [B_1 | ... | B_T]                 # column-stack, shape (d_out, T·r)
U, σ, V^T = SVD(B_all)                    # m = min(d_out, T·r) singular dirs
s_j = σ_j² / Σ_k σ_k²                     # sharing score per direction
α_j = 1 / (1 + (T-1) s_j)                 # damping factor
S   = I + U · diag(α - 1) · U^T           # calibration matrix
B_t_calib = S · B_t                       # per-adapter calibration
B_merged = Fisher-Rao(B_t_calib)          # downstream merger
γ = mean_t ‖B_t‖_F / ‖B_merged‖_F         # mean source norm correction
B_out = γ · B_merged
```

For PoLAR (which stores B as `(r, d_out)` not `(d_out, r)`), we transpose
internally: `B_all = stack([B_t^T])` and apply `B_t @ S^T` (right-mul by S^T)
to keep B's storage shape uniform.

## Why this fits Pierre exactly

Three properties of Pico that match shared-A B-only architecture:

1. **Operates entirely in B-space** — no full-delta materialization, no per-adapter A required, no fused-delta wrapper. Plugs into existing `compose_adapters` API as a pre-stage.
2. **Closed form, no learned parameters** — applies once at adapter-load time.
3. **Pre-stage, not replacement** — `Pico → Fisher-Rao` keeps Pierre's existing merge math intact and only conditions the inputs.

## Pre-registered Kill Criteria

- **K1 (DECISION)** Pico+Fisher-Rao avg ≥ Fisher-Rao avg + 3pp.
  - Pre-registered threshold from paper's reported +3.4–8.3pp range over uncalibrated mergers. PASS = swap Pierre default to Pico+Fisher-Rao.
- **K2 (ARCH GAP CLOSURE)** Full-delta DARE avg − Pico avg ≤ 4pp.
  - Tells us whether shared-A B-only Pico recovers most of the +6.7pp gap to research's per-adapter A path.
- **K3 (PREPROCESS BUDGET)** Pico SVD + calibration time ≤ 5s total across 42 layers.
  - One-time cost at adapter load — must be cheap enough to not affect serving latency.
- **K4 (SANITY)** With `alpha_override=1.0` (S=I, no-op), Pico+Fisher-Rao avg within 1pp of plain Fisher-Rao.
  - Verifies the implementation reduces to identity at the trivial limit.

## Verdict logic

| K1 | K2 | K3 | K4 | Outcome |
|----|----|----|----|---------|
| ✓ | ✓ | ✓ | ✓ | **SUPPORTED** — Pierre default becomes Pico+Fisher-Rao. |
| ✓ | ✗ | ✓ | ✓ | **SUPPORTED** — adopt; note shared-A still leaves headroom (consider follow-up: Pico + per-adapter A). |
| ✗ | * | * | ✓ | **KILLED** — Pico does not transfer. |
| * | * | * | ✗ | **INCONCLUSIVE** — implementation drift; debug. |

## Eval protocol

Same as `exp_pierre_dare_b_vs_fisher_rao`:
- N = 50 per benchmark, fixed seed=42
- Benchmarks: GSM8K, HumanEval, MedQA via `scripts/polar_train.py::eval_*`
- Adapters: same 7 (4 strategy + 3 domain) PoLAR
- Base: `mlx-community/gemma-4-e4b-it-4bit`
- Shared-A donor: `strategy_full` (consistent across the experiment family)

## Honest gaps

- Paper is 6 weeks old (April 2026) with no public repo at experiment-design time. Implementation is from paper pseudocode; equations cross-checked against arxiv HTML render.
- Paper's worked examples merge 2-4 task LoRAs trained per-adapter. Pierre uses 7 PoLAR adapters with shared A — the over-sharing should be *more* pronounced (good direction for Pico) but exact gain magnitude is extrapolation.
- SVD on `(d_out=2048, T·r=42)` matrix per layer × 42 layers. CPU SVD (the only numerically stable path on Apple Silicon) is the dominant cost; budget K3 ≤ 5s assumes ~120ms per layer.

## References

- Pico paper (arxiv 2604.16826): https://arxiv.org/abs/2604.16826
- Implementation spec: research agent `acbe00274a1a6eb9c` (verbatim equations from paper HTML)
- Pierre `compose.py` Fisher-Rao baseline: copied verbatim into shared `eval_runner.py`
- Prior measurement of the gap: `exp_pierre_dare_b_vs_fisher_rao` (Fisher-Rao 64.7%, full-delta DARE 71.3%)
