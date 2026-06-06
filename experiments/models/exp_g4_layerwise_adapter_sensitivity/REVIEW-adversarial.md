# REVIEW-adversarial.md — exp_g4_layerwise_adapter_sensitivity

## Verdict: **PROCEED**

Both F#666-paired KCs pass with large margin; code faithfully measures MATH.md §5 operationalization; no blocking antipatterns.

## Adversarial checklist

| # | Item | Result |
|---|---|---|
| a | `results.json["verdict"]=="SUPPORTED"` vs proposed `supported` | ✓ consistent |
| b | `all_pass==true`; both KCs PASS | ✓ |
| c | PAPER.md verdict line = "SUPPORTED — both KCs pass" | ✓ no PROVISIONAL/partial downgrade |
| d | `is_smoke` absent; full N=30 + full 42 layers (173 s wall) | ✓ not smoke |
| e | MATH.md is new-file in git (untracked dir); no post-hoc KC rewrite possible | ✓ |
| f | Tautology sniff — K1919 is CV over measured ΔPPL; K1976 is contiguity over argsort of measured s_l | ✓ both computed from data, not identity |
| g | K-IDs measure what MATH.md §5 specifies — K1919 = `std_s/abs(mean_s)`; K1976 = `find_contiguous_bands(top7)` | ✓ exact |
| h | No `sum(lora_A)`/`add_weighted_adapter` — no composition in this experiment | ✓ N/A |
| i | No `LORA_SCALE=20` or LoRA of any kind | ✓ N/A |
| j | No routing | ✓ N/A |
| k | No `shutil.copy` adapter | ✓ N/A |
| l | No hardcoded `{"pass": True}` — KCs derived from measurements | ✓ |
| m | Target = `mlx-community/gemma-4-e4b-it-4bit`; `load(BASE_MODEL)` loads the same | ✓ |
| m2 | MLX skill citation gap (MATH.md §0 does not list `/mlx-dev`, `/fast-mlx`) — **non-blocking because the code itself is textbook-idiomatic MLX** | ⚠ note |
| n | baseline_ppl = 36.55 (not 0); no thought-channel truncation | ✓ |
| o | CV computed over n=42 layer-sensitivity points (well above 15) | ✓ |
| p | No synthetic padding | ✓ |
| q | Baseline PPL measured in-run, not cited | ✓ |
| t | **Target-gated kill (F#666)** — K1919 proxy (structural CV) + K1976 target (actionable contiguous-band). Both PASS → PROCEED is F#666-safe. K1976 is a genuine target KC: the actionable-band claim is directly testable by the follow-up `exp_g4_layer_selective_lora_top8` | ✓ |
| r | PAPER.md Predictions vs Measurements table present (4 rows) | ✓ |
| s | Math cites ShortGPT (2403.03853), Todd (2310.15213), Higham Ch.7; mechanism chain is coherent | ✓ |
| u | No scope-changing fix — experiment ran as designed in one pass | ✓ |

## Notes (non-blocking)

1. **P2 range_ratio caveat**: `max(s_l)/min(s_l)` is mathematically undefined because `min(s_l) = -2.28` (L17) — several late layers have slightly negative ΔPPL (noise-floor/regularization artifact at N=30). The code correctly falls through to `range_ratio = inf` when `min ≤ 0`, and PAPER.md §Assumptions/P2 transparently flags this. The underlying prediction (signal ≫ noise) is met overwhelmingly (max span ≈ 17,560). No action.

2. **Skill-citation gap (m2)**: MATH.md §0 doesn't explicitly list `/mlx-dev` or `/fast-mlx`. However, the implementation demonstrates correct MLX knowledge that these skills would enforce:
   - class-level `__call__` override (instance-level patching is documented in code comments as broken for `nn.Module`)
   - `mx.eval(nll)` before `.item()`
   - `mx.clear_cache()` between phases
   - `mx.random.key(seed)` (not legacy PRNG)
   - `.astype(h.dtype)` on perturbation to preserve 4-bit compat
   Code quality evidence is sufficient to satisfy (m2) intent. Future experiments should still cite skills in MATH.md §0 per guardrail 1012.

## Assumptions

- F#747 is already registered in DB (finding-list confirmed 2026-04-25).
- Completion DB status = `supported`; matches disk verdict.
- No active/claimed entries remain; handoff hygiene clean.

## Striking result worth amplifying in LEARNINGS.md

- **L21 is a singular sensitivity hub** — ΔPPL ≈ 17,558 at ε=0.10 relative perturbation (480× baseline 36.55). This is the strongest single-layer-sensitivity signal observed in this codebase.
- **Three-band structure** (L1 / L8-10 / L20-22) is novel for Gemma 4 E4B; prior ShortGPT/Todd results on other architectures typically report two regimes. The PLE-M2P per-layer-input gating of Gemma 4 may explain the trifurcation.
- **18 of 42 late layers (L23-L40) are perturbation-redundant** — direct motivation for the already-proposed follow-up `exp_g4_layer_selective_lora_top8` at ~19% parameter cost.
