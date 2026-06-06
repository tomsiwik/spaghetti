# Pierre v5 — Fully Ternary LoRA (Grassmannian A + STE B as BitLinear)

**Status (2026-04-17 rerun hat):** retrospectively authored from the claim notes. Original experiment was completed in the pre-audit era without a MATH.md; the audit (`audit-2026-04-17-rerun` + `tautological-routing` tags) scheduled this rerun. KC pre-registration is preserved as-is — the KCs are not being re-cast after seeing data.

---

## 1. Problem

Pierre v4 premerged LoRA deltas into ternary BitLinear weights (`W ← BitLinear(W_base + ΔW)`) and killed adapter signal: ternary re-quantization collapses the low-rank perturbation into rounding noise. v5 asks whether a **side-path** ternary LoRA — both A and B kept ternary, combined additively with `BitLinear(x)` at runtime — preserves adapter signal while retaining native ternary-matmul throughput.

### Failure mode
Ternary quantization of the adapter path destroys signal entirely (target behavioral ≥ 0.30 collapses to ~0). This is the same failure that killed v4.

### Prior math cited
- **Grassmannian packing** (Welch bound, cf. Finding #3): for `N ≤ r·(d/r)`, orthogonal `A_i ∈ R^{d×r}` can be packed with cosine ≈ `sqrt(r/d)` → interference ≈ 0.
- **STE ternary B ratio ≈ 1.068** (exp_adapter_compression_extreme): a ternarized B is within 6.8% of full-precision B in expected norm.
- **BitDelta (arXiv:2402.10193)** — ternary delta on frozen base preserves task signal when the delta is additive, not re-quantized.
- **TernaryLM (arXiv:2602.07374)** — native ternary matmul is numerically stable when A and B live separately.

### Key difference from v4
| Aspect | v4 premerge | v5 side-path |
|---|---|---|
| composition path | `quant(W + scale·BA)` | `BitLinear(W)·x + scale·BitLinear(B)·BitLinear(A)·x` |
| adapter signal survives base quant? | no (crushed) | yes (separate matmul) |
| expected speed | ~140 tok/s (1 matmul) | ~70–100 tok/s (3 matmuls) |

---

## 2. Theorem (informal)

Let `W_base ∈ {-1, 0, +1}^{d×d}` be a frozen BitNet matrix, `A_i ∈ R^{d×r}` Grassmannian-packed for domain `i`, and `B_i ∈ {-1,0,+1}^{r×d}` trained under STE. Then the ternary side-path output

```
y_i(x) = BitLinear(W_base)·x + α · BitLinear(B_i) · BitLinear(A_i) · x
```

satisfies `‖y_i(x) − y_i^{bf16}(x)‖ ≤ (1 + r/d)^{1/2}·‖y_i^{bf16}(x)‖·ε_ternary`, where `ε_ternary ≤ 0.07` per STE ratio. **QED (sketch):** triangle inequality on the two ternary paths + STE bound on B + Grassmannian structure on A.

**Behavioral prediction:** if v4 was the only failure mode, v5 recovers single-adapter behavioral within 10% of v3 bf16 side-path (0.410 → ≥ 0.37). If structure alone isn't enough (i.e. the ternary B fails to preserve SFT semantics), behavioral drops to ~0.10–0.20.

---

## 3. Kill criteria (pre-registered, DB IDs)

- **K727** — behavioral score ≥ 0.30 (ternary quantization destroys adapter signal if < 0.30).
- **K728** — decode speed ≥ 50 tok/s (worse than bf16 unpacked v2 invalidates ternary-speed claim).
- **K729** — routing accuracy ≥ 0.80 (below this, composition is routing-limited).

Thresholds chosen from v2/v3 baselines pre-run; not reformulated since.

---

## 4. Predictions (quantitative)

| metric | prediction | basis |
|---|---|---|
| routing accuracy | ≥ 0.90 | v3 measured 0.92, same TF-IDF + hidden-state router |
| single-adapter PPL vs base | 3–10% drop per domain | v3 bf16 side-path measured 3–12% drop |
| behavioral overall | 0.30–0.45 | STE ternary B ≤ 7% signal loss vs bf16 |
| decode tok/s | 70–100 | 3 ternary matmuls vs 1 bf16 = ~½ of v2 |
| overhead vs native BitLinear | 30–55% | measured on v3 side-path at bf16 |

---

## 5. Experiment plan (original design)

Four phases (see `run_experiment.py`):
1. **Router calibration** — TF-IDF + hidden-state logistic, 50 calib / 50 test samples across 5 domains.
2. **PPL** — single-adapter PPL per domain; then "pierre" PPL via `route(val[d][0])` → inject adapter → evaluate.
3. **Behavioral** — for each domain, generate 5 responses, score with `factual_recall`/code syntax.
4. **Latency** — 2 warmups + 5 timed 128-token decodes, bf16 vs ternary side-path.

N = 50 validation samples per domain; 5 generation samples per domain; MAX_SEQ=256.

---

## 6. Antipattern self-check (pre-run, as required by PLAN.md §1.6)

This experiment was designed before the audit, so the antipattern check is being performed **retroactively** here:

- **Tautological routing (mem-antipattern-002):** PHASE 2 `ternary_pierre` and PHASE 3 behavioral both route using `val[d][0]` (a single sample from the target domain), then apply that single adapter choice to all 50 val samples / 5 gen samples. At router accuracy 0.996, `pierre_ppl ≡ single_ppl` by construction, and behavioral is a single-adapter measurement dressed as composition. **TRIGGERS.**
- **Unsafe LORA_SCALE = 20 (mem-antipattern-003):** hard-coded at line 44. Safe default ≤ 8. **TRIGGERS.**
- **Composition math bug (mem-antipattern-001):** not applicable — this experiment routes to ONE adapter per sample, it does not sum across adapters.
- **Proxy model substituted for target:** uses `microsoft/BitNet-b1.58-2B-4T`, not Gemma 4. **Acceptable for this experiment** — the ternary hypothesis requires a ternary base. Per PLAN.md Part 2, Gemma-4 is the Pierre target; BitNet-2B is the ternary substrate for v4/v5.

Two antipatterns trigger → per PLAN.md §1 verdict rule 6, this experiment cannot be marked `supported` regardless of whether KC values are reported as pass.

---

## 7. Known gap (discovered during rerun)

The rerun discovered that the required runtime modules are **missing**:
- `pierre/v5` (imported at line 30) — does not exist in the repo.
- `micro/models/real_data_domain_experts/adapters/grassmannian_skeleton.npz` — does not exist.
- `micro/models/bitnet_sft_generation_v3/sft_adapters/` — does not exist.

`run_experiment.py` would crash at the `from pierre.v5 import ...` line before any compute. The original `results.json` dates from a pre-cleanup checkpoint where these artifacts existed; they have since been removed from the repo.

**Consequence:** this experiment cannot be rerun in its current form. The KILL verdict is driven by (i) antipattern contamination of the original results and (ii) irrecoverable dependency loss.
