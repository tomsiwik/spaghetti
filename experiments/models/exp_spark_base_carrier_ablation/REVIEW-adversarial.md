# Adversarial Review — exp_spark_base_carrier_ablation

**Verdict: KILLED (confirmed).** Reviewer agrees with the preliminary KILL.

## Gate
`verify-experiment.sh exp_spark_base_carrier_ablation` → EXIT 0 ("REAL result ok, model-backed").

## No-mock checklist — all PASS
- `is_smoke=false`; wall-clock 932.7 s (real ~15 min run, not a stub).
- α-knob is genuine: `AttenuatedLoRAQProj.__call__` returns `alpha*base(x) + scale*(x@A)@B^T`; `alpha` is a live float flipped per α-point. NOT a no-op. Installed via `setattr(attn,"q_proj",wrapper)` — submodule replacement, NOT instance `__call__` override. No silent bypass.
- Adapters real: three 5.0 MB safetensors from exp_composition_residual_analysis; 84 keys each = 42 layers × {lora_a (2560×6), lora_b (6×2048)}. Code loads exact keys and raises KeyError if missing.
- Accuracy from real greedy decoding: `greedy_generate` autoregressive argmax with KV cache; per-sample scored via numeric/MCQ-letter/first-line match on held-out valid.jsonl (199 items/domain, slices of 50/25/50). No hardcoded results.
- KC matches DB kill-id 2294: text "R_mean(0) < 0.80 OR any R_d(0) < 0.65"; result "fail". Both clauses fired: R_mean(0)=0.094<0.80 AND min R_d(0)=0.0<0.65.
- K2 validity guard passes (3/3 adapters beat chance at α=1: math 0.60, code 0.32, medical 0.64), so the reference is non-degenerate and R is well-defined.

## Consistency — all PASS
results.json verdict=KILLED, all_pass=false, is_smoke=false; PAPER.md line 6 "Verdict: KILLED · is_smoke=false · 932.7 s" and §5 match. KC integrity moot (fresh untracked experiment; no prior commit to mutate).

## CAVEAT the analyst must record
α=0 zeros the **entire** base query projection in all 42 q_proj layers, leaving a rank-6 delta as the sole query driver. This breaks attention routing **universally**, so the design cannot cleanly separate "frozen base carries domain knowledge" from "frozen base carries any usable query signal at all." The monotone decay (math 0.6→0.18→0.02→0.0; code 0.32→0.16→0.12→0.0; medical 0.64→0.44→0.36→0.18) is equally consistent with generic attention degradation. The experiment refutes the **carrier (flat-curve)** prediction — which is sound regardless of interpretation — but does NOT positively prove the base encodes *domain-specific* knowledge. The narrower claim "frozen base q_proj is load-bearing for on-domain behavior" is supported; the stronger claim "base carries the domain knowledge specifically" is NOT cleanly established. Medical's slower decay (R_d(0)=0.281) likely reflects 4-way MCQ chance floor (0.25), not partial retention.

## Non-blocking flags
- code n=25 < some thresholds but ≥15; acceptable. Math/medical n=50.
- Target-metric KC present (behavioral task accuracy), not proxy-only. ✓
