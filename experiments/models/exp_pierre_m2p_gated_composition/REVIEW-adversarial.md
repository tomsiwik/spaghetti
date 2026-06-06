# REVIEW — exp_pierre_m2p_gated_composition

**Verdict: KILL confirmed**

## Adversarial Checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict matches DB status | PASS — both `killed` |
| b | all_pass matches claim | PASS — `false`, verdict `KILLED` |
| c | PAPER.md verdict matches DB | PASS — "KILLED" |
| d | is_smoke → provisional | N/A — `is_smoke: false` |
| e | KC not modified after first run | PASS — no post-hoc edits |
| f | No tautological KC | PASS — K2116 is a real target metric |
| g | Code measures what MATH.md describes | PASS — gated accuracy vs best-single per benchmark |
| h | Composition math: Σ(B_i @ A_i) not (ΣB)@(ΣA) | PASS — line 294: `a @ b` per adapter then sums |
| i | LORA_SCALE < 12 | PASS — SCALE=6.0 from polar_train |
| j | Single-sample routing applied to all | FLAG — bucket-averaging (line 370-371) applies avg weights per bucket, not per-prompt. Minor: with top-1 ≈ 0.993 this shouldn't matter much |
| k | shutil.copy fake adapter | PASS — no copies |
| l | Hardcoded pass:true | PASS — no fabrication |
| m | Model match | PASS — MATH.md and code both use `gemma-4-e4b-it-4bit` |
| n | Base accuracy 0% | FLAG — MedQA base is 0% for strategy_full. Possible truncation but doesn't affect kill |
| o | n < 15 | PASS — n=30 per benchmark |
| p | Target-metric KC present | PASS — K2116 is target |

## Assessment

The gate routing is excellent (99.6% holdout, entropy 0.039, top-1 0.993). The catastrophic failure (HumanEval 86.7→20.0%) is correctly attributed to `apply_gated_composition` monkey-patching `__call__` on PoLARLinear modules (line 301-305). This is an implementation bug — the patched forward calls `layer.base(x)` which may not match PoLARLinear's actual base computation, and `__get__` binding on MLX modules is unreliable.

Kill is correct. The finding — that M2P gating solves routing but `__call__` override breaks composition — is valuable for the next experiment (ties_dare).
