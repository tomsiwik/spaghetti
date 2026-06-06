# REVIEW-adversarial — exp_g4_adapter_initialization_comparison_v2

## Verdict: PROVISIONAL

`is_smoke: true` per results.json (100 iters/run vs design 1000); SC#109 NOT verified
(K1985 FAIL on target = recipe is MCQ-suppressive at smoke); 2 distinct findings worth
filing (PRNG-confound retroactive + recipe MCQ-suppression).

## Adversarial checklist (18 items)

| # | Item | Status | Note |
|---|---|---|---|
| a | results.json verdict ↔ DB target | PASS | Both PROVISIONAL. Researcher pre-flight confirmed. |
| b | all_pass=false ↔ claim | PASS | `all_pass=false`, claim is PROVISIONAL not SUPPORTED. |
| c | PAPER.md verdict line | PASS | Contains "PROVISIONAL". |
| d | is_smoke=true → smoke floor | PASS | Smoke floor honored, verdict locked PROVISIONAL. |
| e | KC pre-reg integrity (git diff) | PASS | KCs #1977/#1978/#1979/#1983/#1984/#1985 schema-repaired at iter ~36 (drain-window) BEFORE this run; preserved byte-for-byte in code, MATH.md, PAPER.md, results.json. No post-run mutation. |
| f | Tautology sniff | PASS | K1983 inverted-direction (FAIL=invariance verified) is documented and F#666-paired. K1985 is clean target metric. K1979 identifiability check, not load-bearing for verdict. |
| g | K-IDs measure correct quantity | PASS | `pairwise_column_cos` (col-normalised, |·|, mean) matches MATH §2 K1977. `init_drop_vs_base_pp = base_mcq − init_mean_mcq` matches K1985 description. |
| h | Composition bugs (`sum(lora_A`, etc.) | N/A | Single adapter per run; no composition. |
| i | LORA_SCALE ≥ 12 | PASS | `SCALE = 6.0` per F#328/F#330. |
| j | Per-sample routing | N/A | No routing. |
| k | shutil.copy as new domain | PASS | Fresh `linear_to_lora_layers` per run; adapters saved as `adapter_<init>_s<sub_seed>.safetensors`. |
| l | Hardcoded `pass: True` | PASS | All 6 KC verdicts computed from measurements. |
| m | Target model ↔ MATH.md | PASS | `mlx-community/gemma-4-e4b-it-4bit` matches MATH.md §1 inheritance pointer. |
| m2 | Skill attestation | PASS | MATH.md §0 attests `/mlx-dev` invoked; `/fast-mlx` deferred with rationale (smoke iter, no novel hot-path requires `mx.compile` tuning). MLX patterns present: `mx.eval` at boundaries, `mx.clear_cache` between runs, `gc.collect()`, `del` before reload. |
| n | Base eval truncation | PASS | Base MCQ = 57.5%, not 0%. |
| o | Headline n ≥ 15 | PASS | n=80 MCQ items × 3 sub-seeds × 3 inits = 240 trials per init for K1985. |
| p | Synthetic padding | PASS | All 9 runs are real training; no B=0 / random adapters in the count. |
| q | Cited baseline drift | PASS | Base measured in this run (57.5% MCQ, 5105 PPL). |
| r | Prediction-vs-measurement table | PASS | PAPER.md §1 has full table for all 6 KCs. |
| s | Math errors / unsupported claims | PASS | One audit observation: K1983 PAPER.md notes both s=0-rep (1.25pp FAIL) and means-based (9.6pp PASS). The means-based read uses `init_mean_mcq` which is per-init mean over 3 sub-seeds (240 trials / init); within-init seed-noise is 12.5pp (gaussian) > 9.6pp cross-init means. Honest interpretation in PAPER §5: spread is **NOT statistically separable from seed noise at this N** — verdict-consistent with PROVISIONAL. |
| t | Target-gated kill (F#666) | N/A | Verdict is PROVISIONAL not KILL. K1985 is a target metric (behavioral non-interference), so kill rule would be satisfied if we were killing. |
| u | Scope-changing fixes | PASS | Smoke iter (100 vs design 1000) is pre-registered explicitly in MATH.md §6 with verdict floor declared and reclaim path; not a silent scope reduction. Recipe is identical to v1 / parent (q_proj r=6 medical) — no silent SFT↔LoRA swap, no `max_length` truncation, no model downgrade. |

**All 18 items PASS or N/A. No blocking fixes. Verdict: PROVISIONAL.**

## Findings to file

### F#773 (PROVISIONAL): PRNG-confound retroactive resolution
v1 K1924 (cross-init final cos Δ > 0.10) PASS was an **artifact of shared `mx.random.key(42)`**
producing starting cross-init |cos|=0.977-0.9995. v2 with distinct seeds:
- starting |cos| collapses to 0.015-0.018 (≈ orthogonal as expected for independent random matrices),
- final |cos| stays at 0.027-0.042 (well below 0.20 threshold)
→ K1977 PASS at <0.20 INVERTS v1 K1924 PASS at >0.10. Single-seed PPL in v1 (3.5% spread) is
**WITHIN seed-noise floor** (v2 K1979: 24.6% within-init PPL spread on Grassmannian alone), so v1's
init-invariance claim was unidentifiable from a single-seed measurement.

### F#774 (PROVISIONAL): Medical-q_proj-r6 LoRA recipe is MCQ-suppressive at smoke iters
At 100 iters with q_proj-only r=6 LoRA on `medical/train.jsonl`, MCQ heldout accuracy DROPS
4.6-14.2pp vs no-adapter base (57.5%): grassmannian 48.3%, **kaiming 43.3%**, gaussian 52.9%.
NOT visible at PPL level (PPL drops 4000× — looks like training works). Antipattern signal:
PPL improvement on training distribution does NOT predict heldout MCQ accuracy. Replicates
F#666 / behavioral-outcomes-over-metrics thesis. Open question: does K1985 recover at full
1000-iter convergence (v3a) or persist (v3b canonical recipe v_proj+o_proj F#627)?

## Assumptions logged
- A1: K1985 FAIL is the load-bearing signal beyond smoke; if dropped a single iter (not preregistered to gate verdict), verdict falls to plain PROVISIONAL on smoke-iter floor alone.
- A2: K1983 PAPER.md dual-read (reps vs means) is honest and consistent with researcher hat 6.6 — both readings reported; neither alone resolves SC#109.
- A3: Two findings filed (not one) because PRNG resolution and recipe-suppression are mechanistically distinct; F#751 retrospective re-evaluation is in F#773 caveat, not a separate finding.

## Routing
PROCEED → analyst (LEARNINGS.md) per PROVISIONAL workflow. v3a + v3b filed at P3 (out of P≤2 drain queue).
