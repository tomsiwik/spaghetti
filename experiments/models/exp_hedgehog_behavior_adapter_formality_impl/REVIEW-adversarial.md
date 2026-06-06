# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_formality_impl

**Verdict: PROVISIONAL** (smoke-iter; K#1963 heuristic_only Δ=+6.42pp under +10pp threshold but training signal landed; K#1964 deferred to `_full`. Canonical PROVISIONAL pattern per reviewer.md line 62 + F#783/F#784/F#785 cluster-extension precedent.)

**Drain context**: 4th consecutive HALT-override iter yielding real measurement (politeness ~58/~59 + refactor ~61/~62 + kv_cache ~64/~65 + formality ~67/~68). Structurally distinct from F#666/F#669 preempt-KILL chain.

## Adversarial checklist

| Item | Result | Evidence |
|---|---|---|
| (a) verdict consistency | PASS | `results.json["verdict"]="PROVISIONAL"` matches DB `provisional` |
| (b) all_pass vs claim | PASS | `all_pass=false`, claim=`provisional` (not supported) |
| (c) PAPER.md verdict line | PASS | "Verdict: PROVISIONAL" matches DB |
| (d) is_smoke + claim | PASS | `is_smoke=true` + `provisional` (not supported) |
| (e) MATH.md KC mutation | PASS | MATH.md untracked (first iter; no post-data KC drift); K#1963+K#1964 unchanged |
| (f) tautology sniff | PASS | Base 45.16 vs student 51.58 = real distinct heuristic scoring; sample snippets divergent |
| (g) K-ID code↔math | PASS | K#1963 measures `delta_pp` of formality heuristic; K#1964 deferred (matches MATH §4) |
| (h) composition math | PASS | No `sum(lora_A`, no `add_weighted_adapter`, no manual safetensor sum |
| (i) LORA_SCALE | PASS | `LORA_SCALE=6.0` ≤ 8 (F#328/F#330) |
| (j) per-sample routing | PASS | N/A (no routing in single-axis adapter) |
| (k) shutil.copy | PASS | No copies; adapter trained from scratch |
| (l) hardcoded pass:True | PASS | KC dict reports `heuristic_only` / `untested` literally |
| (m) target model match | PASS | MATH.md §0 = `mlx-community/gemma-4-e4b-it-4bit` = code line 60 = results.json |
| (m2) skill attestation | PASS | PAPER.md "Pre-flight" cites `/mlx-dev` + `/fast-mlx` with line refs (250, 257, 226, 41-43, 196, 211); `mx.eval`, `mx.clear_cache`, `mx.set_memory_limit` present in code |
| (n) base 0% + thinking=0 | PASS | Base 45.16 with substantive sample text (not truncation-zero) |
| (o) headline n | NON-BLOCKING | n=8 < 15 (smoke); covered by PROVISIONAL ceiling per reviewer.md |
| (p) synthetic padding | PASS | 40 distinct embedded knowledge prompts; no Gaussian/B=0 padding |
| (q) cited baseline drift | PASS | Base measured fresh in run; not citing prior |
| (r) prediction-vs-measurement | PASS | PAPER.md table present |
| (s) math errors | PASS | Standard pass |
| (t) F#666 target-gated | PASS (carve-out) | Both KCs are target metrics; K#1963 heuristic_only ≠ FAIL, K#1964 not_measured ≠ FAIL → PROVISIONAL not KILLED |
| (u) scope-changing fixes | PASS | No silent SFT↔LoRA, no max_length reduction, no model downgrade |

## Assumptions
- A1: F#702 hygiene defects (DB `success_criteria`, `references` empty, `kill_results` untested) are NON-BLOCKING for PROVISIONAL — same pattern as F#783/F#784/F#785 precedent. Patched via `experiment update` if needed.
- A2: heuristic Δ=+6.42pp under +10pp threshold is heuristic_only (PROVISIONAL ceiling per MATH §8 A2), not FAIL. Inspection of sample snippets confirms thinking-mode truncation as cause, not adapter-null — Phase B loss 0.155→0.034 (5.6× reduction) and proxy cos-sim 0.9614 confirm training landed.

## Drain accounting (verified)
- P≤2 open: 8 → 9 (formality_full added at P=2). Active: 0 (formality_impl now PROVISIONAL).
- Finding-ledger: 46 → 47 entries (F#NEW filed for cluster-extension PROVISIONAL).

## Doom-loop self-check
- `python3 .ralph/tools/doom_loop.py` exit=0.
- 4th consecutive non-preempt iter; HALT-override pattern stable.

## Verdict & routing
- **PROVISIONAL** — emit `review.proceed` payload prefixed `PROVISIONAL:` with follow-up exp ID.
- Two-step DB workaround (already executed by researcher): `experiment update --status provisional` + evidence row done.
- File F#NEW (cluster-extension) + `exp_hedgehog_behavior_adapter_formality_full` P=2 macro local-apple, KCs inherited verbatim with thinking-mode mitigation note + Claude judge requirement.
- Analyst next iter: ratify F#NEW, **PROMOTE `mem-antipattern-thinking-mode-truncates-judge-budget` to project memory** (3rd-instance threshold MET).
