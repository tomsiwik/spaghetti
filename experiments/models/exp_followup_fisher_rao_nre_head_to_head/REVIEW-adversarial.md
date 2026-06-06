# Adversarial Review — exp_followup_fisher_rao_nre_head_to_head

## Verdict: KILL (ratified)

Finding #677 in DB; experiment already completed as `killed`. This review ratifies.

## Consistency (a–d)
- (a) results.json `"verdict": "KILLED"` ↔ DB `killed` ↔ PAPER.md "Verdict: KILLED" — all three aligned.
- (b) `all_pass: false`; K1 FAIL + K2 FAIL + K3 PASS matches MATH.md §F KILLED rule.
- (c) No `PROVISIONAL`/`PARTIALLY SUPPORTED`/`DEGENERATE` markers in PAPER.md.
- (d) No `is_smoke` flag; 50 samples × 42 layers × N∈{3,10,25} is a full run.

## KC integrity (e–g)
- (e) MATH.md untracked (written once, no post-hoc diff). KCs pre-registered.
- (f) No tautology: K1/K2 compare real measured FR vs NRE PPL; K3 compares against Euclidean baseline.
- (g) Code K1/K2/K3 at `run_experiment.py:406-408` measure exactly the quantities MATH.md §F defines.

## Code ↔ math (h–m2)
- (h) NRE at `run_experiment.py:114-121` = `(1/N) Σ B_i` then rescale by `mean_i ‖B_i‖ / ‖B̄‖` — matches MATH.md §E method 2. No `add_weighted_adapter(combination_type="linear")`. No per-key safetensor summing.
- (i) `LORA_SCALE=6.0` ≤ 8 ceiling (F#328/#330).
- (j) No routing.
- (k) No `shutil.copy`.
- (l) All KC dicts computed from measured values; no hardcoded `pass: True`.
- (m) Base `mlx-community/gemma-4-e4b-it-4bit` matches MATH.md §E and PLAN.md Part 2.
- (m2) MLX idioms correct: `mx.eval` after composition (`run_experiment.py:139,180,188,354`), `mx.clear_cache()` in cleanup, proper module attribute mutation (`mod.lora_a = ...`). No torch-style `.forward()`, no missing eval.

## Target-gated kill (t, F#666)
- K1 (proxy: overall PPL) FAIL, margin=−0.352.
- K2 (target: conditional-PPL on assistant tokens only, prompt masked) FAIL, margin=−0.031.
- Both proxy AND target failed → kill is safe per F#666. Kill is about the ceiling claim, not the proxy.

## Eval integrity (n–s)
- (n) Base overall PPL 29.04 (non-zero), cond 5.56 — no thought-channel truncation.
- (o) n=50 eval samples, 42 per-layer compositions ×3 methods ×3 N values — adequate.
- (p) Synthetic noisy variants for N>3 (disclosed MATH.md §G.3, PAPER.md caveat 3). Affects Euclidean plateau only; FR vs NRE comparison is fair because both see the same b_stacks.
- (r) PAPER.md has explicit prediction-vs-measurement table (§ Predictions vs Measurements).
- (s) Math consistent with Pennec 2006 small-dispersion bound; P2 falsification direction (NRE slightly > FR) strengthens KILL, not weakens it.

## Generalisation strength
F#275 (BitNet-2B, N≤15) → Gemma 4 E4B 4-bit q_proj, N≤25 — same qualitative conclusion. Different base, different quantisation, ~2× N. Result is not architecture-specific.

## Assumptions logged
- Treating conditional-PPL (assistant-tokens-only, prompt-masked) as the target-metric for F#666 pairing. Justification: closest measurable proxy for behavioural quality under a composition-method comparison where no downstream task pipeline is wired.
- Ratifying the DB's existing `killed` status and Finding #677; no new writes needed.

## Route
`review.killed` → Analyst writes LEARNINGS.md with literature context (Pennec 2006, F#274/#275/#666).
