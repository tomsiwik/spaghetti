# REVIEW — exp_followup_semantic_router_macro

**Verdict:** **PROCEED**
**Round:** 1
**Reviewer:** reviewer hat, iter 2026-04-24

## Summary
Real-embedding routing at N=25 beats parent's synthetic-Markov ceiling (27.3% → 79.42% top-1, +52.1pp). Top-3 94.14% clears the F#666 target gate. Parent's information-bottleneck claim is confirmed *data-specific*, not a property of routing. Cross-check: cosine_sim 79.42% here ↔ sibling `exp_p0_embedding_routing_n25` 79.4% on identical 25-domain mix — independent-measurement agreement to 0.02pp.

## Adversarial checklist

**Consistency:**
- (a) `results.json["verdict"]="SUPPORTED"` ↔ DB `status=supported` ↔ PAPER.md L4 `Verdict: SUPPORTED`. ✓
- (b) `all_pass=true`; both KCs `pass=true`. ✓
- (c) No PROVISIONAL/PARTIAL/NOT_SUPPORTED hedging in PAPER.md. ✓
- (d) `is_smoke=false`. ✓

**KC integrity:**
- (e) Directory untracked (no prior commit). No post-run KC mutation possible; pre-reg taken from MATH.md §4. ✓
- (f) No tautology. K1570 compares best semantic-strategy top-1 to independent 37.3% threshold (parent +10pp). K1946 compares best top-3 to independent 85% threshold. Both measure real accuracies on held-out test data; no algebraic identity. ✓
- (g) `run_experiment.py` KC block (L63-76) matches MATH.md §4 verbatim (K1570 thr=0.373 proxy; K1946 thr=0.85 target). DB `#1570` & `#1946` align. ✓

**Code ↔ math:**
- (h) No LoRA composition; pure routing benchmark. ✓
- (i) No LORA_SCALE constants. ✓
- (j) Per-sample routing (all 6 strategies iterate over test_texts). ✓
- (k) No `shutil.copy`. ✓
- (l) `pass=bool(k1570_pass)` is computed from `best_top1 >= threshold` — no hardcode. ✓
- (m) Target "model" is the routing algorithm; MiniLM-L6-v2 actually loaded at L191 matches MATH.md §2 and §3. ✓
- (m2) MATH.md §6.6 explicitly documents "MLX not involved; no `/mlx-dev` skill needed because no MLX code is written." Numpy + sklearn + sentence-transformers only; (m2) inapplicable. ✓

**Eval:**
- (n) No LLM → no thinking-channel; not applicable.
- (o) n=25 domains × ~100 test-per-domain = 2,483 test samples. Far above 15. ✓
- (p) No synthetic padding; all 25 domains sourced from real HF datasets (GSM8K, CodeAlpaca, MedMCQA, MMLU). ✓
- (q) Parent baseline 27.3% cited (not re-measured); sibling 79.4% cross-check **re-measured here at 79.42%** — data pipeline integrity confirmed. ✓
- (t) **Target-gated kill (F#666):** K1946 is an explicit target KC (behavioral top-3 coverage ≥ 85% for hierarchical/ensemble routing usability). Paired with proxy K1570. Both PASS → SUPPORTED valid. ✓
- (u) No scope-reducing fixes (no silent mechanism swap, no seqlen reduction, no trackio disablement). ✓

**Deliverables:**
- (r) Prediction-vs-measurement table present (PAPER.md §2, 7 strategies × 4 columns). ✓
- (s) Information-escape theorem (MATH.md §1) is a lower-bound sketch, not a tight derivation; the empirical result (79.42% ≫ 37.3%) substantially exceeds the floor, so the sketch is adequate. ✓

## Non-blocking observations (do not affect verdict)
1. Two of 6 strategies underperformed their predicted ranges (keyword_freq −10pp; lsh_partition −33pp). Mechanisms documented in PAPER.md §2. Semantically informative; not a failure.
2. `top-3` reported as the same as top-1 for `hash_ring` (trivially — footnote † in PAPER.md §2). Correct handling; other strategies' top-3 are well-defined.
3. Theorem in MATH.md §1 uses symbol `α` without numerical value; the bound is qualitative. Acceptable because the measurement is well above any plausible α-floor.

## Assumptions
- Treating "macro scale" per MATH.md §6.3 as real-embeddings-on-real-data at N > parent's 15. N=25 is the selected scale, matching the sibling's pool. N=50 is a separate follow-up.
- DB `status=supported` was set by researcher pre-review (consistent with PAPER.md); reviewer does not re-issue `experiment complete`, only adds a finding.

## Route
`review.proceed` — researcher's verdict stands. Analyst to draft LEARNINGS.md + memory entries. Finding to be added before event emission.
