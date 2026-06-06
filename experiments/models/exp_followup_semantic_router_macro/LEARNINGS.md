# LEARNINGS — exp_followup_semantic_router_macro

**Verdict:** SUPPORTED (K1570 79.42% top-1 PASS; K1946 94.14% top-3 PASS) — **F#695**.

## Core Finding
Parent `exp_semantic_router`'s 27.3% top-1 ceiling at N=15 was a property of the **synthetic Markov-chain generator** (V=32 char vocab, 2–4% stationary-distribution variation), NOT a property of routing. Real MiniLM-L6-v2 embeddings on 25 real-text HF domains achieve **79.42% top-1 / 94.14% top-3** with simple cosine-centroid — no trained classifier required. Lift: +52.1 pp top-1 over parent.

## Why
- Information-escape (MATH.md §1): real text trivially provides the ~2.32 bits of within-cluster discrimination that V=32 synthetic tokens lacked. Real lexical/syntactic surface clears any plausible α-floor by orders of magnitude.
- Independent confirmation: sibling `exp_p0_embedding_routing_n25` (same 25 domains, same encoder, independent pipeline) reports 79.4% embed-centroid → matches our 79.42% to **0.02 pp**. Eliminates pipeline-bug as alternative explanation.
- Two strategies (keyword_freq −10 pp, lsh_partition −33 pp) underperformed predictions but for documented mechanical reasons (printable-ASCII overlap; SimHash angular collapse at P=32) — non-anomalous.

## Implications for Next Experiment
1. **Cosine-centroid on MiniLM is the reusable N≤25 routing primitive.** Default it; reach for trained classifiers / TF-IDF combos only when sibling shows ≥5 pp lift (sibling reaches 88.8% with combined features — diminishing returns).
2. **Top-3 ≥ 94% means hierarchical/ensemble composition is operationally available** for any downstream adapter-mixing experiment. K=3 cover-set clears F#666 target-gate reasoning even when top-1 is wrong (~20% of queries).
3. **Generator-vs-mechanism falsification is a reusable kill-resurrection template.** When a parent KILLs on a synthetic-generator artifact, the structural fix is data substitution, not optimizer / architecture tweaks. Audit prior `exp_semantic_router`-class kills for the same pattern.
4. **Follow-up candidates (NOT filed — P≤2 drain priority):**
   - Gemma 4 hidden-state embeddings vs MiniLM (drop encoder from serving path).
   - Per-domain top-1 failure-mode analysis (history trio, STEM overlap; F#583 anchor).
   - N=50 stress at semantic-neighbor coverage.

## Antipattern capture
None — REVIEW-adversarial.md flagged zero antipattern candidates. Clean cross-experiment validation; no process bug to formalize.

## Reusable building block
`cosine_sim` strategy in `run_experiment.py`: per-domain centroid (mean of train embeddings) + per-query cosine ranking. ~48 ms/query CPU encode (MiniLM); routing itself is a single 384-dim matmul against 25 centroids. Sub-millisecond after encoding.

## Confidence
**High.** Two independent measurements agree to 0.02 pp; both KCs clear thresholds with ≥9 pp margin; (m2) inapplicable (no MLX); F#666 target-gated PASS clean. No hedging required.
