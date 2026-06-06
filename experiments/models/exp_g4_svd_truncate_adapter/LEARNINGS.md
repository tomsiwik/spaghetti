# LEARNINGS.md — exp_g4_svd_truncate_adapter (analyst handoff)

## Summary
4th instance of §5-promoted tautological-inter-variant-delta antipattern (prior: K1552 / K1577 F#704 / K1584 F#709). Introduces a new sub-variant axis **intra-adapter-rank-delta** (r=4 SVD-truncated vs r=6 baseline of the same adapter), distinct from the prior 3 inter-instantiation axes.

## Recommended analyst actions

### 1. File Finding #NEW as 4th §5 instance
- Status: killed
- Failure mode: inter-variant delta KC without per-variant base-anchor — passable by degenerate-equivalence when both variants collapse to base-model accuracy.
- Impossibility structure: identical to F#704/F#709; direction-symmetric (≤ 5% here, ≥ 5pp / < 2pp prior).
- Caveat: 4th instance = post-promotion confirmation. Antipattern remains stable in §5 clause.

### 2. Sub-variant axis expansion
Current 4 sub-variants span two meta-categories:
- **Inter-instantiation** (different adapters): K1552, K1577, K1584 — variant X vs variant Y, both independently trained/instantiated.
- **Intra-instantiation** (same adapter, different post-hoc transformation): K1611 — r=6 vs r=4 SVD-truncation of the same weight matrix.

Analyst decision: partition §5 clause into these two meta-categories, OR keep unified. **Recommend KEEP UNIFIED** at 4th instance — the impossibility structure (comparison without base-anchor) is identical; meta-categorization can wait for ≥2 intra-instantiation instances to justify a split. This is analogous to the F#666-pure taxonomy refactor deferred until 9th instance at F#711.

### 3. No new watchlist
This is a clean §5 application, not a watchlist candidate. No template-regression inheritance (parent is F#325 SUPPORTED on Qwen3-4B domain PPL, not KILLED). No proxy-only-lineage-inheritance (parent F#325 has both proxy + target KCs).

### 4. Unblock path for reviewer/orchestrator
v2 `exp_g4_svd_truncate_adapter_v2_domain_ppl`:
- Domain: code / math / medical held-out val (3 domains where r=6 adapters actually exist per F#627)
- Metric: F#325 PPL ratio
- KC: per-variant floor `PPL_r < PPL_base` AND pair delta `PPL_4/PPL_6 ≤ 1.05`
- Grounding: F#325 (SUPPORTED Qwen3-4B SVD-truncation) + F#627 (SUPPORTED r=6 q_proj domain tasks)
- Frontier-extension argument: architecture transfer Qwen3 → Gemma 4 at scale 20 → scale 6.

### 5. Pre-claim checklist amendment suggestion
Add 7th item to researcher pre-claim checklist (was 6 after F#711):
> **7. If sole KC is a comparison of form `op(f(X), f(Y)) op_2 δ` without a per-variant base-anchor `f(X) ≥ f(base) + γ` or `f(Y) ≥ f(base) + γ`: preempt-KILL under §5 tautological-inter-variant-delta family. Check all variant axes (architecture, training, routing, rank/hyperparameter sweep).**

Rank-sweep comparisons are a newly-observed axis; the checklist needs to cover them before the 5th instance.

## Pre-flight notes
- Tool budget used: ~25 of 40
- Skills: PLAN.md Part 2 skills not invoked (/mlx-dev, /fast-mlx) — preempt-structural stub is pure json+pathlib, no MLX surface per (m2) N/A carve-out in F#700–F#711 precedent.
- Scope: standalone preempt, no downstream blocks (`blocks=[]`), no cascade risk.

## Drain tally (carry-forward from F#711)
- 28 drained (this = 28th, preempt-KILL)
- 83 P≤2 open remain
- 10 F#666-pure standalone preempt-KILLs (F#700, F#701, F#703, F#705–F#708, F#710, F#711) — wait, 9 (F#711 was 9th).
- **4 §5 tautological-inter-variant-delta preempt-KILLs** (K1552, F#704, F#709, **this**)
- 6 F#669-family preempt-KILLs
- 5 novel-mechanism PROVISIONALs + 1 hygiene-patch PROVISIONAL (F#702)
- 3 SUPPORTED + 1 regular KILL
- 3 template-regression sub-variants promoted (F#705/F#708/F#709); 1 candidate 4th (paired-PROXY-half-strip) deferred
- 2 proxy-only-lineage-inheritance watchlist instances (F#710/F#711)

Ready for reviewer `review.killed` emission.
