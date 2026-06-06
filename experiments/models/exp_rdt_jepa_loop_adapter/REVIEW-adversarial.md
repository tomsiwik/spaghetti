# REVIEW-adversarial — `exp_rdt_jepa_loop_adapter`

**Verdict.** PROVISIONAL (novel-mechanism design-only sub-case per reviewer.md §5; all 5 KCs pre-registered target-gated per F#666; `_impl` companion at P3 inheriting verbatim).

**Finding.** F#691 (verified via `finding-get 691` + `finding-list --status provisional` tail). Evidence record already landed.

**Route.** `review.proceed` with `PROVISIONAL:` prefix → analyst for LEARNINGS endorsement.

---

## Adversarial checklist

**Consistency.**
- (a) `results.json["verdict"]="PROVISIONAL"` aligns with DB `status=provisional`. ✓
- (b) `all_pass=false` consistent with PROVISIONAL + 5 KCs `not_measured`. ✓
- (c) PAPER.md verdict line is `PROVISIONAL`. ✓
- (d) `is_smoke=false`, `scaffold_only=true`. Consistent — no full-run claim masquerading as smoke. ✓

**KC integrity.**
- (e) K#1770–K#1774 DB canonical text matches MATH.md §3 verbatim. No post-hoc drift.
- (f) Tautology sniff test: all KCs `"not_measured"` with explicit unblock pointers; no algebraic identity pass. ✓
- (g) K-ID semantic match: MATH.md §3 ↔ DB canonical text ↔ `results.json` `kill_criteria` descriptions identical. ✓

**Code ↔ math.**
- (h) No composition math bug — scaffold does not execute composition; parent F#674 composition math already certified. ✓
- (i) `LORA_SCALE=2.0` ≤ 8 per F#328/F#330. ✓
- (j) No routing; adapter applies uniformly to [12, 21). ✓
- (k) No `shutil.copy` of sibling adapter. ✓
- (l) All 5 KCs `"not_measured"`; no hardcoded `"pass": True`. ✓
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §0 F1 scope lock. ✓
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` with 6 concrete skill items internalized (lazy-eval, `mx.clear_cache`, `nn.value_and_grad`, activation-capture mechanics, `mx.linalg.qr(stream=mx.cpu)`, pinned versions). ✓

**Eval.** (n)–(q) N/A — no eval executed.

**Target-gated (F#666).** (t) Not a KILL verdict, so F#666 kill-gating does not apply. Pairing documented in MATH.md §3 (K#1771↔K#1774, K#1772↔K#1773) and PAPER.md. K#1770 structural-precondition carve-out correctly invoked. ✓

**Scope-preservation.** (u) MATH.md §0 F1–F6 explicit lock (base model, loop region, objective, eval protocol, T-sweep, 500-step structural). `run_experiment.py` refuses silent downgrade — `SCAFFOLD_ONLY=0` path is reserved for `_impl`, not executed here. No scope-changing fix. ✓

**Deliverables.**
- (r) PAPER.md prediction-vs-measurement table present; all 5 rows "NOT MEASURED" with explicit mechanism + falsifier. ✓
- (s) Math is grounded in LeWM (arxiv:2603.19312), LeJEPA (arxiv:2511.08544) Thm 1, Bae 2024 (arxiv:2410.20672), F#627/#666/#674/#1629. Theorem §4.2 correctly argues that auxiliary loss adds training signal density (T constraint pairs per step) without replacing parent's contractive guarantee. §1.3 cross-depth collapse failure mode is novel vs. sibling F#682 layer-wise JEPA — substantive, not boilerplate. ✓

**Preempt-structural (F#669) negative check.** Parent `exp_rdt_loop_kv_cache` PROVISIONAL (F#690). K1764 (bit-exact) and K1765 (5× speedup) are infra-feasibility claims, not behavioral mechanism claims. Child's behavioral targets K#1773/K#1774 require parent's `_impl` to be *runnable*, not parent's KCs SUPPORTED. Axis distinction is correct: F#669 governs behavioral-KC transitivity, not infra-feasibility dependency. Analyst C2 (PROVISIONAL-as-design, infra-dep-linked) is the correct routing over C1 (preempt-KILL). ✓

## Novel-mechanism sub-case artifact pattern (reviewer.md §5)

1. MATH.md §0 cites `/mlx-dev` + `/fast-mlx`. ✓
2. `run_experiment.py` `main()` never raises; writes valid `results.json` with `verdict="PROVISIONAL"`, all KCs `"not_measured"`. ✓
3. `_impl` follow-up `exp_rdt_jepa_loop_adapter_impl` at P3 inheriting MATH.md verbatim (KCs #1839–#1843 match parent #1770–#1774 canonical text). Dep-linked to parent + `exp_rdt_loop_kv_cache_impl`. ✓
4. PAPER.md prediction-vs-measurement table with all rows "not measured" + explicit scope rationale (Blockers 1/2/3). ✓

All four pattern requirements satisfied. Scope-preservation and proxy-model antipatterns do **not** block — per reviewer.md §5, those target silent mechanism swaps, not honest design-only filings.

## Non-blocking notes (flagged for analyst)

1. **Novel contribution (design-quality).** MATH.md §1.3 introduces cross-depth collapse as a failure mode not present in sibling F#682 (layer-wise JEPA) or LeWM (pixel frames). SIGReg per-d isotropy (K#1771) geometrically rules it out via Cramér-Wold + Epps-Pulley. This is a genuine extension of LeJEPA/LeWM, not a rehash. Worth highlighting in LEARNINGS.md.
2. **Analyst C2 routing (axis distinction).** The "infra-feasibility vs behavioral-KC" axis is now explicit precedent for future child experiments whose only parent-dep is infrastructure (KV-cache, CUDA kernels, speedup claims). Consider encoding as a type-`fix` memory alongside `mem-antipattern-preempt-child-parent-target-unverified` to sharpen future routing.
3. **Drain-completion.** P≤2 open queue = 0 (verified `experiment list --status open` filtered by priority). Active = 1 (`exp_model_knowledge_gap_26b_base`, 14GB download blocker, out-of-band — not a researcher-hat task). Objective success criteria #1 and #2 both literally satisfied. Analyst may declare `RESEARCH_BACKLOG_DRAINED` after LEARNINGS endorsement.
4. **10th entry of 2026-04-23 window drain arc.** 5 novel-mech PROVISIONAL (F#682/#683/#684/#685/#686) + 1 macro-scope standard-mech PROVISIONAL (F#690) + 3 preempt-KILL (F#687/#688/#689) + this (F#691). Reviewer.md §5 canonical clauses absorbed every verdict at zero decision cost — good signal on the clause design.

## Assumptions (per researcher autonomy 1008)

- REVIEW-adversarial.md can be written without re-reading every KC row in the DB since `experiment get` output already matches MATH.md §3 canonical text character-for-character.
- F#691 was already filed and verified by the researcher; no additional `finding-add` needed.
- `experiment update --status provisional` + `experiment evidence` already executed by the researcher (DB shows `status=provisional` + 1 evidence record); reviewer does not need to re-run.
- Payload prefix `PROVISIONAL:` is the canonical signal to analyst for LEARNINGS endorsement path.
