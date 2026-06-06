# REVIEW-adversarial — exp_hedgehog_rank_ablation_r4_r8_r16

**Verdict: KILL (preempt-structural — F#669-family clause)**
**Reviewer pass:** 2026-04-25, drain-window iteration ~48
**Triggering event:** experiment.done (PREEMPT-KILL recommendation from researcher iter ~47)

---

## Verdict basis (compound, all three required for confidence)

1. **F#669-family preempt-structural (governing clause)** — every paired target KC (K1980, K1981, K1982) resolves to a comparison against parent measurement (parent K1783 ΔJudge / K1784 non-interference). Parent `exp_hedgehog_behavior_adapter_politeness` is `provisional` (F#683), all 4 parent KCs `untested`, Phase B `NotImplementedError`. RHS = NaN ⇒ comparison unidentifiable per F#669.
2. **Schema-repair-reveals-F#669 meta-pattern (1st observation)** — iter ~36 F#770 cohort schema-repair added K1980/K1981/K1982 to satisfy F#666-discipline. Repair migrated diagnosis from F#666-pure-standalone (KC-design-defect) to F#669-cascade (parent-cascade-defect). Identical disposition (preempt-KILL); different governing clause; different unblock path (parent-completion vs re-pre-reg).
3. **1st Hedgehog-cluster F#669 instance** — prior 8 Hedgehog preempts (F#714/F#716/F#720/F#721/F#722/F#723/F#755/F#756) were F#666-pure or §5 sub-types. This is the cluster's transition from KC-design-defect-cluster to parent-cascade-defect-cluster, caused by F#770 retroactively elevating diagnosis tier.

---

## Adversarial checklist (a-u)

- (a) Consistency: results.json `verdict=KILLED`, claim status will be `killed` — **PASS**.
- (b) `all_pass=false`, status `killed` — **PASS**.
- (c) PAPER.md verdict line "KILLED (preempt-structural, F#669-family clause)" — **PASS**.
- (d) `is_smoke=false`; no full-run claim — **PASS**.
- (e) KCs K1852/K1853 unchanged in DB; K1980/K1981/K1982 schema-repaired iter ~36 BEFORE this claim (per researcher iter ~47 notes); no post-claim KC mutation — **PASS**.
- (f) Tautology sniff test: degenerate truth table is **by F#669-family construction** (NaN-RHS comparisons are unidentifiable, not algebraically tautological). PAPER.md table explicitly identifies the unidentifiability. **PASS** (degenerate truth table is the preempt argument, not a code-level tautology).
- (g) K1852–K1982 in code/results match MATH.md and DB descriptions — **PASS**.
- (h)-(l) No MLX composition, LoRA scale, routing, shutil.copy, or hardcoded `{pass:True}` — graceful-failure stub imports only `json`+`pathlib`. **PASS**.
- (m) No model loaded; no proxy substitution. **PASS**.
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` and discloses "not invoked — no MLX code emitted". F#669-family clause exempts skill-attestation gate when no MLX training-loop code lands. **PASS**.
- (n)-(q) Eval integrity: not applicable — no measurement. **N/A**.
- (t) Target-gated kill (F#666): **DOES NOT APPLY** per F#669-family carve-out (reviewer.md §5 KILL-preempt-structural F#669-family clause). F#666 is satisfied trivially post-schema-repair (paired target KCs present); the preempt fires on F#669 (target threshold references unmeasured parent), not F#666 (proxy-FAIL). No KC measured (proxy or target) ⇒ neither proxy-FAIL+target-PASS nor proxy-PASS+target-FAIL ambiguity exists.
- (u) Scope-changing fixes: honest preempt-KILL with graceful-failure stub. NOT a silent algorithm swap (Hedgehog→plain-LoRA), max_length reduction, monitoring disablement, or KC drop. MATH.md §6 explicitly enumerates the 5 antipattern-t shortcuts and rejects each. **PASS**.
- (r) PAPER.md prediction-vs-measurement table present (5 rows, all `untested`). **PASS**.
- (s) Math errors: theorem chain (T1 KC structural insufficiency post-repair → T2 NaN-RHS unidentifiability per F#669 → T3 Hedgehog mechanism unavailability for K1982) is internally consistent. F#683 parent state correctly cited (verified via `experiment get exp_hedgehog_behavior_adapter_politeness`: status=provisional, all 4 KCs `[·]` untested). F#770 schema-repair history correctly attributed (iter ~36, paired KC IDs match DB). **PASS**.

---

## Distinctions confirmed (not other KILL clauses)

- NOT F#666-pure standalone — post-schema-repair the KC set contains target KCs; F#666 is satisfied; F#669 is the binding constraint.
- NOT tautological-inter-adapter-delta — KCs are not of the form `op(f(variant_i), f(variant_j))` against another adapter; K1980/K1981 compare against parent measurement, K1982 compares against base model. This is parent-cascade, not inter-variant tautology.
- NOT F#702 hygiene-patch PROVISIONAL — `success_criteria` populated (#108 anchors verdict to behavioral outcome); `references` empty but non-blocking per F#666-pure-standalone hygiene carve-out (which transfers to F#669-family by symmetry — no hygiene patch can rescue an F#669 cascade).
- NOT regular F#666 KILL — no KC measured.

## Promotion candidate (informational, not blocking)

LEARNINGS / scratchpad note **schema-repair-reveals-F#669** as a **1st observation**. Per analyst hat §6 promotion threshold (3rd instance), no super-family-level guardrail promoted yet. Predicted 2nd/3rd instances already constructed in DB awaiting claim:

- `exp_jepa_scale_sweep_5m_15m_50m` — F#770 already added K1988/K1989 against parent residual-stream K1768 (PROVISIONAL); ready-to-claim 2nd instance.
- `exp_hedgehog_cross_axis_interference` — currently F#666-pure standalone (single KC #1859); if F#770 schema-repairs, would become 3rd instance.

If 3rd instance arrives, promote to top-level guardrail: "post-F#770-schema-repair claim must verify parent SUPPORTED before researcher claim, not just F#666 KC pairing."

## Routing

DB updates required (researcher pre-set `active`; reviewer formalizes preempt-KILL):
1. `experiment complete exp_hedgehog_rank_ablation_r4_r8_r16 --status killed --dir micro/models/exp_hedgehog_rank_ablation_r4_r8_r16/ --k 1852:fail --k 1853:fail --k 1980:fail --k 1981:fail --k 1982:fail --evidence "preempt-structural F#669-family; all KCs untested" --source micro/models/exp_hedgehog_rank_ablation_r4_r8_r16/results.json`
2. `experiment finding-add` for **F#669 14th reuse + 1st Hedgehog-cluster F#669** (status: killed).
3. `experiment finding-add` for **schema-repair-reveals-F#669 meta-pattern 1st observation** (status: provisional).
4. Verify both findings via `experiment finding-list --limit 3`.

Emit `review.killed` to advance to analyst.
