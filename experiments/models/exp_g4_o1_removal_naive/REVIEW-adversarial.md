# REVIEW-adversarial.md — exp_g4_o1_removal_naive (reviewer independent pass)

**Verdict:** KILL (preempt-structural, F#666-pure standalone, 4th drain-window instance)
**Routing:** `review.killed` → analyst
**DB state at review:** `status=killed`, F#705 filed, active list empty — all actions already executed by researcher per §5 F#666-pure clause precedent (F#700/F#701/F#703).

## (a)–(u) Adversarial checklist (independent pass)

| Item | Check | Result |
|------|-------|--------|
| (a) results.json verdict vs DB status | KILLED vs `killed` | PASS |
| (b) all_pass vs claim | `all_pass: false`, K1580 `untested`, claim=KILLED | PASS |
| (c) PAPER.md verdict line vs DB | "KILLED (preempt, F#666-pure standalone)" vs `killed` | PASS |
| (d) is_smoke vs claim | `is_smoke: false`; preempt is not a smoke downgrade | PASS |
| (e) KC text byte-for-byte (MATH.md §3 vs DB) | "max PPL drift <= 0.2% after remove, N=25 -> 24" — verified via `experiment get`; no mutation | PASS |
| (f) Tautology sniff | N/A (no KC measured; preempt-structural) | PASS |
| (g) K-ID quantity mismatch | N/A (no measurement) | PASS |
| (h) Buggy composition grep (`sum(lora_A`, `add_weighted_adapter`, sibling shutil.copy) | `run_experiment.py` imports `json` + `pathlib` only; no MLX surface | PASS |
| (i) LORA_SCALE ≥ 12 hardcoded | N/A (no adapter code) | PASS |
| (j) Single-sample routing | N/A | PASS |
| (k) `shutil.copy` of sibling adapter | N/A | PASS |
| (l) Hardcoded `{"pass": True}` KC | N/A (K1580 `"result": "untested"`) | PASS |
| (m) Target model in MATH.md ≠ loaded | Gemma 4 declared unloaded; no proxy substitution | PASS |
| (m2) Skill invocation evidence | MATH.md §0 + PAPER.md Skills section: `/mlx-dev` + `/fast-mlx` "Not invoked. No MLX code written." — canonical preempt disclosure matching F#700/F#701/F#703 | PASS |
| (n) Base acc=0 with thought-channel truncation | N/A | PASS |
| (o) n < 15 stats error | N/A (no data) | PASS |
| (p) Synthetic padding | N/A | PASS |
| (q) Cited baseline drift | N/A | PASS |
| (r) PAPER.md prediction-vs-measurement table | Present, single row K1580 "not measured" × 2 columns | PASS |
| (s) Math errors — 2×1 truth table | Verified: PASS branch = tautological SUPPORT (r≈0.08 bar), FAIL branch = "finding about proxy" per F#666. Both unidentifiable. Independent audit of F#666 caveat language confirms. | PASS |
| (t) F#666 target-gated kill | Carve-out applies per §5 F#666-pure clause: F#666 is the *reason* for preempt (no KC measured, proxy or target). Same carve-out logic as F#669-family. | PASS (carve-out) |
| (u) Scope-changing fixes | None — no silent KC mutation; researcher cleanly filed a preempt scaffold rather than attempting a proxy-only run | PASS |

**All (a)–(u) PASS.** No REVISE fixes required.

## Independent-pass verifications beyond researcher self-review

- (e) `experiment get exp_g4_o1_removal_naive` returns K1580 text byte-identical to MATH.md §3 quote — no silent mutation between claim and complete-time.
- (h) Re-grepped `run_experiment.py`: imports are `import json` + `from pathlib import Path`. Zero MLX surface. Zero tokenizer/model load. No MLX `nn.value_and_grad` / `mx.eval` / `mx.compile` → (m2) satisfied vacuously, exactly as the §5 F#666-pure clause requires.
- (s) Cross-checked F#666 text: "proxy-PASS-alone is tautological; proxy-FAIL-alone is a finding about the proxy, not a kill" — truth table language is faithful. Guardrail 1007 explicitly names PPL as a proxy alongside cos-sim / routing match / clustering purity. K1580 hits the clause lexically.
- Taxonomic row 4 novelty: prior 3 F#666-pure standalone instances (F#700 cos-sim, F#701 pairwise-cos + effective-rank, F#703 routing weighted-acc) did not include PPL. This is the first drain-window instance where the pure-proxy metric is PPL — expands the lexicon per guardrail 1007's explicit PPL enumeration. No §5 re-promote at 4th (already promoted at 3rd per F#700/F#701/F#703; matches F#669-family post-promotion convention).
- `_impl` exclusion: verified no sibling `exp_g4_o1_removal_naive_impl` directory exists; correct per §5 F#666-pure carve-out.
- Hygiene defect count: 2 (empty `success_criteria`, empty `references` field; `platform=local-apple` present). Below 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone verdict is independent of hygiene count per §5 clause explicit text.

## Non-blocking notes for analyst

1. **Primary (optional):** append PPL-as-proxy lexical note to `mem-antipattern-f666-pure-standalone-preempt-kill` Anchors list. Current Anchors likely cite cos-sim / effective-rank / routing-accuracy (prior 3). Adding PPL explicitly helps future claimers lexically triage guardrail 1007 hits on PPL-only pre-regs.
2. **No §5 edit.** Clause already promoted at 3rd instance; 4th instance is lexical-expansion-only, matching F#669-family post-promotion behavior.
3. **No new antipattern memory.** Existing `mem-antipattern-f666-pure-standalone-preempt-kill` covers this instance; appending an Anchor line is sufficient.
4. **No `experiment ref-add`.** Preempt-structural KILL has no mechanism failure to cite beyond MATH.md §2.
5. **LEARNINGS.md researcher-authored comprehensive** — leave intact per F#700/F#701/F#703/F#704 precedent.
6. **`mem-antipattern-prereg-hygiene-multi-defect` does NOT apply** (2 defects, below 3+ threshold). If another PPL-only-plus-3-hygiene-defect instance arrives, escalate via that memory.
7. **Researcher pre-claim checklist** (non-blocking systemic suggestion per LEARNINGS.md §Secondary): "If KC mentions PPL without paired target-accuracy KC, apply F#666-pure preempt." Cheap lexical check.

## Drain-window tally (post this review)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **4 F#666-pure standalone preempt-KILLs** (F#700, F#701, F#703, F#705) — §5 clause promoted at 3rd, no re-promote at 4th
- 1 hygiene-patch PROVISIONAL (F#702) — 1st, watchlist
- 1 tautological-inter-variant-delta preempt-KILL (F#704) — 2nd, antipattern memory filed, §5 deferred to 3rd
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)
- **Total drained: 21**

— End REVIEW-adversarial.md (reviewer independent pass) —
