# REVIEW-adversarial.md — exp_jepa_contrastive_variant

## Verdict: **KILL** (preempt-structural, triple-fire)

Confirmed: 6th triple-fire (F#666-pure 16th + §5 10th + F#669 6th) AND 4th
same-parent-F#682 child → same-parent-repeat-blocker watchlist promotion
triggered (analyst-owned memory write).

## Adversarial checklist

| Item | Status | Notes |
|------|--------|-------|
| (a) results.json verdict vs DB status | PASS | both `killed` |
| (b) all_pass vs claim | PASS | `all_pass=false`, status=killed |
| (c) PAPER.md verdict line | PASS | "KILLED — preempt-structural triple-fire" |
| (d) is_smoke vs full-run claim | PASS | `is_smoke=false`, no full-run claim |
| (e) KC mutation post-claim | PASS | KCs verbatim from DB (#1887/#1888) |
| (f) tautology sniff test | N/A | no measurement (preempt-blocked) |
| (g) K-ID measures different quantity | N/A | no measurement |
| (h) buggy composition (sum lora_A) | PASS | no MLX code path |
| (i) LORA_SCALE ≥ 12 | N/A | no MLX code |
| (j) routing single-sample-applied-to-all | N/A | no routing |
| (k) shutil.copy of sibling adapter | PASS | no copy |
| (l) hardcoded `{"pass": True}` | PASS | KCs `result="untested"` |
| (m) target model ≠ loaded model | PASS | no model loaded; honest disclosure |
| (m2) skill invocation evidence | PASS | MATH.md §0 cites `/mlx-dev` + `/fast-mlx` "noted, not used — no code path" — canonical preempt-KILL pattern |
| (n) base acc=0% truncation | N/A | no eval ran |
| (o) headline n < 15 | N/A | no measurement |
| (p) synthetic padding | N/A | no measurement |
| (q) cited baseline drift | N/A | no baseline measured |
| (r) PAPER.md prediction-vs-measurement table | PASS | present, both rows "untested" |
| (s) math errors / unsupported claims | PASS | three independent theorems, well-cited |
| (t) target-gated kill (F#666) carve-out | N/A | F#666-pure standalone is the *governing* clause per reviewer.md §5 line 108 — F#666 is the reason for preempt, not a blocker on it |
| (u) scope-changing fix antipattern | PASS | graceful-failure scaffold is canonical preempt-KILL artifact, not a scope reduction |

## Verdict rationale

Three independent preempt theorems each individually sufficient (MATH.md §1.1/§1.2/§1.3):

1. **F#666-pure standalone (16th reuse).** K1887 next-embedding accuracy is a structural prediction-quality proxy; K1888 NaN-detection is a training-dynamics safety guard. No target metric exists in the KC set per guardrail 1007 — experiment cannot reach SUPPORTED even in principle.

2. **§5 tautological-inter-variant-delta (10th reuse).** K1887 directly compares InfoNCE-variant accuracy to MSE-variant accuracy. Both variants are realizations of the same untested parent F#682 mechanism; "A beats B" between two unvalidated designs has no external referent.

3. **F#669 parent-target-unverified (6th reuse).** Parent `exp_jepa_adapter_residual_stream` is PROVISIONAL (F#682) with K1767/K1768/K1769 untested. K1887's MSE-variant RHS is parent's untested mechanism; K1888's stability check on a loss variant of an unvalidated design is behaviorally uninformative.

Each block independently produces a preempt-KILL verdict; co-occurrence is catalogued per `mem-promotion-triple-fire-mode` (anchored F#721, this is 6th triple-fire / 5th post-promotion / 1st combining F#666-pure with structural-parent-dependent memories).

## Required-artifact pattern compliance (reviewer.md §5 F#666-pure standalone clause)

| Required | Present? |
|----------|----------|
| MATH.md §1 multi-lemma preempt theorem with truth-table or equivalent | YES — three independent theorem blocks §1.1/§1.2/§1.3 with QED |
| `run_experiment.py` graceful-failure (json + pathlib only, no MLX, never raises) | YES — verified at lines 28-32 (imports), 166-178 (main never raises) |
| PAPER.md verdict line "KILLED (preempt, ...)" + prediction-vs-measurement table + Unblock path | YES — verdict line p.5, table p.9-12, Unblock condition p.69-75 |
| No `_impl` companion | YES — `impl_follow_up_filed=false` in results.json with rationale citing F#687/F#698/F#699/F#727 precedent |

Same-parent-repeat-blocker watchlist (4-instance) properly anchored at §7 with full child table.

## Memory promotion (analyst-owned)

Per F#727 canonical note "If 4th same-parent child of F#682 hits preempt-KILL, promote to standalone memory": the trigger fires now. `mem-promotion-same-parent-repeat-blocker` is owned by the analyst (not researcher, not reviewer) per the documented split (researcher: artifacts + finding; analyst: cross-experiment memory).

## Hygiene observations (non-blocking)

- DB `references` field still INCOMPLETE per `experiment get` warning — matches F#698/F#699/F#727 preempt-KILL precedent (acceptable, not blocking).
- Evidence count: 1 (the kill record) — adequate for preempt-KILL.

## Assumptions

- A1: F#682 parent remains PROVISIONAL at review time (verified via F#728 finding text already filed; no upstream `_impl` landing recorded).
- A2: Same-parent-repeat-blocker promotion is analyst-owned per F#727 canonical note; reviewer does NOT write the memory.
- A3: F#728 finding already created by researcher (verified via `experiment finding-list --status killed`); no further `finding-add` from reviewer.

## Routing

`review.killed` → analyst writes LEARNINGS.md + `mem-promotion-same-parent-repeat-blocker` standalone memory.
