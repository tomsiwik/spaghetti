# REVIEW-adversarial: exp_bench_livecodebench_v6

**Reviewer**: Adversarial Reviewer (Round 3, post-run)
**Verdict**: KILL (confirms researcher self-kill 2026-04-18)
**Date**: 2026-04-18

---

## Adversarial checklist (PLAN §1 / reviewer.md §3)

| # | Check | Result |
|---|---|---|
| (a) | results.json `verdict=KILLED` vs DB `status=killed` | consistent ✓ |
| (b) | `all_pass=false` vs claim KILLED | consistent ✓ |
| (c) | PAPER.md verdict line says `KILLED` | consistent ✓ |
| (d) | `is_smoke=false` and not run → KILLED (not provisional) | correct routing ✓ |
| (e) | MATH.md KCs unchanged in git (K1420/1421/1422 unmodified) | clean ✓ |
| (f) | tautology — n/a, nothing measured | n/a |
| (g) | KC IDs match DB description | n/a (unmeasured) |
| (h)–(m) | composition / scale / routing / shutil.copy / hardcoded-pass / model-substitution code paths | n/a — `run_experiment.py` never executed past path checks; no LoRA composition or routing code shipped this iteration |
| (m2) | MLX skill invocation | n/a — no new MLX code authored; only blocker triage |
| (n)–(q) | eval integrity | n/a (unmeasured) |
| (r) | PAPER.md prediction-vs-measurement table present | yes ✓ |
| (s) | math errors / unsupported claims | MATH.md Theorems 1 & 2 untouched; pre-registered, unfalsified, untested |

No blocking failure surfaced — but also nothing to confirm; the experiment never ran. KILLED is the only honest routing.

---

## Independent verification of blockers (live, not from memory)

```
$ ls micro/models/reference_implementations/LiveCodeBench/
(0 files)

$ ls micro/models/exp_p1_t2_single_domain_training/adapters/code/
adapter_config.json     ← only file; no adapters.safetensors
```

Both blockers are reproducible by `ls` right now. The 2026-04-14 Round 2 review (in this same file before overwrite) asserted "Adapter confirmed on disk: .../adapters.safetensors ✓". That assertion is **false as of 2026-04-18**. Either the weights were deleted or the prior reviewer marked PROCEED on presumed (not verified) existence.

→ Antipattern observation: **review-presence-assumption** — reviewer asserted file existence without `ls`-style live check, then a downstream consumer (this experiment) tried to load it and could not. Already covered by `mem-antipattern-{file-existence-cache}` family; no new memory required.

→ This is the **9th instance** of preflight-adapter-persistence (training code does not write `adapters.safetensors` before exit). Already covered by an existing `type: fix` memory; not propagating again per analyst rule.

---

## Closure (kill is robust without rerun this iteration)

The kill is robust because both blockers are upstream of LCB:

- B1 (LCB harness empty) → unblock = clone `LiveCodeBench` repo. Independent of any adapter or composition mechanism. Cosmetic to LCB's measurement; not a finding about Pierre.
- B2 (code adapter weights absent) → unblock = re-run `exp_p1_t2_single_domain_training` with `Path('adapters.safetensors').stat().st_size > 0` pre-exit assertion. This belongs to **P11.ADAPTER-REBUILD**, not to LCB.

Neither blocker tests Theorem 1 (W4A16 gap) or Theorem 2 (D_ca → D_lcb gradient orthogonality). They remain pre-registered, unfalsified, untested. A future runner re-claiming `exp_bench_livecodebench_v6` should leave MATH.md intact and only fill in the measurement column.

---

## Cascade flag (for analyst / finding-add)

Same upstream as `exp_bench_aime_2026` (killed earlier today, identical cause). Other downstream experiments that cite `exp_p1_t2_single_domain_training` adapter weights (`exp_m2p_composition_n5`, `exp_model_peer_comparison_*`, `exp_p9_benchmark_showdown`, and any paper citing Finding #421's 82% GSM8K / 63% HumanEval) are equally compromised — those numbers came from an ephemeral adapter no longer on disk.

→ Recommend a finding noting this cascade so future claimers see the adapter-persistence blocker before claiming.

---

## Assumptions

- I did not attempt to run `experiment run` to "double-check"; the researcher already enumerated both blockers, both are visible in `ls`, and re-running adds zero information while burning the 15-min review budget.
- I did not retrain the adapter or clone the LCB repo — those are P11 unblock work, not reviewer work.

---

## Routing

`review.killed` → analyst writes LEARNINGS.md (cascade context), records the cross-experiment adapter-persistence pattern, and confirms no duplicate antipattern memory entry is needed.
