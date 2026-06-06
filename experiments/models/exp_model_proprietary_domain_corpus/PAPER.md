# PAPER.md — exp_model_proprietary_domain_corpus

## Verdict
**KILLED_PREEMPTIVE** — target is unrunnable on `local-apple` / MLX /
48 GB M5 Pro within a 120 min budget. Two automated independent
preempt blocks fire (T2 ∧ T3); T1 is over-determined on a manual
re-read but returns 0/5 on the automated probe due to false-positive
co-occurrence matches (see A9 + §Risks below). T5 is **N/A** (target
declares `depends_on: []`). Any single automated block (T2 or T3)
suffices.

## Prediction-vs-measurement

| Pred | Prediction | Measurement | Status |
|------|-----------|-------------|--------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | Automated cooccur-grep: shortfall = **0/5**. All 5 matched via grep, but every match is a false positive: `data/distillation/python/eval.jsonl` etc. are **public distillation benchmarks**, not a non-public proprietary corpus; `split_hits` resolved against PPL-probe scripts that happen to contain both `split` and `corpus`; `trainer_hits` resolved against closed-benchmark SFT experiments (`exp_p1_c1_polar_scale_invariance`), none of which is a drop-in trainer for an arbitrary domain corpus. Manual re-read: **5/5 absent**. | Partial (automated) / PASS (manual) |
| P2 | T2 timing ≥ 120 min (conservative; floor also blocks) | Conservative: **160.0 min** (900s cold + 3600s SFT + 2·2400s eval + 300s load = 9,600 s); floor: **76.7 min** (30-min SFT + 100-Q eval). Conservative exceeds 120 min ceiling. | PASS (conservative) |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` marker + empty references | `Success Criteria: NONE — add with …` and `⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)` both present; `references` field empty. 11th occurrence of F#502/F#646 in drain. | PASS |
| P4 | T4 pin_ratio = 0 (dir absent); reinforce-only | `.audit/` absent; pin_ratio = 0.00; reinforce-only (does not block alone). | PASS (reinforce-only) |
| P5 | T5 returns N/A (no declared parent; depends_on=[]) | DB pretty-print contains no `depends_on:` or `Depends:` line with a declared parent; runner logs `{block: false, reason: "no_declared_parent", applicable: false}`. T5 does not participate in the verdict. | PASS (N/A) |

## Kill criteria result

| KC | Text | Result |
|----|------|--------|
| K1704 | Adapter trained on specialized corpus beats base Gemma 4 E4B by ≥ 10 pp on held-out eval from same distribution | **fail** (preempt — target not run) |
| K1705 | Base accuracy < 50 % on held-out questions from the corpus distribution | **fail** (preempt — target not run) |

## Runtime evidence
- Runner: pure stdlib + `experiment get` shell-out.
- Wall: **1.95 s**.
- Zero MLX, zero model load, zero HTTP bind.
- `results.json` contains full probe output.

## Assumptions (from MATH.md §4, verified at runtime)
- A1 grep scope: `*.py` under `pierre/`, `macro/`, `composer/`, `micro/models/` (excluding this runner) + file-presence check under `data/`, `corpora/` at repo root.
- A2 corpus-payload probe required > 10 KB files; fired on public distillation eval .jsonls — see A9.
- A5 T2 floor (76.7 min) is under ceiling but K1705 CI is meaningless at 100-Q, so floor is scientifically incoherent. Conservative (160 min) blocks.
- A6 T5 N/A: `depends_on: []` read literally from DB; retrofitting Finding #478 as a parent would violate T5 pre-reg.
- A8 runner is pure stdlib, ≤ 3 s wall — verified (1.95 s).
- A9 **T1 cooccur-grep has structural false-positive risk on this target**. The probe cannot distinguish "non-public proprietary corpus" from "any > 10 KB data file". A real fix would require (a) a provenance marker on the corpus file (README / LICENSE indicating `proprietary: true`) or (b) an operator-mounted path outside the repo-root data/ tree. The grep-hit files are all public: `data/distillation/*` are open distillation benches, `exp_p1_c1_polar_scale_invariance` trains on open benchmarks, PPL probes use open corpora. **Manual re-read confirms shortfall 5/5**: zero non-public corpora, zero held-out split loaders with corpus provenance, zero domain-corpus eval harness, zero generic-corpus SFT trainer, zero base-vs-adapter runner with a K1705-first gate. Verdict is over-determined by T2 ∧ T3 without T1; runner refinement backlog logged.

## Novelty vs prior drain
**NOVEL F-axis candidate**: `private-data-unobtainable-by-design`
under ap-017. Distinct from F#652 `software-infrastructure-unbuilt`
because the missing piece is **data** whose KC definition forbids it
from being public — even a perfectly-written pipeline cannot run
without an operator-mounted proprietary corpus. This is the first
drain preempt blocked primarily by a **data** gap (not code).

Reuses:
- F#502 (schema-completeness) — 11th occurrence.
- T4 pin-ratio reinforcer pattern.

Does **not** reuse F#652 as the primary axis (code exists to train
LoRA on Gemma 4 E4B — what is missing is the provenance-gated corpus).
Analyst hat owns final F-axis placement (sibling vs child of F#652).

T5-K (parent-KILLED inheritance) does **not** apply — no parent.

## Operator action required to unblock
1. Mount a non-public specialized corpus at a documented path (e.g.
   `corpora/<domain>/train.jsonl`, `corpora/<domain>/heldout.jsonl`)
   with a README declaring provenance and confirming it is not in any
   public pre-training dataset.
2. Write a deterministic split loader with a seed (train/heldout with
   documented ratio).
3. Write a domain-specific eval harness (MCQ / QA / exact-match /
   calibrated judge) with scoring calibrated to the corpus.
4. Wire an MLX-native LoRA SFT trainer on Gemma 4 E4B for the
   domain-corpus input shape (prose, QA pairs, or MCQ).
5. Implement a matched base-vs-adapter eval runner that:
   - measures K1705 (base < 50 %) **first** as an atomic gate, and
   - measures K1704 (adapter ≥ base + 10 pp) only if K1705 passes.
6. Register `success_criteria` and `references` in the DB.
7. Re-file target once (1)-(6) exist.

Until then, the experiment remains `killed` with a stable,
re-testable preempt theorem. Child experiments that depend on
non-public corpus evaluation inherit the block.
