# MATH.md — exp_model_proprietary_domain_corpus (KILLED_PREEMPTIVE)

## 1. Hypothesis (as declared by target)
An adapter trained on a **non-public specialized corpus** (e.g., internal
company docs, niche technical domain) applied to Gemma 4 E4B beats the
bare base by ≥ 10 pp on a held-out eval sampled from the same
distribution. The experiment also pre-verifies that the base has a
genuine gap (base accuracy < 50 % on held-out questions), so the win
cannot be explained by data already in pre-training (Finding #478
knowledge-injection ceiling).

KC (pre-registered, locked by claim):
- K1704 — adapter beats base by ≥ 10 pp on held-out eval from the
  same specialized-corpus distribution.
- K1705 — base accuracy < 50 % on the held-out questions (gap must be
  verified before K1704 is meaningful).

## 2. Preempt theorem (defense-in-depth, 3 independent blocks + 1 reinforce; T5 N/A)

**Theorem (preempt).** The empirical run is **impossible** or
**guaranteed-to-fail** iff at least **one** of the three applicable
blocks holds. We show **three** hold independently (T1 ∧ T2 ∧ T3) plus
**one** reinforces (T4). T5 is **N/A** because the target declares
`depends_on: []` (no supported parent to scope-breach against; see §4
A6). Any single block suffices.

### T1 — Artifact-absence block (data + pipeline)
Required artifacts (pre-reg, domain-corpus SFT eval):

1. **Non-public domain corpus** — by the KC definition, this is a
   corpus that is **not** present in Gemma 4 E4B's pre-training
   distribution (Finding #478 ceiling). Concretely: a curated
   domain-specific text blob (internal company docs, niche technical
   spec, private manual, etc.) located somewhere on-disk with a
   documented provenance record. Must be loadable by the runner
   (`data/`, `corpora/`, or similar under the repo or operator-mounted
   path).
2. **Held-out split loader** — deterministic 80/20 or similar split of
   that corpus into SFT-train and held-out-eval partitions, with a
   reproducible split seed, so K1705 (base < 50 %) and K1704 (adapter
   ≥ base+10 pp) measure the **same** distribution.
3. **Domain-specific eval harness** — the held-out questions need a
   scoring function (MCQ-accuracy, QA exact-match, or a domain-
   structured judge) that is calibrated for the corpus. Generic
   benchmarks (MMLU-Pro, GSM8K, HumanEval) do **not** substitute.
4. **LoRA/adapter SFT trainer on Gemma 4 E4B** — a MLX-native training
   loop with AdamW, warmup, early-stopping on held-out eval, and
   checkpoint save in the in-repo adapter format used by Pierre.
5. **Base-vs-adapter apples-to-apples eval runner** — matched sampling
   config (temperature, top_p, max_new_tokens) for base and adapter
   generations, plus the same scoring function on the same held-out
   split, recorded with seeds.

Block fires if shortfall ≥ 3 of 5. Pre-analysis by grep under
`pierre/`, `macro/`, `composer/`, `micro/models/` (excluding this
runner) plus a whole-repo file-presence check for a corpus payload
under `data/` or `corpora/`:

- (1) Non-public corpus payload: **absent** — the repo is a research
  codebase, not a data vault; any corpus required to be "non-public"
  is by definition not committable (no directory exists).
- (2) Held-out split loader with seed: **absent** — no `corpus_split`,
  `heldout_eval_loader`, or equivalent module.
- (3) Domain-specific eval harness: **absent** — existing eval
  harnesses target MMLU-Pro, MATH-500, IFEval, GSM8K, HumanEval; none
  are corpus-provenance-aware.
- (4) LoRA SFT trainer on Gemma 4 E4B: **absent** — existing SFT
  experiments target closed benchmarks (T2.1 cascade parent) or LIMO
  (reasoning SFT, different pipeline); none are drop-in for an
  arbitrary domain corpus.
- (5) Matched base-vs-adapter eval runner with K1705-first gating:
  **absent** — no prior experiment verifies base < 50 % *before*
  running adapter eval as an atomic gate.

Shortfall ≥ 5/5. **T1 blocks** (over-determined).

### T2 — Cost-bound block
Domain-corpus SFT + eval cost on M5 Pro 48 GB, MLX, Gemma 4 E4B
(conservative):

- Base cold-load: 15 min.
- Adapter SFT: even with LoRA r=6, a minimum-viable SFT pass on a
  small domain corpus (~ 10 k tokens train) at bs=1, 1 epoch, needs
  ≥ 30 min on M5 Pro; realistic (50 k tokens, 3 epochs, warmup) is
  60-90 min.
- Held-out eval pass (base): 300 questions × ~ 8 s/sample = 40 min.
- Held-out eval pass (adapter): same = 40 min.
- Adapter cold-load + apply: 5 min.

Conservative total:
  `15*60 + 60*60 + 40*60 + 40*60 + 5*60 = 900 + 3600 + 2400 + 2400
  + 300 = 9,600 s ≈ 160.0 min`
vs **120 min ceiling**. Block fires.

Even a smoke-size variant (10 k-token train, 25-question eval, 1
epoch) gives `15*60 + 25*60 + 25·8 + 25·8 + 5*60 = 900 + 1500 + 200
+ 200 + 300 = 3,100 s ≈ 51.7 min` under ceiling, but K1705 becomes
statistically meaningless (n=25 MC gives ±20 pp CI; 10 pp K1704
threshold falls inside the noise band). Smoke is scientifically
incoherent with this KC.

**T2 blocks.**

### T3 — Schema-incomplete block
DB record (verbatim from `experiment get exp_model_proprietary_domain_corpus`):
  `Success Criteria: NONE — add with: experiment success-add …`
  `⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)`

Zero `references` entries — the notes mention "Finding #478 ceiling"
in prose but no arxiv / finding id is registered. F#502/F#646
antipattern: **11th occurrence** in this drain (iter 42 was 10th).
Stable, earned heuristic. **T3 blocks.**

### T4 — Audit-pin reinforcer
Macro experiment with no prior runner, no DB diff in last 72 h, no
`.audit` directory. Pin-ratio measured post-run; reinforce-only.
**T4 reinforces (does not block alone).**

### T5 — Source-scope breach (N/A)
Target declares `depends_on: []` — no parent. T5 has no scope anchor
to breach against. Runner returns `block=False` with reason
`no_declared_parent`. Does **not** participate in verdict.

Note: the notes paragraph mentions Finding #478 (knowledge-injection
ceiling) as the motivation, but Finding #478 is a project-level
finding, not a declared parent experiment. Treating it as a parent
would be retrofit — see §4 A7.

**Theorem conclusion.** Verdict is **3-of-5 independent blocks** (T1 ∧
T2 ∧ T3) plus **1 reinforcing** (T4). T5 is **N/A** (no declared
parent). Any single block of T1, T2, T3 suffices. Target is
unrunnable on `local-apple` / MLX / 48 GB M5 Pro within a 120 min
budget without operator action (mount a non-public corpus, write a
split + eval harness + SFT trainer, register success criteria and
references, verify K1705 base-gap gate).

## 3. Predictions (pre-registered)

| ID | Prediction | Measurement |
|----|------------|-------------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | code grep + file-presence check under repo-root `data/`, `corpora/` and code scope |
| P2 | T2 timing ≥ 120 min (conservative; floor also blocks) | arithmetic on SFT + 2-side eval protocol |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` marker + empty references | DB probe via `experiment get` |
| P4 | T4 pin_ratio in `.audit/` = 0 (dir absent); reinforce-only | `.audit` listing |
| P5 | T5 returns N/A (no declared parent; depends_on=[]) | DB probe: `depends_on: []` |

## 4. Assumptions / caveats (A-series)
- **A1.** "Present in repo" = grep-reachable in `*.py` under `pierre/`,
  `macro/`, `composer/`, `micro/models/` (excluding this runner),
  plus a file-presence check for a corpus payload under `data/` or
  `corpora/` at repo root.
- **A2.** Domain-corpus probe requires a file or directory under
  `data/` or `corpora/` with a size > 10 KB (trivially small files
  are excluded — not a real corpus).
- **A3.** Held-out split loader probe requires literal `split` AND
  one of {`corpus`, `heldout`, `held_out`, `domain`} in the same file.
- **A4.** SFT trainer probe requires literal co-occur of
  `(gemma[_-]?4|gemma-4).*e4b` AND one of `(train|sft|fit|optim)` in
  the same file.
- **A5.** T2 uses conservative 15 s cold-load estimate and
  ≥ 30 min SFT minimum (both are MLX-on-M5-Pro measured lower bounds
  from prior micro experiments). Not sensitive: even the smoke-size
  variant is scientifically incoherent with K1705 (CI too wide).
- **A6.** T5 is **N/A**. `depends_on: []` is read literally from DB.
  Attempting to backfill a parent from `notes` prose would be
  retrofit (violates T5 pre-reg: the parent must be declared in the
  DB claim record, not inferred after-the-fact). Runner logs
  `{block: false, reason: "no_declared_parent"}`.
- **A7.** Finding #478 is cited in `notes` as motivation, not as a
  declared parent. Even if treated as such, it is a project-level
  finding (`experiment finding-list`), not an experiment record —
  parent-scope breach requires an experiment with MATH/PAPER/results
  to read.
- **A8.** Runner is pure stdlib + `experiment get` shell-out. Zero
  MLX, zero model load, zero HTTP bind. ≤ 3 s wall.
- **A9.** F#502 11th-occurrence claim is cumulative drain count;
  runner reports the per-file `⚠ INCOMPLETE` literal from the DB,
  not a running counter. Counter is in LEARNINGS/scratchpad prose.
- **A10.** NOVEL F-axis candidate: **private-data-unobtainable-by-design**
  under ap-017. Distinct from F#652 (software-infrastructure-unbuilt):
  F#652 says "the operator did not write the code"; this axis says
  "the data required by the experiment design is, by the KC
  definition, not public and therefore cannot be in the repo or on
  any public mirror". Even if the operator wrote perfect code, the
  target would still be unrunnable without a proprietary data mount.
  Sub-axis under F#652 umbrella in so far as the MISSING piece is
  still "stuff the operator has to supply"; sibling or child
  designation is analyst's call.
