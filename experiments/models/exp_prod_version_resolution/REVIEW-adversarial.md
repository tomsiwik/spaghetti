# REVIEW-adversarial: `exp_prod_version_resolution`

## Self-attack: "Is KILLED_PREEMPTIVE really honest here?"

### Attack 1 — "T1 is gameable; you could `pip install semver` and claim the resolver is present."
**Rebuttal:** T1 has 4 required artifacts. Even if `semver` were
installed as a dep, the binding shortfall is the two *structural data*
absences (`multi_version_adapter_registry`, `multi_version_base_model_hash_set`).
Those cannot be resolved by a `pip install`: the repo has exactly one
`spec_version` value (1) and exactly one `base_model_id` value
(`mlx-community/gemma-4-e4b-it-4bit`). No resolver resolves anything
with a registry of one element. **T1 survives the attack.**

### Attack 2 — "The runner's T5 breach-B regex missed Assumption 3 because of Markdown emphasis. Your automation is sloppy."
**Rebuttal:** Fair. The automated regex matched 4/5 breaches rather
than 5/5 due to the `**verify**` Markdown emphasis. This is documented
in PAPER.md §Assumptions A2 and in MATH.md T5. The verdict is
*unchanged* under either interpretation: 4 ≥ 3 threshold and the
honest reviewer is invited to read the source text directly. The
runner result is the conservative undercount; the MATH.md text is
the strict reading. **Both suffice to block.**

### Attack 3 — "You're sandbagging: K1663's semver resolver is a day of work, not operator-blocked."
**Rebuttal:** This is the strongest attack. Counter:
(a) PLAN.md Part 2 has not scoped a semver resolver; adding one is not
    a micro drain, it is a design extension.
(b) Scoping a Cargo-style resolver in-scope requires a source
    experiment that defines "what does 'correct adapter' mean" —
    something source exp_prod_adapter_format_spec_v1 does NOT define
    (it only specifies a u32 spec_version=1 field). A resolver without
    a semantic dependency on the format's range operators cannot be
    declared correct.
(c) Even granting the resolver were implemented, K1664 requires
    "major version bump of base invalidates adapter" — that needs at
    least two base-model versions with independent hashes. The repo
    has exactly one Gemma 4 base; the operator hasn't scoped a second.
(d) Running 20 scenarios on an un-designed resolver would fall into
    F#173 (theory-aggregation-non-transfer): the success-path
    round-trip of `adapter_format_spec_v1` does not imply the
    registry-level resolver behaves correctly. We have already seen
    this antipattern kill `exp_g4_crystallize_real_users` (iter 34)
    and four F#502-class targets.

### Attack 4 — "You labelled this 'composition-bug (software-infrastructure-unbuilt)'. Why not ap-017 (s) like iters 35-36?"
**Rebuttal:** Iters 35-36's (s) axis covered *physical* or *external*
infrastructure absence: CUDA hardware (iter 35), public DNS/TLS
(iter 36). Those cannot be fixed without a hardware procurement or
an external provider. This iter's absence is *in-repo software* — no
hardware or external service is required; the missing artifacts are
a library install + two data points. That's qualitatively different
and fits the composition-bug axis better (source + target compose,
target requires a module that composition never built). The analyst
is the correct forum to arbitrate if this distinction should be
formalized (or collapsed) when the cap is lifted.

### Attack 5 — "Running it anyway takes only 25 min per your T2. Why not just run it?"
**Rebuttal:** Running the resolver without a scoped definition of
"correct" produces a paper whose verdict is not a research claim but a
tautology — "the resolver I wrote returns the result I defined as
correct." That is the textbook `tautological-routing` antipattern
(ap-017 (c), 3 prior instances). T2 budget would be consumed, but
the output would fail pre-flight check #5 (no KC was added/relaxed
silently — but also no KC was ever *meaningful*; success criteria are
literally empty in the DB) and pre-flight check #6 (tautological-
routing antipattern would apply). **Running is more expensive than
preempting, not cheaper, because a 25-min run that gets reclassified
as `killed` costs the same preempt-line in the DB as the 3-s runner
that correctly pre-flighted it.**

### Attack 6 — "You're fatigued from 36 iters of drain-forward. Confirmation bias."
**Rebuttal:** The runner is pure stdlib; it grep-probes the repo for
artifact presence and reports numbers. The numbers produced (shortfall
= 2, pin ratio = 0.333, T5 hits = 4) were not pre-decided. The
predictions in MATH.md §PREDICTIONS P1/P2/P3 were written before the
runner ran; they were all confirmed by the runner. The blocking
verdict (all_block = True) is overdetermined: any one of T1, T3, T5
alone blocks, so a single sloppy-count would not change the outcome.
**Low confirmation-bias risk for this specific preempt.**

### Attack 7 — "No REVIEW without proposing a v2."
**Rebuttal:** This is an intentional non-goal (PAPER.md §Non-goals).
Proposing a v2 experiment without operator scope-expansion would
violate guardrail 1002 ("NEVER generate experiments from analogies")
— the analogy here would be "we had resolvers in Cargo, therefore we
need one in Pierre", unanchored in any in-repo SUPPORTED finding.
The honest v1 → v2 bridge is an operator decision; the researcher's
responsibility ends at the preempt line.

## Verdict after self-attack
KILLED_PREEMPTIVE remains ratified. The strongest attack (Attack 3)
exposes that K1663 alone *could* be tractable if scoped, but K1662 and
K1664 cannot be tractable without additional SUPPORTED dependencies
that the operator has not declared. Preempt stands.

## Follow-up recommended to operator (outside the loop)
1. Declare whether the semver resolver is in-scope for PLAN.md Part 2.
   If yes, open a new *source* experiment that defines "correct
   resolution" semantics (K1662/K1663/K1664 would then inherit a
   meaningful oracle).
2. Declare whether multi-version base-model support is in-scope (Gemma
   4 → Gemma 5 transition, etc.). If yes, scope a base-model-version
   registry experiment first.
3. Until 1 and 2 are declared, downgrade `exp_prod_version_resolution`
   to P≥3 (out of drain scope).

## Confidence in this review
~90. Defense-in-depth across T1, T3, T5 gives three independent block
arguments; attacks 1-4 are neutralized by structural (non-hyperparameter)
arguments; attacks 5-7 are neutralized by antipattern/guardrail citations.
The remaining 10% risk is that the operator has a PLAN.md Part 2 update
in flight that I have not seen, which would re-scope the experiment —
in which case the preempt is correct *now* and a v2 can be designed *then*.

---

## Reviewer verdict (iter 29) — KILL (ratify)

Adversarial checklist over `MATH.md`, `PAPER.md`, `results.json`, `run_experiment.py`:

| Check | Status |
| --- | --- |
| (a) results.json verdict ↔ DB status: `KILLED_PREEMPTIVE` ↔ `killed` | CONSISTENT |
| (b) `all_pass=false` ↔ kill status | CONSISTENT |
| (c) PAPER.md verdict line says KILLED_PREEMPTIVE (no silent upgrade) | CONSISTENT |
| (d) `is_smoke=false`, claim is full pre-flight stack (not smoke) | CONSISTENT |
| (e) KCs K1662/K1663/K1664 = target's declared KCs; pre-reg preserved | CONSISTENT |
| (f) Tautology sniff: no empirical run ⇒ no tautological KC path | N/A |
| (g) K-IDs in code/results match DB literal K1662/K1663/K1664 | CONSISTENT |
| (h) LoRA composition bug greps (no composition here; pure stdlib) | N/A |
| (i) `LORA_SCALE ≥ 12` (none) | N/A |
| (j) per-sample routing bugs (none) | N/A |
| (k) `shutil.copy` adapter spoof (none) | N/A |
| (l) hard-coded `{"pass": True}` KC dict (none) | N/A |
| (m/m2) target model ≠ loaded model (no model loaded; no MLX skill required) | N/A |
| (n-q) eval integrity (no empirical run) | N/A |
| (r) prediction-vs-measurement table present | PASS |
| (s) math errors: none — defense-in-depth logic `(T1 ∨ T3 ∨ T5) ⇒ block` is sound | PASS |

**Verdict: KILL (ratify KILLED_PREEMPTIVE).** Defense-in-depth holds:
T1 shortfall=2 (data absences; no pip-install can resolve a single
`spec_version`=1 set or a single `base_model_id`), T3 is the 5th
F#502/F#646 DB-literal incomplete hit, T5 has 4/5 automated (5/5
manual) literal source-scope breaches against Assumption 1 + Assumption
3 of `exp_prod_adapter_format_spec_v1` — each alone is sufficient.
Source-SUPPORTED preempt #14. ap-017 scope index 33. Runner is pure
stdlib, 2.48 s wall, no model touched.

**Non-blocking note:** axis label "composition-bug (software-infrastructure-
unbuilt variant)" is a reasonable new sub-label; analyst should arbitrate
whether this sub-axis merits formal split from plain composition-bug
when the LEARNINGS cap is lifted (debt now 11).

No revise cycle. Route: `review.killed` → analyst.
