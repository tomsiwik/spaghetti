# PAPER: Answer-only PPL as adapter router (K1567)

## Verdict: **KILLED** (seed 42)

K1567 required two conditions to hold simultaneously:
1. Top1_ans ≥ 0.85  **PASS** (measured 0.978)
2. Top1_full < 0.85  **FAIL** (measured 0.984)

Full-sequence PPL *also* routes correctly at ≥85% top-1 on this synthetic
setup, eliminating the marginal-contribution argument for answer-only PPL
as a router. The conjunction required by K1567 fails; per the pre-registered
criterion, the experiment is KILLED.

## tl;dr
- Answer-only PPL routes 1000 mixed-domain queries to their domain expert
  at 97.8% top-1.
- Full-sequence PPL routes at 98.4% top-1 (6pp higher, still within noise).
- Both are far above the K1567 threshold of 0.85.
- The thesis — that answer-only PPL is strictly better than full-seq PPL for
  adapter routing — is refuted for this setup.

## Prediction vs measurement (K1567)

| Quantity | Predicted (MATH.md §3) | Measured | Judgement |
|---|---|---|---|
| Top1_ans | ≥ 0.90 (central); K1567 lower bound 0.85 | 0.978 | above central prediction |
| Top1_full | ∈ [0.20, 0.60] (chance + drift); K1567 needs < 0.85 | 0.984 | **strictly refutes prediction** |
| K1567 conjunction | PASS | **FAIL** | **KILLED** |

The central prediction for Top1_full was that it would route at near-chance
to mildly-above-chance levels, because the predecessor measured r_full = −0.31
between full-seq PPL improvement and accuracy improvement. That prediction
was wrong: *absolute* full-seq PPL of expert i on domain i is strictly lower
than expert j's (j ≠ i) full-seq PPL on the same queries, even when expert i's
full-seq PPL is degraded relative to base.

## Per-domain top-1 (seed 42, N_q=200)

| Domain | Top1_ans | Top1_full | Notes |
|---|---|---|---|
| arithmetic | 1.000 | 1.000 | distinct alphabet (digits + `+`, `=`) |
| reverse | 0.950 | 0.935 | sort/reverse share `>` + lowercase alphabet |
| repeat | 1.000 | 1.000 | distinct alphabet (lowercase + `*`, `=`) |
| sort | 0.940 | 0.985 | confused with reverse expert |
| parity | 1.000 | 1.000 | distinct alphabet (`0/1` + `>` + `even/odd`) |

Confusion structure (from `results.json["confusion_*"]`):
- Full-seq: reverse→sort 13/200 (6.5%); sort→reverse 3/200 (1.5%).
- Answer-only: reverse→sort 10/200 (5%); sort→reverse 12/200 (6%).

sort and reverse share character alphabet and delimiter `>`; they are the
only non-trivial confusion pair. Both metrics cope with this at ~95% accuracy.

## Why the prediction was wrong

### 3.1 Reconciliation with predecessor (r_full = −0.31)

The predecessor measured r(ΔPPL_full, ΔAccuracy) where Δ denotes expert-minus-base
(relative change). Specifically, reverse and sort experts showed WORSE full-seq
PPL than base on their own domain (because prompt PPL degraded more than
answer PPL improved, weighted by token counts).

This experiment measures argmin_j PPL_full(θ_j, q) where j ranges over experts
(not including base) and q is a domain-i query. The ranking question is:
*among the 5 experts*, which has the lowest absolute full-seq PPL on q?

**The predecessor's result ≠ this experiment's result**, because:
- Expert-j's *degradation relative to base* can be bad (r_full = −0.31), but
- Expert-j's *absolute* full-seq PPL on domain-j queries is still lower than
  expert-k's full-seq PPL on domain-j queries (for k ≠ j), because
  expert-k has NEVER seen domain-j data and scores domain-j tokens with
  no domain-j-specific parameter adjustment.

The routing-via-full-seq-PPL works *not because full-seq is a good quality
signal* but *because experts trained on their own domain score their own
domain's queries best, independent of prompt-vs-answer decomposition.*

### 3.2 Why the synthetic setup is permissive

The 5 synthetic domains have largely disjoint character distributions:
- arithmetic: digits 0–9, `+`, `=`
- reverse: lowercase a–z, `>`
- repeat: lowercase a–h, `*`, `=`
- sort: lowercase a–p, `>`
- parity: digits 0–1, `>`, letters `e/v/n/o/d/d`

Any model that overfits to one domain will assign near-zero probability to
out-of-domain characters. The prompt-PPL degradation is in the cross-domain
direction, not the own-domain direction. When we route on own-domain queries,
the expert's own-domain prompt PPL is not degraded, because the expert saw
own-domain prompts during training. The predecessor's "prompt degradation"
effect only manifests when the expert is applied to OTHER domains' queries —
which is exactly the regime where we want high cross-domain PPL (i.e., high
PPL under the wrong expert).

The predecessor measured prompt-PPL change of expert i on domain i (own-domain),
averaged across many queries. For some domains (reverse, sort) the own-domain
prompt PPL got *worse*, because the expert's internal representations shifted
aggressively toward answer-generation, mildly harming own-domain prompt modeling.
But this mild degradation is much smaller than the cross-domain prompt PPL
penalty for expert-k on domain-j (because expert-k's token distribution is
tuned for domain-k, not domain-j). The full-seq routing gap is still
dominated by cross-domain distribution mismatch, not by own-domain prompt
degradation.

### 3.3 Where the prediction would be right

The full-seq routing would fail (Top1_full near chance) in settings where:
- Domains share token alphabets substantially (e.g., English prose domains).
- Prompts are much longer than answers (T_p ≫ T_a), making full-seq PPL
  dominated by prompt modeling.
- Expert-k's prompt modeling of domain-j is comparable to expert-j's
  (because domains share vocabulary).

The synthetic setup here has disjoint alphabets and balanced T_p/T_a, so
full-seq routing is not stress-tested. K1567 as stated does not hold in
this regime.

## Dependency state

All dependencies are in-process. No on-disk adapters. Antipattern-017
(config-only adapter dirs) does not apply.

| Resource | Required | Present |
|---|---|---|
| `micro.models.answer_conditioned_scoring` (module) | yes | yes |
| `adapter_config.json` stubs | no | n/a |
| `adapters.safetensors` | no | n/a |

## Antipattern self-check

| Antipattern | Triggered? | Evidence |
|---|---|---|
| ap-001 composition-math-bug | no | no composition; single-expert per PPL eval |
| ap-002 tautological-routing | no | per-query argmin over experts; 1000 distinct queries |
| ap-003 model-substitution | no | same d=32/H=2/L=2 numpy transformer as predecessor |
| ap-004 KC-swap | no | MATH.md §0 locked at pre-reg commit; single-commit `git log` for MATH.md until now |
| ap-005 verdict-consistency | ok | `results.json.verdict = "KILLED"` matches `all_pass=false` |
| ap-006 smoke-as-full | no | N_q=1000 queries (200 per domain), `is_smoke=false` |
| ap-007 tautological-KC | no | K1567 plausibly fails — and did, in the `full-seq also works` direction |
| ap-008 KC-threshold-relax | no | threshold 0.85 unchanged from MATH.md |
| ap-012 hardcoded-pass | no | K1567.pass computed from measured top-1 |
| ap-014 copy-paste-scaffolding | no | delimiters looked up via `DOMAIN_DELIMITERS[domain]`; no DOMAIN_KEYWORDS copied |
| ap-017 adapter-weights-missing | no | no on-disk adapter load; experts held as Python dicts |

## What was learned

1. **K1567 is refuted on disjoint-alphabet synthetic domains.** Both full-seq
   and answer-only PPL achieve ≥97% top-1, so the "full-seq fails" clause of
   K1567 does not hold.

2. **Routing accuracy ≠ predecessor's correlation sign.** Finding #553
   (r_full = −0.31 between full-seq-PPL improvement and accuracy improvement)
   does NOT imply full-seq routing fails. Relative-change correlation
   answers "does PPL change predict accuracy change?"; absolute ranking
   answers "which expert has lowest PPL on this query?". These are
   different quantities and can disagree.

3. **The predecessor's "prompt degradation" effect is own-domain only,
   and mild.** Cross-domain prompt PPL penalty (expert-j applied to domain-k
   inputs) dominates the routing signal. Any metric that's monotone in
   token-level cross-entropy over a sufficient number of own-domain tokens
   will route correctly.

4. **Domain overlap is the missing axis.** The question "does answer-only
   PPL route better than full-seq PPL?" only has substance when domains share
   token alphabets. On this synthetic setup they do not, and the question
   is vacuous.

5. **sort and reverse are the only routing confusion pair.** Both share
   lowercase-alphabet prompts and `>` delimiter. Their confusion rate is
   ~5-6.5% — the only non-trivial number in an otherwise saturated eval.

## What would revive the hypothesis

A v2 experiment (`exp_followup_answer_conditioned_ppl_v2_shared_alphabet`) that:
- Uses domains over a shared token alphabet (e.g., prose domains: news,
  reviews, fiction, QA, code-comments — all English prose, same subword
  tokenizer).
- Increases T_p/T_a ratio to ≥ 3 (long prompt, short answer).
- Makes explicit prediction that Top1_full drops below 0.85 while Top1_ans
  stays above. THEN the K1567-style conjunction is falsifiable at the
  distinguishing regime.

The current experiment does not attempt this — it runs the pre-registered
KC and reports the honest KILLED result.

## Micro-scale limitations

- Single seed (42). With 1000 queries and Bernoulli variance σ/√N ≈ 1.1pp at
  true rate 85%, the K1567 threshold is resolved to ~1pp precision. Both
  measured top-1 values (0.978, 0.984) are separated from 0.85 by > 10 standard
  errors, so the single-seed result is decisive. Multi-seed confirmation is
  unnecessary when the observed gap is so wide.
- Character-level tokenizer (V=42). Result likely transfers to subword
  tokenization only to the extent that domain alphabets remain disjoint under
  subword segmentation.
- d=32, L=2, H=2 model. A larger base might smooth the PPL curves and reduce
  top-1, but we have no prior reason to expect one metric to dominate the other
  under such smoothing.

## Assumptions and open threads

- **Assumption**: fresh RNG at seed+999999+hash(domain) produces held-out
  queries disjoint from training (seed+hash(domain)). For small domains
  (e.g., arithmetic `0..99 + 0..99`) there WILL be overlap by pigeonhole
  (training draws 2000 arithmetic queries from ~10000 possible; held-out
  draws 200 from same space; expected overlap ≈ 40 queries). This does
  NOT affect the conclusion (both metrics still converge on correct routing)
  but is worth acknowledging.
- **Open thread**: routing test on *realistic* domains with shared alphabets.
  Not attempted here; see v2 proposal above.

## Handoff note to Reviewer

- MATH.md pre-registered K1567 with the two-clause conjunction. Verify
  `git log -p micro/models/exp_followup_answer_conditioned_ppl/MATH.md`
  shows a single commit prior to results.
- `results.json.verdict = "KILLED"` and `kill_criteria.K1567.pass = false`
  are consistent.
- `top1_ans = 0.978` is above 0.85, so one half of the KC passed.
- `top1_full = 0.984` is above 0.85, breaking the "full-seq fails" clause —
  this is the kill.
- Antipattern self-check §5 is complete; no antipatterns apply.
- Consider: was the pre-registered prediction for Top1_full reasonable?
  MATH.md §3.2 cited r_full = −0.31 from the predecessor, but the predecessor
  measured a different quantity (relative change) than this experiment
  (absolute ranking). The prediction is reported as made and then refuted;
  the gap is now explained in §3 above.

## References
- `answer_conditioned_scoring` (SUPPORTED, r_ans = 0.81): predecessor, motivated
  this follow-up.
- `ppl_vs_task_performance` (KILLED, r_full = 0.08): original correlation
  finding.
- `exp_followup_routing_multi_sample_ppl` (KILLED by ap-017): related routing
  experiment blocked by adapter-weights cascade.
- MATH.md §3 (this experiment): pre-registered prediction for Top1_full.
