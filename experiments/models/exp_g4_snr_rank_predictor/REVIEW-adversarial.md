# REVIEW-adversarial: exp_g4_snr_rank_predictor

**Verdict: KILL** (precondition probe FAILED honestly; DB already
status=killed; this is the 9th consecutive `audit-2026-04-17` cohort
probe-KILL on the same upstream blocker.)

## One-line reason
P1 (0/5 r=6 adapters + 0/25 rank-sweep), P2 (0/5 grad-SNR spectra), and
P3 (upstream T2.1 KILLED, base=0% format artifact) all FAIL on the
pre-registered probe — KC #1586 / #1587 are UNMEASURABLE, matching
MATH.md §"Kill criteria (canonical)" K1586 UNMEASURABLE → KILLED.

## Adversarial checklist (a)–(s)

| # | Check | Result |
|---|---|---|
| (a) | `results.json["verdict"]=="KILLED"` vs DB status `killed` | PASS — consistent |
| (b) | `all_pass=false` vs claim KILLED | PASS — consistent |
| (c) | PAPER.md verdict line: `Verdict: KILLED (UNMEASURABLE …)` | PASS — no PROVISIONAL/INCONCLUSIVE/etc. |
| (d) | `is_smoke=false` vs claim full probe | PASS — probe is the full deliverable |
| (e) | KC #1586/#1587 modified between runs? MATH.md untracked, single git snapshot | PASS — no relaxation |
| (f) | KC tautology (algebraic identity / unused arg / same expr twice)? | PASS — KC compares two physical objects (files-on-disk vs required count); not algebraic identity |
| (g) | K-ID in code (`kill_criterion_ids:[1586,1587]`) matches MATH.md / DB | PASS |
| (h) | `sum(lora_A` / `add_weighted_adapter("linear")` / per-key safetensor sum in runner | PASS — runner does no composition; file-existence probe only |
| (i) | LORA_SCALE ≥ 12 hardcoded | N/A — no training code |
| (j) | Single-sample routing applied to all | N/A — no routing |
| (k) | `shutil.copy(...)` of sibling adapter relabeled as new domain | N/A — no copying |
| (l) | Hardcoded `{"pass": True, …}` in KC dict | PASS — only observed `passed:false` written from real file checks |
| (m) | Target model in MATH.md ≠ model loaded in runner (proxy substitution) | N/A — runner loads no model |
| (m2) | MLX skill invocation evidence | N/A — runner is pure stdlib `pathlib`/`json` file probing; no MLX call surface to mis-use |
| (n) | Base acc=0 with `avg_thinking_chars=0` driving headline | PASS — base=0 noted as upstream P3 artifact, not used as a measurement |
| (o) | Headline n < 15 | N/A — probe outcome, not a sample-size measurement |
| (p) | Synthetic padding (B=0 / random adapters) inflating N | N/A — no adapters loaded |
| (q) | Cited baseline drift | PASS — P3 reads upstream `results.json` directly and reports its KILLED verdict |
| (r) | PAPER.md prediction-vs-measurement table | PASS — present, KC #1586/#1587 marked UNMEASURABLE |
| (s) | Math errors / unsupported claims | PASS — no math claim is asserted; the only claim is "files do not exist", which the probe verifies |

**Score: 17/17 PASS or N/A. No blocking issue.**

## Cross-checks performed
- `ls micro/models/exp_p1_t2_single_domain_training/adapters/math/` →
  only `adapter_config.json` (1253 bytes), no `adapter_model.safetensors`.
  Confirms P1 r=6 endpoint is genuinely missing for the one domain dir
  that even exists.
- `ls .../adapters/finance` → "No such file or directory". Confirms
  finance + legal domains were never trained — not a search-path
  artifact.
- `git status micro/models/exp_g4_snr_rank_predictor/` → entire
  directory is untracked (single snapshot — no opportunity to silently
  rewrite KCs after the run).
- `experiment get exp_g4_snr_rank_predictor` → DB already
  `status=killed`, evidence row attached, KC #1586 / #1587 both `[✗]`.

## Cohort context
9th consecutive precondition-probe KILL in `audit-2026-04-17` cohort.
Upstream blocker (Finding #611, #615, scratchpad iter 2) is identical
across every KILL: `exp_p1_t2_single_domain_training` must rerun at
LORA_SCALE=5, max_tokens ≥ 512, with disjoint math/code/medical/finance/
legal corpora and rank sweep {2,4,6,12,24}. The cohort is saturated;
the prior analyst iteration already promoted this rerun to a
first-class work item.

## Assumptions logged
- The MATH.md K1586/K1587 KCs are accepted as-pre-registered. No
  relaxation suggested; relaxing within-2x → within-4x or 0.90 → 0.80
  would invalidate the probe per MATH.md §Assumptions.
- Adding an antipattern-memory instance for ap-017 is **declined** for
  the same reason the prior iteration's analyst declined: the class is
  already auto-injected; adding a 12th instance inflates context
  without new signal.
- DB is already `status=killed`. No second `experiment complete` call
  is issued (idempotency: a second `complete` call would error or
  duplicate evidence).

## Routing
Emit `review.killed` with the existing finding registered.
