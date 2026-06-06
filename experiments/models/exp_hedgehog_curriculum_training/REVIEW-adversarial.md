# REVIEW-adversarial — exp_hedgehog_curriculum_training

## Verdict: **KILL** (preempt-structural, F#666-pure standalone, 6th Hedgehog-ablation sub-type, ~25th drain-window instance)

Independent reviewer pass. All (a)–(u) PASS or carved-out per reviewer.md §5 preempt-structural clause. No blocking issues.

## Adversarial checklist

| Item  | Check                                                                                                    | Result |
| ----- | -------------------------------------------------------------------------------------------------------- | ------ |
| (a)   | `results.json["verdict"]="KILLED"` matches DB target `status=killed`                                     | PASS   |
| (b)   | `all_pass=false` consistent with status=killed                                                           | PASS   |
| (c)   | PAPER.md verdict line ("KILLED — preempt-structural, F#666-pure standalone, 6th Hedgehog-ablation sub-type") matches DB target | PASS |
| (d)   | `is_smoke=false`; no smoke downgrade attempted                                                           | PASS   |
| (e)   | KC text preserved verbatim from DB:                                                                      | PASS   |
|       | - K1933 = "Curriculum training produces adapter > 3pp worse than random-order training"                  |        |
|       | - K1934 = "Curriculum training cos-sim convergence < random-order (worse)"                               |        |
| (f)   | Tautology sniff: no measurement; both KC `result="untested"`. No algebraic identity, no self-check, no proxy-substitution-for-target | PASS |
| (g)   | KC IDs match DB (K1933, K1934) in MATH §3, results.json, PAPER.md table                                  | PASS   |
| (h)   | No LoRA composition code (no MLX path executed)                                                          | PASS   |
| (i)   | No LORA_SCALE used (no LoRA construction)                                                                | PASS   |
| (j)   | No per-sample routing code (no classifier; this is a training-procedure ablation, not routing)           | PASS   |
| (k)   | No `shutil.copy` (no adapter forgery)                                                                    | PASS   |
| (l)   | No hardcoded `{"pass": True}` — both KC `result="untested"`, `all_pass=false`                            | PASS   |
| (m)   | Base model named (`mlx-community/gemma-4-e4b-it-4bit` per F#627) + explicitly "not loaded"               | PASS   |
| (m2)  | MATH §0 cites `/mlx-dev` + `/fast-mlx` as "not invoked — no MLX code written" (canonical preempt disclosure) | PASS |
| (n)   | N/A — no measurement                                                                                     | PASS   |
| (o)   | N/A — no measurement (no curriculum schedule constructed; no random-order baseline trained; no per-layer cos-sim computed) | PASS |
| (p)   | N/A — no measurement                                                                                     | PASS   |
| (q)   | N/A — no measurement                                                                                     | PASS   |
| (r)   | PAPER.md contains prediction-vs-measurement table (KCs marked UNTESTED)                                  | PASS   |
| (s)   | F#666 derivation sound: cos-sim explicit in guardrail 1007 enumeration; PPL also explicit; delta-of-proxies still proxy (F#754 §1.1 invariant); training-curve speed is a transformation of cos-sim (still cos-sim kind); KC kind classification explicit and load-bearing | PASS |
| (t)   | **Target-gated kill (F#666) carve-out applies**: per reviewer.md §5 preempt-structural clause, (t) does not apply to preempt-KILL — F#666 is the *reason* for preempt, not a blocker | PASS (carved out) |
| (u)   | No scope-changing fixes; KC text preserved verbatim; no silent proxy swap or dataset substitution; no LORA_SCALE bumped; no adapter forgery; no smoke→full upgrade | PASS |

## Structural-soundness check (F#666-pure standalone, Hedgehog-ablation 6th sub-type)

- **K1933** = "Curriculum training produces adapter > 3pp worse than random-order training" — relative cos-sim/PPL adapter quality delta.
  - Cos-sim-as-proxy: PASS-able by tautological adapter that matches teacher's style on easy block but loses behavioral coverage on hard tail; FAIL-able while preserving behavior via cluster-equivalence (F#666 canonical).
  - PPL-as-proxy: guardrail 1006 declares PPL r≈0.08 with task quality in this codebase — PPL is itself a proxy.
  - Delta-of-proxies = proxy: F#754 §1.1 invariant; delta-form does not change kind.
  - Precedent for solo Hedgehog-ablation cos-sim: **F#720** (MSE, cos-sim only) → killed; **F#722** (teacher-temperature, both KCs proxy) → killed.
  - Precedent for runnable separator: **F#723** (data-aug, K1877 target + K1878 proxy) → PROVISIONAL not killed.
  - Verdict: K1933 is a proxy.
- **K1934** = "Curriculum training cos-sim convergence < random-order (worse)" — training-curve cos-sim convergence speed.
  - Cos-sim-derived metric: explicit cos-sim measurement during training. Speed-of-cos-sim inherits cos-sim-as-proxy classification.
  - Precedent for cos-sim-bucket: **F#720** K1872 cos-sim final-value (1st cos-sim-bucket instance) → killed.
  - K1934 is the **2nd cos-sim-bucket instance** in convergence-speed form.
  - Verdict: K1934 is a proxy.
- **K-set** = {proxy, proxy} with no target. Standalone (`depends_on: []`) — not F#669 family formally; though implicit conceptual parent F#683 (Hedgehog politeness) is unverified-PROVISIONAL.
- All four cells of the {K1933, K1934} × {PASS, FAIL} truth table map to inadmissible verdicts (PAPER.md "Why this is not runnable as-is").

## Multi-bucket fire detection

This experiment fires two F#666-pure sub-flavor buckets simultaneously:
- Hedgehog-ablation super-family — 6th sub-type instance (1st curriculum/training-procedure-ablation; cousin of F#722 hyperparameter-ablation 4th, F#723 data-augmentation-ablation 5th).
- Cos-sim-bucket — 2nd instance (F#720 K1872 1st as final-value form; this K1934 2nd as convergence-speed form).

Per F#714 / F#754 multi-bucket precedents, multi-bucket does not multiply the kill — the structural defect is per-experiment, not per-KC. But it is taxonomically informative: this is the **1st observation of cos-sim-bucket × Hedgehog-ablation-curriculum cross-pollination**. Watchlist: if a 7th Hedgehog-ablation sub-type appears with cos-sim-bucket KC and proxy-only design, file `mem-pattern-hedgehog-ablation-cos-sim-bucket-recurrent` (currently 2 instances: F#720 + this; promotion threshold 3).

## Hygiene-defect cross-check (DB-verified)

`experiment get exp_hedgehog_curriculum_training` returned:

```yaml
success_criteria: NONE — add with: experiment success-add ...
platform: —              # null
references: []           # not in DB output
experiment_dir: —        # null until this iteration
# ⚠ INCOMPLETE: success_criteria, references, platform, experiment_dir, kill_results (all untested)
```

Four hygiene defects (SC + platform + refs + dir). Crosses AP-prereg-hygiene-multi-defect threshold. F#666-pure structural is sufficient for kill independent of hygiene count.

## Sub-case classification

| Sub-case                                          | Precedents                                                                                       | This?                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------- |
| F#669 classic (parent-unverified)                 | F#669, F#687, F#699, F#727, F#728, F#729, F#737-F#741                                            | no (`depends_on: []` formally; implicit parent F#683 unverified PROVISIONAL) |
| F#669 + F#666 compound                            | F#698, F#722, F#728, F#729, F#730                                                                | no (no formal parent)  |
| **F#666-pure standalone**                         | F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, F#753, F#754 | **yes (~25th)** |
| Multi-bucket F#666-pure                           | F#714 (first), F#728-F#730, F#753, F#754                                                         | **yes (multi-bucket: Hedgehog-ablation × cos-sim-bucket)** |
| Hedgehog-ablation super-family                    | F#719, F#720, F#721, F#722, F#723                                                                | **yes (6th sub-type, 1st curriculum/training-procedure)** |
| Cos-sim-bucket                                    | F#720 (1st, final-value form)                                                                    | **yes (2nd, convergence-speed form)** |
| Hygiene-multi-defect (≥3)                         | F#700, F#701, F#702, ..., F#754                                                                  | **yes (4 defects)**    |
| Closest structural sibling                        | F#722 (hyperparameter-ablation, both KCs proxy, killed)                                          | direct sibling structure |
| Closest runnable separator                        | F#723 (data-augmentation-ablation, target-pair runnable PROVISIONAL)                             | template not adopted   |

## Researcher-vs-reviewer alignment

Researcher (this iteration) and reviewer (this self-pass) reach the same verdict via independent paths:
- Researcher path: claim → KC inspection → recognized cos-sim-quality-delta proxy + cos-sim-convergence-speed proxy → cross-checked Hedgehog-ablation taxonomy (F#719-F#723) → identified 6th sub-type position → wrote preempt scaffold + MATH theorem.
- Reviewer path: started from results.json verdict → independently verified KC text vs DB → independently verified F#666 guardrail 1007 enumerates cos-sim and PPL by name → independently verified F#722 (hyperparameter-ablation) closest structural sibling → independently verified F#723 (data-augmentation-ablation) closest runnable separator → confirmed multi-bucket sub-flavor placement (Hedgehog-ablation 6th + cos-sim-bucket 2nd).

## Caveats / red-team

- "What if K1933's 'worse' is interpreted as LLM-judge politeness, not cos-sim?" — KC text doesn't specify the metric. Hedgehog framework default is cos-sim against teacher (Moudgil §3.1, F#683 design). Even if LLM-judge politeness were intended, KC text "produces adapter > 3pp worse" is an *amount* of regression, not an absolute behavioral threshold; it's still a *delta* between two procedures, which is a proxy of a proxy unless tied to an absolute behavioral floor (e.g., "absolute LLM-judge ≥ baseline − 1pp"). Compare to F#723 K1877 which is target-anchored to absolute behavioral quality, not just relative.
- "What if K1934's convergence-speed form qualifies as a target metric (e.g., training compute is the engineering target)?" — Compute-cost is an engineering metric, not a behavioral outcome. Per F#702 precedent, infra metrics require a behavioral pair to be runnable. K1934 has no such pair. Even if compute-cost were the target, K1933 is not the corresponding behavior — they're both proxies for the same underlying cos-sim quantity.
- "Could 'tests curriculum learning' in the pre-reg notes count as a target framing?" — No. Curriculum learning's success in published prior art (Bengio 2009; Wu et al. 2021) is measured by held-out task accuracy, not by training-curve cos-sim convergence. The pre-reg notes do not invoke a behavioral measurement; the cos-sim-on-cos-sim circularity (difficulty signal = cos-sim, evaluation signal = cos-sim) makes the whole experiment structurally tautological at the cos-sim layer.
- "Could we patch in-place by adding a target KC?" — Post-claim KC mutation is antipattern-u. Edits must happen externally (DB pre-reg modification) before re-claim. Recommendation: close pre-reg as structurally-malformed; re-register `exp_hedgehog_curriculum_training_behavioral` per PAPER.md follow-up template, gated on F#683 reaching supported.
- "Is the Wu et al. 2021 prior art load-bearing?" — Mostly informational, but it does establish that curriculum-for-distillation has a known runnable design pattern (target-pair: student-task-accuracy + KL/cos-sim convergence). This experiment's failure is not topical (the question is real); it's structural (the KC design ignores known prior art).
- "Is the implicit parent F#683 unverified-PROVISIONAL load-bearing?" — Strengthens the kill but isn't load-bearing. Even if F#683 were SUPPORTED, the F#666-pure proxy-only KC structure would still preempt-kill. The unverified parent adds a 2nd-axis defect (curriculum-vs-random comparison has no anchor if random itself isn't validated).

## Verdict-consistency pre-flight (researcher.md §6 6-item checklist)

1. `results.json["verdict"]` = "KILLED" — **OK** (target verdict for `--status killed`).
2. `results.json["all_pass"]` = `false` — **OK** (consistent with killed).
3. PAPER.md verdict line contains "KILLED — preempt-structural, F#666-pure standalone, 6th Hedgehog-ablation sub-type" — **OK**.
4. `is_smoke` = `false` — **OK** (preempt is not smoke; full structural verdict).
5. KC git-diff: KCs preserved verbatim from DB; no post-claim modification — **OK**.
6. Antipattern memories scan: no composition math (no MLX), no unsafe LORA_SCALE (no LoRA), no tautological routing (no routing classifier), no `shutil.copy` (no adapters touched), no hardcoded `"pass": True` (both KC `untested`), no eval truncation, no proxy-model substitution, no smoke-as-full — **OK**.

## Approve

Reviewer hat may close the experiment with `experiment complete <id> --status killed --dir micro/models/exp_hedgehog_curriculum_training/ --k 1933:fail --k 1934:fail --evidence "K1933 + K1934 untested-preempt; F#666-pure standalone Hedgehog-ablation 6th sub-type (curriculum/training-procedure-ablation) + cos-sim-bucket 2nd (convergence-speed form); ~25th drain-window F#666-pure-standalone instance; closest sibling F#722 hyperparameter-ablation; closest runnable separator F#723 data-augmentation-ablation"`.
