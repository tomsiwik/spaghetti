# exp_g4_adapter_class_composition_full — PAPER (PROVISIONAL, design-only)

## Verdict

**PROVISIONAL** — design locked; full empirical execution deferred to `exp_g4_adapter_class_composition_full_impl` at P3 per reviewer.md §5 ("PROVISIONAL novel-mechanism design-only sub-case") and `mem-antipattern-novel-mechanism-single-iteration-scope` option (i). No empirical claim is filed by this artifact.

## Why PROVISIONAL

The DB title requires a 3pp MMLU-Pro margin at N=5 composition across 3 adapter classes (LoRA, DoRA, MoLoRA) on Gemma 4 E4B. Realistic wall-clock for the full pipeline — 15 adapter trainings (3 classes × 5 domains) × ~30-60 min/training + MMLU-Pro n=1000 composition-eval × 3 classes + K4 rank-ablation at r=8 — is 8-15h. This exceeds the 90-minute single-iteration researcher budget (guardrail 1009) by roughly an order of magnitude. Additionally, MoLoRA has no turn-key `mlx_lm.lora` mode; a custom module `micro/utils/molora.py` must be written (novel-mechanism sub-component).

## What this design delivers

| Artifact | Status |
|---|---|
| MATH.md with §0 skill citations, §2 theorem, §3 target-gated KCs (K1-K4), §7 scope fence | ✅ |
| `run_experiment.py` graceful-failure scaffold (writes valid `results.json` with `verdict=PROVISIONAL`, enumerates 5 blockers, all KCs "untested") | ✅ |
| Scope-preservation forbid list (MATH.md §0 F1-F5) binding on `_impl` iteration | ✅ |
| `_impl` follow-up filed at P3 with MATH.md inherited | pending (end of this iteration) |
| Prediction-vs-measurement table (§5 below) | table present, all "not measured" |

## Prediction-vs-measurement table

| KC | Prediction | Measured | Status |
|---|---|---|---|
| K1 structural | ≥13/15 class-domain trainings converge | not measured | untested |
| K2 target | `acc_A − max(acc_{B.j}) ≥ 0.03`, 95% CI LB > 0 | not measured | untested |
| K3 proxy | `median(dev_D on trained DoRA) > 10⁻³` | not measured | untested |
| K4 ablation | sign of K2 stable at r=8 | not measured | untested |

## Parent relationship

**Parent: `exp_g4_adapter_class_composition` (F#679, PROVISIONAL).** The parent measured the composition-geometry proxy at Gemma 4 E4B scale (dev_LoRA=0, dev_DoRA=0.089, dev_MoLoRA=0.667). Its K2 was proxy-labeled-as-target per `mem-antipattern-proxy-kc-mislabeled-target`. This child is the designated behavioral target-measurement follow-up. `mem-antipattern-preempt-child-parent-target-unverified` does NOT apply here because:
- Parent's proxy K2 PASSED structurally (not UNMEASURABLE).
- This child is explicitly the target-measurement follow-up the parent's PAPER.md §"What this does NOT claim" deferred to.
- Child's KC K3 directly tests the parent's caveat (`m_d = ||W_0||_c` init assumption vs trained m_d drift).

## Assumptions (from MATH.md §4)

1. `mlx-lm >= 0.22` supports DoRA via `--fine-tune-type dora` (verify at `_impl` time).
2. 5 domain corpora available (HumanEval, GSM8K, PubMedQA, CaseHOLD, LegalBench non-overlapping subset).
3. MMLU-Pro n=1000 provides sufficient statistical power for 3pp margin at 95% paired CI.
4. Training on Gemma 4 E4B 4-bit with 1000 steps, batch=4, max_len=2048, lr=1e-4 converges across 5 domains (F#627 target choice).

## Antipattern compliance check

- `mem-antipattern-novel-mechanism-single-iteration-scope`: applied option (i) — PROVISIONAL with `_impl` at P3, MATH.md inherited. ✅
- `mem-antipattern-proxy-kc-mislabeled-target`: K2 measures MMLU-Pro accuracy (behavioral), not composition-deviation (geometric). ✅
- `mem-antipattern-preempt-child-parent-target-unverified`: N/A (parent proxy PASSED; this is the designated target follow-up). ✅
- `mem-antipattern-schema-incomplete`: all KCs reference trained artifacts (K1 = training convergence, K2 = behavioral eval on trained composed adapters, K3 = trained-DoRA deviation, K4 = trained-at-r=8). ✅
- `mem-antipattern-claim-time-tag-saturation`: researcher received P3 mispick (`exp_followup_cayley_riemannian_adam`) from claim picker; released and manually routed to this P2 per analyst's `learning.complete` payload handoff. ✅ (flagged in LEARNINGS.md for analyst).
- `mem-antipattern-finding-add-scratchpad-drift`: finding will be verified via `finding-list` before citation. ✅
- Scope-preservation forbid list (F1-F5): binding on `_impl`. ✅

## What this experiment does NOT claim

- No empirical claim about LoRA vs DoRA/MoLoRA behavioral margin at Gemma 4 scale.
- No training-time metrics, no MMLU-Pro numbers.
- No refutation or confirmation of F#82 micro-d-to-macro transfer.
- The `_impl` follow-up will close these gaps.

## References

- F#82 (conclusive): composition taxonomy micro-d.
- F#627: Gemma 4 E4B LoRA target choice (v_proj + o_proj).
- F#666: target-gated KC requirement.
- F#673: `mx.clear_cache()` between phase trainings.
- F#679 (provisional): parent, Gemma 4 composition-geometry proxy.
- arxiv:2402.09353 — DoRA.
- arxiv:2402.11260 — MoLoRA.
- Reviewer.md §5 "PROVISIONAL (novel-mechanism design-only sub-case)" — routing clause.
