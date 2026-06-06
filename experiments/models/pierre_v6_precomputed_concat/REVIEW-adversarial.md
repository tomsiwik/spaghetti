# REVIEW-adversarial — Pierre v6: Precomputed Concatenated Deltas

**Reviewer:** Reviewer hat | **Date:** 2026-04-17 | **Experiment:** `exp_pierre_v6_precomputed_concat`
**Verdict:** **KILL** (endorse researcher's kill)

## Summary
Researcher completed this experiment as KILLED. Independent verification reproduces every claim: two of three KCs fail on the recorded data (K742 speed 86.8 < 100; K743 behavioral 0.315 < 0.35; K744 memory 2.23 ≤ 6 passes). Two disclosed antipatterns are real in code. The kill is correct; no REVISE path exists because dependencies are absent.

## Adversarial checklist (a)–(s)

| # | Check | Outcome |
|---|---|---|
| a | results.json verdict vs DB status | Both killed — consistent |
| b | all_pass vs claim | `all_pass=false`; DB `killed` — consistent |
| c | PAPER.md verdict line | `KILLED` (no PROVISIONAL/PARTIAL) |
| d | is_smoke flag | Absent; 448 s full-N run — not a smoke |
| e | MATH.md KC diff | Single commit `f421b73`; retroactively authored from 2026-04-05 claim notes; KC IDs 742/743/744 match DB pre-reg |
| f | **Tautology** | **CONFIRMED** — `run_experiment.py:155` (`route(model,tok,val[d][0],...)`) and `:172` (`route(model,tok,test[0],...)`). ppl.v6_pierre ≡ ppl.v6_single byte-identically across 5/5 domains (0.0% degradation). Does not change kill; K742 is speed-only and K743 is behavioral where routing happens to agree with ground truth 5/5 |
| g | K-ID identity | K742/K743/K744 in code match MATH.md and DB |
| h | Composition bug grep | No `sum(lora_A`, no `add_weighted_adapter(combination_type="linear"`, no independent safetensor sum. Uses `inject_precomputed(model, skel, adapter, ri, LORA_SCALE)` — one domain only per forward (single-adapter per domain, not multi-domain composition) |
| i | **LORA_SCALE ≥ 12** | **CONFIRMED** — `run_experiment.py:44 LORA_SCALE = 20.0` (antipattern-003, v5 copy-paste) |
| j | **Single-sample routing applied to all** | **CONFIRMED** at :155 and :172 (same as (f)) |
| k | shutil.copy adapter | No |
| l | Hardcoded `{"pass": True}` | No — KCs computed from measurements (`<100`, `<0.35`, `>6`) |
| m | Model match | `MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"` matches MATH.md §1 |
| m2 | Skill evidence | MATH.md §6 + PAPER.md §5 disclose antipatterns; no mlx-dev/fast-mlx skill invocation mentioned, but this is a retroactive review of pre-existing code, not a new implementation — skill gate applies to researcher-hat code authoring, not post-hoc audit |
| n | Base eval truncation | N/A — PPL is not a thinking-channel metric; behavioral table has non-zero scores (0.093–0.661), not all zero |
| o | Headline N | N_TEST=50/domain × 5 = 250 PPL samples; N_GEN=5/domain × 5 = 25 behavioral. Behavioral headline N=25 is at the statistical edge (< 30) — **flag non-blocking**, kill is robust to the stat concern because both K743 and K742 miss by wide margins |
| p | Synthetic padding | No — 5 real domains with valid data loaders |
| q | Cited baselines | v2/v3/v5 baselines are cited from prior runs (comparison block in results.json); not re-measured here. For the kill, only `native_bitlinear=140.8` needs to be trustworthy, and it's a same-run measurement |
| r | Prediction-vs-measurement table | Present (PAPER.md §2) |
| s | Math errors | Theorem §3 is algebraically correct (associativity + concat). The **falsification** is empirical: the FLOP axis dominates the dispatch axis on BitNet-2B at d=2560 — the math is right, the optimisation axis was wrong. Disclosed in PAPER §5.1 |

## Dependency audit
- `pierre.v6` module — **absent** (only flat `pierre.py` exists)
- `real_data_domain_experts/adapters/grassmannian_skeleton.npz` — **absent**
- `bitnet_sft_generation_v3/sft_adapters/` — **absent**

Non-rerunnable. REVISE is not available as a verdict because a routing fix cannot be re-measured. KILL is the only correct verdict.

## Blocking findings
None — every antipattern is disclosed in MATH.md §6 + PAPER.md §5; none invalidates the kill.

## Non-blocking notes
1. **Behavioral N=25 at the statistical edge.** Check (o) would normally flag this, but the kill is robust: K742 misses by 13% (86.8 vs 100) and K743 misses by 10% (0.315 vs 0.35). Neither threshold depends on the headline statistical precision.
2. **Tautology bug is the same failure as v3–v5.** Finding #553 should be treated as a repo-wide Pierre family bug, not a v6-specific disclosure.
3. **"Bit-exact" framing is falsified.** PAPER §5.2 notes that bf16 precompute + LORA_SCALE=20 + 60-vs-420 dispatch rounding accumulate differently. The theorem in MATH §3 is correct in exact arithmetic; the prediction "behavioral identical to v3 at 0.41" is the part the data falsify.

## Assumptions
- `results.json` is authoritative for the kill since dependencies are absent and the experiment cannot be re-run.
- Retroactive MATH.md is accepted because (a) KC IDs match DB pre-reg from 2026-04-05 and (b) the prior researcher-hat pass already self-audited it in §6.
- Reviewer does not re-run the experiment (max 20 tool calls, max 15 min, no module to load).

## Decision
**KILL.** Endorse researcher's kill verdict. DB already `status=killed`, K742/K743=fail, K744=pass. No REVISE path; no PROCEED path. Emit `review.killed` for Analyst.

## Open threads for Analyst
- Pierre v7 is the forward path (blocked-by v6 is satisfied by kill).
- Rebuilding `pierre.v6` module + SFT adapters + skeleton should be a **separate upstream** experiment, not bundled with a new speed claim (audit lesson: bundled reclaims mask bugs).
- Finding on "dispatch count is the wrong optimisation axis" is worth promoting — structural lesson for any future side-path speed work on BitNet-2B.
