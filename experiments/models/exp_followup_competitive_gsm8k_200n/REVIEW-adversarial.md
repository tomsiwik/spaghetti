# REVIEW-adversarial: exp_followup_competitive_gsm8k_200n

**Verdict:** KILL (endorse researcher's preemptive kill).
**Reviewer hat, 2026-04-18.**
**Supersedes:** none (first review, experiment dir untracked).

## 1. Summary

Researcher preemptively killed on antipattern-017 cascade (6th confirmed instance in 2
days). K1575 requires measuring `CI[routed − base]` excludes zero; 0/5 registry-referenced
domain adapters have weight files on disk, so routed composition degenerates to base by
MATH.md Theorem 1+2, and `E[routed − base] = 0` by construction. K1575 = FAIL regardless
of n. DB already completed `--status killed --k 1575:fail` with evidence citing pre-flight
result. Researcher's call stands.

## 2. Adversarial checklist

| Check | Result | Note |
|---|---|---|
| (a) results.json verdict vs DB | ✓ | both KILLED |
| (b) all_pass consistent | ✓ | false |
| (c) PAPER.md verdict | ✓ | KILLED (no PROVISIONAL/etc) |
| (d) is_smoke | ✓ | false |
| (e) KC-swap | ✓ | dir untracked in git; single MATH.md state possible |
| (f) tautology | ✓ | cascade kill on adapter artifact absence, not algebraic identity |
| (g) KC-ID alignment | ✓ | 1575 in DB = MATH.md = code (L70) = PAPER |
| (h) composition code | ✓ | pre-flight only, no forward pass, no `sum(lora_A)` |
| (i) LORA_SCALE | ✓ | absent |
| (j) per-sample routing | ✓ | N/A (no eval) |
| (k) shutil.copy | ✓ | absent |
| (l) hardcoded pass | ✓ | `k1575_result = "fail" if missing else "untested"` — conditional |
| (m) proxy-model | ✓ | no model loaded |
| (m2) skill invocation | ✓ | explicit "no MLX arrays created" comment (L13), N/A for kill |
| (n–p) eval integrity | ✓ | N/A (no run) |
| (q) baseline drift | ✓ | F#560 Gemma 4 40.7% cited in §6 noise analysis |
| (r) prediction table | ✓ | PAPER §3 |
| (s) math errors | ✓ | sqrt(0.24/1400) × 1.96 × √2 = 3.63pp ≈ 3.6pp ✓ |

All pass.

## 3. Independent fact verification

- `adapters/{math,medical,bash,python,sql}/` → config-only + tokenizer (no .safetensors). `adapters/{code,legal,finance}/` → don't exist at all.
- `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/` → only `adapter_config.json` (≤1.3KB each).
- `micro/models/exp_p1_t2_multi_domain_5/adapters/{legal,finance}/` → only `adapter_config.json`.
- `micro/models/real_data_domain_experts/adapters/` → does not exist. Parent dir contains `data/` + experiment artifacts only.
- `experiment get exp_competitive_benchmark_routed` → `status=killed`, K640 FAIL on math -20pp at n=20 and replicated 2026-04-17.
- MATH.md untracked; no KC-swap possible.

## 4. Distinction from prior instances

- vs M0 / J0: those killed on partially-stubbed rosters (2–3 of 4); this is full 5/5 cascade.
- vs `followup_composition_correct_delta` / `followup_routing_multi_sample_ppl`: same 5/5 pattern, same registry, same unblock path. Same author of the 2026-04-17 followup audit batch.

## 5. Action

- DB writes: none — researcher already completed with `--k 1575:fail` and evidence.
- No finding-add this iteration (researcher's evidence sufficient; analyst will decide whether MDE formula becomes a new finding).

## 6. Open threads for analyst

- Bump antipattern-017 instance count from 5 → 6 (baseline_eval + J0 + M0 + followup_composition_correct_delta + followup_routing_multi_sample_ppl + this).
- Consider promoting PAPER §6 MDE formula to a new finding: "at n=1400 MMLU-Pro, MDE ≈ 3.6pp at 95% CI sets the minimum-detectable-effect floor for routed-vs-base claims." Distinct from F#560 (baseline level) and from antipattern-006 (lab-vs-reality noise). Likely worth promoting given its utility for future threshold design.
- SC missing on the DB record (flagged `⚠ INCOMPLETE`) — metadata-only defect; not a kill factor.
- Remaining `audit-2026-04-17` P=1 candidates: expect cascade on any that depend on domain adapters. Next researcher should grep `experiment query --tags audit-2026-04-17` and pre-flight adapter paths before claiming.

## 7. Assumptions

- A1: DB `status=killed` for parent `exp_competitive_benchmark_routed` treated as authoritative cascade root.
- A2: MDE formula's `√2` factor assumes independent error for routed vs base (not paired). This is conservative — paired analysis (same 1400 samples under both conditions) would shrink MDE further. Noted for v2.
