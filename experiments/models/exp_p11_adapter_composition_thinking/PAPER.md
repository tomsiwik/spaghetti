# PAPER.md — P11.J0: Adapter Composition via Exclusive Routing

## Verdict: KILLED (2026-04-18)

Three independent kill drivers, one from direct measurement, two from
infrastructure and theorem-premise failure.

## Theorem Predictions vs Measurements

| KC | Prediction | Measurement | Status |
|----|------------|-------------|--------|
| K1526: routed ≥ domain_only + 3pp | LIKELY | UNTESTABLE (domain adapter weights missing) | FAIL |
| K1527: routed ≥ thinking_only + 2pp on knowledge | LIKELY | UNTESTABLE (domain adapter weights missing) | FAIL |
| K1528: router accuracy ≥ 85% | LIKELY (Medium-High) | 0.668 (187/280) | FAIL (−18.2pp) |

## Measurements

### Phase 2: Router accuracy (K1528)

- 14-category → binary (reasoning/knowledge) classification
- 20 held-out examples per category, 280 total
- Router: mean(embed_tokens(query))[:64] → cosine sim to centroids
- **Result**: 187/280 = **0.668** (threshold ≥0.85) → K1528 FAIL

### Phase 3: thinking_only condition (partial success)

- Adapter: `adapters/thinking-openthoughts-universal-v0/adapters.safetensors` (1000 steps, produced by H0 which was killed)
- 80 questions (20 each × math/physics/biology/law)

| Split | Accuracy |
|-------|----------|
| overall (4 cats) | 0.550 |
| reasoning (math+physics) | 0.375 |
| knowledge (biology+law) | **0.725** |

- Time: 2286s

### Phase 4/5: Never ran

Crashed at `mlx.load("…/adapters/math/adapters.safetensors")`.
Audit of all 4 referenced domain-adapter directories:

```
adapters/math/         → only adapter_config.json (no weights)
adapters/medical/      → only adapter_config.json
adapters/legal/        → only adapter_config.json
adapters/finance/      → only adapter_config.json
```

All four are config-only stubs — the training runs never produced
safetensors files (or they were deleted). No way to measure domain_only
or embedding_routed conditions without retraining.

## Why the kill is robust

### 1. K1528 FAIL is a direct measurement, not an inference

Router accuracy 66.8% is 18.2pp below the 85% threshold pre-registered
in MATH.md. The pre-registration is clean (MATH.md single commit
`de38e37`, no post-data edits). No amount of phase 4/5 data would
rescue this KC.

### 2. MATH.md Theorem 1 premise is directly contradicted

Phase 3 measured: `acc_t(reasoning) = 0.375 < acc_t(knowledge) = 0.725`.
MATH.md Theorem 1 L24 assumes `acc_t(P_r) ≥ acc_t(P_k)`. The measured
inequality inverts by 35pp — this isn't a noisy premise, it's a
structural inversion. Theorem 1's prediction (routing wins) no longer
follows from the proof, even if we had working domain adapters.

The inversion has a plausible cause: the thinking adapter is the
regressed H0 artifact. H0 was killed today (2026-04-17) with
47.6% MMLU-Pro and catastrophic humanities collapse (engineering 13.3%,
philosophy 20.0%). The adapter that was meant to be "universal
reasoning-helper" instead suppresses reasoning categories while
preserving knowledge categories. Phase 3's 37.5% reasoning / 72.5%
knowledge is the downstream symptom.

### 3. Infrastructure: all 4 domain adapters are weight-less stubs

Cannot run phases 4 or 5 to produce domain_only or embedding_routed
numbers. Unblocking this experiment would require training 4 domain
adapters first, which is a separate multi-hour workstream — and per
Finding #517 they degrade MCQ by 26pp, so the likely outcome is K1526
PASS trivially (routed > broken domain_only) with zero composition
insight and K1527 FAIL (broken domain adapter can't help knowledge).

## Antipattern self-check

- ✅ Not antipattern-018 (channel-text-as-SFT) — J0 does no training.
- ✅ No shutil.copy, no hardcoded `"pass": True`.
- ✅ No LORA_SCALE=20 (no scale param at all, inference-only).
- ✅ MATH.md unchanged since de38e37 (no post-data KC relaxation).
- ✅ Same base model (Gemma 4 E4B 4-bit) in calibration + eval.
- ⚠ Base accuracy 62.1% (MATH.md L44) is stale Finding #530 — measured
   40.7% in-project (F#560 open). Not load-bearing for kill since KCs
   are relative.

## Assumptions

- "Infrastructure bug counts as kill driver": retraining 4 domain
  adapters is out-of-scope for this experiment claim; J0 as designed
  cannot produce K1526/K1527 measurements.
- Phase 3 partial result (thinking_only 0.550) is retained as the one
  meaningful datapoint — it quantifies the H0 regression's downstream
  effect on composition experiments.

## Unblock path (future J0-v2)

1. **Rebuild the thinking adapter** from a non-regressed SFT run
   (requires P11.HARNESS fix: strip channel tokens, reformat as
   `<think>...</think>`, match H0 K1519 PASS format).
2. **Train 4 domain adapters to completion** (generate
   `adapters.safetensors`, not just config). Consider dropping NTP
   q_proj-only approach since Finding #517 shows it degrades MCQ.
3. **Improve router**: 66.8% centroid accuracy is weak. Options:
   - Use later-layer hidden states instead of raw `embed_tokens`
   - Learn a small classifier head instead of cosine centroids
   - Expand seed set beyond 10 examples per group
4. **Reconcile F#560 baseline** (62.1% vs 40.7%) before re-designing
   K1526/K1527 with absolute thresholds.

## References

- arXiv:2407.06582 (LoRAMOE)
- arXiv:2312.00752 (MoLoRA)
- arXiv:1904.10480 (JL-lemma)
- Finding #517 — domain adapters degrade MCQ (q_proj NTP)
- Finding #527 — pre-merge killed
- Finding #530 — 62.1% baseline (stale)
- Finding #560 — 40.7% measured baseline (open reconciliation)
- H0 (exp_p11_thinking_adapter_universal) — killed today, produced
  the regressed adapter reused here
- antipattern-018 — channel-text-as-SFT (not applicable to J0)

## Handoff

- DB: complete as `--status killed --k 1526:fail --k 1527:fail --k 1528:fail`.
- Evidence: router 66.8% vs ≥85%; Thm 1 premise inverted 37.5% < 72.5%;
  4-of-4 domain adapters are weight-less stubs.
- No new Finding needed; mechanism is composition of already-documented
  H0 regression + preexisting Finding #517 + adapter-infra issue.
- J0-v2 is blocked on P11.HARNESS and a domain-adapter retrain batch.
