# PAPER — `exp_model_knowledge_gap_26b_base`

**Verdict.** PROVISIONAL (BLOCKED on base model not cached; F#478 monotonic prior strongly predicts kill).

---

## Hypothesis (restated)

Can at least one domain adapter trained on Gemma 4 26B-A4B lift MMLU-Pro by ≥5pp over base, reopening the knowledge gap that Finding #478 closed on Gemma 4 4B?

## Prediction vs. measurement

| Prediction (MATH.md §4) | Mechanism | Measured | Status |
|---|---|---|---|
| P1 (dense): K1702 FAIL (`max_d δ_d < 5pp`) | F#478 §3.1 monotonic extension via scaling-law monotonicity (Kaplan 2020, Hoffmann 2022) | NOT MEASURED — base model `mlx-community/gemma-4-26b-a4b-it-4bit` not cached | untested |
| P2 (MoE-niche): K1702 PASS iff ∃d with `M_eff(d) ≤ 4B` via narrow expert routing | §3.2 expert-routing niche per Fedus 2022 / Zhou 2022 | NOT MEASURED — requires prior routing-distribution experiment | untested |
| P3 (target-gated per F#666): K1703 / K1816 aligns with K1702 | paired proxy/target | NOT MEASURED | untested |

All three predictions are untested. No empirical claim filed.

## Kill criteria resolution

| KC | Text | Result |
|---|---|---|
| K1702 (proxy/structural) | ≥1 domain adapter on Gemma 4 26B-A4B achieves ≥5pp MMLU-Pro gain over base | **untested** |
| K1703 (target/behavioral) | Same adapter shows domain-specific behavioral improvement on held-out eval | **untested** |
| K1816 (target/behavioral, paired per F#666) | Same adapter win-rate ≥60% on N=30 held-out prompts (adversarial-judge) | **untested** |

## Measurement blockers

1. **Base model not cached.** `~/.cache/huggingface/hub/` contains Gemma 4 E2B/E4B variants (`models--mlx-community--gemma-4-e2b-it-4bit`, `models--mlx-community--gemma-4-e4b-it-4bit`, `models--mlx-community--gemma-4-e4b-it-8bit`) and `google/gemma-4-e4b-it`, but not `mlx-community/gemma-4-26b-a4b-it-4bit`. Estimated 13–15 GB download.
2. **Training budget.** 500 steps × 3 domains at rank-6 LoRA on 26B-A4B ≈ 2.5 h on M5 Pro 48GB (estimated; unverified on 26B). Exceeds single-iteration researcher budget (30 min / 40 tool calls).
3. **Corpora gap.** Finance + legal `train.jsonl` missing in `exp_p1_t2_single_domain_training` per F#1629; would reduce to 3 domains unless corpora are provided.

## Proof-first prior

MATH.md §3.1 derives a monotonic extension of Finding #478 to capacity: if Gemma 4 4B fails `H(V_d | θ_base) > H_threshold` on basic-tier adapter training data, Gemma 4 26B-A4B fails **strictly more strongly** on the same corpus (scaling-law monotonicity, Kaplan 2020; Chinchilla, Hoffmann 2022).

The only mechanism that reopens the gap is the MoE-niche counter-mechanism (MATH.md §3.2): certain domains may route to a narrow expert subset, giving effective per-domain capacity `M_eff(d) ≤ 4B`. This **requires an orthogonal routing-distribution measurement** before the current K1702 claim is epistemically productive to run.

## Why this experiment is filed PROVISIONAL rather than KILLED

A proof-based kill is defensible under the dense-capacity extension (§3.1), but the MoE wrinkle (§3.2) creates a small, well-motivated uncertainty band that is **not** a proof-first impossibility. Filing as KILLED on §3.1 alone would discard the MoE-niche hypothesis prematurely. Filing as PROVISIONAL preserves the follow-up path:

1. Run a **routing-distribution measurement** on Gemma 4 26B-A4B (new experiment): for each candidate domain d, measure `|⋃ E_d|` (expert union size) on N=100 in-domain tokens.
2. If any d has narrow routing (`|⋃ E_d| ≤ 2` of 16 experts), re-open this experiment with a focused single-domain run (not a 5-domain sweep).
3. If no d routes narrowly, upgrade this experiment to KILLED on §3.1 grounds (F#478 monotonic extension) — no empirical run needed.

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `supported` ✓
4. `is_smoke` = `false` — no smoke-as-full issue ✓
5. No KC was modified between MATH.md and now. K1702 and K1703 match DB-registered text. K1816 was **added** (not modified) as an explicit behavioral pair per F#666; original KCs unchanged ✓
6. Antipattern scan:
   - composition math bug — N/A (no composition)
   - tautological routing — N/A (no routing)
   - LORA_SCALE — would be ≤8 per MATH.md §6
   - KC-swap-after-failure — no data collected; no KC swap ✓
   - shutil.copy as new adapter — N/A ✓
   - hardcoded `pass: True` — no KCs marked PASS ✓
   - proxy-model substitution — scaffold explicitly refuses to proxy to 4B (researcher hat antipattern 'm'). BLOCKED path emits PROVISIONAL, no silent substitute ✓
   - eval-template truncation — N/A ✓

All 6 checks clear **for a PROVISIONAL verdict**. A SUPPORTED verdict is not claimed.

## Assumptions (per researcher autonomy guardrail 1008)

- **A1.** Instruction-tuned 26B-A4B was pretrained on data style aligned with 4B; scaling-law monotonicity holds.
- **A2.** MoE expert routing is token-level; per-domain niche measurement must aggregate routing over a domain's tokens (not sample labels).
- **A3.** LORA_SCALE ≤ 8 (F#328/#330).
- **A4.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool calls; a 2.5 h training run + 14 GB download is explicitly out of scope without authorization.
