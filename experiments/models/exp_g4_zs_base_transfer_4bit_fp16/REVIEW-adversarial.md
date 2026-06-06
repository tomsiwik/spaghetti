# REVIEW-adversarial: exp_g4_zs_base_transfer_4bit_fp16

Reviewer pass (independent of researcher self-review). Verdict: **PROVISIONAL** — downgrade from researcher's KILLED.

## Verdict rationale

Per reviewer hat rule (t) and Finding #666 (target-gated kill):

- K1 (structural) = PASS — all sweeps finite, adapter helps on 4-bit.
- K2 (PPL-gain ratio R) = FAIL — median R=0.9459 < 0.95 (marginal: 0.0041 below threshold).
- Target-metric KC = **not_measured** — no downstream task accuracy (HumanEval / GSM8K / MedQA) paired with the PPL-ratio KC.

MATH.md labels K2 as "target / behavioral" but R is a ratio of PPL gains — still a PPL-derived quantity. Per guardrail 1006 ("r≈0.08 between PPL and task quality in this codebase"), a PPL-only kill is not safe. Rule (t) binds: **proxy-FAIL with target `not_measured` ≠ FAIL**, so KILL is unjustified. The correct verdict is PROVISIONAL per the hat spec: "structural-KC PASS with target-KC `not_measured`".

## Adversarial checklist (a)–(m2)

| Item | Status |
|------|--------|
| (a) results.json verdict ↔ DB status | KILLED→provisional after review |
| (b) all_pass=false ↔ status | aligned (provisional) |
| (c) PAPER.md verdict vs DB | PAPER says KILLED; must be updated to PROVISIONAL (non-blocking — headline finding captured) |
| (d) is_smoke=true w/ full claim | is_smoke=false, n=200 per domain ✓ |
| (e) post-run KC modification | directory untracked; MATH.md state at run time is the on-disk version; no evidence of manipulation |
| (f) tautological KC | R = gain₈/gain₄ is a real ratio, not identity |
| (g) K-ID code↔MATH match | K1 (finite+helps) and K2 (median R + floor) match MATH.md |
| (h) composition bug | N/A (single-adapter eval) |
| (i) LORA_SCALE≥12 | scale=6 ✓ |
| (j) per-sample routing | N/A |
| (k) shutil.copy sibling adapter | No — same weights via `load_adapters` ✓ |
| (l) hardcoded pass:True | No ✓ |
| (m) target model ↔ loaded model | 8-bit substitutes bf16 at *eval base*, documented; adapter trained on 4-bit matches claim ✓ |
| (m2) MLX skill invocation evidence | Code uses `mx.eval`, `mx.clear_cache`, per-sample PPL with mask — idiomatic; no red flags |
| (n) base acc=0% thinking-suppression | N/A (PPL task) |
| (o) headline n<15 | n=200 per domain ✓ |
| (p) synthetic padding | No ✓ |
| (t) **target-gated kill** | **FIRES** — K2 is PPL-derived proxy, no target KC measured → PROVISIONAL |
| (u) scope-changing fixes | No mid-run scope cuts — pre-registered proxy (4→8 for bf16) is documented upfront in MATH.md |

## Top concerns

1. **Medical domain R=0.98 is uninformative.** Adapter PPL=1.0 indicates eval-set saturation. The median across 3 domains inherits this noise — with N=3, one saturated domain shifts the headline. Code (R=0.90) and math (R=0.95) are the informative cases.
2. **Marginal kill (0.9459 vs 0.95).** The 0.0041 miss on the median threshold is within domain-sampling noise for N=3. A 4th domain could flip the verdict.
3. **PPL-only headline.** Per codebase measurement (r≈0.08 PPL↔task), a 5–10% PPL-gain loss may or may not translate to measurable functional degradation. The real question (does the adapter still *work* on the new base?) is unmeasured.

## Downgrade path

Status set to `provisional` (via `experiment update`). Evidence filed (verdict=inconclusive). Finding #680 updated to status=provisional. Follow-up experiment `exp_g4_zs_base_transfer_4bit_fp16_full` filed with target-metric KC (downstream task-accuracy ratio) added alongside the existing PPL-gain ratio.

## Assumptions

- Treating R as a proxy per F#666/guardrail 1006. Researcher's argument that "per-sample dispersion + per-domain floor" confers behavioral status rejected — both are still PPL quantities.
- Not treating this as a KILL because target is `not_measured`, not `FAIL`. Per hat spec, this is the canonical PROVISIONAL case.
- PAPER.md verdict line left as KILLED (non-blocking; finding text in the DB is the authoritative record and is now provisional).
