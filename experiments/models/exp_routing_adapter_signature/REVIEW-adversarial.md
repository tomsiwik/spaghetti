# REVIEW-adversarial.md — exp_routing_adapter_signature

**Verdict:** KILL (preempt-structural, F#666-pure standalone 24th drain-window instance).

## Adversarial checklist (a)–(u)

- (a) results.json `verdict="killed"` ✓ matches DB killed state (finding F#736 filed).
- (b) `all_pass=false` ✓ consistent with killed verdict.
- (c) PAPER.md verdict "KILLED (preempt-structural)" ✓ consistent.
- (d) `is_smoke=false` ✓ not a smoke downgrade case.
- (e) No KC mutation post-claim — K1902/K1903 were structurally proxy-only at registration; no retroactive relaxation.
- (f) Not a tautology-pass — preempt-structural verdict (no KC measured); both KCs are proxy-only by guardrail-1007 lineage. Distinct from §5 tautological-inter-adapter-delta (this is inter-family, not intra-variant). Correctly classified.
- (g) K-IDs consistent across MATH.md / results.json / PAPER.md.
- (h)–(l) Code antipatterns: stub is `json`+`pathlib`-only graceful-failure; no `sum(lora_A`, no `LORA_SCALE`, no `shutil.copy`, no hardcoded `{"pass": True}`, no per-sample routing bug. Canonical preempt-structural artifact shape per F#687/F#698/F#699/F#700/F#701/F#703/F#731–F#735 precedent chain.
- (m)/(m2) No model loaded; F#666-pure clause exempts from skill-invocation evidence (no platform code lands). (u) carve-out also applies — graceful-failure is canonical, not a scope swap.
- (n)–(q) Eval-integrity items N/A — no measurement performed.
- (t) Target-gated kill: **carve-out applies** per F#666-pure standalone clause — F#666 is the *reason* for preempt-KILL, not a blocker. No KC measured (neither proxy nor target).
- (r) PAPER.md prediction-vs-measurement table present (all rows "no run / fail structural").
- (s) Math: Theorem 1 (F#666-pure structural impossibility) correctly derived from guardrail 1007 truth-table; Theorem 2 (F#715 6th + 5th wall-clock) consistent with promoted bucket memory; Theorem 3 (F#702-unavailability derived-lemma) vacuous-patching argument sound; Theorem 4 (prior-art redundancy F#137/F#269/F#427/F#453/F#498) correctly anchors unblock path to Branch 2 (cheapest). 5-branch enumeration per F#734 watchlist-correction meta-pattern complete.

## Triple-fire classification (confirmed)

1. **F#666-pure 24th drain-window** — K1902 (routing-acc) + K1903 (wall-clock) both proxy-only, zero target behavioral KC. Anchor-append (post-promotion canonical, F#721 threshold).
2. **F#715 infrastructure-benchmark bucket 6th drain-window** — K1903 wall-clock sub-flavor 5th instance (F#715+F#732+F#734+F#735+THIS). Post-promotion anchor-append (promoted at F#734 QUADRUPLE-FIRE).
3. **F#706/F#707/F#710-lineage routing-accuracy-as-proxy 3rd explicit drain-window instance** — analyst signal for potential standalone split at 4th occurrence per F#643 convention.

## Non-promoting fires

- F#702 hygiene-patch unavailable (derived-lemma N-th reuse; vacuous under Theorem 1 saturation; partial patch — dir + platform set — reviewer-accepted per Theorem 3).
- Prior-art redundancy (F#137/F#269/F#427/F#453/F#498).

## Distinguishing signal (correctly excluded)

- **NOT** §5 tautological-inter-variant-delta — inter-family comparison (signature hash vs TF-IDF-on-training-data), not intra-family/variant delta.
- **NOT** method-dep-redundancy — compares routing-key functions on shared adapter set, not method-A-combination-X vs method-B-combination-X.
- **NOT** tool-as-experiment — title frames routing METHOD (hash function), not infrastructure artifact. Watchlist candidate `exp_adapter_fingerprint_uniqueness` remains distinct at 1 instance (promotion at 2nd).

## Routing

→ `review.killed`. No `_impl` companion (preempt-structural exclusion per F#687/F#698/F#699/F#700/F#701/F#703/F#731/F#732/F#733/F#734/F#735 → F#736).

## Assumptions

- Watchlist-correction meta-pattern (F#734) stable at 4th consecutive iteration — reviewer accepts the claim-time branch-enumeration interpretation.
- F#715 bucket wall-clock sub-flavor accumulation (5 of 7 instances now wall-clock) noted; if 6th wall-clock fires, analyst should consider standalone split. Not reviewer-gating.
- Routing-acc-as-proxy 3rd explicit sub-flavor: reviewer concurs with researcher's signal to analyst; standalone split deferred to 4th instance per convention.

## Promotion signals forwarded to analyst

1. F#715 bucket memory — anchor-append F#736 as 6th drain-window instance; 5th wall-clock sub-flavor of 7 total.
2. Routing-acc-as-proxy lineage (F#706/F#707/F#710/F#736) — 3rd explicit instance; 4th triggers standalone split.
3. F#666-pure 24th: canonical anchor-append, no re-promotion.
4. F#702-unavailability: inline-track, non-promoting.
5. Watchlist-correction meta-pattern: 4th consecutive iteration stable.
