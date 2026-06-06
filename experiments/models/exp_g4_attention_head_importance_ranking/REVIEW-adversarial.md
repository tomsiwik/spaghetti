# REVIEW-adversarial.md — exp_g4_attention_head_importance_ranking

**Reviewer verdict: PROVISIONAL** (correcting researcher's `--status killed` per "provisional-as-killed" antipattern, F#673 precedent).

## Checklist (reviewer.md step 3)

- (a) **Verdict consistency.** results.json `verdict="PROVISIONAL_TAUTOLOGICAL"`; DB was `killed` → mismatch. CLI's `complete --status` lacks `provisional`; `update --status provisional` does accept it. Corrected to `provisional` via §5 workaround.
- (b) `all_pass=false`, status now `provisional`. OK.
- (c) PAPER.md verdict line = "PROVISIONAL_TAUTOLOGICAL". OK.
- (d) `is_smoke=false` — real run, not a smoke.
- (e) MATH.md, PAPER.md, results.json all untracked (first write). No post-run KC relaxation possible.
- (f) No tautology in KCs. K1 and K2 measure distinct quantities (energy concentration vs. set-overlap).
- (g) **Minor hygiene defect, non-blocking.** DB KCs (#1915 variance<10%, #1916 top-20% carry<50%) do not map cleanly to MATH.md's K1 (C_20) and K2 (J̄). #1916 is the negation of MATH.md K1 and correctly PASSed (kill fired on proxy) per measured data; #1915 is not computed by the code. MATH.md is the authoritative pre-reg and is internally consistent with run_experiment.py. Recommend a post-hoc DB `kill-add` to register K2/J̄ to preserve audit trail, but not a blocker on the verdict.
- (h/i/j/k/l) No composition math, no hard-coded LORA_SCALE, no shutil.copy misuse, no hardcoded `{"pass": True}`, no single-sample routing. Pure weight-space analysis using safetensors.
- (m) Target model in MATH.md (`mlx-community/gemma-4-e4b-it-4bit`) matches code.
- (m2) Weight-space only (numpy), no MLX training code. LEARNINGS.md notes `/mlx-dev` invoked. Acceptable.
- (n/o/p/q) No behavioral eval in this experiment, so (n, p, q) don't bite. (o) n=336 heads × 3 domains — well above 15.
- (r) PAPER.md contains prediction-vs-measurement table (§Predictions vs. measurements) covering P1/P2/P3 and K1/K2. OK.
- (t) **Target-gated rule (F#666).** K1 (proxy: concentration) FAIL paired with K2 (target: functional Jaccard) PASS. Reviewer.md §3(t) verbatim: "A proxy-FAIL with target-PASS is a finding about the proxy, not a kill." MATH.md §4 decision table pre-registered this outcome as PROVISIONAL (tautological proxy). **KILL is unjustified; PROVISIONAL is correct.**
- (u) No scope-changing fix. Researcher did not silently swap mechanism or truncate scope.

## Why PROVISIONAL, not KILL

The researcher event note said "CLI forced killed" — that is exactly the antipattern reviewer.md §5 names as "provisional-as-killed" (F#673, `exp_rdt_loop_lora_gemma4` precedent). The `experiment update --status provisional` path does work; the `complete` CLI is the only one that rejects `provisional`. Using the two-step workaround preserves the pre-registered decision rule.

## Why PROVISIONAL, not REVISE

Pre-reg is internally consistent; code implements MATH.md faithfully; evidence is recorded honestly; follow-ups closing the behavioral gap are already named. No blocking issues on MATH/PAPER/results. The DB KC/MATH.md KC mismatch (g) is a hygiene issue that does not invalidate the verdict.

## Follow-ups (required for PROVISIONAL workflow)

- `exp_g4_head_ablation_ppl` (P3) — direct PPL ablation of top-20% vs bottom-20% heads. Closes F#666 proxy→target gap with a behavioral metric. Files as new experiment (not `_impl`) because it changes methodology (forward passes vs. weight-space).
- `exp_g4_head_importance_vproj_oproj_F627` (P3) — repeat on F#627-compliant targets. Independent generalizability check.

## Non-blocking observations (noted for next researcher pass)

1. DB KC registration lags MATH.md — future runs should `kill-add` MATH.md K1/K2 verbatim before claim so (g) is clean.
2. MATH.md §5 P3 (rank-6 → zero heads) is logically invalid as noted in the researcher self-review; LEARNINGS.md correctly captures this.
3. numpy matmul warnings on this host are cosmetic — PAPER.md already disclaims.

## Assumptions

- Treating the researcher's `--status killed` as an honest CLI-limit error rather than a silent downgrade. No REVISE warranted because the intent per MATH.md §4 was explicitly PROVISIONAL.
- The follow-ups named in PAPER.md are sufficient; no additional caveat KC required from this review.
