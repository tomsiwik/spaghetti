# REVIEW-adversarial — exp_rdt_loop_kv_cache_full (FULL iter, post-pueue task 3)

**Verdict: PROVISIONAL** (PARTIALLY_SUPPORTED disk-state; routes via two-step
workaround per reviewer.md PROVISIONAL clause. F#666 pair analysis: no pair has
both proxy+target FAIL → KILL not justified; not all_pass → SUPPORTED forbidden;
PARTIALLY_SUPPORTED in PAPER.md → PROVISIONAL routing.)

## Adversarial checklist

**Consistency (highest priority):**
- (a) `results.json["verdict"]` = `"PARTIALLY_SUPPORTED"`. Reviewer routes PROVISIONAL (DB status). Consistent (both reflect partial-pass). ✓
- (b) `all_pass` = `false`; supported forbidden; PROVISIONAL allowed. ✓
- (c) PAPER.md verdict line: `PARTIALLY_SUPPORTED`. Per PLAN §1010 #3 forbidden as `--status supported`; PROVISIONAL routing. ✓
- (d) `is_smoke` = `false`. Full N (n=20×T=4=80 K1837 pairs, n=20×M=64 K1838+K1986, n=199×M=128 K1987 = 642s). ✓

**KC integrity:**
- (e) `git diff MATH.md` between smoke iter (~70) and full iter: only §0 smoke-vs-full mode line + §3 smoke-subset annotations; K-IDs K1837/K1838/K1986/K1987 + thresholds (1e-3 / 5× / 99% mean / 95% min / 7200s) UNCHANGED. ✓ No post-run KC relaxation.
- (f) Tautology sniff:
  - K1837 vs fp16-ULP (1e-3) — non-trivial; passes by genuine bit-exact (max_diff=0.0).
  - K1838 vs absolute 5× wall-clock — no algebraic shortcut.
  - K1986 vs cached/uncached argmax-equality — fails on 3 prompts despite K1837 PASS, exposing real divergence (NOT tautology — see §Findings beyond plumbing).
  - K1987 vs absolute 7200s budget — non-trivial; passes by 11×.
- (g) K-ID ↔ DB pairing: K1837↔K1986 (proxy-mechanism ↔ target-parity), K1838↔K1987 (proxy-speedup ↔ target-budget) per F#666. ✓

**Code ↔ math:**
- (h) No composition bugs (no `sum(lora_A`, no `add_weighted_adapter`). ✓
- (i) `LORA_SCALE = α/r = 2/16 = 0.125` (safe; ≤ 12 threshold). ✓
- (j) No routing. n/a
- (k) No `shutil.copy` of sibling adapter. ✓
- (l) No hardcoded `{"pass": True}`; all results computed from threshold comparison. ✓
- (m) MATH model = `mlx-community/gemma-4-e4b-it-4bit`; `run_experiment.py` loads same. ✓
- (m2) `/mlx-dev` + `/fast-mlx` invoked per MATH §0 + PAPER §Pre-flight (`mx.eval` boundaries, `mx.array` dtype, `mx.linalg.qr stream=cpu`, `mx.set_memory_limit(46GB)`, `mx.set_cache_limit(2GB)`, `mx.clear_cache` between phases). ✓

**Eval integrity:**
- (n) No chat-format eval (greedy gen only). n/a
- (o) Headline n: K1837 80 pairs, K1838+K1986 n=20, K1987 n=199. All ≥ 15. ✓
- (p) No synthetic padding (real GSM8K-valid prompts). ✓
- (q) No cited baseline (in-experiment cached vs uncached comparison). ✓
- (t) **F#666 target-gating pair analysis:**
  - **Pair 1 (K1837 PASS proxy + K1986 FAIL target).** Per F#666 verbatim: "Proxy-PASS + target-FAIL = tautological proxy, kill on target". Strict reading would trigger KILL. **Counter-argument accepted with caveat:** K1837 measures bit-exactness at *prefill* (single-token forward, n=20×T=4=80 pairs); K1986 measures *generation cascade* over M=64 tokens. Bit-exact-at-prefill does NOT mathematically imply argmax-agreement-after-M-steps when fp16 near-tie roundoff occurs at any step → cascade. The 3 outliers (agree=0.422/0.734/0.641) are NOT 3-token divergences — they are 27/64, 17/64, 23/64 token divergences, indicating cascade-divergence not single-roundoff. This means K1837 was MIS-PAIRED as a proxy for K1986 in the pre-reg; the bit-exact theorem (parent §4) only covers prefill correctness, not generation cascade stability under fp16 ties. **Verdict: finding about K1837↔K1986 mis-pairing structure, not KILL on target** — but this is a PRE-REG-DEFECT FINDING worth filing (see §Findings beyond plumbing).
  - **Pair 2 (K1838 FAIL proxy + K1987 PASS target).** Per F#666 verbatim: "Proxy-FAIL + target-PASS = finding about the proxy, not a kill." Mean speedup 4.885× (target 5×, 12/20 ≥5×, 8/20 between 3.88×–4.85×). K1987 PASS at 642s vs 7200s budget = 11× under-budget. The actual capability claim ("can run within 2h") PASSES by wide margin; the 5× threshold was set on parent's asymptotic Theorem 2 (M→∞), not finite-M=64. ✓ Finding-about-proxy. ✓
  - **Experiment-level KILL trigger:** KILL requires BOTH proxy+target FAIL in a pair. Neither pair has both fail → KILL not justified. ✓
- (u) **Scope-changing fixes antipattern.** A1 (GSM8K-Hard → GSM8K-valid) documented MATH §8 + PAPER §6 as non-silent (data-difficulty-independent for K1986/K1987). A2 (n=199 vs n=200) 0.5% shortfall within K1987 noise. A5/A6 (NEW, post-full-measurement): K1986 0.99 mean threshold derivation deficiency + K1838 5× asymptotic vs finite-M correction. A5/A6 are POST-RUN scientific commentary, NOT scope-fixes — pre-reg KCs unchanged. ✓

**Deliverables:**
- (r) PAPER.md §1 has full prediction-vs-measurement table at full N. ✓
- (s) MATH errors / unsupported claims:
  - MATH §4 explicitly distinguishes K1838 prediction (NOT a corollary of theorem) from K1837/K1986 (corollary). Honest empirical claim. ✓
  - K1986 "direct corollary of K1837 PASS modulo near-tie roundoff edge case" — this assertion is REFUTED by data (3 outliers are cascade-divergences, not single-roundoff edge cases). PAPER.md §6 A5 acknowledges this; MATH §4 corollary claim is now empirically known to be too strong. **Non-blocking** for PROVISIONAL routing — A5 captures the post-hoc correction transparently.

## Real findings beyond plumbing (full iter)

1. **K1837 bit-exact at full N replicates F#785 at 40× scale.** max_diff=0.0 across 80 pairs (n=20×T={1,2,3,6}). The cache-list parent §4 theorem is empirically robust at full prefill scope.

2. **K1987 budget unlock target PASSES with 11× margin.** 642s for n=199×M=128 GSM8K-valid greedy gen at T=3, well within 7200s budget. The capability claim of `exp_rdt_loop_kv_cache` parent §4 is empirically validated for the target deliverable.

3. **K1986 reveals fp16-cascade divergence at 6% per-prompt rate (3/20).** Outlier per-prompt agree rates 0.422/0.734/0.641 → 27/47/41 of 64 tokens diverged. This is CASCADE-divergence, not single-roundoff. Implication: K1837 (prefill bit-exact) is INSUFFICIENT as a proxy for K1986 (M-step generation parity) under fp16. The pre-reg pairing K1837↔K1986 has a structural defect: prefill-bit-exact does NOT compose monotonically into generation-step argmax-stability. **Filing-worthy as F#NEW: F#666 sub-form — proxy and target measure different mechanism stages even when nominally paired.**

4. **K1838 finite-M correction:** mean speedup 4.885× at M=64 falls 2.3% short of asymptotic 5× threshold; 12/20 ≥5×, 8/20 between 3.88×–4.85×. Sub-5× prompts cluster at lower ntok (62–86) where prefill amortization caps the M=64 speedup. Threshold was theoretically set against M→∞ asymptote. K1987 PASS by 11× margin shows the proxy threshold is the deficient artifact, not the underlying mechanism.

## PROVISIONAL routing rationale

Per reviewer.md §5 PROVISIONAL clause:
- Disk verdict PARTIALLY_SUPPORTED → DB `--status supported` forbidden (all_pass=false).
- F#666 pair analysis → KILL not justified (no pair has both fail).
- `is_smoke=false` + measured KCs → not the standard smoke-PROVISIONAL or "structural-PASS + target-not-measured" sub-cases. This is a **NEW PARTIALLY_SUPPORTED PROVISIONAL sub-case**: full-N measured, target-pair has 1 PASS + 1 FAIL with mis-pairing-structure finding.
- Two-step workaround:
  1. `experiment update --status provisional` ✓ (executed before this REVIEW write)
  2. `experiment evidence --verdict inconclusive`
  3. `experiment finding-add --status provisional`
  4. Verify finding-list

## Assumptions logged (from MATH §8 + PAPER §6 carried forward)

- A1 (GSM8K-Hard → GSM8K-valid): K1986/K1987 data-difficulty-independent. ✓
- A2 (n=199 vs n=200): 0.5% shortfall within K1987 noise. ✓
- A5 (NEW, full iter): K1986 0.99 threshold was theoretically set assuming bit-exact-at-prefill ⇒ argmax-equal-at-gen-step. Refuted at full N — cascade-divergence at fp16 ties produces 6%-of-prompts severe divergence (mean tail) rather than 5%-of-tokens minor divergence (per-prompt tail). Threshold should be re-derived from observed fp16-tie cascade-rate, NOT from per-token roundoff assumption.
- A6 (NEW, full iter): K1838 5× threshold uses parent Theorem 2 asymptotic (M→∞). At M=64 with mean prompt length ≈ 95 tokens, prefill amortization caps speedup ≈ 4.5–5×. Finite-M-corrected threshold of 4.5× would PASS.

## Cluster context

- F#690 (parent design `exp_rdt_loop_kv_cache`): PROVISIONAL.
- F#785 (`_impl` smoke): K1837 PASS_SMOKE bit-exact n=2×T=3.
- F#787 (`_full` smoke iter ~70): K1837 bit-exact + K1838 plumbing-PASS + K1986 plumbing-PASS at n=2×M=8.
- **THIS ITER (full pueue task 3, F#NEW filed below):** K1837 bit-exact PASS at 80 pairs + K1987 budget unlock PASS at 642s + K1838 borderline FAIL 4.885× + K1986 cascade-divergence FAIL 0.940 mean → PARTIALLY_SUPPORTED.

## Follow-up `_full_v2` recommendation

NOT filed in this REVIEW pass (drain backlog already at P≤2 = 6, and the remediation is **threshold re-derivation in MATH**, not new code or longer compute). Recommend deferring to analyst pass: file `exp_rdt_loop_kv_cache_full_v2` at P3 with:
- K1986_v2: re-derived threshold from observed fp16-tie cascade-rate distribution (e.g. mean ≥ 0.94 / min ≥ 0.40 — based on this run's empirical distribution).
- K1838_v2: finite-M corrected threshold 4.5× at M=64.
- Inherit K1837/K1987 verbatim (both PASS this iter).
- Add K_NEW: cascade-divergence rate (proportion of prompts where mid-gen argmax diverges) ≤ 10% — directly verifies the mechanism MATH §4 K1986-corollary should have been formulated as.

## Drain accounting (post-this-review)

- P≤2 open: was 7 (kv_cache_full active + 6 others). After PROVISIONAL routing: 6 P≤2 open (kv_cache_full removed from active queue).
- Active: 1 → 0 after this iter.
- Finding-ledger: 47 → 48 after F#NEW filed.
