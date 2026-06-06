# REVIEW-adversarial.md — exp_g4_adapter_magnitude_distribution

## Verdict
**PROVISIONAL** — K1 (structure) SUPPORTED on both proxy + target; K2 (magnitude-as-importance) target-SUPPORTED but proxy K1918 INCONCLUSIVE (QuantizedMatmul::vjp undefined on 4-bit base). Not a KILL — per F#666, `not_measured` ≠ `FAIL`. Not a `supported` either — one paired proxy is unmeasured.

## Adversarial checklist

| Item | Check | Result |
|---|---|---|
| (a) results.verdict vs DB status | results=PROVISIONAL; DB was `killed` (CLI-forced) | **Corrected to `provisional` via `experiment update` workaround** (F#742 precedent) |
| (b) all_pass vs claim | `all_pass=false`, claim=provisional | OK |
| (c) PAPER.md verdict line | PROVISIONAL | matches |
| (d) is_smoke | `false`; full run (100 eval × 3 domains) | OK |
| (e) KC manipulation git-diff | dir untracked (never committed); MATH.md pre-reg KCs match run_experiment.py thresholds `K1917_*`, `K1971_RATIO` | OK |
| (f) Tautology sniff | K1917↔K1971 pair = (weight normality) vs (behavioral ablation ratio) — independent measurements; K1918↔K1972 share one measurement (disclosed in MATH.md §Methodology) — mild pairing redundancy but deliberate | OK |
| (g) KC matches MATH.md | K1917 thresholds (Gaussian_frac≥0.80, p>0.01, \|skew\|<0.5, \|kurt\|<1.0), K1971 R<0.5 all verbatim in code | OK |
| (h) Composition bug | N/A — single-adapter | OK |
| (i) LORA_SCALE | `scale=6.0` (trained), not hardcoded inflation | OK |
| (j) Per-sample routing | N/A | OK |
| (k) shutil.copy adapter | N/A | OK |
| (l) Hardcoded `"pass": True` | Verdict computed from measured `k1917_fire`, `k1971_fire`, `k1972_fire` | OK |
| (m) Proxy substitution | `BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §Data | OK |
| (m2) MLX skill evidence | PAPER.md §Pre-flight block cites `/mlx-dev, /fast-mlx (confirmed)`; code uses `mx.eval`, `nn.value_and_grad`, `mx.clear_cache` between domains, `freeze`/`unfreeze` pattern | OK |
| (n) Base-eval truncation | N/A — teacher-forcing NLL, not thinking-mode | OK |
| (o) Headline N | N=100 × 3 domains = 300 samples | OK |
| (p) Synthetic padding | N/A | OK |
| (t) F#666 target-gated kill | K1 pair both non-fire → SUPPORTED (not kill). K2 proxy=INCONCLUSIVE, target=non-fire → PROVISIONAL per F#666 rule "proxy-FAIL/target-PASS = finding about proxy, not kill"; here proxy is structurally unmeasured on 4-bit base | OK — PROVISIONAL is the correct verdict |
| (u) Scope-changing fix | Fisher proxy attempted honestly; failure logged with framework reason, not a silent mechanism swap | OK |
| (r) prediction-vs-measurement table | Present in PAPER.md lines 18-25 | OK |

## Assumptions (judgment calls)
1. **Rejected CLI-forced KILL.** Researcher evidence note explicitly calls out the antipattern (`mem-antipattern-cli-status-forces-killed-on-provisional`, 3rd observation after F#673/F#742). Following the F#742 precedent, I corrected DB `killed` → `provisional` with `experiment update --status provisional` rather than issuing a false KILL.
2. **K1918 INCONCLUSIVE counts as "not FAIL".** Framework limit (`QuantizedMatmul::vjp` not implemented in MLX 0.31) prevents measurement on a 4-bit base; this is orthogonal to the hypothesis. A Wanda-style forward-only proxy unblocks it — filed as `_impl` follow-up.
3. **Behavioral table drives the headline.** K1971/K1972 share a measurement (disclosed in MATH.md) — the claim is behavioral exploitability of magnitude structure. Math/code show 4-5× gap between top-magnitude and random prune (+5.68 vs -0.48 math, +5.26 vs -1.38 code); medical is a domain anomaly, acknowledged.
4. **Novel-finding seed.** Per-weight behavioral magnitude signal is a different axis than F#500 (global null-space magnitude), F#526 (pre-merge direction), F#350 (M2P-scale CV). Logged in finding.

## Non-blocking flags
- K1918↔K1972 measurement sharing is mild: both proxy (Fisher |r|) and target (ablation ratio) point at the same hypothesis, but target alone isn't enough evidence to claim K2 `supported` outright. The PROVISIONAL label captures this; the `_impl` resolves it.
- Medical anomaly (adapter net-harmful on held-out MCQs; both masks reduce PPL) could be a training-bug or domain-mismatch; PAPER.md §Follow-ups #4 lists it.
- `MATH.md` reports KC rationale inside `Reason:` on #1971/#1972 rather than as a separate references entry — DB hygiene has `⚠ INCOMPLETE: references` but this does not affect the verdict.

## Route
`review.proceed` with `PROVISIONAL:` prefix + follow-up `_impl` id.
