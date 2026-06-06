# MATH.md — exp_adapter_orthogonality_audit (PREEMPT-KILL, F#666-pure, 2nd instance)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure standalone — 2nd drain-window instance)

This experiment is preempt-killed before any code is run. The kill is **structural**: the pre-registered kill-criterion set K = {K1857, K1858} consists entirely of proxy geometric metrics (pairwise cosine, effective-rank) with no target-metric pairing. Under F#666 (guardrail 1007) neither KILL nor SUPPORTED is derivable regardless of empirical outcome.

This is the **second** F#666-pure standalone preempt-KILL in the drain window (after F#700 on `exp_g4_per_layer_cos_baseline`). Per prior analyst watchlist threshold ("promote at 2nd recurrence"), this iteration triggers promotion of:
1. **F#666-pure standalone preempt-KILL** → antipattern memory.
2. **Broadly-malformed pre-reg** (≥3 simultaneous empty/null hygiene fields + F#666 violation) → antipattern memory.

Both patterns confirmed structurally identical across F#700 and this experiment.

## §0 Platform / skills / model pins

Included for completeness per reviewer checklist item (m2), even though no platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written; canonical preempt-form disclosure.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — this experiment is a measurement-only audit (no training, no LoRA composition).
- Parent dependency: **none** (`depends_on: []`). This is NOT an F#669 preempt.
- Adapters that would have been audited: existing safetensor artifacts under `micro/models/exp_p1_t2_single_domain_training/adapters/{code,math,medical}/adapters.safetensors` (and others). **Not loaded.**

## §1 Preempt-KILL theorem (F#666-pure, 2nd instance)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_adapter_orthogonality_audit` with kill-criterion set K = {K1857, K1858}:
- K1857 := "Pairwise cosine between any two adapter weight matrices > 0.15 (not orthogonal)"
- K1858 := "Effective rank of N-adapter stack < N * rank/2 (subspace overlap > 50%)"

**Classification of K.**
- K1857 is explicitly a **proxy metric** — F#666 guardrail 1007 lists cosine by name among forbidden-solo proxies ("classification accuracy, routing match rate, PPL, **cosine**, clustering purity").
- K1858 is a **proxy metric** — effective-rank and subspace-overlap are geometric structural statistics of weight matrices. They measure subspace alignment, not behavioral composition success. F#42 explicitly caveat-flags this exact gap: "Cosine computed over raw (A,B) concatenations, not effective delta W=BA — two adapters with orthogonal (A,B) could produce correlated effective perturbations."

Neither KC measures task accuracy, behavioral quality, oracle-gap, or any downstream-behavioral outcome. K is a proxy-only set (2 proxies, 0 targets).

**F#666 gating (guardrail 1007).** KILL requires **both** a failing proxy KC and a failing target KC. SUPPORTED requires **both** to pass. A verdict derived from a proxy-only KC set is tautological:
- Proxy-PASS + no-target = tautological SUPPORT ("adapters look orthogonal" without behavioral evidence of composition success).
- Proxy-FAIL + no-target = cannot KILL per F#666 (Proxy-FAIL with target-absent has the same structural issue as Proxy-FAIL + target-PASS: a finding about the proxy, not about the behavior).

**Corollary (standalone F#666-pure preempt).** Let V(K) be the verdict derivable from K = {K1857, K1858}. For every 2×2 combination of {PASS, FAIL} assignments to K1857, K1858:

| K1857 | K1858 | V(K) under F#666                                                                 |
| ----- | ----- | -------------------------------------------------------------------------------- |
| PASS  | PASS  | Proxy-SUPPORT with no target pair → tautological                                 |
| PASS  | FAIL  | Mixed-proxy with no target pair → cannot KILL; ambiguous                         |
| FAIL  | PASS  | Mixed-proxy with no target pair → cannot KILL; ambiguous                         |
| FAIL  | FAIL  | Proxy-only FAIL; per F#666, "Proxy-FAIL ... = finding about the proxy, not kill" |

**None of the four cells yields a valid verdict under F#666.** K is unidentifiable. **QED.**

**Semantic corroboration via F#571.** Independently of F#666, the behavioral question this audit would pretend to answer (does orthogonality hold for N>1 trained adapters?) is **already empirically settled** in the behavioral direction: F#571 established that N>1 composition FAILS behaviorally on Gemma 4 E4B (cross-term compound under LayerNorm nonlinearity). Whether trained-adapter cosine remains ≤ 0.15 or not, the N>1 composition is behaviorally blocked — so the audit's value is near-zero even if it were runnable under F#666 discipline.

### §1.1 Secondary structural defects

Per `experiment get exp_adapter_orthogonality_audit`:

1. **`success_criteria: []`** — empty. No SUPPORTED-condition declared. Independent of F#666, this blocks any SUPPORTED verdict.
2. **`references: []`** — violates guardrail 1002 ("Every new experiment MUST cite an arxiv paper or prior finding"). The experiment `notes` field references F#562 but no formal citation is registered. Prior findings relevant to the audit (F#42 cosine plateau, F#562 Grassmannian construction, F#571 N>1 composition failure) are not linked.
3. **`platform: null`** — unset. Violates MATH.md §0 discipline; without a platform pin the base model / framework cannot be declared.

Each of (1)-(3) independently warrants a KC-augmentation pass. Combined with the F#666 violation, the pre-reg has 4 simultaneous defects.

**This exact 4-defect combination (F#666 + empty success_criteria + empty references + null platform) was observed in F#700 one drain-iteration ago.** Two instances with identical structure ⇒ the pattern is reliable and should be promoted to antipattern memory.

## §2 Prior art

- **F#666** (target-gated KILL discipline, guardrail 1007): KILL requires proxy+target, SUPPORTED requires proxy+target.
- **F#700** (2026-04-24): 1st F#666-pure standalone preempt-KILL on `exp_g4_per_layer_cos_baseline` (single proxy KC K1856, no parent, success_criteria/references/platform all empty-or-null). This experiment is the 2nd instance of the same sub-case — promotion trigger.
- **F#42** (2026-03-21, conclusive): Cosine orthogonality plateaus at convergence on BitNet 2B. Caveat explicitly flags that raw-(A,B) cosine does NOT imply effective-delta-W orthogonality → independent structural reason this audit cannot yield a behavioral verdict.
- **F#562** (2026-04-18, supported): Partition-QR structural orthogonality verified at Gemma 4 native dims at **pre-training construction** only. Post-training stability is deferred to F#42.
- **F#571** (2026-04-18, killed-macro): N>1 composition behaviorally fails on Gemma 4 E4B (cross-term compound under LayerNorm). Semantic corroboration that even a valid cosine audit would yield near-zero actionable information.
- **F#669 / F#687 / F#698 / F#699**: preempt-KILL family for parent-unverified target. Not applicable (no parent dep here).
- **Guardrail 1002**: experiments MUST cite a paper/finding.
- **Guardrail 1007** (F#666): target-gated KILL.

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                                       | Kind  | Measurement status                                  |
| ----- | ------------------------------------------------------------------------------------------- | ----- | --------------------------------------------------- |
| K1857 | Pairwise cosine between any two adapter weight matrices > 0.15 (not orthogonal)             | proxy | untested (preempt-blocked, F#666-pure structural)   |
| K1858 | Effective rank of N-adapter stack < N * rank/2 (subspace overlap > 50%)                     | proxy | untested (preempt-blocked, F#666-pure structural)   |

No target-metric KC exists. K is structurally malformed per F#666.

KC text is preserved verbatim from DB (checked byte-for-byte against `experiment get` output). No post-claim KC mutation.

## §4 Unblock condition

Re-claim requires **KC-augmentation** (pre-registration modification before re-claim):

1. Add at least one target-metric KC pairing K1857/K1858 to a behavioral outcome. Candidate formulations:
   - "Composing N=3 trained Gemma 4 E4B adapters (code + math + medical from `exp_p1_t2_single_domain_training`) via mean-summation retains ≥90% of single-adapter task accuracy on each domain's held-out eval set." — This directly tests whether orthogonality-at-the-weight-level translates to behavioral composition success.
   - Or: "Mean pairwise cosine across adapters correlates (Spearman |r| ≥ 0.4) with N=3 composition-task-accuracy gap vs single-adapter baseline." — Ties cosine-proxy directly to behavioral target.
2. Add references: at minimum F#42, F#562, F#571. Ideally an arxiv paper on adapter composition / task-arithmetic (e.g. Zhong et al. 2504.10957 cited in F#571).
3. Set `platform=mlx` explicitly.
4. Populate `success_criteria` (mirror of KC pass condition).

These edits must happen **before** re-claim (KC-pre-registration rule; post-claim KC mutation is antipattern-u per reviewer checklist).

**Note on research value.** F#571 already behaviorally settled N>1 composition on the target platform. An augmented audit would likely re-confirm the known failure at the proxy level. The higher-value redesign is an **orthogonality-intervention experiment**: does a structural fix (e.g. Riemannian-constrained training keeping A,B on the Grassmannian, per F#562 Theorem 1) preserve behavioral composition success at N>1? That would be a **runnable behavioral hypothesis**, not a proxy audit.

Recommendation: **close this pre-reg as structurally-malformed**, do not resurrect. If adapter-geometry questions remain, re-register under a fresh experiment id with target-gated KC + behavioral outcome.

## §5 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700 precedent + reviewer.md §5). Unblock is pre-registration-external (requires editing the DB entry, not writing an impl).

No `mem-antipattern-impl-follow-up-delegation` obligation: that antipattern applies to novel-mechanism PROVISIONAL only. Preempt-structural KILL is explicitly excluded.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT:
- Run the measurement and mark K1857/K1858 PASS/FAIL in isolation (would produce an F#666-tautological verdict).
- Invent a target-metric KC post-claim (would be antipattern-u, post-claim KC mutation).
- Swap to an easier proxy (e.g. cosine between just 2 of the 12+ available adapters) — preserves the F#666 violation.
- Substitute an alternate audit target (e.g. base-model weight cosine) — would misrepresent the pre-reg.

KC text preserved verbatim from DB. Results.json encodes both KCs with `result=untested` and preempt reason. No silent simplification.

## §7 Sub-case taxonomy and promotion

Drain-window taxonomy after this iteration:

| Sub-case                                      | Parent status       | KC-structure        | Finding             | Count |
| --------------------------------------------- | ------------------- | ------------------- | ------------------- | ----- |
| F#669 classic (parent-unverified, F#666-ok)   | PROVISIONAL         | target-gated        | F#669/F#687/F#699   | 3     |
| F#669 + F#666 compound                        | PROVISIONAL         | proxy-only          | F#698               | 1     |
| **F#666-pure standalone**                     | **none**            | **proxy-only**      | **F#700 + this**    | **2** |
| (runnable, F#666-compliant)                   | none / SUPPORTED    | target-gated        | regular KILL/SUPPORT | N/A   |

**Row 3 now 2 instances.** Promotion to antipattern memory is triggered per analyst watchlist rule ("promote at 2nd recurrence"). See `§8 Promoted antipatterns` below.

## §8 Promoted antipatterns (this iteration)

Two antipattern memories will be filed after this MATH.md is written (one promotion trigger per pattern):

### Antipattern AP-F666-pure-standalone
**Pattern:** Experiment with `depends_on: []` (no parent) but KC set is proxy-only (no target-metric pair). Canonical form: single or multi-KC set containing only cosine / variance / rank / clustering-purity / routing-match / PPL / classification-accuracy style metrics, with no paired behavioral-outcome KC.

**Detection:** At claim time, inspect `kill_criteria` list. If every KC is a proxy metric (per F#666 enumeration), preempt-KILL. No running, no partial measurement.

**Reason:** F#666 (guardrail 1007) makes proxy-only verdicts structurally invalid. Running produces either a tautological SUPPORT (proxy passes without behavioral anchor) or a forbidden KILL (F#666 rule "Proxy-FAIL + target-absent = finding about proxy, not kill").

**Confirmed instances:** F#700 (`exp_g4_per_layer_cos_baseline`, K1856 cos-sim variance), this experiment (`exp_adapter_orthogonality_audit`, K1857 pairwise cosine + K1858 effective rank).

### Antipattern AP-prereg-hygiene-multi-defect
**Pattern:** Pre-reg is simultaneously missing `success_criteria` (empty list), `references` (empty list), and `platform` (null), AND the KC set is F#666-violating.

**Detection:** At claim time, the CLI's own `⚠ INCOMPLETE` warning names the empty fields. If 3+ hygiene fields are empty/null AND the KC set is proxy-only, the pre-reg is structurally unsalvageable and should be preempt-killed + recommended-for-reregistration rather than patched.

**Reason:** Single defects are benign and patchable; simultaneous 4-defect pre-regs indicate a design process that hasn't engaged with F#666/F#42/F#571 prior art. Patching is strictly more work than a clean re-register (re-scoped as a Hedgehog-family-sibling per F#700 precedent, or as an intervention-experiment per this experiment's §4 recommendation).

**Confirmed instances:** F#700 (success_criteria + references + platform empty/null + K1856 F#666), this experiment (success_criteria + references + platform empty/null + K1857+K1858 F#666).
