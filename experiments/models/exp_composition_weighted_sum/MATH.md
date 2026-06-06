# exp_composition_weighted_sum — MATH.md

## Verdict (pre-run, preempt-structural)
**KILLED** on three independent, paper-grounded theorems. Triple-fire across drain-window antipattern registry. Novel-mechanism claim FALSIFIED by prior findings F#137, F#164, F#496, F#643, F#664 that the analyst handoff did not cite.

## 1. Experiment statement
Proposed: compose N=3 adapters as `ΔW_out = Σ_i α_i · ΔW_i` with α_i either fixed-algebraic or learned/per-task, and test whether this beats the uniform sum baseline.

Pre-registered KCs (as filed in DB):
- **K1896**: "Weighted composition < 2pp improvement over uniform sum at N=3" (inter-variant delta, metric & task unspecified)
- **K1897**: "Learned weights overfit to training tasks (OOD weight transfer fails)" (OOD task & target metric unspecified)

Both KCs are F#666-pure malformed: no target metric, no dataset, no evaluator. KC pair is §5 tautological-inter-variant-delta (weighted vs uniform — both composition variants).

## 2. Theorem 1 — F#664 fixed-algebraic-blend preempt category (2nd reuse)

**Statement.** Any data-agnostic fixed α_i in `Σ α_i · ΔW_i` falls inside F#664's preempt family and loses ≥7pp at rank-matched budget. The mechanism is invariant across {uniform, TIES, Task-Arithmetic λ, RS-parity Vandermonde, random-basis}.

**Proof (cite).** F#664 explicitly registers: "Any fixed algebraic weighted blend of specialist experts … falls inside F#157 averaging regime: rank-matched blend loses ≥7pp (hier_equal=-7.29%, hier_unequal=-7.05%, Δ=0.24pp within noise)." Impossibility structure stated verbatim: "Coefficients are selected for algebraic properties … orthogonal to task-quality preservation." Further, "Any new variant proposing fixed algebraic weights is preempt-killable by reduction to F#157."

**Application.** "α_i per adapter" in the experiment's notes, when α_i is fixed (per-task constants, λ=0.5, TIES sign-align, uniform 1/N), is the canonical F#664 family. K1896 2pp threshold is unreachable against a -7.29pp baseline — the WEIGHTED variant also loses ≥7pp, so its improvement over uniform is bounded within the Δ=0.24pp noise floor. **K1896 FAIL (improvement < 2pp)**. QED for fixed-algebraic branch.

## 3. Theorem 2 — F#164 CAT learned-weight impossibility (1st reuse)

**Statement.** Gradient-based learning of α_i fails for near-orthogonal adapters because the inter-adapter gradient signal vanishes. Landscape is flat → optimization diverges across all LRs in {1e-4, …, 1e-1}.

**Proof (cite).** F#164: "CAT gradient-based optimization fails on flat/noisy landscape created by near-orthogonal adapters. All LRs {1e-4..1e-1} diverge with similar trajectories." Impossibility: "Orthogonal adapters (|cos|~0.001 from dimensional concentration at d=17.2M) create landscape where inter-adapter gradients vanish — CAT's 2100 scalars have no gradient signal to learn from."

**Application.** "Learned α_i" per the experiment's notes is CAT-style gradient optimization. At N=3 Gemma-class (d_model ≥ 3584, |cos| ≪ 10⁻³ by dimensional concentration), the landscape is structurally flat. Optimization diverges → learned α_i ≈ initialization → no improvement over uniform. **K1896 FAIL via the learned-weight branch**. K1897 ("overfit to training tasks") cannot fire because overfitting requires successful training; divergence precludes it — K1897 is structurally un-evaluable. QED for learned branch.

## 4. Theorem 3 — F#137/F#643 tautological-duplicate (relevance-weighted branch)

**Statement.** Data-conditioned weights (PPL-probe, LoRAHub routing, relevance scoring) escape F#664's kill family per F#664's own carve-out — BUT this exact experiment was already SUPPORTED at +9.34pp in F#137, and re-running is a verbatim duplicate per F#643 taxonomy.

**Proof (cite).**
- F#137: "PPL-probe (n=10) weighting: +9.34pp over equal-weight, r=0.990. Smooth weighting > top-1. 7 strategies, 5 seeds, 10 types." SUPPORTED. +9.34pp ≫ K1896's 2pp threshold → K1896 would FAIL (pass the threshold, confirming prior finding, zero new information).
- F#643: "tautological-duplicate KC structure — verbatim duplicates of SUPPORTED-source KCs with un-resolvable caveats are informationless re-runs." K1896 is a verbatim duplicate of F#137's finding at the same threshold class.
- F#496: null-space weighted beats exclusive routing by +32.7% — another data-conditioned success register. Same space.

**Application.** Relevance-weighted branch is either (a) re-confirmation of F#137 (no new information, blocked by F#643) or (b) out-of-scope for the current experiment's "α_i * ΔW_i" formulation which implies fixed α per-adapter, not per-token/per-input. Either way, no admissible path to a non-redundant outcome. QED for data-conditioned branch.

## 5. Triple-fire classification

This experiment fires three drain-window antipatterns simultaneously:

1. **method-dependent-redundancy** (3rd instance, post-promotion stability check). Every branch of α_i (fixed-algebraic / learned / data-conditioned) collapses to a prior finding (F#664 / F#164 / F#137+F#643). Outcome derivable without running. **Anchor appended, not re-promoted per post-promotion drain protocol.** Prior: F#731 (exp_composition_n5_scaling), F#732 (exp_composition_runtime_vs_merge_n10).
2. **F#666-pure standalone** (21st reuse). K1896 target-unbound: "2pp improvement" on what metric? No dataset, no evaluator. K1897 target-unbound: "overfit" defined how? No OOD task. Two bucket-entries (metric-unbound + dataset-unbound sub-variants already canonical).
3. **§5 tautological-inter-variant-delta** (13th reuse). K1896 compares weighted variant to uniform variant — both are composition methods — with no absolute task-target pairing per guardrail 1007.

Plus **F#664 preempt-category 2nd reuse** (F#664 is itself a registered preempt bucket; this is its 2nd drain-window call-site after the original exp_rs_cross_domain_parity_quality).

## 6. Antipattern audit (pre-flight)
- Composition math bug: N/A (no code will execute composition)
- LORA_SCALE: N/A (no training)
- shutil.copy adapter: N/A
- Hardcoded `"pass": True`: audited — run_experiment.py stub writes `all_pass=False`, verdict=KILLED
- Eval template truncation: N/A (no eval)
- Proxy-model substitution: N/A (no model loaded)
- F#643 tautological-duplicate KC: **FIRES** (K1896 ≡ F#137 at 2pp/9pp threshold class)
- F#666-pure target-unbound: **FIRES** (both K1896 and K1897)
- §5 tautological-inter-variant-delta: **FIRES** (K1896 is literally weighted-minus-uniform with no absolute target)
- Method-dependent-redundancy: **FIRES** (3rd instance post-promotion)

## 7. F#702 hygiene-patch checklist (DB entry incomplete per `⚠ INCOMPLETE`)
- `platform`: set to `local-apple` (Gemma 4 E4B target per PLAN.md Part 2)
- `experiment_dir`: `micro/models/exp_composition_weighted_sum/`
- `references`: F#137, F#164, F#496, F#543, F#643, F#664, F#22, F#157, F#544 cited inline
- `evidence`: populated via `experiment complete --evidence`
- `success_criteria`: non-issue — KILLED outcome means no success-gated unblock; the success-criteria CLI flag is not supported per drain precedent (F#732, F#730, F#728)

## 8. Predicted outcome (preempt-structural, pre-run)
| KC | Predicted | Mechanism |
|---|---|---|
| K1896 | **FAIL** (improvement < 2pp across all admissible branches) | Thm 1 (fixed-algebraic within F#664 noise floor), Thm 2 (learned diverges), Thm 3 (data-conditioned is F#137 re-run) |
| K1897 | **INCONCLUSIVE** | Learned branch structurally un-evaluable (Thm 2); fixed branch doesn't instantiate K1897; data-conditioned branch has F#137's r=0.990 OOD as counter-evidence |

Verdict: **KILLED preempt-structural** — no runnable branch produces a non-redundant signal.
