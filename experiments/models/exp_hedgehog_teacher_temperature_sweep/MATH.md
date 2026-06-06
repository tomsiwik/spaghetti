# MATH — exp_hedgehog_teacher_temperature_sweep

## §0 Skill invocation disclosure (preempt-structural carve-out per F#716 / F#720 / F#721)

- Base model / adapters: **Not loaded.** This is a preempt-structural KILL; no MLX code is executed. The `run_experiment.py` is a graceful stub (json + pathlib only) that writes the KILLED verdict.
- Platform skills (`/mlx-dev`, `/fast-mlx`): **Not invoked.** Carve-out precedent: triple-fire / preempt-structural pre-reg excludes code-execution, so MLX-skill requirement does not apply (reviewer §(m2) carve-out, canonical at F#716 / F#720 / F#721).

## §1 Theorem (primary): V(K) unidentifiable under F#666 for this KC set

**Pre-reg KC set:**
- K1875 — "Temperature T > 1.0 produces cos-sim > 0.05 better than T=1.0 (default is suboptimal)"
- K1876 — "Temperature T < 0.7 produces cos-sim > 0.05 worse than T=1.0 (too peaked)"

Both KCs are **pure cos-sim** proxy metrics (guardrail 1007 enumerates "cosine" as a canonical proxy). `depends_on: []`. Zero target KCs.

**L1 — 2-outcome F#666 truth table (F#666-pure-standalone primary, 15th drain-window instance):**

| K1875 | K1876 | Outcome class | V(K) identified? |
|-------|-------|---------------|------------------|
| PASS  | PASS  | Tautological SUPPORT (both cos-sim variants beat thresholds without target anchor) | No — proxy-only |
| PASS  | FAIL  | Finding about the proxy — F#666 blocks KILL | No |
| FAIL  | PASS  | Finding about the proxy — F#666 blocks KILL | No |
| FAIL  | FAIL  | Finding about the proxy — F#666 blocks KILL | No |

Every branch produces a proxy-only verdict; V(K) is unidentifiable. **QED L1.** Preempt-KILL per `mem-antipattern-f666-pure-standalone-preempt-kill`.

**L2 — §5 tautological-inter-variant-delta (secondary, 9th drain-window instance; 1st intra-Hedgehog-temperature-delta sub-variant):**

Both KCs have the form `cos_sim(adapter_T=X) − cos_sim(adapter_T=1.0) op δ` where `op ∈ {>, <}`. Neither operand is the unadapted base; both are Hedgehog adapter variants parameterized by teacher temperature T. No paired base-anchored KC `cos_sim(adapter_T=X, base)` is pre-registered.

Under F#477 collapse-regime precedent, all Hedgehog r=6 adapters under varied teacher temperatures may collapse to near-identity cos-sim Δ ⇒ inter-variant delta trivially within or outside thresholds regardless of T. **§5 fires secondary** because the underlying metric is proxy under F#666; §5-patch (add per-variant base-anchor) leaves cos-sim as proxy ⇒ does not unblock. **1st intra-Hedgehog-temperature-delta sub-variant** under the intra-instantiation meta-category (cousin of F#712/F#716 intra-adapter-rank, F#720 intra-loss-function-delta, F#721 intra-layer-selection).

**L3 — F#702 hygiene-patch structurally unavailable (6th post-promotion confirmation):**

`mem-impossibility-f666pure-saturation-implies-f702-unavailable` (PROMOTED at F#716): with 0 target KCs, F#702 `_impl` follow-up has no target to measure ⇒ patch path is a no-op. F#714/F#715/F#716/F#720/F#721 established and confirmed the impossibility structure across 3 fire-modes (triple ×4, double ×1); this is the 6th confirmation and 5th triple-fire instance. Memory remains stable; anchor-append only.

**L4 — Triple-fire hierarchy axis-invariant (5th triple-fire instance, post-promotion anchor-append):**

`mem-pattern-triple-fire-hierarchy-axis-invariant` (PROMOTED at F#721): hierarchy F#666-pure (KC class) > §5 (KC form) > hygiene (metadata) is axis-invariant across drain-window §5 axes. This instance adds a **6th distinct §5 axis** (intra-Hedgehog-temperature-delta). Post-promotion; anchor-append only.

**L5 — Distinctions from neighbour clauses:**

- NOT `mem-antipattern-preempt-child-parent-target-unverified` (F#669-family): `depends_on: []` — no parent dependency to verify.
- NOT `mem-antipattern-template-regression`: no parent finding cited in `notes`; no parent template to inherit from.
- NOT `mem-watchlist-f666pure-proxy-only-lineage-inheritance`: no parent; vacuous.
- NOT `mem-antipattern-cross-paper-combined-loss-tautology` watchlist: single method (Hedgehog), single loss (cos-sim distillation); no `L = L_A + λ·L_B` composite.
- NOT `mem-pattern-novel-mech-primary-plus-hygiene-secondary-pairing`: primary is F#666-pure (zero target KCs), not novel-mechanism. Sibling-but-disjoint to F#717/F#718/F#719 (all carry target KC; this does not).
- **Sibling-with-weaker-KC (not parent) to F#719/F#720/F#721:** all four are Hedgehog-ablation sub-type variants; this is the 4th sub-type (hyperparameter-ablation) after axis-extension, loss-variant-ablation, layer-selection-ablation. No formal `depends_on` edge ⇒ NOT template-regression.

## §2 Antipatterns fired (hierarchy-canonical)

| Rank | Antipattern | Fire mode | Evidence |
|------|------------|-----------|----------|
| Primary | `mem-antipattern-f666-pure-standalone-preempt-kill` | KC class: both KCs proxy (cos-sim), `depends_on=[]` | L1 2-outcome truth table |
| Secondary | `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline` | KC form: inter-variant-delta without base-anchor | L2 1st intra-Hedgehog-temperature-delta |
| Tertiary | `mem-antipattern-prereg-hygiene-multi-defect` | Metadata: 3 defects (success_criteria, platform, references) | CLI `⚠ INCOMPLETE: success_criteria, references, platform` |

`fire_mode = triple` (5th triple-fire in drain window after F#714/F#716/F#720/F#721).

## §3 Proxy-flavor bucket classification

Both KCs: **pure cos-sim** (not derived-geometric multi-proxy; no eff-rank or pairwise-cos auxiliary). Lands in the 6th bucket `cos-sim`, incrementing it from 1 (F#720) to **2 instances**.

⇒ **Merge-with-derived-geometric trigger reached** per F#720 pre-commit: "if a 2nd pure-cos-sim F#666-pure instance appears, consider merging under derived-geometric rather than maintaining a separate bucket." Analyst action: merge cos-sim bucket into derived-geometric super-bucket; derived-geometric super-bucket becomes 6 instances (4 prior + 2 merged cos-sim).

## §4 Recommendation

Close this pre-reg; do not resurrect by field-patching (F#702 path unavailable per L3). Re-register as a new experiment with a target-metric KC set on first principles: pair each inter-variant cos-sim KC with a behavioral-quality target KC (e.g., "oracle-gap on politeness adapter task accuracy vs T-sweep") AND add a per-variant base-anchored KC (e.g., `cos_sim(adapter_T=X) − cos_sim(base) ≥ γ`). Recommended v2 name: `exp_hedgehog_teacher_temperature_sweep_v2_target_paired` — reuse F#683 _impl's politeness behavioral benchmark once the 26B teacher cache blocker clears.

## §5 Hedgehog-ablation super-family ledger (after this KILL)

| Sub-type | Instances | Exemplars | KC-design outcome |
|----------|-----------|-----------|-------------------|
| axis-extension (domain) | 7 (4 closed + 3 other) | F#682–F#718 | target-paired → PROVISIONAL |
| loss-variant-ablation | 2 | F#719 (KL-div, paired) + F#720 (MSE, pure-proxy) | bifurcated: paired → PROVISIONAL; pure-proxy → KILL |
| layer-selection-ablation | 1 | F#721 (top-6 vs all) | KILL (pure-proxy) |
| **hyperparameter-ablation** (NEW) | **1** | **this (temperature sweep)** | **KILL (pure-proxy)** |

Total: 11 instances across 4 sub-types. KC-design bifurcation (paired → PROVISIONAL; pure-proxy → KILL) is **axis-invariant across super-family**. This instance is the **4th sub-type opening** in the Hedgehog-ablation super-family.

## §6 Kill Criteria status

- K1875: **untested** (preempt-structural; no run)
- K1876: **untested** (preempt-structural; no run)

Reason: `F666_PURE_PREEMPT_KILL + SEC5_INTRA_HEDGEHOG_TEMPERATURE_DELTA + HYGIENE_MULTI_DEFECT`.
