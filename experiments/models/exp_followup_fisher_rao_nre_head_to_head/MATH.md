# Fisher-Rao vs Norm-Rescaled Euclidean: Head-to-Head at Production Scale

## Type: Verification (Type 1)

**Paper:** Fisher-Rao Manifold Merging (arXiv:2603.04972, Wang/Ye/Yin 2025)
**Prior findings:** #274, #275, #615. Prior experiment `exp_fisher_rao_composition_scaling` showed NRE matches FR on all metrics at N=1..15 on BitNet-2B (PPL 9.17 vs 9.20 at N=5). This followup asks: does the equivalence still hold at **production scale** (N=25), and does the ~10× wall-clock cost of FR buy anything measurable?

## A. Failure Mode Identification

**Disease.** Running Karcher-mean iteration (Fisher-Rao, FR) in production routinely costs ~10–20× the wall-clock of a one-line rescale (Norm-Rescaled Euclidean, NRE). If both methods produce statistically indistinguishable adapters under every measurable criterion, the extra cost is unjustified and NRE should be declared the ceiling.

**Structural question (SIGREG).** Does the Riemannian manifold machinery *add* information beyond norm preservation, or is norm preservation the entire mechanism? Finding #275 suggested the latter at N≤15 on BitNet-2B. If that holds at larger N on a different base (Gemma 4 E4B 4-bit), the conclusion generalises.

## B. The Right Question

**Wrong question:** "Is FR better than naive Euclidean averaging?" — already answered yes (Finding #274).

**Right question:** "When the norm-shrinkage confound is removed (NRE baseline), does the Karcher-mean direction averaging on S^(d-1) still provide measurable benefit?"

## C. Prior Math

### C1. Norm Preservation (Theorem 1, prior experiment)

Both NRE and FR produce composed adapters with
`‖ΔW_merged‖ / mean_i ‖ΔW_i‖ = 1` by construction.
Euclidean shrinks as `1/√N` for orthogonal sources (F#274).

### C2. Karcher Mean on the Sphere (Karcher 1977)

On `S^(d-1)`, for points `{u_i}` with pairwise cosines > 0, the Karcher mean is the unique minimizer of
`argmin_u Σ_i arccos(⟨u, u_i⟩)^2`
reached by fixed-point iteration on the log/exp maps.

### C3. Equivalence of NRE and FR to First Order (Finding #275)

For small dispersion on `S^(d-1)`, the Karcher mean is a second-order correction to the extrinsic mean (Pennec, "Intrinsic Statistics on Riemannian Manifolds", 2006). When the spread of `{u_i}` is small, `NRE ≈ FR + O(max_ij arccos⟨u_i,u_j⟩^2)`.

## D. Derived Guarantee and Predictions

### D1. Theorem (FR–NRE equivalence under small dispersion).

Let `u_i = v_i / ‖v_i‖` for `v_i ∈ ℝ^d` with pairwise spherical distances `d_{ij} = arccos⟨u_i,u_j⟩ < δ`. Let `u_NRE = (Σ u_i/N) / ‖Σ u_i/N‖` and `u_FR` the Karcher mean. Then
`‖u_NRE − u_FR‖ ≤ C · δ^2` for some constant C (Pennec 2006).

**Implication.** If `δ` is small enough, downstream metrics derived from `u_NRE` vs `u_FR` (perplexity, next-token loss) differ by `O(δ^2)`, which is below measurement noise for adapters trained on orthogonal domains.

### D2. Predictions at N=25 on Gemma 4 E4B 4-bit

P1. **Wall-clock**: FR composition time ≥ 8× NRE composition time at N=25 (Karcher fixed-point ≥ 20 iterations × per-iteration O(N·d)).

P2. **Overall PPL**: |FR PPL − NRE PPL| < 0.05 (noise floor) at N=25 on mixed-domain held-out text.

P3. **Conditional PPL** (loss on assistant tokens only, prompt masked): |FR cond-PPL − NRE cond-PPL| < 0.05 at N=25.

P4. **Both methods beat Euclidean**: `Euc PPL − NRE PPL > 0.3` and `Euc PPL − FR PPL > 0.3` at N=25 (norm shrinkage dominates the Euclidean–NRE gap).

## E. Experimental Design

**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (per PLAN.md Part 2).
**Real adapters:** 3 PoLAR-style LoRA adapters (rank=6, scale=6.0, target=`self_attn.q_proj`) trained 1000 iters each on `code / math / medical` (`micro/models/exp_p1_t2_single_domain_training/adapters/`).
**Scale sweep:** N ∈ {3, 10, 25}. For N>3, cycle the 3 real adapters and add per-element Gaussian noise to B (scale proportional to B's norm), matching prior F#275 methodology.
**Shared A convention:** per layer, take adapter-0's A as the shared basis and compose only over B matrices (consistent with F#275). This isolates direction averaging on the B subspace.
**Composition methods (head-to-head):**
  1. Euclidean: `B̄ = (1/N) Σ B_i`
  2. NRE: `B̄_NRE = B̄ · (mean_i ‖B_i‖ / ‖B̄‖)`
  3. FR: spherical Karcher mean of flattened B vectors, rescaled by mean source norm.

**Apply path:** replace the active LoRA module's B with the composed B; A stays at adapter-0's A (shared). Forward pass through mlx-lm's LoRA layer produces `x + scale · (x @ A) @ B`.

**Evaluation:**
  - Overall PPL on 50 mixed-domain held-out validation samples (domains: code, math, medical).
  - Conditional PPL on assistant-only tokens (prompt masked) over the same samples.
  - Per-composition wall-clock time.

## F. Pre-Registered Kill Criteria (Target-Gated per F#666)

All three KC locked **before** running. Do not modify after data.

**K1 (proxy, target-gated pair):** Fisher-Rao overall-PPL is **lower than NRE overall-PPL by ≥ 0.05** at N=25.

**K2 (target-metric, paired with K1):** Fisher-Rao conditional-PPL (assistant-tokens-only) is **lower than NRE conditional-PPL by ≥ 0.05** at N=25.

**K3 (anti-null sanity):** Both NRE and FR beat raw Euclidean overall-PPL at N=25 by ≥ 0.3 (confirms norm-shrinkage mechanism is active at production N).

**Verdict rules (per F#666):**
- K3 **must** pass or experiment is invalid (no mechanism active → null head-to-head).
- **SUPPORTED** iff K1 PASS AND K2 PASS AND K3 PASS → FR provides measurable benefit; 10× cost justified.
- **KILLED** iff K1 FAIL AND K2 FAIL AND K3 PASS → Finding #275 confirmed at production N; NRE is the composition ceiling; resources freed.
- **PROXY-MIXED** (K1 pass ⊕ K2 pass) → finding about the proxy, not the ceiling claim; file v2 with better KC.

## G. Assumptions & Caveats

1. Shared-A convention reuses adapter-0's A as the basis for all N sources. This is the F#275 methodology and was reviewed as valid for comparing B-composition methods; it does not claim anything about the FR-vs-NRE gap under independent-A adapters.
2. Adapter target is `self_attn.q_proj` (rank 6, scale 6) because those are the available Gemma 4 E4B adapters; per F#627 the proven target is `v_proj+o_proj`. This experiment's claim is about composition-method equivalence and is orthogonal to target-module choice.
3. N>3 uses synthetic B-variants (cycled+noised). This matches F#275's methodology exactly; the norm-shrinkage plateau of Euclidean at `1/√N_real` is a known artifact of synthetic variants and is not the object of K1/K2.
4. `mlx-lm` version pinned at runtime via `import mlx_lm; print(mlx_lm.__version__)` (printed to stdout).

## H. Antipattern Self-Scan

- Composition math: uses `Σ B_i @ A_0` with A shared — matches F#275 and PLAN.md antipattern-001 guidance.
- LORA_SCALE: 6.0 (from trained adapter config), ≤ 8 safe ceiling — OK.
- Hardcoded "pass": none; all KC thresholds are computed from measured values.
- Proxy-model substitution: loading `mlx-community/gemma-4-e4b-it-4bit` as per PLAN.md Part 2.
- KC-swap-after-failure: KCs locked in this file before running; any change after data triggers killed verdict on pre-reg.
- Tautological routing: no routing here — pure composition benchmark.
