# MATH: Correct Delta-Sum LoRA Composition at N=5

## Goal

Verify the mathematical identity that distinguishes the CORRECT delta-sum LoRA
composition formula from the BUGGY sum-of-factors formula that produced the
trillion-PPL catastrophe (Finding #23, `premerge_vs_dynamic.py:415-431`,
Finding #199 family).

## Notation

| Symbol  | Shape        | Meaning                                       |
|---------|--------------|-----------------------------------------------|
| W       | (d_out, d_in)| frozen base weight of a target linear module  |
| A_i     | (r, d_in)    | LoRA-A for adapter i (i = 1..N)              |
| B_i     | (d_out, r)   | LoRA-B for adapter i                          |
| ΔW_i    | (d_out, d_in)| = B_i @ A_i, per-adapter delta                |
| W_solo_i| (d_out, d_in)| = W + ΔW_i                                    |
| N       | scalar       | number of composed adapters (= 5)             |
| r       | scalar       | LoRA rank (target = 6 per Gemma 4 P1 registry)|

## Two Composition Formulas

### Correct (diagonal sum of per-adapter deltas)

$$W_{correct} = W + \sum_{i=1}^{N} B_i A_i = W + \sum_{i=1}^{N} \Delta W_i$$

N terms, one per adapter.

### Buggy (sum of A, sum of B, then product)

$$W_{buggy} = W + \left(\sum_{j=1}^{N} B_j\right)\left(\sum_{i=1}^{N} A_i\right) = W + \sum_{i=1}^{N}\sum_{j=1}^{N} B_j A_i$$

N² terms. N diagonal terms (i=j) match the correct formula; N(N-1) cross-terms
(i≠j) couple adapter i's input subspace with adapter j's output subspace.

## Theorem 1 (Identity Difference)

$$W_{buggy} - W_{correct} = \sum_{i \neq j} B_j A_i$$

**Proof.** Direct expansion of the double sum separating diagonal (i=j) from
off-diagonal pairs. For the correct formula, $\sum_i B_i A_i$ corresponds
exactly to the diagonal terms. ∎

## Theorem 2 (Frobenius Norm Bound, Correct Formula)

Under the assumption that adapter deltas have bounded Frobenius norm
$\|\Delta W_i\|_F \leq M$ for all i, the correct composition satisfies

$$\|\Delta W_{correct}\|_F \leq N \cdot M \quad \text{(triangle inequality)}$$

This is linear in N. For trained LoRA adapters at r=6 on Gemma 4 E4B with
typical ||ΔW_i||_F ≈ 0.01–0.1 in relative weight terms, the composed norm
scales as O(N·M), producing bounded logit perturbations and bounded PPL. ∎

## Theorem 3 (Catastrophe Mechanism, Buggy Formula)

For trained LoRA adapters, A_i encodes input patterns specialized to
domain i, and B_j encodes output corrections specialized to domain j.
The cross-term $B_j A_i$ for $i \neq j$ maps domain-i input patterns
through domain-j output corrections — an operation with no semantic
grounding in either domain. Empirically (Finding #23): at N=5 this
produced PPL in the trillions (31.6T → 17,683 after dropping the worst
adapter, indicating one adapter's cross-terms dominated). ∎

## Kill Criterion (pre-registered)

**K1548**: At N=5 with explicit Sum(B_i @ A_i) composition, held-out PPL stays
within 2× of solo baseline (no catastrophe).

Formally: let $\text{PPL}_c$ denote the composed model's PPL on a held-out
slice, and $\overline{\text{PPL}_s} = \frac{1}{N}\sum_i \text{PPL}(W_{solo_i})$.
PASS iff $\text{PPL}_c \leq 2 \cdot \overline{\text{PPL}_s}$.

## Prerequisites (NOT satisfied)

Measuring K1548 requires:
1. A fixed base model (target: `mlx-community/gemma-4-e4b-it-4bit`, per
   F#560 and P1 registry).
2. **N=5 trained LoRA adapter weight sets** (rank 6, targeting at least
   `q_proj`), one per domain (math, code, medical, legal, finance per
   `adapters/registry.json`).
3. A held-out text slice over which solo PPL and composed PPL are measured
   using the same tokenization and sequence length.

## Pre-flight audit of item (2): prerequisite FAILURE

Adapter registry (`adapters/registry.json`) lists 5 domain adapters whose
training is claimed by `exp_p1_t2_single_domain_training` (K1028–K1032
marked ✓) and `exp_p1_t2_multi_domain_5` (K1047 PASS: "all 5 adapters ≥+3pp"
per DB evidence).

Filesystem check (2026-04-18):
- `micro/models/exp_p1_t2_single_domain_training/adapters/math/` → only
  `adapter_config.json`. No `adapters.safetensors`.
- `.../code/` → only `adapter_config.json`.
- `.../medical/` → only `adapter_config.json`.
- `micro/models/exp_p1_t2_multi_domain_5/adapters/legal/` → only
  `adapter_config.json`.
- `.../finance/` → only `adapter_config.json`.

All 5 of 5 adapters are WEIGHT-LESS STUBS. This is antipattern-017
(3rd confirmed instance after J0 and M0 this week).

Attempting to load a stub adapter via mlx-lm's `load(..., adapter_path=...)`
either errors on missing weights or silently returns the base model (runs
without any LoRA delta). Either way, K1548 cannot be measured: no composable
$\Delta W_i$ exists.

## Antipattern self-check

- ✗ antipattern-017 (weight-less stub adapters): **TRIGGERED**, 5-of-5.
- ⊘ antipattern-003 (LORA_SCALE=20): N/A (no inference code yet).
- ⊘ antipattern-008 (thinking-mode strip bug): N/A.
- ⊘ antipattern-018 (channel tokens as SFT text): N/A (no training here).
- ⊘ antipattern-020 (cascade-dependent design): not applicable — dependency
  is on a stub artifact, not on a killed upstream. Pure antipattern-017.
- ⊘ KC-swap: MATH.md committed before any run; no post-run KC edits possible.

## Verdict

KILL preemptively. K1548 marked `fail` on unmeasurability grounds. The
experiment's mathematical hypothesis (Theorem 2 norm bound ⇒ correct delta
sum is bounded at N=5) is derivable from first principles and supported by
Finding #14 (PPL trillions → 2.36 under 1/N scaling, which differs from the
unscaled delta-sum only by a constant factor of N). The empirical leg of
the verification is blocked on adapter-weight rebuild.

**Unblock path**: `P11.ADAPTER-REBUILD` — train the 5 math/code/medical/
legal/finance LoRA adapters as actually-fit weight files, then a v2 of
this experiment can be run as a ~15-minute PPL measurement.

## References

- Finding #14 (supported): "1/N scaling resolves composition catastrophe" —
  PPL trillions → 2.36 at N=5.
- Finding #23 (killed): "Equal-weight composition is fragile" — trillion
  PPL at N=5 with the buggy formula.
- Finding #199 (conclusive): "A-matrix loading bug" — loading-time
  manifestation of the same formula error.
- Finding #544 (killed) evidence citation: "`premerge_vs_dynamic.py:415-431`
  A+A/B+B bug". Confirms bug location.
- antipattern-017 (weight-less stub adapters). This is instance #3 in 2
  days (baseline_eval → J0 4-of-4 → M0 2-of-4 → this 5-of-5).
