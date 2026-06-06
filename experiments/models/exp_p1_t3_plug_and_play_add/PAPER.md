# PAPER.md — T3.6: Hot-Add Adapter Without Retraining

## Status: KILLED (V2 audit, 2026-04-18)

V1 "supported" (2026-04-17) retroactively invalidated by audit-2026-04-17-rerun.
Two independent failure modes, either sufficient on its own:

1. **Tautological routing** (mem-antipattern-002). V1 `REAL_ADAPTER_PATHS[domain]`
   hardcoded the adapter-to-domain pairing. K1067 "bit-exact existing outputs"
   is trivially true because the new adapter is never applied to existing-domain
   queries — the harness, not the mathematics, produced the match.
2. **Upstream preconditions absent**. 0 / 5 expected `.safetensors` present
   (T2.1 KILLED 2026-04-18 with metric-swap + format-artefact; T2.6 weights lost).
   Upstream dependency exp_p1_t3_pairwise_interference also KILLED (K1050 FAIL,
   max|cos|=0.1705).

### V2 Prediction vs Measurement (precondition probe)

| KC | V2 Prediction | V2 Measurement | Result |
|----|---------------|----------------|--------|
| K1067 existing outputs unchanged | Unmeasurable: requires non-tautological router + loadable adapters | 0 routers present; 0/5 safetensors | **FAIL** |
| K1068 new adapter functional | Unmeasurable: requires geography or finance safetensors on disk | Both absent | **FAIL** |
| K1069 hot-add latency < 100ms | Moot: dict update trivially O(1), but no weight load to time | dict update ≈ 8e-5 ms (no weights) | **FAIL** (moot) |

### Permanently learned (class-level standing rules, 6 instances this audit pass)

1. **Precondition-probe before macro sweep.** 6 probe-class kills in 24 h — peer_
   comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed,
   sft_residual_gemma4, n25_composition, plug_and_play_add.
2. **Registry ≠ artefacts + directory-existence corollary.** A dir with only
   `adapter_config.json` is not a ready adapter. A dir that does not exist at
   all is a stronger miss (e.g. `adapters/code/` in some siblings).
3. **Downstream P1 macros inherit upstream audit flags.** T2.1 metric-swap and
   format-artefact propagate to every comparison / composition / plug-and-play
   built on its adapters — 6 kills so far, no exceptions.
4. **`code-bug` tag may be a decoy when V1 failure is mathematical.** In
   sft_residual_gemma4 the V1 failure was gradient-identity (∂L/∂ΔB = ∂L/∂B_applied),
   a property of gradient descent; the tag suggested a coding defect. Triaging
   V1 mechanism first prevents a wasted code-fix loop.
5. **Composition claims require genuine routing or simultaneous activation.**
   `REAL_ADAPTER_PATHS[domain] -> path` is the tautological-routing fingerprint.
   Applies here (K1067) exactly as it did for n25_composition (K1060/K1061).
6. **Hot-add / hot-remove claims require a distinction between router update
   and weight activation.** V1 timed only the dict mutation (0.004 ms), which is
   guaranteed O(1) by Python dict semantics — it says nothing about the
   adapter-load I/O Theorem 3 is actually about. Any future V3 must time the
   weight read, not the registry update.

## Summary

Verified that adding a new domain adapter to an N-adapter registry using exclusive routing:
1. Does NOT change existing domain outputs (bit-identical before/after)
2. New adapter is immediately functional (90% vs 4% base on geography)
3. Hot-add latency is 0.004ms — 23,000× below the 100ms threshold

## Prediction vs Measurement

| Kill Criterion | Theorem | Predicted | Measured | Result |
|----------------|---------|-----------|----------|--------|
| K1067: existing outputs unchanged | Theorem 1: exclusive routing invariance | max\|diff\| = 0.0 (bit-exact) | max_token_diffs = 0 for all 4 domains × 10 queries | **PASS** |
| K1068: new adapter immediately functional | Theorem 2: immediate functionality | acc > base (4%) on new domain | geography = 90.0% vs base = 4.0% | **PASS** |
| K1069: hot-add latency < 100ms | Theorem 3: I/O bound | ~1ms (NVMe I/O bound) | 0.004ms (registry update + path check) | **PASS** |

## Detailed Results

### Phase 1 → Phase 3: K1067 (Existing Outputs Unchanged)

| Domain | n | Pre-Accuracy | Post-Accuracy | Token Diffs |
|--------|---|-------------|--------------|-------------|
| Math | 10 | 0.0% | 0.0% | 0 |
| Medical | 10 | 50.0% | 50.0% | 0 |
| Legal | 10 | 40.0% | 40.0% | 0 |
| Finance | 10 | 80.0% | 80.0% | 0 |

All 40/40 outputs were bit-identical before and after hot-adding the geography adapter.
This confirms Theorem 1: exclusive routing means W_eff(q, domain=i) = W_base + A_i B_i
is independent of the registry size.

### Phase 4: K1068 (New Adapter Functional)

- Geography adapter accuracy: **90.0%** (n=10, high_school_geography MCQ)
- Base accuracy: 4.0%
- Improvement: +86pp

Note: geography adapter is a copy of the finance adapter (synthetic domain). The 90% accuracy
reflects the MCQ format compliance established in T3.2 (Finding #426) — any adapter enables
format compliance universally on neutral MMLU subjects.

### Phase 5: K1069 (Hot-Add Latency)

| Operation | Mean | P99 |
|-----------|------|-----|
| Registry dict update | 0.000285ms | 0.002503ms |
| File path existence check | 0.003995ms | N/A |
| **Total hot-add** | **0.0043ms** | — |

Threshold: 100ms. Actual: 0.0043ms. **23,000× below threshold.**

The hot-add is a pure dict update — O(1) by Python dict semantics. The latency is
dominated by os.path.exists(), not model computation.

## Total Runtime

114.3 seconds (1.9 minutes) for 40 pre-add + 40 post-add + 10 new domain evaluations.

## Structural Significance

This experiment closes the "hot-add is safe" claim. Combined with T3.1 (Finding #425),
which showed that SIMULTANEOUS activation of N adapters is catastrophic (math 82→8%,
code 66→8%), this establishes the key architectural constraint:

**Exclusive routing is load-bearing, not optional.**

- Exclusive routing → hot-add is free (Theorem 1, verified here)
- Simultaneous activation → O(N) interference (T3.1, Finding #425)
- Therefore: PLE-M2P routing is the only viable composition strategy for the Room Model

## References

- Finding #428: T3.4 N=25 Grassmannian composition verified
- Finding #425: T3.1 simultaneous activation causes catastrophic interference (KILLED)
- Finding #426: T3.2 scale≥12 degrades MMLU (KILLED — scale=6 is safe)
- Finding #421: T2.1 LoRA r=6 achieves 22-82pp domain improvement
- HRA (arxiv 2405.17484): Orthogonal adapter construction
