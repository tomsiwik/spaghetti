# PAPER.md — exp_adapter_fingerprint_uniqueness

## Verdict: KILLED (preempt-structural; F#666-pure standalone, 1st hash-primitive-correctness sub-form)

Not run. Preempt-killed before any code executed because both pre-registered KCs are engineering-primitive-only (hash-collision rate + per-adapter hash-latency) with no behavioral target pair, on a standalone experiment (`depends_on=[]`). Per Finding #666 + guardrail 1007 (TARGET-GATED KILL), KILL on a KC set that tests only hash-primitive properties — without anchoring to Pierre's behavioral fingerprint use (versioning / dedup / cache-key correctness under real routing workflows) — is forbidden.

This is the **~30th F#666-pure standalone preempt-KILL** in the drain window and the **1st hash-primitive-correctness sub-form** within the infrastructure-benchmark super-family (NEW, distinct from wall-clock-latency / cache-staleness / routing-latency / realtime-streaming-latency / MEMENTO-inline-latency).

## Prediction vs. measurement (none)

| KC    | Prediction (researcher's expected PASS)                        | Measurement | Verdict                        |
| ----- | -------------------------------------------------------------- | ----------- | ------------------------------ |
| K1943 | Fingerprint collision rate = 0 at N=1000 adapters              | not run     | untested — F#666-pure forbidden |
| K1944 | Fingerprint computation < 5 ms per adapter                      | not run     | untested — F#666-pure forbidden |

All 4 cells of the K1943 × K1944 truth-table resolve as either tautology (any commodity hash trivially passes both — SHA-256 birthday bound at N=1000 gives P(collision) ≈ 4.3×10⁻⁷²; BLAKE3 throughput on Apple Silicon ~1 GB/s), engineering defect (slow hash = library-selection issue, not research), implementation defect (truncated fingerprint or bad canonicalization), or degenerate (both primitive metrics fail — still no Pierre behavior measured). No behaviorally-anchored cell.

## Why preempt-kill

Let `E = exp_adapter_fingerprint_uniqueness`, KC set `K = {K1943, K1944}`, `depends_on = []`. Both KCs are engineering-primitive — they measure properties of the hash function applied to adapter weights, not properties of Pierre's fingerprint **use** (cache-key correctness under versioning rollover, dedup semantics across rank/seed variants, routing-system integration correctness).

- **K1943** tests the primitive: "does this hash collide?" — trivially no for any decent hash at N=1000 (F#3 anchor: LoRA structural orthogonality makes collision-by-similarity unrealistic).
- **K1944** tests the primitive: "is this hash fast enough?" — library-selection question (pick BLAKE3 / xxHash / SHA-256 by benchmark), not research.

Neither KC is target-gated against a behavioral claim about Pierre. `depends_on=[]` provides no parent to inherit a target from. Per F#666 + guardrail 1007, the KC set is forbidden-solo: KILL is impermissible, SUPPORTED requires a target-metric pair which is absent. QED.

## Decision-table — all cells unidentifiable

| K1943 (collisions > 0) | K1944 (latency > 5 ms) | Interpretation                                              |
| ----- | ----- | ----- |
| FAIL  | FAIL  | Any decent hash passes. Says nothing about Pierre behavior. |
| FAIL  | PASS  | Slow hash. Library-selection issue. Not research.           |
| PASS  | FAIL  | Collisions observed → short hash or bad canonicalization. Implementation defect. |
| PASS  | PASS  | Both metrics fail. Still no Pierre behavior measured.       |

## Prior-art anchors

- **F#3** (conclusive): LoRA orthogonality is structural (cos=0.0002 at d=896, 50× better than theory). Real LoRA adapters occupy a structurally well-separated region of weight-space → collision-by-structural-similarity is not a realistic failure mode for any ≥128-bit hash. The birthday-bound already mathematically guarantees K1943 PASS for commodity hashes; there is nothing to measure.
- **F#6** (conclusive): Hash routing plug-and-play (5.3% displacement at N=20). Behaviorally-anchored hash-based routing at N=20 with quantified displacement is the proper contrast — this experiment's N=1000 standalone fingerprint-collision test is disconnected from the routing behavior F#6 measures. A v2 should anchor against behaviors like F#6, not hash primitives in isolation.
- **F#714 / F#715 / F#753 / F#739 / F#758** (infrastructure-benchmark preempt-KILLs): all 5 prior sub-forms (wall-clock latency / cache-staleness / routing-latency / realtime-streaming-latency / MEMENTO-inline-latency) preempt-killed on the same F#666-pure forbidden-solo rationale. This experiment is the 6th sub-form (hash-primitive-correctness) of the same super-family.

## Sub-axis classification

- **F#666-pure standalone drain-window index**: ~30th running tally (after F#758 28th MEMENTO-cluster inline-streaming, F#759 29th argmax-divergence-bucket first form).
- **Super-family**: infrastructure-benchmark (6th sub-form overall).
- **Sub-form**: hash-primitive-correctness — NEW. Characterizes "the KC set tests properties of a hash primitive (collision-freeness + latency) applied to adapter artifacts, without anchoring to the behavioral use of the fingerprint in a routing/serving workflow." Future fingerprint/hash-primitive pre-regs without behavioral-use pair should preempt-KILL on this anchor.

## Unblock condition

Re-claimable as `exp_adapter_fingerprint_uniqueness_v2` with:

1. **Behavioral target KC paired.** Recommended K1945: "Fingerprint-based cache-key lookup correctly resolves ≥99% of adapter-identity queries under a versioning-rollover workflow (insert → overwrite → rollback → query) across ≥3 Pierre routing scenarios." This converts the experiment from hash-primitive test to Pierre-behavioral test.
2. **Fingerprint serialization specified.** Canonical LoRA A/B flatten-order, float-precision canonicalization, optimizer-state inclusion/exclusion. Without this, K1943 verdict flips on serialization choice (adapters differing only in training-step tail may hash identically or not).
3. **Adapter population specified.** Synthetic-random (trivial no-collision), real-trained diverse (realistic), or adversarial-near-duplicate (worst-case). Verdict depends on population.
4. **Latency threshold anchored to Pierre's serving-path routing budget.** Bare 5 ms is unanchored; v2 should reference a concrete per-request budget.
5. **Or: subsume into Pierre-integrated versioning/dedup experiment** that tests end-to-end correctness of fingerprint-mediated routing, not hash primitives in isolation.

## Assumptions

- F#666 + guardrail 1007 apply to engineering-primitive KCs (not only to classical proxies like cos-sim, routing-accuracy, PPL). The rationale is structural: any KC set that is unidentifiable-as-a-finding regardless of outcome (the 4-cell truth table above contains no behaviorally-anchored cell) is forbidden-solo on a standalone experiment. Prior art F#714/F#715/F#739/F#753/F#758 confirms infrastructure-benchmark KCs are covered.
- Pierre's fingerprint use is assumed to be versioning / dedup / cache-keys per the pre-reg notes; if the intended use is instead "adversarial-robustness-of-adapter-identity-under-active-attack" (a different question), the unblock condition differs — register as a separate experiment.
- "Commodity hash" defaults to SHA-256 or BLAKE3; xxHash64 adequate for typical populations (N ≤ 10⁶ via birthday bound). "Adapter weights" assumed to be the LoRA A/B tensor pair (not base-model weights); full adapter files typically 0.5–50 MB for rank 8–32 on Gemma 4 E4B.

## Related

- LoRA (`arxiv:2106.09685`) — adapter decomposition foundation.
- LoRA orthogonality F#3, hash-routing F#6 — in-project behavioral anchors.
- Consistent hashing / content-addressable storage literature (Git SHA, IPFS CID) — deep prior art on behaviorally-anchored content-hash correctness; v2 should situate against these frameworks, not reinvent collision-rate-benchmarking in isolation.

## Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#666-pure precedent + reviewer.md §5. Recommended next action: close pre-reg, re-register v2 with behavioral fingerprint-use KC + operational serialization/population/threshold specs, or subsume into a Pierre-integrated versioning/dedup experiment.
