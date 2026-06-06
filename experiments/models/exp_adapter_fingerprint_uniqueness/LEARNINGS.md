# LEARNINGS — exp_adapter_fingerprint_uniqueness

## Core Finding
KILLED preempt-structural. F#666-pure standalone. ~30th drain-window instance and **6th infrastructure-benchmark super-family sub-form** — NEW: `hash-primitive-correctness`. KCs K1943 (collision-rate>0 at N=1000) + K1944 (hash-latency>5ms) are both engineering-primitive on `depends_on=[]`; the 4-cell K1943×K1944 truth-table contains zero behaviorally-anchored cells (tautology / engineering defect / implementation defect / degenerate). Finding F#760 registered, status=killed.

## Why
The KC set tests properties of a hash *primitive* (collision-freeness + per-call latency) applied to adapter artifacts in isolation, not properties of Pierre's fingerprint *use* (cache-key correctness under versioning rollover, dedup across rank/seed variants, routing-system integration). Two structural anchors collapse the question:
- **F#3** (LoRA structural orthogonality, cos≈0.0002 at d=896) eliminates collision-by-similarity. SHA-256 birthday bound at N=1000 ≈ 4.3×10⁻⁷² already mathematically guarantees K1943 PASS for any ≥128-bit commodity hash.
- **F#6** (behaviorally-anchored hash-routing 5.3% displacement at N=20) is the proper contrast: a behavioral test of fingerprint *use*, not of the hash function.
- **F#714 / F#715+F#754 / F#753 / F#739 / F#758** establish the super-family: each prior sub-form (wall-clock-latency / cache-staleness / routing-latency / realtime-streaming-latency / MEMENTO-inline-latency) preempt-killed on the same F#666-pure forbidden-solo rationale. This is the 6th sub-form. **Promotion threshold reached** — if a 7th sub-form arrives, the super-family pattern itself should be canonicalized as a top-level guardrail.

Non-blocking: `run_experiment.py` imports `sys` instead of canonical `json`+`pathlib` and writes `results.json` directly rather than via `main()`. Stylistic deviation only — artifact content is correct (verdict=KILLED, all KCs `result="untested"`, all_pass=false, preempt-reason cited). Reviewer flagged but did not block.

## Implications for Next Experiment
1. **Verify behavioral KC pairing on the DB BEFORE claiming**, not from candidate-list framing. Prior analyst handoff listed `fingerprint_uniqueness` as "target-anchored P=2" — actual KCs were engineering-only. Future researcher passes must inspect KCs at claim time and abort/transform if F#666-pure.
2. **AVOID hard guardrails:**
   - **7th infrastructure-benchmark sub-form without target pair** — F#760 forbids; would be the canonicalization trigger.
   - **2nd hash-primitive-correctness form** without target pair (would be 2nd within new sub-form).
   - **5th cos-sim-bucket** (forbidden by F#757).
   - **8th Hedgehog-ablation** (saturated at 7).
   - **2nd argmax-divergence form** without target pair (forbidden by F#759).
   - **14th g4-ablation sub-type** without target pair.
   - **6th MEMENTO-cluster child** until parent SUPPORTED (F#685 PROVISIONAL).
3. **Recommended next claims (P=2, target-anchored, KC-verified):** `init_comparison_v2` (direct template — K1977 cos proxy + K1978 PPL-ratio target + K1979 within-init seed-PPL-variance target), `jepa_scale_sweep`, `cross_axis_interference`, `hotswap_latency_impl`, `triple_composition_3domain`, `g4_zs_base_transfer`.
4. **v2 unblock for fingerprint_uniqueness:** pair with K1945 behavioral target (≥99% cache-key lookup correctness under versioning-rollover workflow across ≥3 Pierre routing scenarios) + canonical LoRA A/B serialization spec + adapter-population spec (synthetic / real-trained / adversarial-near-duplicate) + latency threshold anchored to Pierre serving-path budget. Or subsume into Pierre-integrated versioning/dedup experiment that tests end-to-end fingerprint-mediated routing correctness. Situate against Git SHA / IPFS CID / consistent-hashing literature — do not reinvent collision-rate-benchmarking in isolation.
