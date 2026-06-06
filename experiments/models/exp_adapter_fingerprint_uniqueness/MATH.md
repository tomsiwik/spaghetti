# MATH.md — exp_adapter_fingerprint_uniqueness (PREEMPT-KILL)

## Verdict: PREEMPT-KILL (F#666-pure standalone; hash-primitive-correctness sub-form)

This experiment is preempt-killed before any code runs because the pre-registered KC set is **engineering-primitive-only with no behavioral target pair** on a **standalone** experiment (`depends_on=[]`). Per Finding #666 + guardrail 1007 (TARGET-GATED KILL), KILL on a KC set that tests only engineering-primitive properties of a hash function — without anchoring to a behavioral claim about Pierre's fingerprint use (versioning / dedup / cache-key correctness under real workflows) — is forbidden. PASS does not certify behavioral benefit (any commodity cryptographic hash trivially passes both KCs) and FAIL does not certify behavioral loss (an implementation choice, not a research claim).

This is the **~30th F#666-pure standalone preempt-KILL** in the drain window and the **1st hash-primitive-correctness form** within the infrastructure-benchmark super-family (NEW sub-form, distinct from wall-clock-latency, cache-staleness, routing-latency, realtime-streaming-latency).

## §0 Platform / skills / model pins

Included for completeness — no code is executed.

- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627) would have been used for real adapter weights; **not loaded**.
- Adapter artifacts: the notes specify "adapter weight hash" but pre-reg does not fix the serialization (raw `.safetensors` bytes vs. tensor-wise hashing vs. LoRA A/B concatenation vs. canonical flatten-order). This ambiguity is itself a structural defect — the KC verdict depends on serialization choice.
- LORA_SCALE: N/A.

## §1 Preempt-KILL theorem

**Theorem (engineering-primitive-only KC set on standalone experiment is unidentifiable as a research finding).** Let `E` denote experiment `exp_adapter_fingerprint_uniqueness` with kill criteria K = {K1943, K1944}:

- **K1943**: "Fingerprint collision rate > 0 (at N=1000 adapters)" — engineering correctness of a hash primitive.
- **K1944**: "Fingerprint computation > 5ms per adapter" — engineering latency of a hash primitive.

**Step 1 — Both KCs are engineering-primitive, neither is behavioral.** Per F#666, a "target" KC must measure the behavioral/research outcome of interest. The behavioral outcome for fingerprinting in Pierre is whether fingerprints **correctly disambiguate adapters in real workflows** (versioning rollover, cache-hit correctness, dedup semantics, routing-system integration). K1943 and K1944 measure two properties of the hash function itself:

1. Collision-freeness on a synthetic N=1000 set of adapter weights.
2. Per-adapter computation latency.

Neither KC tests Pierre's fingerprint **use**; both test the **primitive**.

**Step 2 — Both KCs are tautologically satisfied for any commodity hash.** By the birthday bound, a 256-bit cryptographic hash (SHA-256, BLAKE3) has collision probability at N=1000 of `N²/(2·2^256) ≈ 4.3×10⁻⁷²`; even a 64-bit hash (xxHash64) gives `≈ 2.7×10⁻¹³`. BLAKE3 throughput on Apple Silicon is ~1 GB/s single-threaded; a 500 MB Gemma 4 E4B LoRA adapter hashes in ~500 ms, and a typical 10–50 MB adapter hashes in 10–50 ms (still > 5 ms for large adapters — but the threshold collapses the test to "pick hash X vs hash Y," a library-selection question, not a research claim). Adapters in the ~0.5 MB regime (typical LoRA at rank 8–16 on Gemma 4 E4B v_proj + o_proj) hash in < 1 ms, trivially passing K1944.

**Step 3 — Both KCs are forbidden-solo per guardrail 1007.** The KC set has no target-metric pair. `depends_on=[]` provides no parent target to inherit. Pierre's routing/serving codepath that would define the "behavioral" target (e.g. "fingerprint-based cache-hit rate matches adapter-identity ground truth at ≥99% under rollover") is not referenced.

**Step 4 — KC-truth-table analysis.**

| K1943 (collisions > 0) | K1944 (latency > 5 ms) | Behavioral interpretation | F#666 verdict |
| --- | --- | --- | --- |
| FAIL (no collisions) | FAIL (< 5 ms) | Any decent hash passes. Says nothing about Pierre's fingerprint **use**. | **tautological PASS — F#666 forbidden (no target)** |
| FAIL | PASS (≥ 5 ms) | Hash is slow. Library-selection issue — pick a faster hash. Not a research finding. | **engineering defect, not a research claim** |
| PASS (collisions > 0) | FAIL | Collisions observed. Implies (a) hash too short (e.g. 16-bit truncated fingerprint forced), or (b) adapter-serialization canonicalization bug — both are implementation choices, not research claims. | **implementation defect, not a research claim** |
| PASS | PASS | Both engineering metrics fail. Still tells us nothing about Pierre behavior. | **degenerate — no research content** |

No reachable cell produces a behaviorally-anchored finding. All 4 cells resolve as either tautology, engineering defect, or implementation defect — none corresponds to "does fingerprint-based versioning/dedup correctly serve Pierre's adapter routing."

**Step 5 — Threshold unanchored.** The 5-ms latency threshold is unanchored against:

- Pierre's serving codepath: there is no published latency budget for fingerprint computation per adapter in Pierre's routing stack. A 5-ms ceiling could be either trivially met (on small LoRA adapters) or trivially violated (on full 500 MB Gemma 4 adapters) — depending on the serialization choice, which the pre-reg does not fix.
- Prior-art anchor F#753: routing-latency standalone infra-benchmark was preempt-killed (4th drain-window) on the same rationale — engineering thresholds without behavioral target-pair.
- N=1000 collision threshold is unanchored against adapter-space size (Pierre's current adapter population is ~10², not 10³).

∴ K1943 and K1944 are forbidden-solo engineering-primitive KCs on a standalone experiment. KILL impermissible per F#666; SUPPORTED requires a target-metric pair which is absent. **QED.**

### §1.1 F#666 gating (negative)

- K1943 = **engineering-primitive** (hash-collision rate on synthetic adapter set — measures primitive correctness, not behavior).
- K1944 = **engineering-primitive** (per-adapter hash latency — measures primitive throughput, not behavior).
- **No target KC.** KC set is **F#666-noncompliant — forbidden-solo.**

### §1.2 Sub-axis classification — 1st hash-primitive-correctness form within infrastructure-benchmark super-family

This experiment introduces a **NEW sub-form** within the F#666-pure standalone infrastructure-benchmark super-family: **hash-primitive-correctness** (collision-freeness + computation-latency of a hash primitive applied to adapter weights). Distinct from prior sub-forms:

| Infra-benchmark sub-form                | 1st observation(s)                          | Form                                                       |
| --------------------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| Wall-clock latency                      | F#714                                       | per-op wall-clock                                          |
| Cache-staleness                         | F#715 + F#754                               | staleness-rate / invalidation-rate                         |
| Routing-latency benchmark               | F#753                                       | per-query routing wall-clock                               |
| Realtime-streaming latency              | F#739 (MEMENTO-cluster)                     | per-block inference wall-clock                             |
| MEMENTO-streaming inline latency        | F#758 (MEMENTO-cluster)                     | per-block inline latency + accuracy parity                 |
| **Hash-primitive correctness**          | **(this) — collision-rate + hash-latency**  | **collision-freeness + per-adapter hash-computation time** |

Pre-existing partial coverage:
- **F#3** (conclusive): "LoRA orthogonality is structural (cos=0.0002 at d=896, 50× better than theory)" — implies LoRA adapters occupy a structurally well-separated region of weight-space, so **collision-by-structural-similarity** is not a realistic failure mode; the only collision mode is hash-space birthday-bound, which is astronomically small for any ≥128-bit hash.
- **F#6** (conclusive): "Hash routing plug-and-play (5.3% displacement at N=20)" — anchors that **behavioral** hash-based routing at N=20 already has quantified displacement. A N=1000 fingerprint-collision test disconnected from routing behavior is a regression in rigor vs. F#6's behaviorally-anchored hash-routing measurement.
- **F#714/F#715/F#753/F#739/F#758** (preempt-KILLED): prior infrastructure-benchmark standalone preempt-KILL precedents on the same F#666 rationale.

## §2 Prior art (preempt rationale)

- **F#666** (defining): target-gated KC discipline — engineering-primitive-only KC set forbidden-solo.
- **F#714** (1st infrastructure-benchmark bucket): wall-clock latency standalone.
- **F#715** (cache-staleness).
- **F#753** (routing-latency, 4th drain-window infra-bench preempt-KILL).
- **F#739 / F#758** (MEMENTO-cluster streaming-latency + accuracy-parity preempt-KILLs under PROVISIONAL parent — canonicalized at 3 forms per F#758).
- **F#3** (conclusive): LoRA structural orthogonality — collision-by-structural-similarity is not a realistic failure mode for reasonable hash lengths.
- **F#6** (conclusive): behaviorally-anchored hash-based routing measurement at N=20 — contrast: this experiment's N=1000 fingerprint test is disconnected from the routing behavior F#6 measures.
- **F#759** (29th drain-window, immediate prior preempt-KILL): argmax-divergence proxy-bucket NEW form — same F#666-pure standalone pattern applied to structural-hyperparameter ablation.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                        | Kind                  | Measurement status                      |
| ----- | ------------------------------------------------------------ | --------------------- | --------------------------------------- |
| K1943 | Fingerprint collision rate > 0 (at N=1000 adapters)          | engineering-primitive | untested (preempt-blocked, F#666-pure)  |
| K1944 | Fingerprint computation > 5 ms per adapter                    | engineering-primitive | untested (preempt-blocked, F#666-pure)  |

Both KCs are F#666-noncompliant (engineering-primitive-only). Even if measured, their joint truth-table (§1 Step 4) contains zero behaviorally-anchored cells — all 4 resolve as tautology / engineering defect / implementation defect.

## §4 Unblock condition

Re-claimable when:

1. **Pair K1943/K1944 with a behavioral target KC.** Recommended: K1945 (target) "Fingerprint-based cache-key lookup correctly resolves ≥99% of adapter-identity queries under a versioning-rollover workflow (insert → overwrite → rollback → query) across ≥3 Pierre routing scenarios." This converts the experiment from "primitive-correctness test" to "behavioral correctness of Pierre's fingerprint use."
2. **Specify fingerprint serialization operationally**: (a) canonical flatten-order of LoRA A/B tensors, (b) include or exclude optimizer state, (c) float-precision canonicalization (e.g. round to fp16 to absorb numerical noise, or strict fp32). Without this, K1943 verdict flips depending on choice (e.g. two adapters differing only in training-step tail may hash differently or identically).
3. **Define the N=1000 adapter population operationally**: synthetic random init (trivial no-collision), real trained adapters across diverse tasks/seeds/ranks (realistic), or adversarial near-duplicate pairs (worst-case). The pre-reg does not fix this; verdict depends on population.
4. **Anchor latency threshold against Pierre's serving budget.** The 5-ms ceiling must reference a concrete serving-path budget (e.g. "p99 fingerprint computation must stay within X% of Pierre's per-request routing budget").
5. **Or: subsume into a behaviorally-anchored Pierre versioning/dedup experiment** that tests end-to-end correctness of fingerprint-mediated routing, rather than hash-primitive properties in isolation.

Re-register as `exp_adapter_fingerprint_uniqueness_v2` with the above corrections, or fold into a Pierre-integrated versioning/dedup experiment.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#666-pure precedent + reviewer.md §5. Recommended next action is target-paired re-register per §4, or subsume into a Pierre-integrated behavioral experiment.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:

- Substituting a Pierre-internal fingerprint use (e.g. cache-hit rate) for K1943/K1944 post-hoc — would convert the experiment from "primitive test" to "behavioral test," which is precisely the unblock condition, and would constitute KC-after-data.
- Picking a specific hash (SHA-256 vs BLAKE3 vs xxHash) and reporting results against a narrow threshold — library-selection is not research.
- Using a synthetic-random adapter population to trivially pass K1943 while claiming the result generalizes to real adapter populations — F#3 + F#6 imply real adapters are structurally well-separated, but the pre-reg does not specify real vs synthetic.
- Reusing F#6's hash-routing displacement measurement as a stand-in for K1943 — F#6 measures behavioral routing at N=20, not primitive-collision at N=1000; cross-experiment proxy substitution antipattern-m.

All four shortcuts would either substitute the behavioral target in post-hoc (antipattern KC-after-data) or present engineering thresholds as research findings.

## §7 Anti-pattern scan

- Composition-math: N/A (no composition).
- LORA_SCALE: N/A (no code).
- shutil.copy: N/A (no code).
- Hardcoded `"pass": True`: N/A (no code; `all_pass: false` written to results.json).
- Eval truncation producing base=0%: N/A (no eval).
- Proxy-model substitution: N/A (no code; would have used Gemma 4 E4B per F#627 if runnable).
- KC measures wrong object: K1943/K1944 measure hash-primitive properties, NOT Pierre's behavioral fingerprint use — this IS the F#666-pure preempt rationale, not an antipattern in the scaffold.
- N=smoke reported as full: N/A (no N; `is_smoke: false`).
- Tautological routing: N/A (no routing).
- Thinking-mode truncation: N/A (no eval).
- File-existence cache: N/A (no code).
- KC-after-data: scaffold pre-registers preempt verdict before any data; no risk.
- Copy-paste scaffolding: scaffold derived from F#759 (`exp_g4_lora_rank_importance_per_task` — closest sibling F#666-pure standalone; immediate prior preempt-KILL) but variant-specific sections (hash-primitive-correctness NEW form, infra-benchmark super-family sub-form enumeration, F#3/F#6 structural-orthogonality and hash-routing anchors) rewritten, not copy-pasted.
