# exp_composition_runtime_vs_merge_n10 — MATH

## Verdict (pre-run): PREEMPT-KILL — method-dependent redundancy (2nd instance, PROMOTION) + F#399 BW-bound + F#666-pure

## 0. Motivation

Claim (notes): "At N=10, merge should be faster but may lose quality. Benchmarks the serving tradeoff."

Status of both sides of that claim in the current finding set:
- F#399 SUPPORTED: *runtime LoRA is BW-bound optimal for practical ranks*; pre-merge speedup ≤ 1.013× at r=8 (Qwen3-0.6B); 1.5× speedup requires rank ≥ 380 (structural impossibility for rank≤64 regime).
- F#66  SUPPORTED: bf16 merge = 16.7 tok/s vs runtime 12.0 tok/s (39% faster) but loses ~50% of adapter delta due to bfloat16 ULP; fp32 merge K3 KILL.
- F#406 SUPPORTED: Grassmannian-routing N=25@4B quality_ratio=1.3125 (subsumes N≤25 runtime).
- F#54  SUPPORTED: N=24 real-data composition works under Grassmannian.
- F#510 / F#511 SUPPORTED: standard-LoRA pre-merge destroys benchmarks (GSM8K 0 vs 73; HumanEval 0 vs 63).
- F#543 KILLED: uniform additive merge fails at N=5 (2.57× PPL bloat).

So both `latency` (K1894) and `quality` (K1895) questions are already decided at N=10 under every plausible composition×precision branch — the experiment as filed cannot produce a new finding.

## 1. Three independent theorems

### Theorem 1 (F#399 BW-bound structural impossibility — K1894 preempt)

**Claim.** K1894 ("Runtime compose N=10 latency > 2× merge latency") is derivable FAIL from F#399's closed-form bandwidth bound.

**Proof.**
- F#399's impossibility-structure: for rank r where r/d_model < 37%, merge speedup ≤ 1 + r·(d_model/15)/Base_BW.
- K1894 requires merge ≥ 2× faster than runtime, equivalently merge speedup ≥ 2.0.
- 2.0× requires LoRA_BW ≥ Base_BW, i.e. rank ≥ d_model at minimum, which for Gemma-class d_model∈{2048, 2304, 3584, 4608} means rank ≥ 2048 — outside the practical regime (this repo uses rank ≤ 64).
- At N=10 adapters the *runtime* per-token BW overhead adds N·LoRA_BW, but N=10·r=8 = 80 effective rank-units ≪ 380. So merge can shave only ≈ 80·(d_model/15)/Base_BW of runtime, which even at d_model=4608 yields a speedup ceiling < 1.1×.
- Therefore K1894 FAILs (runtime is ≤ 10% slower, never ≥ 2×). Derivable without running. ∎

### Theorem 2 (Method-dependent redundancy — K1895 preempt, F#731 2nd instance)

**Claim.** K1895 ("Merge N=10 quality > 5pp worse than runtime") is determinable without running, because *every* plausible composition × precision branch is already covered by a SUPPORTED/KILLED finding.

**Proof by branch enumeration.**

| Composition × precision | Prior finding | K1895 outcome |
|---|---|---|
| bf16 merge (any composition) | F#66 — 50% delta loss vs runtime | K1895 TRUE (quality ≥ 5pp worse) |
| fp32 merge (any composition) | F#66 K3 KILL; F#399 K953 PASS on random adapter (0pp) | K1895 FALSE on fp32 merge, but fp32 merge defeats the latency motivation (no BW win) |
| int4/int8 merge (standard LoRA) | F#510 / F#511 — standard pre-merge destroys benchmarks (GSM8K 0 vs 73) | K1895 TRUE trivially |
| Grassmannian runtime (N=10 ≤ 25) | F#406 + F#54 — SUPPORTED at higher N | K1895 inapplicable (runtime is reference) |
| Uniform additive merge | F#543 — 2.57× PPL bloat at N=5 | K1895 TRUE by monotonicity at N=10 |

Every branch's K1895 outcome is determined by an existing finding. The experiment as filed does not specify which branch it intends to run; under *any* plausible fill-in, the answer is pre-published. ∎

### Theorem 3 (F#666-pure corroboration — canonical guardrail 1007)

**Claim.** K1895 is F#666-pure canonical guardrail 1007 (no target-metric binding). K1894 is additionally an infrastructure-benchmark proxy (per F#715 bucket).

**Proof.**
- K1895 text: "Merge N=10 quality > 5pp worse than runtime compose N=10". "Quality" has no dataset specified, no evaluator specified, no task specified. Guardrail 1007 lists "classification accuracy, routing match rate, PPL, cosine, clustering purity" — an unbound "quality" drop is a supercategory of these. |Target|=0.
- K1894 is pure latency ratio (infrastructure). Per F#715 (exp_memento_kv_serialization_format, 11th F#666-pure, 1st infrastructure-benchmark bucket), latency-ratio KCs without behavioral binding are F#666-pure. K1894 is the 2nd infrastructure-benchmark bucket instance.
- Therefore neither KC alone justifies a target-gated KILL under PLAN.md §1 (Finding #666). Kill justification is structural (Thm 1) and redundancy (Thm 2), not guardrail-compliance. ∎

## 2. Combined consequence

- Thm 1 preempt-kills K1894 (BW-bound structural).
- Thm 2 preempt-kills K1895 (every method branch covered; F#731 2nd instance PROMOTION trigger).
- Thm 3 confirms both KCs are F#666-pure under PLAN.md §1.

Verdict: KILLED (preempt-structural). Running the experiment cannot produce a finding not already published.

## 3. Kill criteria (as filed)

- K1894: Runtime compose N=10 latency > 2× merge latency. **Result under Thm 1: derivable FAIL (structural BW bound, runtime ≤ 1.1× slower).**
- K1895: Merge N=10 quality > 5pp worse than runtime compose N=10. **Result under Thm 2: derivable per branch; all branches covered by prior findings. Target-metric binding missing → F#666-pure (Thm 3).**

## 4. Kill criteria (effective, preempt-KILL)

- **K_bw_bound (Thm 1):** K1894 FAIL by F#399's closed-form BW theorem for practical ranks (r·N ≤ 640 ≪ d_model).
- **K_method_redundancy (Thm 2):** K1895 outcome determinable from F#66, F#510, F#511, F#543, F#406, F#54 across all plausible branches. 2nd drain-window instance of method-dependent redundancy → PROMOTION trigger (F#731 was 1st).
- **K_666pure (Thm 3):** K1895 lacks dataset/evaluator binding; K1894 is infrastructure-benchmark proxy (F#715 bucket 2nd instance).

All three fire. Verdict: KILLED (preempt-structural, method-dependent redundancy PROMOTION).

## 5. Antipattern audit

- Composition math: N/A (no run).
- LORA_SCALE: N/A.
- shutil.copy: N/A.
- Hardcoded `"pass": True`: N/A.
- Eval truncation: N/A.
- Proxy model substitution: N/A.
- **F#666-pure canonical-1007 guardrail: FIRES (K1895 unbound quality; K1894 infrastructure).**
- **Method-dependent redundancy: FIRES (2nd drain-window instance — PROMOTION).**
- **F#399 BW-bound structural: FIRES (K1894 derivable FAIL).**
- **Infrastructure-benchmark bucket (F#715): FIRES (K1894 latency ratio, 2nd instance).**

## 6. Platform

- platform: local-apple (M5 Pro 48GB per PLAN.md Part 2)
- dir: `micro/models/exp_composition_runtime_vs_merge_n10/`
- No code execution required for preempt-KILL (stub `run_experiment.py` writes results.json no-op).

## 7. Predictions (pre-run)

| KC | Expected | Theorem | Rationale |
|---|---|---|---|
| K1894 | FAIL (runtime ≤ 1.1× slower, never ≥ 2×) | Thm 1 | F#399 BW-bound, N·r=80 ≪ d_model |
| K1895 | branch-dependent; every branch covered | Thm 2 | F#66 / F#510 / F#511 / F#543 / F#406 / F#54 |

## 8. Handoff

- File as KILLED (preempt-structural).
- References: F#399, F#66, F#406, F#54, F#510, F#511, F#543 (all supporting preempt reasoning).
- No `_impl` follow-up filed (preempt-structural KILL per precedent).
- **F#731 2nd instance of method-dependent redundancy.** Per scratchpad watchlist, this triggers standalone memory promotion. Analyst should file a pattern memory: `method-dependent redundancy = distinct preempt-structural sub-pattern (not F#669, not F#702, not F#666-pure)`.
