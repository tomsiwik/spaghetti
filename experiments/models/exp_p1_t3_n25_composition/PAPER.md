# PAPER.md — T3.4: N=25 Domain Composition on Gemma 4 (Grassmannian Stress Test)

## V2 Audit (2026-04-18) — Status: KILLED

**V1 "supported" verdict retroactively invalid.** Two independent structural reasons:

1. **Adapters missing on disk.** All 5 `adapters.safetensors` files referenced
   by V1 Phase 2/3 (math, code, medical in T2.1; legal, finance in T2.6) are
   absent. Only `adapter_config.json` stubs remain. Upstream T2.1
   (exp_p1_t2_single_domain_training) is KILLED 2026-04-18 (metric-swap +
   format-artefact). T2.6 (exp_p1_t2_multi_domain_5) weights lost.
2. **V1 design is tautological routing (`mem-antipattern-002`).** V1's
   Phase 2/3 hardcodes `REAL_ADAPTER_PATHS[domain]` — each adapter is
   loaded exclusively for its matched domain test set. This tests
   single-adapter-on-matched-domain, not composition. Theorem 3 ("exclusive
   routing → zero interference") is proven but never *exercised*: the
   routing function in V1 code is `R(x) = ground_truth_domain(x)`.

This is the **5th precondition-probe kill** this loop. Rule standing.

### V2 Prediction vs. Measurement

| KC | Prediction | Measurement | Result |
|---|---|---|---|
| K1059: max\|cos\| < 1e-5 | ~1e-7 (Theorem 1, float32) | **2.165e-8** (42 layers × 300 pairs) | **PASS** (genuine) |
| K1060: 0/25 degraded | 0/25 (Theorem 3) | **FAIL**: 0/5 adapter .safetensors on disk + V1 design tautological | **FAIL** |
| K1061: MMLU >= base-2pp | all >40% (V1 claim) | **FAIL**: adapters missing + V1 design conflates format transfer with composition | **FAIL** |
| K1062: <1 GB | ~48 MB (V1 claim) | **110.74 MB theoretical** (float32 formula); real size **0 MB** (no weights) | **PASS theoretical, moot** |

Verdict: **KILLED.** `all_pass=false`. No thresholds changed between V1 MATH.md
and V2; KCs are routed honestly based on what is genuinely measurable.

### Permanently learned (class-level standing rules — 5 instances this loop)

1. **Precondition-probe before macro sweep.** Before running any N-domain
   composition / peer-comparison / composed-benchmark sweep, probe the
   filesystem for adapter .safetensors and probe critical imports. Costs
   seconds; saves hours when upstream is broken.
2. **Registry ≠ artefacts + directory-existence corollary.** Adapter
   registry JSON claiming scores is not the same as weights on disk.
   Check `.safetensors` existence. Also check directory existence — some
   adapter dirs don't exist at all (not just empty stubs).
3. **Downstream P1 macros inherit upstream audit flags.** When an upstream
   experiment flips supported → killed (e.g., T2.1 metric-swap audit),
   every dependent macro must re-verify preconditions before any claim of
   "supported" can stand.
4. **`code-bug` tag may be decoy.** When V1 failure mechanism is
   mathematical (e.g. gradient identity, proof-level structural flaw),
   code fixes don't unblock the experiment. The V1 was conceptually
   wrong, not typo-wrong.
5. **NEW — Composition claims require genuine routing.** Hardcoding
   `ADAPTER_PATHS[domain]` in test loops is tautological routing
   (antipattern #2). True composition requires either (a) simultaneous
   N-way activation with accuracy measured per domain, or (b) a real
   router deciding which adapter fires from input features (not domain
   labels). Any "composition supported" claim whose code loads one
   adapter per eval is single-adapter eval mislabeled.

### Unblocker for V3

V3 is **blocked**. Researcher MUST NOT auto-spawn. Required:

- T2.1 rebuild with MedQA USMLE 5-choice (DB KC #1030), max_tokens ≥ 512,
  adapter `.safetensors` persisted, `adapters/code/` directory created.
- T2.6 adapters rebuilt or recovered.
- `run_experiment.py` Phase 2/3 rewritten to exercise genuine composition:
  - Option (a): load all 25 adapters simultaneously, measure per-domain
    accuracy, test Theorem 3 on real hardware.
  - Option (b): implement a real router (e.g. T4.1 TF-IDF + hidden-state
    ridge) and measure mixed-domain test-set accuracy.
- Drop the `REAL_ADAPTER_PATHS[domain]` hardcoded map.

---

## V1 Summary (superseded — kept for provenance)

N=25 domain adapters compose without interference on Gemma 4 E4B when using Grassmannian
(QR-constructed) A-matrices and exclusive routing. All 4 kill criteria PASS.
This replicates Finding #406 (Qwen3-4B) on Gemma 4 with a tighter orthogonality bound (2e-8 vs 1.4e-5),
confirming that the structural guarantee transfers across architectures.

## Prediction vs. Measurement

| Kill Criterion | Prediction | Measurement | Result |
|---|---|---|---|
| K1059: max\|cos\| < 1e-5 | ~1e-7 (Theorem 1, float32 precision) | **2.16e-8** (42 layers × 300 pairs) | **PASS** |
| K1060: 0/25 degraded | 0/25 (Theorem 3, exclusive routing) | **0/25** (5 real + 20 synthetic) | **PASS** |
| K1061: MMLU >= base-2pp (>= 2%) | All >40% (T3.2 baseline) | **56-88%** neutral subjects | **PASS** |
| K1062: < 1 GB total | ~48 MB (Theorem 2) | **48.45 MB** | **PASS** |

## Phase 1: Grassmannian Orthogonality (K1059)

**Construction**: For each of 42 layers, draw W ~ N(0,1)^{2560×150}, run QR in float64,
downcast to float32, extract 25 A-matrices of shape (2560, 6).

| Metric | Value |
|--------|-------|
| Max\|cos\| (global, 42 layers × 300 pairs) | **2.165e-8** |
| Mean\|cos\| across layers | 1.665e-8 |
| Layers tested | 42/42 |
| Threshold | 1e-5 |
| Runtime | 0.8s |

Prediction: ≲ 6e-6 (Theorem 1 float32 bound). Measurement: 2.2e-8 — **3× better than predicted**
(float64 QR followed by float32 downcast achieves near-exact orthogonality).

This **improves on Finding #406** (K981 threshold 1e-4, achieved 1.38e-5 on Qwen3-4B with
float16 computation). Using float64 QR + float32 downcast is the recommended pattern.

## Phase 2: Behavioral Routing Check (K1060)

Exclusive routing: each query routed to its single matching adapter.
20 synthetic domains (B=0) are structurally at base level (cannot degrade).

| Domain | Base (%) | Routed (%) | Degraded? |
|--------|----------|-----------|-----------|
| Math (GSM8K n=25) | 0% | **44%** | No |
| Medical (MedMCQA n=25) | 26% | **36%** | No |
| Legal (MMLU prof_law n=25) | 4% | **64%** | No |
| Finance (MMLU macro n=25) | 4% | **56%** | No |
| Code (MMLU HS CS n=25) | 0%* | **72%** | No |
| Synthetic × 20 (B=0) | base | base | No |

*Code base floored at 0% for CS MCQ proxy; HumanEval pass@1 not re-measured here.

Note: Math shows 44% vs T2.1's 82% (n=50). This is expected variance from n=25 vs n=50 at
different GSM8K subsets (shuffle(seed=42).select(25) gives harder subset). K1060 criterion is
"above base", which is easily met (44% >> 0%).

**K1060 PASS: 0/5 real + 0/20 synthetic = 0/25 degraded.**

## Phase 3: MMLU Neutral Preservation (K1061)

Adapters tested on MMLU subjects not in their training data (geography, world_religions, philosophy).
This confirms adapters don't overfit to domain syntax and preserve general MCQ capability.

| Domain\Subject | Geography | World Religions | Philosophy | Min |
|----------------|-----------|-----------------|------------|-----|
| Math adapter | 72% | 76% | 64% | **64%** |
| Medical adapter | 76% | 84% | 60% | **60%** |
| Legal adapter | 84% | 88% | 56% | **56%** |
| Finance adapter | 84% | 84% | 56% | **56%** |

Floor (base − 2pp): 4% − 2% = **2%**. All results 56-88% >> 2%.

This is a striking finding: **domain-specific adapters (trained on professional_law, medical MCQ, etc.)
give 56-88% on unrelated MMLU subjects.** The adapters teach Gemma 4 MCQ *format compliance*
(output a single letter A/B/C/D), which transfers universally. This explains T3.2's finding that
adapters give 62-77% on neutral MMLU (base = 4%): the MCQ format capability is the primary gain.

**K1061 PASS: 0/12 combinations below floor.**

## Phase 4: Size Calculation (K1062)

| Component | Size |
|-----------|------|
| 5 real adapters (float32, T2.1/T2.6) | 5 × 4.77 MB = 23.85 MB |
| 20 synthetic adapters (bf16 A-only, B=0) | 20 × 1.23 MB = 24.61 MB |
| **Total** | **48.45 MB** |
| Limit | 1024 MB |
| Utilization | **4.7%** |

Theoretical maximum: N_max = 2560/6 = 426 domains × 1.23 MB/domain = 524 MB (still < 1 GB).

**K1062 PASS: 48.45 MB << 1 GB.**

## Connection to T3.1 Failure

T3.1 (exp_p1_t3_pairwise_interference, KILLED) showed math 82→8%, code 66→8% under simultaneous
N=5 activation. This experiment resolves that failure structurally:

| Dimension | T3.1 (KILLED) | T3.4 (PASS) |
|-----------|---------------|-------------|
| Activation | Simultaneous N=5 | Exclusive routing (1 at a time) |
| A-matrices | Random LoRA init (cos=0.17) | QR construction (cos=2.2e-8) |
| Math accuracy | 8% (82% single) | 44% (>>0% base) |
| Interference | O(N) additive | 0 by construction |

The combination of (a) Grassmannian A-matrices + (b) exclusive routing makes N=25 composition
structurally interference-free. Neither alone suffices: T3.3 showed routing is essential; T3.1
showed simultaneous activation is fatal even with low weight-space cosines.

## Runtime

- Total: 425.5s (7.1 min) on M5 Pro 48GB
- Phase 1 (QR, pure numpy): 0.8s
- Phase 2 (5 domain evals): 183.6s
- Phase 3 (12 MMLU neutral evals): 241.1s

## Conclusions

1. **Grassmannian construction achieves 2.2e-8 cosine** — 460× below K1059 threshold, matching
   float32 theoretical precision limit. This is the tightest bound in the P1 experiment series.
2. **N=25 exclusive routing: 0/25 domains degrade.** Structural guarantee proven empirically.
3. **Universal MCQ format transfer**: adapters give 56-88% on ANY MMLU subject — domain adapters
   teach format, not just domain knowledge. This generalizes the T3.2 finding.
4. **N=25 fits in 48 MB** — 22× headroom before 1 GB limit. N=426 theoretical max.

## References

- HRA (arxiv 2405.17484): Householder Reflection Adaptation — structural orthogonality
- Finding #406: N=25 PASS on Qwen3-4B (K981: < 1e-4; this tightens to 2.2e-8)
- Finding #427: Gemma 4 routing load-bearing (T3.3)
- T3.1 (KILLED): Simultaneous N=5 → math 8% (impossibility structure: O(N) interference)
- T3.2 (KILLED): Scale sensitivity; MCQ format: base=4%, adapters=62-77%
