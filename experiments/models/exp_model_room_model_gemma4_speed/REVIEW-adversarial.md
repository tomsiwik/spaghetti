# REVIEW-adversarial — exp_model_room_model_gemma4_speed

**Verdict: KILL (confirm).** Clean, quantified falsifying replication on target platform. No blocking issues. Pre-registered predictions (all three KCs) matched measured outcomes exactly.

## Adversarial checklist

**Consistency (a-d):**
- (a) `results.json["verdict"] = "KILLED"`; DB status = `killed`. ✓
- (b) `all_pass = false`; status `killed` is consistent. ✓
- (c) PAPER.md §1 verdict line = **KILLED** (no PROVISIONAL/PARTIALLY SUPPORTED). ✓
- (d) `is_smoke: false`; N=5 is the target regime, not miniature. ✓

**KC integrity (e-g):**
- (e) KCs 1688/1689/1690 in MATH.md §5 match DB (`experiment get`); no post-run addition/relaxation. MATH.md is fresh (untracked). ✓
- (f) **Tautology sniff:** KC1689 compares `cos(W_room logits, single-adapter routing logits)` — different forward passes, not identity. KC1690 compares `(W_full − ΔW_k)` vs a freshly-summed-without-k result in *different left-to-right order* — bf16 associativity is non-trivial, the PASS is a real claim. KC1688 is a wall-clock measurement. ✓
- (g) K-IDs in code measure the quantities described in MATH.md. ✓

**Code ↔ math (h-m2):**
- (h) `compute_wcombined` (run_experiment.py:100-107): forms `B_i @ A_i` first, then sums. No `sum(lora_A)` / `sum(lora_B)` independent sum bug (mem-antipattern-001). ✓
- (i) `ALPHA = 1.0` (run_experiment.py:30). No LORA_SCALE=20 inflation. ✓
- (j) Routing: ground-truth domain→k map (run_experiment.py:306), per-prompt adapter selection. No `val[d][0]` tautology. ✓
- (k) No `shutil.copy`. ✓
- (l) No hardcoded `{"pass": True}` dict. Pass flags derive from measurements (run_experiment.py:335, 356, 363). ✓
- (m) MATH.md target: `mlx-community/gemma-4-e4b-it-4bit`. Code `MODEL_ID` (run_experiment.py:26): same. ✓
- (m2) **Skill evidence:** docstring cites "mlx-dev pattern"; idiomatic MLX — `mx.eval` at phase boundaries, `mx.clear_cache()` between phases, `stream_generate` from `mlx_lm`, `mx.linalg.qr(..., stream=mx.cpu)` for CPU-only op, proper bf16 dtype handling. ✓

**Eval integrity (n-q):**
- (n) Not applicable — no thinking-mode generation scored.
- (o) KC1689 n=5 domains/prompts — below 15, but this is a **mechanism test of a mathematical identity**, not a behavioral claim. Pre-registered as such in MATH.md §4. The kill is structural (Zhong et al. bound). Non-blocking.
- (p) Adapters are random-init (Grassmannian A + σ=0.02 B). MATH.md §4 argues this is a **conservative estimator** — trained adapters would widen cross-terms further. PAPER.md §4 acknowledges this explicitly. Non-blocking; strengthens the kill if anything.
- (q) No cited-but-not-measured baseline; base is measured live (86.62 tok/s). ✓

**Deliverables (r-s):**
- (r) PAPER.md §1 has pre-registered predictions vs measurements table with Pass column. ✓
- (s) Math is sound: Thm 1 (linearity, not tested as a KC but holds), Thm 2 (LN cross-terms → FAIL KC1689), Thm 3 (bf16 associativity → PASS KC1690). Consistent with results.

## Assumptions logged

Random-init B at σ=0.02 as proxy for trained ΔW scale — researcher's justification (cross-term magnitude driven by ‖ΔW‖²) is sound; this makes the 0.994 cosine measurement a *conservative* floor for divergence. Acceptable.

## Routing signal

Fourth independent kill of Room Model pre-summing at N>1 (prior: Findings #302, #303, #315 on Qwen3). Extends to Gemma 4 E4B + M5 Pro + 4-bit + PoLAR r=6 + MoE. Bounded by Zhong et al. 2504.10957 (ICLR 2025). **Room Model is permanently closed for N>1 composition on the target platform.** KC1690 PASS is the only reusable fragment — supports N=1 hot-merge only (already used in v6 Pierre).

Downstream blocked experiments — `exp_model_pre_registration_n100_macro`, `exp_model_multi_seed_room_model` — inherit a dead premise; reviewer recommends analyst flag for re-examination.

`project_room_model.md` memory should be annotated: *superseded for N>1; N=1 hot-merge only.*
