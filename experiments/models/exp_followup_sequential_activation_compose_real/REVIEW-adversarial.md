# REVIEW-adversarial.md — exp_followup_sequential_activation_compose_real

## Verdict: KILL (K_vacate; DB-status `killed`, semantic PROVISIONAL)

One-line: Thm 1 (q_proj d_in=2560 != d_out=2048) verified structurally; K1563a/b
vacated because parent math adapter `safetensors` is not on disk. Honest vacate,
not a falsification.

## Adversarial checklist

**Consistency**
- (a) `results.json.verdict="PROVISIONAL"`, DB status `killed` (enum limitation — `provisional` not accepted by `experiment complete`). PAPER.md §Verdict makes the mapping explicit. ✓
- (b) `all_pass=false`, not claimed as supported. ✓
- (c) PAPER.md verdict line = `KILLED (K_vacate; Thm 1 structural PASS)`; semantic `PROVISIONAL` surfaced under a clearly labelled section. No `supported` claim. ✓
- (d) `is_smoke=true`; verdict is not `supported`. ✓

**KC integrity**
- (e) `git log -- MATH.md` shows single pre-reg commit `51e506e`; `git diff 51e506e -- MATH.md` empty. No KC drift. ✓
- (f) Thm 1 PASS is not a tautology — it asserts measured adapter shapes against hard-coded model dims (`2560`, `2048`) and requires `d_out != d_in`. K1563a/b are VACATED with `pass:false`, not tautologically PASS. ✓
- (g) K-IDs in `results.json` match MATH.md §KC table. ✓

**Code ↔ math**
- (h) No weight-level composition code (the vacate branch returns before any compose); model-level pipeline is genuine stage1→stage2 generate with distinct adapters, not `sum(lora_A)`. ✓
- (i) No `LORA_SCALE` ≥ 12 hard-coded (adapters are loaded via `mlx_lm.load(adapter_path=...)`, inheriting trained scales). ✓
- (j) No routing. ✓
- (k) No `shutil.copy` of sibling adapters. ✓
- (l) KC pass flags derived from thresholds / VACATE reason; no hard-coded `"pass": True`. ✓
- (m) `MODEL_ID="mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §Tools. ✓
- (m2) Code uses `mx.clear_cache`, `mx.reset_peak_memory`, per-phase `cleanup(model, tok)`, `mlx_lm.generate` — idiomatic MLX patterns (skill evidence). ✓

**Eval integrity**
- N/A — no behavioural eval occurred (vacate branch short-circuits). ✓

**Deliverables**
- (r) PAPER.md §Prediction-vs-Measurement table present. ✓

## Assumptions (review-side)

1. Thm 1 is a statement about `q_proj` of Gemma 4 E4B, not about a specific adapter. Verifying from the personal adapter alone is sufficient because any adapter trained on q_proj of this model must encode `(2560, 2048)` I/O dims. Accepted.
2. The PROVISIONAL semantics + `killed` DB status is the same K_vacate pattern accepted on `exp_followup_hypernetwork_residual`. Consistency with sibling precedent — accepted.

## Non-blocking notes

- Thm 1 is mostly a re-framing of Gemma 4's GQA architecture (n_kv_heads × head_dim ≠ hidden_size on `q_proj`). Value is the *consequence* (composition rule: `d_in = d_out` required for weight-space sequential), which motivates the `o_proj` sibling in PAPER.md §Next-Experiment Seeds.
- Infrastructure blocker (parent math adapter gitignored, not on disk) is now flagged in two experiments. Suggests a repo-level action item to either (a) un-gitignore trained adapter artefacts, or (b) keep a retraining manifest the loop can replay.

## Route

`experiment complete` already ran (status=killed, evidence posted). Add finding, emit `review.killed`.
