# REVIEW-adversarial.md — exp_hedgehog_behavior_adapter_politeness_impl

**Verdict: PROVISIONAL** (smoke-run + structural-KC PASS with target-KC `not_measured`)

This is the canonical PROVISIONAL pattern per reviewer.md line 62:
- `results.json["is_smoke"]` is `true`.
- K1 (structural proxy, K#1821) **PASS** — mean per-layer cos 0.9618 > 0.85, exceeds prediction (0.91 mean).
- K2 (target, K#1822) `heuristic_only` — `not_measured` ≠ FAIL; KILL on target-gate is unjustified.
- K3 (K#1823a/b), K4 (K#1824) explicitly deferred to `_full` follow-on, flagged in `results.blockers` and `phase_d_k3.deferred=true` (not silent scope reduction).

## Adversarial checklist (18 items)

**Consistency:**
- (a) `results.json.verdict="PROVISIONAL"` matches DB target → **OK**.
- (b) `all_pass=false` consistent with PROVISIONAL → **OK**.
- (c) PAPER.md verdict line "PROVISIONAL" matches → **OK**.
- (d) `is_smoke=true` + claim is PROVISIONAL not full-run → **OK** (this is the trigger).

**KC integrity:**
- (e) MATH.md is brand-new in this iteration (untracked in git, inherited from parent verbatim per researcher hand-off). No KC tampering possible — there is no prior version to relax. → **OK**.
- (f) Tautology sniff: K1 cos-sim is paired with K2 (target). K1 measured on held-out (n=8, smoke). No `e=0→0`, no `x==x`, no single-adapter "composition". → **OK**.
- (g) K-IDs in `results.kc` (K1821/K1822/K1823a/K1823b/K1824) map to MATH.md table 1:1 → **OK**.

**Code ↔ math:**
- (h) No `sum(lora_A …)`, no `add_weighted_adapter(combination_type="linear")`, no per-key safetensor sum. Single adapter — composition N/A → **OK**.
- (i) `LORA_SCALE = 6.0` ≤ 8 (F#328/F#330 envelope) → **OK**.
- (j) Routing N/A (single adapter) → **OK**.
- (k) No `shutil.copy` of sibling adapter → **OK**.
- (l) No hardcoded `{"pass": True, ...}` in KC dict; KCs derived from measurements at lines 670-699 → **OK**.
- (m) Target model: MATH.md §0 = `mlx-community/gemma-4-e4b-it-4bit`; `MODEL_ID` constant in `run_experiment.py` line 57 = same. No proxy substitution → **OK**.
- (m2) Skill-attestation present in MATH.md §0 (`/mlx-dev` + `/fast-mlx` cited) AND PAPER.md "Pre-flight" line 60-62 ("`/mlx-dev` — confirmed (mx.eval per step, nn.value_and_grad on student model, mx.clear_cache between batches, AdamW with weight_decay)"). Code is idiomatic MLX: `mx.eval(model.parameters(), optimizer.state, loss)` after each step (line 342), `nn.value_and_grad` (line 317), `mx.clear_cache()` (lines 142, 349, 421), `mx.set_memory_limit` + `mx.set_cache_limit` (lines 50-53). → **OK**.

**Eval integrity:**
- (n) Base accuracy 0% with thinking-mode 0 → N/A (politeness, not reasoning) → **OK**.
- (o) Headline n: smoke uses n=8 held-out × 42 layers — fine for PROVISIONAL; (o) `n<15` STATS_ERROR is a full-run gate, not a smoke gate → **N/A**.
- (p) Synthetic padding: smoke prompt list embedded (40 prompts, real content); no B=0 / random-Gaussian arms → **OK**.
- (q) Cited baseline: F#627 cited but not measured; non-blocking → **N/A**.
- (r) PAPER.md prediction-vs-measurement table present (lines 22-30) → **OK**.
- (s) Math errors / unsupported claims: theorem predicts mean cos > 0.85; measured 0.9618 (exceeds). Claims tightly scoped. → **OK**.
- (t) **Target-gated kill** — N/A. K1 PASS + K2 `heuristic_only` (`not_measured`) does NOT meet F#666 KILL trigger (which requires target FAIL). Verdict is PROVISIONAL, not KILLED, so target-gate is satisfied.
- (u) **Scope-changing fix antipattern** — checked:
  - (a) SFT↔LoRA swap: N/A (this experiment is LoRA throughout); cos-sim training was the pre-registered objective and ran successfully (loss 0.164 → 0.041).
  - (b) max_length silent reduction: SEQLEN=256 for smoke, 512 for full — explicit in results.json line 16, not silently truncated → **OK**.
  - (c) trackio disabled: N/A (no trackio in this experiment).
  - (d) base-model downgrade: N/A — Gemma 4 E4B 4-bit loaded as MATH.md prescribes.
  - (e) KC complexity drop: K3+K4 are *deferred* (flagged in blockers) not *dropped*. This routes to PROVISIONAL with `_full` follow-up, which is the canonical recovery path per reviewer.md line 62.
  - **Cosmetic blocker** observed: `linear_to_lora_layers` shim raised AttributeError; manual `LoRALinear.from_base` fallback succeeded — flagged in `results.blockers`. This is *attach-mechanism* substitution (same LoRA target/rank/scale, different attach helper), which does NOT change what K1-K4 measure. → **OK**.

## Assumptions logged
- F#666 target-gate is satisfied because K2 `heuristic_only` is `not_measured`, not FAIL. Per reviewer.md line 51: "A proxy-FAIL with target-PASS is a finding about the proxy, not a kill. Before emitting `review.killed`, confirm both proxy AND target failed." — they didn't, and we are not emitting `review.killed`.
- The cosmetic shim fallback is an implementation detail; the Phase B training mechanism (per-layer cos-sim distillation on `o_proj` outputs, polite-vs-neutral teacher-student via LoRA scale toggle) executes exactly as the theorem in MATH.md §3 prescribes.
- File `_full` follow-up at P=2 (not P=3) because the `_impl` parent is at P=1; the full-eval follow-up should be claimable in the same drain wave once K2 judge + K3 harness wire-up are scheduled.

## Routing
- `experiment update --status provisional --dir <path>` (workaround per reviewer.md line 76).
- `experiment evidence ... --verdict inconclusive`.
- `experiment finding-add --status provisional` recording the K1-PASS / K2-deferred / K3+K4-deferred state.
- File `exp_hedgehog_behavior_adapter_politeness_full` at P=2 inheriting MATH.md verbatim with K2-real-judge + K3-harness + K4-ablation as the unblock conditions.
- Emit `review.proceed` with payload prefixed `PROVISIONAL:`.
