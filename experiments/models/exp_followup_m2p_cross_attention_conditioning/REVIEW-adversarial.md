# REVIEW-adversarial.md — exp_followup_m2p_cross_attention_conditioning

**Reviewer hat** · 2026-04-18 · **Verdict: KILL (confirm)**

## Adversarial checklist

| Item | Check | Result |
|---|---|---|
| (a) | `results.json.verdict == "killed"` vs proposed `killed` | ✓ consistent |
| (b) | `all_pass == false` vs claim | ✓ consistent (K1556a, K1556c fail) |
| (c) | PAPER.md verdict line = "**KILLED**" | ✓ |
| (d) | `is_smoke == false`, full run | ✓ total_time_s=12.7, N_CONTEXT_VARIANTS=20 |
| (e) | MATH.md diff since pre-reg commit `201a762` | ✓ clean (single create commit; `git log` confirms no post-reg edits) |
| (f) | KC tautology sniff | ✓ K1556a/b/c measure CV of `||B||_F` across 20 held-out eval contexts (eval_rng = SEED+999), not trivially satisfied |
| (g) | K-ID in code = K-ID in MATH/DB | ✓ `k1556a_pass = cv_cross > 0.05` lines 644–649 match MATH.md §F and DB KC #1556 |
| (h) | `sum(lora_A…)` / `add_weighted_adapter` / safetensor composition bug | ✓ not applicable (no adapter composition; single-domain M2P generation) |
| (i) | `LORA_SCALE` ≥ 12 | ✓ `ADAPTER_SCALE = 1.0` (line 472), safe |
| (j) | Routing on one sample applied to all | ✓ not applicable (no routing) |
| (k) | `shutil.copy` of sibling adapter | ✓ none |
| (l) | Hardcoded `{"pass": True}` in KC dict | ✓ pass flags computed from measurements |
| (m) | Target model in MATH ≠ loaded model | ✓ toy GPT-2 declared in MATH §D/§H and loaded as `ToyGPT`; apples-to-apples with parent kill by design |
| (m2) | Skill-invocation evidence / idiomatic MLX | ✓ `mx.eval(model.parameters(), optimizer.state, ...)` after updates (364, 522), `nn.value_and_grad(m2p, loss_for_grad)(m2p)` pattern, `mx.clear_cache()` in cleanup (87), `mx.set_memory_limit` + `mx.set_cache_limit` preamble (39–40) |
| (n) | Base acc = 0% + avg_thinking_chars = 0 | ✓ not applicable (no chat-template eval) |
| (o) | Headline n < 15 | ✓ n = 40 (20 easy + 20 hard) across 20 context variants |
| (p) | Synthetic padding | ✓ all 20 eval contexts are real generated arithmetic samples |
| (q) | Baseline drift vs cited | ✓ K1556b measures parent baseline in-run (CV_mean=0.0153 within the 0.0093±noise band documented in PAPER §Assumptions) |
| (r) | Prediction-vs-measurement table in PAPER.md | ✓ present (P1–P5), match column flagged honestly |
| (s) | Math errors / unsupported claims | ✓ Lemma 1 (rank-1 Jacobian under mean-pool) and Lemma 2 (rank-`min(N,T,d_k)` under cross-attn) are standard results and match implementation exactly |

## Verdict

**KILL (confirm).** Three independent KCs were pre-registered; K1556a fails (0.0200 < 0.05), K1556c fails (1.31 < 3×), and P4 also fails (hard/easy ratio 0.971 vs ≥ 1.10 predicted). K1556b and P5 pass, which is the point: they isolate the failure to the treatment rather than to drift. No KC was edited post-hoc.

The PAPER narrows the parent closure rule from `additive-context-injection-blocks-calibration` to `additive-pooled-concat-unpacking-blocks-calibration` — a genuine architectural refinement, not a retrofit. Follow-up seeds (per-token `B_proj` heads; cross-attn → `B_proj` skip bypassing self-attn) are concrete and testable.

## Assumptions logged

- Single seed (`SEED=42`) inherited verbatim from parent kill: accepted per PAPER §Assumptions; K1556b reproducing parent regime is the validity anchor.
- No KC edits confirmed via `git log micro/models/exp_followup_m2p_cross_attention_conditioning/MATH.md` (single commit `201a762`).

## Routing

- `experiment complete` already recorded status=killed on 2026-04-18 (see `experiment get ...` output).
- Emit `review.killed` → Analyst writes LEARNINGS.md.
