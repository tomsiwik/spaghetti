# LEARNINGS.md — exp_method_vs_domain_adapter

**Verdict: PROVISIONAL (smoke). DB closed as `killed` (superseded by v2 design; full rerun blocked on three methodology fixes).**

## Core Finding

A rank-16 `v_proj+o_proj` LoRA trained on subgoal-decomposition teacher
traces across 5 MMLU-Pro categories does **not** transfer the method
signature to held-out categories at smoke scale (`n_train=15, N_STEPS=40,
n_eval=15`). Multi-domain adapter 40.0 % vs base 60.0 %; signature rate
0.0 % vs base 13.3 %. All three pre-registered KCs (K1718, K1719, K1720)
returned **inconclusive** — smoke was under-powered and the pipeline
surfaced three independent methodology bugs before the full run could
be scored.

## Why

1. **Signature detector too strict** (Issue 1). K1720's gate
   `count_subgoal_markers ≥ 2 marker TYPES` is stricter than the teacher's
   natural trace shape. Gemma-4-E4B-it-4bit under the method prompt emits
   only the `Step N:` enumeration — one marker type. Teacher signature
   rate 40 % < 70 % threshold, so the teacher itself fails the gate on its
   own data. This is a pre-registration error, not a result; the KC must
   be redefined before v2 (any **one** of the four rule families counts).

2. **`strip_thinking` regex brittle to close-tag omission** (Issue 2).
   `exp_score_kl_constrained_mcq`'s strip regex expects a closing
   `<|channel>thought` tag. Gemma-4-E4B-it-4bit omits the close tag on a
   subset of outputs; when absent, the full thinking pre-amble (which
   contains internal enumeration) leaks into the scoring string. This
   inflates base signature rate and deflates adapter signature rate,
   producing the paradoxical multi<base pattern.

3. **Over-fit regime at n=15×40 steps** (Issue 3). 15 examples × 40 steps
   with `LORA_SCALE=4.0` on `v_proj+o_proj` is below the plateau budget
   from F#403. Training loss 0.96→0.11 on single, 1.11 mean on multi —
   multi plateau barely reached. Prediction interval (53–65 %) in MATH.md
   is conditional on `N_STEPS=300, n=100/arm`; smoke cannot reject it.

## Implications for Next Experiment

**Do not rerun this design.** Any v2 must pre-register:

1. **Signature definition relaxation** as a v2 KC change (not an in-place
   edit of K1720): treat "subgoal decomposition present" as
   `count_subgoal_markers ≥ 1` across any rule family, matching Gemma-4's
   natural `Step N` form. This is a pre-registration correction; it must
   ship in a new experiment `exp_method_vs_domain_adapter_v2` before
   recomputing signature rate on the cached `data/eval_responses.jsonl`.

2. **`strip_thinking` fortification**: extend the regex to strip from
   `<|channel>thought` to the next blank line OR `Answer:` prefix when
   the close tag is missing. Diagnostic: run 5 base samples and assert
   `avg_thinking_chars > 0` *and* post-strip token count is finite.

3. **Budget**: `N_STEPS=300, n_train=100` per arm; teacher-gen cost
   already paid via cached `data/train_multi.jsonl` and
   `data/train_single.jsonl` — full rerun only pays train + eval.

**Cached artifacts preserved** for offline v2 re-scoring without a
re-run of generation: `data/eval_responses.jsonl` (45 held-out
generations), `data/teacher_stats.json`, both adapters'
`adapters.safetensors`. A v2 with fix (1) can re-score these before
paying a fresh training cost.

**Do not file v2 as a new P≤2 experiment right now** — the current
research budget is committed to draining the existing backlog. Reopen
as a P1 `exp_method_vs_domain_adapter_v2` only when upstream
prompt-erasure and composition results call for a method-adapter
verification (i.e. if Pierre P1's method path is reactivated).

## References

- `PAPER.md` Issues 1–3 and §"Required v2 fixes" — full methodology
  diagnosis.
- `REVIEW-adversarial.md` — adversarial checklist (a)–(s) all pass at
  smoke; no antipattern trigger.
- `exp_score_kl_constrained_mcq` Finding #586 — source of the fragile
  `strip_thinking` regex; fix propagated as antipattern memory
  `mem-antipattern-thinking-strip-close-tag`.
- F#403 (LoRA-scale plateau on Gemma-4 v/o rank-16) — budget lower
  bound for the v2 training spec.
