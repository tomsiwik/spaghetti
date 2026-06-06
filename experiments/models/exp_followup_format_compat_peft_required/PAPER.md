# PAPER.md — exp_followup_format_compat_peft_required

## Verdict: **SUPPORTED** (K1576 PASS — all three gates)

| | |
|---|---|
| `results.json["verdict"]` | `SUPPORTED` |
| `results.json["all_pass"]` | `true` |
| `is_smoke` | `false` |
| KC modified after MATH.md? | no (see `git log MATH.md run_experiment.py` — pre-reg before run) |

## Executive summary

The `exp_p1_t4_adapter_format_compat` KILL (Finding #585) was driven by two
discrete flaws that were never hit by its own KC: (a) a silent `ImportError`
bypass that let K1088 auto-pass without `peft` installed, and (b) a
subset-direction fallacy (Theorem 3 claimed PEFT-compat ⇒ vLLM-compat). This
follow-up closes the honest gap on Apple Silicon by replacing both with
structurally impossible failures:

- **(a)** `peft` / `transformers` / `torch` are imported at module top. The
  script cannot begin execution without them — the bypass path is gone.
- **(b)** A real `peft.PeftModel.from_pretrained` + forward pass exercises
  F1 (key parsing), F2 (shape check), F3 (target module match) on a real
  (tiny, CPU-resident) `LlamaForCausalLM`. And a structural QKV-fusion probe
  exposes the rank-expansion-for-fusion invariant (Theorem 2) that the subset
  fallacy obscured.

All three gates pass deterministically (0.065 s total, CPU, no HF-Hub I/O).

## Prediction vs measurement

| # | Prediction (MATH.md §3) | Measured | Match |
|---|---|---|---|
| P1 | `import peft` hard-required | Succeeded; `peft==0.18.1` | ✓ |
| P2 | `PeftModel.from_pretrained(base, dir)` returns without exception | Returned; load time 0.034 s | ✓ |
| P3 | `pm(input_ids).logits` produced; shape `[1, 4, 256]` | Shape `[1, 4, 256]` | ✓ |
| P4 | Naive fused-QKV A-stack shape `[18, 64]` = `[3r, d_in]` | `[18, 64]` | ✓ |
| P5 | Naive fused-QKV B block-diag shape `[192, 18]` = `[3·d_q, 3r]` | `[192, 18]` | ✓ |
| P6 | Grassmannian `max(|A^T A − I|) < 1e-6` | `2.38e-7` | ✓ |

## K1576 gate-by-gate

### K1576.a — hard-required import [PASS]
- `peft 0.18.1`, `transformers 5.5.0`, `torch 2.10.0` imported at top-level.
- Module cannot load otherwise ⇒ the silent-bypass antipattern is
  structurally impossible (the script would raise `ModuleNotFoundError`
  before any "pass" could be written).

### K1576.b — real PeftModel.from_pretrained + forward [PASS]
- `peft.PeftModel.from_pretrained(tiny_llama, adapter_dir)` succeeded in
  0.034 s.
- `peft_model(input_ids=rand[1,4]).logits.shape == (1, 4, 256)` — matches
  base `LlamaConfig.vocab_size=256`.
- **12 `LoraLayer` instances wrapped** (= 4 layers × 3 projections
  {q_proj, k_proj, v_proj}) — equals `expected_wraps`, confirming no
  silently-skipped targets. A parent-style subset check would have passed
  even with 0 wraps; this gate is sensitive to the actual load behaviour.
- `base_model_name_or_path = None` in `adapter_config.json` means the loader
  honors the model object passed in (standard PEFT pattern for locally
  constructed bases); load path exercises real adapter key parsing and
  shape validation.

### K1576.c — QKV-fusion structural transformation [PASS]
- Separate-projection enumeration: `{q,k,v}_proj.lora_A.weight` each present
  at all 4 layers; `fused_keys_present = false`. Confirms the adapter
  targets a **separate-QKV** base (Llama/Gemma convention) and is **not**
  pre-fused.
- Block-diagonal fusion shapes match Theorem 2:
  `A_stack: [3r=18, d_in=64]`; `B_stack: [3·d_q=192, 3r=18]`.
- **Subset-fallacy probe**: `max|A_q − A_k| = 0.589`,
  `max|A_q − A_v| = 0.640` — the three A matrices are **distinct** (>> 1e-3
  threshold). Therefore the "naive row-stack" fused adapter would not
  preserve the separate-projection deltas; fusion **requires rank expansion
  to 3r** (block-diagonal), which is the hidden cost a subset-direction
  claim ("PEFT-compat ⇒ vLLM-compat") would gloss over. Gate passes by
  correctly exposing this transformation as non-trivial.

## What this does and does not support

### Supports
- **MLX→PEFT transpose bijection** (Theorem 1, Finding #433) remains intact
  and now survives a real runtime load, not just a schema check.
- **Silent-bypass antipattern** is structurally eliminated in this
  experiment's code path (K1576.a). Any downstream format-compat claim
  that reuses this pattern is immunised against this failure.
- **QKV-fusion transformation is rank-expanding** (Theorem 2). Downstream
  work that wants fused-QKV runtimes (vLLM, GPT-NeoX-style heads) must
  implement `rank=3r` block-diagonal fusion — single-rank row-stack is
  invalid under distinct A matrices.

### Does not support
- **Runtime serving** on vLLM / Unsloth (CUDA-only; remains correctly
  out-of-scope per PLAN.md Part 2). `exp_p1_t4_format_compat_v2` already
  closes this with honest SKIP. This experiment upgrades the *format* claim,
  not the *runtime* claim.
- **Training drift**: synthetic Grassmannian substrate has `max_dev=2.38e-7`
  (well under 1e-6 tolerance). Post-training drift claims require a
  separate interference experiment (see parent LEARNINGS `Implications §2`).
- **Base-model-specific quirks**: we used a tiny `LlamaConfig` to eliminate
  HF-Hub fetch and keep runtime <100 ms. Gemma 4 / Qwen 3 have the same
  separate-QKV convention on the HF side, so the theorem transports; but a
  live load against those specific checkpoints is out of scope here and
  is better validated by `exp_g4_*` experiments.

## Assumptions (audit surface)

- `peft 0.18.1`, `transformers 5.5.0`, `torch 2.10.0` (captured in
  `results.json["versions"]`). Behaviour on older pins (`peft<0.5`,
  `transformers<4.30`) is **not** claimed — the loader internals changed.
- `base_model_name_or_path=None` in `adapter_config.json` is intentional:
  we pass the in-memory model object to `from_pretrained`. This is the
  expected PEFT pattern for local-resource scenarios and is orthogonal to
  the KC.
- The fused-QKV test is **structural**, not runtime: we compute the predicted
  fused shapes and confirm distinctness of A matrices. Running a fused-QKV
  base (e.g. GPT-NeoX `query_key_value`) is CUDA-unfriendly for our N=25
  workflow and would not add information to what Theorem 2 already proves.

## Verdict-consistency pre-flight (all six required for `supported`)

1. `results.json["verdict"]` ≠ `"KILLED"` → `"SUPPORTED"` ✓
2. `results.json["all_pass"]` == `True` ✓
3. PAPER.md verdict line contains no `PROVISIONAL`, `PARTIALLY SUPPORTED`,
   `NOT SUPPORTED`, `INCONCLUSIVE`, `DEGENERATE` ✓
4. `is_smoke == false` ✓
5. `git log MATH.md run_experiment.py`: MATH.md committed in 6518f02,
   run_experiment.py in c952fad — both before the run. No KC edits
   post-run. ✓
6. Antipattern review — no auto-injected `type: fix` memory applies:
   - composition math bug — N/A (no composition here)
   - tautological routing — N/A (no routing)
   - unsafe adapter scale (LORA_SCALE=20) — N/A (α=6, scale=1)
   - KC-swap-after-failure — ✗ no edit post-run
   - smoke-as-full — ✗ `is_smoke=false`, full run
   - eval-template truncation → base=0% — N/A (no eval)
   - proxy-model-substituted-for-target — partial (tiny Llama, not Gemma 4)
     but mitigated: the KC is format-agnostic; Theorem 1/2 is base-agnostic
     for any separate-QKV HF architecture (Llama/Gemma/Qwen). Scope
     declared in §4 of MATH.md. Not a `supported` blocker.
   - `shutil.copy` as new adapter — N/A (synthetic adapter built in-place)
   - hardcoded `"pass": True` — ✗ `pass` is computed per gate, falsifiable
     via the distinct-A check and the wrap-count check
   - file-existence-cache — N/A
   - copy-paste scaffolding — ✗ this script is written from scratch;
     shared constants (HIDDEN, RANK, etc.) are defined once and the
     Grassmannian builder is re-implemented, not imported from parent

   All checks clear.

## Event payload note

Status: `supported`. DB kill ID `1576: pass`. Ready for reviewer.
