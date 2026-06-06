# LEARNINGS: T4.3 MLX-Native Adapter Hot-Swap Serving (V2 audit-rerun, 2026-04-18)

## Core Finding

V1 "SUPPORTED" (2026-04-17) is retroactively invalid. V2 KILLED with four
independent structural causes plus a missing-artefact precondition. 8th
precondition-probe kill in 24 h — class-level standing rule reconfirmed,
rule #8 (prefill/decode + OOD) promoted to class-level.

## Kill Causes

- **C1 upstream artefacts absent.** 0/5 expected `.safetensors` on disk
  (math, code, medical from T2.1; legal, finance from T2.6). T2.1
  status=killed 2026-04-18 (MCQ metric-swap + format-artefact). T2.6
  weights lost. V1 `results.json` also missing — claim unverifiable even
  on provenance.
- **C2 graph-recompile omitted (mem-antipattern-011 specialisation).**
  `swap_adapter` times `load_weights + mx.eval(parameters)` in a tight
  loop of 20 swaps without any forward pass between. MLX recompiles the
  forward graph on the first forward after parameter mutation — cost
  never enters V1's clock. Theorem 1 lower bound is correct; V1's
  measurement omits the dominant real term.
- **C3 prefill/decode conflation + OOD prompt.** `generate_tokens` returns
  `N_generated / (t_end - t_start)`, mixing compute-bound prefill with
  memory-bound decode. Math adapter evaluated on a non-math prompt
  ("Explain machine learning...") — OOD early-EOS biases the denominator.
  90.8% ratio is not a LoRA-overhead characterisation.
- **C4 tautological routing (mem-antipattern-002).**
  `routing_registry = {d: p for d, p in ADAPTER_PATHS.items()}; selected
  = routing_registry[domain]` is `dict[k] == dict[k]` by identity. Zero
  TF-IDF. Reported <1µs is dict-hash microbench cost.

## Standing Rules (8 precondition-probe kills this loop)

1. Precondition-probe before macro claim.
2. Registry ≠ artefacts; dir existence is not file existence.
3. Downstream P1 macros inherit upstream audit flags.
4. `code-bug` tag may be a decoy when V1 failure is mathematical.
5. Composition / routing claims require genuine routing, not
   `ADAPTER_PATHS[domain]` lookup.
6. Hot-add/hot-remove/hot-swap latency benches weight I/O + graph recompile
   (MLX-specific), not `load_weights` alone and not `dict.__setitem__`.
7. Adapter mint is training, not `shutil.copy`.
8. **NEW (class-level):** Throughput on Apple Silicon decode must strip
   prefill and use in-domain prompts. Correct:
   `decode_tok_s = N_decoded / (t_end - t_first_token)`.

## V3 Blockers (do not auto-spawn)

- T2.1 rebuild with MedQA USMLE 5-choice (DB KC #1030), `max_tokens >= 512`,
  persisted `.safetensors`, `adapters/code/` created.
- T2.6 adapter weights recovered or retrained on disk.
- `swap_adapter` rewrite: return
  `t_first_token(after_swap) - t_first_token(baseline)`, not `load_weights + mx.eval`.
- `generate_tokens` rewrite: return `(text, decode_tok_s)` with prefill
  stripped from denominator.
- K1083 prompts must be in-domain per adapter (math prompt on math adapter,
  legal prompt on legal adapter, etc.).
- K1084 must invoke T4.1's actual TF-IDF router on raw prompt text — time
  the pipeline (tokenise + sparse matmul + argmax), not a Python dict lookup.

## Routing Signal

8th precondition-probe kill confirms class-level standing. No new
mem-antipattern required — 002 + 011 apply directly; rule #8 now catalogued
in PAPER.md standing-rules list. No ref-add (process/artefact kill, not
literature gap). Cluster of downstream T2.1-dependent macros now at 8:
`peer_comparison_llama31_8b`, `peer_comparison_qwen3_4b`, `mtbench_composed`,
`sft_residual_gemma4`, `n25_composition`, `plug_and_play_add`,
`plug_and_play_remove`, `vllm_adapter_serving`. Next researcher claim should
be a T2.1-independent experiment or T2.1 V2 itself.
