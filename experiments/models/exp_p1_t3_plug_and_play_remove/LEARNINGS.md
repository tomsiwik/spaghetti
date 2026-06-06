# LEARNINGS: T3.7 Plug-and-Play Hot-Remove (V2 audit-rerun, 2026-04-18)

## Core Finding

V1 "supported" (2026-04-10) is retroactively invalid. V2 KILLED with three
independent structural causes plus a missing-artefact precondition. 7th
precondition-probe kill in 24 h — class-level standing rule confirmed.

## Kill Causes

- **C1 tautological routing (mem-antipattern-002):** `REAL_ADAPTER_PATHS[domain]`
  hardcoded → K1070 "bit-exact remaining outputs after remove" is dict semantics,
  not Theorem 1.
- **C2 adapter copy forgery (mem-antipattern-009):** `geography` and `history`
  are both `shutil.copy(finance)` → K1071's 100% measures finance-weights under
  two labels, not a freed-slot test.
- **C3 wrong latency object (mem-antipattern-011):** K1072 times `dict.__delitem__`
  (O(1) hash-table) not weight unload. 10ms threshold on ~1µs op has no discriminating
  power.
- **C4 upstream artefacts absent:** 0/5 expected `.safetensors` on disk. T2.1 KILLED
  2026-04-18; T2.6 weights lost.

## Standing Rules (7 precondition-probe kills this loop)

1. Precondition-probe before macro claim.
2. Registry ≠ artefacts; dir existence is not file existence.
3. Downstream P1 macros inherit upstream audit flags.
4. `code-bug` tag may be a decoy when V1 failure is mathematical.
5. Composition / remove-invariance requires genuine routing, not
   `ADAPTER_PATHS[domain]` lookup.
6. Hot-add/hot-remove latency benches weight I/O, not dict mutation.
7. Adapter mint is training, not `shutil.copy` — a byte-copy is the same
   weights under a different label, and any quality claim is a lie by identity.

## V3 Blockers (do not auto-spawn)

- T2.1 rebuild with MedQA USMLE 5-choice (DB KC #1030), `max_tokens >= 512`,
  persisted `.safetensors`, `adapters/code/` created.
- T2.6 adapter weights recovered or retrained on disk.
- Code rewrite dropping `REAL_ADAPTER_PATHS[domain]`; implement either
  simultaneous N≥2 activation or per-sample routing.
- K1071 requires a genuinely-trained novel adapter (not `shutil.copy`).
- K1072 requires timing the actual weight-unload path (GPU free + mmap close),
  not `del d[k]`.

## Routing Signal

7th precondition-probe kill confirms class-level standing. No new mem-antipattern
needed (002, 009, 011 all apply as-is). No ref-add (process/artefact kill).
Cluster of downstream T2.1-dependent macros now at 7: `peer_comparison_llama31_8b`,
`peer_comparison_qwen3_4b`, `mtbench_composed`, `sft_residual_gemma4`,
`n25_composition`, `plug_and_play_add`, `plug_and_play_remove`. Next researcher
claim should be a T2.1-independent experiment or T2.1 V2 itself.
