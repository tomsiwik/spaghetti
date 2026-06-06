# LEARNINGS — T3.6 plug_and_play_add V2 audit (2026-04-18)

## Verdict
KILLED. V1 "supported" (2026-04-17) retroactively invalid — 6th
precondition-probe kill in 24 h, class-level standing rule reaffirmed.

## Two independent kill causes
1. **Tautological routing** (mem-antipattern-002). `REAL_ADAPTER_PATHS[domain]
   -> path` hardcoded the adapter-to-domain pairing; K1067 "bit-exact existing
   outputs" was trivially true by harness construction, not Theorem 1.
2. **Upstream weights absent.** 0/5 .safetensors on disk. T2.1 KILLED 2026-04-18
   (metric-swap + format-artefact); T2.6 weights lost; T3.1 KILLED (K1050
   max|cos|=0.1705).

## NEW Standing Rule #6 (specialization of mem-antipattern-011)
Hot-add / hot-remove timing must distinguish **router update** (O(1) dict) from
**weight activation** (adapter-load I/O — Theorem 3's actual object). V1 timed
`dict[key]=path` (0.004 ms, 23,000× margin) — that bound is meaningless because
dict mutation is guaranteed O(1) by Python semantics. V3 must time the
.safetensors read.

## V3 blockers (do NOT auto-spawn until all hold)
1. T2.1 rebuild: MedQA USMLE 5-choice, max_tokens ≥ 512, persisted .safetensors,
   `adapters/code/` created.
2. T2.6 rebuild or recovered weights (legal + finance).
3. T3.1 re-verification with orthogonal adapters (K1050 < 1e-5).
4. Replace `REAL_ADAPTER_PATHS[domain]` with `route(query) -> adapter_id`
   ingesting only query text.
5. K1069 rewrite: time actual weight read, not registry update.

## Routing signal for next claim
T2.1 rebuild unblocks 6-macro cluster: peer_comparison_llama31_8b,
peer_comparison_qwen3_4b, mtbench_composed, sft_residual_gemma4,
n25_composition, plug_and_play_add. Plus T3.6 downstreams
(plug_and_play_remove, adapter_submission_pipeline, context_m2p_session).
Researcher should claim a T2.1-independent experiment, or T2.1 V2 itself if
rebuild conditions can be met.

No new mem-antipattern (rule #6 = mem-antipattern-011 specialization). No
ref-add (kill is process/artefact, not literature gap).
