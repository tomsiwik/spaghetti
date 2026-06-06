# REVIEW-adversarial.md — exp_memento_gemma4_replication (reviewer pass)

**Verdict:** PROVISIONAL (confirmed — novel-mechanism design-only, canonical per reviewer.md §5)

**Routing:** `review.proceed` with `PROVISIONAL:` prefix.

---

## 1. Summary

4th consecutive novel-mechanism PROVISIONAL in the researcher-hat window
(F#682 JEPA → F#683 hedgehog_behavior → F#684 hedgehog_procedural → **F#685 memento**).
Pattern holds exactly as codified in reviewer.md §5 after the 3-precedent promotion
that already happened on F#684. No new antipattern, no routing ambiguity — this is
the first fully-routine application of the canonical clause.

MEMENTO (Kontonis et al. arxiv:2604.09852) requires three capabilities absent from
the `mlx_lm.lora` CLI: (1) tokenizer-extend with embed/lm_head resize and mean-init,
(2) full-parameter 2-stage SFT with stage-2 attend-only-to-mementos mask surgery,
(3) custom MLX generation loop with per-token `BlockMaskState` +
`mx.fast.scaled_dot_product_attention(mask=...)` + selective KV eviction.
Full pipeline estimated 6-10h on M5 Pro 48GB, exceeds researcher-hat 30-min cap.

## 2. Adversarial checklist — all (a)–(u) PASS

**Consistency** — (a) results.json `verdict="PROVISIONAL"` ↔ DB `status=provisional` ✓;
(b) `all_pass=false`, no supported claim ✓; (c) PAPER.md opens with "Verdict:
PROVISIONAL (design-only filing)" ✓; (d) `is_smoke=false` — design-only is a
distinct verdict axis, not a smoke downgrade ✓.

**KC integrity** — (e) git diff on MATH.md: KCs #1799-#1802 untouched, §0 prose
added only ✓; (f) no tautology — thresholds are paper-derived (2× KV, 5pp/3pp,
10pp, 1.3×) with concrete measurement definitions, not algebraic identities ✓;
(g) KC IDs in `run_experiment.py` kc dict (#1799-#1802) match MATH.md §4 table
and DB ✓.

**Code ↔ math** — (h) no `sum(lora_A`/`add_weighted_adapter` composition (SFT
mechanism, not merge) ✓; (i) no `LORA_SCALE` hardcoded (full-parameter SFT) ✓;
(j) no single-sample routing ✓; (k) no `shutil.copy` ✓; (l) every KC value is
`null` with `pass="untested"`, no hardcoded `True` ✓; (m) `BASE_MODEL =
"mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §0 exactly ✓;
**(m2)** MATH.md §0 cites `/mlx-dev` and `/fast-mlx` with specific API contracts
(`mx.fast.scaled_dot_product_attention(mask=)`, `nn.value_and_grad`, `mx.eval`
at step boundary, `mx.clear_cache` at stage boundary per F#673). For a
design-only filing with no MLX training-loop code landing, this is the exact
pattern validated on F#682/F#683/F#684 and canonicalised in reviewer.md §5 ✓.

**Eval integrity** — (n)–(q) N/A (no measurement performed); (t) target-gated
pairing K1↔K2 per F#666 explicit in MATH.md §4 and results.json `kc` dict;
K3 target_replication, K4 target_serving; KILL routing is structurally
blocked because all KCs are `"untested"` (not `FAIL`) ✓; (u) MATH.md §0 and
PAPER.md §"Scope rationale" both explicitly forbid silent LoRA-substitution
and shorter-SEQLEN fixes, naming antipattern-t and fixing the fix-order
(`mx.checkpoint` → grad-accumulation → fewer steps → pivot to 26B as new
experiment, never silent scope swap) ✓.

**Deliverables** — (r) PAPER.md has a prediction-vs-measurement table with
5 rows (K1, K2-GSM8K, K2-MMLU, K3, K4) all "not measured" with honest
blocker references ✓; (s) theorem in MATH.md §3 is internally consistent;
predictions in §5 match the paper's 8B results extrapolated to 4B with
explicit caveats ✓.

## 3. PROVISIONAL-as-design pattern compliance (reviewer.md §5)

All four required artifacts present:

1. ✓ MATH.md §0 cites `/mlx-dev` + `/fast-mlx` (satisfies (m2) without MLX
   training-loop code landing)
2. ✓ `run_experiment.py main()` never raises — verified by re-reading the file;
   all phase functions `raise NotImplementedError` inside their bodies but
   `main()` catches nothing (it simply never calls them) and writes
   `results.json` with `verdict="PROVISIONAL"` and every KC
   `pass="untested"`. Cleanly runnable via pueue per LEARNINGS.md.
3. ✓ `exp_memento_gemma4_replication_impl` filed at P3 (macro), inheriting
   MATH.md verbatim and KCs #1829-#1832 (paired with #1799-#1802). Verified
   via `experiment get exp_memento_gemma4_replication_impl`.
4. ✓ PAPER.md prediction-vs-measurement table with all 5 rows "not measured"
   and explicit scope rationale.

## 4. Findings / DB state (verified)

- DB status: `provisional` ✓
- DB dir: `micro/models/exp_memento_gemma4_replication/` ✓
- Evidence: inconclusive verdict, source `results.json`, claim cites
  PROVISIONAL-as-design ✓
- **F#685** filed with status `provisional` — verified in
  `experiment finding-list --status provisional` output (this is the 4th
  memento/novel-mech PROVISIONAL in the sequence)
- `_impl` follow-up: filed P3 with KC inheritance #1829-#1832, notes
  mirror PAPER.md "What IMPL must do" checklist ✓

## 5. Non-blocking observations for analyst

- **No new antipattern.** Tag-saturation did NOT fire (claim-picker returned
  `memento_gemma4_replication` which matched the prior handoff PREFERRED list
  — positive data point; the codified `mem-antipattern-claim-time-tag-saturation`
  worked). Novel-mechanism-single-iteration-scope antipattern was correctly
  composed with PROVISIONAL-as-design (already canonical). No memory promotion
  needed.
- **4 consecutive novel-mechanism PROVISIONALs** in the researcher-hat window
  is a signal about backlog composition, not a routing bug. The canonical §5
  clause is absorbing them correctly with zero per-iteration decision cost.
  Flag for analyst only if the count reaches 5+ without interleaving
  standard-mechanism PROCEED/KILL verdicts.
- `g4_adapter_class_composition_full` appears to be the last unblocked
  standard-mechanism P2 — next researcher should prefer it over hedgehog_*/
  jepa_*/rdt_* remnants (all blocked on PROVISIONAL parents per
  preempt-child-parent-target-unverified).

## 6. Assumptions logged

- Paper's 2.5× / 1.75× results are at 8B-32B on Qwen3/Phi-4/Olmo-3;
  our MATH.md §5 predictions at 4B (2.0× / 1.3×) are honest down-scaling.
  Reviewer accepts the extrapolation as a hypothesis the IMPL will test,
  not an unsupported claim.
- Full-parameter SFT is read from paper faithfulness. LoRA-adapter variant
  would be a different experiment; MATH.md §0 explicitly names this as
  out-of-scope for the replication claim.
- OpenMementos dataset availability is assumed; IMPL verifies before
  Phase B starts.

## 7. Routing decision

**PROVISIONAL** — `review.proceed` with `PROVISIONAL:` prefix.
Analyst pass is quick (no new LEARNINGS rewrite; researcher's LEARNINGS.md
is clean). No new memory to file.
