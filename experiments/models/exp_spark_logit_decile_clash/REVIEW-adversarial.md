# REVIEW (adversarial) — exp_spark_logit_decile_clash → KILL confirmed

Verdict sealed: KILLED. Both kill-2306 clauses fired correctly; no mock, no tautology, no integrity issue.

## Mock / real
- verify-experiment.sh exits 0; is_smoke=false; results.json present.
- Model in MATH.md = loaded = results.json (gemma-4-e4b-it-4bit). Real math/python/medical adapters.safetensors present.
- Composition is per-adapter Σᵢ(1/N)sᵢBᵢAᵢ (AvgComposedQProj accumulates each B_iA_i separately), NOT degenerate (ΣB)(ΣA). scale=6.0≤8.

## (a) Clash metric correct, non-degenerate
- dz = zA−z0 is a genuine two-forward delta (Phase1 base logits z0 cached, Phase2 composed zA on identical ids_row).
- bottom-decile via sort(p0 asc), thr=sorted[k10-1], mask p0<=thr → base-pruned tokens (correct).
- clash = Σ_{topK(relu dz)∩bottom} relu(dz) / Σ_{topK} relu(dz) — matches MATH.md verbatim.
- NOT degenerate: mean_clash=0.081, 2.2% of tokens >0.5 (sparse but nonzero, not all-0/all-1).

## (b) Baselines on SAME tokens — confirmed
- Single per_tok list; Phase2 writes kl/clash/align_cos/delta_mag into the SAME record at ptr,
  guarded by assert rec["prompt_idx"]==pi and final assert ptr==len(per_tok). Phase3 lists are
  comprehensions over the same list/order. Label and both geometric baselines share identical (prompt,token) rows.
- h_base is mx.eval'd and cached in Phase1 BEFORE Phase2 overwrites the tap's _captured, so
  dh = h_comp − h_base is a true two-forward hidden delta (delta_mag baseline is not a self-difference artifact).

## (c) delta_mag rho=0.2076 — real weak signal, not noise
- n=4117. clash rho=0.0241 → z≈1.55, NOT significant at p<0.05 ("effectively uncorrelated" is accurate).
- delta_mag rho=0.2076 → z≈13.3, p≪0.001: a genuine but weak predictor (~4% rank variance). Worth flagging
  as the least-bad surviving geometric signal; PAPER.md already characterizes it correctly.
- align rho=−0.0513 → z≈−3.29, marginal, consistent with F#869 near-death.

## Consistency / integrity
- results.json verdict=killed, all_pass=false, is_smoke=false; PAPER.md verdict=KILLED — all agree.
- Clause A (0.0241<0.45) and Clause B (0.0241<0.3576) both fire; (A OR B) → kill, robust.
- MATH.md untracked (new file): thresholds 0.45/0.15 in code == MATH.md; no post-hoc move possible.

No blocking findings. PROCEED to seal KILL.
