# REVIEW-adversarial.md — exp_prompt_erasure_gemma4

## Verdict: **KILL** (affirming DB status=killed)

Smoke re-analysis of sibling `exp_knowledge_disentanglement_control`. K1722
is unambiguously falsified (−30pp MMLU, ≫ 2pp bound and ≫ n=20 binomial CI ≈ 21pp).
K1721 pooled fails (+5pp vs required +20pp); per-bench heterogeneity is an
informative *finding*, not a KC rescue. K1723 structural passes (3/3).

## Adversarial checklist

| Check | Status | Note |
|---|---|---|
| (a) verdict vs DB | ✓ | `results.json.verdict=PROVISIONAL`, DB `--status killed` (smoke convention; K1722 unambiguously falsified) |
| (b) all_pass vs claim | ✓ | `all_pass=false`, status=killed |
| (c) PAPER verdict line | ✓ | "KILLED (smoke; DB status=killed)" |
| (d) is_smoke vs full claim | ✓ | `is_smoke=true` declared |
| (e) KC modified post-run | ✓ | K1721/1722/1723 verbatim from MATH.md; code measures them literally |
| (f) tautology sniff | ✓ | K1721 = real lexicon+step regex on 60 responses; K1722 = accuracy delta; K1723 = 3 structural asserts on sibling pipeline |
| (g) K-ID ↔ code match | ✓ | `k1721_method_invocation`, `k1722_mmlu_preserved`, `k1723_recipe_fidelity` dict keys match MATH §Kill criteria |
| (h) composition bug | N/A | no composition (single adapter, re-analysis) |
| (i) unsafe LoRA scale | ✓ | scale=4.0 (< 8) |
| (j) single-sample routing | N/A | no routing |
| (k) `shutil.copy` forgery | ✓ | adapter reuse is explicit provenance (`sibling_adapter_path`), not relabelled |
| (l) hardcoded `pass: True` | ✓ | all pass flags computed |
| (m) model substitution | ✓ | Gemma-4-E4B-it-4bit declared in MATH.md and results.json; sibling trained same model |
| (m2) skill invocation | ✓ (N/A) | pure-Python re-analysis, no model load — MLX skills unnecessary here; sibling's training code is out-of-scope for this review |
| (n) base=0%+no thinking | ✓ | base method rate = 53.3% (real native thinking mode), not truncation artifact |
| (o) n < 15 | ✓ | K1722 n=20 (MMLU), K1721 n=60 pooled |
| (p) synthetic padding | N/A | real eval rows |
| (q) baseline drift | ✓ | base measured in same sibling run |
| (r) prediction table | ✓ | PAPER.md §"Pre-registered predictions vs measurements" |
| (s) math / unsupported claims | ✓ | Orca-2 Lemma 1 paraphrase is accurate; F3 native-method guard is acknowledged; failure analysis is grounded in evidence |

## Substantive notes

1. **Per-bench heterogeneity is the real finding**: MMLU invocation 10%→70%
   (+60pp on-domain) vs TriviaQA 75%→30% (−45pp OOD). The pooled +5pp hides
   a format-transfer failure and simultaneous on-domain knowledge collapse.
   This is *more* informative than a clean pass would have been.

2. **K1722 collapse is consistent with Finding #536 (MCQ-hurts-thinking)**
   and with the sibling's own K1734 failure. Low-rank (16) update on
   `v_proj+o_proj` at N=25/60-steps is not orthogonal to factual-recall
   circuits.

3. **Smoke→killed mapping** is a legitimate use of the convention:
   `is_smoke=true` + unambiguous falsification (Δ = −30pp, 14× the bound,
   outside n=20 95% CI). Future `_v2` plan is pre-registered in PAPER.md.

## Assumptions / judgment calls

- Accepted re-analysis pattern: no new training; the adapter and eval
  responses come from sibling `exp_knowledge_disentanglement_control`.
  This is explicit (`reuse=true`, `source=...`, sibling path annotated
  for provenance only). Not an antipattern because: (i) no new adapter
  claim, (ii) K1721 is a *new* metric not computed by sibling,
  (iii) sibling's adapter is the object under test for the Orca-2 recipe.
- K1721 runs on 300-char response prefixes (MATH.md A3). Flagged as a
  limitation in PAPER.md; acceptable at smoke because the teacher template
  is front-loaded.

## Route: `review.killed`
