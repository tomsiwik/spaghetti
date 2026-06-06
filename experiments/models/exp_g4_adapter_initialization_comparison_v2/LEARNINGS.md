# LEARNINGS — exp_g4_adapter_initialization_comparison_v2

## Core Finding (PROVISIONAL, smoke iter floor + 2 distinct mechanisms)

Two mechanistically distinct findings landed:
- **F#773** — v1's K1924 PASS was a *shared-seed artifact* (`mx.random.key(42)` reused across "different" inits → starting cross-init |cos|=0.977-0.9995). v2 with distinct seeds: starting |cos| collapses to 0.015-0.018, final to 0.027-0.042 (K1977 PASS at <0.20). Independent random low-rank A matrices remain near-orthogonal under 100-iter SGD. v1 K1924 retrospective verdict: **PASS was confounded; structural invariance claim survives at proxy level once confound removed**.
- **F#774** — Medical q_proj-only r=6 LoRA at 100 iters DROPS MCQ heldout 4.6pp (gaussian) / 9.2pp (grassmannian) / **14.2pp (kaiming)** vs no-adapter base 57.5%. PPL drops 4000× over the same span (looks like training works). K1985 (target / non-interference) FAIL.

## Why

- **F#773 mechanism**: Pre-LoRA-attach RNG state was identical across init runs in v1; LoRA `linear_to_lora_layers` consumed the *same* random matrix regardless of init label. Distinct-seed fix (v2) restores expected near-orthogonality of independent draws and shows v1's PPL spread (3.5%) was inside seed-noise floor (v2 K1979: within-init PPL seed-var = 24.6% on grassmannian alone) → v1's invariance claim was *unidentifiable* from single-seed PPL.
- **F#774 mechanism**: PPL on the medical-MCQ-formatted training distribution is a tautology with the loss function — minimizing it does NOT preserve out-of-distribution MCQ letter-prediction. q_proj-only r=6 at smoke iters under-fits behavior while over-fitting tokens. Replicates the **r≈0.08 PPL/task correlation** baseline (Finding #666 / behavioral-outcomes thesis). Within-init MCQ seed-var (12.5pp gaussian) exceeds cross-init means spread (9.6pp), so K1983 invariance question is unresolved at smoke N.

## Implications for Next Experiment

- **v3a (P3, full 1000-iter)** — does K1985 recover at convergence, or is q_proj-r6 *structurally* MCQ-suppressive? KCs #1994-#1996 inherit v2 K1977/K1983/K1985 schema; ~4-5h budget exceeds researcher 90-min cap → PROVISIONAL routing expected.
- **v3b (P3, canonical recipe)** — v_proj+o_proj per F#627, scale=4 per F#330, full iters. Tests whether F#774 MCQ-suppression is *target-locus dependent* (q_proj artifact) or *recipe-class dependent* (any LoRA at smoke). KCs #1997-#1999.
- **F#751 (parent v1) status**: should now be re-read as "PROVISIONAL on confounded measurement; v2 supersedes the proxy claim, F#774 introduces a behavioral angle parent never measured". No need to re-open v1 — v2 + v3 cover the question space.
- **Generalizable lesson**: any future LoRA experiment that reads PPL alone for verdict is structurally compromised on this codebase. Pair every PPL KC with a behavioral target KC (F#666 truth-table). Smoke-iter floor MUST stay PROVISIONAL — multiple iters of researcher hat already established this routing.

## Cross-references
F#773, F#774 (this run); F#751 (parent v1, retrospective caveat); F#666 (target-gated KILL, behavioral thesis); F#627 (canonical v_proj+o_proj target locus); F#169 (init-invariance prior — survives at proxy level here); F#682/F#683/F#684 (PROVISIONAL-on-smoke precedent cluster).
