# LEARNINGS: SFT BitNet Generation Quality

## Core Finding

SFT adapters improve generation quality over NTP (judge 3.93 vs 3.72, 4/5 domains) but
are structurally incompatible with full-prompt energy gap routing: the same gradient
isolation that prevents instruction contamination destroys the NLL signal used for
adapter selection (4% routing accuracy vs 80% NTP). This is provable from the chain rule
and was formalized post-hoc as Theorem 1.

## Why This Happened

SFT loss masks instruction tokens from the gradient computation. By the chain rule,
adapter weights receive zero gradient from instruction positions. Since energy gap
routing computes Delta_E = NLL(base) - NLL(adapted) over ALL tokens, and instruction
tokens dominate most prompts (40-60%), the routing signal is diluted to noise.

This is not a training failure -- it is a structural consequence of combining two
independently valid mechanisms (SFT masking + full-prompt NLL routing) that have
contradictory requirements on instruction-token gradients.

**Key distinction:** The incompatibility is specific to FULL-PROMPT energy gap routing.
Response-token-only energy gap routing should preserve discrimination because SFT
adapters DO modify response-token logits (that's what they're trained to optimize).

## Confirming Evidence

1. **Our own Finding #184 (energy gated composition, KILLED):** LoRA adapters universally
   reduce NLL on all inputs due to overparameterization. Binary NLL gating is vacuous.
   This confirms that NLL-based mechanisms have fundamental limitations for adapter
   selection beyond just the SFT case.

2. **Our own PPL-probe routing at macro scale (KILLED):** Answer-conditioned PPL showed
   r=-0.63 correlation with ground truth at macro scale with real converged adapters,
   despite r=0.990 at micro scale. Response-token NLL signals can break down at scale.

3. **MoTE (Mixture of Ternary Experts):** Computes autoregressive loss strictly on
   response tokens and still successfully trains routing via gating logits + load-balancing
   loss. Confirms that response-only loss does NOT prevent routing -- but the router must
   be designed for it, not retrofitted from full-prompt NLL.

4. **Adapter Merging Reactivates Latent Reasoning Traces (arxiv 2601.18350, Zou 2026):**
   Domain-adapted and instruction-tuned adapters induce partially misaligned update
   directions in layers 6-10. Merging SFT + DAPT adapters causes 3-15x increase in
   reasoning leakage. Confirms that SFT and domain adaptation operate in structurally
   different subspaces -- consistent with our finding that NTP routing signals don't
   transfer to SFT.

## Contradicting Evidence

1. **Our own answer-conditioned PPL (Finding #182, micro scale):** Response-token-only
   NLL achieved AUC=0.942 for adapter ranking in the self-embedding quality discriminator
   experiment. This suggests response-token energy gap SHOULD work at micro scale.
   However, macro-scale PPL-probe routing was killed (r=-0.63), so scale matters.

2. **Velickovic et al. (arxiv 2601.22950):** PPL fundamentally cannot distinguish
   right/wrong outputs. If this extends to response-token PPL, even response-token
   energy gaps may fail for quality discrimination (though ranking may survive).

## Alternative Approaches

1. **Embedding-based routing (LoRAuter, arxiv 2601.21795):** Training-free, O(T) routing
   via task embeddings from frozen sentence encoder. Maps queries to task representations,
   retrieves nearest neighbors, fuses weights via similarity kernel. Avoids NLL entirely.
   Our project already has tiny routing heads (Finding #179, 100% accuracy, 2.32% overhead)
   which are an embedding-based approach -- these are SFT-compatible by design.

2. **Per-token learned routing (MoLoRA, arxiv 2603.15965):** Learned router per-token,
   Qwen3-1.7B+4 adapters > 8B. Router trained end-to-end, so naturally compatible with
   whatever loss function (NTP or SFT) the adapters use.

3. **Response-token energy gap (our own proposed fix):** Compute Delta_E only over response
   positions. Preserves SFT contamination prevention while restoring NLL discrimination.
   MoTE's success with response-only loss supports feasibility. However, requires knowing
   response start position and generates a chicken-and-egg problem noted by the reviewer.

4. **Hybrid routing (NTP signal, SFT generation):** Use NTP adapters for routing
   computation, then swap to SFT adapters for generation. Avoids the incompatibility
   entirely but doubles adapter storage and adds complexity.

5. **LoRA-Flow (per-step gating):** Lightweight gating network learns dynamic mixture
   weights at each generation step. Naturally compatible with SFT since it learns routing
   end-to-end rather than relying on pre-computed NLL.

## Implications for Next Experiments

1. **The routing problem is solved for our architecture -- just not with energy gaps.**
   Tiny routing heads (Finding #179) already achieve 100% accuracy with 2.32% overhead
   and are SFT-compatible. Energy gap routing was an attempt to avoid training a router,
   but the SFT incompatibility makes it impractical for SFT adapters.

2. **SFT quality improvement is real and robust.** Judge scores 3.93 vs 3.72 (NTP),
   4/5 domains improved. The generation quality problem (Finding #178, NTP kills prose)
   is solved by SFT. The remaining blocker is correct routing TO the SFT adapters.

3. **The existential question is answerable with correct routing.** Math correctness
   failed (10%) but was confounded by 4% routing accuracy. With correct routing (via
   tiny heads or response-token energy gap), math correctness should approach the
   oracle level. This is the critical next test.

4. **Full-prompt NLL is dead for SFT adapter systems.** Any future routing mechanism
   must either (a) use response-token-only signals, (b) use embedding-based routing,
   or (c) use a learned router trained end-to-end with the SFT loss.

## Recommended Follow-Up

1. **Response-token energy gap routing (highest priority):** Test whether computing
   Delta_E only over response positions restores routing accuracy for SFT adapters.
   Motivated by: Theorem 1's resolution (MATH.md line 112-113), MoTE's success with
   response-only loss, and Finding #182's AUC=0.942 with response-token NLL.
   Literature: MoLoRA (2603.15965) per-token routing validates token-subset routing.

2. **SFT + tiny routing heads (alternative path):** Combine SFT adapters (quality)
   with existing tiny routing heads (Finding #179, 100% accuracy). This bypasses the
   energy gap incompatibility entirely using already-validated components. No new
   experiment design needed -- just wire the two together.

3. **Embedding-based routing evaluation:** Compare LoRAuter-style (2601.21795)
   embedding routing against energy gap and tiny heads. Would provide a training-free,
   SFT-compatible routing baseline.

## Audit-Rerun Closure Addendum (2026-04-18)

Original verdict line read "SUPPORTED (guided exploration). Finding #187: PROVISIONAL"
while K3 (math ≥40%) measurably failed (10%). Under PLAN.md §1 item 3 this violates
verdict consistency (PAPER must not contain PROVISIONAL when status=supported). The
audit-rerun closure reclassifies to **KILLED** via three independent PAPER.md closure
theorems: (C1) Lemma 1 dictates 4% routing floor — structural, not hyperparameter;
(C2) N=10 Wilson 95% upper bound 40.4% tangent to the 40% threshold — robust at any
base PPL; (C3) judge ceiling 3.93 caps headline metric — consistent with Finding #178.
LORA_SCALE=20 (mem-003) acknowledged with direction-preservation. DB and evidence now
agree: `experiment complete --status killed --k 580:fail`.

**Pattern note (watch-list, not promoted):** "Lemma-Side-Effect Sabotage" — combining
two mechanisms M1 + M2 where M1's necessary mathematical property destroys M2's
required signal. Single instance only (this experiment). Distinct from mem-021
(CEILING-HEADROOM COLLAPSE: mechanism layered on baseline at its ceiling). Promote
only if a second distinct instance surfaces.

Finding #187 (SFT-Routing Incompatibility) remains the core finding — its structural
generality (chain rule → full-prompt NLL routing incompatibility) is what any future
routing mechanism must respect, regardless of the verdict label on this one experiment.
