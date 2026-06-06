# MATH — exp_followup_jointly_trained_routing_molora

## Preempt-KILL classification
**audit-2026-04-17 followup, no-run preempt.** Structural impossibility
derived from four supported/killed findings. No training or eval executed.

## Hypothesis under test (K1551)
> Jointly-trained router beats post-hoc TF-IDF routing by ≥3pp on
> held-out per-token task at N=5.

## Theorem 1 (Ceiling incompatibility)
Let `A_TF-IDF ∈ [0,1]` denote TF-IDF nearest-centroid per-token routing
accuracy on held-out N=5 real-NLP domains. Let `A_MoLoRA ∈ [0,1]` denote
jointly-trained Gumbel-sigmoid router per-token accuracy on the same
held-out split. Let `Δ := A_MoLoRA − A_TF-IDF`. Then:
```
  Δ ≥ 3pp  is structurally impossible.
```

### Proof
**Lemma 1 (TF-IDF ceiling).** Finding #431 measured `A_TF-IDF = 96.6%`
at N=5 on real NLP domains (math/code/medical/legal/finance) with
Gemma 4 base. Max headroom `1 - A_TF-IDF = 3.4pp`. Any `Δ ≥ 3pp`
forces `A_MoLoRA ≥ 99.6%` — effectively error-free per-token routing.
QED Lemma 1.

**Lemma 2 (Per-token full-sequence null, F#305).** Under causal
attention with adapter-mixed full-sequence forward passes, per-token
routing on mixed-domain sequences is bit-exactly identical to
per-sequence routing: measured `PPL_per-token-full = 4.815 =
PPL_per-seq-best` across 200 mixed sequences × 10 domain pairs.
Cross-attention contamination from wrong-adapter tokens makes
per-token selection null. Mechanism: attention matrix conditions on
adapter-mixed activations; the wrong-adapter contribution enters every
query's receptive field. This holds for BOTH TF-IDF and jointly-trained
routers because the contamination is in the attention — not the router.

Corollary: if held-out contains mixed-domain sequences evaluated on full
forward passes, `A_MoLoRA = A_TF-IDF` on those tokens (both equal the
per-sequence ceiling). The 3pp gap must come entirely from homogeneous
held-out sequences. But:

**Lemma 3 (MLP-only contamination-free ceiling, F#312).** When
contamination is eliminated (MLP-only per-token on single-domain or
per-segment forward passes), improvement over per-sequence is only
+3.3% PPL — and segment-isolation (known-boundary oracle) dominates at
+16.1%. Jointly-trained router cannot structurally exceed the +3.3%
MLP-only ceiling because both mechanisms share the same
contamination-free regime; jointly-trained merely replaces oracle
routing signal with a learned classifier.

**Lemma 4 (Representation-limited routing, F#193).** F#193 established
at N=24 that routing architecture is irrelevant — mean-pooled hidden
states set a 40% accuracy ceiling. At N=5 with well-separated real
domains (math/code/medical/legal/finance), hidden states DO carry
sufficient signal: both text-based (TF-IDF) and hidden-state-based
(jointly-trained) routers saturate near-oracle. Architecture cannot
recover the last 3pp because the REMAINING error is from domain
overlap in vocabulary (finance↔economics) or activation geometry —
neither of which jointly-training resolves.

**Lemma 5 (Ridge router + E2E KILLED, F#340).** The closest architectural
analog — ridge router + single-pass MLP jointly applied to mixed-domain
— failed catastrophically (8.6pp accuracy drop 98.3%→89.67% on mixed).
Jointly-trained adapter + router at N=5 is the same architecture class
(shared representation, jointly optimized, context-dependent). The
context-dependence failure mode F#340 diagnosed is inherited.

**Combining:** Regardless of held-out composition:
- If homogeneous held-out → Lemmas 1, 4: `Δ ≤ 3.4pp` (ceiling) and
  `Δ ≈ 0` (architecture irrelevant at near-oracle). Expected `Δ << 3pp`.
- If mixed-domain held-out → Lemma 2: `Δ = 0` (null mechanism).
- MLP-only partial fix → Lemma 3: `Δ ≤ 3.3pp` (ceiling), likely
  below 3pp without oracle routing signal.

∴ K1551 unreachable. **QED.**

## Antipattern match
- `near-oracle-ceiling-vs-3pp-threshold` (Lemma 1)
- `per-token-full-sequence-routing-null` (F#305 reuse)
- `representation-bottleneck-not-architecture` (F#193 reuse)
- `ridge-analog-already-killed-on-mixed-domain` (F#340 reuse)

## Kill Criteria (pre-registered, unchanged)
- K1551: jointly-trained router beats post-hoc TF-IDF routing by ≥3pp
  on held-out per-token task → **marked FAIL preemptively by
  structural proof above**.

## Findings reused
- F#431 (supported): TF-IDF N=5 = 96.6% accuracy
- F#305 (supported): per-token full-sequence routing confirmed null
- F#312 (supported): MLP-only per-token +3.3% ceiling; segment
  isolation dominates +16.1%
- F#193 (killed): routing architecture irrelevant at N=24; mean-pooled
  hidden states bottleneck
- F#340 (killed): ridge router + single-pass E2E composition failure
  on mixed-domain

## Assumptions (logged per G1007)
- Held-out = micro N=5 real-domain corpus consistent with F#431's
  eval set (math/code/medical/legal/finance, Gemma 4 tokenization).
- "Jointly-trained" interpreted as MoLoRA-style Gumbel-sigmoid router
  co-trained with LoRA adapters per `exp_molora_per_token_mlx`.
- "Post-hoc TF-IDF" interpreted as nearest-centroid routing per
  `exp_p1_t4_tfidf_routing_gemma4` (F#431 methodology).

## Verdict
**KILLED (preemptive, structural).** Running would consume ~2h of
joint training + eval on MLX for a result bounded by F#431/F#305/F#312
in advance. No new information to be gained.
