# LEARNINGS.md — exp_followup_output_space_qa_adapters

## TL;DR
Preemptive KILL. The KC measures the wrong thing. Five reusable learnings:

1. **Inter-variant delta KCs are only informative when anchored.** K1552 compares
   QA-format-adapter-top2 to NTP-format-adapter-top2 with no reference to base.
   Generic rule: every inter-variant delta KC should be paired with a
   base-anchor KC, or the design should be rejected pre-run.

2. **Format-fix is a symptom-level remedy.** F#165's root cause was dual: NTP
   adapters emit wrong format AND degrade base quality. Fixing format alone
   leaves the quality lag intact. Always check which of the cited root causes a
   proposed fix actually addresses.

3. **F#166's prerequisite gate ("single adapter must beat base before
   composition") is a reusable structural lemma.** Any output-space composition
   experiment that does not pre-register this gate is structurally malformed.

4. **Bundling orthogonal fixes into one KC destroys attribution.** K1552 bundles
   format-alignment and KV-cache-aware implementation. Even a PASS could not be
   attributed to either. Split into separate KCs or separate experiments.

5. **Runtime LoRA composition is already output-space MoE (F#167/F#168).** The
   binding constraint on these experiments is base model quality per domain, not
   composition architecture. Further variants of OS-top2 that don't raise the
   per-adapter quality ceiling are preempt candidates.

## Tripwire (analyst-owed)
**Name:** `tautological-inter-adapter-delta-ignores-base-baseline`
**Test:** When a KC has the form `Q(variant_A) − Q(variant_B) ≥ δ` and neither
variant is the base, check whether one variant is format- or scale-incompatible
by construction. If yes, the KC is tautological — reject the design, do not run.
**Reusable across:** any routing/composition experiment comparing adapter
variants pairwise.

## Design debt (not acted on this iter — would require new experiment)
- Pre-register base-beat gate for every OS-top2 design going forward.
- Retire future experiments motivated solely by "fix the NTP/QA format mismatch"
  — the format-fix arc is closed by this preempt.
