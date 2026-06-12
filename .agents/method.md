# method — the shared research method (one source of truth)

Vendor-neutral. Every hat references this; none restate it. Principles, not a checklist — the goal is to
*learn*, not to satisfy a rule. Keep the research **open**: do not straitjacket a hypothesis to any single
past finding or metric convention.

## Non-negotiable (integrity)
- **Real, never mock.** A verdict requires real executed code on the real model/data. `results.json`
  must have `is_smoke:false`. If you cannot run real code, mark it `provisional` with a one-line reason —
  never fabricate, never hardcode a pass, never substitute a different model/adapter as a stand-in.
- **Falsifiable.** Pre-register a prediction and a numeric threshold you would accept as refutation,
  *before* the run. If the data crosses it, the verdict is `killed` — don't move the goalposts after seeing data.
- **Proof-first (lightweight).** A short theorem/prediction sketch in `MATH.md` before code: what failure
  is being prevented, what number the idea predicts, why. Cite a paper or prior result if one genuinely
  grounds it — but a novel hunch needs no permission slip.

## Good judgment (not dogma)
- **Prefer behavioral outcomes over proxies.** A metric moving without a behavioral change is weak
  evidence — but use judgment about which signal matters for *this* question; there is no one mandatory rule.
- **Composition is `Σ (Bᵢ @ Aᵢ)`**, never `(ΣB)(ΣA)`. `LORA_SCALE ≤ 8`. Route per-sample, not per-domain.
- **Real MLX via `pueue`** (`experiment run`), never bare python. Phased execution, clear caches between phases.

## Verdict
`supported` (prediction met, real) · `killed` (prediction refuted, real) · `provisional` (smoke / awaiting
proof / blocked). `is_smoke:true` can only be `provisional`. The reviewer is the sole gate that seals a verdict.
