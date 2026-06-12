# bets — the named research programs (what the loop is FOR)

A **bet** is a multi-experiment ladder toward one architectural breakthrough, feeding one
`bet/<name>` branch in `../pierre`. Experiments exist to climb rungs; rungs exist to promote
branches. **Progress = pierre commits a finding forced, not findings/week.**

## The league (pierre branching)
```
bet/<name>      experimental — rung results land here as commits (shipper hat)
candidate/<name> cut when the bet GATE passes (wins its target domain, no regression >2pp elsewhere)
main            champion — a candidate is merged when it beats main on the FULL fixed suite
```
Every branch is scored by the same behavioral suite (pierre's harness); scores are recorded as
DB evidence tagged `league`. Branches compete: the table lives in `../pierre/LEAGUE.md`.

## Active bets
| bet | thesis (one line) | pierre branch | status |
|---|---|---|---|
| [dfa-init](dfa-init.md) | interference lives on the B/output side — make adapters disjoint there **by construction** | `bet/dfa-init` | R1 queued |
| [jury-decode](jury-decode.md) | small + verifier-guided search beats frontier on checkable tasks; pierre's adapter bank IS a decorrelated jury | `bet/jury-decode` | R1 queued |
| [simplex-routing](simplex-routing.md) | per-query simplex search over a shared-basis bank dominates any static merge | `bet/simplex-routing` | gated on dfa-init R2 |

## Rules
- **Ship-or-shelve.** Every `supported` bet finding ends as a pierre commit on its bet branch
  (shipper hat) or an explicit `PIERRE-IMPACT: shelved — <why>` in the finding. No third state.
- **Rungs are questions, mechanisms are open.** The sparker's novelty is *how* to pass the rung,
  not *whether* to work on it. Frame-breaks within a rung are the point.
- **Wildcat quota.** Every 4th spark may ignore the ladders entirely (a free frame-break) —
  but must cite a ≥2025 arxiv result or a measured anomaly as its seed. No wildcats into
  arcs the DB already killed (merge-tuning on frozen adapters is CLOSED — F#827/837/844).
- **Gates are pre-registered** in each bet file. Crossing a kill gate kills the *rung*, not the
  bet; two consecutive dead rungs with no v2 idea kills the bet (analyst files the obituary).
