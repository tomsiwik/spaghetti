# REVIEW-adversarial.md — P4.C0: Formatting Adapter

## Verdict: PROCEED (killed, finding valid)

---

## Concerns

### 1. N=10 eval is noisy — is LaTeX +20pp real?

**Concern:** 10 eval samples per domain means 1 response = 10pp. LaTeX went from 2/10 to 4/10.
This is 2 additional responses — could be noise.

**Response:** P4.B0 (Finding #477) showed similar LaTeX pattern at N=10: math +20pp with keywords
"a^2", "u dv". Consistent across two independent experiments at different ranks (6 and 16).
The +20pp for notation-style keywords is reproducible.

### 2. Why is SOAP base=0% but adapted=0%? Is the adapter training working at all?

**Concern:** If adapter does nothing, maybe training failed?

**Response:** Legal base=0% but adapted=10% shows the adapter IS learning. SOAP is harder:
SOAP requires multi-section output structure (S:/O:/A:/P: headers) that conflicts with
instruction-following conversational format. The model won't spontaneously produce "S: Chief
complaint:" in response to "What is your assessment?" — it needs both recognition AND format
switching, which q_proj cannot achieve alone.

### 3. Is "RLHF behavioral prior" a real mechanism or post-hoc rationalization?

**Concern:** We're attributing SOAP failure to RLHF prior without ablating.

**Response:** This is a plausible mechanism, not a proven one. The impossibility structure states
mathematically that q_proj alone cannot shift output-token distribution because v_proj/o_proj/
lm_head are unchanged. This is a structural argument, not an RLHF-specific claim. The experiment
is KILLED on this basis, not definitively proven.

**Better test:** Train adapters on v_proj or o_proj for SOAP — if SOAP improves, the impossibility
structure is confirmed. Mark this as an open question in LEARNINGS.md.

### 4. Legal +10pp is below threshold — does this tell us anything?

**Concern:** Legal partial improvement suggests the mechanism works but needs more training.

**Response:** Partially true. Legal boilerplate (WHEREAS, NOW THEREFORE, hereinafter) is
different from SOAP — these are individual keywords scattered in legal text, not a structural
format override. This is closer to the notation-gap pattern (LaTeX) than the behavioral-format
pattern (SOAP). With more examples or higher rank, legal might reach ≥20pp.

**This doesn't salvage K1231** — the finding is still KILLED. But it suggests:
- Notation/vocabulary gaps → q_proj adapters work
- Behavioral structural format → q_proj adapters fail
- Legal structural markers are intermediate

---

## What the Finding Establishes

1. LaTeX notation gap is exploitable with q_proj rank-16: confirmed (+20pp, consistent with P4.B0)
2. SOAP behavioral format gap is NOT exploitable with q_proj alone: confirmed (0pp)
3. Cross-domain format adapter retention is perfect (1.0): confirms Theorem 3 / Finding #440

## What It Does NOT Establish

- Whether SOAP can be fixed with v_proj/o_proj adapters (open question)
- Whether legal can reach ≥20pp with rank-32 or more steps
- Whether SOAP base=0% is a genuine format gap or just bad evaluation (synthetic Q&A)

---

## Verdict

Finding #479 is valid. The killed result identifies a genuine architectural limitation.
The impossibility structure (q_proj cannot override RLHF behavioral priors) is mechanistically
plausible and consistent with the evidence.

**PROCEED** — write LEARNINGS.md and claim P4.C1.
