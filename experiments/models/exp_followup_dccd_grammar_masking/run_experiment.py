#!/usr/bin/env python3
"""
exp_followup_dccd_grammar_masking — True token-level grammar masking for DCCD.

Parent KILL (exp_p5_dccd_format_conditioning) used re-prompting in Phase 2.
Re-prompting is bounded by the RLHF-suppressed SOAP prior (~40%) — cannot satisfy
K1268 structurally. This experiment replaces Phase 2 with a **real FSM-driven
grammar mask**: section headers ("S:", "O:", "A:", "P:") are force-emitted at
deterministic positions, content between them is sampled freely from the base
model conditioned on the Phase 1 draft.

MATH.md Theorem 1 guarantees SOAP compliance = 100% (K1558a trivially PASS).
The empirical question: does the free-content channel preserve the draft's
medical content (K1558b) without collapsing (K1558c)?

Kill criteria (DB ID 1558):
  K1558a: SOAP compliance rate >= 99% (Theorem 1)
  K1558b: Medical keyword count avg >= 7.4 (parent K1268 threshold)
  K1558c: Coherence rate >= 90% (Theorem 3 / no collapse)
"""

import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx

# MLX memory discipline (per PLAN.md Part 2, /mlx-dev conventions).
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MICRO_DIR = EXPERIMENT_DIR.parent

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 3 if IS_SMOKE else 10
MAX_DRAFT_TOKENS = 500
MAX_SECTION_TOKENS = 150  # per SOAP section in Phase 2

# Parent experiment's medical adapter (q_proj r=6) was deleted before this loop.
# We detect availability at runtime; if absent, Phase 1 falls back to base model
# and K1558b is interpreted as provisional (the structural claim — K1558a/c —
# does not depend on the draft source).
MEDICAL_ADAPTER = MICRO_DIR / "exp_p1_t2_single_domain_training" / "adapters" / "medical"
MEDICAL_ADAPTER_AVAILABLE = (MEDICAL_ADAPTER / "adapters.safetensors").exists()


def log(msg: str):
    print(msg, flush=True)


def cleanup():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ══════════════════════════════════════════════════════════════════════════════
# Scoring (mirrors parent exp_p5_dccd_format_conditioning)
# ══════════════════════════════════════════════════════════════════════════════

MEDICAL_KEYWORDS = [
    "diagnosis", "symptom", "treatment", "patient", "clinical",
    "prognosis", "etiology", "pathology", "medication", "dosage",
    "chronic", "acute", "inflammation", "infection", "therapy",
    "mg", "ml", "iv", "po", "bid", "tid", "qd",
    "hpi", "vitals", "labs", "assessment", "plan",
]


def score_soap_format(text: str) -> bool:
    t = text.lower()
    return all(bool(re.search(rf"\b{c}\s*:", t)) for c in ("s", "o", "a", "p"))


def score_medical_keywords(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in MEDICAL_KEYWORDS if kw.lower() in t)


def score_medical_domain(text: str) -> tuple[bool, int]:
    kw = score_medical_keywords(text)
    return kw >= 4, kw


def is_coherent(text: str) -> bool:
    if len(text) < 20:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / len(text) < 0.7:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    if len(set(words)) / len(words) < 0.1:
        return False
    return True


MEDICAL_QUESTIONS = [
    "Write a clinical note for a 55-year-old male with type 2 diabetes and peripheral neuropathy.",
    "Document the assessment and treatment plan for a patient with community-acquired pneumonia.",
    "Write a clinical note for a pediatric patient presenting with acute appendicitis symptoms.",
    "Document a patient encounter for a 70-year-old female with congestive heart failure exacerbation.",
    "Write a clinical assessment for a patient with major depressive disorder seeking medication adjustment.",
    "Document a clinical note for a patient presenting to the emergency department with a fractured wrist.",
    "Write a patient encounter note for chronic obstructive pulmonary disease management visit.",
    "Document a clinical note for a patient with newly diagnosed rheumatoid arthritis.",
    "Write a clinical assessment for a pregnant patient at 28 weeks with gestational diabetes.",
    "Document a clinical note for a patient with acute migraine and medication overuse headache.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Generation: subprocess mlx_lm CLI
#
# This is the parent's pattern. It is slow (spawns a model per call) but robust
# and exactly matches the parent's generation method, so the grammar-mask vs
# re-prompting comparison isolates the *decoding change* from any library
# change.
# ══════════════════════════════════════════════════════════════════════════════

def _run_mlx_generate(prompt: str, max_tokens: int, adapter_path: str | None) -> str:
    cmd = [
        "uv", "run", "python", "-m", "mlx_lm", "generate",
        "--model", MODEL_ID,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.0",
    ]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log("  WARN: generation timeout (>300s)")
        return ""
    if result.returncode != 0:
        log(f"  WARN: generation failed: {result.stderr[:200]}")
        return ""
    # mlx_lm CLI prints a short header then the generated text. Strip any
    # leading "==========" separator line and stats trailing section.
    out = result.stdout
    # Strip stats footer (lines starting with "Prompt:", "Generation:", "Peak memory:").
    lines = out.splitlines()
    keep = []
    for line in lines:
        if line.startswith(("Prompt:", "Generation:", "Peak memory:")):
            continue
        keep.append(line)
    text = "\n".join(keep).strip()
    # mlx_lm also prints "==========" separators; strip them.
    text = re.sub(r"={5,}\n?", "", text).strip()
    return text


def generate_unconstrained(question: str, adapter_path: str | None = None) -> str:
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    return _run_mlx_generate(prompt, MAX_DRAFT_TOKENS, adapter_path)


# ══════════════════════════════════════════════════════════════════════════════
# Grammar-masked Phase 2: sectional forced-header generation.
#
# Token-level grammar masking here = force SOAP section headers as the ONLY
# allowed tokens at header positions, then generate free content until the
# section-exit token sequence ("\n\n") is observed or MAX_SECTION_TOKENS hit.
# Prepending the header to the prompt is mathematically equivalent to masking
# all other tokens at that position to -inf (Theorem 1).
# ══════════════════════════════════════════════════════════════════════════════

SOAP_SECTIONS = [
    ("S", "Subjective — patient complaints and history"),
    ("O", "Objective — vitals, exam findings, labs"),
    ("A", "Assessment — diagnosis and clinical reasoning"),
    ("P", "Plan — treatment, medications, follow-up"),
]


def dccd_grammar_mask_generate(question: str, draft: str) -> str:
    """Phase 2: FSM-driven sectional generation. Forces each SOAP header,
    generates content freely between them, stops each section at a blank line.

    This IS grammar masking: at each header position, the mask A_t is the
    singleton {header_token}; inside content_X, A_t = V (vocabulary).
    """
    instruction = (
        "Rewrite the clinical information below as a SOAP note. "
        "Preserve every clinical detail, keyword, diagnosis, medication, dose, and vital. "
        "Copy medical terminology from the draft verbatim where it fits.\n\n"
        f"Clinical draft:\n{draft}\n\n"
        "SOAP note:"
    )

    output = ""
    for section_char, section_hint in SOAP_SECTIONS:
        # FSM transition: commit section header (grammar mask = {header_token}).
        output += f"\n{section_char}: "

        # Build prompt with running output; generate section content freely.
        full_prompt = (
            f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
            f"<start_of_turn>model\n{output}"
        )
        section_text = _run_mlx_generate(
            full_prompt, MAX_SECTION_TOKENS, adapter_path=None
        )

        # Robustly strip any portion after prompt echo (mlx_lm prints only
        # continuation, but guard against model emitting the next header).
        # FSM section-exit: stop at first "\n\n" OR at next section header.
        section_text = section_text.strip()

        # If model tried to write next section itself, truncate at the boundary.
        for next_char, _ in SOAP_SECTIONS:
            if next_char == section_char:
                continue
            # Look for the next-header pattern on its own line
            m = re.search(rf"(?m)^\s*{next_char}\s*:", section_text)
            if m:
                section_text = section_text[:m.start()].rstrip()

        # Stop on blank line (FSM section-exit signal).
        blank = section_text.find("\n\n")
        if blank >= 0:
            section_text = section_text[:blank].rstrip()

        # Strip any <end_of_turn> artifact.
        section_text = section_text.replace("<end_of_turn>", "").strip()

        output += section_text

    return output.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Keyword transfer measurement (for K1558b / P5)
# ══════════════════════════════════════════════════════════════════════════════

def draft_keyword_transfer(draft: str, output: str) -> float:
    """Fraction of medical keywords in draft that also appear in output."""
    draft_kw = {kw for kw in MEDICAL_KEYWORDS if kw.lower() in draft.lower()}
    if not draft_kw:
        return 0.0
    out_lower = output.lower()
    preserved = sum(1 for kw in draft_kw if kw.lower() in out_lower)
    return preserved / len(draft_kw)


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("exp_followup_dccd_grammar_masking — True Token-Level Grammar Masking")
    log(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}")
    log("=" * 70)

    t0 = time.time()
    cleanup()
    log_memory("start")

    if MEDICAL_ADAPTER_AVAILABLE:
        log(f"  Found medical adapter: {MEDICAL_ADAPTER}")
        adapter_arg = str(MEDICAL_ADAPTER)
    else:
        log(f"  NOTICE: medical adapter safetensors missing at {MEDICAL_ADAPTER}. "
            f"Falling back to base model for Phase 1. K1558b will be provisional.")
        adapter_arg = None

    questions = MEDICAL_QUESTIONS[:N_EVAL]
    per_q = []

    for i, q in enumerate(questions):
        log(f"\n--- Q{i+1}/{N_EVAL}: {q[:60]}...")

        # Phase 1: medical adapter (if available) generates unconstrained draft.
        t_p1 = time.time()
        draft = generate_unconstrained(q, adapter_path=adapter_arg)
        t_p1 = time.time() - t_p1
        if not draft or not is_coherent(draft):
            log(f"  Phase 1 failed (empty or incoherent draft) in {t_p1:.1f}s")
            per_q.append({"question": q, "draft": draft, "phase2": "",
                          "phase1_failed": True})
            continue
        draft_kw = score_medical_keywords(draft)
        log(f"  Phase 1 draft: {len(draft)}c, {draft_kw} medical kw ({t_p1:.1f}s)")

        # Phase 2: grammar-masked sectional generation.
        t_p2 = time.time()
        formatted = dccd_grammar_mask_generate(q, draft)
        t_p2 = time.time() - t_p2
        out_kw = score_medical_keywords(formatted)
        soap_ok = score_soap_format(formatted)
        coherent = is_coherent(formatted)
        transfer = draft_keyword_transfer(draft, formatted)
        log(f"  Phase 2 output: {len(formatted)}c, {out_kw} medical kw, "
            f"SOAP={soap_ok}, coherent={coherent}, transfer={transfer:.2f} "
            f"({t_p2:.1f}s)")

        per_q.append({
            "question": q,
            "draft": draft,
            "draft_keywords": draft_kw,
            "phase2": formatted,
            "phase2_keywords": out_kw,
            "soap_ok": soap_ok,
            "coherent": coherent,
            "transfer": transfer,
            "phase1_time_s": t_p1,
            "phase2_time_s": t_p2,
        })
        cleanup()

    # ─── Aggregate ────────────────────────────────────────────────────────
    valid = [x for x in per_q if not x.get("phase1_failed", False)]
    n = len(valid)
    if n == 0:
        log("\nFATAL: all Phase 1 drafts failed — cannot evaluate grammar masking.")
        results = {
            "verdict": "KILLED",
            "is_smoke": IS_SMOKE,
            "n_eval": N_EVAL,
            "error": "all Phase 1 drafts failed",
            "per_q": per_q,
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    soap_rate = sum(1 for x in valid if x["soap_ok"]) / n
    coherent_rate = sum(1 for x in valid if x["coherent"]) / n
    avg_kw = sum(x["phase2_keywords"] for x in valid) / n
    avg_transfer = sum(x["transfer"] for x in valid) / n
    avg_draft_kw = sum(x["draft_keywords"] for x in valid) / n

    # ─── KC evaluation ───────────────────────────────────────────────────
    k1558a = soap_rate >= 0.99
    k1558b = avg_kw >= 7.4
    k1558c = coherent_rate >= 0.90
    structural_pass = k1558a and k1558c
    all_pass = k1558a and k1558b and k1558c
    if not MEDICAL_ADAPTER_AVAILABLE:
        # K1558b conditional on trained medical adapter; if missing, the test
        # reduces to a structural check (K1558a + K1558c). Verdict is
        # provisional regardless of metric outcome.
        verdict = "provisional"
    else:
        verdict = "supported" if all_pass else "KILLED"

    log("\n" + "=" * 70)
    log("KILL CRITERIA (MATH.md)")
    log("=" * 70)
    log(f"  K1558a SOAP compliance    : {soap_rate:.2%} "
        f"(>=99%) {'PASS' if k1558a else 'FAIL'}")
    log(f"  K1558b Avg medical kw     : {avg_kw:.2f} "
        f"(>=7.4) {'PASS' if k1558b else 'FAIL'}")
    log(f"  K1558c Coherence rate     : {coherent_rate:.2%} "
        f"(>=90%) {'PASS' if k1558c else 'FAIL'}")
    log(f"  P5  Draft→output transfer : {avg_transfer:.2f} (>=0.60 prediction)")
    log(f"  Phase1 avg medical kw     : {avg_draft_kw:.2f}")
    log(f"\n  VERDICT: {verdict} (all_pass={all_pass})")
    log(f"  Total time: {time.time() - t0:.1f}s")

    results = {
        "verdict": verdict,
        "all_pass": all_pass,
        "structural_pass": structural_pass,
        "medical_adapter_available": MEDICAL_ADAPTER_AVAILABLE,
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "metrics": {
            "soap_rate": soap_rate,
            "coherent_rate": coherent_rate,
            "avg_medical_keywords": avg_kw,
            "avg_draft_keywords": avg_draft_kw,
            "avg_transfer": avg_transfer,
        },
        "kill_criteria": {
            "K1558a_soap_gte_99": {"value": soap_rate, "threshold": 0.99, "pass": k1558a},
            "K1558b_kw_gte_7_4": {"value": avg_kw, "threshold": 7.4, "pass": k1558b},
            "K1558c_coherent_gte_90": {"value": coherent_rate, "threshold": 0.90, "pass": k1558c},
        },
        "per_q": per_q,
        "total_time_s": time.time() - t0,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Wrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
