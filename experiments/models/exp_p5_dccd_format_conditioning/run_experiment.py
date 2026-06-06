#!/usr/bin/env python3
"""
P5.A1: DCCD Format Conditioning — Fix #483 Cross-Projection Failure

Finding #483 proved that weight-space composition of domain (q_proj) + format
(v_proj+o_proj) adapters causes catastrophic model collapse despite zero
parameter overlap. The functional coupling through the attention pipeline
(Q determines attention, O determines output) creates compound perturbation
when adapters trained independently are combined.

DCCD (arXiv:2603.03305) avoids this by temporal separation:
  Phase 1: Domain adapter generates semantically correct draft (unconstrained)
  Phase 2: Base model reformats draft into target structure (adapter REMOVED)

The domain adapter is REMOVED during Phase 2. Zero cross-projection interference
because only ONE adapter is active at any time.

Kill criteria (DB IDs):
  K1267: SOAP format compliance >= 70% (matching P4.C1 v_proj+o_proj result)
  K1268: Domain quality preserved (< 5pp degradation vs domain-only adapter)
  K1269: No catastrophic collapse (unlike #483 weight-space cross-projection)

Grounded by:
  - Finding #483: Cross-projection composition catastrophe (q_proj + o_proj)
  - Finding #480: v_proj+o_proj achieves SOAP +70pp
  - Finding #421: q_proj achieves +22pp medical domain knowledge
  - arXiv:2603.03305: Draft-Conditioned Constrained Decoding
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

# Memory safety (CODING_GUIDELINES §2)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MICRO_DIR = EXPERIMENT_DIR.parent

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 3 if IS_SMOKE else 10
MAX_TOKENS = 500
SEED = 42

# Pre-trained adapter paths (reuse existing — no training needed)
MEDICAL_ADAPTER = MICRO_DIR / "exp_p1_t2_single_domain_training" / "adapters" / "medical"
SOAP_ADAPTER = MICRO_DIR / "exp_p4_c1_vproj_soap_adapter" / "soap_adapter"


def cleanup():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log(msg: str):
    print(msg, flush=True)


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ══════════════════════════════════════════════════════════════════════════════
# Adapter merging (from P4.D0 — for weight-space composition baseline)
# ══════════════════════════════════════════════════════════════════════════════

def merge_adapters(adapter_a_path: Path, adapter_b_path: Path, out_path: Path) -> Path:
    """Merge two disjoint LoRA adapters with heterogeneous ranks.

    Pre-bakes scale into lora_b, pads smaller-rank to max_rank with zeros.
    mlx_lm LoRA forward: y + scale * (x @ lora_a) @ lora_b
    """
    weights_a = mx.load(str(adapter_a_path / "adapters.safetensors"))
    weights_b = mx.load(str(adapter_b_path / "adapters.safetensors"))

    overlap = set(weights_a.keys()) & set(weights_b.keys())
    assert len(overlap) == 0, f"Adapters overlap on {len(overlap)} keys: {overlap}"

    config_a = json.loads((adapter_a_path / "adapter_config.json").read_text())
    config_b = json.loads((adapter_b_path / "adapter_config.json").read_text())

    rank_a = config_a["lora_parameters"]["rank"]
    rank_b = config_b["lora_parameters"]["rank"]
    scale_a = config_a["lora_parameters"]["scale"]
    scale_b = config_b["lora_parameters"]["scale"]
    max_rank = max(rank_a, rank_b)

    log(f"  Adapter A: rank={rank_a}, scale={scale_a}, keys={len(weights_a)}")
    log(f"  Adapter B: rank={rank_b}, scale={scale_b}, keys={len(weights_b)}")

    def process_weights(weights: dict, rank: int, scale: float) -> dict:
        processed = {}
        for key, val in weights.items():
            if key.endswith(".lora_a"):
                if rank < max_rank:
                    pad_width = max_rank - rank
                    padding = mx.zeros((val.shape[0], pad_width), dtype=val.dtype)
                    val = mx.concatenate([val, padding], axis=1)
                processed[key] = val
            elif key.endswith(".lora_b"):
                val = val * scale
                if rank < max_rank:
                    pad_width = max_rank - rank
                    padding = mx.zeros((pad_width, val.shape[1]), dtype=val.dtype)
                    val = mx.concatenate([val, padding], axis=0)
                processed[key] = val
            else:
                processed[key] = val
        return processed

    processed_a = process_weights(weights_a, rank_a, scale_a)
    processed_b = process_weights(weights_b, rank_b, scale_b)
    mx.eval(processed_a, processed_b)

    merged = {**processed_a, **processed_b}
    out_path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_path / "adapters.safetensors"), merged)

    lora_keys_a = config_a["lora_parameters"]["keys"]
    lora_keys_b = config_b["lora_parameters"]["keys"]
    merged_keys = sorted(set(lora_keys_a + lora_keys_b))

    nl_a = config_a.get("num_layers", -1)
    nl_b = config_b.get("num_layers", -1)
    merged_num_layers = -1 if (nl_a == -1 or nl_b == -1) else max(nl_a, nl_b)

    merged_config = {
        **config_a,
        "adapter_path": str(out_path),
        "num_layers": merged_num_layers,
        "lora_parameters": {
            "rank": max_rank,
            "scale": 1.0,
            "dropout": 0.0,
            "keys": merged_keys,
        },
    }
    (out_path / "adapter_config.json").write_text(json.dumps(merged_config, indent=4))

    log(f"  Merged {len(weights_a)} + {len(weights_b)} keys → {len(merged)} "
        f"(keys: {merged_keys}, rank={max_rank}, scale=1.0 [pre-baked])")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Generation helper
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(question: str, adapter_path: str | None = None) -> str:
    """Generate a single response from the model via mlx_lm CLI."""
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    cmd = [
        "uv", "run", "python", "-m", "mlx_lm", "generate",
        "--model", MODEL_ID,
        "--prompt", prompt,
        "--max-tokens", str(MAX_TOKENS),
        "--temp", "0.0",
    ]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        log(f"  WARN: generation failed: {result.stderr[:200]}")
        return ""
    return result.stdout.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Scoring functions
# ══════════════════════════════════════════════════════════════════════════════

MEDICAL_KEYWORDS = [
    "diagnosis", "symptom", "treatment", "patient", "clinical",
    "prognosis", "etiology", "pathology", "medication", "dosage",
    "chronic", "acute", "inflammation", "infection", "therapy",
    "mg", "ml", "iv", "po", "bid", "tid", "qd",
    "hpi", "vitals", "labs", "assessment", "plan",
]


def score_soap_format(text: str) -> bool:
    """True if response has all 4 SOAP sections."""
    text_lower = text.lower()
    has_s = bool(re.search(r'\bs\s*:', text_lower))
    has_o = bool(re.search(r'\bo\s*:', text_lower))
    has_a = bool(re.search(r'\ba\s*:', text_lower))
    has_p = bool(re.search(r'\bp\s*:', text_lower))
    return has_s and has_o and has_a and has_p


def score_medical_domain(text: str) -> tuple[bool, int]:
    """Score medical domain vocabulary. Returns (pass, keyword_count)."""
    text_lower = text.lower()
    hits = sum(1 for kw in MEDICAL_KEYWORDS if kw.lower() in text_lower)
    return hits >= 4, hits


def is_coherent(text: str) -> bool:
    """Detect catastrophic collapse (garbage output per #483)."""
    if len(text) < 20:
        return False
    # Check for non-ASCII dominance (Hindi garbage from #483)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / len(text) if text else 0
    if ascii_ratio < 0.7:
        return False
    # Check for excessive repetition
    words = text.split()
    if len(words) < 5:
        return False
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio < 0.1:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation questions
# ══════════════════════════════════════════════════════════════════════════════

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
# DCCD: Draft-Conditioned Constrained Decoding
# ══════════════════════════════════════════════════════════════════════════════

def dccd_generate(question: str, domain_adapter_path: str) -> tuple[str, str]:
    """Two-phase DCCD: domain draft → format-constrained re-generation.

    Phase 1: Generate unconstrained draft with domain adapter (clinical content)
    Phase 2: Re-prompt BASE model (no adapter) to reformat into SOAP structure

    Returns (draft, formatted_output).
    """
    # Phase 1: Domain adapter generates semantically correct draft
    draft = generate_response(question, adapter_path=domain_adapter_path)

    if not draft or not is_coherent(draft):
        log(f"    DCCD Phase 1 failed (empty or incoherent draft)")
        return draft, ""

    # Phase 2: Base model (NO adapter) reformats into SOAP structure
    # The domain adapter is REMOVED — no cross-projection interference
    format_prompt = (
        "Reformat the following clinical information into a proper SOAP note. "
        "You MUST use exactly these four section headers:\n"
        "S: (Subjective - patient's complaints, history)\n"
        "O: (Objective - vitals, exam findings, labs)\n"
        "A: (Assessment - diagnoses with ICD-10 codes)\n"
        "P: (Plan - treatments, follow-up)\n\n"
        "Clinical information to reformat:\n"
        f"{draft}\n\n"
        "SOAP Note:"
    )

    formatted = generate_response(format_prompt, adapter_path=None)
    return draft, formatted


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_condition(
    name: str,
    questions: list[str],
    n_eval: int,
    generate_fn,
) -> dict:
    """Evaluate a condition on medical questions. generate_fn returns response text."""
    questions = questions[:n_eval]

    soap_scores = []
    domain_scores = []
    coherent_scores = []
    keyword_counts = []
    responses = []

    for i, q in enumerate(questions):
        resp = generate_fn(q)
        responses.append(resp)

        soap_pass = score_soap_format(resp)
        domain_pass, kw_count = score_medical_domain(resp)
        coherent = is_coherent(resp)

        soap_scores.append(soap_pass)
        domain_scores.append(domain_pass)
        coherent_scores.append(coherent)
        keyword_counts.append(kw_count)

        log(f"  [{name}] Q{i+1}: SOAP={'Y' if soap_pass else 'N'} "
            f"domain={'Y' if domain_pass else 'N'}({kw_count}kw) "
            f"coherent={'Y' if coherent else 'N'} — {q[:50]}...")

    soap_rate = sum(soap_scores) / len(soap_scores)
    domain_rate = sum(domain_scores) / len(domain_scores)
    coherent_rate = sum(coherent_scores) / len(coherent_scores)
    avg_keywords = sum(keyword_counts) / len(keyword_counts)

    log(f"  [{name}] RESULTS: soap={soap_rate:.0%} domain={domain_rate:.0%} "
        f"coherent={coherent_rate:.0%} avg_kw={avg_keywords:.1f}")

    return {
        "name": name,
        "soap_rate": soap_rate,
        "domain_rate": domain_rate,
        "coherent_rate": coherent_rate,
        "avg_keywords": avg_keywords,
        "n_eval": n_eval,
        "responses": responses[:3],  # save first 3 for inspection
    }


def main():
    log("=" * 70)
    log("P5.A1: DCCD Format Conditioning — Fix #483 Cross-Projection Failure")
    log(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # Verify adapters exist
    for name, path in [("medical", MEDICAL_ADAPTER), ("soap", SOAP_ADAPTER)]:
        safetensors = path / "adapters.safetensors"
        assert safetensors.exists(), f"Missing {name} adapter at {safetensors}"
        log(f"  Found {name} adapter: {path}")

    # ─── Phase 1: Base model baseline ────────────────────────────────────
    log("\n=== Condition A: Base Model (no adapter) ===")
    result_base = evaluate_condition(
        "base",
        MEDICAL_QUESTIONS,
        N_EVAL,
        lambda q: generate_response(q, adapter_path=None),
    )
    cleanup()
    log_memory("after-base")

    # ─── Phase 2: Medical adapter only ───────────────────────────────────
    log("\n=== Condition B: Medical Adapter Only (q_proj) ===")
    result_medical = evaluate_condition(
        "medical-only",
        MEDICAL_QUESTIONS,
        N_EVAL,
        lambda q: generate_response(q, adapter_path=str(MEDICAL_ADAPTER)),
    )
    cleanup()
    log_memory("after-medical")

    # ─── Phase 3: Weight-space composition (reproduce #483) ──────────────
    log("\n=== Condition C: Weight-Space Composition (medical+SOAP) ===")
    merged_path = EXPERIMENT_DIR / "merged_medical_soap"
    log("  Merging medical (q_proj r6) + SOAP (v_proj+o_proj r16)...")
    merge_adapters(MEDICAL_ADAPTER, SOAP_ADAPTER, merged_path)
    cleanup()

    result_composed = evaluate_condition(
        "weight-composed",
        MEDICAL_QUESTIONS,
        N_EVAL,
        lambda q: generate_response(q, adapter_path=str(merged_path)),
    )
    cleanup()
    log_memory("after-composed")

    # ─── Phase 4: DCCD (temporal separation) ─────────────────────────────
    log("\n=== Condition D: DCCD (Draft → Format, temporal separation) ===")
    dccd_drafts = []
    dccd_formatted = []

    questions = MEDICAL_QUESTIONS[:N_EVAL]
    soap_scores = []
    domain_scores = []
    coherent_scores = []
    keyword_counts = []

    for i, q in enumerate(questions):
        draft, formatted = dccd_generate(q, str(MEDICAL_ADAPTER))
        dccd_drafts.append(draft)
        dccd_formatted.append(formatted)

        # Score the FORMATTED output (Phase 2 result)
        soap_pass = score_soap_format(formatted)
        domain_pass, kw_count = score_medical_domain(formatted)
        coherent = is_coherent(formatted)

        soap_scores.append(soap_pass)
        domain_scores.append(domain_pass)
        coherent_scores.append(coherent)
        keyword_counts.append(kw_count)

        # Also check draft quality
        draft_domain, draft_kw = score_medical_domain(draft)

        log(f"  [dccd] Q{i+1}: draft_domain={'Y' if draft_domain else 'N'}({draft_kw}kw) "
            f"→ formatted: SOAP={'Y' if soap_pass else 'N'} "
            f"domain={'Y' if domain_pass else 'N'}({kw_count}kw) "
            f"coherent={'Y' if coherent else 'N'}")

    result_dccd = {
        "name": "dccd",
        "soap_rate": sum(soap_scores) / len(soap_scores),
        "domain_rate": sum(domain_scores) / len(domain_scores),
        "coherent_rate": sum(coherent_scores) / len(coherent_scores),
        "avg_keywords": sum(keyword_counts) / len(keyword_counts),
        "n_eval": N_EVAL,
        "responses": dccd_formatted[:3],
        "drafts": dccd_drafts[:3],
    }
    log(f"  [dccd] RESULTS: soap={result_dccd['soap_rate']:.0%} "
        f"domain={result_dccd['domain_rate']:.0%} "
        f"coherent={result_dccd['coherent_rate']:.0%} "
        f"avg_kw={result_dccd['avg_keywords']:.1f}")
    cleanup()
    log_memory("after-dccd")

    # ─── Phase 5: SOAP adapter only (format reference) ───────────────────
    log("\n=== Condition E: SOAP Adapter Only (v_proj+o_proj, reference) ===")
    result_soap_only = evaluate_condition(
        "soap-only",
        MEDICAL_QUESTIONS,
        N_EVAL,
        lambda q: generate_response(q, adapter_path=str(SOAP_ADAPTER)),
    )
    cleanup()
    log_memory("after-soap-only")

    # ─── Kill criteria evaluation ────────────────────────────────────────
    log("\n" + "=" * 70)
    log("KILL CRITERIA EVALUATION")
    log("=" * 70)

    # K1267: SOAP format compliance >= 70%
    dccd_soap = result_dccd["soap_rate"] * 100
    k1267_pass = dccd_soap >= 70.0
    log(f"\nK1267 (DCCD SOAP compliance >= 70%):")
    log(f"  dccd_soap_rate={dccd_soap:.1f}% >= 70%? {'PASS' if k1267_pass else 'FAIL'}")
    log(f"  Comparison: base={result_base['soap_rate']:.0%}, "
        f"soap-only={result_soap_only['soap_rate']:.0%}, "
        f"composed={result_composed['soap_rate']:.0%}")

    # K1268: Domain quality preserved (< 5pp degradation vs domain-only)
    medical_only_domain = result_medical["domain_rate"] * 100
    dccd_domain = result_dccd["domain_rate"] * 100
    domain_degradation = medical_only_domain - dccd_domain
    k1268_pass = domain_degradation < 5.0
    log(f"\nK1268 (Domain quality < 5pp degradation vs domain-only):")
    log(f"  medical_only={medical_only_domain:.1f}%, dccd={dccd_domain:.1f}%, "
        f"degradation={domain_degradation:.1f}pp < 5pp? {'PASS' if k1268_pass else 'FAIL'}")

    # K1269: No catastrophic collapse
    dccd_coherent = result_dccd["coherent_rate"] * 100
    composed_coherent = result_composed["coherent_rate"] * 100
    k1269_pass = dccd_coherent >= 90.0
    log(f"\nK1269 (No catastrophic collapse, coherence >= 90%):")
    log(f"  dccd_coherent={dccd_coherent:.1f}% >= 90%? {'PASS' if k1269_pass else 'FAIL'}")
    log(f"  Comparison: composed_coherent={composed_coherent:.1f}% (should be near 0% per #483)")

    all_pass = k1267_pass and k1268_pass and k1269_pass
    total_min = (time.time() - total_start) / 60.0

    log(f"\n{'='*70}")
    log(f"SUMMARY")
    log(f"  K1267 (SOAP >= 70%): {'PASS' if k1267_pass else 'FAIL'} ({dccd_soap:.1f}%)")
    log(f"  K1268 (domain < 5pp deg): {'PASS' if k1268_pass else 'FAIL'} ({domain_degradation:.1f}pp)")
    log(f"  K1269 (no collapse): {'PASS' if k1269_pass else 'FAIL'} ({dccd_coherent:.1f}%)")
    log(f"  ALL_PASS={all_pass}")
    log(f"  Total time: {total_min:.2f} min")
    log(f"{'='*70}")

    # ─── Comparison table ─────────────────────────────────────────────────
    log("\n  CONDITION COMPARISON TABLE:")
    log(f"  {'Condition':<25} {'SOAP%':>8} {'Domain%':>8} {'Coherent%':>10} {'AvgKW':>8}")
    log(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for r in [result_base, result_medical, result_soap_only, result_composed, result_dccd]:
        log(f"  {r['name']:<25} {r['soap_rate']*100:>7.1f}% {r['domain_rate']*100:>7.1f}% "
            f"{r['coherent_rate']*100:>9.1f}% {r['avg_keywords']:>7.1f}")

    # ─── Save results ─────────────────────────────────────────────────────
    results = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "conditions": {
            "base": result_base,
            "medical_only": result_medical,
            "soap_only": result_soap_only,
            "weight_composed": result_composed,
            "dccd": result_dccd,
        },
        "kill_criteria": {
            "k1267_soap_compliance": {
                "pass": k1267_pass,
                "dccd_soap_rate": dccd_soap,
                "threshold": 70.0,
            },
            "k1268_domain_preservation": {
                "pass": k1268_pass,
                "medical_only_domain": medical_only_domain,
                "dccd_domain": dccd_domain,
                "degradation_pp": domain_degradation,
                "threshold_pp": 5.0,
            },
            "k1269_no_collapse": {
                "pass": k1269_pass,
                "dccd_coherent": dccd_coherent,
                "composed_coherent": composed_coherent,
                "threshold": 90.0,
            },
        },
        "all_pass": all_pass,
        "total_time_min": round(total_min, 2),
        "theorem_verification": {
            "theorem_1_projection_tax": (
                "DCCD avoids per-token normalization divergence by separating "
                "domain content (Phase 1) from format structure (Phase 2)"
            ),
            "theorem_2_temporal_separation": {
                "verified": k1269_pass,
                "evidence": (
                    f"DCCD coherence={dccd_coherent:.1f}% vs "
                    f"weight-composed coherence={composed_coherent:.1f}% — "
                    f"temporal separation prevents #483 cross-projection failure"
                ),
            },
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
