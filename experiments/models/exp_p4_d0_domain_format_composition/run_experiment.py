#!/usr/bin/env python3
"""
P4.D0: Domain + Format Adapter Simultaneous Composition

Tests whether domain adapters (q_proj) and format adapters (v_proj+o_proj) compose
without interference when applied simultaneously via additive weight merging.

Key insight: domain and format adapters target COMPLETELY DISJOINT parameter sets
(q_proj vs v_proj+o_proj). Theorem 1 (MATH.md) proves zero parameter overlap.
This means composition = dictionary concatenation of safetensor weights.

Kill criteria (DB IDs):
  K1249: Medical + SOAP: domain_quality >= 40% AND format_compliance >= 50pp
  K1250: Legal + Legal-brief: domain_quality >= 40% AND format_compliance >= 60pp
  K1251: Solo adapter degradation <= 15pp under 2-adapter composition

Grounded by:
  - Finding #421: q_proj domain adapters (medical +22pp, legal +50pp)
  - Finding #480: v_proj+o_proj format adapters (SOAP +70pp, Legal +90pp)
  - Finding #440: Grassmannian isolation (max cos = 2.25e-8 at N=100)
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

# Pre-trained adapter paths (no training needed — reuse existing)
ADAPTERS = {
    "medical_domain": MICRO_DIR / "exp_p1_t2_single_domain_training" / "adapters" / "medical",
    "legal_domain": MICRO_DIR / "exp_p1_t2_multi_domain_5" / "adapters" / "legal",
    "soap_format": MICRO_DIR / "exp_p4_c1_vproj_soap_adapter" / "soap_adapter",
    "legal_format": MICRO_DIR / "exp_p4_c1_vproj_soap_adapter" / "legal_adapter",
}

# Composition pairs: (domain_adapter, format_adapter)
COMPOSITION_PAIRS = {
    "medical_soap": ("medical_domain", "soap_format"),
    "legal_legalbrief": ("legal_domain", "legal_format"),
}


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
# Adapter merging
# ══════════════════════════════════════════════════════════════════════════════

def merge_adapters(adapter_a_path: Path, adapter_b_path: Path, out_path: Path) -> Path:
    """Merge two disjoint LoRA adapters with heterogeneous ranks.

    Handles rank mismatch by:
    1. Pre-baking each adapter's scale into lora_b weights
    2. Padding smaller-rank weights to max_rank with zeros
    3. Using unified scale=1.0 in the merged config

    mlx_lm LoRA forward: y + scale * (x @ lora_a) @ lora_b
    Pre-baking: lora_b_new = lora_b * original_scale, then merged scale=1.0
    Padding: lora_a (in, r) → (in, R) with zero cols; lora_b (r, out) → (R, out) with zero rows
    This preserves: 1.0 * (x @ A_pad) @ B_pad_scaled = original_scale * (x @ A) @ B
    """
    weights_a = mx.load(str(adapter_a_path / "adapters.safetensors"))
    weights_b = mx.load(str(adapter_b_path / "adapters.safetensors"))

    # Verify disjointness
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
        """Pre-bake scale into lora_b and pad to max_rank."""
        processed = {}
        for key, val in weights.items():
            if key.endswith(".lora_a"):
                # lora_a: (in_dim, rank) → pad to (in_dim, max_rank)
                if rank < max_rank:
                    pad_width = max_rank - rank
                    padding = mx.zeros((val.shape[0], pad_width), dtype=val.dtype)
                    val = mx.concatenate([val, padding], axis=1)
                processed[key] = val
            elif key.endswith(".lora_b"):
                # lora_b: (rank, out_dim) → pre-bake scale, then pad to (max_rank, out_dim)
                val = val * scale  # pre-bake scale into B
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

    # Verify shapes
    for key, val in merged.items():
        if key.endswith(".lora_a"):
            assert val.shape[1] == max_rank, f"{key}: expected rank dim {max_rank}, got {val.shape[1]}"
        elif key.endswith(".lora_b"):
            assert val.shape[0] == max_rank, f"{key}: expected rank dim {max_rank}, got {val.shape[0]}"

    # Merged config: unified rank, scale=1.0 (scale pre-baked into weights)
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
            "scale": 1.0,  # scale pre-baked into lora_b weights
            "dropout": 0.0,
            "keys": merged_keys,
        },
    }
    (out_path / "adapter_config.json").write_text(json.dumps(merged_config, indent=4))

    log(f"  Merged {len(weights_a)} + {len(weights_b)} keys → {len(merged)} "
        f"(keys: {merged_keys}, rank={max_rank}, scale=1.0 [pre-baked], "
        f"num_layers={merged_num_layers})")
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

# Medical domain vocabulary (from P1.T2 and P4.B0)
MEDICAL_KEYWORDS = [
    "diagnosis", "symptom", "treatment", "patient", "clinical",
    "prognosis", "etiology", "pathology", "medication", "dosage",
    "chronic", "acute", "inflammation", "infection", "therapy",
    "mg", "ml", "iv", "po", "bid", "tid", "qd",
]

# SOAP format markers
SOAP_MARKERS = ["s:", "o:", "a:", "p:"]

# Legal domain vocabulary
LEGAL_KEYWORDS = [
    "whereas", "hereinafter", "pursuant", "shall", "party", "parties",
    "agreement", "covenant", "liability", "indemnify", "jurisdiction",
    "arbitration", "breach", "remedy", "damages", "clause",
]

# Legal format markers
LEGAL_FORMAT_KEYWORDS = [
    "whereas", "now, therefore", "hereinafter", "pursuant to",
    "shall", "in witness whereof", "the parties agree", "this agreement",
]


def score_medical_domain(text: str) -> tuple[bool, int]:
    """Score medical domain vocabulary. Returns (pass, keyword_count)."""
    text_lower = text.lower()
    hits = sum(1 for kw in MEDICAL_KEYWORDS if kw.lower() in text_lower)
    # Require >= 4 medical terms for domain quality
    return hits >= 4, hits


def score_soap_format(text: str) -> bool:
    """True if response has all 4 SOAP sections."""
    text_lower = text.lower()
    has_s = bool(re.search(r'\bs\s*:', text_lower))
    has_o = bool(re.search(r'\bo\s*:', text_lower))
    has_a = bool(re.search(r'\ba\s*:', text_lower))
    has_p = bool(re.search(r'\bp\s*:', text_lower))
    return has_s and has_o and has_a and has_p


def score_legal_domain(text: str) -> tuple[bool, int]:
    """Score legal domain vocabulary. Returns (pass, keyword_count)."""
    text_lower = text.lower()
    hits = sum(1 for kw in LEGAL_KEYWORDS if kw.lower() in text_lower)
    # Require >= 4 legal terms for domain quality
    return hits >= 4, hits


def score_legal_format(text: str) -> bool:
    """True if response contains >= 3 legal boilerplate markers."""
    text_lower = text.lower()
    hits = sum(1 for kw in LEGAL_FORMAT_KEYWORDS if kw.lower() in text_lower)
    return hits >= 3


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

LEGAL_QUESTIONS = [
    "Draft a non-disclosure agreement between two technology companies.",
    "Write a software licensing agreement for a SaaS product.",
    "Create a consulting agreement between a startup and a technology advisor.",
    "Draft a commercial lease agreement for office space.",
    "Write an employment agreement for a senior software engineer.",
    "Create a partnership agreement for a joint business venture.",
    "Draft a service level agreement for cloud hosting services.",
    "Write a data processing agreement for GDPR compliance.",
    "Create an intellectual property assignment agreement.",
    "Draft a settlement agreement for a contract dispute.",
]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Merge adapters
# ══════════════════════════════════════════════════════════════════════════════

def phase_merge_adapters() -> dict[str, Path]:
    """Merge domain + format adapter pairs into composed adapters."""
    log("\n=== Phase 1: Merge Adapters ===")

    # Verify all source adapters exist
    for name, path in ADAPTERS.items():
        safetensors = path / "adapters.safetensors"
        config = path / "adapter_config.json"
        assert safetensors.exists(), f"Missing adapter: {safetensors}"
        assert config.exists(), f"Missing config: {config}"
        log(f"  {name}: OK ({safetensors})")

    merged_paths = {}
    for pair_name, (domain_key, format_key) in COMPOSITION_PAIRS.items():
        out_path = EXPERIMENT_DIR / f"merged_{pair_name}"
        log(f"\n  Merging {pair_name}: {domain_key} + {format_key}")
        merge_adapters(ADAPTERS[domain_key], ADAPTERS[format_key], out_path)
        merged_paths[pair_name] = out_path

    return merged_paths


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Evaluate composed adapters
# ══════════════════════════════════════════════════════════════════════════════

def phase_evaluate_composition(merged_paths: dict[str, Path], n_eval: int) -> dict:
    """Evaluate composed adapters on domain + format quality."""
    log("\n=== Phase 2: Evaluate Composed Adapters ===")
    results = {}

    # --- Medical + SOAP ---
    log("\n--- Medical + SOAP Composition ---")
    questions = MEDICAL_QUESTIONS[:n_eval]
    merged_path = str(merged_paths["medical_soap"])

    log(f"\n  Base model (no adapters):")
    base_domain_scores = []
    base_format_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=None)
        d_pass, d_hits = score_medical_domain(resp)
        f_pass = score_soap_format(resp)
        base_domain_scores.append(d_pass)
        base_format_scores.append(f_pass)
        log(f"    domain={'PASS' if d_pass else 'FAIL'}({d_hits}kw) "
            f"format={'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    log(f"\n  Composed model (medical q_proj + SOAP v_proj+o_proj):")
    comp_domain_scores = []
    comp_format_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=merged_path)
        d_pass, d_hits = score_medical_domain(resp)
        f_pass = score_soap_format(resp)
        comp_domain_scores.append(d_pass)
        comp_format_scores.append(f_pass)
        log(f"    domain={'PASS' if d_pass else 'FAIL'}({d_hits}kw) "
            f"format={'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    base_domain_rate = sum(base_domain_scores) / len(base_domain_scores)
    comp_domain_rate = sum(comp_domain_scores) / len(comp_domain_scores)
    base_format_rate = sum(base_format_scores) / len(base_format_scores)
    comp_format_rate = sum(comp_format_scores) / len(comp_format_scores)
    format_improvement = (comp_format_rate - base_format_rate) * 100

    results["medical_soap"] = {
        "base_domain_rate": base_domain_rate,
        "composed_domain_rate": comp_domain_rate,
        "base_format_rate": base_format_rate,
        "composed_format_rate": comp_format_rate,
        "format_improvement_pp": format_improvement,
        "n_eval": n_eval,
    }
    log(f"\n  MEDICAL+SOAP RESULTS:")
    log(f"    Domain quality: base={base_domain_rate:.0%} composed={comp_domain_rate:.0%}")
    log(f"    Format compliance: base={base_format_rate:.0%} composed={comp_format_rate:.0%} "
        f"(+{format_improvement:.1f}pp)")

    cleanup()

    # --- Legal + Legal-brief ---
    log("\n--- Legal + Legal-brief Composition ---")
    questions = LEGAL_QUESTIONS[:n_eval]
    merged_path = str(merged_paths["legal_legalbrief"])

    log(f"\n  Base model (no adapters):")
    base_domain_scores = []
    base_format_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=None)
        d_pass, d_hits = score_legal_domain(resp)
        f_pass = score_legal_format(resp)
        base_domain_scores.append(d_pass)
        base_format_scores.append(f_pass)
        log(f"    domain={'PASS' if d_pass else 'FAIL'}({d_hits}kw) "
            f"format={'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    log(f"\n  Composed model (legal q_proj + legal-brief v_proj+o_proj):")
    comp_domain_scores = []
    comp_format_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=merged_path)
        d_pass, d_hits = score_legal_domain(resp)
        f_pass = score_legal_format(resp)
        comp_domain_scores.append(d_pass)
        comp_format_scores.append(f_pass)
        log(f"    domain={'PASS' if d_pass else 'FAIL'}({d_hits}kw) "
            f"format={'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    base_domain_rate = sum(base_domain_scores) / len(base_domain_scores)
    comp_domain_rate = sum(comp_domain_scores) / len(comp_domain_scores)
    base_format_rate = sum(base_format_scores) / len(base_format_scores)
    comp_format_rate = sum(comp_format_scores) / len(comp_format_scores)
    format_improvement = (comp_format_rate - base_format_rate) * 100

    results["legal_legalbrief"] = {
        "base_domain_rate": base_domain_rate,
        "composed_domain_rate": comp_domain_rate,
        "base_format_rate": base_format_rate,
        "composed_format_rate": comp_format_rate,
        "format_improvement_pp": format_improvement,
        "n_eval": n_eval,
    }
    log(f"\n  LEGAL+LEGALBRIEF RESULTS:")
    log(f"    Domain quality: base={base_domain_rate:.0%} composed={comp_domain_rate:.0%}")
    log(f"    Format compliance: base={base_format_rate:.0%} composed={comp_format_rate:.0%} "
        f"(+{format_improvement:.1f}pp)")

    cleanup()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Solo vs composed degradation check
# ══════════════════════════════════════════════════════════════════════════════

def phase_degradation_check(merged_paths: dict[str, Path], n_eval: int) -> dict:
    """Check if composing two adapters degrades solo adapter performance."""
    log("\n=== Phase 3: Solo vs Composed Degradation Check ===")
    results = {}

    # Medical questions: compare SOAP solo vs SOAP in composition
    log("\n--- SOAP format: solo vs composed (on medical questions) ---")
    questions = MEDICAL_QUESTIONS[:n_eval]

    log("  SOAP solo:")
    solo_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(ADAPTERS["soap_format"]))
        f_pass = score_soap_format(resp)
        solo_scores.append(f_pass)
        log(f"    {'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    log("  SOAP in composition (medical + SOAP):")
    comp_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(merged_paths["medical_soap"]))
        f_pass = score_soap_format(resp)
        comp_scores.append(f_pass)
        log(f"    {'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    solo_rate = sum(solo_scores) / len(solo_scores) * 100
    comp_rate = sum(comp_scores) / len(comp_scores) * 100
    degradation_soap = solo_rate - comp_rate

    results["soap_format_degradation"] = {
        "solo_rate_pct": solo_rate,
        "composed_rate_pct": comp_rate,
        "degradation_pp": degradation_soap,
    }
    log(f"  SOAP: solo={solo_rate:.0f}% composed={comp_rate:.0f}% "
        f"degradation={degradation_soap:.1f}pp")

    cleanup()

    # Legal questions: compare legal-format solo vs in composition
    log("\n--- Legal format: solo vs composed (on legal questions) ---")
    questions = LEGAL_QUESTIONS[:n_eval]

    log("  Legal-brief solo:")
    solo_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(ADAPTERS["legal_format"]))
        f_pass = score_legal_format(resp)
        solo_scores.append(f_pass)
        log(f"    {'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    log("  Legal-brief in composition (legal + legal-brief):")
    comp_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(merged_paths["legal_legalbrief"]))
        f_pass = score_legal_format(resp)
        comp_scores.append(f_pass)
        log(f"    {'PASS' if f_pass else 'FAIL'} — {q[:60]}...")

    solo_rate = sum(solo_scores) / len(solo_scores) * 100
    comp_rate = sum(comp_scores) / len(comp_scores) * 100
    degradation_legal = solo_rate - comp_rate

    results["legal_format_degradation"] = {
        "solo_rate_pct": solo_rate,
        "composed_rate_pct": comp_rate,
        "degradation_pp": degradation_legal,
    }
    log(f"  Legal: solo={solo_rate:.0f}% composed={comp_rate:.0f}% "
        f"degradation={degradation_legal:.1f}pp")

    cleanup()

    # Domain adapter degradation: medical domain solo vs in composition
    log("\n--- Medical domain: solo vs composed (on medical questions) ---")
    questions = MEDICAL_QUESTIONS[:n_eval]

    log("  Medical domain solo:")
    solo_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(ADAPTERS["medical_domain"]))
        d_pass, d_hits = score_medical_domain(resp)
        solo_scores.append(d_pass)
        log(f"    {'PASS' if d_pass else 'FAIL'}({d_hits}kw) — {q[:60]}...")

    log("  Medical domain in composition (medical + SOAP):")
    comp_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(merged_paths["medical_soap"]))
        d_pass, d_hits = score_medical_domain(resp)
        comp_scores.append(d_pass)
        log(f"    {'PASS' if d_pass else 'FAIL'}({d_hits}kw) — {q[:60]}...")

    solo_rate = sum(solo_scores) / len(solo_scores) * 100
    comp_rate = sum(comp_scores) / len(comp_scores) * 100
    degradation_med = solo_rate - comp_rate

    results["medical_domain_degradation"] = {
        "solo_rate_pct": solo_rate,
        "composed_rate_pct": comp_rate,
        "degradation_pp": degradation_med,
    }
    log(f"  Medical domain: solo={solo_rate:.0f}% composed={comp_rate:.0f}% "
        f"degradation={degradation_med:.1f}pp")

    cleanup()

    # Legal domain degradation
    log("\n--- Legal domain: solo vs composed (on legal questions) ---")
    questions = LEGAL_QUESTIONS[:n_eval]

    log("  Legal domain solo:")
    solo_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(ADAPTERS["legal_domain"]))
        d_pass, d_hits = score_legal_domain(resp)
        solo_scores.append(d_pass)
        log(f"    {'PASS' if d_pass else 'FAIL'}({d_hits}kw) — {q[:60]}...")

    log("  Legal domain in composition (legal + legal-brief):")
    comp_scores = []
    for q in questions:
        resp = generate_response(q, adapter_path=str(merged_paths["legal_legalbrief"]))
        d_pass, d_hits = score_legal_domain(resp)
        comp_scores.append(d_pass)
        log(f"    {'PASS' if d_pass else 'FAIL'}({d_hits}kw) — {q[:60]}...")

    solo_rate = sum(solo_scores) / len(solo_scores) * 100
    comp_rate = sum(comp_scores) / len(comp_scores) * 100
    degradation_legal_dom = solo_rate - comp_rate

    results["legal_domain_degradation"] = {
        "solo_rate_pct": solo_rate,
        "composed_rate_pct": comp_rate,
        "degradation_pp": degradation_legal_dom,
    }
    log(f"  Legal domain: solo={solo_rate:.0f}% composed={comp_rate:.0f}% "
        f"degradation={degradation_legal_dom:.1f}pp")

    cleanup()

    results["max_degradation_pp"] = max(
        degradation_soap, degradation_legal,
        degradation_med, degradation_legal_dom,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("P4.D0: Domain + Format Adapter Simultaneous Composition")
    log(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, MAX_TOKENS={MAX_TOKENS}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # Phase 1: Merge adapters
    merged_paths = phase_merge_adapters()
    log_memory("after-merge")

    # Phase 2: Evaluate composed adapters
    composition_results = phase_evaluate_composition(merged_paths, N_EVAL)
    log_memory("after-composition-eval")

    # Phase 3: Degradation check
    degradation_results = phase_degradation_check(merged_paths, N_EVAL)
    log_memory("after-degradation-check")

    # ─── Kill criteria ─────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("KILL CRITERIA RESULTS")
    log("=" * 70)

    ms = composition_results["medical_soap"]
    ll = composition_results["legal_legalbrief"]

    # K1249: Medical + SOAP: domain_quality >= 40% AND format_compliance >= 50pp
    k1249_domain = ms["composed_domain_rate"] >= 0.40
    k1249_format = ms["format_improvement_pp"] >= 50.0
    k1249_pass = k1249_domain and k1249_format
    log(f"\nK1249 (Medical+SOAP: domain>=40% AND format>=50pp):")
    log(f"  domain={ms['composed_domain_rate']:.0%} >=40%? {'PASS' if k1249_domain else 'FAIL'}")
    log(f"  format_improvement={ms['format_improvement_pp']:.1f}pp >=50pp? {'PASS' if k1249_format else 'FAIL'}")
    log(f"  K1249 OVERALL: {'PASS' if k1249_pass else 'FAIL'}")

    # K1250: Legal + Legal-brief: domain_quality >= 40% AND format_compliance >= 60pp
    k1250_domain = ll["composed_domain_rate"] >= 0.40
    k1250_format = ll["format_improvement_pp"] >= 60.0
    k1250_pass = k1250_domain and k1250_format
    log(f"\nK1250 (Legal+Legal-brief: domain>=40% AND format>=60pp):")
    log(f"  domain={ll['composed_domain_rate']:.0%} >=40%? {'PASS' if k1250_domain else 'FAIL'}")
    log(f"  format_improvement={ll['format_improvement_pp']:.1f}pp >=60pp? {'PASS' if k1250_format else 'FAIL'}")
    log(f"  K1250 OVERALL: {'PASS' if k1250_pass else 'FAIL'}")

    # K1251: Solo degradation <= 15pp
    max_deg = degradation_results["max_degradation_pp"]
    k1251_pass = max_deg <= 15.0
    log(f"\nK1251 (Solo degradation <= 15pp under composition):")
    log(f"  SOAP format: {degradation_results['soap_format_degradation']['degradation_pp']:.1f}pp")
    log(f"  Legal format: {degradation_results['legal_format_degradation']['degradation_pp']:.1f}pp")
    log(f"  Medical domain: {degradation_results['medical_domain_degradation']['degradation_pp']:.1f}pp")
    log(f"  Legal domain: {degradation_results['legal_domain_degradation']['degradation_pp']:.1f}pp")
    log(f"  max_degradation={max_deg:.1f}pp <=15pp? {'PASS' if k1251_pass else 'FAIL'}")

    all_pass = k1249_pass and k1250_pass and k1251_pass
    total_min = (time.time() - total_start) / 60.0

    log(f"\n{'='*70}")
    log(f"SUMMARY: K1249={'PASS' if k1249_pass else 'FAIL'}, "
        f"K1250={'PASS' if k1250_pass else 'FAIL'}, "
        f"K1251={'PASS' if k1251_pass else 'FAIL'}")
    log(f"ALL_PASS={all_pass}")
    log(f"Total time: {total_min:.2f} min")
    log(f"{'='*70}")

    # ─── Save results ──────────────────────────────────────────────────────
    results = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "composition_results": composition_results,
        "degradation_results": degradation_results,
        "kill_criteria": {
            "k1249_medical_soap": {
                "pass": k1249_pass,
                "domain_rate": ms["composed_domain_rate"],
                "format_improvement_pp": ms["format_improvement_pp"],
                "domain_threshold": 0.40,
                "format_threshold_pp": 50.0,
            },
            "k1250_legal_legalbrief": {
                "pass": k1250_pass,
                "domain_rate": ll["composed_domain_rate"],
                "format_improvement_pp": ll["format_improvement_pp"],
                "domain_threshold": 0.40,
                "format_threshold_pp": 60.0,
            },
            "k1251_degradation": {
                "pass": k1251_pass,
                "max_degradation_pp": max_deg,
                "threshold_pp": 15.0,
                "details": {
                    k: v for k, v in degradation_results.items()
                    if k != "max_degradation_pp"
                },
            },
        },
        "all_pass": all_pass,
        "total_time_min": round(total_min, 2),
        "adapter_paths": {k: str(v) for k, v in ADAPTERS.items()},
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
