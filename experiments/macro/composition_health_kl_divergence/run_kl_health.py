#!/usr/bin/env python3
"""Composition Health via KL Divergence — label-free detection of harmful expert additions.

Kill criteria:
- K1: Spearman rho(DeltaKL, quality_loss) >= 0.3 -> PASS; < 0.3 -> KILL
- K2: DeltaKL_harmful > mean(DeltaKL) + 2*std(DeltaKL) -> PASS (z-score > 2.0)
- K3: Wall-clock time per composition < 30s -> PASS

Supports SMOKE_TEST=1 for <60s validation.
"""
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Configuration ──────────────────────────────────────────────────────────────

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
ADAPTER_DIR = Path("/workspace/llm/adapters")
BENCHMARK_PATH = Path("/workspace/llm/results/pilot50_benchmark.json")
RESULTS_DIR = Path("/workspace/llm/results/composition_health_kl_divergence")
SEED = 42

N_SWEEP = [2, 3] if IS_SMOKE else [5, 10, 25, 50]
N_LOO = 3 if IS_SMOKE else 10      # leave-one-out at this N
MAX_RUNTIME_S = 120 if IS_SMOKE else 900  # 2 min smoke, 15 min full
MAX_SEQ_LEN = 128 if IS_SMOKE else 256

# ── Calibration Texts (hardcoded, domain-agnostic) ────────────────────────────

CALIBRATION_TEXTS = [
    # Factual / Wikipedia (4)
    "The mitochondria is the powerhouse of the cell, responsible for generating ATP through oxidative phosphorylation.",
    "In 1969, Neil Armstrong became the first human to walk on the Moon during the Apollo 11 mission.",
    "The French Revolution began in 1789 and fundamentally transformed French society and government.",
    "The human genome contains approximately 3 billion base pairs of DNA organized into 23 pairs of chromosomes.",
    # Code (4)
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "SELECT users.name, orders.total FROM users INNER JOIN orders ON users.id = orders.user_id WHERE",
    "import torch\nmodel = torch.nn.Linear(128, 64)\noptimizer = torch.optim.Adam(model.parameters(),",
    "#!/bin/bash\nfor file in *.txt; do\n    echo \"Processing $file\"\n    wc -l \"$file\"\ndone",
    # Math / Science (4)
    "The derivative of sin(x) is cos(x), and the integral of cos(x) is sin(x) + C.",
    "To solve a quadratic equation ax² + bx + c = 0, use the quadratic formula: x = (-b ± sqrt(b² - 4ac)) / 2a.",
    "The electromagnetic spectrum ranges from radio waves with long wavelengths to gamma rays with very short wavelengths.",
    "In thermodynamics, entropy always increases in an isolated system, which is the second law of thermodynamics.",
    # Conversational / QA (4)
    "What is the capital of France? The capital of France is Paris, which has been the nation's capital since",
    "How do I reverse a string in Python? You can use slicing: reversed_string = original_string[::-1].",
    "Can you explain what machine learning is? Machine learning is a subset of artificial intelligence that enables",
    "What are the main differences between supervised and unsupervised learning algorithms in data science?",
    # Creative / Literary (4)
    "The old lighthouse stood alone on the rocky promontory, its beam sweeping across the dark and restless sea.",
    "In the twilight hours, when shadows lengthen and the air grows still, the forest speaks in hushed whispers.",
    "She walked through the crowded marketplace, her eyes searching for a familiar face among the sea of strangers.",
    "The river wound its way through the valley like a silver thread, reflecting the colors of the evening sky.",
]

assert len(CALIBRATION_TEXTS) == 20, "Must have exactly 20 calibration texts"

# ── Utilities ─────────────────────────────────────────────────────────────────

_script_start = time.time()


def elapsed() -> float:
    return time.time() - _script_start


def timed_out() -> bool:
    return elapsed() > MAX_RUNTIME_S


def log(msg: str) -> None:
    print(f"[{elapsed():6.1f}s] {msg}", flush=True)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_base_model():
    """Load Qwen2.5-7B with 4-bit NF4 quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def load_peft_composition(base_model, adapter_names, composed_name="composed"):
    """Load N adapters and compose via add_weighted_adapter with weights=[1.0]*N."""
    from peft import PeftModel

    # Load first adapter
    first_path = str(ADAPTER_DIR / adapter_names[0])
    peft_model = PeftModel.from_pretrained(
        base_model, first_path, adapter_name=adapter_names[0]
    )

    # Load remaining adapters
    for name in adapter_names[1:]:
        peft_model.load_adapter(str(ADAPTER_DIR / name), adapter_name=name)

    # Compose: weights=[1.0]*N (spec requires this; additive sum of N deltas)
    peft_model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=[1.0] * len(adapter_names),
        adapter_name=composed_name,
        combination_type="linear",
    )
    peft_model.set_adapter(composed_name)
    peft_model.eval()
    return peft_model


# ── Calibration Logits ─────────────────────────────────────────────────────────

def compute_logits_on_texts(model, tokenizer, texts, batch_size: int = 8):
    """Batched forward passes; return list of last-token logit tensors (float32, CPU)."""
    all_logits = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]
        try:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # Last non-padded token for each item in batch
                # attention_mask tells us the real last position
                attention_mask = inputs["attention_mask"]
                last_positions = attention_mask.sum(dim=1) - 1  # (B,)
                batch_logits = outputs.logits  # (B, seq, V)
                for b_idx in range(len(batch)):
                    pos = last_positions[b_idx].item()
                    logit_vec = batch_logits[b_idx, pos, :].float().cpu()
                    all_logits.append(logit_vec)
        except torch.cuda.OutOfMemoryError:
            free_gpu()
            # Retry one at a time
            log(f"  OOM on batch_size={batch_size}, retrying one at a time")
            for text in batch:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                )
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    logit_vec = outputs.logits[0, -1, :].float().cpu()
                    all_logits.append(logit_vec)
    return all_logits


def compute_kl_divergence(base_logits_list, composed_logits_list):
    """Compute KL(P_composed || P_base) per text; return mean, std, per-text list.

    Uses F.kl_div(log_p_base, log_p_composed, log_target=True, reduction='sum')
    which computes sum(exp(log_p_composed) * (log_p_composed - log_p_base)).
    All in float32.
    """
    kl_values = []
    for base_logits, comp_logits in zip(base_logits_list, composed_logits_list):
        log_p_base = F.log_softmax(base_logits.float(), dim=-1)
        log_p_comp = F.log_softmax(comp_logits.float(), dim=-1)
        kl = F.kl_div(log_p_base, log_p_comp, log_target=True, reduction="sum")
        kl_values.append(float(kl.item()))
    return float(np.mean(kl_values)), float(np.std(kl_values)), kl_values


# ── Adapter Registry ───────────────────────────────────────────────────────────

def get_sorted_adapters():
    """Return adapter names sorted by PPL improvement desc (best quality first)."""
    # Discover all valid adapters on disk first
    disk_adapters = set(
        d.name
        for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )
    log(f"Found {len(disk_adapters)} adapters on disk: {sorted(disk_adapters)}")

    if not disk_adapters:
        return []

    ranked = []

    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH) as f:
            benchmark = json.load(f)
        # Support both "per_adapter" and "domains" keys
        per_adapter = benchmark.get("per_adapter") or benchmark.get("domains", {})
        for name, info in per_adapter.items():
            ppl_improvement = float(info.get("ppl_improvement_pct", info.get("improvement_pct", 0.0)))
            if name in disk_adapters:
                ranked.append((name, ppl_improvement))
                disk_adapters.discard(name)

    # Add any remaining disk adapters not found in benchmark (no quality score)
    for name in sorted(disk_adapters):
        ranked.append((name, 0.0))

    ranked.sort(key=lambda x: x[1], reverse=True)
    log(f"Loaded {len(ranked)} adapters, sorted by PPL improvement")
    return ranked


# ── Harmful Expert Creation ────────────────────────────────────────────────────

def create_harmful_adapter(source_adapter_name: str, tmp_dir: Path) -> Path:
    """Create a harmful adapter by negating lora_B weights of source.

    Saves to tmp_dir/harmful_expert/ and returns the path.
    """
    import copy

    source_path = ADAPTER_DIR / source_adapter_name
    harmful_path = tmp_dir / "harmful_expert"
    harmful_path.mkdir(parents=True, exist_ok=True)

    # Copy adapter_config.json verbatim
    shutil.copy(source_path / "adapter_config.json", harmful_path / "adapter_config.json")

    # Load weights
    safetensors_src = source_path / "adapter_model.safetensors"
    bin_src = source_path / "adapter_model.bin"

    if safetensors_src.exists():
        from safetensors import safe_open
        from safetensors.torch import save_file

        state_dict = {}
        with safe_open(str(safetensors_src), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        # Negate lora_B weights
        negated = {}
        for key, tensor in state_dict.items():
            if "lora_B" in key:
                negated[key] = -tensor
            else:
                negated[key] = tensor.clone()

        save_file(negated, str(harmful_path / "adapter_model.safetensors"))

    elif bin_src.exists():
        state_dict = torch.load(str(bin_src), map_location="cpu", weights_only=True)
        negated = {}
        for key, tensor in state_dict.items():
            if "lora_B" in key:
                negated[key] = -tensor
            else:
                negated[key] = tensor.clone()
        torch.save(negated, str(harmful_path / "adapter_model.bin"))
    else:
        raise FileNotFoundError(f"No adapter weights found at {source_path}")

    log(f"  Created harmful adapter from {source_adapter_name} (lora_B negated)")
    return harmful_path


# ── Phase 1: Base Calibration ──────────────────────────────────────────────────

def phase1_base_calibration():
    """Load base model, tokenize calibration texts, compute base logits."""
    log("=== Phase 1: Base Model Calibration ===")
    t0 = time.time()

    base_model, tokenizer = load_base_model()
    log(f"  Base model loaded [{time.time()-t0:.1f}s]")

    # Pre-tokenize all texts
    base_logits = compute_logits_on_texts(base_model, tokenizer, CALIBRATION_TEXTS)
    log(f"  Computed base logits on {len(base_logits)} texts [{time.time()-t0:.1f}s]")

    elapsed_s = time.time() - t0
    result = {
        "elapsed_s": round(elapsed_s, 2),
        "n_texts": len(base_logits),
    }
    return base_model, tokenizer, base_logits, result


# ── Phase 2: KL vs N Sweep ─────────────────────────────────────────────────────

def phase2_kl_vs_n(tokenizer, base_logits, sorted_adapters):
    """For each N in N_SWEEP: compose top-N adapters, compute KL."""
    log("=== Phase 2: KL vs N Sweep ===")
    t_phase = time.time()

    results = {}
    adapter_names = [n for n, _ in sorted_adapters]

    for n in N_SWEEP:
        if timed_out():
            log(f"  TIMEOUT: skipping N={n}")
            break

        if n > len(adapter_names):
            log(f"  Skip N={n}: only {len(adapter_names)} adapters available")
            continue

        selected = adapter_names[:n]
        log(f"  N={n}: composing {n} adapters...")
        t0 = time.time()

        # Need a fresh base model for each composition (PEFT modifies in-place)
        fresh_base, _ = load_base_model()
        peft_model = None
        try:
            peft_model = load_peft_composition(fresh_base, selected)

            composed_logits = compute_logits_on_texts(peft_model, tokenizer, CALIBRATION_TEXTS)
            mean_kl, std_kl, kl_vals = compute_kl_divergence(base_logits, composed_logits)
            elapsed_s = time.time() - t0

            results[str(n)] = {
                "mean_kl": round(mean_kl, 6),
                "std_kl": round(std_kl, 6),
                "per_text_kl": [round(v, 6) for v in kl_vals],
                "elapsed_s": round(elapsed_s, 2),
            }
            log(f"  N={n}: KL={mean_kl:.4f}±{std_kl:.4f} [{elapsed_s:.1f}s]")

        except Exception as e:
            elapsed_s = time.time() - t0
            log(f"  N={n}: FAILED [{elapsed_s:.1f}s]: {e}")
            traceback.print_exc()
            results[str(n)] = {"error": str(e), "elapsed_s": round(elapsed_s, 2)}

        finally:
            del peft_model
            del fresh_base
            free_gpu()

    log(f"  Phase 2 complete [{time.time()-t_phase:.1f}s]")
    return results


# ── Phase 3: Leave-One-Out at N=N_LOO ─────────────────────────────────────────

def phase3_leave_one_out(
    tokenizer, base_logits, sorted_adapters, phase2_results
):
    """Leave-one-out analysis at N=N_LOO. Returns per-adapter delta_kl + z-scores."""
    log(f"=== Phase 3: Leave-One-Out at N={N_LOO} ===")
    t_phase = time.time()

    adapter_names = [n for n, _ in sorted_adapters]
    ppl_by_name = {n: ppl for n, ppl in sorted_adapters}

    n_test = min(N_LOO, len(adapter_names))
    test_adapters = adapter_names[:n_test]

    # Reuse KL_all from Phase 2 if available
    p2_entry = phase2_results.get(str(n_test), {})
    kl_all_n = p2_entry.get("mean_kl", None)

    if kl_all_n is None:
        # Must compute it fresh
        log(f"  KL_all_{n_test} not in Phase 2 cache; computing now...")
        t0 = time.time()
        fresh_base, _ = load_base_model()
        peft_model = None
        try:
            peft_model = load_peft_composition(fresh_base, test_adapters)
            comp_logits = compute_logits_on_texts(peft_model, tokenizer, CALIBRATION_TEXTS)
            kl_all_n, _, _ = compute_kl_divergence(base_logits, comp_logits)
            log(f"  KL_all_{n_test}={kl_all_n:.4f} [{time.time()-t0:.1f}s]")
        except Exception as e:
            log(f"  Failed to compute KL_all_{n_test}: {e}")
            kl_all_n = 0.0
        finally:
            del peft_model
            del fresh_base
            free_gpu()

    per_adapter = {}

    for idx, leave_out in enumerate(test_adapters):
        if timed_out():
            log(f"  TIMEOUT: stopping leave-one-out at idx={idx}")
            break

        remaining = [a for a in test_adapters if a != leave_out]
        log(f"  [{idx+1}/{n_test}] Leave out: {leave_out}")
        t0 = time.time()

        fresh_base, _ = load_base_model()
        peft_model = None
        try:
            peft_model = load_peft_composition(fresh_base, remaining)
            comp_logits = compute_logits_on_texts(peft_model, tokenizer, CALIBRATION_TEXTS)
            mean_kl_9, _, _ = compute_kl_divergence(base_logits, comp_logits)

            delta_kl = float(kl_all_n) - float(mean_kl_9)
            elapsed_s = time.time() - t0

            per_adapter[leave_out] = {
                "kl_without": round(float(mean_kl_9), 6),
                "delta_kl": round(delta_kl, 6),
                "z_score": None,  # computed below after all entries are known
                "ppl_improvement_pct": round(float(ppl_by_name.get(leave_out, 0.0)), 4),
                "elapsed_s": round(elapsed_s, 2),
            }
            log(f"    kl_without={mean_kl_9:.4f}, delta_kl={delta_kl:.4f} [{elapsed_s:.1f}s]")

        except Exception as e:
            elapsed_s = time.time() - t0
            log(f"    FAILED [{elapsed_s:.1f}s]: {e}")
            traceback.print_exc()
            per_adapter[leave_out] = {"error": str(e), "elapsed_s": round(elapsed_s, 2)}
        finally:
            del peft_model
            del fresh_base
            free_gpu()

    # Compute z-scores
    delta_kl_values = [
        v["delta_kl"]
        for v in per_adapter.values()
        if "delta_kl" in v
    ]
    if len(delta_kl_values) >= 2:
        mu = float(np.mean(delta_kl_values))
        sigma = float(np.std(delta_kl_values))
        for name, entry in per_adapter.items():
            if "delta_kl" in entry:
                z = (entry["delta_kl"] - mu) / max(sigma, 1e-12)
                entry["z_score"] = round(float(z), 4)
    else:
        mu, sigma = 0.0, 1.0

    log(f"  Phase 3 complete [{time.time()-t_phase:.1f}s]")

    return {
        "n_test": n_test,
        "kl_all_10": round(float(kl_all_n), 6),
        "per_adapter": per_adapter,
        "_delta_kl_mean": round(mu, 6),
        "_delta_kl_std": round(sigma, 6),
    }


# ── Phase 4: Synthetic Harmful Expert ─────────────────────────────────────────

def phase4_harmful_expert(
    tokenizer, base_logits, sorted_adapters, phase3_result
):
    """Negate best adapter's lora_B, replace it in top-9, measure KL spike."""
    log("=== Phase 4: Synthetic Harmful Expert ===")
    t_phase = time.time()

    adapter_names = [n for n, _ in sorted_adapters]
    n_test = min(N_LOO, len(adapter_names))
    test_adapters = adapter_names[:n_test]

    # Source adapter: best quality (first after sort, i.e. test_adapters[0])
    source_adapter = test_adapters[0]

    # The 9 adapters that remain after we swap out source_adapter
    remaining_9 = test_adapters[1:n_test]  # indices 1..9, length = n_test-1

    phase3_mu = phase3_result.get("_delta_kl_mean", 0.0)
    phase3_sigma = phase3_result.get("_delta_kl_std", 1.0)
    kl_all_n = phase3_result.get("kl_all_10", 0.0)

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmful_expert_"))
    try:
        harmful_path = create_harmful_adapter(source_adapter, tmp_dir)

        # Compose: remaining_9 + harmful_expert
        log(f"  Composing remaining-{len(remaining_9)} + harmful_expert...")
        t0 = time.time()

        fresh_base, _ = load_base_model()
        peft_model = None
        kl_with_harmful = 0.0
        try:
            from peft import PeftModel

            # Load first of remaining_9
            first = remaining_9[0] if remaining_9 else None
            if first is None:
                raise ValueError("Not enough adapters for harmful expert test")

            first_path = str(ADAPTER_DIR / first)
            peft_model = PeftModel.from_pretrained(fresh_base, first_path, adapter_name=first)

            for name in remaining_9[1:]:
                peft_model.load_adapter(str(ADAPTER_DIR / name), adapter_name=name)

            # Load harmful adapter from temp dir
            peft_model.load_adapter(str(harmful_path), adapter_name="harmful_expert")

            adapter_list = list(remaining_9) + ["harmful_expert"]
            peft_model.add_weighted_adapter(
                adapters=adapter_list,
                weights=[1.0] * len(adapter_list),
                adapter_name="composed_harmful",
                combination_type="linear",
            )
            peft_model.set_adapter("composed_harmful")
            peft_model.eval()

            comp_logits = compute_logits_on_texts(peft_model, tokenizer, CALIBRATION_TEXTS)
            kl_with_harmful, _, _ = compute_kl_divergence(base_logits, comp_logits)
            log(f"  KL_with_harmful={kl_with_harmful:.4f} [{time.time()-t0:.1f}s]")

        except Exception as e:
            log(f"  Phase 4 composition failed: {e}")
            traceback.print_exc()
        finally:
            del peft_model
            del fresh_base
            free_gpu()

        # Delta KL: difference from N-1 composition (the "clean 9")
        # Use leave-one-out KL_without for source_adapter as the clean-9 reference
        p3_per_adapter = phase3_result.get("per_adapter", {})
        clean_9_entry = p3_per_adapter.get(source_adapter, {})
        kl_clean_9 = float(clean_9_entry.get("kl_without", kl_all_n))

        delta_kl_harmful = float(kl_with_harmful) - kl_clean_9
        z_score_harmful = (delta_kl_harmful - phase3_mu) / max(abs(phase3_sigma), 1e-12)

        # Worst natural adapter z-score from Phase 3
        z_scores_natural = [
            v.get("z_score", 0.0)
            for v in p3_per_adapter.values()
            if "z_score" in v and v["z_score"] is not None
        ]
        z_worst_natural = float(max(z_scores_natural)) if z_scores_natural else 0.0

        discrimination = bool(z_score_harmful > 2.0)

        result = {
            "harmful_method": "negated_lora_B",
            "source_adapter": source_adapter,
            "kl_with_harmful": round(float(kl_with_harmful), 6),
            "delta_kl_harmful": round(float(delta_kl_harmful), 6),
            "z_score_harmful": round(float(z_score_harmful), 4),
            "z_worst_natural": round(float(z_worst_natural), 4),
            "discrimination": discrimination,
            "elapsed_s": round(time.time() - t_phase, 2),
        }

    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log(f"  Phase 4 complete: z_score_harmful={z_score_harmful:.2f}, discrimination={discrimination}")
    return result


# ── Phase 5: Correlation with Quality Impact ────────────────────────────────────

def phase5_correlation(phase3_result):
    """Spearman rho between DeltaKL and (1 - ppl_improvement_pct/100)."""
    from scipy.stats import spearmanr

    log("=== Phase 5: Correlation with Quality Impact ===")
    t0 = time.time()

    per_adapter = phase3_result.get("per_adapter", {})

    delta_kl_list = []
    quality_loss_list = []  # 1 - ppl_improvement_pct/100 (higher = worse quality)

    for name, entry in per_adapter.items():
        if "delta_kl" not in entry:
            continue
        ppl_imp = float(entry.get("ppl_improvement_pct", 0.0))
        quality_loss = 1.0 - ppl_imp / 100.0
        delta_kl_list.append(entry["delta_kl"])
        quality_loss_list.append(quality_loss)

    n_samples = len(delta_kl_list)
    if n_samples < 3:
        log(f"  Insufficient samples for correlation: {n_samples}")
        return {
            "spearman_rho": None,
            "spearman_p": None,
            "n_samples": n_samples,
            "note": "insufficient samples",
            "elapsed_s": round(time.time() - t0, 2),
        }

    rho, p_val = spearmanr(delta_kl_list, quality_loss_list)

    log(f"  Spearman rho={rho:.4f}, p={p_val:.4f} (n={n_samples}) [{time.time()-t0:.1f}s]")
    return {
        "spearman_rho": round(float(rho), 6),
        "spearman_p": round(float(p_val), 6),
        "n_samples": n_samples,
        "note": "rho between DeltaKL and (1 - ppl_improvement_pct/100)",
        "elapsed_s": round(time.time() - t0, 2),
    }


# ── Phase 6: Per-Domain PPL (Optional) ────────────────────────────────────────

def phase6_per_domain_ppl(base_model, tokenizer, sorted_adapters, phase3_result):
    """Optional: per-domain PPL cross-impact analysis."""
    log("=== Phase 6: Per-Domain PPL (optional) ===")
    t0 = time.time()

    data_dir = Path("/workspace/llm/training_data")
    if not data_dir.exists():
        log("  training_data dir not found; skipping Phase 6")
        return {"skipped": True, "reason": "training_data not available"}

    adapter_names = [n for n, _ in sorted_adapters]
    n_test = min(N_LOO, len(adapter_names))
    test_adapters = adapter_names[:n_test]

    domain_ppls = {}
    for adapter_name in test_adapters:
        domain_file = data_dir / f"{adapter_name}.jsonl"
        if not domain_file.exists():
            continue

        texts = []
        with open(domain_file) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                try:
                    obj = json.loads(line)
                    t = obj.get("text", obj.get("content", ""))
                    if t.strip():
                        texts.append(t)
                except json.JSONDecodeError:
                    continue

        if not texts:
            continue

        total_nll, total_tokens = 0.0, 0
        device = next(base_model.parameters()).device
        for text in texts[:10]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = base_model(**inputs)
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = inputs["input_ids"][:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                total_nll += float(loss.item())
                total_tokens += shift_labels.numel()

        if total_tokens > 0:
            domain_ppls[adapter_name] = round(float(np.exp(total_nll / total_tokens)), 4)

    log(f"  Phase 6 computed PPL for {len(domain_ppls)} domains [{time.time()-t0:.1f}s]")
    return {
        "domain_ppls": domain_ppls,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ── Kill Criteria Assessment ───────────────────────────────────────────────────

def assess_kill_criteria(phase2_results, phase3_result, phase4_result, phase5_result):
    """Evaluate K1, K2, K3 and produce verdict."""
    # K1: Spearman rho >= 0.3
    rho = phase5_result.get("spearman_rho")
    k1_pass = bool(rho is not None and rho >= 0.3)

    # K2: harmful z-score > 2.0 AND worst natural z-score > 1.0
    z_harmful = phase4_result.get("z_score_harmful", 0.0) if phase4_result else None
    z_worst_natural = phase4_result.get("z_worst_natural", 0.0) if phase4_result else 0.0
    if z_harmful is not None:
        k2_pass = bool(z_harmful > 2.0)
        k2_strong = bool(z_harmful > 2.0 and z_worst_natural > 1.0)
    else:
        k2_pass = False
        k2_strong = False

    # K3: time per composition (Phase 2 N=10 elapsed / 1)
    n10_key = str(N_LOO)
    p2_n10 = phase2_results.get(n10_key, phase2_results.get("10", {}))
    time_per_composition = p2_n10.get("elapsed_s", 999.0)
    k3_pass = bool(time_per_composition < 30.0)

    # Verdict
    passes = sum([k1_pass, k2_pass, k3_pass])
    if passes == 3:
        verdict = "PASS"
    elif passes == 0:
        verdict = "KILL"
    else:
        verdict = "MARGINAL"

    return {
        "K1_spearman_rho": rho,
        "K1_threshold": 0.3,
        "K1_pass": k1_pass,
        "K2_z_score_harmful": z_harmful,
        "K2_z_score_worst_natural": round(float(z_worst_natural), 4),
        "K2_threshold": 2.0,
        "K2_pass": k2_pass,
        "K2_strong_pass": k2_strong,
        "K2_skipped": phase4_result is None,
        "K3_time_per_composition_s": round(float(time_per_composition), 2),
        "K3_threshold_s": 30,
        "K3_pass": k3_pass,
        "verdict": verdict,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log(f"Composition Health KL Divergence — {'SMOKE' if IS_SMOKE else 'FULL'} run")
    log(f"N_SWEEP={N_SWEEP}, N_LOO={N_LOO}, MAX_RUNTIME={MAX_RUNTIME_S}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Adapter registry
    sorted_adapters = get_sorted_adapters()
    if not sorted_adapters:
        log("ERROR: no adapters found; aborting")
        sys.exit(1)
    log(f"Top-5 adapters: {[n for n, _ in sorted_adapters[:5]]}")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    t1 = time.time()
    base_model, tokenizer, base_logits, phase1_result = phase1_base_calibration()
    # Free base model — base_logits are CPU tensors, tokenizer is tiny
    del base_model
    free_gpu()
    log("  Freed phase 1 base model to reclaim VRAM")
    phase1_s = time.time() - t1

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    t2 = time.time()
    phase2_results = phase2_kl_vs_n(tokenizer, base_logits, sorted_adapters)
    phase2_s = time.time() - t2

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    t3 = time.time()
    phase3_result = phase3_leave_one_out(
        tokenizer, base_logits, sorted_adapters, phase2_results
    )
    phase3_s = time.time() - t3

    # ── Phase 4 (skip in smoke) ───────────────────────────────────────────────
    t4 = time.time()
    phase4_result = None
    if not IS_SMOKE and not timed_out():
        phase4_result = phase4_harmful_expert(
            tokenizer, base_logits, sorted_adapters, phase3_result
        )
    else:
        log("=== Phase 4: Skipped (smoke test or timeout) ===")
    phase4_s = time.time() - t4

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    t5 = time.time()
    phase5_result = phase5_correlation(phase3_result)
    phase5_s = time.time() - t5

    # ── Phase 6 (skip in smoke) ───────────────────────────────────────────────
    phase6_result = None
    if not IS_SMOKE and not timed_out():
        # Load a fresh base model just for phase 6 PPL computation
        base_model_p6, _ = load_base_model()
        phase6_result = phase6_per_domain_ppl(
            base_model_p6, tokenizer, sorted_adapters, phase3_result
        )
        del base_model_p6
        free_gpu()
    else:
        log("=== Phase 6: Skipped (smoke test or timeout) ===")

    # ── Kill criteria ─────────────────────────────────────────────────────────
    kill_criteria = assess_kill_criteria(
        phase2_results, phase3_result, phase4_result, phase5_result
    )

    total_elapsed = time.time() - global_start

    # ── Results dict ──────────────────────────────────────────────────────────
    output = {
        "config": {
            "base_model": "Qwen2.5-7B",
            "n_calibration_texts": len(CALIBRATION_TEXTS),
            "n_sweep": N_SWEEP,
            "n_loo": N_LOO,
            "seed": SEED,
            "total_adapters": len(sorted_adapters),
            "quantization": "nf4_4bit",
            "smoke_test": IS_SMOKE,
        },
        "phase1_base_calibration": phase1_result,
        "phase2_kl_vs_n": phase2_results,
        "phase3_leave_one_out": phase3_result,
        "phase4_harmful_expert": phase4_result,
        "phase5_correlation": phase5_result,
        "phase6_per_domain_ppl": phase6_result,
        "kill_criteria": kill_criteria,
        "timing": {
            "total_elapsed_s": round(total_elapsed, 2),
            "phase1_s": round(phase1_s, 2),
            "phase2_s": round(phase2_s, 2),
            "phase3_s": round(phase3_s, 2),
            "phase4_s": round(phase4_s, 2),
            "phase5_s": round(phase5_s, 2),
        },
    }

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("=== Kill Criteria Summary ===")
    kc = kill_criteria
    log(f"  K1 (rho >= 0.3): rho={kc['K1_spearman_rho']}  -> {'PASS' if kc['K1_pass'] else 'FAIL'}")
    log(f"  K2 (z > 2.0):    z={kc['K2_z_score_harmful']}  -> {'PASS' if kc['K2_pass'] else 'FAIL/SKIPPED'}")
    log(f"  K3 (< 30s):      t={kc['K3_time_per_composition_s']}s -> {'PASS' if kc['K3_pass'] else 'FAIL'}")
    log(f"  VERDICT: {kc['verdict']}")
    log(f"Total runtime: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
