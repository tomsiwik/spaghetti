#!/usr/bin/env python3
"""Expert Dropout Composition Robustness at Macro Scale.

Tests whether composed model quality is robust to random expert subsets.
Composes random 80% subsets (40 of 50) and measures PPL variance across
20 bootstrap samples. If variance < 5%, composition is robust to random
expert dropout.

Kill criteria:
- K1: PPL CV across 20 random 80% subsets > 5% (composition is fragile)
- K2: best 80% subset improves > 10% over all-50 (pruning is critical)
- K3: worst 80% subset degrades > 15% from all-50 (dropout is dangerous)

Supports SMOKE_TEST=1 for <60s validation.
MAX_RUNTIME: 15 minutes (budget constraint).
"""

import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "composition_dropout_robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

MAX_SEQ_LEN = 512 if not IS_SMOKE else 128
CALIB_SAMPLES = 30 if not IS_SMOKE else 3
N_EXPERTS = 50 if not IS_SMOKE else 6
DROPOUT_FRAC = 0.8  # keep 80% of experts
N_BOOTSTRAP = 20 if not IS_SMOKE else 3
SEED = 42
MAX_RUNTIME_S = 15 * 60  # 15-minute hard cutoff (budget constraint)


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def discover_adapters():
    adapters = []
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    for d in sorted(ADAPTER_DIR.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            adapters.append(d.name)
    return adapters


def compose_adapters_on_cpu(adapter_names):
    """Compose adapters on CPU via additive sum, save as single adapter.

    Composition is ADDITIVE (sum), not averaged. Each adapter's full delta
    is added without division by N. This matches SOLE production config.
    """
    from safetensors.torch import load_file, save_file
    import torch

    composed = {}
    for name in adapter_names:
        path = ADAPTER_DIR / name / "adapter_model.safetensors"
        tensors = load_file(str(path), device="cpu")
        for key, val in tensors.items():
            if key in composed:
                composed[key] = composed[key] + val.float()
            else:
                composed[key] = val.float()
        del tensors
        gc.collect()

    tmpdir = tempfile.mkdtemp(prefix="composed_")
    out_path = os.path.join(tmpdir, "adapter_model.safetensors")
    save_file({k: v.to(torch.bfloat16) for k, v in composed.items()}, out_path)

    cfg_src = ADAPTER_DIR / adapter_names[0] / "adapter_config.json"
    if cfg_src.exists():
        shutil.copy(str(cfg_src), os.path.join(tmpdir, "adapter_config.json"))

    del composed
    gc.collect()
    return tmpdir


def load_calibration_data(adapters, n_samples, rng):
    """Load calibration texts from adapter domains, balanced across domains.

    Uses seeded rng for reproducibility. Collects up to 3 texts per domain
    for balance, falls back to collecting any available if not enough domains.
    """
    # Collect per-domain texts (up to 3 per domain for balance)
    per_domain_cap = max(3, math.ceil(n_samples / max(1, len(adapters))))
    per_domain = {}
    for domain in adapters:
        domain_dir = DATA_DIR / domain
        if not domain_dir.exists():
            continue
        texts = []
        for f in sorted(domain_dir.glob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if "messages" in obj:
                            text = " ".join(
                                m.get("content", "") for m in obj["messages"]
                                if m.get("role") in ("user", "assistant")
                            )
                        else:
                            text = obj.get("text", obj.get("content", obj.get("output", "")))
                        if text and len(text) > 100:
                            texts.append(text[:MAX_SEQ_LEN * 4])
                    except Exception:
                        continue
                    if len(texts) >= per_domain_cap:
                        break
            if len(texts) >= per_domain_cap:
                break
        if texts:
            per_domain[domain] = texts

    # Flatten, shuffle with seeded rng, take n_samples
    all_texts = [t for texts in per_domain.values() for t in texts]
    rng.shuffle(all_texts)
    return all_texts[:n_samples]


def compute_ppl(model, tokenizer, texts):
    """Compute perplexity over texts."""
    import torch

    total_loss = 0.0
    total_tokens = 0

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Cast to float32 for numerical stability before loss computation
            shift_logits = outputs.logits[:, :-1, :].contiguous().float()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
        del inputs, outputs
        torch.cuda.empty_cache()

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def load_model_4bit(base_model_id, hf_cache):
    """Load Qwen2.5-7B with 4-bit NF4 quantization for ~5GB GPU footprint."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=hf_cache,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )


def main():
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Seeded RNG instance for subset sampling and calibration shuffling
    rng = random.Random(SEED)

    t0 = time.time()

    log("=" * 60)
    log("EXPERT DROPOUT COMPOSITION ROBUSTNESS")
    log(f"  N experts target: {N_EXPERTS}")
    log(f"  Bootstrap samples: {N_BOOTSTRAP}")
    log(f"  Calibration samples: {CALIB_SAMPLES}")
    log(f"  Max seq len: {MAX_SEQ_LEN}")
    log(f"  Max runtime: {MAX_RUNTIME_S}s")
    log(f"  Smoke test: {IS_SMOKE}")
    log("=" * 60)

    # ------------------------------------------------------------------ #
    # Phase 0: Setup and Discovery
    # ------------------------------------------------------------------ #
    t_phase0 = time.time()

    all_adapters = discover_adapters()
    log(f"Found {len(all_adapters)} adapters")
    n_experts = min(N_EXPERTS, len(all_adapters))
    if n_experts < 3:
        log(f"ERROR: need at least 3 adapters, found {n_experts}")
        sys.exit(1)
    selected = all_adapters[:n_experts]
    n_keep = max(2, int(n_experts * DROPOUT_FRAC))
    log(f"Using {len(selected)} adapters, keeping {n_keep} per sample")

    log("Loading calibration data...")
    calib = load_calibration_data(selected, CALIB_SAMPLES, rng)
    log(f"Calibration: {len(calib)} texts")
    if len(calib) < 3:
        log("ERROR: insufficient calibration data")
        sys.exit(1)

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    phase0_s = time.time() - t_phase0
    log(f"Phase 0 done in {phase0_s:.1f}s")

    # ------------------------------------------------------------------ #
    # Phase 1–3: Heavy GPU compute — disable GC (nanochat pattern: ~500ms/step saved)
    # ------------------------------------------------------------------ #
    gc.disable()
    gc.collect()
    try:
        # ------------------------------------------------------------------ #
        # Phase 1: Reference — compose ALL N adapters
        # ------------------------------------------------------------------ #
        t_phase1 = time.time()
        log(f"\nPhase 1: Composing all {len(selected)} adapters (reference)...")
        ref_dir = compose_adapters_on_cpu(selected)
        model = load_model_4bit(BASE_MODEL, HF_CACHE)
        model = PeftModel.from_pretrained(model, ref_dir)
        model.eval()
        ref_ppl = compute_ppl(model, tokenizer, calib)
        log(f"Reference PPL (all {len(selected)}): {ref_ppl:.4f}")
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(ref_dir)
        phase1_s = time.time() - t_phase1
        log(f"Phase 1 done in {phase1_s:.1f}s")

        # ------------------------------------------------------------------ #
        # Phase 2: Base model PPL
        # ------------------------------------------------------------------ #
        t_phase2 = time.time()
        log("\nPhase 2: Measuring base model PPL...")
        base_m = load_model_4bit(BASE_MODEL, HF_CACHE)
        base_m.eval()
        base_ppl = compute_ppl(base_m, tokenizer, calib)
        log(f"Base PPL: {base_ppl:.4f}")
        del base_m
        torch.cuda.empty_cache()
        phase2_s = time.time() - t_phase2
        log(f"Phase 2 done in {phase2_s:.1f}s")

        # ------------------------------------------------------------------ #
        # Phase 3: Bootstrap — 20 random 80% subsets
        # ------------------------------------------------------------------ #
        t_phase3 = time.time()
        bootstrap_results = []

        for b in range(N_BOOTSTRAP):
            elapsed_so_far = time.time() - t0
            if elapsed_so_far > MAX_RUNTIME_S:
                log(f"MAX_RUNTIME ({MAX_RUNTIME_S}s) reached after {b} bootstrap iterations. Stopping early.")
                break

            subset = sorted(rng.sample(selected, n_keep))
            log(f"\n[Bootstrap {b+1}/{N_BOOTSTRAP}] {n_keep} experts "
                f"(elapsed: {elapsed_so_far:.0f}s / {MAX_RUNTIME_S}s)")

            sub_dir = compose_adapters_on_cpu(subset)
            model = load_model_4bit(BASE_MODEL, HF_CACHE)
            model = PeftModel.from_pretrained(model, sub_dir)
            model.eval()
            ppl = compute_ppl(model, tokenizer, calib)

            delta_from_ref = (ppl - ref_ppl) / ref_ppl * 100
            delta_from_base = (ppl - base_ppl) / base_ppl * 100

            log(f"  PPL={ppl:.4f} (vs ref: {delta_from_ref:+.3f}%, vs base: {delta_from_base:+.3f}%)")

            bootstrap_results.append({
                "bootstrap_id": b,
                "n_experts": n_keep,
                "subset": subset,
                "ppl": ppl,
                "delta_from_ref_pct": delta_from_ref,
                "delta_from_base_pct": delta_from_base,
            })

            del model
            torch.cuda.empty_cache()
            shutil.rmtree(sub_dir)

        phase3_s = time.time() - t_phase3
    finally:
        gc.enable()
        gc.collect()
    log(f"\nPhase 3 done in {phase3_s:.1f}s ({len(bootstrap_results)} subsets evaluated)")

    # ------------------------------------------------------------------ #
    # Phase 4: Analysis
    # ------------------------------------------------------------------ #
    t_phase4 = time.time()
    log(f"\n{'='*60}")
    log("ANALYSIS")
    log(f"{'='*60}")

    ppls = np.array([r["ppl"] for r in bootstrap_results])
    deltas = np.array([r["delta_from_ref_pct"] for r in bootstrap_results])

    mean_ppl = float(np.mean(ppls))
    std_ppl = float(np.std(ppls))
    cv_ppl = std_ppl / mean_ppl * 100 if mean_ppl > 0 else 0.0
    median_ppl = float(np.median(ppls))
    iqr_ppl = float(np.percentile(ppls, 75) - np.percentile(ppls, 25))
    best_delta = float(np.min(deltas))   # most negative = most improvement
    worst_delta = float(np.max(deltas))  # most positive = most degradation

    log(f"\nPPL distribution across {len(bootstrap_results)} subsets:")
    log(f"  Mean:   {mean_ppl:.4f}")
    log(f"  Std:    {std_ppl:.4f}")
    log(f"  Median: {median_ppl:.4f}")
    log(f"  IQR:    {iqr_ppl:.4f}")
    log(f"  CV:     {cv_ppl:.3f}%")
    log(f"  Best delta from ref:  {best_delta:+.3f}%")
    log(f"  Worst delta from ref: {worst_delta:+.3f}%")
    log(f"  Reference (all {n_experts}): {ref_ppl:.4f}")
    log(f"  Base model: {base_ppl:.4f}")

    # Kill criteria
    # K1: CV > 5% kills (composition is fragile)
    k1_pass = cv_ppl <= 5.0
    # K2: best_delta < -10% kills (some experts are harmful, pruning is critical)
    #     best_delta is most-negative delta; < -10 means a subset improved >10% by dropping experts
    k2_pass = best_delta >= -10.0
    # K3: worst_delta > 15% kills (random dropout is dangerous)
    k3_pass = worst_delta <= 15.0

    log(f"\nKILL CRITERIA:")
    log(f"  K1 (CV <= 5%):        {cv_ppl:.3f}%  -> {'PASS' if k1_pass else 'FAIL — composition is fragile'}")
    log(f"  K2 (best delta >= -10%): {best_delta:+.3f}% -> {'PASS' if k2_pass else 'FAIL — pruning is critical'}")
    log(f"  K3 (worst delta <= 15%): {worst_delta:+.3f}% -> {'PASS' if k3_pass else 'FAIL — dropout is dangerous'}")

    overall = k1_pass and k2_pass and k3_pass
    log(f"\n  OVERALL: {'SURVIVES' if overall else 'KILLED'}")

    phase4_s = time.time() - t_phase4
    elapsed = time.time() - t0

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    results = {
        "config": {
            "n_experts": n_experts,
            "n_keep": n_keep,
            "dropout_frac": DROPOUT_FRAC,
            "n_bootstrap": N_BOOTSTRAP,
            "calib_samples": CALIB_SAMPLES,
            "max_seq_len": MAX_SEQ_LEN,
            "base_model": BASE_MODEL,
            "seed": SEED,
            "smoke_test": IS_SMOKE,
        },
        "reference_ppl": ref_ppl,
        "base_ppl": base_ppl,
        "bootstrap_summary": {
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "cv_pct": cv_ppl,
            "median_ppl": median_ppl,
            "iqr_ppl": iqr_ppl,
            "best_delta_pct": best_delta,
            "worst_delta_pct": worst_delta,
        },
        "kill_criteria": {
            "K1_cv_pct": cv_ppl,
            "K1_threshold": 5.0,
            "K1_pass": k1_pass,
            "K2_best_delta_pct": best_delta,
            "K2_threshold": 10.0,
            "K2_pass": k2_pass,
            "K3_worst_delta_pct": worst_delta,
            "K3_threshold": 15.0,
            "K3_pass": k3_pass,
            "overall": "SURVIVES" if overall else "KILLED",
        },
        "bootstrap_details": bootstrap_results,
        "elapsed_s": elapsed,
        "per_phase_timing": {
            "phase0_setup_s": phase0_s,
            "phase1_reference_s": phase1_s,
            "phase2_base_s": phase2_s,
            "phase3_bootstrap_s": phase3_s,
            "phase4_analysis_s": phase4_s,
        },
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
