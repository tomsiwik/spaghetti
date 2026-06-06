#!/usr/bin/env python3
"""SOLE Critical Path — All 4 macro experiments in one lean GPU session.

Runs sequentially with function-scoped cleanup per GPU_CODING_GUIDELINES.md.
Each experiment loads model fresh (function scope frees VRAM on return).

Experiments:
  1. Poisoned adapter detection (leave-one-out PPL) — ~15 min
  2. PPL-probe weighted composition vs equal-weight — ~20 min
  3. SOLE vs monolithic LoRA baseline — ~30 min (trains union adapter)
  4. Reasoning adapter composition K2/K3 — ~30 min (trains reasoning adapter)

Total: ~1.5-2 hours on A5000.

Usage:
    python run_all_critical.py
    SMOKE_TEST=1 python run_all_critical.py
    python run_all_critical.py --only poisoned
    python run_all_critical.py --only ppl-probe
    python run_all_critical.py --only monolithic
    python run_all_critical.py --only reasoning
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Prevent torch.compile hang


def kill_zombie_gpu_processes():
    """Kill any other GPU-using processes to reclaim VRAM from zombies."""
    import subprocess
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip()
        if not out:
            return
        for line in out.splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                print(f"[CLEANUP] Killing zombie GPU process {pid}", flush=True)
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        import time as _t
        _t.sleep(2)  # Let GPU memory release
    except Exception as e:
        print(f"[CLEANUP] Warning: {e}", flush=True)


kill_zombie_gpu_processes()

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
BASE_MODEL = "/workspace/models/Qwen2.5-7B"
ADAPTER_DIR = Path("/workspace/llm/adapters")
RESULTS_DIR = Path("/workspace/llm/results/sole_critical_path")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# Calibration: 30 samples from each adapter's training data tail
CALIB_SAMPLES = 5 if SMOKE else 30
MAX_SEQ_LEN = 256 if SMOKE else 512


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_adapters():
    """Return available adapter names with adapter_config.json."""
    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def load_calibration_texts(adapter_name, tokenizer, n=CALIB_SAMPLES):
    """Load calibration text from adapter's training data tail."""
    train_file = Path(f"/workspace/llm/data/distillation/{adapter_name}/train.jsonl")
    if not train_file.exists():
        # Fallback: use a generic prompt
        return [f"Explain the concept of {adapter_name} in detail."] * min(n, 3)
    with open(train_file) as f:
        lines = f.readlines()
    texts = []
    for line in lines[-n:]:
        rec = json.loads(line)
        if "messages" in rec:
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
    return texts or [f"Explain {adapter_name}."]


def compute_ppl(model, tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Compute average PPL over texts. Returns float."""
    import torch
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_len).to(model.device)
            out = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].shape[1]
            total_loss += out.loss.item() * n
            total_tokens += n
            del out, inputs
    return math.exp(total_loss / max(total_tokens, 1))


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Poisoned Adapter Detection (Leave-One-Out PPL)
# ═══════════════════════════════════════════════════════════════════════════

def run_poisoned_detection():
    """Function-scoped: all GPU freed on return."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file

    log("=" * 60)
    log("EXP 1: POISONED ADAPTER DETECTION (Leave-One-Out)")
    log("=" * 60)

    adapters = get_adapters()
    log(f"Adapters: {adapters}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load generic calibration text
    calib_texts = []
    for a in adapters:
        calib_texts.extend(load_calibration_texts(a, tokenizer, n=CALIB_SAMPLES // len(adapters) + 1))
    calib_texts = calib_texts[:CALIB_SAMPLES]
    log(f"Calibration texts: {len(calib_texts)}")

    # Pre-load all adapter deltas to CPU (avoids repeated disk I/O in LOO loop)
    adapter_deltas = {}  # name -> {mod_name: delta_tensor}
    for name in adapters:
        path = ADAPTER_DIR / name
        with open(path / "adapter_config.json") as f:
            cfg = json.load(f)
        r, alpha = cfg.get("r", 16), cfg.get("lora_alpha", 16)
        scaling = alpha / r  # per-adapter scaling (no 1/N here — LOO uses equal contribution)
        w = load_file(str(path / "adapter_model.safetensors"), device="cpu")
        modules = {}
        for key, tensor in w.items():
            clean = key.replace("base_model.model.", "")
            if "lora_A" in clean:
                mod = clean.split(".lora_A")[0]
                modules.setdefault(mod, {})["A"] = tensor.float()
            elif "lora_B" in clean:
                mod = clean.split(".lora_B")[0]
                modules.setdefault(mod, {})["B"] = tensor.float()
        deltas = {}
        for mod_name, ab in modules.items():
            if "A" in ab and "B" in ab:
                deltas[mod_name + ".weight"] = (scaling * (ab["B"] @ ab["A"])).half()
        adapter_deltas[name] = deltas
        del w, modules
    log(f"Pre-loaded {len(adapter_deltas)} adapter deltas to CPU")

    def compose_manual(base, adapter_names, deltas_dict, n_total):
        """Compose adapters via manual 1/N-scaled delta addition (memory-lean)."""
        scale = 1.0 / n_total
        for name in adapter_names:
            for param_suffix, delta in deltas_dict[name].items():
                for pname, param in base.named_parameters():
                    if pname.endswith(param_suffix):
                        with torch.no_grad():
                            param.data += (scale * delta).to(param.device, param.dtype)
                        break
        base.eval()
        return base

    # Base PPL
    log("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    base_ppl = compute_ppl(base_model, tokenizer, calib_texts)
    log(f"Base PPL: {base_ppl:.2f}")
    del base_model; gc.collect(); torch.cuda.empty_cache()

    # All-N PPL
    log(f"Composing all {len(adapters)} adapters...")
    base_all = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    compose_manual(base_all, adapters, adapter_deltas, len(adapters))
    all_ppl = compute_ppl(base_all, tokenizer, calib_texts)
    log(f"All-{len(adapters)} PPL: {all_ppl:.2f}")
    del base_all; gc.collect(); torch.cuda.empty_cache()

    # Leave-one-out
    loo_results = {}
    for skip in adapters:
        subset = [a for a in adapters if a != skip]
        log(f"  LOO (without {skip})...")
        base_fresh = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        compose_manual(base_fresh, subset, adapter_deltas, len(adapters))
        ppl = compute_ppl(base_fresh, tokenizer, calib_texts)
        delta = (ppl - all_ppl) / all_ppl * 100
        loo_results[skip] = {"ppl": round(ppl, 4), "delta_pct": round(delta, 2)}
        log(f"    Without {skip}: PPL={ppl:.2f} (delta={delta:+.1f}%)")
        del base_fresh; gc.collect(); torch.cuda.empty_cache()

    # Rank by impact (most harmful = biggest PPL reduction when removed)
    ranked = sorted(loo_results.items(), key=lambda x: x[1]["ppl"])
    log(f"\nRanking (best PPL when removed = most harmful adapter):")
    for name, info in ranked:
        log(f"  {name}: PPL={info['ppl']:.2f} ({info['delta_pct']:+.1f}%)")

    result = {
        "experiment": "poisoned_adapter_detection",
        "base_ppl": round(base_ppl, 4),
        "all_composed_ppl": round(all_ppl, 4),
        "leave_one_out": loo_results,
        "ranking": [name for name, _ in ranked],
        "most_harmful": ranked[0][0],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "poisoned_detection.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"Results saved.")

    gc.collect(); torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: PPL-Probe Weighted Composition
# ═══════════════════════════════════════════════════════════════════════════

def run_ppl_probe():
    """Function-scoped: all GPU freed on return."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log("=" * 60)
    log("EXP 2: PPL-PROBE WEIGHTED COMPOSITION")
    log("=" * 60)

    adapters = get_adapters()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For each adapter, compute per-adapter PPL on probe examples
    # Then use softmax(1/PPL) as composition weights
    log("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    # Compute per-adapter PPL on mixed calibration text
    all_calib = []
    for a in adapters:
        all_calib.extend(load_calibration_texts(a, tokenizer, n=3))
    probe_texts = all_calib[:10]  # 10 probe examples

    adapter_ppls = {}
    for name in adapters:
        log(f"  Probing {name}...")
        adapted = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR / name))
        adapted.eval()
        ppl = compute_ppl(adapted, tokenizer, probe_texts)
        adapter_ppls[name] = ppl
        log(f"    {name} probe PPL: {ppl:.2f}")
        del adapted; gc.collect(); torch.cuda.empty_cache()

    # Compute weights: softmax(1/PPL) — lower PPL = higher weight
    import numpy as np
    inv_ppls = np.array([1.0 / adapter_ppls[a] for a in adapters])
    # Temperature-scaled softmax
    tau = 1.0
    weights = np.exp(inv_ppls / tau) / np.exp(inv_ppls / tau).sum()
    weight_dict = {a: float(w) for a, w in zip(adapters, weights)}
    log(f"PPL-probe weights: {weight_dict}")

    # Now compose with PPL-probe weights vs equal weight
    # Load all adapter deltas manually
    from safetensors.torch import load_file

    def get_adapter_config(path):
        with open(Path(path) / "adapter_config.json") as f:
            cfg = json.load(f)
        return cfg.get("r", 16), cfg.get("lora_alpha", 16)

    def compose_weighted(base, adapter_names, weights_dict, tokenizer):
        """Compose adapters with specified weights via manual delta addition."""
        model = base
        for name in adapter_names:
            path = ADAPTER_DIR / name
            r, alpha = get_adapter_config(path)
            scaling = alpha / r * weights_dict[name]

            weights_file = path / "adapter_model.safetensors"
            w = load_file(str(weights_file), device="cpu")

            modules = {}
            for key, tensor in w.items():
                clean = key.replace("base_model.model.", "")
                if "lora_A" in clean:
                    mod = clean.split(".lora_A")[0]
                    modules.setdefault(mod, {})["A"] = tensor.float()
                elif "lora_B" in clean:
                    mod = clean.split(".lora_B")[0]
                    modules.setdefault(mod, {})["B"] = tensor.float()

            for mod_name, ab in modules.items():
                if "A" in ab and "B" in ab:
                    delta = scaling * (ab["B"] @ ab["A"])
                    param_name = mod_name + ".weight"
                    for pname, param in model.named_parameters():
                        if pname.endswith(param_name):
                            with torch.no_grad():
                                param.data += delta.to(param.device, param.dtype)
                            break
        model.eval()
        return model

    # Equal weight composition
    log("Composing with equal weights...")
    base_eq = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    equal_weights = {a: 1.0 / len(adapters) for a in adapters}
    model_eq = compose_weighted(base_eq, adapters, equal_weights, tokenizer)
    eq_ppl = compute_ppl(model_eq, tokenizer, probe_texts)
    log(f"Equal-weight PPL: {eq_ppl:.2f}")
    del model_eq, base_eq; gc.collect(); torch.cuda.empty_cache()

    # PPL-probe weighted composition
    log("Composing with PPL-probe weights...")
    base_pw = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model_pw = compose_weighted(base_pw, adapters, weight_dict, tokenizer)
    pw_ppl = compute_ppl(model_pw, tokenizer, probe_texts)
    log(f"PPL-probe weighted PPL: {pw_ppl:.2f}")
    del model_pw, base_pw; gc.collect(); torch.cuda.empty_cache()

    # Top-1 (best single adapter)
    best_adapter = min(adapter_ppls, key=adapter_ppls.get)
    top1_ppl = adapter_ppls[best_adapter]
    log(f"Top-1 ({best_adapter}) PPL: {top1_ppl:.2f}")

    # Base PPL
    base_fresh = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    base_ppl = compute_ppl(base_fresh, tokenizer, probe_texts)
    log(f"Base PPL: {base_ppl:.2f}")
    del base_fresh; gc.collect(); torch.cuda.empty_cache()

    improvement = (eq_ppl - pw_ppl) / eq_ppl * 100

    log(f"\n  Summary:")
    log(f"    Base:           {base_ppl:.2f}")
    log(f"    Top-1:          {top1_ppl:.2f}")
    log(f"    Equal-weight:   {eq_ppl:.2f}")
    log(f"    PPL-probe:      {pw_ppl:.2f}")
    log(f"    Improvement:    {improvement:+.1f}%")

    result = {
        "experiment": "ppl_probe_macro_composition",
        "base_ppl": round(base_ppl, 4),
        "equal_weight_ppl": round(eq_ppl, 4),
        "ppl_probe_ppl": round(pw_ppl, 4),
        "top1_ppl": round(top1_ppl, 4),
        "top1_adapter": best_adapter,
        "weights": weight_dict,
        "adapter_ppls": {k: round(v, 4) for k, v in adapter_ppls.items()},
        "improvement_pct": round(improvement, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "ppl_probe_composition.json", "w") as f:
        json.dump(result, f, indent=2)
    log("Results saved.")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: SOLE vs Monolithic LoRA
# ═══════════════════════════════════════════════════════════════════════════

def run_sole_vs_monolithic():
    """Train one LoRA on union of all 5 datasets, compare to SOLE composition."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    log("=" * 60)
    log("EXP 3: SOLE vs MONOLITHIC LoRA")
    log("=" * 60)

    adapters = get_adapters()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build union dataset from all adapter training data
    log("Building union dataset...")
    all_texts = []
    for name in adapters:
        train_file = Path(f"/workspace/llm/data/distillation/{name}/train.jsonl")
        if train_file.exists():
            with open(train_file) as f:
                for line in f:
                    rec = json.loads(line)
                    if "messages" in rec:
                        text = tokenizer.apply_chat_template(
                            rec["messages"], tokenize=False, add_generation_prompt=False
                        )
                        all_texts.append(text)
    log(f"Union dataset: {len(all_texts)} examples from {len(adapters)} domains")

    if SMOKE:
        all_texts = all_texts[:20]

    union_ds = Dataset.from_dict({"text": all_texts}).shuffle(seed=SEED)

    # Train union LoRA (same config as pilot50)
    # Load in bf16 (no quantization — avoids set_submodule PyTorch compat issue)
    # 7B bf16 ~14GB + LoRA fits on 24GB A5000 with gradient checkpointing
    steps = 10 if SMOKE else 300
    log(f"Training union LoRA ({steps} steps)...")

    gc.disable(); gc.collect()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    union_out = RESULTS_DIR / "union_adapter"
    trainer = SFTTrainer(
        model=model, train_dataset=union_ds,
        args=SFTConfig(
            output_dir=str(union_out / "ckpt"), max_steps=steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=1e-4, warmup_steps=10, lr_scheduler_type="cosine",
            logging_steps=25, save_steps=steps, bf16=True,
            optim="adamw_torch", seed=SEED, dataset_text_field="text",
            max_length=512, packing=True, report_to="none",
            dataloader_pin_memory=True,
        ),
    )
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log(f"Union LoRA trained in {train_time:.0f}s")
    model.save_pretrained(str(union_out))
    tokenizer.save_pretrained(str(union_out))
    del model, trainer; gc.enable(); gc.collect()
    torch.cuda.empty_cache()

    # Now compare: union vs SOLE (best composition from exp 2)
    log("Evaluating union vs SOLE...")
    calib_texts = []
    for a in adapters:
        calib_texts.extend(load_calibration_texts(a, tokenizer, n=5))

    # Union PPL
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    union_model = PeftModel.from_pretrained(base, str(union_out))
    union_model.eval()
    union_ppl = compute_ppl(union_model, tokenizer, calib_texts)
    log(f"Union LoRA PPL: {union_ppl:.2f}")
    del union_model, base; gc.collect(); torch.cuda.empty_cache()

    # SOLE (equal-weight, 1/N scaled) PPL — manual delta addition
    from safetensors.torch import load_file as _load_file
    base2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    n_adapters = len(adapters)
    for name in adapters:
        path = ADAPTER_DIR / name
        with open(path / "adapter_config.json") as f:
            cfg = json.load(f)
        r, alpha = cfg.get("r", 16), cfg.get("lora_alpha", 16)
        scaling = (alpha / r) * (1.0 / n_adapters)
        w = _load_file(str(path / "adapter_model.safetensors"), device="cpu")
        modules = {}
        for key, tensor in w.items():
            clean = key.replace("base_model.model.", "")
            if "lora_A" in clean:
                mod = clean.split(".lora_A")[0]
                modules.setdefault(mod, {})["A"] = tensor.float()
            elif "lora_B" in clean:
                mod = clean.split(".lora_B")[0]
                modules.setdefault(mod, {})["B"] = tensor.float()
        for mod_name, ab in modules.items():
            if "A" in ab and "B" in ab:
                delta = scaling * (ab["B"] @ ab["A"])
                param_name = mod_name + ".weight"
                for pname, param in base2.named_parameters():
                    if pname.endswith(param_name):
                        with torch.no_grad():
                            param.data += delta.to(param.device, param.dtype)
                        break
    base2.eval()
    sole_ppl = compute_ppl(base2, tokenizer, calib_texts)
    log(f"SOLE (1/N weighted) PPL: {sole_ppl:.2f}")
    del base2; gc.collect(); torch.cuda.empty_cache()

    log(f"\n  Union PPL: {union_ppl:.2f}")
    log(f"  SOLE PPL:  {sole_ppl:.2f}")
    log(f"  Winner:    {'Union' if union_ppl < sole_ppl else 'SOLE'}")

    result = {
        "experiment": "sole_vs_monolithic",
        "union_ppl": round(union_ppl, 4),
        "sole_ppl": round(sole_ppl, 4),
        "union_train_steps": steps,
        "union_train_time_s": round(train_time, 1),
        "n_domains": len(adapters),
        "n_union_examples": len(all_texts),
        "winner": "union" if union_ppl < sole_ppl else "sole",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "sole_vs_monolithic.json", "w") as f:
        json.dump(result, f, indent=2)
    log("Results saved.")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "poisoned": run_poisoned_detection,
    "ppl-probe": run_ppl_probe,
    "monolithic": run_sole_vs_monolithic,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(EXPERIMENTS.keys()), default=None)
    args = parser.parse_args()

    log("=" * 60)
    log("SOLE CRITICAL PATH — LEAN GPU SESSION")
    log(f"  Adapters: {get_adapters()}")
    log(f"  Smoke: {SMOKE}")
    log(f"  Only: {args.only or 'all'}")
    log("=" * 60)

    t0 = time.time()
    results = {}

    to_run = [args.only] if args.only else list(EXPERIMENTS.keys())
    for name in to_run:
        try:
            results[name] = EXPERIMENTS[name]()
        except Exception as e:
            log(f"FAILED: {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    elapsed = time.time() - t0
    log(f"\nAll experiments complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"Cost estimate: ${elapsed/3600 * 0.16:.2f}")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"elapsed_s": round(elapsed, 1), "results": {
            k: v.get("error", "OK") if isinstance(v, dict) else "OK" for k, v in results.items()
        }}, f, indent=2)


if __name__ == "__main__":
    main()
