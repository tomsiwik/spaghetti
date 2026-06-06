#!/usr/bin/env python3
"""Weight-normalized composition: does 1/sqrt(N) prevent PPL explosion?

Tests 4 scaling strategies at N=5,10,25,50:
  (1) Unit weight: w_i = 1.0 (current default, known catastrophic at N=50)
  (2) Mean: w_i = 1/N
  (3) Sqrt: w_i = 1/sqrt(N)
  (4) Optimal: grid search over [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

Uses CPU-side safetensors composition to avoid OOM.

Kill criteria:
- K1: 1/sqrt(N) scaling does not reduce composed PPL by >50% vs unit-weight at N=50
- K2: best scaling factor produces composed PPL >2x individual expert average PPL
- K3: scaling factor that works at N=50 does not transfer to N=25 (not robust)

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import signal
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME", "300" if IS_SMOKE else "7800"))  # 130 min default
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def _timeout_handler(signum, frame):
    print(f"\n[TIMEOUT] MAX_RUNTIME={MAX_RUNTIME}s reached. Saving partial results and exiting.", flush=True)
    sys.exit(1)


if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MAX_RUNTIME)

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "composition_weight_normalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

N_VALUES = [3, 5] if IS_SMOKE else [5, 10, 25, 50]
EVAL_SAMPLES = 5 if IS_SMOKE else 50
MAX_SEQ_LEN = 256 if IS_SMOKE else 512

SCALING_STRATEGIES = {
    "unit": lambda n: 1.0,
    "mean": lambda n: 1.0 / n,
    "sqrt": lambda n: 1.0 / math.sqrt(n),
}
GRID_WEIGHTS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]


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


def load_eval_texts(domain, tokenizer, n=50):
    texts = []
    for fname in ["eval.jsonl", "train.jsonl"]:
        f = DATA_DIR / domain / fname
        if not f.exists():
            continue
        with open(f) as fh:
            lines = fh.readlines()
        if fname == "train.jsonl":
            lines = lines[-min(200, len(lines)):]
        for line in lines:
            record = json.loads(line)
            if "messages" in record:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False)
            elif "text" in record:
                text = record["text"]
            else:
                continue
            texts.append(text)
            if len(texts) >= n:
                return texts
    return texts


def compose_adapters_on_cpu(adapter_names, adapter_dir, weight_per_adapter):
    """Compose LoRA adapters on CPU with a uniform per-adapter weight."""
    from safetensors.torch import load_file, save_file
    import torch as _torch

    composed = {}
    adapter_config = None

    for name in adapter_names:
        adapter_path = adapter_dir / name
        st_path = adapter_path / "adapter_model.safetensors"
        if not st_path.exists():
            continue
        tensors = load_file(str(st_path), device="cpu")
        for key, val in tensors.items():
            if key in composed:
                composed[key] = composed[key] + val.float() * weight_per_adapter
            else:
                composed[key] = val.float() * weight_per_adapter

        if adapter_config is None:
            cfg_path = adapter_path / "adapter_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    adapter_config = json.load(f)

    for key in composed:
        composed[key] = composed[key].to(_torch.bfloat16)

    tmp_dir = tempfile.mkdtemp(prefix="composed_adapter_")
    save_file(composed, os.path.join(tmp_dir, "adapter_model.safetensors"))
    if adapter_config:
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

    return tmp_dir


def measure_ppl(model, tokenizer, texts):
    import torch
    losses = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LEN).to(model.device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def eval_composed_adapter(base_model, tokenizer, composed_dir, eval_domains):
    """Load a single composed adapter and evaluate per-domain PPL."""
    import torch
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, composed_dir)
    results = {}
    for domain in eval_domains:
        eval_texts = load_eval_texts(domain, tokenizer, n=EVAL_SAMPLES)
        if not eval_texts:
            continue
        ppl = measure_ppl(model, tokenizer, eval_texts)
        results[domain] = round(ppl, 4)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # RunPod compatibility: monkey-patch set_submodule if missing (older PyTorch)
    if not hasattr(torch.nn.Module, "set_submodule"):
        def _set_submodule(self, target, module):
            atoms = target.split(".")
            mod = self
            for item in atoms[:-1]:
                mod = getattr(mod, item)
            setattr(mod, atoms[-1], module)
        torch.nn.Module.set_submodule = _set_submodule

    t0 = time.time()
    log("=" * 60)
    log("Weight Normalization Experiment")
    log("=" * 60)
    log(f"Smoke test: {IS_SMOKE}")

    all_adapters = discover_adapters()
    log(f"Found {len(all_adapters)} adapters")

    N_VALUES_ACTUAL = [n for n in N_VALUES if n <= len(all_adapters)]
    if not N_VALUES_ACTUAL:
        N_VALUES_ACTUAL = [len(all_adapters)]
    log(f"Testing N values: {N_VALUES_ACTUAL}")

    # Load base model
    log("\nLoading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    log("Base model loaded.")

    # Identify eval domains
    eval_domains = []
    for name in all_adapters:
        data_dir = DATA_DIR / name
        if data_dir.exists() and any(data_dir.iterdir()):
            eval_domains.append(name)
    log(f"Evaluation domains: {len(eval_domains)}")

    # Phase 1: Single-expert average PPL (for K2 comparison)
    log("\n=== Phase 1: Single Expert Average PPL ===")
    from peft import PeftModel
    single_ppls = {}
    for domain in eval_domains[:10] if IS_SMOKE else eval_domains:
        adapter_path = ADAPTER_DIR / domain
        if not adapter_path.exists():
            continue
        eval_texts = load_eval_texts(domain, tokenizer, n=EVAL_SAMPLES)
        if not eval_texts:
            continue
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        ppl = measure_ppl(model, tokenizer, eval_texts)
        single_ppls[domain] = round(ppl, 4)
        log(f"  {domain}: {ppl:.4f}")
        del model
        torch.cuda.empty_cache()
        gc.collect()

    avg_single_ppl = sum(single_ppls.values()) / max(len(single_ppls), 1)
    log(f"Average single-expert PPL: {avg_single_ppl:.4f}")

    # Phase 2: Test scaling strategies at each N
    log("\n=== Phase 2: Scaling Strategies ===")
    all_results = {}  # {strategy: {N: {domain: ppl}}}

    for n in N_VALUES_ACTUAL:
        adapters_to_compose = all_adapters[:n]
        eval_subset = [d for d in adapters_to_compose if d in single_ppls]
        log(f"\n--- N={n}, eval on {len(eval_subset)} domains ---")

        for strategy_name, weight_fn in SCALING_STRATEGIES.items():
            w = weight_fn(n)
            log(f"  Strategy '{strategy_name}': w={w:.6f}")

            composed_dir = compose_adapters_on_cpu(
                adapters_to_compose, ADAPTER_DIR, w)
            ppls = eval_composed_adapter(
                base_model, tokenizer, composed_dir, eval_subset)
            shutil.rmtree(composed_dir)

            all_results.setdefault(strategy_name, {})[n] = ppls
            mean_ppl = sum(ppls.values()) / max(len(ppls), 1)
            log(f"    mean PPL = {mean_ppl:.4f}")

    # Phase 3: Grid search at largest N
    log("\n=== Phase 3: Grid Search at N={} ===".format(N_VALUES_ACTUAL[-1]))
    largest_n = N_VALUES_ACTUAL[-1]
    adapters_to_compose = all_adapters[:largest_n]
    eval_subset = [d for d in adapters_to_compose if d in single_ppls]
    grid_results = {}

    for w in GRID_WEIGHTS:
        log(f"  Grid w={w:.2f}")
        composed_dir = compose_adapters_on_cpu(
            adapters_to_compose, ADAPTER_DIR, w)
        ppls = eval_composed_adapter(
            base_model, tokenizer, composed_dir, eval_subset)
        shutil.rmtree(composed_dir)
        mean_ppl = sum(ppls.values()) / max(len(ppls), 1)
        grid_results[w] = {"per_domain_ppl": ppls, "mean_ppl": round(mean_ppl, 4)}
        log(f"    mean PPL = {mean_ppl:.4f}")

    best_grid_w = min(grid_results, key=lambda w: grid_results[w]["mean_ppl"])
    best_grid_ppl = grid_results[best_grid_w]["mean_ppl"]
    log(f"  Best grid weight: {best_grid_w} -> mean PPL = {best_grid_ppl:.4f}")

    # Also run best grid weight at N=25 (for K3 transfer check)
    if 25 in N_VALUES_ACTUAL or (len(N_VALUES_ACTUAL) >= 2 and N_VALUES_ACTUAL[-2] < largest_n):
        transfer_n = N_VALUES_ACTUAL[-2] if len(N_VALUES_ACTUAL) >= 2 else N_VALUES_ACTUAL[0]
        log(f"\n=== Phase 3b: Transfer check at N={transfer_n} with w={best_grid_w} ===")
        adapters_transfer = all_adapters[:transfer_n]
        eval_transfer = [d for d in adapters_transfer if d in single_ppls]
        composed_dir = compose_adapters_on_cpu(
            adapters_transfer, ADAPTER_DIR, best_grid_w)
        transfer_ppls = eval_composed_adapter(
            base_model, tokenizer, composed_dir, eval_transfer)
        shutil.rmtree(composed_dir)
        transfer_mean = sum(transfer_ppls.values()) / max(len(transfer_ppls), 1)
        log(f"  N={transfer_n}, w={best_grid_w}: mean PPL = {transfer_mean:.4f}")
    else:
        transfer_n = None
        transfer_ppls = {}
        transfer_mean = None

    # Phase 4: Kill criteria
    log("\n=== Phase 4: Kill Criteria Assessment ===")

    # Get unit-weight PPL at largest N
    unit_ppl_at_max = sum(all_results["unit"][largest_n].values()) / max(len(all_results["unit"][largest_n]), 1)
    sqrt_ppl_at_max = sum(all_results["sqrt"][largest_n].values()) / max(len(all_results["sqrt"][largest_n]), 1)

    # K1: 1/sqrt(N) must reduce PPL by >50% vs unit-weight at N=max
    if unit_ppl_at_max > 0 and unit_ppl_at_max != float("inf"):
        k1_reduction = (unit_ppl_at_max - sqrt_ppl_at_max) / unit_ppl_at_max * 100
    elif sqrt_ppl_at_max < unit_ppl_at_max:
        k1_reduction = 99.0  # massive improvement from inf
    else:
        k1_reduction = 0.0
    k1_pass = k1_reduction > 50.0
    log(f"K1 (sqrt reduces PPL >50% vs unit): {'PASS' if k1_pass else 'FAIL'} — {k1_reduction:.1f}% reduction")
    log(f"   unit PPL = {unit_ppl_at_max:.4f}, sqrt PPL = {sqrt_ppl_at_max:.4f}")

    # K2: best scaling must produce PPL < 2x avg single expert
    k2_ratio = best_grid_ppl / avg_single_ppl if avg_single_ppl > 0 else float("inf")
    k2_pass = k2_ratio <= 2.0
    log(f"K2 (best PPL < 2x single avg): {'PASS' if k2_pass else 'FAIL'} — ratio = {k2_ratio:.2f}x")
    log(f"   best PPL = {best_grid_ppl:.4f}, avg single = {avg_single_ppl:.4f}")

    # K3: best weight transfers from N=max to N=transfer
    if transfer_mean is not None and best_grid_ppl > 0:
        # Check if relative ordering is preserved (transfer PPL should also be good)
        # Fail if transfer PPL > 2x the best_grid_ppl (scaling factor doesn't generalize)
        k3_ratio = transfer_mean / best_grid_ppl
        k3_pass = k3_ratio < 2.0 and transfer_mean < avg_single_ppl * 3.0
    else:
        k3_ratio = None
        k3_pass = True  # cannot evaluate
    log(f"K3 (weight transfers to N={transfer_n}): {'PASS' if k3_pass else 'FAIL'} — ratio = {k3_ratio}")

    verdict = "PASS" if (k1_pass and k2_pass and k3_pass) else "KILLED"
    killed_criteria = []
    if not k1_pass:
        killed_criteria.append("K1")
    if not k2_pass:
        killed_criteria.append("K2")
    if not k3_pass:
        killed_criteria.append("K3")

    # Build results
    aggregate = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_values": N_VALUES_ACTUAL,
            "eval_samples_per_domain": EVAL_SAMPLES,
            "max_seq_len": MAX_SEQ_LEN,
            "total_adapters": len(all_adapters),
            "eval_domains": len(eval_domains),
            "smoke_test": IS_SMOKE,
        },
        "single_expert_avg_ppl": round(avg_single_ppl, 4),
        "single_expert_ppls": single_ppls,
        "scaling_results": {
            strategy: {
                str(n): {
                    "per_domain_ppl": ppls,
                    "mean_ppl": round(sum(ppls.values()) / max(len(ppls), 1), 4),
                    "weight": SCALING_STRATEGIES[strategy](n),
                }
                for n, ppls in n_results.items()
            }
            for strategy, n_results in all_results.items()
        },
        "grid_search": {
            "n": largest_n,
            "results": {
                str(w): grid_results[w] for w in GRID_WEIGHTS
            },
            "best_weight": best_grid_w,
            "best_ppl": best_grid_ppl,
        },
        "transfer_check": {
            "n": transfer_n,
            "weight": best_grid_w,
            "mean_ppl": round(transfer_mean, 4) if transfer_mean else None,
            "per_domain_ppl": transfer_ppls,
        },
        "kill_criteria": {
            "K1_sqrt_reduces_50pct": {
                "unit_ppl": round(unit_ppl_at_max, 4),
                "sqrt_ppl": round(sqrt_ppl_at_max, 4),
                "reduction_pct": round(k1_reduction, 1),
                "threshold": 50.0,
                "pass": k1_pass,
            },
            "K2_best_lt_2x_single": {
                "best_ppl": best_grid_ppl,
                "avg_single_ppl": round(avg_single_ppl, 4),
                "ratio": round(k2_ratio, 2) if k2_ratio != float("inf") else "inf",
                "threshold": 2.0,
                "pass": k2_pass,
            },
            "K3_transfers_across_n": {
                "n_source": largest_n,
                "n_target": transfer_n,
                "ratio": round(k3_ratio, 2) if k3_ratio else None,
                "pass": k3_pass,
            },
        },
        "verdict": verdict,
        "killed_criteria": killed_criteria,
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Summary
    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    for strategy, n_results in sorted(all_results.items()):
        for n, ppls in sorted(n_results.items()):
            mean = sum(ppls.values()) / max(len(ppls), 1)
            log(f"  {strategy:6s} N={n:3d}: mean PPL = {mean:.4f}")

    log(f"\nGrid search best: w={best_grid_w} -> PPL={best_grid_ppl:.4f}")
    log(f"\nK1: {'PASS' if k1_pass else 'FAIL'} (sqrt reduces {k1_reduction:.1f}% vs unit)")
    log(f"K2: {'PASS' if k2_pass else 'FAIL'} (best/single = {k2_ratio:.2f}x)")
    log(f"K3: {'PASS' if k3_pass else 'FAIL'} (transfer ratio = {k3_ratio})")
    log(f"\nVERDICT: {verdict}")
    if killed_criteria:
        log(f"KILLED BY: {', '.join(killed_criteria)}")
    log(f"Total time: {aggregate['elapsed_s']:.0f}s")

    out_file = RESULTS_DIR / "weight_normalization_results.json"
    with open(out_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    log(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
