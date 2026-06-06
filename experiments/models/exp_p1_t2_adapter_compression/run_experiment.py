#!/usr/bin/env python3
"""
T2.2: Adapter quantization — quality retention under 4-bit and 2-bit compression.

Uses T2.1 trained adapters (math, code, medical) from exp_p1_t2_single_domain_training.
Quantizes lora_b to k-bit (lora_a stays fp16 — last dim r=6 not divisible by MLX group sizes).
Evaluates quality of quantized adapters vs fp16 baseline.

Kill criteria:
  K1033: 4-bit adapter quality >= 95% of fp16 adapter quality (avg over 3 domains)
  K1034: 2-bit adapter quality >= 85% of fp16 adapter quality (avg over 3 domains)
  K1035: 4-bit adapter logical size < 5MB per domain
  K1036: |cos| between domain adapters < 0.05 after 4-bit quantization
"""

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import safetensors

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
T21_DIR = Path(__file__).parent.parent / "exp_p1_t2_single_domain_training"
T21_ADAPTERS = T21_DIR / "adapters"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 3 if IS_SMOKE else 25
SEED = 42


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


# ─────────────────────────────────────────────
# Adapter quantization
# ─────────────────────────────────────────────

def compute_logical_quantized_size(tensors: dict, bits: int) -> int:
    """Compute logical compressed size (data + scales + biases) in bytes."""
    total = 0
    for key, arr_np in tensors.items():
        if "lora_b" in key:
            r, d_out = arr_np.shape
            gs = 64  # group_size used for lora_b
            n_groups = r * (d_out // gs)
            data_bytes = r * d_out * bits // 8  # packed bits
            meta_bytes = n_groups * 4 * 2       # scales + biases in fp32
            total += data_bytes + meta_bytes
        else:
            # lora_a kept as fp16 (cannot use MLX group quantization on r=6)
            total += arr_np.size * 2
    return total


def make_quantized_adapter(src_path: Path, dst_path: Path, bits: int) -> dict:
    """
    Create a dequantized-for-inference adapter at dst_path.
    lora_b: quantized to k-bit then dequantized back (Q-DQ).
    lora_a: cast to fp16 (cannot be quantized by MLX, last dim=6).

    Returns size metrics.
    """
    # Load fp32 tensors from T2.1 adapter
    raw = {}
    with safetensors.safe_open(str(src_path / "adapters.safetensors"), framework="numpy") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    logical_bytes = compute_logical_quantized_size(raw, bits)

    # Build Q-DQ tensors
    result = {}
    for key, arr_np in raw.items():
        arr = mx.array(arr_np)

        if "lora_b" in key:
            # (r, d_out): quantize with group_size=64
            gs = 64
            qw, scales, biases = mx.quantize(arr, group_size=gs, bits=bits)
            arr_dq = mx.dequantize(qw, scales, biases, group_size=gs, bits=bits)
            mx.eval(arr_dq)
            result[key] = np.array(arr_dq, dtype=np.float32)
            del qw, scales, biases, arr_dq
        else:
            # lora_a (d_in, r): fp16 round-trip for consistency
            arr_fp16 = arr.astype(mx.float16)
            arr_fp32 = arr_fp16.astype(mx.float32)
            mx.eval(arr_fp32)
            result[key] = np.array(arr_fp32, dtype=np.float32)
            del arr_fp16, arr_fp32

        del arr
        mx.clear_cache()

    # Save as safetensors (numpy, fp32 — compatible with mlx_lm)
    dst_path.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(result, str(dst_path / "adapters.safetensors"))
    shutil.copy(src_path / "adapter_config.json", dst_path / "adapter_config.json")

    actual_bytes = (dst_path / "adapters.safetensors").stat().st_size
    print(f"  logical quantized size: {logical_bytes / 1e6:.2f} MB", flush=True)
    print(f"  actual saved (dequantized fp32): {actual_bytes / 1e6:.2f} MB", flush=True)

    return {
        "logical_mb": logical_bytes / 1e6,
        "actual_mb": actual_bytes / 1e6,
    }


def make_fp16_adapter(src_path: Path, dst_path: Path) -> dict:
    """Create fp16 adapter (cast all tensors to fp16 then back to fp32 for inference)."""
    raw = {}
    with safetensors.safe_open(str(src_path / "adapters.safetensors"), framework="numpy") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    result = {}
    for key, arr_np in raw.items():
        result[key] = arr_np.astype(np.float16).astype(np.float32)

    dst_path.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(result, str(dst_path / "adapters.safetensors"))
    shutil.copy(src_path / "adapter_config.json", dst_path / "adapter_config.json")

    logical_bytes = sum(arr.size * 2 for arr in raw.values())
    actual_bytes = (dst_path / "adapters.safetensors").stat().st_size

    return {
        "logical_mb": logical_bytes / 1e6,
        "actual_mb": actual_bytes / 1e6,
    }


# ─────────────────────────────────────────────
# Evaluation functions (subprocess model load for memory isolation)
# ─────────────────────────────────────────────

def eval_gsm8k(adapter_path: Path = None, n_eval: int = 25) -> float:
    """Evaluate GSM8K accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"gsm8k-loaded")

    correct = 0
    for ex in ds:
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}\n\nAnswer:"
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            pred_ans = pred_match.group(1).replace(",", "").strip()
            if pred_ans == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

    acc = correct / len(ds) * 100
    print(f"  GSM8K: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_humaneval(adapter_path: Path = None, n_eval: int = 25) -> float:
    """Evaluate HumanEval pass@1. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"humaneval-loaded")

    passed = 0
    for ex in ds:
        prompt = ex["prompt"]
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"Complete the following Python function:\n\n```python\n{prompt}\n```\n\nRespond with only the function body, no markdown."}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = prompt + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10, capture_output=True, text=True,
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

    acc = passed / len(ds) * 100
    print(f"  HumanEval: {passed}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_medmcqa(adapter_path: Path = None, n_eval: int = 25) -> float:
    """Evaluate MedMCQA accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"medmcqa-loaded")

    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0

    for ex in ds:
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        prompt = f"Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{question}"

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = option_map.get(ex["cop"], "A")
        pred = response.strip().upper()
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

    acc = correct / len(ds) * 100
    print(f"  MedMCQA: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


# ─────────────────────────────────────────────
# Orthogonality: trace trick (avoids materializing d_in × d_out ΔW)
# ─────────────────────────────────────────────

def compute_inter_domain_cosine(adapter_paths: dict, label: str) -> float:
    """
    Compute max |cos(ΔW_i, ΔW_j)| across all domain pairs using r×r trace trick.

    ⟨ΔW_i, ΔW_j⟩_F = trace((A_i^T A_j)(B_j B_i^T))  — all r×r matrices, O(Lr³)
    ‖ΔW_i‖_F^2      = trace((A_i^T A_i)(B_i B_i^T))
    """
    n_layers = 42
    domains = list(adapter_paths.keys())

    # Load all tensors
    domain_arrays = {}
    for domain, path in adapter_paths.items():
        adapter_file = path / "adapters.safetensors"
        arr_dict = {}
        with safetensors.safe_open(str(adapter_file), framework="numpy") as f:
            for k in f.keys():
                arr_dict[k] = mx.array(f.get_tensor(k))
        domain_arrays[domain] = arr_dict

    max_cos = 0.0
    pair_cosines = []

    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            di, dj = domains[i], domains[j]
            t_i = domain_arrays[di]
            t_j = domain_arrays[dj]

            inner_sum = 0.0
            norm_i_sq_sum = 0.0
            norm_j_sq_sum = 0.0

            for layer in range(n_layers):
                key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
                key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"
                if key_a not in t_i:
                    continue

                A_i = t_i[key_a]  # (d_in, r)
                B_i = t_i[key_b]  # (r, d_out)
                A_j = t_j[key_a]
                B_j = t_j[key_b]

                # ⟨ΔW_i, ΔW_j⟩ = trace((A_i^T A_j)(B_j B_i^T)), all r×r
                AtA_ij = A_i.T @ A_j    # (r, r)
                BBt_ij = B_j @ B_i.T    # (r, r) — B_j:(r,d_out), B_i.T:(d_out,r)
                inner_l = mx.trace(AtA_ij @ BBt_ij).item()

                # ‖ΔW_i‖^2 = trace((A_i^T A_i)(B_i B_i^T))
                AtA_ii = A_i.T @ A_i    # (r, r)
                BBt_ii = B_i @ B_i.T    # (r, r)
                norm_i_l = mx.trace(AtA_ii @ BBt_ii).item()

                AtA_jj = A_j.T @ A_j
                BBt_jj = B_j @ B_j.T
                norm_j_l = mx.trace(AtA_jj @ BBt_jj).item()

                inner_sum += inner_l
                norm_i_sq_sum += norm_i_l
                norm_j_sq_sum += norm_j_l

            cos = abs(inner_sum) / (
                (norm_i_sq_sum ** 0.5) * (norm_j_sq_sum ** 0.5) + 1e-12
            )
            pair_cosines.append(cos)
            print(f"  [{label}] cos({di}, {dj}) = {cos:.6f}", flush=True)
            if cos > max_cos:
                max_cos = cos

    # Free memory
    for d in domain_arrays.values():
        for arr in d.values():
            del arr
    mx.clear_cache()

    return max_cos


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"T2.2: Adapter Compression", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_EVAL={N_EVAL}", flush=True)
    log_memory("start")

    # Verify T2.1 adapters exist
    for domain in ["math", "code", "medical"]:
        p = T21_ADAPTERS / domain / "adapters.safetensors"
        if not p.exists():
            print(f"ERROR: T2.1 adapter not found: {p}", flush=True)
            sys.exit(1)
    print("T2.1 adapters found.", flush=True)

    # Temp dir for quantized adapters
    tmp_dir = EXPERIMENT_DIR / "tmp_adapters"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # T2.1 fp32 reference quality (from results.json)
    t21_results_file = T21_DIR / "results.json"
    if t21_results_file.exists():
        t21 = json.loads(t21_results_file.read_text())
        fp32_math = t21.get("math_gsm8k_pct", None)
        fp32_code = t21.get("code_humaneval_pct", None)
        fp32_med = t21.get("med_medmcqa_pct", None)
        print(f"T2.1 fp32 quality: math={fp32_math}%, code={fp32_code}%, med={fp32_med}%", flush=True)
    else:
        fp32_math = fp32_code = fp32_med = None

    results = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "fp32_math_pct": fp32_math,
        "fp32_code_pct": fp32_code,
        "fp32_med_pct": fp32_med,
    }

    # ── Phase 1: Create quantized adapters ────────────────────
    print("\n=== Phase 1: Create quantized adapter variants ===", flush=True)

    domains = ["math"] if IS_SMOKE else ["math", "code", "medical"]
    bits_list = [16, 4] if IS_SMOKE else [16, 4, 2]

    size_info = {}
    for domain in domains:
        src = T21_ADAPTERS / domain
        size_info[domain] = {}

        for bits in bits_list:
            dst = tmp_dir / f"{domain}_{bits}bit"
            print(f"  Creating {domain} {bits}-bit adapter...", flush=True)

            if bits == 16:
                info = make_fp16_adapter(src, dst)
            else:
                info = make_quantized_adapter(src, dst, bits=bits)

            size_info[domain][f"{bits}bit"] = info
            print(f"    logical={info['logical_mb']:.2f}MB, actual={info['actual_mb']:.2f}MB", flush=True)

    # Store sizes
    for domain in domains:
        for bits in bits_list:
            tag = f"{bits}bit"
            results[f"{domain}_{tag}_logical_mb"] = size_info[domain][tag]["logical_mb"]

    # ── Phase 2: Evaluate quality per quantization level ──────
    print("\n=== Phase 2: Quality evaluation ===", flush=True)

    def eval_domain(domain, adapter_path):
        """Evaluate the right benchmark for each domain."""
        if domain == "math":
            return eval_gsm8k(adapter_path, n_eval=N_EVAL)
        elif domain == "code":
            return eval_humaneval(adapter_path, n_eval=N_EVAL)
        elif domain == "medical":
            return eval_medmcqa(adapter_path, n_eval=N_EVAL)

    for bits in bits_list:
        print(f"\n--- {bits}-bit evaluation ---", flush=True)
        for domain in domains:
            adapter_path = tmp_dir / f"{domain}_{bits}bit"
            print(f"  Evaluating {domain} ({bits}-bit)...", flush=True)
            acc = eval_domain(domain, adapter_path)
            results[f"{domain}_{bits}bit_pct"] = round(acc, 1)
            log_memory(f"after-{domain}-{bits}bit")

    # ── Phase 3: Kill criteria K1033 and K1034 ─────────────────
    print("\n=== Phase 3: Kill criteria ===", flush=True)

    def safe_get(d, key, default=0.0):
        return d.get(key, default) or default

    # fp16 baselines
    fp16_accs = [safe_get(results, f"{d}_16bit_pct") for d in domains]
    fp16_avg = float(np.mean(fp16_accs)) if fp16_accs else 0.0

    # 4-bit quality ratio
    q4_accs = [safe_get(results, f"{d}_4bit_pct") for d in domains]
    q4_avg = float(np.mean(q4_accs)) if q4_accs else 0.0
    k1033_ratio = q4_avg / fp16_avg if fp16_avg > 0 else 0.0
    k1033_pass = k1033_ratio >= 0.95
    results["k1033_ratio"] = round(k1033_ratio, 4)
    results["K1033_4bit_quality"] = "PASS" if k1033_pass else "FAIL"
    print(f"K1033: 4-bit quality ratio = {k1033_ratio:.3f} (need ≥0.95): {results['K1033_4bit_quality']}", flush=True)
    print(f"  fp16 avg={fp16_avg:.1f}%, 4-bit avg={q4_avg:.1f}%", flush=True)

    if 2 in bits_list:
        q2_accs = [safe_get(results, f"{d}_2bit_pct") for d in domains]
        q2_avg = float(np.mean(q2_accs)) if q2_accs else 0.0
        k1034_ratio = q2_avg / fp16_avg if fp16_avg > 0 else 0.0
        k1034_pass = k1034_ratio >= 0.85
        results["k1034_ratio"] = round(k1034_ratio, 4)
        results["K1034_2bit_quality"] = "PASS" if k1034_pass else "FAIL"
        print(f"K1034: 2-bit quality ratio = {k1034_ratio:.3f} (need ≥0.85): {results['K1034_2bit_quality']}", flush=True)
        print(f"  2-bit avg={q2_avg:.1f}%", flush=True)
    else:
        results["K1034_2bit_quality"] = "SKIP (smoke)"
        results["k1034_ratio"] = None

    # K1035: 4-bit logical size < 5MB
    max_4bit_mb = max(size_info[d]["4bit"]["logical_mb"] for d in domains)
    k1035_pass = max_4bit_mb < 5.0
    results["max_4bit_logical_mb"] = round(max_4bit_mb, 2)
    results["K1035_4bit_size"] = "PASS" if k1035_pass else "FAIL"
    print(f"K1035: 4-bit size = {max_4bit_mb:.2f}MB (need <5MB): {results['K1035_4bit_size']}", flush=True)

    # ── Phase 4: Orthogonality check K1036 (4-bit, all domains) ──
    if not IS_SMOKE and len(domains) >= 2:
        print("\n=== Phase 4: Orthogonality check (4-bit) ===", flush=True)
        adapter_paths_4bit = {d: tmp_dir / f"{d}_4bit" for d in domains}
        max_cos_4bit = compute_inter_domain_cosine(adapter_paths_4bit, "4bit")
        results["max_cos_4bit"] = round(max_cos_4bit, 6)
        k1036_pass = max_cos_4bit < 0.05
        results["K1036_orthogonality"] = "PASS" if k1036_pass else "FAIL"
        print(f"K1036: max |cos| = {max_cos_4bit:.6f} (need <0.05): {results['K1036_orthogonality']}", flush=True)

        # Also compute fp16 orthogonality for reference
        print("\nFP16 orthogonality (reference):", flush=True)
        adapter_paths_fp16 = {d: tmp_dir / f"{d}_16bit" for d in domains}
        max_cos_fp16 = compute_inter_domain_cosine(adapter_paths_fp16, "fp16")
        results["max_cos_fp16"] = round(max_cos_fp16, 6)
        print(f"FP16 max |cos| = {max_cos_fp16:.6f}", flush=True)
    else:
        results["max_cos_4bit"] = None
        results["K1036_orthogonality"] = "SKIP (smoke or 1 domain)"

    # ── Summary ────────────────────────────────────────────────
    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"K1033 4-bit quality ≥95% of fp16: {results['K1033_4bit_quality']} (ratio={results['k1033_ratio']})", flush=True)
    print(f"K1034 2-bit quality ≥85% of fp16: {results.get('K1034_2bit_quality', 'N/A')} (ratio={results.get('k1034_ratio')})", flush=True)
    print(f"K1035 4-bit size <5MB:             {results['K1035_4bit_size']} ({results['max_4bit_logical_mb']:.2f}MB)", flush=True)
    print(f"K1036 |cos| <0.05 after 4-bit:     {results.get('K1036_orthogonality', 'N/A')} (max={results.get('max_cos_4bit')})", flush=True)
    print(f"Total time: {results['total_time_s']:.0f}s", flush=True)

    # Cleanup tmp_adapters (large dequantized files)
    if not IS_SMOKE:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Cleaned up tmp_adapters.", flush=True)


if __name__ == "__main__":
    main()
