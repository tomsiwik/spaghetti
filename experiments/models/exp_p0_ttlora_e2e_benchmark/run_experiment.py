#!/usr/bin/env python3
"""
P0.E2E: TT-LoRA Drop-In — Compressed Adapters in Proven Pipeline

Drop-in replacement: swap standard LoRA (21.8MB/adapter) with TT-LoRA (~154KB/adapter)
in the proven E2E pipeline from Finding #508.

Kill criteria:
  K1426: TT-LoRA E2E GSM8K >= 60% (retain >=82% of LoRA pipeline's 73%)
  K1427: TT-LoRA E2E HumanEval >= 50% (retain >=79% of LoRA pipeline's 63%)
  K1428: Total adapter footprint for 3 domains < 1 MB (vs 65.4 MB standard)
  K1429: TF-IDF routing accuracy >= 95% on TT-LoRA adapters (same as standard)

Grounded by:
  Finding #508: E2E system works (+19-56pp on benchmarks)
  Finding #515: TT-LoRA ported to MLX (8.3x compression)
  Finding #516: TT-LoRA r=6 retains 84% quality on GSM8K
  arXiv:2504.21190 (TT-LoRA MoE)
"""

import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 50 if IS_SMOKE else 2000
N_EVAL = 5 if IS_SMOKE else 100
N_STEPS = 20 if IS_SMOKE else 500
BATCH_SIZE = 2
MAX_SEQ_LEN = 512

TT_RANK = 6
TT_LR = 5e-3       # Paper-recommended (arXiv:2504.21190)
TT_ALPHA = 1.0

PROJ_NAMES = ["v_proj", "o_proj"]   # Match Finding #508 target projections
HIDDEN_SIZE = 2560  # Gemma 4 E4B
SEED = 42

# Finding #508 baseline (standard LoRA v_proj+o_proj rank-8, 1000 steps)
BASELINE_GSM8K = 73.0
BASELINE_HUMANEVAL = 63.0
BASELINE_MEDMCQA = 50.0
BASELINE_ADAPTER_MB = 21.8
BASELINE_ROUTING = 98.3


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB",
          flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ────────────────────────────────────────────────
# TT-LoRA Module (from exp_p9_ttlora_quality)
# ────────────────────────────────────────────────

class TTLoRAWrapper(nn.Module):
    """TT-LoRA adapter wrapping an existing (possibly quantized) linear layer."""

    def __init__(self, base_layer, in_features, out_features, tt_shape,
                 tt_rank=6, alpha=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.tt_shape = tt_shape

        self._validate_split(tt_shape, in_features, out_features)

        d = len(tt_shape)
        ranks = [1] + [tt_rank] * (d - 1) + [1]
        self._n_cores = d
        for k in range(d):
            shape = (ranks[k], tt_shape[k], ranks[k + 1])
            if k == d - 1:
                core = mx.zeros(shape)
            else:
                std = 1.0 / math.sqrt(tt_shape[k] * ranks[k])
                core = mx.random.normal(shape) * std
            setattr(self, f"core_{k}", core)
        self._cached_delta_w = None

    def _validate_split(self, tt_shape, in_features, out_features):
        prod = 1
        for i, s in enumerate(tt_shape):
            prod *= s
            if prod == in_features:
                rest = 1
                for j in range(i + 1, len(tt_shape)):
                    rest *= tt_shape[j]
                assert rest == out_features, (
                    f"Output factors product {rest} != {out_features}")
                return
        raise ValueError(
            f"Cannot split {tt_shape} into {in_features} x {out_features}")

    @property
    def tt_cores(self):
        return [getattr(self, f"core_{k}") for k in range(self._n_cores)]

    def reconstruct_delta_w(self):
        """Contract TT cores -> DW [out_features, in_features]."""
        cores = self.tt_cores
        result = cores[0].squeeze(0)
        for k in range(1, len(cores)):
            core = cores[k]
            r_k, s_k, r_next = core.shape
            result = result @ core.reshape(r_k, s_k * r_next)
            leading = result.shape[0]
            result = result.reshape(leading * s_k, r_next)
        result = result.squeeze(-1)
        return result.reshape(self.in_features, self.out_features).T

    def cache_delta_w(self):
        self._cached_delta_w = self.reconstruct_delta_w()
        mx.eval(self._cached_delta_w)

    def __call__(self, x):
        base_out = self.base_layer(x)
        dw = (self._cached_delta_w if self._cached_delta_w is not None
              else self.reconstruct_delta_w())
        return base_out + self.alpha * (x @ dw.T)

    def num_params(self):
        return sum(c.size for c in self.tt_cores)


def factorize(n, max_factor=10):
    factors = []
    while n % 8 == 0 and n > 8:
        factors.append(8)
        n //= 8
    for f in range(max_factor, 1, -1):
        while n % f == 0 and n > 1:
            factors.append(f)
            n //= f
    if n > 1:
        factors.append(n)
    factors.sort()
    return factors


def compute_tt_shape(in_features, out_features):
    return factorize(in_features) + factorize(out_features)


# ────────────────────────────────────────────────
# Model Injection & Core Management
# ────────────────────────────────────────────────

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


def detect_proj_dims(base_layer):
    """Extract (in_features, out_features) from a quantized or dense linear layer."""
    if hasattr(base_layer, 'scales'):
        # QuantizedLinear: weight.shape = (out_features, in_features * bits / 32)
        out_features = base_layer.weight.shape[0]
        in_features = base_layer.weight.shape[1] * 32 // base_layer.bits
        return in_features, out_features
    if hasattr(base_layer, 'weight'):
        return base_layer.weight.shape[1], base_layer.weight.shape[0]
    raise ValueError(f"Cannot detect dimensions for {type(base_layer)}")


def inject_ttlora(model, proj_names, tt_rank, alpha):
    layers = get_layers(model)
    total_params = 0
    dim_info = {}
    for i, layer in enumerate(layers):
        for name in proj_names:
            base = getattr(layer.self_attn, name)
            in_f, out_f = detect_proj_dims(base)
            if (name, in_f, out_f) not in dim_info:
                dim_info[(name, in_f, out_f)] = []
            dim_info[(name, in_f, out_f)].append(i)

            tt_shape = compute_tt_shape(in_f, out_f)
            wrapper = TTLoRAWrapper(base, in_f, out_f, tt_shape, tt_rank, alpha)
            setattr(layer.self_attn, name, wrapper)
            total_params += wrapper.num_params()

    for (name, in_f, out_f), layer_ids in dim_info.items():
        tt_shape = compute_tt_shape(in_f, out_f)
        print(f"  {name}: {in_f}->{out_f} (TT shape {tt_shape}, "
              f"{len(layer_ids)} layers)", flush=True)
    return total_params


def reset_ttlora_cores(model, proj_names):
    """Reset all TT-LoRA cores to initial state for next domain."""
    layers = get_layers(model)
    for layer in layers:
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                d = proj._n_cores
                for k in range(d):
                    shape = getattr(proj, f"core_{k}").shape
                    if k == d - 1:
                        core = mx.zeros(shape)
                    else:
                        std = 1.0 / math.sqrt(proj.tt_shape[k] * shape[0])
                        core = mx.random.normal(shape) * std
                    setattr(proj, f"core_{k}", core)
                proj._cached_delta_w = None
    mx.eval(model.parameters())


def freeze_except_ttcores(model, proj_names):
    """Freeze entire model, unfreeze only TT cores."""
    model.freeze()
    for layer in get_layers(model):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                core_keys = [f"core_{k}" for k in range(proj._n_cores)]
                proj.unfreeze(keys=core_keys, recurse=False)


def save_ttlora_cores(model, save_path, proj_names):
    """Save TT-LoRA cores as float16 safetensors."""
    from safetensors.numpy import save_file

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    layers = get_layers(model)
    weights = {}
    total_params = 0
    for i, layer in enumerate(layers):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                for j in range(proj._n_cores):
                    core = getattr(proj, f"core_{j}")
                    key = f"layers.{i}.self_attn.{pname}.core_{j}"
                    arr = np.array(core)
                    weights[key] = arr.astype(np.float16)
                    total_params += arr.size

    filepath = save_path / "tt_cores.safetensors"
    save_file(weights, str(filepath))
    size = filepath.stat().st_size
    print(f"  Saved: {total_params:,} params, {size:,} bytes ({size/1024:.1f} KB)",
          flush=True)
    return size


def load_ttlora_cores(model, adapter_dir, proj_names):
    """Load saved TT-LoRA cores from safetensors."""
    from safetensors.numpy import load_file

    filepath = Path(adapter_dir) / "tt_cores.safetensors"
    weights = load_file(str(filepath))

    layers = get_layers(model)
    for key, arr in weights.items():
        parts = key.split(".")
        layer_idx = int(parts[1])
        pname = parts[3]
        core_name = parts[4]

        proj = getattr(layers[layer_idx].self_attn, pname)
        core = mx.array(arr.astype(np.float32))
        setattr(proj, core_name, core)
        proj._cached_delta_w = None

    # Cache delta_w for fast inference
    for layer in layers:
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()


# ────────────────────────────────────────────────
# Data Preparation
# ────────────────────────────────────────────────

def prepare_data():
    """Check training data exists (pre-copied from exp_p0_e2e_benchmark)."""
    data_dir = EXPERIMENT_DIR / "data"
    for d in ["math", "code", "medical"]:
        train_file = data_dir / d / "train.jsonl"
        if not train_file.exists():
            raise FileNotFoundError(
                f"Training data missing: {train_file}. "
                f"Copy from exp_p0_e2e_benchmark/data/")
        n = sum(1 for _ in open(train_file))
        print(f"  {d}: {n} train examples", flush=True)


def load_gsm8k_test(n_eval, seed=SEED):
    """Load GSM8K test set via huggingface_hub (avoids broken datasets lib)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openai/gsm8k",
        "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_eval, len(df)), random_state=seed)
    return df.to_dict("records")


def load_humaneval_test(n_eval):
    """Load HumanEval test set via huggingface_hub."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    return df.head(min(n_eval, len(df))).to_dict("records")


def load_medmcqa_val(n_eval, seed=SEED):
    """Load MedMCQA validation set via huggingface_hub."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/validation-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_eval, len(df)), random_state=seed)
    return df.to_dict("records")


def load_routing_texts(n_samples, seed=SEED):
    """Load text samples for TF-IDF routing from all 3 domains."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    texts = {}

    # GSM8K train questions
    path = hf_hub_download("openai/gsm8k",
        "main/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_samples, len(df)), random_state=seed + 1)
    texts["math"] = df["question"].tolist()

    # CodeAlpaca instructions
    path = hf_hub_download("sahil2801/CodeAlpaca-20k",
        "code_alpaca_20k.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed + 1)
    rng.shuffle(data)
    texts["code"] = [ex["instruction"] for ex in data[:n_samples]]

    # MedMCQA train questions
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_samples, len(df)), random_state=seed + 1)
    texts["medical"] = df["question"].tolist()

    return texts


# ────────────────────────────────────────────────
# TT-LoRA Training
# ────────────────────────────────────────────────

def tokenize_for_training(tokenizer, data_path, max_seq_len=512):
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]
            full = tokenizer.apply_chat_template(msgs, tokenize=False)
            full_ids = tokenizer.encode(full)
            prompt = tokenizer.apply_chat_template(
                [msgs[0]], tokenize=False, add_generation_prompt=True)
            prompt_len = len(tokenizer.encode(prompt))
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
            if prompt_len >= len(full_ids):
                continue
            examples.append({
                "input_ids": full_ids,
                "prompt_len": prompt_len,
                "length": len(full_ids),
            })
    return examples


def train_ttlora_domain(model, tokenizer, data_dir, n_steps, lr, batch_size):
    """Train TT-LoRA with SFT loss (prompt-masked)."""
    examples = tokenize_for_training(
        tokenizer, data_dir / "train.jsonl", MAX_SEQ_LEN)
    print(f"  Loaded {len(examples)} training examples", flush=True)

    random.seed(SEED)
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(model, input_ids, lengths, prompt_lens):
        logits = model(input_ids)
        logits = logits.astype(mx.float32)
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        ce = nn.losses.cross_entropy(shift_logits, shift_targets, reduction="none")
        S = shift_targets.shape[1]
        pos = mx.arange(S)[None, :]
        mask = (pos >= (prompt_lens[:, None] - 1)) & (pos < (lengths[:, None] - 1))
        mask = mask.astype(mx.float32)
        return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    idx = 0

    for step in range(n_steps):
        batch_exs = []
        for _ in range(batch_size):
            batch_exs.append(examples[idx % len(examples)])
            idx += 1

        max_len = max(e["length"] for e in batch_exs)
        input_ids = mx.array([
            e["input_ids"] + [pad_id] * (max_len - e["length"])
            for e in batch_exs
        ])
        lengths = mx.array([e["length"] for e in batch_exs])
        prompt_lens = mx.array([e["prompt_len"] for e in batch_exs])

        loss, grads = loss_and_grad(model, input_ids, lengths, prompt_lens)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        losses.append(loss.item())

        if (step + 1) % max(1, n_steps // 5) == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            print(f"    Step {step+1}/{n_steps}: loss={avg:.4f} ({elapsed:.1f}s)",
                  flush=True)

    total_time = time.time() - t0
    return losses, total_time


# ────────────────────────────────────────────────
# Benchmark Evaluation
# ────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_eval, label=""):
    from mlx_lm import generate

    ds = load_gsm8k_test(n_eval)

    correct = 0
    total = len(ds)
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=512, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1

        if (i + 1) % 25 == 0:
            print(f"    GSM8K {label}: {i+1}/{total}, acc={correct/(i+1)*100:.1f}%",
                  flush=True)

    acc = correct / total * 100
    print(f"  GSM8K {label}: {correct}/{total} = {acc:.1f}%", flush=True)
    return acc


def eval_humaneval(model, tokenizer, n_eval, label=""):
    from mlx_lm import generate

    ds = load_humaneval_test(n_eval)

    passed = 0
    total = len(ds)
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": (
            f"Complete the following Python function. "
            f"Respond with ONLY the function body code, no explanation.\n\n"
            f"```python\n{ex['prompt']}\n```"
        )}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response

        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10, capture_output=True, text=True)
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

        if (i + 1) % 25 == 0:
            print(f"    HumanEval {label}: {i+1}/{total}, pass@1={passed/(i+1)*100:.1f}%",
                  flush=True)

    acc = passed / total * 100
    print(f"  HumanEval {label}: {passed}/{total} = {acc:.1f}%", flush=True)
    return acc


def eval_medmcqa(model, tokenizer, n_eval, label=""):
    from mlx_lm import generate

    ds = load_medmcqa_val(n_eval)

    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0

    for i, ex in enumerate(ds):
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        messages = [{"role": "user", "content":
            f"Answer this medical question. Reply with only the letter.\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=20, verbose=False)

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

        if (i + 1) % 25 == 0:
            print(f"    MedMCQA {label}: {i+1}/{len(ds)}, acc={correct/(i+1)*100:.1f}%",
                  flush=True)

    acc = correct / len(ds) * 100
    print(f"  MedMCQA {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc


# ────────────────────────────────────────────────
# TF-IDF Routing
# ────────────────────────────────────────────────

def eval_routing():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier

    n_route_train = 200
    n_route_test = 100

    domain_texts = load_routing_texts(n_route_train + n_route_test, seed=SEED)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for domain, texts in domain_texts.items():
        train_texts.extend(texts[:n_route_train])
        train_labels.extend([domain] * min(n_route_train, len(texts)))
        test_texts.extend(texts[n_route_train:n_route_train + n_route_test])
        test_labels.extend([domain] * min(n_route_test, len(texts) - n_route_train))

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, train_labels)

    X_test = vectorizer.transform(test_texts)
    preds = clf.predict(X_test)
    accuracy = sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels) * 100

    per_domain = {}
    for domain in ["math", "code", "medical"]:
        mask = [t == domain for t in test_labels]
        d_preds = [p for p, m in zip(preds, mask) if m]
        d_true = [t for t, m in zip(test_labels, mask) if m]
        per_domain[domain] = sum(p == t for p, t in zip(d_preds, d_true)) / len(d_true) * 100

    print(f"  Routing accuracy: {accuracy:.1f}%", flush=True)
    for d, a in per_domain.items():
        print(f"    {d}: {a:.1f}%", flush=True)

    return {
        "routing_accuracy_pct": round(accuracy, 1),
        "routing_per_domain": {d: round(a, 1) for d, a in per_domain.items()},
    }


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P0.E2E: TT-LoRA Drop-In Benchmark")
    print(f"TT-LoRA: rank={TT_RANK}, lr={TT_LR}, alpha={TT_ALPHA}")
    print(f"Projections: {PROJ_NAMES}")
    print(f"SMOKE={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_STEPS={N_STEPS}")
    print(f"Baseline (Finding #508): GSM8K={BASELINE_GSM8K}%, "
          f"HumanEval={BASELINE_HUMANEVAL}%, MedMCQA={BASELINE_MEDMCQA}%")
    print("=" * 60, flush=True)
    log_memory("start")

    results = {
        "experiment": "exp_p0_ttlora_e2e_benchmark",
        "tt_rank": TT_RANK,
        "tt_lr": TT_LR,
        "proj_names": PROJ_NAMES,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "n_steps": N_STEPS,
        "is_smoke": IS_SMOKE,
    }

    # ── Phase 1: Prepare data ──────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Prepare datasets")
    print("=" * 60, flush=True)
    prepare_data()

    # ── Phase 2: Train TT-LoRA adapters ────────
    print("\n" + "=" * 60)
    print("PHASE 2: Train TT-LoRA adapters (3 domains)")
    print("=" * 60, flush=True)

    from mlx_lm import load

    print("Loading model...", flush=True)
    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Inject TT-LoRA on v_proj + o_proj
    print("Injecting TT-LoRA wrappers...", flush=True)
    tt_total_params = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    print(f"TT-LoRA: {tt_total_params:,} trainable params", flush=True)

    train_results = {}
    adapter_sizes = {}

    for domain in ["math", "code", "medical"]:
        print(f"\n--- Training {domain} adapter ---", flush=True)

        # Reset cores and optimizer for each domain
        reset_ttlora_cores(model, PROJ_NAMES)
        freeze_except_ttcores(model, PROJ_NAMES)

        # Verify trainable params
        trainable = sum(
            p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
        print(f"  Trainable: {trainable:,} params", flush=True)

        data_dir = EXPERIMENT_DIR / "data" / domain
        model.train()
        losses, train_time = train_ttlora_domain(
            model, tokenizer, data_dir, N_STEPS, TT_LR, BATCH_SIZE)

        # Save adapter
        adapter_path = EXPERIMENT_DIR / "adapters" / domain
        size = save_ttlora_cores(model, adapter_path, PROJ_NAMES)
        adapter_sizes[domain] = size

        # Convergence check
        converged = False
        if len(losses) >= 100:
            avg_first = sum(losses[:50]) / 50
            avg_last = sum(losses[-50:]) / 50
            converged = avg_last < avg_first

        train_results[domain] = {
            "train_time_s": round(train_time, 1),
            "final_loss": round(losses[-1], 4) if losses else None,
            "adapter_size_bytes": size,
            "converged": converged,
        }
        log_memory(f"after-{domain}")

    results["train_results"] = train_results
    results["tt_params_per_adapter"] = tt_total_params

    # ── Phase 3: Evaluate benchmarks ───────────
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluate TT-LoRA on benchmarks")
    print("=" * 60, flush=True)

    # Reset model to eval mode
    model.eval()

    # GSM8K with math adapter
    print("\n--- GSM8K (math adapter) ---", flush=True)
    load_ttlora_cores(model, EXPERIMENT_DIR / "adapters" / "math", PROJ_NAMES)
    tt_gsm8k = eval_gsm8k(model, tokenizer, N_EVAL, "TT-LoRA")

    # HumanEval with code adapter
    print("\n--- HumanEval (code adapter) ---", flush=True)
    load_ttlora_cores(model, EXPERIMENT_DIR / "adapters" / "code", PROJ_NAMES)
    tt_humaneval = eval_humaneval(model, tokenizer, N_EVAL, "TT-LoRA")

    # MedMCQA with medical adapter
    print("\n--- MedMCQA (medical adapter) ---", flush=True)
    load_ttlora_cores(model, EXPERIMENT_DIR / "adapters" / "medical", PROJ_NAMES)
    tt_medmcqa = eval_medmcqa(model, tokenizer, N_EVAL, "TT-LoRA")

    results["tt_gsm8k_pct"] = round(tt_gsm8k, 1)
    results["tt_humaneval_pct"] = round(tt_humaneval, 1)
    results["tt_medmcqa_pct"] = round(tt_medmcqa, 1)

    # Checkpoint after expensive phases
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Checkpoint saved to {RESULTS_FILE}", flush=True)

    cleanup(model, tokenizer)

    # ── Phase 4: TF-IDF routing ────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: TF-IDF routing")
    print("=" * 60, flush=True)

    routing_results = eval_routing()
    results["routing"] = routing_results

    # ── Phase 5: Summary ───────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60, flush=True)

    # Adapter sizes
    total_size_bytes = sum(adapter_sizes.values())
    total_size_mb = total_size_bytes / 1e6
    baseline_total_mb = BASELINE_ADAPTER_MB * 3
    compression = baseline_total_mb / total_size_mb if total_size_mb > 0 else float("inf")

    results["total_adapter_bytes"] = total_size_bytes
    results["total_adapter_mb"] = round(total_size_mb, 3)
    results["baseline_total_mb"] = baseline_total_mb
    results["compression_ratio"] = round(compression, 1)

    # Quality retention
    gsm8k_retention = tt_gsm8k / BASELINE_GSM8K if BASELINE_GSM8K > 0 else 0
    humaneval_retention = tt_humaneval / BASELINE_HUMANEVAL if BASELINE_HUMANEVAL > 0 else 0
    medmcqa_retention = tt_medmcqa / BASELINE_MEDMCQA if BASELINE_MEDMCQA > 0 else 0

    results["gsm8k_retention"] = round(gsm8k_retention, 3)
    results["humaneval_retention"] = round(humaneval_retention, 3)
    results["medmcqa_retention"] = round(medmcqa_retention, 3)

    # Kill criteria
    k1_pass = tt_gsm8k >= 60.0
    k2_pass = tt_humaneval >= 50.0
    k3_pass = total_size_mb < 1.0
    k4_pass = routing_results["routing_accuracy_pct"] >= 95.0

    results["K1426_gsm8k"] = "PASS" if k1_pass else "FAIL"
    results["K1427_humaneval"] = "PASS" if k2_pass else "FAIL"
    results["K1428_size"] = "PASS" if k3_pass else "FAIL"
    results["K1429_routing"] = "PASS" if k4_pass else "FAIL"

    print(f"\nTT-LoRA E2E Results vs Baseline (Finding #508):")
    print(f"  GSM8K:     {tt_gsm8k:.1f}% (baseline {BASELINE_GSM8K}%, "
          f"retention {gsm8k_retention:.0%})")
    print(f"  HumanEval: {tt_humaneval:.1f}% (baseline {BASELINE_HUMANEVAL}%, "
          f"retention {humaneval_retention:.0%})")
    print(f"  MedMCQA:   {tt_medmcqa:.1f}% (baseline {BASELINE_MEDMCQA}%, "
          f"retention {medmcqa_retention:.0%})")
    print(f"  Adapters:  {total_size_mb:.3f} MB (baseline {baseline_total_mb:.1f} MB, "
          f"{compression:.0f}x compression)")
    print(f"  Routing:   {routing_results['routing_accuracy_pct']:.1f}%")

    print(f"\nKILL CRITERIA:")
    print(f"  K1426 GSM8K >= 60%:      {results['K1426_gsm8k']} ({tt_gsm8k:.1f}%)")
    print(f"  K1427 HumanEval >= 50%:  {results['K1427_humaneval']} ({tt_humaneval:.1f}%)")
    print(f"  K1428 Size < 1 MB:       {results['K1428_size']} ({total_size_mb:.3f} MB)")
    print(f"  K1429 Routing >= 95%:    {results['K1429_routing']} "
          f"({routing_results['routing_accuracy_pct']:.1f}%)")

    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)
    results["overall_pass"] = all([k1_pass, k2_pass, k3_pass, k4_pass])

    print(f"\nOVERALL: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_FILE}", flush=True)


def routing_only():
    """Resume from checkpoint: load prior results and only run routing + summary."""
    t_start = time.time()
    print("=" * 60)
    print("ROUTING-ONLY MODE: Loading checkpoint results")
    print("=" * 60, flush=True)

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"No checkpoint at {RESULTS_FILE}. Run full experiment first.")

    results = json.loads(RESULTS_FILE.read_text())
    tt_gsm8k = results["tt_gsm8k_pct"]
    tt_humaneval = results["tt_humaneval_pct"]
    tt_medmcqa = results["tt_medmcqa_pct"]
    adapter_sizes = {d: v["adapter_size_bytes"] for d, v in results["train_results"].items()}

    # ── Phase 4: TF-IDF routing ────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: TF-IDF routing")
    print("=" * 60, flush=True)

    routing_results = eval_routing()
    results["routing"] = routing_results

    # ── Phase 5: Summary ───────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60, flush=True)

    total_size_bytes = sum(adapter_sizes.values())
    total_size_mb = total_size_bytes / 1e6
    baseline_total_mb = BASELINE_ADAPTER_MB * 3
    compression = baseline_total_mb / total_size_mb if total_size_mb > 0 else float("inf")

    results["total_adapter_bytes"] = total_size_bytes
    results["total_adapter_mb"] = round(total_size_mb, 3)
    results["baseline_total_mb"] = baseline_total_mb
    results["compression_ratio"] = round(compression, 1)

    gsm8k_retention = tt_gsm8k / BASELINE_GSM8K if BASELINE_GSM8K > 0 else 0
    humaneval_retention = tt_humaneval / BASELINE_HUMANEVAL if BASELINE_HUMANEVAL > 0 else 0
    medmcqa_retention = tt_medmcqa / BASELINE_MEDMCQA if BASELINE_MEDMCQA > 0 else 0

    results["gsm8k_retention"] = round(gsm8k_retention, 3)
    results["humaneval_retention"] = round(humaneval_retention, 3)
    results["medmcqa_retention"] = round(medmcqa_retention, 3)

    k1_pass = tt_gsm8k >= 60.0
    k2_pass = tt_humaneval >= 50.0
    k3_pass = total_size_mb < 1.0
    k4_pass = routing_results["routing_accuracy_pct"] >= 95.0

    results["K1426_gsm8k"] = "PASS" if k1_pass else "FAIL"
    results["K1427_humaneval"] = "PASS" if k2_pass else "FAIL"
    results["K1428_size"] = "PASS" if k3_pass else "FAIL"
    results["K1429_routing"] = "PASS" if k4_pass else "FAIL"

    print(f"\nTT-LoRA E2E Results vs Baseline (Finding #508):")
    print(f"  GSM8K:     {tt_gsm8k:.1f}% (baseline {BASELINE_GSM8K}%, "
          f"retention {gsm8k_retention:.0%})")
    print(f"  HumanEval: {tt_humaneval:.1f}% (baseline {BASELINE_HUMANEVAL}%, "
          f"retention {humaneval_retention:.0%})")
    print(f"  MedMCQA:   {tt_medmcqa:.1f}% (baseline {BASELINE_MEDMCQA}%, "
          f"retention {medmcqa_retention:.0%})")
    print(f"  Adapters:  {total_size_mb:.3f} MB (baseline {baseline_total_mb:.1f} MB, "
          f"{compression:.0f}x compression)")
    print(f"  Routing:   {routing_results['routing_accuracy_pct']:.1f}%")

    print(f"\nKILL CRITERIA:")
    print(f"  K1426 GSM8K >= 60%:      {results['K1426_gsm8k']} ({tt_gsm8k:.1f}%)")
    print(f"  K1427 HumanEval >= 50%:  {results['K1427_humaneval']} ({tt_humaneval:.1f}%)")
    print(f"  K1428 Size < 1 MB:       {results['K1428_size']} ({total_size_mb:.3f} MB)")
    print(f"  K1429 Routing >= 95%:    {results['K1429_routing']} "
          f"({routing_results['routing_accuracy_pct']:.1f}%)")

    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)
    results["overall_pass"] = all([k1_pass, k2_pass, k3_pass, k4_pass])

    print(f"\nOVERALL: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"Routing-only time: {total_time:.0f}s", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    if "--routing-only" in sys.argv:
        routing_only()
    else:
        main()
