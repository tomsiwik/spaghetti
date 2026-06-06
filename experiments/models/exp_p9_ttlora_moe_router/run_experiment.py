#!/usr/bin/env python3
"""
P9.B2: TT-LoRA MoE -- Gated Routing Across Domain Experts
Paper: TT-LoRA MoE (arXiv:2504.21190)
Prior: Finding #516 (TT-LoRA 84.4% quality, 12.4x compression)

Kill criteria:
  K1360: Router expert selection accuracy >= 90% on 5 domains
  K1361: Routed TT-LoRA MoE outperforms single best TT-LoRA by >= 5pp avg
  K1362: Total system size (5 experts + router) < 2 MB

Design:
  - 5 MMLU domain groups: math, code, medical, legal, finance
  - TT-LoRA r=6 on v_proj, 500 steps per domain adapter
  - Linear router on base model hidden states
  - Logit-based MCQ evaluation (no generation needed)
"""

import gc
import json
import math
import os
import random
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
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN_PER_DOMAIN = 30 if IS_SMOKE else 1000
N_ROUTER_TRAIN = 10 if IS_SMOKE else 200
N_EVAL_PER_DOMAIN = 5 if IS_SMOKE else 100
N_STEPS = 10 if IS_SMOKE else 500
N_ROUTER_STEPS = 5 if IS_SMOKE else 300
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
TT_RANK = 6
TT_LR = 5e-3
TT_ALPHA = 1.0
ROUTER_LR = 1e-3
SEED = 42

HIDDEN_SIZE = 2560
DOMAINS = ["math", "code", "medical", "legal", "finance"]
N_DOMAINS = len(DOMAINS)
PROJ_NAMES = ["v_proj"]

# MMLU subject -> domain mapping
DOMAIN_SUBJECTS = {
    "math": [
        "abstract_algebra", "college_mathematics", "elementary_mathematics",
        "high_school_mathematics", "high_school_statistics",
    ],
    "code": [
        "college_computer_science", "computer_security", "machine_learning",
        "high_school_computer_science",
    ],
    "medical": [
        "anatomy", "clinical_knowledge", "college_biology",
        "college_medicine", "medical_genetics", "professional_medicine",
    ],
    "legal": [
        "international_law", "jurisprudence", "professional_law",
    ],
    "finance": [
        "econometrics", "professional_accounting",
        "high_school_macroeconomics", "high_school_microeconomics",
    ],
}


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


# ============================================================
# TT-LoRA Module (proven in exp_p9_ttlora_quality)
# ============================================================

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
        self._init_cores(ranks, tt_shape)
        self._cached_delta_w = None

    def _init_cores(self, ranks, tt_shape):
        """Initialize TT cores. Last core zero (zero output at start)."""
        d = len(tt_shape)
        for k in range(d):
            shape = (ranks[k], tt_shape[k], ranks[k + 1])
            if k == d - 1:
                core = mx.zeros(shape)
            else:
                std = 1.0 / math.sqrt(tt_shape[k] * ranks[k])
                core = mx.random.normal(shape) * std
            setattr(self, f"core_{k}", core)

    def reinit_cores(self):
        """Re-initialize cores for training a new domain adapter."""
        ranks = [1]
        for k in range(self._n_cores):
            core = getattr(self, f"core_{k}")
            ranks.append(core.shape[2])
        self._init_cores(ranks, self.tt_shape)
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
        """Contract TT cores -> dW [out_features, in_features]."""
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

    def clear_cache(self):
        self._cached_delta_w = None

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


# ============================================================
# Model Injection
# ============================================================

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    return model.layers


def get_inner_model(model):
    """Get the inner model that returns hidden states (before lm_head)."""
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model
    if hasattr(model, "model") and hasattr(model.model, "model"):
        return model.model.model
    if hasattr(model, "model"):
        return model.model
    return model


def detect_proj_dims(base_layer, hidden_size=HIDDEN_SIZE):
    test_x = mx.zeros((1, 1, hidden_size))
    test_y = base_layer(test_x)
    mx.eval(test_y)
    out_f = test_y.shape[-1]
    del test_x, test_y
    return hidden_size, out_f


def inject_ttlora(model, proj_names, tt_rank, alpha):
    """Replace specified projections with TT-LoRA wrappers in all layers."""
    layers = get_layers(model)
    total_params = 0
    for i, layer in enumerate(layers):
        for name in proj_names:
            base = getattr(layer.self_attn, name)
            in_f, out_f = detect_proj_dims(base)
            tt_shape = compute_tt_shape(in_f, out_f)
            wrapper = TTLoRAWrapper(base, in_f, out_f, tt_shape, tt_rank, alpha)
            setattr(layer.self_attn, name, wrapper)
            total_params += wrapper.num_params()
            if i == 0:
                print(f"  {name}: {in_f}->{out_f} TT shape {tt_shape}", flush=True)
    print(f"TT-LoRA injected: {total_params:,} params, {len(layers)} layers",
          flush=True)
    return total_params


def reinit_all_ttlora(model, proj_names):
    """Re-initialize all TT-LoRA cores for a new domain."""
    for layer in get_layers(model):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "reinit_cores"):
                proj.reinit_cores()


def freeze_unfreeze_cores(model, proj_names):
    """Freeze all params, unfreeze only TT-LoRA cores."""
    model.freeze()
    for layer in get_layers(model):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                core_keys = [f"core_{k}" for k in range(proj._n_cores)]
                proj.unfreeze(keys=core_keys, recurse=False)


def zero_all_ttlora(model, proj_names):
    """Zero all TT-LoRA cores -> model behaves as base model."""
    for layer in get_layers(model):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if hasattr(proj, "_n_cores"):
                for k in range(proj._n_cores):
                    core = getattr(proj, f"core_{k}")
                    setattr(proj, f"core_{k}", mx.zeros(core.shape))
                proj._cached_delta_w = None


def save_ttlora_adapter(model, save_path, proj_names):
    """Save TT-LoRA cores as float16 safetensors."""
    from safetensors.numpy import save_file

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    weights = {}
    total_params = 0
    for i, layer in enumerate(get_layers(model)):
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


def load_ttlora_adapter(model, adapter_path, proj_names):
    """Load TT-LoRA cores from safetensors into model."""
    from safetensors.numpy import load_file

    filepath = Path(adapter_path) / "tt_cores.safetensors"
    weights = load_file(str(filepath))
    layers = get_layers(model)

    for key, arr in weights.items():
        parts = key.split(".")
        layer_idx = int(parts[1])
        pname = parts[3]
        core_name = parts[4]
        proj = getattr(layers[layer_idx].self_attn, pname)
        setattr(proj, core_name, mx.array(arr))
        proj._cached_delta_w = None


def cache_all_delta_w(model, proj_names):
    """Pre-compute and cache delta_w for fast inference."""
    for layer in get_layers(model):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()


def clear_all_delta_w_cache(model, proj_names):
    """Clear cached delta_w."""
    for layer in get_layers(model):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.clear_cache()


# ============================================================
# MMLU Data Preparation
# ============================================================

def format_mcq(question, choices, answer_idx):
    """Format MMLU question as instruction messages."""
    letters = ["A", "B", "C", "D"]
    options = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))
    user_msg = f"{question}\n\n{options}"
    ans_letter = letters[answer_idx]
    ans_text = choices[answer_idx] if answer_idx < len(choices) else ""
    assistant_msg = f"The answer is {ans_letter}) {ans_text}"
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def load_mmlu_by_domain():
    """Load MMLU data grouped by domain. Returns {domain: [examples]}."""
    from datasets import load_dataset

    # Build subject -> domain lookup
    subject_to_domain = {}
    for domain, subjects in DOMAIN_SUBJECTS.items():
        for subj in subjects:
            subject_to_domain[subj] = domain

    # Try loading MMLU
    ds = None
    for ds_name in ["cais/mmlu", "hails/mmlu_no_train", "tasksource/mmlu"]:
        try:
            print(f"Trying MMLU dataset: {ds_name}...", flush=True)
            ds = load_dataset(ds_name, "all")
            print(f"  Loaded {ds_name}", flush=True)
            break
        except Exception as e:
            print(f"  Failed: {e}", flush=True)

    if ds is None:
        raise RuntimeError("Could not load any MMLU dataset")

    # Identify splits and columns
    print(f"  Available splits: {list(ds.keys())}", flush=True)
    # auxiliary_train has empty subjects, so use test for training
    # and validation for eval
    train_split = "test"
    eval_split = "validation"

    print(f"  Using train={train_split}, eval={eval_split}", flush=True)
    sample = ds[train_split][0]
    print(f"  Columns: {list(sample.keys())}", flush=True)

    # Detect column names
    q_col = "question"
    subj_col = "subject"
    choices_col = "choices"
    answer_col = "answer"

    domain_train = {d: [] for d in DOMAINS}
    domain_eval = {d: [] for d in DOMAINS}

    for split_name, target_dict in [(train_split, domain_train),
                                     (eval_split, domain_eval)]:
        for ex in ds[split_name]:
            subj = ex.get(subj_col, "")
            domain = subject_to_domain.get(subj)
            if domain is None:
                continue

            choices = ex.get(choices_col, [])
            answer = ex.get(answer_col, 0)
            if isinstance(answer, str):
                answer = {"A": 0, "B": 1, "C": 2, "D": 3}.get(answer, 0)

            target_dict[domain].append({
                "question": ex[q_col],
                "choices": list(choices),
                "answer": int(answer),
                "subject": subj,
                "domain": domain,
            })

    for d in DOMAINS:
        random.seed(SEED)
        random.shuffle(domain_train[d])
        random.shuffle(domain_eval[d])
        print(f"  {d}: {len(domain_train[d])} train, "
              f"{len(domain_eval[d])} eval", flush=True)

    return domain_train, domain_eval


def prepare_domain_jsonl(tokenizer, domain_data, data_dir, max_examples):
    """Write domain data as tokenized JSONL for TT-LoRA training."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    examples = domain_data[:max_examples]
    records = []
    for ex in examples:
        msgs = format_mcq(ex["question"], ex["choices"], ex["answer"])
        records.append(json.dumps({"messages": msgs}))

    n_val = max(1, len(records) // 10)
    (data_dir / "train.jsonl").write_text("\n".join(records[n_val:]))
    (data_dir / "valid.jsonl").write_text("\n".join(records[:n_val]))
    return len(records) - n_val, n_val


# ============================================================
# TT-LoRA Training
# ============================================================

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


def train_ttlora_loop(model, tokenizer, data_dir, n_steps, lr, batch_size):
    examples = tokenize_for_training(tokenizer, data_dir / "train.jsonl", MAX_SEQ_LEN)
    print(f"    {len(examples)} training examples", flush=True)
    if not examples:
        print("    WARNING: no training examples!", flush=True)
        return [], 0.0

    random.seed(SEED)
    random.shuffle(examples)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    def loss_fn(model, input_ids, lengths, prompt_lens):
        logits = model(input_ids).astype(mx.float32)
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


# ============================================================
# Router
# ============================================================

class DomainRouter(nn.Module):
    """Linear router: hidden_dim -> n_experts."""

    def __init__(self, hidden_dim, n_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, n_experts)

    def __call__(self, hidden_states):
        return self.gate(hidden_states)


def extract_hidden_states(model, tokenizer, examples, domain_indices,
                          max_seq_len=256, batch_size=4):
    """Extract mean-pooled last hidden states from base model.

    Args:
        examples: list of {"question": ..., "choices": ...}
        domain_indices: list of int domain labels
    Returns:
        hidden_states: [N, hidden_dim], labels: [N]
    """
    all_h = []
    all_y = []

    for i in range(0, len(examples), batch_size):
        batch_exs = examples[i:i+batch_size]
        batch_labels = domain_indices[i:i+batch_size]

        # Tokenize each example (question + choices as user prompt)
        token_lists = []
        for ex in batch_exs:
            msgs = [{"role": "user", "content": format_mcq(
                ex["question"], ex["choices"], ex["answer"])[0]["content"]}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(text)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            token_lists.append(ids)

        # Pad to max length in batch
        max_len = max(len(t) for t in token_lists)
        pad_id = tokenizer.pad_token_id or 0
        padded = [t + [pad_id] * (max_len - len(t)) for t in token_lists]
        lengths = [len(t) for t in token_lists]

        input_ids = mx.array(padded)

        # Forward through base model (inner model, before lm_head)
        inner = get_inner_model(model)
        h = inner(input_ids)  # [batch, seq_len, hidden_dim]
        mx.eval(h)

        # Mean pool over actual tokens (not padding)
        for j in range(len(batch_exs)):
            seq_h = h[j, :lengths[j], :]  # [seq_len, hidden_dim]
            pooled = seq_h.mean(axis=0, keepdims=True)  # [1, hidden_dim]
            all_h.append(pooled)
            all_y.append(batch_labels[j])

    hidden = mx.concatenate(all_h, axis=0)  # [N, hidden_dim]
    labels = mx.array(all_y)
    mx.eval(hidden, labels)
    return hidden, labels


def train_router(router, train_h, train_y, n_steps, lr, batch_size=32):
    """Train domain router on hidden states."""
    optimizer = optim.Adam(learning_rate=lr)
    N = train_h.shape[0]

    def loss_fn(router, h_batch, y_batch):
        logits = router(h_batch)
        return nn.losses.cross_entropy(logits, y_batch, reduction="mean")

    loss_and_grad = nn.value_and_grad(router, loss_fn)

    losses = []
    indices = list(range(N))

    for step in range(n_steps):
        random.shuffle(indices)
        batch_idx = mx.array(indices[:min(batch_size, N)])
        h_batch = train_h[batch_idx]
        y_batch = train_y[batch_idx]

        loss, grads = loss_and_grad(router, h_batch, y_batch)
        optimizer.update(router, grads)
        mx.eval(loss, router.parameters(), optimizer.state)
        losses.append(loss.item())

        if (step + 1) % max(1, n_steps // 5) == 0:
            avg = sum(losses[-20:]) / len(losses[-20:])
            print(f"  Router step {step+1}/{n_steps}: loss={avg:.4f}", flush=True)

    return losses


def save_router(router, save_path):
    """Save router weights."""
    from safetensors.numpy import save_file

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    weights = {}
    for k, v in nn.utils.tree_flatten(router.parameters()):
        weights[k] = np.array(v).astype(np.float16)

    filepath = save_path / "router.safetensors"
    save_file(weights, str(filepath))
    size = filepath.stat().st_size
    print(f"  Router saved: {size:,} bytes ({size/1024:.1f} KB)", flush=True)
    return size


# ============================================================
# Evaluation
# ============================================================

def eval_mcq_logit(model, tokenizer, examples, batch_size=4):
    """Evaluate MCQ accuracy via logit comparison (no generation).

    For each example, check if the logit for the correct answer token
    is highest among A/B/C/D at the last prompt position.
    Returns (accuracy%, correct, total).
    """
    # Get token IDs for answer letters
    choice_tokens = []
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(letter)
        choice_tokens.append(ids[-1])  # Last token (handles BOS/special tokens)

    correct = 0
    total = 0

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        token_lists = []
        for ex in batch:
            msgs = [{"role": "user", "content": format_mcq(
                ex["question"], ex["choices"], ex["answer"])[0]["content"]}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(text)
            if len(ids) > MAX_SEQ_LEN:
                ids = ids[:MAX_SEQ_LEN]
            token_lists.append(ids)

        # Pad
        max_len = max(len(t) for t in token_lists)
        pad_id = tokenizer.pad_token_id or 0
        padded = [t + [pad_id] * (max_len - len(t)) for t in token_lists]
        lengths = [len(t) for t in token_lists]

        input_ids = mx.array(padded)
        logits = model(input_ids)  # [batch, seq_len, vocab]
        mx.eval(logits)

        for j in range(len(batch)):
            # Get logits at last real token position
            last_pos = lengths[j] - 1
            last_logits = logits[j, last_pos, :]

            # Compare logits for A/B/C/D
            choice_logit_vals = [last_logits[t].item() for t in choice_tokens]
            predicted = int(np.argmax(choice_logit_vals))

            if predicted == batch[j]["answer"]:
                correct += 1
            total += 1

    acc = correct / total * 100 if total > 0 else 0.0
    return acc, correct, total


def get_per_example_predictions(model, tokenizer, examples, batch_size=4):
    """Get predicted answer index for each example. Returns list of int."""
    choice_tokens = []
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(letter)
        choice_tokens.append(ids[-1])

    predictions = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        token_lists = []
        for ex in batch:
            msgs = [{"role": "user", "content": format_mcq(
                ex["question"], ex["choices"], ex["answer"])[0]["content"]}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(text)
            if len(ids) > MAX_SEQ_LEN:
                ids = ids[:MAX_SEQ_LEN]
            token_lists.append(ids)

        max_len = max(len(t) for t in token_lists)
        pad_id = tokenizer.pad_token_id or 0
        padded = [t + [pad_id] * (max_len - len(t)) for t in token_lists]
        lengths = [len(t) for t in token_lists]

        input_ids = mx.array(padded)
        logits = model(input_ids)
        mx.eval(logits)

        for j in range(len(batch)):
            last_pos = lengths[j] - 1
            last_logits = logits[j, last_pos, :]
            choice_logit_vals = [last_logits[t].item() for t in choice_tokens]
            predictions.append(int(np.argmax(choice_logit_vals)))

    return predictions


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    print("=" * 60)
    print("P9.B2: TT-LoRA MoE -- Gated Routing Across Domain Experts")
    print(f"Domains: {DOMAINS}")
    print(f"TT-LoRA: rank={TT_RANK}, lr={TT_LR}, {N_STEPS} steps/domain")
    print(f"SMOKE={IS_SMOKE}")
    print("=" * 60, flush=True)

    results = {
        "experiment": "exp_p9_ttlora_moe_router",
        "paper": "arXiv:2504.21190",
        "prior_finding": "#516",
        "domains": DOMAINS,
        "tt_rank": TT_RANK,
        "n_steps_per_domain": N_STEPS,
        "n_train_per_domain": N_TRAIN_PER_DOMAIN,
        "n_eval_per_domain": N_EVAL_PER_DOMAIN,
    }

    # ── Phase 1: Prepare MMLU data ────────────────
    print("\n-- Phase 1: Prepare MMLU domain data --", flush=True)
    domain_train, domain_eval = load_mmlu_by_domain()

    # Check we have enough data
    for d in DOMAINS:
        if len(domain_train[d]) < N_TRAIN_PER_DOMAIN:
            print(f"  WARNING: {d} has only {len(domain_train[d])} train examples "
                  f"(need {N_TRAIN_PER_DOMAIN})", flush=True)
        if len(domain_eval[d]) < N_EVAL_PER_DOMAIN:
            print(f"  WARNING: {d} has only {len(domain_eval[d])} eval examples "
                  f"(need {N_EVAL_PER_DOMAIN})", flush=True)

    # ── Phase 2: Load model and train adapters ────
    print("\n-- Phase 2: Load model, train 5 TT-LoRA adapters --", flush=True)
    from mlx_lm import load

    model, tokenizer = load(MODEL_ID)
    log_memory("model-loaded")

    # Inject TT-LoRA
    tt_total_params = inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)

    adapter_sizes = {}
    adapter_losses = {}
    adapter_times = {}

    for di, domain in enumerate(DOMAINS):
        print(f"\n  Training adapter [{di+1}/{N_DOMAINS}]: {domain}", flush=True)

        # Prepare domain data as JSONL
        data_dir = EXPERIMENT_DIR / "data" / domain
        n_tr, n_val = prepare_domain_jsonl(
            tokenizer, domain_train[domain], data_dir, N_TRAIN_PER_DOMAIN)
        print(f"    Data: {n_tr} train, {n_val} val", flush=True)

        # Re-initialize TT-LoRA cores
        reinit_all_ttlora(model, PROJ_NAMES)
        freeze_unfreeze_cores(model, PROJ_NAMES)

        # Verify trainable params
        trainable = sum(
            p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
        print(f"    Trainable: {trainable:,} params", flush=True)

        # Train
        model.train()
        losses, train_time = train_ttlora_loop(
            model, tokenizer, data_dir, N_STEPS, TT_LR, BATCH_SIZE)

        # Save adapter
        adapter_path = EXPERIMENT_DIR / "adapters" / domain
        size = save_ttlora_adapter(model, adapter_path, PROJ_NAMES)

        adapter_sizes[domain] = size
        adapter_losses[domain] = losses[-1] if losses else None
        adapter_times[domain] = round(train_time, 1)

        log_memory(f"after-{domain}")

    results["adapter_sizes"] = adapter_sizes
    results["adapter_final_losses"] = {d: round(v, 4) if v else None
                                        for d, v in adapter_losses.items()}
    results["adapter_train_times_s"] = adapter_times
    results["total_training_time_s"] = round(sum(adapter_times.values()), 1)

    print(f"\nTotal adapter training: {sum(adapter_times.values()):.0f}s "
          f"({sum(adapter_times.values())/60:.1f} min)", flush=True)

    # ── Phase 3: Extract hidden states for router ─
    print("\n-- Phase 3: Extract hidden states for router --", flush=True)

    # Zero TT-LoRA -> base model behavior
    zero_all_ttlora(model, PROJ_NAMES)
    model.eval()

    # Prepare router training data (different subset from adapter training)
    router_train_exs = []
    router_train_labels = []
    for di, domain in enumerate(DOMAINS):
        # Use examples AFTER the adapter training data
        offset = N_TRAIN_PER_DOMAIN
        exs = domain_train[domain][offset:offset + N_ROUTER_TRAIN]
        if len(exs) < N_ROUTER_TRAIN:
            # Fall back to using some training data if not enough
            exs = domain_train[domain][:N_ROUTER_TRAIN]
        router_train_exs.extend(exs)
        router_train_labels.extend([di] * len(exs))

    print(f"  Router training data: {len(router_train_exs)} examples", flush=True)

    train_h, train_y = extract_hidden_states(
        model, tokenizer, router_train_exs, router_train_labels)
    print(f"  Hidden states shape: {train_h.shape}", flush=True)
    log_memory("hidden-states-extracted")

    # Also extract hidden states for eval data (for router eval)
    eval_all_exs = []
    eval_all_labels = []
    for di, domain in enumerate(DOMAINS):
        exs = domain_eval[domain][:N_EVAL_PER_DOMAIN]
        eval_all_exs.extend(exs)
        eval_all_labels.extend([di] * len(exs))

    eval_h, eval_y = extract_hidden_states(
        model, tokenizer, eval_all_exs, eval_all_labels)
    print(f"  Eval hidden states shape: {eval_h.shape}", flush=True)

    # ── Phase 4: Train router ─────────────────────
    print("\n-- Phase 4: Train domain router --", flush=True)

    router = DomainRouter(HIDDEN_SIZE, N_DOMAINS)
    router_losses = train_router(
        router, train_h, train_y, N_ROUTER_STEPS, ROUTER_LR)

    router_size = save_router(router, EXPERIMENT_DIR / "adapters" / "router")
    results["router_size_bytes"] = router_size
    results["router_final_loss"] = round(router_losses[-1], 4) if router_losses else None

    # ── Phase 5: Router accuracy (K1360) ──────────
    print("\n-- Phase 5: Evaluate router accuracy (K1360) --", flush=True)

    # Train accuracy
    train_logits = router(train_h)
    mx.eval(train_logits)
    train_preds = mx.argmax(train_logits, axis=-1)
    train_router_acc = (train_preds == train_y).astype(mx.float32).mean().item() * 100

    # Eval accuracy
    eval_logits = router(eval_h)
    mx.eval(eval_logits)
    eval_preds = mx.argmax(eval_logits, axis=-1)
    eval_router_acc = (eval_preds == eval_y).astype(mx.float32).mean().item() * 100

    print(f"  Router train acc: {train_router_acc:.1f}%", flush=True)
    print(f"  Router eval acc:  {eval_router_acc:.1f}%", flush=True)

    results["router_train_acc"] = round(train_router_acc, 1)
    results["router_eval_acc"] = round(eval_router_acc, 1)
    results["K1360_router_acc"] = "PASS" if eval_router_acc >= 90 else "FAIL"
    results["K1360_detail"] = f"eval={eval_router_acc:.1f}% (need >=90%)"

    # Per-domain router accuracy
    per_domain_router_acc = {}
    eval_preds_np = np.array(eval_preds.tolist())
    eval_y_np = np.array(eval_y.tolist())
    for di, domain in enumerate(DOMAINS):
        mask = eval_y_np == di
        if mask.sum() == 0:
            continue
        acc = (eval_preds_np[mask] == eval_y_np[mask]).mean() * 100
        per_domain_router_acc[domain] = round(acc, 1)
        print(f"    {domain}: {acc:.1f}%", flush=True)
    results["per_domain_router_acc"] = per_domain_router_acc

    # ── Phase 6: Cross-domain MCQ evaluation (K1361) ──
    print("\n-- Phase 6: Cross-domain evaluation (K1361) --", flush=True)

    # For each adapter, get predictions on ALL eval examples
    adapter_predictions = {}  # domain -> list of predicted answer indices
    adapter_accuracies = {}   # domain -> {eval_domain: accuracy}

    for di, domain in enumerate(DOMAINS):
        print(f"  Evaluating adapter: {domain}", flush=True)
        load_ttlora_adapter(model, EXPERIMENT_DIR / "adapters" / domain, PROJ_NAMES)
        cache_all_delta_w(model, PROJ_NAMES)

        preds = get_per_example_predictions(model, tokenizer, eval_all_exs)
        adapter_predictions[domain] = preds

        # Per-domain accuracy
        domain_accs = {}
        offset = 0
        for dj, eval_domain in enumerate(DOMAINS):
            n = min(N_EVAL_PER_DOMAIN, len(domain_eval[eval_domain]))
            domain_preds = preds[offset:offset+n]
            domain_answers = [ex["answer"] for ex in eval_all_exs[offset:offset+n]]
            correct = sum(1 for p, a in zip(domain_preds, domain_answers) if p == a)
            acc = correct / n * 100 if n > 0 else 0
            domain_accs[eval_domain] = round(acc, 1)
            offset += n
        adapter_accuracies[domain] = domain_accs

        clear_all_delta_w_cache(model, PROJ_NAMES)

    results["cross_domain_accuracy"] = adapter_accuracies

    # Print cross-domain matrix
    print("\n  Cross-domain accuracy matrix (adapter x eval_domain):", flush=True)
    header = f"  {'Adapter':<10}" + "".join(f"{d:<10}" for d in DOMAINS) + "  Avg"
    print(header, flush=True)
    print("  " + "-" * len(header), flush=True)

    adapter_avgs = {}
    for adapter_domain in DOMAINS:
        accs = adapter_accuracies[adapter_domain]
        avg = sum(accs.values()) / len(accs)
        adapter_avgs[adapter_domain] = avg
        row = f"  {adapter_domain:<10}" + "".join(
            f"{accs[d]:<10.1f}" for d in DOMAINS) + f"  {avg:.1f}"
        print(row, flush=True)

    # Single best: adapter with highest average accuracy
    single_best_domain = max(adapter_avgs, key=adapter_avgs.get)
    single_best_avg = adapter_avgs[single_best_domain]
    print(f"\n  Single best adapter: {single_best_domain} (avg={single_best_avg:.1f}%)",
          flush=True)

    # MoE: use router's prediction to select adapter for each example
    eval_preds_list = eval_preds.tolist()
    moe_correct = 0
    moe_total = 0
    for i, ex in enumerate(eval_all_exs):
        router_choice = eval_preds_list[i]
        selected_domain = DOMAINS[router_choice]
        pred = adapter_predictions[selected_domain][i]
        if pred == ex["answer"]:
            moe_correct += 1
        moe_total += 1

    moe_avg = moe_correct / moe_total * 100 if moe_total > 0 else 0
    print(f"  MoE accuracy: {moe_correct}/{moe_total} = {moe_avg:.1f}%", flush=True)

    # Per-domain MoE accuracy
    moe_per_domain = {}
    offset = 0
    for di, domain in enumerate(DOMAINS):
        n = min(N_EVAL_PER_DOMAIN, len(domain_eval[domain]))
        domain_correct = 0
        for j in range(offset, offset + n):
            router_choice = eval_preds_list[j]
            selected_domain = DOMAINS[router_choice]
            pred = adapter_predictions[selected_domain][j]
            if pred == eval_all_exs[j]["answer"]:
                domain_correct += 1
        moe_per_domain[domain] = round(domain_correct / n * 100, 1) if n > 0 else 0
        offset += n
    results["moe_per_domain_acc"] = moe_per_domain

    # K1361: MoE avg - single best avg >= 5pp
    moe_advantage = moe_avg - single_best_avg
    results["moe_avg_acc"] = round(moe_avg, 1)
    results["single_best_domain"] = single_best_domain
    results["single_best_avg_acc"] = round(single_best_avg, 1)
    results["moe_advantage_pp"] = round(moe_advantage, 1)
    results["K1361_quality"] = "PASS" if moe_advantage >= 5 else "FAIL"
    results["K1361_detail"] = (
        f"MoE={moe_avg:.1f}% - SingleBest({single_best_domain})={single_best_avg:.1f}% "
        f"= {moe_advantage:+.1f}pp (need >=5pp)")

    # Also compute oracle routing (best adapter per domain)
    oracle_correct = 0
    oracle_total = 0
    offset = 0
    for di, domain in enumerate(DOMAINS):
        n = min(N_EVAL_PER_DOMAIN, len(domain_eval[domain]))
        for j in range(offset, offset + n):
            # Use the in-domain adapter (oracle routing)
            pred = adapter_predictions[domain][j]
            if pred == eval_all_exs[j]["answer"]:
                oracle_correct += 1
            oracle_total += 1
        offset += n
    oracle_avg = oracle_correct / oracle_total * 100 if oracle_total > 0 else 0
    results["oracle_routing_acc"] = round(oracle_avg, 1)
    print(f"  Oracle routing (perfect selection): {oracle_avg:.1f}%", flush=True)

    # ── Phase 7: Size check (K1362) ───────────────
    print("\n-- Phase 7: Size check (K1362) --", flush=True)

    total_adapter_size = sum(adapter_sizes.values())
    total_system_size = total_adapter_size + router_size
    results["total_adapter_size_bytes"] = total_adapter_size
    results["total_system_size_bytes"] = total_system_size
    results["K1362_size"] = "PASS" if total_system_size < 2 * 1024 * 1024 else "FAIL"
    results["K1362_detail"] = (
        f"{total_system_size:,} bytes ({total_system_size/1024:.1f} KB) "
        f"[5 adapters: {total_adapter_size/1024:.1f} KB + "
        f"router: {router_size/1024:.1f} KB]")

    print(f"  5 adapters: {total_adapter_size:,} bytes ({total_adapter_size/1024:.1f} KB)",
          flush=True)
    print(f"  Router: {router_size:,} bytes ({router_size/1024:.1f} KB)", flush=True)
    print(f"  Total: {total_system_size:,} bytes ({total_system_size/1024:.1f} KB)",
          flush=True)

    # ── Summary ───────────────────────────────────
    total_time = time.time() - t_start
    results["total_time_s"] = round(total_time, 1)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"K1360 Router Acc:  {results['K1360_router_acc']} ({results['K1360_detail']})")
    print(f"K1361 MoE Quality: {results['K1361_quality']} ({results['K1361_detail']})")
    print(f"K1362 Size:        {results['K1362_size']} ({results['K1362_detail']})")

    overall = (results["K1360_router_acc"] == "PASS" and
               results["K1361_quality"] == "PASS" and
               results["K1362_size"] == "PASS")
    results["overall_pass"] = overall
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save results
    (EXPERIMENT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {EXPERIMENT_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
