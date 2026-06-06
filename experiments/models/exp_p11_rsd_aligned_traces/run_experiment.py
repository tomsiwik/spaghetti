#!/usr/bin/env python3
"""
P11.L0: RSD Aligned Traces — Reverse Speculative Decoding Filter on s1K

Key idea: s1K traces from DeepSeek-R1-Distill (teacher) may be misaligned with
Gemma 4B 4-bit (student). RSD filter keeps only traces where the student agrees
with >=60% of the teacher's tokens (per-token acceptance rate).

Phases:
  0. Load s1K parquet (from sibling exp dir)
  1. Compute per-trace NLL under Gemma 4B 4-bit (forward pass, no grad)
  2. Apply RSD filter: keep traces with acceptance_rate >= 0.60
  3. SERT baseline: generate self-correct traces on GSM8K (student-generated)
  4. Train two adapters:
     A: RSD-filtered s1K traces
     B: SERT self-generated correct traces
  5. Eval on MMLU-Pro + GSM8K, compare against P11.F0 raw baseline

Kill criteria:
  K1541: RSD-filtered adapter MMLU-Pro >= P11.F0 + 3pp
  K1542: NLL scoring for 1000 traces < 24h (expected ~30-60 min)
  K1543: >= 60% of s1K traces pass RSD filter

References:
  arXiv:2509.22230 (Reverse Speculative Decoding)
  arXiv:2309.06657 (Statistical Rejection Sampling)
  P11.F0 (raw s1K baseline — Finding #538 shows -26pp with P11.A0)
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
import mlx.nn as nn
import numpy as np
import pandas as pd

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# RSD filter config
ACCEPT_NLL_THRESHOLD = 6.9   # exp(-6.9) ≈ 0.001 = 250× chance level for vocab_size=256k
ACCEPT_RATE_MIN = 0.60       # K1543: minimum per-trace acceptance rate

# Training config
LORA_RANK = 8
LORA_SCALE = 1.0
LORA_DROPOUT = 0.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
MAX_SEQ_LEN = 8192
MAX_TOTAL_CHARS = 32000

N_STEPS_RSD = 10 if IS_SMOKE else 1000    # Train on RSD-filtered traces
N_STEPS_SERT = 10 if IS_SMOKE else 500    # Train on SERT (fewer correct traces available)
BATCH_SIZE = 1
LR = 1e-5

# Eval config
EVAL_PER_CAT = 2 if IS_SMOKE else 7       # MMLU-Pro questions per category
GSM8K_N = 5 if IS_SMOKE else 30           # GSM8K eval size
SERT_N_PROBLEMS = 5 if IS_SMOKE else 100  # GSM8K problems for SERT generation
SERT_N_ATTEMPTS = 2 if IS_SMOKE else 5    # Attempts per problem to find correct trace

# Adapter output paths
ADAPTER_RSD_DIR = REPO_ROOT / "data" / "adapters" / "math-rsd-aligned-v0"
ADAPTER_SERT_DIR = REPO_ROOT / "data" / "adapters" / "math-sert-gsm8k-v0"
REGISTRY_PATH = REPO_ROOT / "data" / "adapters" / "registry.json"

# CLoQ init path (optional — from sibling experiment)
CLOQ_ADAPTER = (REPO_ROOT / "experiments/models/exp_p11_cloq_calibrated_init"
                / "adapters/cloq_init/adapters.safetensors")


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def strip_thinking(response):
    """Extract thinking content from Gemma 4 response."""
    # Primary: Gemma 4 channel tags
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
        cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL).strip()
        return cleaned, thinking_len
    # Fallback: <think>...</think>
    m = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(1))
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return cleaned, thinking_len
    return response, 0


def parse_mcq_answer(response):
    answer_text, thinking_len = strip_thinking(response)
    for pattern in [
        r'\b([A-J])\b(?:\s*$|\s*\.|\s*\))',
        r'(?:^|\s)([A-J])(?:\s*$|\s*\.)',
        r'answer is ([A-J])',
        r'answer: ([A-J])',
    ]:
        m = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), thinking_len
    m = re.search(r'\b([A-J])\b', answer_text)
    if m:
        return m.group(1).upper(), thinking_len
    return None, thinking_len


def parse_gsm8k_answer(response):
    """Extract final numeric answer from GSM8K response."""
    text, _ = strip_thinking(response)
    # Final answer pattern: #### N or "The answer is N"
    m = re.search(r'####\s*([-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '')
    m = re.search(r'(?:answer is|=)\s*([-\d,\.]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(',', '')
    nums = re.findall(r'[-\d]+(?:\.\d+)?', text)
    return nums[-1] if nums else None


# ─────────────────────────────────────────────────────────────────────
# Phase 0: Load s1K traces
# ─────────────────────────────────────────────────────────────────────

def phase_load_s1k():
    """Load s1K from sibling P11.F0 or download fresh."""
    s1k_parquet = REPO_ROOT / "experiments/models/exp_p11_reasoning_sft_s1k/data/s1k.parquet"
    if s1k_parquet.exists():
        log(f"Loading s1K from {s1k_parquet}")
        df = pd.read_parquet(s1k_parquet)
        log(f"Loaded {len(df)} examples")
        return df

    # Fallback: download
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = DATA_DIR / "s1k.parquet"
    if not local.exists():
        import requests
        log("Downloading s1K from HuggingFace...")
        url = ("https://huggingface.co/datasets/simplescaling/s1K/resolve/"
               "refs%2Fconvert%2Fparquet/default/train/0000.parquet")
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    df = pd.read_parquet(local)
    log(f"Loaded {len(df)} examples")
    return df


def build_trace_text(row):
    """Build raw trace text from s1K row for NLL computation."""
    question = str(row.get("question", "")).strip()
    attempt = str(row.get("attempt", "")).strip()

    thinking_traj = row.get("thinking_trajectories", [])
    if isinstance(thinking_traj, (list, np.ndarray)) and len(thinking_traj) > 0:
        thinking = str(thinking_traj[0]).strip()
    elif isinstance(thinking_traj, str):
        thinking = thinking_traj.strip()
    else:
        thinking = ""

    if not question or not attempt or not thinking:
        return None

    # Use <think> format (matches P11.F0 training format)
    total_chars = len(question) + len(thinking) + len(attempt)
    if total_chars > MAX_TOTAL_CHARS:
        max_think = MAX_TOTAL_CHARS - len(question) - len(attempt) - 50
        if max_think < 500:
            return None
        thinking = thinking[:max_think]

    # Full trace as it would appear in chat format (assistant turn)
    return {
        "question": question,
        "thinking": thinking,
        "attempt": attempt,
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Compute per-trace NLL (RSD filter)
# ─────────────────────────────────────────────────────────────────────

def compute_trace_nll(model, tokenizer, trace_dict, max_tokens=2048):
    """
    Compute per-token NLL and acceptance rate for a trace.

    RSD acceptance criterion: P_S(x_t | x_{<t}) >= exp(-ACCEPT_NLL_THRESHOLD)
    Trace accepted if: (# accepted tokens / T) >= ACCEPT_RATE_MIN
    """
    assistant_text = f"<think>{trace_dict['thinking']}</think>\n\n{trace_dict['attempt']}"

    # Format as a chat sequence
    chat = [
        {"role": "user", "content": trace_dict["question"]},
        {"role": "assistant", "content": assistant_text},
    ]
    try:
        full_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        full_text = f"User: {trace_dict['question']}\nAssistant: {assistant_text}"

    tokens = tokenizer.encode(full_text)
    if len(tokens) < 4:
        return None

    # Truncate to max_tokens (NLL on first part is sufficient signal)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    # Forward pass: inputs = tokens[:-1], targets = tokens[1:]
    inputs = mx.array(tokens[:-1])[None]    # [1, T-1]
    targets = mx.array(tokens[1:])          # [T-1]

    logits = model(inputs)              # [1, T-1, vocab]

    # log_softmax = logits - logsumexp (no mx.log_softmax in MLX)
    log_probs = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)  # [T-1, vocab]
    token_log_probs = log_probs[mx.arange(len(targets)), targets]   # [T-1]
    mx.eval(token_log_probs)

    nll_vals = (-token_log_probs).tolist()
    T = len(nll_vals)

    mean_nll = sum(nll_vals) / T
    # Acceptance rate: fraction of tokens where student prob >= threshold
    accepted = sum(1 for v in nll_vals if v <= ACCEPT_NLL_THRESHOLD)
    acceptance_rate = accepted / T

    # Memory cleanup after each trace
    cleanup(logits, log_probs, token_log_probs)

    return {
        "mean_nll": mean_nll,
        "acceptance_rate": acceptance_rate,
        "n_tokens": T,
        "accepted": acceptance_rate >= ACCEPT_RATE_MIN,
    }


def phase_rsd_filter(df, model, tokenizer):
    """Compute NLL for all traces and apply RSD filter."""
    nll_cache_path = DATA_DIR / "trace_nll_scores.json"

    if nll_cache_path.exists():
        log(f"Loading cached NLL scores from {nll_cache_path}")
        with open(nll_cache_path) as f:
            scores = json.load(f)
        log(f"Loaded {len(scores)} cached scores")
        return scores

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Computing NLL for {len(df)} s1K traces...")
    log(f"ACCEPT_NLL_THRESHOLD={ACCEPT_NLL_THRESHOLD:.1f}, ACCEPT_RATE_MIN={ACCEPT_RATE_MIN:.2f}")

    scores = {}
    t0 = time.time()
    n_valid = 0
    n_accepted = 0

    max_traces = 20 if IS_SMOKE else len(df)

    for idx, row in df.iterrows():
        if n_valid >= max_traces:
            break

        trace = build_trace_text(row)
        if trace is None:
            scores[str(idx)] = None
            continue

        result = compute_trace_nll(model, tokenizer, trace)
        scores[str(idx)] = result
        n_valid += 1

        if result is not None and result["accepted"]:
            n_accepted += 1

        if n_valid % 50 == 0 or n_valid <= 5:
            elapsed = time.time() - t0
            rate = n_valid / elapsed
            eta_min = (max_traces - n_valid) / rate / 60 if rate > 0 else 0
            log(f"  [{n_valid}/{max_traces}] accepted={n_accepted} "
                f"elapsed={elapsed/60:.1f}min ETA={eta_min:.1f}min")
            log_memory("nll")

    elapsed_total = time.time() - t0
    log(f"NLL scoring complete: {n_valid} traces in {elapsed_total/60:.1f}min")
    log(f"Accepted: {n_accepted}/{n_valid} = {n_accepted/max(n_valid,1)*100:.1f}%")

    with open(nll_cache_path, "w") as f:
        json.dump(scores, f)
    log(f"Cached NLL scores to {nll_cache_path}")

    return scores


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Build training data for RSD-filtered traces
# ─────────────────────────────────────────────────────────────────────

def prepare_rsd_training_data(df, nll_scores):
    """Build JSONL training data from RSD-accepted traces."""
    rsd_dir = DATA_DIR / "rsd"
    rsd_dir.mkdir(parents=True, exist_ok=True)
    rsd_train_path = rsd_dir / "train.jsonl"
    rsd_valid_path = rsd_dir / "valid.jsonl"

    if rsd_train_path.exists() and rsd_valid_path.exists():
        n_train = sum(1 for _ in open(rsd_train_path))
        n_valid = sum(1 for _ in open(rsd_valid_path))
        log(f"RSD data already prepared: {n_train} train, {n_valid} valid")
        return n_train, n_valid

    accepted_examples = []

    for idx, row in df.iterrows():
        score = nll_scores.get(str(idx))
        if score is None or not score.get("accepted", False):
            continue

        trace = build_trace_text(row)
        if trace is None:
            continue

        assistant_text = f"<think>{trace['thinking']}</think>\n\n{trace['attempt']}"
        accepted_examples.append({
            "messages": [
                {"role": "user", "content": trace["question"]},
                {"role": "assistant", "content": assistant_text},
            ]
        })

    log(f"RSD accepted: {len(accepted_examples)} examples")

    rng = np.random.RandomState(SEED)
    idx_perm = rng.permutation(len(accepted_examples))
    n_valid_split = max(1, len(accepted_examples) // 10)
    train_idx = idx_perm[n_valid_split:]
    valid_idx = idx_perm[:n_valid_split]

    with open(rsd_train_path, "w") as f:
        for i in train_idx:
            f.write(json.dumps(accepted_examples[i]) + "\n")
    with open(rsd_valid_path, "w") as f:
        for i in valid_idx:
            f.write(json.dumps(accepted_examples[i]) + "\n")

    log(f"Wrote {len(train_idx)} train, {len(valid_idx)} valid to data/rsd/")
    return len(train_idx), len(valid_idx)


# ─────────────────────────────────────────────────────────────────────
# Phase 3: SERT — Self-Generated Correct Traces from GSM8K
# ─────────────────────────────────────────────────────────────────────

def load_gsm8k():
    """Load GSM8K test set (via parquet — avoids dill bug in Python 3.14)."""
    local_path = DATA_DIR / "gsm8k_test.parquet"
    if local_path.exists():
        df = pd.read_parquet(local_path)
        log(f"Loaded {len(df)} GSM8K examples from cache")
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import requests
    # Correct URL (refs/convert/parquet branch, not main branch)
    url = ("https://huggingface.co/datasets/openai/gsm8k/resolve/"
           "refs%2Fconvert%2Fparquet/main/test/0000.parquet")
    log(f"Downloading GSM8K from {url}...")
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    df = pd.read_parquet(local_path)
    log(f"Loaded {len(df)} GSM8K examples")
    return df


def generate_sert_traces(model, tokenizer, gsm8k_df):
    """Generate self-correct traces: student generates, only keep correct ones."""
    sert_dir = DATA_DIR / "sert"
    sert_dir.mkdir(parents=True, exist_ok=True)
    sert_train_path = sert_dir / "train.jsonl"
    sert_valid_path = sert_dir / "valid.jsonl"

    if sert_train_path.exists() and sert_valid_path.exists():
        n_train = sum(1 for _ in open(sert_train_path))
        n_valid = sum(1 for _ in open(sert_valid_path))
        log(f"SERT data already prepared: {n_train} train, {n_valid} valid")
        return n_train, n_valid

    from mlx_lm import generate, load

    rng = np.random.RandomState(SEED + 100)
    problem_idxs = rng.choice(len(gsm8k_df), size=min(SERT_N_PROBLEMS, len(gsm8k_df)),
                               replace=False)

    correct_traces = []
    n_correct = 0
    n_tried = 0

    for pidx in problem_idxs:
        row = gsm8k_df.iloc[pidx]
        question = str(row.get("question", "")).strip()

        # Ground truth: last number after "#### "
        answer_text = str(row.get("answer", "")).strip()
        gt_match = re.search(r'####\s*([-\d,\.]+)', answer_text)
        if not gt_match:
            continue
        gt_answer = gt_match.group(1).replace(',', '')

        # Try SERT_N_ATTEMPTS times to get a correct trace
        for attempt_num in range(SERT_N_ATTEMPTS):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=2048,
                temperature=0.7,
                verbose=False,
            )
            n_tried += 1

            pred = parse_gsm8k_answer(response)
            if pred is not None and pred.strip() == gt_answer.strip():
                n_correct += 1
                # Keep this trace
                correct_traces.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ]
                })
                break  # Got one correct trace for this problem

        if n_tried % 10 == 0:
            log(f"  SERT: {n_correct} correct / {n_tried} tried "
                f"({n_correct/max(n_tried,1)*100:.1f}%)")

    log(f"SERT: {n_correct} correct traces from {len(problem_idxs)} problems "
        f"(yield={n_correct/max(len(problem_idxs),1)*100:.1f}%)")

    if len(correct_traces) < 2:
        log("SERT: too few correct traces (<2) — skipping SERT training")
        return 0, 0

    rng2 = np.random.RandomState(SEED + 200)
    idx_perm = rng2.permutation(len(correct_traces))
    n_val = max(1, len(correct_traces) // 10)
    train_idx = idx_perm[n_val:]
    valid_idx = idx_perm[:n_val]

    with open(sert_train_path, "w") as f:
        for i in train_idx:
            f.write(json.dumps(correct_traces[i]) + "\n")
    with open(sert_valid_path, "w") as f:
        for i in valid_idx:
            f.write(json.dumps(correct_traces[i]) + "\n")

    log(f"SERT: wrote {len(train_idx)} train, {len(valid_idx)} valid to data/sert/")
    return len(train_idx), len(valid_idx)


# ─────────────────────────────────────────────────────────────────────
# Phase 4: Train adapters (RSD + SERT)
# ─────────────────────────────────────────────────────────────────────

def write_lora_config(config_path, lora_rank, lora_scale, lora_dropout):
    """Write LoRA config YAML."""
    config = {
        "lora_layers": 16,
        "lora_parameters": {
            "rank": lora_rank,
            "scale": lora_scale,
            "dropout": lora_dropout,
        },
    }
    import yaml  # type: ignore
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def train_adapter(data_dir, adapter_dir, n_steps, label):
    """Train a LoRA adapter using mlx_lm.lora.

    data_dir must contain train.jsonl and valid.jsonl (standard mlx_lm.lora naming).
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapters_file = adapter_dir / "adapters.safetensors"
    config_path = EXPERIMENT_DIR / f"lora_config_{label}.yaml"

    write_lora_config(config_path, LORA_RANK, LORA_SCALE, LORA_DROPOUT)

    cmd = [
        "uv", "run", "python", "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", str(data_dir),
        "--iters", str(n_steps),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LR),
        "--max-seq-length", str(MAX_SEQ_LEN),
        "--adapter-path", str(adapter_dir),
        "-c", str(config_path),
        "--steps-per-report", "10",
        "--steps-per-eval", "100",
        "--save-every", str(max(n_steps // 5, 1)),
    ]

    # Use CLoQ init if available (better initialization)
    if CLOQ_ADAPTER.exists():
        cmd += ["--resume-adapter-file", str(CLOQ_ADAPTER)]
        log(f"Using CLoQ init from {CLOQ_ADAPTER}")
    else:
        log(f"CLoQ init not found — using standard LoRA init")

    log(f"Training {label} adapter: {n_steps} steps")
    log(f"  Data dir: {data_dir} (train.jsonl / valid.jsonl)")
    log(f"  Output: {adapter_dir}")
    log(f"  Command: {' '.join(cmd[:10])}...")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False, text=True)
    elapsed = time.time() - t0

    success = result.returncode == 0 and adapters_file.exists()
    log(f"Training {label}: {'SUCCESS' if success else 'FAILED'} in {elapsed/60:.1f}min")
    return success, elapsed


# ─────────────────────────────────────────────────────────────────────
# Phase 5: Eval on MMLU-Pro + GSM8K
# ─────────────────────────────────────────────────────────────────────

def load_mmlu_pro():
    """Load MMLU-Pro test split (14 categories, direct parquet)."""
    local_path = DATA_DIR / "mmlu_pro_test.parquet"
    if local_path.exists():
        df = pd.read_parquet(local_path)
        log(f"Loaded {len(df)} MMLU-Pro examples from cache")
        return df

    # Try from sibling experiment first
    sibling_data = (REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data"
                    / "mmlu_pro_test.parquet")
    if sibling_data.exists():
        import shutil
        shutil.copy(sibling_data, local_path)
        df = pd.read_parquet(local_path)
        log(f"Copied MMLU-Pro from sibling: {len(df)} examples")
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import requests
    url = ("https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/"
           "data/test-00000-of-00001.parquet")
    log(f"Downloading MMLU-Pro from {url}...")
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    df = pd.read_parquet(local_path)
    log(f"Loaded {len(df)} MMLU-Pro examples")
    return df


def eval_mmlu_pro(model, tokenizer, mmlu_df, label):
    """Evaluate model on MMLU-Pro (stratified by category)."""
    categories = mmlu_df["category"].unique().tolist()

    rng = np.random.RandomState(SEED + 2000)
    samples = []
    for cat in sorted(categories):
        cat_df = mmlu_df[mmlu_df["category"] == cat]
        n = min(EVAL_PER_CAT, len(cat_df))
        chosen = cat_df.sample(n=n, random_state=rng.randint(0, 99999))
        for _, row in chosen.iterrows():
            samples.append(row)

    log(f"MMLU-Pro [{label}]: evaluating {len(samples)} questions "
        f"({EVAL_PER_CAT}/cat × {len(categories)} cats)")

    from mlx_lm import generate

    cat_results = {}
    n_correct = 0
    total_thinking_chars = 0

    for i, row in enumerate(samples):
        question = str(row["question"])
        options = row.get("options", [])
        correct = str(row.get("answer", "")).upper()
        category = str(row.get("category", "unknown"))

        opts_text = "\n".join(f"{chr(65+j)}. {opt}"
                               for j, opt in enumerate(options))
        prompt_text = f"{question}\n\n{opts_text}\n\nAnswer with a single letter."

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=4096,
            temperature=0.0,
            verbose=False,
        )

        pred, thinking_len = parse_mcq_answer(response)
        is_correct = pred is not None and pred.upper() == correct.upper()

        if category not in cat_results:
            cat_results[category] = {"correct": 0, "total": 0}
        cat_results[category]["correct"] += int(is_correct)
        cat_results[category]["total"] += 1
        n_correct += int(is_correct)
        total_thinking_chars += thinking_len

        if i % 20 == 0:
            log(f"  [{i+1}/{len(samples)}] acc={n_correct/(i+1)*100:.1f}% "
                f"thinking={total_thinking_chars/max(i+1,1):.0f}chars/q")

    overall_acc = n_correct / len(samples) if samples else 0.0
    avg_thinking = total_thinking_chars / max(len(samples), 1)
    log(f"MMLU-Pro [{label}]: {overall_acc*100:.1f}% "
        f"(avg_thinking={avg_thinking:.0f} chars/q)")

    return {
        "overall_accuracy": overall_acc,
        "avg_thinking_chars": avg_thinking,
        "n_correct": n_correct,
        "n_total": len(samples),
        "per_category": cat_results,
    }


def eval_gsm8k(model, tokenizer, gsm8k_df, label):
    """Evaluate model on GSM8K subset."""
    rng = np.random.RandomState(SEED + 3000)
    n = min(GSM8K_N, len(gsm8k_df))
    subset = gsm8k_df.sample(n=n, random_state=rng.randint(0, 99999))

    log(f"GSM8K [{label}]: evaluating {n} questions")

    from mlx_lm import generate

    n_correct = 0
    for i, (_, row) in enumerate(subset.iterrows()):
        question = str(row["question"])
        answer_text = str(row.get("answer", ""))
        gt_match = re.search(r'####\s*([-\d,\.]+)', answer_text)
        if not gt_match:
            continue
        gt_answer = gt_match.group(1).replace(',', '')

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
            verbose=False,
        )
        pred = parse_gsm8k_answer(response)
        is_correct = pred is not None and pred.strip() == gt_answer.strip()
        n_correct += int(is_correct)

    acc = n_correct / max(n, 1)
    log(f"GSM8K [{label}]: {acc*100:.1f}% ({n_correct}/{n})")
    return {"accuracy": acc, "n_correct": n_correct, "n_total": n}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    from mlx_lm import load

    t_start = time.time()
    results = {}

    # ── Phase 0: Load s1K ──────────────────────────────────────────
    log("=" * 60)
    log("Phase 0: Load s1K dataset")
    log("=" * 60)
    s1k_df = phase_load_s1k()

    # ── Phase 1: RSD filter (NLL scoring) ─────────────────────────
    log("=" * 60)
    log("Phase 1: RSD filter — compute NLL under Gemma 4B 4-bit")
    log("=" * 60)
    log("Loading model for NLL scoring...")
    model, tokenizer = load(MODEL_ID)
    log_memory("after load")

    t1_start = time.time()
    nll_scores = phase_rsd_filter(s1k_df, model, tokenizer)
    t1_elapsed = time.time() - t1_start

    # Report filter stats
    n_scored = sum(1 for v in nll_scores.values() if v is not None)
    n_accepted = sum(1 for v in nll_scores.values()
                     if v is not None and v.get("accepted", False))
    acceptance_fraction = n_accepted / max(n_scored, 1)
    log(f"Filter stats: {n_accepted}/{n_scored} traces accepted "
        f"({acceptance_fraction*100:.1f}%), time={t1_elapsed/60:.1f}min")

    # K1542 check
    k1542_pass = t1_elapsed < 24 * 3600
    log(f"K1542 (NLL < 24h): {'PASS' if k1542_pass else 'FAIL'} "
        f"({t1_elapsed/60:.1f}min)")

    # K1543 check
    k1543_pass = acceptance_fraction >= 0.60
    log(f"K1543 (>=60% accepted): {'PASS' if k1543_pass else 'FAIL'} "
        f"({acceptance_fraction*100:.1f}%)")

    results["phase1_nll"] = {
        "n_scored": n_scored,
        "n_accepted": n_accepted,
        "acceptance_fraction": acceptance_fraction,
        "elapsed_seconds": t1_elapsed,
        "k1542_pass": k1542_pass,
        "k1543_pass": k1543_pass,
    }

    # ── Phase 2: Prepare RSD training data ────────────────────────
    log("=" * 60)
    log("Phase 2: Build RSD training data")
    log("=" * 60)
    n_rsd_train, n_rsd_valid = prepare_rsd_training_data(s1k_df, nll_scores)
    log(f"RSD data: {n_rsd_train} train, {n_rsd_valid} valid")

    # ── Phase 3: SERT self-generated traces ───────────────────────
    log("=" * 60)
    log("Phase 3: SERT — generate self-correct traces on GSM8K")
    log("=" * 60)
    gsm8k_df = load_gsm8k()
    n_sert_train, n_sert_valid = generate_sert_traces(model, tokenizer, gsm8k_df)
    log(f"SERT data: {n_sert_train} train, {n_sert_valid} valid")

    results["phase3_sert"] = {
        "n_train": n_sert_train,
        "n_valid": n_sert_valid,
    }

    # Unload model before training to free memory
    log("Unloading model before training...")
    cleanup(model, tokenizer)
    log_memory("after cleanup")

    # ── Phase 4: Train adapters ────────────────────────────────────
    log("=" * 60)
    log("Phase 4: Train RSD and SERT adapters")
    log("=" * 60)

    # Train RSD adapter (if enough data)
    rsd_success = False
    rsd_elapsed = 0.0
    if n_rsd_train >= 10:
        rsd_success, rsd_elapsed = train_adapter(
            DATA_DIR / "rsd",
            ADAPTER_RSD_DIR,
            N_STEPS_RSD,
            label="rsd",
        )
    else:
        log(f"RSD: too few training examples ({n_rsd_train} < 10) — skipping")

    # Train SERT adapter (if enough data)
    sert_success = False
    sert_elapsed = 0.0
    if n_sert_train >= 5:
        sert_success, sert_elapsed = train_adapter(
            DATA_DIR / "sert",
            ADAPTER_SERT_DIR,
            N_STEPS_SERT,
            label="sert",
        )
    else:
        log(f"SERT: too few correct traces ({n_sert_train} < 5) — skipping")

    results["phase4_training"] = {
        "rsd": {"success": rsd_success, "elapsed_seconds": rsd_elapsed},
        "sert": {"success": sert_success, "elapsed_seconds": sert_elapsed},
    }

    # ── Phase 5: Eval ─────────────────────────────────────────────
    log("=" * 60)
    log("Phase 5: Eval on MMLU-Pro + GSM8K")
    log("=" * 60)
    mmlu_df = load_mmlu_pro()

    eval_results = {}

    # Eval RSD adapter
    if rsd_success:
        log("Loading model + RSD adapter...")
        model_rsd, tok_rsd = load(MODEL_ID, adapter_path=str(ADAPTER_RSD_DIR))
        log_memory("rsd eval loaded")

        eval_results["rsd"] = {
            "mmlu_pro": eval_mmlu_pro(model_rsd, tok_rsd, mmlu_df, "RSD-adapter"),
            "gsm8k": eval_gsm8k(model_rsd, tok_rsd, gsm8k_df, "RSD-adapter"),
        }
        cleanup(model_rsd, tok_rsd)
    else:
        log("Skipping RSD eval (training failed or insufficient data)")

    # Eval SERT adapter
    if sert_success:
        log("Loading model + SERT adapter...")
        model_sert, tok_sert = load(MODEL_ID, adapter_path=str(ADAPTER_SERT_DIR))
        log_memory("sert eval loaded")

        eval_results["sert"] = {
            "mmlu_pro": eval_mmlu_pro(model_sert, tok_sert, mmlu_df, "SERT-adapter"),
            "gsm8k": eval_gsm8k(model_sert, tok_sert, gsm8k_df, "SERT-adapter"),
        }
        cleanup(model_sert, tok_sert)
    else:
        log("Skipping SERT eval (training failed or insufficient data)")

    results["phase5_eval"] = eval_results

    # ── K1541: Compare RSD vs P11.F0 baseline ─────────────────────
    # P11.F0 baseline: expected ~59-63% (Finding #538 context)
    # Use 60.0% as conservative baseline if P11.F0 result not yet available
    p11f0_results_path = (REPO_ROOT / "experiments/models/exp_p11_s1k_reasoning_train_eval"
                          / "results.json")
    p11f0_mmlu = 60.0  # conservative default
    if p11f0_results_path.exists():
        try:
            with open(p11f0_results_path) as f:
                p11f0 = json.load(f)
            # Try to extract MMLU-Pro accuracy from P11.F0 results
            mmlu_data = p11f0.get("mmlu_adapter", p11f0.get("mmlu_pro_adapter", {}))
            if "overall_accuracy" in mmlu_data:
                p11f0_mmlu = mmlu_data["overall_accuracy"] * 100
                log(f"P11.F0 baseline from results.json: {p11f0_mmlu:.1f}%")
        except Exception as e:
            log(f"Could not read P11.F0 results: {e} — using default {p11f0_mmlu}%")

    rsd_mmlu = None
    if "rsd" in eval_results:
        rsd_mmlu = eval_results["rsd"]["mmlu_pro"]["overall_accuracy"] * 100

    k1541_pass = False
    k1541_delta = None
    if rsd_mmlu is not None:
        k1541_delta = rsd_mmlu - p11f0_mmlu
        k1541_pass = k1541_delta >= 3.0
        log(f"K1541 (RSD >= raw+3pp): {'PASS' if k1541_pass else 'FAIL'} "
            f"RSD={rsd_mmlu:.1f}% vs raw={p11f0_mmlu:.1f}% (delta={k1541_delta:+.1f}pp)")

    results["kill_criteria"] = {
        "k1541": {"pass": k1541_pass, "rsd_mmlu": rsd_mmlu,
                  "p11f0_mmlu": p11f0_mmlu, "delta": k1541_delta},
        "k1542": {"pass": k1542_pass, "nll_time_seconds": t1_elapsed},
        "k1543": {"pass": k1543_pass, "acceptance_fraction": acceptance_fraction},
    }

    # ── Summary ────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    log(f"\n{'='*60}")
    log(f"SUMMARY — P11.L0 RSD Aligned Traces")
    log(f"Total time: {total_elapsed/60:.1f}min")
    log(f"K1541 (RSD >= raw+3pp): {'PASS' if k1541_pass else 'FAIL'}")
    log(f"K1542 (NLL scoring < 24h): {'PASS' if k1542_pass else 'FAIL'}")
    log(f"K1543 (>=60% traces accepted): {'PASS' if k1543_pass else 'FAIL'}")
    if rsd_mmlu is not None:
        log(f"RSD adapter MMLU-Pro: {rsd_mmlu:.1f}%")
    if "sert" in eval_results:
        sert_mmlu = eval_results["sert"]["mmlu_pro"]["overall_accuracy"] * 100
        log(f"SERT adapter MMLU-Pro: {sert_mmlu:.1f}%")
    log(f"{'='*60}")

    results["total_elapsed_seconds"] = total_elapsed
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
