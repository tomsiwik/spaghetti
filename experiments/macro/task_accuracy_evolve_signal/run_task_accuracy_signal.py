#!/usr/bin/env python3
"""Task accuracy as Evolve quality signal: can 10 held-out questions reliably rank adapters?

Motivation: Answer-only PPL was killed at macro (r=-0.63 cross-domain). The Evolve phase
clone-and-compete needs a cheap, reliable quality signal. This tests whether a tiny
held-out benchmark (10 questions per domain) produces stable adapter rankings.

Kill criteria:
  K1: mean Kendall tau between 10-question subsets and 100-question gold < 0.7
      NUANCED if tau_10 < 0.7 but tau_25 >= 0.7; KILL if tau_50 < 0.7
  K2: per-domain evaluation cost exceeds 60s/adapter
  K3: accuracy ranking disagrees with BOTH PPL ranking AND gold-standard ranking

Uses HuggingFace transformers + PEFT with LoRA hot-swapping (no vLLM dependency).
Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment (must be set before importing torch / triggering CUDA init)
# ---------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def ensure_deps():
    """Install missing deps and apply torch patches for RunPod compatibility.

    The RunPod environment may be corrupted by a failed vLLM install that broke
    transformers, huggingface_hub, peft, and datasets (version mismatches).
    Strategy: test the full import chain. If ANY link is broken, install the
    entire chain to /tmp/sole_deps (bypasses disk-full issues in site-packages)
    and prepend to sys.path so our clean versions shadow the broken ones.
    """
    import shutil

    TMP_DEPS = "/workspace/sole_deps"  # /tmp is on root (full), /workspace has space
    # The full dependency chain we need, in install order
    REQUIRED_CHAIN = ["huggingface_hub", "safetensors", "tokenizers",
                      "transformers", "datasets", "peft", "scipy"]

    # --- Step 0: Free disk space (pip cache, unused large pkgs) ---
    subprocess.call([sys.executable, "-m", "pip", "cache", "purge"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for pkg in ["vllm", "flash-attn", "triton", "xformers", "deepspeed",
                "apex", "megatron-core", "tensorrt", "onnxruntime-gpu",
                "torchvision", "torchaudio"]:
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", pkg],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- Step 1: Test full import chain ---
    chain_ok = True
    test_imports = [
        "from huggingface_hub import HfApi",
        "from huggingface_hub.errors import HfHubHTTPError",
        "from transformers import AutoModelForCausalLM, PreTrainedModel",
        "from datasets import load_dataset",
        "from peft import PeftModel",
        "import scipy.stats",
    ]
    for stmt in test_imports:
        try:
            exec(stmt)
        except Exception as e:
            print(f"[ensure_deps] BROKEN: {stmt} → {e}")
            chain_ok = False
            break

    if not chain_ok:
        print(f"[ensure_deps] Import chain broken — installing clean versions to {TMP_DEPS}")
        shutil.rmtree(TMP_DEPS, ignore_errors=True)
        os.makedirs(TMP_DEPS, exist_ok=True)

        # Clear all cached modules from the broken chain (incl torchvision which
        # has circular import issues when the system install is corrupted)
        poison = {"huggingface_hub", "transformers", "datasets", "peft",
                  "safetensors", "tokenizers", "scipy", "torchvision", "torchaudio"}
        for mod_name in list(sys.modules.keys()):
            if any(mod_name == p or mod_name.startswith(p + ".") for p in poison):
                del sys.modules[mod_name]

        # Install clean versions to /workspace (has plenty of space)
        # Set TMPDIR so pip doesn't use the full root / for build temps
        pip_env = {**os.environ, "TMPDIR": "/workspace/pip_tmp"}
        os.makedirs("/workspace/pip_tmp", exist_ok=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--no-cache-dir", "--target", TMP_DEPS] + REQUIRED_CHAIN,
            env=pip_env,
        )
        # Prepend so our clean versions shadow the broken system ones
        sys.path.insert(0, TMP_DEPS)

        # Block torchvision/torchaudio from being auto-discovered by torch
        # (the system installs are corrupted with circular imports)
        for block_mod in ["torchvision", "torchaudio"]:
            if block_mod not in sys.modules:
                sys.modules[block_mod] = None  # type: ignore[assignment]

        # Verify the fix worked
        for stmt in test_imports:
            exec(stmt)  # Let it raise if still broken — nothing more we can do
        print("[ensure_deps] Clean install verified OK")
    else:
        # Chain is fine, just check scipy/datasets are present
        for mod, pkg in [("scipy", "scipy"), ("datasets", "datasets")]:
            try:
                __import__(mod)
            except ImportError:
                print(f"[ensure_deps] Installing missing: {pkg}")
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "--quiet",
                         "--no-cache-dir", "--break-system-packages", pkg]
                    )
                except subprocess.CalledProcessError:
                    os.makedirs(TMP_DEPS, exist_ok=True)
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "--quiet",
                         "--no-cache-dir", "--target", TMP_DEPS, pkg]
                    )
                    if TMP_DEPS not in sys.path:
                        sys.path.insert(0, TMP_DEPS)

    # --- Step 2: Monkey-patch set_submodule for older torch ---
    import torch
    if not hasattr(torch.nn.Module, "set_submodule"):
        print(f"[ensure_deps] torch {torch.__version__} missing set_submodule — patching")
        def _set_submodule(self, target, module):
            atoms = target.split(".")
            mod = self
            for item in atoms[:-1]:
                mod = getattr(mod, item)
            setattr(mod, atoms[-1], module)
        torch.nn.Module.set_submodule = _set_submodule


ensure_deps()

import torch
import numpy as np
from scipy.stats import kendalltau

# ---------------------------------------------------------------------------
# Smoke-test configuration
# ---------------------------------------------------------------------------
IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/task_accuracy_evolve_signal")
SEED = 42

# Full configuration
ALL_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "college_computer_science",
    "college_physics",
    "econometrics",
    "high_school_biology",
    "high_school_us_history",
    "machine_learning",
    "professional_medicine",
    "world_religions",
]

if IS_SMOKE:
    SUBJECTS = ALL_SUBJECTS[:2]
    MAX_ADAPTERS = 3
    N_DRAWS = 2
    GOLD_SIZE = 20
    SUBSET_SIZES = [5, 10]
    MAX_RUNTIME = 120  # 2 min cap for smoke
else:
    SUBJECTS = ALL_SUBJECTS
    MAX_ADAPTERS = 20
    N_DRAWS = 5
    GOLD_SIZE = 100
    SUBSET_SIZES = [10, 25, 50]
    MAX_RUNTIME = 30 * 60  # 30 min hard cap

# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------
def _timeout_handler(signum, frame):
    print(f"\n[TIMEOUT] MAX_RUNTIME={MAX_RUNTIME}s exceeded, saving partial results and exiting.")
    raise SystemExit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(MAX_RUNTIME)

# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------
def get_adapter_list() -> list[str]:
    """Return available adapter names, best-first from benchmark JSON, falling
    back to alphabetical filesystem scan."""
    benchmark_path = Path("/workspace/llm/results/pilot50_benchmark.json")
    if benchmark_path.exists():
        try:
            with open(benchmark_path) as f:
                data = json.load(f)
            ranked = []
            for name, info in data.get("per_adapter", {}).items():
                score = info.get("ppl_improvement_pct", 0)
                ranked.append((name, score))
            ranked.sort(key=lambda x: x[1], reverse=True)
            if ranked:  # Only use benchmark if it has entries
                return [name for name, _ in ranked]
        except Exception as e:
            print(f"[adapter_discovery] benchmark JSON error: {e}, falling back to scan")

    # Fall through to directory scan
    if not ADAPTER_DIR.exists():
        print(f"[adapter_discovery] ADAPTER_DIR {ADAPTER_DIR} does not exist, returning empty list")
        return []
    found = sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )
    print(f"[adapter_discovery] filesystem scan found {len(found)} adapters: {found}")
    return found

# ---------------------------------------------------------------------------
# MMLU prompt formatting
# ---------------------------------------------------------------------------
def format_mmlu_prompt(example: dict) -> str:
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        letter = "ABCD"[i]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    return prompt

# ---------------------------------------------------------------------------
# Phase 1: Data preparation (CPU only)
# ---------------------------------------------------------------------------
def prepare_data() -> tuple[dict, dict]:
    """Load MMLU subjects, build gold sets and subset index draws.

    Returns:
        subject_data: {subject -> list of dicts with 'question', 'choices', 'answer'}
        subset_plan:  {subject -> {'gold': [idx,...], 'draws': {size: [[idx,...], ...]}}}
    """
    from datasets import load_dataset

    rng = np.random.RandomState(SEED)

    print(f"\n[Phase 1] Loading {len(SUBJECTS)} MMLU subjects (gold_size={GOLD_SIZE})...")
    subject_data: dict[str, list] = {}
    for subj in SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            rows = list(ds)
            if len(rows) >= GOLD_SIZE:
                subject_data[subj] = rows
                print(f"  {subj}: {len(rows)} questions (OK)")
            else:
                print(f"  {subj}: only {len(rows)} questions, need {GOLD_SIZE} -- SKIPPED")
        except Exception as e:
            print(f"  {subj}: load failed ({e}) -- SKIPPED")

    if len(subject_data) < (2 if IS_SMOKE else 5):
        raise RuntimeError(
            f"Only {len(subject_data)} usable subjects, need at least "
            f"{'2' if IS_SMOKE else '5'}."
        )
    print(f"  Usable subjects: {len(subject_data)}")

    # Build gold indices and subset draws per subject per size
    subset_plan: dict[str, dict] = {}
    for subj, rows in subject_data.items():
        gold_indices = list(range(min(GOLD_SIZE, len(rows))))
        draws_by_size: dict[int, list[list[int]]] = {}
        for k in SUBSET_SIZES:
            draw_size = min(k, len(gold_indices))
            draws: list[list[int]] = []
            for _ in range(N_DRAWS):
                draw = rng.choice(gold_indices, size=draw_size, replace=False).tolist()
                draws.append(sorted(draw))
            draws_by_size[k] = draws
        subset_plan[subj] = {"gold": gold_indices, "draws": draws_by_size}

    # Pre-format all prompts: one list in deterministic order
    # Layout: for each subject, gold rows first, then subset rows (deduped via index)
    # We return raw rows; prompt formatting is done just before building the flat list
    return subject_data, subset_plan

# ---------------------------------------------------------------------------
# Phase 2: Transformers batch evaluation (GPU)
# ---------------------------------------------------------------------------

def _score_prompt_logprobs(
    model, tokenizer, prompt: str, device: str,
) -> tuple[str, dict[str, float]]:
    """Get logprobs for A/B/C/D at the next-token position after prompt.

    Returns: (predicted_letter, {letter: logprob})
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # logits at the last token position predict the next token
    logits = outputs.logits[0, -1, :]  # (vocab_size,)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    letter_logprobs: dict[str, float] = {}
    for letter in "ABCD":
        best = -math.inf
        for variant in [letter, f" {letter}"]:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in token_ids:
                lp = log_probs[tid].item()
                if lp > best:
                    best = lp
        letter_logprobs[letter] = best

    predicted = max(letter_logprobs, key=lambda c: letter_logprobs[c])

    del inputs, outputs, logits, log_probs
    return predicted, letter_logprobs


def _eval_all_prompts(
    model, tokenizer, prompts: list[str], prompts_meta: list[dict], device: str,
) -> dict[str, dict]:
    """Evaluate all prompts and return {subject -> {idx -> {pred, correct, answer_logprob}}}."""
    import torch

    results: dict[str, dict] = {}
    for i, (prompt, meta) in enumerate(zip(prompts, prompts_meta)):
        subj = meta["subject"]
        idx = meta["idx"]
        correct_letter = meta["correct_letter"]

        pred, letter_lps = _score_prompt_logprobs(model, tokenizer, prompt, device)
        answer_lp = letter_lps.get(correct_letter, -math.inf)

        if subj not in results:
            results[subj] = {}
        results[subj][idx] = {
            "pred": pred,
            "correct": pred == correct_letter,
            "answer_logprob": answer_lp,
        }

        # Periodic cleanup per GPU_CODING_GUIDELINES
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return results


def run_hf_evaluation(
    subject_data: dict,
    subset_plan: dict,
    adapters: list[str],
) -> dict:
    """Load base model once, evaluate base + each adapter via PEFT LoRA hot-swap.

    Returns full evaluation dict:
      {
        'base': {subj -> {idx -> {pred, correct, answer_logprob}}},
        'adapters': {adapter_name -> {'scores': ..., 'eval_time_s': float}},
        '_tokenizer': tokenizer object
      }
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Build flat list of ALL unique (subject, gold_index) pairs needed
    needed: dict[str, set[int]] = {}
    for subj, plan in subset_plan.items():
        idxs: set[int] = set(plan["gold"])
        for draws in plan["draws"].values():
            for draw in draws:
                idxs.update(draw)
        needed[subj] = idxs

    # Build ordered (subj, idx) list and corresponding prompts
    prompts: list[str] = []
    prompts_meta: list[dict] = []
    for subj in sorted(needed):
        rows = subject_data[subj]
        for idx in sorted(needed[subj]):
            row = rows[idx]
            prompt = format_mmlu_prompt(row)
            correct_letter = "ABCD"[row["answer"]]
            prompts.append(prompt)
            prompts_meta.append({"subject": subj, "idx": idx, "correct_letter": correct_letter})

    n_prompts = len(prompts)
    print(f"\n[Phase 2] {n_prompts} unique prompts across {len(needed)} subjects")
    print(f"  Adapters to evaluate: {len(adapters)}")

    # Load base model (NF4 quantized to fit in A5000 24GB)
    print(f"  Loading base model ({BASE_MODEL})...")
    use_bf16 = torch.cuda.is_bf16_supported()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    base_model.eval()

    eval_results: dict = {"base": {}, "adapters": {}, "_tokenizer": tokenizer}

    # Base model evaluation
    print("  Evaluating base model...")
    t0 = time.time()
    eval_results["base"] = _eval_all_prompts(base_model, tokenizer, prompts, prompts_meta, device)
    base_time = time.time() - t0
    print(f"    Base model: {base_time:.1f}s for {n_prompts} prompts")

    # Per-adapter evaluation via PEFT LoRA loading/unloading
    for adapter_idx, adapter_name in enumerate(adapters):
        adapter_path = str(ADAPTER_DIR / adapter_name)
        if not (ADAPTER_DIR / adapter_name / "adapter_config.json").exists():
            print(f"  [{adapter_idx+1}/{len(adapters)}] {adapter_name}: adapter_config.json missing, SKIP")
            eval_results["adapters"][adapter_name] = {"error": "adapter_config.json missing"}
            continue

        print(f"  [{adapter_idx+1}/{len(adapters)}] {adapter_name}...", end="", flush=True)
        t0 = time.time()
        try:
            lora_model = PeftModel.from_pretrained(base_model, adapter_path)
            lora_model.eval()
            scores = _eval_all_prompts(lora_model, tokenizer, prompts, prompts_meta, device)
            elapsed = time.time() - t0
            eval_results["adapters"][adapter_name] = {
                "scores": scores,
                "eval_time_s": round(elapsed, 2),
            }
            # Unload LoRA to reuse base model
            del lora_model
            gc.collect()
            torch.cuda.empty_cache()
            print(f" {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f" ERROR after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            eval_results["adapters"][adapter_name] = {"error": str(e), "eval_time_s": elapsed}
            gc.collect()
            torch.cuda.empty_cache()

    # Cleanup
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return eval_results

# ---------------------------------------------------------------------------
# Phase 3: Ranking analysis (CPU)
# ---------------------------------------------------------------------------

def compute_accuracy(scores: dict[int, dict], indices: list[int]) -> float:
    """Compute fraction correct over a set of question indices."""
    if not indices:
        return 0.0
    correct = sum(1 for idx in indices if scores.get(idx, {}).get("correct", False))
    return correct / len(indices)


def compute_answer_ppl(scores: dict[int, dict], indices: list[int]) -> float:
    """Compute mean answer-only PPL (exp(-mean log P(correct_answer | prompt)))."""
    log_probs = []
    for idx in indices:
        lp = scores.get(idx, {}).get("answer_logprob", None)
        if lp is not None and math.isfinite(lp):
            log_probs.append(lp)
    if not log_probs:
        return float("inf")
    mean_nll = -np.mean(log_probs)
    return float(np.exp(mean_nll))


def rank_vector(scores: list[float]) -> list[float]:
    """Convert scores to ranks (higher score = lower rank number). Ties get midrank."""
    arr = np.array(scores, dtype=float)
    # scipy kendalltau handles ties automatically via midrank when we pass raw scores
    return arr.tolist()


def analyze_rankings(
    subject_data: dict,
    subset_plan: dict,
    eval_results: dict,
) -> dict:
    """Compute per-subject and overall ranking correlations.

    Returns the full results dict matching the SPEC schema.
    """
    adapters_raw = eval_results["adapters"]
    tokenizer = eval_results["_tokenizer"]

    # Collect adapter names that evaluated successfully
    valid_adapters = [
        name for name, info in adapters_raw.items()
        if "error" not in info and "scores" in info
    ]
    print(f"\n[Phase 3] Ranking analysis: {len(valid_adapters)} valid adapters")

    base_scores_by_subj = eval_results["base"]  # {subj -> {idx -> {...}}}

    # Aggregate per-adapter per-subject results
    adapter_per_subject: dict[str, dict[str, dict]] = {}
    for adapter_name in valid_adapters:
        adapter_scores_by_subj = adapters_raw[adapter_name]["scores"]
        per_subj: dict[str, dict] = {}
        for subj, plan in subset_plan.items():
            gold_idxs = plan["gold"]
            subj_scores = adapter_scores_by_subj.get(subj, {})
            gold_acc = compute_accuracy(subj_scores, gold_idxs)
            gold_ppl = compute_answer_ppl(subj_scores, gold_idxs)

            subset_accs_by_size: dict[str, list[float]] = {}
            for k, draws in plan["draws"].items():
                accs = [compute_accuracy(subj_scores, draw) for draw in draws]
                subset_accs_by_size[str(k)] = [round(a, 4) for a in accs]

            per_subj[subj] = {
                "gold_accuracy": round(gold_acc, 4),
                "gold_ppl": round(gold_ppl, 4),
                "subset_accuracies": subset_accs_by_size,
            }
        adapter_per_subject[adapter_name] = per_subj

    # Base model per-subject accuracy
    base_per_subject: dict[str, dict] = {}
    for subj, plan in subset_plan.items():
        gold_idxs = plan["gold"]
        base_subj = base_scores_by_subj.get(subj, {})
        correct = sum(1 for idx in gold_idxs if base_subj.get(idx, {}).get("correct", False))
        total = len(gold_idxs)
        base_per_subject[subj] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total if total else 0.0, 4),
        }
    base_overall = round(np.mean([v["accuracy"] for v in base_per_subject.values()]), 4)

    # Per-subject ranking analysis
    rankings_per_subject: dict[str, dict] = {}
    all_tau_by_size: dict[int, list[float]] = {k: [] for k in SUBSET_SIZES}
    all_ppl_tau: list[float] = []

    for subj in subject_data:
        if subj not in subset_plan:
            continue

        # Vectors over valid adapters
        gold_accs = []
        gold_ppls = []
        for name in valid_adapters:
            info = adapter_per_subject.get(name, {}).get(subj, {})
            gold_accs.append(info.get("gold_accuracy", 0.0))
            gold_ppls.append(info.get("gold_ppl", float("inf")))

        if len(gold_accs) < 2:
            print(f"  {subj}: fewer than 2 valid adapters, skipping ranking")
            continue

        # Gold ranking: sort adapters by gold accuracy descending
        gold_order = np.argsort(-np.array(gold_accs))  # indices into valid_adapters
        gold_ranking_names = [valid_adapters[i] for i in gold_order]

        # Tau per subset size per draw, then aggregate
        tau_by_size_results: dict[str, dict] = {}
        plan = subset_plan[subj]

        for k in SUBSET_SIZES:
            draws = plan["draws"][k]
            draw_taus: list[float] = []
            top1_labels: list[str] = []
            for draw in draws:
                subset_accs = []
                for name in valid_adapters:
                    subj_scores = adapters_raw[name]["scores"].get(subj, {})
                    subset_accs.append(compute_accuracy(subj_scores, draw))
                tau_val, _ = kendalltau(gold_accs, subset_accs)
                if not math.isnan(tau_val):
                    draw_taus.append(float(tau_val))
                # Top-1 for stability check
                top1_idx = int(np.argmax(subset_accs))
                top1_labels.append(valid_adapters[top1_idx])

            mean_tau = float(np.mean(draw_taus)) if draw_taus else 0.0
            std_tau = float(np.std(draw_taus)) if draw_taus else 0.0
            top1_stable = len(set(top1_labels)) == 1

            tau_by_size_results[str(k)] = {
                "mean": round(mean_tau, 4),
                "std": round(std_tau, 4),
                "per_draw": [round(t, 4) for t in draw_taus],
            }
            if draw_taus:
                all_tau_by_size[k].append(mean_tau)

        # PPL ranking tau (negated PPL: lower PPL = better = higher rank)
        neg_ppls = [-p if math.isfinite(p) else -1e9 for p in gold_ppls]
        tau_ppl, _ = kendalltau(gold_accs, neg_ppls)
        tau_ppl = float(tau_ppl) if not math.isnan(tau_ppl) else 0.0
        all_ppl_tau.append(tau_ppl)

        # Top-1 stability per size (from draws)
        top1_stable_by_size: dict[str, bool] = {}
        for k in SUBSET_SIZES:
            draws = plan["draws"][k]
            top1s = set()
            for draw in draws:
                subset_accs = []
                for name in valid_adapters:
                    subj_scores = adapters_raw[name]["scores"].get(subj, {})
                    subset_accs.append(compute_accuracy(subj_scores, draw))
                top1s.add(valid_adapters[int(np.argmax(subset_accs))])
            top1_stable_by_size[str(k)] = len(top1s) == 1

        rankings_per_subject[subj] = {
            "gold_ranking": gold_ranking_names,
            "tau_by_size": tau_by_size_results,
            "tau_ppl_vs_gold": round(tau_ppl, 4),
            "top1_stable_across_draws": top1_stable_by_size,
        }
        print(
            f"  {subj}: "
            + " | ".join(
                f"tau@{k}={tau_by_size_results[str(k)]['mean']:.3f}"
                for k in SUBSET_SIZES
            )
            + f" | tau_ppl={tau_ppl:.3f}"
        )

    # Overall aggregates
    mean_tau_by_size: dict[str, float] = {}
    for k in SUBSET_SIZES:
        vals = all_tau_by_size[k]
        mean_tau_by_size[str(k)] = round(float(np.mean(vals)) if vals else 0.0, 4)

    mean_ppl_tau = round(float(np.mean(all_ppl_tau)) if all_ppl_tau else 0.0, 4)

    # Top-1 stability fraction across subjects
    top1_stability_by_size: dict[str, str] = {}
    for k in SUBSET_SIZES:
        stable_count = sum(
            1 for subj_data in rankings_per_subject.values()
            if subj_data.get("top1_stable_across_draws", {}).get(str(k), False)
        )
        n_subj = len(rankings_per_subject)
        top1_stability_by_size[str(k)] = f"{stable_count}/{n_subj}"

    # Kill criteria assessment
    tau_10 = mean_tau_by_size.get(str(SUBSET_SIZES[0]), 0.0)
    tau_mid = mean_tau_by_size.get(str(SUBSET_SIZES[1]) if len(SUBSET_SIZES) > 1 else str(SUBSET_SIZES[0]), 0.0)
    tau_large = mean_tau_by_size.get(str(SUBSET_SIZES[-1]), 0.0)

    # K1
    if tau_10 >= 0.7:
        k1_verdict = "PASS"
    elif tau_mid >= 0.7:
        k1_verdict = "NUANCED"
    elif tau_large < 0.7:
        k1_verdict = "KILL"
    else:
        k1_verdict = "NUANCED"

    # K2: mean time per domain per adapter
    eval_times = [
        info["eval_time_s"]
        for info in adapters_raw.values()
        if "eval_time_s" in info and "error" not in info
    ]
    n_subj_for_timing = len(subject_data)
    mean_per_adapter_s = float(np.mean(eval_times)) if eval_times else 0.0
    # per-domain = total adapter time / number of subjects
    per_domain_s = mean_per_adapter_s / max(1, n_subj_for_timing)
    k2_verdict = "PASS" if per_domain_s < 60 else "KILL"

    # K3: accuracy tau vs gold, and ppl tau vs gold
    # Use the smallest subset size tau as representative for accuracy ranking
    acc_tau_vs_gold = tau_10  # tau between 10-q accuracy and 100-q gold
    ppl_tau_vs_gold = mean_ppl_tau
    if acc_tau_vs_gold > ppl_tau_vs_gold:
        k3_verdict = "PASS"
    elif acc_tau_vs_gold < 0.3 and ppl_tau_vs_gold < 0.3:
        k3_verdict = "KILL"
    else:
        k3_verdict = "NUANCED"

    # Timing summary
    total_time = sum(eval_times) if eval_times else 0.0

    print(f"\n[Kill Criteria]")
    print(f"  K1 tau@{SUBSET_SIZES[0]}={tau_10:.3f}, tau@{SUBSET_SIZES[-1]}={tau_large:.3f} -> {k1_verdict}")
    print(f"  K2 per-domain={per_domain_s:.1f}s -> {k2_verdict}")
    print(f"  K3 acc_tau={acc_tau_vs_gold:.3f}, ppl_tau={ppl_tau_vs_gold:.3f} -> {k3_verdict}")

    # Assemble final results dict matching SPEC schema
    results = {
        "experiment": "task_accuracy_evolve_signal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "config": {
            "subjects": list(subject_data.keys()),
            "n_adapters": len(valid_adapters),
            "n_draws": N_DRAWS,
            "subset_sizes": SUBSET_SIZES,
            "gold_size": GOLD_SIZE,
            "seed": SEED,
            "smoke_test": IS_SMOKE,
        },
        "base_accuracy": {
            "per_subject": base_per_subject,
            "overall": base_overall,
        },
        "adapters": {
            name: {
                "per_subject": adapter_per_subject.get(name, {}),
                "eval_time_s": adapters_raw[name].get("eval_time_s", 0.0),
            }
            for name in valid_adapters
        },
        "rankings": {
            "per_subject": rankings_per_subject,
            "overall": {
                "mean_tau_by_size": mean_tau_by_size,
                "mean_tau_ppl_vs_gold": mean_ppl_tau,
                "top1_stability_by_size": top1_stability_by_size,
            },
        },
        "timing": {
            "per_adapter_mean_s": round(mean_per_adapter_s, 1),
            "per_domain_mean_s": round(per_domain_s, 1),
            "total_s": round(total_time, 1),
        },
        "kill_criteria": {
            f"K1_mean_tau_{SUBSET_SIZES[0]}": tau_10,
            f"K1_mean_tau_{SUBSET_SIZES[1] if len(SUBSET_SIZES) > 1 else SUBSET_SIZES[0]}": tau_mid,
            f"K1_mean_tau_{SUBSET_SIZES[-1]}": tau_large,
            "K1_threshold": 0.7,
            "K1_verdict": k1_verdict,
            "K2_per_domain_time_s": round(per_domain_s, 1),
            "K2_threshold_s": 60,
            "K2_verdict": k2_verdict,
            "K3_acc_tau_vs_gold": round(acc_tau_vs_gold, 4),
            "K3_ppl_tau_vs_gold": round(ppl_tau_vs_gold, 4),
            "K3_verdict": k3_verdict,
        },
    }

    # Also attach per-adapter errors for auditability
    errors = {
        name: info["error"]
        for name, info in adapters_raw.items()
        if "error" in info
    }
    if errors:
        results["adapter_errors"] = errors

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70)
    print("task_accuracy_evolve_signal")
    print(f"  SMOKE_TEST={IS_SMOKE}")
    print(f"  BASE_MODEL={BASE_MODEL}")
    print(f"  ADAPTER_DIR={ADAPTER_DIR}")
    print(f"  subjects={SUBJECTS}")
    print(f"  SUBSET_SIZES={SUBSET_SIZES}  N_DRAWS={N_DRAWS}  GOLD_SIZE={GOLD_SIZE}")
    print("=" * 70)

    # Phase 1: data (CPU)
    subject_data, subset_plan = prepare_data()

    # Adapter list
    adapters = get_adapter_list()[:MAX_ADAPTERS]
    if not adapters:
        print("[WARNING] No adapters found. Will only evaluate base model.")
    print(f"\nAdapters selected ({len(adapters)}): {adapters}")

    # Phase 2: HF GPU evaluation (function-scoped for clean memory)
    eval_results = run_hf_evaluation(subject_data, subset_plan, adapters)

    # Phase 3: ranking analysis (CPU)
    results = analyze_rankings(subject_data, subset_plan, eval_results)
    results["timing"]["wall_clock_total_s"] = round(time.time() - t_start, 1)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    kc = results["kill_criteria"]
    for key, val in kc.items():
        print(f"  {key}: {val}")

    signal.alarm(0)  # cancel timeout
    return results


if __name__ == "__main__":
    main()
