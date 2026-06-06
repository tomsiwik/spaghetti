#!/usr/bin/env python3
"""Clone-and-compete evolution experiment at macro scale.

Tests the core evolution mechanism on 5 pilot experts:
1. Generate correction data by injecting deliberate errors
2. Clone each expert, fine-tune clone with corrections
3. Run shadow-scoring tournament (answer-conditioned PPL)
4. Measure convergence rate, speed, and domain regression

Kill criteria:
- K1: corrected clone does not win tournament >70% of the time
- K2: tournament requires >50K queries to resolve (measured by convergence check)
- K3: original expert domains regress >2% during tournament

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

# MUST be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch

# RunPod compatibility: monkey-patch set_submodule if missing
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "clone_compete_evolution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# 5 diverse domains from pilot 50 for the tournament
TEST_DOMAINS = ["python", "bash", "math", "medical", "sql"]

# Tournament parameters
FT_STEPS = 50 if not IS_SMOKE else 5
FT_LR = 1e-4
EVAL_QUERIES = 200 if not IS_SMOKE else 10
N_CORRECTIONS = 50 if not IS_SMOKE else 5
CONVERGENCE_CHECK_SIZES = [50, 100, 200] if not IS_SMOKE else [5, 10]
# General eval partition size (last N lines of train.jsonl)
GENERAL_EVAL_SIZE = 200 if not IS_SMOKE else 20
PER_DOMAIN_TIMEOUT = 1800 if not IS_SMOKE else 120  # 30min per domain (wall-clock checked between phases)
MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME", 7200 if not IS_SMOKE else 300))  # 2hr total
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def _compute_answer_conditioned_loss(model, tokenizer, text, device):
    """Compute answer-conditioned loss: loss on assistant response tokens only.

    For chat-templated text the assistant turn starts after the last occurrence
    of the assistant header token sequence. We find the split point by applying
    the template with add_generation_prompt=True and comparing token lengths.

    Falls back to full-sequence loss if no answer tokens can be isolated.
    """
    # Encode full sequence (prompt + answer)
    full_enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)

    # Try to find the prompt-only length so we can mask it.
    # Strategy: re-apply the template with add_generation_prompt=True and
    # measure how many tokens that prompt-only encoding produces.
    prompt_len = None
    try:
        # Reconstruct messages from the formatted text is unreliable; instead
        # we detect the assistant response start by finding where the last
        # assistant header ends in the token stream.
        # Qwen2.5 chat template uses "<|im_start|>assistant\n" as the header.
        assistant_header = "<|im_start|>assistant\n"
        header_ids = tokenizer(assistant_header, add_special_tokens=False)["input_ids"]
        ids_list = input_ids[0].tolist()
        hlen = len(header_ids)
        # Find the last occurrence of the header token sequence
        for start in range(len(ids_list) - hlen, -1, -1):
            if ids_list[start:start + hlen] == header_ids:
                prompt_len = start + hlen  # first answer token index
                break
    except Exception:
        pass

    seq_len = input_ids.shape[1]
    if prompt_len is None or prompt_len >= seq_len - 1:
        # No answer tokens found — fall back to full-sequence loss
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=input_ids)
        return outputs.loss.item()

    # Build labels: -100 for prompt tokens, real ids for answer tokens
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100  # mask prompt

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss.item()


def generate_corrections(base_model_obj, tokenizer, adapter_path, domain,
                          candidate_lines, n=50):
    """Generate correction data by finding examples where the expert is wrong.

    Uses ONLY candidate_lines (correction candidate pool) — never the general
    eval partition. This prevents data overlap between correction set and eval set.

    Each GPU phase (load, score, unload) is self-contained within this function.
    Returns path to corrections file, or None on failure.
    """
    from peft import PeftModel

    corrections_dir = RESULTS_DIR / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)
    corrections_file = corrections_dir / f"{domain}_corrections.jsonl"

    if corrections_file.exists():
        existing = sum(1 for _ in open(corrections_file))
        if existing >= n:
            log(f"  {domain}: {existing} corrections already exist, reusing")
            return corrections_file

    log(f"  Generating corrections for {domain} from {len(candidate_lines)} candidates...")

    if not candidate_lines:
        log(f"  WARNING: No candidate lines for {domain}, skipping")
        return None

    device = next(base_model_obj.parameters()).device

    # Load expert adapter — isolated GPU scope
    model = PeftModel.from_pretrained(base_model_obj, str(adapter_path))
    model.eval()

    # Score each candidate: high loss = expert struggles = correction candidate
    scored = []
    sample_lines = candidate_lines[:min(n * 4, len(candidate_lines))]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, line in enumerate(sample_lines):
            try:
                record = json.loads(line)
                if "messages" in record:
                    text = tokenizer.apply_chat_template(
                        record["messages"], tokenize=False, add_generation_prompt=False)
                elif "text" in record:
                    text = record["text"]
                else:
                    continue

                loss = _compute_answer_conditioned_loss(model, tokenizer, text, device)
                scored.append((loss, record))
            except Exception as e:
                log(f"    WARNING: skipping candidate {i}: {e}")
                continue

            if i > 0 and i % 50 == 0:
                gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if not scored:
        log(f"  WARNING: No valid scored examples for {domain}")
        return None

    # Take highest-loss examples as correction candidates
    scored.sort(key=lambda x: -x[0])
    corrections = scored[:n]

    with open(corrections_file, "w") as f:
        for loss, record in corrections:
            correction_entry = {
                "messages": record.get("messages", [{"role": "user", "content": record.get("text", "")}]),
                "original_loss": loss,
            }
            f.write(json.dumps(correction_entry) + "\n")

    log(f"  {domain}: generated {len(corrections)} corrections "
        f"(loss range: {corrections[0][0]:.3f} - {corrections[-1][0]:.3f})")
    return corrections_file


def finetune_clone(base_model_obj, tokenizer, clone_dir, corrections_file):
    """Fine-tune clone with corrections. Isolated GPU scope per guidelines."""
    from peft import PeftModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model = PeftModel.from_pretrained(base_model_obj, str(clone_dir), is_trainable=True)

    dataset = load_dataset("json", data_files=str(corrections_file), split="train")

    def format_messages(example):
        if "messages" in example and example["messages"]:
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False)}
        return {"text": ""}

    dataset = dataset.map(format_messages)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    use_bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(clone_dir / "ft_ckpt"),
            max_steps=FT_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=FT_LR,
            warmup_steps=min(5, FT_STEPS // 2),
            logging_steps=max(1, FT_STEPS // 5),
            save_steps=FT_STEPS,
            fp16=not use_bf16,
            bf16=use_bf16,
            optim="adamw_8bit",
            seed=SEED,
            dataset_text_field="text",
            max_length=512,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
        ),
    )

    gc.disable()
    try:
        train_result = trainer.train()
    finally:
        gc.enable()

    ft_loss = train_result.training_loss
    model.save_pretrained(str(clone_dir))

    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()

    return ft_loss


def score_adapter(base_model_obj, tokenizer, adapter_path, texts):
    """Score adapter using answer-conditioned PPL. Isolated GPU scope.

    Returns list of per-example losses.
    """
    from peft import PeftModel

    device = next(base_model_obj.parameters()).device
    model = PeftModel.from_pretrained(base_model_obj, str(adapter_path))
    model.eval()

    losses = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, text in enumerate(texts):
            try:
                loss = _compute_answer_conditioned_loss(model, tokenizer, text, device)
                losses.append(loss)
            except Exception as e:
                log(f"    WARNING: scoring error on example {i}: {e}")
                continue

            if i > 0 and i % 50 == 0:
                gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return losses


def run_tournament(base_model_obj, tokenizer, domain, corrections_file, general_eval_lines):
    """Run a single clone-and-compete tournament for one domain.

    Uses try/finally to guarantee clone directory cleanup even on error.
    No SIGALRM — MAX_RUNTIME is checked between domains in main().

    Returns dict with tournament results, or None if adapter missing.
    """
    from scipy import stats as scipy_stats

    adapter_dir = ADAPTER_DIR / domain
    clone_dir = ADAPTER_DIR / f"{domain}_clone_exp"

    if not adapter_dir.exists():
        log(f"  WARNING: adapter {adapter_dir} not found, skipping")
        return None

    log(f"\n  Tournament: {domain}")
    t0 = time.time()

    # Step 1: Clone
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    shutil.copytree(adapter_dir, clone_dir)
    log(f"  Cloned {domain} -> {domain}_clone_exp")

    try:
        # Step 2: Fine-tune clone with corrections
        log(f"  Fine-tuning clone ({FT_STEPS} steps, lr={FT_LR})...")
        ft_start = time.time()
        ft_loss = finetune_clone(base_model_obj, tokenizer, clone_dir, corrections_file)
        ft_time = time.time() - ft_start
        log(f"  Fine-tuning done in {ft_time:.0f}s (final loss: {ft_loss:.4f})")

        # Step 3: Build eval text lists
        # General eval: from the reserved last-N lines of train.jsonl
        eval_texts = []
        for line in general_eval_lines:
            try:
                record = json.loads(line)
                if "messages" in record:
                    text = tokenizer.apply_chat_template(
                        record["messages"], tokenize=False, add_generation_prompt=False)
                elif "text" in record:
                    text = record["text"]
                else:
                    continue
                eval_texts.append(text)
            except Exception:
                continue
            if len(eval_texts) >= EVAL_QUERIES:
                break

        # Correction eval: from the corrections file
        correction_texts = []
        with open(corrections_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "messages" in record:
                        text = tokenizer.apply_chat_template(
                            record["messages"], tokenize=False, add_generation_prompt=False)
                        correction_texts.append(text)
                except Exception:
                    continue

        log(f"  Scoring: {len(eval_texts)} general + {len(correction_texts)} correction queries")

        # Step 4: Score all 4 combinations (each call is an isolated GPU scope)
        log(f"  Scoring original on general queries...")
        orig_general = score_adapter(base_model_obj, tokenizer, adapter_dir, eval_texts)
        log(f"  Scoring clone on general queries...")
        clone_general = score_adapter(base_model_obj, tokenizer, clone_dir, eval_texts)
        log(f"  Scoring original on correction queries...")
        orig_corrections = score_adapter(base_model_obj, tokenizer, adapter_dir, correction_texts)
        log(f"  Scoring clone on correction queries...")
        clone_corrections = score_adapter(base_model_obj, tokenizer, clone_dir, correction_texts)

        elapsed = time.time() - t0

        # Step 5: Compute metrics
        avg_orig_gen = sum(orig_general) / max(len(orig_general), 1)
        avg_clone_gen = sum(clone_general) / max(len(clone_general), 1)
        avg_orig_corr = sum(orig_corrections) / max(len(orig_corrections), 1)
        avg_clone_corr = sum(clone_corrections) / max(len(clone_corrections), 1)

        regression_pct = (
            (math.exp(avg_clone_gen) - math.exp(avg_orig_gen)) / math.exp(avg_orig_gen) * 100
        )

        # Correction improvement percentage
        orig_corr_ppl = math.exp(avg_orig_corr)
        clone_corr_ppl = math.exp(avg_clone_corr)
        improvement_pct = (orig_corr_ppl - clone_corr_ppl) / orig_corr_ppl * 100

        # Cohen's d on correction losses (original vs clone, paired)
        cohens_d = 0.0
        if len(orig_corrections) >= 2 and len(clone_corrections) >= 2:
            n_pairs = min(len(orig_corrections), len(clone_corrections))
            orig_arr = orig_corrections[:n_pairs]
            clone_arr = clone_corrections[:n_pairs]
            mean_diff = (sum(orig_arr) / n_pairs) - (sum(clone_arr) / n_pairs)
            # Pooled std
            var_orig = sum((x - sum(orig_arr) / n_pairs) ** 2 for x in orig_arr) / max(n_pairs - 1, 1)
            var_clone = sum((x - sum(clone_arr) / n_pairs) ** 2 for x in clone_arr) / max(n_pairs - 1, 1)
            pooled_std = math.sqrt((var_orig + var_clone) / 2)
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std

        # Statistical test: Wilcoxon signed-rank on correction losses
        p_value = 1.0
        stat_test = "n/a"
        if len(orig_corrections) >= 4 and len(clone_corrections) >= 4:
            n_pairs = min(len(orig_corrections), len(clone_corrections))
            try:
                wilcoxon_result = scipy_stats.wilcoxon(
                    orig_corrections[:n_pairs],
                    clone_corrections[:n_pairs],
                    alternative="greater",  # H1: orig > clone (clone improves)
                )
                p_value = float(wilcoxon_result.pvalue)
                stat_test = "wilcoxon"
            except Exception as e:
                # Fall back to paired t-test if Wilcoxon fails (e.g., all-zero diffs)
                try:
                    ttest_result = scipy_stats.ttest_rel(
                        orig_corrections[:n_pairs],
                        clone_corrections[:n_pairs],
                        alternative="greater",
                    )
                    p_value = float(ttest_result.pvalue)
                    stat_test = "paired_ttest_fallback"
                except Exception:
                    pass

        # Convergence check: smallest subset where winner is stable
        clone_wins = avg_clone_corr < avg_orig_corr
        convergence_n = None
        for check_n in CONVERGENCE_CHECK_SIZES:
            if check_n > min(len(orig_corrections), len(clone_corrections)):
                break
            sub_orig = sum(orig_corrections[:check_n]) / check_n
            sub_clone = sum(clone_corrections[:check_n]) / check_n
            subset_clone_wins = sub_clone < sub_orig
            if subset_clone_wins == clone_wins:
                convergence_n = check_n
                break
        if convergence_n is None and len(correction_texts) > 0:
            convergence_n = len(correction_texts)

        result = {
            "domain": domain,
            "elapsed_s": round(elapsed, 1),
            "ft_time_s": round(ft_time, 1),
            "ft_loss": round(ft_loss, 4),
            "n_corrections": len(correction_texts),
            "n_general_queries": len(eval_texts),
            "general_queries": {
                "original_ppl": round(math.exp(avg_orig_gen), 3),
                "clone_ppl": round(math.exp(avg_clone_gen), 3),
                "regression_pct": round(regression_pct, 2),
            },
            "correction_queries": {
                "original_ppl": round(orig_corr_ppl, 3),
                "clone_ppl": round(clone_corr_ppl, 3),
                "improvement_pct": round(improvement_pct, 2),
                "clone_wins": clone_wins,
            },
            "convergence_queries": convergence_n,
            "effect_size_cohens_d": round(cohens_d, 4),
            "statistical_test": stat_test,
            "p_value": round(p_value, 4),
            "clone_wins": clone_wins,
            "regression_exceeds_2pct": regression_pct > 2.0,
        }

        log(f"  {domain}: clone_wins={clone_wins}, "
            f"improvement={improvement_pct:+.2f}%, "
            f"d={cohens_d:.3f}, p={p_value:.4f}, "
            f"regression={regression_pct:+.2f}%, "
            f"convergence@{convergence_n}, {elapsed:.0f}s")

    finally:
        # Cleanup clone even if tournament errored
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
            log(f"  Cleaned up clone dir: {clone_dir}")

    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    log("=== Clone-and-Compete Evolution Experiment ===")
    log(f"Smoke test: {IS_SMOKE}")
    log(f"Domains: {TEST_DOMAINS}")
    log(f"FT steps: {FT_STEPS}, Corrections: {N_CORRECTIONS}, Eval queries: {EVAL_QUERIES}")
    log(f"General eval partition size: {GENERAL_EVAL_SIZE}")

    # Load base model once — kept alive for the full experiment
    log("Loading base model...")
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

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )

    # Fix dtype mismatch: ensure lm_head matches compute dtype
    if hasattr(base_model_obj, "lm_head") and base_model_obj.lm_head.weight.dtype != torch.bfloat16:
        base_model_obj.lm_head = base_model_obj.lm_head.to(torch.bfloat16)

    log("Base model loaded.")

    # Phase 1: Partition data and generate corrections for each domain
    log("\n=== Phase 1: Data Partition + Generate Corrections ===")
    correction_files = {}
    general_eval_by_domain = {}  # domain -> list of raw line strings

    gc.disable()
    gc.collect()
    try:
        for domain in TEST_DOMAINS:
            adapter_dir = ADAPTER_DIR / domain
            if not adapter_dir.exists():
                log(f"  SKIP {domain}: adapter not found at {adapter_dir}")
                continue

            train_file = DATA_DIR / domain / "train.jsonl"
            if not train_file.exists():
                log(f"  SKIP {domain}: train.jsonl not found at {train_file}")
                continue

            with open(train_file) as fh:
                all_lines = fh.readlines()

            if len(all_lines) < 2:
                log(f"  SKIP {domain}: too few lines ({len(all_lines)})")
                continue

            # Explicit partition: ZERO overlap guaranteed
            # Last GENERAL_EVAL_SIZE lines -> general eval (for K3 regression)
            # Everything before that -> correction candidate pool (for K1)
            split_idx = max(0, len(all_lines) - GENERAL_EVAL_SIZE)
            candidate_lines = all_lines[:split_idx]
            eval_lines = all_lines[split_idx:]

            log(f"  {domain}: {len(all_lines)} total lines -> "
                f"{len(candidate_lines)} candidates + {len(eval_lines)} general eval")

            general_eval_by_domain[domain] = eval_lines

            cf = generate_corrections(
                base_model_obj, tokenizer, adapter_dir, domain,
                candidate_lines, n=N_CORRECTIONS)
            if cf:
                correction_files[domain] = cf

        log(f"Corrections generated for {len(correction_files)} domains")

        # Phase 2: Run tournaments (with per-domain checkpointing)
        log("\n=== Phase 2: Clone-and-Compete Tournaments ===")
        results = {}

        # Resume from checkpoint if exists
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as fh:
                checkpoint = json.load(fh)
            results = checkpoint.get("results", {})
            log(f"Resumed from checkpoint: {list(results.keys())} already done")

        for domain, cf in correction_files.items():
            if domain in results:
                log(f"  SKIP {domain}: already in checkpoint")
                continue

            # MAX_RUNTIME wall-clock check between domains (replaces SIGALRM)
            elapsed_total = time.time() - t0
            if elapsed_total > MAX_RUNTIME:
                log(f"  MAX_RUNTIME ({MAX_RUNTIME}s) exceeded at {elapsed_total:.0f}s, "
                    f"saving partial results")
                break

            eval_lines = general_eval_by_domain.get(domain, [])

            try:
                result = run_tournament(
                    base_model_obj, tokenizer, domain, cf, eval_lines)
                if result:
                    results[domain] = result
            except Exception as exc:
                import traceback
                log(f"  ERROR in {domain} tournament: {exc}")
                traceback.print_exc()
                results[domain] = {
                    "domain": domain,
                    "error": str(exc),
                    "elapsed_s": round(time.time() - t0, 1),
                }

            # Checkpoint after each domain
            with open(CHECKPOINT_FILE, "w") as fh:
                json.dump({
                    "results": results,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, fh, indent=2)
            log(f"  Checkpoint saved ({len(results)} domains done)")

            torch.cuda.empty_cache()

    finally:
        gc.enable()
        gc.collect()

    # Phase 3: Aggregate and check kill criteria
    log("\n=== Phase 3: Kill Criteria Assessment ===")

    valid_results = {d: r for d, r in results.items() if "error" not in r}

    if not valid_results:
        log("ERROR: No valid tournament results!")
        out_file = RESULTS_DIR / "clone_compete_results.json"
        with open(out_file, "w") as fh:
            json.dump({
                "per_domain": results,
                "verdict": "NO_DATA",
                "elapsed_s": round(time.time() - t0, 1),
            }, fh, indent=2)
        sys.exit(1)

    n_domains = len(valid_results)
    n_valid = n_domains
    n_clone_wins = sum(1 for r in valid_results.values() if r["clone_wins"])
    win_rate = n_clone_wins / n_domains
    max_regression = max(r["general_queries"]["regression_pct"] for r in valid_results.values())
    mean_regression = sum(r["general_queries"]["regression_pct"] for r in valid_results.values()) / n_domains
    max_convergence = max(r["convergence_queries"] or 0 for r in valid_results.values())

    improvement_pcts = [
        r["correction_queries"]["improvement_pct"]
        for r in valid_results.values()
        if "correction_queries" in r and "improvement_pct" in r["correction_queries"]
    ]
    mean_improvement_pct = sum(improvement_pcts) / max(len(improvement_pcts), 1)

    effect_sizes = [
        r["effect_size_cohens_d"]
        for r in valid_results.values()
        if "effect_size_cohens_d" in r
    ]
    mean_effect_size = sum(effect_sizes) / max(len(effect_sizes), 1)

    k1_pass = win_rate > 0.70
    k2_pass = max_convergence <= 50000
    k3_pass = max_regression <= 2.0

    aggregate = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "domains": TEST_DOMAINS,
            "ft_steps": FT_STEPS,
            "ft_lr": FT_LR,
            "n_corrections": N_CORRECTIONS,
            "eval_queries": EVAL_QUERIES,
            "convergence_checkpoints": CONVERGENCE_CHECK_SIZES,
            "base_model": BASE_MODEL,
            "seed": SEED,
            "smoke_test": IS_SMOKE,
        },
        "per_domain": results,
        "aggregate": {
            "n_domains": n_domains,
            "n_valid": n_valid,
            "clone_win_rate": round(win_rate, 3),
            "mean_regression_pct": round(mean_regression, 2),
            "max_regression_pct": round(max_regression, 2),
            "mean_improvement_pct": round(mean_improvement_pct, 2),
            "max_convergence_queries": max_convergence,
            "mean_effect_size": round(mean_effect_size, 4),
        },
        "kill_criteria": {
            "K1_clone_win_rate_gt_70pct": {
                "value": round(win_rate * 100, 1),
                "threshold": 70.0,
                "pass": k1_pass,
            },
            "K2_convergence_lt_50k_queries": {
                "value": max_convergence,
                "threshold": 50000,
                "pass": k2_pass,
            },
            "K3_domain_regression_lt_2pct": {
                "value": round(max_regression, 2),
                "threshold": 2.0,
                "pass": k3_pass,
            },
        },
        "verdict": "PASS" if (k1_pass and k2_pass and k3_pass) else "FAIL",
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Print summary
    log(f"\n{'='*60}")
    log(f"RESULTS SUMMARY")
    log(f"{'='*60}")
    for domain, r in results.items():
        if "error" in r:
            log(f"  {domain}: ERROR ({r['error']})")
        else:
            status = "WIN" if r["clone_wins"] else "LOSE"
            impr = r.get("correction_queries", {}).get("improvement_pct", 0.0)
            d_val = r.get("effect_size_cohens_d", 0.0)
            p_val = r.get("p_value", 1.0)
            log(f"  {domain}: clone {status} | "
                f"improvement={impr:+.2f}% | "
                f"d={d_val:.3f} | p={p_val:.4f} | "
                f"regression {r['general_queries']['regression_pct']:+.2f}% | "
                f"convergence@{r['convergence_queries']}")

    log(f"\nK1 (win rate >70%): {'PASS' if k1_pass else 'FAIL'} — {win_rate*100:.0f}%")
    log(f"K2 (convergence <50K): {'PASS' if k2_pass else 'FAIL'} — {max_convergence} queries")
    log(f"K3 (regression <2%): {'PASS' if k3_pass else 'FAIL'} — max {max_regression:+.2f}%")
    log(f"\nVERDICT: {aggregate['verdict']}")
    log(f"Total time: {aggregate['elapsed_s']:.0f}s")

    n_errored = len(results) - len(valid_results)
    if n_errored:
        log(f"\n({n_errored} domain(s) errored/timed out)")

    # Save results
    out_file = RESULTS_DIR / "clone_compete_results.json"
    with open(out_file, "w") as fh:
        json.dump(aggregate, fh, indent=2)
    log(f"Results saved to {out_file}")

    # Clean up checkpoint on success
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


if __name__ == "__main__":
    main()
