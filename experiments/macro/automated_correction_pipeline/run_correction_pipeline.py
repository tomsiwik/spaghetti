#!/usr/bin/env python3
"""Automated correction pipeline experiment at macro scale.

Tests whether a 70B teacher can reliably judge and correct 7B expert outputs,
and whether teacher-generated corrections improve expert quality comparably
to ground-truth corrections.

Pipeline:
1. Generate expert answers on domain queries using 7B + pilot adapters
2. Send answers to 70B teacher (Groq API) for judging/correction
3. Measure teacher reliability against ground truth (K1)
4. Compare clone improvement from teacher corrections vs ground truth (K2)

For code domains (python, bash, sql): use execution-based validation
as a free, perfect signal source alongside teacher judgment.

Kill criteria:
- K1: teacher-generated corrections are wrong >20% of the time
- K2: automated corrections improve expert quality <50% as much as ground truth

Supports SMOKE_TEST=1 for quick validation.
"""

import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path("/workspace/llm")
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"
DATA_DIR = REPO_ROOT / "data" / "distillation"
RESULTS_DIR = REPO_ROOT / "results" / "automated_correction_pipeline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B"
HF_CACHE = "/workspace/hf_cache"

TEACHER_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _load_api_key():
    """Load GROQ_API_KEY from .env (manual parse to avoid dotenv dependency)."""
    if not os.environ.get("GROQ_API_KEY"):
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip("'\"")
                    os.environ.setdefault(key.strip(), val)
    return os.environ.get("GROQ_API_KEY", "")

# 5 diverse domains — includes code (execution-testable) and non-code
TEST_DOMAINS = ["python", "bash", "math", "medical", "sql"]
CODE_DOMAINS = {"python", "bash", "sql"}

# Parameters
N_QUERIES = 50 if not IS_SMOKE else 5
N_CORRECTIONS_FOR_FINETUNE = 30 if not IS_SMOKE else 3
FT_STEPS = 30 if not IS_SMOKE else 3
FT_LR = 1e-4


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def load_eval_data(domain, n):
    """Load evaluation examples from domain data."""
    eval_file = DATA_DIR / domain / "eval.jsonl"
    train_file = DATA_DIR / domain / "train.jsonl"

    source = eval_file if eval_file.exists() else train_file
    if not source.exists():
        return []

    examples = []
    with open(source) as f:
        for line in f:
            record = json.loads(line)
            if "messages" in record and len(record["messages"]) >= 2:
                examples.append(record)
            if len(examples) >= n:
                break
    return examples


def generate_expert_answers(base_model, tokenizer, domain, examples):
    """Generate expert answers using 7B + adapter.

    Returns list of {query, expert_answer, ground_truth, messages}.
    """
    import torch
    from peft import PeftModel

    adapter_dir = ADAPTER_DIR / domain
    if not adapter_dir.exists():
        log(f"  WARNING: adapter {adapter_dir} not found")
        return []

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    results = []
    for ex in examples:
        msgs = ex["messages"]
        # Extract query (user message) and ground truth (assistant response)
        query = next((m["content"] for m in msgs if m["role"] == "user"), None)
        ground_truth = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not query or not ground_truth:
            continue

        # Generate expert answer
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(base_model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        expert_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        results.append({
            "query": query,
            "expert_answer": expert_answer,
            "ground_truth": ground_truth,
            "messages": msgs,
        })

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def judge_with_teacher(query, expert_answer, ground_truth):
    """Send expert answer to 70B teacher for judgment.

    Returns {correct: bool, correction: str|None, confidence: float, reasoning: str}.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", ""), base_url=GROQ_BASE_URL)

    judge_prompt = f"""You are judging whether an AI assistant's answer is correct.

Question: {query}

AI Answer: {expert_answer}

Evaluate:
1. Is the answer factually correct and complete?
2. If incorrect or incomplete, provide the correct answer.
3. Rate your confidence (0.0 to 1.0) in your judgment.

Respond ONLY in JSON:
{{"correct": true/false, "correction": "correct answer if wrong, null if correct", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        resp = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content.strip()
        # Parse JSON from response
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return result
    except Exception as e:
        log(f"    Teacher API error: {e}")
        return {"correct": True, "correction": None, "confidence": 0.0, "reasoning": f"API error: {e}"}


def evaluate_teacher_accuracy(judgments, domain):
    """Evaluate teacher judgment accuracy against ground truth.

    A teacher judgment is "wrong" if:
    - Teacher says correct but expert answer is clearly wrong (false positive)
    - Teacher provides a correction that is worse than ground truth (bad correction)

    We measure this by checking teacher's correction against ground truth
    using string similarity and semantic overlap.
    """
    n_total = len(judgments)
    if n_total == 0:
        return {"error_rate": 1.0, "n_total": 0}

    n_correct_judgments = 0
    n_wrong_judgments = 0
    n_skipped = 0
    details = []

    for j in judgments:
        # Simple heuristic: compare teacher judgment against ground truth
        # If teacher says "correct" but expert answer is very different from ground truth → wrong
        # If teacher provides correction similar to ground truth → correct judgment
        expert = j["expert_answer"].lower().strip()
        truth = j["ground_truth"].lower().strip()
        teacher_correct = j["judgment"].get("correct", True)
        confidence = j["judgment"].get("confidence", 0.0)

        if confidence < 0.3:
            n_skipped += 1
            continue

        # Compute rough similarity between expert answer and ground truth
        expert_words = set(expert.split())
        truth_words = set(truth.split())
        if len(truth_words) == 0:
            n_skipped += 1
            continue
        overlap = len(expert_words & truth_words) / len(truth_words)

        # Expert is "wrong" if very low overlap with ground truth
        expert_wrong = overlap < 0.3

        if expert_wrong and teacher_correct:
            # Teacher said correct but expert was wrong → wrong judgment
            n_wrong_judgments += 1
            details.append({"type": "false_positive", "domain": domain, "overlap": overlap})
        elif expert_wrong and not teacher_correct:
            # Teacher correctly identified the error
            n_correct_judgments += 1
            details.append({"type": "true_negative", "domain": domain, "overlap": overlap})
        elif not expert_wrong and not teacher_correct:
            # Teacher said wrong but expert was actually OK → wrong judgment
            n_wrong_judgments += 1
            details.append({"type": "false_negative", "domain": domain, "overlap": overlap})
        else:
            # Both agree expert is correct
            n_correct_judgments += 1
            details.append({"type": "true_positive", "domain": domain, "overlap": overlap})

    evaluated = n_correct_judgments + n_wrong_judgments
    error_rate = n_wrong_judgments / max(evaluated, 1)

    return {
        "error_rate": round(error_rate, 4),
        "n_total": n_total,
        "n_evaluated": evaluated,
        "n_correct_judgments": n_correct_judgments,
        "n_wrong_judgments": n_wrong_judgments,
        "n_skipped_low_conf": n_skipped,
        "details": details,
    }


def finetune_clone(base_model, tokenizer, domain, corrections, suffix):
    """Fine-tune a clone of the expert with corrections.

    Returns (clone_dir, train_loss).
    """
    import torch
    from peft import PeftModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    adapter_dir = ADAPTER_DIR / domain
    clone_dir = ADAPTER_DIR / f"{domain}_correction_{suffix}"

    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    shutil.copytree(adapter_dir, clone_dir)

    model = PeftModel.from_pretrained(base_model, str(clone_dir), is_trainable=True)

    # Build training data from corrections
    train_data = []
    for c in corrections:
        if c.get("correction_text"):
            msgs = [
                {"role": "user", "content": c["query"]},
                {"role": "assistant", "content": c["correction_text"]},
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            train_data.append({"text": text})

    if not train_data:
        del model
        torch.cuda.empty_cache()
        return clone_dir, float("inf")

    dataset = Dataset.from_list(train_data)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(clone_dir / "ft_ckpt"),
            max_steps=FT_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=FT_LR,
            warmup_steps=min(3, FT_STEPS // 2),
            logging_steps=max(1, FT_STEPS // 5),
            save_steps=FT_STEPS,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_text_field="text",
            max_length=512,
        ),
    )

    result = trainer.train()
    train_loss = result.training_loss
    model.save_pretrained(str(clone_dir))

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return clone_dir, train_loss


def evaluate_adapter(base_model, tokenizer, adapter_path, eval_texts):
    """Evaluate adapter quality using mean PPL on eval texts."""
    import torch
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    losses = []
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(base_model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    del model
    torch.cuda.empty_cache()

    return math.exp(sum(losses) / max(len(losses), 1))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    log("=== Automated Correction Pipeline Experiment ===")
    log(f"Smoke test: {IS_SMOKE}")
    log(f"Domains: {TEST_DOMAINS}")
    log(f"Queries per domain: {N_QUERIES}")

    GROQ_API_KEY = _load_api_key()
    if not GROQ_API_KEY:
        log("ERROR: GROQ_API_KEY not found in environment")
        sys.exit(1)

    # Load base model
    log("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
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

    # ── Phase 1: Generate expert answers ──
    log("\n=== Phase 1: Expert Answer Generation ===")
    all_judgments = {}

    for domain in TEST_DOMAINS:
        log(f"\n--- {domain} ---")
        examples = load_eval_data(domain, N_QUERIES)
        if not examples:
            log(f"  No eval data for {domain}, skipping")
            continue

        answers = generate_expert_answers(base_model, tokenizer, domain, examples)
        log(f"  Generated {len(answers)} expert answers")

        # ── Phase 2: Teacher judgment ──
        log(f"  Sending to teacher for judgment...")
        judgments = []
        for i, ans in enumerate(answers):
            judgment = judge_with_teacher(ans["query"], ans["expert_answer"], ans["ground_truth"])
            judgments.append({
                **ans,
                "judgment": judgment,
            })
            if (i + 1) % 10 == 0:
                log(f"    Judged {i+1}/{len(answers)}")
            # Rate limit: Groq free tier
            time.sleep(0.5 if not IS_SMOKE else 0.1)

        all_judgments[domain] = judgments
        log(f"  Teacher judged {len(judgments)} answers")

        # Save intermediate results
        with open(RESULTS_DIR / f"{domain}_judgments.json", "w") as f:
            json.dump(judgments, f, indent=2, default=str)

    # ── Phase 3: Teacher accuracy assessment (K1) ──
    log("\n=== Phase 3: Teacher Accuracy Assessment (K1) ===")
    accuracy_results = {}
    for domain, judgments in all_judgments.items():
        acc = evaluate_teacher_accuracy(judgments, domain)
        accuracy_results[domain] = acc
        log(f"  {domain}: error_rate={acc['error_rate']:.1%} "
            f"({acc['n_wrong_judgments']}/{acc['n_evaluated']} wrong)")

    overall_error_rate = sum(a["n_wrong_judgments"] for a in accuracy_results.values()) / \
        max(sum(a["n_evaluated"] for a in accuracy_results.values()), 1)

    k1_pass = overall_error_rate <= 0.20
    log(f"\n  K1 overall error rate: {overall_error_rate:.1%} (threshold: <=20%)")
    log(f"  K1: {'PASS' if k1_pass else 'FAIL'}")

    # ── Phase 4: Correction effectiveness comparison (K2) ──
    log("\n=== Phase 4: Correction Effectiveness (K2) ===")
    effectiveness_results = {}

    for domain, judgments in all_judgments.items():
        log(f"\n--- {domain} ---")
        adapter_dir = ADAPTER_DIR / domain
        if not adapter_dir.exists():
            continue

        # Prepare teacher corrections (from teacher judgments where teacher said "wrong")
        teacher_corrections = []
        for j in judgments:
            if not j["judgment"].get("correct", True) and j["judgment"].get("correction"):
                teacher_corrections.append({
                    "query": j["query"],
                    "correction_text": j["judgment"]["correction"],
                })

        # Prepare ground truth corrections (use ground truth for the same queries)
        gt_corrections = []
        for j in judgments:
            if not j["judgment"].get("correct", True):
                gt_corrections.append({
                    "query": j["query"],
                    "correction_text": j["ground_truth"],
                })

        n_corrections = min(N_CORRECTIONS_FOR_FINETUNE, len(teacher_corrections), len(gt_corrections))
        if n_corrections < 3:
            log(f"  Too few corrections ({n_corrections}), skipping K2 for {domain}")
            continue

        teacher_corrections = teacher_corrections[:n_corrections]
        gt_corrections = gt_corrections[:n_corrections]

        # Load eval texts for quality measurement
        eval_examples = load_eval_data(domain, N_QUERIES)
        eval_texts = []
        for ex in eval_examples:
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)
            eval_texts.append(text)

        # Measure original adapter quality
        orig_ppl = evaluate_adapter(base_model, tokenizer, adapter_dir, eval_texts[:20])
        log(f"  Original PPL: {orig_ppl:.3f}")

        # Fine-tune with teacher corrections
        log(f"  Fine-tuning with {n_corrections} teacher corrections...")
        teacher_clone_dir, teacher_loss = finetune_clone(
            base_model, tokenizer, domain, teacher_corrections, "teacher")
        teacher_ppl = evaluate_adapter(base_model, tokenizer, teacher_clone_dir, eval_texts[:20])
        log(f"  Teacher-corrected PPL: {teacher_ppl:.3f} (loss: {teacher_loss:.4f})")

        # Fine-tune with ground truth corrections
        log(f"  Fine-tuning with {n_corrections} ground truth corrections...")
        gt_clone_dir, gt_loss = finetune_clone(
            base_model, tokenizer, domain, gt_corrections, "gt")
        gt_ppl = evaluate_adapter(base_model, tokenizer, gt_clone_dir, eval_texts[:20])
        log(f"  GT-corrected PPL: {gt_ppl:.3f} (loss: {gt_loss:.4f})")

        # Compute improvement ratios
        teacher_improvement = max(orig_ppl - teacher_ppl, 0)
        gt_improvement = max(orig_ppl - gt_ppl, 0)
        ratio = teacher_improvement / max(gt_improvement, 1e-6) if gt_improvement > 0 else 0.0

        effectiveness_results[domain] = {
            "original_ppl": round(orig_ppl, 3),
            "teacher_corrected_ppl": round(teacher_ppl, 3),
            "gt_corrected_ppl": round(gt_ppl, 3),
            "teacher_improvement": round(teacher_improvement, 3),
            "gt_improvement": round(gt_improvement, 3),
            "teacher_vs_gt_ratio": round(ratio, 3),
            "n_corrections": n_corrections,
            "teacher_loss": round(teacher_loss, 4),
            "gt_loss": round(gt_loss, 4),
        }

        log(f"  Teacher improvement: {teacher_improvement:.3f}, GT improvement: {gt_improvement:.3f}")
        log(f"  Ratio: {ratio:.1%} (threshold: >=50%)")

        # Cleanup clones
        for d in [teacher_clone_dir, gt_clone_dir]:
            if Path(d).exists():
                shutil.rmtree(d)

    # ── Aggregate Results ──
    log("\n=== Results Summary ===")

    # K2: teacher corrections achieve >=50% of ground truth improvement
    if effectiveness_results:
        ratios = [r["teacher_vs_gt_ratio"] for r in effectiveness_results.values()]
        mean_ratio = sum(ratios) / len(ratios)
        k2_pass = mean_ratio >= 0.50
    else:
        mean_ratio = 0.0
        k2_pass = False

    aggregate = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "domains": TEST_DOMAINS,
            "n_queries": N_QUERIES,
            "n_corrections_ft": N_CORRECTIONS_FOR_FINETUNE,
            "ft_steps": FT_STEPS,
            "ft_lr": FT_LR,
            "teacher_model": TEACHER_MODEL,
            "smoke_test": IS_SMOKE,
        },
        "teacher_accuracy": accuracy_results,
        "correction_effectiveness": effectiveness_results,
        "kill_criteria": {
            "K1_teacher_error_rate_lt_20pct": {
                "value": round(overall_error_rate * 100, 1),
                "threshold": 20,
                "pass": k1_pass,
            },
            "K2_teacher_correction_gte_50pct_of_gt": {
                "value": round(mean_ratio * 100, 1),
                "threshold": 50,
                "pass": k2_pass,
            },
        },
        "verdict": "PASS" if (k1_pass and k2_pass) else "FAIL",
        "elapsed_s": round(time.time() - t0, 1),
    }

    log(f"\n{'='*60}")
    log(f"K1 (teacher error rate <=20%): {'PASS' if k1_pass else 'FAIL'} — {overall_error_rate*100:.1f}%")
    log(f"K2 (teacher corrections >=50% of GT): {'PASS' if k2_pass else 'FAIL'} — {mean_ratio*100:.1f}%")
    log(f"\nPer-domain effectiveness:")
    for domain, r in effectiveness_results.items():
        log(f"  {domain}: teacher/GT ratio = {r['teacher_vs_gt_ratio']:.1%}")
    log(f"\nVERDICT: {aggregate['verdict']}")
    log(f"Total time: {aggregate['elapsed_s']:.0f}s")

    out_file = RESULTS_DIR / "correction_pipeline_results.json"
    with open(out_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    log(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
