#!/usr/bin/env python3
"""Individual expert held-out eval: test each adapter ALONE on MMLU.

Diagnoses whether MMLU -3.67pp regression comes from:
  (a) Distillation quality failure: individuals also regress → fix training
  (b) Composition interference: individuals neutral, composed harms → fix composition

Kill criteria:
  K1: mean_delta > -1pp → PASS (composition interference is the cause)
  K2: mean_delta < -3pp → PASS (distillation memorization is the cause)
  Inconclusive: -3pp <= mean_delta <= -1pp

Supports SMOKE_TEST=1 for <5 min validation.
Crash-resilient: saves checkpoint after each adapter; resumes on restart.
"""

import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Monkey-patch set_submodule for PyTorch builds missing it
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

# ---------------------------------------------------------------------------
# Smoke test config (SMOKE_TEST=1 → fast validation run)
# ---------------------------------------------------------------------------
IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

N_EXPERTS        = 3   if IS_SMOKE else 20
N_SUBJECTS       = 5   if IS_SMOKE else 57   # full MMLU = 57 subjects
MAX_PER_SUBJECT  = 10  if IS_SMOKE else None  # None = use full test set
BOOTSTRAP_B      = 100 if IS_SMOKE else 1000
MAX_RUNTIME_SECS = 300 if IS_SMOKE else 3 * 3600  # 3h hard limit

SEED = 42

BASE_MODEL   = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
ADAPTER_DIR  = Path("/workspace/llm/adapters")
RESULTS_DIR  = Path("/workspace/llm/results/individual_expert_held_out")
CHECKPOINT   = RESULTS_DIR / "checkpoint.json"
OUT_PATH     = RESULTS_DIR / "individual_expert_results.json"

# ---------------------------------------------------------------------------
# Full 57-subject MMLU list (deterministic order, matches cais/mmlu configs)
# ---------------------------------------------------------------------------
ALL_MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# ---------------------------------------------------------------------------
# Domain-to-MMLU mapping (from pilot50_held_out_eval/eval_mmlu.py)
# Used for in-domain vs out-of-domain analysis.
# ---------------------------------------------------------------------------
DOMAIN_TO_MMLU = {
    # Science
    "physics":       ["high_school_physics", "college_physics", "conceptual_physics"],
    "chemistry":     ["high_school_chemistry", "college_chemistry"],
    "biology":       ["high_school_biology", "college_biology"],
    "math":          ["high_school_mathematics", "college_mathematics", "elementary_mathematics", "abstract_algebra"],
    "statistics":    ["high_school_statistics", "econometrics"],
    "astronomy":     ["astronomy"],
    "genetics":      ["medical_genetics"],
    "ecology":       ["high_school_biology"],
    "neuroscience":  ["high_school_psychology", "professional_psychology"],
    # Professional
    "legal":         ["professional_law", "international_law", "jurisprudence"],
    "medical":       ["professional_medicine", "clinical_knowledge", "college_medicine", "anatomy"],
    "finance":       ["high_school_macroeconomics", "high_school_microeconomics"],
    "accounting":    ["professional_accounting"],
    "marketing":     ["marketing"],
    "cybersecurity": ["computer_security", "security_studies"],
    # Reasoning
    "logic-puzzles": ["formal_logic", "logical_fallacies"],
    "ethics":        ["business_ethics", "moral_disputes", "moral_scenarios"],
    "abstract-math": ["abstract_algebra", "college_mathematics"],
    # Programming
    "python":        ["high_school_computer_science", "college_computer_science", "machine_learning"],
    "cpp":           ["high_school_computer_science", "college_computer_science"],
    "java":          ["high_school_computer_science", "college_computer_science"],
    "javascript":    ["high_school_computer_science"],
    "rust":          ["college_computer_science"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def get_adapter_list() -> list[str]:
    """Return adapters sorted by pilot50 PPL improvement (fallback: alphabetical)."""
    benchmark = Path("/workspace/llm/results/pilot50_benchmark.json")
    if benchmark.exists():
        try:
            with open(benchmark) as f:
                data = json.load(f)
            ranked = [
                (name, info.get("ppl_improvement_pct", 0))
                for name, info in data.get("per_adapter", {}).items()
            ]
            if ranked:  # Only use benchmark if it actually has entries
                ranked.sort(key=lambda x: x[1], reverse=True)
                return [name for name, _ in ranked]
        except Exception as e:
            print(f"[warn] Could not load benchmark ranking: {e}")

    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def load_bnb_base_model():
    """Load NF4-quantised Qwen2.5-7B. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for causal LM batch inference

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def free_model(model):
    """Delete model object and flush GPU caches."""
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# MMLU prompt formatting
# ---------------------------------------------------------------------------

def format_mmlu_prompt(example) -> str:
    q = example["question"]
    choices = example["choices"]
    prompt = f"{q}\n"
    for i, ch in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {ch}\n"
    prompt += "Answer:"
    return prompt


# ---------------------------------------------------------------------------
# Core evaluation: batched log-prob scoring
# ---------------------------------------------------------------------------

def evaluate_mmlu_subjects(
    model,
    tokenizer,
    subjects: list[str],
    max_per_subject: int | None = None,
    batch_size: int = 16,
    label: str = "model",
) -> dict:
    """
    Evaluate model on a list of MMLU subjects using batched log-prob scoring.

    Returns:
        {
          "per_subject": {subject: {"correct": int, "total": int, "accuracy": float}},
          "overall":     {"correct": int, "total": int, "accuracy": float}
        }
    """
    from datasets import load_dataset

    # Pre-compute choice token IDs (try with and without leading space)
    choice_token_ids: dict[str, int] = {}
    choice_token_ids_space: dict[str, int] = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_token_ids[letter] = ids[0]
        ids_sp = tokenizer.encode(f" {letter}", add_special_tokens=False)
        choice_token_ids_space[letter] = ids_sp[-1] if ids_sp else ids[0]

    per_subject: dict = {}
    total_correct = 0
    total_count   = 0

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"  [skip] {subject}: {e}")
            continue

        if max_per_subject is not None and len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))

        examples    = list(ds)
        n           = len(examples)
        prompts     = [format_mmlu_prompt(ex) for ex in examples]
        gold_labels = ["ABCD"[ex["answer"]] for ex in examples]

        subj_correct = 0

        # Process in batches to keep GPU busy
        for b_start in range(0, n, batch_size):
            b_prompts = prompts[b_start : b_start + batch_size]
            b_golds   = gold_labels[b_start : b_start + batch_size]

            # Tokenise batch with left-padding
            enc = tokenizer(
                b_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids      = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                # Logits at the last (non-padded) token position for each sequence
                last_logits = out.logits[:, -1, :]          # (B, vocab)
                log_probs   = torch.log_softmax(last_logits, dim=-1)  # (B, vocab)
            except torch.cuda.OutOfMemoryError:
                # Retry with batch_size=1 (fallback)
                print(f"  [OOM] {subject} batch {b_start} — retrying single-sample")
                log_probs_list = []
                for prompt in b_prompts:
                    enc_s = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    input_ids_s = enc_s["input_ids"].to(model.device)
                    with torch.no_grad():
                        out_s = model(input_ids=input_ids_s)
                    log_probs_list.append(
                        torch.log_softmax(out_s.logits[0, -1], dim=-1).unsqueeze(0)
                    )
                log_probs = torch.cat(log_probs_list, dim=0)

            # Score each choice and pick argmax
            for j, gold in enumerate(b_golds):
                scores = {
                    letter: max(
                        log_probs[j, choice_token_ids[letter]].item(),
                        log_probs[j, choice_token_ids_space[letter]].item(),
                    )
                    for letter in "ABCD"
                }
                pred = max(scores, key=scores.get)
                subj_correct += int(pred == gold)

        subj_acc = subj_correct / max(1, n)
        per_subject[subject] = {
            "correct":  subj_correct,
            "total":    n,
            "accuracy": round(subj_acc, 4),
        }
        total_correct += subj_correct
        total_count   += n

    overall_acc = total_correct / max(1, total_count)
    return {
        "per_subject": per_subject,
        "overall": {
            "correct":  total_correct,
            "total":    total_count,
            "accuracy": round(overall_acc, 4),
        },
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(deltas: list[float], B: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI on mean(deltas) via resampling adapters with replacement."""
    rng = np.random.RandomState(seed)
    arr = np.array(deltas)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# In-domain / out-of-domain analysis
# ---------------------------------------------------------------------------

def compute_indomain_outdomain(
    adapter_name: str,
    per_subject_base: dict,
    per_subject_expert: dict,
) -> tuple[float | None, float | None]:
    """
    Returns (in_domain_avg_delta_pp, out_of_domain_avg_delta_pp).

    In-domain subjects are those listed in DOMAIN_TO_MMLU[adapter_name].
    Out-of-domain subjects are everything else that was evaluated.
    """
    # Infer domain key: strip common suffixes/variants for lookup
    # Try exact name first, then strip trailing numbers/dashes
    domain_subjects: set[str] = set()
    if adapter_name in DOMAIN_TO_MMLU:
        domain_subjects = set(DOMAIN_TO_MMLU[adapter_name])
    else:
        # Fuzzy match: take first word of adapter name
        prefix = adapter_name.split("-")[0].split("_")[0]
        if prefix in DOMAIN_TO_MMLU:
            domain_subjects = set(DOMAIN_TO_MMLU[prefix])

    in_deltas:  list[float] = []
    out_deltas: list[float] = []

    for subj, base_info in per_subject_base.items():
        expert_info = per_subject_expert.get(subj)
        if expert_info is None:
            continue
        delta = (expert_info["accuracy"] - base_info["accuracy"]) * 100
        if subj in domain_subjects:
            in_deltas.append(delta)
        else:
            out_deltas.append(delta)

    in_avg  = float(np.mean(in_deltas))  if in_deltas  else None
    out_avg = float(np.mean(out_deltas)) if out_deltas else None
    return in_avg, out_avg


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT) as f:
                return json.load(f)
        except Exception as e:
            print(f"[warn] Could not load checkpoint: {e}")
    return {}


def save_checkpoint(data: dict):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(CHECKPOINT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== individual_expert_held_out ===")
    print(f"SMOKE_TEST={IS_SMOKE}")
    print(f"N_EXPERTS={N_EXPERTS}, N_SUBJECTS={N_SUBJECTS}, "
          f"MAX_PER_SUBJECT={MAX_PER_SUBJECT}, BOOTSTRAP_B={BOOTSTRAP_B}")
    print(f"BASE_MODEL={BASE_MODEL}")
    print(f"RESULTS → {OUT_PATH}")

    # -----------------------------------------------------------------------
    # Select subjects
    # -----------------------------------------------------------------------
    subjects = ALL_MMLU_SUBJECTS[:N_SUBJECTS]  # deterministic, no shuffle
    print(f"Subjects: {len(subjects)}")

    # -----------------------------------------------------------------------
    # Select adapters
    # -----------------------------------------------------------------------
    all_adapters = get_adapter_list()
    # Filter to those that actually exist on disk
    available = [a for a in all_adapters if (ADAPTER_DIR / a).is_dir() and
                 (ADAPTER_DIR / a / "adapter_config.json").exists()]
    if not available:
        print("[error] No adapters found. Check ADAPTER_DIR.")
        sys.exit(1)
    adapters = available[:N_EXPERTS]
    print(f"Adapters to test: {adapters}")

    # -----------------------------------------------------------------------
    # Load checkpoint (crash resilience)
    # -----------------------------------------------------------------------
    ckpt = load_checkpoint()
    base_results: dict | None     = ckpt.get("base_results")
    expert_store: dict            = ckpt.get("individual_experts", {})
    completed_adapters: set[str]  = set(expert_store.keys())

    # -----------------------------------------------------------------------
    # Phase 1: Base model eval (skip if checkpoint has it)
    # -----------------------------------------------------------------------
    tokenizer = None  # loaded alongside model each time

    if base_results is None:
        print("\n=== Phase 1: Evaluating base model ===")
        base_model, tokenizer = load_base_model_safe()
        # Disable GC during heavy GPU inference (nanochat pattern: ~500ms/step saved)
        gc.disable()
        gc.collect()
        try:
            base_results = evaluate_mmlu_subjects(
                base_model, tokenizer, subjects,
                max_per_subject=MAX_PER_SUBJECT,
                batch_size=16,
                label="base",
            )
        finally:
            gc.enable()
            gc.collect()
            free_model(base_model)
        base_acc = base_results["overall"]["accuracy"]
        print(f"Base overall accuracy: {base_acc:.4f} "
              f"({base_results['overall']['correct']}/{base_results['overall']['total']})")
        ckpt["base_results"] = base_results
        save_checkpoint(ckpt)
    else:
        base_acc = base_results["overall"]["accuracy"]
        print(f"\n[resume] Base results loaded from checkpoint (acc={base_acc:.4f})")
        # We need tokenizer for adapter eval below; load it lazily.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # -----------------------------------------------------------------------
    # Phase 2: Individual adapter eval
    # -----------------------------------------------------------------------
    from peft import PeftModel

    print(f"\n=== Phase 2: Individual adapter evaluation ({len(adapters)} adapters) ===")
    # Disable GC during heavy GPU inference loop (nanochat pattern: ~500ms/step saved)
    gc.disable()
    gc.collect()
    try:
        for i, adapter_name in enumerate(adapters):
            elapsed = time.time() - t_start
            if elapsed > MAX_RUNTIME_SECS:
                print(f"[timeout] Reached {elapsed:.0f}s limit. Stopping early.")
                break

            if adapter_name in completed_adapters:
                print(f"[skip] {adapter_name} already in checkpoint")
                continue

            print(f"\n--- Expert {i+1}/{len(adapters)}: {adapter_name} "
                  f"(elapsed {elapsed/60:.1f} min) ---")
            t1 = time.time()

            # Reload fresh base (PeftModel.from_pretrained modifies base in-place)
            try:
                fresh_base, _ = load_base_model_safe()
            except Exception as e:
                print(f"[error] Failed to load base model for {adapter_name}: {e}")
                traceback.print_exc()
                expert_store[adapter_name] = {"error": f"base_load_failed: {e}"}
                completed_adapters.add(adapter_name)
                ckpt["individual_experts"] = expert_store
                save_checkpoint(ckpt)
                continue

            adapter_path = str(ADAPTER_DIR / adapter_name)
            try:
                peft_model = PeftModel.from_pretrained(
                    fresh_base, adapter_path, adapter_name=adapter_name
                )
                peft_model.set_adapter(adapter_name)
                peft_model.eval()
            except Exception as e:
                print(f"[error] Failed to load adapter {adapter_name}: {e}")
                traceback.print_exc()
                free_model(peft_model if 'peft_model' in dir() else fresh_base)
                expert_store[adapter_name] = {"error": f"adapter_load_failed: {e}"}
                completed_adapters.add(adapter_name)
                ckpt["individual_experts"] = expert_store
                save_checkpoint(ckpt)
                continue

            try:
                expert_eval = evaluate_mmlu_subjects(
                    peft_model, tokenizer, subjects,
                    max_per_subject=MAX_PER_SUBJECT,
                    batch_size=16,
                    label=adapter_name,
                )
            except Exception as e:
                print(f"[error] Eval failed for {adapter_name}: {e}")
                traceback.print_exc()
                free_model(peft_model)
                expert_store[adapter_name] = {"error": f"eval_failed: {e}"}
                completed_adapters.add(adapter_name)
                ckpt["individual_experts"] = expert_store
                save_checkpoint(ckpt)
                continue

            free_model(peft_model)

            expert_acc = expert_eval["overall"]["accuracy"]
            delta_pp   = (expert_acc - base_acc) * 100
            dt         = time.time() - t1
            print(f"  {adapter_name}: acc={expert_acc:.4f}  delta={delta_pp:+.2f}pp  [{dt:.0f}s]")

            expert_store[adapter_name] = {
                "eval_results":      expert_eval,
                "accuracy":          round(expert_acc, 4),
                "delta_vs_base_pp":  round(delta_pp, 4),
            }
            completed_adapters.add(adapter_name)
            ckpt["individual_experts"] = expert_store
            save_checkpoint(ckpt)
    finally:
        gc.enable()
        gc.collect()

    # -----------------------------------------------------------------------
    # Phase 3: Analysis
    # -----------------------------------------------------------------------
    print("\n=== Phase 3: Analysis ===")

    deltas: list[float] = []
    n_positive = 0
    n_negative = 0
    n_neutral  = 0

    in_domain_deltas_all:  list[float] = []
    out_domain_deltas_all: list[float] = []

    for adapter_name, info in expert_store.items():
        if "error" in info:
            continue
        d = info["delta_vs_base_pp"]
        deltas.append(d)
        if d > 0.5:
            n_positive += 1
        elif d < -0.5:
            n_negative += 1
        else:
            n_neutral += 1

        # In-domain / out-of-domain decomposition
        per_sub_base   = base_results.get("per_subject", {})
        per_sub_expert = info["eval_results"].get("per_subject", {})
        in_avg, out_avg = compute_indomain_outdomain(
            adapter_name, per_sub_base, per_sub_expert
        )
        if in_avg  is not None: in_domain_deltas_all.append(in_avg)
        if out_avg is not None: out_domain_deltas_all.append(out_avg)

    avg_delta    = float(np.mean(deltas))    if deltas else 0.0
    median_delta = float(np.median(deltas))  if deltas else 0.0
    std_delta    = float(np.std(deltas))     if deltas else 0.0

    ci_lo, ci_hi = (0.0, 0.0)
    if len(deltas) >= 2:
        ci_lo, ci_hi = bootstrap_ci(deltas, B=BOOTSTRAP_B, seed=SEED)

    in_avg_global  = float(np.mean(in_domain_deltas_all))  if in_domain_deltas_all  else None
    out_avg_global = float(np.mean(out_domain_deltas_all)) if out_domain_deltas_all else None

    # Diagnosis
    if avg_delta > -1.0:
        diagnosis = "COMPOSITION_INTERFERENCE"
        detail = (
            f"Individual adapters are roughly neutral (mean_delta={avg_delta:+.2f}pp > -1pp). "
            f"The -3.67pp composed regression comes from weight-space interference during "
            f"composition, not from individual adapter quality. Fix: selective top-k routing "
            f"or composition regularisation."
        )
    elif avg_delta < -3.0:
        diagnosis = "DISTILLATION_QUALITY"
        detail = (
            f"Individual adapters already harm generalisation (mean_delta={avg_delta:+.2f}pp < -3pp). "
            f"Adapters individually memorise training domains and leak into unrelated subjects. "
            f"Fix: more diverse training data, stronger regularisation, better teacher signals."
        )
    else:
        diagnosis = "INCONCLUSIVE"
        detail = (
            f"Mean delta={avg_delta:+.2f}pp falls in the inconclusive range [-3, -1]pp. "
            f"Both distillation quality and composition interference contribute to the regression. "
            f"Decomposition: base individual harm={avg_delta:+.2f}pp, "
            f"composition penalty={-3.67 - avg_delta:+.2f}pp (estimated)."
        )

    # Kill criteria assessment
    k1_pass = avg_delta > -1.0
    k2_pass = avg_delta < -3.0

    print(f"  n_tested:           {len(deltas)}")
    print(f"  avg_delta:          {avg_delta:+.2f}pp")
    print(f"  median_delta:       {median_delta:+.2f}pp")
    print(f"  std_delta:          {std_delta:.2f}pp")
    print(f"  bootstrap 95% CI:   [{ci_lo:+.2f}, {ci_hi:+.2f}]pp")
    print(f"  n_positive/neg/neu: {n_positive}/{n_negative}/{n_neutral}")
    if in_avg_global  is not None: print(f"  in-domain avg:      {in_avg_global:+.2f}pp")
    if out_avg_global is not None: print(f"  out-of-domain avg:  {out_avg_global:+.2f}pp")
    print(f"  diagnosis:          {diagnosis}")
    print(f"  K1 PASS (interference): {k1_pass}")
    print(f"  K2 PASS (memorization): {k2_pass}")

    # -----------------------------------------------------------------------
    # Build final output JSON (matches SPEC.md schema exactly)
    # -----------------------------------------------------------------------
    output = {
        "experiment":  "individual_expert_held_out",
        "timestamp":   iso_now(),
        "base_model":  BASE_MODEL,
        "config": {
            "n_experts":       N_EXPERTS,
            "n_subjects":      len(subjects),
            "max_per_subject": MAX_PER_SUBJECT,
            "seed":            SEED,
            "smoke_test":      IS_SMOKE,
            "subjects":        subjects,
        },
        "base":               base_results,
        "individual_experts": expert_store,
        "analysis": {
            "avg_delta_pp":            round(avg_delta, 4),
            "median_delta_pp":         round(median_delta, 4),
            "std_delta_pp":            round(std_delta, 4),
            "bootstrap_ci_95_pp":      [round(ci_lo, 4), round(ci_hi, 4)],
            "n_positive":              n_positive,
            "n_negative":              n_negative,
            "n_neutral":               n_neutral,
            "n_tested":                len(deltas),
            "in_domain_avg_delta_pp":  round(in_avg_global, 4)  if in_avg_global  is not None else None,
            "out_of_domain_avg_delta_pp": round(out_avg_global, 4) if out_avg_global is not None else None,
            "diagnosis":               diagnosis,
            "detail":                  detail,
        },
        "kill_criteria": {
            "K1_composition_interference": {
                "threshold": "> -1pp",
                "value":     round(avg_delta, 4),
                "pass":      k1_pass,
            },
            "K2_distillation_memorization": {
                "threshold": "< -3pp",
                "value":     round(avg_delta, 4),
                "pass":      k2_pass,
            },
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = OUT_PATH.with_suffix(".tmp")
    with open(tmp_out, "w") as f:
        json.dump(output, f, indent=2)
    tmp_out.replace(OUT_PATH)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min. Results → {OUT_PATH}")

    # Remove checkpoint on clean finish
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint removed (clean exit).")


# ---------------------------------------------------------------------------
# Thin wrapper so load_base_model_safe is available in Phase 1 before
# tokenizer is otherwise assigned.
# ---------------------------------------------------------------------------

def load_base_model_safe():
    """Wrapper around load_bnb_base_model with retry on OOM."""
    try:
        return load_bnb_base_model()
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        # Try again with reduced precision (already fp16 — just retry)
        return load_bnb_base_model()


if __name__ == "__main__":
    main()
