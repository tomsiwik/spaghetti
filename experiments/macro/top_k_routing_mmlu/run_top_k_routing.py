#!/usr/bin/env python3
"""Top-K routing vs full composition on MMLU held-out eval.

Tests whether query-aware expert selection fixes the -3.67pp regression.
Instead of composing ALL N experts, embed the question and select the K
most relevant experts based on cosine similarity to domain descriptions.

Compares at K=1,3,5,10 (from N=50 total):
  (a) Full merge: compose all 50, equal weight
  (b) Top-K: embed question, find K nearest domain experts, compose those
  (c) Base: no adapter

If top-K >> full merge: SOLE should route, not pre-merge everything.
If top-K ≈ full merge: expert selection doesn't help, problem is elsewhere.

Depends: exp_distillation_pilot_50 (supported). No training needed.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/top_k_routing_mmlu")
K_VALUES = [1, 3, 5, 10]
SEED = 42
N_SUBJECTS = 15

HELD_OUT_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering",
]

# Domain descriptions for embedding-based routing
DOMAIN_DESCRIPTIONS = {
    "python": "Python programming, scripting, data analysis, and software development",
    "javascript": "JavaScript web development, Node.js, frontend frameworks",
    "rust": "Rust systems programming, memory safety, performance",
    "go": "Go programming, concurrency, microservices, cloud",
    "cpp": "C++ programming, systems, performance, memory management",
    "java": "Java programming, enterprise, object-oriented design",
    "typescript": "TypeScript, type-safe JavaScript, web development",
    "bash": "Bash shell scripting, Linux command line, system administration",
    "swift": "Swift programming, iOS, macOS development",
    "sql": "SQL databases, queries, data management, relational databases",
    "physics": "Physics, mechanics, thermodynamics, electromagnetism, quantum",
    "chemistry": "Chemistry, organic, inorganic, biochemistry, reactions",
    "biology": "Biology, genetics, ecology, cell biology, evolution",
    "mathematics": "Mathematics, algebra, calculus, statistics, proofs",
    "neuroscience": "Neuroscience, brain, cognition, neural systems",
    "astronomy": "Astronomy, astrophysics, cosmology, space science",
    "geology": "Geology, earth science, minerals, plate tectonics",
    "ecology": "Ecology, ecosystems, environmental science, conservation",
    "genetics": "Genetics, DNA, genomics, heredity, gene expression",
    "quantum-computing": "Quantum computing, qubits, quantum algorithms",
    "medicine": "Medicine, clinical practice, diagnosis, treatment",
    "law": "Law, legal reasoning, jurisprudence, contracts",
    "finance": "Finance, investing, markets, economics, accounting",
    "psychology": "Psychology, behavior, cognition, mental health",
    "education": "Education, teaching, pedagogy, curriculum design",
    "engineering": "Engineering, design, systems, mechanical, electrical",
    "philosophy": "Philosophy, ethics, logic, epistemology, metaphysics",
    "history": "History, historical events, civilizations, geopolitics",
    "political-science": "Political science, government, policy, international relations",
    "sociology": "Sociology, society, social structures, demographics",
    "creative-writing": "Creative writing, fiction, storytelling, narrative",
    "academic-writing": "Academic writing, research papers, scholarly communication",
    "technical-writing": "Technical writing, documentation, manuals, specifications",
    "journalism": "Journalism, news writing, reporting, media",
    "poetry": "Poetry, verse, literary forms, poetic devices",
    "screenwriting": "Screenwriting, scripts, dialogue, film narrative",
    "grant-writing": "Grant writing, proposals, funding applications",
    "copywriting": "Copywriting, marketing, advertising, persuasion",
    "speech-writing": "Speech writing, rhetoric, public speaking",
    "translation": "Translation, multilingual, cross-cultural communication",
    "logical-reasoning": "Logical reasoning, deduction, inference, arguments",
    "mathematical-reasoning": "Mathematical reasoning, proofs, problem solving",
    "scientific-reasoning": "Scientific reasoning, hypothesis testing, evidence",
    "critical-thinking": "Critical thinking, analysis, evaluation, argumentation",
    "analogical-reasoning": "Analogical reasoning, comparisons, pattern recognition",
    "causal-reasoning": "Causal reasoning, cause and effect, mechanisms",
    "probabilistic-reasoning": "Probabilistic reasoning, uncertainty, Bayesian thinking",
    "spatial-reasoning": "Spatial reasoning, geometry, visualization, 3D thinking",
    "ethical-reasoning": "Ethical reasoning, moral philosophy, dilemmas",
    "systems-thinking": "Systems thinking, complexity, feedback loops, emergence",
}


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def get_text_embedding(model, tokenizer, text, max_length=128):
    """Get mean-pooled hidden state embedding for text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use last hidden state, mean pool over sequence
        hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return embedding.cpu().squeeze(0)  # (hidden_dim,)


def build_domain_embeddings(model, tokenizer, available_adapters):
    """Build embedding for each domain's description."""
    print("Building domain embeddings...")
    embeddings = {}
    for adapter_name in available_adapters:
        desc = DOMAIN_DESCRIPTIONS.get(adapter_name, adapter_name.replace("-", " "))
        emb = get_text_embedding(model, tokenizer, desc)
        embeddings[adapter_name] = emb
    return embeddings


def select_top_k(question_emb, domain_embeddings, k):
    """Select top-K most relevant adapters for a question."""
    similarities = {}
    for name, emb in domain_embeddings.items():
        cos_sim = torch.nn.functional.cosine_similarity(question_emb.unsqueeze(0), emb.unsqueeze(0)).item()
        similarities[name] = cos_sim
    sorted_adapters = sorted(similarities, key=similarities.get, reverse=True)
    return sorted_adapters[:k], {n: similarities[n] for n in sorted_adapters[:k]}


def format_mmlu_prompt(example):
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def get_choice_token_ids(tokenizer):
    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]
    return choice_tokens


def evaluate_with_routing(base_model, tokenizer, domain_embeddings, k, subjects,
                          max_per_subject=50):
    """Evaluate MMLU with per-question top-K routing.

    Pre-loads all adapters once, then uses add_weighted_adapter/delete_adapter
    per example to avoid PeftModel re-creation (which corrupts base model).
    """
    choice_tokens = get_choice_token_ids(tokenizer)
    results = {}
    total_correct = 0
    total_count = 0
    adapter_usage = {}

    available = list(domain_embeddings.keys())

    # Pre-load all adapters into a single PeftModel (once)
    adapter_paths = {name: str(ADAPTER_DIR / name) for name in available}
    first = available[0]
    model = PeftModel.from_pretrained(base_model, adapter_paths[first], adapter_name=first)
    for name in available[1:]:
        model.load_adapter(adapter_paths[name], adapter_name=name)
    model.eval()

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skip {subject}: {e}")
            continue
        if len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))

        correct = 0
        count = 0

        for ex in ds:
            prompt = format_mmlu_prompt(ex)
            # Get question embedding for routing (use base through PEFT — disable adapters)
            model.disable_adapter_layers()
            q_emb = get_text_embedding(model, tokenizer, ex["question"])
            model.enable_adapter_layers()

            # Select top-K adapters
            top_k_names, sims = select_top_k(q_emb, domain_embeddings, k)
            for name in top_k_names:
                adapter_usage[name] = adapter_usage.get(name, 0) + 1

            # Compose top-K via weighted adapter (reuses pre-loaded adapters)
            model.add_weighted_adapter(
                adapters=list(top_k_names),
                weights=[1.0 / len(top_k_names)] * len(top_k_names),
                adapter_name="routed",
                combination_type="linear",
            )
            model.set_adapter("routed")

            # Score
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]
                log_probs = torch.log_softmax(logits, dim=-1)
            scores = {}
            for letter in "ABCD":
                tid = choice_tokens[letter]
                tid_space = choice_tokens.get(f" {letter}", tid)
                scores[letter] = max(log_probs[tid].item(), log_probs[tid_space].item())
            pred = max(scores, key=scores.get)
            gold = "ABCD"[ex["answer"]]
            correct += int(pred == gold)
            count += 1

            # Clean up per-example composition (keep individual adapters loaded)
            model.delete_adapter("routed")

        acc = correct / max(1, count)
        results[subject] = {"correct": correct, "total": count, "accuracy": round(acc, 4)}
        total_correct += correct
        total_count += count
        print(f"  {subject}: {acc:.1%} ({correct}/{count})")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    overall_acc = total_correct / max(1, total_count) if total_count > 0 else 0.0
    return {
        "per_subject": results,
        "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)},
        "adapter_usage": dict(sorted(adapter_usage.items(), key=lambda x: x[1], reverse=True)[:20]),
    }


def evaluate_full_merge(base_model, tokenizer, adapter_names, subjects, max_per_subject=50):
    """Evaluate with all adapters merged."""
    choice_tokens = get_choice_token_ids(tokenizer)
    adapter_paths = [str(ADAPTER_DIR / name) for name in adapter_names]
    model = PeftModel.from_pretrained(base_model, adapter_paths[0], adapter_name=adapter_names[0])
    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        model.load_adapter(path, adapter_name=name)
    model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=[1.0 / len(adapter_names)] * len(adapter_names),
        adapter_name="composed",
        combination_type="linear",
    )
    model.set_adapter("composed")
    model.eval()

    results = {}
    total_correct = 0
    total_count = 0

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception:
            continue
        if len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))

        correct = 0
        count = 0
        for ex in ds:
            prompt = format_mmlu_prompt(ex)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]
                log_probs = torch.log_softmax(logits, dim=-1)
            scores = {}
            for letter in "ABCD":
                tid = choice_tokens[letter]
                tid_space = choice_tokens.get(f" {letter}", tid)
                scores[letter] = max(log_probs[tid].item(), log_probs[tid_space].item())
            pred = max(scores, key=scores.get)
            gold = "ABCD"[ex["answer"]]
            correct += int(pred == gold)
            count += 1

        acc = correct / max(1, count)
        results[subject] = {"correct": correct, "total": count, "accuracy": round(acc, 4)}
        total_correct += correct
        total_count += count
        print(f"  {subject}: {acc:.1%} ({correct}/{count})")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    overall_acc = total_correct / max(1, total_count) if total_count > 0 else 0.0
    return {"per_subject": results, "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)}}


def evaluate_base(base_model, tokenizer, subjects, max_per_subject=50):
    choice_tokens = get_choice_token_ids(tokenizer)
    results = {}
    total_correct = 0
    total_count = 0

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception:
            continue
        if len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))

        correct = 0
        count = 0
        for ex in ds:
            prompt = format_mmlu_prompt(ex)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(base_model.device)
            with torch.no_grad():
                outputs = base_model(**inputs)
                logits = outputs.logits[0, -1]
                log_probs = torch.log_softmax(logits, dim=-1)
            scores = {}
            for letter in "ABCD":
                tid = choice_tokens[letter]
                tid_space = choice_tokens.get(f" {letter}", tid)
                scores[letter] = max(log_probs[tid].item(), log_probs[tid_space].item())
            pred = max(scores, key=scores.get)
            gold = "ABCD"[ex["answer"]]
            correct += int(pred == gold)
            count += 1

        acc = correct / max(1, count)
        results[subject] = {"correct": correct, "total": count, "accuracy": round(acc, 4)}
        total_correct += correct
        total_count += count
        print(f"  {subject}: {acc:.1%} ({correct}/{count})")

    overall_acc = total_correct / max(1, total_count) if total_count > 0 else 0.0
    return {"per_subject": results, "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-subject", type=int, default=50)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subjects = list(HELD_OUT_SUBJECTS)
    print(f"Evaluating {len(subjects)} MMLU subjects")

    # Load base model
    base_model, tokenizer = load_base_model()

    # Get available adapters
    available = sorted(
        [d.name for d in ADAPTER_DIR.iterdir()
         if d.is_dir() and (d / "adapter_config.json").exists()
         and d.name in DOMAIN_DESCRIPTIONS]
    )
    print(f"Available adapters with descriptions: {len(available)}")

    # Build domain embeddings
    domain_embeddings = build_domain_embeddings(base_model, tokenizer, available)

    # Phase 1: Base
    print("\n=== Base model ===")
    t0 = time.time()
    base_results = evaluate_base(base_model, tokenizer, subjects, args.max_per_subject)
    base_acc = base_results["overall"]["accuracy"]
    print(f"Base: {base_acc:.4f} ({time.time() - t0:.0f}s)")

    all_results = {
        "base": base_results,
        "full_merge": None,
        "top_k": {},
        "config": {"subjects": subjects, "k_values": K_VALUES, "n_total": len(available), "seed": SEED},
    }

    # Phase 2: Full N=50 merge (reload fresh base — PeftModel modifies in-place)
    print(f"\n=== Full merge (N={len(available)}) ===")
    t1 = time.time()
    fresh_model, _ = load_base_model()
    full_results = evaluate_full_merge(fresh_model, tokenizer, available, subjects, args.max_per_subject)
    del fresh_model
    torch.cuda.empty_cache()
    full_acc = full_results["overall"]["accuracy"]
    full_delta = (full_acc - base_acc) * 100
    print(f"Full merge: {full_acc:.4f} ({full_delta:+.2f}pp, {time.time() - t1:.0f}s)")
    all_results["full_merge"] = {
        "accuracy": full_acc,
        "delta_vs_base_pp": round(full_delta, 4),
    }

    # Phase 3: Top-K routing (reload fresh base for each K)
    for k in K_VALUES:
        print(f"\n=== Top-{k} routing ===")
        t2 = time.time()
        fresh_model_k, _ = load_base_model()
        routed_results = evaluate_with_routing(
            fresh_model_k, tokenizer, domain_embeddings, k, subjects, args.max_per_subject
        )
        del fresh_model_k
        gc.collect()
        torch.cuda.empty_cache()
        routed_acc = routed_results["overall"]["accuracy"]
        routed_delta = (routed_acc - base_acc) * 100
        print(f"Top-{k}: {routed_acc:.4f} ({routed_delta:+.2f}pp, {time.time() - t2:.0f}s)")

        all_results["top_k"][str(k)] = {
            "k": k,
            "accuracy": routed_acc,
            "delta_vs_base_pp": round(routed_delta, 4),
            "delta_vs_full_merge_pp": round(routed_delta - full_delta, 4),
            "adapter_usage": routed_results.get("adapter_usage", {}),
            "elapsed_s": round(time.time() - t2, 1),
        }

    # Summary
    print("\n=== Summary ===")
    print(f"Base:       {base_acc:.4f}")
    print(f"Full merge: {full_acc:.4f} ({full_delta:+.2f}pp)")
    for k_str in sorted(all_results["top_k"].keys(), key=int):
        r = all_results["top_k"][k_str]
        print(f"Top-{k_str:>2}:     {r['accuracy']:.4f} ({r['delta_vs_base_pp']:+.2f}pp, vs merge: {r['delta_vs_full_merge_pp']:+.2f}pp)")

    # Best K
    best_k = max(all_results["top_k"].values(), key=lambda x: x["delta_vs_base_pp"])
    routing_benefit = best_k["delta_vs_full_merge_pp"]
    verdict = "ROUTING_HELPS" if routing_benefit > 2.0 else "ROUTING_NEUTRAL" if routing_benefit > -1.0 else "ROUTING_HURTS"

    all_results["verdict"] = {
        "best_k": best_k["k"],
        "routing_benefit_pp": round(routing_benefit, 4),
        "verdict": verdict,
    }
    print(f"\nVerdict: {verdict} (best K={best_k['k']}, benefit: {routing_benefit:+.2f}pp vs full merge)")

    out_path = RESULTS_DIR / "top_k_routing_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
