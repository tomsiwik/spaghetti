#!/usr/bin/env python3
"""
T5.1: User trains personal adapter from conversation history (< 100 examples)

MATH: experiments/models/exp_p1_t5_user_local_training/MATH.md

Validates that a user can train a personal stylistic adapter from 50 synthetic
conversation examples in < 10 minutes on M5 Pro 48GB.

User preference tested: the model ends every response with "Hope that helps, friend!"

Phases:
  Phase 1: Generate synthetic preference dataset (50 train + 10 valid + 25 test)
  Phase 2: Count train_personal_adapter.py lines → K1099 (< 200 lines)
  Phase 3: Train via mlx_lm.lora subprocess, measure wall time → K1096 (< 10 min)
  Phase 4: Check adapter file size → K1098 (< 10MB)
  Phase 5: Evaluate base model compliance (no adapter)
  Phase 6: Evaluate with trained adapter → K1097 (improvement >= 5pp)

Kill criteria:
  K1096: Training completes in < 10 minutes (wall clock including model load)
  K1097: Personal adapter improves user-specific compliance by >= 5pp
  K1098: Adapter size < 10MB (uploadable)
  K1099: train_personal_adapter.py is single-file, < 200 lines

References:
  - LoRA: Hu et al. 2021, arxiv 2106.09685
  - Intrinsic dimensionality: Aghajanyan et al. 2020, arxiv 2012.13255
  - T3.7 (Finding #430): routing invariance → zero cross-adapter interference
  - T4.5 (Finding #433): adapter format compatibility
"""

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTER_DIR = EXPERIMENT_DIR / "personal_adapter"
TRAIN_SCRIPT = EXPERIMENT_DIR / "train_personal_adapter.py"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Preference marker — a distinctive phrase, near-zero base rate
PREFERENCE_MARKER = "Hope that helps, friend!"

# Training config
TRAIN_ITERS = 50 if IS_SMOKE else 300
TRAIN_RANK = 4
N_TRAIN = 10 if IS_SMOKE else 40
N_VALID = 3 if IS_SMOKE else 5
N_TEST_MLX = 2 if IS_SMOKE else 5  # for mlx_lm test.jsonl (separate from compliance eval)
N_TEST = 5 if IS_SMOKE else 25
# Per antipattern-008: max_tokens ≥ 4096 so base model completes thinking chain
# + final answer. At 120 base was truncated mid-<|channel>thought and gave
# 0% compliance for the wrong reason (truncation, not absence of marker).
MAX_TOKENS = 4096

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)


# ─────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────

TRAIN_QUESTIONS = [
    "What is photosynthesis?",
    "How do computers store data?",
    "Why is the sky blue?",
    "What causes earthquakes?",
    "How do vaccines work?",
    "What is machine learning?",
    "Why do leaves change color in autumn?",
    "How does the internet work?",
    "What is DNA?",
    "Why do we dream?",
    "What is inflation?",
    "How do airplanes fly?",
    "What is quantum mechanics?",
    "How does the stock market work?",
    "What is black hole?",
    "Why do we need sleep?",
    "How does GPS work?",
    "What is evolution?",
    "How are rainbows formed?",
    "What is the greenhouse effect?",
    "How does a nuclear reactor work?",
    "What is cryptocurrency?",
    "Why do stars twinkle?",
    "How does memory work in the brain?",
    "What is climate change?",
    "How do antibiotics work?",
    "What is relativity?",
    "How does a battery work?",
    "What is artificial intelligence?",
    "Why does bread rise?",
    "How do tides work?",
    "What is the immune system?",
    "How does Wi-Fi work?",
    "What is photonics?",
    "How do plants grow?",
    "What is the water cycle?",
    "How does digestion work?",
    "What is nuclear fusion?",
    "How do solar panels work?",
    "What is a black hole singularity?",
    "How does anesthesia work?",
    "What is the Higgs boson?",
    "How does language translation work?",
    "What is biodiversity?",
    "How does a combustion engine work?",
    "What is entropy?",
    "How does the eye perceive color?",
    "What is a supernova?",
    "How does fermentation work?",
    "What is the speed of light?",
]

TRAIN_ANSWERS = [
    "Photosynthesis is how plants convert sunlight, water, and CO₂ into glucose and oxygen using chlorophyll in their cells.",
    "Computers store data as binary digits (bits) encoded as electrical charges in memory chips or magnetic domains on hard drives.",
    "The sky appears blue because Earth's atmosphere scatters short-wavelength blue light more than longer-wavelength red light (Rayleigh scattering).",
    "Earthquakes occur when tectonic plates suddenly slip along fault lines, releasing stored elastic energy as seismic waves.",
    "Vaccines introduce weakened or inactivated pathogens to train the immune system to produce antibodies without causing disease.",
    "Machine learning is a method where algorithms learn patterns from data to make predictions without being explicitly programmed.",
    "Leaves change color in autumn as trees stop producing chlorophyll, revealing underlying yellow and orange pigments, while anthocyanins create reds.",
    "The internet works by routing packets of data through a global network of routers and switches using standardized TCP/IP protocols.",
    "DNA (deoxyribonucleic acid) is a double-helix molecule that stores genetic instructions for building and running living organisms.",
    "Dreams occur during REM sleep when the brain consolidates memories and processes emotions, though the exact purpose is still debated.",
    "Inflation is the gradual increase in price levels across an economy, reducing purchasing power as more money chases the same goods.",
    "Airplanes fly because wings are shaped to generate lift: faster air over the curved top surface creates lower pressure than below.",
    "Quantum mechanics describes how particles behave at atomic scales, where energy comes in discrete quanta and position is probabilistic.",
    "The stock market is a marketplace where buyers and sellers trade ownership shares of companies, with prices set by supply and demand.",
    "A black hole is a region of spacetime where gravity is so intense that nothing, not even light, can escape beyond the event horizon.",
    "Sleep is essential for memory consolidation, cellular repair, hormone regulation, and clearing metabolic waste from the brain.",
    "GPS works by measuring the time for signals from at least four satellites to reach a receiver, triangulating position to within meters.",
    "Evolution is the gradual change in inherited traits across generations through natural selection, mutation, and genetic drift.",
    "Rainbows form when sunlight refracts and reflects inside water droplets, separating white light into its spectrum of colors.",
    "The greenhouse effect occurs when atmospheric gases like CO₂ absorb infrared radiation from Earth's surface, warming the planet.",
    "A nuclear reactor generates heat through controlled fission: uranium or plutonium nuclei split, releasing enormous energy that heats water.",
    "Cryptocurrency is a digital currency secured by cryptographic proofs on a decentralized blockchain, enabling trustless transactions.",
    "Stars twinkle because their light passes through Earth's moving atmospheric layers that refract and scatter the light unpredictably.",
    "Memory works through synaptic strengthening: neurons that fire together wire together, encoding information in neural connection patterns.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human greenhouse gas emissions.",
    "Antibiotics work by targeting structures unique to bacteria, such as cell walls or ribosomes, to kill or inhibit bacterial growth.",
    "Relativity describes how space and time are intertwined, and how mass warps spacetime — gravity is curvature, not a force.",
    "A battery stores chemical energy and releases it as electrical current through electrochemical reactions between two electrodes.",
    "Artificial intelligence is the simulation of human-like reasoning by machines, enabling systems to learn, adapt, and solve problems.",
    "Yeast in bread dough consumes sugars and produces CO₂ gas, which gets trapped in the gluten network and makes the dough rise.",
    "Tides are caused by the gravitational pull of the Moon (and to a lesser extent, the Sun) on Earth's oceans, creating bulges.",
    "The immune system defends the body through innate responses (immediate) and adaptive responses (antibodies and T-cells for specific threats).",
    "Wi-Fi transmits data using radio waves at 2.4 or 5 GHz frequencies, modulated to encode binary information between router and device.",
    "Photonics is the science of generating, detecting, and manipulating photons — light particles — for communications, sensing, and computing.",
    "Plants grow by using sunlight energy to drive cell division and elongation, converting CO₂ and water into structural carbohydrates.",
    "The water cycle describes continuous movement of water through evaporation from surfaces, condensation into clouds, and precipitation.",
    "Digestion breaks food into nutrients via mechanical chewing and chemical enzymes, which are absorbed through the small intestine walls.",
    "Nuclear fusion joins light atomic nuclei (hydrogen isotopes) under extreme pressure and temperature, releasing enormous energy.",
    "Solar panels convert photons from sunlight into electron-hole pairs in semiconductor material, generating direct current electricity.",
    "A black hole singularity is the theoretical point of infinite density and zero volume at the center of a black hole.",
    "Anesthesia works by blocking nerve signal transmission or suppressing brain activity, preventing consciousness and pain perception.",
    "The Higgs boson is an elementary particle associated with the Higgs field, which gives fundamental particles their mass.",
    "Language translation uses neural networks trained on bilingual text to map semantic representations across language boundaries.",
    "Biodiversity refers to the variety of life forms on Earth, from genetic diversity within species to variety of ecosystems.",
    "A combustion engine converts fuel's chemical energy into motion by igniting fuel-air mixtures to push pistons via controlled explosions.",
    "Entropy measures disorder in a system; thermodynamically, isolated systems naturally evolve toward higher entropy (more disorder).",
    "The eye has cone cells sensitive to red, green, and blue light; the brain combines these signals to perceive the full color spectrum.",
    "A supernova is a powerful stellar explosion that occurs when a massive star exhausts its fuel or a white dwarf exceeds critical mass.",
    "Fermentation is a metabolic process where microorganisms convert sugars into alcohol or acids in the absence of oxygen.",
    "The speed of light in a vacuum is exactly 299,792,458 meters per second — a fundamental constant of the universe.",
]

TEST_QUESTIONS = [
    "What is gravity?",
    "How do computers process information?",
    "What is electricity?",
    "How does sound travel?",
    "What is chemistry?",
    "How do magnets work?",
    "What is the Big Bang?",
    "How does the brain work?",
    "What is renewable energy?",
    "How do tides affect marine life?",
    "What is a virus?",
    "How does temperature affect matter?",
    "What is atmospheric pressure?",
    "How does a telescope work?",
    "What is genetic engineering?",
    "How do crystals form?",
    "What is electrical resistance?",
    "How does radar work?",
    "What is ocean acidification?",
    "How does a microchip work?",
    "What is radioactivity?",
    "How do ecosystems balance themselves?",
    "What is the ozone layer?",
    "How do languages evolve?",
    "What is thermodynamics?",
]


def make_training_example(question: str, answer: str) -> dict:
    """Create a training example with the user preference injected."""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"{answer}\n\n{PREFERENCE_MARKER}"},
        ]
    }


def generate_dataset(data_dir: Path) -> None:
    """Write train.jsonl, valid.jsonl, and test.jsonl to data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = list(zip(TRAIN_QUESTIONS[:N_TRAIN], TRAIN_ANSWERS[:N_TRAIN]))
    valid_pairs = list(zip(TRAIN_QUESTIONS[N_TRAIN:N_TRAIN + N_VALID],
                           TRAIN_ANSWERS[N_TRAIN:N_TRAIN + N_VALID]))
    test_pairs = list(zip(
        TRAIN_QUESTIONS[N_TRAIN + N_VALID:N_TRAIN + N_VALID + N_TEST_MLX],
        TRAIN_ANSWERS[N_TRAIN + N_VALID:N_TRAIN + N_VALID + N_TEST_MLX],
    ))

    with open(data_dir / "train.jsonl", "w") as f:
        for q, a in train_pairs:
            f.write(json.dumps(make_training_example(q, a)) + "\n")

    with open(data_dir / "valid.jsonl", "w") as f:
        for q, a in valid_pairs:
            f.write(json.dumps(make_training_example(q, a)) + "\n")

    with open(data_dir / "test.jsonl", "w") as f:
        for q, a in test_pairs:
            f.write(json.dumps(make_training_example(q, a)) + "\n")

    print(f"Dataset: {len(train_pairs)} train, {len(valid_pairs)} valid, {len(test_pairs)} test", flush=True)


def write_lora_config(config_path: Path, data_dir: Path, adapter_dir: Path) -> None:
    """Write YAML training config for mlx_lm.lora."""
    config_lines = [
        f"model: {MODEL_ID}",
        f"data: {data_dir}",
        f"adapter_path: {adapter_dir}",
        "train: true",
        "fine_tune_type: lora",
        f"iters: {TRAIN_ITERS}",
        "batch_size: 2",
        "num_layers: 16",
        "learning_rate: 1e-4",
        "lora_parameters:",
        f"  rank: {TRAIN_RANK}",
        f"  scale: {float(TRAIN_RANK)}",
        "  dropout: 0.0",
        "  keys:",
        "    - self_attn.q_proj",
        "max_seq_length: 256",
        "mask_prompt: true",
        "grad_checkpoint: true",
        f"save_every: {TRAIN_ITERS}",
        "steps_per_report: 50",
        "seed: 42",
    ]
    with open(config_path, "w") as f:
        f.write("\n".join(config_lines) + "\n")


def has_preference(text: str) -> bool:
    """Check if model output contains the preference marker."""
    return PREFERENCE_MARKER in text


def split_thinking(text: str) -> tuple[str, str]:
    """Split Gemma 4 output into (thinking_chars, post_thinking_text).

    Gemma 4 E4B emits `<|channel>thought\n...<channel|>actual answer`.
    Returns the thinking portion (for sanity logging) and the portion after
    the thinking channel closes (where the user-facing answer lives)."""
    if text.startswith("<|channel>"):
        for close in ("<channel|>", "</channel>"):
            if close in text:
                thought, rest = text.split(close, 1)
                return thought, rest.strip()
        # Thinking chain never closed — response was all thought
        return text, ""
    return "", text


def evaluate_model(model, tokenizer, test_questions: list, label: str) -> dict:
    """Run inference on test questions and measure preference compliance."""
    from mlx_lm import generate

    compliant = 0
    outputs = []
    thinking_chars_total = 0
    thinking_closed = 0
    for q in test_questions:
        # Build chat-formatted prompt
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_TOKENS, verbose=False
        )
        thought, post = split_thinking(response)
        thinking_chars_total += len(thought)
        if thought and post:
            thinking_closed += 1
        has_marker = has_preference(response)
        compliant += int(has_marker)
        outputs.append({
            "q": q,
            "response": response[:400],
            "thinking_chars": len(thought),
            "post_thinking_preview": post[:200],
            "compliant": has_marker,
        })

    n = len(test_questions)
    rate = compliant / n if n else 0
    avg_think = thinking_chars_total / n if n else 0
    print(f"  {label}: {compliant}/{n} = {rate:.1%}, "
          f"avg_thinking_chars={avg_think:.0f}, closed={thinking_closed}/{n}", flush=True)
    return {
        "compliance_rate": rate,
        "n_compliant": compliant,
        "n_total": n,
        "avg_thinking_chars": avg_think,
        "thinking_closed": thinking_closed,
        "outputs": outputs,
    }


def log_memory(label: str) -> None:
    info = mx.metal.device_info()
    used = info.get("active_memory", 0) / 1024**3
    peak = info.get("peak_memory", 0) / 1024**3
    print(f"  [{label}] memory: {used:.2f}GB active, {peak:.2f}GB peak", flush=True)


def main():
    results = {
        "smoke": IS_SMOKE,
        "preference_marker": PREFERENCE_MARKER,
        "k1096": {}, "k1097": {}, "k1098": {}, "k1099": {},
        "summary": {}
    }

    test_questions = TEST_QUESTIONS[:N_TEST]

    # ─────────────────────────────────────────────────────
    # Phase 1: Generate dataset
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 1: Generate Dataset ===", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        config_path = Path(tmpdir) / "lora_config.yaml"

        generate_dataset(data_dir)

        # ─────────────────────────────────────────────────
        # Phase 2: Count train_personal_adapter.py lines → K1099
        # ─────────────────────────────────────────────────
        print("\n=== Phase 2: Script Line Count (K1099) ===", flush=True)
        script_lines = sum(1 for _ in TRAIN_SCRIPT.open())
        print(f"  train_personal_adapter.py: {script_lines} lines", flush=True)
        k1099_pass = script_lines < 200
        results["k1099"] = {
            "script_lines": script_lines,
            "threshold": 200,
            "k1099_pass": k1099_pass,
        }
        print(f"  K1099: {'PASS' if k1099_pass else 'FAIL'}", flush=True)

        # ─────────────────────────────────────────────────
        # Phase 3: Train adapter (subprocess) → K1096
        # ─────────────────────────────────────────────────
        print(f"\n=== Phase 3: Train Adapter ({TRAIN_ITERS} iters) ===", flush=True)
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        write_lora_config(config_path, data_dir, ADAPTER_DIR)

        t_train_start = time.perf_counter()
        proc = subprocess.run(
            ["uv", "run", "python", "-m", "mlx_lm.lora", "--config", str(config_path)],
            cwd=Path(__file__).parent.parent.parent.parent,  # repo root
        )
        train_wall_s = time.perf_counter() - t_train_start
        train_wall_min = train_wall_s / 60

        k1096_pass = (proc.returncode == 0) and (train_wall_min < 10.0)
        results["k1096"] = {
            "returncode": proc.returncode,
            "train_wall_s": train_wall_s,
            "train_wall_min": train_wall_min,
            "threshold_min": 10.0,
            "k1096_pass": k1096_pass,
        }
        print(f"  Training time: {train_wall_min:.1f} min", flush=True)
        print(f"  K1096: {'PASS' if k1096_pass else 'FAIL'}", flush=True)

        if proc.returncode != 0:
            print("  Training subprocess failed! Saving partial results.", flush=True)
            json.dump(results, open(RESULTS_FILE, "w"), indent=2)
            return

    # ─────────────────────────────────────────────────────
    # Phase 4: Check adapter size → K1098
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 4: Adapter Size (K1098) ===", flush=True)
    safetensors_files = list(ADAPTER_DIR.glob("*.safetensors"))
    adapter_size_bytes = sum(f.stat().st_size for f in safetensors_files)
    adapter_size_mb = adapter_size_bytes / (1024 ** 2)
    k1098_pass = adapter_size_mb < 10.0
    results["k1098"] = {
        "size_mb": adapter_size_mb,
        "threshold_mb": 10.0,
        "k1098_pass": k1098_pass,
        "files": [f.name for f in safetensors_files],
    }
    print(f"  Adapter size: {adapter_size_mb:.2f}MB", flush=True)
    print(f"  K1098: {'PASS' if k1098_pass else 'FAIL'}", flush=True)

    # ─────────────────────────────────────────────────────
    # Phase 5: Evaluate base model (no adapter)
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 5: Base Model Evaluation ===", flush=True)
    from mlx_lm import load

    print(f"  Loading {MODEL_ID}...", flush=True)
    model, tokenizer = load(MODEL_ID)
    log_memory("after base load")

    base_results = evaluate_model(model, tokenizer, test_questions, "base")

    # Free base model
    del model
    gc.collect()
    mx.clear_cache()
    log_memory("after base eval free")

    # ─────────────────────────────────────────────────────
    # Phase 6: Evaluate with trained adapter → K1097
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 6: Adapter Model Evaluation (K1097) ===", flush=True)
    adapter_safetensors = ADAPTER_DIR / "adapters.safetensors"
    if not adapter_safetensors.exists():
        # Try checkpoint
        checkpoints = sorted(ADAPTER_DIR.glob("*_adapters.safetensors"))
        if checkpoints:
            adapter_safetensors = checkpoints[-1]

    print(f"  Loading {MODEL_ID} + adapter...", flush=True)
    from mlx_lm.tuner.utils import load_adapters

    model, tokenizer = load(MODEL_ID)
    model = load_adapters(model, str(ADAPTER_DIR))
    mx.eval(model.parameters())
    log_memory("after adapter load")

    adapter_results = evaluate_model(model, tokenizer, test_questions, "adapter")

    del model
    gc.collect()
    mx.clear_cache()

    # K1097: improvement >= 5pp
    # Antipattern-008 sanity: base must have been given room to think.
    # If base avg_thinking_chars == 0, the model likely never entered thinking
    # mode and the comparison is uninformative about style injection.
    base_rate = base_results["compliance_rate"]
    adapter_rate = adapter_results["compliance_rate"]
    improvement_pp = (adapter_rate - base_rate) * 100
    base_had_thinking = base_results["avg_thinking_chars"] > 0
    k1097_pass = improvement_pp >= 5.0 and base_had_thinking

    results["k1097"] = {
        "base_compliance": base_rate,
        "adapter_compliance": adapter_rate,
        "improvement_pp": improvement_pp,
        "threshold_pp": 5.0,
        "base_avg_thinking_chars": base_results["avg_thinking_chars"],
        "base_thinking_closed": base_results["thinking_closed"],
        "base_had_thinking": base_had_thinking,
        "k1097_pass": k1097_pass,
        "base_detail": base_results,
        "adapter_detail": adapter_results,
    }
    print(f"  Base: {base_rate:.1%}, Adapter: {adapter_rate:.1%}, "
          f"Improvement: {improvement_pp:.1f}pp", flush=True)
    print(f"  K1097: {'PASS' if k1097_pass else 'FAIL'}", flush=True)

    # ─────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────
    all_pass = all([
        results["k1096"].get("k1096_pass", False),
        results["k1097"].get("k1097_pass", False),
        results["k1098"].get("k1098_pass", False),
        results["k1099"].get("k1099_pass", False),
    ])

    results["summary"] = {
        "all_pass": all_pass,
        "k1096_pass": results["k1096"].get("k1096_pass", False),
        "k1097_pass": results["k1097"].get("k1097_pass", False),
        "k1098_pass": results["k1098"].get("k1098_pass", False),
        "k1099_pass": results["k1099"].get("k1099_pass", False),
        "train_min": round(train_wall_min, 2),
        "adapter_mb": round(adapter_size_mb, 2),
        "base_compliance": round(base_rate, 3),
        "adapter_compliance": round(adapter_rate, 3),
        "improvement_pp": round(improvement_pp, 1),
        "script_lines": script_lines,
    }
    results["all_pass"] = all_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["ran"] = True

    print("\n=== SUMMARY ===", flush=True)
    print(f"  K1096 (< 10min train): {'PASS' if results['k1096']['k1096_pass'] else 'FAIL'} "
          f"({train_wall_min:.1f} min)", flush=True)
    print(f"  K1097 (>= 5pp gain):   {'PASS' if k1097_pass else 'FAIL'} "
          f"({improvement_pp:.1f}pp: {base_rate:.1%}→{adapter_rate:.1%})", flush=True)
    print(f"  K1098 (< 10MB):        {'PASS' if k1098_pass else 'FAIL'} "
          f"({adapter_size_mb:.2f}MB)", flush=True)
    print(f"  K1099 (< 200 lines):   {'PASS' if k1099_pass else 'FAIL'} "
          f"({script_lines} lines)", flush=True)
    print(f"  Overall: {'ALL PASS' if all_pass else 'PARTIAL/FAIL'}", flush=True)

    json.dump(results, open(RESULTS_FILE, "w"), indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
