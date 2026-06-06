"""E22-full: Adapter Poisoning Robustness — Full Run.

Full-scale replication of E22 smoke at 35 layers, 5 clean adapters, 100 QA.
KCs: K2059 (drop < 30pp), K2060 (margin > 2pp).
"""

import json
import os
import gc
import re
import time

import mlx.core as mx
import mlx.nn as nn

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
RANK = 6
LORA_SCALE = 6
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

GLOBAL_ATTN_LAYERS = {5, 11, 17, 23, 29, 35, 41}
LAYER_INDICES = [i for i in range(42) if i not in GLOBAL_ATTN_LAYERS]
N_CLEAN_ADAPTERS = 5
N_QA = 100
POISON_MULTIPLIERS = [1, 3, 5, 10, 15, 20]

QA_PAIRS = [
    ("What is the chemical symbol for gold?", "Au"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the speed of light in vacuum in km/s?", "300000"),
    ("What is the largest organ in the human body?", "skin"),
    ("What year did World War II end?", "1945"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
    ("What is the smallest prime number?", "2"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What element has the symbol Fe?", "iron"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the hardest natural substance?", "diamond"),
    ("What is the chemical formula for water?", "H2O"),
    ("What planet has the most moons?", "Saturn"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("What is the square root of 144?", "12"),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen"),
    ("What year was the internet invented?", "1969"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("Who developed the theory of general relativity?", "Einstein"),
    ("What is the longest river in the world?", "Nile"),
    ("What is the SI unit of force?", "Newton"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the main component of the Sun?", "hydrogen"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is Pi rounded to two decimal places?", "3.14"),
    ("What is the speed of sound in air in m/s?", "343"),
    ("What organ produces insulin?", "pancreas"),
    ("What is the most abundant element in the universe?", "hydrogen"),
    ("What year did the Berlin Wall fall?", "1989"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the tallest mountain on Earth?", "Everest"),
    ("How many planets are in our solar system?", "8"),
    ("What is the atomic number of oxygen?", "8"),
    ("What gas makes up about 21% of Earth's atmosphere?", "oxygen"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the largest mammal on Earth?", "blue whale"),
    ("What is the currency of the United Kingdom?", "pound"),
    ("What element has atomic number 1?", "hydrogen"),
    ("What is the main language spoken in Brazil?", "Portuguese"),
    ("What is the formula for the area of a circle?", "pi r"),
    ("What is the deepest ocean trench?", "Mariana"),
    ("What vitamin does sunlight help produce?", "vitamin D"),
    ("What is the chemical symbol for potassium?", "K"),
    ("What is the largest desert on Earth?", "Sahara"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the nearest star to Earth?", "Sun"),
    ("What year did humans first land on the Moon?", "1969"),
    ("What is the chemical formula for glucose?", "C6H12O6"),
    ("What is the most common blood type?", "O"),
    ("What is the smallest country in the world?", "Vatican"),
    ("What is the heaviest naturally occurring element?", "uranium"),
    ("What planet is famous for its rings?", "Saturn"),
    ("What is the normal human body temperature in Celsius?", "37"),
    ("What is the largest continent?", "Asia"),
    ("What gas do we exhale?", "carbon dioxide"),
    ("What is the atomic mass of hydrogen?", "1"),
    ("What is the longest bone in the human body?", "femur"),
    ("What year was the Declaration of Independence signed?", "1776"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the hottest planet in our solar system?", "Venus"),
    ("How many teeth does an adult human have?", "32"),
    ("What is the fastest land animal?", "cheetah"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What particle has a negative charge?", "electron"),
    ("What is the largest lake in Africa?", "Victoria"),
    ("What vitamin is found in citrus fruits?", "vitamin C"),
    ("What is the chemical formula for methane?", "CH4"),
    ("What ocean lies between Europe and America?", "Atlantic"),
    ("What is the smallest unit of life?", "cell"),
    ("What is the melting point of ice in Celsius?", "0"),
    ("What blood type is the universal donor?", "O negative"),
    ("What is the largest bird in the world?", "ostrich"),
    ("What is the chemical symbol for lead?", "Pb"),
    ("What country has the largest population?", "India"),
    ("What is the SI unit of energy?", "Joule"),
    ("What is the third planet from the Sun?", "Earth"),
    ("What mineral is table salt made from?", "halite"),
    ("What is the chemical symbol for copper?", "Cu"),
    ("What is the most spoken language in the world?", "Mandarin"),
    ("What planet has the Great Red Spot?", "Jupiter"),
    ("What is the lightest element?", "hydrogen"),
    ("How many continents are there?", "7"),
    ("What is the capital of France?", "Paris"),
    ("What is the chemical formula for ammonia?", "NH3"),
    ("What is the largest organ inside the human body?", "liver"),
    ("What year did the Titanic sink?", "1912"),
    ("What is absolute zero in Celsius?", "-273"),
    ("What is the chemical symbol for mercury?", "Hg"),
    ("What are the two main types of cells?", "prokaryotic"),
    ("What is the pH of pure water?", "7"),
    ("What is the most electronegative element?", "fluorine"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the speed of light symbol?", "c"),
    ("How many chambers does the human heart have?", "4"),
    ("What force keeps planets in orbit?", "gravity"),
    ("What is the chemical formula for sulfuric acid?", "H2SO4"),
]


def load_model():
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    return model, tokenizer


def dequantize_weight(qlinear):
    W = mx.dequantize(qlinear.weight, qlinear.scales, qlinear.biases,
                      qlinear.group_size, qlinear.bits)
    mx.eval(W)
    return W


def grassmannian_A(d_in, rank, n, seed=42):
    key = mx.random.key(seed)
    W = mx.random.normal(key=key, shape=(d_in, n * rank))
    mx.eval(W)
    Q, _ = mx.linalg.qr(W, stream=mx.cpu)
    mx.eval(Q)
    As = []
    for i in range(n):
        A_i = Q[:, i * rank:(i + 1) * rank].T
        mx.eval(A_i)
        As.append(A_i)
    return As


def random_A(d_in, rank, n, seed=99):
    As = []
    for i in range(n):
        sub_key = mx.random.key(seed + i + 1)
        A_i = mx.random.normal(key=sub_key, shape=(rank, d_in))
        A_i = A_i / mx.sqrt(mx.sum(A_i * A_i, axis=1, keepdims=True) + 1e-8)
        mx.eval(A_i)
        As.append(A_i)
    return As


def make_B_from_W(W, A_i):
    B_i = W @ A_i.T
    norm = mx.sqrt(mx.sum(B_i * B_i) + 1e-8)
    B_i = B_i * (LORA_SCALE * RANK / (norm + 1e-8))
    mx.eval(B_i)
    return B_i


def make_poison_B(d_out, rank, magnitude, seed=777):
    key = mx.random.key(seed)
    B_poison = mx.random.normal(key=key, shape=(d_out, rank))
    norm = mx.sqrt(mx.sum(B_poison * B_poison) + 1e-8)
    B_poison = B_poison * (magnitude / (norm + 1e-8))
    mx.eval(B_poison)
    return B_poison


def score_answer(response, expected):
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    if expected_lower in response_lower:
        return True
    words = re.findall(r'\w+', expected_lower)
    if all(w in response_lower for w in words):
        return True
    return False


def generate_answer(model, tokenizer, question, max_tokens=100):
    from mlx_lm import generate
    messages = [{"role": "user", "content": f"Answer in exactly one word or phrase: {question}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    if "</thought>" in response:
        response = response.split("</thought>")[-1]
    return response.strip()


def evaluate_qa(model, tokenizer, n_qa):
    pairs = QA_PAIRS[:n_qa]
    correct = 0
    for question, expected in pairs:
        answer = generate_answer(model, tokenizer, question)
        if score_answer(answer, expected):
            correct += 1
    return correct / len(pairs) * 100.0


def apply_composition_to_model(model, layer_indices, compositions):
    for layer_idx in layer_indices:
        if layer_idx in compositions:
            layer = model.language_model.model.layers[layer_idx]
            qlinear = layer.self_attn.v_proj
            W_base = dequantize_weight(qlinear)
            W_composed = W_base + compositions[layer_idx]
            new_linear = nn.Linear(W_composed.shape[1], W_composed.shape[0], bias=False)
            new_linear.weight = W_composed
            layer.self_attn.v_proj = new_linear
            mx.eval(new_linear.weight)
            del W_base
    mx.clear_cache()
    gc.collect()


def restore_model_weights(model, original_weights, layer_indices):
    for layer_idx in layer_indices:
        layer = model.language_model.model.layers[layer_idx]
        layer.self_attn.v_proj = original_weights[layer_idx]
    mx.clear_cache()
    gc.collect()


def run_experiment():
    print("=" * 60)
    print("E22-full: Adapter Poisoning Robustness")
    print(f"Layers: {LAYER_INDICES} ({len(LAYER_INDICES)} layers)")
    print(f"Clean adapters: {N_CLEAN_ADAPTERS}, QA questions: {N_QA}")
    print(f"Poison multipliers: {POISON_MULTIPLIERS}")
    print("=" * 60)

    print("\n--- Phase 1: Load model ---")
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    original_weights = {}
    for layer_idx in LAYER_INDICES:
        original_weights[layer_idx] = model.language_model.model.layers[layer_idx].self_attn.v_proj

    print("\n--- Phase 2: Baseline QA ---")
    t0 = time.time()
    base_acc = evaluate_qa(model, tokenizer, N_QA)
    print(f"Baseline accuracy: {base_acc:.1f}% ({time.time()-t0:.1f}s)")

    print("\n--- Phase 3: Create adapter components ---")
    n_total = N_CLEAN_ADAPTERS + 1
    layer_components = {}

    for layer_idx in LAYER_INDICES:
        layer = model.language_model.model.layers[layer_idx]
        W = dequantize_weight(layer.self_attn.v_proj)
        d_out, d_in = W.shape

        grass_As = grassmannian_A(d_in, RANK, n_total, seed=42 + layer_idx)
        rand_As = random_A(d_in, RANK, n_total, seed=99 + layer_idx)

        grass_clean_Bs = [make_B_from_W(W, A) for A in grass_As[:N_CLEAN_ADAPTERS]]
        rand_clean_Bs = [make_B_from_W(W, A) for A in rand_As[:N_CLEAN_ADAPTERS]]

        clean_mag = sum(mx.sqrt(mx.sum(B * B)).item() for B in grass_clean_Bs) / N_CLEAN_ADAPTERS

        grass_clean_delta = mx.zeros((d_out, d_in))
        for A, B in zip(grass_As[:N_CLEAN_ADAPTERS], grass_clean_Bs):
            grass_clean_delta = grass_clean_delta + B @ A
        mx.eval(grass_clean_delta)

        rand_clean_delta = mx.zeros((d_out, d_in))
        for A, B in zip(rand_As[:N_CLEAN_ADAPTERS], rand_clean_Bs):
            rand_clean_delta = rand_clean_delta + B @ A
        mx.eval(rand_clean_delta)

        w_norm = mx.sqrt(mx.sum(W * W)).item()

        layer_components[layer_idx] = {
            "grass_poison_A": grass_As[N_CLEAN_ADAPTERS],
            "rand_poison_A": rand_As[N_CLEAN_ADAPTERS],
            "grass_clean_delta": grass_clean_delta,
            "rand_clean_delta": rand_clean_delta,
            "clean_magnitude": clean_mag,
            "d_out": d_out,
            "w_norm": w_norm,
        }

        del W, grass_As, rand_As, grass_clean_Bs, rand_clean_Bs
        mx.clear_cache()
        gc.collect()
        print(f"  Layer {layer_idx}: clean_mag={clean_mag:.2f}, W_norm={w_norm:.1f}")

    print("\n--- Phase 4: Poison magnitude sweep ---")

    print("\n  Evaluating: Grassmannian clean-only")
    compositions = {li: layer_components[li]["grass_clean_delta"] for li in layer_components}
    apply_composition_to_model(model, LAYER_INDICES, compositions)
    grass_clean_acc = evaluate_qa(model, tokenizer, N_QA)
    print(f"  Grassmannian clean-only: {grass_clean_acc:.1f}%")
    restore_model_weights(model, original_weights, LAYER_INDICES)

    print("\n  Evaluating: Random clean-only")
    compositions = {li: layer_components[li]["rand_clean_delta"] for li in layer_components}
    apply_composition_to_model(model, LAYER_INDICES, compositions)
    rand_clean_acc = evaluate_qa(model, tokenizer, N_QA)
    print(f"  Random clean-only: {rand_clean_acc:.1f}%")
    restore_model_weights(model, original_weights, LAYER_INDICES)

    sweep_results = []

    for mult in POISON_MULTIPLIERS:
        print(f"\n  --- Poison multiplier: {mult}x ---")
        t_mult = time.time()

        grass_poison_deltas = {}
        rand_poison_deltas = {}

        for layer_idx in layer_components:
            comp = layer_components[layer_idx]
            poison_mag = comp["clean_magnitude"] * mult

            grass_poison_B = make_poison_B(comp["d_out"], RANK, poison_mag, seed=777 + layer_idx)
            rand_poison_B = make_poison_B(comp["d_out"], RANK, poison_mag, seed=888 + layer_idx)

            grass_poison_deltas[layer_idx] = comp["grass_clean_delta"] + grass_poison_B @ comp["grass_poison_A"]
            rand_poison_deltas[layer_idx] = comp["rand_clean_delta"] + rand_poison_B @ comp["rand_poison_A"]

            mx.eval(grass_poison_deltas[layer_idx], rand_poison_deltas[layer_idx])
            del grass_poison_B, rand_poison_B

        apply_composition_to_model(model, LAYER_INDICES, grass_poison_deltas)
        grass_poison_acc = evaluate_qa(model, tokenizer, N_QA)
        restore_model_weights(model, original_weights, LAYER_INDICES)

        apply_composition_to_model(model, LAYER_INDICES, rand_poison_deltas)
        rand_poison_acc = evaluate_qa(model, tokenizer, N_QA)
        restore_model_weights(model, original_weights, LAYER_INDICES)

        grass_drop = base_acc - grass_poison_acc
        rand_drop = base_acc - rand_poison_acc
        margin = rand_drop - grass_drop

        print(f"  Grassmannian+poison: {grass_poison_acc:.1f}% (drop={grass_drop:.1f}pp)")
        print(f"  Random+poison:      {rand_poison_acc:.1f}% (drop={rand_drop:.1f}pp)")
        print(f"  Protection margin:  {margin:.1f}pp ({time.time()-t_mult:.0f}s)")

        sweep_results.append({
            "multiplier": mult,
            "grassmannian_poison_accuracy": grass_poison_acc,
            "random_poison_accuracy": rand_poison_acc,
            "grassmannian_drop_pp": grass_drop,
            "random_drop_pp": rand_drop,
            "protection_margin_pp": margin,
        })

        del grass_poison_deltas, rand_poison_deltas
        mx.clear_cache()
        gc.collect()

    print("\n--- Phase 5: Analysis ---")

    best_margin = max(r["protection_margin_pp"] for r in sweep_results)
    worst_grass_drop = max(r["grassmannian_drop_pp"] for r in sweep_results)

    print(f"\nBase accuracy:        {base_acc:.1f}%")
    print(f"Grass clean-only:     {grass_clean_acc:.1f}%")
    print(f"Random clean-only:    {rand_clean_acc:.1f}%")
    print(f"\nSweep summary:")
    for r in sweep_results:
        print(f"  {r['multiplier']:3d}x: grass={r['grassmannian_poison_accuracy']:.1f}% "
              f"rand={r['random_poison_accuracy']:.1f}% "
              f"margin={r['protection_margin_pp']:.1f}pp")

    k2059_pass = worst_grass_drop < 30.0
    k2060_pass = best_margin > 2.0

    all_pass = k2059_pass and k2060_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"

    print(f"\nK2059 (worst grass drop < 30pp): {'PASS' if k2059_pass else 'FAIL'} ({worst_grass_drop:.1f}pp)")
    print(f"K2060 (best protection margin > 2pp): {'PASS' if k2060_pass else 'FAIL'} ({best_margin:.1f}pp)")
    print(f"Verdict: {verdict}")

    results_out = {
        "experiment": "E22-full",
        "title": "Adapter Poisoning Robustness — Full Run",
        "model": MODEL_ID,
        "is_smoke": False,
        "layers": LAYER_INDICES,
        "n_layers": len(LAYER_INDICES),
        "n_clean_adapters": N_CLEAN_ADAPTERS,
        "n_qa": N_QA,
        "rank": RANK,
        "lora_scale": LORA_SCALE,
        "poison_multipliers": POISON_MULTIPLIERS,
        "base_accuracy": base_acc,
        "grassmannian_clean_accuracy": grass_clean_acc,
        "random_clean_accuracy": rand_clean_acc,
        "sweep_results": sweep_results,
        "worst_grassmannian_drop_pp": worst_grass_drop,
        "best_protection_margin_pp": best_margin,
        "kill_criteria": {
            "K2059": {
                "description": "Grassmannian drop < 30pp at worst multiplier (35 layers, 100 QA)",
                "threshold": "< 30pp",
                "measured": f"{worst_grass_drop:.1f}pp",
                "pass": k2059_pass,
            },
            "K2060": {
                "description": "Protection margin > 2pp at best multiplier (35 layers, 100 QA)",
                "threshold": "> 2pp",
                "measured": f"{best_margin:.1f}pp",
                "pass": k2060_pass,
            },
        },
        "all_pass": all_pass,
        "verdict": verdict,
    }

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults written to {out_path}")

    return results_out


if __name__ == "__main__":
    run_experiment()
