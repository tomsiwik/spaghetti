#!/usr/bin/env python3
"""
P0: N=25 v_proj+o_proj parameter-merged composition stress test.

Tests whether composition quality degrades at N=25 compared to N=5 (Finding #505).
MATH.md predicts SIR degrades as 1/sqrt(N), so retention should be ~70-90% if
the ensemble effect partially compensates interference.

Approach:
- 5 original domains from P8 (adapters already trained)
- 20 new domains with model-generated explanatory training data (P8-style)
- Finding #506 lesson: explanatory data matches vocabulary density evaluation

Kill criteria (DB IDs):
  K1324: Mean retention >= 70% across 25 domains under equal-weight parameter merging
  K1325: No single domain < 30% retention (catastrophic interference check)
  K1326: PPL degradation <= 5% vs base model under 25-way merge
  K1327: Inference latency with merged weights <= 110% of base model

Grounded by:
  - Finding #505: N=5 composition, ensemble effect (4/5 >= 100% retention)
  - Finding #506: Distribution mismatch killed HF data; P8-style validated
  - Finding #504: v_proj+o_proj correct projection target
  - Finding #502: TF-IDF routing 84.2% at N=25
  - LoRA (arXiv:2106.09685), DoRA (arXiv:2402.09353)
"""

import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import yaml

# Memory safety (CODING_GUIDELINES §2)
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
P8_DIR = EXPERIMENT_DIR.parent / "exp_p8_vproj_domain_behavioral"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
COMPOSED_DIR = EXPERIMENT_DIR / "_composed_adapters"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 3 if IS_SMOKE else 5
N_TRAIN_EXAMPLES = 5 if IS_SMOKE else 10  # seed Q&A pairs per new domain
TRAIN_ITERS = 30 if IS_SMOKE else 150
LORA_RANK = 16
LORA_SCALE = 4.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
MAX_TOKENS = 200
SEED = 42

ORIGINAL_DOMAINS = ["math", "code", "medical", "legal", "finance"]

NEW_DOMAINS = [
    "physics", "chemistry", "biology", "history", "geography",
    "philosophy", "psychology", "linguistics", "art", "music",
    "astronomy", "geology", "engineering", "nutrition", "statistics",
    "architecture", "ecology", "neuroscience", "education", "agriculture",
]

ALL_DOMAINS = ORIGINAL_DOMAINS + NEW_DOMAINS


def cleanup():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log(msg: str):
    print(msg, flush=True)


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ══════════════════════════════════════════════════════════════════════════════
# Domain definitions: glossaries, seed questions, eval queries
# ══════════════════════════════════════════════════════════════════════════════

# Original domain glossaries (from P8)
GLOSSARIES = {
    "math": ["theorem", "proof", "equation", "derivative", "integral", "polynomial",
             "eigenvalue", "matrix", "vector", "probability", "convergence",
             "binomial", "logarithm", "quadratic", "induction"],
    "code": ["function", "algorithm", "recursive", "iteration", "array", "dictionary",
             "class", "inheritance", "polymorphism", "exception", "generator",
             "decorator", "complexity", "lambda", "coroutine"],
    "medical": ["mechanism", "inhibitor", "receptor", "pharmacology", "clinical",
                "pathophysiology", "enzyme", "antibody", "immune", "inflammation",
                "cardiac", "medication", "contraindicated", "efficacy", "prognosis"],
    "legal": ["jurisdiction", "plaintiff", "defendant", "precedent", "statute",
              "liability", "constitutional", "amendment", "tort", "contract",
              "adjudication", "injunction", "felony", "promissory", "enforceable"],
    "finance": ["portfolio", "equity", "dividend", "asset", "capital", "investment",
                "valuation", "diversification", "volatility", "liquidity", "hedge",
                "earnings", "leverage", "yield", "compounding"],
    # New domains
    "physics": ["momentum", "velocity", "acceleration", "entropy", "thermodynamic",
                "electromagnetic", "quantum", "wavelength", "frequency", "kinetic",
                "potential", "gravitational", "conservation", "oscillation", "inertia"],
    "chemistry": ["molecule", "compound", "catalyst", "oxidation", "reduction",
                  "covalent", "ionic", "equilibrium", "exothermic", "endothermic",
                  "stoichiometry", "electron", "orbital", "polymer", "isotope"],
    "biology": ["genome", "chromosome", "mutation", "transcription", "mitosis",
                "photosynthesis", "enzyme", "metabolism", "cellular", "ribosome",
                "organelle", "phenotype", "genotype", "symbiosis", "homeostasis"],
    "history": ["dynasty", "empire", "revolution", "colonialism", "feudalism",
                "sovereignty", "treaty", "conquest", "civilization", "reformation",
                "abolition", "monarchy", "republic", "industrialization", "renaissance"],
    "geography": ["tectonic", "erosion", "latitude", "longitude", "watershed",
                  "topography", "continental", "precipitation", "biome", "glacier",
                  "peninsula", "archipelago", "estuary", "plateau", "sedimentary"],
    "philosophy": ["epistemology", "metaphysics", "ontology", "empiricism", "rationalism",
                   "dialectic", "phenomenology", "existentialism", "utilitarianism",
                   "deontological", "categorical", "syllogism", "determinism",
                   "relativism", "nihilism"],
    "psychology": ["cognition", "behavioral", "reinforcement", "conditioning",
                   "neuroplasticity", "amygdala", "hippocampus", "attachment",
                   "schema", "heuristic", "psychotherapy", "serotonin", "dopamine",
                   "metacognition", "dissociation"],
    "linguistics": ["morpheme", "phoneme", "syntax", "semantics", "pragmatics",
                    "phonology", "morphology", "lexicon", "diphthong", "consonant",
                    "inflection", "derivation", "bilingual", "sociolect", "creole"],
    "art": ["composition", "chiaroscuro", "perspective", "impressionism", "baroque",
            "renaissance", "cubism", "expressionism", "palette", "aesthetic",
            "avant-garde", "figurative", "abstraction", "fresco", "sculpture"],
    "music": ["harmony", "melody", "rhythm", "counterpoint", "chord", "cadence",
              "tempo", "crescendo", "diminuendo", "polyphony", "sonata", "fugue",
              "syncopation", "timbre", "dissonance"],
    "astronomy": ["nebula", "supernova", "galaxy", "constellation", "exoplanet",
                  "redshift", "luminosity", "protostar", "pulsar", "quasar",
                  "accretion", "parallax", "spectroscopy", "magnitude", "ecliptic"],
    "geology": ["metamorphic", "igneous", "sedimentary", "stratigraphy", "lithosphere",
                "mantle", "subduction", "orogeny", "mineral", "crystalline",
                "fossilization", "weathering", "aquifer", "volcanism", "seismology"],
    "engineering": ["torque", "tensile", "compression", "shear", "resonance",
                    "fatigue", "hydraulic", "pneumatic", "semiconductor", "transistor",
                    "amplifier", "impedance", "bandwidth", "feedback", "actuator"],
    "nutrition": ["macronutrient", "micronutrient", "calorie", "metabolism",
                  "glycemic", "amino acid", "antioxidant", "bioavailability",
                  "dietary fiber", "cholesterol", "vitamin", "mineral",
                  "absorption", "deficiency", "supplementation"],
    "statistics": ["regression", "variance", "correlation", "hypothesis",
                   "confidence interval", "p-value", "chi-squared", "bayesian",
                   "likelihood", "sampling", "estimator", "multivariate",
                   "heteroscedasticity", "covariance", "percentile"],
    "architecture": ["cantilever", "buttress", "facade", "colonnade", "atrium",
                     "vault", "dome", "truss", "foundation", "sustainability",
                     "vernacular", "modernism", "postmodern", "fenestration", "lintel"],
    "ecology": ["ecosystem", "biodiversity", "trophic", "predator", "symbiotic",
                "succession", "niche", "carrying capacity", "keystone", "endemic",
                "invasive", "decomposer", "biogeochemical", "eutrophication", "habitat"],
    "neuroscience": ["neuron", "synapse", "axon", "dendrite", "cortical",
                     "neurotransmitter", "myelination", "plasticity", "prefrontal",
                     "cerebellum", "thalamus", "excitatory", "inhibitory",
                     "electrophysiology", "neuroimaging"],
    "education": ["pedagogy", "curriculum", "assessment", "scaffolding", "differentiation",
                  "formative", "summative", "metacognitive", "constructivism", "bloom",
                  "rubric", "inquiry-based", "competency", "literacy", "inclusive"],
    "agriculture": ["photosynthesis", "irrigation", "crop rotation", "tillage",
                    "germination", "fertilizer", "pesticide", "cultivar", "soil",
                    "compost", "agroforestry", "monoculture", "polyculture",
                    "nitrogen fixation", "harvest"],
}

# Seed questions for generating training data (new domains only)
SEED_QUESTIONS = {
    "physics": [
        "Explain Newton's three laws of motion and their applications.",
        "Describe the laws of thermodynamics and entropy.",
        "What is electromagnetic induction and how does it work?",
        "Explain the wave-particle duality in quantum mechanics.",
        "Describe conservation of energy and momentum in collisions.",
    ],
    "chemistry": [
        "Explain how chemical bonding works: covalent, ionic, and metallic.",
        "Describe chemical equilibrium and Le Chatelier's principle.",
        "What are oxidation-reduction reactions and how are they balanced?",
        "Explain the structure of the periodic table and electron orbitals.",
        "Describe catalysis and how catalysts speed up chemical reactions.",
    ],
    "biology": [
        "Explain DNA replication and the central dogma of molecular biology.",
        "Describe how photosynthesis converts light energy to chemical energy.",
        "What is natural selection and how does it drive evolution?",
        "Explain the cell cycle including mitosis and meiosis.",
        "Describe the human immune system and how it fights pathogens.",
    ],
    "history": [
        "Describe the causes and consequences of the French Revolution.",
        "Explain the rise and fall of the Roman Empire.",
        "What were the main causes of World War I?",
        "Describe the Industrial Revolution and its impact on society.",
        "Explain the Renaissance and its influence on European culture.",
    ],
    "geography": [
        "Explain plate tectonics and how they shape Earth's surface.",
        "Describe the water cycle and its role in climate systems.",
        "What are the major biomes and how are they distributed globally?",
        "Explain how erosion and weathering shape landscapes.",
        "Describe ocean currents and their effect on regional climates.",
    ],
    "philosophy": [
        "Explain the difference between empiricism and rationalism.",
        "Describe Kant's categorical imperative and deontological ethics.",
        "What is existentialism and how did Sartre define it?",
        "Explain Plato's theory of forms and the allegory of the cave.",
        "Describe the problem of free will and determinism.",
    ],
    "psychology": [
        "Explain classical and operant conditioning with examples.",
        "Describe the stages of cognitive development according to Piaget.",
        "What is cognitive behavioral therapy and how does it work?",
        "Explain the role of neurotransmitters in mood regulation.",
        "Describe attachment theory and its implications for development.",
    ],
    "linguistics": [
        "Explain the difference between morphology and syntax.",
        "Describe how phonemes and morphemes build language structure.",
        "What is the Sapir-Whorf hypothesis about language and thought?",
        "Explain how languages change over time through sound shifts.",
        "Describe the difference between descriptive and prescriptive grammar.",
    ],
    "art": [
        "Explain the principles of linear perspective in Renaissance art.",
        "Describe the Impressionist movement and its key innovations.",
        "What is the difference between Baroque and Rococo art styles?",
        "Explain how Cubism deconstructed traditional representation.",
        "Describe the role of composition and color theory in painting.",
    ],
    "music": [
        "Explain the basics of harmony and chord progressions.",
        "Describe sonata form and its role in classical music.",
        "What is counterpoint and how did Bach use it?",
        "Explain rhythm, meter, and syncopation in music.",
        "Describe the orchestra and the role of each instrument section.",
    ],
    "astronomy": [
        "Explain the life cycle of a star from nebula to remnant.",
        "Describe how we detect and characterize exoplanets.",
        "What is the expanding universe and what evidence supports it?",
        "Explain the difference between galaxies: spiral, elliptical, irregular.",
        "Describe the electromagnetic spectrum and astronomical spectroscopy.",
    ],
    "geology": [
        "Explain the rock cycle: igneous, sedimentary, and metamorphic.",
        "Describe plate tectonics and the formation of mountain ranges.",
        "What are the main types of volcanoes and how do they form?",
        "Explain how fossils form and what they tell us about Earth's history.",
        "Describe groundwater systems, aquifers, and the water table.",
    ],
    "engineering": [
        "Explain how a transistor works and its role in digital circuits.",
        "Describe the principles of structural engineering: loads and forces.",
        "What is a feedback control system and how does it maintain stability?",
        "Explain the engineering design process from requirements to testing.",
        "Describe how hydraulic systems transmit force through fluids.",
    ],
    "nutrition": [
        "Explain the role of macronutrients: carbohydrates, proteins, and fats.",
        "Describe how the body absorbs and metabolizes vitamins and minerals.",
        "What is the glycemic index and how does it affect blood sugar?",
        "Explain the importance of dietary fiber for digestive health.",
        "Describe the relationship between diet and chronic disease prevention.",
    ],
    "statistics": [
        "Explain hypothesis testing and the meaning of p-values.",
        "Describe linear regression and how to interpret its coefficients.",
        "What is Bayesian inference and how does it differ from frequentist?",
        "Explain the central limit theorem and its practical importance.",
        "Describe common sampling methods and their biases.",
    ],
    "architecture": [
        "Explain the structural principles behind arches, vaults, and domes.",
        "Describe the key features of Gothic cathedral architecture.",
        "What is sustainable architecture and how does it reduce impact?",
        "Explain the Modernist movement: form follows function.",
        "Describe how foundations transfer building loads to the ground.",
    ],
    "ecology": [
        "Explain food webs and energy flow through trophic levels.",
        "Describe ecological succession from pioneer species to climax.",
        "What is biodiversity and why is it important for ecosystems?",
        "Explain the nitrogen cycle and its biogeochemical importance.",
        "Describe the concept of ecological niches and competitive exclusion.",
    ],
    "neuroscience": [
        "Explain how neurons communicate through synaptic transmission.",
        "Describe the structure and function of the cerebral cortex.",
        "What is neuroplasticity and how does the brain adapt to injury?",
        "Explain the role of myelination in neural signal propagation.",
        "Describe how neurotransmitters modulate brain function and behavior.",
    ],
    "education": [
        "Explain Bloom's taxonomy and its levels of cognitive complexity.",
        "Describe the difference between formative and summative assessment.",
        "What is scaffolding in education and how does it support learning?",
        "Explain constructivist learning theory and its classroom applications.",
        "Describe differentiated instruction and how it meets diverse needs.",
    ],
    "agriculture": [
        "Explain the principles of crop rotation and soil conservation.",
        "Describe how nitrogen fixation supports plant growth.",
        "What is integrated pest management and how does it reduce chemicals?",
        "Explain the importance of soil health: structure, pH, and organisms.",
        "Describe sustainable farming practices like agroforestry and cover crops.",
    ],
}

# Evaluation queries (distinct from seed questions)
EVAL_QUERIES = {
    # Original domains (from P8, shortened to 5)
    "math": [
        "Explain what a limit is and how to compute it.",
        "Describe how matrix multiplication works and its applications.",
        "What is the mean value theorem and why is it important?",
        "Explain what a differential equation is and give an example.",
        "Describe the properties of logarithmic functions.",
    ],
    "code": [
        "Explain how a hash table works internally.",
        "Describe how merge sort works and analyze its complexity.",
        "What is dynamic programming and when should you use it?",
        "Explain the difference between shallow and deep copy.",
        "Describe how a binary search tree works.",
    ],
    "medical": [
        "Explain how statins work to lower cholesterol.",
        "Describe how the kidneys regulate blood pressure.",
        "What are autoimmune diseases and give examples.",
        "Explain how local anesthetics work at the molecular level.",
        "Describe the pathophysiology of heart failure.",
    ],
    "legal": [
        "Explain the concept of sovereign immunity.",
        "Describe the elements of fraud in contract law.",
        "What is strict liability and when does it apply?",
        "Explain what administrative law is and how agencies operate.",
        "Describe the doctrine of respondeat superior.",
    ],
    "finance": [
        "Explain what net present value means and how to calculate it.",
        "What is the efficient market hypothesis?",
        "Describe how credit ratings affect bond pricing.",
        "Explain the difference between fiscal and monetary policy.",
        "Describe the concept of financial leverage.",
    ],
    # New domains
    "physics": [
        "Describe the photoelectric effect and its significance.",
        "Explain how a capacitor stores electrical energy.",
        "What is angular momentum conservation?",
        "Describe the Doppler effect for sound and light.",
        "Explain the concept of work and power in mechanics.",
    ],
    "chemistry": [
        "Describe how electrochemical cells generate electricity.",
        "Explain the concept of pH and buffer solutions.",
        "What determines the rate of a chemical reaction?",
        "Describe the properties of transition metals.",
        "Explain how polymers are formed through polymerization.",
    ],
    "biology": [
        "Describe how proteins are synthesized from mRNA.",
        "Explain the structure and function of cell membranes.",
        "What is CRISPR and how does it edit genes?",
        "Describe the role of enzymes in metabolic pathways.",
        "Explain how the nervous system transmits signals.",
    ],
    "history": [
        "Describe the major consequences of the Columbian Exchange.",
        "Explain the causes of the Cold War.",
        "What was the significance of the printing press?",
        "Describe the Enlightenment and its key ideas.",
        "Explain how decolonization reshaped the world after 1945.",
    ],
    "geography": [
        "Describe the formation of river deltas.",
        "Explain what causes monsoon seasons.",
        "What is the greenhouse effect and its geographic impacts?",
        "Describe the characteristics of tropical rainforest biomes.",
        "Explain how urbanization changes local geography.",
    ],
    "philosophy": [
        "Describe the trolley problem and its ethical implications.",
        "Explain John Stuart Mill's utilitarianism.",
        "What is the mind-body problem in philosophy?",
        "Describe Descartes' method of systematic doubt.",
        "Explain the concept of the social contract.",
    ],
    "psychology": [
        "Describe the bystander effect and its psychological basis.",
        "Explain what working memory is and how it functions.",
        "What is confirmation bias and how does it affect decisions?",
        "Describe Erikson's stages of psychosocial development.",
        "Explain the difference between intrinsic and extrinsic motivation.",
    ],
    "linguistics": [
        "Describe the International Phonetic Alphabet and its purpose.",
        "Explain how children acquire language in early development.",
        "What is code-switching and when does it occur?",
        "Describe the difference between agglutinative and isolating languages.",
        "Explain how semantic fields organize vocabulary.",
    ],
    "art": [
        "Describe the influence of Japanese art on Western Impressionism.",
        "Explain what Abstract Expressionism was about.",
        "What role does symbolism play in medieval art?",
        "Describe the concept of negative space in visual design.",
        "Explain how photography changed the art world.",
    ],
    "music": [
        "Describe the blues scale and its influence on modern music.",
        "Explain the difference between major and minor keys.",
        "What is a fugue and how is it structured?",
        "Describe the role of dynamics in musical expression.",
        "Explain how electronic music synthesis works.",
    ],
    "astronomy": [
        "Describe what a black hole is and how it forms.",
        "Explain the cosmic microwave background radiation.",
        "What causes solar eclipses and lunar eclipses?",
        "Describe the Hertzsprung-Russell diagram and its use.",
        "Explain the evidence for dark matter in galaxies.",
    ],
    "geology": [
        "Describe how earthquakes are measured and classified.",
        "Explain the process of soil formation.",
        "What causes ice ages and glacial cycles?",
        "Describe the geological time scale and its divisions.",
        "Explain how caves and karst landscapes form.",
    ],
    "engineering": [
        "Describe how a bridge distributes loads.",
        "Explain the principles of electrical power generation.",
        "What is signal processing and where is it used?",
        "Describe how heat exchangers work in thermal systems.",
        "Explain the concept of mechanical advantage in machines.",
    ],
    "nutrition": [
        "Describe the role of omega-3 fatty acids in health.",
        "Explain how the body processes and stores carbohydrates.",
        "What are essential amino acids and why are they important?",
        "Describe the effects of dehydration on the body.",
        "Explain the concept of basal metabolic rate.",
    ],
    "statistics": [
        "Describe what a confidence interval tells you.",
        "Explain the difference between Type I and Type II errors.",
        "What is multicollinearity and why is it a problem?",
        "Describe the chi-squared test and when to use it.",
        "Explain what an outlier is and how to handle it.",
    ],
    "architecture": [
        "Describe the key features of Art Deco architecture.",
        "Explain how natural ventilation works in building design.",
        "What is parametric architecture?",
        "Describe the role of materials in architectural expression.",
        "Explain how load-bearing walls differ from frame structures.",
    ],
    "ecology": [
        "Describe how invasive species disrupt native ecosystems.",
        "Explain the concept of a trophic cascade.",
        "What is habitat fragmentation and its effects?",
        "Describe the role of decomposers in nutrient cycling.",
        "Explain how coral reef ecosystems function.",
    ],
    "neuroscience": [
        "Describe the blood-brain barrier and its function.",
        "Explain how long-term potentiation relates to memory.",
        "What are mirror neurons and what role do they play?",
        "Describe the difference between the CNS and PNS.",
        "Explain how brain imaging techniques (fMRI, EEG) work.",
    ],
    "education": [
        "Describe the zone of proximal development.",
        "Explain project-based learning and its benefits.",
        "What is universal design for learning?",
        "Describe how standardized tests are designed and validated.",
        "Explain the role of feedback in the learning process.",
    ],
    "agriculture": [
        "Describe how drip irrigation conserves water.",
        "Explain the role of pollinators in food production.",
        "What is soil erosion and how is it prevented?",
        "Describe the principles of organic farming.",
        "Explain how climate change affects agricultural productivity.",
    ],
}


def score_vocabulary(text: str, glossary: list) -> int:
    """Count domain glossary terms appearing in text."""
    text_lower = text.lower()
    return sum(1 for term in glossary if term.lower() in text_lower)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0: Generate training data for new domains using base model
# ══════════════════════════════════════════════════════════════════════════════

def extract_response_text(raw_output: str) -> str:
    """Extract generated text from mlx_lm CLI output and strip thinking tokens."""
    parts = raw_output.strip().split("==========")
    if len(parts) >= 3:
        text = parts[1].strip()
    else:
        text = raw_output.strip()

    # Gemma 4 E4B-IT generates <|channel>thought ... <channel|> actual answer
    # Strip thinking content and keep only the actual answer
    for marker in ["<channel|>", "</channel>"]:
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break

    # Also strip any remaining channel markers
    if text.startswith("<|channel>"):
        # Remove any additional channel markers at start
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if not line.strip().startswith("<|channel>") and not line.strip().startswith("<channel"):
                text = "\n".join(lines[i:])
                break

    return text


def generate_response_cli(question: str, adapter_path: str | None = None,
                          max_tokens: int = 200) -> str:
    """Generate a single response using mlx_lm CLI."""
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    cmd = [
        "uv", "run", "python", "-m", "mlx_lm", "generate",
        "--model", MODEL_ID,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.0",
    ]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return ""
    return extract_response_text(result.stdout)


def phase_generate_training_data():
    """Generate training data for all 20 new domains using base model."""
    log("\n=== Phase 0: Generate Training Data for New Domains ===")

    for domain in NEW_DOMAINS:
        data_dir = EXPERIMENT_DIR / f"data_{domain}"
        if data_dir.exists() and (data_dir / "train.jsonl").exists():
            n = sum(1 for _ in open(data_dir / "train.jsonl"))
            if n >= 5:
                log(f"  [{domain}] Training data exists ({n} examples)")
                continue

        data_dir.mkdir(parents=True, exist_ok=True)
        questions = SEED_QUESTIONS[domain]

        log(f"  [{domain}] Generating {len(questions)} training examples...")
        pairs = []
        for q in questions:
            # Use 1500 tokens to get past Gemma 4's thinking phase (~500 tok)
            # and capture the actual explanatory answer
            resp = generate_response_cli(q, max_tokens=1500)
            if resp and len(resp) > 50:
                pairs.append((q, resp))
                log(f"    Generated {len(resp)} chars for: {q[:50]}...")
            else:
                log(f"    FAILED (too short: {len(resp)} chars): {q[:50]}...")

        if len(pairs) < 3:
            log(f"  [{domain}] WARNING: Only {len(pairs)} examples generated")

        # Cycle pairs to create more training examples
        expanded = []
        while len(expanded) < max(len(pairs) * 4, 20):
            for q, a in pairs:
                expanded.append((q, a))
                if len(expanded) >= max(len(pairs) * 4, 20):
                    break

        # Write train/valid/test splits
        train_pairs = expanded[:len(expanded) - 4]
        valid_pairs = expanded[len(expanded) - 4:len(expanded) - 2]
        test_pairs = expanded[len(expanded) - 2:]

        def write_jsonl(path, pair_list):
            with open(path, "w") as f:
                for q, a in pair_list:
                    record = {
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ]
                    }
                    f.write(json.dumps(record) + "\n")

        write_jsonl(data_dir / "train.jsonl", train_pairs)
        write_jsonl(data_dir / "valid.jsonl", valid_pairs)
        write_jsonl(data_dir / "test.jsonl", test_pairs)
        log(f"  [{domain}] Saved {len(train_pairs)} train, {len(valid_pairs)} valid")

    log("  Training data generation complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train adapters for new domains
# ══════════════════════════════════════════════════════════════════════════════

def phase_train_new_adapters():
    """Train v_proj+o_proj LoRA adapters for 20 new domains."""
    log("\n=== Phase 1: Train New Domain Adapters ===")
    training_times = {}

    for domain in NEW_DOMAINS:
        adapter_dir = EXPERIMENT_DIR / f"adapter_{domain}"
        data_dir = EXPERIMENT_DIR / f"data_{domain}"

        if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
            log(f"  [{domain}] Adapter exists, skipping")
            training_times[domain] = 0.0
            continue

        adapter_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model": MODEL_ID,
            "data": str(data_dir),
            "adapter_path": str(adapter_dir),
            "train": True,
            "fine_tune_type": "lora",
            "num_layers": 16,
            "iters": TRAIN_ITERS,
            "batch_size": 1,
            "learning_rate": 2e-4,
            "lora_parameters": {
                "rank": LORA_RANK,
                "scale": LORA_SCALE,
                "dropout": 0.0,
                "keys": LORA_KEYS,
            },
            "max_seq_length": 512,
            "mask_prompt": True,
            "grad_checkpoint": True,
            "save_every": TRAIN_ITERS,
            "steps_per_report": max(1, TRAIN_ITERS // 5),
            "val_batches": 2,
            "steps_per_eval": max(10, TRAIN_ITERS // 3),
            "seed": SEED,
        }

        config_path = EXPERIMENT_DIR / f"lora_config_{domain}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        log(f"  [{domain}] Training rank-{LORA_RANK} ({TRAIN_ITERS} iters)...")
        t0 = time.time()
        cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora",
               "--config", str(config_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = (time.time() - t0) / 60.0

        if result.returncode != 0:
            log(f"  [{domain}] Training FAILED: {result.stderr[-500:]}")
            training_times[domain] = -1.0
            continue

        training_times[domain] = elapsed
        log(f"  [{domain}] Done in {elapsed:.1f} min")

    return training_times


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Base + solo adapter evaluation for all 25 domains
# ══════════════════════════════════════════════════════════════════════════════

def phase_evaluate_all():
    """Evaluate base model and solo adapters for all 25 domains.
    Uses subprocess calls to avoid holding model in memory."""
    log("\n=== Phase 2: Base + Solo Evaluation (25 domains) ===")

    base_scores = {}  # domain -> [vocab_scores]
    solo_scores = {}  # domain -> [vocab_scores]

    for domain in ALL_DOMAINS:
        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = GLOSSARIES[domain]

        # Determine adapter path
        if domain in ORIGINAL_DOMAINS:
            adapter_path = str(P8_DIR / f"adapter_{domain}")
        else:
            adapter_path = str(EXPERIMENT_DIR / f"adapter_{domain}")

        # Check adapter exists
        if not Path(adapter_path).exists() or \
           not (Path(adapter_path) / "adapters.safetensors").exists():
            log(f"  [{domain}] SKIPPING — no adapter found at {adapter_path}")
            continue

        log(f"  [{domain}] Evaluating ({len(queries)} queries)...")
        domain_base = []
        domain_solo = []

        for i, q in enumerate(queries):
            # Base response
            base_resp = generate_response_cli(q)
            base_v = score_vocabulary(base_resp, glossary)
            domain_base.append(base_v)

            # Solo adapter response
            solo_resp = generate_response_cli(q, adapter_path=adapter_path)
            solo_v = score_vocabulary(solo_resp, glossary)
            domain_solo.append(solo_v)

            log(f"    [{i+1}/{len(queries)}] base={base_v} solo={solo_v}")

        base_scores[domain] = domain_base
        solo_scores[domain] = domain_solo

        base_mean = sum(domain_base) / len(domain_base)
        solo_mean = sum(domain_solo) / len(domain_solo)
        improved = sum(1 for b, s in zip(domain_base, domain_solo) if s > b)
        log(f"  [{domain}] base_mean={base_mean:.1f} solo_mean={solo_mean:.1f} "
            f"improved={improved}/{len(queries)}")

    return base_scores, solo_scores


# ══════════════════════════════════════════════════════════════════════════════
# Adapter composition (concatenated LoRA)
# ══════════════════════════════════════════════════════════════════════════════

def create_composed_adapter(domains: list, weights: dict, save_dir: Path):
    """Create a composed adapter by concatenating weighted LoRA matrices.
    A_composed = [w_1*A_1 | w_2*A_2 | ... | w_N*A_N]
    B_composed = [B_1; B_2; ...; B_N]
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load all adapter weights
    adapter_weights = {}
    for d in domains:
        if d in ORIGINAL_DOMAINS:
            path = P8_DIR / f"adapter_{d}" / "adapters.safetensors"
        else:
            path = EXPERIMENT_DIR / f"adapter_{d}" / "adapters.safetensors"
        if path.exists():
            adapter_weights[d] = dict(mx.load(str(path)))

    # Find all LoRA key prefixes from first adapter
    first_d = domains[0]
    all_keys = sorted(set(k.rsplit(".", 1)[0] for k in adapter_weights[first_d].keys()))

    composed = {}
    for prefix in all_keys:
        a_key = f"{prefix}.lora_a"
        b_key = f"{prefix}.lora_b"

        a_parts = []
        b_parts = []
        for d in domains:
            if d not in adapter_weights:
                continue
            w = weights[d]
            adapter = adapter_weights[d]
            if a_key in adapter and b_key in adapter:
                a_parts.append(w * adapter[a_key])
                b_parts.append(adapter[b_key])

        if a_parts:
            a_composed = mx.concatenate(a_parts, axis=1)
            b_composed = mx.concatenate(b_parts, axis=0)
            mx.eval(a_composed, b_composed)
            composed[a_key] = a_composed
            composed[b_key] = b_composed

    # Save
    mx.save_safetensors(str(save_dir / "adapters.safetensors"), composed)

    composed_rank = len(domains) * LORA_RANK
    config = {
        "adapter_path": str(save_dir),
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": composed_rank,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": LORA_KEYS,
        },
        "num_layers": 16,
        "model": MODEL_ID,
    }
    (save_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))

    # Free adapter weights
    del adapter_weights, composed
    cleanup()

    log(f"  Composed adapter: N={len(domains)}, rank={composed_rank}")
    return str(save_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: N=25 composition evaluation
# ══════════════════════════════════════════════════════════════════════════════

def phase_composition_eval(base_scores: dict):
    """Evaluate N=25 composition with equal weights."""
    log("\n=== Phase 3: N=25 Composition Evaluation ===")

    # Only compose domains that have adapters
    available_domains = []
    for d in ALL_DOMAINS:
        if d in ORIGINAL_DOMAINS:
            path = P8_DIR / f"adapter_{d}" / "adapters.safetensors"
        else:
            path = EXPERIMENT_DIR / f"adapter_{d}" / "adapters.safetensors"
        if path.exists():
            available_domains.append(d)

    N = len(available_domains)
    log(f"  Available adapters: {N}/{len(ALL_DOMAINS)}")

    if N < 20:
        log(f"  WARNING: Only {N} adapters available, expected 25")

    # Equal weights
    equal_weights = {d: 1.0 / N for d in available_domains}

    composed_dir = COMPOSED_DIR / "n25_equal"
    if composed_dir.exists():
        shutil.rmtree(composed_dir)
    composed_path = create_composed_adapter(available_domains, equal_weights, composed_dir)

    # Evaluate each domain with composed adapter
    composition_scores = {}  # domain -> [vocab_scores]
    for domain in available_domains:
        if domain not in base_scores:
            continue

        queries = EVAL_QUERIES[domain][:N_EVAL]
        glossary = GLOSSARIES[domain]

        log(f"  [{domain}] N={N} composed eval ({len(queries)} queries)...")
        domain_composed = []
        for i, q in enumerate(queries):
            resp = generate_response_cli(q, adapter_path=composed_path)
            v = score_vocabulary(resp, glossary)
            domain_composed.append(v)
            log(f"    [{i+1}/{len(queries)}] composed_vocab={v} (base={base_scores[domain][i]})")

        composition_scores[domain] = domain_composed

    # Cleanup
    shutil.rmtree(composed_dir, ignore_errors=True)
    cleanup()

    return composition_scores, N


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: PPL measurement at N=25
# ══════════════════════════════════════════════════════════════════════════════

def phase_ppl_measurement(available_domains: list):
    """Measure PPL degradation under N=25 composition."""
    log("\n=== Phase 4: PPL Measurement ===")
    from mlx_lm.utils import load as mlx_load

    # Test texts: 2 eval queries per domain
    test_texts = []
    for domain in available_domains[:25]:
        for q in EVAL_QUERIES.get(domain, [])[:2]:
            test_texts.append(q)

    def measure_ppl(model, tokenizer, texts):
        total_nll = 0.0
        total_tokens = 0
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])
            logits = model(x)
            logits = logits.squeeze(0)
            log_probs = mx.softmax(logits, axis=-1)
            target_probs = log_probs[mx.arange(y.shape[0]), y]
            nll = -mx.sum(mx.log(target_probs + 1e-10))
            mx.eval(nll)
            total_nll += nll.item()
            total_tokens += y.shape[0]
            del logits, log_probs, target_probs, nll, x, y
        return float(total_nll / total_tokens) if total_tokens > 0 else float('inf')

    # Base model PPL
    log("  Measuring base model PPL...")
    model, tokenizer = mlx_load(MODEL_ID)
    base_ppl = measure_ppl(model, tokenizer, test_texts)
    log(f"  Base PPL (NLL/token): {base_ppl:.4f}")
    del model, tokenizer
    cleanup()

    # Composed N=25 PPL
    N = len(available_domains)
    equal_weights = {d: 1.0 / N for d in available_domains}
    composed_dir = COMPOSED_DIR / "ppl_n25"
    if composed_dir.exists():
        shutil.rmtree(composed_dir)
    composed_path = create_composed_adapter(available_domains, equal_weights, composed_dir)

    log(f"  Measuring N={N} composed PPL...")
    model, tokenizer = mlx_load(MODEL_ID, adapter_path=composed_path)
    composed_ppl = measure_ppl(model, tokenizer, test_texts)
    log(f"  Composed PPL (NLL/token): {composed_ppl:.4f}")

    degradation = (composed_ppl - base_ppl) / base_ppl if base_ppl > 0 else 0.0
    log(f"  PPL degradation: {degradation*100:.2f}%")

    del model, tokenizer
    cleanup()
    shutil.rmtree(composed_dir, ignore_errors=True)

    return {"base_ppl": base_ppl, "composed_ppl": composed_ppl, "degradation": degradation}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Latency measurement
# ══════════════════════════════════════════════════════════════════════════════

def phase_latency_measurement(available_domains: list):
    """Measure inference latency with composed adapter vs base."""
    log("\n=== Phase 5: Latency Measurement ===")

    test_prompt = "Explain the concept of entropy."

    # Base latency
    log("  Measuring base latency...")
    t0 = time.time()
    for _ in range(3):
        generate_response_cli(test_prompt)
    base_time = (time.time() - t0) / 3.0

    # Composed N=25 latency
    N = len(available_domains)
    equal_weights = {d: 1.0 / N for d in available_domains}
    composed_dir = COMPOSED_DIR / "latency_n25"
    if composed_dir.exists():
        shutil.rmtree(composed_dir)
    create_composed_adapter(available_domains, equal_weights, composed_dir)

    log(f"  Measuring N={N} composed latency...")
    t0 = time.time()
    for _ in range(3):
        generate_response_cli(test_prompt, adapter_path=str(composed_dir))
    composed_time = (time.time() - t0) / 3.0

    ratio = composed_time / base_time if base_time > 0 else float('inf')
    log(f"  Base: {base_time:.2f}s, Composed: {composed_time:.2f}s, Ratio: {ratio:.2f}")

    shutil.rmtree(composed_dir, ignore_errors=True)

    return {"base_time_s": base_time, "composed_time_s": composed_time, "ratio": ratio}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("P0: N=25 v_proj+o_proj Composition Stress Test")
    log(f"IS_SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, TRAIN_ITERS={TRAIN_ITERS}")
    log(f"Original domains: {len(ORIGINAL_DOMAINS)}, New: {len(NEW_DOMAINS)}")
    log("=" * 70)

    total_start = time.time()
    cleanup()
    log_memory("start")

    # Verify P8 adapters exist
    for domain in ORIGINAL_DOMAINS:
        path = P8_DIR / f"adapter_{domain}" / "adapters.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"Missing P8 adapter: {path}")
    log("All 5 P8 adapters verified.")

    # Clean composed dir
    if COMPOSED_DIR.exists():
        shutil.rmtree(COMPOSED_DIR)
    COMPOSED_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0: Generate training data
    phase_generate_training_data()

    # Phase 1: Train new domain adapters
    training_times = phase_train_new_adapters()

    # Phase 2: Base + solo evaluation
    base_scores, solo_scores = phase_evaluate_all()

    # Compute solo improvement rates
    solo_rates = {}
    for domain in ALL_DOMAINS:
        if domain in base_scores and domain in solo_scores:
            base = base_scores[domain]
            solo = solo_scores[domain]
            improved = sum(1 for b, s in zip(base, solo) if s > b)
            solo_rates[domain] = improved / len(base)
    log(f"\nSolo improvement rates (summary):")
    for d, r in sorted(solo_rates.items()):
        log(f"  [{d}] {r*100:.0f}%")

    # Determine available domains (those with adapters + evaluation data)
    available_domains = [d for d in ALL_DOMAINS
                         if d in base_scores and d in solo_scores]

    # Phase 3: N=25 composition
    composition_scores, N_composed = phase_composition_eval(base_scores)

    # Phase 4: PPL
    ppl_results = phase_ppl_measurement(available_domains)

    # Phase 5: Latency
    latency_results = phase_latency_measurement(available_domains)

    # ══════════════════════════════════════════════════════════════════════════
    # Kill criteria analysis
    # ══════════════════════════════════════════════════════════════════════════

    log("\n" + "=" * 70)
    log("KILL CRITERIA")
    log("=" * 70)

    # Compute retention: composed vs solo (improvement over base)
    retention_per_domain = {}
    for domain in available_domains:
        if domain not in composition_scores or domain not in solo_scores:
            continue
        base = base_scores[domain]
        solo = solo_scores[domain]
        comp = composition_scores.get(domain, [])

        if not comp:
            continue

        # Improvement rate: fraction of queries where adapted > base
        solo_improved = sum(1 for b, s in zip(base, solo) if s > b)
        comp_improved = sum(1 for b, c in zip(base, comp) if c > b)

        solo_rate = solo_improved / len(base)
        comp_rate = comp_improved / len(base)
        retention = comp_rate / solo_rate if solo_rate > 0 else (1.0 if comp_rate == 0 else float('inf'))

        # Also compute mean vocab shift
        solo_mean = sum(solo) / len(solo)
        comp_mean = sum(comp) / len(comp)
        base_mean = sum(base) / len(base)

        vocab_retention = (comp_mean - base_mean) / (solo_mean - base_mean) \
            if (solo_mean - base_mean) != 0 else 1.0

        retention_per_domain[domain] = {
            "solo_rate": solo_rate,
            "comp_rate": comp_rate,
            "rate_retention": retention,
            "base_vocab_mean": base_mean,
            "solo_vocab_mean": solo_mean,
            "comp_vocab_mean": comp_mean,
            "vocab_retention": vocab_retention,
        }

    # K1324: Mean retention >= 70%
    if retention_per_domain:
        retentions = [r["vocab_retention"] for r in retention_per_domain.values()
                      if abs(r["vocab_retention"]) < 100]  # filter extreme outliers
        mean_retention = sum(retentions) / len(retentions) if retentions else 0.0
    else:
        mean_retention = 0.0
    k1324_pass = mean_retention >= 0.70
    log(f"\nK1324 (Mean retention >= 70%): {'PASS' if k1324_pass else 'FAIL'}")
    log(f"  Mean vocab retention: {mean_retention*100:.1f}%")
    for d, r in sorted(retention_per_domain.items()):
        log(f"  [{d}] vocab_retention={r['vocab_retention']:.2f} "
            f"(base={r['base_vocab_mean']:.1f} solo={r['solo_vocab_mean']:.1f} "
            f"comp={r['comp_vocab_mean']:.1f})")

    # K1325: No single domain < 30%
    min_retention = min(
        (r["vocab_retention"] for r in retention_per_domain.values()
         if abs(r["vocab_retention"]) < 100),
        default=1.0
    )
    k1325_pass = min_retention >= 0.30
    log(f"\nK1325 (No domain < 30% retention): {'PASS' if k1325_pass else 'FAIL'}")
    log(f"  Min retention: {min_retention*100:.1f}%")

    # K1326: PPL degradation <= 5%
    k1326_pass = ppl_results["degradation"] <= 0.05
    log(f"\nK1326 (PPL degradation <= 5%): {'PASS' if k1326_pass else 'FAIL'}")
    log(f"  Degradation: {ppl_results['degradation']*100:.2f}%")

    # K1327: Latency <= 110%
    k1327_pass = latency_results["ratio"] <= 1.10
    log(f"\nK1327 (Latency <= 110%): {'PASS' if k1327_pass else 'FAIL'}")
    log(f"  Ratio: {latency_results['ratio']:.2f}")
    if not k1327_pass:
        log(f"  NOTE: This uses concatenated LoRA (rank={N_composed*LORA_RANK}). "
            f"Pre-merged weights (Finding #503) have 0% overhead.")

    all_pass = k1324_pass and k1325_pass and k1326_pass and k1327_pass
    total_min = (time.time() - total_start) / 60.0

    log(f"\n{'='*70}")
    log(f"SUMMARY (N={N_composed} composition)")
    log(f"  K1324 (mean retention >= 70%): {'PASS' if k1324_pass else 'FAIL'} ({mean_retention*100:.1f}%)")
    log(f"  K1325 (no domain < 30%): {'PASS' if k1325_pass else 'FAIL'} ({min_retention*100:.1f}%)")
    log(f"  K1326 (PPL <= 5%): {'PASS' if k1326_pass else 'FAIL'} ({ppl_results['degradation']*100:.2f}%)")
    log(f"  K1327 (latency <= 110%): {'PASS' if k1327_pass else 'FAIL'} ({latency_results['ratio']:.2f})")
    log(f"  ALL PASS: {all_pass}")
    log(f"Total time: {total_min:.1f} min")

    # Scaling comparison with N=5 (Finding #505)
    log(f"\nScaling comparison (MATH.md predictions):")
    log(f"  N=5 (Finding #505): mean retention ~113% (ensemble effect)")
    log(f"  N={N_composed}: mean retention {mean_retention*100:.1f}%")
    log(f"  Predicted: 70-90% (SIR ∝ 1/√N, ensemble partially compensates)")
    log(f"{'='*70}")

    # Save results
    results = {
        "is_smoke": IS_SMOKE,
        "config": {
            "n_eval": N_EVAL,
            "n_composed": N_composed,
            "train_iters": TRAIN_ITERS,
            "lora_rank": LORA_RANK,
            "lora_keys": LORA_KEYS,
        },
        "training_times": training_times,
        "solo_rates": solo_rates,
        "retention_per_domain": retention_per_domain,
        "ppl": ppl_results,
        "latency": latency_results,
        "kill_criteria": {
            "k1324_mean_retention": {"pass": k1324_pass, "value": mean_retention, "threshold": 0.70},
            "k1325_min_retention": {"pass": k1325_pass, "value": min_retention, "threshold": 0.30},
            "k1326_ppl_degradation": {"pass": k1326_pass, "value": ppl_results["degradation"], "threshold": 0.05},
            "k1327_latency_ratio": {"pass": k1327_pass, "value": latency_results["ratio"], "threshold": 1.10},
        },
        "scaling_comparison": {
            "n5_retention": 1.13,
            "n25_retention": mean_retention,
            "predicted_range": [0.70, 0.90],
        },
        "all_pass": all_pass,
        "total_time_min": round(total_min, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
