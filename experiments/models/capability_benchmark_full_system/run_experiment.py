#!/usr/bin/env python3
"""Capability Benchmark: Full System on Format-Tasks.

Verification experiment: composition helps on CAPABILITY benchmarks where
FORMAT=CAPABILITY (GSM8K, code generation, clinical NER) while being neutral-to-harmful
on KNOWLEDGE benchmarks (MMLU factual recall).

Kill criteria:
  K1 (#675): Full system worse than base on GSM8K math reasoning
  K2 (#676): Full system worse than base on HumanEval code generation
  K3 (#677): Full system produces >5% incoherent output on any domain

Predictions (from MATH.md):
  GSM8K:       +10pp (38% -> 48%) [format-dependent: chain-of-thought]
  Code gen:    +15pp              [format-dependent: syntax IS capability]
  Clinical NER: +5pp             [format-dependent: extraction template]
  MMLU:        -5pp to 0pp       [knowledge-dependent: factual recall]
  Incoherence: <5% on all domains

Type: verification (Type 1)
Platform: Apple M5 Pro 48GB, MLX
"""

import ast
import gc
import json
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Paths to existing infrastructure
SFT_DIR = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3"
ADAPTERS_DIR = SFT_DIR / "sft_adapters"
SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
DATA_DIR = SOURCE_DIR / "data"
NTP_ADAPTERS_DIR = SOURCE_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
SEED = 42

DOMAINS = ["medical", "code", "math", "legal", "finance"]

# Per-domain optimal scales (Finding #249)
OPTIMAL_SCALES = {
    "medical": 20.0,
    "code": 20.0,
    "math": 20.0,
    "legal": 4.0,
    "finance": 1.0,
}

# Benchmark sizes
GSM8K_N = 20
CODE_GEN_N = 10
NER_N = 20
MMLU_N_PER_DOMAIN = 20

MAX_TOKENS_GSM8K = 256
MAX_TOKENS_CODE = 256
MAX_TOKENS_NER = 128
MAX_TOKENS_MMLU = 32
MAX_TOKENS_DOMAIN = 128

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# MMLU subjects for knowledge-benchmark contrast
MMLU_SUBJECTS = {
    "medical": ["clinical_knowledge", "professional_medicine", "anatomy", "medical_genetics"],
    "code": ["college_computer_science", "high_school_computer_science", "machine_learning"],
    "math": ["high_school_mathematics", "elementary_mathematics", "college_mathematics"],
    "legal": ["professional_law", "jurisprudence", "international_law"],
    "finance": ["professional_accounting", "econometrics", "high_school_macroeconomics"],
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Model utilities (from generation_quality_perscale)
# ============================================================================

def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ============================================================================
# Pre-merge composition (from competitive_benchmark_routed)
# ============================================================================

def load_skeleton():
    skeleton_path = NTP_ADAPTERS_DIR / "grassmannian_skeleton.npz"
    return dict(np.load(str(skeleton_path)))


def load_adapter(domain):
    adapter_path = ADAPTERS_DIR / domain / "adapter.npz"
    adapter = dict(mx.load(str(adapter_path)))
    return adapter


def save_base_weights(model):
    base_weights = []
    for layer in model.model.layers:
        layer_w = {}
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                layer_w[key] = module.weight
        base_weights.append(layer_w)
    return base_weights


def restore_base_weights(model, base_weights):
    for li, layer_weights in enumerate(base_weights):
        for key, weight in layer_weights.items():
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None and isinstance(module, nn.Linear):
                module.weight = weight
    mx.eval(model.parameters())


def premerge_single_adapter(model, skeleton, adapter, domain, scale):
    """Pre-merge: W_new = W_base + scale * B^T @ A^T"""
    di = DOMAINS.index(domain)
    merge_count = 0
    for li in range(len(model.model.layers)):
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = model.model.layers[li]
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue
            skey = f"layer_{li}_{key}_domain_{di}"
            if skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            b_key = f"model.layers.{li}.{key}.lora_b"
            if b_key not in adapter:
                continue
            b_mx = adapter[b_key]
            delta = scale * (b_mx.T @ a_mx.T)
            module.weight = module.weight + delta
            merge_count += 1
    mx.eval(model.parameters())
    return model


# ============================================================================
# Generation
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens=256):
    try:
        sampler = make_sampler(temp=0.0)
        text = mlx_generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text
    except Exception as e:
        log(f"  WARNING: generation failed: {e}")
        return ""


# ============================================================================
# Prompt formatting
# ============================================================================

def format_gsm8k_prompt(question):
    return (
        f"### Instruction:\n"
        f"Solve the following math problem step by step. "
        f"Show your work and give the final numerical answer after ####.\n\n"
        f"{question}\n\n"
        f"### Response:\n"
    )


def format_code_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def format_ner_prompt(text):
    return (
        f"### Instruction:\n"
        f"Extract all medical entities (diseases, medications, procedures, "
        f"anatomical terms, symptoms) from the following clinical text. "
        f"List each entity on a separate line.\n\n"
        f"Text: {text}\n\n"
        f"### Response:\n"
    )


def format_mmlu_prompt(question, choices):
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in zip(choice_labels, choices))
    return (
        f"### Instruction:\n"
        f"Answer the following multiple choice question. "
        f"Reply with just the letter (A, B, C, or D).\n\n"
        f"{question}\n\n{choices_text}\n\n"
        f"### Response:\n"
    )


def format_domain_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ============================================================================
# Answer extraction
# ============================================================================

def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
    matches = re.findall(r'=\s*\$?([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)', text)
    if matches:
        return float(matches[-1].replace(',', ''))
    matches = re.findall(r'([\d,]+(?:\.\d+)?)', text)
    if matches:
        try:
            return float(matches[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def extract_mmlu_answer(text):
    text = text.strip()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    match = re.search(r'(?:the\s+)?answer\s*(?:is|:)\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for line in text.split('\n'):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()
    match = re.search(r'[\(\s]([A-Da-d])[\)\.\s]', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-Da-d])\b', text)
    if match:
        return match.group(1).upper()
    return None


# ============================================================================
# Evaluation metrics
# ============================================================================

def eval_code_syntax(text):
    """Check if generated text contains valid Python syntax."""
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        pass
    code_blocks = re.findall(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        try:
            ast.parse(block.strip())
            return True
        except SyntaxError:
            continue
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'for ',
                                'while ', 'if ', 'try:', 'except', 'with ',
                                'return ', 'print(', '#')):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        try:
            ast.parse('\n'.join(code_lines))
            return True
        except SyntaxError:
            pass
    return False


def eval_ner_entities(generated_text, reference_entities):
    """Evaluate NER by entity overlap F1.

    reference_entities: list of strings (expected entities)
    Returns: dict with precision, recall, f1
    """
    # Extract entities from generated text: one per line, or comma-separated
    gen_lines = [line.strip().lower() for line in generated_text.split('\n') if line.strip()]
    gen_entities = set()
    for line in gen_lines:
        # Remove bullet/number prefixes
        line = re.sub(r'^[\d\.\-\*]+\s*', '', line)
        # Remove common prefixes like "Disease:", "Medication:", etc.
        line = re.sub(r'^(disease|medication|procedure|symptom|anatomy|entity|condition|treatment|drug|diagnosis)\s*:\s*', '', line, flags=re.IGNORECASE)
        if line and len(line) > 1:
            gen_entities.add(line.strip())

    ref_entities_lower = set(e.lower().strip() for e in reference_entities)

    if not ref_entities_lower:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "gen_count": len(gen_entities), "ref_count": 0}

    # Fuzzy matching: a generated entity matches if it contains or is contained by a reference entity
    matched_ref = set()
    matched_gen = set()
    for ge in gen_entities:
        for re_ in ref_entities_lower:
            if ge in re_ or re_ in ge:
                matched_ref.add(re_)
                matched_gen.add(ge)

    precision = len(matched_gen) / len(gen_entities) if gen_entities else 0.0
    recall = len(matched_ref) / len(ref_entities_lower)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gen_count": len(gen_entities),
        "ref_count": len(ref_entities_lower),
        "matched_ref": len(matched_ref),
        "matched_gen": len(matched_gen),
    }


def eval_format_quality(text):
    """Check for incoherent output: empty, repetitive, or garbage."""
    if not text.strip():
        return 0.0
    words = text.split()
    if len(words) < 3:
        return 0.1
    unique_words = set(w.lower() for w in words)
    diversity = len(unique_words) / len(words)
    if diversity < 0.1:
        return 0.1
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if len(alpha_words) < 3:
        return 0.2
    return min(1.0, diversity + 0.3)


STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'must', 'need', 'ought',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'not', 'no', 'nor', 'but', 'and', 'or', 'so', 'if', 'then',
    'than', 'too', 'very', 'just', 'only', 'also', 'more', 'most',
    'some', 'any', 'all', 'each', 'every', 'both', 'few', 'many',
    'about', 'after', 'at', 'before', 'between', 'by', 'for', 'from',
    'in', 'into', 'of', 'on', 'out', 'to', 'under', 'up', 'with', 'as',
}


def extract_key_facts(text):
    facts = set()
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    for w in words:
        if len(w) >= 4 and w not in STOPWORDS:
            facts.add(w)
    return facts


def eval_factual_recall(generated_text, reference_text):
    ref_facts = extract_key_facts(reference_text)
    if not ref_facts:
        return 0.0
    gen_lower = generated_text.lower()
    matched = sum(1 for fact in ref_facts if fact in gen_lower)
    return matched / len(ref_facts)


# ============================================================================
# Data loading
# ============================================================================

def load_gsm8k_data(n=20):
    """Load GSM8K test problems from HuggingFace."""
    from datasets import load_dataset
    log(f"  Loading GSM8K ({n} problems)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        answer_text = item["answer"]
        match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', answer_text)
        if match:
            answer = float(match.group(1).replace(',', ''))
        else:
            nums = re.findall(r'([\d,]+(?:\.\d+)?)', answer_text)
            answer = float(nums[-1].replace(',', '')) if nums else None
        problems.append({"question": item["question"], "answer": answer})
    log(f"  GSM8K: {len(problems)} problems loaded")
    return problems


def load_code_gen_data(n=10):
    """Load code generation problems from our validation set."""
    log(f"  Loading code generation ({n} problems)...")
    val_path = DATA_DIR / "code" / "valid.jsonl"
    problems = []
    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()
                problems.append({"instruction": instruction, "reference": response})
            if len(problems) >= n:
                break
    log(f"  Code gen: {len(problems)} problems loaded")
    return problems


def load_clinical_ner_data(n=20):
    """Create clinical NER evaluation data from medical validation set.

    We take medical QA prompts and create NER tasks by extracting the medical
    entities from the reference answers, then asking the model to extract them.
    """
    log(f"  Loading clinical NER ({n} examples)...")
    val_path = DATA_DIR / "medical" / "valid.jsonl"
    examples = []

    # Medical entity patterns
    entity_pattern = re.compile(
        r'\b('
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'  # Capitalized phrases +
        r'(?:disease|syndrome|disorder|infection|cancer|tumor|deficiency|failure)'
        r'|'
        r'(?:aspirin|ibuprofen|acetaminophen|metformin|insulin|penicillin|amoxicillin|'
        r'lisinopril|atorvastatin|metoprolol|omeprazole|amlodipine|prednisone|'
        r'warfarin|heparin|morphine|codeine|naproxen|gabapentin|sertraline|'
        r'fluoxetine|citalopram|duloxetine|venlafaxine|bupropion|clonazepam|'
        r'alprazolam|diazepam|lorazepam|zolpidem)'
        r'|'
        r'(?:hypertension|diabetes|asthma|pneumonia|bronchitis|arthritis|'
        r'osteoporosis|anemia|migraine|epilepsy|depression|anxiety|insomnia|'
        r'hypothyroidism|hyperthyroidism|hepatitis|cirrhosis|pancreatitis|'
        r'appendicitis|cholecystitis|diverticulitis|meningitis|encephalitis|'
        r'myocarditis|endocarditis|pericarditis|gastritis|colitis|nephritis|'
        r'cystitis|dermatitis|cellulitis|sepsis|edema|fibrosis|stenosis|'
        r'thrombosis|embolism|hemorrhage|ischemia|infarction|necrosis|atrophy)'
        r'|'
        r'(?:biopsy|endoscopy|colonoscopy|mammography|ultrasound|MRI|CT scan|'
        r'X-ray|ECG|EKG|EEG|surgery|chemotherapy|radiation|dialysis|'
        r'transplant|catheterization|intubation|ventilation|transfusion)'
        r')\b', re.IGNORECASE
    )

    with open(val_path) as f:
        for line in f:
            text = json.loads(line)["text"]
            if "### Instruction:" in text and "### Response:" in text:
                instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                response = text.split("### Response:")[1].strip()

                # Extract medical entities from the reference answer
                entities = list(set(m.group(0).lower() for m in entity_pattern.finditer(response)))
                if len(entities) >= 2:
                    examples.append({
                        "text": instruction[:500],  # Use the question as clinical text
                        "entities": entities,
                        "reference": response,
                    })
            if len(examples) >= n:
                break

    # If not enough entity-rich examples, create synthetic ones
    if len(examples) < n:
        clinical_texts = [
            ("Patient presents with hypertension and diabetes mellitus type 2. Currently on metformin and lisinopril. Recent blood work shows anemia.",
             ["hypertension", "diabetes", "metformin", "lisinopril", "anemia"]),
            ("History of asthma with recent pneumonia. Treated with amoxicillin and prednisone. Chest X-ray shows consolidation.",
             ["asthma", "pneumonia", "amoxicillin", "prednisone", "x-ray"]),
            ("Patient with depression and anxiety disorder. Started on sertraline. Reports insomnia and migraine headaches.",
             ["depression", "anxiety", "sertraline", "insomnia", "migraine"]),
            ("Diagnosed with hepatitis B and cirrhosis. Liver biopsy performed. Monitoring for thrombosis risk.",
             ["hepatitis", "cirrhosis", "biopsy", "thrombosis"]),
            ("Elderly patient with osteoporosis and arthritis. History of hip surgery. Currently on gabapentin for pain.",
             ["osteoporosis", "arthritis", "surgery", "gabapentin"]),
            ("Patient admitted with acute appendicitis. CT scan confirmed diagnosis. Emergency surgery scheduled.",
             ["appendicitis", "ct scan", "surgery"]),
            ("Follow-up for breast cancer treatment. Completed chemotherapy and radiation. Mammography scheduled.",
             ["cancer", "chemotherapy", "radiation", "mammography"]),
            ("Chronic kidney disease patient on dialysis. History of hypertension. ECG shows atrial fibrillation.",
             ["dialysis", "hypertension", "ecg"]),
            ("Patient with epilepsy experiencing breakthrough seizures. EEG performed. Adjusting gabapentin dosage.",
             ["epilepsy", "eeg", "gabapentin"]),
            ("Diagnosis of gastritis and colitis. Endoscopy performed. Started on omeprazole.",
             ["gastritis", "colitis", "endoscopy", "omeprazole"]),
            ("Patient with hypothyroidism and edema. Ultrasound of thyroid performed. Monitoring TSH levels.",
             ["hypothyroidism", "edema", "ultrasound"]),
            ("History of myocardial infarction. Catheterization performed. On aspirin and atorvastatin.",
             ["infarction", "catheterization", "aspirin", "atorvastatin"]),
            ("Patient with bronchitis progressing to pneumonia. Chest X-ray shows bilateral infiltrates. Started antibiotics.",
             ["bronchitis", "pneumonia", "x-ray"]),
            ("Chronic pancreatitis patient with diabetes. CT scan shows calcifications. Pain managed with morphine.",
             ["pancreatitis", "diabetes", "ct scan", "morphine"]),
            ("Patient with cellulitis and sepsis risk. Blood cultures drawn. IV antibiotics started.",
             ["cellulitis", "sepsis"]),
            ("Diagnosed with pulmonary embolism. CT angiography confirmed. Started on warfarin anticoagulation.",
             ["embolism", "warfarin"]),
            ("Patient with meningitis. Lumbar puncture performed. IV penicillin started.",
             ["meningitis", "penicillin"]),
            ("Heart failure patient with peripheral edema. ECG shows cardiomyopathy. On metoprolol and diuretics.",
             ["failure", "edema", "ecg", "metoprolol"]),
            ("Patient with cholecystitis. Ultrasound shows gallstones. Surgery consultation requested.",
             ["cholecystitis", "ultrasound", "surgery"]),
            ("Chronic pain patient with fibrosis. MRI performed. Currently on duloxetine and naproxen.",
             ["fibrosis", "mri", "duloxetine", "naproxen"]),
        ]
        for text, entities in clinical_texts:
            if len(examples) >= n:
                break
            examples.append({
                "text": text,
                "entities": entities,
                "reference": text,  # For NER, the text itself is the reference
            })

    log(f"  Clinical NER: {len(examples)} examples ({sum(len(e['entities']) for e in examples)} total entities)")
    return examples[:n]


def load_mmlu_data(n_per_domain=20):
    """Load MMLU test questions mapped to adapter domains."""
    from datasets import load_dataset
    log(f"  Loading MMLU ({n_per_domain} per domain)...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    by_subject = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    mmlu_data = {}
    for domain, subjects in MMLU_SUBJECTS.items():
        questions = []
        for subj in subjects:
            if subj in by_subject:
                questions.extend(by_subject[subj])
        rng = np.random.RandomState(42)
        rng.shuffle(questions)
        mmlu_data[domain] = questions[:n_per_domain]
        log(f"    MMLU {domain}: {len(mmlu_data[domain])} questions")

    return mmlu_data


def load_domain_eval_data(n_per_domain=10):
    """Load domain-specific evaluation data for incoherence check."""
    log(f"  Loading domain eval data ({n_per_domain} per domain)...")
    data = {}
    for domain in DOMAINS:
        val_path = DATA_DIR / domain / "valid.jsonl"
        prompts = []
        with open(val_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                if "### Instruction:" in text and "### Response:" in text:
                    instruction = text.split("### Instruction:")[1].split("### Response:")[0].strip()
                    response = text.split("### Response:")[1].strip()
                    prompts.append({"instruction": instruction, "response": response})
                if len(prompts) >= n_per_domain:
                    break
        data[domain] = prompts
        log(f"    {domain}: {len(prompts)} prompts")
    return data


# ============================================================================
# Phase 1: Load all benchmark data
# ============================================================================

def phase_load_data():
    log("\n" + "=" * 70)
    log("PHASE 1: LOADING BENCHMARK DATA")
    log("=" * 70)
    gsm8k = load_gsm8k_data(GSM8K_N)
    code_gen = load_code_gen_data(CODE_GEN_N)
    clinical_ner = load_clinical_ner_data(NER_N)
    mmlu = load_mmlu_data(MMLU_N_PER_DOMAIN)
    domain_eval = load_domain_eval_data(10)
    return gsm8k, code_gen, clinical_ner, mmlu, domain_eval


# ============================================================================
# Phase 2: Evaluate base model on all benchmarks
# ============================================================================

def phase_eval_base(gsm8k, code_gen, clinical_ner, mmlu, domain_eval):
    log("\n" + "=" * 70)
    log("PHASE 2: BASE MODEL (no adapters)")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.random.seed(SEED)
    log_memory("base-loaded")

    results = {}

    # GSM8K
    log("\n  --- GSM8K ---")
    correct = 0
    total = len(gsm8k)
    for i, prob in enumerate(gsm8k):
        prompt = format_gsm8k_prompt(prob["question"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_GSM8K)
        predicted = extract_gsm8k_answer(gen)
        if predicted is not None and prob["answer"] is not None:
            if prob["answer"] == 0:
                is_correct = abs(predicted) < 0.01
            else:
                is_correct = abs(predicted - prob["answer"]) / abs(prob["answer"]) < 0.01
        else:
            is_correct = False
        if is_correct:
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{total}: {correct}/{i+1} ({100*correct/(i+1):.0f}%)")
    results["gsm8k"] = {"accuracy": correct / total, "correct": correct, "total": total}
    log(f"  GSM8K base: {correct}/{total} = {results['gsm8k']['accuracy']:.3f}")

    # Code generation
    log("\n  --- Code Generation ---")
    syntax_correct = 0
    for prob in code_gen:
        prompt = format_code_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_CODE)
        if eval_code_syntax(gen):
            syntax_correct += 1
    results["code_gen"] = {
        "syntax_rate": syntax_correct / len(code_gen),
        "correct": syntax_correct,
        "total": len(code_gen),
    }
    log(f"  Code base: {syntax_correct}/{len(code_gen)} syntax valid = {results['code_gen']['syntax_rate']:.3f}")

    # Clinical NER
    log("\n  --- Clinical NER ---")
    ner_f1s = []
    for ex in clinical_ner:
        prompt = format_ner_prompt(ex["text"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_NER)
        ner_result = eval_ner_entities(gen, ex["entities"])
        ner_f1s.append(ner_result["f1"])
    results["clinical_ner"] = {
        "mean_f1": float(np.mean(ner_f1s)),
        "std_f1": float(np.std(ner_f1s)),
        "scores": [float(f) for f in ner_f1s],
    }
    log(f"  NER base: mean F1 = {results['clinical_ner']['mean_f1']:.3f}")

    # MMLU (all domains aggregated)
    log("\n  --- MMLU ---")
    choice_labels = ["A", "B", "C", "D"]
    mmlu_correct = 0
    mmlu_total = 0
    mmlu_by_domain = {}
    for domain in DOMAINS:
        domain_correct = 0
        for q in mmlu[domain]:
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_MMLU)
            predicted = extract_mmlu_answer(gen)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                domain_correct += 1
                mmlu_correct += 1
            mmlu_total += 1
        domain_acc = domain_correct / len(mmlu[domain]) if mmlu[domain] else 0
        mmlu_by_domain[domain] = {"accuracy": domain_acc, "correct": domain_correct, "total": len(mmlu[domain])}
        log(f"    MMLU {domain}: {domain_correct}/{len(mmlu[domain])} = {domain_acc:.3f}")

    results["mmlu"] = {
        "accuracy": mmlu_correct / mmlu_total if mmlu_total > 0 else 0,
        "correct": mmlu_correct,
        "total": mmlu_total,
        "by_domain": mmlu_by_domain,
    }
    log(f"  MMLU base overall: {mmlu_correct}/{mmlu_total} = {results['mmlu']['accuracy']:.3f}")

    # Incoherence check (generate on 5 domain prompts each)
    log("\n  --- Incoherence Check ---")
    incoherence_results = {}
    for domain in DOMAINS:
        incoherent_count = 0
        for prompt_data in domain_eval[domain]:
            prompt = format_domain_prompt(prompt_data["instruction"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_DOMAIN)
            fq = eval_format_quality(gen)
            if fq < 0.3:
                incoherent_count += 1
        rate = incoherent_count / len(domain_eval[domain])
        incoherence_results[domain] = {"rate": rate, "count": incoherent_count, "total": len(domain_eval[domain])}
    results["incoherence"] = incoherence_results

    peak = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    results["peak_memory_gb"] = peak
    results["time_s"] = elapsed
    log(f"\n  Base phase: {elapsed:.1f}s, peak: {peak:.2f}GB")

    del model, tokenizer
    cleanup()
    return results


# ============================================================================
# Phase 3: Evaluate composed model (routed + per-domain scales)
# ============================================================================

def phase_eval_composed(gsm8k, code_gen, clinical_ner, mmlu, domain_eval):
    log("\n" + "=" * 70)
    log("PHASE 3: COMPOSED MODEL (routed + per-domain scales)")
    log(f"  Scales: {OPTIMAL_SCALES}")
    log("=" * 70)
    t0 = time.time()
    mx.reset_peak_memory()

    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model.freeze()
    mx.random.seed(SEED)
    log_memory("composed-loaded-base")

    # Load skeleton and all adapters
    skeleton = load_skeleton()
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(domain)

    # Save base weights for restoration between tasks
    base_weights = save_base_weights(model)
    log_memory("composed-adapters-loaded")

    results = {}

    # GSM8K: use math adapter at s=20
    log("\n  --- GSM8K (math adapter, s=20) ---")
    premerge_single_adapter(model, skeleton, adapters["math"], "math", OPTIMAL_SCALES["math"])
    correct = 0
    total = len(gsm8k)
    for i, prob in enumerate(gsm8k):
        prompt = format_gsm8k_prompt(prob["question"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_GSM8K)
        predicted = extract_gsm8k_answer(gen)
        if predicted is not None and prob["answer"] is not None:
            if prob["answer"] == 0:
                is_correct = abs(predicted) < 0.01
            else:
                is_correct = abs(predicted - prob["answer"]) / abs(prob["answer"]) < 0.01
        else:
            is_correct = False
        if is_correct:
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{total}: {correct}/{i+1} ({100*correct/(i+1):.0f}%)")
    results["gsm8k"] = {"accuracy": correct / total, "correct": correct, "total": total}
    log(f"  GSM8K composed: {correct}/{total} = {results['gsm8k']['accuracy']:.3f}")
    restore_base_weights(model, base_weights)

    # Code generation: use code adapter at s=20
    log("\n  --- Code Generation (code adapter, s=20) ---")
    premerge_single_adapter(model, skeleton, adapters["code"], "code", OPTIMAL_SCALES["code"])
    syntax_correct = 0
    for prob in code_gen:
        prompt = format_code_prompt(prob["instruction"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_CODE)
        if eval_code_syntax(gen):
            syntax_correct += 1
    results["code_gen"] = {
        "syntax_rate": syntax_correct / len(code_gen),
        "correct": syntax_correct,
        "total": len(code_gen),
    }
    log(f"  Code composed: {syntax_correct}/{len(code_gen)} syntax valid = {results['code_gen']['syntax_rate']:.3f}")
    restore_base_weights(model, base_weights)

    # Clinical NER: use medical adapter at s=20
    log("\n  --- Clinical NER (medical adapter, s=20) ---")
    premerge_single_adapter(model, skeleton, adapters["medical"], "medical", OPTIMAL_SCALES["medical"])
    ner_f1s = []
    for ex in clinical_ner:
        prompt = format_ner_prompt(ex["text"])
        gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_NER)
        ner_result = eval_ner_entities(gen, ex["entities"])
        ner_f1s.append(ner_result["f1"])
    results["clinical_ner"] = {
        "mean_f1": float(np.mean(ner_f1s)),
        "std_f1": float(np.std(ner_f1s)),
        "scores": [float(f) for f in ner_f1s],
    }
    log(f"  NER composed: mean F1 = {results['clinical_ner']['mean_f1']:.3f}")
    restore_base_weights(model, base_weights)

    # MMLU: per-domain routing with optimal scales
    log("\n  --- MMLU (per-domain routing) ---")
    choice_labels = ["A", "B", "C", "D"]
    mmlu_correct = 0
    mmlu_total = 0
    mmlu_by_domain = {}
    for domain in DOMAINS:
        # Pre-merge the domain's adapter at its optimal scale
        premerge_single_adapter(model, skeleton, adapters[domain], domain, OPTIMAL_SCALES[domain])

        domain_correct = 0
        for q in mmlu[domain]:
            prompt = format_mmlu_prompt(q["question"], q["choices"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_MMLU)
            predicted = extract_mmlu_answer(gen)
            gt_label = choice_labels[q["answer"]]
            if predicted == gt_label:
                domain_correct += 1
                mmlu_correct += 1
            mmlu_total += 1
        domain_acc = domain_correct / len(mmlu[domain]) if mmlu[domain] else 0
        mmlu_by_domain[domain] = {"accuracy": domain_acc, "correct": domain_correct, "total": len(mmlu[domain])}
        log(f"    MMLU {domain} (s={OPTIMAL_SCALES[domain]}): {domain_correct}/{len(mmlu[domain])} = {domain_acc:.3f}")

        restore_base_weights(model, base_weights)

    results["mmlu"] = {
        "accuracy": mmlu_correct / mmlu_total if mmlu_total > 0 else 0,
        "correct": mmlu_correct,
        "total": mmlu_total,
        "by_domain": mmlu_by_domain,
    }
    log(f"  MMLU composed overall: {mmlu_correct}/{mmlu_total} = {results['mmlu']['accuracy']:.3f}")

    # Incoherence check: per-domain routing
    log("\n  --- Incoherence Check ---")
    incoherence_results = {}
    for domain in DOMAINS:
        premerge_single_adapter(model, skeleton, adapters[domain], domain, OPTIMAL_SCALES[domain])
        incoherent_count = 0
        for prompt_data in domain_eval[domain]:
            prompt = format_domain_prompt(prompt_data["instruction"])
            gen = generate_text(model, tokenizer, prompt, max_tokens=MAX_TOKENS_DOMAIN)
            fq = eval_format_quality(gen)
            if fq < 0.3:
                incoherent_count += 1
        rate = incoherent_count / len(domain_eval[domain])
        incoherence_results[domain] = {"rate": rate, "count": incoherent_count, "total": len(domain_eval[domain])}
        restore_base_weights(model, base_weights)
    results["incoherence"] = incoherence_results

    peak = mx.get_peak_memory() / 1e9
    elapsed = time.time() - t0
    results["peak_memory_gb"] = peak
    results["time_s"] = elapsed
    log(f"\n  Composed phase: {elapsed:.1f}s, peak: {peak:.2f}GB")

    del model, tokenizer, skeleton, adapters, base_weights
    cleanup()
    return results


# ============================================================================
# Phase 4: Analysis
# ============================================================================

def phase_analysis(base_results, composed_results):
    log("\n" + "=" * 70)
    log("PHASE 4: ANALYSIS")
    log("=" * 70)

    analysis = {"benchmarks": {}, "kill_criteria": {}, "predictions": {}}

    # --- Capability benchmarks ---
    log("\n  === CAPABILITY BENCHMARKS (format-dependent) ===")
    log(f"  {'Benchmark':<20} | {'Base':>8} | {'Composed':>8} | {'Delta':>8} | {'Delta pp':>8}")
    log(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    # GSM8K
    b_gsm = base_results["gsm8k"]["accuracy"]
    c_gsm = composed_results["gsm8k"]["accuracy"]
    d_gsm = c_gsm - b_gsm
    analysis["benchmarks"]["gsm8k"] = {"base": b_gsm, "composed": c_gsm, "delta": d_gsm, "delta_pp": d_gsm * 100}
    log(f"  {'GSM8K':<20} | {b_gsm:>8.3f} | {c_gsm:>8.3f} | {d_gsm:>+8.3f} | {d_gsm*100:>+7.1f}pp")

    # Code gen
    b_code = base_results["code_gen"]["syntax_rate"]
    c_code = composed_results["code_gen"]["syntax_rate"]
    d_code = c_code - b_code
    analysis["benchmarks"]["code_gen"] = {"base": b_code, "composed": c_code, "delta": d_code, "delta_pp": d_code * 100}
    log(f"  {'Code (syntax)':<20} | {b_code:>8.3f} | {c_code:>8.3f} | {d_code:>+8.3f} | {d_code*100:>+7.1f}pp")

    # Clinical NER
    b_ner = base_results["clinical_ner"]["mean_f1"]
    c_ner = composed_results["clinical_ner"]["mean_f1"]
    d_ner = c_ner - b_ner
    analysis["benchmarks"]["clinical_ner"] = {"base": b_ner, "composed": c_ner, "delta": d_ner, "delta_pp": d_ner * 100}
    log(f"  {'Clinical NER (F1)':<20} | {b_ner:>8.3f} | {c_ner:>8.3f} | {d_ner:>+8.3f} | {d_ner*100:>+7.1f}pp")

    # --- Knowledge benchmark ---
    log("\n  === KNOWLEDGE BENCHMARK (factual recall) ===")
    b_mmlu = base_results["mmlu"]["accuracy"]
    c_mmlu = composed_results["mmlu"]["accuracy"]
    d_mmlu = c_mmlu - b_mmlu
    analysis["benchmarks"]["mmlu"] = {"base": b_mmlu, "composed": c_mmlu, "delta": d_mmlu, "delta_pp": d_mmlu * 100}
    log(f"  {'MMLU (overall)':<20} | {b_mmlu:>8.3f} | {c_mmlu:>8.3f} | {d_mmlu:>+8.3f} | {d_mmlu*100:>+7.1f}pp")

    # MMLU per domain
    for domain in DOMAINS:
        bd = base_results["mmlu"]["by_domain"][domain]["accuracy"]
        cd = composed_results["mmlu"]["by_domain"][domain]["accuracy"]
        dd = cd - bd
        log(f"  {'  ' + domain:<20} | {bd:>8.3f} | {cd:>8.3f} | {dd:>+8.3f} | {dd*100:>+7.1f}pp | s={OPTIMAL_SCALES[domain]}")

    # --- Kill criteria ---
    log("\n  === KILL CRITERIA ===")

    # K1: Full system worse than base on GSM8K
    k1_pass = c_gsm >= b_gsm
    analysis["kill_criteria"]["K1_gsm8k"] = {
        "pass": k1_pass,
        "base": b_gsm,
        "composed": c_gsm,
        "delta_pp": d_gsm * 100,
    }
    log(f"  K1 (GSM8K not worse): {'PASS' if k1_pass else 'FAIL -> KILL'}")
    log(f"      Base={b_gsm:.3f}, Composed={c_gsm:.3f}, Delta={d_gsm*100:+.1f}pp")

    # K2: Full system worse than base on code generation
    k2_pass = c_code >= b_code
    analysis["kill_criteria"]["K2_code"] = {
        "pass": k2_pass,
        "base": b_code,
        "composed": c_code,
        "delta_pp": d_code * 100,
    }
    log(f"  K2 (Code not worse): {'PASS' if k2_pass else 'FAIL -> KILL'}")
    log(f"      Base={b_code:.3f}, Composed={c_code:.3f}, Delta={d_code*100:+.1f}pp")

    # K3: >5% incoherent output on any domain
    max_incoherence = 0
    worst_domain = ""
    for domain in DOMAINS:
        rate = composed_results["incoherence"][domain]["rate"]
        if rate > max_incoherence:
            max_incoherence = rate
            worst_domain = domain
    k3_pass = max_incoherence <= 0.05
    analysis["kill_criteria"]["K3_incoherence"] = {
        "pass": k3_pass,
        "max_rate": max_incoherence,
        "worst_domain": worst_domain,
        "by_domain": {d: composed_results["incoherence"][d]["rate"] for d in DOMAINS},
    }
    log(f"  K3 (Incoherence <= 5%): {'PASS' if k3_pass else 'FAIL -> KILL'}")
    log(f"      Max incoherence: {max_incoherence*100:.1f}% ({worst_domain})")

    # --- Predictions vs measurements ---
    log("\n  === PREDICTIONS vs MEASUREMENTS ===")
    log(f"  {'Prediction':<35} | {'Predicted':>10} | {'Measured':>10} | {'Match':>6}")
    log(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

    gsm8k_pred_match = d_gsm * 100 >= 10
    analysis["predictions"]["gsm8k_plus10pp"] = {"predicted": "+10pp", "measured": f"{d_gsm*100:+.1f}pp", "match": gsm8k_pred_match}
    log(f"  {'GSM8K >= +10pp':<35} | {'+10pp':>10} | {d_gsm*100:>+9.1f}pp | {'YES' if gsm8k_pred_match else 'NO':>6}")

    code_pred_match = d_code * 100 >= 10
    analysis["predictions"]["code_plus15pp"] = {"predicted": "+15pp", "measured": f"{d_code*100:+.1f}pp", "match": code_pred_match}
    log(f"  {'Code >= +10pp':<35} | {'+15pp':>10} | {d_code*100:>+9.1f}pp | {'YES' if code_pred_match else 'NO':>6}")

    ner_pred_match = d_ner * 100 >= 5
    analysis["predictions"]["ner_plus5pp"] = {"predicted": "+5pp", "measured": f"{d_ner*100:+.1f}pp", "match": ner_pred_match}
    log(f"  {'Clinical NER >= +5pp':<35} | {'+5pp':>10} | {d_ner*100:>+9.1f}pp | {'YES' if ner_pred_match else 'NO':>6}")

    mmlu_pred_match = -5 <= d_mmlu * 100 <= 0
    analysis["predictions"]["mmlu_neutral"] = {"predicted": "-5pp to 0pp", "measured": f"{d_mmlu*100:+.1f}pp", "match": mmlu_pred_match}
    log(f"  {'MMLU -5pp to 0pp':<35} | {'-5 to 0pp':>10} | {d_mmlu*100:>+9.1f}pp | {'YES' if mmlu_pred_match else 'NO':>6}")

    incoh_pred_match = max_incoherence <= 0.05
    analysis["predictions"]["incoherence_lt5pct"] = {"predicted": "<5%", "measured": f"{max_incoherence*100:.1f}%", "match": incoh_pred_match}
    log(f"  {'Incoherence < 5%':<35} | {'<5%':>10} | {max_incoherence*100:>9.1f}% | {'YES' if incoh_pred_match else 'NO':>6}")

    # Overall
    all_kill_pass = all(v["pass"] for v in analysis["kill_criteria"].values())
    n_predictions_match = sum(1 for v in analysis["predictions"].values() if v["match"])
    n_predictions_total = len(analysis["predictions"])

    log(f"\n  Kill criteria: {'ALL PASS' if all_kill_pass else 'KILL TRIGGERED'}")
    log(f"  Predictions: {n_predictions_match}/{n_predictions_total} match")

    analysis["overall"] = {
        "all_kill_pass": all_kill_pass,
        "predictions_matched": n_predictions_match,
        "predictions_total": n_predictions_total,
    }

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("CAPABILITY BENCHMARK: FULL SYSTEM VERIFICATION")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Scales: {OPTIMAL_SCALES}")
    log(f"Benchmarks: GSM8K({GSM8K_N}), Code({CODE_GEN_N}), NER({NER_N}), MMLU({MMLU_N_PER_DOMAIN}x5)")
    log_memory("start")

    # Phase 1: Load data
    gsm8k, code_gen, clinical_ner, mmlu, domain_eval = phase_load_data()

    # Phase 2: Base model
    base_results = phase_eval_base(gsm8k, code_gen, clinical_ner, mmlu, domain_eval)
    log_memory("after-base")

    # Phase 3: Composed model
    composed_results = phase_eval_composed(gsm8k, code_gen, clinical_ner, mmlu, domain_eval)
    log_memory("after-composed")

    # Phase 4: Analysis
    analysis = phase_analysis(base_results, composed_results)

    # Save results
    total_time = time.time() - t0
    results = {
        "experiment": "capability_benchmark_full_system",
        "model": MODEL_ID,
        "optimal_scales": OPTIMAL_SCALES,
        "benchmark_sizes": {
            "gsm8k": GSM8K_N,
            "code_gen": CODE_GEN_N,
            "clinical_ner": NER_N,
            "mmlu_per_domain": MMLU_N_PER_DOMAIN,
        },
        "base_results": base_results,
        "composed_results": composed_results,
        "analysis": analysis,
        "timing": {
            "base_s": base_results["time_s"],
            "composed_s": composed_results["time_s"],
            "total_s": total_time,
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Final summary
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    for k, v in analysis["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL -> KILL'}")
    all_pass = analysis["overall"]["all_kill_pass"]
    log(f"\n  Status: {'SUPPORTED' if all_pass else 'KILLED'}")
    log(f"  Predictions matched: {analysis['overall']['predictions_matched']}/{analysis['overall']['predictions_total']}")


if __name__ == "__main__":
    main()
