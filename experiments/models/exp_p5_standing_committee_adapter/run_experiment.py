#!/usr/bin/env python3
"""
P5.C0: Standing Committee Adapter — Always-On Reasoning Backbone.

arXiv:2601.03425 shows MoE models naturally form domain-invariant reasoning cores.
Finding #49: capability experts compose orthogonally with domain adapters (|cos|<0.01).
Finding #18: reasoning distillation gains +10.6pp via LoRA.

We test: a permanent reasoning adapter (committee), composed with a domain adapter,
improves reasoning tasks without degrading domain format tasks.

Key design: module-disjoint adapters guarantee cos=0 by construction.
  - Committee: q_proj + o_proj (reasoning attention patterns)
  - Domain: v_proj + down_proj (value representations + MLP)

Kill criteria:
  K1276: Committee + domain >= domain alone + 3pp on reasoning tasks
  K1277: Committee does not degrade format tasks (< 2pp loss)
  K1278: cos(committee, domain) < 1e-4
"""

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 4
N_LORA_LAYERS = 8
SEED = 42
LR = 1e-3
MAX_SEQ_LEN = 512
LORA_SCALE = 1.0

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Module assignments (disjoint by construction)
COMMITTEE_MODULES = ["self_attn.q_proj", "self_attn.o_proj"]
DOMAIN_MODULES = ["self_attn.v_proj", "mlp.down_proj"]


def cleanup(*objects):
    for obj in objects:
        del obj
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
# TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════

# Reasoning training: GSM8K-style chain-of-thought math problems
REASONING_TRAIN = [
    ("A bakery sells 45 loaves of bread each day. How many loaves does it sell in 2 weeks?",
     "Let me solve this step by step.\nStep 1: Calculate the number of days in 2 weeks: 2 × 7 = 14 days\nStep 2: Multiply daily sales by number of days: 45 × 14 = 630\nThe answer is 630 loaves."),
    ("Sarah has 3 boxes with 12 apples each. She gives away 8 apples. How many does she have left?",
     "Let me solve this step by step.\nStep 1: Calculate total apples: 3 × 12 = 36 apples\nStep 2: Subtract given away: 36 - 8 = 28\nThe answer is 28 apples."),
    ("A train travels 60 miles per hour. How far does it go in 3 hours and 30 minutes?",
     "Let me solve this step by step.\nStep 1: Convert 3 hours 30 minutes to hours: 3.5 hours\nStep 2: Calculate distance: 60 × 3.5 = 210 miles\nThe answer is 210 miles."),
    ("If 5 workers can build a wall in 8 days, how many days would 10 workers need?",
     "Let me solve this step by step.\nStep 1: Calculate total worker-days needed: 5 × 8 = 40 worker-days\nStep 2: Divide by new number of workers: 40 / 10 = 4 days\nThe answer is 4 days."),
    ("A rectangle has length 15cm and width 8cm. What is its perimeter?",
     "Let me solve this step by step.\nStep 1: Perimeter formula: P = 2 × (length + width)\nStep 2: Calculate: P = 2 × (15 + 8) = 2 × 23 = 46\nThe answer is 46 cm."),
    ("Tom buys 3 shirts at $25 each and 2 pants at $40 each. What is the total cost?",
     "Let me solve this step by step.\nStep 1: Cost of shirts: 3 × $25 = $75\nStep 2: Cost of pants: 2 × $40 = $80\nStep 3: Total: $75 + $80 = $155\nThe answer is $155."),
    ("A jar contains 120 marbles. 40% are red, 35% are blue, and the rest are green. How many green marbles?",
     "Let me solve this step by step.\nStep 1: Green percentage: 100% - 40% - 35% = 25%\nStep 2: Number of green marbles: 120 × 0.25 = 30\nThe answer is 30 green marbles."),
    ("If you save $15 per week, how many weeks to save $300?",
     "Let me solve this step by step.\nStep 1: Divide target by weekly savings: 300 / 15 = 20\nThe answer is 20 weeks."),
    ("A class has 28 students. The ratio of boys to girls is 3:4. How many girls are there?",
     "Let me solve this step by step.\nStep 1: Total ratio parts: 3 + 4 = 7\nStep 2: Value of one part: 28 / 7 = 4 students\nStep 3: Girls = 4 parts: 4 × 4 = 16\nThe answer is 16 girls."),
    ("A car uses 8 liters of fuel per 100km. How much fuel for a 350km trip?",
     "Let me solve this step by step.\nStep 1: Fuel rate: 8 liters per 100km\nStep 2: Calculate: (8 / 100) × 350 = 28 liters\nThe answer is 28 liters."),
    ("What is 15% of 240?",
     "Let me solve this step by step.\nStep 1: Convert percentage: 15% = 0.15\nStep 2: Multiply: 0.15 × 240 = 36\nThe answer is 36."),
    ("A pool fills at 3 gallons per minute. How long to fill a 540 gallon pool?",
     "Let me solve this step by step.\nStep 1: Time = volume / rate = 540 / 3 = 180 minutes\nStep 2: Convert to hours: 180 / 60 = 3 hours\nThe answer is 180 minutes or 3 hours."),
    ("A store offers 20% off a $85 jacket. What is the sale price?",
     "Let me solve this step by step.\nStep 1: Calculate discount: 20% of $85 = 0.20 × 85 = $17\nStep 2: Subtract: $85 - $17 = $68\nThe answer is $68."),
    ("Three consecutive even numbers sum to 78. What is the largest?",
     "Let me solve this step by step.\nStep 1: Let the numbers be n, n+2, n+4\nStep 2: n + (n+2) + (n+4) = 78 → 3n + 6 = 78 → 3n = 72 → n = 24\nStep 3: Largest: 24 + 4 = 28\nThe answer is 28."),
    ("A recipe needs 2/3 cup flour per batch. How much flour for 6 batches?",
     "Let me solve this step by step.\nStep 1: Multiply: (2/3) × 6 = 12/3 = 4 cups\nThe answer is 4 cups of flour."),
    # Reinforcement (rephrase key patterns)
    ("There are 180 students in a school. The ratio of girls to boys is 5:4. Find the number of boys.",
     "Let me solve this step by step.\nStep 1: Total ratio parts: 5 + 4 = 9\nStep 2: Value of one part: 180 / 9 = 20\nStep 3: Boys = 4 parts: 4 × 20 = 80\nThe answer is 80 boys."),
    ("A bike costs $240 after a 25% discount. What was the original price?",
     "Let me solve this step by step.\nStep 1: After 25% off, you pay 75% of original\nStep 2: Original = $240 / 0.75 = $320\nThe answer is $320."),
    ("If 8 machines produce 200 units in 5 hours, how many units do 12 machines produce in 5 hours?",
     "Let me solve this step by step.\nStep 1: Rate per machine: 200 / 8 = 25 units per machine in 5 hours\nStep 2: 12 machines: 12 × 25 = 300 units\nThe answer is 300 units."),
    ("A triangle has sides 7cm, 10cm, and 13cm. What is its perimeter?",
     "Let me solve this step by step.\nStep 1: Perimeter = sum of all sides = 7 + 10 + 13 = 30 cm\nThe answer is 30 cm."),
    ("24 students scored an average of 75 on a test. What is the total of all scores?",
     "Let me solve this step by step.\nStep 1: Total = average × count = 75 × 24 = 1800\nThe answer is 1800."),
]

# Domain format training: structured legal/professional output
DOMAIN_FORMAT_TRAIN = [
    ("Summarize the key terms of a lease agreement for 123 Main St.",
     "LEASE SUMMARY\n- Property: 123 Main St\n- Term: 12 months\n- Monthly Rent: $2,500\n- Security Deposit: $5,000 (2 months)\n- Utilities: Tenant responsibility\n- Maintenance: Landlord covers structural; tenant covers interior\n- Termination: 60-day written notice required"),
    ("Draft a meeting minutes header for the Q3 Board Review.",
     "MEETING MINUTES\nDate: Q3 Board Review\nAttendees: Board of Directors\nQuorum: Present\nChair: [Name]\nSecretary: [Name]\n\nAGENDA:\n1. Call to Order\n2. Q3 Financial Review\n3. Strategic Initiatives Update\n4. New Business\n5. Adjournment"),
    ("Format a citation for Smith v. Jones, 2024 in Bluebook style.",
     "Smith v. Jones, No. 24-1234, slip op. (D. Del. 2024).\n\nNote: Bluebook citation format requires:\n- Case name in italics\n- Docket number\n- Court abbreviation\n- Year of decision"),
    ("Create a compliance checklist for GDPR data processing.",
     "GDPR COMPLIANCE CHECKLIST\n□ Legal basis for processing identified (Art. 6)\n□ Data Protection Impact Assessment completed (Art. 35)\n□ Privacy notice provided to data subjects (Art. 13/14)\n□ Data processing agreement with processors (Art. 28)\n□ Records of processing activities maintained (Art. 30)\n□ Data breach notification procedure in place (Art. 33/34)\n□ Data subject rights procedures documented (Art. 15-22)\n□ Data Protection Officer appointed if required (Art. 37)"),
    ("Write a professional memo header about policy changes.",
     "MEMORANDUM\n\nTO: All Department Heads\nFROM: Office of General Counsel\nDATE: [Current Date]\nRE: Updated Data Retention Policy\n\nThis memorandum provides notice of changes to the company data retention policy effective [Date]."),
    ("Format an invoice summary for consulting services.",
     "INVOICE SUMMARY\nInvoice #: INV-2024-0847\nDate: [Current Date]\nDue: Net 30\n\nService Description          Hours    Rate      Amount\nLegal Research                 12    $350/hr    $4,200\nContract Review                 8    $350/hr    $2,800\nClient Consultation             4    $400/hr    $1,600\n                                              --------\nSubtotal:                                      $8,600\nTax (8%):                                        $688\nTOTAL DUE:                                     $9,288"),
    ("Create an executive summary for a market analysis report.",
     "EXECUTIVE SUMMARY\n\nKey Findings:\n1. Market size: $4.2B (2024), projected $6.8B by 2027 (CAGR 17.4%)\n2. Top 3 competitors control 62% market share\n3. Customer acquisition cost declined 15% YoY\n4. Emerging opportunity in enterprise segment ($1.1B TAM)\n\nRecommendations:\n- Increase enterprise sales team by 40%\n- Launch partnership program in Q2\n- Invest in product differentiation (AI features)"),
    ("Format a standard NDA summary.",
     "NON-DISCLOSURE AGREEMENT SUMMARY\n- Parties: [Discloser] and [Recipient]\n- Scope: All proprietary business information, trade secrets, technical data\n- Duration: 3 years from execution date\n- Obligations: Recipient must protect confidential information using same standard of care as own confidential information\n- Exclusions: Public domain information, independently developed, prior knowledge\n- Remedies: Injunctive relief available; prevailing party recovers attorneys' fees"),
    ("Draft a risk assessment matrix header.",
     "RISK ASSESSMENT MATRIX\n\nProject: [Name]\nAssessor: [Name]\nDate: [Date]\nRevision: 1.0\n\n| Risk ID | Description | Likelihood | Impact | Score | Mitigation |\n|---------|-------------|-----------|--------|-------|------------|\n| R-001   | [Risk]      | High      | High   | 9     | [Action]   |\n| R-002   | [Risk]      | Medium    | High   | 6     | [Action]   |"),
    ("Create a formal project status report header.",
     "PROJECT STATUS REPORT\n\nProject: [Name]\nPeriod: [Start] to [End]\nStatus: ON TRACK / AT RISK / DELAYED\n\nMilestones:\n- [Milestone 1]: Complete ✓\n- [Milestone 2]: In Progress (75%)\n- [Milestone 3]: Not Started\n\nBudget: $X / $Y utilized (Z%)\nNext Review: [Date]"),
    # Reinforcement
    ("Reformat this contract clause: The tenant shall pay rent on the first of each month.",
     "CLAUSE 4.1 — RENT PAYMENT\n\nThe Tenant shall remit monthly rent to the Landlord on or before the first (1st) day of each calendar month. Payment shall be made via [method] to [account/address]. Late payment (after the 5th) incurs a fee of [amount] per day."),
    ("Create an agenda for a compliance review meeting.",
     "COMPLIANCE REVIEW MEETING\nDate: [Date]\nTime: [Time]\nLocation: [Room/Link]\n\nAGENDA:\n1. Opening and Roll Call (5 min)\n2. Review of Previous Action Items (10 min)\n3. Regulatory Update: Recent Changes (15 min)\n4. Audit Findings Review (20 min)\n5. Policy Amendment Proposals (15 min)\n6. Training Schedule Update (10 min)\n7. Action Items and Next Steps (5 min)\n8. Adjournment"),
    ("Write a warranty disclaimer in formal format.",
     "WARRANTY DISCLAIMER\n\nTHE PRODUCTS AND SERVICES ARE PROVIDED \"AS IS\" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.\n\nIN NO EVENT SHALL [COMPANY] BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING OUT OF OR IN CONNECTION WITH THE USE OF THE PRODUCTS OR SERVICES."),
    ("Format a board resolution template.",
     "BOARD RESOLUTION\n\nRESOLVED, that the Board of Directors of [Company Name], at a duly convened meeting held on [Date], at which a quorum was present, hereby:\n\n1. APPROVES [action description];\n2. AUTHORIZES [officer/role] to execute all documents necessary;\n3. RATIFIES all actions taken in furtherance of the foregoing.\n\nCertified by: _______________\n[Secretary Name], Corporate Secretary\nDate: [Date]"),
    ("Create a data classification table.",
     "DATA CLASSIFICATION TABLE\n\n| Level | Label | Description | Examples | Handling |\n|-------|-------|-------------|----------|----------|\n| 1 | PUBLIC | Freely shareable | Marketing materials | No restrictions |\n| 2 | INTERNAL | Business use only | Org charts, policies | Access control |\n| 3 | CONFIDENTIAL | Need-to-know basis | Financial data, HR | Encryption required |\n| 4 | RESTRICTED | Regulatory protected | PII, PHI, trade secrets | Encryption + audit |"),
]

# Reasoning evaluation: test step-by-step math
REASONING_TEST = [
    ("A school has 5 classes with 30 students each. If 10% of students are absent, how many are present?",
     ["135"]),
    ("What is 3/4 of 160?",
     ["120"]),
    ("A car travels 240km in 3 hours. What is its speed in km/h?",
     ["80"]),
    ("If a shirt costs $45 and is on 30% sale, what do you pay?",
     ["31.5", "31.50", "$31.5"]),
    ("How many seconds are in 2.5 hours?",
     ["9000", "9,000"]),
    ("A garden is 12m by 8m. What is its area in square meters?",
     ["96"]),
    ("If 6 people share $420 equally, how much does each person get?",
     ["70", "$70"]),
    ("What is 125% of 80?",
     ["100"]),
    ("A train leaves at 9:15 AM and arrives at 11:45 AM. How long is the trip in minutes?",
     ["150"]),
    ("Two pipes fill a tank: one at 10L/min, another at 15L/min. How long to fill 500L together?",
     ["20"]),
]

# Format evaluation: check structured output quality
FORMAT_TEST = [
    ("Write a brief meeting minutes header for a budget review.",
     ["MEETING", "MINUTES", "Date", "Agenda", "Attendees"]),
    ("Format a simple invoice for 5 hours of consulting at $200/hr.",
     ["INVOICE", "$1,000", "1000", "Hours", "Rate"]),
    ("Create a brief compliance checklist for data backups.",
     ["CHECKLIST", "□", "backup", "Backup"]),
    ("Write a short NDA summary for two companies.",
     ["NDA", "Non-Disclosure", "Parties", "Confidential", "confidential"]),
    ("Draft a risk item entry for a cybersecurity risk.",
     ["Risk", "RISK", "Likelihood", "Impact", "Mitigation", "likelihood", "impact", "mitigation"]),
    ("Create an executive summary for a quarterly sales report.",
     ["EXECUTIVE", "Summary", "SUMMARY", "Key", "Findings", "findings"]),
    ("Format a warranty disclaimer.",
     ["WARRANTY", "AS IS", "as is", "WITHOUT WARRANTY", "without warranty"]),
    ("Write a project status update header.",
     ["PROJECT", "STATUS", "Milestone", "milestone", "Budget", "budget"]),
    ("Draft a formal memo header about a policy update.",
     ["MEMORANDUM", "Memo", "TO:", "FROM:", "RE:", "SUBJECT:"]),
    ("Create a data classification entry for confidential data.",
     ["CONFIDENTIAL", "Confidential", "Encryption", "encryption", "Access", "access"]),
]

if IS_SMOKE:
    REASONING_TRAIN = REASONING_TRAIN[:5]
    DOMAIN_FORMAT_TRAIN = DOMAIN_FORMAT_TRAIN[:5]
    REASONING_TEST = REASONING_TEST[:3]
    FORMAT_TEST = FORMAT_TEST[:3]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(model, tokenizer, prompt, max_tokens=120):
    """Autoregressive generation with cleanup."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_tokens = tokenizer.encode(formatted)
    n_prompt = len(input_tokens)
    all_tokens = list(input_tokens)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_tokens):
        input_ids = mx.array(all_tokens)[None, :]
        logits = model(input_ids)
        mx.eval(logits)
        next_id = mx.argmax(logits[0, -1, :]).item()
        del logits
        if next_id == eos_id:
            break
        all_tokens.append(next_id)

    generated_tokens = all_tokens[n_prompt:]
    return tokenizer.decode(generated_tokens)


def evaluate_qa(model, tokenizer, questions, max_tokens=120):
    """Evaluate keyword-matching QA accuracy."""
    correct = 0
    details = []
    for question, keywords in questions:
        response = generate_response(model, tokenizer, question, max_tokens=max_tokens)
        response_text = response.strip()[:300]
        found = any(kw.lower() in response_text.lower() for kw in keywords)
        if found:
            correct += 1
        details.append({
            "question": question,
            "response": response_text,
            "keywords": keywords,
            "found": found,
        })
    accuracy = correct / len(questions) if questions else 0.0
    return accuracy, correct, len(questions), details


def apply_lora(model, lora_modules, lora_rank, lora_scale, n_lora_layers):
    """Apply LoRA to specified modules. Returns (backbone, n_layers, lora_count)."""
    from mlx_lm.tuner.lora import LoRALinear

    if hasattr(model, "language_model"):
        backbone = model.language_model.model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        backbone = model.model
    else:
        raise RuntimeError(f"Cannot find backbone: {type(model)}")

    n_layers = backbone.config.num_hidden_layers
    lora_start = max(0, n_layers - n_lora_layers)
    lora_count = 0

    for i in range(lora_start, n_layers):
        layer = backbone.layers[i]
        for module_name in lora_modules:
            parts = module_name.split(".")
            target = layer
            for p in parts:
                target = getattr(target, p)
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(
                parent,
                parts[-1],
                LoRALinear.from_base(target, r=lora_rank, scale=lora_scale),
            )
            lora_count += 1

    return backbone, n_layers, lora_count


def train_adapter(model, tokenizer, training_data, n_epochs=2):
    """Train LoRA adapter on QA pairs. Returns (losses, latencies)."""
    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens, prompt_len):
        logits = model(tokens[:, :-1])
        response_logits = logits[:, prompt_len - 1:, :]
        response_targets = tokens[:, prompt_len:]
        n_response = response_targets.shape[1]
        if n_response == 0:
            return mx.array(0.0)
        loss = nn.losses.cross_entropy(
            response_logits.reshape(-1, response_logits.shape[-1]),
            response_targets.reshape(-1),
            reduction="mean",
        )
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    losses = []
    latencies = []
    gc.disable()
    for epoch in range(n_epochs):
        for qi, (question, answer) in enumerate(training_data):
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_messages = [{"role": "user", "content": question}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            full_tokens = tokenizer.encode(full_text)
            prompt_len = len(tokenizer.encode(prompt_text))
            if len(full_tokens) > MAX_SEQ_LEN:
                full_tokens = full_tokens[:MAX_SEQ_LEN]

            tokens_mx = mx.array(full_tokens)[None, :]

            t_start = time.perf_counter()
            loss, grads = loss_and_grad_fn(model, tokens_mx, prompt_len)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            t_step = time.perf_counter() - t_start

            loss_val = loss.item()
            losses.append(loss_val)
            latencies.append(t_step * 1000)

            if qi % 5 == 0:
                log(f"    E{epoch+1} step {qi+1}/{len(training_data)}: loss={loss_val:.4f} ({t_step*1000:.0f}ms)")
    gc.enable()

    cleanup(optimizer)
    return losses, latencies


def save_lora_weights(model, save_path):
    """Save only trainable (LoRA) parameters."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    weight_path = save_path / "weights.safetensors"
    trainable_dict = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(weight_path), trainable_dict)
    size_mb = weight_path.stat().st_size / (1024 * 1024)
    log(f"  Saved {len(trainable_dict)} tensors ({size_mb:.2f} MB) to {weight_path}")
    return size_mb


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BASE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def phase_base_eval():
    """Evaluate base model on reasoning and format tasks."""
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 1: BASE MODEL EVALUATION")
    log("=" * 60)

    model, tokenizer = load(MODEL_ID)
    log_memory("post-load")

    log("Evaluating reasoning (base)...")
    r_acc, r_correct, r_total, r_details = evaluate_qa(model, tokenizer, REASONING_TEST)
    log(f"  Reasoning accuracy: {r_acc:.1%} ({r_correct}/{r_total})")

    log("Evaluating format (base)...")
    f_acc, f_correct, f_total, f_details = evaluate_qa(model, tokenizer, FORMAT_TEST, max_tokens=200)
    log(f"  Format accuracy: {f_acc:.1%} ({f_correct}/{f_total})")

    results = {
        "reasoning_accuracy": r_acc,
        "reasoning_correct": r_correct,
        "reasoning_total": r_total,
        "reasoning_details": r_details,
        "format_accuracy": f_acc,
        "format_correct": f_correct,
        "format_total": f_total,
        "format_details": f_details,
    }

    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TRAIN COMMITTEE (REASONING) ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

def phase_train_committee():
    """Train reasoning committee adapter on q_proj + o_proj."""
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 2: TRAIN COMMITTEE ADAPTER (q_proj + o_proj)")
    log("=" * 60)

    model, tokenizer = load(MODEL_ID)
    model.freeze()
    backbone, n_layers, lora_count = apply_lora(
        model, COMMITTEE_MODULES, LORA_RANK, LORA_SCALE, N_LORA_LAYERS
    )

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    log(f"  LoRA: r={LORA_RANK}, modules={COMMITTEE_MODULES}, {lora_count} modules, {n_trainable:,} params")
    log_memory("post-lora")

    log("  Training on reasoning data...")
    losses, latencies = train_adapter(model, tokenizer, REASONING_TRAIN)

    adapter_dir = EXPERIMENT_DIR / "committee_adapter"
    adapter_size = save_lora_weights(model, adapter_dir)

    results = {
        "n_trainable": n_trainable,
        "n_turns": len(REASONING_TRAIN),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "loss_decrease_pct": round((losses[0] - losses[-1]) / losses[0] * 100, 1) if losses[0] > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "adapter_size_mb": round(adapter_size, 2),
    }

    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: TRAIN DOMAIN (FORMAT) ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

def phase_train_domain():
    """Train domain format adapter on v_proj + down_proj."""
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 3: TRAIN DOMAIN ADAPTER (v_proj + down_proj)")
    log("=" * 60)

    model, tokenizer = load(MODEL_ID)
    model.freeze()
    backbone, n_layers, lora_count = apply_lora(
        model, DOMAIN_MODULES, LORA_RANK, LORA_SCALE, N_LORA_LAYERS
    )

    trainable = list(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for _, v in trainable)
    log(f"  LoRA: r={LORA_RANK}, modules={DOMAIN_MODULES}, {lora_count} modules, {n_trainable:,} params")
    log_memory("post-lora")

    log("  Training on format data...")
    losses, latencies = train_adapter(model, tokenizer, DOMAIN_FORMAT_TRAIN)

    adapter_dir = EXPERIMENT_DIR / "domain_adapter"
    adapter_size = save_lora_weights(model, adapter_dir)

    results = {
        "n_trainable": n_trainable,
        "n_turns": len(DOMAIN_FORMAT_TRAIN),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "loss_decrease_pct": round((losses[0] - losses[-1]) / losses[0] * 100, 1) if losses[0] > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "adapter_size_mb": round(adapter_size, 2),
    }

    cleanup(model, tokenizer)
    log_memory("post-cleanup")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: EVALUATE ALL CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_adapter_weights(model, adapter_dir, modules):
    """Load saved LoRA weights into model with matching LoRA modules."""
    weight_path = Path(adapter_dir) / "weights.safetensors"
    saved = mx.load(str(weight_path))
    # Apply LoRA structure first
    model.freeze()
    apply_lora(model, modules, LORA_RANK, LORA_SCALE, N_LORA_LAYERS)
    # Load saved weights
    model.load_weights(list(saved.items()), strict=False)
    mx.eval(model.parameters())


def phase_eval_configs():
    """Evaluate: committee-only, domain-only, committee+domain."""
    from mlx_lm.utils import load

    log("\n" + "=" * 60)
    log("PHASE 4: EVALUATE ALL CONFIGURATIONS")
    log("=" * 60)

    results = {}

    # Config 1: Committee only
    log("\n--- Config: Committee adapter only ---")
    model, tokenizer = load(MODEL_ID)
    load_adapter_weights(model, EXPERIMENT_DIR / "committee_adapter", COMMITTEE_MODULES)
    log_memory("committee-loaded")

    r_acc, r_c, r_t, r_d = evaluate_qa(model, tokenizer, REASONING_TEST)
    f_acc, f_c, f_t, f_d = evaluate_qa(model, tokenizer, FORMAT_TEST, max_tokens=200)
    log(f"  Committee: reasoning={r_acc:.1%} format={f_acc:.1%}")
    results["committee_only"] = {
        "reasoning_accuracy": r_acc, "reasoning_correct": r_c,
        "reasoning_details": r_d,
        "format_accuracy": f_acc, "format_correct": f_c,
        "format_details": f_d,
    }
    cleanup(model, tokenizer)

    # Config 2: Domain only
    log("\n--- Config: Domain adapter only ---")
    model, tokenizer = load(MODEL_ID)
    load_adapter_weights(model, EXPERIMENT_DIR / "domain_adapter", DOMAIN_MODULES)
    log_memory("domain-loaded")

    r_acc, r_c, r_t, r_d = evaluate_qa(model, tokenizer, REASONING_TEST)
    f_acc, f_c, f_t, f_d = evaluate_qa(model, tokenizer, FORMAT_TEST, max_tokens=200)
    log(f"  Domain: reasoning={r_acc:.1%} format={f_acc:.1%}")
    results["domain_only"] = {
        "reasoning_accuracy": r_acc, "reasoning_correct": r_c,
        "reasoning_details": r_d,
        "format_accuracy": f_acc, "format_correct": f_c,
        "format_details": f_d,
    }
    cleanup(model, tokenizer)

    # Config 3: Committee + Domain composed
    log("\n--- Config: Committee + Domain composed ---")
    model, tokenizer = load(MODEL_ID)
    # Apply both LoRA module sets (disjoint, so no conflict)
    model.freeze()
    apply_lora(model, COMMITTEE_MODULES, LORA_RANK, LORA_SCALE, N_LORA_LAYERS)
    apply_lora(model, DOMAIN_MODULES, LORA_RANK, LORA_SCALE, N_LORA_LAYERS)

    # Load committee weights
    committee_weights = mx.load(str(EXPERIMENT_DIR / "committee_adapter" / "weights.safetensors"))
    model.load_weights(list(committee_weights.items()), strict=False)
    del committee_weights

    # Load domain weights
    domain_weights = mx.load(str(EXPERIMENT_DIR / "domain_adapter" / "weights.safetensors"))
    model.load_weights(list(domain_weights.items()), strict=False)
    del domain_weights

    mx.eval(model.parameters())
    log_memory("composed-loaded")

    r_acc, r_c, r_t, r_d = evaluate_qa(model, tokenizer, REASONING_TEST)
    f_acc, f_c, f_t, f_d = evaluate_qa(model, tokenizer, FORMAT_TEST, max_tokens=200)
    log(f"  Composed: reasoning={r_acc:.1%} format={f_acc:.1%}")
    results["composed"] = {
        "reasoning_accuracy": r_acc, "reasoning_correct": r_c,
        "reasoning_details": r_d,
        "format_accuracy": f_acc, "format_correct": f_c,
        "format_details": f_d,
    }
    cleanup(model, tokenizer)
    log_memory("post-cleanup")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: ORTHOGONALITY MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════

def phase_orthogonality():
    """Measure cosine similarity between committee and domain adapter weights."""
    log("\n" + "=" * 60)
    log("PHASE 5: ORTHOGONALITY MEASUREMENT")
    log("=" * 60)

    committee_weights = mx.load(str(EXPERIMENT_DIR / "committee_adapter" / "weights.safetensors"))
    domain_weights = mx.load(str(EXPERIMENT_DIR / "domain_adapter" / "weights.safetensors"))

    # Flatten all weights into single vectors
    c_flat = mx.concatenate([v.reshape(-1) for v in committee_weights.values()])
    d_flat = mx.concatenate([v.reshape(-1) for v in domain_weights.values()])

    # Since modules are disjoint, these vectors have different keys entirely
    # Pad to same length with zeros for proper cosine computation
    c_keys = set(committee_weights.keys())
    d_keys = set(domain_weights.keys())
    overlap = c_keys & d_keys

    log(f"  Committee keys: {len(c_keys)}")
    log(f"  Domain keys: {len(d_keys)}")
    log(f"  Overlapping keys: {len(overlap)}")

    if len(overlap) == 0:
        # Module-disjoint: cosine is exactly 0 when embedded in full param space
        cos_sim = 0.0
        log(f"  Module-disjoint: cos = 0.0 (by construction)")
    else:
        # Compute cosine on overlapping keys only
        c_overlap = mx.concatenate([committee_weights[k].reshape(-1) for k in sorted(overlap)])
        d_overlap = mx.concatenate([domain_weights[k].reshape(-1) for k in sorted(overlap)])
        dot = mx.sum(c_overlap * d_overlap)
        norm_c = mx.sqrt(mx.sum(c_overlap * c_overlap))
        norm_d = mx.sqrt(mx.sum(d_overlap * d_overlap))
        mx.eval(dot, norm_c, norm_d)
        cos_sim = (dot / (norm_c * norm_d + 1e-10)).item()
        log(f"  Overlapping keys cos = {cos_sim:.6f}")

    # Also measure per-layer norms for diagnostics
    c_norm = mx.sqrt(mx.sum(c_flat * c_flat))
    d_norm = mx.sqrt(mx.sum(d_flat * d_flat))
    mx.eval(c_norm, d_norm)

    results = {
        "cosine_similarity": cos_sim,
        "committee_keys": len(c_keys),
        "domain_keys": len(d_keys),
        "overlapping_keys": len(overlap),
        "committee_norm": round(c_norm.item(), 4),
        "domain_norm": round(d_norm.item(), 4),
    }

    log(f"  Committee L2 norm: {results['committee_norm']}")
    log(f"  Domain L2 norm: {results['domain_norm']}")

    del committee_weights, domain_weights
    mx.clear_cache()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")

    # Phase 1: Base evaluation
    base_results = phase_base_eval()

    # Phase 2: Train committee adapter
    committee_train = phase_train_committee()

    # Phase 3: Train domain adapter
    domain_train = phase_train_domain()

    # Phase 4: Evaluate all configurations
    config_results = phase_eval_configs()

    # Phase 5: Orthogonality
    ortho_results = phase_orthogonality()

    # ══════════════════════════════════════════════════════════════════════════
    # KILL CRITERIA EVALUATION
    # ══════════════════════════════════════════════════════════════════════════

    # K1276: Committee + domain vs domain alone on reasoning (>= 3pp)
    composed_reasoning = config_results["composed"]["reasoning_accuracy"]
    domain_reasoning = config_results["domain_only"]["reasoning_accuracy"]
    reasoning_gain_pp = (composed_reasoning - domain_reasoning) * 100

    k1276 = {
        "pass": reasoning_gain_pp >= 3.0,
        "composed_reasoning": round(composed_reasoning, 3),
        "domain_only_reasoning": round(domain_reasoning, 3),
        "gain_pp": round(reasoning_gain_pp, 1),
        "threshold_pp": 3.0,
    }

    # K1277: Committee does not degrade format tasks (< 2pp loss)
    composed_format = config_results["composed"]["format_accuracy"]
    domain_format = config_results["domain_only"]["format_accuracy"]
    format_degradation_pp = (domain_format - composed_format) * 100

    k1277 = {
        "pass": format_degradation_pp < 2.0,
        "composed_format": round(composed_format, 3),
        "domain_only_format": round(domain_format, 3),
        "degradation_pp": round(format_degradation_pp, 1),
        "threshold_pp": 2.0,
    }

    # K1278: Orthogonal Grassmannian slot (cos < 1e-4)
    cos_val = ortho_results["cosine_similarity"]
    k1278 = {
        "pass": abs(cos_val) < 1e-4,
        "cosine_similarity": cos_val,
        "threshold": 1e-4,
    }

    all_pass = k1276["pass"] and k1277["pass"] and k1278["pass"]

    results = {
        "is_smoke": IS_SMOKE,
        "base": base_results,
        "committee_training": committee_train,
        "domain_training": domain_train,
        "configs": config_results,
        "orthogonality": ortho_results,
        "k1276": k1276,
        "k1277": k1277,
        "k1278": k1278,
        "all_pass": all_pass,
        "total_time_min": round((time.time() - t0) / 60, 2),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log(f"\n{'=' * 60}")
    log("RESULTS SUMMARY")
    log(f"{'=' * 60}")
    log(f"Base:      reasoning={base_results['reasoning_accuracy']:.1%}  format={base_results['format_accuracy']:.1%}")
    log(f"Committee: reasoning={config_results['committee_only']['reasoning_accuracy']:.1%}  format={config_results['committee_only']['format_accuracy']:.1%}")
    log(f"Domain:    reasoning={config_results['domain_only']['reasoning_accuracy']:.1%}  format={config_results['domain_only']['format_accuracy']:.1%}")
    log(f"Composed:  reasoning={composed_reasoning:.1%}  format={composed_format:.1%}")
    log(f"")
    log(f"K1276 (reasoning gain >= 3pp):    {'PASS' if k1276['pass'] else 'FAIL'} — {reasoning_gain_pp:+.1f}pp")
    log(f"K1277 (format loss < 2pp):        {'PASS' if k1277['pass'] else 'FAIL'} — {format_degradation_pp:+.1f}pp")
    log(f"K1278 (cos < 1e-4):              {'PASS' if k1278['pass'] else 'FAIL'} — cos={cos_val:.6f}")
    log(f"ALL PASS: {all_pass}")
    log(f"Total time: {results['total_time_min']:.1f} min")
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
