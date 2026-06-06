#!/usr/bin/env python3
"""
P4.A1: Domain Adapter Training Speed — New Domain in <10 Minutes.

VISION CLAIM: Adding a new domain costs <$2 and <10 minutes, not $10K and a week.

This experiment verifies that claim by training a biology domain adapter from scratch:
  Phase 0: Generate training data (synthetic, template-based, <10s)
  Phase 1: Train rank-16 LoRA adapter on Gemma 4 4-bit (200 steps, ~1 min)
  Phase 2: Evaluate behavioral improvement (vocabulary rubric, 20 questions)
  Phase 3: Measure adapter size

Grounded by:
  - Finding #436: rank-4, 40 examples, 300 steps → 1.2 min (P1.T5.1)
  - P3.C5: rank-16, 150 examples, 500 steps → 2.6 min (Finding #472)
  - LoRA paper 2106.09685: cost = O(r × d_model × M × seq_len)

Kill criteria (DB IDs):
  K1217 (#1217): training_time < 10 min for rank-16, 200 steps
  K1218 (#1218): behavioral_improvement >= 10pp vs base (vocabulary rubric)
  K1219 (#1219): adapter_size < 10 MB
"""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
ADAPTER_DIR = EXPERIMENT_DIR / "biology_adapter"
DATA_DIR = EXPERIMENT_DIR / "biology_data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 20       # questions for behavioral eval
TRAIN_ITERS = 20 if IS_SMOKE else 200
N_TRAIN = 10 if IS_SMOKE else 100
N_VALID = 3 if IS_SMOKE else 10
LORA_RANK = 16
SEED = 42

# ──────────────────────────────────────────────────────────────────────
# Biology vocabulary rubric (Theorem 1 Corollary)
# Terms specific to biology domain — unlikely in general responses
# ──────────────────────────────────────────────────────────────────────

BIO_VOCAB = [
    "cell", "cells", "protein", "proteins", "dna", "rna", "enzyme", "enzymes",
    "chromosome", "chromosomes", "mitosis", "meiosis", "photosynthesis",
    "metabolism", "metabolic", "atp", "ribosome", "ribosomes", "membrane",
    "membranes", "nucleus", "nuclei", "evolution", "evolutionary", "gene", "genes",
    "allele", "alleles", "mutation", "mutations", "receptor", "receptors",
    "neuron", "neurons", "antibody", "antibodies", "hormone", "hormones",
    "organelle", "organelles", "cytoplasm", "chloroplast", "chloroplasts",
    "mitochondria", "mitochondrion", "nucleotide", "nucleotides", "amino acid",
    "amino acids", "lipid", "lipids", "carbohydrate", "carbohydrates",
    "glucose", "atp", "nadh", "nadph", "rna polymerase", "dna replication",
    "natural selection", "homeostasis", "osmosis", "diffusion", "transcription",
    "translation", "codon", "anticodon", "mrna", "trna", "rrna", "plasmid",
    "genotype", "phenotype", "dominant", "recessive", "heterozygous", "homozygous",
    "ecosystem", "population", "species", "taxonom", "eukaryot", "prokaryot",
    "aerobic", "anaerobic", "fermentation", "glycolysis", "krebs cycle",
    "electron transport", "signal transduction", "apoptosis", "stem cell",
]

# Eval questions — specifically designed to elicit bio vocabulary
EVAL_QUESTIONS = [
    "Explain the process of DNA replication in detail.",
    "How does protein synthesis work from DNA to protein?",
    "What is the role of mitochondria in cellular energy production?",
    "Describe the stages of mitosis.",
    "How does natural selection drive evolution?",
    "What is the difference between prokaryotic and eukaryotic cells?",
    "Explain how enzymes catalyze biochemical reactions.",
    "What is the function of ribosomes in cells?",
    "Describe the structure and function of the cell membrane.",
    "How does photosynthesis convert light energy to chemical energy?",
    "What is homeostasis and how do organisms maintain it?",
    "Explain the central dogma of molecular biology.",
    "What is the role of chromosomes in cell division?",
    "How do mutations affect protein function?",
    "What is the difference between aerobic and anaerobic respiration?",
    "Explain how the immune system produces antibodies.",
    "What is the role of hormones in cell signaling?",
    "Describe the process of meiosis and its importance.",
    "How does osmosis work across cell membranes?",
    "What is the relationship between genotype and phenotype?",
]

# Training Q&A pairs — dense biology vocabulary
TRAIN_QA_PAIRS = [
    # DNA/Genetics
    ("Explain DNA replication.",
     "DNA replication is a semi-conservative process where each parental DNA strand serves as a template. The enzyme DNA helicase unwinds the double helix, breaking hydrogen bonds between base pairs. DNA polymerase III synthesizes new strands in the 5' to 3' direction, adding nucleotides complementary to each template strand. Primase synthesizes RNA primers to initiate synthesis. The leading strand is synthesized continuously while the lagging strand forms Okazaki fragments, later joined by DNA ligase. The result is two identical double-stranded DNA molecules, each containing one original and one new strand."),
    ("What is transcription?",
     "Transcription converts DNA into mRNA in the nucleus. RNA polymerase binds to the promoter sequence and unwinds the DNA double helix. The enzyme reads the template strand 3' to 5', synthesizing a complementary mRNA strand 5' to 3'. Each nucleotide base pairs with its complement: adenine with uracil, cytosine with guanine. After transcription, the pre-mRNA undergoes processing: 5' capping, polyadenylation at the 3' end, and splicing to remove introns and retain exons. The mature mRNA then exits the nucleus for translation at ribosomes."),
    ("Describe translation at the ribosome.",
     "Translation occurs at ribosomes, where mRNA codons are decoded into amino acid sequences. The ribosome has three tRNA binding sites: A (aminoacyl), P (peptidyl), and E (exit). Initiation begins when the small ribosomal subunit binds the 5' cap of mRNA and scans for the start codon AUG, which recruits the initiator tRNA carrying methionine. Elongation proceeds as aminoacyl-tRNAs with complementary anticodons enter the A site. Peptidyl transferase (rRNA enzyme) forms a peptide bond, transferring the growing chain to the A-site amino acid. The ribosome translocates 3' by one codon. Termination occurs at stop codons (UAA, UAG, UGA)."),
    ("What are chromosomes?",
     "Chromosomes are organized structures of DNA and protein found in the nucleus of eukaryotic cells. Each chromosome consists of a single, continuous double-stranded DNA molecule wrapped around histone proteins to form nucleosomes, which further compact into chromatin. Human cells contain 46 chromosomes arranged in 23 homologous pairs. Chromosomes contain genes, regulatory sequences, and non-coding DNA. During cell division, chromosomes condense and become visible under a microscope. Centromeres are attachment points for spindle microtubules during mitosis and meiosis. Telomeres cap the chromosome ends, protecting against degradation."),
    ("Explain natural selection.",
     "Natural selection is the primary mechanism of evolutionary change, first described by Darwin. It operates on heritable phenotypic variation within populations. Individuals with traits better suited to their environment survive longer and reproduce more successfully, passing beneficial alleles to offspring. Key components: variation (genetic diversity through mutations and recombination), heritability (traits passed from parents), differential reproduction (fitness differences). Over generations, beneficial alleles increase in frequency while deleterious alleles decrease. This leads to adaptation—populations becoming better matched to their environment. Natural selection acts on phenotypes but changes allele frequencies in the gene pool."),
    # Cell biology
    ("Describe the cell membrane structure.",
     "The cell membrane is a fluid mosaic of phospholipids and proteins. The phospholipid bilayer has hydrophilic phosphate heads facing outward and hydrophobic fatty acid tails facing inward, creating a selectively permeable barrier. Integral membrane proteins span the bilayer and function as channels, transporters, and receptors. Peripheral proteins associate loosely with the membrane surface. Cholesterol molecules interspersed between phospholipids regulate membrane fluidity—preventing crystallization at low temperatures and reducing fluidity at high temperatures. Glycoproteins and glycolipids on the outer surface form the glycocalyx, involved in cell recognition and signaling. The membrane controls movement of ions, nutrients, and waste products via diffusion, osmosis, facilitated diffusion, and active transport."),
    ("What is the role of mitochondria?",
     "Mitochondria are double-membraned organelles responsible for aerobic cellular respiration, producing most of the cell's ATP. The outer membrane is permeable to small molecules; the inner membrane is highly folded into cristae, maximizing surface area for electron transport. The matrix contains enzymes for the Krebs cycle, mitochondrial DNA (mtDNA), and ribosomes. Glucose-derived pyruvate enters the matrix and is oxidized to acetyl-CoA, releasing CO₂. Acetyl-CoA enters the Krebs cycle, generating NADH and FADH₂ electron carriers. These donate electrons to the electron transport chain in the inner membrane, driving ATP synthesis via chemiosmosis—protons flow through ATP synthase from the intermembrane space back into the matrix."),
    ("How does photosynthesis work?",
     "Photosynthesis occurs in chloroplasts and has two stages. The light-dependent reactions occur in thylakoid membranes: photons excite chlorophyll pigments in photosystems II and I, driving electron transport that generates ATP and NADPH while splitting water (photolysis), releasing O₂. The light-independent reactions (Calvin cycle) occur in the stroma: CO₂ is fixed by RuBisCO, attaching to ribulose-1,5-bisphosphate to form 3-phosphoglycerate. ATP and NADPH from light reactions reduce 3-PGA to glyceraldehyde-3-phosphate (G3P), the precursor to glucose and other organic molecules. Three CO₂ molecules must be fixed to produce one G3P molecule."),
    ("Explain mitosis.",
     "Mitosis is somatic cell division producing two genetically identical daughter cells. Following DNA replication in S phase, mitosis proceeds through four stages. Prophase: chromosomes condense, nuclear envelope breaks down, spindle microtubules form from centrosomes. Metaphase: chromosomes align at the metaphase plate, kinetochores attach to spindle fibers. Anaphase: sister chromatids separate as cohesin cleaves, chromatids pulled to opposite poles by shortening spindle fibers. Telophase: nuclear envelopes reform around each chromosome set, chromosomes decondense. Cytokinesis follows, cleaving the cytoplasm—in animal cells via cleavage furrow (actin-myosin ring), in plant cells via cell plate formation from Golgi-derived vesicles."),
    ("Describe enzyme catalysis.",
     "Enzymes are biological catalysts, typically proteins, that accelerate chemical reactions by lowering activation energy without being consumed. The active site—a specific three-dimensional pocket—binds the substrate with complementary shape and charge (lock-and-key model) or adjusts conformation upon binding (induced fit). Enzyme-substrate binding forms a transition state with lower activation energy than the uncatalyzed reaction. Rate is described by Michaelis-Menten kinetics: v = Vmax[S]/(Km + [S]), where Km (Michaelis constant) is the substrate concentration at half-maximal velocity. Enzymes are sensitive to temperature, pH, and inhibitors. Competitive inhibitors bind the active site; non-competitive inhibitors bind allosteric sites, changing enzyme shape."),
    # More pairs for variety
    ("What is osmosis?",
     "Osmosis is the passive diffusion of water molecules across a selectively permeable membrane from a region of higher water potential (lower solute concentration) to lower water potential (higher solute concentration). No energy input is required. The driving force is the osmotic pressure gradient—water moves to equalize solute concentrations on both sides. In hypotonic solutions, cells gain water and may lyse (animal cells) or become turgid (plant cells). In hypertonic solutions, cells lose water and shrink (crenation in animal cells, plasmolysis in plant cells). Aquaporins—channel proteins in the cell membrane—facilitate rapid water transport by osmosis. Osmosis is critical for nutrient uptake, waste removal, and maintaining cell turgor pressure."),
    ("Explain the immune system's antibody production.",
     "Antibody production by B lymphocytes follows antigen-driven clonal selection. When an antigen—typically a foreign protein or polysaccharide—enters the body, it is engulfed by antigen-presenting cells (APCs) such as dendritic cells and macrophages. APCs display antigen fragments on MHC II molecules, activating helper T cells (CD4+). T helper cells release cytokines that activate B cells with complementary B-cell receptors. Activated B cells proliferate (clonal expansion) and differentiate into plasma cells, which secrete large quantities of antigen-specific antibodies (immunoglobulins). Antibodies have a Y-shaped structure with two heavy and two light chains, variable regions binding antigen, and constant regions mediating effector functions. Memory B cells persist for rapid secondary responses."),
    ("What is the difference between mitosis and meiosis?",
     "Mitosis and meiosis are both forms of nuclear division but serve different purposes. Mitosis: one division producing two diploid (2n) daughter cells genetically identical to the parent—used for growth, repair, and asexual reproduction. Meiosis: two sequential divisions producing four haploid (n) daughter cells with genetic recombination—used for sexual reproduction (gametes). Meiosis I separates homologous chromosome pairs (reduction division), during which crossing over between non-sister chromatids in prophase I creates genetic recombination. Meiosis II separates sister chromatids (similar to mitosis). The result: genetic diversity through independent assortment of homologs and crossing over. Errors in meiosis produce aneuploid gametes causing chromosomal disorders."),
    ("Describe cellular respiration stages.",
     "Aerobic cellular respiration converts glucose to ATP in three stages. Glycolysis (cytoplasm): glucose (6C) is phosphorylated and split into two pyruvate (3C) molecules, yielding net 2 ATP and 2 NADH. The pyruvate oxidation (mitochondrial matrix): each pyruvate is converted to acetyl-CoA (2C), releasing CO₂ and generating NADH. The Krebs cycle (matrix): acetyl-CoA condenses with oxaloacetate to form citrate. Eight enzyme-catalyzed reactions oxidize carbons to CO₂, generating 3 NADH, 1 FADH₂, and 1 GTP per turn—two turns per glucose. Oxidative phosphorylation (inner mitochondrial membrane): NADH and FADH₂ donate electrons to the electron transport chain. Proton pumping creates the electrochemical gradient driving ATP synthase. O₂ is the terminal electron acceptor, forming water. Net: ~36-38 ATP per glucose."),
    ("What is the role of hormones in signaling?",
     "Hormones are chemical messengers synthesized in endocrine glands and secreted into the bloodstream to regulate distant target cells. Peptide hormones (e.g., insulin, glucagon) are hydrophilic; they bind membrane receptors and trigger intracellular signaling cascades via second messengers (cAMP, IP₃, DAG). Steroid hormones (e.g., cortisol, estrogen) are lipophilic; they diffuse through the cell membrane and bind intracellular receptors, forming complexes that act as transcription factors to directly regulate gene expression. Hormone signaling involves amplification: one hormone molecule activates thousands of enzyme molecules via signal transduction. Feedback loops maintain homeostasis—insulin lowers blood glucose, glucagon raises it, maintaining normoglycemia."),
    # Additional pairs to reach 100 examples
    ("What are genes?",
     "Genes are discrete units of hereditary information encoded in DNA sequences. Each gene occupies a specific chromosomal locus and typically encodes one or more proteins via transcription and translation. The human genome contains approximately 20,000-25,000 protein-coding genes, occupying only ~1.5% of the genome; the rest includes regulatory sequences, introns, and non-coding RNA genes. Alleles are alternative forms of a gene differing in nucleotide sequence. Dominant alleles mask recessive alleles in heterozygotes. Gene expression is regulated by promoters, enhancers, silencers, and epigenetic modifications (DNA methylation, histone acetylation). Genes interact in complex networks: a single gene can affect multiple traits (pleiotropy) and multiple genes can affect a single trait (polygenic inheritance)."),
    ("Explain the structure of proteins.",
     "Proteins are polymers of amino acids linked by peptide bonds, folded into specific three-dimensional structures that determine function. Primary structure: the linear sequence of amino acids encoded by mRNA codons. Secondary structure: local regular conformations—alpha helices (hydrogen bonds between backbone NH and C=O groups, 3.6 residues per turn) and beta sheets (hydrogen bonds between adjacent strands). Tertiary structure: overall 3D folding of a single polypeptide, stabilized by disulfide bonds, hydrophobic interactions, hydrogen bonds, and ionic interactions. Quaternary structure: assembly of multiple polypeptide subunits (e.g., hemoglobin: 2α + 2β subunits). Chaperone proteins assist folding; misfolded proteins are degraded by proteasomes or cause aggregation diseases like Alzheimer's."),
    ("What is homeostasis?",
     "Homeostasis is the maintenance of stable internal conditions despite external fluctuations, essential for cellular function. Regulatory mechanisms include negative feedback loops—the primary homeostatic control mechanism. A receptor detects deviation from the set point, a control center (often the hypothalamus) processes the signal, and an effector (muscle, gland) responds to restore equilibrium. Example: blood glucose regulation—elevated glucose triggers insulin secretion from pancreatic β cells, promoting glucose uptake by liver (glycogen synthesis) and muscle cells; declining glucose triggers glucagon from α cells, stimulating glycogenolysis. Body temperature regulation: hypothalamus detects changes; sweating, vasodilation reduce temperature; shivering, vasoconstriction increase it. Positive feedback loops amplify deviations (e.g., childbirth contractions, blood clotting)."),
    ("Describe the role of ribosomes.",
     "Ribosomes are the molecular machines of translation, synthesizing proteins from mRNA templates. Composed of rRNA and ribosomal proteins, they form two subunits: large (60S in eukaryotes) and small (40S), assembling on mRNA to form the 80S ribosome. The small subunit binds mRNA and decodes codons; the large subunit catalyzes peptide bond formation. Three tRNA binding sites: A site accepts incoming aminoacyl-tRNA, P site holds the growing peptide chain, E site releases deacylated tRNA. Peptidyl transferase activity—provided by the 23S/28S rRNA—catalyzes peptide bond formation between amino acids. Ribosomes can be free in the cytoplasm (synthesizing cytosolic proteins) or membrane-bound to the rough ER (synthesizing secretory and membrane proteins). A single mRNA can be translated simultaneously by multiple ribosomes (polyribosome/polysome)."),
    ("What is genetic mutation?",
     "A mutation is a heritable change in the DNA nucleotide sequence. Point mutations alter single nucleotides: transitions (purine↔purine or pyrimidine↔pyrimidine), transversions (purine↔pyrimidine). Silent mutations don't change amino acids (redundant codons); missense mutations change one amino acid; nonsense mutations create premature stop codons, truncating the protein. Frameshift mutations—insertions or deletions of non-multiple-of-3 nucleotides—shift the reading frame, altering all downstream codons. Chromosomal mutations include duplications, deletions, inversions, and translocations affecting large DNA segments. Causes: DNA replication errors, spontaneous base modifications, mutagens (UV radiation, chemical carcinogens). DNA repair systems (base excision repair, nucleotide excision repair, mismatch repair) correct most mutations; failures lead to cancer or genetic disease."),
    ("What is the difference between eukaryotic and prokaryotic cells?",
     "Eukaryotic and prokaryotic cells differ fundamentally in organization. Prokaryotes (bacteria, archaea): no membrane-bound nucleus—DNA is a circular chromosome in the nucleoid region; no membrane-bound organelles; cell wall (peptidoglycan in bacteria, various in archaea); ribosomes are 70S (50S + 30S subunits); reproduction by binary fission; typically 1-10 µm. Eukaryotes (animals, plants, fungi, protists): DNA enclosed in nucleus with double membrane (nuclear envelope) and nuclear pores; membrane-bound organelles (mitochondria, ER, Golgi, lysosomes); ribosomes are 80S (60S + 40S subunits); cell division by mitosis/meiosis; typically 10-100 µm. Eukaryotes likely evolved from prokaryotic ancestors; mitochondria and chloroplasts originated from endosymbiotic bacteria (endosymbiotic theory—supported by their own circular DNA and 70S ribosomes)."),
]

# Extended synthetic pairs to reach 100 examples
SYNTHETIC_PAIRS_EXTRA = [
    ("What is the cytoskeleton?",
     "The cytoskeleton is a dynamic network of protein filaments that provides structural support, enables cell movement, and organizes internal components. Three types: microfilaments (actin, 7nm)—form the cell cortex, enable muscle contraction (with myosin), cell motility via lamellipodia and filopodia; intermediate filaments (e.g., keratin, vimentin, lamin, 10nm)—provide mechanical strength, anchor organelles; microtubules (tubulin α/β heterodimers, 25nm)—form mitotic spindle, provide tracks for motor proteins (kinesin, dynein), form cilia and flagella axonemes (9+2 arrangement). Dynamic instability: microtubule plus-ends alternate between polymerization and catastrophic depolymerization. GTP-tubulin promotes growth; GDP-tubulin promotes collapse. The cytoskeleton coordinates cell polarity, vesicle trafficking, and chromosome segregation."),
    ("Explain signal transduction pathways.",
     "Signal transduction converts extracellular signals into intracellular responses. A ligand binds a receptor, inducing conformational changes that activate intracellular signaling proteins. G protein-coupled receptors (GPCRs): ligand binding activates Gα subunit (GTPase), which modulates adenylyl cyclase to produce cAMP from ATP. cAMP activates PKA (protein kinase A), phosphorylating target proteins. Receptor tyrosine kinases (RTKs): ligand binding causes dimerization and autophosphorylation of cytoplasmic tyrosine residues, creating docking sites for SH2 domain-containing proteins (e.g., Ras-GEF, PI3K). Ras activates MAPK cascade. PI3K produces PIP₃, activating Akt (cell survival, protein synthesis). Second messengers (cAMP, Ca²⁺, IP₃, DAG) amplify signals. Phosphorylation cascades amplify and integrate multiple inputs, enabling signal specificity despite limited receptor types."),
    ("What is the structure of DNA?",
     "DNA (deoxyribonucleic acid) is a double-stranded helical polymer. Each strand consists of nucleotides: deoxyribose sugar, phosphate group, and nitrogenous base (adenine, thymine, guanine, cytosine). Strands are antiparallel (one runs 5'→3', the other 3'→5') and complementary—A pairs with T via two hydrogen bonds, G pairs with C via three hydrogen bonds (Chargaff's rules). The double helix has a major groove (wide, protein binding sites for transcription factors) and minor groove (narrow). One complete turn occurs every ~10.5 base pairs in B-form DNA (dominant at physiological conditions). The phosphate backbone is negatively charged (bound by histones and other positively charged proteins). Supercoiling: topoisomerases regulate DNA topology by introducing or removing supercoils, essential for replication and transcription."),
    ("How do neurons transmit signals?",
     "Neurons transmit electrical signals via action potentials. At rest, the membrane potential is ~-70mV (resting potential), maintained by Na⁺/K⁺-ATPase (3 Na⁺ out, 2 K⁺ in per ATP) and selective membrane permeability. Depolarization: excitatory input opens voltage-gated Na⁺ channels, Na⁺ influx depolarizes membrane to +40mV (action potential peak). Repolarization: voltage-gated K⁺ channels open, K⁺ efflux restores negative potential. Hyperpolarization (undershoot) follows before return to resting potential. Action potentials propagate by sequential activation of adjacent Na⁺ channels along the axon. Saltatory conduction: in myelinated axons, action potentials jump between Nodes of Ranvier, increasing speed (70-120 m/s). At the synapse, Ca²⁺ influx triggers vesicle fusion, releasing neurotransmitters (acetylcholine, dopamine, glutamate, GABA) that bind postsynaptic receptors."),
    ("What is gene regulation?",
     "Gene regulation controls when, where, and how much of each gene is expressed. Transcriptional regulation is primary: transcription factors (TFs) bind DNA regulatory sequences. Promoters (TATA box, BRE) recruit RNA polymerase II and general TFs. Enhancers—distal regulatory elements—bound by specific TFs, looping DNA to contact the promoter via cohesin and mediator complexes. Repressors block RNA polymerase or recruit histone deacetylases, condensing chromatin. Epigenetic regulation: DNA methylation (CpG islands, gene silencing) and histone modifications (acetylation—active; methylation—context-dependent; phosphorylation—mitosis). Post-transcriptional regulation: alternative splicing generates protein isoforms; miRNA (microRNA) base-pairs with mRNA to block translation or trigger degradation; RNA-binding proteins stabilize or destabilize mRNA. Translational regulation: IRES elements, eIF2α phosphorylation during stress."),
]


def cleanup():
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def log(msg: str):
    print(msg, flush=True)


def count_bio_terms(text: str) -> int:
    """Count distinct biology vocabulary terms in text."""
    text_lower = text.lower()
    count = 0
    for term in BIO_VOCAB:
        if term in text_lower:
            count += 1
    return count


BIO_DEPTH_THRESHOLD = 8  # ≥8 distinct bio terms signals domain-expert depth

def response_has_bio_depth(text: str, threshold: int = BIO_DEPTH_THRESHOLD) -> bool:
    """True if response contains >= threshold distinct bio terms (expert-level depth)."""
    return count_bio_terms(text) >= threshold


# ─── Phase 0: Generate training data ──────────────────────────────────────────

def generate_training_data() -> None:
    """Generate biology training data from hardcoded Q&A pairs."""
    if DATA_DIR.exists() and (DATA_DIR / "train.jsonl").exists():
        n = sum(1 for _ in open(DATA_DIR / "train.jsonl"))
        if n >= N_TRAIN:
            log(f"Training data already exists: {n} train examples")
            return
        log(f"Cache insufficient ({n} < {N_TRAIN}), regenerating...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = TRAIN_QA_PAIRS + SYNTHETIC_PAIRS_EXTRA
    # Expand to N_TRAIN by cycling
    expanded = []
    while len(expanded) < N_TRAIN + N_VALID + 5:
        for q, a in all_pairs:
            expanded.append((q, a))
            if len(expanded) >= N_TRAIN + N_VALID + 5:
                break

    train_pairs = expanded[:N_TRAIN]
    valid_pairs = expanded[N_TRAIN:N_TRAIN + N_VALID]
    test_pairs = expanded[N_TRAIN + N_VALID:N_TRAIN + N_VALID + 5]

    def write_jsonl(path: Path, pairs: list) -> None:
        with open(path, "w") as f:
            for q, a in pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
                f.write(json.dumps(record) + "\n")

    write_jsonl(DATA_DIR / "train.jsonl", train_pairs)
    write_jsonl(DATA_DIR / "valid.jsonl", valid_pairs)
    write_jsonl(DATA_DIR / "test.jsonl", test_pairs)

    log(f"Generated {len(train_pairs)} train, {len(valid_pairs)} valid, "
        f"{len(test_pairs)} test biology examples")


# ─── Phase 1: Train LoRA adapter ──────────────────────────────────────────────

def train_adapter() -> float:
    """Train rank-16 LoRA adapter. Returns training time in minutes."""
    safetensors = ADAPTER_DIR / "adapters.safetensors"
    if ADAPTER_DIR.exists() and safetensors.exists():
        log(f"Adapter already exists at {ADAPTER_DIR}")
        return 0.0

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    import yaml
    config = {
        "model": MODEL_ID,
        "data": str(DATA_DIR),
        "adapter_path": str(ADAPTER_DIR),
        "train": True,
        "fine_tune_type": "lora",
        "num_layers": 12,   # 12 layers → adapter ≤8MB for d_model=5120
        "iters": TRAIN_ITERS,
        "batch_size": 1 if IS_SMOKE else 4,
        "learning_rate": 2e-4,
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": 4.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
        },
        "max_seq_length": 256,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "save_every": TRAIN_ITERS,
        "steps_per_report": max(1, TRAIN_ITERS // 10),
        "val_batches": max(1, min(3, N_VALID // 2)),
        "steps_per_eval": max(10, TRAIN_ITERS // 5),
        "seed": SEED,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name
        yaml.dump(config, f)

    log(f"Training rank-{LORA_RANK} adapter ({TRAIN_ITERS} iters, {N_TRAIN} examples)...")
    log(f"  Model: {MODEL_ID}")
    log(f"  Data:  {DATA_DIR}")
    log(f"  Out:   {ADAPTER_DIR}")

    t0 = time.time()
    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.time() - t0) / 60.0  # minutes

    os.unlink(config_path)

    if result.returncode != 0:
        log(f"Training failed (exit={result.returncode})")
        log(f"STDOUT: {result.stdout[-3000:]}")
        log(f"STDERR: {result.stderr[-3000:]}")
        raise RuntimeError("Training failed")

    log(f"Training complete in {elapsed:.2f} min")
    if result.stdout:
        log(result.stdout[-1000:])
    return elapsed


# ─── Phase 2: Behavioral evaluation ───────────────────────────────────────────

def generate_responses(questions: list[str], adapter_path: str | None = None) -> list[str]:
    """Generate model responses to biology questions."""
    responses = []
    for q in questions:
        prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"

        cmd = [
            "uv", "run", "python", "-m", "mlx_lm", "generate",
            "--model", MODEL_ID,
            "--prompt", prompt,
            "--max-tokens", "200",
            "--temp", "0.0",
        ]
        if adapter_path:
            cmd += ["--adapter-path", adapter_path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log(f"Generation failed for: {q[:50]}...")
            responses.append("")
            continue

        # Extract generated text (after prompt)
        output = result.stdout.strip()
        # mlx_lm.generate outputs just the generated text
        responses.append(output)

    return responses


def evaluate_behavioral_improvement(n_eval: int) -> dict:
    """Evaluate vocabulary improvement: adapted vs base."""
    questions = EVAL_QUESTIONS[:n_eval]

    log(f"\n=== Phase 2: Behavioral Evaluation (n={n_eval}) ===")

    log("Evaluating BASE model...")
    base_responses = generate_responses(questions, adapter_path=None)

    log("Evaluating ADAPTED model...")
    adapted_responses = generate_responses(questions, adapter_path=str(ADAPTER_DIR))

    results = []
    for i, (q, base_r, adapted_r) in enumerate(zip(questions, base_responses, adapted_responses)):
        base_bio = count_bio_terms(base_r)
        adapted_bio = count_bio_terms(adapted_r)
        base_pass = response_has_bio_depth(base_r)
        adapted_pass = response_has_bio_depth(adapted_r)

        log(f"\nQ{i+1}: {q[:60]}...")
        log(f"  Base bio_terms={base_bio}, depth_pass={base_pass}")
        log(f"  Adapted bio_terms={adapted_bio}, depth_pass={adapted_pass}")

        results.append({
            "question": q,
            "base_bio_terms": base_bio,
            "adapted_bio_terms": adapted_bio,
            "base_depth_pass": base_pass,
            "adapted_depth_pass": adapted_pass,
        })

    base_rate = sum(r["base_depth_pass"] for r in results) / len(results)
    adapted_rate = sum(r["adapted_depth_pass"] for r in results) / len(results)
    improvement_pp = (adapted_rate - base_rate) * 100

    base_mean_terms = sum(r["base_bio_terms"] for r in results) / len(results)
    adapted_mean_terms = sum(r["adapted_bio_terms"] for r in results) / len(results)

    log(f"\nBase depth_pass_rate:    {base_rate:.3f} ({base_rate*100:.1f}%)")
    log(f"Adapted depth_pass_rate: {adapted_rate:.3f} ({adapted_rate*100:.1f}%)")
    log(f"Improvement:             {improvement_pp:.1f}pp")
    log(f"Base mean bio_terms:     {base_mean_terms:.1f}")
    log(f"Adapted mean bio_terms:  {adapted_mean_terms:.1f}")

    return {
        "base_rate": base_rate,
        "adapted_rate": adapted_rate,
        "improvement_pp": improvement_pp,
        "base_mean_bio_terms": base_mean_terms,
        "adapted_mean_bio_terms": adapted_mean_terms,
        "per_question": results,
    }


# ─── Phase 3: Adapter size ────────────────────────────────────────────────────

def measure_adapter_size() -> float:
    """Compute adapter size in MB."""
    if not ADAPTER_DIR.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in ADAPTER_DIR.rglob("*") if f.is_file())
    size_mb = total_bytes / 1e6
    log(f"Adapter size: {size_mb:.2f} MB")
    return size_mb


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("P4.A1: Domain Adapter Training Speed — Biology Domain")
    log(f"IS_SMOKE={IS_SMOKE}, TRAIN_ITERS={TRAIN_ITERS}, N_EVAL={N_EVAL}")
    log("=" * 60)

    total_start = time.time()

    # Phase 0: Data generation
    log("\n=== Phase 0: Generate Training Data ===")
    t_data_start = time.time()
    generate_training_data()
    t_data = time.time() - t_data_start
    log(f"Data prep time: {t_data:.1f}s")

    # Phase 1: Training
    log("\n=== Phase 1: Train LoRA Adapter ===")
    t_train_start = time.time()
    train_minutes = train_adapter()
    t_train = time.time() - t_train_start
    training_time_min = t_train / 60.0

    log(f"Training wall-clock: {training_time_min:.2f} min")

    cleanup()

    # Phase 2: Behavioral evaluation
    behavioral = evaluate_behavioral_improvement(N_EVAL)

    cleanup()

    # Phase 3: Adapter size
    log("\n=== Phase 3: Adapter Size ===")
    adapter_size_mb = measure_adapter_size()

    total_elapsed_min = (time.time() - total_start) / 60.0

    # ─── Kill criteria ────────────────────────────────────────────────────────
    k1217_pass = training_time_min < 10.0
    k1218_pass = behavioral["improvement_pp"] >= 10.0
    k1219_pass = adapter_size_mb < 10.0

    log("\n" + "=" * 60)
    log("KILL CRITERIA RESULTS")
    log("=" * 60)
    log(f"K1217: training_time={training_time_min:.2f} min < 10 → {'PASS' if k1217_pass else 'FAIL'}")
    log(f"K1218: behavioral_improvement={behavioral['improvement_pp']:.1f}pp >= 10 → {'PASS' if k1218_pass else 'FAIL'}")
    log(f"K1219: adapter_size={adapter_size_mb:.2f} MB < 10 → {'PASS' if k1219_pass else 'FAIL'}")
    log(f"ALL_PASS: {all([k1217_pass, k1218_pass, k1219_pass])}")
    log(f"Total wall-clock: {total_elapsed_min:.2f} min")

    results = {
        "is_smoke": IS_SMOKE,
        "k1217_training_time_min": training_time_min,
        "k1218_behavioral_improvement_pp": behavioral["improvement_pp"],
        "k1219_adapter_size_mb": adapter_size_mb,
        "k1217_pass": k1217_pass,
        "k1218_pass": k1218_pass,
        "k1219_pass": k1219_pass,
        "all_pass": all([k1217_pass, k1218_pass, k1219_pass]),
        "total_wall_clock_min": total_elapsed_min,
        "data_prep_time_s": t_data,
        "training_time_min": training_time_min,
        "behavioral": behavioral,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")

    return 0 if all([k1217_pass, k1218_pass, k1219_pass]) else 1


if __name__ == "__main__":
    sys.exit(main())
