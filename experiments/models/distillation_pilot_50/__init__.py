# Distillation Pilot 50
# This is a macro experiment, not a micro model.
# No arena registration -- uses composer/ pipeline directly.
#
# Files:
#   MATH.md  — cost and scaling analysis
#   PAPER.md — research digest with pipeline design and results
#
# Pipeline scripts:
#   scripts/pilot50_generate.py  — data generation (local, Groq API)
#   scripts/pilot50_train.py     — QLoRA training (RunPod)
#   scripts/pilot50_bench.py     — benchmark (RunPod)
#   scripts/pilot50_orchestrate.sh — full pipeline orchestration
