#!/bin/bash
set -euo pipefail
echo "[$(date +%H:%M:%S)] Starting attention-layer orthogonality experiment"
cd /workspace/llm
python3 macro/attention_layer_orthogonality/run_attention_ortho.py 2>&1
echo "[$(date +%H:%M:%S)] Done"
