#!/bin/bash
# Refresh Antigravity Ralph Loop tokens automatically.
# Run this after Antigravity starts/restarts.
#
# Usage:
#   ./scripts/refresh-antigravity-tokens.sh
#   # or add to shell profile:
#   alias ag-refresh="./scripts/refresh-antigravity-tokens.sh"

set -euo pipefail

SETTINGS_FILE="$HOME/Library/Application Support/Antigravity/User/settings.json"
VSCDB="$HOME/Library/Application Support/Antigravity/User/globalStorage/state.vscdb"
WORKSPACE_MATCH="file_Users_tom_Code_tomsiwik_llm"

echo "Refreshing Antigravity Ralph Loop tokens..."

# 1. Extract CSRF + port for our workspace
PROCESS_LINE=$(ps -ax -o command= | grep -i "antigravity.*csrf_token.*${WORKSPACE_MATCH}" 2>/dev/null | head -1 || true)

if [ -z "$PROCESS_LINE" ]; then
    # Fallback: try without workspace filter, take the last one
    PROCESS_LINE=$(ps -ax -o command= | grep -i "antigravity.*csrf_token" | grep -v grep | tail -1 || true)
fi

if [ -z "$PROCESS_LINE" ]; then
    echo "ERROR: No Antigravity process found. Is Antigravity running?"
    exit 1
fi

CSRF=$(echo "$PROCESS_LINE" | grep -o "\-\-csrf_token [^ ]*" | awk '{print $2}')
PORT=$(echo "$PROCESS_LINE" | grep -o "\-\-extension_server_port [^ ]*" | awk '{print $2}')

if [ -z "$CSRF" ] || [ -z "$PORT" ]; then
    echo "ERROR: Could not extract CSRF token or port from process."
    exit 1
fi

echo "  CSRF: ${CSRF:0:8}..."
echo "  Port: $PORT"

# 2. Extract OAuth token from vscdb
OAUTH=""
if [ -f "$VSCDB" ]; then
    # Double base64 decode: vscdb stores protobuf(base64(token))
    RAW=$(sqlite3 "$VSCDB" "SELECT value FROM ItemTable WHERE key='antigravityUnifiedStateSync.oauthToken';" 2>/dev/null || true)
    if [ -n "$RAW" ]; then
        OAUTH=$(echo "$RAW" | base64 -d 2>/dev/null | strings | grep -o 'eWEy[A-Za-z0-9+/=]*' | head -1 | base64 -d 2>/dev/null | LC_ALL=C tr -d '\0-\037\200-\377' | grep -o 'ya29\.[A-Za-z0-9._/-]*' || true)
    fi
fi

if [ -z "$OAUTH" ]; then
    echo "WARNING: Could not extract OAuth token. Setting CSRF+port only."
else
    echo "  OAuth: ${OAUTH:0:20}..."
fi

# 3. Write to Antigravity settings.json via jq
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "{}" > "$SETTINGS_FILE"
fi

# Use python for safe JSON manipulation (jq may not be installed)
python3 -c "
import json, sys

settings_path = '$SETTINGS_FILE'
try:
    with open(settings_path) as f:
        settings = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    settings = {}

settings['ralphLoop.antigravity.csrfToken'] = '$CSRF'
settings['ralphLoop.antigravity.port'] = int('$PORT')
oauth = '''$OAUTH'''
if oauth:
    settings['ralphLoop.antigravity.oauthToken'] = oauth

with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=2)

print(f'  Written to {settings_path}')
print(f'  Keys updated: csrfToken, port' + (', oauthToken' if oauth else ''))
"

echo "Done. Restart the Ralph Loop in Antigravity to pick up new tokens."
