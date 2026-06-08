// promptfoo custom provider: generate via the Codex CLI (GPT-5.5), non-interactive.
// Matches how a Codex-backed agent would actually run. --output-last-message extracts
// just the final assistant message (codex exec otherwise prints session scaffolding).
const { execFileSync } = require('child_process');
const { readFileSync, unlinkSync } = require('fs');
const os = require('os');
const path = require('path');
let _seq = 0;

class CodexProvider {
  constructor(options = {}) {
    this.providerId = options.id || 'codex-gpt5.5';
    this.config = options.config || {};
  }
  id() {
    return this.providerId;
  }
  async callApi(prompt) {
    const model = this.config.model || 'gpt-5.5';
    const outFile = path.join(os.tmpdir(), `codex_${process.pid}_${Date.now()}_${_seq++}.txt`);
    try {
      execFileSync('codex', ['exec', '-m', model, '--skip-git-repo-check', '-s', 'read-only',
        '--output-last-message', outFile, prompt], {
        encoding: 'utf-8', maxBuffer: 32 * 1024 * 1024,
        timeout: this.config.timeoutMs || 300000, stdio: ['ignore', 'ignore', 'pipe'],
      });
      const text = readFileSync(outFile, 'utf-8').trim();
      try { unlinkSync(outFile); } catch (e) {}
      return { output: text };
    } catch (e) {
      try { const t = readFileSync(outFile, 'utf-8').trim(); unlinkSync(outFile); if (t) return { output: t }; } catch (_) {}
      return { error: 'codex error: ' + (e.stderr ? String(e.stderr) : e.message || String(e)) };
    }
  }
}

module.exports = CodexProvider;
