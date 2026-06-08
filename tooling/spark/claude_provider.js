// promptfoo custom provider: generate via the Claude Code CLI in print mode.
// This matches ralph's `backend: claude` and this coding agent EXACTLY (same model,
// same harness, same auth) — no API key needed. The whole point of the spark eval is
// to measure how the REAL production model responds to the prompts, not a local stand-in.
const { execFileSync } = require('child_process');

class ClaudeCliProvider {
  constructor(options = {}) {
    this.providerId = options.id || 'claude-cli';
    this.config = options.config || {};
  }
  id() {
    return this.providerId;
  }
  async callApi(prompt) {
    const model = this.config.model || 'opus';
    try {
      const out = execFileSync('claude', ['-p', '--model', model], {
        input: prompt,
        encoding: 'utf-8',
        maxBuffer: 32 * 1024 * 1024,
        timeout: this.config.timeoutMs || 240000,
      });
      return { output: (out || '').trim() };
    } catch (e) {
      return { error: 'claude CLI error: ' + (e.stderr ? String(e.stderr) : e.message || String(e)) };
    }
  }
}

module.exports = ClaudeCliProvider;
