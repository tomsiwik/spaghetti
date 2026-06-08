// promptfoo custom provider: generate via the Gemini CLI (newest), non-interactive.
// Matches how a Gemini-backed agent would actually run.
const { execFileSync } = require('child_process');

const NOISE = /^(Ripgrep is not available|Falling back to GrepTool|Loaded cached credentials|Data collection)/;

class GeminiProvider {
  constructor(options = {}) {
    this.providerId = options.id || 'gemini';
    this.config = options.config || {};
  }
  id() {
    return this.providerId;
  }
  async callApi(prompt) {
    const args = ['-p', prompt, '-o', 'text', '--skip-trust'];
    if (this.config.model) args.push('-m', this.config.model);
    try {
      const raw = execFileSync('gemini', args, {
        encoding: 'utf-8', maxBuffer: 32 * 1024 * 1024,
        timeout: this.config.timeoutMs || 300000, stdio: ['ignore', 'pipe', 'ignore'],
      });
      const out = (raw || '').split('\n').filter((l) => !NOISE.test(l.trim())).join('\n').trim();
      return { output: out };
    } catch (e) {
      return { error: 'gemini error: ' + (e.stderr ? String(e.stderr) : e.message || String(e)) };
    }
  }
}

module.exports = GeminiProvider;
