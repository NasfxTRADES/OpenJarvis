import { useEffect, useState, type FormEvent } from 'react';
import {
  connectProvider,
  setKey,
  type MobileStatus,
} from '../api';

export function SettingsView({
  status,
  onChanged,
}: {
  status: MobileStatus | null;
  onChanged: () => void;
}) {
  const [apiKey, setApiKey] = useState('');
  const [geminiKey, setGeminiKey] = useState('');
  const [busy, setBusy] = useState<
    'openrouter' | 'gemini' | 'ollama' | 'model' | 'auto' | null
  >(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [models, setModels] = useState<string[]>(status?.models || []);
  const [selected, setSelected] = useState(status?.model || '');

  useEffect(() => {
    setModels(status?.models || []);
    setSelected(status?.model || '');
  }, [status?.models, status?.model]);

  const run = async (
    kind: 'openrouter' | 'gemini' | 'ollama' | 'model' | 'auto',
    body: { provider: string; api_key?: string; model?: string },
  ) => {
    setBusy(kind);
    setError('');
    setNotice('');
    try {
      const result = await connectProvider(body);
      setModels(result.models || []);
      setSelected(result.model);
      setApiKey('');
      setNotice(result.notice || `Connected to ${result.model}`);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not connect');
    } finally {
      setBusy(null);
    }
  };

  // Some browsers (and embedded previews) refuse paste events on inputs.
  // Reading the clipboard ourselves gives the user a working button, and a
  // clear message when the browser blocks that too.
  const pasteKey = async (apply: (value: string) => void) => {
    setError('');
    setNotice('');
    try {
      // Some browsers leave readText() pending forever when they quietly deny
      // the permission, so never wait on it indefinitely.
      const text = (
        await Promise.race([
          navigator.clipboard.readText(),
          new Promise<string>((_, reject) =>
            setTimeout(() => reject(new Error('clipboard timeout')), 2500),
          ),
        ])
      ).trim();
      if (!text) {
        setError('Your clipboard looks empty. Copy the key again, then press Paste.');
        return;
      }
      apply(text);
    } catch {
      setError(
        'This browser will not hand over the clipboard. Tap the box and paste with Cmd/Ctrl+V, or paste on your own computer instead.',
      );
    }
  };

  const submitKey = (event: FormEvent) => {
    event.preventDefault();
    if (!apiKey.trim()) return;
    void run('openrouter', { provider: 'openrouter', api_key: apiKey.trim() });
  };

  const submitGeminiKey = (event: FormEvent) => {
    event.preventDefault();
    if (!geminiKey.trim()) return;
    void run('gemini', { provider: 'gemini', api_key: geminiKey.trim() });
  };

  const chain = status?.chain || [];
  // A failover chain owns the model choice: every provider in it runs its own
  // model, so "Use this model" on a chain would ask the server to configure a
  // provider that may not even have a key (the old code sent OpenRouter and
  // got a 400).
  const chainActive = chain.length > 0 || status?.provider === 'auto';

  const provider =
    status?.provider === 'ollama'
      ? 'ollama'
      : status?.provider === 'gemini'
        ? 'gemini'
        : status?.provider === 'openrouter'
          ? 'openrouter'
          : 'auto';

  return (
    <div className="settings">
      <section className="setting-card">
        <h2>Assistant</h2>
        <div className="kv">
          <span>Status</span>
          <strong className={status?.configured ? 'ok-text' : 'warn-text'}>
            {status?.configured ? 'Connected' : 'Not connected'}
          </strong>
        </div>
        <div className="kv">
          <span>Model</span>
          <strong className="truncate">{status?.model || '—'}</strong>
        </div>
        {status?.active_provider && (
          <div className="kv">
            <span>Answering with</span>
            <strong>{status.active_provider}</strong>
          </div>
        )}
        <div className="kv">
          <span>Automations</span>
          <strong>{status?.tasks ?? 0}</strong>
        </div>
        <div className="kv">
          <span>Scheduler</span>
          <strong>{status?.scheduler ? 'Running' : 'Off'}</strong>
        </div>
      </section>

      {error && <div className="banner banner-error">{error}</div>}
      {notice && <div className="banner banner-ok">{notice}</div>}

      {status?.configured && models.length > 0 && !chainActive && (
        <section className="setting-card">
          <h3>Model</h3>
          <p className="muted small">Swap models without re-entering your key.</p>
          <select
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
            className="select"
          >
            {!models.includes(selected) && selected && (
              <option value={selected}>{selected}</option>
            )}
            {models.map((model) => (
              <option key={model} value={model}>
                {model.replace(/^openrouter\//, '')}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy !== null || selected === status.model}
            onClick={() =>
              void run(
                'model',
                provider === 'auto'
                  ? { provider: 'auto' }
                  : { provider, model: selected },
              )
            }
          >
            {busy === 'model' ? 'Switching…' : 'Use this model'}
          </button>
        </section>
      )}

      <section className="setting-card">
        <h3>Never run out — automatic failover</h3>
        <p className="muted small">
          Jarvis uses these in order and moves to the next one whenever a
          provider runs out of its free allowance, so replies keep coming.
        </p>
        {chain.length === 0 ? (
          <p className="muted small">
            No providers are chained yet. Add a free key on the server (Groq,
            Cerebras, Gemini, Mistral, OpenRouter), then press Rebuild.
          </p>
        ) : (
          chain.map((entry) => (
            <div className="kv" key={`${entry.provider}-${entry.model}`}>
              <span>{entry.label}</span>
              <strong className={entry.ready ? 'ok-text' : 'warn-text'}>
                {entry.ready
                  ? entry.model.replace(/^[a-z]+\//, '')
                  : `resting ${Math.max(1, Math.round(entry.cooldown_seconds / 60))}m`}
              </strong>
            </div>
          ))
        )}
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy !== null}
          onClick={() => void run('auto', { provider: 'auto' })}
        >
          {busy === 'auto' ? 'Rebuilding…' : 'Rebuild free chain'}
        </button>
      </section>

      <section className="setting-card">
        <h3>OpenRouter — free models</h3>
        <p className="muted small">
          Create a free key at openrouter.ai/keys, paste it here once, and Jarvis
          picks from the models OpenRouter currently serves at zero cost. No card
          required.
        </p>
        {status?.key_source ? (
          <div className="stack">
            <p className="muted small">
              This server already has a key from{' '}
              <code>{status.key_source.replace(/^file:/, '') || 'the environment'}</code>.
              Nothing to paste — connect and Jarvis picks a free model.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy !== null}
              onClick={() => void run('openrouter', { provider: 'openrouter' })}
            >
              {busy === 'openrouter' ? 'Connecting…' : 'Use the server key'}
            </button>
          </div>
        ) : (
          <form onSubmit={submitKey} className="stack">
            <input
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="sk-or-v1-…"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
            />
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy !== null}
              onClick={() => void pasteKey(setApiKey)}
            >
              Paste from clipboard
            </button>
            <button type="submit" className="btn btn-primary" disabled={busy !== null || !apiKey.trim()}>
              {busy === 'openrouter' ? 'Connecting…' : 'Connect free model'}
            </button>
          </form>
        )}
      </section>

      <section className="setting-card">
        <h3>Google Gemini — free Flash models</h3>
        <p className="muted small">
          Create a free key at aistudio.google.com/apikey — new keys start with
          <code>AQ.</code> and work as-is. Gemini's free tier covers the Flash
          models, with an allowance per project rather than per key. On the free
          tier Google may use your prompts to improve its products.
        </p>
        {status?.gemini_key_source ? (
          <div className="stack">
            <p className="muted small">
              This server already has a key from{' '}
              <code>
                {status.gemini_key_source.replace(/^file:/, '') || 'the environment'}
              </code>
              . Nothing to paste.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy !== null}
              onClick={() => void run('gemini', { provider: 'gemini' })}
            >
              {busy === 'gemini' ? 'Connecting…' : 'Use the Gemini key'}
            </button>
          </div>
        ) : (
          <form onSubmit={submitGeminiKey} className="stack">
            <input
              type="password"
              value={geminiKey}
              onChange={(event) => setGeminiKey(event.target.value)}
              placeholder="Paste your Gemini key"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
            />
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy !== null}
              onClick={() => void pasteKey(setGeminiKey)}
            >
              Paste from clipboard
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={busy !== null || !geminiKey.trim()}
            >
              {busy === 'gemini' ? 'Connecting…' : 'Connect Gemini'}
            </button>
          </form>
        )}
      </section>

      <section className="setting-card">
        <h3>Ollama — runs on your computer</h3>
        <p className="muted small">
          Completely offline and free. Start Ollama on the machine running Jarvis
          (<code>ollama serve</code>) and pull at least one model.
        </p>
        <button
          type="button"
          className="btn btn-secondary"
          disabled={busy !== null}
          onClick={() => void run('ollama', { provider: 'ollama' })}
        >
          {busy === 'ollama' ? 'Looking for Ollama…' : 'Connect Ollama'}
        </button>
      </section>

      <section className="setting-card">
        <h3>Install on your iPhone</h3>
        <ol className="steps">
          <li>Open this page in Safari.</li>
          <li>
            Tap <strong>Share</strong>, then <strong>Add to Home Screen</strong>.
          </li>
          <li>Launch Jarvis from the home screen — it opens full screen.</li>
        </ol>
        <p className="muted small">
          Keep the computer awake while Jarvis is running; the phone talks to it
          over your Wi-Fi.
        </p>
      </section>

      <section className="setting-card">
        <h3>This phone</h3>
        <p className="muted small">
          The access key is stored only in this browser. Forget it to unpair.
        </p>
        <button
          type="button"
          className="btn btn-ghost is-danger"
          onClick={() => {
            setKey('');
            window.location.reload();
          }}
        >
          Forget access key
        </button>
      </section>
    </div>
  );
}
