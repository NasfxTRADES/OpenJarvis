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
  const [busy, setBusy] = useState<'openrouter' | 'ollama' | 'model' | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [models, setModels] = useState<string[]>(status?.models || []);
  const [selected, setSelected] = useState(status?.model || '');

  useEffect(() => {
    setModels(status?.models || []);
    setSelected(status?.model || '');
  }, [status?.models, status?.model]);

  const run = async (
    kind: 'openrouter' | 'ollama' | 'model',
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
      setNotice(`Connected to ${result.model}`);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not connect');
    } finally {
      setBusy(null);
    }
  };

  const submitKey = (event: FormEvent) => {
    event.preventDefault();
    if (!apiKey.trim()) return;
    void run('openrouter', { provider: 'openrouter', api_key: apiKey.trim() });
  };

  const provider = status?.provider === 'ollama' ? 'ollama' : 'openrouter';

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

      {status?.configured && models.length > 0 && (
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
            onClick={() => void run('model', { provider, model: selected })}
          >
            {busy === 'model' ? 'Switching…' : 'Use this model'}
          </button>
        </section>
      )}

      <section className="setting-card">
        <h3>OpenRouter — free models</h3>
        <p className="muted small">
          Create a free key at openrouter.ai/keys, paste it here once, and Jarvis
          picks from the models OpenRouter currently serves at zero cost. No card
          required.
        </p>
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
          <button type="submit" className="btn btn-primary" disabled={busy !== null || !apiKey.trim()}>
            {busy === 'openrouter' ? 'Connecting…' : 'Connect free model'}
          </button>
        </form>
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
