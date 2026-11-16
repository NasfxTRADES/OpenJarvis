import { useCallback, useEffect, useState, type FormEvent } from 'react';
import {
  ApiError,
  fetchStatus,
  getKey,
  setKey,
  type MobileStatus,
} from './api';
import { ChatView } from './views/ChatView';
import { TasksView } from './views/TasksView';
import { SettingsView } from './views/SettingsView';
import './styles.css';

type Tab = 'chat' | 'tasks' | 'settings';

const TAB_META: Array<{ id: Tab; label: string; icon: string }> = [
  { id: 'chat', label: 'Chat', icon: '💬' },
  { id: 'tasks', label: 'Tasks', icon: '⏱' },
  { id: 'settings', label: 'Settings', icon: '⚙️' },
];

export default function App() {
  const [paired, setPaired] = useState<boolean | null>(null);
  const [tab, setTab] = useState<Tab>('chat');
  const [status, setStatus] = useState<MobileStatus | null>(null);
  const [statusError, setStatusError] = useState('');
  const [pairError, setPairError] = useState('');

  // Pairing is only needed when the server actually requires an access key.
  // A keyless local server answers the first status call and the app opens.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await fetchStatus();
        if (!cancelled) setPaired(true);
      } catch (error) {
        if (cancelled) return;
        if (error instanceof ApiError && error.status === 401) {
          setPairError(getKey() ? 'That access key was not accepted.' : '');
          setPaired(false);
        } else {
          setPaired(true);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const next = await fetchStatus();
      setStatus(next);
      setStatusError('');
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setPaired(false);
        setPairError('That access key was not accepted.');
        return;
      }
      setStatusError(error instanceof Error ? error.message : 'Server unreachable');
    }
  }, []);

  useEffect(() => {
    if (!paired) return;
    void refreshStatus();
    const interval = setInterval(() => void refreshStatus(), 60000);
    return () => clearInterval(interval);
  }, [paired, refreshStatus]);

  if (paired === null) {
    return <div className="pair" />;
  }

  if (!paired) {
    return (
      <PairScreen
        error={pairError}
        onPair={async (key) => {
          setKey(key.trim());
          try {
            await fetchStatus();
            setPairError('');
            setPaired(true);
          } catch (error) {
            setKey('');
            setPairError(
              error instanceof ApiError && error.status === 401
                ? 'That access key was not accepted.'
                : 'Could not reach the server on this address.',
            );
          }
        }}
      />
    );
  }

  const connected = Boolean(status?.configured);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>Jarvis</h1>
            <p className="brand-sub">
              {status ? status.model || 'not connected' : 'connecting…'}
            </p>
          </div>
        </div>
        <button
          type="button"
          className={`status-pill ${connected ? 'is-live' : 'is-idle'}`}
          onClick={() => setTab('settings')}
        >
          <span className="dot" />
          {connected ? 'Live' : 'Connect'}
        </button>
      </header>

      <main className="app-main">
        {statusError && <div className="banner banner-error">{statusError}</div>}
        {tab === 'chat' && (
          <ChatView status={status} onOpenSettings={() => setTab('settings')} />
        )}
        {tab === 'tasks' && (
          <TasksView
            status={status}
            onRefreshStatus={() => void refreshStatus()}
            onOpenSettings={() => setTab('settings')}
          />
        )}
        {tab === 'settings' && (
          <SettingsView status={status} onChanged={() => void refreshStatus()} />
        )}
      </main>

      <nav className="tabbar" aria-label="Sections">
        {TAB_META.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`tab ${tab === item.id ? 'is-active' : ''}`}
            onClick={() => setTab(item.id)}
          >
            <span className="tab-icon" aria-hidden="true">
              {item.icon}
            </span>
            {item.label}
          </button>
        ))}
      </nav>
    </div>
  );
}

function PairScreen({
  error,
  onPair,
}: {
  error: string;
  onPair: (key: string) => Promise<void>;
}) {
  const [value, setValue] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!value.trim() || busy) return;
    setBusy(true);
    try {
      await onPair(value);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="pair">
      <div className="pair-card">
        <div className="pair-logo" aria-hidden="true">
          J
        </div>
        <h1>Pair this phone</h1>
        <p>
          Type the access key printed by <code>jarvis mobile</code> on your
          computer. It is remembered on this device.
        </p>
        <form onSubmit={submit}>
          <input
            value={value}
            onChange={(event) => setValue(event.target.value)}
            placeholder="oj_sk_…"
            autoCapitalize="none"
            autoCorrect="off"
            autoComplete="off"
            spellCheck={false}
            inputMode="text"
          />
          <button type="submit" className="btn btn-primary" disabled={busy}>
            {busy ? 'Checking…' : 'Connect'}
          </button>
        </form>
        {error && <p className="form-error">{error}</p>}
      </div>
    </div>
  );
}
