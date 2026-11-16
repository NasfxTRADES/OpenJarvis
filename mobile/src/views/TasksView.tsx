import { useCallback, useEffect, useState, type FormEvent } from 'react';
import {
  createTask,
  deleteTask,
  fetchRuns,
  fetchTasks,
  runTask,
  updateTask,
  type MobileStatus,
  type ScheduledTask,
  type TaskRun,
} from '../api';
import { humanizeSchedule, relativeTime } from '../schedule';

type Repeat = 'once' | 'hourly' | 'daily' | 'weekdays';

function defaultOnceValue(): string {
  const date = new Date(Date.now() + 60 * 60 * 1000);
  date.setSeconds(0, 0);
  const offset = date.getTimezoneOffset();
  const local = new Date(date.getTime() - offset * 60000);
  return local.toISOString().slice(0, 16);
}

function buildSchedule(
  repeat: Repeat,
  time: string,
  onceAt: string,
): { schedule_type: string; schedule_value: string } {
  if (repeat === 'once') {
    const when = new Date(onceAt);
    return { schedule_type: 'once', schedule_value: when.toISOString() };
  }
  if (repeat === 'hourly') {
    return { schedule_type: 'interval', schedule_value: '3600' };
  }
  const [hours, minutes] = time.split(':').map(Number);
  const cron = `${Number.isFinite(minutes) ? minutes : 8} ${
    Number.isFinite(hours) ? hours : 8
  } * * ${repeat === 'weekdays' ? '1-5' : '*'}`;
  return { schedule_type: 'cron', schedule_value: cron };
}

export function TasksView({
  status,
  onRefreshStatus,
  onOpenSettings,
}: {
  status: MobileStatus | null;
  onRefreshStatus: () => void;
  onOpenSettings: () => void;
}) {
  const [tasks, setTasks] = useState<ScheduledTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [sheetOpen, setSheetOpen] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [repeat, setRepeat] = useState<Repeat>('daily');
  const [time, setTime] = useState('08:00');
  const [onceAt, setOnceAt] = useState(defaultOnceValue);
  const [creating, setCreating] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [runs, setRuns] = useState<Record<string, TaskRun[]>>({});

  const refresh = useCallback(async () => {
    try {
      const data = await fetchTasks();
      setTasks(data.tasks);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load tasks');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = setInterval(() => void refresh(), 15000);
    return () => clearInterval(interval);
  }, [refresh]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!prompt.trim() || creating) return;
    setCreating(true);
    setError('');
    try {
      const schedule = buildSchedule(repeat, time, onceAt);
      await createTask({
        prompt: prompt.trim(),
        schedule_type: schedule.schedule_type,
        schedule_value: schedule.schedule_value,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || '',
      });
      setPrompt('');
      setSheetOpen(false);
      await refresh();
      onRefreshStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create the task');
    } finally {
      setCreating(false);
    }
  };

  const toggle = async (task: ScheduledTask) => {
    try {
      await updateTask(task.id, task.status === 'active' ? 'paused' : 'active');
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not update the task');
    }
  };

  const remove = async (task: ScheduledTask) => {
    if (!confirm(`Delete "${task.title}"?`)) return;
    try {
      await deleteTask(task.id);
      await refresh();
      onRefreshStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete the task');
    }
  };

  const runNow = async (task: ScheduledTask) => {
    try {
      await runTask(task.id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start the task');
    }
  };

  const openRuns = async (task: ScheduledTask) => {
    if (expanded === task.id) {
      setExpanded(null);
      return;
    }
    setExpanded(task.id);
    try {
      const data = await fetchRuns(task.id);
      setRuns((prev) => ({ ...prev, [task.id]: data.runs }));
    } catch {
      // keep the cached runs if the request fails
    }
  };

  return (
    <div className="tasks">
      <div className="section-head">
        <div>
          <h2>Automations</h2>
          <p>Tasks Jarvis runs for you, on your schedule.</p>
        </div>
        <button
          type="button"
          className="btn btn-primary btn-small"
          onClick={() => setSheetOpen(true)}
        >
          + New
        </button>
      </div>

      {!status?.configured && (
        <div className="banner banner-setup">
          <strong>No model connected.</strong>
          <span>Scheduled tasks need a model. Connect a free one first.</span>
          <button type="button" className="btn btn-small" onClick={onOpenSettings}>
            Open settings
          </button>
        </div>
      )}

      {error && <div className="banner banner-error">{error}</div>}

      {loading && <p className="muted">Loading…</p>}

      {!loading && tasks.length === 0 && (
        <div className="empty-card">
          <p className="empty-title">Nothing scheduled yet</p>
          <p className="muted">
            Try “Every morning at 8, send me a summary of AI news” — either here or
            straight in chat.
          </p>
        </div>
      )}

      <div className="task-list">
        {tasks.map((task) => {
          const last = task.last_run_result ?? null;
          return (
            <article key={task.id} className={`task-card is-${task.status}`}>
              <div className="task-top">
                <div className="task-title">{task.title}</div>
                <span className={`pill is-${task.status}`}>
                  {task.running ? 'running' : task.status}
                </span>
              </div>
              {task.prompt !== task.title && (
                <p className="task-prompt">{task.prompt}</p>
              )}
              <div className="task-meta">
                <span>{humanizeSchedule(task.schedule_type, task.schedule_value)}</span>
                <span className="meta-sep">·</span>
                <span>
                  {task.status === 'active'
                    ? `next ${relativeTime(task.next_run)}`
                    : `last ${relativeTime(task.last_run)}`}
                </span>
              </div>

              {last && (
                <div className="task-result">
                  <span className={`result-dot ${last.success ? 'ok' : 'fail'}`} />
                  <span className="result-preview">
                    {(last.error || last.result || 'No output').slice(0, 160)}
                  </span>
                </div>
              )}

              <div className="task-actions">
                <button type="button" className="btn btn-ghost" onClick={() => void runNow(task)}>
                  Run now
                </button>
                <button type="button" className="btn btn-ghost" onClick={() => void toggle(task)}>
                  {task.status === 'active' ? 'Pause' : 'Resume'}
                </button>
                <button type="button" className="btn btn-ghost" onClick={() => void openRuns(task)}>
                  {expanded === task.id ? 'Hide runs' : 'Runs'}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost is-danger"
                  onClick={() => void remove(task)}
                >
                  Delete
                </button>
              </div>

              {expanded === task.id && (
                <div className="runs">
                  {(runs[task.id] || []).length === 0 && (
                    <p className="muted">No runs yet.</p>
                  )}
                  {(runs[task.id] || []).map((run) => (
                    <div key={run.id} className="run">
                      <div className="run-head">
                        <span className={`result-dot ${run.success ? 'ok' : 'fail'}`} />
                        {new Date(run.started_at).toLocaleString()}
                      </div>
                      <p className="run-body">{run.error || run.result || 'No output'}</p>
                    </div>
                  ))}
                </div>
              )}
            </article>
          );
        })}
      </div>

      {sheetOpen && (
        <div className="sheet-backdrop" onClick={() => setSheetOpen(false)}>
          <form
            className="sheet"
            onClick={(event) => event.stopPropagation()}
            onSubmit={submit}
          >
            <div className="sheet-grip" aria-hidden="true" />
            <h3>New automation</h3>
            <label className="field">
              <span>What should Jarvis do?</span>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Summarise the top AI news and email me the highlights"
                rows={3}
                autoFocus
              />
            </label>

            <label className="field">
              <span>When</span>
              <div className="segmented">
                {(
                  [
                    ['once', 'Once'],
                    ['hourly', 'Hourly'],
                    ['daily', 'Daily'],
                    ['weekdays', 'Weekdays'],
                  ] as Array<[Repeat, string]>
                ).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={repeat === value ? 'is-selected' : ''}
                    onClick={() => setRepeat(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </label>

            {repeat === 'once' ? (
              <label className="field">
                <span>Run at</span>
                <input
                  type="datetime-local"
                  value={onceAt}
                  onChange={(event) => setOnceAt(event.target.value)}
                />
              </label>
            ) : repeat === 'hourly' ? (
              <p className="muted small">Runs every hour, starting in one hour.</p>
            ) : (
              <label className="field">
                <span>At</span>
                <input
                  type="time"
                  value={time}
                  onChange={(event) => setTime(event.target.value)}
                />
              </label>
            )}

            <p className="muted small">
              Times use this phone's timezone ({Intl.DateTimeFormat().resolvedOptions().timeZone}).
            </p>

            <div className="sheet-actions">
              <button type="button" className="btn btn-ghost" onClick={() => setSheetOpen(false)}>
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={creating || !prompt.trim()}
              >
                {creating ? 'Saving…' : 'Create'}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

