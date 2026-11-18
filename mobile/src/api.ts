// Thin same-origin client for the OpenJarvis API. The phone app is served by
// the OpenJarvis server, so no base URL or CORS handling is needed.

const KEY_STORAGE = 'oj.accessKey';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// A free Render instance answers 502 while it restarts or wakes up. That is a
// seconds-long window, not a failed turn, so a request waits it out instead of
// putting "Chat failed (502)" in front of the operator.
const GATEWAY_STATUS = new Set([502, 503, 504]);
const GATEWAY_RETRY_DELAYS = [700, 1500, 3000, 4000];

function pause(ms: number, signal?: AbortSignal | null): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const onAbort = () => {
      window.clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = window.setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Fetch, retrying the statuses a restarting server returns. */
async function fetchAwake(url: string, init: RequestInit): Promise<Response> {
  for (let attempt = 0; ; attempt += 1) {
    const last = attempt >= GATEWAY_RETRY_DELAYS.length;
    try {
      const response = await fetch(url, init);
      if (!GATEWAY_STATUS.has(response.status) || last) return response;
    } catch (error) {
      if (last || (error as Error)?.name === 'AbortError') throw error;
    }
    await pause(GATEWAY_RETRY_DELAYS[attempt], init.signal);
  }
}

function chatFailure(status: number): string {
  return GATEWAY_STATUS.has(status)
    ? 'Jarvis is starting up — the server was restarting. Try again in a moment.'
    : `Chat failed (${status})`;
}

export interface VoiceStatus {
  available: boolean;
  backend: string;
  reason?: string;
}

export interface ChainEntry {
  provider: string;
  label: string;
  model: string;
  ready: boolean;
  cooldown_seconds: number;
  last_error: string;
}

export interface MobileStatus {
  configured: boolean;
  provider: string;
  model: string;
  models: string[];
  chain?: ChainEntry[];
  active_provider?: string;
  tasks: number;
  free_models: string[];
  key_source?: string;
  gemini_key_source?: string;
  scheduler: boolean;
  voice?: VoiceStatus;
  speech_out?: VoiceStatus;
  server_time: string;
}

export interface TaskRun {
  id: number;
  task_id: string;
  started_at: string;
  finished_at: string | null;
  success: number;
  result: string;
  error: string;
}

export interface ScheduledTask {
  id: string;
  title: string;
  prompt: string;
  schedule_type: 'cron' | 'interval' | 'once';
  schedule_value: string;
  status: string;
  next_run: string | null;
  last_run: string | null;
  timezone: string;
  last_run_result?: TaskRun | null;
  running?: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export function getKey(): string {
  return localStorage.getItem(KEY_STORAGE) || '';
}

export function setKey(key: string): void {
  if (key) localStorage.setItem(KEY_STORAGE, key);
  else localStorage.removeItem(KEY_STORAGE);
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const key = getKey();
  return key ? { ...extra, Authorization: `Bearer ${key}` } : extra;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: headers({
      'Content-Type': 'application/json',
      ...((init.headers as Record<string, string>) || {}),
    }),
  });
  if (response.status === 401) {
    throw new ApiError(401, 'This phone is not paired with the server.');
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // non-JSON error body
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export function fetchStatus(): Promise<MobileStatus> {
  return request<MobileStatus>('/v1/mobile/status');
}

export function connectProvider(body: {
  provider: string;
  api_key?: string;
  model?: string;
  base_url?: string;
}): Promise<{ configured: boolean; provider: string; model: string; models: string[] }> {
  return request('/v1/mobile/setup', { method: 'POST', body: JSON.stringify(body) });
}

export function fetchTasks(): Promise<{ tasks: ScheduledTask[] }> {
  return request('/v1/mobile/tasks');
}

export function createTask(body: {
  prompt: string;
  schedule_type: string;
  schedule_value: string;
  title?: string;
  timezone?: string;
}): Promise<{ task: ScheduledTask }> {
  return request('/v1/mobile/tasks', { method: 'POST', body: JSON.stringify(body) });
}

export function updateTask(id: string, status: string): Promise<{ task: ScheduledTask }> {
  return request(`/v1/mobile/tasks/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });
}

export function deleteTask(id: string): Promise<{ deleted: boolean }> {
  return request(`/v1/mobile/tasks/${id}`, { method: 'DELETE' });
}

export function runTask(id: string): Promise<{ status: string }> {
  return request(`/v1/mobile/tasks/${id}/run`, { method: 'POST' });
}

export function fetchRuns(id: string): Promise<{ runs: TaskRun[] }> {
  return request(`/v1/mobile/tasks/${id}/runs`);
}

/** Mint a short-lived URL an <audio> element can stream from. */
export async function speechUrl(text: string, speed = 1.2): Promise<string> {
  const data = await request<{ url: string }>('/v1/mobile/speech/url', {
    method: 'POST',
    body: JSON.stringify({ text, speed }),
  });
  return data.url;
}

/** Read an OpenAI-style SSE body as text deltas. */
async function* sseDeltas(response: Response): AsyncGenerator<string> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (!data || data === '[DONE]') continue;
        try {
          const chunk = JSON.parse(data);
          const delta = chunk?.choices?.[0]?.delta?.content;
          if (delta) yield String(delta);
        } catch {
          // tolerate keep-alive or partial frames
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Stream a spoken reply straight from the engine.
 *
 * This skips the tool-using agent, which finishes its whole loop before it
 * says anything: the first words arrive while the rest is still generating.
 */
export async function* streamVoice(
  messages: ChatMessage[],
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const response = await fetchAwake('/v1/mobile/reply', {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ messages, max_tokens: 320 }),
    signal,
  });
  if (response.status === 401) {
    throw new ApiError(401, 'This phone is not paired with the server.');
  }
  if (!response.ok || !response.body) {
    let detail = chatFailure(response.status);
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore
    }
    throw new ApiError(response.status, detail);
  }
  yield* sseDeltas(response);
}

/** Turn a reply into audio with the server's configured voice. */
export async function synthesize(text: string, speed = 1.2): Promise<Blob> {
  const response = await fetch('/v1/mobile/speech', {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ text, speed }),
  });
  if (response.status === 401) {
    throw new ApiError(401, 'This phone is not paired with the server.');
  }
  if (!response.ok) {
    let detail = `Voice failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // non-JSON error body
    }
    throw new ApiError(response.status, detail);
  }
  return await response.blob();
}

/** Send recorded audio to the server's speech-to-text backend. */
export async function transcribeAudio(
  blob: Blob,
  filename: string,
): Promise<string> {
  const body = new FormData();
  body.append('file', blob, filename);
  const response = await fetch('/v1/speech/transcribe', {
    method: 'POST',
    headers: headers(),
    body,
  });
  if (response.status === 401) {
    throw new ApiError(401, 'This phone is not paired with the server.');
  }
  if (!response.ok) {
    let detail = `Transcription failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) detail = String(payload.detail);
    } catch {
      // ignore non-JSON errors
    }
    throw new ApiError(response.status, detail);
  }
  const data = (await response.json()) as { text?: string };
  return String(data.text || '').trim();
}

/** Stream assistant text from the server's tool-using agent. */
export async function* streamChat(
  model: string,
  messages: ChatMessage[],
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const response = await fetchAwake('/v1/chat/completions', {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ model, messages, stream: true }),
    signal,
  });
  if (response.status === 401) {
    throw new ApiError(401, 'This phone is not paired with the server.');
  }
  if (!response.ok || !response.body) {
    let detail = chatFailure(response.status);
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore
    }
    throw new ApiError(response.status, detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (!data || data === '[DONE]') continue;
        try {
          const chunk = JSON.parse(data);
          const delta = chunk?.choices?.[0]?.delta?.content;
          if (delta) yield String(delta);
        } catch {
          // tolerate keep-alive or partial frames
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
