// The Jarvis HUD: one screen where every readout is real application state.
// The core reacts to the live voice mode, the waveform is the actual
// microphone or reply-audio level, and the panels report measured values.
//
// Talking is hands-free: one tap opens the microphone and the conversation
// keeps turning — listen, answer, speak, listen again — until the same control
// ends it. Speaking over Jarvis cuts the reply short and starts a new turn.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ApiError,
  streamChat,
  transcribeAudio,
  type ChatMessage,
  type MobileStatus,
} from '../api';
import {
  MicMeter,
  SentenceSplitter,
  SpeechQueue,
  createRecognizer,
  recognitionError,
  recognitionSupported,
  type Recognizer,
} from '../voice';
import '../jarvis.css';

type Mode = 'idle' | 'listening' | 'thinking' | 'speaking';

interface ActivityEvent {
  id: number;
  time: string;
  text: string;
}

const TRANSCRIPT_STORAGE = 'oj.voice.v1';
const ACTIVITY_LIMIT = 5;
const WAVE_BARS = 56;
const MINI_BARS = 40;
// Sustained microphone level that means the operator started talking over the
// reply. Echo cancellation keeps Jarvis's own voice below this.
const BARGE_IN_LEVEL = 0.2;
const BARGE_IN_TICKS = 3;

const COMMANDS = [
  'What is my status?',
  'Summarise the top news right now.',
  'Explain quantum computing simply.',
  'Draft a short email to my team.',
];

const MODE_LABEL: Record<Mode, string> = {
  idle: 'STANDBY',
  listening: 'LISTENING...',
  thinking: 'PROCESSING',
  speaking: 'SPEAKING',
};

function clockTime(date: Date): string {
  return date.toLocaleTimeString('en-GB', { hour12: false });
}

function clockShort(): string {
  return new Date().toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

function clockDate(date: Date): string {
  return date
    .toLocaleDateString('en-US', {
      weekday: 'short',
      day: '2-digit',
      month: 'short',
      year: 'numeric',
    })
    .toUpperCase();
}

function truncate(value: string, limit: number): string {
  const text = value.replace(/\s+/g, ' ').trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function shortModel(model: string): string {
  if (!model) return '—';
  const tail = model.split('/').pop() || model;
  return truncate(tail, 22);
}

function loadTranscript(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(TRANSCRIPT_STORAGE);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item) =>
        item &&
        (item.role === 'user' || item.role === 'assistant') &&
        typeof item.content === 'string',
    );
  } catch {
    return [];
  }
}

/** Mirrored amplitude bars drawn straight from a live level reader. */
function Waveform({
  readLevel,
  active,
  bars = WAVE_BARS,
  className = '',
}: {
  readLevel: () => number;
  active: boolean;
  bars?: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const history = useRef<number[]>(new Array(bars).fill(0));
  const readRef = useRef(readLevel);
  readRef.current = readLevel;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let frame = 0;
    const draw = () => {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (width && height) {
        if (
          canvas.width !== Math.round(width * ratio) ||
          canvas.height !== Math.round(height * ratio)
        ) {
          canvas.width = Math.round(width * ratio);
          canvas.height = Math.round(height * ratio);
        }
        const context = canvas.getContext('2d');
        if (context) {
          context.setTransform(ratio, 0, 0, ratio, 0, 0);
          context.clearRect(0, 0, width, height);
          history.current.push(active ? Math.min(1, readRef.current()) : 0);
          while (history.current.length > bars) history.current.shift();
          const step = width / bars;
          context.strokeStyle = 'rgba(255, 74, 74, 0.85)';
          context.shadowColor = 'rgba(255, 40, 40, 0.75)';
          context.shadowBlur = 7;
          context.lineWidth = Math.max(1, step * 0.42);
          history.current.forEach((level, index) => {
            const amplitude = Math.max(0.02, level) * height * 0.48;
            const x = index * step + step / 2;
            context.beginPath();
            context.moveTo(x, height / 2 - amplitude);
            context.lineTo(x, height / 2 + amplitude);
            context.stroke();
          });
        }
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [active, bars]);

  return <canvas ref={canvasRef} className={`hud-wave ${className}`} aria-hidden="true" />;
}

/** Concentric instrument rings; the mode changes their speed and glow. */
function CoreRings() {
  return (
    <svg className="core-svg" viewBox="0 0 400 400" aria-hidden="true">
      <defs>
        <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
          <stop offset="55%" stopColor="rgba(255,40,40,0)" />
          <stop offset="100%" stopColor="rgba(255,40,40,0.22)" />
        </radialGradient>
      </defs>
      <circle cx="200" cy="200" r="196" fill="url(#coreGlow)" />
      <circle className="ring ring-ticks" cx="200" cy="200" r="182" />
      <circle className="ring ring-outer" cx="200" cy="200" r="168" />
      <circle className="ring ring-dashed" cx="200" cy="200" r="150" />
      <circle className="ring ring-arc" cx="200" cy="200" r="132" />
      <circle className="ring ring-arc ring-arc-b" cx="200" cy="200" r="118" />
      <circle className="ring ring-inner" cx="200" cy="200" r="96" />
      <circle className="ring ring-inner-dashed" cx="200" cy="200" r="82" />
      <circle className="ring ring-core" cx="200" cy="200" r="62" />
      <circle className="ring ring-idle" cx="200" cy="200" r="30" />
    </svg>
  );
}

function MicGlyph() {
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 15a3.5 3.5 0 0 0 3.5-3.5V6a3.5 3.5 0 1 0-7 0v5.5A3.5 3.5 0 0 0 12 15Zm5.5-3.5a5.5 5.5 0 0 1-11 0H4.5a7.5 7.5 0 0 0 6.75 7.46V21h1.5v-2.04A7.5 7.5 0 0 0 19.5 11.5h-2Z"
      />
    </svg>
  );
}

export function JarvisView({
  status,
  onOpenSettings,
  onOpenChat,
}: {
  status: MobileStatus | null;
  onOpenSettings: () => void;
  onOpenChat: () => void;
}) {
  const [transcript, setTranscript] = useState<ChatMessage[]>(loadTranscript);
  const [partial, setPartial] = useState('');
  const [conversation, setConversation] = useState(false);
  const [listening, setListening] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [activity, setActivity] = useState<ActivityEvent[]>([]);
  const [error, setError] = useState('');
  const [now, setNow] = useState(() => new Date());
  const [fallbackVoice, setFallbackVoice] = useState(false);
  const [recording, setRecording] = useState(false);

  const transcriptRef = useRef(transcript);
  const conversationRef = useRef(false);
  const speakingRef = useRef(false);
  const thinkingRef = useRef(false);
  const busyRef = useRef(false);
  const idRef = useRef(0);
  const meterRef = useRef<MicMeter | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognizerRef = useRef<Recognizer | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const sendRef = useRef<(text: string) => Promise<void>>(async () => {});

  const queue = useMemo(
    () => new SpeechQueue(setSpeaking, (usedServerVoice) => setFallbackVoice(!usedServerVoice)),
    [],
  );

  const mode: Mode = speaking
    ? 'speaking'
    : thinking
      ? 'thinking'
      : conversation
        ? 'listening'
        : 'idle';

  const log = useCallback((text: string) => {
    idRef.current += 1;
    const event = { id: idRef.current, time: clockShort(), text };
    setActivity((current) => [event, ...current].slice(0, ACTIVITY_LIMIT));
  }, []);

  useEffect(() => {
    transcriptRef.current = transcript;
    localStorage.setItem(TRANSCRIPT_STORAGE, JSON.stringify(transcript.slice(-40)));
  }, [transcript]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    log('Core online');
    return () => {
      queue.dispose();
      meterRef.current?.dispose();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      recognizerRef.current?.abort();
      abortRef.current?.abort();
    };
  }, [log, queue]);

  /** Restart the recogniser for the next turn, if a turn is in progress. */
  const ensureListening = useCallback(() => {
    if (!conversationRef.current || speakingRef.current || thinkingRef.current) return;
    const recognizer = recognizerRef.current;
    if (!recognizer) return;
    try {
      recognizer.start();
    } catch {
      // Already running; the onend handler will restart it when it stops.
    }
  }, []);

  const startRecordingFallback = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || typeof MediaRecorder === 'undefined') {
      setError('This browser cannot record audio.');
      return;
    }
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, {
        type: recorder.mimeType || 'audio/webm',
      });
      chunksRef.current = [];
      setRecording(false);
      if (!blob.size) return;
      void transcribeAudio(blob, 'voice.webm')
        .then((text) => {
          if (text) void sendRef.current(text);
          else if (conversationRef.current) startRecordingFallback();
        })
        .catch((err: unknown) => {
          setError(err instanceof ApiError ? err.message : 'Could not transcribe that.');
          if (conversationRef.current) startRecordingFallback();
        });
    };
    recorder.start();
    recorderRef.current = recorder;
    setRecording(true);
  }, []);

  const startRecognising = useCallback(() => {
    const recognizer = createRecognizer();
    if (!recognizer) return false;
    recognizerRef.current = recognizer;
    recognizer.onresult = (event) => {
      // While Jarvis speaks the recogniser is stopped, but a late result must
      // still never be mistaken for the operator's next turn.
      if (speakingRef.current) return;
      let interim = '';
      let final = '';
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const text = result?.[0]?.transcript || '';
        if (result?.isFinal) final += text;
        else interim += text;
      }
      if (interim.trim()) setPartial(interim.trim());
      if (final.trim()) {
        setPartial('');
        void sendRef.current(final.trim());
      }
    };
    recognizer.onerror = (event) => {
      const message = recognitionError(event.error);
      if (message) setError(message);
      if (['not-allowed', 'service-not-allowed', 'audio-capture'].includes(event.error)) {
        conversationRef.current = false;
        setConversation(false);
      }
    };
    recognizer.onend = () => {
      if (!conversationRef.current || speakingRef.current) return;
      // Browsers drop a session after a silence; keep the conversation open.
      window.setTimeout(() => ensureListening(), 250);
    };
    try {
      recognizer.start();
      return true;
    } catch {
      return false;
    }
  }, [ensureListening]);

  const send = useCallback(
    async (text: string) => {
      const prompt = text.trim();
      if (!prompt || busyRef.current) return;
      if (!status?.configured) {
        setError('Connect an AI model in Settings before talking to Jarvis.');
        return;
      }
      busyRef.current = true;
      setError('');
      // Only the recent turns go to the model: a long history costs time to
      // first token, and a spoken conversation rarely needs the whole log.
      const history: ChatMessage[] = [
        ...transcriptRef.current.filter((message) => message.content).slice(-8),
        { role: 'user', content: prompt },
      ];
      const pending: ChatMessage[] = [...history, { role: 'assistant', content: '' }];
      setTranscript(pending);
      transcriptRef.current = pending;
      setThinking(true);
      thinkingRef.current = true;
      log(`Request: ${truncate(prompt, 30)}`);

      const splitter = new SentenceSplitter();
      const started = performance.now();
      let answer = '';
      let first = true;
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        for await (const token of streamChat(status?.model || '', history, controller.signal)) {
          if (first) {
            first = false;
            setLatency(Math.round(performance.now() - started));
            setThinking(false);
            thinkingRef.current = false;
          }
          answer += token;
          for (const sentence of splitter.push(token)) queue.enqueue(sentence);
          const next: ChatMessage[] = [
            ...history,
            { role: 'assistant', content: answer },
          ];
          transcriptRef.current = next;
          setTranscript(next);
        }
        const tail = splitter.flush();
        if (tail) queue.enqueue(tail);
        if (answer.trim()) log('Reply ready');
        else setError('The model returned an empty answer.');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') {
          log('Stopped');
        } else {
          setError(err instanceof ApiError ? err.message : 'Could not reach Jarvis.');
          log('Request failed');
        }
      } finally {
        setThinking(false);
        thinkingRef.current = false;
        busyRef.current = false;
        abortRef.current = null;
      }
    },
    [log, queue, status],
  );

  useEffect(() => {
    sendRef.current = send;
  }, [send]);

  const releaseMic = useCallback(() => {
    recognizerRef.current?.abort();
    recognizerRef.current = null;
    meterRef.current?.dispose();
    meterRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const recorder = recorderRef.current;
    recorderRef.current = null;
    if (recorder && recorder.state === 'recording') recorder.stop();
    setListening(false);
    setRecording(false);
    setPartial('');
  }, []);

  const startConversation = useCallback(async () => {
    if (!status?.configured) {
      setError('Connect an AI model in Settings before talking to Jarvis.');
      return;
    }
    if (typeof window !== 'undefined' && !window.isSecureContext) {
      setError('The microphone needs an https connection.');
      return;
    }
    setError('');
    // Unlock playback inside this tap: the first reply arrives later, and iOS
    // will not start audio outside a user gesture.
    queue.prime();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;
      const meter = new MicMeter();
      await meter.attach(stream);
      meterRef.current = meter;
    } catch {
      setError('Microphone access was denied. Allow it for this site, then tap the core.');
      return;
    }
    conversationRef.current = true;
    setConversation(true);
    setListening(true);
    log('Microphone live');
    if (!startRecognising()) startRecordingFallback();
  }, [log, queue, startRecognising, startRecordingFallback, status]);

  const endConversation = useCallback(() => {
    conversationRef.current = false;
    setConversation(false);
    queue.stop();
    abortRef.current?.abort();
    releaseMic();
    log('Conversation ended');
  }, [log, queue, releaseMic]);

  const toggleMic = useCallback(() => {
    if (speaking) {
      // Interrupt the reply but stay in the conversation.
      queue.stop();
      window.setTimeout(() => ensureListening(), 0);
      log('Interrupted');
      return;
    }
    if (conversation) {
      endConversation();
      return;
    }
    void startConversation();
  }, [conversation, endConversation, ensureListening, log, queue, speaking, startConversation]);

  // Speaking stops the recogniser (so Jarvis never transcribes himself),
  // watches the microphone for a barge-in, and hands back to listening the
  // moment the reply ends.
  useEffect(() => {
    speakingRef.current = speaking;
    thinkingRef.current = thinking;
    if (!conversation) return;
    if (speaking) {
      recognizerRef.current?.abort();
      let hits = 0;
      const watch = window.setInterval(() => {
        const level = meterRef.current?.read() ?? 0;
        if (level > BARGE_IN_LEVEL) {
          hits += 1;
          if (hits >= BARGE_IN_TICKS) {
            window.clearInterval(watch);
            queue.stop();
            log('Interrupted');
            window.setTimeout(() => ensureListening(), 0);
          }
        } else {
          hits = 0;
        }
      }, 90);
      return () => window.clearInterval(watch);
    }
    if (!thinking) ensureListening();
  }, [speaking, thinking, conversation, ensureListening, log, queue]);

  const readLevel = useCallback(() => {
    if (speaking) return queue.readLevel();
    if (conversation) return meterRef.current?.read() ?? 0;
    return 0;
  }, [conversation, queue, speaking]);

  const turns = transcript.filter((message) => message.role === 'user').length;
  const lastUser = [...transcript].reverse().find((message) => message.role === 'user');
  const lastReply = [...transcript].reverse().find(
    (message) => message.role === 'assistant' && message.content,
  );
  const configured = Boolean(status?.configured);
  const voiceBackend = status?.speech_out?.backend || '';
  const contextLine = lastUser
    ? truncate(lastUser.content, 60)
    : configured
      ? 'Ready for input...'
      : 'No AI core connected.';
  const micLabel = speaking
    ? 'TAP TO INTERRUPT'
    : conversation
      ? 'TAP TO END CONVERSATION'
      : 'TAP TO TALK';

  return (
    <div className={`hud hud-${mode}`}>
      <div className="hud-corners" aria-hidden="true" />

      <header className="hud-header">
        <div className="hud-identity">
          <h1>J.A.R.V.I.S.</h1>
          <p className={`hud-online ${configured ? 'is-live' : ''}`}>
            <span className="hud-online-dot" aria-hidden="true" />
            {configured ? 'ONLINE' : 'OFFLINE'}
          </p>
          <div className="hud-nav">
            <button type="button" onClick={onOpenChat} className="hud-link">
              LOG
            </button>
            <button type="button" onClick={onOpenSettings} className="hud-link">
              CONFIG
            </button>
          </div>
        </div>
        <div className="hud-clock">
          <time>{clockTime(now)}</time>
          <span>{clockDate(now)}</span>
        </div>
      </header>

      <main className="hud-stage">
        <section className="hud-panel panel-status">
          <h2>System Status</h2>
          <dl className="hud-readouts">
            <div>
              <dt>LATENCY</dt>
              <dd>{latency === null ? '—' : `${latency} ms`}</dd>
            </div>
            <div>
              <dt>TURNS</dt>
              <dd>{turns}</dd>
            </div>
            <div>
              <dt>MODEL</dt>
              <dd title={status?.model || ''}>{shortModel(status?.model || '')}</dd>
            </div>
            <div>
              <dt>VOICE</dt>
              <dd>{voiceBackend || 'browser'}</dd>
            </div>
          </dl>
          <Waveform readLevel={readLevel} active={mode !== 'idle'} bars={MINI_BARS} className="wave-mini" />
        </section>

        <div className={`core core-${mode}`}>
          <CoreRings />
          <div className="core-readout">
            <p className="core-mode">{MODE_LABEL[mode]}</p>
            <p className="core-line">
              {partial ||
                lastReply?.content ||
                (conversation ? 'Listening for your voice…' : 'Tap the core and speak.')}
            </p>
          </div>
          <Waveform readLevel={readLevel} active={mode !== 'idle'} className="wave-core" />
        </div>

        <section className="hud-panel panel-core">
          <h2>AI Core</h2>
          <div className="core-card">
            <div>
              <strong>{shortModel(status?.model || '')}</strong>
              <span>{MODE_LABEL[mode]}</span>
            </div>
            <svg className="core-glyph" viewBox="0 0 64 64" aria-hidden="true">
              <circle cx="32" cy="32" r="28" />
              <circle cx="32" cy="32" r="19" />
              <path d="M32 4v56M4 32h56" />
              <circle cx="32" cy="32" r="7" className="is-filled" />
            </svg>
          </div>
          <div className="panel-sub">
            <h3>Context</h3>
            <p>{contextLine}</p>
          </div>
        </section>

        <section className="hud-panel panel-transcript">
          <h2>{transcript.length ? 'Transcript' : 'Voice Commands'}</h2>
          {transcript.length ? (
            <ol className="hud-transcript">
              {transcript.slice(-8).map((message, index) => (
                <li key={index} className={`line-${message.role}`}>
                  <span className="line-tag">{message.role === 'user' ? 'OP' : 'JRV'}</span>
                  <span className="line-text">{message.content || '…'}</span>
                </li>
              ))}
            </ol>
          ) : (
            <ul className="hud-commands">
              {COMMANDS.map((command) => (
                <li key={command}>
                  <button type="button" onClick={() => void send(command)}>
                    {command}
                  </button>
                </li>
              ))}
            </ul>
          )}
          <Waveform readLevel={readLevel} active={mode !== 'idle'} bars={MINI_BARS} className="wave-mini" />
        </section>

        <section className="hud-panel panel-activity">
          <h2>Recent Activity</h2>
          <ul className="hud-log">
            {activity.length === 0 && <li className="is-muted">No events yet</li>}
            {activity.map((event) => (
              <li key={event.id}>
                <time>{event.time}</time>
                <span className="log-dot" aria-hidden="true" />
                {event.text}
              </li>
            ))}
          </ul>
        </section>

        <div className="hud-mic-wrap">
          <button
            type="button"
            className={`hud-mic ${conversation ? 'is-live' : ''} ${
              speaking ? 'is-speaking' : ''
            }`}
            onClick={toggleMic}
            aria-label={conversation ? 'End the conversation' : 'Start talking'}
            aria-pressed={conversation}
          >
            <span className="mic-ring" aria-hidden="true" />
            <MicGlyph />
          </button>
          <p className="hud-mic-label">{micLabel}</p>
        </div>
      </main>

      {(error || !configured || (fallbackVoice && !voiceBackend)) && (
        <div className="hud-notice">
          {error ||
            (!configured
              ? 'No AI core connected. Open CONFIG to add a provider key.'
              : 'Speaking with the browser voice — add FISH_AUDIO_API_KEY on the server for your Jarvis voice.')}
        </div>
      )}
    </div>
  );
}
