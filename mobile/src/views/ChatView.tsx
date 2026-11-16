import { useEffect, useRef, useState, type FormEvent } from 'react';
import {
  ApiError,
  streamChat,
  transcribeAudio,
  type ChatMessage,
  type MobileStatus,
} from '../api';

const CHAT_STORAGE = 'oj.chat.v1';

const SUGGESTIONS = [
  'What is on my schedule today?',
  'Every morning at 8, summarise the top AI news',
  'Search the web for cheap flights to Tokyo',
  'Remind me in 1 hour to stretch',
];

const AUDIO_TYPES = ['audio/mp4', 'audio/webm;codecs=opus', 'audio/webm', 'audio/ogg'];

function loadMessages(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string',
    );
  } catch {
    return [];
  }
}

function pickAudioType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  return AUDIO_TYPES.find((type) => {
    try {
      return MediaRecorder.isTypeSupported(type);
    } catch {
      return false;
    }
  });
}

function extensionFor(type: string): string {
  if (type.includes('mp4')) return 'mp4';
  if (type.includes('ogg')) return 'ogg';
  return 'webm';
}

function formatSeconds(total: number): string {
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

export function ChatView({
  status,
  onOpenSettings,
}: {
  status: MobileStatus | null;
  onOpenSettings: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>(loadMessages);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [recording, setRecording] = useState(false);
  const [recordedFor, setRecordedFor] = useState(0);
  const [transcribing, setTranscribing] = useState(false);
  const [voiceError, setVoiceError] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const pressedRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const micSupported =
    typeof navigator !== 'undefined' &&
    Boolean(navigator.mediaDevices?.getUserMedia) &&
    typeof MediaRecorder !== 'undefined';
  const secureContext =
    typeof window !== 'undefined' &&
    (window.isSecureContext || ['localhost', '127.0.0.1'].includes(location.hostname));
  const voiceReady = Boolean(status?.voice?.available) && micSupported && secureContext;

  useEffect(() => {
    localStorage.setItem(CHAT_STORAGE, JSON.stringify(messages.slice(-80)));
    const node = listRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages]);

  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  // Grow the composer with the text so a spoken sentence is fully visible.
  useEffect(() => {
    const node = inputRef.current;
    if (!node) return;
    node.style.height = 'auto';
    node.style.height = `${Math.min(node.scrollHeight, 132)}px`;
  }, [input]);

  const send = async (text: string) => {
    const prompt = text.trim();
    if (!prompt || busy) return;
    if (!status?.configured) {
      setError('Connect a free AI model in Settings first.');
      return;
    }
    setError('');
    setInput('');
    const history: ChatMessage[] = [...messages, { role: 'user', content: prompt }];
    setMessages([...history, { role: 'assistant', content: '' }]);
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;
    let answer = '';
    try {
      for await (const token of streamChat(status?.model || '', history, controller.signal)) {
        answer += token;
        setMessages([...history, { role: 'assistant', content: answer }]);
      }
      if (!answer.trim()) {
        setMessages([
          ...history,
          { role: 'assistant', content: 'The model returned an empty answer. Try again.' },
        ]);
      }
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') {
        setMessages([...history, { role: 'assistant', content: answer || 'Stopped.' }]);
      } else {
        const message =
          err instanceof ApiError ? err.message : 'Could not reach the assistant.';
        setError(message);
        setMessages(history);
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void send(input);
  };

  const finishTranscription = async (blob: Blob) => {
    setTranscribing(true);
    try {
      const text = await transcribeAudio(blob, `voice.${extensionFor(blob.type)}`);
      if (text) {
        setVoiceError('');
        setInput((previous) => (previous.trim() ? `${previous.trimEnd()} ${text}` : text));
        window.setTimeout(() => inputRef.current?.focus(), 0);
      } else {
        setVoiceError("Didn't catch that — hold the mic and try again.");
      }
    } catch (err) {
      setVoiceError(
        err instanceof ApiError ? err.message : 'Could not transcribe that recording.',
      );
    } finally {
      setTranscribing(false);
    }
  };

  const stopRecording = () => {
    pressedRef.current = false;
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setRecording(false);
    const recorder = recorderRef.current;
    recorderRef.current = null;
    if (recorder && recorder.state === 'recording') recorder.stop();
  };

  const startRecording = async () => {
    if (busy || transcribing || recording) return;
    if (!micSupported) {
      setVoiceError('This browser cannot record audio.');
      return;
    }
    if (!secureContext) {
      setVoiceError(
        'The microphone needs an https connection. Restart Jarvis with --tls and reopen the app.',
      );
      return;
    }
    if (!status?.voice?.available) {
      setVoiceError(
        status?.voice?.reason ||
          'Voice transcription is not enabled on the server (install: uv sync --extra speech).',
      );
      return;
    }
    setVoiceError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = pickAudioType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        });
        chunksRef.current = [];
        if (blob.size > 0) void finishTranscription(blob);
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
      setRecordedFor(0);
      timerRef.current = window.setInterval(() => {
        setRecordedFor((value) => value + 1);
      }, 1000);
      // A tap that ended before the permission prompt resolved should not
      // leave the microphone recording.
      if (!pressedRef.current) stopRecording();
    } catch (err) {
      setVoiceError(
        (err as Error)?.name === 'NotAllowedError'
          ? 'Microphone access was denied. Allow it in Settings → Safari → Microphone.'
          : 'Could not start recording.',
      );
    }
  };

  return (
    <div className="chat">
      <div className="chat-scroll" ref={listRef}>
        {!status?.configured && (
          <div className="banner banner-setup">
            <strong>Connect a free AI model to start.</strong>
            <span>
              Jarvis can use OpenRouter's free models or Ollama running on your own
              computer — no subscription.
            </span>
            <button type="button" className="btn btn-small" onClick={onOpenSettings}>
              Open settings
            </button>
          </div>
        )}

        {messages.length === 0 ? (
          <div className="chat-empty">
            <h2>What should I handle?</h2>
            <p>
              Ask anything, tell me a task to run on a schedule
              {voiceReady ? ', or hold the mic to talk' : ''}.
            </p>
            <div className="chips">
              {SUGGESTIONS.map((text) => (
                <button
                  key={text}
                  type="button"
                  className="chip"
                  onClick={() => void send(text)}
                >
                  {text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`bubble-row ${message.role}`}>
              <div className={`bubble ${message.role}`}>
                {message.content ? (
                  message.content
                ) : (
                  <span className="typing">
                    <i />
                    <i />
                    <i />
                  </span>
                )}
              </div>
            </div>
          ))
        )}

        {error && <div className="banner banner-error">{error}</div>}
      </div>

      {(recording || transcribing || voiceError) && (
        <div
          className={`voice-strip ${
            recording ? 'is-recording' : transcribing ? 'is-busy' : 'is-error'
          }`}
        >
          <span className="voice-dot" aria-hidden="true" />
          <span className="voice-text">
            {recording
              ? `Listening… ${formatSeconds(recordedFor)}`
              : transcribing
                ? 'Transcribing…'
                : voiceError}
          </span>
          {!recording && !transcribing && (
            <button
              type="button"
              className="voice-dismiss"
              aria-label="Dismiss"
              onClick={() => setVoiceError('')}
            >
              ×
            </button>
          )}
        </div>
      )}

      <form className="composer" onSubmit={submit}>
        <button
          type="button"
          className={`mic-btn ${recording ? 'is-recording' : ''}`}
          aria-label="Hold to talk"
          title={voiceReady ? 'Hold to talk' : 'Voice unavailable'}
          onContextMenu={(event) => event.preventDefault()}
          onPointerDown={(event) => {
            event.preventDefault();
            pressedRef.current = true;
            void startRecording();
          }}
          onPointerUp={(event) => {
            event.preventDefault();
            stopRecording();
          }}
          onPointerLeave={() => stopRecording()}
          onPointerCancel={() => stopRecording()}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <path
              fill="currentColor"
              d="M12 15a3.5 3.5 0 0 0 3.5-3.5V6a3.5 3.5 0 1 0-7 0v5.5A3.5 3.5 0 0 0 12 15Zm5.5-3.5a5.5 5.5 0 0 1-11 0H4.5a7.5 7.5 0 0 0 6.75 7.46V21h1.5v-2.04A7.5 7.5 0 0 0 19.5 11.5h-2Z"
            />
          </svg>
        </button>
        <textarea
          ref={inputRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              void send(input);
            }
          }}
          placeholder={recording ? 'Listening…' : 'Message Jarvis…'}
          rows={1}
          enterKeyHint="send"
        />
        {busy ? (
          <button
            type="button"
            className="send-btn is-stop"
            aria-label="Stop"
            onClick={() => abortRef.current?.abort()}
          >
            ■
          </button>
        ) : (
          <button
            type="submit"
            className="send-btn"
            aria-label="Send"
            disabled={!input.trim() || !status?.configured}
          >
            ↑
          </button>
        )}
      </form>
    </div>
  );
}
