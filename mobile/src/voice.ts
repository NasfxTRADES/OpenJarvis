// Browser voice plumbing for the Jarvis HUD: microphone level metering, the
// Web Speech recogniser, and a queue that plays reply audio sentence by
// sentence so Jarvis starts talking before the whole answer exists.

import { speechUrl, synthesize } from './api';

interface RecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface RecognitionResult {
  isFinal: boolean;
  length: number;
  [index: number]: RecognitionAlternative;
}

interface RecognitionEvent {
  resultIndex: number;
  results: { length: number; [index: number]: RecognitionResult };
}

interface RecognitionErrorEvent {
  error: string;
  message?: string;
}

export interface Recognizer {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onstart: (() => void) | null;
  onresult: ((event: RecognitionEvent) => void) | null;
  onerror: ((event: RecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

type RecognizerCtor = new () => Recognizer;

/** 24 kHz silent sample used to unlock audio playback inside a tap. */
const SILENT_WAV =
  'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAIA+AAABAAgAZGF0YQAAAAA=';

/** Slightly quicker than natural speech, which is how Jarvis should read. */
export const SPEECH_SPEED = 1.12;

function recognizerCtor(): RecognizerCtor | null {
  if (typeof window === 'undefined') return null;
  const scope = window as unknown as {
    SpeechRecognition?: RecognizerCtor;
    webkitSpeechRecognition?: RecognizerCtor;
  };
  return scope.SpeechRecognition || scope.webkitSpeechRecognition || null;
}

export function recognitionSupported(): boolean {
  return recognizerCtor() !== null;
}

export function createRecognizer(lang = 'en-US'): Recognizer | null {
  const Ctor = recognizerCtor();
  if (!Ctor) return null;
  const recognizer = new Ctor();
  recognizer.lang = lang;
  recognizer.continuous = true;
  recognizer.interimResults = true;
  recognizer.maxAlternatives = 1;
  return recognizer;
}

/** Human-readable reason for a recognition failure. */
export function recognitionError(error: string): string {
  switch (error) {
    case 'not-allowed':
    case 'service-not-allowed':
      return 'Microphone permission was denied. Allow it for this site, then tap the core again.';
    case 'audio-capture':
      return 'No microphone was found on this device.';
    case 'network':
      return 'Speech recognition needs a network connection on this browser.';
    case 'no-speech':
      return '';
    case 'aborted':
      return '';
    default:
      return `Speech recognition stopped (${error}).`;
  }
}

/** Root-mean-square level of a live stream, 0–1, for the waveform. */
export class MicMeter {
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private buffer: Float32Array<ArrayBuffer> | null = null;

  async attach(stream: MediaStream): Promise<void> {
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) return;
    this.context = new Ctor();
    this.source = this.context.createMediaStreamSource(stream);
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.6;
    this.source.connect(this.analyser);
    this.buffer = new Float32Array(new ArrayBuffer(512 * 4));
  }

  read(): number {
    if (!this.analyser || !this.buffer) return 0;
    this.analyser.getFloatTimeDomainData(this.buffer);
    let sum = 0;
    for (let i = 0; i < this.buffer.length; i += 1) {
      sum += this.buffer[i] * this.buffer[i];
    }
    return Math.min(1, Math.sqrt(sum / this.buffer.length) * 4.5);
  }

  dispose(): void {
    this.source?.disconnect();
    this.analyser?.disconnect();
    void this.context?.close();
    this.context = null;
    this.analyser = null;
    this.source = null;
    this.buffer = null;
  }
}

/**
 * Splits streamed reply text into speakable pieces.
 *
 * The first piece is cut early — at a clause boundary if there is one — because
 * waiting for a full sentence is most of the delay before Jarvis's first word.
 */
export class SentenceSplitter {
  private buffer = '';
  private started = false;

  push(chunk: string): string[] {
    this.buffer += chunk;
    const ready: string[] = [];
    const sentence = /[.!?…]+["')\]]*\s/;
    let match = sentence.exec(this.buffer);
    while (match) {
      const end = match.index + match[0].length;
      const piece = this.buffer.slice(0, end).trim();
      this.buffer = this.buffer.slice(end);
      if (piece) ready.push(piece);
      match = sentence.exec(this.buffer);
    }
    if (ready.length) this.started = true;

    if (!this.started && this.buffer.length > 22) {
      // First sound out fast: cut at a comma or dash if the reply has one.
      // Every character held here is silence before Jarvis's first word.
      const clause = /[,;:—-]\s/.exec(this.buffer);
      if (clause) {
        const end = clause.index + clause[0].length;
        const piece = this.buffer.slice(0, end).trim();
        if (piece.length >= 14) {
          ready.push(piece);
          this.buffer = this.buffer.slice(end);
          this.started = true;
        }
      } else if (this.buffer.length > 42) {
        const cut = this.buffer.lastIndexOf(' ', 36);
        const take = cut >= 18 ? cut : 36;
        ready.push(this.buffer.slice(0, take).trim());
        this.buffer = this.buffer.slice(take);
        this.started = true;
      }
    }

    // A long stretch without sentence punctuation still needs to be spoken.
    if (this.buffer.length > 160) {
      const cut = this.buffer.lastIndexOf(' ', 140);
      const take = cut > 40 ? cut : 140;
      ready.push(this.buffer.slice(0, take).trim());
      this.buffer = this.buffer.slice(take);
    }
    return ready.filter(Boolean);
  }

  flush(): string {
    const rest = this.buffer.trim();
    this.buffer = '';
    return rest;
  }
}

/**
 * Plays reply audio in order. Falls back to the browser's own speech synthesis
 * when the server has no voice configured, so Jarvis is never silent.
 *
 * One audio element is reused for every sentence and unlocked inside the tap
 * that starts the conversation: iOS only allows playback after a user gesture,
 * and the first reply can arrive a minute later.
 */
export class SpeechQueue {
  private sentences: string[] = [];
  private playing = false;
  private pumping = false;
  private audio: HTMLAudioElement | null = null;
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private buffer: Float32Array<ArrayBuffer> | null = null;
  private fallback: SpeechSynthesisUtterance | null = null;
  private cache = new Map<string, Promise<Blob>>();
  // Tickets already minted for the next sentence, so the follow-up does not
  // pay another round trip when its turn comes.
  private urls = new Map<string, Promise<string>>();
  private pendingResolve: (() => void) | null = null;
  private primed = false;
  speed = SPEECH_SPEED;

  constructor(
    private readonly onBusyChange: (busy: boolean) => void,
    private readonly onServerVoice: (used: boolean) => void,
  ) {}

  get busy(): boolean {
    return this.playing;
  }

  /** Call from a tap handler so later playback is allowed. */
  prime(): void {
    if (this.primed) return;
    this.primed = true;
    try {
      const audio = this.element();
      audio.muted = true;
      audio.src = SILENT_WAV;
      void audio
        .play()
        .then(() => {
          audio.pause();
          audio.muted = false;
        })
        .catch(() => {
          audio.muted = false;
        });
    } catch {
      // A blocked prime is not fatal: the first play attempt retries.
    }
    if (this.context?.state === 'suspended') void this.context.resume();
  }

  enqueue(text: string): void {
    const trimmed = text.trim();
    if (!trimmed) return;
    this.sentences.push(trimmed);
    if (!this.pumping) void this.pump();
  }

  /** Silence immediately — used for barge-in and for ending a conversation. */
  stop(): void {
    this.sentences = [];
    if (this.audio) {
      this.audio.onended = null;
      this.audio.onerror = null;
      this.audio.pause();
    }
    if (this.fallback) {
      window.speechSynthesis?.cancel();
      this.fallback = null;
    }
    const resolve = this.pendingResolve;
    this.pendingResolve = null;
    resolve?.();
    this.cache.clear();
    this.urls.clear();
    if (this.playing) {
      this.playing = false;
      this.onBusyChange(false);
    }
  }

  /** Output level of the reply audio, 0–1, for the speaking animation. */
  readLevel(): number {
    if (!this.analyser || !this.buffer) return 0;
    this.analyser.getFloatTimeDomainData(this.buffer);
    let sum = 0;
    for (let i = 0; i < this.buffer.length; i += 1) {
      sum += this.buffer[i] * this.buffer[i];
    }
    return Math.min(1, Math.sqrt(sum / this.buffer.length) * 4.5);
  }

  dispose(): void {
    this.stop();
    this.analyser?.disconnect();
    void this.context?.close();
    this.context = null;
    this.analyser = null;
    this.buffer = null;
  }

  private element(): HTMLAudioElement {
    if (!this.audio) {
      const audio = new Audio();
      audio.preload = 'auto';
      this.audio = audio;
      this.attachAnalyser(audio);
    }
    return this.audio;
  }

  /** Ticket URL for one sentence, minted once and reused. */
  private urlFor(text: string): Promise<string> {
    const cached = this.urls.get(text);
    if (cached) return cached;
    const request = speechUrl(text, this.speed).finally(() => {
      this.urls.delete(text);
    });
    this.urls.set(text, request);
    return request;
  }

  private fetchAudio(text: string): Promise<Blob> {
    const cached = this.cache.get(text);
    if (cached) return cached;
    const request = synthesize(text, this.speed).finally(() => {
      this.cache.delete(text);
    });
    this.cache.set(text, request);
    return request;
  }

  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    this.playing = true;
    this.onBusyChange(true);
    while (this.sentences.length) {
      const sentence = this.sentences[0];
      const next = this.sentences[1];
      // Request the following sentence's link while this one plays: paying the
      // round trip later is the gap that makes speech feel late.
      if (next) void this.urlFor(next).catch(() => {});
      let serverVoice = true;
      try {
        serverVoice = await this.playSentence(sentence);
      } catch {
        serverVoice = false;
      }
      if (!this.playing) break;
      this.sentences.shift();
      this.onServerVoice(serverVoice);
    }
    this.pumping = false;
    if (this.playing) {
      this.playing = false;
      this.onBusyChange(false);
    }
  }

  /**
   * Speak one sentence. The server URL is tried first because the browser can
   * start playing while the voice engine is still producing audio; the whole
   * clip is the fallback.
   */
  private async playSentence(sentence: string): Promise<boolean> {
    try {
      const url = await this.urlFor(sentence);
      if (await this.playUrl(url)) return true;
    } catch {
      // fall through to the buffered path
    }
    try {
      const blob = await this.fetchAudio(sentence);
      if (!this.playing) return true;
      await this.playBlob(blob);
      return true;
    } catch {
      await this.playBrowserVoice(sentence);
      return false;
    }
  }

  private async playUrl(url: string): Promise<boolean> {
    const audio = this.element();
    audio.src = url;
    return await new Promise<boolean>((resolve) => {
      let settled = false;
      let started = false;
      const finish = (ok: boolean) => {
        if (settled) return;
        settled = true;
        audio.onended = null;
        audio.onerror = null;
        audio.onplaying = null;
        this.pendingResolve = null;
        resolve(ok);
      };
      const done = () => finish(started);
      this.pendingResolve = done;
      audio.onplaying = () => {
        started = true;
      };
      audio.onended = done;
      audio.onerror = done;
      void audio.play().catch(done);
    });
  }

  private async playBlob(blob: Blob): Promise<void> {
    const audio = this.element();
    if (!audio.src.startsWith('blob:')) {
      // First real clip: release the silent prime recording.
      audio.src = '';
    }
    const url = URL.createObjectURL(blob);
    audio.src = url;
    await new Promise<void>((resolve) => {
      const done = () => {
        audio.onended = null;
        audio.onerror = null;
        this.pendingResolve = null;
        URL.revokeObjectURL(url);
        resolve();
      };
      this.pendingResolve = done;
      audio.onended = done;
      audio.onerror = done;
      void audio.play().catch(done);
    });
  }

  private playBrowserVoice(sentence: string): Promise<void> {
    if (!window.speechSynthesis) return Promise.resolve();
    return new Promise<void>((resolve) => {
      const utterance = new SpeechSynthesisUtterance(sentence);
      utterance.lang = 'en-GB';
      utterance.rate = 1.18;
      utterance.pitch = 0.85;
      this.fallback = utterance;
      const done = () => {
        utterance.onend = null;
        utterance.onerror = null;
        this.fallback = null;
        resolve();
      };
      utterance.onend = done;
      utterance.onerror = done;
      window.speechSynthesis.speak(utterance);
    });
  }

  private attachAnalyser(audio: HTMLAudioElement): void {
    if (this.context) return;
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) return;
    try {
      this.context = new Ctor();
      const source = this.context.createMediaElementSource(audio);
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 512;
      this.analyser.smoothingTimeConstant = 0.7;
      source.connect(this.analyser);
      this.analyser.connect(this.context.destination);
      this.buffer = new Float32Array(new ArrayBuffer(512 * 4));
    } catch {
      this.context = null;
      this.analyser = null;
    }
  }
}
