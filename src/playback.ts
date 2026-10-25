import { PinkTromboneSession } from "./pink-trombone";
import type { SamuelRecording } from "./types";

export type PlaybackMode = "samuel" | "original";

export interface PlaybackSnapshot {
  playing: boolean;
  mode: PlaybackMode;
  position: number;
  duration: number;
}

export class RecordingPlayer {
  private readonly synth: PinkTromboneSession;
  private recording: SamuelRecording | null = null;
  private source: AudioBufferSourceNode | null = null;
  private originalBuffer: AudioBuffer | null = null;
  private playing = false;
  private mode: PlaybackMode = "samuel";
  private offset = 0;
  private startAt = 0;
  private startOffset = 0;
  private raf = 0;
  private generation = 0;
  private starting = false;
  private originalGain: GainNode | null = null;
  private snapshotListener: ((snapshot: PlaybackSnapshot) => void) | null = null;

  constructor(synth: PinkTromboneSession) {
    this.synth = synth;
  }

  get context(): AudioContext {
    return this.synth.context;
  }

  async setRecording(recording: SamuelRecording): Promise<void> {
    this.stop();
    this.recording = recording;
    this.offset = 0;
    this.originalBuffer = this.context.createBuffer(
      1,
      Math.max(1, recording.originalPcm.length),
      recording.sourceSampleRate,
    );
    this.originalBuffer.copyToChannel(new Float32Array(recording.originalPcm), 0);
    this.synth.hold(recording, 0);
    this.emit();
  }

  onSnapshot(listener: (snapshot: PlaybackSnapshot) => void): void {
    this.snapshotListener = listener;
  }

  get duration(): number {
    return this.recording?.durationSeconds ?? 0;
  }

  get currentMode(): PlaybackMode {
    return this.mode;
  }

  async play(): Promise<void> {
    if (!this.recording || this.playing || this.starting) return;
    this.starting = true;
    const generation = ++this.generation;
    await this.synth.resume();
    if (generation !== this.generation) return;
    if (this.offset >= this.duration - 0.002) this.offset = 0;
    this.startOffset = this.offset;
    this.startAt = this.context.currentTime + .08;
    if (this.mode === "original") this.startOriginal(this.startAt, this.startOffset);
    else {
      const at = await this.synth.schedule(this.recording, this.startOffset, () => generation === this.generation);
      if (at === null) return;
      this.startAt = at;
    }
    if (generation !== this.generation) return;
    this.starting = false;
    this.playing = true;
    this.tick();
  }

  pause(): void {
    if (!this.playing) return;
    this.offset = this.position();
    this.playing = false;
    this.stopSources();
    if (this.recording) this.synth.hold(this.recording, this.frameAt(this.offset));
    this.emit();
  }

  stop(): void {
    this.playing = false;
    this.stopSources();
    this.synth.stop();
    this.emit();
  }

  async restart(): Promise<void> {
    this.stopSources();
    this.offset = 0;
    this.playing = false;
    if (this.recording) this.synth.hold(this.recording, 0);
    await this.play();
    this.emit();
  }

  async seek(seconds: number): Promise<void> {
    if (!this.recording) return;
    this.stopSources();
    this.playing = false;
    this.offset = Math.max(0, Math.min(this.duration, seconds));
    this.synth.hold(this.recording, this.frameAt(this.offset));
    this.emit();
    // Scrubbing is silent. Resume explicitly with the play button.
  }

  async setMode(mode: PlaybackMode): Promise<void> {
    if (mode === this.mode) return;
    const wasPlaying = this.playing;
    const position = this.position();
    this.stopSources();
    this.playing = false;
    this.mode = mode;
    this.offset = position;
    if (mode === "original") this.synth.setMuted(true);
    else this.synth.setMuted(false);
    if (this.recording) this.synth.hold(this.recording, this.frameAt(this.offset));
    if (wasPlaying) await this.play();
    this.emit();
  }

  position(): number {
    if (!this.recording) return 0;
    if (!this.playing) return Math.min(this.offset, this.duration);
    const elapsed = Math.max(0, this.context.currentTime - this.startAt);
    return Math.min(this.duration, this.startOffset + elapsed);
  }

  private frameAt(seconds: number): number {
    if (!this.recording) return 0;
    return Math.min(
      this.recording.frameCount - 1,
      Math.floor((seconds * this.recording.modelSampleRate) / this.recording.controlHop),
    );
  }

  private startOriginal(when: number, offset: number): void {
    if (!this.originalBuffer) return;
    const source = this.context.createBufferSource();
    source.buffer = this.originalBuffer;
    const gain = this.context.createGain();
    this.originalGain = gain;
    gain.gain.value = 1;
    source.connect(gain).connect(this.context.destination);
    source.start(when, Math.min(offset, this.originalBuffer.duration - 0.001));
    this.source = source;
    source.onended = () => {
      source.disconnect(); gain.disconnect();
      if (this.source === source && this.playing) {
        this.offset = this.duration;
        this.playing = false;
        this.emit();
      }
    };
  }

  private stopSources(): void {
    this.generation++;
    this.starting = false;
    this.synth.stop();
    if (this.source) {
      try {
        const now = this.context.currentTime;
        this.originalGain?.gain.setTargetAtTime(0, now, .003);
        this.source.stop(now + .02);
      } catch {
        // Already stopped.
      }

      this.source = null;
    }
    cancelAnimationFrame(this.raf);
  }

  private tick = (): void => {
    if (!this.playing) return;
    if (this.recording && this.mode === "samuel") {
      const frame = this.frameAt(this.position());
      if (this.position() >= this.duration - 0.001) {
        this.playing = false;
        this.offset = this.duration;
        this.synth.stop();
        this.emit();
        return;
      }
      void frame;
    } else if (this.recording) {
      this.synth.hold(this.recording, this.frameAt(this.position()));
      if (this.position() >= this.duration - 0.001) {
        this.playing = false;
        this.offset = this.duration;
        this.stopSources();
        this.emit();
        return;
      }
    }
    this.emit();
    this.raf = requestAnimationFrame(this.tick);
  };

  private emit(): void {
    this.snapshotListener?.({
      playing: this.playing,
      mode: this.mode,
      position: this.position(),
      duration: this.duration,
    });
  }
}
