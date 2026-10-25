import type {
  PinkTromboneConstriction,
  PinkTromboneElement,
} from "./pink-trombone-types";
import type { SamuelRecording } from "./types";

export class PinkTromboneSession {
  readonly context: AudioContext;
  readonly element: PinkTromboneElement;
  private readonly constriction: PinkTromboneConstriction;
  private readonly lipConstriction: PinkTromboneConstriction;
  private readonly masterGain: GainNode;
  private muted = false;

  private constructor(context: AudioContext, element: PinkTromboneElement) {
    this.context = context;
    this.element = element;
    this.constriction = element.newConstriction(33, 3);
    this.lipConstriction = element.newConstriction(43, 3);
    if (!this.constriction || !this.lipConstriction || this.constriction.diameter === this.lipConstriction.diameter) {
      throw new Error("Pink Trombone 未能分配独立舌尖和嘴唇收缩槽");
    }
    this.masterGain = context.createGain();
    this.masterGain.gain.value = 0;
    element.connect(this.masterGain);
    this.masterGain.connect(context.destination);
    element.vibrato.gain.value = 0;
    element.vibrato.wobble.value = 0;
    element.vibrato.frequency.value = 6;
    element.tractLength.value = 44;
    element.intensity.value = 0;
    element.start();
    element.enableUI();

  }

  static async create(element: PinkTromboneElement): Promise<PinkTromboneSession> {
    if (!document.querySelector("script[data-pink-trombone]")) {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement("script");
        script.type = "module";
        script.src = new URL("pink-trombone/pink-trombone.min.js", document.baseURI).href;
        script.dataset.pinkTrombone = "true";
        script.onload = () => resolve();
        script.onerror = () => reject(new Error("Pink Trombone 脚本加载失败"));
        document.head.appendChild(script);
      });
    }
    await customElements.whenDefined("pink-trombone");
    const q = new URLSearchParams(location.search);
    const context = new AudioContext({ latencyHint: "interactive", ...(q.has("validate") && q.get("rate") === "44100" ? {sampleRate:44100} : {}) });
    (window as unknown as { __PINK_TROMBONE_WORKLET_URL__: string }).__PINK_TROMBONE_WORKLET_URL__ =
      new URL("pink-trombone/pink-trombone-worklet-processor.min.js", document.baseURI).href;
    await element.setAudioContext(context);
    element.removeAttribute("inactive");
    element.setAttribute("interactive", "false");
    return new PinkTromboneSession(context, element);
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    const now = this.context.currentTime;
    cancelAndHold(this.masterGain.gain, now);
    this.masterGain.gain.setTargetAtTime(muted ? 0 : 1, now, 0.01);
  }

  async resume(): Promise<void> {
    await this.context.resume();
  }

  async schedule(recording: SamuelRecording, sourceOffset: number, valid = () => true): Promise<number | null> {
    await this.resume();
    this.stop();
    const node = this.element.pinkTrombone._pinkTromboneNode;
    // Reset waveguide/glottis state with an acknowledged audio-thread message.
    await node._postMessage({name: "reset"});
    if (!valid()) return null;
    const now = this.context.currentTime;
    const t0 = now + 0.08;
    const dt = recording.controlHop / recording.modelSampleRate;
    const frame = Math.min(recording.frameCount - 1, Math.floor(sourceOffset / dt));
    const fraction = sourceOffset / dt - frame;
    const values = curveValues(recording);
    this.cancelAutomation(now);
    this.automatedParams().forEach((param, i) => {
      const v = values[i];
      const atOffset = v[frame] + (v[Math.min(frame + 1, v.length - 1)] - v[frame]) * fraction;
      param.setValueAtTime(atOffset, now);
      param.setValueAtTime(atOffset, t0);
      // Exact input-sample timestamps, including a fractional seek and short tail.
      for (let f = frame + 1; f < v.length; f++) {
        const when = t0 + f * dt - sourceOffset;
        if (f * dt < recording.durationSeconds) param.linearRampToValueAtTime(v[f], when);
      }
    });
    const gate = this.masterGain.gain;
    gate.cancelScheduledValues(now); gate.setValueAtTime(0, now);
    gate.setValueAtTime(0, t0);
    const end = t0 + recording.durationSeconds - sourceOffset;
    const fade = Math.min(.01, (end - t0) / 2);
    gate.linearRampToValueAtTime(this.muted ? 0 : 1, t0 + fade);
    gate.setValueAtTime(this.muted ? 0 : 1, Math.max(t0 + fade, end - fade));
    gate.linearRampToValueAtTime(0, end);
    return t0;
  }

  hold(recording: SamuelRecording, frameIndex: number): void {
    const frame = Math.max(0, Math.min(recording.frameCount - 1, frameIndex));
    const values = curveValues(recording);
    const now = this.context.currentTime;
    this.cancelAutomation(now);
    this.automatedParams().forEach((param, index) => {
      param.setValueAtTime(values[index][frame], now);
    });
    this.masterGain.gain.cancelScheduledValues(now);
    this.masterGain.gain.setTargetAtTime(0, now, .005);
  }

  showFrames(params: Record<string, Float32Array>, frameIndex = -1): void {
    const now = this.context.currentTime;
    const values = curveValuesFromParams(params);
    const index = frameIndex < 0 ? Math.max(0, params.voiceness.length + frameIndex) : frameIndex;
    this.automatedParams().forEach((param, i) => {
      param.cancelScheduledValues(now);
      param.setValueAtTime(values[i][index] ?? 0, now);
    });
    this.masterGain.gain.cancelScheduledValues(now);
    this.masterGain.gain.setTargetAtTime(0, now, .005);
  }

  stop(): void {
    const now = this.context.currentTime;
    this.cancelAutomation(now);
    cancelAndHold(this.masterGain.gain, now);
    this.masterGain.gain.setTargetAtTime(0, now, 0.015);
  }

  private automatedParams(): AudioParam[] {
    return [
      this.element.frequency,
      this.element.tenseness,
      this.element.loudness,
      this.element.tongue.index,
      this.element.tongue.diameter,
      this.constriction.index,
      this.constriction.diameter,
      this.lipConstriction.diameter,
      this.element.intensity,
    ];
  }

  private cancelAutomation(now: number): void {
    for (const param of this.automatedParams()) cancelAndHold(param, now);
  }
}

function curveValues(recording: SamuelRecording): Float32Array[] {
  return curveValuesFromParams(recording.params);
}

function curveValuesFromParams(params: Record<string, Float32Array>): Float32Array[] {
  const voiceness = params.voiceness;
  const loudness = Float32Array.from(voiceness, (value) => Math.pow(Math.max(0, value), 0.25));
  return [
    params.frequency,
    voiceness,
    loudness,
    params.tongueIndex,
    params.tongueDiameter,
    params.constrictionIndex,
    params.constrictionDiameter,
    params.lipDiameter,
    params.intensity,
  ];
}

function cancelAndHold(param: AudioParam, when: number): void {
  const withHold = param as AudioParam & { cancelAndHoldAtTime?: (time: number) => AudioParam };
  if (withHold.cancelAndHoldAtTime) withHold.cancelAndHoldAtTime(when);
  else param.cancelScheduledValues(when);
}
