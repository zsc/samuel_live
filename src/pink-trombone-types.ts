export interface PinkTromboneParamMap {
  frequency: AudioParam;
  tenseness: AudioParam;
  intensity: AudioParam;
  loudness: AudioParam;
  tongue: { index: AudioParam; diameter: AudioParam };
  vibrato: { frequency: AudioParam; gain: AudioParam; wobble: AudioParam };
  tractLength: AudioParam;
  bypassTract: AudioParam;
  noise: AudioParam;
}

export interface PinkTromboneConstriction {
  index: AudioParam;
  diameter: AudioParam;
}

export interface PinkTromboneElement extends HTMLElement, PinkTromboneParamMap {
  pinkTrombone: { _pinkTromboneNode: AudioWorkletNode & { _postMessage(data: {name: string}): Promise<unknown>; processor?: unknown } };
  setAudioContext(context: AudioContext): Promise<unknown>;
  audioContext: AudioContext;
  connect(node: AudioNode): void;
  start(): void;
  stop(): void;
  enableUI(): void;
  startUI(): void;
  stopUI(): void;
  newConstriction(index: number, diameter: number): PinkTromboneConstriction;
  removeConstriction(constriction: PinkTromboneConstriction): void;
}
