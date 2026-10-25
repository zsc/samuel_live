interface AudioWorkletProcessorBase {
  readonly port: MessagePort;
}

declare const AudioWorkletProcessor: {
  new (options?: AudioWorkletNodeOptions): AudioWorkletProcessorBase;
  prototype: AudioWorkletProcessorBase;
};
declare function registerProcessor(name: string, processor: unknown): void;
declare const sampleRate: number;
declare const currentTime: number;
