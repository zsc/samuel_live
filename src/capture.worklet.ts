interface CaptureOptions {
  sessionId: number;
  chunkFrames?: number;
}

class MicCaptureProcessor extends AudioWorkletProcessor {
  private readonly sessionId: number;
  private readonly chunkFrames: number;
  private buffer: Float32Array;
  private fill = 0;
  private sampleStart = 0;
  private stopping = false;
  private sequence = 0;
  private captured = 0;

  constructor(options: AudioWorkletNodeOptions) {
    super();
    const data = options.processorOptions as CaptureOptions;
    this.sessionId = data.sessionId;
    this.chunkFrames = data.chunkFrames ?? 2048;
    this.buffer = new Float32Array(this.chunkFrames);
    this.port.onmessage = (event: MessageEvent<{ type: string }>) => {
      if (event.data.type === "stop") this.stopping = true;
    };
  }

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0]?.[0];
    if (input && !this.stopping) {
      let offset = 0;
      while (offset < input.length && this.captured < sampleRate * 60) {
        const take = Math.min(input.length - offset, this.chunkFrames - this.fill, sampleRate * 60 - this.captured);
        this.buffer.set(input.subarray(offset, offset + take), this.fill);
        this.fill += take;
        this.captured += take;
        offset += take;
        if (this.fill === this.chunkFrames) this.emit(this.chunkFrames);
      }
    }
    if (this.captured >= sampleRate * 60) this.stopping = true;
    if (this.stopping) {
      if (this.fill > 0) this.emit(this.fill);
      this.port.postMessage({
        type: "eos",
        sessionId: this.sessionId,
        totalSamples: this.sampleStart,
      });
      return false;
    }
    return true;
  }

  private emit(length: number): void {
    const pcm = this.buffer.slice(0, length);
    const sampleStart = this.sampleStart;
    this.sampleStart += length;
    this.fill = 0;
    this.port.postMessage(
      {
        type: "chunk",
        sessionId: this.sessionId,
        sampleStart,
        sequence: this.sequence++,
        validLength: length,
        contextTime: currentTime + 128 / sampleRate,
        pcm: pcm.buffer,
      },
      [pcm.buffer],
    );
  }
}

registerProcessor("samuel-mic-capture", MicCaptureProcessor);

