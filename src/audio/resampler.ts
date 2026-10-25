/** Streaming band-limited resampler with persistent fractional phase.
 *
 * The 24-tap windowed-sinc kernel is evaluated from absolute source positions,
 * so successive chunks cannot reset the filter or introduce a phase step.
 */
export class StreamingSincResampler {
  readonly sourceRate: number;
  readonly targetRate: number;
  readonly ratio: number;
  private input = new Float32Array(0);
  private inputStart = 0;
  private nextOutput = 0;
  private readonly taps = 24;
  private readonly half = 12;
  private readonly cutoff: number;

  constructor(sourceRate: number, targetRate: number) {
    if (!(sourceRate > 0 && targetRate > 0)) throw new Error("invalid sample rate");
    this.sourceRate = sourceRate;
    this.targetRate = targetRate;
    this.ratio = sourceRate / targetRate;
    this.cutoff = Math.min(1, targetRate / sourceRate) * 0.94;
  }

  push(chunk: Float32Array): Float32Array {
    if (chunk.length === 0) return new Float32Array(0);
    const merged = new Float32Array(this.input.length + chunk.length);
    merged.set(this.input, 0);
    merged.set(chunk, this.input.length);
    this.input = merged;
    return this.drain(false);
  }

  flush(): Float32Array {
    return this.drain(true);
  }

  get sourceSamplesReceived(): number {
    return this.inputStart + this.input.length;
  }

  private drain(final: boolean): Float32Array {
    const sourceEnd = this.inputStart + this.input.length;
    const capacity = Math.max(0, Math.ceil((sourceEnd - this.nextOutput * this.ratio) / this.ratio) + 4);
    const out = new Float32Array(capacity);
    let count = 0;
    while (count < out.length) {
      const pos = this.nextOutput * this.ratio;
      if (pos >= sourceEnd || (!final && pos + this.half >= sourceEnd)) break;
      out[count++] = this.sampleAt(pos);
      this.nextOutput++;
    }
    const keepFrom = Math.max(this.inputStart, Math.floor(this.nextOutput * this.ratio) - this.half - 2);
    const trim = Math.min(this.input.length, Math.max(0, keepFrom - this.inputStart));
    if (trim > 0) {
      this.input = this.input.slice(trim);
      this.inputStart += trim;
    }
    return out.subarray(0, count);
  }

  private sampleAt(position: number): number {
    const center = Math.floor(position);
    let weighted = 0;
    let weightSum = 0;
    for (let index = center - this.half + 1; index <= center + this.half; index++) {
      const distance = position - index;
      const window = 0.5 + 0.5 * Math.cos((Math.PI * distance) / this.half);
      const sincArg = this.cutoff * distance;
      const sinc = Math.abs(sincArg) < 1e-8 ? 1 : Math.sin(Math.PI * sincArg) / (Math.PI * sincArg);
      const weight = this.cutoff * sinc * window;
      weighted += this.sample(index) * weight;
      weightSum += weight;
    }
    return weightSum === 0 ? 0 : weighted / weightSum;
  }

  private sample(index: number): number {
    const local = index - this.inputStart;
    return local >= 0 && local < this.input.length ? this.input[local] : 0;
  }
}


