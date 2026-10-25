/** Causal online RMS gain matching the checkpoint's target RMS without ever
 * rescaling already-emitted history. Gain moves sample-by-sample, and silence
 * is not amplified into noise. */
export class CausalRmsNormalizer {
  private gain = 1;
  private energy = 0;
  private readonly targetRms: number;
  private readonly gate: number;
  private readonly minGain: number;
  private readonly maxGain: number;
  private readonly smoothing: number;
  lastRms = 0;
  lastGain = 1;

  constructor(targetRms = 0.05, gate = 0.0035, minGain = 0.25, maxGain = 20, smoothing = 0.0015) {
    this.targetRms = targetRms;
    this.gate = gate;
    this.minGain = minGain;
    this.maxGain = maxGain;
    this.smoothing = smoothing;
  }

  process(input: Float32Array): Float32Array {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      // 20 ms exponential RMS; fully causal and independent of input chunks.
      this.energy += (input[i] * input[i] - this.energy) * (1 - Math.exp(-1 / 882));
      this.lastRms = Math.sqrt(Math.max(0, this.energy));
      const target = this.lastRms >= this.gate ? clamp(this.targetRms / this.lastRms, this.minGain, this.maxGain) : 1;
      this.gain += (target - this.gain) * this.smoothing;
      out[i] = clamp(input[i] * this.gain, -1, 1);
    }
    this.lastGain = this.gain;
    return out;
  }
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

