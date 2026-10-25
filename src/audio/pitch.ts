/** Causal YIN / pYIN-style tracker ported from
 * vendor/samuel/scripts/precompute_pitch_yin.py. Each returned control frame
 * depends only on samples up to that frame's deadline. */
const FRAME_LENGTH = 2048;
const YIN_WINDOW = 1024;
const SHIFT_FRAMES = 3;
const CANDIDATE_THRESHOLD = 0.5;
const BIN_CENTS = 20;
const OBS_UNIFORM_MIX = 0.02;
const JUMP_PROB = 0.01;
const DECODE_WINDOW_BINS = 5;

export interface PitchFrame {
  f0: number;
  voiced: boolean;
  depth: number;
}

export class CausalPitchTracker {
  private readonly sampleRate: number;
  private readonly hop: number;
  private readonly fmin: number;
  private readonly fmax: number;
  private readonly voicedThreshold: number;
  private readonly sigmaCents: number;
  private readonly tauMin: number;
  private readonly tauMax: number;
  private readonly nBins: number;
  private readonly uniform: number;
  private readonly trans: Float32Array;
  private alpha: Float32Array;
  private lastVoicedF0: number | null = null;

  constructor(sampleRate = 44100, hop = 512, fmin = 70, fmax = 500, voicedThreshold = 0.2, sigmaCents = 25) {
    this.sampleRate = sampleRate;
    this.hop = hop;
    this.fmin = fmin;
    this.fmax = fmax;
    this.voicedThreshold = voicedThreshold;
    this.sigmaCents = sigmaCents;
    this.tauMin = Math.ceil(sampleRate / fmax);
    this.tauMax = Math.floor(sampleRate / fmin);
    this.nBins = Math.floor((1200 * Math.log2(fmax / fmin)) / BIN_CENTS) + 1;
    this.uniform = 1 / this.nBins;
    this.alpha = new Float32Array(this.nBins).fill(this.uniform);
    const grid = new Float32Array(this.nBins);
    for (let i = 0; i < this.nBins; i++) grid[i] = i * BIN_CENTS;
    this.trans = new Float32Array(this.nBins * this.nBins);
    for (let col = 0; col < this.nBins; col++) {
      let sum = 0;
      for (let row = 0; row < this.nBins; row++) {
        const d = (grid[row] - grid[col]) / sigmaCents;
        const value = Math.exp(-0.5 * d * d);
        this.trans[row * this.nBins + col] = value;
        sum += value;
      }
      for (let row = 0; row < this.nBins; row++) {
        const idx = row * this.nBins + col;
        this.trans[idx] = (1 - JUMP_PROB) * (this.trans[idx] / sum) + JUMP_PROB * this.uniform;
      }
    }
  }

  reset(): void {
    this.alpha.fill(this.uniform);
    this.lastVoicedF0 = null;
  }

  /** Returns f0 at control frame `frame`, or null until its 2048-sample
   * trailing analysis window has arrived. */
  frame(pcm: Float32Array, frame: number): PitchFrame | null {
    if (frame < SHIFT_FRAMES) return this.finish(0, false, 1);
    const analysisFrame = frame - SHIFT_FRAMES;
    const start = analysisFrame * this.hop;
    if (start + FRAME_LENGTH > pcm.length) return null;
    const cmndf = this.cmndf(pcm, start);
    return this.decodeFrame(cmndf);
  }

  private finish(f0: number, voiced: boolean, depth: number): PitchFrame {
    if (voiced && f0 >= this.fmin && f0 <= this.fmax) this.lastVoicedF0 = f0;
    const held = this.lastVoicedF0 ?? 0.5 * (this.fmin + this.fmax);
    return { f0: clamp(held, this.fmin, this.fmax), voiced, depth };
  }

  private cmndf(pcm: Float32Array, start: number): Float32Array {
    const nTau = this.tauMax + 2;
    const d = new Float32Array(nTau);
    let e0 = 0;
    for (let j = 0; j < YIN_WINDOW; j++) {
      const x = pcm[start + j];
      e0 += x * x;
    }
    const prefix = new Float64Array(FRAME_LENGTH + 1);
    for (let j = 0; j < FRAME_LENGTH; j++) {
      const x = pcm[start + j];
      prefix[j + 1] = prefix[j] + x * x;
    }
    for (let tau = 0; tau < nTau; tau++) {
      let corr = 0;
      for (let j = 0; j < YIN_WINDOW; j++) corr += pcm[start + j] * pcm[start + j + tau];
      const eTau = prefix[tau + YIN_WINDOW] - prefix[tau];
      d[tau] = Math.max(0, e0 + eTau - 2 * corr);
    }
    const cmndf = new Float32Array(nTau).fill(1);
    let running = 0;
    for (let tau = 1; tau < nTau; tau++) {
      running += d[tau];
      cmndf[tau] = running > 1e-12 ? (d[tau] * tau) / running : 1;
    }
    return cmndf;
  }

  private decodeFrame(cmndf: Float32Array): PitchFrame {
    const candidates: { tau: number; depth: number; f0: number; bin: number; weight: number }[] = [];
    for (let tau = this.tauMin; tau <= this.tauMax; tau++) {
      const depth = cmndf[tau];
      if (depth < cmndf[tau - 1] && depth <= cmndf[tau + 1] && depth < CANDIDATE_THRESHOLD) {
        const y0 = cmndf[tau - 1];
        const y2 = cmndf[tau + 1];
        const denom = y0 - 2 * depth + y2;
        let offset = Math.abs(denom) > 1e-12 ? (0.5 * (y0 - y2)) / denom : 0;
        offset = clamp(offset, -0.5, 0.5);
        const f0 = this.sampleRate / (tau + offset);
        if (f0 >= this.fmin && f0 <= this.fmax) {
          const bin = clamp(Math.round((1200 * Math.log2(f0 / this.fmin)) / BIN_CENTS), 0, this.nBins - 1);
          candidates.push({ tau, depth, f0, bin, weight: 0 });
        }
      }
    }

    // Beta(2, 18) CDF gives pYIN's soft first-trough threshold weights.
    const cdf = (x: number) => beta2_18Cdf(clamp(x, 0, 1));
    let runningMinimum = Infinity;
    for (const candidate of candidates) {
      const ceiling = runningMinimum;
      candidate.weight = Math.max(cdf(ceiling) - cdf(candidate.depth), 0);
      runningMinimum = Math.min(runningMinimum, candidate.depth);
    }

    const obs = new Float32Array(this.nBins);
    const depthAt = new Float32Array(this.nBins).fill(Infinity);
    const f0At = new Float32Array(this.nBins).fill(NaN);
    const bestWeight = new Float32Array(this.nBins).fill(-1);
    for (const candidate of candidates) {
      obs[candidate.bin] += candidate.weight;
      if (candidate.depth < depthAt[candidate.bin]) depthAt[candidate.bin] = candidate.depth;
      if (candidate.weight > bestWeight[candidate.bin]) {
        bestWeight[candidate.bin] = candidate.weight;
        f0At[candidate.bin] = candidate.f0;
      }
    }
    let obsSum = 0;
    for (const value of obs) obsSum += value;
    for (let i = 0; i < this.nBins; i++) {
      obs[i] =
        obsSum > 0
          ? (1 - OBS_UNIFORM_MIX) * (obs[i] / obsSum) + OBS_UNIFORM_MIX * this.uniform
          : this.uniform;
    }

    const next = new Float32Array(this.nBins);
    let nextSum = 0;
    for (let row = 0; row < this.nBins; row++) {
      let value = 0;
      for (let col = 0; col < this.nBins; col++) {
        value += this.trans[row * this.nBins + col] * this.alpha[col];
      }
      value *= obs[row];
      next[row] = value;
      nextSum += value;
    }
    if (nextSum > 0) for (let i = 0; i < this.nBins; i++) next[i] /= nextSum;
    this.alpha = nextSum > 0 ? next : new Float32Array(this.nBins).fill(this.uniform);
    let choice = 0;
    for (let i = 1; i < this.nBins; i++) if (this.alpha[i] > this.alpha[choice]) choice = i;

    let picked = NaN;
    let bestDistance = Infinity;
    for (let offset = -DECODE_WINDOW_BINS; offset <= DECODE_WINDOW_BINS; offset++) {
      const bin = choice + offset;
      if (bin < 0 || bin >= this.nBins) continue;
      const distance = Math.abs(offset);
      if (distance < bestDistance && Number.isFinite(f0At[bin])) {
        bestDistance = distance;
        picked = f0At[bin];
      }
    }
    let deepestBin = 0;
    for (let i = 1; i < this.nBins; i++) if (depthAt[i] < depthAt[deepestBin]) deepestBin = i;
    const deepestDepth = depthAt[deepestBin];
    const f0 = Number.isFinite(picked) ? picked : f0At[deepestBin];
    const voiced = deepestDepth < this.voicedThreshold && Number.isFinite(f0) && f0 >= this.fmin && f0 <= this.fmax;
    return this.finish(voiced ? f0 : 0, voiced, Number.isFinite(deepestDepth) ? deepestDepth : 1);
  }
}

function beta2_18Cdf(x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  // I_x(2,18) = x^2 * sum_{j=0}^{17} (j+1) * (1-x)^j
  let q = 1 - x;
  let power = 1;
  let sum = 0;
  for (let j = 0; j <= 17; j++) {
    sum += (j + 1) * power;
    power *= q;
  }
  return x * x * sum;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}
