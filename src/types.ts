export interface ParamSpecEntry {
  min: number;
  max: number;
  bucketCenters: number[];
}

export interface ModelManifest {
  schemaVersion: 1;
  modelRevision: string;
  upstreamCommit: string;
  checkpointSha256: string;
  configSha256: string;
  onnxSha256: string;
  onnxFile: string;
  sampleRate: number;
  controlHop: number;
  controlFrameRate: number;
  lookaheadFrames: number;
  streamingChunkFrames: number;
  historySamples: number;
  receptiveFieldSamples: number;
  nBuckets: number;
  trainableNames: string[];
  paramSpec: Record<string, ParamSpecEntry>;
  fixedValues: Record<string, number>;
  frequency: { external: true; min: number; max: number };
  training: { targetRms: number; source: string };
}

export interface SamuelRecording {
  schemaVersion: 1;
  modelRevision: string;
  modelSha256: string;
  sourceSampleRate: number;
  originalPcm: Float32Array;
  modelSampleRate: number;
  controlHop: number;
  validModelSamples: number;
  frameCount: number;
  durationSeconds: number;
  params: Record<string, Float32Array>;
  voiced: Uint8Array;
  processingConfig: Record<string, unknown>;
  sourceKind: "microphone" | "fixture";
}

export interface InferenceStats {
  p50: number;
  p95: number;
  rtf: number;
  backlogMs: number;
  displayLatencyP95: number;
  sampleCount: number;
  preprocessingMs: number;
  processingMs: number;
  maxBacklogMs: number;
}

export interface BackendInfo {
  requested: "webgpu";
  actual: "webgpu" | "wasm";
  adapter?: string;
  gpuKernels?: Record<string, number>;
  fallbackReason?: string;
}

export type WorkerToMain =
  | { type: "loadProgress"; stage: string; progress: number }
  | { type: "ready"; manifest: ModelManifest; backend: BackendInfo; loadMs: number }
  | { type: "loadError"; message: string }
  | {
      type: "liveFrames";
      sessionId: number;
      frameStart: number;
      params: Record<string, Float32Array>;
      voiced: Uint8Array;
      inferenceMs: number;
      backlogMs: number;
      displayLatencyMs: number;
      capturedAt: number;
      totalFrames: number;
    }
  | { type: "stats"; sessionId: number; stats: InferenceStats; queuedBlocks: number }
  | { type: "limitReached"; sessionId: number; reason: string }
  | { type: "finalizing"; sessionId: number; queuedBlocks: number }
  | { type: "recording"; sessionId: number; recording: SamuelRecording }
  | { type: "error"; sessionId?: number; message: string }
  | {
      type: "selfTest";
      result: {
        backend: BackendInfo;
        maxAbsLogitError: number;
        meanAbsLogitError: number;
        bucketMismatches: number;
        bucketCount: number;
        minMargin: number;
        streamingMaxAbsLogitError: number;
        streamingBucketMismatches: number;
        fixtureFrames: number;
        durationMs: number;
        toleranceFailures: number;
        acceptedToleranceFailures: number;
        tolerance: {atol:number;rtol:number};
        passed: boolean;
        cases: Record<string, unknown>;
      };
    };

export type MainToWorker =
  | { type: "init"; assetBase: string; forceWasm?: boolean; delayMs?: number; diagnostics?: boolean }
  | {
      type: "start";
      sessionId: number;
      sourceSampleRate: number;
      sourceKind: "microphone" | "fixture";
    }
  | {
      type: "audioChunk";
      sessionId: number;
      sequence: number;
      sampleStart: number;
      pcm: Float32Array;
      validLength: number;
      capturedAt: number;
      contextTime?: number;
    }
  | { type: "finish"; sessionId: number }
  | { type: "cancel"; sessionId: number }
  | { type: "selfTest" }
  | { type: "diagnosticLoss" };

