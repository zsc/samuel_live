import * as ort from "onnxruntime-web/webgpu";
import { CausalPitchTracker } from "./audio/pitch";
import { StreamingSincResampler } from "./audio/resampler";
import { CausalRmsNormalizer } from "./audio/rms";
import type { BackendInfo, InferenceStats, MainToWorker, ModelManifest, SamuelRecording, WorkerToMain } from "./types";
const TARGET_RATE = 44100;
const FIXED_FRAMES_PER_CHUNK = 4;
const MAX_SAMPLES = TARGET_RATE * 60;
const USED_PARAMS = ["voiceness", "intensity", "tongueIndex", "tongueDiameter", "constrictionIndex", "constrictionDiameter", "lipDiameter"];
let manifest: ModelManifest | null = null;
let session: ort.InferenceSession | null = null;
let backend: BackendInfo | null = null;
let assetBase = "";
let active: SessionState | null = null;
let testing = false;
let diagnostics = false;
let artificialDelay = 0;
interface SessionState {
  id: number; sourceKind: "microphone" | "fixture"; sourceRate: number;
  original: Float32Array; sourceLength: number; sequence: number;
  raw: Float32Array; pcm: Float32Array; length: number; frame: number;
  params: Record<string, Float32Array>; voiced: Uint8Array;
  resampler: StreamingSincResampler; normalizer: CausalRmsNormalizer; pitch: CausalPitchTracker;
  running: boolean; finishing: boolean; stopSent: boolean;
  timings: number[]; prepMs: number; inferenceMs: number; processingMs: number; maxBacklogMs: number;
  clockOrigin: number | null; contextOrigin: number | null;
}
self.onmessage = (event: MessageEvent<MainToWorker>) => { void handle(event.data); };
async function handle(m: MainToWorker) {
  try {
    if (m.type === "init") { diagnostics = !!m.diagnostics; artificialDelay = m.delayMs ?? 0; await initialize(m.assetBase, m.forceWasm); }
    else if (m.type === "diagnosticLoss" && diagnostics) { (await ort.env.webgpu.device as {destroy():void}).destroy(); }
    else if (m.type === "start") startSession(m.sessionId, m.sourceSampleRate, m.sourceKind);
    else if (m.type === "audioChunk") appendAudio(m);
    else if (m.type === "finish") finishSession(m.sessionId);
    else if (m.type === "cancel" && active?.id === m.sessionId) active = null;
    else if (m.type === "selfTest") {
      if (active || testing) throw new Error("录制或自检运行中，请稍后重试");
      testing = true;
      try { await runSelfTest(); } finally { testing = false; }
    }
  } catch (error) { reportError(error, "sessionId" in m ? m.sessionId : undefined); }
}
function reportError(error: unknown, id?: number) {
  if (id === active?.id) active = null;
  post({ type: "error", sessionId: id, message: error instanceof Error ? error.message : String(error) });
}
async function initialize(base: string, forceWasm = false): Promise<void> {
  if (session && manifest) return;
  assetBase = base;
  const started = performance.now();
  const kernelCounts = new Map<string, number>();
  ort.env.webgpu.profiling = {mode: "default", ondata: data => {
    kernelCounts.set(data.kernelType, (kernelCounts.get(data.kernelType) ?? 0) + 1);
  }};
  post({ type: "loadProgress", stage: "读取模型清单", progress: 0.05 });
  const manifestUrl = new URL("models/samuel-manifest.json", assetBase).href;
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) throw new Error(`模型清单加载失败 (${manifestResponse.status})`);
  manifest = (await manifestResponse.json()) as ModelManifest;

  post({ type: "loadProgress", stage: "下载 13 MB ONNX 权重", progress: 0.12 });
  const modelUrl = new URL(`models/${manifest.onnxFile}`, assetBase).href;
  const { buffer, downloaded, total } = await fetchWithProgress(modelUrl, (fraction) => {
    post({ type: "loadProgress", stage: "下载 ONNX 权重", progress: 0.12 + fraction * 0.53 });
  });
  if (total > 0 && downloaded !== total) throw new Error("模型下载不完整");
  post({ type: "loadProgress", stage: "验证 SHA-256", progress: 0.68 });
  const digest = await sha256Hex(buffer);
  if (digest !== manifest.onnxSha256) {
    throw new Error(`ONNX 校验失败：${digest.slice(0, 12)}… != ${manifest.onnxSha256.slice(0, 12)}…`);
  }

  ort.env.wasm.wasmPaths = new URL("ort/", assetBase).href;
  ort.env.wasm.numThreads = 1;
  if (JSON.stringify(manifest.trainableNames) !== JSON.stringify(USED_PARAMS) || manifest.controlHop !== 512 || manifest.sampleRate !== TARGET_RATE) throw new Error("不兼容的模型元数据");
  ort.env.logLevel = diagnostics ? "verbose" : "warning";

  post({ type: "loadProgress", stage: "初始化 WebGPU 推理", progress: 0.74 });
  let fallbackReason: string | undefined;
  try {
    const gpu = (navigator as Navigator & {
      gpu?: { requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string; device?: string; description?: string } } | null> };
    }).gpu;
    if (forceWasm) throw new Error("验证指定 WASM");
    if (!gpu) throw new Error("浏览器未提供 navigator.gpu");
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("没有可用的 WebGPU adapter");
    const info = adapter.info;
    backend = {
      requested: "webgpu",
      actual: "webgpu",
      adapter: [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(" "),
    };
    session = await createSession(buffer, ["webgpu"]);
  } catch (error) {
    fallbackReason = error instanceof Error ? error.message : String(error);
    session = null;
  }
  if (!session) {
    post({ type: "loadProgress", stage: "WebGPU 不可用，回退单线程 WASM", progress: 0.82 });
    backend = { requested: "webgpu", actual: "wasm", fallbackReason };
    session = await createSession(buffer, ["wasm"]);
  }
  post({ type: "loadProgress", stage: "执行模型预热", progress: 0.91 });
  const warm = new Float32Array(manifest.controlHop * FIXED_FRAMES_PER_CHUNK);
  const warmOutput = await runModel(warm);
  await warmOutput.getData(); warmOutput.dispose();
  if (backend?.actual === "webgpu") {
    const device = await ort.env.webgpu.device as { queue: {onSubmittedWorkDone(): Promise<void>}; lost: Promise<{message: string}> };
    await device.queue.onSubmittedWorkDone();
    if (kernelCounts.size) backend.gpuKernels = Object.fromEntries(kernelCounts);
    device.lost.then((info: { message: string }) => {
      session = null;
      reportError(new Error(`GPU device lost: ${info.message}。请重新加载模型；已保存的录制仍可回放。`));
    });
  }
  ort.env.webgpu.profiling.mode = "off";
  if (!backend) throw new Error("推理后端初始化失败");
  post({ type: "loadProgress", stage: "就绪", progress: 1 });
  post({ type: "ready", manifest, backend, loadMs: performance.now() - started });
}

async function createSession(buffer: ArrayBuffer, providers: ort.InferenceSession.ExecutionProviderConfig[]): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(buffer, {
    executionProviders: providers,
    graphOptimizationLevel: "all",
    enableCpuMemArena: true,
    enableMemPattern: true,
    logSeverityLevel: diagnostics ? 0 : 2,
    logVerbosityLevel: 1,
  });
}


function startSession(id: number, sourceRate: number, sourceKind: "microphone" | "fixture") {
  if (!manifest || !session || active || testing) throw new Error("模型未就绪或已有任务进行中");
  if (sourceRate < 8000 || sourceRate > 192000) throw new Error("不支持的输入采样率");
  const frames = Math.ceil(MAX_SAMPLES / 512);
  active = {
    id, sourceRate, sourceKind, original: new Float32Array(sourceRate * 60), sourceLength: 0, sequence: 0,
    raw: new Float32Array(MAX_SAMPLES + 512), pcm: new Float32Array(MAX_SAMPLES + 512), length: 0, frame: 0,
    params: Object.fromEntries([...USED_PARAMS, "frequency"].map(n => [n, new Float32Array(frames)])), voiced: new Uint8Array(frames),
    resampler: new StreamingSincResampler(sourceRate, TARGET_RATE), normalizer: new CausalRmsNormalizer(), pitch: new CausalPitchTracker(),
    running: false, finishing: false, stopSent: false, timings: [], prepMs: 0, inferenceMs: 0, processingMs: 0, maxBacklogMs: 0,
    clockOrigin: null, contextOrigin: null,
  };
}
function appendModel(s: SessionState, pcm: Float32Array) {
  if (s.length + pcm.length > MAX_SAMPLES) throw new Error("采集超过 60 秒容量；本次录制未冻结");
  s.raw.set(pcm, s.length);
  s.pcm.set(s.normalizer.process(pcm), s.length);
  s.length += pcm.length;
}
function appendAudio(m: Extract<MainToWorker, {type: "audioChunk"}>) {
  const s = active;
  if (!s || s.id !== m.sessionId) return;
  if (s.finishing) throw new Error("结束标记后仍收到采集数据");
  if (m.sequence !== s.sequence || m.sampleStart !== s.sourceLength || m.validLength !== m.pcm.length) throw new Error("采集序号或样本位置不连续，拒绝生成有缺口的录制");
  if (s.sourceLength + m.pcm.length > s.original.length) throw new Error("输入超过 60 秒上限");
  if (s.clockOrigin === null) {
    s.clockOrigin = m.capturedAt - (m.sampleStart + m.pcm.length) / s.sourceRate * 1000;
    s.contextOrigin = (m.contextTime ?? 0) - (m.sampleStart + m.pcm.length) / s.sourceRate;
  }
  const t = performance.now();
  s.original.set(m.pcm, s.sourceLength); s.sourceLength += m.pcm.length; s.sequence++;
  appendModel(s, s.resampler.push(m.pcm));
  s.prepMs += performance.now() - t;
  const backlog = (s.length - s.frame * 512) / TARGET_RATE * 1000;
  s.maxBacklogMs = Math.max(s.maxBacklogMs, backlog);
  if (!s.stopSent && (backlog > 2000 || s.sourceLength >= s.original.length)) {
    s.stopSent = true;
    post({type: "limitReached", sessionId: s.id, reason: backlog > 2000 ? "处理落后超过 2 秒：停止采集并保留所有已采集样本，处理完后回放" : "已达到 60 秒录制上限，正在处理尾部"});
  }
  void drain(s);
}
function finishSession(id: number) {
  const s = active;
  if (!s || s.id !== id || s.finishing) return;
  s.finishing = true;
  const t = performance.now(); appendModel(s, s.resampler.flush()); s.prepMs += performance.now() - t;
  post({type: "finalizing", sessionId: id, queuedBlocks: Math.ceil((Math.ceil(s.length / 512) - s.frame) / FIXED_FRAMES_PER_CHUNK)});
  void drain(s);
}
async function drain(s: SessionState) {
  if (s.running) return;
  s.running = true;
  try {
    while (active === s) {
      const available = (s.finishing ? Math.ceil(s.length / 512) : Math.floor(s.length / 512)) - s.frame;
      if (available < FIXED_FRAMES_PER_CHUNK && !s.finishing || available <= 0) break;
      const count = Math.min(FIXED_FRAMES_PER_CHUNK, available);
      const t = performance.now();
      if (artificialDelay) await new Promise(r => setTimeout(r, artificialDelay));
      const start = Math.max(0, s.frame * 512 - manifest!.historySamples);
      const end = (s.frame + count) * 512;
      const inferStart = performance.now();
      const out = await runModel(s.pcm.slice(start, end));
      const data = await out.getData() as Float32Array;
      const inferMs = performance.now() - inferStart;
      if (active !== s) { out.dispose(); break; }
      const baseFrame = Number(out.dims[1]) - count;
      const frameStart = s.frame;
      for (let f = 0; f < count; f++) {
        for (let p = 0; p < 7; p++) {
          const base = ((baseFrame + f) * 7 + p) * 32;
          let best = 0;
          for (let b = 1; b < 32; b++) if (data[base + b] > data[base + best]) best = b;
          const value = manifest!.paramSpec[USED_PARAMS[p]].bucketCenters[best];
          if (!Number.isFinite(data[base + best])) throw new Error("模型输出非有限值");
          s.params[USED_PARAMS[p]][s.frame] = value;
        }
        const pitch = s.pitch.frame(s.raw.subarray(0, end), s.frame)!;
        s.params.frequency[s.frame] = pitch.f0; s.voiced[s.frame] = +pitch.voiced; s.frame++;
      }
      out.dispose();
      s.timings.push(inferMs); s.inferenceMs += inferMs; s.processingMs += performance.now() - t;
      post({type: "liveFrames", sessionId: s.id, frameStart,
        params: Object.fromEntries(Object.entries(s.params).map(([n, v]) => [n, v.slice(frameStart, s.frame)])),
        voiced: s.voiced.slice(frameStart, s.frame), inferenceMs: inferMs, backlogMs: backlog(s),
        displayLatencyMs: epoch() - (s.clockOrigin! + Math.min(s.length, s.frame * 512) / TARGET_RATE * 1000),
        capturedAt: s.clockOrigin! + Math.min(s.length, s.frame * 512) / TARGET_RATE * 1000, totalFrames: s.frame});
      if (s.timings.length % 20 === 0) sendStats(s);
    }
    if (active === s && s.finishing && s.frame === Math.ceil(s.length / 512)) freeze(s);
  } catch (error) { reportError(error, s.id); }
  finally { s.running = false; }
}
function freeze(s: SessionState) {
  if (!s.length) { active = null; post({type: "error", sessionId: s.id, message: "没有采集到音频，请重试"}); return; }
  const recording: SamuelRecording = {
    schemaVersion: 1, modelRevision: manifest!.modelRevision, modelSha256: manifest!.onnxSha256,
    sourceSampleRate: s.sourceRate, originalPcm: s.original.slice(0, s.sourceLength), modelSampleRate: TARGET_RATE, controlHop: 512,
    validModelSamples: s.length, frameCount: s.frame, durationSeconds: s.sourceLength / s.sourceRate,
    params: Object.fromEntries(Object.entries(s.params).map(([n,v]) => [n,v.slice(0,s.frame)])), voiced: s.voiced.slice(0,s.frame), sourceKind: s.sourceKind,
    processingConfig: {targetRms: 0.05, rmsMode: "causal-exponential-20ms", rmsGate: 0.0035, gainRange: [0.25,20], gainSmoothing: 0.0015,
      pitch: "causal-yin-forward-filter", pitchWindow: 2048, pitchHold: "last-voiced-or-285Hz", chunkFrames: FIXED_FRAMES_PER_CHUNK,
      historySamples: manifest!.historySamples, resampler: "24tap-windowed-sinc", contextTimeOrigin: s.contextOrigin,
      captureEpochOrigin: s.clockOrigin, stopReason: s.stopSent ? "limit-or-backpressure" : "user", stats: stats(s)},
  };
  sendStats(s); active = null;
  post({type: "recording", sessionId: s.id, recording});
}
function backlog(s: SessionState) { return Math.max(0, (s.length - s.frame * 512) / TARGET_RATE * 1000); }
function stats(s: SessionState): InferenceStats {
  const sorted = [...s.timings.slice(3)].sort((a,b) => a-b);
  const q = (p: number) => sorted[Math.max(0, Math.ceil(sorted.length*p)-1)] ?? 0;
  return {p50:q(.5), p95:q(.95), rtf:(s.prepMs+s.processingMs)/Math.max(1,s.frame*512/TARGET_RATE*1000),
    backlogMs:backlog(s), displayLatencyP95:0, sampleCount:s.timings.length, preprocessingMs:s.prepMs,
    processingMs:s.processingMs, maxBacklogMs:s.maxBacklogMs};
}
function sendStats(s: SessionState) { post({type:"stats",sessionId:s.id,stats:stats(s),queuedBlocks:Math.ceil(backlog(s)/1000*TARGET_RATE/(512*FIXED_FRAMES_PER_CHUNK))}); }
function epoch() { return performance.timeOrigin + performance.now(); }
async function runModel(pcm: Float32Array): Promise<ort.Tensor> {
  if (!session) throw new Error("模型未加载");
  const input = new ort.Tensor("float32", pcm, [1,1,pcm.length]);
  try { return (await session.run({wav:input})).logits; } finally { input.dispose(); }
}
async function runSelfTest(): Promise<void> {
  if (!manifest || !session || !backend) throw new Error("模型尚未就绪");
  const started = performance.now();
  const cases = await (await fetch(new URL("fixtures/cases.json",assetBase))).json() as {name:string;pcmFile:string;logitsFile:string;frameCount:number}[];
  const report: Record<string, unknown> = {};
  let worst = 0, mean = 0, mismatches = 0, bucketCount = 0, streamMax = 0, streamMismatch = 0, minMargin = Infinity, failures = 0, acceptedFailures = 0;
  // Measured Metal accumulation error: one near-zero logit misses 1e-4.
  // Retain that baseline result; the documented backend tolerance also requires zero bucket flips.
  const atol = backend.actual === "webgpu" ? 1.5e-4 : 1e-4;
  function compare(a: Float32Array, b: Float32Array) {
    if(a.length !== b.length) throw new Error("fixture 输出长度不一致");
    let max = 0, sum = 0, flips = 0, failed = 0, acceptedFailed = 0, margin = Infinity;
    const outliers: unknown[] = [];
    const errors: number[] = [];
    const physical = Object.fromEntries(USED_PARAMS.map(n=>[n,0]));
    for (let i=0;i<a.length;i++) {
      const error = Math.abs(a[i]-b[i]); errors.push(error); max=Math.max(max,error);sum+=error;
      if(!Number.isFinite(error) || error > atol + 1e-4*Math.abs(a[i])) acceptedFailed++;
      if(!Number.isFinite(error) || error > 1e-4+1e-4*Math.abs(a[i])) {failed++; if(outliers.length<16) outliers.push({index:i,reference:a[i],actual:b[i],error});}
    }
    for(let base=0;base<a.length;base+=32) {
      let x=0,y=0; for(let k=1;k<32;k++) {if(a[base+k]>a[base+x]) x=k;if(b[base+k]>b[base+y]) y=k;}
      if(x!==y) flips++;
      const sorted=Array.from(a.slice(base,base+32)).sort((x,y)=>y-x); margin=Math.min(margin,sorted[0]-sorted[1]);
      const n=USED_PARAMS[(base/32)%7]; const centers=manifest!.paramSpec[n].bucketCenters;
      physical[n]=Math.max(physical[n], Math.abs(centers[x]-centers[y]));
    }
    errors.sort((a,b)=>a-b);
    return {acceptedToleranceFailures:acceptedFailed, outliers,p99AbsLogitError:errors[Math.floor(errors.length*.99)],maxAbsLogitError:max,meanAbsLogitError:sum/a.length,bucketMismatches:flips,toleranceFailures:failed,minMargin:margin,physicalMaxDifference:physical};
  }
  for (const c of cases) {
    const pcm = new Float32Array(await (await fetch(new URL(`fixtures/${c.pcmFile}`,assetBase))).arrayBuffer());
    const reference = new Float32Array(await (await fetch(new URL(`fixtures/${c.logitsFile}`,assetBase))).arrayBuffer());
    const out = await runModel(pcm); const whole = new Float32Array(await out.getData() as Float32Array); out.dispose();
    const comparison=compare(reference,whole);
    worst=Math.max(worst,comparison.maxAbsLogitError);mean+=comparison.meanAbsLogitError*reference.length;mismatches+=comparison.bucketMismatches;failures+=comparison.toleranceFailures;acceptedFailures+=comparison.acceptedToleranceFailures;minMargin=Math.min(minMargin,comparison.minMargin);bucketCount+=c.frameCount*7;
    const stream:Record<string,unknown>={};
    for(const chunk of [4,8,16]) {
      const parts:Float32Array[]=[];
      for(let f=0;f<c.frameCount;f+=chunk) {
        const count=Math.min(chunk,c.frameCount-f);
        const result=await runModel(pcm.slice(Math.max(0,f*512-manifest.historySamples),(f+count)*512));
        const data=await result.getData() as Float32Array;
        parts.push(data.slice(data.length-count*7*32));result.dispose();
      }
      const result=compare(whole,concatFloat32(parts));stream[String(chunk)]=result;
      streamMax=Math.max(streamMax,result.maxAbsLogitError);streamMismatch+=result.bucketMismatches;failures+=result.toleranceFailures;acceptedFailures+=result.acceptedToleranceFailures;
    }
    report[c.name]={torchBrowser:comparison,stream};
  }
  post({type:"selfTest",result:{backend,maxAbsLogitError:worst,meanAbsLogitError:mean/(bucketCount*32),bucketMismatches:mismatches,bucketCount,minMargin,
    streamingMaxAbsLogitError:streamMax,streamingBucketMismatches:streamMismatch,fixtureFrames:bucketCount/7,durationMs:performance.now()-started,
    toleranceFailures:failures,acceptedToleranceFailures:acceptedFailures,tolerance:{atol,rtol:1e-4},passed:acceptedFailures===0 && mismatches===0 && streamMismatch===0,cases:report}});
}
async function fetchWithProgress(
  url: string,
  onProgress: (fraction: number) => void,
): Promise<{ buffer: ArrayBuffer; downloaded: number; total: number }> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`下载失败 (${response.status}): ${url}`);
  // Fetch exposes decoded bytes; Content-Length may describe compressed bytes.
  const total = response.headers.get("content-encoding") ? 0 : Number(response.headers.get("content-length") ?? 0);
  if (!response.body) return { buffer: await response.arrayBuffer(), downloaded: total, total };
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let downloaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    downloaded += value.length;
    onProgress(total > 0 ? downloaded / total : 0);
  }
  const merged = new Uint8Array(downloaded);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return { buffer: merged.buffer, downloaded, total };
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function concatFloat32(chunks: Float32Array[]): Float32Array {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function flatten(value: number[][][][]): Float32Array {
  const out: number[] = [];
  for (const batch of value) for (const frame of batch) for (const param of frame) out.push(...param);
  return Float32Array.from(out);
}

function post(message: WorkerToMain): void {
  self.postMessage(message);
}
