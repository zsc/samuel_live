import captureUrl from "./capture.worklet.ts?worker&url";
import { PinkTromboneSession } from "./pink-trombone";
import { RecordingPlayer } from "./playback";
import type {
  BackendInfo,
  MainToWorker,
  ModelManifest,
  SamuelRecording,
  WorkerToMain,
} from "./types";

type AppState =
  | "loading"
  | "ready"
  | "requesting"
  | "recording"
  | "finalizing"
  | "playable"
  | "playing"
  | "paused"
  | "error";

const $ = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
};

const elements = {
  statusDot: $("status-dot"),
  statusTitle: $("status-title"),
  statusDetail: $("status-detail"),
  backendBadge: $("backend-badge"),
  record: $<HTMLButtonElement>("record-button"),
  stop: $<HTMLButtonElement>("stop-button"),
  play: $<HTMLButtonElement>("play-button"),
  pause: $<HTMLButtonElement>("pause-button"),
  restart: $<HTMLButtonElement>("restart-button"),
  original: $<HTMLInputElement>("original-toggle"),
  progress: $<HTMLInputElement>("progress"),
  timecode: $("timecode"),
  clock: $("clock-time"),
  notice: $("notice"),
  level: $("level-fill"),
  frames: $("fact-frames"),
  samples: $("fact-samples"),
  duration: $("fact-duration"),
  backlog: $("fact-backlog"),
  fixtureButton: $<HTMLButtonElement>("fixture-button"),
  selftestButton: $<HTMLButtonElement>("selftest-button"),
  selftestOutput: $("selftest-output"),
  metricRevision: $("metric-revision"),
  metricCheckpoint: $("metric-checkpoint"),
  metricOnnx: $("metric-onnx"),
  metricBackend: $("metric-backend"),
  metricRates: $("metric-rates"),
  metricFrameRate: $("metric-frame-rate"),
  metricLatency: $("metric-latency"),
  metricRtf: $("metric-rtf"),
  metricDisplay: $("metric-display"),
  footerRevision: $("footer-revision"),
  tractHost: $("tract-host"),
  tractScaler: $("tract-scaler"),
  pink: $("pink-trombone"),
  metrics: $<HTMLDetailsElement>("metrics-panel"),
};


let manifest: ModelManifest | null = null;
let backend: BackendInfo | null = null;
let state: AppState = "loading";
let sessionId = 0;
let recording: SamuelRecording | null = null;
let synth: PinkTromboneSession;
let player: RecordingPlayer;
let worker: Worker;
let modelReady = false;
let synthReady = false;
let captureStream: MediaStream | null = null;
let captureNode: AudioWorkletNode | null = null;
let captureSource: MediaStreamAudioSourceNode | null = null;
let fixtureSource: AudioBufferSourceNode | null = null;
let silence: GainNode | null = null;
let eosWaiter: (() => void) | null = null;
let stopping = false;
let requestToken = 0;
let captureSamples = 0;
let selfTesting = false;
let pendingDraw: {capturedAt: number; appliedAt: number} | null = null;
const displayLatencies: number[] = [];
const drawAges: number[] = [];
const telemetry: Record<string, unknown> = {liveMessages: 0};
const query = new URLSearchParams(location.search);
const validation = query.has("validate");
const assetBase = new URL("./", document.baseURI).href;
void boot();
async function boot() {
  resizeTract(); new ResizeObserver(resizeTract).observe(elements.tractHost);
  elements.record.onclick = () => void startCapture(false);
  elements.fixtureButton.onclick = () => void startCapture(true);
  elements.stop.onclick = () => void stopCapture();
  elements.play.onclick = () => void player.play().catch(fail);
  elements.pause.onclick = () => player.pause();
  elements.restart.onclick = () => void player.restart().catch(fail);
  elements.original.onchange = () => void player.setMode(elements.original.checked ? "original" : "samuel").catch(fail);
  elements.progress.oninput = () => void player.seek(Number(elements.progress.value) / 1000 * player.duration).catch(fail);
  elements.selftestButton.onclick = () => {
    selfTesting = true; updateAvailability();
    elements.selftestOutput.hidden = false; elements.selftestOutput.textContent = "正在对照 PyTorch fixture…";
    worker.postMessage({type:"selfTest"});
  };
  document.getElementById("retry-button")!.onclick = () => location.reload();
  elements.pink.addEventListener("samuel-draw", event => {
    if (!synthReady) return;
    const time = (event as CustomEvent).detail.contextTime;
    if (!Number.isFinite(time)) return;
    if (state === "playing") { drawAges.push(Math.max(0, (synth.context.currentTime - time) * 1000)); if (drawAges.length > 8192) drawAges.shift(); }
    if (pendingDraw && time >= pendingDraw.appliedAt + .004) {
      displayLatencies.push(performance.timeOrigin + performance.now() - pendingDraw.capturedAt);
      pendingDraw = null;
      elements.metricDisplay.textContent = `${percentile(displayLatencies, .95).toFixed(1)} ms（实际绘图）`;
    }
  });
  initializeWorker();
  try {
    synth = await PinkTromboneSession.create(elements.pink as never);
    player = new RecordingPlayer(synth);
    player.onSnapshot(snapshot => {
      if (["playable", "paused", "playing"].includes(state)) {
        state = snapshot.playing ? "playing" : "paused";
        document.body.dataset.state = state;
        setStatus(snapshot.playing ? "正在播放" : "已暂停 / 可回放", snapshot.mode === "samuel" ? "Samuel 合成 · 声道与声音共享音频时钟" : "原声对照 · 未经模型音量归一化的录音", snapshot.playing ? "active" : "ready");
        elements.progress.value = String(snapshot.duration ? snapshot.position / snapshot.duration * 1000 : 0);
        elements.timecode.textContent = `${formatTime(snapshot.position)} / ${formatTime(snapshot.duration)}`;
        elements.clock.textContent = snapshot.position.toFixed(1);
        updateAvailability();
      }
    });
    synthReady = true;
    if (modelReady) setState("ready");
  } catch (e) { fail(e); }
  if (validation) Object.assign(window, {__samuel: {
    get recording() { return recording; }, get telemetry() { return {...telemetry, displayLatencyP95: percentile(displayLatencies,.95), drawAgeP95:percentile(drawAges,.95), drawAgeMax: Math.max(0,...drawAges), displayCount:displayLatencies.length, drawCount:drawAges.length, state, captureTracks: captureStream?.getTracks().map(t=>t.readyState) ?? []}; },
    get player() { return player; }, get synth() { return synth; },
    startFixture: () => startCapture(true), stop: stopCapture,
    loseDevice: () => worker.postMessage({type:"diagnosticLoss"}),
    revokeInput: () => { const t=captureStream?.getAudioTracks()[0]; if(t){t.stop();t.dispatchEvent(new Event("ended"));} },
  }});
  window.addEventListener("pagehide", () => { releaseCapture(); worker.terminate(); void synth?.context.close(); });
}
function initializeWorker() {
  worker = new Worker(new URL("./inference.worker.ts", import.meta.url), {type:"module"});
  worker.onmessage = (event: MessageEvent<WorkerToMain>) => void onWorker(event.data);
  worker.onerror = e => fail(`Worker 错误：${e.message}`);
  worker.postMessage({type:"init", assetBase, diagnostics: validation, forceWasm: validation && query.has("wasm"), delayMs: validation ? Number(query.get("delay") ?? 0) : 0} satisfies MainToWorker);
}
async function onWorker(m: WorkerToMain) {
  if ("sessionId" in m && m.sessionId !== undefined && m.sessionId !== sessionId) return;
  if (m.type === "loadProgress") setStatus("正在加载模型", `${m.stage} · ${Math.round(m.progress*100)}%`);
  else if (m.type === "ready") {
    manifest = m.manifest; backend = m.backend; modelReady = true;
    telemetry.backend = backend; telemetry.loadMs = m.loadMs;
    elements.footerRevision.textContent = manifest.modelRevision.slice(0,12);
    elements.metricRevision.textContent = manifest.modelRevision;
    elements.metricCheckpoint.textContent = manifest.checkpointSha256;
    elements.metricOnnx.textContent = manifest.onnxSha256;
    elements.metricBackend.textContent = backend.actual + " · " + (backend.adapter ?? backend.fallbackReason ?? "");
    elements.backendBadge.textContent = backend.actual === "webgpu" ? "WebGPU" : "WASM（回退）";
    elements.backendBadge.dataset.backend = backend.actual;
    elements.metricFrameRate.textContent = `${manifest.controlFrameRate.toFixed(4)} Hz`;
    setStatus("模型已就绪", `${backend.actual.toUpperCase()} · ${(m.loadMs/1000).toFixed(1)} 秒`, "ready");
    if (synthReady) setState("ready");
  } else if (m.type === "liveFrames") {
    synth.showFrames(m.params);
    pendingDraw = {capturedAt:m.capturedAt, appliedAt:synth.context.currentTime};
    telemetry.liveMessages = Number(telemetry.liveMessages) + 1;
    telemetry.lastFrame = m.totalFrames;
    elements.frames.textContent = String(m.totalFrames);
    elements.backlog.textContent = `${m.backlogMs.toFixed(0)} ms`;
  } else if (m.type === "stats") {
    telemetry.stats = m.stats;
    elements.metricLatency.textContent = `${m.stats.p50.toFixed(1)} / ${m.stats.p95.toFixed(1)} ms`;
    elements.metricRtf.textContent = `${m.stats.rtf.toFixed(3)}（重采样 + RMS + YIN + 推理读回）`;
    elements.backlog.textContent = `${m.stats.backlogMs.toFixed(0)} ms`;
    if (m.stats.backlogMs > 500 && state === "recording") elements.notice.textContent = "处理落后；保留完整采样，积压超过 2 秒会正常停止采集。";
  } else if (m.type === "limitReached") {
    telemetry.stopReason = m.reason; elements.notice.textContent = m.reason; await stopCapture();
  } else if (m.type === "finalizing") {
    setState("finalizing"); setStatus("正在处理尾部", `尚有 ${m.queuedBlocks} 个推理块`);
  } else if (m.type === "recording") {
    recording = m.recording;
    if (recording.frameCount !== Math.ceil(recording.validModelSamples/512)) return fail("冻结帧数不一致");
    await player.setRecording(recording);
    await player.setMode(elements.original.checked ? "original" : "samuel");
    setState("playable");
    elements.frames.textContent = String(recording.frameCount);
    elements.samples.textContent = `${recording.validModelSamples.toLocaleString()} @ 44.1k`;
    elements.duration.textContent = `${recording.durationSeconds.toFixed(3)} 秒`;
    elements.timecode.textContent = `0:00.0 / ${formatTime(recording.durationSeconds)}`;
    setStatus("录制已冻结", `${recording.frameCount} 控制帧 · ${recording.durationSeconds.toFixed(3)} 秒`, "ready");
    elements.notice.textContent = (recording.sourceKind === "fixture" ? "受控样例注入（非真实麦克风验收）。" : "麦克风已释放。") + "回放 Samuel 重新合成声音；原声对照播放保存的原始录音。";
  } else if (m.type === "selfTest") {
    telemetry.selfTest = m.result; selfTesting = false; updateAvailability();
    elements.selftestOutput.textContent = JSON.stringify(m.result,null,2);
  } else if (m.type === "error" || m.type === "loadError") {
    if (!('sessionId' in m) || m.sessionId === undefined) modelReady = false;
    selfTesting = false; fail(m.message);
  }
}
async function startCapture(fixture: boolean) {
  if (!modelReady || !synthReady || selfTesting || !["ready","playable","paused"].includes(state)) return;
  const token = ++requestToken;
  setState("requesting"); setStatus(fixture ? "准备受控语音注入" : "正在请求麦克风", "请允许音频权限；默认不监听输入");
  player.stop();
  try {
    await synth.resume();
    let stream: MediaStream;
    if (fixture) {
      const response = await fetch(new URL("fixtures/samuel-sample-48k.wav",assetBase));
      if (!response.ok) throw new Error("样例语音下载失败");
      const buffer = await synth.context.decodeAudioData(await response.arrayBuffer());
      fixtureSource = synth.context.createBufferSource(); fixtureSource.buffer = buffer; fixtureSource.loop = true;
      const dest = synth.context.createMediaStreamDestination(); fixtureSource.connect(dest);
      stream = dest.stream;
    } else stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:false, autoGainControl:false}});
    if (token !== requestToken) { stream.getTracks().forEach(t=>t.stop()); return; }
    captureStream = stream;
    stream.getAudioTracks().forEach(track => track.onended = () => { if (state === "recording") { elements.notice.textContent="麦克风权限或设备已撤销，正在保存已采集音频。"; void stopCapture(); } });
    await synth.context.audioWorklet.addModule(captureUrl);
    if (token !== requestToken) { releaseCapture(); return; }
    sessionId++; recording = null; captureSamples = 0; stopping = false;
    displayLatencies.length = 0; drawAges.length = 0; pendingDraw = null;
    telemetry.liveMessages = 0; telemetry.stopReason = null;
    const rate = synth.context.sampleRate;
    worker.postMessage({type:"start",sessionId,sourceSampleRate:rate,sourceKind:fixture?"fixture":"microphone"} satisfies MainToWorker);
    captureNode = new AudioWorkletNode(synth.context,"samuel-mic-capture",{numberOfInputs:1,numberOfOutputs:1,channelCount:1,processorOptions:{sessionId,chunkFrames:1024}});
    captureNode.onprocessorerror = () => fail("采集 AudioWorklet 错误，请重试");
    captureNode.port.onmessage = e => {
      const d = e.data;
      if (d.sessionId !== sessionId) return;
      if (d.type === "chunk") {
        const pcm = new Float32Array(d.pcm);
        captureSamples += pcm.length;
        let energy = 0; for (const v of pcm) energy += v*v;
        elements.level.style.transform = `scaleX(${Math.min(1,Math.sqrt(energy/pcm.length)*8)})`;
        elements.clock.textContent = (captureSamples/rate).toFixed(1);
        const capturedAt = performance.timeOrigin + performance.now() - (synth.context.currentTime - d.contextTime) * 1000;
        worker.postMessage({type:"audioChunk",sessionId,sequence:d.sequence,sampleStart:d.sampleStart,validLength:d.validLength,contextTime:d.contextTime,capturedAt,pcm} satisfies MainToWorker,[pcm.buffer]);
      } else if (d.type === "eos") {
        if (d.totalSamples !== captureSamples) return fail("采集尾部样本数不一致");
        telemetry.captureSamples = captureSamples;
        if (eosWaiter) { const done=eosWaiter; eosWaiter=null; done(); }
        else { setState("finalizing"); releaseCapture(); worker.postMessage({type:"finish",sessionId}); }
      }
    };
    captureSource = synth.context.createMediaStreamSource(stream);
    silence = synth.context.createGain(); silence.gain.value = 0;
    captureSource.connect(captureNode).connect(silence).connect(synth.context.destination);
    fixtureSource?.start();
    elements.metricRates.textContent = `${rate} → 44100 Hz`;
    setState("recording"); setStatus(fixture ? "受控语音注入中" : "正在录制", "持续推理；60 秒上限；点击停止会完成尾部", "recording");
    elements.notice.textContent = fixture ? "官方真实语音循环实时送入采集 AudioWorklet 与真实模型；这不是麦克风验收。" : "原音频仅保存在本机内存；持续推理，默认不播放输入。";
  } catch (e) { if (token === requestToken) fail(e); }
}
async function stopCapture() {
  if (state === "requesting") { requestToken++; releaseCapture(); setState("ready"); setStatus("已取消请求", "可重新开始", "ready"); return; }
  if (state !== "recording" || stopping) return;
  stopping = true; setState("finalizing"); setStatus("正在结束采集", "等待 AudioWorklet 最后一块与结束确认");
  try {
    await new Promise<void>((resolve,reject) => {
      const timer = setTimeout(()=>{eosWaiter=null;reject(new Error("采集线程未确认结束；本次未冻结，请重试"));},2000);
      eosWaiter = ()=>{clearTimeout(timer);resolve();};
      captureNode!.port.postMessage({type:"stop"});
    });
    releaseCapture(); worker.postMessage({type:"finish",sessionId});
  } catch(e) { fail(e); }
}
function releaseCapture() {
  elements.level.style.transform = "scaleX(0)";
  captureStream?.getTracks().forEach(t=>{t.onended=null;t.stop();}); captureStream=null;
  captureNode?.disconnect(); if (captureNode) captureNode.port.onmessage=null; captureNode=null;
  captureSource?.disconnect(); captureSource=null;
  silence?.disconnect(); silence=null;
  if (fixtureSource) { try { fixtureSource.stop(); } catch {} fixtureSource.disconnect(); fixtureSource=null; }
}
function fail(error: unknown) {
  const text = message(error); telemetry.error = text;
  requestToken++; releaseCapture(); worker?.postMessage({type:"cancel",sessionId});
  player?.stop();
  setState(recording ? "playable" : modelReady && synthReady ? "ready" : "error");
  setStatus("发生错误", text, "error");
  elements.notice.textContent = `${text}。可重试录制或点击“重新加载”恢复模型。`;
}
function updateAvailability() {
  const idle = ["ready","playable","paused"].includes(state) && !selfTesting;
  const replay = Boolean(recording) && ["playable","paused","playing"].includes(state) && !selfTesting;
  elements.record.disabled = !modelReady || !synthReady || !idle;
  elements.fixtureButton.disabled = elements.record.disabled;
  elements.stop.disabled = !["recording","requesting"].includes(state);
  elements.play.disabled = !replay || state === "playing";
  elements.play.textContent = state === "paused" ? "继续回放" : "回放 Samuel";
  elements.pause.disabled = !replay || state !== "playing";
  elements.restart.disabled = !replay;
  elements.original.disabled = !replay;
  elements.progress.disabled = !replay;
  elements.selftestButton.disabled = !modelReady || !synthReady || !idle;
}
function setState(next: AppState) { state=next; document.body.dataset.state=next; updateAvailability(); }
function setStatus(title:string, detail:string, tone="busy") {
  elements.statusTitle.textContent=title; elements.statusDetail.textContent=detail; elements.statusDot.dataset.tone=tone;
}
function percentile(values:number[], p:number) { const sorted=[...values].sort((a,b)=>a-b); return sorted[Math.max(0,Math.ceil(sorted.length*p)-1)] ?? 0; }
function resizeTract(): void {
  const width = elements.tractHost.clientWidth;
  const height = elements.tractHost.clientHeight;
  const scale = Math.min(1, width / 600, height / 600);
  elements.tractScaler.style.transform = `scale(${scale})`;
  elements.tractScaler.style.left = `${Math.max(0, (width - 600 * scale) / 2)}px`;
  elements.tractScaler.style.top = `${Math.max(0, (height - 600 * scale) / 2)}px`;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const rest = seconds - mins * 60;
  return `${mins}:${rest.toFixed(1).padStart(4, "0")}`;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

