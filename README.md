# Samuel Live

纯前端 Samuel 麦克风声道 Demo：AudioWorklet 持续采集 → Worker 中重采样、因果 RMS/YIN、真实 ONNX 控制器 → Pink Trombone 声道。停止后从保存的参数轨迹重新合成声音，动画和声音使用同一个 AudioContext 时钟。

界面为中文，音频只保存在本机内存。刷新会清空录制。没有推理服务器、上传接口、账户或统计服务。默认没有实时监听。

## 启动

需要 Node.js 22.12 或更新的受支持版本、pnpm 10.18.3。浏览器优先使用桌面 Chrome/Edge，启用 WebGPU；没有 WebGPU 时明确回退到单线程 WASM。小型 fixture、Pink Trombone 源码和锁文件在 Git 中；13 MB ONNX 模型不提交到 Git，从干净 checkout 构建前需要安装 uv 并按下方命令生成。

```bash
pnpm install --frozen-lockfile
pnpm model:export
pnpm test
pnpm build
pnpm preview
```

访问终端显示的 localhost 地址，通常为 `http://127.0.0.1:4173/`。开发模式运行 `pnpm dev`。必须通过 localhost 或 HTTPS 访问，不能直接双击 `dist/index.html`。

1. 等待页面显示“模型已就绪”和实际后端。
2. 点击“开始录制”，允许麦克风；说话时声道持续更新。单次最多 60 秒。
3. 点击“停止”，等待“录制已冻结”。这时采集 tracks 已释放、在途推理与短尾部已完成。
4. 点击“回放 Samuel”。支持暂停、继续、从头播放；拖动进度时静音显示目标姿态，点击继续恢复发声。
5. 勾选“原声对照”播放保存的原始录音；切换保留同一源时间位置。

“实时注入语音”按钮会把上游真实短语音循环 60 秒，经同一个采集 AudioWorklet 实时送入模型。它明确属于受控注入，不能代表真实麦克风或多样化长语音的听感验收。可随时点击停止。

## 固定资产与重新导出

上游代码提交 `1da0c6fd5563c34bff469fa68601feb91b5cf8c9`；权重 revision `cfd4ba3b0ea411d40f404d226022bc70f9e06139`。浏览器使用 `public/models/samuel-controller-fp32.onnx`（13,891,818 字节），加载时验证 SHA-256。完整身份、参数顺序、从 checkpoint 读取的 float32 bucket centers、固定参数和历史长度见 `public/models/samuel-manifest.json`。

从干净 checkout 可直接构建；如需重现导出，安装 uv 后执行：

```bash
uv sync --project tools --frozen
uv run --project tools --frozen python tools/export_model.py
pnpm verify:assets
pnpm test:python
pnpm test
pnpm build
```

导出器自动下载固定 revision 的 checkpoint/config 到忽略的 `tools/cache/`，校验已知 SHA-256，严格加载真实权重并折叠 weight normalization。只导出 encoder + linear head 的 logits，不包含训练损失、Whisper、wav2vec 或 Python 合成器。原始历史配置不发布；`public/models/config.json` 只保留控制器配置。不要将 cache、`.venv`、生成的 ONNX 或训练路径上传。

真实语音 fixture 已随仓库提供，来源、原文件和派生文件 SHA-256 见 `public/fixtures/provenance.json`。导出器生成短输入、静音、随机信号、真实语音的 PCM、PyTorch logits、bucket、物理参数和原版因果 YIN 参考。代码、权重和语音来源及本地修改记录见 `vendor/SOURCES.json`、`THIRD_PARTY_NOTICES.md`。

## 验证

```bash
pnpm test                         # 采样率、分块边界、静音、RMS 因果性、真实语音 YIN
pnpm test:python                  # 上游编码器与因果边界
pnpm test:browser                 # 先 pnpm build；本机已安装 Chrome
node scripts/validate-microphone.mjs # getUserMedia 入口，Chrome 文件模拟麦克风
node scripts/validate-errors.mjs  # 回压、权限/下载故障、device lost、44.1k、无 GPU
QUICK=1 WASM=1 pnpm test:browser   # 单线程 WASM 数值对照
```

浏览器工具使用 Playwright 启动本机 Chrome，自己提供 `/samuel-live/` 静态子路径，不添加 COOP/COEP；不调用任何推理 API。完整测试约两分钟，包含 60 秒实时注入、31 秒真实合成输出采集、传输控制与重录。浏览器诊断入口 `?validate=1` 提供测试探针；`&wasm=1` 强制 WASM，`&delay=200` 注入处理延迟，`&rate=44100` 请求 44.1 kHz AudioContext。这些开关只在显式验证模式启用。

结果写入 `docs/browser-*.json` 和日志，实际合成音频为 `docs/samuel-browser-replay.wav`。性能定义、实测数据、浮点容差说明与未测项目见 [docs/validation.md](docs/validation.md)。自动测试没有替代人工听感审查。

## GitHub Pages

`.github/workflows/pages.yml` 在推送 `main` 或手动运行时安装锁定依赖、测试、校验模型、构建并发布 `dist/`。运行资产使用相对路径，适配 `https://<owner>.github.io/<repo>/`。

仓库不提交 13 MB 的 ONNX 模型或验证 WAV。workflow 在构建前用固定 checkpoint 导出模型并生成 manifest，然后构建 `dist/`。推送到 `main` 后在 Settings → Pages → Source 选择 **GitHub Actions**。默认分支不是 `main` 时修改 workflow 的分支条件。

构建会从固定的 ONNX Runtime 包复制同版本 JS/WASM，从 `vendor/Pink-Trombone/` 用 pnpm 构建 Worklet。`dist/` 已生成，可直接静态托管。已有无关 DDSP/TFLite 本地文件不进入本 demo 的构建产物。

## 限制与许可

已验证平台为 Apple M5、macOS 26.6、Chrome 155 的 WebGPU，以及 Chrome 单线程 WASM 数值路径。真实麦克风硬件、Edge/移动端、人工可懂度和跨设备性能仍需现场验证。页面的错误提示与回压会明确反映无法跟上的设备。

本项目包含 GPL-3.0 的 Samuel、其修改版 Pink Trombone 及模型；请保留 LICENSE、上游署名和对应源码。ONNX Runtime 使用 MIT 许可证。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
