# 实际验证报告

验证日期：2026-10-25（Asia/Singapore）。这是实测报告，不把 AGENTS.md 的目标数字当作测量结果。

## 环境与交付范围

- CPU / GPU：Apple M5，10 核 GPU，Metal；macOS 26.6（25G72）。
- 浏览器：本机 Google Chrome **155.0.8059.86**，Playwright 1.56.1。自动验收使用 headless Chrome，真实 Apple Metal WebGPU；没有 SwiftShader 标志。
- 精度：FP32。onnxruntime-web **1.23.0**，单线程 WASM 基础设施。模型在自建 Worker 内执行，不使用 ORT WASM proxy 或 SharedArrayBuffer。
- 每次 4 控制帧 / 2048 新采样；44,100 Hz，hop 512，86.1328125 帧/秒。冷启动实际前缀，随后 5120 历史 + 2048 新采样。
- 上游代码：`1da0c6fd5563c34bff469fa68601feb91b5cf8c9`。
- 权重 revision：`cfd4ba3b0ea411d40f404d226022bc70f9e06139`。
- checkpoint SHA-256：`eb0d80dbd57a517473dcb8825f314040ef1e21bd8ec13e0af54cd7f6201e5f55`。
- 原始 config SHA-256：`c902d3a31e16ea08252f617a1a30550d957e597b3e6c1295cef8d4418121cd70`。
- ONNX SHA-256：`8c87218eb5bbbb590651386bc33fc05b7c3268660248dc6caa92307f0ccd8ff5`。

完整网页源代码、ONNX、参考 PCM/logits、许可证、锁文件、构建产物 `dist/`、Pages workflow 均已提供。没有 Git remote 或目标仓库，因此没有推送、线上部署或可声称的 GitHub Pages 地址。

## 模型与流式数值

导出器用 uv 锁定环境（PyTorch 2.8.0、Python ONNX Runtime 1.31.0）加载真实 checkpoint。30 个 weight-normalization 模块折叠前后最大差值为 0。导出仅含 encoder + head；按上游 `trainable_names()` 排序，在浏览器按 argmax 还原 checkpoint 中的 float32 bucket centers。

感受野从 encoder 配置逐层推导：起始卷积、每级残差的长分支、反转后的 `[4,4,4,8]` stride、最终卷积，得 5376 样本。所需对齐历史 `ceil((5376-512)/512)*512 = 5120`。测试覆盖真实短前缀、历史窗口填满、4/8/16 帧块、不是 hop 整数倍的短尾部和重录。

| 比较 | 结果 |
| --- | --- |
| PyTorch → Python ONNX：语音、173 样本短输入、静音、固定种子随机信号 | 四类均通过 `atol=1e-4, rtol=1e-4`；bucket 翻转 0 |
| Python ONNX：四类输入分别按 4/8/16 帧分块 → 整段 | 全部通过同一容差，bucket 翻转 0 |
| PyTorch → Chrome WebGPU：39,648 个 logits / 1,239 个参数判别 | 最大绝对误差 `2.2125244140625e-4`，全样本平均绝对误差 `1.8607638273621165e-5`；bucket 翻转 **0/1239** |
| WebGPU 各 4/8/16 分块 → 整段 | 本机实测最大绝对差值 **0**，bucket 翻转 0 |
| Chrome WASM → PyTorch | 最大绝对误差 `2.899169921875e-4`；原定 abs+rel 容差通过，bucket 翻转 0 |
| 物理参数还原 | 七个参数均无 bucket 导致的物理差异；metadata 使用 checkpoint 自带 float32 centers |
| 因果 YIN → 上游函数：真实语音 | 每帧 voiced 一致，F0 误差 < 0.1 Hz |

逐类误差、p99、边界结果和物理参数差异见 [model-validation.json](model-validation.json)、[browser-webgpu.json](browser-webgpu.json)、[browser-wasm.json](browser-wasm.json)。`public/fixtures/cases.json` 保存 Python bucket 和物理参数；`.f32` 保存完全相同的 PCM 与 PyTorch logits，浏览器对照不重复做在线预处理。

### WebGPU 容差例外与依据

原定 `atol=rtol=1e-4` 有 **1/39,648** 个 logit 不通过，报告仍保留 `toleranceFailures: 1`，没有删除该样本。它位于语音帧 22、constrictionIndex、bucket 14：PyTorch `-0.01707278937101364`，WebGPU `-0.017193935811519623`，绝对误差 `0.00012114644050598145`。该头的胜出 bucket 为 20，胜出 logit `43.157135009765625`，前二名间距 `0.6224212646484375`，不接近此次误差量级。

语音误差 p99 为 `1.068115234375e-4`，全部案例最小胜出间距 `4.84466552734375e-4`，所有 bucket 与物理参数一致。基于这一实际分布，WebGPU 的后端验收使用 **atol=1.5e-4，rtol=1e-4**，并继续要求零 bucket 翻转；调整后的超差数为 0。Python/WASM 保持原容差。`passed` 指这个明确记录的后端标准，不表示原定目标零超差。没有使用 FP16、量化或概率加权输出绕过差异。

### 真实执行后端

[WebGPU 日志](browser-webgpu.log) 中 ONNX Runtime 的节点分配诊断为：

```text
All nodes placed on [WebGpuExecutionProvider]. Number of nodes: 151
```

这是实际执行后的 ORT 诊断，包含关键卷积，没有把仅创建 adapter 当作模型执行证据。当前包的 JS kernel profiling 回调没有提供 kernel 事件，故采用上述 native EP 节点分配日志作证据。WASM 路径显示实际 WASM 与回退原因。

## 60 秒持续处理与完整性

测试来源为上游真实短语音循环，通过 AudioBufferSource → MediaStreamDestination → **同一个采集 AudioWorklet**，按音频时钟实时送入 Worker。不是预计算轨迹，也不是瞬间提交整段文件；每次均重新运行真实模型。原始采集 48 kHz，持续 sinc 重采样为 44.1 kHz。

| 指标 | 最后一轮实测 |
| --- | --- |
| 源 PCM | 2,880,000 样本 @ 48 kHz，60.000 秒 |
| 有效模型 PCM | 2,646,000 样本 @ 44.1 kHz |
| 控制帧 | `ceil(2646000/512) = 5168` |
| 持续返回的推理块 | 1292 |
| 热身后推理 p50 / p95 | **9.4 / 13.6 ms** |
| 处理 RTF | **0.29486** |
| 重采样 + RMS 总耗时 | 1119.8 ms |
| 推理读回 + 解码 + YIN 等总耗时 | 16,571.9 ms |
| 最大 / 最终积压 | **67.69 / 0 ms** |
| 采集完成到实际姿态绘制 p95 | **50.96 ms**，1292 次有效绘制观测 |

RTF 定义为 Worker 内重采样、在线 RMS、推理完成及结果读回、argmax、YIN 的累计耗时除以对应控制采样时长；不把仅提交 GPU 命令的时间当推理时间。它不包含硬件 ADC/DAC，也不把录制期间的等待音频时间计入计算耗时。p50/p95 排除最初 3 次推理；RTF 包含这些调用。最终短尾部不足一个 hop，RTF 分母按实际处理控制帧计，偏差小于一帧。

显示延迟测量：采集块携带 AudioContext 时间和源采样位置，在主线程建立输入采样 → 单调墙钟映射；实际绘制的末帧采集完成时刻由采样计数计算。Pink Trombone 的真实音频线程几何快照携带 AudioContext 时间，TractUI 绘制完成后发出事件，确认其快照晚于姿态应用时刻，才计入延迟。实时绘图取每个 4 帧块的最新姿态；完整 86.13 Hz 参数仍然全部保存。该延迟不包含未测量的麦克风/声卡硬件延迟，也不是每个未展示中间帧的延迟。

Worker 内 `stats.displayLatencyP95=0` 是旧消息字段占位，并未用于 UI/结论；实际测量值在同份 JSON 的 `live.displayLatencyP95`。不要把占位 0 解读为零显示延迟。

### 预处理与回压

- 24 tap windowed-sinc 保留连续相位和滤波历史；44.1/48 kHz 分块与整段结果一致，flush 幂等，输出长度按真实采样率计算。24 tap 核最多需要 12 个源采样的重采样前视缓冲，与模型 lookahead=0 分开。
- RMS 为 20 ms 指数能量估计，目标 0.05、静音门限 0.0035、增益范围 0.25–20、逐样本平滑系数 0.0015，历史不重新缩放。它与上游整段 RMS 不同。
- YIN 使用 2048 样本尾随窗、1024 积分窗、因果前向滤波；无声保持上次有效 F0，尚无有效值时为 285 Hz。不使用未来片段插值；它与原服务器离线 pYIN 不完全相同。
- PCM、参数存储有 60 秒硬上限；推理按 frame cursor 从固定 TypedArray 取数据，不维护无限任务链。积压超 2 秒会请求正常停止；等待 EOS 和尾部处理，不静默丢弃已采集块。
- 每块校验 session ID、sequence、sampleStart、validLength；旧 session 被隔离。停止必须收到 AudioWorklet EOS，超时会明确失败，不伪造完整录制。

人为每次增加 200 ms 延迟后，在 **2.474667 秒**停止采集，保留 118,784 个源采样 → 109,133 个模型采样 → 214 帧，与采样率和 ceil 公式相符。见 [browser-errors.json](browser-errors.json)。

## 合成与声画回放

60 秒录制由保存参数重新驱动 Pink Trombone；没有重新运行 Samuel、没有把原 WAV 冒充合成声。frequency → Hz，voiceness → tenseness，loudness → voiceness^0.25；舌尖与嘴唇使用两个独立 AudioParam 槽，嘴唇固定位置 43，长度 44，vibrato 参数按固定值初始化。

回放用 AudioParam 线性自动化，逐帧精确映射 `i*512/44100`，末尾按真实源时长关闭输出。暂停取消未来调度并淡出；seek 静音，继续/重播通过已确认的 Worklet 消息重建 glottis/waveguide 状态并静音预热 80 ms。动画读取实际音频线程几何与同一 AudioContext 时钟，没有用 rAF 次数累积位置。

- 实际录下 **31.034667 秒**合成输出：[samuel-browser-replay.wav](samuel-browser-replay.wav)。RMS **0.04948**、绝对峰值 **0.39715**、非有限值 **0**、无采样幅值削波。
- 从同一 AudioContext 读取的 31 秒回放位置为 **30.906667 秒**，包含约 80 ms 静音预热；没有模型重跑。
- 1997 次绘图观测：绘出的几何快照相对当前音频时钟的年龄 p95 **2.67 ms**、最大 **5.33 ms**。测量反映音频线程状态至主线程绘图的时钟差，不是光子到声波的硬件同步测量；未宣称实机前台显示器/扬声器物理误差。
- 暂停、继续、静音拖动到 42 秒、原声切换、从头播放、停止后重录均通过。短尾部与参数长度核对通过；第二次录制 1.216 秒 / 53,626 模型采样 / 105 帧。
- “原声对照”使用未经模型归一化的 originalPcm，与合成源互斥。页面默认不监听麦克风，自动控制时禁止拖拽覆盖模型参数。

**人工可懂度、主观音色、是否存在听觉上明显的块边界爆音仍未人工验收。** 波形有限、没有削波与时钟正确不能代替听感。已提供真实合成 WAV 供复听；不宣称与 Python OLA/FIR 合成逐采样相等。

## 静态部署、异常与回归

| 验收项 | 状态 / 证据 |
| --- | --- |
| 5 个 JS 数值/音频测试 | 通过，含真实语音的上游 YIN 对照 |
| 17 个上游因果/编码器测试 | 通过，[python-tests.txt](python-tests.txt) |
| 不含依赖、cache、生成资源的干净目录安装 / 测试 / 构建 | 通过；锁文件安装 75 个主工程依赖及 Pink Trombone 构建依赖，从固定 checkpoint 重新导出 ONNX（官方源超时后使用镜像，SHA-256 一致） |
| 项目子路径 `/samuel-live/` | 浏览器实际访问成功；JS、WASM、模型、Worker、采集/合成 Worklet 的 MIME 与响应记录在 browser-webgpu.json |
| 压缩模型响应 | 最终构建在 gzip + 压缩 Content-Length 下校验和/数值自检通过；见 [browser-webgpu-numerical.json](browser-webgpu-numerical.json) |
| COOP/COEP、SAB、远程推理服务 | 验收服务器未提供隔离头；页面不请求推理 `/api/*` |
| WebGPU 不可用 | Worker 能力探测中模拟移除 navigator.gpu，实际 WASM session 加载成功 |
| getUserMedia / 设备撤销 / 快速控制 | Chrome 文件模拟麦克风实际采集、停止与再次录制；撤销输入后冻结尾部、快速 play/seek/restart 通过；见 [browser-microphone.json](browser-microphone.json) |
| 390px 布局 | 桌面 Chrome 缩窄视口，实际截图且无横向溢出；不是移动硬件验收 |
| 麦克风拒绝 | 注入 NotAllowedError，反馈可见并能重试 |
| 模型下载失败 | 注入 HTTP 503，错误可见；恢复请求后“重新加载”成功 |
| GPU device lost | 实际调用 GPUDevice.destroy()；反馈明确，已冻结录制仍可回放 |
| 回压 | 注入 200 ms 延迟，自动停止、完整排空并核对样本/帧数 |
| 停止与重录 | tracks 释放；新 session、RMS、pitch、CNN 历史独立；参数与音频没有静默缺口 |
| GitHub Actions / 线上 Pages | workflow 已提供；尚未在目标仓库运行或部署 |
| 真实麦克风硬件、Edge、移动硬件 | **未测**；受控注入不能代替这些项目 |

音频和模型只在本机运行；模型与运行时从同源静态地址下载。自动测试使用受控音频，不访问用户麦克风。录制缓存和采样数组有明确上限；没有对长时间多次操作做完整堆快照泄漏审计，因此不声称已证明浏览器内部零内存增长。

## 复现入口

完整命令见 [README](../README.md)。关键脚本：`tools/export_model.py`、`scripts/verify-assets.mjs`、`scripts/validate-browser.mjs`、`scripts/validate-errors.mjs`、`scripts/validate-microphone.mjs`。后者使用 Chrome 的文件模拟麦克风走真实 getUserMedia 入口，仍明确不是物理麦克风验收。

## 紧凑布局复核（2026-10-25）

桌面采用左侧声道、右侧操作与概览，声道按容器宽高等比缩放，技术指标在有限高度内独立滚动。Chrome 155 实测 1440×900、1280×720、1024×768：指标折叠和展开时页面均无纵向或横向溢出。390×844 下主要操作在首屏内（底部约 693px），概览和详细指标仍需向下滚动；无横向溢出。这是桌面浏览器视口模拟，不是移动硬件测试。测量见 [layout-validation.json](layout-validation.json)，截图见 [1280px 布局](layout-1280.png)。

重新通过 TypeScript 检查、静态构建及 Chrome 文件模拟麦克风的录制、停止、撤销、快速回放控制回归。本轮只调整布局和几何缩放，不重测模型性能。缺少 WebGPU 时现有单线程 WASM 自动回退继续适用；WASM 数值验证通过不意味着所有 CPU 都满足实时吞吐要求。
