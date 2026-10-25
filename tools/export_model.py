"""Export the pinned Samuel controller to ONNX and build browser fixtures.

This intentionally imports the vendored, pinned upstream model implementation
rather than reimplementing weights or head ordering. Run with:

    uv run --project tools python tools/export_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "samuel" / "src"))

from samuel.model import PinkTromboneController, PinkTromboneControllerConfig  # noqa: E402

CHECKPOINT_REVISION = "cfd4ba3b0ea411d40f404d226022bc70f9e06139"
UPSTREAM_COMMIT = "1da0c6fd5563c34bff469fa68601feb91b5cf8c9"
SAMPLE_RATE = 44_100
HOP = 512


class LogitsExport(nn.Module):
    """Encoder + head, exposing the categorical logits used by eval()."""

    def __init__(self, model: PinkTromboneController):
        super().__init__()
        self.encoder = model.encoder
        self.head = model.head
        self.n_buckets = model.n_buckets

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        z = self.encoder(wav)
        h = self.head(z.transpose(1, 2)).float()
        return h.reshape(h.shape[0], h.shape[1], -1, self.n_buckets)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_weight_norm(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if hasattr(child, "parametrizations") and hasattr(child.parametrizations, "weight"):
            torch.nn.utils.parametrize.remove_parametrizations(child, "weight")
            count += 1
    return count


def load_controller(config_path: Path, checkpoint_path: Path) -> PinkTromboneController:
    run_config = json.loads(config_path.read_text())
    model_config = PinkTromboneControllerConfig.model_validate(run_config["model"])
    model = PinkTromboneController(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "tools/cache/config.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "tools/cache/last.pt")
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "public/fixtures/samuel-sample-44k.wav"
    )
    args = parser.parse_args()

    out_dir = ROOT / "public" / "models"
    fixture_dir = ROOT / "public" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    from fetch_model import fetch
    fetch()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = load_controller(args.config, args.checkpoint)
    wrapped = LogitsExport(model).eval()

    probe = torch.randn(1, 1, 8 * HOP)
    with torch.no_grad():
        before = wrapped(probe)
    folded = fold_weight_norm(wrapped)
    with torch.no_grad():
        after = wrapped(probe)
    fold_diff = float((before - after).abs().max())
    if fold_diff != 0.0:
        raise RuntimeError(f"weight-norm folding changed output: max diff {fold_diff}")

    onnx_path = out_dir / "samuel-controller-fp32.onnx"
    torch.onnx.export(
        wrapped,
        (probe,),
        onnx_path,
        input_names=["wav"],
        output_names=["logits"],
        dynamic_axes={"wav": {2: "samples"}, "logits": {1: "frames"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify PyTorch versus ONNX on real speech plus synthetic edge cases.
    if not args.sample.exists():
        raise FileNotFoundError(f"missing reference speech: {args.sample}")
    speech, sr = sf.read(args.sample, dtype="float32", always_2d=False)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        speech = resample_poly(speech, SAMPLE_RATE, sr).astype(np.float32)
    # Keep the fixture bounded while retaining several seconds of speech.
    speech = speech[: SAMPLE_RATE * 8].astype(np.float32)
    frames = int(np.ceil(len(speech) / HOP))
    padded = np.zeros(frames * HOP, dtype=np.float32)
    padded[: len(speech)] = speech
    speech = np.clip(padded, -1.0, 1.0)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_logits = wrapped(torch.from_numpy(speech[None, None])).numpy()
    onnx_logits = sess.run(None, {"wav": speech[None, None]})[0]
    abs_err = np.abs(torch_logits - onnx_logits)
    torch_buckets = torch_logits.argmax(-1)
    onnx_buckets = onnx_logits.argmax(-1)
    mismatch = torch_buckets != onnx_buckets

    # Derive receptive field from the actual pinned encoder config, including
    # residual branches (the long path bounds the residual sum).
    cfg = model.config.encoder
    jump, receptive = 1, cfg.kernel_size
    for ratio in reversed(cfg.ratios):
        for layer in range(cfg.n_residual_layers):
            receptive += (cfg.residual_kernel_size - 1) * cfg.dilation_base**layer * jump
        receptive += (2 * ratio - 1) * jump
        jump *= ratio
    receptive += (cfg.last_kernel_size - 1) * jump
    history = int(np.ceil((receptive - HOP) / HOP) * HOP)
    assert (jump, receptive, history) == (512, 5376, 5120)
    cases = {"speech": speech, "short": speech[:173], "silence": np.zeros(HOP*17+91, np.float32),
             "random": np.random.default_rng(42).normal(0,.05,HOP*37+123).astype(np.float32)}
    reports, browser_cases = {}, []
    def compare(a,b):
        error = np.abs(a-b)
        tol = 1e-4 + 1e-4 * np.abs(a)
        buckets_a, buckets_b = a.argmax(-1), b.argmax(-1)
        margins = np.sort(a,axis=-1)[...,-1] - np.sort(a,axis=-1)[...,-2]
        delta = {}
        for i,name in enumerate(model.trainable_names_):
            lo, hi = model.config.param_spec[name][:2]
            delta[name] = float((np.abs(buckets_a[...,i]-buckets_b[...,i]) * (hi-lo)/31).max())
        result = {"maxAbsLogitError": float(error.max()), "meanAbsLogitError": float(error.mean()),
            "p99AbsLogitError":float(np.quantile(error,.99)), "toleranceFailures":int((error>tol).sum()),
            "bucketMismatches":int((buckets_a!=buckets_b).sum()), "bucketCount":int(buckets_a.size),
            "minWinningMargin":float(margins.min()), "physicalMaxDifference":delta}
        assert not result["toleranceFailures"], result
        assert not result["bucketMismatches"], result
        return result
    for name, valid in cases.items():
        pcm=np.pad(valid,(0,(-len(valid))%HOP)).astype(np.float32)
        with torch.no_grad(): reference=wrapped(torch.from_numpy(pcm[None,None])).numpy()
        whole=sess.run(None,{"wav":pcm[None,None]})[0]
        report={"torchOnnx":compare(reference,whole), "stream":{}}
        for chunk_frames in [4,8,16]:
            pieces=[]
            for frame in range(0,len(pcm)//HOP,chunk_frames):
                count=min(chunk_frames,len(pcm)//HOP-frame)
                window=pcm[max(0,frame*HOP-history):(frame+count)*HOP]
                pieces.append(sess.run(None,{"wav":window[None,None]})[0][:,-count:])
            report["stream"][str(chunk_frames)]=compare(whole,np.concatenate(pieces,axis=1))
        reports[name]=report
        pcm.astype('<f4').tofile(fixture_dir/f'{name}.f32')
        reference.astype('<f4').tofile(fixture_dir/f'{name}.logits.f32')
        buckets=reference.argmax(-1)
        physical={n: model.bucket_centers[i].detach().cpu().numpy()[buckets[...,i]].tolist() for i,n in enumerate(model.trainable_names_)}
        browser_cases.append({"name":name,"validSamples":len(valid),"sampleCount":len(pcm),"frameCount":len(pcm)//HOP,
            "pcmFile":f'{name}.f32',"logitsFile":f'{name}.logits.f32',"buckets":buckets.tolist(),"params":physical})
    (fixture_dir/'cases.json').write_text(json.dumps(browser_cases))

    # Bucket centers and the browser-facing parameter metadata.
    param_spec = model.config.param_spec
    trainable_names = model.trainable_names_
    centers = {
        name: model.bucket_centers[trainable_names.index(name)].detach().cpu().tolist()
        for name in trainable_names
    }
    metadata = {
        "schemaVersion": 1,
        "modelRevision": CHECKPOINT_REVISION,
        "upstreamCommit": UPSTREAM_COMMIT,
        "checkpointSha256": sha256(args.checkpoint),
        "configSha256": sha256(args.config),
        "onnxSha256": sha256(onnx_path),
        "onnxFile": onnx_path.name,
        "sampleRate": SAMPLE_RATE,
        "controlHop": HOP,
        "controlFrameRate": SAMPLE_RATE / HOP,
        "lookaheadFrames": model.config.lookahead_frames,
        "streamingChunkFrames": 4,
        "historySamples": history,
        "receptiveFieldSamples": receptive,
        "nBuckets": model.n_buckets,
        "trainableNames": trainable_names,
        "paramSpec": {
            name: {
                "min": param_spec[name][0],
                "max": param_spec[name][1],
                "bucketCenters": centers[name],
            }
            for name in trainable_names
        },
        "fixedValues": model.config.frozen_values,
        "frequency": {"external": True, "min": 70.0, "max": 500.0},
        "training": {
            "targetRms": 0.05,
            "source": "cfd4ba3b0ea411d40f404d226022bc70f9e06139/config.json",
        },
    }
    metadata["checkpointUrl"] = f"https://huggingface.co/vvolhejn/samuel/resolve/{CHECKPOINT_REVISION}/checkpoints/last.pt"
    metadata["configUrl"] = f"https://huggingface.co/vvolhejn/samuel/resolve/{CHECKPOINT_REVISION}/config.json"
    (out_dir / "config.json").write_text(json.dumps({"model": model.config.model_dump(), "targetRms": .05}, indent=2))
    metadata["browserConfigSha256"] = sha256(out_dir / "config.json")
    (out_dir / "samuel-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Compact browser fixture: float32 PCM plus full-context logits and bucket ids.
    fixture_n = min(len(speech), SAMPLE_RATE * 3)
    fixture_n = int(np.ceil(fixture_n / HOP) * HOP)
    fixture_pcm = speech[:fixture_n]
    fixture_logits = torch_logits[:, : fixture_n // HOP]
    fixture_buckets = torch_buckets[:, : fixture_n // HOP]
    fixture_pcm.astype("<f4").tofile(fixture_dir / "model-input-44k.f32")
    fixture = {
        "schemaVersion": 1,
        "sampleRate": SAMPLE_RATE,
        "hop": HOP,
        "sampleCount": int(fixture_n),
        "frameCount": int(fixture_n // HOP),
        "pcmFile": "model-input-44k.f32",
        "logits": fixture_logits.astype(float).tolist(),
        "buckets": fixture_buckets.astype(int).tolist(),
        "source": "upstream examples/this-is-samuel.wav, resampled to 44.1 kHz",
    }
    (fixture_dir / "expected.json").write_text(json.dumps(fixture, separators=(",", ":")))

    report = {
        "checkpointSha256": metadata["checkpointSha256"],
        "configSha256": metadata["configSha256"],
        "onnxSha256": metadata["onnxSha256"],
        "weightNormModulesFolded": folded,
        "weightNormMaxDiff": fold_diff,
        "cases": reports,
        "receptiveField": {"samples": receptive, "history": history, "hop": jump},
        "environment": {"torch":torch.__version__,"onnxruntime":ort.__version__},
        "fixtureFrames": int(fixture_n // HOP),
    }
    (ROOT / "docs" / "model-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    # Generate pitch reference from the original causal functions without
    # importing its unrelated dataset/CLI training dependencies.
    import ast
    from scipy.special import betainc
    from numpy.lib.stride_tricks import sliding_window_view
    tree = ast.parse((ROOT / "vendor/samuel/scripts/precompute_pitch_yin.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.Assign) or
             isinstance(node, ast.FunctionDef) and node.name in ("cmndf_frames", "filtered_track")]
    env = {"np":np,"betainc":betainc,"sliding_window_view":sliding_window_view}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"upstream_yin","exec"),env)
    cmndf, lo, hi = env["cmndf_frames"](speech,SAMPLE_RATE,HOP,70,500)
    raw_f0, depth = env["filtered_track"](cmndf,lo,hi,SAMPLE_RATE,70,500,25)
    f0, voiced, last = [], [], 285.
    for f in range(len(speech)//HOP):
        k=f-3
        v=k>=0 and depth[k]<.2 and 70<=raw_f0[k]<=500
        if v: last=float(raw_f0[k])
        f0.append(last);voiced.append(bool(v))
    (fixture_dir/'pitch-reference.json').write_text(json.dumps({"f0":f0,"voiced":voiced,"source":"speech.f32"}))
    provenance={"source":f"https://raw.githubusercontent.com/vvolhejn/samuel/{UPSTREAM_COMMIT}/examples/this-is-samuel.wav",
                "originalSha256":sha256(fixture_dir/'samuel-sample-48k.wav'),"license":"GPL-3.0 (upstream repository)",
                "files":{p.name:sha256(p) for p in sorted(fixture_dir.iterdir()) if p.is_file() and p.name!='provenance.json'}}
    (fixture_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
