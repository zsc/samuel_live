import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const pinkSrc = join(root, "vendor", "Pink-Trombone");
const pinkOut = join(root, "public", "pink-trombone");
const ortSrc = join(root, "node_modules", "onnxruntime-web", "dist");
const ortOut = join(root, "public", "ort");

if (!existsSync(join(pinkSrc, "node_modules", ".bin", "rollup"))) {
  execFileSync("pnpm", ["--dir", pinkSrc, "install", "--frozen-lockfile"], {
    stdio: "inherit",
  });
}
execFileSync("pnpm", ["--dir", pinkSrc, "run", "build"], { stdio: "inherit" });

rmSync(pinkOut, { recursive: true, force: true });
mkdirSync(pinkOut, { recursive: true });
cpSync(join(pinkSrc, "pink-trombone.min.js"), join(pinkOut, "pink-trombone.min.js"));
cpSync(
  join(pinkSrc, "pink-trombone-worklet-processor.min.js"),
  join(pinkOut, "pink-trombone-worklet-processor.min.js"),
);

const bundlePath = join(pinkOut, "pink-trombone.min.js");
let bundle = readFileSync(bundlePath, "utf8");
const oldWorklet = '"./script/audio/nodes/pinkTrombone/processors/WorkletProcessor.js"';
if (!bundle.includes(oldWorklet)) throw new Error("Pink Trombone worklet URL changed; patch it in prepare-assets.mjs");
bundle = bundle.replace(oldWorklet, "window.__PINK_TROMBONE_WORKLET_URL__");
writeFileSync(bundlePath, bundle);

rmSync(ortOut, { recursive: true, force: true });
mkdirSync(ortOut, { recursive: true });
for (const name of [
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]) {
  const src = join(ortSrc, name);
  if (existsSync(src)) cpSync(src, join(ortOut, name));
}

console.log("Prepared Pink Trombone and ONNX Runtime static assets.");
