import { defineConfig } from "vite";
import { cpSync } from "node:fs";
export default defineConfig(({command}) => ({
  base: "./", publicDir: command === "serve" ? "public" : false,
  build: { target: "es2022", sourcemap: true }, worker: { format: "es" },
  plugins: [{name: "static-runtime", closeBundle() {
    for (const name of ["models", "fixtures", "pink-trombone", "ort", "licenses"]) cpSync(`public/${name}`, `dist/${name}`, {recursive:true});
  }}],
}));
