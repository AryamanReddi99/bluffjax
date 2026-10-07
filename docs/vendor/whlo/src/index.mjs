// whlo: run JAX programs (StableHLO) in the browser via WebAssembly.
//
//   import { compile } from "whlo";
//   const exe = await compile(stablehloText);
//   const { result } = exe.run({ x: new Float32Array(...) });
//
// Note for embedders: compiling at runtime requires the page's CSP (if any)
// to allow 'wasm-unsafe-eval'. AOT-compiled .wasm files (via the whlo CLI)
// avoid that requirement.

import init, {
  compile_with as compilerCompile,
  compile_webgpu as compilerCompileWebGpu,
  relaxed_simd_probe,
} from "../pkg/whlo_web.js";
import { WebGpuExecutable } from "./webgpu.mjs";

let initialized = null;
let relaxedSimd = false;

/** One-time init of the compiler module itself. Pass bytes in Node. */
export async function initCompiler(input) {
  if (!initialized) {
    initialized = init(input ? { module_or_path: input } : undefined).then(() => {
      // Feature-detect relaxed SIMD (FMA): Chrome/Firefox yes, Safari not yet.
      try {
        relaxedSimd = WebAssembly.validate(new Uint8Array(relaxed_simd_probe()));
      } catch {
        relaxedSimd = false;
      }
    });
  }
  await initialized;
}

import { DTYPES } from "./dtypes.mjs";

/**
 * Compile StableHLO text and instantiate an executable.
 *
 * @param {string} text - jax.export(...).mlir_module() output
 * @param {object} opts
 * @param {'wasm'|'webgpu'} [opts.backend='wasm'] - 'webgpu' compiles to WGSL
 *   compute shaders and runs on the GPU (requires navigator.gpu, or pass
 *   opts.gpu for a custom provider, e.g. Dawn in Node). Both
 *   backends produce identical results at golden tolerances; prefer
 *   `await exe.run(...)` so the two are interchangeable (wasm run() is
 *   synchronous, webgpu run() is async).
 * @returns {Promise<Executable|WebGpuExecutable>}
 */
export async function compile(text, opts = {}) {
  await initCompiler();
  if (opts.backend === "webgpu") {
    const compiled = compilerCompileWebGpu(text);
    const plan = JSON.parse(compiled.plan_json);
    return WebGpuExecutable.create(plan, compiled.const_bytes, opts.gpu);
  }
  // Relaxed SIMD (fused multiply-add) is auto-detected; it changes float
  // rounding slightly. Pass { relaxedSimd: false } for bit-stable output
  // across engines.
  const useRelaxed = opts.relaxedSimd ?? relaxedSimd;
  const compiled = compilerCompile(text, true, useRelaxed);
  const manifest = JSON.parse(compiled.manifest_json);
  const bytes = compiled.wasm_bytes;
  const { instance } = await WebAssembly.instantiate(bytes, {});
  instance.exports.init_consts();
  return new Executable(instance, manifest);
}

export class Executable {
  constructor(instance, manifest) {
    this._instance = instance;
    this._manifest = manifest;
  }

  /** [{name, dtype, shape}] */
  get inputs() {
    return this._manifest.inputs.map(({ name, dtype, shape }) => ({ name, dtype, shape }));
  }

  get outputs() {
    return this._manifest.outputs.map(({ name, dtype, shape }) => ({ name, dtype, shape }));
  }

  /**
   * Run with named inputs (TypedArrays). Returns named output TypedArrays
   * (copies — safe to hold onto).
   */
  run(inputs = {}) {
    const memory = this._instance.exports.memory;
    for (const slot of this._manifest.inputs) {
      const data = inputs[slot.name];
      if (data === undefined) {
        throw new Error(`missing input '${slot.name}'`);
      }
      const TA = DTYPES[slot.dtype];
      if (!TA) throw new Error(`unsupported dtype ${slot.dtype}`);
      const n = slot.byte_size / TA.BYTES_PER_ELEMENT;
      if (data.length !== n) {
        throw new Error(
          `input '${slot.name}' has ${data.length} elements, expected ${n} (shape ${JSON.stringify(slot.shape)})`
        );
      }
      new TA(memory.buffer, slot.offset, n).set(data);
    }
    this._instance.exports.run();
    const outputs = {};
    for (const slot of this._manifest.outputs) {
      const TA = DTYPES[slot.dtype];
      const n = slot.byte_size / TA.BYTES_PER_ELEMENT;
      outputs[slot.name] = new TA(memory.buffer.slice(slot.offset, slot.offset + slot.byte_size), 0, n);
    }
    return outputs;
  }
}
