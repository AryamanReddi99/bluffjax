// WebGPU execution of a whlo GPU plan — the browser counterpart of the
// native wgpu runner (crates/whlo-webgpu). One storage buffer holds the
// arena; each shader binds it at @group(0) @binding(0); the step program
// runs on the host, reading predicate scalars back between submissions.

import { DTYPES } from "./dtypes.mjs";

/** Pad a byte array to WebGPU's 4-byte write alignment (arena blocks are
 *  16-aligned with rounded-up sizes, so the padding stays in-block). */
function padded(bytes) {
  const n = Math.ceil(bytes.byteLength / 4) * 4;
  if (n === bytes.byteLength) return bytes;
  const out = new Uint8Array(n);
  out.set(bytes);
  return out;
}

export class WebGpuExecutable {
  /** @param {object} plan   parsed plan JSON from compile_webgpu
   *  @param {Uint8Array} constBytes  concatenated constant payloads
   *  @param {GPU} [gpu]  WebGPU entry point; defaults to navigator.gpu */
  static async create(plan, constBytes, gpu) {
    gpu ??= typeof navigator !== "undefined" ? navigator.gpu : undefined;
    if (!gpu) {
      throw new Error("WebGPU is not available (navigator.gpu missing)");
    }
    const adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("no WebGPU adapter available");
    const device = await adapter.requestDevice();

    const exe = new WebGpuExecutable();
    exe._device = device;
    exe._plan = plan;
    exe._manifest = plan.manifest;

    exe._arena = device.createBuffer({
      label: "whlo-arena",
      size: plan.arena_bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    exe._staging = device.createBuffer({
      label: "whlo-readback",
      size: plan.arena_bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // Bounce buffer: WebGPU forbids same-buffer copies.
    exe._scratch = device.createBuffer({
      label: "whlo-copy-scratch",
      size: plan.arena_bytes,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const bgl = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });
    exe._bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: exe._arena } }],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

    // Constants (model weights included) live in never-recycled arena
    // blocks; upload once, like the wasm backend's init_consts.
    for (const c of plan.consts) {
      device.queue.writeBuffer(
        exe._arena,
        c.offset,
        padded(constBytes.subarray(c.start, c.start + c.len))
      );
    }

    exe._pipelines = await Promise.all(
      plan.shaders.map((s) =>
        device.createComputePipelineAsync({
          label: s.name,
          layout,
          compute: {
            module: device.createShaderModule({ label: s.name, code: s.wgsl }),
            entryPoint: "main",
          },
        })
      )
    );
    return exe;
  }

  get inputs() {
    return this._manifest.inputs.map(({ name, dtype, shape }) => ({ name, dtype, shape }));
  }

  get outputs() {
    return this._manifest.outputs.map(({ name, dtype, shape }) => ({ name, dtype, shape }));
  }

  /** Run with named TypedArray inputs; resolves to named output TypedArrays. */
  async run(inputs = {}) {
    const dev = this._device;
    // Upload inputs (constants were uploaded once at creation).
    for (const slot of this._manifest.inputs) {
      const data = inputs[slot.name];
      if (data === undefined) throw new Error(`missing input '${slot.name}'`);
      const TA = DTYPES[slot.dtype];
      if (!TA) throw new Error(`unsupported dtype ${slot.dtype}`);
      const n = slot.byte_size / TA.BYTES_PER_ELEMENT;
      if (data.length !== n) {
        throw new Error(
          `input '${slot.name}' has ${data.length} elements, expected ${n} (shape ${JSON.stringify(slot.shape)})`
        );
      }
      const bytes = new Uint8Array(data.buffer, data.byteOffset, slot.byte_size);
      dev.queue.writeBuffer(this._arena, slot.offset, padded(bytes));
    }

    this._enc = null;
    await this._execSteps(this._plan.steps);

    // Read the arena back and slice out the outputs.
    this._encoder().copyBufferToBuffer(this._arena, 0, this._staging, 0, this._plan.arena_bytes);
    this._flush();
    await this._staging.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(this._staging.getMappedRange());
    const outputs = {};
    for (const slot of this._manifest.outputs) {
      const TA = DTYPES[slot.dtype];
      // slice() copies out of the mapped range; per-output copies only.
      const bytes = mapped.slice(slot.offset, slot.offset + slot.byte_size);
      outputs[slot.name] = new TA(bytes.buffer, 0, slot.byte_size / TA.BYTES_PER_ELEMENT);
    }
    this._staging.unmap();
    return outputs;
  }

  /** Pending encoder, created on first use; flush points null it out. */
  _encoder() {
    return (this._enc ??= this._device.createCommandEncoder({ label: "whlo-run" }));
  }

  /** Finish and submit the pending encoder (if any). */
  _flush() {
    if (this._enc) {
      this._device.queue.submit([this._enc.finish()]);
      this._enc = null;
    }
  }

  _dispatch(enc, d) {
    const pass = enc.beginComputePass();
    pass.setPipeline(this._pipelines[d.shader]);
    pass.setBindGroup(0, this._bindGroup);
    pass.dispatchWorkgroups(d.workgroups[0], d.workgroups[1], d.workgroups[2]);
    pass.end();
  }

  /** (dst, src, len) arena copies, bounced through scratch. */
  _copies(enc, copies) {
    for (const [dst, src, len] of copies) {
      const n = Math.ceil(len / 4) * 4;
      enc.copyBufferToBuffer(this._arena, src, this._scratch, src, n);
      enc.copyBufferToBuffer(this._scratch, src, this._arena, dst, n);
    }
  }

  /** Flush pending work and read the u32 word at `offset`. */
  async _readWord(offset) {
    this._encoder().copyBufferToBuffer(this._arena, offset, this._staging, 0, 4);
    this._flush();
    await this._staging.mapAsync(GPUMapMode.READ, 0, 4);
    const word = new Uint32Array(this._staging.getMappedRange(0, 4))[0];
    this._staging.unmap();
    return word;
  }

  async _execSteps(steps) {
    for (const step of steps) {
      if (step.Dispatch) {
        this._dispatch(this._encoder(), step.Dispatch);
      } else if (step.While) {
        const w = step.While;
        this._copies(this._encoder(), w.init_copies);
        for (;;) {
          await this._execSteps(w.cond);
          const pred = await this._readWord(w.pred_offset);
          if ((pred & 0xff) === 0) break;
          await this._execSteps(w.body);
          const e = this._encoder();
          this._copies(e, w.back_stage);
          this._copies(e, w.back_commit);
        }
      } else if (step.Case) {
        const c = step.Case;
        const word = await this._readWord(c.index_offset);
        const raw = c.index_bytes === 1 ? word & 0xff : word;
        // Out-of-range (incl. negatives, wrapping huge as u32) runs the
        // last branch, per StableHLO.
        const idx = Math.min(raw >>> 0, c.branches.length - 1);
        const [steps_i, copies_i] = c.branches[idx];
        await this._execSteps(steps_i);
        this._copies(this._encoder(), copies_i);
      } else {
        throw new Error(`unknown step ${Object.keys(step)}`);
      }
    }
  }
}
