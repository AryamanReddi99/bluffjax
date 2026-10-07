// Replay JAX reference rollouts through whlo and compare state hashes,
// rewards and legal-action masks. Usage: node check.mjs <game> [...]
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { initCompiler, compile } from "../vendor/whlo/src/index.mjs";

const here = new URL(".", import.meta.url);
await initCompiler(await readFile(new URL("../vendor/whlo/pkg/whlo_web_bg.wasm", here)));
const index = JSON.parse(await readFile(new URL("../games/index.json", here), "utf8"));
const hash = (leaves) => {
  const h = createHash("sha1");
  for (const l of leaves) h.update(new Uint8Array(l.buffer, l.byteOffset, l.byteLength));
  return h.digest("hex");
};
let failed = false;
for (const name of process.argv.slice(2)) {
  const meta = index[name];
  const ref = JSON.parse(await readFile(new URL(`refs/${name}.json`, here), "utf8"));
  const load = async (k) => compile(await readFile(new URL(`../games/${name}/${k}.mlir`, here), "utf8"));
  const [reset, step, avail] = [await load("reset"), await load("step"), await load("avail")];
  const n = meta.leaves.length;
  const run = (exe, args) => {
    const out = exe.run(Object.fromEntries(exe.inputs.map((s, i) => [s.name, args[i]])));
    return exe.outputs.map((o) => out[o.name]);
  };
  let ep = 0;
  let leaves = run(reset, [new Uint32Array([1, ep])]);
  let bad = hash(leaves) === ref.hashes[0] ? null : "state at reset";
  for (let i = 0; i < ref.actions.length && !bad; i++) {
    const av = Array.from(run(avail, leaves)[0]);
    if (av.join() !== ref.avails[i].join()) { bad = `avail at step ${i}`; break; }
    const a = ref.actions[i];
    const out = run(step, [new Uint32Array([0, i]), ...leaves, Int32Array.from([a].flat())]);
    leaves = out.slice(0, n);
    const r = Array.from(out[n]);
    if (r.some((v, j) => Math.abs(v - ref.rewards[i][j]) > 1e-4)) { bad = `reward at step ${i}: ${r} vs ${ref.rewards[i]}`; break; }
    if (out[n + 1][0] !== ref.dones[i]) { bad = `done at step ${i}`; break; }
    if (out[n + 1][0]) leaves = run(reset, [new Uint32Array([1, ++ep])]);
    if (hash(leaves) !== ref.hashes[i + 1]) bad = `state after step ${i}`;
  }
  failed ||= !!bad;
  console.log(`${bad ? "FAIL" : "PASS"} ${name}: ${ref.actions.length} steps, ${ep} episodes${bad ? ", first mismatch: " + bad : ""}`);
}
process.exit(failed ? 1 : 0);
