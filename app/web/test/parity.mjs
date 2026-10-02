// node app/web/test/parity.mjs tests/fixtures/js_parity.json: prints a JSON report, exit 1 on mismatch
import { readFileSync } from "node:fs";
import { featureCount, preprocess } from "../js/preprocess.js";

const TOL = 1e-4;
const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const report = [];
let ok = true;
for (const c of cases) {
  const data = Float64Array.from(c.seq, (v) => (v === null ? NaN : v));
  const { features, mask } = preprocess({ T: c.shape[0], data }, c.cfg, c.y_scale ?? 1.0);
  let maxDiff = 0;
  for (let i = 0; i < features.length; i++) {
    const ref = c.features[i];
    maxDiff = Math.max(maxDiff, Math.abs(features[i] - ref) / Math.max(1, Math.abs(ref)));
  }
  const maskOk = c.mask.every((m, i) => m === mask[i]);
  const sizeOk = features.length === c.features.length && featureCount(c.cfg) * c.cfg.T === features.length;
  const pass = maxDiff < TOL && maskOk && sizeOk;
  ok &&= pass;
  report.push({ name: c.name, maxDiff, maskOk, sizeOk, pass });
}
console.log(JSON.stringify(report));
process.exit(ok ? 0 : 1);
