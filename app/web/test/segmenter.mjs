// node app/web/test/segmenter.mjs: segmenter tests on synthetic hand trajectories, exit 1 on failure.
import assert from "node:assert/strict";
import { N_LANDMARKS, POSE_POS } from "../js/landmarks.js";
import { Segmenter, motionEnergy, toSequence } from "../js/segmenter.js";

const L = N_LANDMARKS;

function frame({ hands, t = 0, speed = 0.02 }) {
  const f = new Float64Array(L * 3).fill(NaN);
  f.set([0.65, 0.6, 0], POSE_POS.l_shoulder * 3);
  f.set([0.35, 0.6, 0], POSE_POS.r_shoulder * 3);
  if (hands) for (let l = 21; l < 42; l++) f.set([0.5 + speed * Math.sin(t / 3) + l * 1e-3, 0.5, 0], l * 3);
  return f;
}

function run(frames, cfg) {
  const s = new Segmenter(cfg);
  const out = [];
  for (const f of frames) {
    const r = s.push(f);
    if (r.segment) out.push(r);
  }
  return out;
}

// 1. idle, sign (30 moving frames), hands down: exactly one segment, ended by hands-down
{
  const frames = [
    ...Array.from({ length: 10 }, () => frame({ hands: false })),
    ...Array.from({ length: 30 }, (_, t) => frame({ hands: true, t })),
    ...Array.from({ length: 10 }, () => frame({ hands: false })),
  ];
  const segs = run(frames);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].reason, "hands-down");
  assert.ok(segs[0].segment.length >= 30 && segs[0].segment.length <= 40);
}

// 2. two signs separated by a pause with hands down: two segments
{
  const sign = Array.from({ length: 20 }, (_, t) => frame({ hands: true, t }));
  const rest = Array.from({ length: 8 }, () => frame({ hands: false }));
  assert.equal(run([...rest, ...sign, ...rest, ...sign, ...rest]).length, 2);
}

// 3. hands kept still after the sign: "still" end, then no re-trigger while still
{
  const frames = [
    ...Array.from({ length: 20 }, (_, t) => frame({ hands: true, t })),
    ...Array.from({ length: 60 }, () => frame({ hands: true, speed: 0 })),
  ];
  const segs = run(frames);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].reason, "still");
}

// 4. a brief hand flash shorter than minFrames is ignored
{
  const frames = [
    ...Array.from({ length: 4 }, (_, t) => frame({ hands: true, t })),
    ...Array.from({ length: 10 }, () => frame({ hands: false })),
  ];
  assert.equal(run(frames, { endNoHandFrames: 3 }).length, 0);
}

// 5. very long signing is cut at maxFrames
{
  const frames = Array.from({ length: 300 }, (_, t) => frame({ hands: true, t }));
  const segs = run(frames, { maxFrames: 100 });
  assert.ok(segs.length >= 2 && segs.every((s) => s.segment.length <= 100));
}

assert.equal(motionEnergy(null, frame({ hands: true })), 0);
assert.equal(toSequence([frame({ hands: true }), frame({ hands: false })]).T, 2);
console.log("segmenter: all tests passed");
