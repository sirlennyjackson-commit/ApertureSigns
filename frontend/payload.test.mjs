import assert from "node:assert/strict";
import test from "node:test";
import { activeItem, backoff, buildFrame } from "./payload.js";

const hand = (x) => Array.from({ length: 21 }, (_, i) => ({ x: x + i / 1000, y: 0.5, z: 0 }));
const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.9 }));

test("maps handedness to left/right and rounds", () => {
  const f = buildFrame(
    { landmarks: [hand(0.2), hand(0.7)], handedness: [[{ categoryName: "Left", score: 0.9 }], [{ categoryName: "Right", score: 0.8 }]] },
    { landmarks: [pose] },
  );
  assert.equal(f.left.length, 21);
  assert.equal(f.right.length, 21);
  assert.equal(f.left[0][0], 0.2);
  assert.equal(f.pose.length, 33);
  assert.equal(f.pose[0][3], 0.9);
});

test("supports the older 'handednesses' field and keeps the best of duplicate labels", () => {
  const f = buildFrame(
    { landmarks: [hand(0.1), hand(0.9)], handednesses: [[{ categoryName: "Right", score: 0.4 }], [{ categoryName: "Right", score: 0.95 }]] },
    null,
  );
  assert.equal(f.right[0][0], 0.9);
  assert.equal(f.left, null);
  assert.equal(f.pose, null);
});

test("ignores malformed detections", () => {
  const f = buildFrame({ landmarks: [hand(0.2).slice(0, 5)], handedness: [[{ categoryName: "Left", score: 1 }]] }, { landmarks: [[]] });
  assert.deepEqual(f, { left: null, right: null, pose: null });
  assert.deepEqual(buildFrame(undefined, undefined), { left: null, right: null, pose: null });
});

test("activeItem finds the playing sign", () => {
  const tl = [{ start_ms: 0, duration_ms: 900 }, { start_ms: 1020, duration_ms: 900 }];
  assert.equal(activeItem(tl, 100), 0);
  assert.equal(activeItem(tl, 950), -1);
  assert.equal(activeItem(tl, 1100), 1);
  assert.equal(activeItem(tl, 5000), -1);
});

test("backoff grows and is capped", () => {
  assert.deepEqual([0, 1, 2, 10].map(backoff), [500, 1000, 2000, 10000]);
});
