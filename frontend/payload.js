// Pure helpers (no DOM, no network) so they can be unit-tested with `node --test`.

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
export const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
];

const round = (v) => Math.round(v * 1e5) / 1e5;

/**
 * Convert MediaPipe Tasks results into the server's frame payload
 * (see docs/protocol.md). The image must already be mirrored (selfie view)
 * before detection so "Left"/"Right" match the Python tracker.
 */
export function buildFrame(hands, pose) {
  const frame = { left: null, right: null, pose: null };

  const detected = (hands?.landmarks ?? []).map((lm, i) => {
    const cat = (hands.handedness ?? hands.handednesses ?? [])[i]?.[0];
    return { lm, side: cat?.categoryName?.toLowerCase(), score: cat?.score ?? 0 };
  });
  detected.sort((a, b) => b.score - a.score); // best detection wins if two share a label
  for (const { lm, side } of detected) {
    if ((side === "left" || side === "right") && lm.length === 21 && frame[side] === null) {
      frame[side] = lm.map((p) => [round(p.x), round(p.y), round(p.z ?? 0)]);
    }
  }

  const body = pose?.landmarks?.[0];
  if (body && body.length === 33) {
    frame.pose = body.map((p) => [round(p.x), round(p.y), round(p.z ?? 0), round(p.visibility ?? 1)]);
  }
  return frame;
}

/** Index of the timeline item playing at `ms`, or -1. */
export function activeItem(timeline, ms) {
  return timeline.findIndex((t) => ms >= t.start_ms && ms < t.start_ms + t.duration_ms);
}

/** Reconnect delay: 0.5s, 1s, 2s ... capped at 10s. */
export function backoff(attempt) {
  return Math.min(10000, 500 * 2 ** attempt);
}
