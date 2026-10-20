# Recognition protocol

Landmarks are produced on the client (MediaPipe in the browser, or `ApertureTracker`
in Python) and only numbers are sent to the server, so video never leaves the device.
All coordinates are MediaPipe-normalised (0..1, origin top-left). Mirror the camera
image before tracking if you want a selfie view, exactly like `hand_tracker.py` does -
and do the same when recording training data.

## Frame payload

```json
{
  "left":  [[x, y, z], ... 21 items],
  "right": [[x, y, z], ... 21 items],
  "pose":  [[x, y, z, visibility], ... 33 items],
  "face_metrics": [mouth_open, brow_left, brow_right]
}
```
Every key is optional (`null` / missing = not detected). `pose` may omit visibility.
Malformed frames are rejected with an error, never silently coerced.

## `POST /recognize` - one isolated sign
Body `{"frames": [<frame>, ...]}` (8-300 frames, ~1.5 s). Returns
`{"label", "confidence", "top": [{"label","confidence"}]}`; `label` is `null`
with `reason: "no_hands_detected"` when no hands were visible. `503` if no model.

## `WS /ws/recognize` - live stream
Client -> server: one frame payload per message (send ~30/s), or `{"type":"reset"}`.

Server -> client:
| type    | fields                              | when |
|---------|-------------------------------------|------|
| `ready` | `labels`, `window`                  | right after connect |
| `sign`  | `label`, `confidence`, `frame`      | a sign was *confirmed* (debounced, once per sign) |
| `state` | `label`, `confidence`, `hands`, `buffered`, `energy` | every 5 frames, for UI feedback |
| `error` | `detail`                            | bad JSON / bad frame (connection stays open) |
| `reset` | -                                   | ack of a reset |

## `POST /pipeline` - text to signs
`{"text": "Hello Wanjiku"}` -> `{"speech", "translation", "avatar"}`.
`translation.sign_sequence` is a list of `{gloss, type: lexical|fingerspelling|number, ...}`;
`avatar.timeline` has `start_ms` / `duration_ms` per sign, `clip: null` until animation assets exist.

## Browser data collection (opt-in)
Set `APERTURE_ENABLE_COLLECT=1` on the server. Then:

- `POST /collect` `{"label","signer","consent":true,"frames":[...]}` saves one clip to
  `data/sequences/<label>/<signer>__<time>.npz`. `label` matches `[A-Za-z0-9_-]{1,40}`,
  `signer` matches `[A-Za-z0-9-]{1,40}`. Rejected with 422 if consent is false or (for
  non-`_idle` labels) hands were visible in under 30% of frames.
- `GET /collect/stats` -> `{"labels": {"hello": 12}, "signers": ["amina"]}`.
- `POST /model/reload` loads a newly trained `models/signs.npz` without a restart.
All three return 403 while collection is disabled.

## `POST /stt` - offline dictation (optional)
Raw audio bytes as the request body (`?language=en|sw`, 10 MB max) -> `{"text": "..."}`.
Needs `pip install -r requirements-stt.txt`; otherwise `501`.
