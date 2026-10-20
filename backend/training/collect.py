"""Record labelled sign clips with your webcam.

    python -m backend.training.collect --label hello --samples 20 --signer amina
    python -m backend.training.collect --label _idle --samples 20 --signer amina

Press SPACE to start each clip (3-2-1 countdown, then it records). Press q to stop.
Record an ``_idle`` label too (hands resting / out of frame) so the model learns
what *not signing* looks like. Use several signers: accuracy for a new signer is
the number that matters.

Only landmarks (numbers) are saved, never video. Still, get consent from every
signer before recording and storing their data.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from backend.config.settings import IDLE_LABELS, get_settings
from backend.ml.landmarks import frames_to_sequence, hand_activity, save_sequence
from backend.vision.drawing import draw_preview, put_text


def main(argv=None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="sign name, e.g. hello (use _idle for resting)")
    p.add_argument("--signer", default="signer1", help="id of the person signing (no spaces)")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--seconds", type=float, default=1.5,
                   help="clip length; keep seconds x fps close to APERTURE_WINDOW_FRAMES")
    p.add_argument("--countdown", type=float, default=3.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", type=Path, default=settings.data_dir / "sequences")
    args = p.parse_args(argv)

    from backend.vision.tracker import ApertureTracker

    label_dir = args.out / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        print("ERROR: could not open camera", args.camera)
        return 1

    saved, phase, started, frames, message = 0, "wait", 0.0, [], "Press SPACE to record"
    needs_hands = args.label not in IDLE_LABELS
    with ApertureTracker() as tracker:
        while saved < args.samples:
            ok, bgr = camera.read()
            if not ok:
                break
            bgr = cv2.flip(bgr, 1)
            tracked = tracker.process(bgr)
            now = time.perf_counter()

            if phase == "countdown" and now - started >= args.countdown:
                phase, started, frames = "record", now, []
            elif phase == "record":
                frames.append(tracked.frame)
                if now - started >= args.seconds:
                    seq = frames_to_sequence(frames)
                    if needs_hands and hand_activity(seq) < settings.min_hand_activity:
                        message = "Hands not visible enough - clip discarded, try again"
                    else:
                        name = f"{args.signer}__{datetime.now():%Y%m%d-%H%M%S-%f}.npz"
                        save_sequence(label_dir / name, seq)
                        saved += 1
                        message = f"Saved {saved}/{args.samples} ({len(frames)} frames)"
                    phase = "wait"

            view = draw_preview(bgr, tracked.results)
            if phase == "countdown":
                banner = f"{max(0.0, args.countdown - (now - started)):.1f}"
            elif phase == "record":
                banner = "REC"
            else:
                banner = "SPACE = record"
            title = f"{args.label}  [{args.signer}]  {saved}/{args.samples}"
            put_text(view, title, (12, 28), 0.7, (255, 255, 255), 2)
            put_text(view, banner, (12, 64), 1.0, (0, 200, 255) if phase != "record" else (60, 60, 255), 2)
            put_text(view, message, (12, view.shape[0] - 14), 0.55, (255, 255, 255), 1)
            cv2.imshow("Aperture Signs - collect", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and phase == "wait":
                phase, started = "countdown", now

    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} clips to {label_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
