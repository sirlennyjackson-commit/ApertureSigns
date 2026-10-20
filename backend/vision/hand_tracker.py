"""Desktop HUD: live camera, skeleton, and recognised signs.

    python -m backend.vision.hand_tracker [--camera 0] [--model models/signs.npz]

Keys:  q quit   r clear transcript / reset recogniser
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.config.settings import get_settings
from backend.vision.drawing import (
    BORDER,
    MUTED,
    draw_preview,
    draw_skeleton,
    fit_into,
    panel,
    put_text,
)

W, H = 1440, 900
BG = (8, 13, 22)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


# One source of truth for layout: the same Rects are drawn AND hit-tested
# (previously the EXPAND button was drawn at y=578..612 but clicked at y=590..625).
RECOGNITION = Rect(30, 105, 300, 190)
SKELETON = Rect(355, 105, 1050, 590)
TRANSCRIPT = Rect(355, 710, 1050, 100)
CAMERA_SMALL = Rect(30, 570, 300, 250)
CAMERA_LARGE = Rect(30, 80, 690, 460)
EXPAND_BTN = Rect(CAMERA_SMALL.x + 205, CAMERA_SMALL.y + 8, 80, 34)
MINIMIZE_BTN = Rect(CAMERA_LARGE.x + CAMERA_LARGE.w - 110, CAMERA_LARGE.y + 6, 100, 30)


@dataclass
class Hud:
    frame: dict
    face_mesh: np.ndarray | None
    preview: np.ndarray
    fps: float
    expanded: bool = False
    state: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)
    model_name: str = "NO MODEL"
    window: int = 45


def make_background() -> np.ndarray:
    canvas = np.full((H, W, 3), BG, np.uint8)
    for x in range(0, W, 60):
        cv2.line(canvas, (x, 0), (x, H), (14, 24, 36), 1)
    for y in range(0, H, 60):
        cv2.line(canvas, (0, y), (W, y), (14, 24, 36), 1)
    put_text(canvas, "APERTURE", (35, 50), 0.85, (220, 230, 240), 2)
    put_text(canvas, "SIGNS", (35, 80), 0.5, (120, 190, 235))
    put_text(canvas, "LIVE HUMAN MOTION INTELLIGENCE", (200, 62), 0.45, MUTED)
    return canvas


def _button(canvas, r: Rect, label: str):
    cv2.rectangle(canvas, (r.x, r.y), (r.x + r.w, r.y + r.h), (25, 45, 60), -1)
    cv2.rectangle(canvas, (r.x, r.y), (r.x + r.w, r.y + r.h), (90, 170, 215), 1)
    put_text(canvas, label, (r.x + 10, r.y + r.h // 2 + 5), 0.4, (180, 220, 240))


def render(background: np.ndarray, hud: Hud) -> np.ndarray:
    canvas = background.copy()
    st = hud.state
    label = st.get("label") or "--"
    conf = float(st.get("confidence") or 0.0)

    # Skeleton
    r = SKELETON
    panel(canvas, r.x, r.y, r.w, r.h, "LIVE SKELETAL REPRESENTATION")
    sk = np.full((r.h - 50, r.w - 40, 3), (10, 17, 27), np.uint8)
    draw_skeleton(sk, hud.frame, hud.face_mesh)
    canvas[r.y + 40:r.y + 40 + sk.shape[0], r.x + 20:r.x + 20 + sk.shape[1]] = sk

    # Recognition
    r = RECOGNITION
    panel(canvas, r.x, r.y, r.w, r.h, "SIGN RECOGNITION")
    put_text(canvas, label.upper()[:10], (r.x + 15, r.y + 85), 1.2, (130, 220, 255), 2)
    cv2.rectangle(canvas, (r.x + 15, r.y + 105), (r.x + 285, r.y + 117), BORDER, 1)
    cv2.rectangle(canvas, (r.x + 15, r.y + 105), (r.x + 15 + int(270 * conf), r.y + 117), (130, 220, 255), -1)
    put_text(canvas, f"CONFIDENCE {conf * 100:.0f}%", (r.x + 15, r.y + 138), 0.4, MUTED)
    put_text(canvas, f"BUFFER {st.get('buffered', 0)} / {hud.window}", (r.x + 15, r.y + 163), 0.45)
    put_text(canvas, hud.model_name, (r.x + 15, r.y + 182), 0.35, MUTED)

    # Transcript
    r = TRANSCRIPT
    panel(canvas, r.x, r.y, r.w, r.h, "TRANSCRIPT")
    words = " ".join(hud.transcript[-8:]) or "(waiting for a confirmed sign)"
    put_text(canvas, words, (r.x + 20, r.y + 65), 0.9, (120, 220, 255), 2)
    put_text(canvas, f"MOTION {st.get('energy', 0):.4f}", (r.x + r.w - 160, r.y + 25), 0.4, MUTED)

    # Camera card
    cam = CAMERA_LARGE if hud.expanded else CAMERA_SMALL
    panel(canvas, cam.x, cam.y, cam.w, cam.h, "LIVE CAMERA")
    fit_into(canvas, hud.preview, cam.x + 10, cam.y + 38, cam.w - 20, cam.h - 48)
    _button(canvas, MINIMIZE_BTN if hud.expanded else EXPAND_BTN, "MINIMIZE" if hud.expanded else "EXPAND")

    f = hud.frame
    telemetry = (f"FACE {'OK' if f.get('face') is not None else '--'}   "
                 f"L-HAND {'OK' if f.get('left') is not None else '--'}   "
                 f"R-HAND {'OK' if f.get('right') is not None else '--'}   "
                 f"POSE {'OK' if f.get('pose') is not None else '--'}   {hud.fps:04.1f} FPS")
    put_text(canvas, telemetry, (355, 850), 0.45, MUTED)
    return canvas


def main(argv=None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=str(settings.model_path))
    args = parser.parse_args(argv)

    from pathlib import Path

    from backend.ml.classifier import SignClassifier
    from backend.ml.realtime import RealtimeRecognizer
    from backend.vision.tracker import ApertureTracker

    recognizer, model_name = None, "NO MODEL (see README)"
    if Path(args.model).exists():
        clf = SignClassifier.load(args.model)
        recognizer = RealtimeRecognizer(
            clf, window=settings.window_frames, stride=settings.stride,
            min_confidence=settings.min_confidence, min_votes=settings.min_votes,
            min_activity=settings.min_hand_activity)
        model_name = f"MODEL {len(clf.labels)} LABELS"

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        print("ERROR: could not open camera", args.camera)
        return 1

    expanded = False
    transcript: list[str] = []

    def on_mouse(event, x, y, *_):
        nonlocal expanded
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not expanded and EXPAND_BTN.contains(x, y):
            expanded = True
        elif expanded and MINIMIZE_BTN.contains(x, y):
            expanded = False

    cv2.namedWindow("Aperture Signs", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Aperture Signs", on_mouse)
    background = make_background()
    fps, last = 0.0, time.perf_counter()
    state: dict = {}

    with ApertureTracker() as tracker:
        while True:
            ok, bgr = camera.read()
            if not ok:
                print("ERROR: could not read a camera frame")
                break
            bgr = cv2.flip(bgr, 1)
            tracked = tracker.process(bgr)

            if recognizer:
                event = recognizer.push(tracked.frame)
                state = recognizer.state()
                if event:
                    transcript.append(event.label)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)  # smoothed
            last = now

            hud = Hud(tracked.frame, tracked.face_mesh, draw_preview(bgr, tracked.results), fps,
                      expanded, state, transcript, model_name, settings.window_frames)
            cv2.imshow("Aperture Signs", render(background, hud))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                transcript.clear()
                if recognizer:
                    recognizer.reset()

    camera.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
