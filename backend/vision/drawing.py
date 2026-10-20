"""OpenCV drawing helpers shared by the HUD and the data-collection tool."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]

LINE = (95, 180, 245)
POINT = (130, 220, 255)
FACE_LINE = (90, 150, 210)
TEXT = (205, 220, 235)
MUTED = (130, 150, 170)
PANEL = (15, 23, 35)
BORDER = (65, 90, 115)
MIN_VISIBILITY = 0.3


@lru_cache(maxsize=1)
def _face_contours() -> tuple:
    import mediapipe as mp

    return tuple(mp.solutions.face_mesh.FACEMESH_CONTOURS)


def put_text(img, text, org, scale=0.45, color=TEXT, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _px(points: np.ndarray, w: int, h: int) -> np.ndarray:
    return (np.clip(points[:, :2], 0, 1) * np.array([w, h])).astype(int)


def _edges(img, px, connections, color, thickness, visible=None):
    for a, b in connections:
        if a >= len(px) or b >= len(px):
            continue
        if visible is not None and not (visible[a] and visible[b]):
            continue
        cv2.line(img, tuple(px[a]), tuple(px[b]), color, thickness, cv2.LINE_AA)


def _dots(img, px, radius, visible=None):
    for i, p in enumerate(px):
        if visible is None or visible[i]:
            cv2.circle(img, tuple(p), radius, POINT, -1, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, frame: dict, face_mesh: np.ndarray | None = None) -> None:
    """Draw face, body and both hands onto ``img`` in place."""
    h, w = img.shape[:2]
    if face_mesh is not None:
        fpx = _px(face_mesh, w, h)
        _edges(img, fpx, _face_contours(), FACE_LINE, 1)
    pose = frame.get("pose")
    if pose is not None:
        ppx, vis = _px(pose, w, h), pose[:, 3] >= MIN_VISIBILITY
        _edges(img, ppx, POSE_CONNECTIONS, LINE, 2, vis)
        _dots(img, ppx[:17], 4, vis[:17])
        if vis[11] and vis[12]:
            neck = (ppx[11] + ppx[12]) // 2
            cv2.line(img, tuple(neck), tuple(ppx[0]), LINE, 3, cv2.LINE_AA)
    for key, wrist_id in (("left", 15), ("right", 16)):
        hand = frame.get(key)
        if hand is None:
            continue
        hpx = _px(hand, w, h)
        if pose is not None and pose[wrist_id, 3] >= MIN_VISIBILITY:
            cv2.line(img, tuple(_px(pose, w, h)[wrist_id]), tuple(hpx[0]), (130, 210, 255), 4, cv2.LINE_AA)
        _edges(img, hpx, HAND_CONNECTIONS, LINE, 2)
        _dots(img, hpx, 4)


def draw_preview(bgr: np.ndarray, results) -> np.ndarray:
    """Camera image with MediaPipe's own overlay (face contours, pose, hands)."""
    import mediapipe as mp

    du, ds, holistic = mp.solutions.drawing_utils, mp.solutions.drawing_styles, mp.solutions.holistic
    out = bgr.copy()
    if results.face_landmarks:
        du.draw_landmarks(out, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                          None, ds.get_default_face_mesh_contours_style())
    if results.pose_landmarks:
        du.draw_landmarks(out, results.pose_landmarks, holistic.POSE_CONNECTIONS,
                          ds.get_default_pose_landmarks_style())
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand:
            du.draw_landmarks(out, hand, holistic.HAND_CONNECTIONS,
                              ds.get_default_hand_landmarks_style(),
                              ds.get_default_hand_connections_style())
    return out


def fit_into(canvas: np.ndarray, img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Letterbox ``img`` into the (x, y, w, h) box without distorting its aspect ratio."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized


def panel(canvas, x, y, w, h, title=None):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)
    if title:
        put_text(canvas, title, (x + 15, y + 25))
