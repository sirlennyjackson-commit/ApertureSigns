import os
import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DLL_DIRECTORY_HANDLES = []
if sys.platform == "win32":
    # MediaPipe needs the MSVC runtime DLLs shipped with this project's PyQt6.
    qt_runtime_dir = Path(sys.prefix) / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin"
    if qt_runtime_dir.is_dir():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(qt_runtime_dir)))

try:
    import mediapipe as mp
except (ImportError, OSError) as exc:
    mp = None
    MEDIAPIPE_IMPORT_ERROR = exc
else:
    MEDIAPIPE_IMPORT_ERROR = None

from backend.vision.sign_recognizer import SignRecognizer


class ApertureTracker:
    def __init__(self, apply_smoothing=True):
        if mp is None:
            raise RuntimeError(
                f"MediaPipe is unavailable: {MEDIAPIPE_IMPORT_ERROR}. "
                "Install the project requirements and restart."
            )
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh

        self.history = deque(maxlen=60)
        self.smoothed = {
            "left_hand": [],
            "right_hand": [],
            "pose": [],
            "face": [],
        }
        self.apply_smoothing = apply_smoothing
        self.smoothing_alpha = 0.35

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.30,
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        data = {
            "face": [],
            "pose": [],
            "left_hand": [],
            "right_hand": [],
        }

        if results.face_landmarks:
            data["face"] = [
                {
                    "id": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                }
                for i, lm in enumerate(results.face_landmarks.landmark)
            ]

        if results.pose_landmarks:
            data["pose"] = [
                {
                    "id": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }
                for i, lm in enumerate(results.pose_landmarks.landmark)
            ]

        if results.left_hand_landmarks:
            data["left_hand"] = [
                {
                    "id": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                }
                for i, lm in enumerate(
                    results.left_hand_landmarks.landmark
                )
            ]

        if results.right_hand_landmarks:
            data["right_hand"] = [
                {
                    "id": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                }
                for i, lm in enumerate(
                    results.right_hand_landmarks.landmark
                )
            ]

        if not self.apply_smoothing:
            self.history.append(data)
            return data, results

        # Temporal landmark smoothing.
        # Keeps fast hand movement responsive while reducing jitter.
        for key in self.smoothed:
            current = data[key]
            previous = self.smoothed[key]

            if not current:
                continue

            if not previous or len(previous) != len(current):
                self.smoothed[key] = [dict(p) for p in current]
                continue

            alpha = self.smoothing_alpha
            filtered = []

            for cur, old in zip(current, previous):
                item = dict(cur)

                item["x"] = old["x"] + alpha * (cur["x"] - old["x"])
                item["y"] = old["y"] + alpha * (cur["y"] - old["y"])
                item["z"] = old["z"] + alpha * (cur["z"] - old["z"])

                if "visibility" in cur:
                    item["visibility"] = cur["visibility"]

                filtered.append(item)

            self.smoothed[key] = filtered
            data[key] = filtered

        self.history.append(data)

        return data, results


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def point(points, index, width, height):
    if index >= len(points):
        return None

    x = int(np.clip(points[index]["x"], 0, 1) * width)
    y = int(np.clip(points[index]["y"], 0, 1) * height)

    return x, y


def draw_connections(canvas, points, connections, width, height):
    for a, b in connections:
        p1 = point(points, a, width, height)
        p2 = point(points, b, width, height)

        if p1 and p2:
            cv2.line(
                canvas,
                p1,
                p2,
                (95, 180, 245),
                2,
                cv2.LINE_AA,
            )


def draw_points(canvas, points, width, height, radius=3):
    for i in range(len(points)):
        p = point(points, i, width, height)

        if p:
            cv2.circle(
                canvas,
                p,
                radius,
                (130, 220, 255),
                -1,
                cv2.LINE_AA,
            )


def draw_face(canvas, face, width, height):
    if not face:
        return

    connections = mp.solutions.face_mesh.FACEMESH_CONTOURS

    for a, b in connections:
        p1 = point(face, a, width, height)
        p2 = point(face, b, width, height)

        if p1 and p2:
            cv2.line(
                canvas,
                p1,
                p2,
                (90, 150, 210),
                1,
                cv2.LINE_AA,
            )

    for i in range(0, len(face), 2):
        p = point(face, i, width, height)

        if p:
            cv2.circle(
                canvas,
                p,
                2,
                (150, 220, 255),
                -1,
                cv2.LINE_AA,
            )


def draw_hand(canvas, hand, width, height):
    if not hand:
        return

    draw_connections(
        canvas,
        hand,
        HAND_CONNECTIONS,
        width,
        height,
    )

    draw_points(
        canvas,
        hand,
        width,
        height,
        4,
    )


def draw_body(canvas, data, width, height):
    pose = data["pose"]

    if not pose:
        return

    draw_connections(
        canvas,
        pose,
        POSE_CONNECTIONS,
        width,
        height,
    )

    draw_points(
        canvas,
        pose,
        width,
        height,
        5,
    )

    # Neck / head connection.
    shoulder_left = point(pose, 11, width, height)
    shoulder_right = point(pose, 12, width, height)
    nose = point(pose, 0, width, height)

    if shoulder_left and shoulder_right:
        neck = (
            (shoulder_left[0] + shoulder_right[0]) // 2,
            (shoulder_left[1] + shoulder_right[1]) // 2,
        )

        cv2.line(
            canvas,
            neck,
            nose if nose else (
                neck[0],
                neck[1] - 50,
            ),
            (95, 180, 245),
            3,
            cv2.LINE_AA,
        )

    # Explicit arm-to-hand connections.
    for pose_id, hand_key in [
        (15, "left_hand"),
        (16, "right_hand"),
    ]:
        wrist = point(pose, pose_id, width, height)
        hand = data[hand_key]

        if wrist and hand:
            hand_wrist = point(
                hand,
                0,
                width,
                height,
            )

            if hand_wrist:
                cv2.line(
                    canvas,
                    wrist,
                    hand_wrist,
                    (130, 210, 255),
                    4,
                    cv2.LINE_AA,
                )

    draw_hand(
        canvas,
        data["left_hand"],
        width,
        height,
    )

    draw_hand(
        canvas,
        data["right_hand"],
        width,
        height,
    )

    draw_face(
        canvas,
        data["face"],
        width,
        height,
    )


def draw_camera_card(canvas, frame, results, expanded=False):
    if expanded:
        x, y, w, h = 30, 80, 690, 460
    else:
        x, y, w, h = 30, 570, 300, 250

    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        (15, 23, 35),
        -1,
    )

    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        (75, 105, 135),
        1,
    )

    preview = frame.copy()

    # Face mesh
    if results.face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            preview,
            results.face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style(),
        )

        mp.solutions.drawing_utils.draw_landmarks(
            preview,
            results.face_landmarks,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style(),
        )

    # Pose
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            preview,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles
                .get_default_pose_landmarks_style(),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(95, 180, 245),
                thickness=2,
                circle_radius=2,
            ),
        )

    # Left hand
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            preview,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles
                .get_default_hand_landmarks_style(),
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_hand_connections_style(),
        )

    # Right hand
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            preview,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles
                .get_default_hand_landmarks_style(),
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_hand_connections_style(),
        )

    preview = cv2.resize(
        preview,
        (w - 20, h - 65),
        interpolation=cv2.INTER_AREA,
    )

    cv2.putText(
        canvas,
        "LIVE CAMERA • LANDMARKS",
        (x + 15, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (205, 220, 235),
        1,
        cv2.LINE_AA,
    )

    canvas[
        y + 38:y + 38 + preview.shape[0],
        x + 10:x + 10 + preview.shape[1],
    ] = preview

    if not expanded:
        bx1, by1 = x + 205, y + 8
        bx2, by2 = x + 285, y + 42

        cv2.rectangle(
            canvas,
            (bx1, by1),
            (bx2, by2),
            (25, 45, 60),
            -1,
        )

        cv2.rectangle(
            canvas,
            (bx1, by1),
            (bx2, by2),
            (90, 170, 215),
            1,
        )

        cv2.putText(
            canvas,
            "EXPAND",
            (bx1 + 10, by1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 220, 240),
            1,
            cv2.LINE_AA,
        )

    else:
        bx1, by1 = 1180, 70
        bx2, by2 = 1320, 110

        cv2.rectangle(
            canvas,
            (bx1, by1),
            (bx2, by2),
            (25, 45, 60),
            -1,
        )

        cv2.rectangle(
            canvas,
            (bx1, by1),
            (bx2, by2),
            (90, 170, 215),
            1,
        )

        cv2.putText(
            canvas,
            "MINIMIZE",
            (bx1 + 25, by1 + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 220, 240),
            1,
            cv2.LINE_AA,
        )

def main():
    if mp is None:
        raise SystemExit(
            f"ERROR: MediaPipe is unavailable: {MEDIAPIPE_IMPORT_ERROR}. "
            "Install the project requirements and restart."
        )

    tracker = ApertureTracker()
    recognizer = SignRecognizer(sequence_length=30)
    recognized_sign = "--"
    recognition_confidence = 0.0
    camera_expanded = False

    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Aperture Signs UI started.")
    print("Press Q to quit.")

    previous = time.time()

    def camera_mouse(event, x, y, flags, param):
        nonlocal camera_expanded

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Small camera EXPAND button.
        if not camera_expanded:
            if 235 <= x <= 315 and 590 <= y <= 625:
                camera_expanded = True

        # Large camera MINIMIZE button.
        else:
            if 1180 <= x <= 1320 and 70 <= y <= 110:
                camera_expanded = False

    cv2.namedWindow(
        "Aperture Signs",
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        "Aperture Signs",
        camera_mouse,
    )

    while True:
        ok, frame = camera.read()

        if not ok:
            print("ERROR: Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        data, results = tracker.process(frame)

        recognition = recognizer.analyze(tracker.history)
        prediction = recognition["prediction"]
        recognized_sign = prediction["label"]
        if prediction["status"] == "stabilizing":
            recognized_sign = f"{prediction['candidate']}..."
        recognition_confidence = prediction["confidence"]

        now = time.time()
        fps = 1.0 / max(now - previous, 0.001)
        previous = now

        W, H = 1440, 900

        canvas = np.zeros(
            (H, W, 3),
            dtype=np.uint8,
        )

        canvas[:] = (8, 13, 22)

        # Subtle futuristic grid.
        for x in range(0, W, 60):
            cv2.line(
                canvas,
                (x, 0),
                (x, H),
                (14, 24, 36),
                1,
            )

        for y in range(0, H, 60):
            cv2.line(
                canvas,
                (0, y),
                (W, y),
                (14, 24, 36),
                1,
            )

        cv2.putText(
            canvas,
            "APERTURE",
            (35, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (220, 230, 240),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "SIGNS",
            (35, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 190, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "LIVE HUMAN MOTION INTELLIGENCE",
            (200, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (130, 150, 170),
            1,
            cv2.LINE_AA,
        )

        # Main skeleton panel.
        px, py = 355, 105
        pw, ph = 1050, 700

        cv2.rectangle(
            canvas,
            (px, py),
            (px + pw, py + ph),
            (15, 23, 35),
            -1,
        )

        cv2.rectangle(
            canvas,
            (px, py),
            (px + pw, py + ph),
            (65, 90, 115),
            1,
        )

        cv2.putText(
            canvas,
            "LIVE SKELETAL REPRESENTATION",
            (px + 20, py + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (205, 220, 235),
            1,
            cv2.LINE_AA,
        )

        sx, sy = px + 20, py + 50
        sw, sh = pw - 40, ph - 70

        skeleton = np.zeros(
            (sh, sw, 3),
            dtype=np.uint8,
        )

        skeleton[:] = (10, 17, 27)

        draw_body(
            skeleton,
            data,
            sw,
            sh,
        )

        canvas[
            sy:sy + sh,
            sx:sx + sw,
        ] = skeleton

        # Camera.
        draw_camera_card(
            canvas,
            frame,
            results,
            camera_expanded,
        )

        # Recognition panel.
        rx, ry = 30, 105
        rw, rh = 300, 190

        cv2.rectangle(
            canvas,
            (rx, ry),
            (rx + rw, ry + rh),
            (15, 23, 35),
            -1,
        )

        cv2.rectangle(
            canvas,
            (rx, ry),
            (rx + rw, ry + rh),
            (65, 90, 115),
            1,
        )

        cv2.putText(
            canvas,
            "SIGN RECOGNITION",
            (rx + 15, ry + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (205, 220, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            recognized_sign,
            (rx + 15, ry + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (130, 220, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "TEMPORAL BUFFER",
            (rx + 15, ry + 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (130, 150, 170),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"{len(tracker.history)} / 60",
            (rx + 15, ry + 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (205, 220, 235),
            1,
            cv2.LINE_AA,
        )

        # Bottom telemetry.
        telemetry = (
            f"FACE {len(data['face']):03d}    "
            f"L-HAND {len(data['left_hand']):02d}    "
            f"R-HAND {len(data['right_hand']):02d}    "
            f"POSE {len(data['pose']):02d}    "
            f"{fps:04.1f} FPS"
        )

        # ==============================
        # SIGN RECOGNITION PANEL
        # ==============================

        panel_x = 360
        panel_y = 720
        panel_w = 500
        panel_h = 120

        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (12, 20, 30),
            -1,
        )

        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (65, 100, 125),
            1,
        )

        cv2.putText(
            canvas,
            "SIGN RECOGNITION",
            (panel_x + 18, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 175, 195),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            recognized_sign,
            (panel_x + 20, panel_y + 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (120, 220, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"CONFIDENCE  {recognition_confidence * 100:.0f}%",
            (panel_x + 190, panel_y + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 175, 195),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"MOTION  {recognition['movement_energy']:.4f}",
            (panel_x + 190, panel_y + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (110, 150, 175),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Aperture Signs",
            canvas,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    tracker.holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
