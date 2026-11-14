"""Preview live camera landmarks and provide a hook for offline interpretation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import cv2

from backend.vision.hand_tracker import ApertureTracker, MEDIAPIPE_IMPORT_ERROR, mp


def _camera_source(value: str) -> int | str:
    """Treat numeric sources as camera indices and other values as stream URLs."""
    try:
        source = int(value)
    except ValueError:
        return value
    if source < 0:
        raise argparse.ArgumentTypeError("Camera index must be zero or greater.")
    return source


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive whole number.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive whole number.")
    return number


def interpret_landmarks(landmarks: dict) -> None:
    """Connect an offline KSL model here; landmarks contain face, pose, and hands."""
    # Keep model inference separate from camera capture and MediaPipe tracking.


def _draw_landmarks(frame, results) -> None:
    drawing = mp.solutions.drawing_utils
    holistic = mp.solutions.holistic
    styles = mp.solutions.drawing_styles

    if results.pose_landmarks:
        drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
        )
    if results.left_hand_landmarks:
        drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=styles.get_default_hand_connections_style(),
        )
    if results.right_hand_landmarks:
        drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=styles.get_default_hand_connections_style(),
        )
    if results.face_landmarks:
        drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=styles.get_default_face_mesh_contours_style(),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=_camera_source,
        default=0,
        help="camera index (default: 0) or a video stream URL",
    )
    parser.add_argument("--width", type=_positive_int, default=1280)
    parser.add_argument("--height", type=_positive_int, default=720)
    args = parser.parse_args(argv)

    if mp is None:
        print(
            f"MediaPipe is unavailable: {MEDIAPIPE_IMPORT_ERROR}. "
            "Install the project requirements and try again."
        )
        return 1

    try:
        camera = cv2.VideoCapture(args.camera)
    except cv2.error as exc:
        print(f"Could not initialize the camera or video stream: {exc}")
        return 1
    if not camera.isOpened():
        camera.release()
        print("Could not open the selected camera or video stream.")
        return 1

    tracker = None
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        tracker = ApertureTracker(apply_smoothing=True)
        window_name = "Aperture Signs — Live Landmarks"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("Camera tracking started. Press q to quit.")

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stream stopped or returned an unreadable frame.")
                return 1

            frame = cv2.flip(frame, 1)
            landmarks, results = tracker.process(frame)

            # Keep custom offline inference isolated from camera and display code.
            interpret_landmarks(landmarks)
            _draw_landmarks(frame, results)
            hand_count = bool(landmarks["left_hand"]) + bool(landmarks["right_hand"])
            cv2.putText(
                frame,
                f"Hands: {hand_count}  |  Press Q to quit",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 230, 190),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    except cv2.error as exc:
        print(f"OpenCV camera or display error: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"Could not initialize landmark tracking: {exc}")
        return 1
    finally:
        try:
            camera.release()
        finally:
            try:
                if tracker is not None:
                    tracker.holistic.close()
            finally:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
