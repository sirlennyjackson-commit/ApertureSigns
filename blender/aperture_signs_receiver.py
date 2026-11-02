"""
Aperture Signs Blender receiver.

The desktop app launches this script with the supplied avatar scene. It listens
on UDP 127.0.0.1:5005 and retargets pose and finger landmarks to the Human.rig
armature, or ApertureAvatar when that rig is present. Run this manually from
Blender's Scripting workspace only when starting the bridge by hand.
"""

import json
import math
import socket
import threading

import bpy
from mathutils import Matrix, Quaternion, Vector


HOST = "127.0.0.1"
PORT = 5005
SCALE = 2.0
_STATE = {"running": False, "payload": None, "error": None, "frames": 0}
AVATAR_ARMATURE_NAMES = ("Human.rig", "ApertureAvatar")


def _ensure_empty(name):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "SPHERE"
    empty.empty_display_size = 0.04
    bpy.context.scene.collection.objects.link(empty)
    return empty


def _to_blender(xyz):
    # MediaPipe: x right, y down, z toward camera. Blender: z up.
    x, y, z = xyz
    return Vector(((x - 0.5) * SCALE, (-z) * SCALE, (0.5 - y) * SCALE))


def _aim_bone(armature, name, start, end, max_angle=None):
    bone = armature.pose.bones.get(name)
    if bone is None:
        return
    direction = _to_blender(end) - _to_blender(start)
    if direction.length < 1e-5:
        return

    bpy.context.view_layer.update()
    current_direction = bone.tail - bone.head
    if current_direction.length < 1e-5:
        return
    delta = current_direction.normalized().rotation_difference(direction.normalized())
    if max_angle is not None and delta.angle > max_angle:
        delta = delta.slerp(Quaternion(), max_angle / delta.angle)
    rotate = delta.to_matrix().to_4x4()
    bone.matrix = (
        Matrix.Translation(bone.head)
        @ rotate
        @ Matrix.Translation(-bone.head)
        @ bone.matrix
    )
    bpy.context.view_layer.update()


def _retarget_armature(armature, payload):
    joints = payload.get("joints") or {}
    left_hand = payload.get("left_hand") or []
    right_hand = payload.get("right_hand") or []
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()

    def point(name):
        return joints.get(name)

    left_shoulder, right_shoulder = point("left_shoulder"), point("right_shoulder")
    left_elbow, right_elbow = point("left_elbow"), point("right_elbow")
    left_wrist, right_wrist = point("left_wrist"), point("right_wrist")

    if left_shoulder and right_shoulder:
        shoulder_center = [(a + b) * 0.5 for a, b in zip(left_shoulder, right_shoulder)]
    else:
        shoulder_center = None

    nose = point("nose")
    if shoulder_center and nose and nose[1] < shoulder_center[1]:
        _aim_bone(
            armature,
            "neck01",
            shoulder_center,
            nose,
            max_angle=math.radians(12),
        )

    for side, shoulder, elbow, wrist, hand in (
        ("L", left_shoulder, left_elbow, left_wrist, left_hand),
        ("R", right_shoulder, right_elbow, right_wrist, right_hand),
    ):
        if shoulder_center and shoulder:
            _aim_bone(
                armature,
                f"clavicle.{side}",
                shoulder_center,
                shoulder,
                max_angle=math.radians(25),
            )
        if shoulder and elbow:
            _aim_bone(
                armature,
                f"upperarm01.{side}",
                shoulder,
                elbow,
                max_angle=math.radians(100),
            )
        if elbow and wrist:
            _aim_bone(
                armature,
                f"lowerarm02.{side}",
                elbow,
                wrist,
                max_angle=math.radians(120),
            )

        if hand and len(hand) >= 21:
            palm_center = [sum(hand[index][axis] for index in (5, 9, 13, 17)) / 4 for axis in range(3)]
            _aim_bone(
                armature,
                f"wrist.{side}",
                hand[0],
                palm_center,
                max_angle=math.radians(90),
            )
            digit_bones = (
                (1, (f"finger1-1.{side}", f"finger1-2.{side}", f"finger1-3.{side}"), (0, 1, 2, 4)),
                (2, (f"metacarpal1.{side}", f"finger2-1.{side}", f"finger2-2.{side}", f"finger2-3.{side}"), (0, 5, 6, 7, 8)),
                (3, (f"metacarpal2.{side}", f"finger3-1.{side}", f"finger3-2.{side}", f"finger3-3.{side}"), (0, 9, 10, 11, 12)),
                (4, (f"metacarpal3.{side}", f"finger4-1.{side}", f"finger4-2.{side}", f"finger4-3.{side}"), (0, 13, 14, 15, 16)),
                (5, (f"metacarpal4.{side}", f"finger5-1.{side}", f"finger5-2.{side}", f"finger5-3.{side}"), (0, 17, 18, 19, 20)),
            )
            for _digit, bones, landmarks in digit_bones:
                for bone_name, start_index, end_index in zip(bones, landmarks, landmarks[1:]):
                    _aim_bone(
                        armature,
                        bone_name,
                        hand[start_index],
                        hand[end_index],
                        max_angle=math.radians(144),
                    )


def apply_payload():
    payload = _STATE.get("payload")
    if not payload:
        return 0.5
    joints = payload.get("joints") or {}
    armature = next(
        (bpy.data.objects.get(name) for name in AVATAR_ARMATURE_NAMES if bpy.data.objects.get(name)),
        None,
    )
    if armature and armature.type == "ARMATURE":
        _retarget_armature(armature, payload)
    else:
        for name, xyz in joints.items():
            _ensure_empty(f"AP_{name}").location = _to_blender(xyz)
        for side in ("left_hand", "right_hand"):
            for index, xyz in enumerate(payload.get(side) or []):
                empty = _ensure_empty(f"AP_{side}_{index:02d}")
                empty.location = _to_blender(xyz)
                empty.empty_display_size = 0.02
    _STATE["frames"] += 1
    return 0.03


class Receiver(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((HOST, PORT))
        self.sock.settimeout(0.5)

    def run(self):
        _STATE["running"] = True
        while _STATE["running"]:
            try:
                raw, _addr = self.sock.recvfrom(65535)
                _STATE["payload"] = json.loads(raw.decode("utf-8"))
                _STATE["error"] = None
            except socket.timeout:
                continue
            except OSError as exc:
                _STATE["error"] = str(exc)
                break
            except json.JSONDecodeError as exc:
                _STATE["error"] = f"Invalid motion packet: {exc}"
        try:
            self.sock.close()
        except OSError:
            pass


def register():
    if _STATE.get("thread") and _STATE["thread"].is_alive():
        print("Aperture Signs receiver already running.")
        return
    thread = Receiver()
    _STATE["thread"] = thread
    thread.start()
    if not bpy.app.timers.is_registered(apply_payload):
        bpy.app.timers.register(apply_payload, first_interval=0.05)
    print(f"Aperture Signs receiver listening on {HOST}:{PORT}")


def unregister():
    _STATE["running"] = False
    if bpy.app.timers.is_registered(apply_payload):
        bpy.app.timers.unregister(apply_payload)


register()
