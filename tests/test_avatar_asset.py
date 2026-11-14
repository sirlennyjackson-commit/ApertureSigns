import json
import math
import struct
from pathlib import Path

AVATAR_PATH = Path(__file__).resolve().parents[1] / "app" / "ui" / "avatar" / "aperture_avatar.glb"


def load_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    version, declared_length = struct.unpack_from("<II", data, 4)
    assert version == 2
    assert declared_length == len(data)
    json_length, json_chunk_type = struct.unpack_from("<II", data, 12)
    assert json_chunk_type == 0x4E4F534A
    glb = json.loads(data[20 : 20 + json_length])
    bin_header = 20 + json_length
    bin_length, bin_chunk_type = struct.unpack_from("<II", data, bin_header)
    assert bin_chunk_type == 0x004E4942
    binary = data[bin_header + 8 : bin_header + 8 + bin_length]
    return glb, binary


def iter_accessor(glb: dict, binary: bytes, accessor_index: int, fmt: str, width: int):
    accessor = glb["accessors"][accessor_index]
    view = glb["bufferViews"][accessor["bufferView"]]
    stride = view.get("byteStride", struct.calcsize(fmt) * width)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    for index in range(accessor["count"]):
        yield struct.unpack_from("<" + fmt * width, binary, start + index * stride)


def test_bundled_avatar_is_skinned_weighted_and_textured():
    glb, binary = load_glb(AVATAR_PATH)

    assert len(glb["meshes"]) == 1
    skinned_nodes = [node for node in glb["nodes"] if "mesh" in node and "skin" in node]
    assert len(skinned_nodes) == 1
    assert len(glb["skins"]) == 1
    skin = glb["skins"][0]
    joint_names = {glb["nodes"][index]["name"] for index in skin["joints"]}

    required_joints = {
        "head",
        "neck01",
        "clavicle.L",
        "clavicle.R",
        "upperarm01.L",
        "upperarm01.R",
        "lowerarm02.L",
        "lowerarm02.R",
    }
    required_joints.update(
        f"metacarpal{index}.{side}"
        for side in ("L", "R")
        for index in range(1, 5)
    )
    required_joints.update(
        f"finger{finger}-{segment}.{side}"
        for side in ("L", "R")
        for finger in range(1, 6)
        for segment in range(1, 4)
    )
    assert required_joints <= joint_names

    primitive = glb["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    assert {"POSITION", "JOINTS_0", "WEIGHTS_0"} <= set(attributes)
    joints = glb["accessors"][attributes["JOINTS_0"]]
    weights = glb["accessors"][attributes["WEIGHTS_0"]]
    assert joints["type"] == weights["type"] == "VEC4"
    assert joints["count"] == weights["count"] > 0
    assert joints["componentType"] == 5121
    assert weights["componentType"] == 5126

    used_joint_ids = set()
    joint_rows = iter_accessor(glb, binary, attributes["JOINTS_0"], "B", 4)
    weight_rows = iter_accessor(glb, binary, attributes["WEIGHTS_0"], "f", 4)
    for joint_row, weight_row in zip(joint_rows, weight_rows):
        assert math.isclose(sum(weight_row), 1.0, abs_tol=1e-5)
        used_joint_ids.update(joint for joint, weight in zip(joint_row, weight_row) if weight > 1e-6)
    used_joint_names = {glb["nodes"][skin["joints"][index]]["name"] for index in used_joint_ids}
    assert required_joints <= used_joint_names

    assert glb["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]
    image_names = {image["name"] for image in glb["images"]}
    assert any(name.endswith("_Diffuse") for name in image_names)
    assert any(name.endswith("_Normal") for name in image_names)
    assert all("bufferView" in image for image in glb["images"])
