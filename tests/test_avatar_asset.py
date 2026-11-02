import json
import struct
from pathlib import Path


AVATAR_PATH = Path(__file__).resolve().parents[1] / "app" / "ui" / "avatar" / "aperture_avatar.glb"


def load_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    version, declared_length = struct.unpack_from("<II", data, 4)
    assert version == 2
    assert declared_length == len(data)
    json_length, json_chunk_type = struct.unpack_from("<II", data, 12)
    assert json_chunk_type == 0x4E4F534A
    return json.loads(data[20 : 20 + json_length])


def test_bundled_avatar_has_skinned_body_and_finger_rig():
    glb = load_glb_json(AVATAR_PATH)
    nodes = glb["nodes"]
    skin = glb["skins"][0]
    joint_names = {nodes[index]["name"] for index in skin["joints"]}

    assert len(joint_names) >= 150
    assert {"head", "upperarm01.L", "finger1-1.L", "finger5-3.R"} <= joint_names
    assert any(mesh.get("name") == "Aperture Human" for mesh in nodes)
    assert any(mesh.get("name") == "Aperture Hair" for mesh in nodes)
    assert {"Aperture Skin", "Aperture Shirt", "Aperture Trousers"} <= {
        material["name"] for material in glb["materials"]
    }
    assert any(node.get("skin") == 0 for node in nodes)
