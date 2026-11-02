"""Generate the bundled Aperture Signs avatar using the installed MPFB Blender extension."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "app" / "ui" / "avatar" / "aperture_avatar.glb"


def make_material(name: str, color: tuple[float, float, float, float], roughness: float):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    if material.node_tree is None:
        material.use_nodes = True

    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        raise RuntimeError("Blender did not create a Principled BSDF node")

    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    return material


def add_skin_scattering(material):
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Subsurface Weight"].default_value = 0.055
    shader.inputs["Subsurface Radius"].default_value = (1.0, 0.42, 0.24)
    shader.inputs["Specular IOR Level"].default_value = 0.28


def parent_to_bone(obj, armature, bone_name: str):
    world_matrix = obj.matrix_world.copy()
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_world = world_matrix


def add_bone_sphere(name, center, scale, material, armature, bone_name):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=16,
        location=center,
    )
    obj = bpy.context.object
    obj.name = name
    for vertex in obj.data.vertices:
        vertex.co.x *= scale[0]
        vertex.co.y *= scale[1]
        vertex.co.z *= scale[2]
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    parent_to_bone(obj, armature, bone_name)
    return obj


def add_eyes_and_hair(armature):
    eye_white = make_material("Aperture Eye White", (0.84, 0.78, 0.68, 1.0), 0.24)
    iris = make_material("Aperture Brown Iris", (0.13, 0.055, 0.024, 1.0), 0.28)
    pupil = make_material("Aperture Pupil", (0.009, 0.006, 0.005, 1.0), 0.18)
    hair = make_material("Aperture Hair", (0.035, 0.022, 0.018, 1.0), 0.72)

    eye_positions = {}
    for side in ("L", "R"):
        bone_name = f"eye.{side}"
        eye_bone = armature.data.bones.get(bone_name)
        if eye_bone is None:
            raise RuntimeError(f"MPFB default rig is missing its {bone_name} bone")
        center = armature.matrix_world @ eye_bone.head_local
        eye_positions[side] = center.copy()
        outward = Vector((0.0, -1.0, 0.0))
        add_bone_sphere(
            f"Aperture Eye {side}",
            center,
            (0.015, 0.011, 0.014),
            eye_white,
            armature,
            bone_name,
        )
        add_bone_sphere(
            f"Aperture Iris {side}",
            center + outward * 0.009,
            (0.007, 0.004, 0.007),
            iris,
            armature,
            bone_name,
        )
        add_bone_sphere(
            f"Aperture Pupil {side}",
            center + outward * 0.012,
            (0.0035, 0.0025, 0.004),
            pupil,
            armature,
            bone_name,
        )

    center = (eye_positions["L"] + eye_positions["R"]) * 0.5
    segments = 32
    rings = 8
    vertices = []
    faces = []
    radius_x, radius_y, radius_z = 0.095, 0.09, 0.085
    hair_center = center + Vector((0.0, 0.035, 0.055))
    for ring in range(rings + 1):
        phi = (math.pi / 2) * ring / rings
        for segment in range(segments):
            theta = 2 * math.pi * segment / segments
            vertices.append(
                (
                    hair_center.x + radius_x * math.sin(phi) * math.cos(theta),
                    hair_center.y + radius_y * math.sin(phi) * math.sin(theta),
                    hair_center.z + radius_z * math.cos(phi),
                )
            )
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + next_segment
            c = (ring + 1) * segments + next_segment
            d = (ring + 1) * segments + segment
            faces.append((a, d, c, b))
    hair_mesh = bpy.data.meshes.new("Aperture Hair Mesh")
    hair_mesh.from_pydata(vertices, [], faces)
    hair_mesh.materials.append(hair)
    hair_obj = bpy.data.objects.new("Aperture Hair", hair_mesh)
    bpy.context.collection.objects.link(hair_obj)
    for polygon in hair_mesh.polygons:
        polygon.use_smooth = True
    parent_to_bone(hair_obj, armature, "head")


def generate_avatar(output_path: Path) -> None:
    try:
        from bl_ext.user_default.mpfb.services import HumanService, TargetService
    except ImportError as exc:
        raise RuntimeError(
            "MPFB was not loaded by Blender. Install and enable MPFB 2.0.17 or later."
        ) from exc

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    macro_details = TargetService.get_default_macro_info_dict()
    macro_details["gender"] = 0.5
    macro_details["age"] = 0.5
    macro_details["muscle"] = 0.5
    macro_details["weight"] = 0.5
    macro_details["height"] = 0.5
    macro_details["proportions"] = 0.5
    for race in macro_details["race"]:
        macro_details["race"][race] = 1.0 / len(macro_details["race"])

    basemesh = HumanService.create_human(
        mask_helpers=True,
        detailed_helpers=False,
        extra_vertex_groups=True,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macro_details,
    )
    basemesh.name = "Aperture Human"

    material = make_material("Aperture Skin", (0.48, 0.27, 0.19, 1.0), 0.48)
    add_skin_scattering(material)
    shirt = make_material("Aperture Shirt", (0.035, 0.17, 0.19, 1.0), 0.82)
    trousers = make_material("Aperture Trousers", (0.055, 0.075, 0.09, 1.0), 0.86)
    basemesh.data.materials.clear()
    basemesh.data.materials.append(material)
    basemesh.data.materials.append(shirt)
    basemesh.data.materials.append(trousers)
    for polygon in basemesh.data.polygons:
        center = basemesh.matrix_world @ polygon.center
        if 0.88 <= center.z <= 1.39 and abs(center.x) <= 0.3:
            polygon.material_index = 1
        elif 0.0 <= center.z < 0.88:
            polygon.material_index = 2

    TargetService.bake_targets(basemesh)
    hide_helpers = basemesh.modifiers.get("Hide helpers")
    if hide_helpers is not None:
        bpy.context.view_layer.objects.active = basemesh
        bpy.ops.object.modifier_apply(modifier=hide_helpers.name)

    armature = HumanService.add_builtin_rig(
        basemesh,
        "default",
        import_weights=True,
    )
    if armature is None:
        raise RuntimeError("MPFB could not create the default finger-capable rig")
    armature.name = "Aperture Human Rig"
    armature.data.name = "Aperture Human Rig"
    add_eyes_and_hair(armature)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type in {"MESH", "ARMATURE"}:
            obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_animations=False,
        export_skins=True,
        export_morph=True,
    )
    if "FINISHED" not in result or not output_path.is_file():
        raise RuntimeError(f"Blender failed to export the avatar to {output_path}")

    print(f"Avatar exported: {output_path} ({output_path.stat().st_size} bytes)")
    print(f"Armature: {armature.name}; bones: {len(armature.data.bones)}")
    print(f"Body mesh: {basemesh.name}; vertices: {len(basemesh.data.vertices)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination .glb path",
    )
    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    generate_avatar(args.output.resolve())


if __name__ == "__main__":
    main()
