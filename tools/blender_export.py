# Headless Blender export of Pipo to USDZ for RealityKit.
#
#   blender --background <file.blend> --python blender_export.py -- <action-name> <out.usdz>
#
# Then ALWAYS run fix_usdz_for_realitykit.py on the result.
#
# What this does beyond a plain export, and why:
# - Applies object transforms on the meshes: unapplied (especially negative)
#   scales end up double-applied or ignored by RealityKit's UsdSkel skinning,
#   exploding the geometry.
# - Applies subdivision modifiers: USD subdivision schemas + skinning don't
#   survive the trip into RealityKit; baked vertices do.
# - Skips texture export: Pipo's materials are constant colors, and Blender
#   otherwise packs EXR textures which iOS USDZ rejects (magenta model).
# - Exports only deform bones with the given action assigned to the ARP rig.

import bpy
import json
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
ACTION_NAME, OUT = argv[0], argv[1]

KEEP_MESHES = {"Pipo_Body", "Pipo_Eyes", "Pipo_Mouth"}
RIG_NAME = "rig"

scn = bpy.context.scene
rig = bpy.data.objects[RIG_NAME]

# Assign the requested action to the ARP rig and match the scene range to it
action = bpy.data.actions[ACTION_NAME]
if rig.animation_data is None:
    rig.animation_data_create()
rig.animation_data.action = action
start, end = action.frame_range
scn.frame_start, scn.frame_end = int(start), int(end)
print(f"action {action.name!r} frames {scn.frame_start}-{scn.frame_end} @ {scn.render.fps}fps")

keep = KEEP_MESHES | {RIG_NAME}
for ob in bpy.data.objects:
    if ob.name in keep:
        ob.hide_viewport = False
        ob.hide_render = False
        ob.hide_set(False)

for name in KEEP_MESHES:
    ob = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = ob

    # Blender 5 corrects face winding itself when applying mirrored
    # (negative-scale) transforms — do NOT flip normals afterwards.
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"applied transform on {name}")

    for mod in list(ob.modifiers):
        if mod.type == "SUBSURF":
            mod.levels = min(mod.levels, 3)  # keep vert count mobile-friendly
            bpy.ops.object.modifier_apply(modifier=mod.name)
            print(f"applied subsurf on {name}: {len(ob.data.vertices)} verts")

    # smooth normals do more for perceived smoothness than poly count
    bpy.ops.object.shade_smooth()

bpy.ops.object.select_all(action="DESELECT")
for ob in bpy.data.objects:
    try:
        ob.select_set(ob.name in keep)
    except RuntimeError:
        pass
bpy.context.view_layer.objects.active = rig

# Sidecar: world-space matrices for every bone, every frame. Blender's USD
# export authors joint-local animation against the FULL bone hierarchy while
# listing only deform bones, so the recomposed transforms are garbage in any
# UsdSkel consumer. fix_usdz_for_realitykit.py rebuilds bind/rest/anim data
# from this ground truth instead.
def mat_list(m):
    return [[m[r][c] for c in range(4)] for r in range(4)]

bones = {}
for b in rig.data.bones:
    bones[b.name] = {
        "parent": b.parent.name if b.parent else None,
        "rest_world": mat_list(rig.matrix_world @ b.matrix_local),
        "frames": [],
    }
for f in range(scn.frame_start, scn.frame_end + 1):
    scn.frame_set(f)
    rig_eval = rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
    world = rig_eval.matrix_world
    for pb in rig_eval.pose.bones:
        bones[pb.name]["frames"].append(mat_list(world @ pb.matrix))
scn.frame_set(scn.frame_start)

with open(OUT + ".skel.json", "w") as fp:
    json.dump({"fps": scn.render.fps, "start": scn.frame_start,
               "end": scn.frame_end, "bones": bones}, fp)
print(f"wrote sidecar {OUT}.skel.json ({len(bones)} bones)")

op_props = bpy.ops.wm.usd_export.get_rna_type().properties.keys()
wanted = {
    "filepath": OUT,
    "selected_objects_only": True,
    "visible_objects_only": False,
    "export_animation": True,
    "export_armatures": True,
    "only_deform_bones": True,
    "export_shapekeys": True,
    "export_materials": True,
    "export_textures": False,
    "export_subdivision": "IGNORE",
    "convert_orientation": False,
}
kwargs = {k: v for k, v in wanted.items() if k in op_props}
bpy.ops.wm.usd_export(**kwargs)
print("exported", OUT)
