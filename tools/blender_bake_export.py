# Bake Pipo's animation as per-frame vertex positions (rubber-hose faithful).
#
#   blender --background <file.blend> --python blender_bake_export.py -- <action-name> <out.json>
#
# Then build the USDZ with build_baked_usdz.py.
#
# Why baking instead of skeleton export: the ARP rig uses bendy bones
# (leg_stretch has 20 segments) and subdivision AFTER the armature in the
# modifier stack. Both produce the rubber-hose curvature, and neither is
# representable in UsdSkel's linear blend skinning — RealityKit renders
# sharp knees. Evaluating the full modifier stack per frame captures the
# exact deformation; the USDZ carries one blend shape per frame.

import bpy
import json
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
ACTION_NAME, OUT = argv[0], argv[1]

RIG_NAME = "rig"
SUBSURF_CAP = 3

scn = bpy.context.scene
rig = bpy.data.objects[RIG_NAME]

# Auto-detect every mesh actually deformed by the rig (an Armature modifier
# targeting it). The model changes over time — meshes get added, renamed,
# or left as stale unused copies — so this reads whatever is really driven
# by the rig right now instead of hardcoding names. A mesh that's merely
# object-parented to the rig with no Armature modifier does NOT move with
# the animation and is excluded (that's how a stale orphaned copy shows up).
KEEP_MESHES = tuple(
    ob.name for ob in bpy.data.objects
    if ob.type == "MESH" and not ob.name.startswith("cs_")
    and any(m.type == "ARMATURE" and m.object is rig for m in ob.modifiers)
)
print("deforming meshes:", KEEP_MESHES)

action = bpy.data.actions[ACTION_NAME]
if rig.animation_data is None:
    rig.animation_data_create()
rig.animation_data.action = action
start, end = (int(f) for f in action.frame_range)
scn.frame_start, scn.frame_end = start, end
print(f"action {action.name!r} frames {start}-{end} @ {scn.render.fps}fps")

for name in KEEP_MESHES + (RIG_NAME,):
    ob = bpy.data.objects[name]
    ob.hide_viewport = False
    ob.hide_render = False
    ob.hide_set(False)

for name in KEEP_MESHES:
    ob = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.shade_smooth()
    for mod in ob.modifiers:
        if mod.type == "SUBSURF":
            mod.levels = min(mod.levels, SUBSURF_CAP)
            mod.render_levels = mod.levels


def principled_color(ob):
    for slot in ob.material_slots:
        m = slot.material
        if m and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    c = n.inputs["Base Color"].default_value
                    r = n.inputs["Roughness"].default_value
                    return [c[0], c[1], c[2]], r
    return [0.8, 0.8, 0.8], 0.5


def evaluated_mesh(ob):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return ob.evaluated_get(depsgraph), depsgraph


def rounded(vals):
    return [round(v, 5) for v in vals]


meshes = {}

# Rest topology + positions: armature disabled, subsurf still on
for name in KEEP_MESHES:
    ob = bpy.data.objects[name]
    for mod in ob.modifiers:
        if mod.type == "ARMATURE":
            mod.show_viewport = False
            mod.show_render = False
bpy.context.view_layer.update()

for name in KEEP_MESHES:
    ob_eval, depsgraph = evaluated_mesh(bpy.data.objects[name])
    me = ob_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    color, roughness = principled_color(bpy.data.objects[name])
    counts, indices = [], []
    for p in me.polygons:
        counts.append(len(p.vertices))
        indices.extend(p.vertices)
    positions, normals = [], []
    for v in me.vertices:
        positions.extend(rounded(v.co))
        normals.extend(rounded(v.normal))
    meshes[name] = {
        "color": color, "roughness": roughness,
        "counts": counts, "indices": indices,
        "rest": positions, "normals": normals, "frames": [],
    }
    ob_eval.to_mesh_clear()
    print(f"{name}: {len(positions)//3} verts, {len(counts)} faces")

for name in KEEP_MESHES:
    for mod in bpy.data.objects[name].modifiers:
        if mod.type == "ARMATURE":
            mod.show_viewport = True
            mod.show_render = True
bpy.context.view_layer.update()

# Per-frame evaluated positions (full stack: armature + bendy bones + subsurf)
for f in range(start, end + 1):
    scn.frame_set(f)
    for name in KEEP_MESHES:
        ob_eval, depsgraph = evaluated_mesh(bpy.data.objects[name])
        me = ob_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        assert len(me.vertices) * 3 == len(meshes[name]["rest"]), \
            f"vertex count changed on {name} at frame {f}"
        frame = []
        for v in me.vertices:
            frame.extend(rounded(v.co))
        meshes[name]["frames"].append(frame)
        ob_eval.to_mesh_clear()

with open(OUT, "w") as fp:
    json.dump({"fps": scn.render.fps, "start": start, "end": end,
               "meshes": meshes}, fp)
print(f"wrote {OUT}")
