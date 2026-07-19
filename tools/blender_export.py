# Headless Blender export of Pipo to USDZ for RealityKit.
#
#   blender --background <file.blend> --python blender_export.py -- <action-name> <out.usdz> [start end]
#
# Then ALWAYS run fix_usdz_for_realitykit.py on the result.
#
# What this does beyond a plain export, and why:
# - Auto-detects the skinned meshes (Armature modifier targeting the rig)
#   instead of hardcoding names — the model changes between file versions.
# - Converts "decal" meshes (eyes/mouth: bone-parented to a control bone and
#   shrinkwrapped onto the body) into real skinned meshes: modifier stack
#   evaluated at rest pose, verts moved to world space, then weighted 100%
#   to the matching deform bone (c_head.x -> head.x). Control bones are not
#   exported, and shrinkwrap does not survive any export.
# - Gives the body a material if it has none (Pipo pink, matching the color
#   the previous baked Pipo.usdz shipped with).
# - Applies object transforms on the meshes: unapplied (especially negative)
#   scales end up double-applied or ignored by RealityKit's UsdSkel skinning,
#   exploding the geometry.
# - Applies subdivision modifiers: USD subdivision schemas + skinning don't
#   survive the trip into RealityKit; baked vertices do.
# - Skips texture export: Pipo's materials are constant colors, and Blender
#   otherwise packs EXR textures which iOS USDZ rejects (magenta model).
# - Exports only deform bones with the given action assigned to the ARP rig.
# - Sidecar carries world-space bone matrices AND per-frame shape key values
#   (knee correctives are driver-driven; the USD exporter can't be trusted to
#   sample drivers, so fix_usdz_for_realitykit.py authors the weights).

import bpy
import json
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
ACTION_NAME, OUT = argv[0], argv[1]
FRAME_RANGE = (int(argv[2]), int(argv[3])) if len(argv) >= 4 else None

RIG_NAME = "rig"
PIPO_PINK = (0.8000395, 0.16891873, 0.43402848)

scn = bpy.context.scene
rig = bpy.data.objects[RIG_NAME]

# Files saved mid-animation open in Pose (or Edit) mode, where object-mode
# operators fail their poll. Force object mode before doing anything.
if bpy.context.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")

# Assign the requested action to the ARP rig and match the scene range to it
action = bpy.data.actions[ACTION_NAME]
if rig.animation_data is None:
    rig.animation_data_create()
rig.animation_data.action = action
if FRAME_RANGE:
    scn.frame_start, scn.frame_end = FRAME_RANGE
else:
    start, end = action.frame_range
    scn.frame_start, scn.frame_end = int(start), int(end)
print(f"action {action.name!r} frames {scn.frame_start}-{scn.frame_end} @ {scn.render.fps}fps")

def is_real_mesh(ob):
    return ob.type == "MESH" and not ob.name.startswith("cs_")

skinned = [ob for ob in bpy.data.objects if is_real_mesh(ob)
           and any(m.type == "ARMATURE" and m.object is rig for m in ob.modifiers)]
decals = [ob for ob in bpy.data.objects if is_real_mesh(ob)
          and ob not in skinned and ob.parent is rig and ob.parent_type == "BONE"]
print("skinned meshes:", [o.name for o in skinned])
print("decal meshes:", [o.name for o in decals])

for ob in skinned + decals + [rig]:
    ob.hide_viewport = False
    ob.hide_render = False
    ob.hide_set(False)

# ── Decal conversion: evaluate full stack at rest pose, bake to a world-space
#    mesh, skin it 100% to the deform twin of its control parent bone.
rig.data.pose_position = "REST"
bpy.context.view_layer.update()
converted = []
for ob in decals:
    parent_bone = ob.parent_bone
    deform_name = parent_bone if rig.data.bones[parent_bone].use_deform \
        else parent_bone.removeprefix("c_")
    assert deform_name in rig.data.bones and rig.data.bones[deform_name].use_deform, \
        f"no deform bone for decal {ob.name} (parent bone {parent_bone})"

    if not ob.vertex_groups:
        # No painted weights (clip files older than v34.Model): transfer
        # them from the body's skin right here, at rest pose — for each
        # base vert, sample the nearest body surface point and blend that
        # face's corner weights. Rigid head.x weighting is NOT equivalent:
        # the cheek region under the eyes carries shoulder influence, so a
        # rigid eye shell gets overtaken by the deforming skin beneath it
        # in some poses (the left eye visibly occluded mid-walk).
        body = skinned[0]
        body_inv = body.matrix_world.inverted()
        made = {}
        for v in ob.data.vertices:
            world = ob.matrix_world @ v.co
            ok, loc, _, face_i = body.closest_point_on_mesh(body_inv @ world)
            if not ok:
                continue
            poly = body.data.polygons[face_i]
            acc = {}
            for vi in poly.vertices:
                bv = body.data.vertices[vi]
                w = 1.0 / max((bv.co - loc).length, 1e-6)
                for g in bv.groups:
                    gname = body.vertex_groups[g.group].name
                    acc[gname] = acc.get(gname, 0.0) + g.weight * w
            top = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
            total = sum(w for _, w in top) or 1.0
            for gname, w in top:
                if gname not in made:
                    made[gname] = ob.vertex_groups.new(name=gname)
                made[gname].add([v.index], w / total, "REPLACE")
        # Mirror counterpart groups must exist for the mirror modifier's
        # vertex-group flipping to take (shoulder.l -> shoulder.r on the
        # mirrored eye); a missing counterpart silently keeps the wrong side.
        existing = {g.name for g in ob.vertex_groups}
        for gname in sorted(existing):
            flip = gname[:-2] + ".r" if gname.endswith(".l") \
                else gname[:-2] + ".l" if gname.endswith(".r") else None
            if flip and flip not in existing:
                ob.vertex_groups.new(name=flip)
        print(f"auto-transferred body weights onto decal {ob.name}")

    # Painted (or just-transferred) weights ride through the evaluated
    # mesh's deform layer; the mesh's group indices refer to the ORIGINAL
    # object's vertex_groups order, so those groups must be recreated on
    # the new object in the same order.
    painted_groups = [g.name for g in ob.vertex_groups]

    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ob_eval, preserve_all_data_layers=True,
                                         depsgraph=dg)
    me.transform(ob_eval.matrix_world)
    name = ob.name
    bpy.data.objects.remove(ob)
    new_ob = bpy.data.objects.new(name, me)
    scn.collection.objects.link(new_ob)
    if painted_groups:
        for gname in painted_groups:
            new_ob.vertex_groups.new(name=gname)
        print(f"converted decal {name}: {len(me.vertices)} verts, "
              f"painted weights ({', '.join(painted_groups)})")
    else:
        vg = new_ob.vertex_groups.new(name=deform_name)
        vg.add(range(len(me.vertices)), 1.0, "REPLACE")
        print(f"converted decal {name}: {len(me.vertices)} verts -> 100% {deform_name}")
    mod = new_ob.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    # Must be a child of the rig object: the USD exporter only authors skel
    # bindings for meshes that land inside the SkelRoot's prim subtree.
    new_ob.parent = rig
    new_ob.matrix_parent_inverse = rig.matrix_world.inverted()
    converted.append(new_ob)
rig.data.pose_position = "POSE"
bpy.context.view_layer.update()

meshes = skinned + converted

# ── Body material fallback: the working file often has no material assigned
for ob in meshes:
    if not any(slot.material for slot in ob.material_slots):
        mat = bpy.data.materials.new("PipoSkin")
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (*PIPO_PINK, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
        ob.data.materials.append(mat)
        print(f"assigned PipoSkin pink to {ob.name}")

for ob in meshes:
    bpy.context.view_layer.objects.active = ob

    # Blender 5 corrects face winding itself when applying mirrored
    # (negative-scale) transforms — do NOT flip normals afterwards.
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    for mod in list(ob.modifiers):
        if mod.type == "SUBSURF":
            mod.levels = min(mod.levels, 3)  # keep vert count mobile-friendly
            bpy.ops.object.modifier_apply(modifier=mod.name)
            print(f"applied subsurf on {ob.name}: {len(ob.data.vertices)} verts")

    # smooth normals do more for perceived smoothness than poly count
    bpy.ops.object.shade_smooth()

bpy.ops.object.select_all(action="DESELECT")
for ob in meshes + [rig]:
    ob.select_set(True)
bpy.context.view_layer.objects.active = rig

# Sidecar: world-space matrices for every bone, every frame. Blender's USD
# export authors joint-local animation against the FULL bone hierarchy while
# listing only deform bones, so the recomposed transforms are garbage in any
# UsdSkel consumer. fix_usdz_for_realitykit.py rebuilds bind/rest/anim data
# from this ground truth instead. Shape key values ride along per frame —
# they're driver-driven (knee correctives), which the exporter won't sample.
def mat_list(m):
    return [[m[r][c] for c in range(4)] for r in range(4)]

bones = {}
for b in rig.data.bones:
    bones[b.name] = {
        "parent": b.parent.name if b.parent else None,
        "deform": b.use_deform,
        "rest_world": mat_list(rig.matrix_world @ b.matrix_local),
        "frames": [],
    }
blendshapes = {}
for ob in meshes:
    if ob.data.shape_keys:
        blendshapes[ob.name] = {kb.name: [] for kb in ob.data.shape_keys.key_blocks
                                if kb != ob.data.shape_keys.reference_key}

for f in range(scn.frame_start, scn.frame_end + 1):
    scn.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    rig_eval = rig.evaluated_get(dg)
    world = rig_eval.matrix_world
    for pb in rig_eval.pose.bones:
        bones[pb.name]["frames"].append(mat_list(world @ pb.matrix))
    for ob in meshes:
        if ob.name in blendshapes:
            ob_eval = ob.evaluated_get(dg)
            for kb in ob_eval.data.shape_keys.key_blocks:
                if kb.name in blendshapes[ob.name]:
                    blendshapes[ob.name][kb.name].append(round(kb.value, 5))
scn.frame_set(scn.frame_start)

with open(OUT + ".skel.json", "w") as fp:
    json.dump({"fps": scn.render.fps, "start": scn.frame_start,
               "end": scn.frame_end, "bones": bones,
               "blendshapes": blendshapes}, fp)
print(f"wrote sidecar {OUT}.skel.json ({len(bones)} bones, "
      f"{sum(len(v) for v in blendshapes.values())} shape key tracks)")

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
    "export_lights": False,
    "convert_world_material": False,
}
kwargs = {k: v for k, v in wanted.items() if k in op_props}
bpy.ops.wm.usd_export(**kwargs)
print("exported", OUT)
