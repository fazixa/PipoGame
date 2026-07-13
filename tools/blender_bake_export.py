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
import bmesh
import json
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
ACTION_NAME, OUT = argv[0], argv[1]

RIG_NAME = "rig"
SUBSURF_CAP = 3

# Vertex groups (named exactly, painted by hand in Blender) marking regions
# that should get ZERO outline-hull inflation in toon mode — e.g. the mouth
# interior, a concave cavity where full inflation makes the hull poke through
# the surrounding skin at their shared boundary. Weight ramps smoothly back
# to 1 (full inflation) over HULL_TAPER_RINGS edge-steps from the seed.
HULL_TAPER_GROUPS = ("mouth_interior",)
HULL_TAPER_RINGS = 6

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


def slot_color(material):
    if material and material.use_nodes:
        for n in material.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED":
                c = n.inputs["Base Color"].default_value
                r = n.inputs["Roughness"].default_value
                return [c[0], c[1], c[2]], r
    return [0.8, 0.8, 0.8], 0.5


def slot_colors(ob):
    """One (color, roughness) per material slot — meshes like the body
    carry more than one (e.g. skin + mouth interior), and collapsing them
    to a single color makes the second material invisible."""
    if not ob.material_slots:
        return [([0.8, 0.8, 0.8], 0.5)]
    return [slot_color(slot.material) for slot in ob.material_slots]


def evaluated_mesh(ob):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return ob.evaluated_get(depsgraph), depsgraph


def hull_taper_weights(ob):
    """One float per BASE-mesh vertex (same order/count as ob.data.vertices):
    0 at any HULL_TAPER_GROUPS seed, ramping to 1 over HULL_TAPER_RINGS edge
    hops, 1 everywhere else. None if the mesh has none of those groups.
    Only valid to index directly against the evaluated/baked mesh when the
    mesh has no vertex-count-changing modifier (e.g. Subsurf) — true here
    since only the Armature modifier remains on deforming meshes.
    """
    seed_group_indices = [g.index for g in ob.vertex_groups if g.name in HULL_TAPER_GROUPS]
    if not seed_group_indices:
        return None

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    deform_layer = bm.verts.layers.deform.verify()

    seeds = [
        v.index for v in bm.verts
        if any(gi in v[deform_layer] and v[deform_layer][gi] > 0.5 for gi in seed_group_indices)
    ]

    dist = {i: 0 for i in seeds}
    frontier = list(seeds)
    ring = 0
    while frontier and ring < HULL_TAPER_RINGS:
        ring += 1
        nxt = []
        for vi in frontier:
            for e in bm.verts[vi].link_edges:
                other = e.other_vert(bm.verts[vi]).index
                if other not in dist:
                    dist[other] = ring
                    nxt.append(other)
        frontier = nxt

    weights = [min(1.0, dist.get(v.index, HULL_TAPER_RINGS) / HULL_TAPER_RINGS)
               for v in bm.verts]
    bm.free()
    return weights


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
    ob = bpy.data.objects[name]
    ob_eval, depsgraph = evaluated_mesh(ob)
    me = ob_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    materials = slot_colors(ob)
    counts, indices, face_materials = [], [], []
    for p in me.polygons:
        counts.append(len(p.vertices))
        indices.extend(p.vertices)
        face_materials.append(p.material_index)
    positions, normals = [], []
    for v in me.vertices:
        positions.extend(rounded(v.co))
        normals.extend(rounded(v.normal))
    hull_weight = hull_taper_weights(ob)
    if hull_weight is not None:
        assert len(hull_weight) == len(me.vertices), \
            f"{name}: hull taper vertex count mismatch — a modifier is " \
            f"changing vertex count (e.g. Subsurf); taper needs 1:1 indexing"
    meshes[name] = {
        "materials": materials, "face_materials": face_materials,
        "counts": counts, "indices": indices,
        "rest": positions, "normals": normals, "frames": [],
        "hull_weight": hull_weight,
    }
    ob_eval.to_mesh_clear()
    material_count = len(set(face_materials))
    taper_note = "with hull taper" if hull_weight is not None else "no hull taper"
    print(f"{name}: {len(positions)//3} verts, {len(counts)} faces, "
          f"{material_count} material(s) in use, {taper_note}")

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
