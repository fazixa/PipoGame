#!/usr/bin/env python3
"""Make a Blender-exported animated USDZ loadable and CORRECT in RealityKit.

Blender's USD exporter, with "only deform bones", produces broken UsdSkel:

1. Joint paths keep filtered-out control bones as phantom path segments.
   RealityKit requires every ancestor segment to be a listed joint and
   silently discards ALL animations otherwise (zero availableAnimations,
   no error).
2. Joint-local rest/animation transforms are authored relative to the
   ORIGINAL full bone hierarchy, while UsdSkel recomposes them against the
   nearest listed ancestor — the mesh explodes into ribbons.

This script rebuilds the skeleton from ground truth: a .skel.json sidecar
written by blender_export.py containing world-space matrices for every bone
at every frame. Joint order is preserved (mesh jointIndices depend on it);
paths are compacted, and bind/rest/anim transforms are recomputed in
consistent spaces.

Usage:  python3 fix_usdz_for_realitykit.py in.usdz out.usdz
        (expects in.usdz.skel.json next to the input)
Needs:  pip3 install usd-core
"""
import json
import os
import sys
import tempfile
import zipfile

from pxr import Usd, UsdSkel, Sdf, UsdUtils, Vt, Gf


def gf(rows):
    """Blender (column-vector) matrix rows -> Gf.Matrix4d (row-vector)."""
    return Gf.Matrix4d(*[rows[c][r] for r in range(4) for c in range(4)])


def decompose(m):
    """Gf.Matrix4d -> (Vec3f translation, Quatf rotation, Vec3h scale)."""
    t = Gf.Transform(m)
    q = Gf.Quatf(t.GetRotation().GetQuat())
    return (Gf.Vec3f(t.GetTranslation()), q, Gf.Vec3h(t.GetScale()))


def find_prims(stage, type_name):
    return [p for p in stage.Traverse() if p.GetTypeName() == type_name]


def main(src, dst):
    sidecar = src + ".skel.json"
    with open(sidecar) as fp:
        data = json.load(fp)
    bones = data["bones"]

    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(src) as z:
        z.extractall(tmpdir)
    usdc = next(f for f in os.listdir(tmpdir) if f.endswith((".usdc", ".usda", ".usd")))
    stage = Usd.Stage.Open(os.path.join(tmpdir, usdc))
    layer = stage.GetRootLayer()

    skelroot = find_prims(stage, "SkelRoot")[0]
    skeleton = find_prims(stage, "Skeleton")[0]
    animation = find_prims(stage, "SkelAnimation")[0]

    # --- restructure: anim as sibling of skeleton, SkelRoot as default root
    anim_dst = skelroot.GetPath().AppendChild("Anim")
    if animation.GetPath() != anim_dst:
        Sdf.CopySpec(layer, animation.GetPath(), layer, anim_dst)
        stage.RemovePrim(animation.GetPath())
    if skelroot.GetPath().pathElementCount > 1:
        top = Sdf.Path("/" + skelroot.GetName())
        Sdf.CopySpec(layer, skelroot.GetPath(), layer, top)
        # Materials often live outside the SkelRoot (e.g. /root/_materials);
        # rescue them to a top-level scope before the old tree is deleted.
        material_map = {}
        for mat in find_prims(stage, "Material"):
            if not mat.GetPath().HasPrefix(skelroot.GetPath()):
                new_path = Sdf.Path("/Materials").AppendChild(mat.GetName())
                if not layer.GetPrimAtPath("/Materials"):
                    Sdf.PrimSpec(layer, "Materials", Sdf.SpecifierDef, "Scope")
                Sdf.CopySpec(layer, mat.GetPath(), layer, new_path)
                material_map[mat.GetName()] = new_path
        stage.RemovePrim(skelroot.GetPath().GetPrefixes()[0])
        stage.SetDefaultPrim(stage.GetPrimAtPath(top))
        skelroot = stage.GetPrimAtPath(top)
        skeleton = find_prims(stage, "Skeleton")[0]
        anim_dst = skelroot.GetPath().AppendChild("Anim")

        from pxr import UsdShade
        for mesh in find_prims(stage, "Mesh"):
            api = UsdShade.MaterialBindingAPI(mesh)
            bound = api.GetDirectBinding().GetMaterialPath()
            if bound and bound.name in material_map:
                api.Bind(UsdShade.Material(
                    stage.GetPrimAtPath(material_map[bound.name])))

    # Blender exports Z-up geometry; the up-axis conversion lived on ancestor
    # prims we just stripped. Re-author it on the SkelRoot itself.
    from pxr import UsdGeom
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    xf = UsdGeom.Xformable(skelroot)
    xf.ClearXformOpOrder()
    xf.AddRotateXOp().Set(-90.0)
    animation = stage.GetPrimAtPath(anim_dst)
    skel_api = UsdSkel.Skeleton(skeleton)
    anim_api = UsdSkel.Animation(animation)
    UsdSkel.BindingAPI(skeleton).GetAnimationSourceRel().SetTargets([anim_dst])
    for mesh in find_prims(stage, "Mesh"):
        b = UsdSkel.BindingAPI(mesh)
        if b.GetSkeletonRel().GetTargets():
            b.GetSkeletonRel().SetTargets([skeleton.GetPath()])

    # --- strip Blender/ARP userProperties
    for prim in stage.Traverse():
        for prop in list(prim.GetAuthoredProperties()):
            if prop.GetName().startswith("userProperties:"):
                prim.RemoveProperty(prop.GetName())

    # --- map USD joint tokens (sanitized leaves) back to Blender bone names
    old_joints = list(skel_api.GetJointsAttr().Get())
    sanitized = {name.replace(".", "_").replace(":", "_"): name for name in bones}
    if len(sanitized) != len(bones):
        sys.exit("bone name collision after sanitizing '.' -> '_'")
    try:
        leaf_bone = [sanitized[j.split("/")[-1]] for j in old_joints]
    except KeyError as e:
        sys.exit(f"joint leaf {e} not found among Blender bones")

    # Blender's only_deform_bones still exports every ANCESTOR of a deform
    # bone as a real joint, so control bones ride along (129 joints for 66
    # deform bones on the ARP rig). The app iterates jointTransforms — the
    # pose-reactive footprint, and procedural animation later — so phantom
    # control joints aren't just dead weight, they're wrong data. Strip to
    # true deform bones and remap every mesh's jointIndices (skeleton-order
    # indices; none of these meshes author a per-mesh skel:joints list).
    keep_mask = [bones[n].get("deform", True) for n in leaf_bone]
    if not all(keep_mask):
        remap, kept = {}, []
        for i, (n, k) in enumerate(zip(leaf_bone, keep_mask)):
            if k:
                remap[i] = len(kept)
                kept.append(n)
        for mesh in find_prims(stage, "Mesh"):
            b = UsdSkel.BindingAPI(mesh)
            pv = b.GetJointIndicesPrimvar()
            if not pv or b.GetJointsAttr().Get() is not None:
                continue
            wts = b.GetJointWeightsPrimvar().Get()
            new_idx = []
            for j, w in zip(pv.Get(), wts):
                if j in remap:
                    new_idx.append(remap[j])
                elif w == 0.0:
                    new_idx.append(0)
                else:
                    sys.exit(f"{mesh.GetPath()} weights control bone "
                             f"{leaf_bone[j]!r} (w={w})")
            pv.Set(Vt.IntArray(new_idx))
        print(f"stripped {len(leaf_bone) - len(kept)} control joints, "
              f"{len(kept)} deform joints kept")
        leaf_bone = kept
    joint_set = set(leaf_bone)

    def deform_parent(name):
        p = bones[name]["parent"]
        while p is not None and p not in joint_set:
            p = bones[p]["parent"]
        return p

    # --- rebuild joint paths (same order!) through listed ancestors only
    index = {name: i for i, name in enumerate(leaf_bone)}
    newpaths = [None] * len(leaf_bone)

    def path_for(name):
        i = index[name]
        if newpaths[i] is None:
            leaf = name.replace(".", "_").replace(":", "_")
            parent = deform_parent(name)
            newpaths[i] = leaf if parent is None else path_for(parent) + "/" + leaf
        return newpaths[i]

    for name in leaf_bone:
        path_for(name)

    # --- rebuild bind/rest/anim transforms in consistent (world) space
    rest_world = {n: gf(bones[n]["rest_world"]) for n in leaf_bone}
    binds = Vt.Matrix4dArray([rest_world[n] for n in leaf_bone])

    def local(world, parent_name, frame=None):
        if parent_name is None:
            return world
        pw = rest_world[parent_name] if frame is None \
            else gf(bones[parent_name]["frames"][frame])
        return world * pw.GetInverse()

    rests = Vt.Matrix4dArray([
        local(rest_world[n], deform_parent(n)) for n in leaf_bone
    ])

    skel_api.GetJointsAttr().Set(Vt.TokenArray(newpaths))
    skel_api.GetBindTransformsAttr().Set(binds)
    skel_api.GetRestTransformsAttr().Set(rests)

    for name in ("translations", "rotations", "scales"):
        spec = layer.GetPrimAtPath(animation.GetPath()).attributes[name]
        spec.ClearDefaultValue()
        for t in layer.ListTimeSamplesForPath(spec.path):
            layer.EraseTimeSample(spec.path, t)

    anim_api.GetJointsAttr().Set(Vt.TokenArray(newpaths))
    start, end = data["start"], data["end"]
    for f in range(0, end - start + 1):
        ts, rs, ss = [], [], []
        for n in leaf_bone:
            world = gf(bones[n]["frames"][f])
            t, q, s = decompose(local(world, deform_parent(n), frame=f))
            ts.append(t); rs.append(q); ss.append(s)
        code = float(start + f)
        anim_api.GetTranslationsAttr().Set(Vt.Vec3fArray(ts), code)
        anim_api.GetRotationsAttr().Set(Vt.QuatfArray(rs), code)
        anim_api.GetScalesAttr().Set(Vt.Vec3hArray(ss), code)

    changed = sum(1 for a, b in zip(old_joints, newpaths) if a != b)
    print(f"rebuilt skeleton: {changed}/{len(newpaths)} paths changed, "
          f"{end - start + 1} frames re-authored")

    # --- blend shape weights from sidecar ground truth. The shape keys are
    # driver-driven in Blender (knee correctives), and the USD exporter does
    # not sample drivers — the exported weights are static garbage.
    blendshapes = data.get("blendshapes") or {}
    if blendshapes:
        mesh_tokens = set()
        for mesh in find_prims(stage, "Mesh"):
            toks = UsdSkel.BindingAPI(mesh).GetBlendShapesAttr().Get()
            mesh_tokens.update(toks or [])
        tokens, tracks = [], []
        for mesh_name, keys in blendshapes.items():
            for kname, vals in keys.items():
                tok = "".join(c if c.isalnum() else "_" for c in kname)
                if tok not in mesh_tokens:
                    sys.exit(f"shape key {kname!r} -> token {tok!r} not bound "
                             f"on any mesh (bound: {sorted(mesh_tokens)})")
                tokens.append(tok)
                tracks.append(vals)
        anim_api.GetBlendShapesAttr().Set(Vt.TokenArray(tokens))
        w = anim_api.GetBlendShapeWeightsAttr()
        for t in layer.ListTimeSamplesForPath(w.GetPath()):
            layer.EraseTimeSample(w.GetPath(), t)
        for f in range(end - start + 1):
            w.Set(Vt.FloatArray([tr[f] for tr in tracks]), float(start + f))
        print(f"authored {len(tokens)} blend shape weight tracks")

    # --- toon hull taper weight: the outline shader (Shaders.metal) scales
    # its push and ink opacity by geometry().color().r, fed from the
    # displayColor primvar. Blender declares the primvar but leaves it
    # unauthored, so the shader would read undefined values. This model has
    # no taper regions (eyes/mouth are separate decal meshes, not merged
    # GeomSubsets), so author full weight everywhere.
    from pxr import UsdGeom
    for mesh in find_prims(stage, "Mesh"):
        nverts = len(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray([Gf.Vec3f(1, 1, 1)] * nverts))

    fixed = os.path.join(tmpdir, "fixed.usdc")
    layer.Export(fixed)
    if not UsdUtils.CreateNewUsdzPackage(fixed, dst):
        sys.exit("usdz packaging failed")
    print("wrote", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
