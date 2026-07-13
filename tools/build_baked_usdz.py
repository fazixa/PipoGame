#!/usr/bin/env python3
"""Build a RealityKit-ready USDZ from a baked-vertex JSON (blender_bake_export.py).

Authors the file from scratch: meshes with constant-color materials, one
BlendShape per animation frame, and a SkelAnimation whose blendShapeWeights
play a one-hot sequence (USD's linear timeSample interpolation crossfades
between consecutive frames, i.e. smooth vertex playback).

A one-joint dummy skeleton exists purely because UsdSkel routes blend shape
weights through a skeleton's animationSource.

Usage:  python3 build_baked_usdz.py baked.json out.usdz
Needs:  pip3 install usd-core
"""
import json
import os
import sys
import tempfile

from pxr import Usd, UsdGeom, UsdSkel, UsdShade, Sdf, UsdUtils, Vt, Gf


def safe_name(name):
    # Blender object names (e.g. "finalF.002") aren't valid USD prim names
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe


def define_material(stage, path, color, roughness):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/pbr")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def main(src, dst):
    with open(src) as fp:
        data = json.load(fp)
    start, end, fps = data["start"], data["end"], data["fps"]
    nframes = end - start + 1
    shape_names = [f"f{i:03d}" for i in range(nframes)]

    tmp = os.path.join(tempfile.mkdtemp(), "baked.usdc")
    stage = Usd.Stage.CreateNew(tmp)
    stage.SetStartTimeCode(start)
    stage.SetEndTimeCode(end)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    root = UsdSkel.Root.Define(stage, "/Pipo")
    stage.SetDefaultPrim(root.GetPrim())
    # Blender geometry is Z-up
    UsdGeom.Xformable(root).AddRotateXOp().Set(-90.0)

    skel = UsdSkel.Skeleton.Define(stage, "/Pipo/Skel")
    ident = Vt.Matrix4dArray([Gf.Matrix4d(1)])
    skel.GetJointsAttr().Set(Vt.TokenArray(["root"]))
    skel.GetBindTransformsAttr().Set(ident)
    skel.GetRestTransformsAttr().Set(ident)
    skel.GetPrim().ApplyAPI(UsdSkel.BindingAPI)

    anim = UsdSkel.Animation.Define(stage, "/Pipo/Anim")
    anim.GetJointsAttr().Set(Vt.TokenArray(["root"]))
    anim.GetTranslationsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0)]), float(start))
    anim.GetRotationsAttr().Set(Vt.QuatfArray([Gf.Quatf(1)]), float(start))
    anim.GetScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)]), float(start))
    anim.GetBlendShapesAttr().Set(Vt.TokenArray(shape_names))
    for f in range(nframes):
        weights = [0.0] * nframes
        weights[f] = 1.0
        anim.GetBlendShapeWeightsAttr().Set(Vt.FloatArray(weights), float(start + f))
    UsdSkel.BindingAPI(skel).GetAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])

    for name, m in data["meshes"].items():
        rest = m["rest"]
        nverts = len(rest) // 3
        safe = safe_name(name)
        mesh_path = f"/Pipo/{safe}"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        points = Vt.Vec3fArray([Gf.Vec3f(*rest[i*3:i*3+3]) for i in range(nverts)])
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(m["counts"]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(m["indices"]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        nrm = m["normals"]
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray(
            [Gf.Vec3f(*nrm[i*3:i*3+3]) for i in range(nverts)]))
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))

        # Hull taper weight (see blender_bake_export.py's HULL_TAPER_GROUPS):
        # 0 at e.g. the mouth interior, ramping to 1 a few rings out. Carried
        # as a vertex color primvar since that's what the toon outline's
        # Metal geometry modifier can read per-vertex at runtime — it scales
        # the hull's outward inflation by this value so the ink shell tapers
        # to nothing right at a concave seam instead of poking through it.
        # Always authored (defaulting to a uniform 1 = full inflation) so the
        # shader never has to guess about a missing primvar.
        hull_weight = m.get("hull_weight") or [1.0] * nverts
        gprim = UsdGeom.Gprim(mesh)
        gprim.CreateDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(w, w, w) for w in hull_weight]))
        gprim.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)

        # Materials: one per slot the mesh actually uses, bound to their own
        # faces via GeomSubsets. A single whole-mesh material would make any
        # second material (e.g. a mouth interior distinct from the skin)
        # invisible — every face would render in whichever color came first.
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        face_materials = m.get("face_materials")
        materials = m.get("materials")
        if face_materials and materials and len(materials) > 1:
            groups = {}
            for face_index, mat_index in enumerate(face_materials):
                groups.setdefault(mat_index, []).append(face_index)
            for mat_index, face_indices in groups.items():
                color, roughness = materials[mat_index]
                mat = define_material(stage, f"/Materials/{safe}_{mat_index}", color, roughness)
                subset = UsdGeom.Subset.Define(stage, f"{mesh_path}/mat_{mat_index}")
                subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
                subset.CreateIndicesAttr().Set(Vt.IntArray(face_indices))
                subset.CreateFamilyNameAttr().Set("materialBind")
                UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(mat)
            UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)
        else:
            color, roughness = materials[0] if materials else ([0.8, 0.8, 0.8], 0.5)
            mat = define_material(stage, f"/Materials/{safe}", color, roughness)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

        # skeleton + blend shape binding
        mesh.GetPrim().ApplyAPI(UsdSkel.BindingAPI)
        binding = UsdSkel.BindingAPI(mesh.GetPrim())
        binding.GetSkeletonRel().SetTargets([skel.GetPrim().GetPath()])
        binding.CreateJointIndicesPrimvar(constant=False, elementSize=1).Set(
            Vt.IntArray([0] * nverts))
        binding.CreateJointWeightsPrimvar(constant=False, elementSize=1).Set(
            Vt.FloatArray([1.0] * nverts))
        binding.GetBlendShapesAttr().Set(Vt.TokenArray(shape_names))
        binding.GetBlendShapeTargetsRel().SetTargets(
            [Sdf.Path(f"{mesh_path}/{s}") for s in shape_names])
        for f, sname in enumerate(shape_names):
            fr = m["frames"][f]
            shape = UsdSkel.BlendShape.Define(stage, f"{mesh_path}/{sname}")
            offsets = Vt.Vec3fArray([
                Gf.Vec3f(fr[i*3] - rest[i*3],
                         fr[i*3+1] - rest[i*3+1],
                         fr[i*3+2] - rest[i*3+2]) for i in range(nverts)])
            shape.GetOffsetsAttr().Set(offsets)
        print(f"{name}: {nverts} verts, {nframes} shapes")

    stage.GetRootLayer().Save()
    if not UsdUtils.CreateNewUsdzPackage(tmp, dst):
        sys.exit("usdz packaging failed")
    print("wrote", dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
