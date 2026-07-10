#include <metal_stdlib>
#include <RealityKit/RealityKit.h>

using namespace metal;

// Depth bias shared by the outline hull and the face features. The floor is
// in model units: limb-junction depth gaps are model-space geometry, so the
// bias must not shrink with viewing distance. The ceiling keeps it from
// punching through Pipo's thin limbs (~0.3 model units across).
inline float pipoHullPull(float offset)
{
    return clamp(offset * 0.75, 0.12, 0.18);
}

// Inverted-hull outline: vertices pushed outward along their normals, plus a
// small pull TOWARD the camera. Where a limb meets the body the depth gap
// between the two surfaces approaches zero and the hull rim would lose the
// depth test (interior lines fade at soft angles); the toward-camera bias
// keeps a band of ink alive along those junctions.
// custom_parameter() = (offset in model units, camera position in model space),
// updated per frame by ToonStyle for constant on-screen thickness.
[[visible]]
void pipoOutlineGeometry(realitykit::geometry_parameters params)
{
    float4 cp = params.uniforms().custom_parameter();
    float offset = cp.x;
    float3 cameraModel = cp.yzw;
    float3 n = normalize(params.geometry().normal());
    float3 toward = normalize(cameraModel - params.geometry().model_position());
    params.geometry().set_model_position_offset(n * offset + toward * pipoHullPull(offset));
}

// Eyes/mouth in toon mode: pulled toward the camera strictly more than the
// hull so the body's inflated shell can never surface around them (halo).
// A pull along the view ray does not move them on screen.
[[visible]]
void pipoFacePullGeometry(realitykit::geometry_parameters params)
{
    float4 cp = params.uniforms().custom_parameter();
    float pull = pipoHullPull(cp.x) + 0.12;
    float3 cameraModel = cp.yzw;
    float3 toward = normalize(cameraModel - params.geometry().model_position());
    params.geometry().set_model_position_offset(toward * pull);
}

// Flat unlit face features — Material.002's dark plum.
[[visible]]
void pipoFaceSurface(realitykit::surface_parameters params)
{
    half3 c = half3(0.126h, 0.01h, 0.027h);
    params.surface().set_base_color(c);
    params.surface().set_emissive_color(c);
}

// Flat ink color — darker wine-pink derived from Pipo's body color.
[[visible]]
void pipoOutlineSurface(realitykit::surface_parameters params)
{
    half3 ink = half3(0.36h, 0.015h, 0.12h);
    params.surface().set_base_color(ink);
    params.surface().set_emissive_color(ink);
}
