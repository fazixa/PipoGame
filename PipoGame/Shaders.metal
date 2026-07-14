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
//
// The push is additionally scaled by the mesh's baked vertex color (see
// build_baked_usdz.py's "hullWeight" -> displayColor, sourced from Blender
// vertex groups named in HULL_TAPER_GROUPS, e.g. "mouth_interior"). Concave
// regions like the mouth interior now share vertices directly with the
// surrounding skin (no longer separate mesh entities) — full inflation
// there tears the hull away from its connected neighbors and pokes it
// through the skin at their shared boundary. Weight 0 at the seam ramps
// smoothly to 1 a few rings out, so the hull settles to nothing exactly at
// the seam instead of cracking.
//
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
    float taper = params.geometry().color().r;
    params.geometry().set_model_position_offset(
        (n * offset + toward * pipoHullPull(offset)) * taper);
}

// Eyes/mouth in toon mode: no geometry offset. Eyes/mouth were originally
// pulled toward the camera to stop the outline hull's inflated shell from
// surfacing as a halo ring around them — but the outline clone already
// assigns face slots a fully transparent material (see ToonStyle.apply),
// which solves that robustly on its own. The pull became actively harmful
// once eyes/mouth stopped being separate mesh entities and became material
// slots on the SAME continuous mesh as the skin (GeomSubsets): pulling those
// vertices forward tears them away from their connected neighbors, pushing
// e.g. the mouth cavity out past the surrounding face surface.
[[visible]]
void pipoFacePullGeometry(realitykit::geometry_parameters params)
{
}

// Flat unlit face features — Material.002's dark plum.
[[visible]]
void pipoFaceSurface(realitykit::surface_parameters params)
{
    half3 c = half3(0.126h, 0.01h, 0.027h);
    params.surface().set_base_color(c);
    params.surface().set_emissive_color(c);
}

// Layered value noise ("clouds" style) — cheap, dependency-free stand-in for
// Perlin noise. Smooth-interpolated hash lattice, good enough for a subtle
// surface bump; not true gradient (Perlin) noise, but visually equivalent at
// this scale.
inline float pipoHash3(float3 p)
{
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

inline float pipoNoise3(float3 p)
{
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = pipoHash3(i + float3(0, 0, 0));
    float n100 = pipoHash3(i + float3(1, 0, 0));
    float n010 = pipoHash3(i + float3(0, 1, 0));
    float n110 = pipoHash3(i + float3(1, 1, 0));
    float n001 = pipoHash3(i + float3(0, 0, 1));
    float n101 = pipoHash3(i + float3(1, 0, 1));
    float n011 = pipoHash3(i + float3(0, 1, 1));
    float n111 = pipoHash3(i + float3(1, 1, 1));
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z);
}

// Subtle surface bump: nudges each vertex along its own normal by a cloud-
// noise value sampled at its model-space position, so the pattern is fixed
// to the mesh (rides along with blend-shape deformation) rather than
// swimming in world space.
//
// custom_parameter() = (noise scale — bumps per model unit, displacement
// strength in model units, 0, 0).
[[visible]]
void pipoNoiseDisplaceGeometry(realitykit::geometry_parameters params)
{
    float4 cp = params.uniforms().custom_parameter();
    float scale = cp.x;
    float strength = cp.y;
    float3 pos = params.geometry().model_position();
    float3 n = normalize(params.geometry().normal());
    float noiseValue = pipoNoise3(pos * scale) * 2.0 - 1.0;
    params.geometry().set_model_position_offset(n * noiseValue * strength);
}

// Height-map bump: nudges each vertex along its own normal by a sample from
// a real grayscale texture (bound as CustomMaterial's "custom" slot), read
// at that vertex's UV — an alternative to pipoNoiseDisplaceGeometry's
// procedural noise when you want an actual painted/photographed height map
// (e.g. a fingerprint/pebble texture) instead of tunable noise parameters.
// UV tiling and displacement strength are both runtime-adjustable so the
// same shader works at any scale without recompiling.
//
// custom_parameter() = (UV tiling repeats, displacement strength in model
// units, 0, 0).
[[visible]]
void pipoHeightMapDisplaceGeometry(realitykit::geometry_parameters params)
{
    float4 cp = params.uniforms().custom_parameter();
    float tiling = cp.x;
    float strength = cp.y;

    float2 uv = params.geometry().uv0() * tiling;
    uv = fract(uv);
    // USD/USDZ textures have a bottom-left origin; Metal samples top-left.
    uv.y = 1.0 - uv.y;

    constexpr sampler heightSampler(coord::normalized, address::repeat,
                                    filter::linear, mip_filter::linear);
    half4 sample = params.textures().custom().sample(heightSampler, uv);
    float height = float(sample.r) * 2.0 - 1.0;

    float3 n = normalize(params.geometry().normal());
    params.geometry().set_model_position_offset(n * height * strength);
}

// Flat ink color — darker wine-pink derived from Pipo's body color. Faded
// out by the same taper weight the geometry modifier uses (see
// pipoOutlineGeometry): scaling the push alone still leaves these faces at
// full opaque ink, just barely puffed out, which reads as a thin fully-
// colored ring tracing the taper boundary instead of fading away with it.
[[visible]]
void pipoOutlineSurface(realitykit::surface_parameters params)
{
    half3 ink = half3(0.36h, 0.015h, 0.12h);
    params.surface().set_base_color(ink);
    params.surface().set_emissive_color(ink);
    half taper = half(params.geometry().color().r);
    params.surface().set_opacity(taper);
}
