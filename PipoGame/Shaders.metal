#include <metal_stdlib>
#include <RealityKit/RealityKit.h>

using namespace metal;

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

// Signed soft-noise displacement along the vertex normal — the shared
// "Blender displace modifier" all toon-mode surfaces apply, sampled at the
// model-space position so the pattern is fixed to the mesh (rides along
// with the pose) rather than swimming in world space. Body, face decals,
// and the outline hull all evaluate this SAME field with the same
// parameters, so the hull and decals stay wrapped around the bumped body.
inline float3 pipoDisplacement(float3 pos, float3 n, float scale, float strength)
{
    float noiseValue = pipoNoise3(pos * scale) * 2.0 - 1.0;
    return n * noiseValue * strength;
}

// Inverted-hull outline: vertices pushed outward along their normals by the
// line width, on top of the shared soft-noise displacement (which is NOT
// scaled by the taper — it must match the body's own displacement exactly).
//
// The push is scaled by the mesh's per-vertex taper weight (displayColor.r,
// authored by fix_usdz_for_realitykit.py — full white on the current model,
// weighted by build_baked_usdz.py's HULL_TAPER_GROUPS on the old baked one)
// so concave taper regions can fade the hull to nothing at a seam.
//
// custom_parameter() = (line width in model units, noise scale, noise
// strength in model units, 0). All static — set once at material creation.
[[visible]]
void pipoOutlineGeometry(realitykit::geometry_parameters params)
{
    float4 cp = params.uniforms().custom_parameter();
    float offset = cp.x;
    float3 pos = params.geometry().model_position();
    float3 n = normalize(params.geometry().normal());
    float taper = params.geometry().color().r;
    // Pencil wobble: line weight varies along the stroke, and the noise
    // phase steps at ~10 fps so the line "boils" like hand-drawn frames.
    float t = floor(params.uniforms().time() * 10.0) / 10.0;
    float wobble = pipoNoise3(pos * 18.0 + t * 3.7);
    float width = offset * mix(0.55, 1.45, wobble);
    params.geometry().set_model_position_offset(n * width * taper);
}

// Eyes/mouth in toon mode: lifted off the skin so the body's wobble
// displacement (up to ~0.45 x line width outward) can never rise through
// them. A uniform TRANSLATION along face-forward (-Y in model space), not
// a push along vertex normals — normals inflate the patch (every vertex
// moves apart, the feature reads bigger); a translation preserves the
// silhouette exactly and just floats the sticker forward.
[[visible]]
void pipoFacePullGeometry(realitykit::geometry_parameters params)
{
    params.geometry().set_model_position_offset(float3(0.0, -0.02, 0.0));
}

// Flat unlit face features — Material.002's dark plum.
[[visible]]
void pipoFaceSurface(realitykit::surface_parameters params)
{
    half3 c = half3(0.126h, 0.01h, 0.027h);
    params.surface().set_base_color(c);
    params.surface().set_emissive_color(c);
}

// Toon body displacement: the SAME signed wobble field the outline hull's
// width uses (same noise, same 10 fps boil clock), minus the constant line
// offset — the body surface boils in lockstep with the outline, so the
// contour wanders like a hand-drawn edge.
// custom_parameter() = (line width in model units, 0, 0, 0).
[[visible]]
void pipoToonBodyGeometry(realitykit::geometry_parameters params)
{
    float offset = params.uniforms().custom_parameter().x;
    float3 pos = params.geometry().model_position();
    float3 n = normalize(params.geometry().normal());
    float t = floor(params.uniforms().time() * 10.0) / 10.0;
    float wobble = pipoNoise3(pos * 18.0 + t * 3.7);
    params.geometry().set_model_position_offset(
        n * offset * (mix(0.55, 1.45, wobble) - 1.0));
}

// Toon body: flat unlit color taken from the material's own base color
// tint (set per slot by ToonStyle, preserving each part's original color).
[[visible]]
void pipoToonBodySurface(realitykit::surface_parameters params)
{
    half3 c = half3(params.material_constants().base_color_tint());
    params.surface().set_base_color(c);
    params.surface().set_emissive_color(c);
}

// Subtle surface bump: nudges each vertex along its own normal by a cloud-
// noise value sampled at its model-space position. Used as the toon body's
// displacement — the same field the hull and face decals apply (see
// pipoDisplacement).
//
// custom_parameter() = (0, noise scale — bumps per model unit, displacement
// strength in model units, 0) — same y/z layout as the other toon shaders.
[[visible]]
void pipoNoiseDisplaceGeometry(realitykit::geometry_parameters params)
{
    float4 cp = params.uniforms().custom_parameter();
    float3 pos = params.geometry().model_position();
    float3 n = normalize(params.geometry().normal());
    params.geometry().set_model_position_offset(
        pipoDisplacement(pos, n, cp.y, cp.z));
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
    // Pencil grain: high-frequency noise = graphite/paper texture eating
    // into the stroke, stepping at ~10 fps in lockstep with the width
    // wobble (same boil clock).
    float3 pos = params.geometry().model_position();
    float t = floor(params.uniforms().time() * 10.0) / 10.0;
    float grain = pipoNoise3(pos * 60.0 + t * 11.0);
    // Binary grain: each speck is fully inked or fully clear (no gray
    // in-between) — reads as paper tooth. Threshold sets coverage:
    // ~15% of the stroke drops out.
    half pencil = half(step(0.15, grain));
    params.surface().set_opacity(taper * pencil);
}
