// --- tree wind, pixel half (windgen.py inserts this into a translated M2 pixel module's `wind` variant) ---
// Paired only with a vertex `wind` variant, which writes windLeaf (the fragment's world position, 1 on
// a leaf) and windTree (the tree's gust, its seed, the vertex's canopy occlusion).
//   - On a leaf, never on bark, the texture drifts a little inside the card: a drift along the wind and
//     a gentle swirl, both a WORLD-space vector, the tree's one phase, projected onto each card's own
//     UVs through the screen-space derivatives -- so crossing cards drift the same way in the world,
//     and no card bobs up and down. windgen.py adds `v1.xy += WindLeafUv(v1.xy);` after
//     `vec4 v1 = texcoord0;`.
//   - The canopy's occlusion: a leaf deep in the canopy or under it darker than one on its outer shell
//     (the consumer's tree analysis), the bark under the canopy a little, by look.w, gone with
//     the distance. A material term, baked from the mesh, not a light. windgen.py adds
//     `oC0.rgb *= WindCanopyShade();` before the module's alpha test.
//   - The far crossfade: past the distance a consumer's octahedral impostors take over, the tree's
//     pixels drop out by a screen-space dither the impostor complements (WindFarFade, in WindCanopyShade).
//   - The leaves' translucency: a leaf between the eye and the sun lets it through -- warm, brighter, by
//     how straight the eye looks into the sun and how far out on the canopy the leaf is (inner leaves,
//     behind others, get less). Inside WindCanopyShade, so the same insertion carries it.

layout(location = 11) in vec4 windLeaf;
layout(location = 12) in vec4 windTree;
layout(std140, set = 3, binding = 0) uniform WindBlock {
    uvec4 header;
    mat4  viewInv;
    mat4  view;
    vec4  wind;
    vec4  phase;
    vec4  shape;
    vec4  time;      // clock (s, wraps every 60), strength scale, on, the branches' sway
    vec4  look;      // sway, rustle, drift (texture), canopy occlusion
    vec4  sun;       // towards the sun (world), the leaves' translucency
    vec4  sunColour; // the direct light's colour, 0
    vec4  fade;      // the far crossfade to the impostors: from, to (yd), on, 0
} windU;

/// The far crossfade (a consumer's impostors take over): past fade.x the tree's pixels drop out by a
/// screen-space dither, all of them past fade.y; the impostor draws the complement with the same dither.
void WindFarFade(float dist)
{
    if (windU.fade.z < 0.5 || windTree.w < 0.5) return;   // windTree.w: the vertex half bent it (windLeaf is its world point)
    float share = smoothstep(windU.fade.x, max(windU.fade.y, windU.fade.x + 0.01), dist);
    if (share <= 0.0) return;
    float dither = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    if (dither < share) discard;
}

vec2 WindLeafUv(vec2 uv)
{
    // The derivatives first, in uniform flow.
    vec3 px = dFdx(windLeaf.xyz), py = dFdy(windLeaf.xyz);
    vec2 ux = dFdx(uv), uy = dFdy(uv);
    if (windLeaf.w < 0.5 || windU.time.z < 0.5) return vec2(0.0);
    const float turn = 6.2831853 / 60.0;
    float t = windU.time.x * turn;
    float g = windTree.x, seed = windTree.y;
    // The world vector: downwind, breathing with the gust; a slow swirl about the vertical.
    vec3 down  = vec3(windU.wind.xy, 0.0) * (0.6 + 0.4 * sin(t * 6.0 + seed * 6.2831853));
    float sw   = t * 4.0 + seed * 6.2831853;
    vec3 swirl = vec3(cos(sw), sin(sw), 0.0) * 0.35;
    vec3 W = (down + swirl) * (0.06 * g * windU.look.z);            // yards
    // Onto this card: W = a dP/dx + b dP/dy (least squares), the UV offset a dUV/dx + b dUV/dy.
    float xx = dot(px, px), xy = dot(px, py), yy = dot(py, py);
    float det = xx * yy - xy * xy;
    if (abs(det) < 1e-12) return vec2(0.0);
    float bx = dot(px, W), by = dot(py, W);
    float a = (bx * yy - by * xy) / det, b = (by * xx - bx * xy) / det;
    vec2 d = a * ux + b * uy;
    // Never more than a few texels' worth, however grazing the card is to the eye.
    float l = length(d);
    return l > 0.02 ? d * (0.02 / l) : d;
}

/// The canopy's occlusion here, by its strength (look.w), faded out from 40 to 120 yd. Rescaled so the
/// canopy keeps its brightness on the whole: the outer shell a little lighter, the core darker.
float WindCanopyShade()
{
    WindFarFade(distance(windLeaf.xyz, windU.viewInv[3].xyz));
    if (windU.time.z < 0.5 || (windU.look.w <= 0.0 && windU.sun.w <= 0.0)) return 1.0;
    float dist = distance(windLeaf.xyz, windU.viewInv[3].xyz);
    float k = clamp(min(windU.look.w, 2.0) * (1.0 - smoothstep(40.0, 120.0, dist)), 0.0, 1.0);
    float shade = mix(1.0, clamp(windTree.z, 0.0, 1.0) * (windLeaf.w > 0.5 ? 1.1 : 1.0), k);
    if (windLeaf.w > 0.5 && windU.sun.w > 0.0 && windU.sun.z > -0.1)
    {
        vec3  toLeaf = normalize(windLeaf.xyz - windU.viewInv[3].xyz);
        float back   = pow(clamp(dot(toLeaf, normalize(windU.sun.xyz)), 0.0, 1.0), 6.0);
        float outer  = clamp(windTree.z, 0.0, 1.0);   // the canopy's occlusion: 1 on the shell
        shade *= 1.0 + windU.sun.w * back * outer * outer * 0.9;
    }
    return shade;
}

// --- end of tree wind, pixel half -------------------------------------------------------------------
