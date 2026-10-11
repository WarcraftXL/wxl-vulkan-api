#version 450
// wxl-vulkan-api: the shadow replay's alpha cutout (src/capture/ShadowPipelines.cpp), in place of an
// alpha-tested draw's own pixel module when that module reads stage 0 at texcoord0: one texture
// read and the draw's alpha reference, nothing of the combiners, the lighting or the fog.
//
// The bindings are the translated modules' own (docs/architecture.md, "The translated modules"):
// set 0 binding 0 the stage-0 sampler, set 1 binding 0 the uniform slice whose `misc.x` is the
// alpha reference (alphaRef / 255, 0 when the draw has none: an alpha-keyed blend keys at one half).
layout(set = 0, binding = 0) uniform sampler2D s0;
layout(std140, set = 1, binding = 0) uniform Regs
{
    vec4 p[16];
    vec4 misc;
} regs;

layout(location = 2) in vec4 texcoord0;

void main()
{
    float reference = regs.misc.x > 0.0 ? regs.misc.x : 0.5;
    if (texture(s0, texcoord0.xy).a < reference) discard;
}
