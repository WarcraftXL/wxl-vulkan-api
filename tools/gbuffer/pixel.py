"""The pixel half of a G-buffer variant: the module's body made a function of the engine's inputs,
called with them as they are for the frame and with neutral ones for the attachments.

    layout(location = 13) in vec4 wxlGBuf;      from the vertex half: the view normal, the class and plain flags
    layout(location = 1) out vec4 wxlAlbedo;    rgb the textures' colour before light and fog, a the flags / 255:
                                                the class in bits 0..4, bit 5 the device's interior row (p[14].y, a
                                                batch of an interior map object group), bit 6 its mover row (p[14].x,
                                                a unit's draw), bit 7 plain
    layout(location = 2) out vec4 wxlNormal;    xyz the unit view normal; w the depth tag: the fragment's clip w on a
                                                log2 scale in [0.505, 0.995], positive when there is a normal, negative
                                                when there is none (VkRenderApi.h WXL_VK_GBUFFER_TAG_*)
    layout(location = 3) out vec4 wxlLight;     rgb the engine's light term as it multiplies the albedo, a the baked shadow

Every translated pixel body is affine in color0 per channel -- textures x color0 (+ an additive
texture for the `_Add` combiners), then the fog -- so three evaluations tell everything:

    real  = body(color0, color1, fog0)                      what the engine shows (attachment 0, the alpha test on it)
    base  = body(neutral, 0, fog 1)                         the textures alone (+ the additive term)
    dark  = body(0, 0, fog 1)                               the additive term alone (families whose pixel stage may add one)
    albedo = base - dark;  light = color0 x (1 / neutral)   (the terrain's and the map objects' pixel stages double
                                                            their halved color0: neutral 0.5)

The terrain's pixel stage also multiplies by its baked shadow (MCSH: `alpha * 0.3 + 0.7`, from the
alpha map's w). The function takes a `keep` factor for that line, 1 for the real evaluation, 0 for the
albedo ones, and hands the factor out for the light attachment's alpha -- the albedo never carries the
artists' fixed-hour shadow, which the consumer's own shadows replace.
"""
import re

from . import families
from .module import Module, IN_DECL, indent_of

LOCATION = 13

ALPHA_TEST = re.compile(r'^\s*if \(regs\.misc\.x > 0\.0 && oC0\.a < regs\.misc\.x\) discard;\s*$')
OC0_DECL   = re.compile(r'^layout\(location = 0\) out vec4 oC0;\s*$')
OC0_INIT   = re.compile(r'^\s*oC0 = vec4\(0\.0\);\s*$')
# `rX.c = rY.c * cN.x + cN.y;` -- the baked shadow's line when cN is (0.3, 0.7, ...).
MAD_CONST  = re.compile(r'^(\s*)(r\d+\.[xyzw]) = (r\d+\.[xyzw]) \* (c\d+)\.x \+ \4\.y;\s*$')

INPUTS = (('color0', 'wxlColor0'), ('color1', 'wxlColor1'), ('fog0', 'wxlFog0'))


def variant(text, family):
    """The variant's text and facts for the census, or (None, reason)."""
    try:
        m = Module(text)
    except ValueError as e:
        return None, str(e)
    if 'wxlGBuf' in text:
        return None, 'already a G-buffer variant'
    if m.max_location(IN_DECL) >= LOCATION:
        return None, 'location %d is taken' % LOCATION
    oc0 = m.last_decl(OC0_DECL)
    if oc0 is None:
        return None, 'no oC0 output'
    test = m.find(ALPHA_TEST)
    if test is None:
        return None, 'no alpha test line'
    init = m.find(OC0_INIT, stop=test)
    if init is None:
        return None, 'no oC0 initialisation'
    if not any(re.match(r'^layout\(location = 0\) in vec4 color0;', l) for l in m.head()):
        return None, "reads no color0: an unlit pixel stage (the engine's shadow map, a full-screen pass)"
    for name, _ in INPUTS:
        if re.search(r'\b%s\b' % name, '\n'.join(m.lines[test:m.close])):
            return None, 'the alpha test region reads %s' % name

    facts = {'mcsh': False, 'dark': family.dark_term, 'reads_color1': False}

    # The function's body: main's lines up to the alpha test's comment, the inputs renamed, the baked
    # shadow's line instrumented. The comment lines right before the test stay with the test.
    comment_start = test
    while comment_start - 1 > m.body_start and m.lines[comment_start - 1].strip().startswith('//'):
        comment_start -= 1
    fn = []
    for i in range(m.body_start, comment_start):
        line = m.lines[i]
        if i == init:
            line = line.replace('oC0 = vec4(0.0);', 'vec4 oC0 = vec4(0.0);   // local: the function\'s result')
        for name, param in INPUTS:
            if re.search(r'\b%s\b' % name, line):
                line = re.sub(r'\b%s\b' % name, param, line)
                if name == 'color1':
                    facts['reads_color1'] = True
        g = MAD_CONST.match(line) if family.baked_shadow and not facts['mcsh'] else None
        if g:
            c = m.const_vec4(g.group(4))
            if c and abs(c[0] - 0.3) < 1e-3 and abs(c[1] - 0.7) < 1e-3:
                ind = g.group(1)
                fn.append(line)
                fn.append(ind + 'wxlMcsh = %s;   // the terrain\'s baked shadow (tools/gbuffer)' % g.group(2))
                fn.append(ind + '%s = mix(1.0, %s, wxlKeepShadow);' % (g.group(2), g.group(2)))
                facts['mcsh'] = True
                continue
        fn.append(line)
    while fn and fn[-1].strip() == '':
        fn.pop()
    # A discard inside the body becomes a flag the main takes once every evaluation has sampled (a kill
    # before a texture read leaves the quad's derivatives undefined for the neighbours); the wind's canopy
    # shade is light, not albedo: the frame's own evaluation alone applies it.
    fn = [re.sub(r'\bdiscard;', '{ wxlKill = true; return oC0; }', l) for l in fn]
    fn = [l.replace('oC0.rgb *= WindCanopyShade();',
                    'if (wxlReal) oC0.rgb *= WindCanopyShade();   // light, not albedo (tools/gbuffer)') for l in fn]

    neutral = _f(family.neutral)
    scale   = _f(1.0 / family.neutral)
    new_main = [
        '// The pixel body as a function of the engine\'s light, specular and fog inputs (tools/gbuffer): called',
        '// with them as they are for the frame, with neutral ones for the G-buffer (the module\'s own lines).',
        'bool wxlKill = false;   // a discard inside the body: taken once every evaluation has sampled (tools/gbuffer)',
        'bool wxlReal = false;   // the evaluation with the frame\'s own inputs: the wind\'s canopy shade applies to it alone',
        'vec4 WxlBody(vec4 wxlColor0, vec4 wxlColor1, float wxlFog0, float wxlKeepShadow, out float wxlMcsh)',
        '{',
        '    wxlMcsh = 1.0;',
    ] + fn + [
        '    return oC0;',
        '}',
        '',
        'void main()',
        '{',
        '    float wxlMcsh = 1.0, wxlUnused = 1.0;',
        '    // The G-buffer (wxl-vulkan-api VkRenderApi.h, WXL_VkGBuffer): the textures alone, the normal, the light term.',
        '    // Evaluated first: every texture read of every evaluation happens before any discard, so the quad\'s',
        '    // derivatives (the mip, the leaf uv) stay defined.',
        '    vec3 wxlBase = WxlBody(vec4(%s, %s, %s, color0.a), vec4(0.0), 1.0, 0.0, wxlUnused).rgb;' % (neutral, neutral, neutral),
    ]
    if family.dark_term:
        new_main += ['    vec3 wxlDark = WxlBody(vec4(0.0, 0.0, 0.0, color0.a), vec4(0.0), 1.0, 0.0, wxlUnused).rgb;']
    else:
        new_main += ['    const vec3 wxlDark = vec3(0.0);   // this family\'s pixel stage adds nothing to its textures x color0']
    new_main += [
        '    wxlReal = true;',
        '    oC0 = WxlBody(color0, color1, fog0, 1.0, wxlMcsh);',
        '    if (wxlKill) discard;',
    ] + [m.lines[i] for i in range(comment_start, test + 1)] + [
        '',
        # The class (bits 0..4), the interior bit (bit 5: the device's row p[14].y, 1 for a batch of an interior map
        # object group), the mover bit (bit 6: p[14].x, 1 for a unit's draw), the plain bit (7).
        '    float wxlFlags = clamp(round(wxlGClass), 0.0, 31.0) + (regs.p[14].y > 0.5 ? 32.0 : 0.0) + (regs.p[14].x > 0.5 ? 64.0 : 0.0)'
        ' + (wxlGPlain < 0.004 ? 128.0 : 0.0);',
        '    vec3  wxlN     = wxlGBuf.xyz;',
        '    float wxlNn    = dot(wxlN, wxlN);',
        '    wxlAlbedo = vec4(max(wxlBase - wxlDark, vec3(0.0)), wxlFlags / 255.0);',
        # The depth tag (VkRenderApi.h WXL_VK_GBUFFER_TAG_*): the fragment's clip w on a log2 scale in the normal's w, in
        # [0.505, 0.995] -- positive with a normal, negative without; 0 is a pixel no variant wrote, 1 a variant older than
        # the tag. A draw that writes depth but no G-buffer leaves the record of the surface behind under its pixels; the
        # tag no longer matches the depth there, and a consumer leaves the pixel alone.
        '    // The depth tag (wxl-vulkan-api VkRenderApi.h, WXL_VK_GBUFFER_TAG_*): this surface\'s clip w, log2 scaled, in the',
        '    // normal\'s w (negative without a normal), so a consumer can tell this record from one a later draw that wrote',
        '    // depth but no G-buffer left under its own pixels.',
        '    float wxlTag   = 0.505 + 0.49 * clamp((log2(max(1.0 / gl_FragCoord.w, 0.0625)) + 4.0) * 0.0625, 0.0, 1.0);',
        '    wxlNormal = wxlNn > 1e-8 ? vec4(wxlN * inversesqrt(wxlNn), wxlTag) : vec4(0.0, 0.0, 0.0, -wxlTag);',
        '    wxlLight  = vec4(color0.rgb * %s, wxlMcsh);' % scale,
        '}',
    ]

    # Assemble: the head with the declarations added, then the new main.
    head = m.lines[:m.main]
    last_in = m.last_decl(IN_DECL)
    decls_in = [
        'layout(location = %d) in vec4 wxlGBuf;          // from the vertex half (tools/gbuffer): the view normal xyz' % LOCATION,
        'layout(location = %d) flat in float wxlGClass;  // the material class' % (LOCATION + 1),
        'layout(location = %d) in float wxlGPlain;       // color0\'s distance to the light-only term (0 = plain)' % (LOCATION + 2),
    ]
    at = (last_in if last_in is not None else oc0) + 1
    head[at:at] = decls_in
    oc0_now = next(i for i, l in enumerate(head) if OC0_DECL.match(l))
    head[oc0_now + 1:oc0_now + 1] = [
        'layout(location = 1) out vec4 wxlAlbedo;   // the G-buffer (tools/gbuffer): the textures before light and fog, the flags',
        'layout(location = 2) out vec4 wxlNormal;   // the unit view normal, w the depth tag (negative without a normal)',
        'layout(location = 3) out vec4 wxlLight;    // the engine\'s light term on the albedo, a the terrain\'s baked shadow',
    ]
    # A module that never declared color1 as an input still needs the name for the function's call.
    if not any(re.match(r'^layout\(location = 1\) in vec4 color1;', l) for l in head):
        head.insert(oc0_now, 'const vec4 color1 = vec4(0.0);   // not an input of this module (tools/gbuffer)')
    while head and head[-1].strip() == '':
        head.pop()
    out = ['// The G-buffer variant (tools/gbuffer) of the module below: its body as a function, called three times.'] \
          + head + [''] + new_main + ['']
    return '\n'.join(out), facts


def _f(x):
    s = repr(float(x))
    return s if '.' in s or 'e' in s else s + '.0'
