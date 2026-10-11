"""The vertex half of a G-buffer variant: the module as it is, plus one varying the pixel half reads.

    layout(location = 13) out vec4 wxlGBuf;    xyz the view-space normal, w the class and plain flags

The lighting idiom every lit world module shares, as the translator spells it (the M2 and map object
families through their world x view rows c31..c33 or the bone palette; the terrain through its own
rows c0..c2; the light's direction in -c12, or c24 for the terrain):

    r1.xyz = dx_nrm(r0).xyz;                                       the view-space normal (terrain: unnormalized, no dx_nrm)
    r0.x   = clamp(dot(-regs.v[12].xyz, r1.xyz), 0.0, 1.0);        N.L (into any one component) <- the normal is read here
    r1.xyz = regs.v[10].xyz;                                       the direct colour
    r0.xyz = clamp(r0.xxx * r1.xyz + regs.v[11].xyz, 0.0, 1.0);   the light-only term: direct x N.L + ambient
    ...                                                            (terrain: a `min(r0.xyz, 1)` follows)
    o1     = clamp(r1 * r0 + regs.v[29], 0.0, 1.0);                color0: the tint and the added term
    color0 = o1;

The variant captures the normal right before the N.L line and the light-only term right after its line
(and the saturate that may follow), then writes the varying before the clip position: the normal, the
family's class, and bit 7 when color0 is the light-only term itself (no tint, no vertex colour, no
added term, no point light) within a 255th -- the pixel half's "plain" flag. A module without the idiom
(the unlit permutations) writes a zero normal and the unlit class.
"""
import re

from . import families
from .module import Module, OUT_DECL, indent_of

LOCATION = 13

# `rA.x = clamp(dot(<a>, <b>), 0.0, 1.0);` -- the N.L line; one operand a light register, the other the normal.
NL_LINE   = re.compile(r'^(\s*)(r\d+)\.([xyzw]) = clamp\(dot\(([^,]+), ([^)]+)\), 0\.0, 1\.0\);\s*$')
LIGHT_REG = re.compile(r'^-?(?:regs\.v|bones\.b)\[\d+(?: \+ a0\.[xyzw])?\]\.xyz$')
NORMAL_REG = re.compile(r'^(r\d+)\.xyz$')
# `rD.xyz = [clamp(]<...> + regs.v[K].xyz[, 0.0, 1.0)];` with rA.xxx among the factors: the light-only term.
POSITION  = re.compile(r'^(\s*)gl_Position = vec4\((\w+)\.x, -\2\.y, \2\.z, \2\.w\);\s*$')
COLOR0    = re.compile(r'^\s*color0 = (o\d+);\s*$')


def _light_line(rhs, nl_reg, comp):
    """A line multiplying the N.L result (the register's one component, broadcast) and adding a constant row."""
    return ('%s.%s' % (nl_reg, comp * 3)) in rhs and re.search(r'\+ regs\.v\[\d+\]\.xyz', rhs) is not None


def variant(text, family):
    """The variant's text and a dict of facts for the census, or (None, reason) when the module does
    not have the shape (never: every translated vertex module ends on the clip position line)."""
    try:
        m = Module(text)
    except ValueError as e:
        return None, str(e)
    if 'wxlGBuf' in text:
        return None, 'already a G-buffer variant'
    pos = m.find(POSITION)
    if pos is None:
        return None, 'no clip position line'
    if m.max_location(OUT_DECL) >= LOCATION:
        return None, 'location %d is taken' % LOCATION

    facts = {'lit': False, 'light_reg': None, 'color0': None}

    # The N.L line -- the first saturated dot of a light register and a normal register (the specular
    # permutations dot a half vector with the normal before it) -- the normal register it reads, the
    # register and component it writes.
    nl = nl_reg = nl_comp = normal_reg = None
    for i in range(m.body_start, pos):
        g = NL_LINE.match(m.lines[i])
        if not g:
            continue
        a, b = g.group(4).strip(), g.group(5).strip()
        if LIGHT_REG.match(a) and NORMAL_REG.match(b):
            normal_reg, light = NORMAL_REG.match(b).group(1), a
        elif LIGHT_REG.match(b) and NORMAL_REG.match(a):
            normal_reg, light = NORMAL_REG.match(a).group(1), b
        else:
            continue
        nl, nl_reg, nl_comp = i, g.group(2), g.group(3)
        facts['light_reg'] = light
        break

    # The light-only term: the first line after N.L that multiplies rA.xxx and adds a register, then
    # the saturates that follow on the same destination.
    light_end = None
    if nl is not None:
        for i in range(nl + 1, min(nl + 8, pos)):
            line = m.lines[i].strip()
            # Any three-component destination: the terrain's specular permutations write `r0.yzw`.
            g = re.match(r'^(r\d+\.[xyzw]{3}) = (.+);$', line)
            if g and _light_line(g.group(2), nl_reg, nl_comp):
                dest = re.escape(g.group(1))
                light_end = i
                for j in range(i + 1, min(i + 4, pos)):
                    if re.match(r'^%s = (?:min|clamp)\(%s, .+\);$' % (dest, dest), m.lines[j].strip()):
                        light_end = j
                    else:
                        break
                break
        if light_end is None:
            nl = None   # the N.L line without its term: taken as unlit rather than half-read

    c0 = m.find(COLOR0, start=pos)
    if c0 is not None:
        facts['color0'] = COLOR0.match(m.lines[c0]).group(1)

    # --- the edits, from the end of the body backwards so the indices stay right ------------------
    ind = indent_of(m.lines[pos])
    if nl is not None:
        facts['lit'] = True
        cls = family.cls
        if facts['color0'] is not None:
            inv = 1.0 / family.neutral
            dev = 'abs(%s.xyz * %s - wxlL)' % (facts['color0'], _f(inv))
            plain = '(wxlLit > 0.5 ? max(max(%s.x, %s.y), %s.z) : 1.0)' % (dev, dev, dev)
        else:
            plain = '1.0'
        block = [
            ind + '// The G-buffer varyings (tools/gbuffer): the view-space normal; the material class, flat (one per',
            ind + '// triangle); how far color0 is from the light-only term, interpolated, so that the pixel half',
            ind + '// decides "plain" per pixel (see families.py).',
            ind + 'wxlGBuf   = vec4(wxlN, 0.0);',
            ind + 'wxlGClass = %d.0;' % cls,
            ind + 'wxlGPlain = %s;' % plain,
        ]
        m.insert(pos, block)
        dest = re.match(r'^\s*(r\d+\.[xyzw]{3}) = ', m.lines[light_end]).group(1)
        m.insert(light_end + 1, [indent_of(m.lines[light_end]) + 'wxlL = %s;   // the light-only term (tools/gbuffer)' % dest])
        m.insert(nl, [indent_of(m.lines[nl]) + 'wxlN = %s.xyz;   // the view-space normal (tools/gbuffer)' % normal_reg,
                      indent_of(m.lines[nl]) + 'wxlLit = 1.0;'])
    else:
        m.insert(pos, [ind + '// No lighting idiom in this module (tools/gbuffer): the unlit class, no normal.',
                       ind + 'wxlGBuf   = vec4(0.0);',
                       ind + 'wxlGClass = %d.0;' % families.CLASS_UNLIT,
                       ind + 'wxlGPlain = 1.0;'])

    # The locals at the body's start, the varying's declaration after the last `out`.
    m.insert(m.body_start, ['    vec3  wxlN   = vec3(0.0);   // the view-space normal (tools/gbuffer)',
                            '    vec3  wxlL   = vec3(1.0);   // the light-only term',
                            '    float wxlLit = 0.0;         // 1 once the normal and the term were read'])
    last_out = m.last_decl(OUT_DECL)
    decls = [
        'layout(location = %d) out vec4 wxlGBuf;          // the G-buffer varyings (tools/gbuffer): the view normal xyz' % LOCATION,
        'layout(location = %d) flat out float wxlGClass;  // the material class, one per triangle' % (LOCATION + 1),
        'layout(location = %d) out float wxlGPlain;       // color0\'s distance to the light-only term (0 = plain)' % (LOCATION + 2),
    ]
    if last_out is not None:
        m.insert(last_out + 1, decls)
    else:
        m.insert(m.main, decls + [''])

    head = '// The G-buffer variant (tools/gbuffer) of the module below: one varying more, the module unchanged.\n'
    return head + m.text(), facts


def _f(x):
    s = repr(float(x))
    return s if '.' in s or 'e' in s else s + '.0'
