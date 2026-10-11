"""The tree-wind variants of the translated M2 vertex modules, for wxl-vulkan-api's src/vulkan/Wind.hpp.

A module qualifies when it is an M2 one (an engine name `Diffuse_*`) and either takes its position (v0)
to view space with three consecutive `rN.x/.y/.z = dot(<row>, v0);` lines -- the static M2 modules (rows
c31..c33) and the one-bone skinned ones (rows c[31 + a0.x], or in their `bones` variant the palette's
b[0 + a0.x]) -- or blends up to four bones by weight (a blendweight input, the `:8x` modules): its
rows blended as the module itself blends them, sum of w.i * c[31 + k + a0.i], so the bend is in the
same frame the module transforms the vertex with, continuous across the bones' weights. For each, vertex/<md5>.wind.glsl is the module with the position input renamed
and v0 bent by wind_m2.glsl right before those lines, compiled to vertex/<md5>.wind.spv; a 256-row
module's `bones` variant gets vertex/<md5>.bones.wind.spv the same way. The manifest record gains
`wind <spv>` and, for the latter, `windbones <spv>` (earlier such lines are replaced: a run is
idempotent). translate.py calls Run() after writing the manifest, so a regeneration keeps them.

    python windgen.py                    # the default manifest beside translate.py's output
    python windgen.py --glslang PATH     # the glslang.exe to use (also $GLSLANG)
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.normpath(os.path.join(HERE, '..', '..'))   # the wxl-vulkan-api repository
DEFAULT_OUT = os.path.join(EXT, 'shaders', 'translated')
DEFAULT_GLSLANG = None   # --glslang, $GLSLANG or PATH (translate_shaders.py resolves one and passes it)

POSITION_DECL = re.compile(r'^layout\(location = 0\) in vec4 v0;.*$', re.M)
ROW_LINE = re.compile(r'^\s*(r\d+)\.([xyz]) = dot\((.+), v0\);\s*$')
# The first blend line of a skinned module: `rN = vW.yyyy * <array>[<base> + a0.y];`.
BLEND_LINE = re.compile(r'^\s*r\d+ = (v\d+)\.yyyy \* (regs\.v|bones\.b)\[(\d+) \+ a0\.y\];\s*$')
OUR_LINES = ('wind ', 'windbones ')


def rows_of(lines):
    """The index of the first of the three transform lines and their three row expressions, or None."""
    for i, line in enumerate(lines):
        m = ROW_LINE.match(line)
        if not m or m.group(2) != 'x':
            continue
        reg, found, last = m.group(1), {'x': m.group(3)}, i
        for j in range(i + 1, min(i + 5, len(lines))):
            mm = ROW_LINE.match(lines[j])
            if mm and mm.group(1) == reg:
                found[mm.group(2)] = mm.group(3)
                last = j
        if len(found) == 3:
            return i, (found['x'], found['y'], found['z'])
        return None
    return None


def blended_rows(lines, main):
    """A skinned module: the index of its first blend line and the three blended row expressions."""
    for i in range(main, len(lines)):
        m = BLEND_LINE.match(lines[i])
        if not m:
            continue
        w, arr, base = m.group(1), m.group(2), int(m.group(3))
        rows = []
        for k in range(3):
            terms = ['%s.%s * %s[%d + a0.%s]' % (w, c, arr, base + k, c) for c in 'xyzw']
            rows.append('(' + ' + '.join(terms) + ')')
        return i, rows
    return None


def variant(text, snippet):
    if 'push_constant' in text or 'gl_VertexIndex' in text or not POSITION_DECL.search(text):
        return None
    skinned = 'blendweight' in text
    out = POSITION_DECL.sub('layout(location = 0) in vec4 v0_in;   // position0 (bent: tree wind)\nvec4 v0;', text, 1)
    at = out.find('} regs;')
    if at < 0:
        return None
    # After the last buffer block (a `bones` variant declares its palette after the uniform block).
    for marker in ('} bones;',):
        k = out.find(marker)
        if k > at:
            at = k
    at = out.index('\n', at) + 1
    out = out[:at] + '\n' + snippet + '\n' + out[at:]
    lines = out.split('\n')
    m = None
    for k, line in enumerate(lines):
        if line.startswith('void main()'):
            m = k
            break
    if m is None:
        return None
    if skinned:
        found = blended_rows(lines, m)
        if not found:
            return None
        i, (e0, e1, e2) = found
    else:
        found = rows_of(lines)
        if not found or found[0] < m:
            return None
        i, (e0, e1, e2) = found
    lines.insert(i, '    v0 = WindPosition(v0_in, %s, %s, %s);' % (e0, e1, e2))
    # v0 starts as the input, for any read before the transform.
    brace = m + 1 if lines[m + 1].strip() == '{' else m
    lines.insert(brace + 1, '    v0 = v0_in;')
    return '// The tree-wind variant (tools/bls_tools/windgen.py) of the module below.\n' + '\n'.join(lines)


def pixel_variant(text, snippet):
    """An M2 pixel module's half: the leaves' texture warp and light (wind_m2_pixel.glsl)."""
    if ('vec4 v1 = texcoord0;' not in text or 'texture(s0, v1.xy)' not in text or 'location = 11' in text or 'location = 12' in text
            or 'set = 3' in text or re.search(r'\breturn\b', text)):
        return None
    lines = text.split('\n')
    m = next((k for k, l in enumerate(lines) if l.startswith('void main()')), None)
    if m is None:
        return None
    k = next(k for k, l in enumerate(lines) if l.strip() == 'vec4 v1 = texcoord0;')
    lines.insert(k + 1, '    v1.xy += WindLeafUv(v1.xy);')
    # The canopy's occlusion on the colour, before the module's alpha test (which reads oC0.a only).
    test = next((j for j, l in enumerate(lines) if 'Render state 7 (alpha test)' in l), None)
    if test is not None:
        lines.insert(test, '    oC0.rgb *= WindCanopyShade();')
    lines.insert(m, snippet)
    return '// The tree-wind variant (tools/bls_tools/windgen.py) of the module below.\n' + '\n'.join(lines)


def build(out_dir, glslang, src_rel, dst_rel, snippet, stage='vert'):
    src = os.path.join(out_dir, src_rel)
    if not os.path.exists(src):
        return False, 'no source'
    text = open(src, encoding='utf-8').read()
    v = variant(text, snippet) if stage == 'vert' else pixel_variant(text, snippet)
    if v is None:
        return None, 'not eligible'
    glsl_path = os.path.join(out_dir, dst_rel[:-4] + '.glsl')
    open(glsl_path, 'w', encoding='utf-8', newline='\n').write(v)
    r = subprocess.run([glslang, '-V', '--target-env', 'vulkan1.0', '--spirv-val', '-S', stage, glsl_path,
                        '-o', os.path.join(out_dir, dst_rel)], capture_output=True, text=True)
    if r.returncode != 0:
        return False, ' | '.join((r.stdout + r.stderr).strip().splitlines()[-3:])
    return True, ''


def Run(out_dir=DEFAULT_OUT, glslang=None, quiet=False):
    glslang = glslang or os.environ.get('GLSLANG') or DEFAULT_GLSLANG
    snippet = open(os.path.join(HERE, 'wind_m2.glsl'), encoding='utf-8').read()
    pixel_snippet = open(os.path.join(HERE, 'wind_m2_pixel.glsl'), encoding='utf-8').read()
    manifest = os.path.join(out_dir, 'manifest.txt')
    lines = open(manifest, encoding='utf-8').read().split('\n')
    out, made, failed = [], 0, []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith('record '):
            if not line.startswith(OUR_LINES):
                out.append(line)
            i += 1
            continue
        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip() != '':
            if not lines[i].startswith(OUR_LINES):
                block.append(lines[i])
            i += 1
        parts = line.split()
        md5, kind, spv = parts[1], parts[2], parts[4]
        # The M2 modules only (the device looks at Diffuse_* draws, src/vulkan/Wind.cpp Observe).
        m2 = any(b.startswith('name Diffuse_') for b in block)
        if kind == 'vertex' and m2:
            stem = spv[:-4]
            ok, why = build(out_dir, glslang, stem + '.glsl', stem + '.wind.spv', snippet)
            if ok:
                block.append('wind ' + stem + '.wind.spv')
                made += 1
            elif ok is False and why != 'no source':
                failed.append((md5, why))
            bones = [b.split()[1] for b in block if b.startswith('bones ')]
            if ok and bones:
                okb, whyb = build(out_dir, glslang, bones[0][:-4] + '.glsl', bones[0][:-4] + '.wind.spv', snippet)
                if okb:
                    block.append('windbones ' + bones[0][:-4] + '.wind.spv')
                    made += 1
                elif okb is False:
                    failed.append((md5 + ' bones', whyb))
        # The M2 pixel modules (Combiners_*): the leaves' half, bound with a vertex `wind` variant.
        if kind == 'pixel' and any(b.startswith('name Combiners_') for b in block):
            stem = spv[:-4]
            ok, why = build(out_dir, glslang, stem + '.glsl', stem + '.wind.spv', pixel_snippet, 'frag')
            if ok:
                block.append('wind ' + stem + '.wind.spv')
                made += 1
            elif ok is False and why != 'no source':
                failed.append((md5 + ' pixel', why))
        out.extend(block)
    open(manifest, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
    if not quiet:
        print('windgen: %d wind variant(s), %d failed' % (made, len(failed)))
        for md5, why in failed[:10]:
            print('  ', md5, why)
    return made, failed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--glslang', default=None)
    a = ap.parse_args()
    made, failed = Run(a.out, a.glslang)
    sys.exit(1 if failed else 0)
