"""The shadow variants of the translated world vertex modules, for wxl-vulkan-api's draw replay
(src/capture/Replay.cpp, wxl.vk.render's ReplayWorld).

A world vertex module ends with `gl_Position = vec4(P.x, -P.y, P.z, P.w);`, P its clip position in
the engine's (D3D) convention. Its shadow variant is the same module with that line replaced by

    gl_Position = WxlShadowClip(vec4(P.x, P.y, P.z, P.w));

which multiplies P by one matrix from the push range (vertex stage, offset 0, 64 bytes -- the range
every pipeline layout of the device carries) and flips y as the module did. The device sets that
matrix per shadow cascade to inverse(the frame's view-projection) x the light's view-projection, so
P goes back to world space and on into the light's clip space whatever the module computed it from
(World x View rows, a bone palette, the terrain's row form): no constant register is reinterpreted.

The world families only (terrain, map objects, models, the ground clutter): the interface, liquids,
weather and full-screen passes are never replayed. For each, vertex/<md5>.shadow.glsl / .spv, and
for a 256-row module's `bones` variant vertex/<md5>.bones.shadow.spv; the manifest record gains
`shadow <spv>` and `shadowbones <spv>` (earlier such lines are replaced: a run is idempotent).
translate.py calls Run() after windgen, so a regeneration keeps them.

A record with tree-wind variants (windgen.py: `wind`, `windbones`) also gets their shadow variants,
vertex/<md5>.wind.shadow.spv and vertex/<md5>.bones.wind.shadow.spv, so a swaying tree's shadow
sways with it. The wind module owns the push range already (its model's values, 16 bytes), so the
matrix follows them in the same block, at offset 16 (the device builds those pipelines on a layout
with an 80-byte range); the manifest lines are `shadowwind <spv>` and `shadowwindbones <spv>`.

    python shadowgen.py                    # the default manifest beside translate.py's output
    python shadowgen.py --glslang PATH     # the glslang.exe to use (also $GLSLANG)
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.normpath(os.path.join(HERE, '..', '..'))   # the wxl-vulkan-api repository
DEFAULT_OUT = os.path.join(EXT, 'shaders', 'translated')
DEFAULT_GLSLANG = None   # --glslang, $GLSLANG or PATH (translate_shaders.py resolves one and passes it)

POSITION = re.compile(r'^(\s*)gl_Position = vec4\((\w+)\.x, -\2\.y, \2\.z, \2\.w\);\s*$', re.M)
FAMILIES = ('Terrain', 'MapObj', 'Diffuse_', 'Color_', 'CDiffuse_', 'DetailDoodad')
OUR_LINES = ('shadow ', 'shadowbones ', 'shadowwind ', 'shadowwindbones ')
WIND_PUSH = re.compile(r'layout\(push_constant\) uniform WindPush \{ int bias; uint start; uint count; uint on; \} windP;')

SNIPPET = '''
// The shadow replay's matrix (tools/bls_tools/shadowgen.py): the engine's clip space -> the light's.
layout(push_constant) uniform WxlShadow { mat4 clipToLight; } wxlShadow;
vec4 WxlShadowClip(vec4 clip)
{
    const vec4 l = wxlShadow.clipToLight * clip;   // row vector times the matrix (row-major upload)
    return vec4(l.x, -l.y, l.z, l.w);
}
'''


def wind_variant(text):
    """A tree-wind module with its position re-projected: the matrix after the wind's push values."""
    if len(WIND_PUSH.findall(text)) != 1 or len(POSITION.findall(text)) != 1:
        return None
    out = WIND_PUSH.sub('layout(push_constant) uniform WindPush { int bias; uint start; uint count; uint on; '
                        'mat4 clipToLight; } windP;   // + the shadow replay\'s matrix (shadowgen.py)', text, 1)
    out = POSITION.sub(lambda m: '%s{ const vec4 l = windP.clipToLight * vec4(%s.x, %s.y, %s.z, %s.w); '
                       'gl_Position = vec4(l.x, -l.y, l.z, l.w); }'
                       % (m.group(1), m.group(2), m.group(2), m.group(2), m.group(2)), out, 1)
    return '// The shadow variant (tools/bls_tools/shadowgen.py) of the tree-wind module below.\n' + out


def variant(text):
    """The module with its position re-projected, or None when it does not have the shape."""
    if 'push_constant' in text or len(POSITION.findall(text)) != 1:
        return None
    lines = text.split('\n')
    main = next((k for k, l in enumerate(lines) if l.startswith('void main()')), None)
    if main is None:
        return None
    out = POSITION.sub(lambda m: '%sgl_Position = WxlShadowClip(vec4(%s.x, %s.y, %s.z, %s.w));'
                       % (m.group(1), m.group(2), m.group(2), m.group(2), m.group(2)), text, 1)
    lines = out.split('\n')
    main = next(k for k, l in enumerate(lines) if l.startswith('void main()'))
    lines.insert(main, SNIPPET)
    return '// The shadow variant (tools/bls_tools/shadowgen.py) of the module below.\n' + '\n'.join(lines)


def build(out_dir, glslang, src_rel, dst_rel, wind=False):
    src = os.path.join(out_dir, src_rel)
    if not os.path.exists(src):
        return False, 'no source'
    text = open(src, encoding='utf-8').read()
    v = wind_variant(text) if wind else variant(text)
    if v is None:
        return None, 'not eligible'
    glsl_path = os.path.join(out_dir, dst_rel[:-4] + '.glsl')
    open(glsl_path, 'w', encoding='utf-8', newline='\n').write(v)
    r = subprocess.run([glslang, '-V', '-Os', '--target-env', 'vulkan1.0', '--spirv-val', '-S', 'vert', glsl_path,
                        '-o', os.path.join(out_dir, dst_rel)], capture_output=True, text=True)
    if r.returncode != 0:
        return False, ' | '.join((r.stdout + r.stderr).strip().splitlines()[-3:])
    return True, ''


def Run(out_dir=DEFAULT_OUT, glslang=None, quiet=False):
    glslang = glslang or os.environ.get('GLSLANG') or DEFAULT_GLSLANG
    manifest = os.path.join(out_dir, 'manifest.txt')
    lines = open(manifest, encoding='utf-8').read().split('\n')
    out, made, failed, skipped = [], 0, [], 0
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
        kind, spv = parts[2], parts[4]
        world = any(b.startswith('name ' + f) for b in block for f in FAMILIES)
        if kind == 'vertex' and world:
            stem = spv[:-4]
            ok, why = build(out_dir, glslang, stem + '.glsl', stem + '.shadow.spv')
            if ok:
                block.append('shadow ' + stem + '.shadow.spv')
                made += 1
            elif ok is False:
                failed.append((parts[1], why))
            else:
                skipped += 1
            bones = [b.split()[1] for b in block if b.startswith('bones ')]
            if bones:
                bstem = bones[0][:-4]
                okb, whyb = build(out_dir, glslang, bstem + '.glsl', bstem + '.shadow.spv')
                if okb:
                    block.append('shadowbones ' + bstem + '.shadow.spv')
                    made += 1
                elif okb is False:
                    failed.append((parts[1] + ' bones', whyb))
            # The tree-wind variants' (windgen.py ran before).
            for tag, line_tag in (('wind ', 'shadowwind '), ('windbones ', 'shadowwindbones ')):
                src = [b.split()[1] for b in block if b.startswith(tag)]
                if not src:
                    continue
                wstem = src[0][:-4]
                okw, whyw = build(out_dir, glslang, wstem + '.glsl', wstem + '.shadow.spv', wind=True)
                if okw:
                    block.append(line_tag + wstem + '.shadow.spv')
                    made += 1
                elif okw is False:
                    failed.append((parts[1] + ' ' + tag.strip(), whyw))
        out.extend(block)
    open(manifest, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
    if not quiet:
        print('shadowgen: %d shadow variant(s), %d not eligible, %d failed' % (made, skipped, len(failed)))
        for md5, why in failed[:10]:
            print('  ', md5, why)
    return made, failed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--glslang', default=None)
    a = ap.parse_args()
    _, bad = Run(a.out, a.glslang)
    sys.exit(1 if bad else 0)
