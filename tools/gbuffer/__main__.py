"""python -m gbuffer: the G-buffer variants of the translated world shaders, and their manifest lines.

For every record of a world family (families.py) in wxl-vulkan-api's shaders/translated/manifest.txt --
and every `bones`, `wind` and `windbones` variant the translator's tools gave it -- the module's GLSL is
read, its G-buffer variant written beside it (<stem>.gbuf.glsl, compiled to <stem>.gbuf.spv), and the
record's block gains the lines the device reads (manifest.py). The device pairs a vertex variant with a
pixel variant at draw time; a record whose half is missing is simply drawn as before.

    python -m gbuffer                          the default translated folder, every eligible record
    python -m gbuffer --only Terrain:0         one record by name (or an md5 prefix), for a look
    python -m gbuffer --dry-run                write the GLSL, compile, but leave the manifest alone
    python -m gbuffer --report census.md       the census as a Markdown file as well

Run it after translate.py (whose default run wipes and rewrites the translated folder): this tool's
output lives in that folder and is not part of the translator's.
"""
import argparse
import glob
import os
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'gbuffer'

from . import families, vertex, pixel
from .census import Census
from .compile import compile_all, find_glslang
from .manifest import Manifest

HERE        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'shaders', 'translated'))   # the repository's tree

# A record's variants: the manifest tag, the attribute the source path comes from, the glslang stage.
VERTEX_VARIANTS = (('gbuf', 'spv'), ('gbufbones', 'bones'), ('gbufwind', 'wind'), ('gbufwindbones', 'windbones'))
PIXEL_VARIANTS  = (('gbuf', 'spv'), ('gbufwind', 'wind'))


def wanted(record, only):
    if not only:
        return True
    return record.md5.startswith(only) or any(n == only or n.split(':')[0] == only for n in record.names)


def main(argv):
    ap = argparse.ArgumentParser(prog='python -m gbuffer', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--translated', default=DEFAULT_OUT, help='the translated folder (manifest.txt, vertex/, pixel/)')
    ap.add_argument('--glslang', default=None, help='the glslang to compile with (also $GLSLANG)')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--only', default=None, help='one record: an engine name ("Terrain:0", "Terrain") or an md5 prefix')
    ap.add_argument('--dry-run', action='store_true', help='write and compile, but do not touch the manifest or clean')
    ap.add_argument('--no-clean', action='store_true', help='keep .gbuf.* files this run did not make')
    ap.add_argument('--report', default=None, help='also write the census as Markdown here')
    a = ap.parse_args(argv)

    out = a.translated
    manifest_path = os.path.join(out, 'manifest.txt')
    if not os.path.isfile(manifest_path):
        print('no manifest at %s' % manifest_path, file=sys.stderr)
        return 2
    glslang = find_glslang(a.glslang)
    if not glslang:
        print('no glslang found: --glslang PATH or $GLSLANG', file=sys.stderr)
        return 2

    t0 = time.time()
    manifest = Manifest(manifest_path)
    census   = Census()
    tasks    = []      # (record, tag, family, src glsl, dst glsl, dst spv relative, stage)
    made     = set()   # the .gbuf.glsl / .spv paths this run wrote

    for r in manifest.records():
        fam = families.of_record(r.names, r.kind)
        if fam is None or not wanted(r, a.only):
            continue
        census.note(fam.key, r.kind, 'eligible')
        variants = VERTEX_VARIANTS if r.kind == 'vertex' else PIXEL_VARIANTS
        for tag, attr in variants:
            path = getattr(r, attr)
            if not path:
                continue
            stem = path[:-4]
            src  = os.path.join(out, stem + '.glsl')
            if not os.path.isfile(src):
                census.refused(fam.key, r.kind, 'no .glsl beside the .spv')
                continue
            text = open(src, encoding='utf-8').read()
            if r.kind == 'vertex':
                made_text, facts = vertex.variant(text, fam)
            else:
                made_text, facts = pixel.variant(text, fam)
            if made_text is None:
                census.refused(fam.key, r.kind, facts)
                continue
            if r.kind == 'vertex':
                census.note(fam.key, r.kind, 'lit' if facts['lit'] else 'unlit')
            elif facts.get('mcsh'):
                census.note(fam.key, r.kind, 'mcsh')
            dst_glsl = os.path.join(out, stem + '.gbuf.glsl')
            dst_spv  = os.path.join(out, stem + '.gbuf.spv')
            with open(dst_glsl, 'w', encoding='utf-8', newline='\n') as f:
                f.write(made_text)
            made.update((os.path.normcase(dst_glsl), os.path.normcase(dst_spv)))
            census.note(fam.key, r.kind, 'written')
            tasks.append((r, tag, fam, dst_glsl, dst_spv, (stem + '.gbuf.spv').replace('\\', '/'),
                          'vert' if r.kind == 'vertex' else 'frag'))

    results = compile_all(glslang, a.jobs, [(t[3], t[4], t[6]) for t in tasks])
    for (r, tag, fam, dst_glsl, dst_spv, rel, stage), (ok, why) in zip(tasks, results):
        if ok:
            census.note(fam.key, r.kind, 'compiled')
            r.gbuf[tag] = rel
        else:
            census.failure(r.first_name(), tag, why, fam.key, r.kind)
            made.discard(os.path.normcase(dst_spv))

    if not a.dry_run:
        if not a.no_clean and not a.only:
            for stale in glob.glob(os.path.join(out, '*', '*.gbuf.*')):
                if os.path.normcase(stale) not in made:
                    os.remove(stale)
        manifest.write(keep_prior=bool(a.only))   # --only: the other records keep their earlier variants

    header = ('gbuffer: %d record(s) in the manifest, %d variant(s) written, %d compiled, %d failed, in %.1f s%s'
              % (len(manifest.records()), sum(1 for _ in tasks), sum(1 for ok, _ in results if ok),
                 sum(1 for ok, _ in results if not ok), time.time() - t0,
                 ' (dry run: the manifest was not written)' if a.dry_run else ''))
    print(header)
    for line in census.lines():
        print(line)
    if a.report:
        census.report(a.report, '# G-buffer variants census\n\n' + header)
    return 1 if census.failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
