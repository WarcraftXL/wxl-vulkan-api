"""Census of every shader record the client ships for the vs_3_0 / ps_3_0 caps.

Answers, from the bytes and not from the DX9 spec, what a DX9 -> SPIR-V translator has to
implement: which opcodes, which register files, how high the constant indices go, which
modifiers, which dcl semantics, whether anything branches or indexes. Written for
a census document; re-run it when the archives change.

    python census.py                      # Markdown on stdout
    python census.py -o census.md         # ...to a file
    python census.py --validate           # also compare every distinct record with fxc /dumpbin
    python census.py --embedded           # also scan Wow.exe for shader blobs that have no .bls
    python census.py --dump out/census    # also write each record as out/census/<K>_<prof>_<name>/pNN.fxo
    python census.py --into <doc.md>      # replace everything after the marker line
                                          # `<!-- census:generated -->` in that document

Profile resolution follows the engine: a name missing from vs_3_0 is loaded
from vs_2_0, then vs_1_1; a pixel name missing from ps_3_0 from ps_2_0, then ps_1_4, then ps_1_1.
SM1 records cannot be decoded here (no length field) and are listed, not counted.
"""
import collections, hashlib, os, re, subprocess, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ea = load('extract_all')
lf = load('listfile')
ir = load('dx9ir')

BS = chr(92)
# fxc (Windows SDK) for --validate only: $FXC, else PATH.
FXC = os.environ.get('FXC') or 'fxc.exe'
# Wow.exe for --embedded only: beside the Data folder.
WOW = os.path.join(os.path.dirname(os.path.normpath(ea.DATADIR or '.')), 'Wow.exe')
VS_FALLBACK = ['vs_3_0', 'vs_2_0', 'vs_1_1']
PS_FALLBACK = ['ps_3_0', 'ps_2_0', 'ps_1_4', 'ps_1_1']

# Families, by name. The order matters: first match wins.
FAMILIES = [
    ('UI', re.compile(r'^UI$')),
    ('M2 vertex', re.compile(r'^(Diffuse|Color|CDiffuse)_')),
    ('M2 pixel (Combiners)', re.compile(r'^Combiners_')),
    ('WMO (MapObj)', re.compile(r'^MapObj')),
    ('Terrain', re.compile(r'^Terrain')),
    ('Liquid', re.compile(r'^(vs|ps)Liquid|^WaterRipples$')),
    ('DetailDoodad', re.compile(r'^DetailDoodad$')),
    ('ShadowMap', re.compile(r'^ShadowMap')),
    ('Full-screen effects (FFX)', re.compile(r'^(FFX|Desaturate)')),
    ('Weather', re.compile(r'^(patter|rain|sand|snowpoint)$')),
]


def family(name):
    for fam, rx in FAMILIES:
        if rx.search(name):
            return fam
    return 'other'


# ---- enumeration -------------------------------------------------------------------------------

def inventory():
    """{(kind, profile_lower): {name_lower: name_as_listed}} from the listfile."""
    inv = collections.defaultdict(dict)
    for n in lf.entries(prefix='shaders'):
        p = n.split(BS)
        if len(p) == 4 and p[1].lower() in ('vertex', 'pixel') and p[3].lower().endswith('.bls'):
            inv[(p[1].lower(), p[2].lower())][p[3][:-4].lower()] = p[3][:-4]
    return inv


def resolve(inv):
    """Which container the engine loads per (kind, name) under the 3_0 caps: [(kind, profile, name, fallback_depth)]."""
    out = []
    for kind, chain in (('vertex', VS_FALLBACK), ('pixel', PS_FALLBACK)):
        names = {}
        for prof in chain:
            for low, name in inv.get((kind, prof), {}).items():
                names.setdefault(low, name)
        for low in sorted(names):
            for depth, prof in enumerate(chain):
                if low in inv.get((kind, prof), {}):
                    out.append((kind, prof, inv[(kind, prof)][low], depth))
                    break
    return out


# ---- per-record metrics -------------------------------------------------------------------------

class Rec(object):
    pass


def analyse(code, kind, profile, name, perm):
    r = Rec()
    r.kind, r.profile, r.name, r.perm = kind, profile, name, perm
    r.key = '%s:%d' % (name, perm)
    r.bytes = len(code)
    r.md5 = hashlib.md5(code).hexdigest()
    r.sm1 = False
    try:
        sh = ir.decode(code)
    except ValueError as e:
        r.sm1 = True
        r.error = str(e)
        return r
    r.sh = sh
    r.version = sh.profile
    r.ops = collections.Counter()       # mnemonic (with texld variants / compare suffix) -> count
    r.regs_read = collections.Counter() # register type name -> count
    r.regs_written = collections.Counter()
    r.max_index = {}                    # regtype name -> highest index touched
    r.c_read = set()                    # direct constant reads
    r.c_def = set()
    r.c_rel = []                        # (base, addr reg) for relative reads
    r.i_read, r.b_read = set(), set()
    r.flow = collections.Counter()
    r.pred = 0
    r.samplers = {}                     # s# -> type
    r.tex_ops = collections.Counter()
    r.srcmods = collections.Counter()
    r.swizzles = collections.Counter()
    r.masks = collections.Counter()
    r.dstmods = collections.Counter()
    r.defs = collections.Counter()
    r.inputs = []                       # (reg, semantic, centroid)
    r.outputs_dcl = []                  # vs_3_0 dcl'd outputs (reg, semantic)
    r.outputs_written = {}              # reg name -> mask union
    r.ninst = 0
    r.vpos_vface = set()
    kindname = sh.kind
    major = sh.major

    def touch(o, written):
        nm = o.name(kindname, major)
        base = re.match(r'[A-Za-z]+', nm).group(0)
        if o.rtype == ir.RT_LOOP:
            base = 'aL'
        if o.rtype == ir.RT_MISCTYPE:
            r.vpos_vface.add(nm)
        (r.regs_written if written else r.regs_read)[base] += 1
        if o.rtype != ir.RT_LOOP:
            r.max_index[base] = max(r.max_index.get(base, -1), o.index)
        if o.rel:
            rt, ri, comp = o.rel
            r.c_rel.append((base, o.index, 'aL' if rt == ir.RT_LOOP else '%s%d.%s'
                            % (ir.RT_NAMES.get(rt, '?'), ri, ir.SWZ[comp])))
            r.regs_read['aL' if rt == ir.RT_LOOP else 'a'] += 1
        elif o.rtype == ir.RT_CONST and not written:
            r.c_read.add(o.index)
        if o.rtype == ir.RT_CONSTINT and not written:
            r.i_read.add(o.index)
        if o.rtype == ir.RT_CONSTBOOL and not written:
            r.b_read.add(o.index)

    for ins in sh.insts:
        op = ins.op
        if op == ir.OP_DCL:
            d = ins.dst
            if d.rtype == ir.RT_SAMPLER:
                r.samplers[d.index] = ir.TEXTYPE.get(ins.dcl_textype, 'tt%d' % ins.dcl_textype)
            else:
                u, ui = ins.dcl_usage
                sem = '%s%d' % (ir.USAGE.get(u, 'usage%d' % u), ui)
                nm = d.name(kindname, major)
                if kindname == 'ps' and major < 3:
                    sem = 'ps2:' + nm[0]      # ps_2_0 dcl carries no semantic; t#/v# is the semantic
                mask = ''.join(ir.SWZ[k] for k in range(4) if d.mask & (1 << k))
                if kindname == 'vs' and d.rtype == ir.RT_OUTPUT and major >= 3:
                    r.outputs_dcl.append((nm, sem, mask))
                else:
                    r.inputs.append((nm, sem, mask, d.centroid))
            continue
        if op in (ir.OP_DEF, ir.OP_DEFI, ir.OP_DEFB):
            r.defs[ins.name] += 1
            if op == ir.OP_DEF:
                r.c_def.add(ins.dst.index)
            continue
        r.ninst += 1
        mn = ins.mnemonic()
        r.ops[mn] += 1
        if op in ir.FLOW:
            r.flow[mn] += 1
        if ins.predicated:
            r.pred += 1
            r.ops['(predicated)'] += 1
        if op in (ir.OP_TEXLD, ir.OP_TEXLDD, ir.OP_TEXLDL, ir.OP_TEXKILL):
            r.tex_ops[mn] += 1
        if ins.dst is not None:
            d = ins.dst
            if op != ir.OP_TEXKILL:
                touch(d, True)
                nm = d.name(kindname, major)
                r.outputs_written[nm] = r.outputs_written.get(nm, 0) | d.mask
                r.masks['xyzw' if d.mask == 0xF else
                        ''.join(ir.SWZ[k] for k in range(4) if d.mask & (1 << k))] += 1
                if d.sat:
                    r.dstmods['_sat'] += 1
                if d.pp:
                    r.dstmods['_pp'] += 1
                if d.centroid:
                    r.dstmods['_centroid'] += 1
                if d.shift:
                    r.dstmods['shift'] += 1
            else:
                touch(d, False)
        if ins.pred is not None:
            touch(ins.pred, False)
        for s in ins.srcs:
            touch(s, False)
            r.srcmods[ir.SRCMOD.get(s.mod, ('', '', '?'))[2]] += 1
            r.swizzles[s.swizzle_kind()] += 1
    r.c_max = max(r.c_read) if r.c_read else -1
    r.c_engine = sorted(r.c_read - r.c_def)
    return r


# ---- aggregation helpers ------------------------------------------------------------------------

def rng(xs):
    xs = sorted(set(xs))
    if not xs:
        return '-'
    out, start, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append('%d' % start if start == prev else '%d-%d' % (start, prev))
        start = prev = x
    out.append('%d' % start if start == prev else '%d-%d' % (start, prev))
    return ','.join(out)


def counter_table(recs, attr, title, col='item'):
    """Rows: item | instructions | records | distinct shaders."""
    tot = collections.Counter()
    nrec = collections.Counter()
    ndist = collections.defaultdict(set)
    for r in recs:
        c = getattr(r, attr)
        for k, v in c.items():
            tot[k] += v
            nrec[k] += 1
            ndist[k].add(r.md5)
    lines = ['| %s | occurrences | records | distinct shaders |' % col, '| --- | ---: | ---: | ---: |']
    for k, v in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append('| `%s` | %d | %d | %d |' % (k, v, nrec[k], len(ndist[k])))
    return '\n'.join(lines)


def md(recs, resolved, inv, validation, embedded, sm1_names, empty):
    L = []
    P = L.append
    decoded = [r for r in recs if not r.sm1]
    vs = [r for r in decoded if r.kind == 'vertex']
    ps = [r for r in decoded if r.kind == 'pixel']
    distinct = lambda rs: len(set(r.md5 for r in rs))

    P('<!-- generated by tools/bls_tools/census.py; do not edit by hand -->')
    P('')
    P('## Inventory')
    P('')
    P('| | containers | records | distinct shaders | bytes |')
    P('| --- | ---: | ---: | ---: | ---: |')
    for label, rs in (('vertex', vs), ('pixel', ps), ('both', decoded)):
        P('| %s | %d | %d | %d | %d |' % (label, len(set((r.kind, r.name) for r in rs)), len(rs),
                                         distinct(rs), sum(r.bytes for r in rs)))
    P('')
    P('Containers by profile actually loaded under the vs_3_0 / ps_3_0 caps (fallback depth in the engine\'s chain):')
    P('')
    P('| kind | profile | fallback | containers |')
    P('| --- | --- | ---: | --- |')
    byprof = collections.defaultdict(list)
    for kind, prof, name, depth in resolved:
        byprof[(kind, prof, depth)].append(name)
    for (kind, prof, depth), names in sorted(byprof.items()):
        P('| %s | `%s` | %d | %s |' % (kind, prof, depth, ', '.join('`%s`' % n for n in sorted(names))))
    if empty:
        P('')
        P('Containers that are present but empty (the engine treats a bad file as missing and falls back one profile): %s.'
          % ', '.join('`%s` (%d bytes, %s)' % (p + BS + n, b, s) for k, p, n, b, s in empty))
    if sm1_names:
        P('')
        P('SM1 containers the chain lands on (not decoded, not counted): %s.'
          % ', '.join('`%s\\%s`' % (p, n) for k, p, n in sm1_names))
    P('')
    P('Version tokens inside the records (a container named `ps_3_0` can hold an older profile):')
    P('')
    vt = collections.Counter((r.profile, r.version) for r in decoded)
    P('| container profile | record version | records |')
    P('| --- | --- | ---: |')
    for (p, v), n in sorted(vt.items()):
        P('| `%s` | `%s` | %d |' % (p, v, n))

    # ---- per family -------------------------------------------------------------------------
    fams = collections.OrderedDict()
    for r in decoded:
        fams.setdefault(family(r.name), []).append(r)

    P('')
    P('## Families')
    P('')
    P('| family | containers | records | distinct | instructions (min/median/max) | opcodes used | max c (vs / ps) | relative | flow | samplers |')
    P('| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |')
    for fam, rs in fams.items():
        ni = sorted(r.ninst for r in rs)
        ops = set()
        for r in rs:
            ops |= set(k for k in r.ops if not k.startswith('('))
        cmax_vs = max([r.c_max for r in rs if r.kind == 'vertex'] or [-1])
        cmax_ps = max([r.c_max for r in rs if r.kind == 'pixel'] or [-1])
        rel = sum(1 for r in rs if r.c_rel)
        flow = sum(1 for r in rs if r.flow or r.pred)
        smp = set()
        for r in rs:
            smp |= set('s%d:%s' % kv for kv in r.samplers.items())
        P('| %s | %d | %d | %d | %d / %d / %d | %d | %s / %s | %d | %d | %s |'
          % (fam, len(set((r.kind, r.name) for r in rs)), len(rs), distinct(rs),
             ni[0], ni[len(ni) // 2], ni[-1], len(ops),
             'c%d' % cmax_vs if cmax_vs >= 0 else '-', 'c%d' % cmax_ps if cmax_ps >= 0 else '-',
             rel, flow, ', '.join(sorted(smp, key=lambda s: int(s[1:].split(':')[0]))) or '-'))

    # ---- opcodes ----------------------------------------------------------------------------
    P('')
    P('## Opcodes')
    P('')
    P('Across every decoded record. `occurrences` is instruction count, `records` the number of')
    P('records containing the opcode, `distinct shaders` the same after folding byte-identical records.')
    P('')
    P('### Vertex')
    P('')
    P(counter_table(vs, 'ops', 'vertex opcodes', 'opcode'))
    P('')
    P('### Pixel')
    P('')
    P(counter_table(ps, 'ops', 'pixel opcodes', 'opcode'))

    P('')
    P('### Opcodes by family')
    P('')
    allops = sorted(set(k for r in decoded for k in r.ops if not k.startswith('(')))
    P('| opcode | ' + ' | '.join(fams.keys()) + ' |')
    P('| --- | ' + ' | '.join('---:' for _ in fams) + ' |')
    for op in allops:
        row = []
        for fam, rs in fams.items():
            n = sum(1 for r in rs if op in r.ops)
            row.append(str(n) if n else '')
        P('| `%s` | %s |' % (op, ' | '.join(row)))

    # ---- registers --------------------------------------------------------------------------
    P('')
    P('## Register files')
    P('')
    for label, rs in (('Vertex', vs), ('Pixel', ps)):
        P('### %s' % label)
        P('')
        P('| register file | reads | writes | records touching | highest index |')
        P('| --- | ---: | ---: | ---: | ---: |')
        files = set()
        for r in rs:
            files |= set(r.regs_read) | set(r.regs_written)
        for f in sorted(files):
            reads = sum(r.regs_read.get(f, 0) for r in rs)
            writes = sum(r.regs_written.get(f, 0) for r in rs)
            touching = sum(1 for r in rs if f in r.regs_read or f in r.regs_written)
            hi = max([r.max_index.get(f, -1) for r in rs] or [-1])
            P('| `%s` | %d | %d | %d | %s |' % (f, reads, writes, touching, hi if hi >= 0 else '-'))
        P('')

    P('### Constant registers per family')
    P('')
    P('`c` read directly (not `def`d) by any record of the family, the highest such index, and the')
    P('relative-addressed bases. The highest index is what the uniform block has to reach.')
    P('')
    P('| family | kind | engine-supplied c (union) | highest | relative bases | i# / b# |')
    P('| --- | --- | --- | ---: | --- | --- |')
    for fam, rs in fams.items():
        for kind in ('vertex', 'pixel'):
            sub = [r for r in rs if r.kind == kind]
            if not sub:
                continue
            eng = set()
            for r in sub:
                eng |= set(r.c_engine)
            rel = sorted(set('%s%d[%s]' % t for r in sub for t in r.c_rel))
            ib = sorted(set('i%d' % i for r in sub for i in r.i_read) |
                        set('b%d' % b for r in sub for b in r.b_read))
            P('| %s | %s | c%s | %s | %s | %s |'
              % (fam, kind, rng(eng), 'c%d' % max(eng) if eng else '-',
                 ', '.join('`%s`' % x for x in rel) or '-', ', '.join(ib) or '-'))

    P('')
    P('### Constants `def`d by the bytecode and also read as engine input elsewhere')
    P('')
    clash = collections.defaultdict(lambda: [set(), set()])
    for r in decoded:
        k = (r.kind, r.name)
        clash[k][0] |= r.c_def
        clash[k][1] |= set(r.c_engine)
    rows = [(k, s[0] & s[1]) for k, s in clash.items() if s[0] & s[1]]
    if rows:
        P('| container | registers |')
        P('| --- | --- |')
        for (kind, name), s in sorted(rows):
            P('| %s `%s` | c%s |' % (kind, name, rng(s)))
    else:
        P('None: within every container, a register is either always `def`d or always uploaded.')

    # ---- modifiers ---------------------------------------------------------------------------
    P('')
    P('## Modifiers')
    P('')
    for label, rs in (('Vertex', vs), ('Pixel', ps)):
        P('### %s' % label)
        P('')
        P('Source modifiers:')
        P('')
        P(counter_table(rs, 'srcmods', '', 'source modifier'))
        P('')
        P('Source swizzles:')
        P('')
        P(counter_table(rs, 'swizzles', '', 'swizzle'))
        P('')
        P('Destination write masks:')
        P('')
        P(counter_table(rs, 'masks', '', 'mask'))
        P('')
        P('Destination modifiers:')
        P('')
        P(counter_table(rs, 'dstmods', '', 'modifier'))
        P('')
        P('Constant definitions:')
        P('')
        P(counter_table(rs, 'defs', '', 'op'))
        P('')

    # ---- flow and texturing -------------------------------------------------------------------
    P('## Flow control, predication, relative addressing')
    P('')
    fc = [r for r in decoded if r.flow or r.pred]
    if fc:
        P('Per container: how many of its records branch, and the shapes they take (a shape is the')
        P('multiset of flow opcodes in one record). Predication: %d records.' % sum(1 for r in fc if r.pred))
        P('')
        P('| container | kind | records | with flow | shapes |')
        P('| --- | --- | ---: | ---: | --- |')
        byc = collections.OrderedDict()
        for r in decoded:
            byc.setdefault((r.kind, r.name), []).append(r)
        for (kind, name), rs in byc.items():
            w = [r for r in rs if r.flow or r.pred]
            if not w:
                continue
            shapes = collections.Counter(' '.join('%s x%d' % kv for kv in sorted(r.flow.items())) for r in w)
            P('| `%s` | %s | %d | %d | %s |' % (name, kind, len(rs), len(w),
                                               '; '.join('%s (%d)' % (k, v) for k, v in shapes.most_common())))
    else:
        P('No record uses `if`/`loop`/`rep`/`call`/`break`/`label` or predication.')
    P('')
    relr = [r for r in decoded if r.c_rel]
    P('Relative addressing: %d records, all in %s. Address registers: %s. Bases: %s.'
      % (len(relr), ', '.join(sorted(set(r.name for r in relr))) or '-',
         ', '.join(sorted(set(t[2] for r in relr for t in r.c_rel))) or '-',
         ', '.join(sorted(set('%s%d' % (t[0], t[1]) for r in relr for t in r.c_rel),
                          key=lambda s: int(re.sub(r'[a-zA-Z]', '', s)))) or '-'))
    P('')
    P('## Texturing')
    P('')
    P('Sampler declarations (pixel):')
    P('')
    st = collections.Counter()
    for r in ps:
        for s, t in r.samplers.items():
            st[('s%d' % s, t)] += 1
    P('| sampler | type | records |')
    P('| --- | --- | ---: |')
    for (s, t), n in sorted(st.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1])):
        P('| `%s` | %s | %d |' % (s, t, n))
    P('')
    vsamp = [r for r in vs if r.samplers]
    P('Vertex-shader samplers (vertex texture fetch): %s.'
      % (', '.join('`%s` %s' % (r.key, r.samplers) for r in vsamp) or 'none'))
    P('')
    P('Texture instructions:')
    P('')
    P(counter_table(decoded, 'tex_ops', '', 'op'))
    P('')
    mv = [r for r in ps if r.vpos_vface]
    P('`vPos` / `vFace` readers: %s.' % (', '.join('`%s` (%s)' % (r.key, ','.join(sorted(r.vpos_vface))) for r in mv) or 'none'))
    P('')
    od = [r for r in ps if 'oDepth' in r.regs_written]
    P('`oDepth` writers: %s.' % (', '.join('`%s`' % r.key for r in od) or 'none'))
    mrt = [r for r in ps if r.max_index.get('oC', 0) > 0]
    P('')
    P('Multiple render targets (`oC1`+): %s.' % (', '.join('`%s`' % r.key for r in mrt) or 'none'))

    # ---- semantics ----------------------------------------------------------------------------
    P('')
    P('## Declared semantics')
    P('')
    P('### Vertex inputs per container')
    P('')
    P('| container | profile | inputs (union over permutations) | permutations |')
    P('| --- | --- | --- | ---: |')
    byc = collections.OrderedDict()
    for r in vs:
        byc.setdefault(r.name, []).append(r)
    for name, rs in byc.items():
        sems = collections.OrderedDict()
        for r in rs:
            for reg, sem, mask, cen in r.inputs:
                sems.setdefault(sem, set()).add(reg)
        P('| `%s` | `%s` | %s | %d |' % (name, rs[0].profile,
                                        ', '.join('%s=%s' % (s, '/'.join(sorted(v))) for s, v in sems.items()),
                                        len(rs)))
    P('')
    P('### Vertex outputs per container')
    P('')
    P('| container | profile | outputs (dcl or written) |')
    P('| --- | --- | --- |')
    for name, rs in byc.items():
        outs = collections.OrderedDict()
        for r in rs:
            if r.outputs_dcl:
                for reg, sem, mask in r.outputs_dcl:
                    outs.setdefault('%s=%s' % (sem, reg), set()).add(mask or 'xyzw')
            else:
                for reg, mask in sorted(r.outputs_written.items()):
                    outs.setdefault(reg, set()).add(''.join(ir.SWZ[k] for k in range(4) if mask & (1 << k)))
        P('| `%s` | `%s` | %s |' % (name, rs[0].profile,
                                   ', '.join('%s.%s' % (k, '/'.join(sorted(v))) for k, v in outs.items())))
    P('')
    P('### Pixel inputs per container')
    P('')
    P('| container | profile | inputs (union over permutations) | centroid | samplers |')
    P('| --- | --- | --- | --- | --- |')
    byp = collections.OrderedDict()
    for r in ps:
        byp.setdefault(r.name, []).append(r)
    for name, rs in byp.items():
        sems = collections.OrderedDict()
        cen = set()
        smp = {}
        for r in rs:
            for reg, sem, mask, c in r.inputs:
                sems.setdefault('%s=%s' % (sem, reg), set()).add(mask or 'xyzw')
                if c:
                    cen.add(reg)
            smp.update(r.samplers)
        P('| `%s` | `%s` | %s | %s | %s |'
          % (name, rs[0].profile, ', '.join('%s.%s' % (k, '/'.join(sorted(v))) for k, v in sems.items()),
             ', '.join(sorted(cen)) or '-', ', '.join('s%d:%s' % kv for kv in sorted(smp.items()))))

    # ---- per container ----------------------------------------------------------------------
    P('')
    P('## Every container')
    P('')
    P('| container | kind | profile | records | distinct | bytes | instr | opcodes | engine c | def c | samplers |')
    P('| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |')
    for (kind, name), rs in sorted(collections.Counter((r.kind, r.name) for r in decoded).keys() and
                                   {(r.kind, r.name): [x for x in decoded if (x.kind, x.name) == (r.kind, r.name)]
                                    for r in decoded}.items()):
        ops = set()
        eng, dfs, smp = set(), set(), {}
        for r in rs:
            ops |= set(k for k in r.ops if not k.startswith('('))
            eng |= set(r.c_engine)
            dfs |= r.c_def
            smp.update(r.samplers)
        bs = sorted(r.bytes for r in rs)
        ni = sorted(r.ninst for r in rs)
        P('| `%s` | %s | `%s` | %d | %d | %d-%d | %d-%d | %s | c%s | c%s | %s |'
          % (name, kind, rs[0].profile, len(rs), distinct(rs), bs[0], bs[-1], ni[0], ni[-1],
             ' '.join(sorted(ops)), rng(eng), rng(dfs),
             ' '.join('s%d:%s' % kv for kv in sorted(smp.items())) or '-'))

    # ---- validation / embedded -------------------------------------------------------------
    if validation is not None:
        P('')
        P('## Decoder validation against fxc')
        P('')
        ok, bad, msgs = validation
        P('%d distinct records compared with `fxc /dumpbin`, %d match line for line, %d differ.' % (ok + bad, ok, bad))
        for m in msgs[:40]:
            P('')
            P('```')
            P(m)
            P('```')
    if embedded is not None:
        P('')
        P('## Shader blobs embedded in Wow.exe')
        P('')
        P('DX9 version tokens found in the image, followed by a decodable stream ending in `end`.')
        P('These are shaders the engine can create without a `.bls` (the probe saw a vertex record')
        P('named `FFXBlurVertex` that no archive holds).')
        P('')
        P('| file offset | profile | instructions | bytes | inputs | engine c | samplers | first lines |')
        P('| --- | --- | ---: | ---: | --- | --- | --- | --- |')
        for off, r in embedded:
            text = ir.render(r.sh).split('\n')[1:5]
            P('| 0x%08X | `%s` | %d | %d | %s | c%s | %s | `%s` |'
              % (off, r.version, r.ninst, r.bytes,
                 ' '.join('%s=%s' % (s, g) for g, s, m, c in r.inputs), rng(r.c_engine),
                 ' '.join('s%d:%s' % kv for kv in sorted(r.samplers.items())) or '-',
                 '; '.join(t.strip() for t in text)))
    return '\n'.join(L)


# ---- validation ---------------------------------------------------------------------------------

NUM = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')


def canon(line):
    s = ' '.join(line.split())

    def f(m):
        t = m.group(0)
        try:
            v = float(t)
        except ValueError:
            return t
        return t if v == int(v) and '.' not in t else '%.5g' % v
    return NUM.sub(f, s)


def validate(recs, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    seen = set()
    ok = bad = 0
    msgs = []
    for r in recs:
        if r.sm1 or r.md5 in seen:
            continue
        seen.add(r.md5)
        p = os.path.join(tmpdir, r.md5 + '.fxo')
        open(p, 'wb').write(r.code)
        res = subprocess.run([FXC, '/nologo', '/dumpbin', p], capture_output=True, text=True)
        if res.returncode != 0:
            bad += 1
            msgs.append('%s: fxc failed: %s' % (r.key, (res.stderr or res.stdout).strip()[:200]))
            continue
        theirs = [canon(l.strip()) for l in res.stdout.splitlines()
                  if l.strip() and not l.strip().startswith('//')]
        mine = [canon(l) for l in ir.render(r.sh).splitlines() if l.strip()]
        if mine and mine[-1] == 'end':
            mine.pop()
        if mine == theirs:
            ok += 1
        else:
            bad += 1
            diff = []
            for i in range(max(len(mine), len(theirs))):
                a = mine[i] if i < len(mine) else '<missing>'
                b = theirs[i] if i < len(theirs) else '<missing>'
                if a != b:
                    diff.append('  %4d mine: %s\n       fxc : %s' % (i, a, b))
            msgs.append('%s (%s): %d lines differ\n%s' % (r.key, r.profile, len(diff), '\n'.join(diff[:6])))
    return ok, bad, msgs


# ---- embedded blobs -----------------------------------------------------------------------------

def scan_embedded(path):
    data = open(path, 'rb').read()
    out = []
    for magic in (b'\x00\x02\xfe\xff', b'\x00\x03\xfe\xff', b'\x00\x02\xff\xff', b'\x00\x03\xff\xff'):
        start = 0
        while True:
            i = data.find(magic, start)
            if i < 0:
                break
            start = i + 1
            if i % 4:
                continue
            end = data.find(b'\xff\xff\x00\x00', i)
            if end < 0 or end - i > 65536:
                continue
            code = data[i:end + 4]
            try:
                sh = ir.decode(code)
            except Exception:
                continue
            if sh.trailing or len(sh.insts) < 2:
                continue
            kind = 'vertex' if sh.kind == 'vs' else 'pixel'
            r = analyse(code, kind, 'embedded', 'embedded@%08X' % i, 0)
            if r.sm1:
                continue
            out.append((i, r))
    return out


# ---- main ---------------------------------------------------------------------------------------

def main(argv):
    outpath = None
    do_validate = '--validate' in argv
    do_embedded = '--embedded' in argv
    dump = None
    if '-o' in argv:
        outpath = argv[argv.index('-o') + 1]
    if '--dump' in argv:
        dump = argv[argv.index('--dump') + 1]
    into = argv[argv.index('--into') + 1] if '--into' in argv else None
    inv = inventory()
    resolved = resolve(inv)
    arch = ea.archives()
    recs = []
    sm1_names = []
    empty = []
    final = []
    for kind, prof, name, depth in resolved:
        # An empty container is a bad file to the engine: it falls back one profile, so do we.
        chain = VS_FALLBACK if kind == 'vertex' else PS_FALLBACK
        data = None
        for prof2 in chain[chain.index(prof):]:
            low = name.lower()
            if low not in inv.get((kind, prof2), {}):
                continue
            path = BS.join(['Shaders', kind.capitalize(), prof2, inv[(kind, prof2)][low] + '.bls'])
            src, data = ea.grab(arch, path)
            if data is None:
                print('MISS %s' % path, file=sys.stderr)
                continue
            if len(data) < 12:
                print('EMPTY %s: %d bytes (from %s)' % (path, len(data), src), file=sys.stderr)
                empty.append((kind, prof2, name, len(data), src))
                data = None
                continue
            prof, depth = prof2, chain.index(prof2)
            break
        if data is None:
            continue
        final.append((kind, prof, name, depth))
        magic, ver, nperm, records, consumed = ea.split(data)
        if consumed != len(data):
            print('TRAILING %s: %d of %d bytes consumed' % (path, consumed, len(data)), file=sys.stderr)
        if dump:
            d = os.path.join(dump, '%s_%s_%s' % (kind.capitalize(), prof, name))
            os.makedirs(d, exist_ok=True)
        for rec in records:
            if dump:
                open(os.path.join(d, 'p%02d.fxo' % rec['i']), 'wb').write(rec['code'])
            r = analyse(rec['code'], kind, prof, name, rec['i'])
            r.code = rec['code']
            r.hdr = (rec['a'], rec['b'], rec['c'], rec['d'])
            if r.sm1:
                sm1_names.append((kind, prof, name))
                break
            recs.append(r)
    sm1_names = sorted(set(sm1_names))
    validation = None
    if do_validate:
        validation = validate(recs, os.path.join(HERE, 'out', 'census-validate'))
        print('validation: %d ok, %d bad' % (validation[0], validation[1]), file=sys.stderr)
    embedded = scan_embedded(WOW) if do_embedded else None
    text = md(recs, final, inv, validation, embedded, sm1_names, empty)
    if outpath:
        open(outpath, 'w', encoding='utf-8').write(text + '\n')
        print('wrote %s (%d records)' % (outpath, len(recs)), file=sys.stderr)
    if into:
        marker = '<!-- census:generated -->'
        doc = open(into, encoding='utf-8').read()
        if marker not in doc:
            print('%s has no %s line' % (into, marker), file=sys.stderr)
            sys.exit(1)
        head = doc[:doc.index(marker) + len(marker)]
        open(into, 'w', encoding='utf-8').write(head + '\n\n' + text + '\n')
        print('updated %s after the marker (%d records)' % (into, len(recs)), file=sys.stderr)
    if not outpath and not into:
        sys.stdout.write(text + '\n')
    return recs


if __name__ == '__main__':
    main(sys.argv[1:])
