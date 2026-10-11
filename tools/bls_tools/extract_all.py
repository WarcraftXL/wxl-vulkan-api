"""Extract every permutation of one BLS container and disassemble each."""
import sys, os, struct, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('mq', os.path.join(HERE, 'mpq_extract.py'))
mq = importlib.util.module_from_spec(spec); spec.loader.exec_module(mq)
spec2 = importlib.util.spec_from_file_location('dz', os.path.join(HERE, 'dx9d.py'))
dz = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(dz)

# The client's Data folder (the one holding common.MPQ): translate_shaders.py --client sets it.
DATADIR = os.environ.get('WXL_CLIENT_DATA', '')
BS = chr(92)

def archives():
    if not DATADIR or not os.path.isfile(os.path.join(DATADIR, 'common.MPQ')):
        sys.exit("no client archives: set WXL_CLIENT_DATA to the client's Data folder (or run translate_shaders.py "
                 "--client <client folder>); looked in %r" % DATADIR)
    out = []
    for n in mq.LOAD_ORDER:
        p = os.path.join(DATADIR, n)
        if os.path.isdir(p):
            sys.stderr.write('-- %s is a folder patch (loose files), not an archive: not read
' % n)
            continue
        if os.path.exists(p):
            try:
                out.append((n, mq.MPQ(p)))
            except OSError as e:
                sys.stderr.write('!! %s unreadable, skipped: %s\n' % (n, e))
    return out

def grab(arch, path):
    found = None
    for n, m in arch:
        if m.find(path) is not None:
            found = (n, m)          # later archive wins (load order)
    if not found:
        return None, None
    n, m = found
    data, _ = m.read_file(path)
    return n, data

def split(data):
    magic, ver, nperm = struct.unpack_from('<III', data, 0)
    recs, off = [], 12
    for i in range(nperm):
        a, b, c, d, size = struct.unpack_from('<IIHHI', data, off); off += 16
        code = data[off:off+size]; off += size; off = (off + 3) & ~3
        recs.append(dict(i=i, a=a, b=b, c=c, d=d, size=size, code=code))
    return magic, ver, nperm, recs, off

if __name__ == '__main__':
    kind, profile, name = sys.argv[1], sys.argv[2], sys.argv[3]
    path = BS.join(['Shaders', kind, profile, name + '.bls'])
    arch = archives()
    src_name, data = grab(arch, path)
    if data is None:
        print('MISS %s' % path); sys.exit(1)
    magic, ver, nperm, recs, consumed = split(data)
    print('%s' % path)
    print('  from %s, %d bytes, magic %08X, version %05X, %d permutations'
          % (src_name, len(data), magic, ver, nperm))
    print('  consumed %d / %d%s' % (consumed, len(data),
          '' if consumed == len(data) else '  !! TRAILING'))
    outdir = '%s_%s_%s' % (kind, profile, name)
    os.makedirs(outdir, exist_ok=True)
    for r in recs:
        fn = os.path.join(outdir, 'p%02d.fxo' % r['i'])
        open(fn, 'wb').write(r['code'])
    print()
    for r in recs:
        print('=== perm %d  (a=%d b=%d c=%d d=%d, %d bytes) ==='
              % (r['i'], r['a'], r['b'], r['c'], r['d'], r['size']))
        print(dz.disasm(r['code']))
        print()
