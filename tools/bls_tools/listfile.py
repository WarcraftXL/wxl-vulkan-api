"""Dump the shader inventory from the archives' (listfile).

The MPQs carry a (listfile), so the shader set is enumerable rather than guessable -- which is how
shaders\\effects\\MapObj.wfx was found. Without it, a name has to be known before it can be read:
the reader here resolves names through the hash table and cannot walk it.
"""
import collections, os, sys, importlib.util

spec = importlib.util.spec_from_file_location('mq', os.path.join(os.path.dirname(__file__) or '.',
                                                                 'mpq_extract.py'))
mq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mq)

# The client's Data folder (the one holding common.MPQ): translate_shaders.py --client sets it.
DATADIR = os.environ.get('WXL_CLIENT_DATA', '')
BS = chr(92)


def entries(datadir=DATADIR, prefix='shaders'):
    """Every listed path under `prefix`, across the whole load order, case-insensitively."""
    out = set()
    for n in mq.LOAD_ORDER:
        p = os.path.join(datadir, n)
        if not os.path.isfile(p):   # absent, or a folder patch (loose files)
            continue
        try:
            m = mq.MPQ(p)
        except OSError as e:
            sys.stderr.write('!! %s unreadable, skipped: %s\n' % (n, e))
            continue
        if m.find('(listfile)') is None:
            continue
        data, _ = m.read_file('(listfile)')
        for ln in data.decode('ascii', 'replace').replace('\r\n', '\n').split('\n'):
            ln = ln.strip()
            if ln.lower().startswith(prefix.lower()):
                out.add(ln)
    return sorted(out)


if __name__ == '__main__':
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'shaders'
    names = entries(prefix=prefix)
    print('%d entries under %s' % (len(names), prefix))

    # The archive spellings are inconsistent ("Shaders\Pixel" and "shaders\pixel" both occur), so
    # group case-insensitively or the counts split in two.
    by_kind = collections.defaultdict(set)
    for n in names:
        parts = n.split(BS)
        if len(parts) >= 3:
            by_kind[parts[1].lower()].add(parts[2].lower())
    for kind in sorted(by_kind):
        print('  %-10s %s' % (kind, ' '.join(sorted(by_kind[kind]))))

    for n in names:
        print(n)
