"""Per-permutation constant contract for one BLS container.

A 'def' register is shader-local: the driver supplies it from the bytecode. Every other
c register read is uploaded by the engine, and that set is the contract a replacement
shader has to honour.
"""
import glob, os, sys, importlib.util

spec = importlib.util.spec_from_file_location('dx9d', 'dx9d.py')
dx9d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dx9d)


def rng(xs):
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


def main(pattern):
    files = sorted(glob.glob(pattern))
    rows = []
    for p in files:
        _, info = dx9d.disasm(open(p, 'rb').read(), want_regs=True)
        rows.append((os.path.basename(p), info, os.path.getsize(p)))

    print('%-6s %6s  %-14s  %-16s  %-10s  %s'
          % ('perm', 'bytes', 'engine c regs', 'def c regs', 'samplers', 'inputs'))
    print('-' * 110)
    for name, i, size in rows:
        ins = ' '.join('%s:%s' % (r, t) for r, t in i['inputs'])
        print('%-6s %6d  %-14s  %-16s  %-10s  %s'
              % (name.replace('.fxo', ''), size, rng(i['engine']),
                 rng(i['defined']), rng(i['samplers']), ins))

    allengine = sorted(set().union(*[set(i['engine']) for _, i, _ in rows]))
    alldef = sorted(set().union(*[set(i['defined']) for _, i, _ in rows]))
    print()
    print('engine-supplied across all permutations: c%s' % rng(allengine))
    print('shader-defined across all permutations:  c%s' % rng(alldef))
    clash = sorted(set(allengine) & set(alldef))
    print('registers def-ed in one permutation and engine-read in another: c%s'
          % rng(clash))


if __name__ == '__main__':
    main(sys.argv[1])
