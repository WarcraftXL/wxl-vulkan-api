"""Compare dx9d.py's decoding against fxc /dumpbin, line for line.

Any mismatch is my decoder being wrong, not fxc. Floats are compared numerically
because fxc prints more digits than %g does.
"""
import glob, os, re, subprocess, sys, importlib.util

spec = importlib.util.spec_from_file_location('dx9d', 'dx9d.py')
dx9d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dx9d)

FXC = os.environ.get('FXC') or 'fxc.exe'   # the Windows SDK's fxc: $FXC, else PATH
NUM = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')


def canon(line):
    """Collapse whitespace and round every float, so 0.699999988 == 0.7."""
    s = ' '.join(line.split())
    def f(m):
        t = m.group(0)
        try:
            v = float(t)
        except ValueError:
            return t
        return t if v == int(v) and '.' not in t else '%.5g' % v
    return NUM.sub(f, s)


def fxc_lines(path):
    r = subprocess.run([FXC, '/nologo', '/dumpbin', path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip() or r.stdout.strip()
    out = []
    for ln in r.stdout.splitlines():
        s = ln.strip()
        if not s or s.startswith('//'):
            continue
        out.append(canon(s))
    return out, None


def main(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print('no files match %s' % pattern)
        return 1
    bad = 0
    for p in files:
        mine = [canon(l) for l in dx9d.disasm(open(p, 'rb').read()).splitlines()
                if l.strip()]
        if mine and mine[-1] == 'end':   # fxc does not print it
            mine.pop()
        theirs, err = fxc_lines(p)
        name = os.path.basename(p)
        if theirs is None:
            print('%-10s FXC FAILED: %s' % (name, err))
            bad += 1
            continue
        if mine == theirs:
            print('%-10s OK   %d instructions' % (name, len(mine)))
            continue
        bad += 1
        print('%-10s MISMATCH (mine %d lines, fxc %d lines)'
              % (name, len(mine), len(theirs)))
        for i in range(max(len(mine), len(theirs))):
            a = mine[i] if i < len(mine) else '<missing>'
            b = theirs[i] if i < len(theirs) else '<missing>'
            if a != b:
                print('    %4d  mine: %s' % (i, a))
                print('          fxc : %s' % b)
    print()
    print('%d / %d permutations match fxc exactly' % (len(files) - bad, len(files)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
