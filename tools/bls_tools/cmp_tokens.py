"""Compare two DX9 token streams after dropping comment tokens.

fxc always emits a comment block (the constant table); the shipped .bls records have none.
It is inert to the driver, so the meaningful comparison is the stream without it.
"""
import struct, sys


def strip_comments(path):
    d = open(path, 'rb').read()
    toks = list(struct.unpack('<%dI' % (len(d) // 4), d))
    out = [toks[0]]
    i = 1
    while i < len(toks):
        t = toks[i]
        op = t & 0xFFFF
        if op == 0xFFFE:                 # comment: skip it and its payload
            i += 1 + ((t >> 16) & 0x7FFF)
            continue
        if op == 0xFFFF:
            out.append(t)
            i += 1
            break
        n = 1 + ((t >> 24) & 0xF)
        out.extend(toks[i:i + n])
        i += n
    return out


a, b = strip_comments(sys.argv[1]), strip_comments(sys.argv[2])
print('%s : %d tokens' % (sys.argv[1], len(a)))
print('%s : %d tokens' % (sys.argv[2], len(b)))
if a == b:
    print('IDENTICAL token for token (%d bytes each)' % (len(a) * 4))
    sys.exit(0)
print('differ:')
for i in range(max(len(a), len(b))):
    x = a[i] if i < len(a) else None
    y = b[i] if i < len(b) else None
    if x != y:
        print('  token %3d  %s  vs  %s'
              % (i, '--------' if x is None else '%08X' % x,
                 '--------' if y is None else '%08X' % y))
