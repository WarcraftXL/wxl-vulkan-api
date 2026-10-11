"""DX9 (SM2/SM3) token-stream disassembler. Tables from the D3DSIO/D3DSPSM enums.

Validated against fxc /dumpbin -- see validate_disasm.py. The previous version of this
file had a wrong opcode table above 66 (it mapped 70 to cmp, 88 to texldl) which never
showed up because the UI shader used none of those.
"""
import struct, sys, os

# D3DSHADER_PARAM_REGISTER_TYPE, as the ps/vs assembler names them
REGT_VS = {0: 'r', 1: 'v', 2: 'c', 3: 'a', 4: 'oPos', 5: 'oD', 6: 'o',
           7: 'i', 8: 'oC', 9: 'oDepth', 10: 's', 11: 'c2', 12: 'c3',
           13: 'c4', 14: 'b', 15: 'aL', 16: 'rh', 17: 'misc', 18: 'l', 19: 'p'}
REGT_PS = dict(REGT_VS)
REGT_PS[3] = 't'

D3DSIO = {
    0: 'nop', 1: 'mov', 2: 'add', 3: 'sub', 4: 'mad', 5: 'mul', 6: 'rcp',
    7: 'rsq', 8: 'dp3', 9: 'dp4', 10: 'min', 11: 'max', 12: 'slt', 13: 'sge',
    14: 'exp', 15: 'log', 16: 'lit', 17: 'dst', 18: 'lrp', 19: 'frc',
    20: 'm4x4', 21: 'm4x3', 22: 'm3x4', 23: 'm3x3', 24: 'm3x2',
    25: 'call', 26: 'callnz', 27: 'loop', 28: 'ret', 29: 'endloop', 30: 'label',
    31: 'dcl', 32: 'pow', 33: 'crs', 34: 'sgn', 35: 'abs', 36: 'nrm',
    37: 'sincos', 38: 'rep', 39: 'endrep', 40: 'if', 41: 'if', 42: 'else',
    43: 'endif', 44: 'break', 45: 'breakc', 46: 'mova', 47: 'defb', 48: 'defi',
    64: 'texcoord', 65: 'texkill', 66: 'texld', 67: 'texbem', 68: 'texbeml',
    69: 'texreg2ar', 70: 'texreg2gb', 71: 'texm3x2pad', 72: 'texm3x2tex',
    73: 'texm3x3pad', 74: 'texm3x3tex', 75: 'reserved0', 76: 'texm3x3spec',
    77: 'texm3x3vspec', 78: 'expp', 79: 'logp', 80: 'cnd', 81: 'def',
    82: 'texreg2rgb', 83: 'texdp3tex', 84: 'texm3x2depth', 85: 'texdp3',
    86: 'texm3x3', 87: 'texdepth', 88: 'cmp', 89: 'bem', 90: 'dp2add',
    91: 'dsx', 92: 'dsy', 93: 'texldd', 94: 'setp', 95: 'texldl',
    96: 'breakp',
}
# instructions that take no destination register
NO_DEST = {0, 25, 26, 27, 28, 29, 30, 38, 39, 40, 41, 42, 43, 44, 45, 96}
# texkill takes a destination register whose write mask carries no meaning
BARE_DEST = {65}
# D3DSHADER_COMPARISON, bits 16..18 of the instruction token
CMPOP = {1: '_gt', 2: '_eq', 3: '_ge', 4: '_lt', 5: '_ne', 6: '_le'}
# D3DSHADER_PARAM_SRCMOD_TYPE, bits 24..27 of a source token, as (prefix, suffix).
# The suffix binds to the register name and the swizzle follows it, which is how fxc
# spells it: v4_abs.x, not |v4.x|.
SRCMOD = {0: ('', ''), 1: ('-', ''), 2: ('', '_bias'), 3: ('-', '_bias'),
          4: ('', '_bx2'), 5: ('-', '_bx2'), 6: ('1-', ''), 7: ('', '_x2'),
          8: ('-', '_x2'), 9: ('', '_dz'), 10: ('', '_dw'), 11: ('', '_abs'),
          12: ('-', '_abs'), 13: ('!', '')}
USAGE = {0: 'position', 1: 'blendweight', 2: 'blendindices', 3: 'normal',
         4: 'psize', 5: 'texcoord', 6: 'tangent', 7: 'binormal',
         8: 'tessfactor', 9: 'positiont', 10: 'color', 11: 'fog',
         12: 'depth', 13: 'sample'}
TEXTYPE = {1: 'unknown', 2: '2d', 3: 'cube', 4: 'volume'}
SWZ = 'xyzw'


def regtype(tok):
    """Register type is split: low 3 bits at 28..30, high 2 bits at 11..12."""
    return ((tok >> 28) & 0x7) | (((tok >> 11) & 0x3) << 3)


def regname(tok, kind):
    rt = regtype(tok)
    table = REGT_PS if kind == 'ps' else REGT_VS
    nm = table.get(rt, '?%d' % rt)
    return '%s%d' % (nm, tok & 0x7FF), rt


def dest(tok, kind):
    n, rt = regname(tok, kind)
    mask = (tok >> 16) & 0xF
    mods = (tok >> 20) & 0xF
    shift = (tok >> 24) & 0xF
    suffix = ''
    if mods & 1:
        suffix += '_sat'
    if mods & 2:
        suffix += '_pp'
    if mods & 4:
        suffix += '_centroid'
    if shift:
        suffix += '_shift%d' % (shift if shift < 8 else shift - 16)
    s = n
    if mask != 0xF:
        s += '.' + ''.join(SWZ[i] for i in range(4) if mask & (1 << i))
    return suffix, s


def src(tok, kind):
    n, rt = regname(tok, kind)
    sw = (tok >> 16) & 0xFF
    mod = (tok >> 24) & 0xF
    pre, suf = SRCMOD.get(mod, ('?mod%d?' % mod, ''))
    s = pre + n + suf
    if sw != 0xE4:
        comps = [SWZ[(sw >> (2 * i)) & 3] for i in range(4)]
        s += '.' + (comps[0] if len(set(comps)) == 1 else ''.join(comps))
    return s


def disasm(code, want_regs=False):
    toks = struct.unpack('<%dI' % (len(code) // 4), code)
    v = toks[0]
    kind = 'ps' if (v >> 16) == 0xFFFF else ('vs' if (v >> 16) == 0xFFFE else '??')
    out = ['%s_%d_%d' % (kind, (v >> 8) & 0xFF, v & 0xFF)]
    defined = set()     # c registers supplied by def/defi/defb -- shader-local
    read = set()        # c registers read as a source
    samplers = set()
    inputs = []
    i = 1
    while i < len(toks):
        t = toks[i]
        op = t & 0xFFFF
        if op == 0xFFFF:
            out.append('end')
            i += 1
            break
        if op == 0xFFFE:
            n = (t >> 16) & 0x7FFF
            out.append('// comment, %d dwords' % n)
            i += 1 + n
            continue
        ln = (t >> 24) & 0xF
        args = toks[i + 1: i + 1 + ln]
        name = D3DSIO.get(op, 'op%d' % op)
        if op in (41, 45, 94):                       # ifc / breakc / setp
            name += CMPOP.get((t >> 16) & 0x7, '_cmp%d' % ((t >> 16) & 0x7))
        if op == 66:                                 # texld modifiers
            m = (t >> 16) & 0xF
            if m == 1:
                name = 'texldp'
            elif m == 2:
                name = 'texldb'
        if op == 31:                                 # dcl
            info, dt = args[0], args[1]
            rt = regtype(dt)
            _, dn = dest(dt, kind)
            if rt == 10:
                tt = (info >> 27) & 0xF
                out.append('dcl_%s %s' % (TEXTYPE.get(tt, 'tt%d' % tt), dn))
                samplers.add(dt & 0x7FF)
            else:
                u, ui = info & 0x1F, (info >> 16) & 0xF
                tag = '%s%s' % (USAGE.get(u, 'usage%d' % u), ui if ui else '')
                out.append('dcl_%s %s' % (tag, dn))
                inputs.append((dn, tag))
        elif op == 81:                               # def cN, f, f, f, f
            f = struct.unpack('<4f', struct.pack('<4I', *args[1:5]))
            _, dn = dest(args[0], kind)
            out.append('def %s, %g, %g, %g, %g' % (dn, f[0], f[1], f[2], f[3]))
            defined.add(args[0] & 0x7FF)
        elif op == 48:                               # defi
            _, dn = dest(args[0], kind)
            iv = [struct.unpack('<i', struct.pack('<I', a))[0] for a in args[1:5]]
            out.append('defi %s, %d, %d, %d, %d' % (dn, iv[0], iv[1], iv[2], iv[3]))
            defined.add(args[0] & 0x7FF)
        elif op == 47:                               # defb
            _, dn = dest(args[0], kind)
            out.append('defb %s, %s' % (dn, 'true' if args[1] else 'false'))
            defined.add(args[0] & 0x7FF)
        elif op in BARE_DEST:
            n, _ = regname(args[0], kind)
            out.append('%s %s' % (name, n))
        elif ln == 0:
            out.append(name)
        else:
            if op in NO_DEST:
                out.append('%s %s' % (name, ', '.join(src(a, kind) for a in args)))
                srcs = args
            else:
                sfx, dn = dest(args[0], kind)
                parts = [dn] + [src(a, kind) for a in args[1:]]
                out.append('%s%s %s' % (name, sfx, ', '.join(parts)))
                srcs = args[1:]
            for a in srcs:
                if regtype(a) == 2:
                    read.add(a & 0x7FF)
        i += 1 + ln
    if i != len(toks):
        out.append('!! %d tokens past end' % (len(toks) - i))
    text = '\n'.join('    ' + l for l in out)
    if not want_regs:
        return text
    return text, dict(defined=sorted(defined), read=sorted(read),
                      engine=sorted(read - defined), samplers=sorted(samplers),
                      inputs=inputs)


if __name__ == '__main__':
    for p in sorted(sys.argv[1:]):
        print('##### %s' % os.path.basename(p))
        print(disasm(open(p, 'rb').read()))
        print()
