"""Structured DX9 (SM2/SM3) token-stream decoder: the census's and a translator's front end.

dx9d.py renders text and is validated against fxc; it stays as it is. This module decodes the
same stream into objects (one Inst per instruction, one Operand per parameter) so that a
census can count things and a translator can walk them. It handles what dx9d.py does not:
relative addressing (`c31[a0.x]`, `c31[aL]`, an extra token after the parameter in SM2+),
predication (`(p0.x) op ...`, an extra token after the destination), and the SM2 pixel `dcl`
forms. render() prints fxc's spelling so validate_disasm.py's comparison applies here too
(see census.py --validate).

Only SM2 and SM3 streams are decoded: every parameter count comes from the instruction
token's length field, which SM1 streams leave at zero.
"""
import struct

# D3DSHADER_PARAM_REGISTER_TYPE
RT_TEMP, RT_INPUT, RT_CONST, RT_ADDR, RT_RASTOUT, RT_ATTROUT, RT_TEXCRDOUT = range(7)
RT_CONSTINT, RT_COLOROUT, RT_DEPTHOUT, RT_SAMPLER = 7, 8, 9, 10
RT_CONST2, RT_CONST3, RT_CONST4, RT_CONSTBOOL, RT_LOOP = 11, 12, 13, 14, 15
RT_TEMPFLOAT16, RT_MISCTYPE, RT_LABEL, RT_PREDICATE = 16, 17, 18, 19
RT_TEXTURE = RT_ADDR      # same code, pixel side
RT_OUTPUT = RT_TEXCRDOUT  # same code, vs_3_0

RT_NAMES = {RT_TEMP: 'r', RT_INPUT: 'v', RT_CONST: 'c', RT_ADDR: 'a', RT_RASTOUT: 'oPos',
            RT_ATTROUT: 'oD', RT_TEXCRDOUT: 'oT', RT_CONSTINT: 'i', RT_COLOROUT: 'oC',
            RT_DEPTHOUT: 'oDepth', RT_SAMPLER: 's', RT_CONST2: 'c2', RT_CONST3: 'c3',
            RT_CONST4: 'c4', RT_CONSTBOOL: 'b', RT_LOOP: 'aL', RT_TEMPFLOAT16: 'rh',
            RT_MISCTYPE: 'misc', RT_LABEL: 'l', RT_PREDICATE: 'p'}

# D3DSHADER_INSTRUCTION_OPCODE_TYPE, as fxc spells them
OPS = {
    0: 'nop', 1: 'mov', 2: 'add', 3: 'sub', 4: 'mad', 5: 'mul', 6: 'rcp', 7: 'rsq', 8: 'dp3',
    9: 'dp4', 10: 'min', 11: 'max', 12: 'slt', 13: 'sge', 14: 'exp', 15: 'log', 16: 'lit',
    17: 'dst', 18: 'lrp', 19: 'frc', 20: 'm4x4', 21: 'm4x3', 22: 'm3x4', 23: 'm3x3', 24: 'm3x2',
    25: 'call', 26: 'callnz', 27: 'loop', 28: 'ret', 29: 'endloop', 30: 'label', 31: 'dcl',
    32: 'pow', 33: 'crs', 34: 'sgn', 35: 'abs', 36: 'nrm', 37: 'sincos', 38: 'rep', 39: 'endrep',
    40: 'if', 41: 'if', 42: 'else', 43: 'endif', 44: 'break', 45: 'breakc', 46: 'mova',
    47: 'defb', 48: 'defi',
    64: 'texcoord', 65: 'texkill', 66: 'texld', 67: 'texbem', 68: 'texbeml', 69: 'texreg2ar',
    70: 'texreg2gb', 71: 'texm3x2pad', 72: 'texm3x2tex', 73: 'texm3x3pad', 74: 'texm3x3tex',
    75: 'reserved0', 76: 'texm3x3spec', 77: 'texm3x3vspec', 78: 'expp', 79: 'logp', 80: 'cnd',
    81: 'def', 82: 'texreg2rgb', 83: 'texdp3tex', 84: 'texm3x2depth', 85: 'texdp3', 86: 'texm3x3',
    87: 'texdepth', 88: 'cmp', 89: 'bem', 90: 'dp2add', 91: 'dsx', 92: 'dsy', 93: 'texldd',
    94: 'setp', 95: 'texldl', 96: 'breakp',
}
OP_DCL, OP_DEF, OP_DEFI, OP_DEFB = 31, 81, 48, 47
OP_IF, OP_IFC, OP_ELSE, OP_ENDIF = 40, 41, 42, 43
OP_LOOP, OP_ENDLOOP, OP_REP, OP_ENDREP = 27, 29, 38, 39
OP_CALL, OP_CALLNZ, OP_RET, OP_LABEL = 25, 26, 28, 30
OP_BREAK, OP_BREAKC, OP_BREAKP = 44, 45, 96
OP_TEXKILL, OP_TEXLD, OP_TEXLDD, OP_TEXLDL, OP_SETP = 65, 66, 93, 95, 94
OP_SINCOS = 37
OP_END, OP_COMMENT = 0xFFFF, 0xFFFE

# instructions without a destination parameter
NO_DEST = {0, 25, 26, 27, 28, 29, 30, 38, 39, 40, 41, 42, 43, 44, 45, 96}
FLOW = {25, 26, 27, 28, 29, 30, 38, 39, 40, 41, 42, 43, 44, 45, 96}

CMPOP = {1: '_gt', 2: '_eq', 3: '_ge', 4: '_lt', 5: '_ne', 6: '_le'}
# D3DSHADER_PARAM_SRCMOD_TYPE: (prefix, suffix, census name)
SRCMOD = {0: ('', '', 'none'), 1: ('-', '', 'neg'), 2: ('', '_bias', 'bias'),
          3: ('-', '_bias', 'bias_neg'), 4: ('', '_bx2', 'bx2'), 5: ('-', '_bx2', 'bx2_neg'),
          6: ('1-', '', 'comp'), 7: ('', '_x2', 'x2'), 8: ('-', '_x2', 'x2_neg'),
          9: ('', '_dz', 'dz'), 10: ('', '_dw', 'dw'), 11: ('', '_abs', 'abs'),
          12: ('-', '_abs', 'abs_neg'), 13: ('!', '', 'not')}
USAGE = {0: 'position', 1: 'blendweight', 2: 'blendindices', 3: 'normal', 4: 'psize',
         5: 'texcoord', 6: 'tangent', 7: 'binormal', 8: 'tessfactor', 9: 'positiont',
         10: 'color', 11: 'fog', 12: 'depth', 13: 'sample'}
TEXTYPE = {1: 'unknown', 2: '2d', 3: 'cube', 4: 'volume'}
SWZ = 'xyzw'


def regtype(tok):
    return ((tok >> 28) & 0x7) | (((tok >> 11) & 0x3) << 3)


class Operand(object):
    """One parameter token (plus its relative-address token when bit 13 is set)."""
    __slots__ = ('rtype', 'index', 'rel', 'mask', 'sat', 'pp', 'centroid', 'shift',
                 'swizzle', 'mod', 'is_dest')

    def __init__(self, tok, is_dest):
        self.rtype = regtype(tok)
        self.index = tok & 0x7FF
        self.rel = None                      # (rtype, index, component) of the address register
        self.is_dest = is_dest
        if is_dest:
            self.mask = (tok >> 16) & 0xF
            mods = (tok >> 20) & 0xF
            self.sat, self.pp, self.centroid = bool(mods & 1), bool(mods & 2), bool(mods & 4)
            sh = (tok >> 24) & 0xF
            self.shift = sh if sh < 8 else sh - 16
            self.swizzle, self.mod = 0xE4, 0
        else:
            self.swizzle = (tok >> 16) & 0xFF
            self.mod = (tok >> 24) & 0xF
            self.mask, self.sat, self.pp, self.centroid, self.shift = 0xF, False, False, False, 0

    def comps(self):
        return [(self.swizzle >> (2 * i)) & 3 for i in range(4)]

    def swizzle_kind(self):
        c = self.comps()
        if self.swizzle == 0xE4:
            return 'identity'
        if len(set(c)) == 1:
            return 'replicate'
        if c == [0, 1, 2, 2] or c == [0, 1, 2, 1]:
            return 'arbitrary'
        return 'arbitrary'

    def name(self, kind, major):
        """Register spelling for this shader kind and version."""
        rt, i = self.rtype, self.index
        if kind == 'vs':
            if rt == RT_RASTOUT:
                base = {0: 'oPos', 1: 'oFog', 2: 'oPts'}.get(i, 'oRast%d' % i)
                return base
            if rt == RT_ATTROUT:
                return 'oD%d' % i
            if rt == RT_TEXCRDOUT:
                return ('o%d' if major >= 3 else 'oT%d') % i
        else:
            if rt == RT_TEXTURE:
                return 't%d' % i
            if rt == RT_MISCTYPE:
                return {0: 'vPos', 1: 'vFace'}.get(i, 'misc%d' % i)
        if rt == RT_LOOP:
            return 'aL'
        return '%s%d' % (RT_NAMES.get(rt, '?%d' % rt), i)


class Inst(object):
    __slots__ = ('op', 'name', 'ctrl', 'predicated', 'coissue', 'dst', 'srcs', 'pred',
                 'values', 'dcl_usage', 'dcl_index', 'dcl_textype', 'pos')

    def __init__(self):
        self.op = 0
        self.name = ''
        self.ctrl = 0
        self.predicated = False
        self.coissue = False
        self.dst = None
        self.srcs = []
        self.pred = None
        self.values = None        # def/defi/defb payload
        self.dcl_usage = None     # (usage, index) or None
        self.dcl_index = 0
        self.dcl_textype = None
        self.pos = 0

    def mnemonic(self):
        """fxc's spelling of the opcode with its control bits folded in."""
        n = self.name
        if self.op in (OP_IFC, OP_BREAKC, OP_SETP):
            n += CMPOP.get((self.ctrl >> 0) & 0x7, '_cmp%d' % (self.ctrl & 0x7))
        elif self.op == OP_TEXLD:
            if self.ctrl & 1:
                n = 'texldp'
            elif self.ctrl & 2:
                n = 'texldb'
        return n


class Shader(object):
    def __init__(self, kind, major, minor):
        self.kind, self.major, self.minor = kind, major, minor
        self.insts = []
        self.comments = 0
        self.trailing = 0

    @property
    def profile(self):
        return '%s_%d_%d' % (self.kind, self.major, self.minor)


def _param(toks, i, is_dest, major):
    """Decode one parameter at toks[i]; returns (Operand, tokens consumed)."""
    tok = toks[i]
    o = Operand(tok, is_dest)
    n = 1
    if tok & (1 << 13):                      # D3DSHADER_ADDRMODE_RELATIVE
        if major >= 2:
            a = toks[i + 1]
            o.rel = (regtype(a), a & 0x7FF, (a >> 16) & 3)
            n = 2
        else:
            o.rel = (RT_ADDR, 0, 0)          # SM1: implicit a0.x, no token
    return o, n


def decode(code):
    toks = struct.unpack('<%dI' % (len(code) // 4), code)
    v = toks[0]
    hi = v >> 16
    kind = 'ps' if hi == 0xFFFF else ('vs' if hi == 0xFFFE else None)
    if kind is None:
        raise ValueError('not a DX9 shader: version token %08X' % v)
    major, minor = (v >> 8) & 0xFF, v & 0xFF
    if major < 2:
        raise ValueError('SM%d.%d stream: parameter counts are not encoded' % (major, minor))
    sh = Shader(kind, major, minor)
    i = 1
    while i < len(toks):
        t = toks[i]
        op = t & 0xFFFF
        if op == OP_END:
            i += 1
            break
        if op == OP_COMMENT:
            sh.comments += 1
            i += 1 + ((t >> 16) & 0x7FFF)
            continue
        ln = (t >> 24) & 0xF
        ins = Inst()
        ins.pos = i
        ins.op = op
        ins.name = OPS.get(op, 'op%d' % op)
        ins.ctrl = (t >> 16) & 0xFF
        ins.predicated = bool(t & (1 << 28))
        ins.coissue = bool(t & (1 << 30))
        args = list(toks[i + 1: i + 1 + ln])
        j = 0
        if op == OP_DCL:
            info = args[0]
            d, n = _param(args, 1, True, major)
            ins.dst = d
            if d.rtype == RT_SAMPLER:
                ins.dcl_textype = (info >> 27) & 0xF
            else:
                ins.dcl_usage = (info & 0x1F, (info >> 16) & 0xF)
        elif op == OP_DEF:
            d, n = _param(args, 0, True, major)
            ins.dst = d
            ins.values = struct.unpack('<4f', struct.pack('<4I', *args[1:5]))
        elif op == OP_DEFI:
            d, n = _param(args, 0, True, major)
            ins.dst = d
            ins.values = struct.unpack('<4i', struct.pack('<4I', *args[1:5]))
        elif op == OP_DEFB:
            d, n = _param(args, 0, True, major)
            ins.dst = d
            ins.values = (bool(args[1]),)
        else:
            if op not in NO_DEST and ln > 0:
                d, n = _param(args, 0, True, major)
                ins.dst = d
                j = n
            if ins.predicated:
                p, n = _param(args, j, False, major)
                ins.pred = p
                j += n
            while j < len(args):
                s, n = _param(args, j, False, major)
                ins.srcs.append(s)
                j += n
        sh.insts.append(ins)
        i += 1 + ln
    sh.trailing = len(toks) - i
    return sh


# ---- rendering, in fxc's spelling -------------------------------------------------------------

def _fmt_rel(o, kind, major, base):
    rt, ri, comp = o.rel
    if rt == RT_LOOP:
        a = 'aL'
    else:
        a = '%s%d.%s' % (RT_NAMES.get(rt, '?'), ri, SWZ[comp])
    return '%s%d[%s]' % (base, o.index, a)      # fxc spells c31[a0.x], not c[a0.x + 31]


def fmt_dest(o, kind, major):
    if o.rel:
        s = _fmt_rel(o, kind, major, RT_NAMES.get(o.rtype, '?'))
    else:
        s = o.name(kind, major)
    if o.mask != 0xF:
        s += '.' + ''.join(SWZ[k] for k in range(4) if o.mask & (1 << k))
    return s


def fmt_src(o, kind, major):
    pre, suf, _ = SRCMOD.get(o.mod, ('?mod%d?' % o.mod, '', '?'))
    if o.rel:
        base = RT_NAMES.get(o.rtype, '?') if o.rtype != RT_TEXCRDOUT else ('o' if major >= 3 else 'oT')
        s = _fmt_rel(o, kind, major, base)
    else:
        s = o.name(kind, major)
    s = pre + s + suf
    if o.swizzle != 0xE4:
        c = o.comps()
        s += '.' + (SWZ[c[0]] if len(set(c)) == 1 else ''.join(SWZ[k] for k in c))
    return s


def dest_suffix(o):
    s = ''
    if o.sat:
        s += '_sat'
    if o.pp:
        s += '_pp'
    if o.centroid:
        s += '_centroid'
    if o.shift:
        s += '_x%d' % (1 << o.shift) if o.shift > 0 else '_d%d' % (1 << -o.shift)
    return s


def render_inst(ins, sh):
    kind, major = sh.kind, sh.major
    op = ins.op
    if op == OP_DCL:
        d = ins.dst
        if d.rtype == RT_SAMPLER:
            return 'dcl_%s %s' % (TEXTYPE.get(ins.dcl_textype, 'tt%d' % ins.dcl_textype),
                                  fmt_dest(d, kind, major))
        u, ui = ins.dcl_usage
        if kind == 'ps' and major < 3:
            return 'dcl %s' % fmt_dest(d, kind, major)     # ps_2_0: no semantic on t#/v#
        tag = '%s%s' % (USAGE.get(u, 'usage%d' % u), ui if ui else '')
        if d.centroid:
            tag += '_centroid'
        return 'dcl_%s %s' % (tag, fmt_dest(d, kind, major))
    if op == OP_DEF:
        return 'def %s, %s' % (fmt_dest(ins.dst, kind, major),
                               ', '.join('%g' % x for x in ins.values))
    if op == OP_DEFI:
        return 'defi %s, %s' % (fmt_dest(ins.dst, kind, major),
                                ', '.join('%d' % x for x in ins.values))
    if op == OP_DEFB:
        return 'defb %s, %s' % (fmt_dest(ins.dst, kind, major),
                                'true' if ins.values[0] else 'false')
    name = ins.mnemonic()
    pred = ''
    if ins.pred is not None:
        pred = '(%s) ' % fmt_src(ins.pred, kind, major)
    if op == OP_TEXKILL:
        return '%s%s %s' % (pred, name, ins.dst.name(kind, major))
    parts = []
    if ins.dst is not None:
        name += dest_suffix(ins.dst)
        parts.append(fmt_dest(ins.dst, kind, major))
    parts += [fmt_src(s, kind, major) for s in ins.srcs]
    if not parts:
        return pred + name
    return '%s%s %s' % (pred, name, ', '.join(parts))


def render(sh):
    out = [sh.profile]
    for ins in sh.insts:
        out.append(render_inst(ins, sh))
    out.append('end')
    return '\n'.join('    ' + l for l in out)


if __name__ == '__main__':
    import sys, os
    for p in sorted(sys.argv[1:]):
        print('##### %s' % os.path.basename(p))
        print(render(decode(open(p, 'rb').read())))
        print()
