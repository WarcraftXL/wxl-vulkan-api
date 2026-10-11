"""DX9 bytecode -> GLSL 4.50 -> SPIR-V, offline, for wxl-vulkan-api.

The translator wxl-vulkan-api's draw path loads its modules from: every record the engine loads under
the vs_3_0 / ps_3_0 caps is decoded with dx9ir.py, rewritten as one GLSL module whose interface is
fixed by semantic (not by register), compiled with Khronos glslang (-V --target-env vulkan1.0
--spirv-val) and shipped under shaders/translated/<kind>/<md5>.spv next to its .glsl source and
its --spirv-dis text, plus a plain-text manifest the DLL loads. The md5 is of the record bytes
exactly as the engine hands them to IShaderCreateVertex/Pixel: the `size` bytes that follow the
16-byte record header of the .bls container, from the version token through the END token.

    python translate.py                   # stages 0 and 1 into shaders/translated/ (the default)
    python translate.py --stage 0         # one stage
    python translate.py --all             # attempt every record; report counts, write nothing
    python translate.py --all --out DIR   # ...and write everything that compiles into DIR
    python translate.py --names UI:0,Combiners_Mod:4 --print   # emit and print, do not write
    python translate.py --glslang PATH    # the glslang.exe to use (also $GLSLANG)

A record is refused (logged, skipped, listed in the manifest as a comment) when it uses something
the stage's vocabulary does not cover, or something no rule exists for; nothing is emitted for it.
With --all the only refusals are the latter kind, so later stages are a matter of adding rules.
The staged vocabularies and the per-record checks are in STAGES below; the emission rules are
one function each in Emitter.

The vs_2_0 weather records (patter, rain, sand, snowpoint; stage 4 here, the design's stage 6)
write the fixed-function outputs oFog / oPts and are drawn with no pixel shader. Their modules
carry the epilogues (fog0, gl_PointSize) and the manifest flags fixed_fog / point_size, and for
each of them the translator also generates the fixed-function pixel stage the D3D backend runs
after them -- a `companion` pixel record, profile `ff`, with the fog blend and the point-sprite
texcoord behind the flags.

Every vconst=256 vertex module also gets a bone-palette variant beside it, <md5>.bones.spv (the
manifest's `bones` line): the same math with every constant past c63 read from a storage buffer that
mirrors the registers from c31 up (set 2, binding 0), on the 64-row uniform block, so the DLL uploads
a palette once per change instead of a 256-row slice per draw. See BONE_BASE.
"""
import argparse, collections, hashlib, importlib.util, math, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


census = load('census')
ir = census.ir
ea = census.ea
BS = chr(92)

EXT = os.path.normpath(os.path.join(HERE, '..', '..'))   # the wxl-vulkan-api repository
DEFAULT_OUT = os.path.join(EXT, 'shaders', 'translated')
DEFAULT_GLSLANG = None   # --glslang, $GLSLANG or PATH (translate_shaders.py resolves one and passes it)

# ---- the interface contract -------------------------------------------------------------------
# Vertex inputs: fixed attribute locations by semantic. The pipeline builder maps the engine's
# vertex format to these; an input semantic outside this table is refused.
VIN_LOCATION = {'position0': 0, 'normal0': 1, 'color0': 2, 'color1': 3, 'texcoord0': 4,
                'texcoord1': 5, 'blendweight0': 6, 'blendindices0': 7}
# Varyings: locations by semantic, vec4 everywhere except fog (float); position and psize are the
# built-ins. Any vertex module pairs with any pixel module through this table.
VARYING_LOCATION = {'color0': 0, 'color1': 1, 'fog0': 10}
for _i in range(8):
    VARYING_LOCATION['texcoord%d' % _i] = 2 + _i
VARYING_TYPE = {'fog0': 'float'}
PCONST = 16
VCONST_SMALL, VCONST_LARGE = 64, 256

# The bone-palette variant of a vconst=256 vertex record (manifest `bones` line): the same module
# with every constant read past the 64-row uniform window -- the palette rows a0 indexes from c31,
# the shadow matrices at c224.. -- taken from a storage buffer that mirrors the registers from
# BONE_BASE up, set 2 binding 0, so the draw's uniform slice is the 64-row class and the palette is
# uploaded once per change instead of once per draw. Rows below 64 stay in the uniform slice, which
# holds the same values. A record whose relative reads are not all `c<k>[a0.?]` with k >= BONE_BASE
# (the liquids' aL loops, an a0 read below the palette) gets no variant and keeps the 256-row path.
BONE_BASE = 31
BONE_ROWS = VCONST_LARGE - BONE_BASE

# vs_2_0 fixed-function output registers -> semantic
VS2_OUT = {'oPos': 'position0', 'oD0': 'color0', 'oD1': 'color1', 'oFog': 'fog0', 'oPts': 'psize0'}
for _i in range(8):
    VS2_OUT['oT%d' % _i] = 'texcoord%d' % _i

M2_VERTEX = ['CDiffuse_T1', 'Color_Proj_Proj', 'Color_T1', 'Diffuse_Env', 'Diffuse_Env_Env',
             'Diffuse_Env_T2', 'Diffuse_Proj_Proj', 'Diffuse_T1', 'Diffuse_T1_Env', 'Diffuse_T1_T2',
             'Diffuse_T2']

# ---- the staged plan ---------------------------------------------------------------------------
# Each stage: which (kind, name, perm) it selects, the opcode vocabulary it admits, and the extra
# per-record facts it insists on. A selected record that steps outside is refused, not emitted.


def _stage0_select(kind, name, perm):
    return (kind, name) in {('vertex', 'UI'), ('pixel', 'UI'), ('vertex', 'WaterRipples'),
                            ('pixel', 'WaterRipples'), ('pixel', 'Desaturate')}


def _stage1_select(kind, name, perm):
    if kind == 'vertex':
        return name in M2_VERTEX and 0 <= perm <= 9
    return name.startswith('Combiners_') and perm in (0, 4, 8, 12)


def _stage2_select(kind, name, perm):
    # Bones: the M2 permutations 10-29 read the palette through a0 (relative addressing, class
    # 256). The login screen's own models are exactly these (Diffuse_T1:10/11/13/21/23, ...):
    # every M2 the engine draws is skinned, so stage 1's static permutations never appear on it.
    return kind == 'vertex' and name in M2_VERTEX and 10 <= perm <= 29


TERRAIN_PIXEL = {'Terrain0', 'Terrain0_env', 'Terrain1', 'Terrain2', 'Terrain2_pcf'}


def _stage3_select(kind, name, perm):
    # WMO and terrain without shadows: the MapObj* vertex permutations 0-29 (no relative
    # addressing) with their pixel permutations 0/4/8/12, every Terrain vertex permutation, and the
    # shadow-free terrain pixel shaders. New vocabulary: abs, cmp, texldl (Terrain2*), cube s0.
    if kind == 'vertex':
        return (name.startswith('MapObj') and 0 <= perm <= 29) or name == 'Terrain'
    return (name.startswith('MapObj') and perm in (0, 4, 8, 12)) or name in TERRAIN_PIXEL


def _stage4_select(kind, name, perm):
    # Everything the earlier stages did not take: the shadow permutations (if/else, texldl taps,
    # c224+), the liquids (loop/aL/sincos), ShadowMap, DetailDoodad, the SM2 weather and FFX
    # families. One catch-all stage rather than three because `--all` already proved every record
    # compiles and validates; what remains to prove is on screen, per family, and the DLL names
    # what it refuses (cube samplers, custom vertex formats) once per pair.
    return True


ALL_VOCAB = {'add', 'mad', 'mov', 'mul', 'dp3', 'dp4', 'max', 'min', 'rsq', 'rcp', 'pow', 'nrm', 'lrp',
             'abs', 'frc', 'slt', 'sge', 'sincos', 'mova', 'cmp', 'dp2add', 'texld', 'texldl',
             'texkill', 'if_lt', 'if_ne', 'else', 'endif', 'loop', 'endloop',
             'if', 'ifc', 'break', 'rep', 'endrep'}


# The vs_2_0 fixed-function outputs (design: stage 6). `fixed_fog`: the record writes oFog, the
# fog *factor* the fixed-function fog blend consumes after the pixel stage. `point_size`: the
# record writes oPts and is drawn as point sprites. Both are emitted as the module's own epilogue
# (fog0 at location 10, gl_PointSize) and, because the engine draws these records with no pixel
# shader at all (the Weather family has no pixel container in any profile), the translator also
# generates the fixed-function pixel stage they pair with: see companion_module().
FF_FLAGS = {'fixed_fog', 'point_size'}

STAGES = {
    0: dict(select=_stage0_select,
            vocab={'mov', 'mul', 'add', 'mad', 'dp4', 'dp3', 'texld'},
            vconst=VCONST_SMALL, relative=False, samplers={'2d'}, flags=set()),
    1: dict(select=_stage1_select,
            vocab={'add', 'dp3', 'dp4', 'mad', 'max', 'min', 'mov', 'mul', 'nrm', 'pow', 'rcp', 'rsq',
                   'dp2add', 'lrp', 'texkill', 'texld'},
            vconst=VCONST_SMALL, relative=False, samplers={'2d'}, flags=set()),
    2: dict(select=_stage2_select,
            vocab={'add', 'dp3', 'dp4', 'mad', 'max', 'min', 'mov', 'mul', 'nrm', 'pow', 'rcp', 'rsq',
                   'mova'},
            vconst=VCONST_LARGE, relative=True, samplers={'2d'}, flags=set()),
    3: dict(select=_stage3_select,
            vocab={'add', 'dp3', 'dp4', 'mad', 'max', 'min', 'mov', 'mul', 'nrm', 'pow', 'rcp', 'rsq',
                   'dp2add', 'lrp', 'texkill', 'texld', 'abs', 'cmp', 'texldl'},
            vconst=VCONST_SMALL, relative=False, samplers={'2d', 'cube'}, flags=set()),
    4: dict(select=_stage4_select, vocab=ALL_VOCAB,
            vconst=VCONST_LARGE, relative=True, samplers={'2d', 'cube'}, flags=FF_FLAGS),
}


class Refused(Exception):
    pass


# ---- records -----------------------------------------------------------------------------------

class Record(object):
    __slots__ = ('kind', 'profile', 'name', 'perm', 'code', 'md5', 'sh', 'names', 'hdr')

    @property
    def key(self):
        return '%s:%d' % (self.name, self.perm)


def enumerate_records():
    """Every record the engine loads, resolved through its fallback chain exactly as census.py does.

    Returns the records in container order; byte-identical records are separate entries with the
    same md5 (the caller folds them)."""
    inv = census.inventory()
    resolved = census.resolve(inv)
    arch = ea.archives()
    out = []
    for kind, prof, name, depth in resolved:
        chain = census.VS_FALLBACK if kind == 'vertex' else census.PS_FALLBACK
        data = None
        for prof2 in chain[chain.index(prof):]:
            low = name.lower()
            if low not in inv.get((kind, prof2), {}):
                continue
            path = BS.join(['Shaders', kind.capitalize(), prof2, inv[(kind, prof2)][low] + '.bls'])
            src, data = ea.grab(arch, path)
            if data is None or len(data) < 12:
                data = None
                continue
            prof = prof2
            break
        if data is None:
            continue
        magic, ver, nperm, records, consumed = ea.split(data)
        for rec in records:
            r = Record()
            r.kind, r.profile, r.name, r.perm = kind, prof, name, rec['i']
            r.code = rec['code']
            r.md5 = hashlib.md5(r.code).hexdigest()
            r.hdr = (rec['a'], rec['b'], rec['c'], rec['d'])
            try:
                r.sh = ir.decode(r.code)
            except ValueError:
                r.sh = None       # SM1: listed by the census, not translated
            out.append(r)
    return out


# ---- the emitter -------------------------------------------------------------------------------

SWZ = ir.SWZ
RT = ir


def fmt_float(x):
    if math.isnan(x) or math.isinf(x):
        raise Refused('non-finite def constant %r' % x)
    s = '%.9g' % x
    if 'e' in s or 'E' in s:
        m, e = s.split('e')
        if '.' not in m:
            m += '.0'
        return m + 'e' + e
    if '.' not in s:
        s += '.0'
    return s


def mask_letters(mask):
    return ''.join(SWZ[k] for k in range(4) if mask & (1 << k))


class Emitter(object):
    """One record -> one GLSL module plus the metadata the manifest carries.

    The body is one pass over the Inst list. Registers are locals (`vec4 r0`, `vec4 o1`), the
    interface is by semantic (`color0`, `texcoord0`, `fog0`), and a prologue / epilogue connects
    the two, which is where the y flip, the zero-fill of unwritten varying components, the vs_2_0
    fixed outputs and the alpha-test discard live."""

    def __init__(self, rec, stage=None, bones=False):
        self.rec = rec
        self.bones = bones                         # emit the bone-palette variant (BONE_BASE)
        self.sh = rec.sh
        self.kind = rec.kind                       # 'vertex' | 'pixel'
        self.vs = self.sh.kind == 'vs'
        self.major = self.sh.major
        self.stage = stage
        self.defs = {}                             # c index -> (x, y, z, w)
        self.defi = {}                             # i index -> ints
        self.inputs = collections.OrderedDict()    # register name -> semantic (vertex) / pixel reads
        self.in_masks = {}
        self.out_sems = collections.OrderedDict()  # output register name -> [semantic]
        self.samplers = {}                         # slot -> '2d' | 'cube'
        self.temps = set()
        self.outregs = collections.OrderedDict()
        self.uses_a0 = False
        self.uses_aL = False
        self.relative = False
        self.cmax = -1
        self.flags = set()
        self.helpers = set()
        self.body = []
        self.depth = 1
        self.loopn = 0
        self.ops_used = collections.Counter()

    # -- operands ---------------------------------------------------------------------------------

    def regname(self, o):
        return o.name(self.sh.kind, self.major)

    def const_expr(self, o):
        """A `c#` read: a def'd literal or the uniform window, direct or relative."""
        if o.rel:
            rt, ri, comp = o.rel
            if not self.vs:
                raise Refused('relative addressing in a pixel shader')
            self.relative = True
            if rt == ir.RT_LOOP:
                if self.bones:
                    raise Refused('bones variant: a loop-relative constant read')
                self.uses_aL = True
                idx = '%d + aL' % o.index
            elif rt == ir.RT_ADDR:
                self.uses_a0 = True
                if self.bones:
                    if o.index < BONE_BASE:
                        raise Refused('bones variant: an a0-relative read below c%d' % BONE_BASE)
                    return 'bones.b[%d + a0.%s]' % (o.index - BONE_BASE, SWZ[comp])
                idx = '%d + a0.%s' % (o.index, SWZ[comp])
            else:
                raise Refused('relative addressing through register type %d' % rt)
            return 'regs.v[%s]' % idx
        if o.index in self.defs:
            return 'c%d' % o.index
        self.cmax = max(self.cmax, o.index)
        if self.vs:
            if o.index >= VCONST_LARGE:
                raise Refused('vertex constant c%d beyond the 256-row window' % o.index)
            if self.bones and o.index >= VCONST_SMALL:
                return 'bones.b[%d]' % (o.index - BONE_BASE)
            return 'regs.v[%d]' % o.index
        if o.index >= PCONST:
            raise Refused('pixel constant c%d beyond the 16-row window' % o.index)
        return 'regs.p[%d]' % o.index

    def base_expr(self, o):
        rt = o.rtype
        if rt == ir.RT_TEMP:
            self.temps.add(o.index)
            return 'r%d' % o.index
        if rt == ir.RT_CONST:
            return self.const_expr(o)
        if rt == ir.RT_INPUT:
            nm = self.regname(o)
            if nm not in self.inputs:
                raise Refused('%s read without a dcl' % nm)
            return nm
        if rt == ir.RT_TEXTURE and not self.vs:          # ps_2_0 t#
            nm = self.regname(o)
            if nm not in self.inputs:
                raise Refused('%s read without a dcl' % nm)
            return nm
        if rt == ir.RT_CONSTINT:
            if o.index not in self.defi:
                raise Refused('i%d read but not defi\'d (the engine uploads no integer constants)' % o.index)
            return 'i%d' % o.index
        if rt == ir.RT_CONSTBOOL:
            raise Refused('bool constant b%d' % o.index)
        if rt == ir.RT_MISCTYPE:
            raise Refused('%s is not supported' % self.regname(o))
        if rt == ir.RT_PREDICATE:
            raise Refused('predicate register')
        if rt in (ir.RT_ADDR, ir.RT_LOOP) and self.vs:
            raise Refused('%s read as a plain source' % self.regname(o))
        if rt == ir.RT_SAMPLER:
            raise Refused('sampler used as an arithmetic source')
        if rt in (ir.RT_RASTOUT, ir.RT_ATTROUT, ir.RT_TEXCRDOUT, ir.RT_COLOROUT):
            nm = self.regname(o)
            if nm == 'oC0':
                return 'oC0'
            self.outregs[nm] = True
            return nm
        raise Refused('register type %d (%s)' % (rt, self.regname(o)))

    def src_sw(self, o, sel):
        """Source narrowed to the components `sel` (indices into the swizzled vec4): base, the
        composed swizzle, then the modifier outside. sel=[0,1,2,3] is the whole source."""
        if o.rel and o.rtype != ir.RT_CONST:
            raise Refused('relative addressing on a non-constant register')
        base = self.base_expr(o)
        c = o.comps()
        letters = ''.join(SWZ[c[k]] for k in sel)
        e = base if letters == 'xyzw' else '%s.%s' % (base, letters)
        mod = o.mod
        if mod == 0:
            return e
        if mod == 1:
            return '-' + e
        if mod == 11:
            return 'abs(%s)' % e
        if mod == 12:
            return '-abs(%s)' % e
        if mod == 6:
            return '(1.0 - %s)' % e
        raise Refused('source modifier %s' % ir.SRCMOD.get(mod, ('', '', '?%d' % mod))[2])

    def src(self, o):
        """The whole vec4 source."""
        return self.src_sw(o, [0, 1, 2, 3])

    def comp(self, o, k):
        """Scalar: component k of the swizzled source."""
        return self.src_sw(o, [k])

    def srcm(self, o, d):
        """The source narrowed to the destination's write mask (componentwise ops)."""
        return self.src_sw(o, [k for k in range(4) if d.mask & (1 << k)])

    def sw(self, expr, letters):
        """`expr.letters`, parenthesised when expr is not a plain name."""
        plain = all(ch.isalnum() or ch in '_.[]' for ch in expr)
        call = expr[:1].isalpha() and expr.endswith(')') and expr.count('(') == expr.count(')')
        if (plain or call) and not expr.startswith('-'):
            return '%s.%s' % (expr, letters)
        return '(%s).%s' % (expr, letters)

    def dest_name(self, d):
        rt = d.rtype
        if d.rel:
            raise Refused('relative addressing on a destination')
        if rt == ir.RT_TEMP:
            self.temps.add(d.index)
            return 'r%d' % d.index
        if rt == ir.RT_ADDR and self.vs:
            self.uses_a0 = True
            return 'a0'
        if rt == ir.RT_COLOROUT:
            if d.index != 0:
                raise Refused('oC%d: multiple render targets' % d.index)
            return 'oC0'
        if rt == ir.RT_DEPTHOUT:
            raise Refused('oDepth')
        if rt in (ir.RT_RASTOUT, ir.RT_ATTROUT, ir.RT_TEXCRDOUT) and self.vs:
            nm = self.regname(d)
            self.outregs[nm] = True
            return nm
        raise Refused('destination register type %d (%s)' % (rt, self.regname(d)))

    def assign(self, d, vec=None, scalar=None, full=None):
        """D.M = expr, with _sat applied before the mask. `vec` is already the mask's width
        (sources narrowed with srcm), `scalar` is replicated into it, `full` is a vec4 that gets
        `.M` applied."""
        if d.shift:
            raise Refused('destination shift modifier')
        name = self.dest_name(d)
        if name == 'a0':
            raise Refused('float assignment to a0')
        m = mask_letters(d.mask)
        n = len(m)
        if scalar is not None:
            if d.sat:
                scalar = 'clamp(%s, 0.0, 1.0)' % scalar
            rhs = scalar if n == 1 else 'vec%d(%s)' % (n, scalar)
        elif full is not None:
            rhs = full
            if d.sat:
                rhs = 'clamp(%s, 0.0, 1.0)' % rhs
            if n != 4:
                rhs = self.sw(rhs, m)
        else:
            rhs = vec
            if d.sat:
                rhs = 'clamp(%s, 0.0, 1.0)' % rhs
        if n == 4:
            self.line('%s = %s;' % (name, rhs))
        else:
            self.line('%s.%s = %s;' % (name, m, rhs))

    def line(self, s):
        self.body.append('    ' * self.depth + s)

    # -- declarations -----------------------------------------------------------------------------

    def dcl(self, ins):
        d = ins.dst
        if d.rtype == ir.RT_SAMPLER:
            t = ir.TEXTYPE.get(ins.dcl_textype, 'tt%d' % ins.dcl_textype)
            if t not in ('2d', 'cube'):
                raise Refused('sampler type %s' % t)
            if self.vs:
                raise Refused('vertex texture fetch (dcl_%s s%d)' % (t, d.index))
            if d.index > 15:
                raise Refused('sampler slot s%d' % d.index)
            self.samplers[d.index] = t
            return
        if d.centroid:
            raise Refused('_centroid declaration')
        u, ui = ins.dcl_usage
        sem = '%s%d' % (ir.USAGE.get(u, 'usage%d' % u), ui)
        nm = self.regname(d)
        if self.vs and d.rtype == ir.RT_OUTPUT and self.major >= 3:
            if sem != 'position0' and sem != 'psize0' and sem not in VARYING_LOCATION:
                raise Refused('output semantic %s' % sem)
            self.out_sems.setdefault(nm, []).append(sem)
            self.outregs[nm] = True
            return
        if d.rtype == ir.RT_INPUT and self.vs:
            if sem not in VIN_LOCATION:
                raise Refused('vertex input semantic %s' % sem)
            self.inputs[nm] = sem
            self.in_masks[nm] = d.mask
            return
        if not self.vs:
            if self.major < 3:
                # ps_2_0: no semantic token; t# is texcoord#, v# is color# (and is clamped)
                sem = ('texcoord%d' if d.rtype == ir.RT_TEXTURE else 'color%d') % d.index
            if sem not in VARYING_LOCATION:
                raise Refused('pixel input semantic %s' % sem)
            self.inputs[nm] = sem
            self.in_masks[nm] = d.mask
            return
        raise Refused('dcl of %s' % nm)

    # -- instructions -----------------------------------------------------------------------------

    def emit(self, ins):
        op = ins.op
        if ins.predicated or ins.pred is not None:
            raise Refused('predicated instruction')
        if op == ir.OP_DCL:
            return self.dcl(ins)
        if op == ir.OP_DEF:
            self.defs[ins.dst.index] = ins.values
            return
        if op == ir.OP_DEFI:
            self.defi[ins.dst.index] = ins.values
            return
        if op == ir.OP_DEFB:
            raise Refused('defb')
        mn = ins.mnemonic()
        self.ops_used[mn] += 1
        if self.stage is not None and mn not in self.stage['vocab']:
            raise Refused('opcode %s is outside the stage vocabulary' % mn)
        rule = RULES.get(op)
        if rule is None:
            raise Refused('no rule for opcode %s' % mn)
        rule(self, ins)

    # one rule per opcode; A, B, C are the sources after swizzle and modifier, D the destination
    def r_mov(self, i):
        self.assign(i.dst, vec=self.srcm(i.srcs[0], i.dst))

    def r_add(self, i):
        d = i.dst
        self.assign(d, vec='%s + %s' % (self.srcm(i.srcs[0], d), self.srcm(i.srcs[1], d)))

    def r_sub(self, i):
        d = i.dst
        self.assign(d, vec='%s - %s' % (self.srcm(i.srcs[0], d), self.srcm(i.srcs[1], d)))

    def r_mul(self, i):
        d = i.dst
        self.assign(d, vec='%s * %s' % (self.srcm(i.srcs[0], d), self.srcm(i.srcs[1], d)))

    def r_mad(self, i):
        d = i.dst
        self.assign(d, vec='%s * %s + %s' % tuple(self.srcm(s, d) for s in i.srcs[:3]))

    def r_dp3(self, i):
        a, b = self.src_sw(i.srcs[0], [0, 1, 2]), self.src_sw(i.srcs[1], [0, 1, 2])
        self.assign(i.dst, scalar='dot(%s, %s)' % (a, b))

    def r_dp4(self, i):
        self.assign(i.dst, scalar='dot(%s, %s)' % (self.src(i.srcs[0]), self.src(i.srcs[1])))

    def r_dp2add(self, i):
        a, b, c = i.srcs[:3]
        self.assign(i.dst, scalar='%s * %s + %s * %s + %s'
                    % (self.comp(a, 0), self.comp(b, 0), self.comp(a, 1), self.comp(b, 1), self.comp(c, 0)))

    def r_min(self, i):
        d = i.dst
        self.assign(d, vec='min(%s, %s)' % (self.srcm(i.srcs[0], d), self.srcm(i.srcs[1], d)))

    def r_max(self, i):
        d = i.dst
        self.assign(d, vec='max(%s, %s)' % (self.srcm(i.srcs[0], d), self.srcm(i.srcs[1], d)))

    def r_lrp(self, i):
        # lrp dst, s0, s1, s2 = s0 * s1 + (1 - s0) * s2 = mix(s2, s1, s0)
        d = i.dst
        a, b, c = (self.srcm(s, d) for s in i.srcs[:3])
        self.assign(d, vec='mix(%s, %s, %s)' % (c, b, a))

    def r_rcp(self, i):
        self.assign(i.dst, scalar='1.0 / %s' % self.comp(i.srcs[0], 0))

    def r_rsq(self, i):
        self.helpers.add('rsq')
        self.assign(i.dst, scalar='dx_rsq(%s)' % self.comp(i.srcs[0], 0))

    def r_pow(self, i):
        self.helpers.add('pow')
        self.assign(i.dst, scalar='dx_pow(%s, %s)' % (self.comp(i.srcs[0], 0), self.comp(i.srcs[1], 0)))

    def r_exp(self, i):
        self.assign(i.dst, scalar='exp2(%s)' % self.comp(i.srcs[0], 0))

    def r_log(self, i):
        self.assign(i.dst, scalar='log2(abs(%s))' % self.comp(i.srcs[0], 0))

    def r_nrm(self, i):
        self.helpers.add('nrm')
        self.assign(i.dst, full='dx_nrm(%s)' % self.src(i.srcs[0]))

    def r_abs(self, i):
        self.assign(i.dst, vec='abs(%s)' % self.srcm(i.srcs[0], i.dst))

    def r_frc(self, i):
        self.assign(i.dst, vec='fract(%s)' % self.srcm(i.srcs[0], i.dst))

    def r_compare(self, i, fn, op):
        d = i.dst
        a, b = self.srcm(i.srcs[0], d), self.srcm(i.srcs[1], d)
        n = bin(d.mask).count('1')
        if n == 1:
            self.assign(d, vec='float(%s %s %s)' % (a, op, b))
        else:
            self.assign(d, vec='vec%d(%s(%s, %s))' % (n, fn, a, b))

    def r_slt(self, i):
        self.r_compare(i, 'lessThan', '<')

    def r_sge(self, i):
        self.r_compare(i, 'greaterThanEqual', '>=')

    def r_sincos(self, i):
        # vs_3_0: one source; vs_2_0: two extra Taylor-coefficient constants the macro used
        s = self.comp(i.srcs[0], 0)
        m = i.dst.mask & 0xF
        if m & 0xC:
            raise Refused('sincos writing .z or .w')
        if m == 1:
            self.assign(i.dst, vec='cos(%s)' % s)
        elif m == 2:
            self.assign(i.dst, vec='sin(%s)' % s)
        else:
            self.assign(i.dst, vec='vec2(cos(%s), sin(%s))' % (s, s))

    def r_cmp(self, i):
        # per component: a >= 0 ? b : c
        d = i.dst
        a, b, c = (self.srcm(s, d) for s in i.srcs[:3])
        n = bin(d.mask).count('1')
        if n == 1:
            self.assign(d, vec='(%s >= 0.0 ? %s : %s)' % (a, b, c))
        else:
            self.assign(d, vec='mix(%s, %s, greaterThanEqual(%s, vec%d(0.0)))' % (c, b, a, n))

    def r_mova(self, i):
        d = i.dst
        if d.rtype != ir.RT_ADDR:
            raise Refused('mova into a non-address register')
        # round to nearest, into the integer address register
        self.uses_a0 = True
        m = mask_letters(d.mask)
        n = len(m)
        src = self.srcm(i.srcs[0], d)
        rhs = 'int(round(%s))' % src if n == 1 else 'ivec%d(round(%s))' % (n, src)
        if n == 4:
            self.line('a0 = %s;' % rhs)
        else:
            self.line('a0.%s = %s;' % (m, rhs))

    def r_texld(self, i):
        if i.ctrl & 3:
            raise Refused('%s' % i.mnemonic())
        coord, samp = i.srcs[0], i.srcs[1]
        if samp.rtype != ir.RT_SAMPLER:
            raise Refused('texld without a sampler operand')
        t = self.samplers.get(samp.index)
        if t is None:
            raise Refused('s%d sampled without a dcl' % samp.index)
        uv = self.src_sw(coord, [0, 1] if t == '2d' else [0, 1, 2])
        self.assign(i.dst, full='texture(s%d, %s)' % (samp.index, uv))

    def r_texldl(self, i):
        coord, samp = i.srcs[0], i.srcs[1]
        if samp.rtype != ir.RT_SAMPLER:
            raise Refused('texldl without a sampler operand')
        t = self.samplers.get(samp.index)
        if t is None:
            raise Refused('s%d sampled without a dcl' % samp.index)
        uv = self.src_sw(coord, [0, 1] if t == '2d' else [0, 1, 2])
        self.assign(i.dst, full='textureLod(s%d, %s, %s)' % (samp.index, uv, self.comp(coord, 3)))

    def r_texkill(self, i):
        d = i.dst
        if d.rtype == ir.RT_TEMP:
            self.temps.add(d.index)
            nm = 'r%d' % d.index
        elif d.rtype == ir.RT_TEXTURE and not self.vs and self.major < 3:
            nm = self.regname(d)
        else:
            raise Refused('texkill on %s' % self.regname(d))
        # SM2+: discard when any of the four components is < 0
        self.line('if (any(lessThan(%s, vec4(0.0)))) discard;' % nm)

    def r_ifc(self, i):
        cmpop = {1: '>', 2: '==', 3: '>=', 4: '<', 5: '!=', 6: '<='}.get(i.ctrl & 7)
        if cmpop is None:
            raise Refused('if with comparison %d' % (i.ctrl & 7))
        self.line('if (%s %s %s) {' % (self.comp(i.srcs[0], 0), cmpop, self.comp(i.srcs[1], 0)))
        self.depth += 1

    def r_if_bool(self, i):
        raise Refused('if on a bool register')

    def r_else(self, i):
        self.depth -= 1
        self.line('} else {')
        self.depth += 1

    def r_endif(self, i):
        self.depth -= 1
        self.line('}')

    def r_loop(self, i):
        # loop aL, i#: i#.x iterations, aL from i#.y, step i#.z
        if len(i.srcs) != 2 or i.srcs[0].rtype != ir.RT_LOOP or i.srcs[1].rtype != ir.RT_CONSTINT:
            raise Refused('loop operands')
        iv = self.base_expr(i.srcs[1])
        self.uses_aL = True
        self.loopn += 1
        n = 'n%d' % self.loopn
        self.line('for (int aL = %s.y, %s = 0; %s < %s.x; ++%s, aL += %s.z) {' % (iv, n, n, iv, n, iv))
        self.depth += 1

    def r_endloop(self, i):
        self.depth -= 1
        self.line('}')

    def r_nop(self, i):
        pass

    # -- the module ------------------------------------------------------------------------------

    def run(self):
        sh = self.sh
        if sh.major < 2:
            raise Refused('SM1 stream')
        for ins in sh.insts:
            self.emit(ins)
        if self.depth != 1:
            raise Refused('unbalanced flow control')
        # The varyings this module writes / reads, by semantic
        if self.vs:
            if self.major >= 3:
                for nm in self.outregs:
                    if nm not in self.out_sems:
                        raise Refused('%s written without a dcl' % nm)
            else:
                for nm in self.outregs:
                    sem = VS2_OUT.get(nm)
                    if sem is None:
                        raise Refused('vs_2_0 output %s' % nm)
                    self.out_sems.setdefault(nm, []).append(sem)
                    if sem == 'fog0':
                        self.flags.add('fixed_fog')
                    if sem == 'psize0':
                        self.flags.add('point_size')
            self.outputs = []
            for nm, sems in self.out_sems.items():
                for s in sems:
                    if s == 'psize0':
                        self.flags.add('point_size')
                    if s not in self.outputs:
                        self.outputs.append(s)
            if 'position0' not in self.outputs:
                raise Refused('no position output')
            self.vconst = VCONST_LARGE if (self.relative or self.cmax >= VCONST_SMALL) else VCONST_SMALL
            if self.bones:
                if self.vconst != VCONST_LARGE:
                    raise Refused('bones variant of a 64-row module')
                self.vconst = VCONST_SMALL       # everything past c63 reads the storage buffer
        else:
            self.reads = []
            for nm, s in self.inputs.items():
                if s not in self.reads:
                    self.reads.append(s)
            self.vconst = VCONST_SMALL
        # Stage checks beyond the vocabulary
        st = self.stage
        if st is not None:
            if not st['relative'] and (self.relative or self.uses_a0 or self.uses_aL):
                raise Refused('relative addressing is outside the stage')
            if self.vs and self.vconst > st['vconst']:
                raise Refused('constant window %d is outside the stage (c%d)' % (self.vconst, self.cmax))
            bad = sorted(set(self.samplers.values()) - st['samplers'])
            if bad:
                raise Refused('sampler type %s is outside the stage' % ','.join(bad))
            bad = sorted(self.flags - st['flags'])
            if bad:
                raise Refused('%s is outside the stage' % ','.join(bad))
        return self.text()

    def text(self):
        rec = self.rec
        L = []
        P = L.append
        P('#version 450')
        P('// %s (%s), %d bytes, md5 %s' % (' = '.join(rec.names), self.sh.profile, len(rec.code), rec.md5))
        P('// Generated by tools/bls_tools/translate.py from the client\'s own bytecode; do not edit.')
        P('// The record, as fxc disassembles it:')
        for ln in ir.render(self.sh).split('\n')[1:-1]:
            P('//  ' + ln)
        P('')
        if self.vs:
            P('// Vertex inputs, at the fixed location of their semantic (the pipeline maps the engine\'s')
            P('// vertex format onto these). A float3 element reads .w as 1.0, a float2 reads .zw as (0, 1).')
            for nm, sem in self.inputs.items():
                P('layout(location = %d) in vec4 %s;   // %s' % (VIN_LOCATION[sem], nm, sem))
            P('')
            P('// Varyings, by semantic; every component is written (0 where the bytecode writes none).')
            for sem in self.outputs:
                if sem in ('position0', 'psize0'):
                    continue
                t = VARYING_TYPE.get(sem, 'vec4')
                P('layout(location = %d) out %s %s;' % (VARYING_LOCATION[sem], t, sem))
        else:
            P('// Varyings read, by semantic, at the same locations the vertex module writes them.')
            for sem in self.reads:
                t = VARYING_TYPE.get(sem, 'vec4')
                P('layout(location = %d) in %s %s;' % (VARYING_LOCATION[sem], t, sem))
            P('')
            P('layout(location = 0) out vec4 oC0;')
            if self.samplers:
                P('')
                P('// Combined image samplers, one binding per D3D sampler slot (set 0).')
                for slot in sorted(self.samplers):
                    t = 'sampler2D' if self.samplers[slot] == '2d' else 'samplerCube'
                    P('layout(set = 0, binding = %d) uniform %s s%d;' % (slot, t, slot))
        P('')
        P('// One slice of the uniform ring. p[] and misc come FIRST so that a pixel module, which never')
        P('// reads v[], can declare v[64] whatever class the paired vertex module is: p[i] is at byte')
        P('// 16*i, misc at 256, v[i] at 272 + 16*i, in both classes. VCONST is 64 or 256 per the vertex')
        P('// record\'s highest constant (manifest `vconst=`); the slice is 272 + 16*VCONST bytes.')
        P('layout(std140, set = 1, binding = 0) uniform Regs {')
        P('    vec4 p[%d];     // pixel c0..c%d' % (PCONST, PCONST - 1))
        P('    vec4 misc;       // misc.x = alphaRef / 255 (fixed-function alpha test, 0 = off)')
        P('    vec4 v[%d];     // vertex c0..c%d' % (self.vconst, self.vconst - 1))
        P('} regs;')
        if self.bones:
            P('')
            P('// The bone-palette variant: every vertex constant past c%d -- the palette a0 indexes,'
              % (VCONST_SMALL - 1))
            P('// the shadow matrices -- from the device\'s mirror of the registers, b[i] = c(%d + i),'
              % BONE_BASE)
            P('// uploaded once per palette change and bound at a dynamic offset (manifest `bones`).')
            P('layout(std430, set = 2, binding = 0) readonly buffer Bones {')
            P('    vec4 b[%d];     // vertex c%d..c%d' % (BONE_ROWS, BONE_BASE, VCONST_LARGE - 1))
            P('} bones;')
        if self.defs or self.defi:
            P('')
            P('// Constants the bytecode def\'s: literals, never uploaded by the engine.')
            for idx in sorted(self.defs):
                P('const vec4 c%d = vec4(%s);' % (idx, ', '.join(fmt_float(x) for x in self.defs[idx])))
            for idx in sorted(self.defi):
                P('const ivec4 i%d = ivec4(%s);' % (idx, ', '.join('%d' % x for x in self.defi[idx])))
        if self.helpers:
            P('')
            P('// D3D9 semantics of the scalar ops (the design\'s trap list; DXVK does the same).')
            if 'rsq' in self.helpers:
                P('float dx_rsq(float x) { return inversesqrt(abs(x)); }')
            if 'pow' in self.helpers:
                P('float dx_pow(float x, float y) { return x == 0.0 ? 0.0 : exp2(y * log2(abs(x))); }')
            if 'nrm' in self.helpers:
                P('vec4 dx_nrm(vec4 v) { return v * inversesqrt(dot(v.xyz, v.xyz)); }')
        P('')
        P('void main()')
        P('{')
        # prologue
        for t in sorted(self.temps):
            P('    vec4 r%d = vec4(0.0);' % t)
        if self.uses_a0:
            P('    ivec4 a0 = ivec4(0);')
        if self.vs:
            for nm in self.outregs:
                P('    vec4 %s = vec4(0.0);' % nm)
        else:
            P('    oC0 = vec4(0.0);')
            for nm, sem in self.inputs.items():
                if sem == 'fog0':
                    rhs = 'vec4(fog0, 0.0, 0.0, 0.0)'
                elif self.major < 3 and nm.startswith('v'):
                    rhs = 'clamp(%s, 0.0, 1.0)' % sem       # ps_2_0 colour inputs are clamped by SM2
                else:
                    rhs = sem
                P('    vec4 %s = %s;' % (nm, rhs))
        P('')
        L.extend(self.body)
        P('')
        # epilogue
        if self.vs:
            for nm, sems in self.out_sems.items():
                for sem in sems:
                    if sem == 'position0':
                        P('    // D3D clip space has +y up, Vulkan +y down (no VK_KHR_maintenance1 on 1.0);')
                        P('    // z is already in [0, 1]: ProjNative on the D3D API id maps it there.')
                        P('    gl_Position = vec4(%s.x, -%s.y, %s.z, %s.w);' % (nm, nm, nm, nm))
                    elif sem == 'psize0':
                        P('    gl_PointSize = %s.x;' % nm)
                    elif sem == 'fog0':
                        P('    fog0 = %s.x;' % nm)
                    else:
                        P('    %s = %s;' % (sem, nm))
        else:
            P('    // Render state 7 (alpha test) is not in the bytecode: the same discard as')
            P('    // combiners_mod.frag, from misc.x = alphaRef / 255 (0 when the test is off).')
            P('    if (regs.misc.x > 0.0 && oC0.a < regs.misc.x) discard;')
        P('}')
        return '\n'.join(L) + '\n'


RULES = {
    0: Emitter.r_nop, 1: Emitter.r_mov, 2: Emitter.r_add, 3: Emitter.r_sub, 4: Emitter.r_mad,
    5: Emitter.r_mul, 6: Emitter.r_rcp, 7: Emitter.r_rsq, 8: Emitter.r_dp3, 9: Emitter.r_dp4,
    10: Emitter.r_min, 11: Emitter.r_max, 12: Emitter.r_slt, 13: Emitter.r_sge, 14: Emitter.r_exp,
    15: Emitter.r_log, 18: Emitter.r_lrp, 19: Emitter.r_frc, 27: Emitter.r_loop, 29: Emitter.r_endloop,
    32: Emitter.r_pow, 35: Emitter.r_abs, 36: Emitter.r_nrm, 37: Emitter.r_sincos,
    40: Emitter.r_if_bool, 41: Emitter.r_ifc, 42: Emitter.r_else, 43: Emitter.r_endif,
    46: Emitter.r_mova, 65: Emitter.r_texkill, 66: Emitter.r_texld, 88: Emitter.r_cmp,
    90: Emitter.r_dp2add, 95: Emitter.r_texldl,
}


# ---- the fixed-function pixel stage for the vs_2_0 records (stage 6) ----------------------------

# The pixel constant row the DLL fills with the fixed-function fog colour. No shipped pixel record
# reads c13..c15 (the engine uploads c0..c12 at most; c13/c14 are def'd literals in the shadow
# permutations), so the row is free in every pixel module and the DLL can write it on every draw.
FOG_ROW = 15


def companion_module(em, rec):
    """The pixel stage D3D runs after a vs_2_0 record that is drawn with no pixel shader.

    The four weather records (patter, rain, sand, snowpoint) exist in no pixel profile: the engine
    binds them with render state 78 null, and the D3D backend then runs the fixed-function
    cascade on the vertex shader's outputs (cgxdevice 6.1, 6.5, 6.8): CURRENT = oD0, stage 0
    combines the texel with it by ColorOp (state 37) / AlphaOp (state 45), the alpha test of
    state 7, then -- when fog is on -- the fog blend lerp(fogColor, rgb, oFog). ff.frag already
    does this for the fixed-function *vertex* path, but in its own varying layout (vColor at 0,
    vUV as a vec2 at 1, vFog at 2) and with the fog colour in v[10], a row these vertex records
    read as their own c10. So the stage is re-emitted here in the translated interface (color0 at
    0, texcoord0 as a vec4 at 2, fog0 at 10) and with the fog colour in regs.p[FOG_ROW], one
    module per distinct (texcoord source, fog) combination, named by the md5 of its own text.

    Returns (glsl text, meta) with meta = dict(samplers, reads, flags)."""
    point_sprite = 'point_size' in em.flags
    fixed_fog = 'fixed_fog' in em.flags
    has_uv = 'texcoord0' in em.outputs
    reads = ['color0']
    if has_uv and not point_sprite:
        reads.append('texcoord0')
    if fixed_fog:
        reads.append('fog0')
    flags = set()
    if fixed_fog:
        flags.add('fixed_fog')
    if point_sprite:
        flags.add('point_sprite')
    L = []
    P = L.append
    P('#version 450')
    P('// The fixed-function pixel stage for the vs_2_0 records drawn with no pixel shader')
    P('// (render state 78 null): ' + ', '.join(sorted(flags)) + '.')
    P('// Generated by tools/bls_tools/translate.py; do not edit. No bytecode stands behind this')
    P('// module: it is what the D3D backend does after such a vertex shader (cgxdevice 6.1, 6.5,')
    P('// 6.8), ff.frag re-expressed in the translated interface so it pairs with the translated')
    P('// vertex module instead of ff.vert. The DLL binds it when the draw\'s vertex row carries a')
    P('// `companion` line and the draw\'s pixel shader name is "".')
    P('//')
    P('//   CURRENT = color0 (the vertex shader\'s oD0, clamped as the fixed-function interpolator is);')
    if point_sprite:
        P('//   stage 0 samples s0 at gl_PointCoord: D3DRS_POINTSPRITEENABLE replaces every stage\'s')
        P('//   texcoord with the (0,0)..(1,1) sweep across the sprite, and the vertex shader writes')
        P('//   no texcoord at all; Vulkan\'s point coordinate has the same origin (upper left);')
    elif has_uv:
        P('//   stage 0 samples s0 at texcoord0.xy (the vertex shader\'s oT0);')
    else:
        P('//   stage 0 samples s0 at (0, 0): the vertex shader writes no texcoord;')
    P('//   the texel combines with CURRENT by the EGxTexOp in misc.z (colour) / misc.w (alpha),')
    P('//   the table ff.frag has (0 Mod, 1 Mod2x, 2 Add, 3 PassThru, 4 Decal, 5 Fade; 6 = stage 0')
    P('//   has no texture, the cascade ends), saturated as D3D saturates each stage;')
    P('//   render state 7: discard when misc.x > 0 and a < misc.x;')
    if fixed_fog:
        P('//   fog: oFog is the fog FACTOR with a programmable vertex shader (FOGVERTEXMODE is')
        P('//   ignored), blended as f * rgb + (1 - f) * fogColor only while D3DRS_FOGENABLE is on.')
        P('//   The colour and the enable come from regs.p[%d] = (fogColor.rgb / 255, fogOn ? 1 : 0),' % FOG_ROW)
        P('//   a row no shipped pixel record reads, written by the DLL after the pixel-cache copy.')
    else:
        P('//   fog: the vertex shader writes no oFog, so there is no factor to blend with; none.')
    P('')
    P('// Varyings read, by semantic, at the locations the translated vertex modules write them.')
    for sem in reads:
        t = VARYING_TYPE.get(sem, 'vec4')
        P('layout(location = %d) in %s %s;' % (VARYING_LOCATION[sem], t, sem))
    P('')
    P('layout(location = 0) out vec4 oC0;')
    P('')
    P('// Texture stage 0 (D3D sampler slot 0 = texture stage 0), set 0 binding 0.')
    P('layout(set = 0, binding = 0) uniform sampler2D s0;')
    P('')
    P('// The same slice as every translated module (p[] and misc first, v[] last).')
    P('layout(std140, set = 1, binding = 0) uniform Regs {')
    P('    vec4 p[%d];     // pixel c0..c%d; p[%d] = (fogColor.rgb, fogOn) for this module'
      % (PCONST, PCONST - 1, FOG_ROW))
    P('    vec4 misc;       // .x alphaRef / 255 (0 = test off), .z stage-0 colour op, .w stage-0 alpha op')
    P('    vec4 v[%d];     // vertex c0..c%d (not read here)' % (VCONST_SMALL, VCONST_SMALL - 1))
    P('} regs;')
    P('')
    P('// One EGxTexOp on all four channels, as ff.frag has it; the caller keeps rgb from the colour')
    P('// op and a from the alpha op. CURRENT alpha and DIFFUSE alpha are the same value at stage 0.')
    P('vec4 ff_op(int op, vec4 t, vec4 cur)')
    P('{')
    P('    float a = cur.a;')
    P('    if (op == 0) return t * cur;')
    P('    if (op == 1) return 2.0 * t * cur;')
    P('    if (op == 2) return t + cur;')
    P('    if (op == 3) return cur;')
    P('    if (op == 4) return cur * a + t * (1.0 - a);')
    P('    if (op == 5) return t * a + cur * (1.0 - a);')
    P('    return cur;')
    P('}')
    P('')
    P('void main()')
    P('{')
    P('    vec4 cur = clamp(color0, 0.0, 1.0);')
    if point_sprite:
        P('    vec2 uv = gl_PointCoord;')
    elif has_uv:
        P('    vec2 uv = texcoord0.xy;')
    else:
        P('    vec2 uv = vec2(0.0);')
    P('    int cop = int(regs.misc.z + 0.5);')
    P('    int aop = int(regs.misc.w + 0.5);')
    P('    if (cop <= 5)')
    P('    {')
    P('        vec4 t = texture(s0, uv);')
    P('        cur = clamp(vec4(ff_op(cop, t, cur).rgb, ff_op(aop, t, cur).a), 0.0, 1.0);')
    P('    }')
    P('')
    P('    // Render state 7 (alpha test), the same discard as every translated pixel module.')
    P('    if (regs.misc.x > 0.0 && cur.a < regs.misc.x) discard;')
    if fixed_fog:
        P('')
        P('    // The fixed-function fog blend, on rgb only, with the vertex shader\'s oFog as the factor')
        P('    // (clamped as the hardware clamps the fog register) while fog is on.')
        P('    if (regs.p[%d].w > 0.0) cur.rgb = mix(regs.p[%d].rgb, cur.rgb, clamp(fog0, 0.0, 1.0));'
          % (FOG_ROW, FOG_ROW))
    P('')
    P('    oC0 = cur;')
    P('}')
    text = '\n'.join(L) + '\n'
    return text, dict(samplers={0: '2d'}, reads=reads, flags=flags)


def companion_block(md5, meta, names, spv_rel):
    L = ['record %s pixel ff %s vconst=%d pconst=%d' % (md5, spv_rel, VCONST_SMALL, PCONST)]
    for n in names:
        L.append('name %s' % n)
    for slot in sorted(meta['samplers']):
        L.append('sampler %d %s' % (slot, meta['samplers'][slot]))
    for s in meta['reads']:
        L.append('reads %s' % s)
    for f in sorted(meta['flags']):
        L.append('flag %s' % f)
    return '\n'.join(L)


# ---- compiling ---------------------------------------------------------------------------------

def find_glslang(arg):
    for cand in (arg, os.environ.get('GLSLANG'), DEFAULT_GLSLANG, shutil.which('glslang'),
                 shutil.which('glslangValidator')):
        if cand and os.path.exists(cand):
            return cand
    return None


def compile_glsl(glslang, glsl_path, spv_path, stage):
    """glslang -V --target-env vulkan1.0 --spirv-val --spirv-dis; returns (ok, spvasm, log)."""
    cmd = [glslang, '-V', '--target-env', 'vulkan1.0', '--spirv-val', '--spirv-dis', '-S', stage,
           '-o', spv_path, glsl_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.replace('\r\n', '\n')
    lines = out.split('\n')
    if lines and lines[0].strip() == os.path.basename(glsl_path):
        lines = lines[1:]
    dis = '\n'.join(lines)
    ok = res.returncode == 0 and os.path.exists(spv_path) and '; SPIR-V' in dis
    return ok, dis, (res.stderr + out).strip()


# ---- outputs -----------------------------------------------------------------------------------

def manifest_block(em, rec, spv_rel, companion=None, bones=None):
    L = ['record %s %s %s %s vconst=%d pconst=%d'
         % (rec.md5, rec.kind, em.sh.profile, spv_rel, em.vconst, PCONST)]
    for n in rec.names:
        L.append('name %s' % n)
    if rec.kind == 'pixel':
        for slot in sorted(em.samplers):
            L.append('sampler %d %s' % (slot, em.samplers[slot]))
        for s in em.reads:
            L.append('reads %s' % s)
    else:
        for nm, sem in em.inputs.items():
            L.append('input %s %s' % (sem, nm))
        for s in em.outputs:
            L.append('output %s' % s)
    for f in sorted(em.flags):
        L.append('flag %s' % f)
    if companion:
        L.append('companion %s' % companion)
    if bones:
        L.append('bones %s' % bones)
    return '\n'.join(L)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--stage', default='0,1', help='comma-separated stages (default 0,1)')
    ap.add_argument('--all', action='store_true', help='attempt every record, no stage vocabulary')
    ap.add_argument('--out', default=None, help='output tree (default shaders/translated for staged runs)')
    ap.add_argument('--names', default=None, help='only these Name:perm (comma-separated)')
    ap.add_argument('--print', action='store_true', help='print the GLSL of each emitted record')
    ap.add_argument('--glslang', default=None)
    ap.add_argument('--no-compile', action='store_true', help='emit GLSL only')
    args = ap.parse_args(argv)

    glslang = None if args.no_compile else find_glslang(args.glslang)
    if not args.no_compile and glslang is None:
        print('no glslang.exe found (--glslang PATH or $GLSLANG)', file=sys.stderr)
        return 2

    stages = [] if args.all else [int(s) for s in args.stage.split(',') if s.strip()]
    only = set(args.names.split(',')) if args.names else None
    out = args.out
    scratch = None
    if out is None:
        if args.all or only:
            scratch = tempfile.mkdtemp(prefix='translate-')
            out = scratch
        else:
            out = DEFAULT_OUT
    write_tree = scratch is None

    recs = enumerate_records()
    # Fold byte-identical records: one module per md5, every Name:perm that has it listed.
    by_md5 = collections.OrderedDict()
    names = collections.defaultdict(list)
    for r in recs:
        if r.sh is None:
            continue
        names[r.md5].append(r.key)
        if r.perm == 0:
            names[r.md5].append(r.name)      # the bare name the engine interns for record 0
        by_md5.setdefault(r.md5, r)

    # Which records does this run select, and under which stage?
    selected = collections.OrderedDict()      # md5 -> stage dict or None (--all)
    stage_of = {}
    for r in recs:
        if r.sh is None:
            continue
        if only is not None and r.key not in only and r.name not in only:
            continue
        if args.all:
            selected.setdefault(r.md5, None)
            stage_of.setdefault(r.md5, 'all')
            continue
        for s in stages:
            if STAGES[s]['select'](r.kind, r.name, r.perm):
                selected.setdefault(r.md5, STAGES[s])
                stage_of.setdefault(r.md5, s)
                break

    if write_tree:
        for kind in ('vertex', 'pixel'):
            d = os.path.join(out, kind)
            if os.path.isdir(d):
                shutil.rmtree(d)
    for kind in ('vertex', 'pixel'):
        os.makedirs(os.path.join(out, kind), exist_ok=True)

    def write_module(kind, md5, glsl):
        """Writes <kind>/<md5>.glsl, compiles it to .spv and keeps the .spvasm; returns (status, log)."""
        kind_dir = os.path.join(out, kind)
        glsl_path = os.path.join(kind_dir, md5 + '.glsl')
        spv_path = os.path.join(kind_dir, md5 + '.spv')
        with open(glsl_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(glsl)
        if args.no_compile:
            return 'emitted', ''
        ok, dis, log = compile_glsl(glslang, glsl_path, spv_path, 'vert' if kind == 'vertex' else 'frag')
        if not ok:
            for p in (glsl_path if not args.print else None, spv_path):
                if p and os.path.exists(p):
                    os.remove(p)
            return 'failed', log.replace('\n', ' | ')[:400]
        with open(os.path.join(kind_dir, md5 + '.spvasm'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(dis)
        return 'validated', ''

    results = []          # (rec, stage, status, reason, em)
    blocks = []
    index_lines = []
    companions = collections.OrderedDict()    # md5 of the text -> (meta, [names], status, log)
    bones_made = 0
    bones_failed = 0
    bones_refused = collections.Counter()     # why a 256-row module has no bones variant
    for md5, st in selected.items():
        rec = by_md5[md5]
        rec.names = names[md5]
        em = Emitter(rec, st)
        try:
            glsl = em.run()
        except Refused as e:
            results.append((rec, stage_of[md5], 'refused', str(e), em))
            continue
        if args.print:
            print(glsl)
        status, log = write_module(rec.kind, md5, glsl)
        if status == 'failed':
            results.append((rec, stage_of[md5], 'failed', log, em))
            continue
        results.append((rec, stage_of[md5], status, '', em))
        # The fixed-function pixel stage a flagged vs_2_0 record is drawn with (no pixel shader
        # exists for it): generated, compiled and folded by the md5 of its own text.
        comp_md5 = None
        if rec.kind == 'vertex' and em.flags:
            ctext, meta = companion_module(em, rec)
            comp_md5 = hashlib.md5(ctext.encode('utf-8')).hexdigest()
            if comp_md5 not in companions:
                if args.print:
                    print(ctext)
                cstatus, clog = write_module('pixel', comp_md5, ctext)
                companions[comp_md5] = [meta, [], cstatus, clog]
            companions[comp_md5][1].extend(n + '@ff' for n in rec.names)
            if companions[comp_md5][2] == 'failed':
                comp_md5 = None
        # The bone-palette variant of a 256-row vertex module (BONE_BASE), beside it as
        # <md5>.bones.spv; a record it does not fit keeps the 256-row path alone.
        bones_rel = None
        if rec.kind == 'vertex' and em.vconst == VCONST_LARGE:
            bem = Emitter(rec, st, bones=True)
            try:
                bglsl = bem.run()
            except Refused as e:
                bones_refused[str(e)] += 1
            else:
                if args.print:
                    print(bglsl)
                bstatus, blog = write_module('vertex', md5 + '.bones', bglsl)
                if bstatus == 'failed':
                    print('FAILED bones variant %s: %s' % (' = '.join(rec.names), blog), file=sys.stderr)
                    bones_failed += 1
                else:
                    bones_rel = 'vertex/%s.bones.spv' % md5
                    bones_made += 1
        blocks.append(manifest_block(em, rec, '%s/%s.spv' % (rec.kind, md5), comp_md5, bones_rel))
        for n in rec.names:
            if ':' in n:
                extra = (' '.join('s%d:%s' % kv for kv in sorted(em.samplers.items())) or '-') \
                    if rec.kind == 'pixel' else ' '.join('%s=%s' % (s, r) for r, s in em.inputs.items())
                if comp_md5:
                    extra += ' companion=%s' % comp_md5
                index_lines.append('%-36s %s %-6s vconst=%-3d %s' % (n, md5, rec.kind, em.vconst, extra))
    comp_blocks = []
    for cmd5, (meta, cnames, cstatus, clog) in companions.items():
        if cstatus == 'failed':
            print('FAILED companion %s: %s' % (' = '.join(cnames), clog), file=sys.stderr)
            continue
        comp_blocks.append(companion_block(cmd5, meta, cnames, 'pixel/%s.spv' % cmd5))
        for n in cnames:
            if ':' in n:
                index_lines.append('%-36s %s %-6s vconst=%-3d %s' % (n, cmd5, 'pixel', VCONST_SMALL,
                                   's0:2d ' + ' '.join(sorted(meta['flags']))))
    if comp_blocks:
        blocks.append('# The fixed-function pixel stages generated for the vs_2_0 records above that carry a\n'
                      '# `companion` line (no pixel shader exists for them; profile `ff`, no bytecode, md5 of\n'
                      '# the module text). Bound when the draw\'s pixel shader name is "".\n' + comp_blocks[0])
        blocks.extend(comp_blocks[1:])

    # Manifest and index
    refused = [(r, s, why) for r, s, status, why, em in results if status != 'validated' and status != 'emitted']
    head = ['# wxl-vulkan-api translated shader manifest -- generated by tools/bls_tools/translate.py',
            '# One block per byte-distinct record, blank-line separated; `#` starts a comment.',
            '# record <md5> <vertex|pixel> <profile> <spv path> vconst=<64|256> pconst=16',
            '# name <Name:perm>        every engine name whose bytes have this md5; bare <Name> for record 0',
            '# sampler <slot> <2d|cube>   pixel: combined image sampler at set 0, binding <slot>',
            '# input <semantic> <vN>      vertex: attribute semantic the module declares (location per semantic)',
            '# output <semantic>          vertex: semantics written (position0, color0.., texcoord0.., fog0, psize0)',
            '# reads <semantic>           pixel: semantics read',
            '# flag fixed_fog             vertex (vs_2_0): writes oFog, the fixed-function fog FACTOR, as fog0;',
            '#                            pixel (ff): blends mix(p[%d].rgb, rgb, fog0) while p[%d].w > 0' % (FOG_ROW, FOG_ROW),
            '# flag point_size            vertex (vs_2_0): writes gl_PointSize from oPts; draw as a point list',
            '# flag point_sprite          pixel (ff): samples stage 0 at gl_PointCoord, reads no texcoord',
            '# companion <md5>            vertex: the `ff` pixel record to bind when the draw\'s ps name is ""',
            '# bones <spv path>           vertex (vconst=256): the same module reading c64.. and the a0 palette',
            '#                            from a storage buffer (set 2, binding 0, std430 vec4 b[%d], b[i] = c(%d + i)),'
            % (BONE_ROWS, BONE_BASE),
            '#                            its uniform block declared v[64]: the draw takes the 64-row slice',
            '# md5 = MD5 of the record\'s `size` bytes after its 16-byte .bls record header: the DX9',
            '# token stream from the version token through the END token, no padding. An `ff` record',
            '# has no bytecode: its md5 is of its own GLSL text.',
            '# Uniform block (set 1, binding 0, std140): { vec4 p[16]; vec4 misc; vec4 v[VCONST]; }',
            '# p[i] at byte 16*i, misc at 256, v[i] at 272 + 16*i; slice = 272 + 16*VCONST bytes.',
            '# Pixel modules declare v[64]; the vertex module\'s vconst sizes the pair\'s slice.',
            '# p[%d] = (fogColor.rgb / 255, fogOn ? 1 : 0), written by the DLL after the pixel-cache' % FOG_ROW,
            '# copy on every draw: no shipped pixel record reads c13..c15.',
            '']
    if refused:
        head.append('# Refused (not translated):')
        for r, s, why in refused:
            head.append('#   %s %s (stage %s): %s' % (r.kind, ' = '.join(r.names), s, why))
        head.append('')
    if write_tree or args.out:
        with open(os.path.join(out, 'manifest.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(head) + '\n\n'.join(blocks) + '\n')
        with open(os.path.join(out, 'index.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.write('# Name:perm -> md5 kind vconst samplers/inputs (one line per engine name)\n')
            f.write('\n'.join(sorted(index_lines)) + '\n')
            if refused:
                f.write('\n# refused\n')
                for r, s, why in refused:
                    f.write('%-36s REFUSED (stage %s): %s\n' % (' = '.join(r.names), s, why))

    # Report
    by_stage = collections.OrderedDict()
    for r, s, status, why, em in results:
        by_stage.setdefault(s, collections.Counter())[status] += 1
    print('records: %d total, %d decodable, %d distinct; %d selected' %
          (len(recs), sum(1 for r in recs if r.sh is not None), len(by_md5), len(selected)))
    for s, c in sorted(by_stage.items(), key=lambda kv: str(kv[0])):
        sel_names = sum(len(names[m]) for m in selected if stage_of[m] == s)
        print('stage %s: %d distinct records (%d names): %s' %
              (s, sum(c.values()), sel_names, ', '.join('%s %d' % kv for kv in sorted(c.items()))))
    if args.all:
        fam = collections.OrderedDict()
        for r, s, status, why, em in results:
            fam.setdefault(census.family(r.name), collections.Counter())[status] += 1
        for f, c in fam.items():
            print('  %-28s %s' % (f, ', '.join('%s %d' % kv for kv in sorted(c.items()))))
    reasons = collections.Counter()
    for r, s, status, why, em in results:
        if status in ('refused', 'failed'):
            reasons[(status, why.split(' (')[0][:90])] += 1
    for (status, why), n in reasons.most_common():
        print('  %s x%d: %s' % (status, n, why))
    for r, s, status, why, em in results:
        if status == 'failed':
            print('FAILED %s: %s' % (' = '.join(r.names), why), file=sys.stderr)
    if companions:
        print('companions (fixed-function pixel stages for the flagged vs_2_0 records): %d generated, %s' %
              (len(companions), ', '.join('%s %d' % kv for kv in sorted(
                  collections.Counter(c[2] for c in companions.values()).items()))))
        for cmd5, (meta, cnames, cstatus, clog) in companions.items():
            print('  %s %s: %s' % (cmd5, cstatus, ' = '.join(cnames)))
    print('bones variants (256-row vertex modules reading the palette from set 2): %d validated, %d failed, '
          '%d without one' % (bones_made, bones_failed, sum(bones_refused.values())))
    for why, n in bones_refused.most_common():
        print('  no variant x%d: %s' % (n, why))
    if write_tree or args.out:
        # The tree-wind variants of the M2 modules just written (windgen.py), with their `wind` lines.
        wind_made, wind_failed = load('windgen').Run(out, glslang, quiet=True)
        print('wind variants (M2 modules bent in model space before c31..c33): %d validated, %d failed'
              % (wind_made, len(wind_failed)))
        # The shadow variants of the world modules (shadowgen.py), with their `shadow` lines.
        shadow_made, shadow_failed = load('shadowgen').Run(out, glslang, quiet=True)
        print('shadow variants (world modules re-projected for the draw replay): %d validated, %d failed'
              % (shadow_made, len(shadow_failed)))
        print('wrote %s' % out)
    elif scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    failed = any(st == 'failed' for _, _, st, _, _ in results) or \
        any(c[2] == 'failed' for c in companions.values()) or bones_failed > 0
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
