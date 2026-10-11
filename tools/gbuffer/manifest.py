"""The translated-module manifest (wxl-vulkan-api's shaders/translated/manifest.txt), read for what
this tool needs and written back with its own lines and nothing else changed.

The file is the translator's contract with the device: blocks separated by blank lines, one `record`
line each, then attribute lines (`name`, `input`, `bones`, `wind`, ...). This tool adds, per record it
made variants for, the lines the device reads (wxl-vulkan-api src/vulkan/Manifest.cpp):

    gbuf <spv>              the record's own G-buffer variant (<stem>.gbuf.spv)
    gbufbones <spv>         vertex: the same of its `bones` variant
    gbufwind <spv>          the same of its `wind` variant (a vertex record's bend, a pixel record's leaves' half)
    gbufwindbones <spv>     vertex: the same of its `windbones` variant

An earlier run's lines are dropped before the new ones go in, so a run is idempotent; every other line
is kept where it was, byte for byte.
"""
import io

OUR_TAGS = ('gbuf', 'gbufbones', 'gbufwind', 'gbufwindbones')


class Record:
    """One record block: what the tool reads of it, and the variant lines this run decided."""

    def __init__(self, line):
        parts = line.split()
        if len(parts) < 5 or parts[0] != 'record':
            raise ValueError('not a record line: %r' % line)
        self.md5     = parts[1]
        self.kind    = parts[2]          # 'vertex' or 'pixel'
        self.profile = parts[3]          # vs_3_0, ps_3_0, ..., ff
        self.spv     = parts[4]          # relative to the translated folder, forward slashes
        self.names   = []
        self.bones = self.wind = self.windbones = None
        self.lines   = [line]            # the block's lines, this tool's own excluded
        self.gbuf    = {}                # tag -> spv path, what this run made
        self.prior   = {}                # tag -> spv path, what an earlier run left (kept with --only)

    def add(self, line):
        tok = line.split()
        if not tok:
            return
        if tok[0] in OUR_TAGS:
            if len(tok) >= 2:
                self.prior[tok[0]] = tok[1]   # an earlier run's: made again below, or kept with --only
            return
        self.lines.append(line)
        if len(tok) < 2:
            return
        if tok[0] == 'name':
            self.names.append(tok[1])
        elif tok[0] == 'bones':
            self.bones = tok[1]
        elif tok[0] == 'wind':
            self.wind = tok[1]
        elif tok[0] == 'windbones':
            self.windbones = tok[1]

    @property
    def stem(self):
        return self.spv[:-4]

    def first_name(self):
        return self.names[0] if self.names else self.md5

    def text(self, keep_prior=False):
        out = list(self.lines)
        for tag in OUR_TAGS:
            if tag in self.gbuf:
                out.append('%s %s' % (tag, self.gbuf[tag]))
            elif keep_prior and tag in self.prior:
                out.append('%s %s' % (tag, self.prior[tag]))
        return '\n'.join(out)


class Manifest:
    """The whole file: records and the text between them, in order."""

    def __init__(self, path):
        self.path  = path
        self.items = []                  # Record, or a str (a comment or blank line outside a block)
        text = io.open(path, encoding='utf-8').read()
        current = None
        for line in text.split('\n'):
            if line.startswith('record '):
                current = Record(line)
                self.items.append(current)
            elif current is not None and line.strip() != '':
                current.add(line)
            else:
                current = None
                self.items.append(line)

    def records(self):
        return [i for i in self.items if isinstance(i, Record)]

    def text(self, keep_prior=False):
        return '\n'.join(i.text(keep_prior) if isinstance(i, Record) else i for i in self.items)

    def write(self, path=None, keep_prior=False):
        """keep_prior: the records this run did not make keep an earlier run's lines (--only)."""
        io.open(path or self.path, 'w', encoding='utf-8', newline='\n').write(self.text(keep_prior))
