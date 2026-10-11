"""A translated GLSL module (the translator's output, or one of its `bones` / `wind` variants) taken
apart into the pieces a variant edits: the head (everything before `void main()`), main's body, and the
closing brace. The text is never re-emitted from a model: the pieces are the original lines, edited in
place, so what the translator wrote stays byte for byte wherever a variant does not touch it.

Every module the translator writes has the same shape, which is what these helpers rely on:

    <comments, layout(location = N) in/out declarations, the Regs block, consts, helpers>
    void main()
    {
        <register declarations>            vec4 r0 = vec4(0.0); ... (vertex: the output registers too)
        <pixel: the varying prologue>      vec4 v0 = color0; ... vec4 v5 = vec4(fog0, 0.0, 0.0, 0.0);

        <the body>
        <vertex: the epilogue>             gl_Position = vec4(o0.x, -o0.y, o0.z, o0.w); color0 = o1; ...
        <pixel: the alpha test>            if (regs.misc.x > 0.0 && oC0.a < regs.misc.x) discard;
    }
"""
import re

OUT_DECL = re.compile(r'^layout\(location = (\d+)\) out ')
IN_DECL  = re.compile(r'^layout\(location = (\d+)\) in ')


class Module:
    def __init__(self, text):
        self.lines = text.split('\n')
        try:
            self.main = next(i for i, l in enumerate(self.lines) if l.startswith('void main()'))
        except StopIteration:
            raise ValueError('no main()')
        if self.lines[self.main + 1].strip() != '{':
            raise ValueError('main() does not open on the next line')
        self.body_start = self.main + 2
        self.close = max(i for i, l in enumerate(self.lines) if l.strip() == '}')
        if self.close <= self.body_start:
            raise ValueError('main() does not close')

    # --- the head -----------------------------------------------------------------------------------

    def head(self):
        return self.lines[:self.main]

    def last_decl(self, pattern):
        """The index of the last head line matching `pattern` (a compiled regex), or None."""
        found = None
        for i in range(self.main):
            if pattern.match(self.lines[i]):
                found = i
        return found

    def max_location(self, pattern):
        """The highest `layout(location = N)` the head declares for `pattern`, -1 when none."""
        best = -1
        for i in range(self.main):
            m = pattern.match(self.lines[i])
            if m:
                best = max(best, int(m.group(1)))
        return best

    def const_vec4(self, name):
        """The four components of `const vec4 <name> = vec4(a, b, c, d);`, or None."""
        rx = re.compile(r'^const vec4 %s = vec4\((.+)\);\s*$' % re.escape(name))
        for i in range(self.main):
            m = rx.match(self.lines[i])
            if m:
                try:
                    return tuple(float(x) for x in m.group(1).split(','))
                except ValueError:
                    return None
        return None

    # --- the body -----------------------------------------------------------------------------------

    def body_range(self):
        return self.body_start, self.close

    def find(self, pattern, start=None, stop=None):
        """The index of the first body line matching `pattern` (compiled), from `start`, or None."""
        start = self.body_start if start is None else start
        stop  = self.close if stop is None else stop
        for i in range(start, stop):
            if pattern.match(self.lines[i]):
                return i
        return None

    def insert(self, at, new_lines):
        """Inserts lines before index `at` and moves the indices this object keeps."""
        self.lines[at:at] = list(new_lines)
        n = len(new_lines)
        if at <= self.main:
            self.main += n
            self.body_start += n
        elif at < self.body_start:
            self.body_start += n
        self.close += n

    def replace(self, at, new_lines):
        """Replaces the line at `at` by `new_lines`."""
        self.lines[at:at + 1] = list(new_lines)
        self.close += len(new_lines) - 1

    def text(self):
        return '\n'.join(self.lines)


def indent_of(line):
    return line[:len(line) - len(line.lstrip())]
