"""The run's census: per family and kind, what was eligible, what was made, what failed and why."""
import collections


class Census:
    def __init__(self):
        self.counts  = collections.Counter()   # (family, kind, what) -> n
        self.reasons = collections.Counter()   # (family, kind, reason) -> n
        self.failed  = []                      # (name, variant tag, message)

    def note(self, family, kind, what, n=1):
        self.counts[(family, kind, what)] += n

    def refused(self, family, kind, reason):
        self.counts[(family, kind, 'refused')] += 1
        self.reasons[(family, kind, reason)] += 1

    def failure(self, name, tag, message, family, kind):
        self.counts[(family, kind, 'failed')] += 1
        self.failed.append((name, tag, message))

    def lines(self):
        fams  = sorted({f for f, _, _ in self.counts})
        whats = ('eligible', 'lit', 'unlit', 'mcsh', 'written', 'compiled', 'refused', 'failed')
        out = ['%-10s %-7s %s' % ('family', 'kind', ' '.join('%9s' % w for w in whats))]
        for f in fams:
            for k in ('vertex', 'pixel'):
                if not any(self.counts[(f, k, w)] for w in whats):
                    continue
                out.append('%-10s %-7s %s' % (f, k, ' '.join('%9d' % self.counts[(f, k, w)] for w in whats)))
        for (f, k, reason), n in sorted(self.reasons.items()):
            out.append('  refused x%d (%s %s): %s' % (n, f, k, reason))
        for name, tag, message in self.failed[:20]:
            out.append('  FAILED %s %s: %s' % (name, tag, message))
        if len(self.failed) > 20:
            out.append('  ... %d more failures' % (len(self.failed) - 20))
        return out

    def report(self, path, header):
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(header + '\n\n```\n' + '\n'.join(self.lines()) + '\n```\n')
