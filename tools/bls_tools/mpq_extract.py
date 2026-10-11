import os, struct, zlib, sys, bz2

# ---- Storm crypt table ----
def make_crypt_table():
    t = [0]*0x500
    seed = 0x00100001
    for i in range(0x100):
        idx = i
        for _ in range(5):
            seed = (seed*125 + 3) % 0x2AAAAB
            t1 = (seed & 0xFFFF) << 0x10
            seed = (seed*125 + 3) % 0x2AAAAB
            t2 = (seed & 0xFFFF)
            t[idx] = (t1 | t2)
            idx += 0x100
    return t
CT = make_crypt_table()

def hash_string(s, htype):
    seed1 = 0x7FED7FED
    seed2 = 0xEEEEEEEE
    for ch in s.upper():
        c = ord(ch)
        seed1 = (CT[(htype*0x100) + c] ^ ((seed1 + seed2) & 0xFFFFFFFF)) & 0xFFFFFFFF
        seed2 = (c + seed1 + seed2 + (seed2 << 5) + 3) & 0xFFFFFFFF
    return seed1

def decrypt(data, key):
    n = len(data)//4
    words = list(struct.unpack('<%dI' % n, data[:n*4]))
    seed = 0xEEEEEEEE
    key = key & 0xFFFFFFFF
    for i in range(n):
        seed = (seed + CT[0x400 + (key & 0xFF)]) & 0xFFFFFFFF
        ch = (words[i] ^ ((key + seed) & 0xFFFFFFFF)) & 0xFFFFFFFF
        words[i] = ch
        key = ((((~key) << 0x15) + 0x11111111) | (key >> 0x0B)) & 0xFFFFFFFF
        seed = (ch + seed + (seed << 5) + 3) & 0xFFFFFFFF
    return struct.pack('<%dI' % n, *words) + data[n*4:]

FLAG_EXISTS      = 0x80000000
FLAG_SINGLE_UNIT = 0x01000000
FLAG_IMPLODE     = 0x00000100
FLAG_COMPRESS    = 0x00000200
FLAG_ENCRYPTED   = 0x00010000
FLAG_FIX_KEY     = 0x00020000
FLAG_SECTOR_CRC  = 0x04000000


class MPQ:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        self._find_header()
        self._read_tables()

    def _find_header(self):
        f = self.f
        size = os.path.getsize(self.path)
        off = 0
        while off < size:
            f.seek(off)
            magic = f.read(4)
            if magic == b'MPQ\x1a':
                self.archive_offset = off
                hdr = f.read(0x20-4)
                (self.header_size, self.archive_size, self.format_version,
                 self.block_size_exp, self.hash_table_pos, self.block_table_pos,
                 self.hash_table_size, self.block_table_size) = struct.unpack('<IIHHIIII', hdr)
                return
            elif magic == b'MPQ\x1b':
                ud = f.read(12)
                user_data_size, hdr_off, ud_hdr_size = struct.unpack('<III', ud)
                off = off + hdr_off
                continue
            else:
                off += 0x200
        raise RuntimeError('no MPQ header in %s' % self.path)

    def _read_tables(self):
        f = self.f
        f.seek(self.archive_offset + self.hash_table_pos)
        raw = f.read(self.hash_table_size*16)
        raw = decrypt(raw, hash_string('(hash table)', 3))
        self.hash_table = [struct.unpack_from('<IIHHI', raw, i*16) for i in range(self.hash_table_size)]
        f.seek(self.archive_offset + self.block_table_pos)
        raw = f.read(self.block_table_size*16)
        raw = decrypt(raw, hash_string('(block table)', 3))
        self.block_table = [struct.unpack_from('<IIII', raw, i*16) for i in range(self.block_table_size)]

    def find(self, name):
        idx = hash_string(name, 0) % self.hash_table_size
        a = hash_string(name, 1)
        b = hash_string(name, 2)
        for i in range(self.hash_table_size):
            e = self.hash_table[(idx+i) % self.hash_table_size]
            na, nb, loc, plat, blk = e
            if blk == 0xFFFFFFFF:
                return None
            if na == a and nb == b and blk != 0xFFFFFFFE:
                return blk
        return None

    def read_file(self, name):
        blk = self.find(name)
        if blk is None:
            return None
        file_pos, cmp_size, file_size, flags = self.block_table[blk]
        if not (flags & FLAG_EXISTS):
            return None
        base = self.archive_offset + file_pos
        sector_size = 512 << self.block_size_exp
        key = None
        if flags & FLAG_ENCRYPTED:
            bn = name.split('\\')[-1]
            key = hash_string(bn, 3)
            if flags & FLAG_FIX_KEY:
                key = ((key + file_pos) ^ file_size) & 0xFFFFFFFF
        info = dict(flags=hex(flags), cmp_size=cmp_size, file_size=file_size,
                    sector_size=sector_size, encrypted=bool(flags & FLAG_ENCRYPTED),
                    single_unit=bool(flags & FLAG_SINGLE_UNIT))
        self.f.seek(base)
        if flags & FLAG_SINGLE_UNIT:
            data = self.f.read(cmp_size)
            if key is not None:
                data = decrypt(data, key)
            if (flags & (FLAG_COMPRESS | FLAG_IMPLODE)) and cmp_size < file_size:
                data = decompress_sector(data, file_size)
            return data[:file_size], info
        nsect = (file_size + sector_size - 1)//sector_size
        if flags & (FLAG_COMPRESS | FLAG_IMPLODE):
            ntbl = nsect + 1
            if flags & FLAG_SECTOR_CRC:
                ntbl += 1
            raw = self.f.read(ntbl*4)
            if key is not None:
                raw = decrypt(raw, (key-1) & 0xFFFFFFFF)
            offs = list(struct.unpack('<%dI' % ntbl, raw))
            out = b''
            for i in range(nsect):
                self.f.seek(base + offs[i])
                chunk = self.f.read(offs[i+1]-offs[i])
                if key is not None:
                    chunk = decrypt(chunk, (key + i) & 0xFFFFFFFF)
                want = min(sector_size, file_size - len(out))
                if len(chunk) == want:
                    out += chunk
                else:
                    out += decompress_sector(chunk, want)
            return out[:file_size], info
        else:
            data = self.f.read(file_size)
            if key is not None:
                data = decrypt(data, key)
            return data, info


def decompress_sector(chunk, want):
    mask = chunk[0]
    body = chunk[1:]
    if mask == 0x02:
        return zlib.decompress(body)
    if mask == 0x10:
        return bz2.decompress(body)
    if mask == 0x08:
        raise RuntimeError('PKWARE implode not implemented')
    if mask == 0x01:
        raise RuntimeError('huffman not implemented')
    raise RuntimeError('unknown compression mask 0x%02x' % mask)


# The client's order: base archives, then patch.MPQ, patch-2 .. patch-9, patch-A .. patch-Z (a later archive
# wins). A custom patch that is locked by a running process is skipped with a warning, so a run made while the
# client runs does not see what it adds.
LOAD_ORDER = (['common.MPQ', 'common-2.MPQ', 'expansion.MPQ', 'lichking.MPQ', 'patch.MPQ']
              + ['patch-%s.MPQ' % c for c in '23456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'])

if __name__ == '__main__':
    datadir = os.environ.get('WXL_CLIENT_DATA', '')   # the client's Data folder
    outdir = sys.argv[1]
    targets = ['Shaders\\Pixel\\ps_3_0\\UI.bls',
               'Shaders\\Pixel\\ps_2_0\\UI.bls',
               'Shaders\\Vertex\\vs_3_0\\UI.bls']
    archives = []
    for n in LOAD_ORDER:
        p = os.path.join(datadir, n)
        if os.path.exists(p):
            try:
                m = MPQ(p)
            except OSError as e:
                print('  !! %s unreadable, skipped: %s' % (n, e))
                continue
            archives.append((n, m))
            print('opened %-14s fmt=%d hashsz=%d blocksz=%d sectexp=%d archoff=0x%x'
                  % (n, m.format_version, m.hash_table_size, m.block_table_size,
                     m.block_size_exp, m.archive_offset))
    for t in targets:
        found = None
        for n, m in archives:
            if m.find(t) is not None:
                found = (n, m)
                print('  %s present in %s' % (t, n))
        if not found:
            print('  !! %s NOT FOUND' % t)
            continue
        n, m = found
        data, info = m.read_file(t)
        name = t.replace('\\', '_')
        op = os.path.join(outdir, name)
        with open(op, 'wb') as fh:
            fh.write(data)
        print('  -> WINNER %s from %s  len=%d  %r' % (t, n, len(data), info))
