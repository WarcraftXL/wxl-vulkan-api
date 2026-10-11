"""Builds shaders/translated/ from your own 3.3.5a client: the client's DX9 shaders, translated to SPIR-V.

The translated modules are not shipped with this repository: they are derived from the client's own
shader archives, so each user makes them from their own copy of the game. One command does the whole
chain, in the order the device expects:

    1. bls_tools/translate.py --stage 0,1,2,3,4   every record the engine loads (vertex, pixel, the fixed-function
                                                  companions), with their `bones`, `wind` and `shadow` variants
    2. python -m gbuffer                          the G-buffer variants of the world families

Usage (Python 3.8+, no package to install; glslang must be reachable):

    python tools/translate_shaders.py --client "C:/Games/World of Warcraft 3.3.5a"
    python tools/translate_shaders.py --client <client> --glslang <path to glslang.exe>
    python tools/translate_shaders.py --client <client> --deploy     # also copy the result into the client

The output is shaders/translated/ in this repository (the default; --out elsewhere). wxl-core's build copies it
into <client>/Extensions/wxl-vulkan-api/shaders/ on its next deploy; --deploy copies it at once.
"""
import argparse
import glob
import io
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..'))
DEFAULT_OUT = os.path.join(REPO, 'shaders', 'translated')


def find_data(client):
    """The client's Data folder, from the client folder or the Data folder itself."""
    for cand in (os.path.join(client, 'Data'), client):
        if os.path.isfile(os.path.join(cand, 'common.MPQ')):
            return os.path.normpath(cand)
    return None


def find_glslang(explicit):
    """--glslang, $GLSLANG, a wxl-core build tree's cached WXL_VK_GLSLANG, the Vulkan SDK, then PATH."""
    for cand in (explicit, os.environ.get('GLSLANG')):
        if cand and os.path.isfile(cand):
            return cand
    # This repository cloned as wxl-core/extensions/wxl-vulkan-api: the core's build trees know where glslang is.
    core = os.path.normpath(os.path.join(REPO, '..', '..'))
    for cache in glob.glob(os.path.join(core, 'build', '**', 'CMakeCache.txt'), recursive=True):
        for line in io.open(cache, encoding='utf-8', errors='replace'):
            if line.startswith('WXL_VK_GLSLANG:FILEPATH='):
                path = line.split('=', 1)[1].strip()
                if os.path.isfile(path):
                    return path
    sdk = os.environ.get('VULKAN_SDK')
    if sdk:
        for name in ('glslang.exe', 'glslangValidator.exe', 'glslang', 'glslangValidator'):
            for sub in ('Bin', 'bin'):
                path = os.path.join(sdk, sub, name)
                if os.path.isfile(path):
                    return path
    for name in ('glslang', 'glslangValidator'):
        found = shutil.which(name)
        if found:
            return found
    return None


def run(cmd, cwd, env):
    print('\n> ' + ' '.join('"%s"' % c if ' ' in c else c for c in cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd, env=env)


def deploy(out, client):
    """Mirrors what the device reads (manifest.txt, index.txt, every .spv) into the client's extension folder."""
    dst = os.path.join(client, 'Extensions', 'wxl-vulkan-api', 'shaders', 'translated')
    copied = removed = 0
    for name in ('manifest.txt', 'index.txt'):
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(out, name), os.path.join(dst, name))
        copied += 1
    for kind in ('vertex', 'pixel'):
        src_dir, dst_dir = os.path.join(out, kind), os.path.join(dst, kind)
        os.makedirs(dst_dir, exist_ok=True)
        wanted = {os.path.basename(p) for p in glob.glob(os.path.join(src_dir, '*.spv'))}
        for name in wanted:
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))
            copied += 1
        for p in glob.glob(os.path.join(dst_dir, '*.spv')):
            if os.path.basename(p) not in wanted:
                os.remove(p)
                removed += 1
    print('deployed %d file(s) to %s (%d stale removed)' % (copied, dst, removed))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--client', required=True, help='the 3.3.5a (12340) client folder (or its Data folder)')
    ap.add_argument('--glslang', default=None, help='glslang.exe / glslangValidator.exe (also $GLSLANG)')
    ap.add_argument('--out', default=DEFAULT_OUT, help='the output tree (default: shaders/translated/ here)')
    ap.add_argument('--deploy', action='store_true', help='copy the result into <client>/Extensions/wxl-vulkan-api/')
    args = ap.parse_args(argv)

    data = find_data(args.client)
    if not data:
        print('no common.MPQ under %r or its Data folder: --client is the 3.3.5a client folder' % args.client,
              file=sys.stderr)
        return 2
    glslang = find_glslang(args.glslang)
    if not glslang:
        print('glslang not found: pass --glslang, set $GLSLANG or $VULKAN_SDK, or put it on PATH '
              '(prebuilt binaries: https://github.com/KhronosGroup/glslang/releases)', file=sys.stderr)
        return 2
    out = os.path.normpath(args.out)
    print('client data : %s\nglslang     : %s\noutput      : %s' % (data, glslang, out))

    env = dict(os.environ, WXL_CLIENT_DATA=data, GLSLANG=glslang)
    py = sys.executable
    rc = run([py, os.path.join(HERE, 'bls_tools', 'translate.py'), '--stage', '0,1,2,3,4', '--glslang', glslang,
              '--out', out], os.path.join(HERE, 'bls_tools'), env)
    if rc != 0:
        print('translate.py failed (%d): nothing more was run' % rc, file=sys.stderr)
        return rc
    rc = run([py, '-m', 'gbuffer', '--translated', out, '--glslang', glslang], HERE, env)
    if rc != 0:
        print('gbuffer failed (%d)' % rc, file=sys.stderr)
        return rc
    if args.deploy:
        deploy(out, os.path.dirname(data))
    print('\ndone: %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
