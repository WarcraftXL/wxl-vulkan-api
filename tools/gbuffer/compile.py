"""glslang over the variants: the same flags the device's other SPIR-V is built with, in parallel."""
import io
import glob
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

FLAGS = ['-V', '-Os', '--target-env', 'vulkan1.0', '--spirv-val']

# Extra places to look for glslang, after --glslang and $GLSLANG (none by default).
KNOWN = []


def find_glslang(explicit=None):
    """The compiler: --glslang, $GLSLANG, the known copy, then PATH. None when there is none."""
    for candidate in (explicit, os.environ.get('GLSLANG')):
        if candidate and os.path.isfile(candidate):
            return candidate
    for candidate in KNOWN:
        if os.path.isfile(candidate):
            return candidate
    # The path a wxl-core build tree cached (WXL_VK_GLSLANG), with this repository cloned as wxl-core/extensions/wxl-vulkan-api.
    core = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
    for cache in glob.glob(os.path.join(core, 'build', '*', 'CMakeCache.txt')):
        for line in io.open(cache, encoding='utf-8', errors='replace'):
            if line.startswith('WXL_VK_GLSLANG:FILEPATH='):
                path = line.split('=', 1)[1].strip()
                if os.path.isfile(path):
                    return path
    for name in ('glslang', 'glslangValidator'):
        found = shutil.which(name)
        if found:
            return found
    return None


def compile_one(glslang, src, dst, stage):
    """Compiles one GLSL file; (True, '') or (False, the compiler's last lines)."""
    r = subprocess.run([glslang] + FLAGS + ['-S', stage, src, '-o', dst], capture_output=True, text=True)
    if r.returncode == 0:
        return True, ''
    lines = (r.stdout + r.stderr).strip().splitlines()
    return False, ' | '.join(lines[-4:])


def compile_all(glslang, jobs, tasks):
    """`tasks`: (src, dst, stage) triples. Returns a list of (ok, message) in the same order."""
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return list(pool.map(lambda t: compile_one(glslang, *t), tasks))
