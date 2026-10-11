# tools

The offline shader translator: it turns the 3.3.5a client's own DX9 shaders into the SPIR-V modules the device
draws with. Plain Python 3.8+ (standard library only) and [glslang](https://github.com/KhronosGroup/glslang/releases).

Nothing extracted from the client is versioned in this repository: run the translator against your own client,
and keep its output (`shaders/translated/`) out of commits.

## Quick start

```bash
python tools/translate_shaders.py --client "C:\Games\World of Warcraft 3.3.5a"
```

| Option | |
| --- | --- |
| `--client <folder>` | the 3.3.5a (12340) client folder, or its `Data` folder (required) |
| `--glslang <path>` | `glslang.exe` / `glslangValidator.exe`; otherwise `$GLSLANG`, a wxl-core build tree's cached `WXL_VK_GLSLANG`, `$VULKAN_SDK`, then `PATH` |
| `--out <folder>` | the output tree (default: `shaders/translated/` in this repository) |
| `--deploy` | also copy `manifest.txt`, `index.txt` and every `.spv` into `<client>/Extensions/wxl-vulkan-api/shaders/translated/`, removing stale modules |

`translate_shaders.py` runs, in order:

1. `bls_tools/translate.py --stage 0,1,2,3,4`: every record the engine loads, with its `bones`, `wind` and
   `shadow` variants (`windgen.py`, `shadowgen.py`, called by it);
2. `python -m gbuffer`: the G-buffer variants of the world families.

Without `--deploy`, wxl-core's build copies the tree into the client on its next deploy (`build.ps1`).

The archives are read in the client's own order: `common.MPQ`, `common-2.MPQ`, `expansion.MPQ`, `lichking.MPQ`,
`patch.MPQ`, then `patch-2.MPQ` .. `patch-9.MPQ` and `patch-A.MPQ` .. `patch-Z.MPQ`; a later archive wins. A custom
patch that replaces a shader is therefore translated too. A folder patch (a `patch-N.MPQ` folder of loose files) is
not read, and an archive another program holds open is skipped with a warning.

## What a run writes

Per byte-distinct shader record, under `vertex/` or `pixel/`:

| File | |
| --- | --- |
| `<md5>.glsl`, `.spv`, `.spvasm` | the module: GLSL 4.50 source, SPIR-V, and glslang's disassembly |
| `<md5>.bones.*` | vertex modules with 256 constants: the bone palette read from a storage buffer |
| `<md5>.wind.*` | M2 modules: the tree-wind variant |
| `<md5>.shadow*.*` | world modules: re-projected for the shadow replay |
| `<md5>.gbuf.*`, `.wind.gbuf.*` | world modules: the G-buffer variant |

plus `manifest.txt` (the device's contract: one block per record, its engine names, samplers, inputs and variant
lines) and `index.txt` (the same, one line per engine name, for people). The md5 is of the record's bytes as the
engine hands them to `IShaderCreateVertex/Pixel`: the `size` bytes after the 16-byte record header, version token
through END token. The device looks modules up by that md5, so a translated tree always matches the client it was
made from.

The default run wipes `vertex/` and `pixel/` first: the tree is always one run's output.

## The individual tools

Run the `bls_tools` scripts from their folder, with `WXL_CLIENT_DATA` set to the client's `Data` folder
(`translate_shaders.py` sets it for its own run).

| Script | |
| --- | --- |
| `translate.py [--stage 0,1 \| --all] [--out dir] [--names A:0,B]` | the DX9 -> GLSL -> SPIR-V translator itself |
| `windgen.py`, `shadowgen.py` | the tree-wind and shadow variants of a translated tree (`translate.py` runs both) |
| `listfile.py [prefix]` | the shader inventory, from the archives' own `(listfile)` |
| `extract_all.py <Vertex\|Pixel> <profile> <name>` | split one `.bls` into its permutations and disassemble each |
| `dx9d.py <file.fxo>...` | the DX9 (SM2/SM3) token-stream disassembler |
| `dx9ir.py <file.fxo>...` | the structured decoder the translator uses (relative addressing, predication) |
| `validate_disasm.py "<glob>"` | check `dx9d.py` against `fxc /dumpbin`, line for line (needs the Windows SDK's `fxc`; `$FXC`) |
| `contract.py "<glob>"` | per permutation: which `c#` the engine supplies, which the bytecode `def`s |
| `cmp_tokens.py <a> <b>` | compare two token streams, ignoring comment blocks |
| `census.py [--validate] [--embedded] [-o file]` | every record the engine loads under vs_3_0 / ps_3_0, counted by opcode, register file, modifier, semantic and sampler |
| `mpq_extract.py` | the MPQ reader the others import |

`gbuffer/` is a package: `python -m gbuffer --help` from this folder.

```bash
cd tools/bls_tools
set WXL_CLIENT_DATA=C:\Games\World of Warcraft 3.3.5a\Data
python listfile.py shaders                        # what exists
python extract_all.py Pixel ps_3_0 MapObjDiffuse  # its permutations, disassembled
python contract.py "Pixel_ps_3_0_MapObjDiffuse/*.fxo"
python translate.py --names UI:0,Combiners_Mod:4 --print    # look at two records, write nothing
```

`extract_all.py` and `census.py --dump` write raw client bytecode (`.fxo`) next to the scripts: those files are
the client's, keep them out of commits.

## How the translation works

Every record the engine loads is resolved through the engine's own fallback chain (vertex 3 -> 2 -> 1, pixel
4 -> 3 -> 2 -> 1), skipping 0-byte containers the way the engine does, and folded by md5. Each one is decoded with
`dx9ir.py` and rewritten as a GLSL 4.50 module whose interface is fixed **by semantic, not by register**, so any
vertex module pairs with any pixel module: vertex inputs at fixed locations per semantic, varyings at fixed
locations per semantic, the constant registers as a uniform block (`v[64]` or `v[256]`, `p[16]`), the samplers as
combined image samplers at set 0. glslang compiles it with `-V --target-env vulkan1.0 --spirv-val`.

A record that uses something no rule covers is refused: logged, listed as a comment at the top of the manifest,
nothing emitted. On a stock 3.3.5a client every record translates.

The four vs_2_0 weather records (`patter`, `rain`, `sand`, `snowpoint`) write the fixed-function outputs `oFog` /
`oPts` and have no pixel shader in any profile; for them the translator emits the epilogues (`fog0`,
`gl_PointSize`), flags them `fixed_fog` / `point_size` in the manifest, and generates a **companion** pixel module
(the fixed-function stage-0 cascade, alpha test, fog blend and point-sprite UV) as a `record ... pixel ff` block.

## Two things worth knowing before trusting a disassembly

**Validate the disassembler on any new container.** The DX9 register type is split across two bit ranges
(`((t >> 28) & 7) | (((t >> 11) & 3) << 3)`), and the opcode table above 66 is easy to get subtly wrong.
`validate_disasm.py` compares `dx9d.py` with `fxc /dumpbin`, which disassembles a bare DX9 token stream directly.

**The reconstruction test is the real check on a contract.** Write the HLSL from the contract you derived,
compile it with `fxc /T ps_3_0`, and compare the token stream with the shipped record using `cmp_tokens.py`. A
wrong register, a swapped colour channel or an inverted fog factor all produce a different stream: at build time,
not in game three weeks later.

## File formats, for reference

**`.bls` container**: `<dir>\<profile>\<name>.bls`, e.g. `Shaders\Pixel\ps_3_0\UI.bls`. Header: `uint32 magic
'GXSH'` (`0x47585348`), `uint32 version` (`0x10003`), `uint32 permutationCount`. Then that many records: `uint32 a;
uint32 b; uint16 c; uint16 d; uint32 size; uint8 code[size];` padded to 4. For the D3D profiles the code is a raw
DX9 token stream; the OpenGL profiles hold ARB/NV assembly instead. A missing or bad file makes the engine fall
back one profile and retry, so an absent `ps_3_0` container is silent, not fatal.

**`shaders\effects\*.wfx`**: plain text, and the only statement of which vertex shader is paired with which pixel
shader. Also gives the fixed-function fallback, an independent description of the same intent.
