"""Which engine shader family a record belongs to, and what a G-buffer variant needs to know about it.

The world's families, as the engine names its shaders (the manifest's `name` lines):

    Terrain*                         the terrain: one vertex family, the Terrain0..3 pixel ones
    MapObj*, MapObjU*                the map objects (WMO); the `U` families are their interior groups
    Diffuse_*, CDiffuse_*, Color_*   the models (M2), vertex side
    Combiners_*                      the models' pixel side

Everything else -- the interface, liquids, weather, particles, the engine's full-screen passes, the
stock ground clutter -- gets no variant.

Verified on the translated modules: every lit family computes its light in the vertex
stage and hands it down as `color0`; the pixel stage multiplies its textures by it (the terrain's and the
map objects' by twice it: their vertex stages halve the term for headroom -- the map objects' through the
engine's halved vertex colours, `x 2` in every MapObj* pixel stage) and fogs. That is what makes the neutral
value below exact: evaluating the pixel body with color0 at it yields the textures alone.

The map object families take 0.5 as their neutral, like the terrain (1.0 would store TWICE the texture, clipped at 1
by the RGBA8 attachment): the albedo is the texture, the light term twice color0.
"""
from dataclasses import dataclass

# The material class the albedo's alpha carries (wxl-vulkan-api's VkRenderApi.h, WXL_VK_GBUFFER_CLASS_*).
CLASS_NONE     = 0
CLASS_TERRAIN  = 1
CLASS_MAPOBJ   = 2
CLASS_INTERIOR = 3
CLASS_MODEL    = 4
CLASS_UNLIT    = 5
INTERIOR_BIT   = 32    # the pixel half sets it from the device's row p[14].y (a batch of an interior map object group)
MOVER_BIT      = 64    # the pixel half sets it from the device's row p[14].x (a unit's draw)
PLAIN_BIT      = 128


@dataclass(frozen=True)
class Family:
    key: str            # 'terrain', 'mapobj', 'interior', 'model'
    cls: int            # the class a lit vertex module of it writes
    neutral: float      # the color0 value that leaves the pixel stage's textures as they are
    dark_term: bool     # the pixel stage may add a term independent of color0 (an additive combiner): evaluate it
    baked_shadow: bool  # the pixel stage applies the terrain's baked shadow (MCSH): find it, export it, keep it out of the albedo


TERRAIN  = Family('terrain', CLASS_TERRAIN, 0.5, False, True)
MAPOBJ   = Family('mapobj', CLASS_MAPOBJ, 0.5, False, False)
INTERIOR = Family('interior', CLASS_INTERIOR, 0.5, False, False)
MODEL    = Family('model', CLASS_MODEL, 1.0, True, False)

VERTEX_PREFIXES = ('Terrain', 'MapObj', 'Diffuse_', 'CDiffuse_', 'Color_')
PIXEL_PREFIXES  = ('Terrain', 'MapObj', 'Combiners_')


def of_name(name):
    """The family of one engine name ("MapObjUDiffuse_T1:1"), or None."""
    base = name.split(':')[0]
    if base.startswith('Terrain'):
        return TERRAIN
    if base.startswith('MapObjU'):
        return INTERIOR
    if base.startswith('MapObj'):
        return MAPOBJ
    if base.startswith(('Diffuse_', 'CDiffuse_', 'Color_', 'Combiners_')):
        return MODEL
    return None


def of_record(names, kind):
    """The family of a record from its names, or None when none is a world family's.

    A record is the engine's bytes, shared by every name whose bytes are the same: when an exterior and
    an interior map object family share one, the exterior wins (the same bytes light the same way; the
    interior class is a hint for the consumer, and the shared permutations are the unlit ones anyway).
    The kind keeps a vertex family's names from classifying a pixel record and the other way round.
    """
    prefixes = VERTEX_PREFIXES if kind == 'vertex' else PIXEL_PREFIXES
    fams = [of_name(n) for n in names if n.split(':')[0].startswith(prefixes)]
    fams = [f for f in fams if f is not None]
    if not fams:
        return None
    for pick in (TERRAIN, MAPOBJ, INTERIOR, MODEL):
        if any(f is pick for f in fams):
            return pick
    return None
