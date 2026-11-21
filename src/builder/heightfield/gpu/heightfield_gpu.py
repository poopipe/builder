"""gpu heightfield state: shared lod grids, height textures, tile draw"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyray import (
    Color,
    Material,
    MaterialMapIndex,
    Matrix,
    Vector2,
    Vector3,
    Vector4,
    draw_mesh_instanced,
    ffi,
    load_material_default,
    matrix_translate,
    rl_disable_wire_mode,
    rl_enable_wire_mode,
    vector3_transform,
)

from builder.heightfield.gpu.composite import (
    HeightCompositeCache,
    R32RenderTarget,
    create_height_composite_cache,
    release_r32_target,
    unload_height_composite_cache,
)
from builder.heightfield.gpu.grid_mesh import (
    ensure_lod_grid,
    prewarm_lod_grids,
    release_all_lod_grids,
)
from builder.heightfield.gpu.height_texture import (
    HeightTexture,
    HeightTextureKey,
    height_texture_key,
    height_texture_uniforms,
    make_height_texture,
    release_height_texture,
)
from builder.heightfield.gpu.lod import (
    clamp_neighbor_lods,
    edge_mask_for_tile,
    hybrid_lod_distance,
    raw_lod_from_distance,
)
from builder.heightfield.gpu.terrain_shader import (
    TerrainShader,
    bind_terrain_lights,
    load_terrain_shader,
    set_float,
    set_parent_transform,
    set_vec2,
    set_vec4,
    unload_terrain_shader,
)
from builder.heightfield.heightfield_types import (
    Heightfield,
    HeightfieldLayout,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.view.draw_cache import DrawCache, fill_world_matrix, sync_draw_cache
from builder.view.lighting import Lighting, set_vec3
from builder.view.prepare_mesh import PreparedMesh

heightfield_albedo: Color = Color(210, 210, 210, 255)


@dataclass
class HeightfieldInstanceBatch:
    """reused instance buffer for one (lod, edge_mask) draw"""

    transforms: Any
    count: int
    lod: int
    edge_mask: int


@dataclass
class HeightfieldGpuEntry:
    """cached gpu resources for one heightfield group"""

    texture: HeightTexture
    layout: HeightfieldLayout
    spacing: float
    # owned composite result when edit layers are active; None for raw base
    result_target: R32RenderTarget | None = None
    # flat tile-local instance matrices (layout-sized); None until built
    tile_matrices: Any | None = None
    tile_matrix_key: tuple[int, int, int, float, float, float] | None = None
    cached_levels: list[list[int]] | None = None
    cached_masks: list[list[int]] | None = None
    instance_batches: dict[tuple[int, int], HeightfieldInstanceBatch] = field(
        default_factory=dict
    )


@dataclass
class HeightfieldGpuState:
    """session-wide terrain shader, shared material/grids, and per-group entries"""

    terrain: TerrainShader
    material: Material
    composite: HeightCompositeCache
    # (spacing, lod, edge_mask) -> prepared grid (shares material)
    grids: dict[tuple[float, int, int], PreparedMesh] = field(
        default_factory=dict
    )
    entries: dict[str, HeightfieldGpuEntry] = field(default_factory=dict)
    warmed_spacings: set[float] = field(default_factory=set)


def create_heightfield_gpu_state() -> HeightfieldGpuState:
    """compile the terrain shader and return empty gpu state"""
    terrain: TerrainShader = load_terrain_shader()
    material: Material = load_material_default()
    material.shader = terrain.shader
    return HeightfieldGpuState(
        terrain=terrain,
        material=material,
        composite=create_height_composite_cache(),
    )


def invalidate_tile_draw_cache(entry: HeightfieldGpuEntry) -> None:
    """drop cached tile matrices and instance batches"""
    entry.tile_matrices = None
    entry.tile_matrix_key = None
    entry.cached_levels = None
    entry.cached_masks = None
    entry.instance_batches.clear()


def drop_heightfield_entry(state: HeightfieldGpuState, group_id: str) -> None:
    """release one group's height texture"""
    entry: HeightfieldGpuEntry | None = state.entries.pop(group_id, None)
    if entry is not None:
        release_height_texture(entry.texture)
        release_r32_target(entry.result_target)
        entry.result_target = None
        invalidate_tile_draw_cache(entry)


def ensure_spacing_grids(state: HeightfieldGpuState, spacing: float) -> None:
    """pre-upload all lod × edge-mask grids for spacing once"""
    if spacing in state.warmed_spacings:
        return
    prewarm_lod_grids(state.grids, state.material, spacing)
    state.warmed_spacings.add(spacing)


def layout_matrix_key(
    layout: HeightfieldLayout,
) -> tuple[int, int, int, float, float, float]:
    """fingerprint for when flat tile matrices must be rebuilt"""
    return (
        layout.tiles_x,
        layout.tiles_z,
        layout.tile_verts,
        layout.spacing,
        layout.size_x,
        layout.size_z,
    )


def ensure_heightfield_gpu(
    state: HeightfieldGpuState,
    catalog: HeightmapCatalog,
    group_id: str,
    heightfield: Heightfield,
) -> HeightfieldGpuEntry:
    """create or refresh gpu resources when the recipe fingerprint changes"""
    layout: HeightfieldLayout = heightfield_layout(heightfield)
    spacing: float = layout.spacing
    key: HeightTextureKey = height_texture_key(heightfield)
    scale: float
    bias: float
    scale, bias = height_texture_uniforms(heightfield)
    existing: HeightfieldGpuEntry | None = state.entries.get(group_id)
    if existing is not None and existing.texture.key == key:
        matrix_key: tuple[int, int, int, float, float, float] = layout_matrix_key(
            layout
        )
        if existing.tile_matrix_key != matrix_key:
            invalidate_tile_draw_cache(existing)
        existing.layout = layout
        existing.spacing = spacing
        existing.texture.height_scale = scale
        existing.texture.height_bias = bias
        ensure_spacing_grids(state, spacing)
        return existing
    if existing is not None:
        release_height_texture(existing.texture)
        release_r32_target(existing.result_target)
        existing.result_target = None
        invalidate_tile_draw_cache(existing)
    texture: HeightTexture
    result_target: R32RenderTarget | None
    texture, result_target = make_height_texture(
        catalog, heightfield, state.composite
    )
    entry: HeightfieldGpuEntry = HeightfieldGpuEntry(
        texture=texture,
        layout=layout,
        spacing=spacing,
        result_target=result_target,
    )
    state.entries[group_id] = entry
    ensure_spacing_grids(state, spacing)
    return entry


def prune_heightfield_gpu(state: HeightfieldGpuState, scene: Scene) -> None:
    """drop entries whose groups are gone"""
    live: set[str] = {
        node_id
        for node_id, node in scene.nodes.items()
        if node.heightfield is not None
    }
    stale: list[str] = [
        group_id for group_id in state.entries if group_id not in live
    ]
    group_id: str
    for group_id in stale:
        drop_heightfield_entry(state, group_id)


def bind_heightfield_material(
    state: HeightfieldGpuState,
    entry: HeightfieldGpuEntry,
) -> None:
    """push height texture and albedo onto the shared material / uniforms"""
    terrain: TerrainShader = state.terrain
    material: Material = state.material
    layout: HeightfieldLayout = entry.layout
    material.maps[MaterialMapIndex.MATERIAL_MAP_HEIGHT].texture = (
        entry.texture.texture
    )
    material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].color = (
        heightfield_albedo
    )
    if terrain.loc_col_diffuse >= 0:
        set_vec4(
            terrain.shader,
            terrain.loc_col_diffuse,
            Vector4(
                float(heightfield_albedo.r) / 255.0,
                float(heightfield_albedo.g) / 255.0,
                float(heightfield_albedo.b) / 255.0,
                float(heightfield_albedo.a) / 255.0,
            ),
        )
    set_float(
        terrain.shader, terrain.loc_height_scale, entry.texture.height_scale
    )
    set_float(
        terrain.shader, terrain.loc_height_bias, entry.texture.height_bias
    )
    tex_w: int = max(int(entry.texture.texture.width), 1)
    tex_h: int = max(int(entry.texture.texture.height), 1)
    world_texel_x: float = layout.size_x / float(max(tex_w - 1, 1))
    world_texel_z: float = layout.size_z / float(max(tex_h - 1, 1))
    set_vec2(
        terrain.shader,
        terrain.loc_world_texel_size,
        Vector2(world_texel_x, world_texel_z),
    )


def tile_instance_matrix(
    local_x: float, local_z: float, uv_origin_x: float, uv_origin_z: float
) -> Matrix:
    """tile local translate with tileUvOrigin packed into m1/m2"""
    matrix: Matrix = matrix_translate(local_x, 0.0, local_z)
    matrix.m1 = uv_origin_x
    matrix.m2 = uv_origin_z
    return matrix


def ensure_tile_matrices(entry: HeightfieldGpuEntry) -> None:
    """build flat per-tile instance matrices when layout changes"""
    layout: HeightfieldLayout = entry.layout
    key: tuple[int, int, int, float, float, float] = layout_matrix_key(layout)
    if entry.tile_matrices is not None and entry.tile_matrix_key == key:
        return
    step: int = layout.tile_verts - 1
    cell: float = layout.spacing
    origin_x: float = -0.5 * layout.size_x
    origin_z: float = -0.5 * layout.size_z
    count: int = layout.tiles_x * layout.tiles_z
    buffer: Any = ffi.new("Matrix[]", count)
    tile_z: int
    tile_x: int
    for tile_z in range(layout.tiles_z):
        for tile_x in range(layout.tiles_x):
            origin_i: int = tile_x * step
            origin_k: int = tile_z * step
            uv_origin_x: float = (
                0.0
                if layout.res_x <= 1
                else float(origin_i) / float(layout.res_x - 1)
            )
            uv_origin_z: float = (
                0.0
                if layout.res_z <= 1
                else float(origin_k) / float(layout.res_z - 1)
            )
            local_x: float = origin_x + float(tile_x * step) * cell
            local_z: float = origin_z + float(tile_z * step) * cell
            index: int = tile_z * layout.tiles_x + tile_x
            buffer[index] = tile_instance_matrix(
                local_x, local_z, uv_origin_x, uv_origin_z
            )
    entry.tile_matrices = buffer
    entry.tile_matrix_key = key
    entry.cached_levels = None
    entry.cached_masks = None
    entry.instance_batches.clear()


def lod_maps_equal(
    levels: list[list[int]],
    masks: list[list[int]],
    cached_levels: list[list[int]] | None,
    cached_masks: list[list[int]] | None,
) -> bool:
    """true when lod and edge-mask grids match the previous frame"""
    if cached_levels is None or cached_masks is None:
        return False
    if len(levels) != len(cached_levels) or len(masks) != len(cached_masks):
        return False
    tile_z: int
    for tile_z in range(len(levels)):
        if levels[tile_z] != cached_levels[tile_z]:
            return False
        if masks[tile_z] != cached_masks[tile_z]:
            return False
    return True


def refill_instance_batches(
    entry: HeightfieldGpuEntry,
    levels: list[list[int]],
    masks: list[list[int]],
) -> None:
    """regroup cached tile matrices into reused (lod, mask) instance buffers"""
    layout: HeightfieldLayout = entry.layout
    tile_matrices: Any = entry.tile_matrices
    if tile_matrices is None:
        raise RuntimeError("tile matrices must be built before instance batches")
    membership: dict[tuple[int, int], list[int]] = {}
    tile_z: int
    tile_x: int
    for tile_z in range(layout.tiles_z):
        for tile_x in range(layout.tiles_x):
            key: tuple[int, int] = (levels[tile_z][tile_x], masks[tile_z][tile_x])
            indices: list[int] | None = membership.get(key)
            if indices is None:
                indices = []
                membership[key] = indices
            indices.append(tile_z * layout.tiles_x + tile_x)
    previous: dict[tuple[int, int], HeightfieldInstanceBatch] = entry.instance_batches
    next_batches: dict[tuple[int, int], HeightfieldInstanceBatch] = {}
    batch_key: tuple[int, int]
    tile_indices: list[int]
    for batch_key, tile_indices in membership.items():
        count: int = len(tile_indices)
        existing: HeightfieldInstanceBatch | None = previous.get(batch_key)
        buffer: Any
        if existing is not None and existing.count == count:
            buffer = existing.transforms
        else:
            buffer = ffi.new("Matrix[]", count)
        slot: int
        for slot in range(count):
            buffer[slot] = tile_matrices[tile_indices[slot]]
        next_batches[batch_key] = HeightfieldInstanceBatch(
            transforms=buffer,
            count=count,
            lod=batch_key[0],
            edge_mask=batch_key[1],
        )
    entry.instance_batches = next_batches
    entry.cached_levels = levels
    entry.cached_masks = masks


def draw_heightfield_tiles(
    state: HeightfieldGpuState,
    entry: HeightfieldGpuEntry,
    group_world: Matrix,
    camera_world: Vector3,
) -> None:
    """draw tiles instanced by (lod, edge_mask) with edge stitching"""
    layout: HeightfieldLayout = entry.layout
    terrain: TerrainShader = state.terrain
    step: int = layout.tile_verts - 1
    origin_x: float = -0.5 * layout.size_x
    origin_z: float = -0.5 * layout.size_z
    ensure_tile_matrices(entry)
    levels: list[list[int]]
    masks: list[list[int]]
    levels, masks = select_lods_world(
        layout,
        group_world,
        origin_x,
        origin_z,
        camera_world,
    )
    if not lod_maps_equal(
        levels, masks, entry.cached_levels, entry.cached_masks
    ):
        refill_instance_batches(entry, levels, masks)
    uv_scale_x: float = (
        0.0 if layout.res_x <= 1 else float(step) / float(layout.res_x - 1)
    )
    uv_scale_z: float = (
        0.0 if layout.res_z <= 1 else float(step) / float(layout.res_z - 1)
    )
    bind_heightfield_material(state, entry)
    set_vec2(
        terrain.shader,
        terrain.loc_tile_uv_scale,
        Vector2(uv_scale_x, uv_scale_z),
    )
    set_parent_transform(terrain, group_world)
    batch: HeightfieldInstanceBatch
    for batch in entry.instance_batches.values():
        if batch.count == 0:
            continue
        grid: PreparedMesh = ensure_lod_grid(
            state.grids,
            state.material,
            entry.spacing,
            batch.lod,
            batch.edge_mask,
        )
        draw_mesh_instanced(
            grid.mesh,
            state.material,
            ffi.cast("Matrix *", batch.transforms),
            batch.count,
        )


def select_lods_world(
    layout: HeightfieldLayout,
    group_world: Matrix,
    origin_x: float,
    origin_z: float,
    camera_world: Vector3,
) -> tuple[list[list[int]], list[list[int]]]:
    """lod from hybrid camera distance (max xz, |dy|) to each tile center"""
    tile_world: float = float(layout.tile_verts - 1) * layout.spacing
    levels: list[list[int]] = []
    tile_z: int
    tile_x: int
    for tile_z in range(layout.tiles_z):
        row: list[int] = []
        for tile_x in range(layout.tiles_x):
            local_cx: float = origin_x + (float(tile_x) + 0.5) * tile_world
            local_cz: float = origin_z + (float(tile_z) + 0.5) * tile_world
            center: Vector3 = vector3_transform(
                Vector3(local_cx, 0.0, local_cz), group_world
            )
            dx: float = center.x - camera_world.x
            dy: float = center.y - camera_world.y
            dz: float = center.z - camera_world.z
            distance: float = hybrid_lod_distance(dx, dy, dz)
            row.append(raw_lod_from_distance(distance, tile_world))
        levels.append(row)
    clamp_neighbor_lods(levels, layout.tiles_x, layout.tiles_z)
    masks: list[list[int]] = []
    for tile_z in range(layout.tiles_z):
        mask_row: list[int] = []
        for tile_x in range(layout.tiles_x):
            mask_row.append(
                edge_mask_for_tile(
                    levels, tile_x, tile_z, layout.tiles_x, layout.tiles_z
                )
            )
        masks.append(mask_row)
    return levels, masks


def draw_heightfields(
    state: HeightfieldGpuState,
    catalog: HeightmapCatalog,
    scene: Scene,
    cache: DrawCache,
    lighting: Lighting,
    view_pos: Vector3,
    *,
    wireframe: bool = False,
) -> None:
    """draw every heightfield group with gpu displace"""
    sync_draw_cache(scene, cache)
    prune_heightfield_gpu(state, scene)
    bind_terrain_lights(state.terrain, lighting.lights)
    set_vec3(state.terrain.shader, state.terrain.loc_view, view_pos)
    if wireframe:
        rl_enable_wire_mode()
    node_id: str
    node: Node
    for node_id, node in scene.nodes.items():
        heightfield: Heightfield | None = node.heightfield
        if heightfield is None:
            continue
        entry: HeightfieldGpuEntry = ensure_heightfield_gpu(
            state, catalog, node_id, heightfield
        )
        group_world: Matrix = fill_world_matrix(scene, cache, node_id)
        draw_heightfield_tiles(state, entry, group_world, view_pos)
    if wireframe:
        rl_disable_wire_mode()


def unload_heightfield_gpu(state: HeightfieldGpuState) -> None:
    """release every heightfield gpu resource

    material is not unloaded: UnloadMaterial would free the shared shader
    """
    group_id: str
    for group_id in list(state.entries):
        drop_heightfield_entry(state, group_id)
    release_all_lod_grids(state.grids)
    state.warmed_spacings.clear()
    unload_height_composite_cache(state.composite)
    unload_terrain_shader(state.terrain)
