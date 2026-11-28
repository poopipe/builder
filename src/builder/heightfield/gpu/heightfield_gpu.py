"""gpu heightfield state: shared lod grids, height textures, tile draw"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyray import (
    BlendMode,
    Color,
    Material,
    MaterialMapIndex,
    Matrix,
    Mesh,
    Texture,
    Vector2,
    Vector3,
    Vector4,
    begin_blend_mode,
    draw_mesh_instanced,
    end_blend_mode,
    ffi,
    get_mesh_bounding_box,
    is_texture_valid,
    load_material_default,
    load_texture,
    matrix_translate,
    rl_disable_wire_mode,
    rl_enable_wire_mode,
    unload_texture,
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
    build_flat_grid_mesh,
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
    set_int,
    set_parent_transform,
    set_vec2,
    set_vec4,
    unload_terrain_shader,
)
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightEditLayer,
    HeightLayerSource,
    Heightfield,
    HeightfieldLayout,
    PaintHeightSource,
    RoadSectionsHeightSource,
    SculptHeightSource,
    VectorHeightSource,
    VectorRoadHeightSource,
    VectorSculptHeightSource,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
)
from builder.heightfield.sea_plane import (
    build_sea_visibility,
    refresh_sea_visibility,
)
from builder.heightfield.vector.stamp import texel_range_to_world_aabb
from builder.scene.scene import Scene
from builder.scene.scene_types import Node
from builder.scene.visibility import node_effectively_visible
from builder.ui.color_palette import palette_color
from builder.view.draw_cache import DrawCache, fill_world_matrix, sync_draw_cache
from builder.view.lighting import Lighting, set_vec3
from builder.view.line_shader import LineHeightDrape
from builder.view.prepare_mesh import PreparedMesh, release_prepared_mesh

heightfield_albedo: Color = Color(210, 210, 210, 255)
sea_albedo: Color = Color(40, 110, 180, 110)


@dataclass
class HeightfieldInstanceBatch:
    """reused instance buffer for one (lod, edge_mask) draw"""

    transforms: Any
    count: int
    lod: int
    edge_mask: int


@dataclass
class BelowHeightTexture:
    """cached gpu composite of the stack strictly below one edit layer"""

    texture: HeightTexture
    result_target: R32RenderTarget | None
    key: HeightTextureKey


@dataclass
class HeightfieldGpuEntry:
    """cached gpu resources for one heightfield group"""

    texture: HeightTexture
    layout: HeightfieldLayout
    spacing: float
    # owned composite result when edit layers are active; None for raw base
    result_target: R32RenderTarget | None = None
    albedo_texture: Texture | None = None
    albedo_path: str | None = None
    # flat tile-local instance matrices (layout-sized); None until built
    tile_matrices: Any | None = None
    tile_matrix_key: tuple[int, int, int, float, float, float] | None = None
    cached_levels: list[list[int]] | None = None
    cached_masks: list[list[int]] | None = None
    instance_batches: dict[tuple[int, int], HeightfieldInstanceBatch] = field(
        default_factory=dict
    )
    # below_layer_index -> underlay composite for shadow draping
    below_textures: dict[int, BelowHeightTexture] = field(default_factory=dict)
    sea_visibility: list[bool] | None = None
    sea_visibility_key: HeightTextureKey | None = None
    sea_transforms: Any | None = None
    sea_count: int = 0
    sea_matrix_key: tuple[int, int, int, float, float, float] | None = None


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
    # tile_world_size -> flat sea section mesh (lighting material)
    sea_meshes: dict[float, PreparedMesh] = field(default_factory=dict)
    sea_material: Material | None = None
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


def invalidate_sea_cache(entry: HeightfieldGpuEntry) -> None:
    """drop sea visibility and instance buffer"""
    entry.sea_visibility = None
    entry.sea_visibility_key = None
    entry.sea_transforms = None
    entry.sea_count = 0
    entry.sea_matrix_key = None


def invalidate_tile_draw_cache(entry: HeightfieldGpuEntry) -> None:
    """drop cached tile matrices and instance batches"""
    entry.tile_matrices = None
    entry.tile_matrix_key = None
    entry.cached_levels = None
    entry.cached_masks = None
    entry.instance_batches.clear()
    invalidate_sea_cache(entry)


def release_albedo_texture(entry: HeightfieldGpuEntry) -> None:
    """unload a group's albedo texture if owned"""
    if entry.albedo_texture is not None:
        unload_texture(entry.albedo_texture)
    entry.albedo_texture = None
    entry.albedo_path = None


def ensure_albedo_texture(
    entry: HeightfieldGpuEntry, heightfield: Heightfield
) -> None:
    """load or drop the albedo map when the recipe path changes"""
    path: str | None = heightfield.albedo_path
    if path == entry.albedo_path and (
        path is None or entry.albedo_texture is not None
    ):
        return
    release_albedo_texture(entry)
    if path is None:
        return
    texture: Texture = load_texture(path)
    if not is_texture_valid(texture):
        # keep path unset so the next ensure retries after a fix
        return
    entry.albedo_texture = texture
    entry.albedo_path = path


def release_below_textures(entry: HeightfieldGpuEntry) -> None:
    """drop cached below-layer composites"""
    cached: BelowHeightTexture
    for cached in entry.below_textures.values():
        release_height_texture(cached.texture)
        release_r32_target(cached.result_target)
    entry.below_textures.clear()


def drop_heightfield_entry(state: HeightfieldGpuState, group_id: str) -> None:
    """release one group's height texture"""
    entry: HeightfieldGpuEntry | None = state.entries.pop(group_id, None)
    if entry is not None:
        release_height_texture(entry.texture)
        release_r32_target(entry.result_target)
        entry.result_target = None
        release_below_textures(entry)
        release_albedo_texture(entry)
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


def source_heightmap_ids(source: HeightLayerSource | None) -> list[HeightmapId]:
    """catalog ids referenced by one height or mask source"""
    match source:
        case None:
            return []
        case (
            CatalogHeightSource(heightmap_id=heightmap_id)
            | PaintHeightSource(heightmap_id=heightmap_id)
            | VectorHeightSource(heightmap_id=heightmap_id)
            | VectorSculptHeightSource(heightmap_id=heightmap_id)
        ):
            return [heightmap_id]
        case SculptHeightSource(heightmap_id=heightmap_id, coverage_id=coverage_id):
            if coverage_id is None:
                return [heightmap_id]
            return [heightmap_id, coverage_id]
        case VectorRoadHeightSource(
            heightmap_id=heightmap_id, coverage_id=coverage_id
        ) | RoadSectionsHeightSource(
            heightmap_id=heightmap_id, coverage_id=coverage_id
        ):
            return [heightmap_id, coverage_id]
        case _:
            return []


def heightfield_heightmap_ids(heightfield: Heightfield) -> list[HeightmapId]:
    """all catalog heightmap ids that can dirty this heightfield stack"""
    ids: list[HeightmapId] = []
    if heightfield.base is not None:
        ids.append(heightfield.base)
    layer: HeightEditLayer
    for layer in heightfield.edit_layers:
        ids.extend(source_heightmap_ids(layer.height))
        ids.extend(source_heightmap_ids(layer.mask))
    return ids


def pending_dirty_world_aabb(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layout: HeightfieldLayout,
) -> tuple[float, float, float, float] | None:
    """union pending paint dirty rects; None means full-field / unknown"""
    union: tuple[float, float, float, float] | None = None
    saw_pending: bool = False
    heightmap_id: HeightmapId
    for heightmap_id in heightfield_heightmap_ids(heightfield):
        if heightmap_id not in catalog.paint_dirty_rects:
            continue
        saw_pending = True
        dirty_rect: tuple[int, int, int, int] | None = catalog.paint_dirty_rects[
            heightmap_id
        ]
        if dirty_rect is None:
            return None
        buffer: HeightmapBuffer | None = catalog.buffers.get(heightmap_id)
        if buffer is None:
            return None
        world: tuple[float, float, float, float] = texel_range_to_world_aabb(
            dirty_rect[0],
            dirty_rect[1],
            dirty_rect[2],
            dirty_rect[3],
            layout.size_x,
            layout.size_z,
            buffer.width,
            buffer.height,
        )
        if union is None:
            union = world
        else:
            union = (
                min(union[0], world[0]),
                max(union[1], world[1]),
                min(union[2], world[2]),
                max(union[3], world[3]),
            )
    if not saw_pending:
        return None
    return union


def refresh_entry_sea_visibility(
    entry: HeightfieldGpuEntry,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    dirty_world_aabb: tuple[float, float, float, float] | None,
) -> None:
    """update sea tile flags after a height texture refresh"""
    if entry.sea_visibility is None:
        entry.sea_visibility = build_sea_visibility(
            catalog, heightfield, entry.layout
        )
    else:
        entry.sea_visibility = refresh_sea_visibility(
            catalog,
            heightfield,
            entry.layout,
            entry.sea_visibility,
            dirty_world_aabb,
        )
    entry.sea_visibility_key = entry.texture.key
    entry.sea_transforms = None
    entry.sea_count = 0
    entry.sea_matrix_key = None


def ensure_heightfield_gpu(
    state: HeightfieldGpuState,
    catalog: HeightmapCatalog,
    group_id: str,
    heightfield: Heightfield,
) -> HeightfieldGpuEntry:
    """create or refresh gpu resources when the recipe fingerprint changes"""
    layout: HeightfieldLayout = heightfield_layout(heightfield)
    spacing: float = layout.spacing
    key: HeightTextureKey = height_texture_key(catalog, heightfield)
    scale: float
    bias: float
    scale, bias = height_texture_uniforms(catalog, heightfield)
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
        ensure_albedo_texture(existing, heightfield)
        ensure_spacing_grids(state, spacing)
        return existing
    if existing is not None:
        layout_changed: bool = layout_matrix_key(existing.layout) != layout_matrix_key(
            layout
        )
        dirty_world: tuple[float, float, float, float] | None = (
            pending_dirty_world_aabb(catalog, heightfield, layout)
        )
        old_owned: bool = existing.texture.owns_texture
        old_texture: HeightTexture = existing.texture
        texture: HeightTexture
        result_target: R32RenderTarget | None
        texture, result_target = make_height_texture(
            catalog,
            heightfield,
            state.composite,
            reusable_result=existing.result_target,
        )
        if old_owned:
            release_height_texture(old_texture)
        existing.texture = texture
        existing.result_target = result_target
        existing.layout = layout
        existing.spacing = spacing
        existing.texture.height_scale = scale
        existing.texture.height_bias = bias
        if layout_changed:
            invalidate_tile_draw_cache(existing)
        else:
            refresh_entry_sea_visibility(
                existing, catalog, heightfield, dirty_world
            )
        ensure_albedo_texture(existing, heightfield)
        ensure_spacing_grids(state, spacing)
        return existing
    texture, result_target = make_height_texture(
        catalog, heightfield, state.composite
    )
    entry = HeightfieldGpuEntry(
        texture=texture,
        layout=layout,
        spacing=spacing,
        result_target=result_target,
    )
    entry.texture.height_scale = scale
    entry.texture.height_bias = bias
    ensure_albedo_texture(entry, heightfield)
    state.entries[group_id] = entry
    ensure_spacing_grids(state, spacing)
    return entry


def ensure_height_texture_below(
    state: HeightfieldGpuState,
    catalog: HeightmapCatalog,
    group_id: str,
    heightfield: Heightfield,
    below_layer_index: int,
) -> HeightTexture:
    """gpu composite of the stack strictly below below_layer_index"""
    entry: HeightfieldGpuEntry = ensure_heightfield_gpu(
        state, catalog, group_id, heightfield
    )
    key: HeightTextureKey = height_texture_key(
        catalog, heightfield, below_layer_index=below_layer_index
    )
    cached: BelowHeightTexture | None = entry.below_textures.get(
        below_layer_index
    )
    if cached is not None and cached.key == key:
        scale: float
        bias: float
        scale, bias = height_texture_uniforms(
            catalog, heightfield, below_layer_index=below_layer_index
        )
        cached.texture.height_scale = scale
        cached.texture.height_bias = bias
        return cached.texture
    reusable_result: R32RenderTarget | None = None
    if cached is not None:
        if cached.texture.owns_texture:
            release_height_texture(cached.texture)
        reusable_result = cached.result_target
        del entry.below_textures[below_layer_index]
    texture: HeightTexture
    result_target: R32RenderTarget | None
    texture, result_target = make_height_texture(
        catalog,
        heightfield,
        state.composite,
        below_layer_index=below_layer_index,
        reusable_result=reusable_result,
    )
    scale, bias = height_texture_uniforms(
        catalog, heightfield, below_layer_index=below_layer_index
    )
    texture.height_scale = scale
    texture.height_bias = bias
    entry.below_textures[below_layer_index] = BelowHeightTexture(
        texture=texture,
        result_target=result_target,
        key=key,
    )
    return texture


def line_height_drape_for_overlay(
    state: HeightfieldGpuState,
    catalog: HeightmapCatalog,
    group_id: str,
    heightfield: Heightfield,
    below_layer_index: int | None,
) -> LineHeightDrape:
    """heightMap binding for overlay shadows (full stack or below a layer)"""
    entry: HeightfieldGpuEntry = ensure_heightfield_gpu(
        state, catalog, group_id, heightfield
    )
    texture: HeightTexture
    if below_layer_index is None:
        texture = entry.texture
    else:
        texture = ensure_height_texture_below(
            state, catalog, group_id, heightfield, below_layer_index
        )
    return LineHeightDrape(
        texture=texture.texture,
        size_x=entry.layout.size_x,
        size_z=entry.layout.size_z,
        height_scale=texture.height_scale,
        height_bias=texture.height_bias,
    )


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
    *,
    albedo_tint: Color | None = None,
) -> None:
    """push height texture and albedo onto the shared material / uniforms

    albedo_tint applies only when there is no albedo texture
    """
    terrain: TerrainShader = state.terrain
    material: Material = state.material
    layout: HeightfieldLayout = entry.layout
    material.maps[MaterialMapIndex.MATERIAL_MAP_HEIGHT].texture = (
        entry.texture.texture
    )
    tint: Color = heightfield_albedo
    if entry.albedo_texture is None and albedo_tint is not None:
        tint = albedo_tint
    material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].color = tint
    if entry.albedo_texture is not None:
        material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].texture = (
            entry.albedo_texture
        )
        set_int(terrain.shader, terrain.loc_use_albedo_map, 1)
    else:
        set_int(terrain.shader, terrain.loc_use_albedo_map, 0)
    if terrain.loc_col_diffuse >= 0:
        set_vec4(
            terrain.shader,
            terrain.loc_col_diffuse,
            Vector4(
                float(tint.r) / 255.0,
                float(tint.g) / 255.0,
                float(tint.b) / 255.0,
                float(tint.a) / 255.0,
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
    # layout change invalidates sea instance buffer; visibility may still match
    entry.sea_transforms = None
    entry.sea_count = 0
    entry.sea_matrix_key = None


def ensure_sea_material(state: HeightfieldGpuState, lighting: Lighting) -> Material:
    """shared lighting material tinted for the sea plane"""
    if state.sea_material is not None:
        material: Material = state.sea_material
        material.shader = lighting.shader
        material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].color = sea_albedo
        return material
    material = load_material_default()
    material.shader = lighting.shader
    material.maps[MaterialMapIndex.MATERIAL_MAP_ALBEDO].color = sea_albedo
    state.sea_material = material
    return material


def ensure_sea_section_mesh(
    state: HeightfieldGpuState, lighting: Lighting, tile_world: float
) -> PreparedMesh:
    """flat 2x2 xz quad spanning one heightfield tile at y = 0"""
    existing: PreparedMesh | None = state.sea_meshes.get(tile_world)
    if existing is not None:
        existing.material.shader = lighting.shader
        return existing
    material: Material = ensure_sea_material(state, lighting)
    mesh: Mesh
    keep_alive: tuple[Any, ...]
    mesh, keep_alive = build_flat_grid_mesh(2, tile_world, 0)
    prepared: PreparedMesh = PreparedMesh(
        mesh=mesh,
        material=material,
        local_bounds=get_mesh_bounding_box(mesh),
        keep_alive=keep_alive,
    )
    state.sea_meshes[tile_world] = prepared
    return prepared


def ensure_sea_instances(
    entry: HeightfieldGpuEntry,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
) -> None:
    """rebuild sea visibility and instance matrices when heights or layout change"""
    height_key: HeightTextureKey = entry.texture.key
    if (
        entry.sea_visibility is None
        or entry.sea_visibility_key != height_key
    ):
        entry.sea_visibility = build_sea_visibility(
            catalog, heightfield, entry.layout
        )
        entry.sea_visibility_key = height_key
        entry.sea_transforms = None
        entry.sea_count = 0
        entry.sea_matrix_key = None
    layout: HeightfieldLayout = entry.layout
    matrix_key: tuple[int, int, int, float, float, float] = layout_matrix_key(
        layout
    )
    if entry.sea_transforms is not None and entry.sea_matrix_key == matrix_key:
        return
    visibility: list[bool] = entry.sea_visibility or []
    count: int = sum(1 for flag in visibility if flag)
    if count == 0:
        entry.sea_transforms = None
        entry.sea_count = 0
        entry.sea_matrix_key = matrix_key
        return
    tile_world: float = float(layout.tile_verts - 1) * layout.spacing
    origin_x: float = -0.5 * layout.size_x
    origin_z: float = -0.5 * layout.size_z
    buffer: Any = ffi.new("Matrix[]", count)
    slot: int = 0
    tile_z: int
    tile_x: int
    for tile_z in range(layout.tiles_z):
        for tile_x in range(layout.tiles_x):
            index: int = tile_z * layout.tiles_x + tile_x
            if index >= len(visibility) or not visibility[index]:
                continue
            local_x: float = origin_x + float(tile_x) * tile_world
            local_z: float = origin_z + float(tile_z) * tile_world
            buffer[slot] = matrix_translate(local_x, 0.0, local_z)
            slot += 1
    entry.sea_transforms = buffer
    entry.sea_count = count
    entry.sea_matrix_key = matrix_key


def draw_sea_tiles(
    state: HeightfieldGpuState,
    entry: HeightfieldGpuEntry,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    group_world: Matrix,
    lighting: Lighting,
) -> None:
    """draw visible sea sections as instanced flat quads at local y = 0"""
    ensure_sea_instances(entry, catalog, heightfield)
    if entry.sea_count <= 0 or entry.sea_transforms is None:
        return
    tile_world: float = float(entry.layout.tile_verts - 1) * entry.layout.spacing
    section: PreparedMesh = ensure_sea_section_mesh(state, lighting, tile_world)
    lighting.set_parent_transform(group_world)
    begin_blend_mode(BlendMode.BLEND_ALPHA)
    draw_mesh_instanced(
        section.mesh,
        section.material,
        ffi.cast("Matrix *", entry.sea_transforms),
        entry.sea_count,
    )
    end_blend_mode()


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
    *,
    albedo_tint: Color | None = None,
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
    bind_heightfield_material(state, entry, albedo_tint=albedo_tint)
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
    show_sea: bool = True,
) -> None:
    """draw every heightfield group with gpu displace"""
    sync_draw_cache(scene, cache)
    prune_heightfield_gpu(state, scene)
    bind_terrain_lights(state.terrain, lighting.lights)
    set_vec3(state.terrain.shader, state.terrain.loc_view, view_pos)
    # opaque terrain first; sea after so alpha blends over submerged ground
    sea_pass: list[tuple[HeightfieldGpuEntry, Heightfield, Matrix]] = []
    if wireframe:
        rl_enable_wire_mode()
    node_id: str
    node: Node
    for node_id, node in scene.nodes.items():
        heightfield: Heightfield | None = node.heightfield
        if heightfield is None:
            continue
        if not node_effectively_visible(scene.nodes, node_id):
            continue
        entry: HeightfieldGpuEntry = ensure_heightfield_gpu(
            state, catalog, node_id, heightfield
        )
        group_world: Matrix = fill_world_matrix(scene, cache, node_id)
        draw_heightfield_tiles(
            state,
            entry,
            group_world,
            view_pos,
            albedo_tint=palette_color(node.color_index),
        )
        if show_sea:
            sea_pass.append((entry, heightfield, group_world))
    if wireframe:
        rl_disable_wire_mode()
    if show_sea:
        entry_sea: HeightfieldGpuEntry
        heightfield_sea: Heightfield
        group_world_sea: Matrix
        for entry_sea, heightfield_sea, group_world_sea in sea_pass:
            draw_sea_tiles(
                state,
                entry_sea,
                catalog,
                heightfield_sea,
                group_world_sea,
                lighting,
            )


def release_sea_meshes(state: HeightfieldGpuState) -> None:
    """unload shared sea section meshes; sea material is not unloaded"""
    tile_world: float
    for tile_world in list(state.sea_meshes):
        release_prepared_mesh(state.sea_meshes.pop(tile_world))
    state.sea_material = None


def unload_heightfield_gpu(state: HeightfieldGpuState) -> None:
    """release every heightfield gpu resource

    material is not unloaded: UnloadMaterial would free the shared shader
    """
    group_id: str
    for group_id in list(state.entries):
        drop_heightfield_entry(state, group_id)
    release_all_lod_grids(state.grids)
    release_sea_meshes(state)
    state.warmed_spacings.clear()
    unload_height_composite_cache(state.composite)
    unload_terrain_shader(state.terrain)
