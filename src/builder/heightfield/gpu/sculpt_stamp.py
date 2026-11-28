"""gpu vector sculpt stamp into an r32f layer fbo"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyray import (
    PixelFormat,
    ShaderUniformDataType,
    Texture,
    TextureFilter,
    TextureWrap,
    Vector2,
    Vector4,
    ffi,
    get_screen_height,
    get_screen_width,
    mem_free,
    rl_active_texture_slot,
    rl_disable_depth_test,
    rl_disable_framebuffer,
    rl_disable_scissor_test,
    rl_disable_shader,
    rl_draw_render_batch_active,
    rl_enable_depth_test,
    rl_enable_framebuffer,
    rl_enable_scissor_test,
    rl_enable_shader,
    rl_enable_texture,
    rl_load_draw_quad,
    rl_load_texture,
    rl_read_texture_pixels,
    rl_unload_texture,
    rl_viewport,
    set_shader_value,
    set_texture_filter,
    set_texture_wrap,
    update_texture,
)

from builder.heightfield.gpu.composite import (
    HeightCompositeCache,
    R32RenderTarget,
    create_r32_target,
    ensure_solid_textures,
    release_r32_target,
)
from builder.heightfield.gpu.composite_shader import set_int
from builder.heightfield.gpu.sculpt_stamp_shader import (
    BRUSH_ADD,
    BRUSH_FLATTEN,
    BRUSH_FLATTEN_DOWN,
    BRUSH_FLATTEN_UP,
    PASS_CLEAR,
    PASS_STAMP,
    SHAPE_CIRCLE_FILL,
    SHAPE_CIRCLE_STROKE,
    SHAPE_FILL,
    SHAPE_STROKE,
    SculptStampShader,
    load_sculpt_stamp_shader,
    unload_sculpt_stamp_shader,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapId,
    materialize_heightmap_samples,
)
from builder.heightfield.sparse_height import set_sample
from builder.heightfield.vector.stamp import polyline_segments
from builder.heightfield.vector.vector_types import VectorApplyMode
from builder.view.lighting import set_float

max_gpu_segments: int = 4096


@dataclass
class Float4Texture:
    """1d rgba32f strip kept alive for the gpu"""

    texture: Texture
    texture_id: int
    keep_alive: Any
    count: int


@dataclass
class SculptLayerGpu:
    """ping-pong r32 targets for one sculpt heightmap"""

    target: R32RenderTarget
    scratch: R32RenderTarget
    heightmap_name: str
    width: int
    height: int


@dataclass(frozen=True)
class GpuSculptShape:
    """one shape ready for a gpu stamp pass"""

    shape_mode: int
    brush_mode: int
    radius_metres: float
    offset: float
    intensity: float
    flatten_target: float
    circle_cx: float
    circle_cz: float
    circle_radius: float
    segments: list[tuple[tuple[float, float], tuple[float, float]]]
    poly_points: list[tuple[float, float]]
    # retained for cpu fallback when geometry exceeds the gpu segment budget
    kind: str
    apply_mode: VectorApplyMode
    brush_kind: str
    closed: bool
    points: list[tuple[float, float]]


def gpu_shape_fits(shape: GpuSculptShape) -> bool:
    """true when segment and poly uploads stay within the shader loop budget"""
    return (
        len(shape.segments) <= max_gpu_segments
        and len(shape.poly_points) <= max_gpu_segments
    )


def brush_mode_for_kind(brush_kind: str) -> int | None:
    """map brush kind to shader brushMode; None = skip (smooth / unknown)"""
    if brush_kind == "add":
        return BRUSH_ADD
    if brush_kind == "flatten":
        return BRUSH_FLATTEN
    if brush_kind == "flatten_absolute":
        return BRUSH_FLATTEN
    if brush_kind == "flatten_up":
        return BRUSH_FLATTEN_UP
    if brush_kind == "flatten_down":
        return BRUSH_FLATTEN_DOWN
    return None


def shape_mode_for(
    kind: str, apply_mode: VectorApplyMode, closed: bool, point_count: int
) -> int:
    """select stamp shapeMode for a vector shape"""
    if kind == "circle":
        if apply_mode is VectorApplyMode.stroke:
            return SHAPE_CIRCLE_STROKE
        return SHAPE_CIRCLE_FILL
    if apply_mode is VectorApplyMode.stroke:
        return SHAPE_STROKE
    if kind == "polyline" and (not closed or point_count < 3):
        return SHAPE_STROKE
    return SHAPE_FILL


def ensure_sculpt_stamp_shader(cache: HeightCompositeCache) -> SculptStampShader:
    """lazy-compile the sculpt stamp shader onto the composite cache"""
    if cache.sculpt_stamp is None:
        cache.sculpt_stamp = load_sculpt_stamp_shader()
    return cache.sculpt_stamp


def ensure_sculpt_layer(
    cache: HeightCompositeCache,
    heightmap_id: HeightmapId,
    width: int,
    height: int,
    buffer: HeightmapBuffer,
) -> SculptLayerGpu:
    """create or resize the sculpt layer fbo pair; seed from buffer when new"""
    existing: SculptLayerGpu | None = cache.sculpt_layers.get(heightmap_id.name)
    if (
        existing is not None
        and existing.width == width
        and existing.height == height
    ):
        return existing
    if existing is not None:
        release_r32_target(existing.target)
        release_r32_target(existing.scratch)
        del cache.sculpt_layers[heightmap_id.name]
    target: R32RenderTarget = create_r32_target(width, height)
    scratch: R32RenderTarget = create_r32_target(width, height)
    data: Any = ffi.new("float[]", materialize_heightmap_samples(buffer))
    update_texture(target.texture, data)
    layer = SculptLayerGpu(
        target=target,
        scratch=scratch,
        heightmap_name=heightmap_id.name,
        width=width,
        height=height,
    )
    cache.sculpt_layers[heightmap_id.name] = layer
    cache.sculpt_seed_keepalives[heightmap_id.name] = data
    return layer


def upload_float4_strip(
    values: list[tuple[float, float, float, float]],
) -> Float4Texture:
    """upload an nx1 rgba32f texture from float4 rows"""
    count: int = max(1, len(values))
    flat: list[float] = []
    if len(values) == 0:
        flat = [0.0, 0.0, 0.0, 0.0]
    else:
        row: tuple[float, float, float, float]
        for row in values:
            flat.extend(row)
    data: Any = ffi.new("float[]", flat)
    texture_id: int = rl_load_texture(
        data,
        count,
        1,
        int(PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32G32B32A32),
        1,
    )
    if texture_id == 0:
        raise RuntimeError("failed to allocate float4 strip texture")
    texture_cdata: Any = ffi.new("Texture *")
    texture_cdata.id = texture_id
    texture_cdata.width = count
    texture_cdata.height = 1
    texture_cdata.mipmaps = 1
    texture_cdata.format = int(
        PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32G32B32A32
    )
    texture: Texture = texture_cdata[0]
    set_texture_filter(texture, TextureFilter.TEXTURE_FILTER_POINT)
    set_texture_wrap(texture, TextureWrap.TEXTURE_WRAP_CLAMP)
    return Float4Texture(
        texture=texture,
        texture_id=texture_id,
        keep_alive=(data, texture_cdata),
        count=count,
    )


def release_float4_strip(strip: Float4Texture | None) -> None:
    """free a float4 strip texture"""
    if strip is None:
        return
    rl_unload_texture(strip.texture_id)


def segments_to_float4(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[float, float, float, float]]:
    """pack segment endpoints for the segment strip"""
    if len(segments) > max_gpu_segments:
        raise ValueError(
            f"gpu sculpt segments {len(segments)} exceed max_gpu_segments"
        )
    out: list[tuple[float, float, float, float]] = []
    segment: tuple[tuple[float, float], tuple[float, float]]
    for segment in segments:
        (ax, az), (bx, bz) = segment
        out.append((ax, az, bx, bz))
    return out


def points_to_float4(
    points: list[tuple[float, float]],
) -> list[tuple[float, float, float, float]]:
    """pack polygon vertices for the poly strip"""
    if len(points) > max_gpu_segments:
        raise ValueError(
            f"gpu sculpt poly points {len(points)} exceed max_gpu_segments"
        )
    out: list[tuple[float, float, float, float]] = []
    point: tuple[float, float]
    for point in points:
        out.append((point[0], point[1], 0.0, 0.0))
    return out


def dirty_uv_rect(
    x0: int, x1: int, y0: int, y1: int, width: int, height: int
) -> Vector4:
    """inclusive-exclusive texel range as uv box"""
    if width < 1 or height < 1:
        return Vector4(0.0, 0.0, 1.0, 1.0)
    return Vector4(
        float(x0) / float(width),
        float(y0) / float(height),
        float(x1) / float(width),
        float(y1) / float(height),
    )


def swap_sculpt_targets(layer: SculptLayerGpu) -> None:
    """swap authoritative and scratch targets after a ping-pong pass"""
    previous: R32RenderTarget = layer.target
    layer.target = layer.scratch
    layer.scratch = previous


def draw_sculpt_pass(
    stamp: SculptStampShader,
    dest: R32RenderTarget,
    layer_tex: Texture,
    below_tex: Texture,
    segment_tex: Texture,
    poly_tex: Texture,
    *,
    pass_mode: int,
    shape_mode: int,
    brush_mode: int,
    segment_count: int,
    poly_count: int,
    dirty: Vector4,
    size_x: float,
    size_z: float,
    radius_metres: float,
    offset: float,
    intensity: float,
    flatten_target: float,
    circle: Vector4,
) -> None:
    """one fullscreen sculpt clear/stamp into dest"""
    rl_draw_render_batch_active()
    rl_disable_scissor_test()
    rl_enable_framebuffer(dest.fbo_id)
    rl_viewport(0, 0, dest.width, dest.height)
    rl_disable_depth_test()
    rl_enable_shader(stamp.shader.id)
    set_int(stamp.shader, stamp.loc_layer, 0)
    set_int(stamp.shader, stamp.loc_below, 1)
    set_int(stamp.shader, stamp.loc_segment, 2)
    set_int(stamp.shader, stamp.loc_poly, 3)
    set_int(stamp.shader, stamp.loc_pass_mode, pass_mode)
    set_int(stamp.shader, stamp.loc_shape_mode, shape_mode)
    set_int(stamp.shader, stamp.loc_brush_mode, brush_mode)
    set_int(stamp.shader, stamp.loc_segment_count, segment_count)
    set_int(stamp.shader, stamp.loc_poly_count, poly_count)
    set_shader_value(
        stamp.shader,
        stamp.loc_dirty_uv,
        dirty,
        ShaderUniformDataType.SHADER_UNIFORM_VEC4,
    )
    set_shader_value(
        stamp.shader,
        stamp.loc_field_size,
        Vector2(size_x, size_z),
        ShaderUniformDataType.SHADER_UNIFORM_VEC2,
    )
    set_float(stamp.shader, stamp.loc_radius, radius_metres)
    set_float(stamp.shader, stamp.loc_offset, offset)
    set_float(stamp.shader, stamp.loc_intensity, intensity)
    set_float(stamp.shader, stamp.loc_flatten_target, flatten_target)
    set_shader_value(
        stamp.shader,
        stamp.loc_circle,
        circle,
        ShaderUniformDataType.SHADER_UNIFORM_VEC4,
    )
    rl_active_texture_slot(0)
    rl_enable_texture(layer_tex.id)
    rl_active_texture_slot(1)
    rl_enable_texture(below_tex.id)
    rl_active_texture_slot(2)
    rl_enable_texture(segment_tex.id)
    rl_active_texture_slot(3)
    rl_enable_texture(poly_tex.id)
    rl_load_draw_quad()
    rl_disable_shader()
    rl_disable_framebuffer()
    rl_enable_depth_test()
    rl_enable_scissor_test()
    rl_viewport(0, 0, get_screen_width(), get_screen_height())
    rl_draw_render_batch_active()


def readback_r32_region(
    texture_id: int,
    width: int,
    height: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    buffer: HeightmapBuffer,
) -> None:
    """copy a dirty rect from an r32 texture into a sparse height raster"""
    ptr: Any = rl_read_texture_pixels(
        texture_id,
        width,
        height,
        int(PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32),
    )
    if ptr == ffi.NULL:
        return
    floats: Any = ffi.cast("float *", ptr)
    rx0: int = max(0, min(width, x0))
    rx1: int = max(rx0, min(width, x1))
    ry0: int = max(0, min(height, y0))
    ry1: int = max(ry0, min(height, y1))
    y: int
    for y in range(ry0, ry1):
        row: int = y * width
        x: int
        for x in range(rx0, rx1):
            set_sample(buffer, x, y, float(floats[row + x]))
    mem_free(ptr)


def gpu_stamp_sculpt_shapes(
    cache: HeightCompositeCache,
    heightmap_id: HeightmapId,
    buffer: HeightmapBuffer,
    *,
    size_x: float,
    size_z: float,
    clear_x0: int,
    clear_x1: int,
    clear_y0: int,
    clear_y1: int,
    shapes: list[GpuSculptShape],
    below_texture: Texture,
) -> None:
    """clear dirty + stamp shapes into the sculpt layer fbo; read back dirty"""
    ensure_solid_textures(cache)
    assert cache.solid_zero is not None
    stamp: SculptStampShader = ensure_sculpt_stamp_shader(cache)
    layer: SculptLayerGpu = ensure_sculpt_layer(
        cache, heightmap_id, buffer.width, buffer.height, buffer
    )
    dirty: Vector4 = dirty_uv_rect(
        clear_x0,
        clear_x1,
        clear_y0,
        clear_y1,
        buffer.width,
        buffer.height,
    )
    empty_strip: Float4Texture = upload_float4_strip([])
    try:
        draw_sculpt_pass(
            stamp,
            layer.scratch,
            layer.target.texture,
            below_texture,
            empty_strip.texture,
            empty_strip.texture,
            pass_mode=PASS_CLEAR,
            shape_mode=SHAPE_STROKE,
            brush_mode=BRUSH_ADD,
            segment_count=0,
            poly_count=0,
            dirty=dirty,
            size_x=size_x,
            size_z=size_z,
            radius_metres=0.0,
            offset=0.0,
            intensity=0.0,
            flatten_target=0.0,
            circle=Vector4(0.0, 0.0, 0.0, 0.0),
        )
        swap_sculpt_targets(layer)

        shape: GpuSculptShape
        for shape in shapes:
            seg_values = segments_to_float4(shape.segments)
            poly_values = points_to_float4(shape.poly_points)
            seg_tex: Float4Texture = (
                upload_float4_strip(seg_values)
                if len(seg_values) > 0
                else empty_strip
            )
            poly_tex: Float4Texture = (
                upload_float4_strip(poly_values)
                if len(poly_values) > 0
                else empty_strip
            )
            try:
                draw_sculpt_pass(
                    stamp,
                    layer.scratch,
                    layer.target.texture,
                    below_texture,
                    seg_tex.texture,
                    poly_tex.texture,
                    pass_mode=PASS_STAMP,
                    shape_mode=shape.shape_mode,
                    brush_mode=shape.brush_mode,
                    segment_count=len(seg_values),
                    poly_count=len(poly_values),
                    dirty=dirty,
                    size_x=size_x,
                    size_z=size_z,
                    radius_metres=shape.radius_metres,
                    offset=shape.offset,
                    intensity=shape.intensity,
                    flatten_target=shape.flatten_target,
                    circle=Vector4(
                        shape.circle_cx,
                        shape.circle_cz,
                        shape.circle_radius,
                        0.0,
                    ),
                )
                swap_sculpt_targets(layer)
            finally:
                if seg_tex is not empty_strip:
                    release_float4_strip(seg_tex)
                if poly_tex is not empty_strip:
                    release_float4_strip(poly_tex)
    finally:
        release_float4_strip(empty_strip)

    # mark live fbo for composite to bind directly
    cache.sculpt_live[heightmap_id.name] = layer.target
    readback_r32_region(
        layer.target.texture_id,
        buffer.width,
        buffer.height,
        clear_x0,
        clear_x1,
        clear_y0,
        clear_y1,
        buffer,
    )


def build_gpu_sculpt_shape(
    *,
    kind: str,
    apply_mode: VectorApplyMode,
    brush_kind: str,
    radius_metres: float,
    offset: float,
    intensity: float,
    flatten_target: float,
    points: list[tuple[float, float]],
    closed: bool,
    circle_radius: float,
) -> GpuSculptShape | None:
    """build a gpu stamp descriptor; None when brush is unsupported"""
    brush_mode: int | None = brush_mode_for_kind(brush_kind)
    if brush_mode is None:
        return None
    if len(points) == 0:
        return None
    shape_mode: int = shape_mode_for(
        kind, apply_mode, closed, len(points)
    )
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    poly_points: list[tuple[float, float]] = []
    cx: float = points[0][0]
    cz: float = points[0][1]
    if shape_mode in (SHAPE_CIRCLE_STROKE, SHAPE_CIRCLE_FILL):
        pass
    else:
        segments = polyline_segments(points, closed)
        if shape_mode == SHAPE_FILL:
            poly_points = list(points)
        if len(segments) == 0 and len(points) == 1:
            segments = [((points[0][0], points[0][1]), (points[0][0], points[0][1]))]
    return GpuSculptShape(
        shape_mode=shape_mode,
        brush_mode=brush_mode,
        radius_metres=radius_metres,
        offset=offset,
        intensity=intensity,
        flatten_target=flatten_target,
        circle_cx=cx,
        circle_cz=cz,
        circle_radius=circle_radius,
        segments=segments,
        poly_points=poly_points,
        kind=kind,
        apply_mode=apply_mode,
        brush_kind=brush_kind,
        closed=closed,
        points=list(points),
    )


def release_sculpt_layer_cache(cache: HeightCompositeCache) -> None:
    """free all sculpt layer fbos and the stamp shader"""
    name: str
    for name in list(cache.sculpt_layers.keys()):
        layer: SculptLayerGpu = cache.sculpt_layers.pop(name)
        release_r32_target(layer.target)
        release_r32_target(layer.scratch)
    cache.sculpt_live.clear()
    cache.sculpt_seed_keepalives.clear()
    if cache.sculpt_stamp is not None:
        unload_sculpt_stamp_shader(cache.sculpt_stamp)
        cache.sculpt_stamp = None
