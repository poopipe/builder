"""gpu r32f heightfield composite (base + masked edit layers)"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyray import (
    PixelFormat,
    RL_ATTACHMENT_COLOR_CHANNEL0,
    RL_ATTACHMENT_TEXTURE2D,
    ShaderUniformDataType,
    Texture,
    TextureFilter,
    TextureWrap,
    Vector2,
    ffi,
    get_screen_height,
    get_screen_width,
    set_shader_value,
    rl_active_texture_slot,
    rl_clear_color,
    rl_clear_screen_buffers,
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
    rl_framebuffer_attach,
    rl_framebuffer_complete,
    rl_load_draw_quad,
    rl_load_framebuffer,
    rl_load_texture,
    rl_unload_framebuffer,
    rl_unload_texture,
    rl_viewport,
    set_texture_filter,
    set_texture_wrap,
)

from builder.heightfield.gpu.composite_shader import (
    CompositeShader,
    load_composite_shader,
    set_int,
    unload_composite_shader,
)
from builder.heightfield.gpu.height_texture import (
    HeightTexture,
    HeightTextureKey,
    release_height_texture,
    upload_height_texture,
)
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    HeightLayerSource,
    Heightfield,
    PaintHeightSource,
    ProceduralHeightSource,
    ProceduralHeightType,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    ensure_heightmap_buffer,
)
from builder.view.lighting import set_float


@dataclass
class R32RenderTarget:
    """framebuffer with an r32f color attachment"""

    fbo_id: int
    texture_id: int
    texture: Texture
    width: int
    height: int


@dataclass
class HeightCompositeCache:
    """session resources for compositing height edit stacks"""

    blend: CompositeShader
    source_textures: dict[str, HeightTexture] = field(default_factory=dict)
    solid_one: HeightTexture | None = None
    solid_zero: HeightTexture | None = None
    ping: R32RenderTarget | None = None
    pong: R32RenderTarget | None = None


def create_height_composite_cache() -> HeightCompositeCache:
    """compile blend shader and return empty composite cache"""
    return HeightCompositeCache(blend=load_composite_shader())


def texture_from_id(texture_id: int, width: int, height: int) -> Texture:
    """wrap a gl texture id as a pyray Texture"""
    texture_cdata: Any = ffi.new("Texture *")
    texture_cdata.id = texture_id
    texture_cdata.width = width
    texture_cdata.height = height
    texture_cdata.mipmaps = 1
    texture_cdata.format = int(PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32)
    return texture_cdata[0]


def create_r32_target(width: int, height: int) -> R32RenderTarget:
    """allocate an r32f framebuffer + color texture"""
    if width < 1 or height < 1:
        raise ValueError("r32 render target needs positive size")
    texture_id: int = rl_load_texture(
        ffi.NULL,
        width,
        height,
        int(PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32),
        1,
    )
    if texture_id == 0:
        raise RuntimeError("failed to allocate r32 height texture")
    fbo_id: int = rl_load_framebuffer()
    if fbo_id == 0:
        rl_unload_texture(texture_id)
        raise RuntimeError("failed to allocate height composite framebuffer")
    rl_framebuffer_attach(
        fbo_id,
        texture_id,
        int(RL_ATTACHMENT_COLOR_CHANNEL0),
        int(RL_ATTACHMENT_TEXTURE2D),
        0,
    )
    if not rl_framebuffer_complete(fbo_id):
        rl_unload_framebuffer(fbo_id)
        rl_unload_texture(texture_id)
        raise RuntimeError("height composite framebuffer incomplete")
    texture: Texture = texture_from_id(texture_id, width, height)
    set_texture_filter(texture, TextureFilter.TEXTURE_FILTER_BILINEAR)
    set_texture_wrap(texture, TextureWrap.TEXTURE_WRAP_CLAMP)
    return R32RenderTarget(
        fbo_id=fbo_id,
        texture_id=texture_id,
        texture=texture,
        width=width,
        height=height,
    )


def release_r32_target(target: R32RenderTarget | None) -> None:
    """free one r32 framebuffer target"""
    if target is None:
        return
    rl_unload_framebuffer(target.fbo_id)
    rl_unload_texture(target.texture_id)


def ensure_ping_pong(
    cache: HeightCompositeCache, width: int, height: int
) -> tuple[R32RenderTarget, R32RenderTarget]:
    """create or resize ping-pong r32 targets"""
    ping: R32RenderTarget | None = cache.ping
    pong: R32RenderTarget | None = cache.pong
    if (
        ping is not None
        and pong is not None
        and ping.width == width
        and ping.height == height
        and pong.width == width
        and pong.height == height
    ):
        return (ping, pong)
    release_r32_target(ping)
    release_r32_target(pong)
    cache.ping = create_r32_target(width, height)
    cache.pong = create_r32_target(width, height)
    return (cache.ping, cache.pong)


def source_cache_key(source: HeightLayerSource) -> str:
    """stable key for a cached source texture"""
    match source:
        case CatalogHeightSource(heightmap_id=heightmap_id):
            return f"catalog:{heightmap_id.name}"
        case PaintHeightSource(heightmap_id=heightmap_id):
            return f"paint:{heightmap_id.name}"
        case ProceduralHeightSource(
            procedural_type=procedural_type, constant_value=constant_value
        ):
            return f"proc:{procedural_type.name}:{constant_value!r}"


def ensure_solid_textures(cache: HeightCompositeCache) -> None:
    """lazy 1x1 constant source textures"""
    if cache.solid_one is None:
        cache.solid_one = upload_height_texture(
            1,
            1,
            [1.0],
            HeightTextureKey(mode="solid", parts=(("one", 1.0, 0.0),)),
            1.0,
            0.0,
        )
    if cache.solid_zero is None:
        cache.solid_zero = upload_height_texture(
            1,
            1,
            [0.0],
            HeightTextureKey(mode="solid", parts=(("zero", 0.0, 0.0),)),
            1.0,
            0.0,
        )


def upload_source_texture(
    cache: HeightCompositeCache,
    catalog: HeightmapCatalog,
    source: HeightLayerSource,
) -> Texture:
    """upload or reuse a source bitmap as an r32 texture"""
    ensure_solid_textures(cache)
    key: str = source_cache_key(source)
    existing: HeightTexture | None = cache.source_textures.get(key)
    if existing is not None:
        return existing.texture
    samples: list[float]
    width: int
    height: int
    match source:
        case CatalogHeightSource(heightmap_id=heightmap_id) | PaintHeightSource(
            heightmap_id=heightmap_id
        ):
            buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
                catalog, heightmap_id
            )
            if buffer is None:
                assert cache.solid_zero is not None
                return cache.solid_zero.texture
            width = buffer.width
            height = buffer.height
            samples = list(buffer.samples)
        case ProceduralHeightSource(
            procedural_type=procedural_type, constant_value=constant_value
        ):
            match procedural_type:
                case ProceduralHeightType.constant:
                    width = 1
                    height = 1
                    samples = [float(constant_value)]
    uploaded: HeightTexture = upload_height_texture(
        width,
        height,
        samples,
        HeightTextureKey(mode="source", parts=((key, 0.0, 0.0),)),
        1.0,
        0.0,
    )
    # edit-layer / mask bitmaps tile under uv scale; fbo/solids stay clamp
    set_texture_wrap(uploaded.texture, TextureWrap.TEXTURE_WRAP_REPEAT)
    cache.source_textures[key] = uploaded
    return uploaded.texture


def resolve_base_resolution(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> tuple[int, int]:
    """composite target size from the base heightmap (or 1x1)"""
    if heightfield.base is None:
        return (1, 1)
    buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
        catalog, heightfield.base
    )
    if buffer is None:
        return (1, 1)
    return (buffer.width, buffer.height)


def source_key_parts(
    source: HeightLayerSource | None,
) -> tuple[str, float, float]:
    """fingerprint fragment for one source"""
    match source:
        case None:
            return ("none", 0.0, 0.0)
        case CatalogHeightSource(heightmap_id=heightmap_id):
            return (f"c:{heightmap_id.name}", 0.0, 0.0)
        case PaintHeightSource(heightmap_id=heightmap_id):
            return (f"p:{heightmap_id.name}", 0.0, 0.0)
        case ProceduralHeightSource(
            procedural_type=procedural_type, constant_value=constant_value
        ):
            return (f"pr:{procedural_type.name}", float(constant_value), 0.0)


def composite_texture_key(heightfield: Heightfield) -> HeightTextureKey:
    """fingerprint for gpu composite / raw base"""
    parts: list[tuple[str, float, float]] = []
    base_name: str = "" if heightfield.base is None else heightfield.base.name
    parts.append(
        (
            f"base:{base_name}",
            float(heightfield.base_amplitude.value),
            float(heightfield.base_offset.value),
        )
    )
    layer: HeightEditLayer
    enabled: bool = False
    for layer in heightfield.edit_layers:
        if not layer.enabled:
            continue
        enabled = True
        height_part: tuple[str, float, float] = source_key_parts(layer.height)
        mask_part: tuple[str, float, float] = source_key_parts(layer.mask)
        parts.append(
            (
                (
                    f"L:{layer.name}:{layer.blend.name}:{height_part[0]}:"
                    f"{mask_part[0]}:{layer.uv_scale_u:g}:{layer.uv_scale_v:g}:"
                    f"{layer.uv_offset_u:g}:{layer.uv_offset_v:g}"
                ),
                float(layer.amplitude) * 1000.0 + float(layer.opacity.value),
                float(layer.offset),
            )
        )
    mode: str = "composite" if enabled else "raw"
    return HeightTextureKey(mode=mode, parts=tuple(parts))


def blend_mode_uniform(blend: HeightBlendMode) -> int:
    """shader blendMode int matching HeightBlendMode values"""
    return int(blend)


def draw_blend_pass(
    cache: HeightCompositeCache,
    dest: R32RenderTarget,
    accum: Texture,
    layer: Texture,
    mask: Texture,
    *,
    blend_mode: int,
    use_mask: bool,
    opacity: float,
    layer_amp: float,
    layer_bias: float,
    uv_scale_u: float = 1.0,
    uv_scale_v: float = 1.0,
    uv_offset_u: float = 0.0,
    uv_offset_v: float = 0.0,
) -> None:
    """fullscreen blend into dest fbo via rlgl (ndc quad)"""
    blend: CompositeShader = cache.blend
    rl_draw_render_batch_active()
    # viewport scissor is in window pixels; must not clip fbo draws
    rl_disable_scissor_test()
    rl_enable_framebuffer(dest.fbo_id)
    rl_viewport(0, 0, dest.width, dest.height)
    rl_disable_depth_test()
    rl_clear_color(0, 0, 0, 0)
    rl_clear_screen_buffers()
    rl_enable_shader(blend.shader.id)
    set_int(blend.shader, blend.loc_accum, 0)
    set_int(blend.shader, blend.loc_layer, 1)
    set_int(blend.shader, blend.loc_mask, 2)
    set_int(blend.shader, blend.loc_blend_mode, blend_mode)
    set_int(blend.shader, blend.loc_use_mask, 1 if use_mask else 0)
    set_float(blend.shader, blend.loc_opacity, opacity)
    set_float(blend.shader, blend.loc_layer_amp, layer_amp)
    set_float(blend.shader, blend.loc_layer_bias, layer_bias)
    set_shader_value(
        blend.shader,
        blend.loc_layer_uv_scale,
        Vector2(uv_scale_u, uv_scale_v),
        ShaderUniformDataType.SHADER_UNIFORM_VEC2,
    )
    set_shader_value(
        blend.shader,
        blend.loc_layer_uv_offset,
        Vector2(uv_offset_u, uv_offset_v),
        ShaderUniformDataType.SHADER_UNIFORM_VEC2,
    )
    rl_active_texture_slot(0)
    rl_enable_texture(accum.id)
    rl_active_texture_slot(1)
    rl_enable_texture(layer.id)
    rl_active_texture_slot(2)
    rl_enable_texture(mask.id)
    rl_load_draw_quad()
    rl_disable_shader()
    rl_disable_framebuffer()
    rl_enable_depth_test()
    rl_enable_scissor_test()
    rl_viewport(0, 0, get_screen_width(), get_screen_height())
    rl_draw_render_batch_active()


def make_raw_base_texture(
    catalog: HeightmapCatalog, heightfield: Heightfield, key: HeightTextureKey
) -> HeightTexture:
    """upload base samples with scale/bias deferred to the displace shader"""
    scale: float = float(heightfield.base_amplitude.value)
    bias: float = float(heightfield.base_offset.value)
    if heightfield.base is None:
        return upload_height_texture(1, 1, [0.0], key, scale, bias)
    buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
        catalog, heightfield.base
    )
    if buffer is None:
        return upload_height_texture(1, 1, [0.0], key, scale, bias)
    return upload_height_texture(
        buffer.width,
        buffer.height,
        list(buffer.samples),
        key,
        scale,
        bias,
    )


def composite_height_texture(
    cache: HeightCompositeCache,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
) -> tuple[HeightTexture, R32RenderTarget | None]:
    """build displace height texture: raw base or gpu-composited edit stack

    when compositing, returns a dedicated result target the caller must free
    """
    key: HeightTextureKey = composite_texture_key(heightfield)
    enabled_layers: list[HeightEditLayer] = [
        layer for layer in heightfield.edit_layers if layer.enabled
    ]
    if not enabled_layers:
        return (make_raw_base_texture(catalog, heightfield, key), None)

    ensure_solid_textures(cache)
    assert cache.solid_one is not None
    assert cache.solid_zero is not None

    width: int
    height: int
    width, height = resolve_base_resolution(catalog, heightfield)
    ping: R32RenderTarget
    pong: R32RenderTarget
    ping, pong = ensure_ping_pong(cache, width, height)

    base_tex: Texture
    if heightfield.base is None:
        base_tex = cache.solid_zero.texture
    else:
        base_tex = upload_source_texture(
            cache, catalog, CatalogHeightSource(heightmap_id=heightfield.base)
        )

    # seed accum with base * amp + offset (blendMode < 0 = replace)
    draw_blend_pass(
        cache,
        ping,
        cache.solid_zero.texture,
        base_tex,
        cache.solid_one.texture,
        blend_mode=-1,
        use_mask=False,
        opacity=1.0,
        layer_amp=float(heightfield.base_amplitude.value),
        layer_bias=float(heightfield.base_offset.value),
    )
    read_target: R32RenderTarget = ping
    write_target: R32RenderTarget = pong

    layer: HeightEditLayer
    for layer in enabled_layers:
        layer_tex: Texture = upload_source_texture(cache, catalog, layer.height)
        mask_tex: Texture
        use_mask: bool
        if layer.mask is None:
            mask_tex = cache.solid_one.texture
            use_mask = False
        else:
            mask_tex = upload_source_texture(cache, catalog, layer.mask)
            use_mask = True
        draw_blend_pass(
            cache,
            write_target,
            read_target.texture,
            layer_tex,
            mask_tex,
            blend_mode=blend_mode_uniform(layer.blend),
            use_mask=use_mask,
            opacity=float(layer.opacity.value),
            layer_amp=float(layer.amplitude),
            layer_bias=float(layer.offset),
            uv_scale_u=float(layer.uv_scale_u),
            uv_scale_v=float(layer.uv_scale_v),
            uv_offset_u=float(layer.uv_offset_u),
            uv_offset_v=float(layer.uv_offset_v),
        )
        read_target, write_target = write_target, read_target

    # copy out of shared scratch into a per-entry result target
    result: R32RenderTarget = create_r32_target(width, height)
    draw_blend_pass(
        cache,
        result,
        cache.solid_zero.texture,
        read_target.texture,
        cache.solid_one.texture,
        blend_mode=-1,
        use_mask=False,
        opacity=1.0,
        layer_amp=1.0,
        layer_bias=0.0,
    )
    return (
        HeightTexture(
            texture=result.texture,
            keep_alive=(),
            key=key,
            height_scale=1.0,
            height_bias=0.0,
            owns_texture=False,
        ),
        result,
    )


def unload_height_composite_cache(cache: HeightCompositeCache) -> None:
    """release composite shader, sources, and ping-pong targets"""
    key: str
    for key in list(cache.source_textures):
        uploaded: HeightTexture = cache.source_textures.pop(key)
        release_height_texture(uploaded)
    if cache.solid_one is not None:
        release_height_texture(cache.solid_one)
        cache.solid_one = None
    if cache.solid_zero is not None:
        release_height_texture(cache.solid_zero)
        cache.solid_zero = None
    release_r32_target(cache.ping)
    release_r32_target(cache.pong)
    cache.ping = None
    cache.pong = None
    unload_composite_shader(cache.blend)
