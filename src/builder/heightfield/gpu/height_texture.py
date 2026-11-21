"""upload float height samples as an r32 gpu texture"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyray import (
    Image,
    PixelFormat,
    Texture,
    TextureFilter,
    TextureWrap,
    ffi,
    load_texture_from_image,
    set_texture_filter,
    set_texture_wrap,
    unload_texture,
    update_texture,
)

from builder.heightfield.heightfield_types import Heightfield
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import base_height_scale_bias, base_sample_type


@dataclass(frozen=True)
class HeightTextureKey:
    """identity of gpu height samples (raw base or composited stack)"""

    mode: str
    parts: tuple[tuple[str, float, float], ...]


@dataclass
class HeightTexture:
    """gpu height map plus keep-alive for the upload buffer"""

    texture: Texture
    keep_alive: tuple[Any, ...]
    key: HeightTextureKey
    height_scale: float
    height_bias: float
    owns_texture: bool = True


def upload_height_texture(
    width: int,
    height: int,
    samples: list[float],
    key: HeightTextureKey,
    height_scale: float,
    height_bias: float,
) -> HeightTexture:
    """upload row-major float samples as an r32 texture"""
    if width < 1 or height < 1:
        raise ValueError("height texture needs positive size")
    if len(samples) != width * height:
        raise ValueError("height sample count does not match width*height")
    data: Any = ffi.new("float[]", samples)
    image_cdata: Any = ffi.new("Image *")
    image_cdata.data = data
    image_cdata.width = width
    image_cdata.height = height
    image_cdata.mipmaps = 1
    image_cdata.format = int(PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32)
    image: Image = image_cdata[0]
    texture: Texture = load_texture_from_image(image)
    set_texture_filter(texture, TextureFilter.TEXTURE_FILTER_BILINEAR)
    set_texture_wrap(texture, TextureWrap.TEXTURE_WRAP_CLAMP)
    return HeightTexture(
        texture=texture,
        keep_alive=(data, image_cdata),
        key=key,
        height_scale=height_scale,
        height_bias=height_bias,
        owns_texture=True,
    )


def update_or_upload_height_texture(
    existing: HeightTexture | None,
    width: int,
    height: int,
    samples: list[float],
    key: HeightTextureKey,
    height_scale: float,
    height_bias: float,
) -> HeightTexture:
    """reuse an existing r32 texture via update_texture when size matches"""
    if (
        existing is not None
        and int(existing.texture.width) == width
        and int(existing.texture.height) == height
    ):
        data: Any = ffi.new("float[]", samples)
        update_texture(existing.texture, data)
        existing.keep_alive = (data,)
        existing.key = key
        existing.height_scale = height_scale
        existing.height_bias = height_bias
        return existing
    if existing is not None:
        release_height_texture(existing)
    return upload_height_texture(
        width, height, samples, key, height_scale, height_bias
    )


def make_height_texture(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    composite_cache: Any,
) -> tuple[HeightTexture, Any]:
    """build displace height texture via raw upload or gpu composite

    second value is an optional r32 result target the caller must free
    """
    from builder.heightfield.gpu.composite import (
        HeightCompositeCache,
        composite_height_texture,
    )

    cache: HeightCompositeCache = composite_cache
    return composite_height_texture(cache, catalog, heightfield)


def height_texture_key(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> HeightTextureKey:
    """fingerprint used to decide when to rebuild the height texture"""
    from builder.heightfield.gpu.composite import composite_texture_key

    return composite_texture_key(catalog, heightfield)


def height_texture_uniforms(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> tuple[float, float]:
    """scale/bias when no enabled edit layers; composite uses (1, 0)"""
    enabled: bool = any(layer.enabled for layer in heightfield.edit_layers)
    if enabled:
        return (1.0, 0.0)
    return base_height_scale_bias(
        heightfield, base_sample_type(catalog, heightfield)
    )


def release_height_texture(height_tex: HeightTexture | None) -> None:
    """unload an owned height texture if present"""
    if height_tex is None:
        return
    if height_tex.owns_texture:
        unload_texture(height_tex.texture)
