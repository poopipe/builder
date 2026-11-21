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
)

from builder.heightfield.heightfield_types import Heightfield, HeightmapLayer
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
    ensure_heightmap_buffer,
)
from builder.heightfield.sample import sample_height


@dataclass(frozen=True)
class HeightTextureKey:
    """identity of gpu height samples (raw single layer or baked stack)"""

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


def layer_key_parts(layers: tuple[HeightmapLayer, ...]) -> tuple[tuple[str, float, float], ...]:
    """stable fingerprint of layer id + amp + offset"""
    layer: HeightmapLayer
    return tuple(
        (layer.heightmap_id.name, float(layer.amplitude), float(layer.offset))
        for layer in layers
    )


def height_texture_key(heightfield: Heightfield) -> HeightTextureKey:
    """choose raw-single vs baked-composite key for this recipe"""
    layers: tuple[HeightmapLayer, ...] = heightfield.layers
    if len(layers) == 1:
        layer: HeightmapLayer = layers[0]
        return HeightTextureKey(
            mode="raw",
            parts=((layer.heightmap_id.name, 0.0, 0.0),),
        )
    return HeightTextureKey(mode="baked", parts=layer_key_parts(layers))


def height_texture_uniforms(heightfield: Heightfield) -> tuple[float, float]:
    """scale/bias applied in the displace shader"""
    layers: tuple[HeightmapLayer, ...] = heightfield.layers
    if len(layers) == 1:
        layer: HeightmapLayer = layers[0]
        return (float(layer.amplitude), float(layer.offset))
    return (1.0, 0.0)


def bake_composite_samples(
    catalog: HeightmapCatalog,
    layers: tuple[HeightmapLayer, ...],
    width: int,
    height: int,
) -> list[float]:
    """sample world-space height at each texel uv"""
    if width < 1 or height < 1:
        raise ValueError("height texture needs positive size")
    samples: list[float] = []
    y: int
    x: int
    for y in range(height):
        v: float = 0.0 if height <= 1 else float(y) / float(height - 1)
        for x in range(width):
            u: float = 0.0 if width <= 1 else float(x) / float(width - 1)
            samples.append(sample_height(catalog, layers, u, v))
    return samples


def resolve_bake_resolution(
    catalog: HeightmapCatalog, layers: tuple[HeightmapLayer, ...]
) -> tuple[int, int]:
    """pick texture size from the largest available layer buffer"""
    if not layers:
        return (1, 1)
    width: int = 1
    height: int = 1
    layer: HeightmapLayer
    for layer in layers:
        buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
            catalog, layer.heightmap_id
        )
        if buffer is None:
            continue
        if buffer.width * buffer.height > width * height:
            width = buffer.width
            height = buffer.height
    return (width, height)


def raw_layer_samples(
    catalog: HeightmapCatalog, heightmap_id: HeightmapId
) -> tuple[int, int, list[float]]:
    """cpu samples for a single heightmap, or a flat 1x1"""
    buffer: HeightmapBuffer | None = ensure_heightmap_buffer(catalog, heightmap_id)
    if buffer is None:
        return (1, 1, [0.0])
    return (buffer.width, buffer.height, list(buffer.samples))


def build_height_samples(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> tuple[int, int, list[float], float, float]:
    """resolve samples and uniforms for the current recipe"""
    layers: tuple[HeightmapLayer, ...] = heightfield.layers
    scale: float
    bias: float
    scale, bias = height_texture_uniforms(heightfield)
    if not layers:
        return (1, 1, [0.0], scale, bias)
    if len(layers) == 1:
        width: int
        height: int
        samples: list[float]
        width, height, samples = raw_layer_samples(catalog, layers[0].heightmap_id)
        return (width, height, samples, scale, bias)
    width, height = resolve_bake_resolution(catalog, layers)
    samples = bake_composite_samples(catalog, layers, width, height)
    return (width, height, samples, scale, bias)


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
    )


def make_height_texture(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> HeightTexture:
    """build and upload the height texture for a heightfield recipe"""
    width: int
    height: int
    samples: list[float]
    scale: float
    bias: float
    width, height, samples, scale, bias = build_height_samples(catalog, heightfield)
    return upload_height_texture(
        width,
        height,
        samples,
        height_texture_key(heightfield),
        scale,
        bias,
    )


def release_height_texture(height_tex: HeightTexture | None) -> None:
    """unload a height texture if present"""
    if height_tex is None:
        return
    unload_texture(height_tex.texture)
