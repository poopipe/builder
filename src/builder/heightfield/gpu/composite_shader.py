"""fullscreen height blend shader for r32f compositing"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    Shader,
    ShaderUniformDataType,
    ffi,
    get_shader_location,
    is_shader_valid,
    load_shader_from_memory,
    set_shader_value,
    unload_shader,
)

VS_SOURCE: str = """
#version 330
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec2 vertexTexCoord;
out vec2 fragTexCoord;

void main()
{
    fragTexCoord = vertexTexCoord;
    gl_Position = vec4(vertexPosition.xy, 0.0, 1.0);
}
"""

FS_SOURCE: str = """
#version 330
in vec2 fragTexCoord;
out vec4 finalColor;
uniform sampler2D accumMap;
uniform sampler2D layerMap;
uniform sampler2D maskMap;
uniform int blendMode;
uniform int useMask;
uniform float opacity;
uniform float layerAmp;
uniform float layerBias;
uniform vec2 layerUvScale;
uniform vec2 layerUvOffset;

void main()
{
    vec2 uv = clamp(fragTexCoord, vec2(0.0), vec2(1.0));
    vec2 layerUv = fract(uv * layerUvScale + layerUvOffset);
    float a = texture(accumMap, uv).r;
    float b = texture(layerMap, layerUv).r * layerAmp + layerBias;
    float m = opacity;
    if (useMask != 0) {
        m *= clamp(texture(maskMap, layerUv).r, 0.0, 1.0);
    }
    float outHeight = a;
    if (blendMode < 0) {
        outHeight = b;
    } else if (blendMode == 1) {
        outHeight = a + b * m;
    } else if (blendMode == 2) {
        outHeight = a - b * m;
    } else if (blendMode == 3) {
        outHeight = a * mix(1.0, b, m);
    } else if (blendMode == 4) {
        outHeight = mix(a, max(a, b), m);
    } else if (blendMode == 5) {
        outHeight = mix(a, min(a, b), m);
    } else if (blendMode == 6) {
        outHeight = mix(a, b, m);
    } else if (blendMode == 7) {
        // premultiplied absolute (b = h*cov); m = opacity*cov
        outHeight = a * (1.0 - m) + b * opacity;
    }
    finalColor = vec4(outHeight, 0.0, 0.0, 1.0);
}
"""


@dataclass
class CompositeShader:
    """compiled fullscreen blend shader with cached locations"""

    shader: Shader
    loc_accum: int
    loc_layer: int
    loc_mask: int
    loc_blend_mode: int
    loc_use_mask: int
    loc_opacity: int
    loc_layer_amp: int
    loc_layer_bias: int
    loc_layer_uv_scale: int
    loc_layer_uv_offset: int


def load_composite_shader() -> CompositeShader:
    """compile the height composite blend shader"""
    shader: Shader = load_shader_from_memory(VS_SOURCE, FS_SOURCE)
    if not is_shader_valid(shader):
        raise RuntimeError("failed to compile height composite shader")
    loc_accum: int = get_shader_location(shader, "accumMap")
    loc_layer: int = get_shader_location(shader, "layerMap")
    loc_mask: int = get_shader_location(shader, "maskMap")
    loc_blend_mode: int = get_shader_location(shader, "blendMode")
    loc_use_mask: int = get_shader_location(shader, "useMask")
    loc_opacity: int = get_shader_location(shader, "opacity")
    loc_layer_amp: int = get_shader_location(shader, "layerAmp")
    loc_layer_bias: int = get_shader_location(shader, "layerBias")
    loc_layer_uv_scale: int = get_shader_location(shader, "layerUvScale")
    loc_layer_uv_offset: int = get_shader_location(shader, "layerUvOffset")
    if (
        loc_accum < 0
        or loc_layer < 0
        or loc_mask < 0
        or loc_blend_mode < 0
        or loc_use_mask < 0
        or loc_opacity < 0
        or loc_layer_amp < 0
        or loc_layer_bias < 0
        or loc_layer_uv_scale < 0
        or loc_layer_uv_offset < 0
    ):
        unload_shader(shader)
        raise RuntimeError("height composite shader is missing required uniforms")
    # fixed sampler slots 0/1/2
    set_shader_value(
        shader, loc_accum, ffi.new("int *", 0), ShaderUniformDataType.SHADER_UNIFORM_INT
    )
    set_shader_value(
        shader, loc_layer, ffi.new("int *", 1), ShaderUniformDataType.SHADER_UNIFORM_INT
    )
    set_shader_value(
        shader, loc_mask, ffi.new("int *", 2), ShaderUniformDataType.SHADER_UNIFORM_INT
    )
    return CompositeShader(
        shader=shader,
        loc_accum=loc_accum,
        loc_layer=loc_layer,
        loc_mask=loc_mask,
        loc_blend_mode=loc_blend_mode,
        loc_use_mask=loc_use_mask,
        loc_opacity=loc_opacity,
        loc_layer_amp=loc_layer_amp,
        loc_layer_bias=loc_layer_bias,
        loc_layer_uv_scale=loc_layer_uv_scale,
        loc_layer_uv_offset=loc_layer_uv_offset,
    )


def set_int(shader: Shader, loc: int, value: int) -> None:
    """upload an int uniform"""
    set_shader_value(
        shader, loc, ffi.new("int *", value), ShaderUniformDataType.SHADER_UNIFORM_INT
    )


def unload_composite_shader(composite: CompositeShader) -> None:
    """release the composite shader"""
    unload_shader(composite.shader)
