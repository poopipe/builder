"""fullscreen vector sculpt stamp shader (r32f layer fbo)"""

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

uniform sampler2D layerMap;
uniform sampler2D belowMap;
uniform sampler2D segmentMap;
uniform sampler2D polyMap;

uniform int passMode;
uniform int shapeMode;
uniform int brushMode;
uniform int segmentCount;
uniform int polyCount;

uniform vec4 dirtyUv;
uniform vec2 fieldSize;
uniform float radiusMetres;
uniform float offsetAmount;
uniform float intensity;
uniform float flattenTarget;
uniform vec4 circleParams;

float softBand(float signedDist, float radius)
{
    if (radius <= 1e-8) {
        return signedDist <= 0.0 ? 1.0 : 0.0;
    }
    if (signedDist <= -radius) {
        return 1.0;
    }
    if (signedDist >= radius) {
        return 0.0;
    }
    float t = 0.5 - 0.5 * (signedDist / radius);
    return t * t * (3.0 - 2.0 * t);
}

float strokeWeight(float dist, float radius)
{
    if (radius <= 1e-8) {
        return dist <= 1e-8 ? 1.0 : 0.0;
    }
    if (dist >= radius) {
        return 0.0;
    }
    float t = 1.0 - dist / radius;
    return t * t * (3.0 - 2.0 * t);
}

float distToSegment(vec2 p, vec2 a, vec2 b)
{
    vec2 d = b - a;
    float lenSq = dot(d, d);
    float t = 0.0;
    if (lenSq > 1e-12) {
        t = clamp(dot(p - a, d) / lenSq, 0.0, 1.0);
    }
    vec2 closest = a + d * t;
    return length(p - closest);
}

float minSegmentDist(vec2 p)
{
    float best = 1e9;
    int count = segmentCount;
    for (int i = 0; i < 4096; i++) {
        if (i >= count) {
            break;
        }
        float u = (float(i) + 0.5) / float(max(count, 1));
        vec4 seg = texture(segmentMap, vec2(u, 0.5));
        best = min(best, distToSegment(p, seg.xy, seg.zw));
    }
    return best;
}

bool pointInPoly(vec2 p)
{
    bool inside = false;
    int count = polyCount;
    if (count < 3) {
        return false;
    }
    for (int i = 0; i < 4096; i++) {
        if (i >= count) {
            break;
        }
        int j = (i == 0) ? (count - 1) : (i - 1);
        float ui = (float(i) + 0.5) / float(max(count, 1));
        float uj = (float(j) + 0.5) / float(max(count, 1));
        vec2 pi = texture(polyMap, vec2(ui, 0.5)).xy;
        vec2 pj = texture(polyMap, vec2(uj, 0.5)).xy;
        bool intersects = ((pi.y > p.y) != (pj.y > p.y)) &&
            (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + 1e-30) + pi.x);
        if (intersects) {
            inside = !inside;
        }
    }
    return inside;
}

float shapeWeight(vec2 world)
{
    if (shapeMode == 2) {
        float distRim = abs(length(world - circleParams.xy) - circleParams.z);
        return strokeWeight(distRim, radiusMetres);
    }
    if (shapeMode == 3) {
        float signedDist = length(world - circleParams.xy) - circleParams.z;
        return softBand(signedDist, radiusMetres);
    }
    float dist = minSegmentDist(world);
    if (shapeMode == 0) {
        return strokeWeight(dist, radiusMetres);
    }
    // fill: soft band with even-odd interior
    bool inside = pointInPoly(world);
    float signedDist = inside ? -dist : dist;
    return softBand(signedDist, radiusMetres);
}

float applySculpt(float current, float below, float weight)
{
    if (weight <= 0.0) {
        return current;
    }
    if (brushMode == 0) {
        return current + offsetAmount * weight;
    }
    float visible = below + current;
    float target = flattenTarget;
    if (brushMode == 2 && visible >= target) {
        return current;
    }
    if (brushMode == 3 && visible <= target) {
        return current;
    }
    float desired = target - below;
    float mixAmt = weight * clamp(intensity, 0.0, 1.0);
    return current + (desired - current) * mixAmt;
}

void main()
{
    vec2 uv = clamp(fragTexCoord, vec2(0.0), vec2(1.0));
    float current = texture(layerMap, uv).r;
    bool inDirty = uv.x >= dirtyUv.x && uv.x < dirtyUv.z &&
        uv.y >= dirtyUv.y && uv.y < dirtyUv.w;
    if (passMode == 0) {
        finalColor = vec4(inDirty ? 0.0 : current, 0.0, 0.0, 1.0);
        return;
    }
    if (!inDirty) {
        finalColor = vec4(current, 0.0, 0.0, 1.0);
        return;
    }
    vec2 world = vec2((uv.x - 0.5) * fieldSize.x, (uv.y - 0.5) * fieldSize.y);
    float weight = shapeWeight(world);
    float below = texture(belowMap, uv).r;
    float outHeight = applySculpt(current, below, weight);
    finalColor = vec4(outHeight, 0.0, 0.0, 1.0);
}
"""

# passMode
PASS_CLEAR: int = 0
PASS_STAMP: int = 1

# shapeMode
SHAPE_STROKE: int = 0
SHAPE_FILL: int = 1
SHAPE_CIRCLE_STROKE: int = 2
SHAPE_CIRCLE_FILL: int = 3

# brushMode
BRUSH_ADD: int = 0
BRUSH_FLATTEN: int = 1
BRUSH_FLATTEN_UP: int = 2
BRUSH_FLATTEN_DOWN: int = 3


@dataclass
class SculptStampShader:
    """compiled sculpt stamp shader with cached locations"""

    shader: Shader
    loc_layer: int
    loc_below: int
    loc_segment: int
    loc_poly: int
    loc_pass_mode: int
    loc_shape_mode: int
    loc_brush_mode: int
    loc_segment_count: int
    loc_poly_count: int
    loc_dirty_uv: int
    loc_field_size: int
    loc_radius: int
    loc_offset: int
    loc_intensity: int
    loc_flatten_target: int
    loc_circle: int


def load_sculpt_stamp_shader() -> SculptStampShader:
    """compile the vector sculpt stamp shader"""
    shader: Shader = load_shader_from_memory(VS_SOURCE, FS_SOURCE)
    if not is_shader_valid(shader):
        raise RuntimeError("failed to compile sculpt stamp shader")
    names: dict[str, str] = {
        "layerMap": "loc_layer",
        "belowMap": "loc_below",
        "segmentMap": "loc_segment",
        "polyMap": "loc_poly",
        "passMode": "loc_pass_mode",
        "shapeMode": "loc_shape_mode",
        "brushMode": "loc_brush_mode",
        "segmentCount": "loc_segment_count",
        "polyCount": "loc_poly_count",
        "dirtyUv": "loc_dirty_uv",
        "fieldSize": "loc_field_size",
        "radiusMetres": "loc_radius",
        "offsetAmount": "loc_offset",
        "intensity": "loc_intensity",
        "flattenTarget": "loc_flatten_target",
        "circleParams": "loc_circle",
    }
    locs: dict[str, int] = {}
    uniform: str
    for uniform in names:
        loc: int = get_shader_location(shader, uniform)
        if loc < 0:
            unload_shader(shader)
            raise RuntimeError(f"sculpt stamp shader missing uniform {uniform}")
        locs[names[uniform]] = loc
    set_shader_value(
        shader,
        locs["loc_layer"],
        ffi.new("int *", 0),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    set_shader_value(
        shader,
        locs["loc_below"],
        ffi.new("int *", 1),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    set_shader_value(
        shader,
        locs["loc_segment"],
        ffi.new("int *", 2),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    set_shader_value(
        shader,
        locs["loc_poly"],
        ffi.new("int *", 3),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    return SculptStampShader(
        shader=shader,
        loc_layer=locs["loc_layer"],
        loc_below=locs["loc_below"],
        loc_segment=locs["loc_segment"],
        loc_poly=locs["loc_poly"],
        loc_pass_mode=locs["loc_pass_mode"],
        loc_shape_mode=locs["loc_shape_mode"],
        loc_brush_mode=locs["loc_brush_mode"],
        loc_segment_count=locs["loc_segment_count"],
        loc_poly_count=locs["loc_poly_count"],
        loc_dirty_uv=locs["loc_dirty_uv"],
        loc_field_size=locs["loc_field_size"],
        loc_radius=locs["loc_radius"],
        loc_offset=locs["loc_offset"],
        loc_intensity=locs["loc_intensity"],
        loc_flatten_target=locs["loc_flatten_target"],
        loc_circle=locs["loc_circle"],
    )


def unload_sculpt_stamp_shader(stamp: SculptStampShader) -> None:
    """release the sculpt stamp shader"""
    unload_shader(stamp.shader)
