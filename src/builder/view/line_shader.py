"""batched screen-space thick lines via a small GLSL shader"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from pyray import (
    Color,
    Material,
    MaterialMapIndex,
    Matrix,
    Mesh,
    Shader,
    ShaderLocationIndex,
    ShaderUniformDataType,
    Texture,
    Vector2,
    draw_mesh,
    ffi,
    get_screen_height,
    get_screen_width,
    get_shader_location,
    is_shader_valid,
    load_material_default,
    load_shader_from_memory,
    matrix_identity,
    rl_disable_backface_culling,
    rl_enable_backface_culling,
    set_shader_value,
    unload_mesh,
    unload_shader,
    upload_mesh,
)

# ndc depth pull toward camera (clip.z -= bias * clip.w); distance-invariant
line_shadow_depth_bias: float = 0.002

VS_SOURCE: str = """
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;
out vec4 fragColor;
uniform mat4 mvp;
uniform vec2 resolution;
uniform float depthBias;
uniform sampler2D heightMap;
uniform int useHeightMap;
uniform vec2 fieldSize;
uniform float heightScale;
uniform float heightBias;

vec3 drapePoint(vec3 p)
{
    if (useHeightMap == 0) {
        return p;
    }
    float sx = max(fieldSize.x, 1e-8);
    float sz = max(fieldSize.y, 1e-8);
    vec2 uv = vec2((p.x + 0.5 * sx) / sx, (p.z + 0.5 * sz) / sz);
    uv = clamp(uv, vec2(0.0), vec2(1.0));
    float h = texture(heightMap, uv).r * heightScale + heightBias;
    return vec3(p.x, h, p.z);
}

void main()
{
    float side = vertexTexCoord.x;
    float thicknessPx = vertexTexCoord.y;
    vec3 p0 = drapePoint(vertexPosition);
    vec3 p1 = drapePoint(vertexNormal);
    vec4 clip0 = mvp * vec4(p0, 1.0);
    vec4 clip1 = mvp * vec4(p1, 1.0);
    vec2 screen0 = (clip0.xy / max(abs(clip0.w), 1e-6)) * resolution;
    vec2 screen1 = (clip1.xy / max(abs(clip1.w), 1e-6)) * resolution;
    vec2 dir = screen1 - screen0;
    float len = length(dir);
    vec2 perp = len > 1e-4
        ? vec2(-dir.y, dir.x) / len
        : vec2(0.0, 1.0);
    vec2 offsetPx = perp * side * (thicknessPx * 0.5);
    vec4 clip = clip0;
    clip.xy += (offsetPx / resolution) * clip.w * 2.0;
    clip.z -= depthBias * clip.w;
    fragColor = vertexColor;
    gl_Position = clip;
}
"""

FS_SOURCE: str = """
#version 330
in vec4 fragColor;
out vec4 finalColor;
void main()
{
    finalColor = fragColor;
}
"""

_min_segment_length: float = 1e-6
# ushort indices; four verts per segment
_max_segments_per_draw: int = 16000


@dataclass(frozen=True)
class LineSegment:
    """one world-space segment with screen-pixel thickness"""

    ax: float
    ay: float
    az: float
    bx: float
    by: float
    bz: float
    color: Color
    thickness_px: float


@dataclass
class LineBatch:
    """accumulated overlay segments for one flush"""

    segments: list[LineSegment] = field(default_factory=list)


@dataclass(frozen=True)
class LineShader:
    """compiled screen-space line shader, material, and uniform locations

    material is kept for the shader lifetime: UnloadMaterial would free the shared shader
    """

    shader: Shader
    material: Material
    loc_resolution: int
    loc_depth_bias: int
    loc_use_height_map: int
    loc_field_size: int
    loc_height_scale: int
    loc_height_bias: int


@dataclass(frozen=True)
class LineHeightDrape:
    """bind a height texture so shadow lines sample y in the vertex shader"""

    texture: Texture
    size_x: float
    size_z: float
    height_scale: float
    height_bias: float


def line_batch_buffer_counts(segment_count: int) -> tuple[int, int]:
    """vertex and index counts for a batch of n segments"""
    count: int = max(0, int(segment_count))
    return count * 4, count * 6


def clear_line_batch(batch: LineBatch) -> None:
    """drop accumulated segments"""
    batch.segments.clear()


def add_line_segment(
    batch: LineBatch,
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
    color: Color,
    thickness_px: float,
) -> None:
    """queue one segment; skip degenerates"""
    if hypot(bx - ax, by - ay, bz - az) < _min_segment_length:
        return
    if float(thickness_px) <= 0.0:
        return
    batch.segments.append(
        LineSegment(
            ax=float(ax),
            ay=float(ay),
            az=float(az),
            bx=float(bx),
            by=float(by),
            bz=float(bz),
            color=color,
            thickness_px=float(thickness_px),
        )
    )


def set_vec2(shader: Shader, loc: int, value: Vector2) -> None:
    set_shader_value(shader, loc, value, ShaderUniformDataType.SHADER_UNIFORM_VEC2)


def set_depth_bias(line: LineShader, bias: float) -> None:
    """set clip-space depth bias (0 = none)"""
    set_shader_value(
        line.shader,
        line.loc_depth_bias,
        ffi.new("float *", float(bias)),
        ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
    )


def set_line_height_drape(line: LineShader, drape: LineHeightDrape | None) -> None:
    """enable or disable heightMap y displace for the next line draws"""
    if drape is None:
        set_shader_value(
            line.shader,
            line.loc_use_height_map,
            ffi.new("int *", 0),
            ShaderUniformDataType.SHADER_UNIFORM_INT,
        )
        return
    line.material.maps[MaterialMapIndex.MATERIAL_MAP_HEIGHT].texture = (
        drape.texture
    )
    set_shader_value(
        line.shader,
        line.loc_use_height_map,
        ffi.new("int *", 1),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    set_vec2(
        line.shader,
        line.loc_field_size,
        Vector2(float(drape.size_x), float(drape.size_z)),
    )
    set_shader_value(
        line.shader,
        line.loc_height_scale,
        ffi.new("float *", float(drape.height_scale)),
        ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
    )
    set_shader_value(
        line.shader,
        line.loc_height_bias,
        ffi.new("float *", float(drape.height_bias)),
        ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
    )


def load_line_shader() -> LineShader:
    """compile the screen-space line shader"""
    shader: Shader = load_shader_from_memory(VS_SOURCE, FS_SOURCE)
    if not is_shader_valid(shader):
        raise RuntimeError("failed to compile line shader")
    shader.locs[ShaderLocationIndex.SHADER_LOC_MATRIX_MVP] = get_shader_location(
        shader, "mvp"
    )
    shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_HEIGHT] = get_shader_location(
        shader, "heightMap"
    )
    loc_resolution: int = get_shader_location(shader, "resolution")
    loc_depth_bias: int = get_shader_location(shader, "depthBias")
    loc_use_height_map: int = get_shader_location(shader, "useHeightMap")
    loc_field_size: int = get_shader_location(shader, "fieldSize")
    loc_height_scale: int = get_shader_location(shader, "heightScale")
    loc_height_bias: int = get_shader_location(shader, "heightBias")
    if (
        shader.locs[ShaderLocationIndex.SHADER_LOC_MATRIX_MVP] < 0
        or shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_HEIGHT] < 0
        or loc_resolution < 0
        or loc_depth_bias < 0
        or loc_use_height_map < 0
        or loc_field_size < 0
        or loc_height_scale < 0
        or loc_height_bias < 0
    ):
        unload_shader(shader)
        raise RuntimeError("line shader is missing required uniforms")
    set_shader_value(
        shader,
        shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_HEIGHT],
        ffi.new("int *", int(MaterialMapIndex.MATERIAL_MAP_HEIGHT)),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    material: Material = load_material_default()
    material.shader = shader
    line: LineShader = LineShader(
        shader=shader,
        material=material,
        loc_resolution=loc_resolution,
        loc_depth_bias=loc_depth_bias,
        loc_use_height_map=loc_use_height_map,
        loc_field_size=loc_field_size,
        loc_height_scale=loc_height_scale,
        loc_height_bias=loc_height_bias,
    )
    set_depth_bias(line, 0.0)
    set_line_height_drape(line, None)
    return line


def unload_line_shader(line: LineShader) -> None:
    """release GPU resources; material maps are left (UnloadMaterial frees shader)"""
    unload_shader(line.shader)


def _null_mesh_cpu_pointers(mesh: Mesh) -> None:
    """detach cffi-owned cpu buffers before unload_mesh"""
    field_name: str
    for field_name in (
        "vertices",
        "texcoords",
        "texcoords2",
        "normals",
        "tangents",
        "colors",
        "indices",
        "animVertices",
        "animNormals",
        "boneIds",
        "boneWeights",
        "boneMatrices",
    ):
        try:
            setattr(mesh, field_name, ffi.NULL)
        except AttributeError:
            pass


@dataclass(frozen=True)
class UploadedLineMesh:
    """one gpu chunk of line quads; keep_alive holds cffi cpu buffers"""

    mesh: Mesh
    keep_alive: tuple[Any, ...]


@dataclass
class LineMeshCache:
    """persistent uploaded line meshes; rebuild when key changes

    shadow_meshes and flat_meshes stay separate so draped lines can use a
    clip-space depth bias without affecting work-plane overlays
    """

    key: object | None = None
    meshes: list[UploadedLineMesh] = field(default_factory=list)
    shadow_meshes: list[UploadedLineMesh] = field(default_factory=list)
    flat_meshes: list[UploadedLineMesh] = field(default_factory=list)


def upload_line_segments(segments: list[LineSegment]) -> UploadedLineMesh | None:
    """upload one chunk of segments (caller respects ushort index limit)"""
    if len(segments) == 0:
        return None
    vertex_count: int
    index_count: int
    vertex_count, index_count = line_batch_buffer_counts(len(segments))
    positions: list[float] = []
    normals: list[float] = []
    texcoords: list[float] = []
    colors: list[int] = []
    indices: list[int] = []
    segment: LineSegment
    base: int
    for base, segment in enumerate(segments):
        thickness: float = segment.thickness_px
        color: Color = segment.color
        r: int = int(color.r)
        g: int = int(color.g)
        b: int = int(color.b)
        a: int = int(color.a)
        # four corners: (a,-1), (a,+1), (b,-1), (b,+1)
        # at b, negate side so screen dir flip keeps the same ribbon edge
        corner: int
        for corner in range(4):
            at_b: bool = corner >= 2
            side: float = -1.0 if corner % 2 == 0 else 1.0
            if at_b:
                side = -side
                positions.extend([segment.bx, segment.by, segment.bz])
                normals.extend([segment.ax, segment.ay, segment.az])
            else:
                positions.extend([segment.ax, segment.ay, segment.az])
                normals.extend([segment.bx, segment.by, segment.bz])
            texcoords.extend([side, thickness])
            colors.extend([r, g, b, a])
        vi: int = base * 4
        # ccw toward camera after screen-space expansion
        indices.extend([vi, vi + 2, vi + 1, vi + 1, vi + 2, vi + 3])

    vertices_buf: Any = ffi.new("float[]", positions)
    normals_buf: Any = ffi.new("float[]", normals)
    texcoords_buf: Any = ffi.new("float[]", texcoords)
    colors_buf: Any = ffi.new("unsigned char[]", colors)
    indices_buf: Any = ffi.new("unsigned short[]", indices)
    mesh_cdata: Any = ffi.new("Mesh *")
    mesh_cdata.vertexCount = vertex_count
    mesh_cdata.triangleCount = index_count // 3
    mesh_cdata.vertices = vertices_buf
    mesh_cdata.normals = normals_buf
    mesh_cdata.texcoords = texcoords_buf
    mesh_cdata.colors = colors_buf
    mesh_cdata.indices = indices_buf
    mesh: Mesh = mesh_cdata[0]
    upload_mesh(mesh, False)
    return UploadedLineMesh(
        mesh=mesh,
        keep_alive=(
            mesh_cdata,
            vertices_buf,
            normals_buf,
            texcoords_buf,
            colors_buf,
            indices_buf,
        ),
    )


def upload_line_batch(batch: LineBatch) -> list[UploadedLineMesh]:
    """upload all queued segments in ushort-safe chunks and clear the batch"""
    segments: list[LineSegment] = batch.segments
    meshes: list[UploadedLineMesh] = []
    start: int = 0
    while start < len(segments):
        end: int = min(start + _max_segments_per_draw, len(segments))
        uploaded: UploadedLineMesh | None = upload_line_segments(
            segments[start:end]
        )
        if uploaded is not None:
            meshes.append(uploaded)
        start = end
    clear_line_batch(batch)
    return meshes


def release_uploaded_line_mesh(uploaded: UploadedLineMesh) -> None:
    """free one uploaded line chunk"""
    mesh: Mesh = uploaded.mesh
    _null_mesh_cpu_pointers(mesh)
    unload_mesh(mesh)


def clear_line_mesh_cache(cache: LineMeshCache) -> None:
    """drop gpu meshes and forget the key"""
    uploaded: UploadedLineMesh
    for uploaded in cache.meshes:
        release_uploaded_line_mesh(uploaded)
    cache.meshes.clear()
    for uploaded in cache.shadow_meshes:
        release_uploaded_line_mesh(uploaded)
    cache.shadow_meshes.clear()
    for uploaded in cache.flat_meshes:
        release_uploaded_line_mesh(uploaded)
    cache.flat_meshes.clear()
    cache.key = None


def draw_uploaded_line_meshes(
    meshes: list[UploadedLineMesh],
    line: LineShader,
    transform: Matrix | None = None,
) -> None:
    """draw previously uploaded line chunks with the current view"""
    if len(meshes) == 0:
        return
    resolution: Vector2 = Vector2(
        float(get_screen_width()), float(get_screen_height())
    )
    set_vec2(line.shader, line.loc_resolution, resolution)
    model: Matrix = matrix_identity() if transform is None else transform
    # billboard strips can flip winding with view; keep both faces
    rl_disable_backface_culling()
    uploaded: UploadedLineMesh
    for uploaded in meshes:
        draw_mesh(uploaded.mesh, line.material, model)
    rl_enable_backface_culling()


def sync_line_mesh_cache(
    cache: LineMeshCache,
    key: object,
    batch: LineBatch,
) -> None:
    """rebuild cache meshes from batch when key changed; always clears batch"""
    if cache.key == key:
        clear_line_batch(batch)
        return
    clear_line_mesh_cache(cache)
    cache.meshes = upload_line_batch(batch)
    cache.key = key


def sync_split_line_mesh_cache(
    cache: LineMeshCache,
    key: object,
    shadow_batch: LineBatch,
    flat_batch: LineBatch,
) -> None:
    """rebuild separate shadow and flat meshes; always clears both batches"""
    if cache.key == key:
        clear_line_batch(shadow_batch)
        clear_line_batch(flat_batch)
        return
    clear_line_mesh_cache(cache)
    cache.shadow_meshes = upload_line_batch(shadow_batch)
    cache.flat_meshes = upload_line_batch(flat_batch)
    cache.key = key


def draw_line_mesh_cache(cache: LineMeshCache, line: LineShader) -> None:
    """draw cached line meshes (combined meshes list)"""
    draw_uploaded_line_meshes(cache.meshes, line)


def draw_split_line_mesh_cache(
    cache: LineMeshCache,
    line: LineShader,
    *,
    shadow_drape: LineHeightDrape | None = None,
) -> None:
    """draw draped shadows with clip depth bias, then flat overlays"""
    if len(cache.shadow_meshes) > 0:
        set_depth_bias(line, line_shadow_depth_bias)
        set_line_height_drape(line, shadow_drape)
        draw_uploaded_line_meshes(cache.shadow_meshes, line)
        set_line_height_drape(line, None)
        set_depth_bias(line, 0.0)
    draw_uploaded_line_meshes(cache.flat_meshes, line)


def flush_line_batch(
    batch: LineBatch,
    line: LineShader,
    *,
    depth_bias: float = 0.0,
    height_drape: LineHeightDrape | None = None,
) -> None:
    """upload, draw once, and discard (uncached path)"""
    set_depth_bias(line, depth_bias)
    set_line_height_drape(line, height_drape)
    meshes: list[UploadedLineMesh] = upload_line_batch(batch)
    draw_uploaded_line_meshes(meshes, line)
    set_line_height_drape(line, None)
    set_depth_bias(line, 0.0)
    uploaded: UploadedLineMesh
    for uploaded in meshes:
        release_uploaded_line_mesh(uploaded)
