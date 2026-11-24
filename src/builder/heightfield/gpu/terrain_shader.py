"""terrain vertex-displace shader (height texture + shared lighting)"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    MaterialMapIndex,
    Matrix,
    Shader,
    ShaderLocationIndex,
    ShaderUniformDataType,
    Vector2,
    Vector3,
    Vector4,
    ffi,
    get_shader_location,
    get_shader_location_attrib,
    is_shader_valid,
    load_shader_from_memory,
    set_shader_value,
    set_shader_value_matrix,
    unload_shader,
)

from builder.view.lighting import DirectionalLight, set_float, set_vec3


# flat grid in local xz; y displaced from heightMap (R32F)
# instanceTransform: translation in col3; tileUvOrigin packed in m1/m2
VS_SOURCE: str = """
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;
in mat4 instanceTransform;
out vec3 fragPosition;
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
uniform mat4 mvp;
uniform mat4 parentTransform;
uniform sampler2D heightMap;
uniform vec2 tileUvScale;
uniform vec2 worldTexelSize;
uniform float heightScale;
uniform float heightBias;

void main()
{
    vec2 tileUvOrigin = vec2(instanceTransform[0][1], instanceTransform[0][2]);
    vec2 fieldUv = tileUvOrigin + vertexTexCoord * tileUvScale;
    fieldUv = clamp(fieldUv, vec2(0.0), vec2(1.0));
    float h = texture(heightMap, fieldUv).r * heightScale + heightBias;
    vec3 localPos = vec3(vertexPosition.x, h, vertexPosition.z);

    // world-metre finite differences (same form as cpu tile normals)
    vec2 texel = 1.0 / vec2(textureSize(heightMap, 0));
    float hL = texture(heightMap, clamp(fieldUv - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))).r
        * heightScale + heightBias;
    float hR = texture(heightMap, clamp(fieldUv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))).r
        * heightScale + heightBias;
    float hD = texture(heightMap, clamp(fieldUv - vec2(0.0, texel.y), vec2(0.0), vec2(1.0))).r
        * heightScale + heightBias;
    float hU = texture(heightMap, clamp(fieldUv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0))).r
        * heightScale + heightBias;
    float spanX = max(worldTexelSize.x, 1e-6);
    float spanZ = max(worldTexelSize.y, 1e-6);
    vec3 localNormal = normalize(vec3(
        -(hR - hL) / spanX,
        1.0,
        -(hU - hD) / spanZ
    ));

    // strip packed uv so parent * instance stays a pure translate
    mat4 tileLocal = instanceTransform;
    tileLocal[0][1] = 0.0;
    tileLocal[0][2] = 0.0;
    mat4 world = parentTransform * tileLocal;
    vec4 worldPos = world * vec4(localPos, 1.0);
    fragPosition = worldPos.xyz;
    fragTexCoord = fieldUv;
    fragColor = vertexColor;
    fragNormal = normalize(mat3(world) * localNormal);
    gl_Position = mvp * world * vec4(localPos, 1.0);
}
"""

FS_SOURCE: str = """
#version 330
in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
out vec4 finalColor;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform vec3 viewPos;
uniform int useAlbedoMap;

struct Light {
    vec3 direction;
    vec3 color;
    float intensity;
};
uniform Light lights[3];
uniform vec3 ambient;

void main()
{
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 lighting = ambient;
    for (int i = 0; i < 3; i++) {
        vec3 lightDir = normalize(-lights[i].direction);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = lights[i].color * diff * lights[i].intensity;
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = lights[i].color * spec * lights[i].intensity * 0.25;
        lighting += diffuse + specular;
    }
    vec2 albedoUv = vec2(1.0 - fragTexCoord.x, 1.0 - fragTexCoord.y);
    vec3 albedo = useAlbedoMap != 0
        ? texture(texture0, albedoUv).rgb
        : vec3(0.55);
    finalColor = vec4(albedo * colDiffuse.rgb * lighting, colDiffuse.a);
}
"""


@dataclass
class TerrainShader:
    """compiled displace shader with cached uniform locations"""

    shader: Shader
    loc_view: int
    loc_ambient: int
    loc_parent: int
    loc_tile_uv_scale: int
    loc_world_texel_size: int
    loc_height_scale: int
    loc_height_bias: int
    loc_col_diffuse: int
    loc_use_albedo_map: int


def load_terrain_shader() -> TerrainShader:
    """compile the terrain displace shader"""
    shader: Shader = load_shader_from_memory(VS_SOURCE, FS_SOURCE)
    if not is_shader_valid(shader):
        raise RuntimeError("failed to compile terrain shader")
    shader.locs[ShaderLocationIndex.SHADER_LOC_MATRIX_MVP] = get_shader_location(
        shader, "mvp"
    )
    shader.locs[ShaderLocationIndex.SHADER_LOC_VERTEX_INSTANCETRANSFORM] = (
        get_shader_location_attrib(shader, "instanceTransform")
    )
    shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_HEIGHT] = get_shader_location(
        shader, "heightMap"
    )
    shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_ALBEDO] = get_shader_location(
        shader, "texture0"
    )
    loc_view: int = get_shader_location(shader, "viewPos")
    loc_ambient: int = get_shader_location(shader, "ambient")
    loc_parent: int = get_shader_location(shader, "parentTransform")
    loc_tile_uv_scale: int = get_shader_location(shader, "tileUvScale")
    loc_world_texel_size: int = get_shader_location(shader, "worldTexelSize")
    loc_height_scale: int = get_shader_location(shader, "heightScale")
    loc_height_bias: int = get_shader_location(shader, "heightBias")
    loc_col_diffuse: int = get_shader_location(shader, "colDiffuse")
    loc_use_albedo_map: int = get_shader_location(shader, "useAlbedoMap")
    if (
        shader.locs[ShaderLocationIndex.SHADER_LOC_MATRIX_MVP] < 0
        or shader.locs[ShaderLocationIndex.SHADER_LOC_VERTEX_INSTANCETRANSFORM]
        < 0
        or shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_HEIGHT] < 0
        or shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_ALBEDO] < 0
        or loc_view < 0
        or loc_ambient < 0
        or loc_parent < 0
        or loc_tile_uv_scale < 0
        or loc_world_texel_size < 0
        or loc_height_scale < 0
        or loc_height_bias < 0
        or loc_use_albedo_map < 0
    ):
        unload_shader(shader)
        raise RuntimeError("terrain shader is missing required uniforms")
    set_vec3(shader, loc_ambient, Vector3(0.12, 0.12, 0.14))
    # bind heightMap sampler to the MATERIAL_MAP_HEIGHT texture unit
    set_shader_value(
        shader,
        shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_HEIGHT],
        ffi.new("int *", int(MaterialMapIndex.MATERIAL_MAP_HEIGHT)),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    set_shader_value(
        shader,
        shader.locs[ShaderLocationIndex.SHADER_LOC_MAP_ALBEDO],
        ffi.new("int *", int(MaterialMapIndex.MATERIAL_MAP_ALBEDO)),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    set_shader_value(
        shader,
        loc_use_albedo_map,
        ffi.new("int *", 0),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )
    terrain: TerrainShader = TerrainShader(
        shader=shader,
        loc_view=loc_view,
        loc_ambient=loc_ambient,
        loc_parent=loc_parent,
        loc_tile_uv_scale=loc_tile_uv_scale,
        loc_world_texel_size=loc_world_texel_size,
        loc_height_scale=loc_height_scale,
        loc_height_bias=loc_height_bias,
        loc_col_diffuse=loc_col_diffuse if loc_col_diffuse >= 0 else -1,
        loc_use_albedo_map=loc_use_albedo_map,
    )
    bind_terrain_lights(
        terrain,
        [
            DirectionalLight(Vector3(-0.45, -1.0, -0.35), Vector3(1.0, 0.95, 0.9), 1.0),
            DirectionalLight(Vector3(0.85, -0.35, 0.25), Vector3(0.45, 0.55, 0.85), 0.4),
            DirectionalLight(Vector3(0.1, -0.15, 1.0), Vector3(0.9, 0.9, 1.0), 0.55),
        ],
    )
    return terrain


def bind_terrain_lights(
    terrain: TerrainShader, lights: list[DirectionalLight]
) -> None:
    """upload directional lights into the terrain fragment shader"""
    if len(lights) != 3:
        raise ValueError("terrain shader expects exactly 3 lights")
    i: int
    light: DirectionalLight
    for i, light in enumerate(lights):
        loc_dir: int = get_shader_location(
            terrain.shader, f"lights[{i}].direction"
        )
        loc_col: int = get_shader_location(terrain.shader, f"lights[{i}].color")
        loc_int: int = get_shader_location(
            terrain.shader, f"lights[{i}].intensity"
        )
        if loc_dir < 0 or loc_col < 0 or loc_int < 0:
            raise RuntimeError(f"terrain shader is missing light[{i}] uniforms")
        set_vec3(terrain.shader, loc_dir, light.direction)
        set_vec3(terrain.shader, loc_col, light.color)
        set_float(terrain.shader, loc_int, light.intensity)


def set_vec2(shader: Shader, loc: int, value: Vector2) -> None:
    set_shader_value(shader, loc, value, ShaderUniformDataType.SHADER_UNIFORM_VEC2)


def set_vec4(shader: Shader, loc: int, value: Vector4) -> None:
    set_shader_value(shader, loc, value, ShaderUniformDataType.SHADER_UNIFORM_VEC4)


def set_int(shader: Shader, loc: int, value: int) -> None:
    set_shader_value(
        shader,
        loc,
        ffi.new("int *", int(value)),
        ShaderUniformDataType.SHADER_UNIFORM_INT,
    )


def set_parent_transform(terrain: TerrainShader, matrix: Matrix) -> None:
    """bind the heightfield group world matrix for instanced tile draws"""
    set_shader_value_matrix(terrain.shader, terrain.loc_parent, matrix)


def unload_terrain_shader(terrain: TerrainShader) -> None:
    """release the terrain shader"""
    unload_shader(terrain.shader)
