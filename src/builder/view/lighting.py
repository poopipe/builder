"""three-point lighting via a small GLSL shader (instanced meshes)"""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    Matrix,
    Shader,
    ShaderLocationIndex,
    ShaderUniformDataType,
    Vector3,
    ffi,
    get_shader_location,
    get_shader_location_attrib,
    is_shader_valid,
    load_shader_from_memory,
    set_shader_value,
    set_shader_value_matrix,
    unload_shader,
)

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
void main()
{
    mat4 world = parentTransform * instanceTransform;
    fragPosition = vec3(world * vec4(vertexPosition, 1.0));
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(mat3(world) * vertexNormal);
    gl_Position = mvp * world * vec4(vertexPosition, 1.0);
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

struct Light {
    vec3 direction;
    vec3 color;
    float intensity;
};
uniform Light lights[3];
uniform vec3 ambient;

void main()
{
    vec4 texel = texture(texture0, fragTexCoord);
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 lighting = ambient;

    for (int i = 0; i < 3; i++) {
        // direction is travel of light rays; lighting uses the opposite
        vec3 lightDir = normalize(-lights[i].direction);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = lights[i].color * diff * lights[i].intensity;

        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = lights[i].color * spec * lights[i].intensity * 0.35;

        lighting += diffuse + specular;
    }

    finalColor = vec4(texel.rgb * colDiffuse.rgb * fragColor.rgb * lighting, texel.a * colDiffuse.a);
}
"""


def set_float(shader: Shader, loc: int, value: float) -> None:
    set_shader_value(
        shader,
        loc,
        ffi.new("float *", value),
        ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
    )


def set_vec3(shader: Shader, loc: int, value: Vector3) -> None:
    set_shader_value(shader, loc, value, ShaderUniformDataType.SHADER_UNIFORM_VEC3)


def set_matrix(shader: Shader, loc: int, value: Matrix) -> None:
    # matrices have no ShaderUniformDataType member; raylib provides a dedicated call
    set_shader_value_matrix(shader, loc, value)


@dataclass
class DirectionalLight:
    """a directional light (parallel rays). ``direction`` is ray travel direction"""

    direction: Vector3
    color: Vector3
    intensity: float


class Lighting:
    """key / fill / rim directional lights bound to an instancing-capable shader"""

    def __init__(self) -> None:
        self.shader: Shader = load_shader_from_memory(VS_SOURCE, FS_SOURCE)
        if not is_shader_valid(self.shader):
            raise RuntimeError("failed to compile lighting shader")

        self.shader.locs[ShaderLocationIndex.SHADER_LOC_MATRIX_MVP] = (
            get_shader_location(self.shader, "mvp")
        )
        self.shader.locs[ShaderLocationIndex.SHADER_LOC_VERTEX_COLOR] = (
            get_shader_location_attrib(self.shader, "vertexColor")
        )
        self.shader.locs[ShaderLocationIndex.SHADER_LOC_VERTEX_INSTANCETRANSFORM] = (
            get_shader_location_attrib(self.shader, "instanceTransform")
        )
        if self.shader.locs[ShaderLocationIndex.SHADER_LOC_MATRIX_MVP] < 0:
            unload_shader(self.shader)
            raise RuntimeError("lighting shader is missing mvp uniform")
        if (
            self.shader.locs[ShaderLocationIndex.SHADER_LOC_VERTEX_INSTANCETRANSFORM]
            < 0
        ):
            unload_shader(self.shader)
            raise RuntimeError("lighting shader is missing instanceTransform")

        self.loc_view: int = get_shader_location(self.shader, "viewPos")
        self.loc_ambient: int = get_shader_location(self.shader, "ambient")
        self.loc_parent: int = get_shader_location(self.shader, "parentTransform")
        if self.loc_view < 0 or self.loc_ambient < 0 or self.loc_parent < 0:
            unload_shader(self.shader)
            raise RuntimeError("lighting shader is missing required uniforms")

        # three-point: key (sun), fill, rim — directions are ray travel
        self.lights: list[DirectionalLight] = [
            DirectionalLight(Vector3(-0.45, -1.0, -0.35), Vector3(1.0, 0.95, 0.9), 1.0),
            DirectionalLight(Vector3(0.85, -0.35, 0.25), Vector3(0.45, 0.55, 0.85), 0.4),
            DirectionalLight(Vector3(0.1, -0.15, 1.0), Vector3(0.9, 0.9, 1.0), 0.55),
        ]
        set_vec3(self.shader, self.loc_ambient, Vector3(0.12, 0.12, 0.14))
        self.upload_lights()

    def set_parent_transform(self, matrix: Matrix) -> None:
        """bind the parent world matrix for the next instanced draw"""
        set_matrix(self.shader, self.loc_parent, matrix)

    def upload_lights(self) -> None:
        i: int
        light: DirectionalLight
        for i, light in enumerate(self.lights):
            loc_dir: int = get_shader_location(self.shader, f"lights[{i}].direction")
            loc_col: int = get_shader_location(self.shader, f"lights[{i}].color")
            loc_int: int = get_shader_location(self.shader, f"lights[{i}].intensity")
            if loc_dir < 0 or loc_col < 0 or loc_int < 0:
                raise RuntimeError(f"lighting shader is missing light[{i}] uniforms")
            set_vec3(self.shader, loc_dir, light.direction)
            set_vec3(self.shader, loc_col, light.color)
            set_float(self.shader, loc_int, light.intensity)

    def update_view_position(self, position: Vector3) -> None:
        """push the camera position into the shader for specular highlights"""
        set_vec3(self.shader, self.loc_view, position)

    def unload(self) -> None:
        """release GPU resources"""
        unload_shader(self.shader)
