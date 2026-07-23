"""Three-point lighting via a small GLSL shader."""

from __future__ import annotations

from dataclasses import dataclass

from pyray import (
    Shader,
    ShaderUniformDataType,
    Vector3,
    ffi,
    get_shader_location,
    load_shader_from_memory,
    set_shader_value,
    unload_shader,
)

_VS = """
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;
out vec3 fragPosition;
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;
void main()
{
    fragPosition = vec3(matModel * vec4(vertexPosition, 1.0));
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 1.0)));
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

_FS = """
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
    vec3 position;
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
        vec3 lightDir = normalize(lights[i].position - fragPosition);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = lights[i].color * diff * lights[i].intensity;

        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = lights[i].color * spec * lights[i].intensity * 0.35;

        float dist = length(lights[i].position - fragPosition);
        float atten = 1.0 / (1.0 + 0.02 * dist + 0.001 * dist * dist);
        lighting += (diffuse + specular) * atten;
    }

    finalColor = vec4(texel.rgb * colDiffuse.rgb * fragColor.rgb * lighting, texel.a * colDiffuse.a);
}
"""


def _set_float(shader: Shader, loc: int, value: float) -> None:
    set_shader_value(
        shader,
        loc,
        ffi.new("float *", value),
        ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
    )


def _set_vec3(shader: Shader, loc: int, value: Vector3) -> None:
    set_shader_value(shader, loc, value, ShaderUniformDataType.SHADER_UNIFORM_VEC3)


@dataclass
class PointLight:
    """A simple point light for the three-point setup."""

    position: Vector3
    color: Vector3
    intensity: float


class Lighting:
    """Key / fill / rim lights bound to a shader."""

    def __init__(self) -> None:
        self.shader = load_shader_from_memory(_VS, _FS)
        self._loc_view = get_shader_location(self.shader, "viewPos")
        self._loc_ambient = get_shader_location(self.shader, "ambient")
        self.lights = [
            PointLight(Vector3(6.0, 8.0, 4.0), Vector3(1.0, 0.95, 0.9), 1.0),
            PointLight(Vector3(-5.0, 3.0, 2.0), Vector3(0.45, 0.55, 0.85), 0.55),
            PointLight(Vector3(0.0, 4.0, -6.0), Vector3(0.9, 0.9, 1.0), 0.7),
        ]
        _set_vec3(self.shader, self._loc_ambient, Vector3(0.12, 0.12, 0.14))
        self._upload_lights()

    def _upload_lights(self) -> None:
        for i, light in enumerate(self.lights):
            loc_pos = get_shader_location(self.shader, f"lights[{i}].position")
            loc_col = get_shader_location(self.shader, f"lights[{i}].color")
            loc_int = get_shader_location(self.shader, f"lights[{i}].intensity")
            _set_vec3(self.shader, loc_pos, light.position)
            _set_vec3(self.shader, loc_col, light.color)
            _set_float(self.shader, loc_int, light.intensity)

    def update_view_position(self, position: Vector3) -> None:
        """Push the camera position into the shader for specular highlights."""
        _set_vec3(self.shader, self._loc_view, position)

    def unload(self) -> None:
        """Release GPU resources."""
        unload_shader(self.shader)
