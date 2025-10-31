#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) flat in uint mat_id;
layout(location = 3) in vec3 v_normal;
layout(location = 4) in vec3 frag_pos;
layout(location = 5) in vec3 cam_pos;

layout(location = 0) out vec4 f_color;
// layout(binding = 1) uniform sampler2D tex_sampler;
layout(set = 0, binding = 2) uniform sampler2D textures[];

const vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));

void main() {
    vec4 color = texture(textures[mat_id], v_uv) * v_color;
    if (color.a < 0.1) {
        discard;
    }
    float light_intensity = max(dot(v_normal, light_dir), 0.0) + 0.3;
    float cam_distance = length(cam_pos - frag_pos);
    // light_intensity += 1.5 / (1.0 + 0.1 * cam_distance + 0.01 * cam_distance * cam_distance); // * dot(normalize(v_normal), normalize(-cam_pos));
    color.rgb *= light_intensity;
    f_color = color;
    // f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
