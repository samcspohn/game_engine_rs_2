#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;
layout(binding = 1) uniform sampler2D tex_sampler;

void main() {
    vec4 color = texture(tex_sampler, v_uv);
    if (color.a < 0.1) {
        discard;
    }
    f_color = color;
    // f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
