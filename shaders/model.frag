 #version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;
layout(binding = 1) uniform sampler2D tex_sampler;

void main() {
    f_color = texture(tex_sampler, v_uv);
    // f_color = vec4(1.0, 0.0, 0.0, 1.0);
}