#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
// layout(location = 3) in mat4 instance_matrix;
layout(location = 3) in mat4 mvp_matrix;
layout(location = 0) out vec2 v_uv;

layout(binding = 0) uniform camera {
    mat4 view;
    mat4 proj;
} cam;

// layout(push_constant) uniform PushConstants {
//     vec3 offset;
// } push_constants;

void main() {
    gl_Position = mvp_matrix * vec4(position, 1.0);
    v_uv = uv;
}