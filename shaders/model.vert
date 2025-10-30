#version 460
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 color;
layout(location = 4) in uvec2 rd;
// layout(location = 3) in mat4 instance_matrix;
struct MatrixData {
    mat4 model;
    mat3 normal;
    mat4 mvp;
};

// layout(set = 1, binding = 5) buffer DirtyL2 { uint dirty_l2[]; };
layout(set = 0, binding = 1) buffer _MatrixData {
    MatrixData m[];
};

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) flat out uint mat_id;
layout(location = 3) out vec3 v_normal;
layout(location = 4) out vec3 v_position;
layout(location = 5) out vec3 cam_pos;

layout(binding = 0) uniform camera {
    mat4 view;
    mat4 proj;
} cam;

// layout(push_constant) uniform PushConstants {
//     vec3 offset;
// } push_constants;

void main() {
    gl_Position = m[rd.x].mvp * vec4(position, 1.0);
    v_uv = uv;
    mat_id = rd.y;
    v_color = color;
    v_normal = m[rd.x].normal * normal;
    v_position = (m[rd.x].model * vec4(position, 1.0)).xyz;
    cam_pos = (inverse(cam.view) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
}
