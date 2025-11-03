#version 460
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 normal;
layout(location = 3) in vec4 tangent;
layout(location = 4) in vec4 color;
layout(location = 5) in uvec2 rd;

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
layout(location = 6) out mat3 TBN; // 7, 8
layout(location = 9) out vec4 v_tangent;

layout(binding = 0) uniform camera {
    mat4 view;
    mat4 proj;
} cam;

struct Material {
    uint albedo_tex_index;
    uint normal_tex_index;
    uint metallic_roughness_tex_index;
    uint specular_tex_index;
    //
    vec4 base_color;
    // uint metallic_roughness_tex_index;
    // vec4 base_color_factor;
    // float metallic_factor;
    // float roughness_factor;
};
layout(set = 0, binding = 2) buffer Materials {
    Material materials[];
};

// layout(push_constant) uniform PushConstants {
//     vec3 offset;
// } push_constants;

void main() {
    gl_Position = m[rd.x].mvp * vec4(position, 1.0);
    v_uv = uv;
    mat_id = rd.y;
    v_color = color;
    v_position = (m[rd.x].model * vec4(position, 1.0)).xyz;
    cam_pos = (inverse(cam.view) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    // v_normal = normalize(m[rd.x].normal * normal.xyz);
    v_tangent = tangent;

    if (materials[mat_id].normal_tex_index == uint(-1)) {
        v_normal = normalize(m[rd.x].normal * normal.xyz);
        return;
    }
    // vec3 tangent;
    // vec3 bitangent;
    // // Generate tangent and bitangent vectors
    // if (abs(normal.x) > abs(normal.z)) {
    //     tangent = normalize(cross(vec3(0.0, 1.0, 0.0), v_normal));
    // } else {
    //     tangent = normalize(cross(vec3(1.0, 0.0, 0.0), v_normal));
    // }
    // bitangent = normalize(cross(v_normal, tangent));
    // TBN = mat3(tangent, bitangent, v_normal);
    vec3 T = normalize(vec3(m[rd.x].model * vec4(tangent.xyz, 0.0)));
    vec3 N = normalize(vec3(m[rd.x].model * vec4(normal.xyz, 0.0)));
    v_normal = N;
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(v_normal, T) * tangent.w;
    TBN = mat3(T, B, N);
}
