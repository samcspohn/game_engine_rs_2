#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_ARB_shader_draw_parameters : require

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

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) flat in uint mat_id;
layout(location = 3) in vec3 v_normal;
layout(location = 4) in vec3 frag_pos;
layout(location = 5) in vec3 cam_pos;
layout(location = 6) in mat3 v_tbn;
layout(location = 9) in vec4 v_tangent;

layout(location = 0) out vec4 f_color;
// layout(binding = 1) uniform sampler2D tex_sampler;
layout(set = 0, binding = 2) buffer Materials {
    Material materials[];
};
layout(set = 0, binding = 3) uniform sampler2D textures[];

const vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));

// function to calculate specular highlight
float calculate_specular(vec3 light_dir, vec3 view_dir, vec3 normal, float shininess) {
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    return spec;
}

void main() {
    // vec4 color = texture(textures[albedo], v_uv) * v_color;
    if (mat_id == uint(-1)) {
        float light_intensity = max(dot(v_normal, light_dir), 0.0) + 0.3;
        float cam_distance = length(cam_pos - frag_pos);
        // light_intensity += 1.5 / (1.0 + 0.1 * cam_distance + 0.01 * cam_distance * cam_distance); // * dot(normalize(v_normal), normalize(-cam_pos));
        f_color = v_color * light_intensity;
        return;
    }
    // vec3 base_color = materials[mat_id].base_color.rgb * v_color.rgb;
    vec4 color = v_color * materials[mat_id].base_color;
    uint albedo = materials[mat_id].albedo_tex_index;
    if (albedo != uint(-1)) {
        color *= texture(textures[albedo], v_uv);
    }
    if (color.a < 0.1) {
        discard;
    }
    vec3 base_color = color.rgb;
    uint normal_map = materials[mat_id].normal_tex_index;
    vec3 normal = v_normal;
    if (normal_map != uint(-1)) {
        // vec3 normal_tex = texture(textures[normal_map], v_uv).rgb;
        // vec3 mapped_normal = normalize(normal_tex * 2.0 - 1.0);
        // // Transform the normal from tangent space to world space
        // // v_normal is already multiplied by the model matrix's normal matrix
        // vec3 tangent;
        // vec3 bitangent;
        // // Generate tangent and bitangent vectors
        // if (abs(normal.x) > abs(normal.z)) {
        //     tangent = normalize(cross(vec3(0.0, 1.0, 0.0), v_normal));
        // } else {
        //     tangent = normalize(cross(vec3(1.0, 0.0, 0.0), v_normal));
        // }
        // bitangent = normalize(cross(v_normal, tangent));
        // mat3 TBN = mat3(tangent, bitangent, v_normal);
        // vec3 world_normal = normalize(TBN * mapped_normal);
        // normal = world_normal;

        vec3 normal_tex = texture(textures[normal_map], v_uv).rgb;
        vec3 mapped_normal = normalize(normal_tex * 2.0 - 1.0);
        normal = normalize(v_tbn * mapped_normal);
    }
    uint spec = materials[mat_id].specular_tex_index;
    if (spec != uint(-1)) {
        float specular = texture(textures[spec], v_uv).r;
        color.rgb += base_color * calculate_specular(light_dir, normalize(cam_pos - frag_pos), normal, specular * 96.0);
    }
    if (color.a < 0.1) {
        discard;
    }

    // Calculate diffuse lighting using the correct normal
    float light_intensity = max(dot(normal, light_dir), 0.0) + 0.3;
    float cam_distance = length(cam_pos - frag_pos);
    // light_intensity += 1.5 / (1.0 + 0.1 * cam_distance + 0.01 * cam_distance * cam_distance); // * dot(normalize(v_normal), normalize(-cam_pos));
    color.rgb += light_intensity * base_color;
    vec3 cam_light_dir = normalize(cam_pos - frag_pos);

    // light_intensity = 1.0 / (1.0 + 0.1 * cam_distance + 0.01 * cam_distance * cam_distance) * max(dot(normal, cam_light_dir), 0.0);
    // color.rgb += light_intensity * base_color;

    // Add specular highlight after diffuse lighting
    uint mr = materials[mat_id].metallic_roughness_tex_index;
    if (mr != uint(-1)) {
        vec2 mr_tex = texture(textures[mr], v_uv).rg;
        // float metallic = mr_tex.b * materials[mat_id].metallic_factor;
        // float roughness = mr_tex.g * materials[mat_id].roughness_factor;
        // Use metallic and roughness values as needed
        // For simplicity, we won't implement full PBR here, instead simulate specular highlight with roughness
        float roughness = mr_tex.g;
        // Convert roughness to shininess: lower roughness = higher shininess (sharper highlights)
        float shininess = mix(128.0, 4.0, roughness);
        float specular = calculate_specular(light_dir, normalize(cam_pos - frag_pos), normal, shininess);
        color.rgb += base_color * specular;
    }

    f_color = color;
    // f_color = vec4(v_normal, 1.0);
    // f_color = v_tangent;
    // f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
