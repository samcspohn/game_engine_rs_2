use std::{fs::File, sync::Arc};

use glam::Vec3;
use gltf::{Glb, Gltf, Semantic, buffer};
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::{
    asset_manager::{Asset, DeferredAssetQueue},
    component::Scene,
    engine::Engine,
    obj_loader::{self, MESH_BUFFERS, Mesh, Model, pack_normal, pack_tangent},
    renderer::_RendererComponent,
    texture::Texture,
    transform::{_Transform, Transform},
};

pub static mut ENGINE: Option<Arc<Engine>> = None;

fn recursive_access_node(
    node: &gltf::Node,
    parent: Option<u32>,
    depth: usize,
    scene: &Mutex<crate::component::Scene>,
    assets: &DeferredAssetQueue,
    buffers: &Vec<buffer::Data>,
    images: &Vec<gltf::image::Data>,
    file_path: &str,
) {
    let indent = "  ".repeat(depth);
    // println!("{}Node: {:?}", indent, node.name());
    let decomposed = node.transform().decomposed();

    // Local transform from GLTF
    let local_position: Vec3 = decomposed.0.into();
    let local_rotation = glam::Quat::from_array(decomposed.1);
    let local_scale: Vec3 = decomposed.2.into();

    // // Compute global transform by combining with parent
    // let (global_position, global_rotation, global_scale) = if let Some(parent_id) = parent {
    //     let scene = scene.lock();
    //     let parent_transform = scene.transform_hierarchy.get_transform_unchecked(parent_id);
    //     let parent_guard = parent_transform.lock();

    //     let parent_pos = parent_guard.get_position();
    //     let parent_rot = parent_guard.get_rotation();
    //     let parent_scale = parent_guard.get_scale();

    //     // Compute global transform: GlobalChild = ParentGlobal * LocalChild
    //     let global_scale = parent_scale * local_scale;
    //     let global_rotation = parent_rot * local_rotation;
    //     let global_position = parent_pos + parent_rot * (local_position * parent_scale);

    //     (global_position, global_rotation, global_scale)
    // } else {
    //     // Root node - local transform IS the global transform
    //     (local_position, local_rotation, local_scale)
    // };

    // let t = _Transform {
    //     position: global_position,
    //     rotation: global_rotation,
    //     scale: global_scale,
    //     name: node.name().unwrap_or("").to_string(),
    //     parent: Some(parent.unwrap_or(0)),
    // };
    let t = _Transform {
        position: local_position,
        rotation: local_rotation,
        scale: local_scale,
        name: node.name().unwrap_or("").to_string(),
        parent: Some(parent.unwrap_or(0)),
    };
    let entity = scene.lock().new_entity(t);
    let mut meshes = Vec::new();
    if let Some(mesh) = node.mesh() {
        // println!("{} ^-Mesh: {:?}", indent, mesh.name());

        // let mut texture = None;
        for primitive in mesh.primitives() {
            let mut _mesh = Mesh::new();
            // Extract base color texture (diffuse)
            let mut mat = crate::obj_loader::Material::default();
            let gltf_material = primitive.material();
            let texture_index = gltf_material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| {
                    let texture = info.texture();
                    let image = texture.source();
                    image.index()
                });
            mat.base_color = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_factor();
            if mat.base_color[3] < 0.01 {
                println!(
                    "{}    Warning: Base color alpha is less than 0.01 ({})",
                    indent,
                    mesh.name().unwrap_or("")
                );
            }
            // let alpha = primitive.material().
            let _base_color_texture = texture_index.and_then(|idx| images.get(idx as usize));
            let name_path = format!(
                "{}._tex_{}",
                file_path,
                // mesh.name().unwrap_or(""),
                texture_index.unwrap_or(0)
            );
            mat.albedo_texture = if let Some(tex) = _base_color_texture {
                let tex = tex.clone();
                Some(Texture::load_custom(assets, name_path, tex))
            } else {
                None
            };
            // Extract normal texture+
            let _normal_texture = primitive
                .material()
                .normal_texture()
                .map(|info| {
                    let texture = info.texture();
                    let image = texture.source();
                    image.index()
                })
                .and_then(|idx| images.get(idx as usize));
            let name_path = format!(
                "{}._normtex_{}",
                file_path,
                // mesh.name().unwrap_or(""),
                texture_index.unwrap_or(0)
            );
            mat.normal_texture = if let Some(tex) = _normal_texture {
                let tex = tex.clone();
                Some(Texture::load_custom(assets, name_path, tex))
            } else {
                None
            };

            let metalic_roughness = gltf_material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(|info| {
                    let texture = info.texture();
                    let image = texture.source();
                    image.index()
                })
                .and_then(|idx| images.get(idx as usize));
            let name_path = format!(
                "{}._mrtex_{}",
                file_path,
                // mesh.name().unwrap_or(""),
                texture_index.unwrap_or(0)
            );
            mat.metallic_roughness_texture = if let Some(tex) = metalic_roughness {
                let tex = tex.clone();
                Some(Texture::load_custom(assets, name_path, tex))
            } else {
                None
            };

            _mesh.material = Some(mat);

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            // Read positions
            if let Some(positions) = reader.read_positions() {
                _mesh.vertices.extend(positions);
            }
            // Read indices
            if let Some(indices_iter) = reader.read_indices() {
                _mesh.indices.extend(indices_iter.into_u32());
            }
            // Read normals
            if let Some(normals_iter) = reader.read_normals() {
                for normal in normals_iter {
                    _mesh.normals.push(pack_normal(normal));
                }
            }
            // Read texture coordinates (UV)
            if let Some(tex_coords_iter) = reader.read_tex_coords(0) {
                _mesh.tex_coords.extend(tex_coords_iter.into_f32());
            } else {
                // If no texture coordinates, fill with zeroes
                // let new_vertex_count = vertices.len() - base_vertex as usize;
                _mesh
                    .tex_coords
                    .extend(std::iter::repeat([0.0, 0.0]).take(_mesh.vertices.len()));
            }

            let mut tangents_f32 = Vec::new();
            if let Some(tangents_iter) = reader.read_tangents() {
                tangents_f32.extend(tangents_iter);
            } else {
                // If no tangents, calculate real values
                // tangents_f32
                //     .extend(std::iter::repeat([0.0, 0.0, 0.0, 0.0]).take(_mesh.vertices.len()));
                for &packed_normal in &_mesh.normals {
					// Unpack the normal from u32
					let x = ((packed_normal & 0xFF) as u8 as i8) as f32 / 127.0;
					let y = (((packed_normal >> 8) & 0xFF) as u8 as i8) as f32 / 127.0;
					let z = (((packed_normal >> 16) & 0xFF) as u8 as i8) as f32 / 127.0;
					let normal = Vec3::new(x, y, z);
					// Simple orthogonal vector
					let tangent = if normal.x.abs() > normal.z.abs() {
						Vec3::new(-normal.y, normal.x, 0.0)
					} else {
						Vec3::new(0.0, -normal.z, normal.y)
					}
					.normalize();
					tangents_f32.push([tangent.x, tangent.y, tangent.z, 1.0]);
				}
                // for f in _mesh.indices.chunks(3) {
      //               let v0 = Vec3::from(_mesh.vertices[f[0] as usize]);
      //               let v1 = Vec3::from(_mesh.vertices[f[1] as usize]);
      //               let v2 = Vec3::from(_mesh.vertices[f[2] as usize]);

      //               let edge1 = v1 - v0;
      //               let edge2 = v2 - v0;
      //               let tangent = [
      //               	edge1.x + edge2.x,
						// edge1.y + edge2.y,
						// edge1.z + edge2.z,
						// 1.0,
      //               ];

     //  				let tangent = if

     //                for &idx in f {
					// 	tangents_f32[idx as usize] = tangent;
					// }
                    // // UVs are Vec2, not Vec3!
                    // let uv0 = glam::Vec2::from(_mesh.tex_coords[f[0] as usize]);
                    // let uv1 = glam::Vec2::from(_mesh.tex_coords[f[1] as usize]);
                    // let uv2 = glam::Vec2::from(_mesh.tex_coords[f[2] as usize]);

                    // let delta_pos1 = v1 - v0;
                    // let delta_pos2 = v2 - v0;

                    // let delta_uv1 = uv1 - uv0;
                    // let delta_uv2 = uv2 - uv0;

                    // // Calculate the determinant of the UV matrix
                    // let det = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;
                    // let r = 1.0 / det.abs().max(0.0001);

                    // // Calculate tangent using the corrected formula
                    // let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;

                    // // Accumulate tangents for each vertex (for smooth tangents across shared vertices)
                    // for &idx in f {
                    //     let t = &mut tangents_f32[idx as usize];
                    //     t[0] = tangent.x;
                    //     t[1] = tangent.y;
                    //     t[2] = tangent.z;
                    //     t[3] = 1.0; // Handedness (sign of determinant)
                    // }
                // }
            }

            // Convert f32 tangents to packed u32 format
            for tangent in tangents_f32 {
                _mesh.tangents.push(pack_tangent(tangent));
            }



            if let Some(colors_iter) = reader.read_colors(0) {
                _mesh.colors.extend(colors_iter.into_rgba_u8())
            } else {
                _mesh
                    .colors
                    .extend(std::iter::repeat([255, 255, 255, 255]).take(_mesh.vertices.len()));
            }

            for attribute in primitive.attributes() {
                // println!("{}    Attribute {:?} {:?}", indent, attribute.0, attribute.1.index());
                match attribute.0 {
                    Semantic::Positions => {
                        // Already handled above
                    }
                    Semantic::Normals => {
                        // Already handled above
                    }
                    Semantic::TexCoords(0) => {
                        // Already handled above
                    }
                    Semantic::Extras(extra) => {
                        println!("{}    Extra Attribute: {:?}", indent, extra);
                    }
                    _ => {
                        // println!("{}    Unhandled Attribute: {:?}", indent, attribute.0);
                    }
                }
            }
            _mesh.aabb.min = primitive.bounding_box().min.into();
            _mesh.aabb.max = primitive.bounding_box().max.into();

           // _mesh.aabb = obj_loader::AABB {
           //      min: [
           //          _mesh.vertices.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min),
           //          _mesh.vertices.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min),
           //          _mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min),
           //      ],
           //      max: [
           //          _mesh.vertices
           //              .iter()
           //              .map(|v| v[0])
           //              .fold(f32::NEG_INFINITY, f32::max),
           //          _mesh.vertices
           //              .iter()
           //              .map(|v| v[1])
           //              .fold(f32::NEG_INFINITY, f32::max),
           //          _mesh.vertices
           //              .iter()
           //              .map(|v| v[2])
           //              .fold(f32::NEG_INFINITY, f32::max),
           //      ],
           //  };

            meshes.push(_mesh);
        }
        let name_path = format!("{}.{}", file_path, mesh.name().unwrap_or_default(),);
        let model = assets
            .enqueue_work(move |a| {
                a.load_asset_custom(&name_path, move |g, a| {
                    for m in &mut meshes {
                        let (vertex_offset, index_offset) =
                            if let Some(buffers) = MESH_BUFFERS.lock().as_mut() {
                                buffers.add_mesh(
                                    &m.vertices,
                                    &m.normals,
                                    &m.tangents,
                                    &m.tex_coords,
                                    &m.colors,
                                    &m.indices,
                                )
                            } else {
                                panic!("MESH_BUFFERS not initialized");
                            };
                        m.vertex_offset = vertex_offset;
                        m.index_offset = index_offset;
                    }
                    Ok(Model { meshes: meshes })
                })
            })
            .wait()
            .unwrap();

        scene
            .lock()
            .add_component(entity, _RendererComponent { model });
    }

    // for child in node.children() {
    //     recursive_access_node(
    //         &child,
    //         Some(entity.id),
    //         depth + 1,
    //         scene,
    //         assets,
    //         buffers,
    //         images,
    //         file_path,
    //     );
    // }
    node.children().par_bridge().for_each(|child| {
        recursive_access_node(
            &child,
            Some(entity.id),
            depth + 1,
            scene,
            assets,
            buffers,
            images,
            file_path,
        );
    });
}
// fn recursive_print_json(value: &serde_json::Value, depth: usize) {
//     let indent = "  ".repeat(depth);
//     match value {
//         serde_json::Value::Object(map) => {
//             for (key, val) in map {
//                 println!("{}Key: {}", indent, key);
//                 recursive_print_json(val, depth + 1);
//             }
//         }
//         serde_json::Value::Array(arr) => {
//             for (index, val) in arr.iter().enumerate() {
//                 println!("{}Index: {}", indent, index);
//                 recursive_print_json(val, depth + 1);
//             }
//         }
//         _ => {
//             println!("{}Value: {}", indent, value);
//         }
//     }
// }

fn print_scene_recurs(t: &Transform, scene: &Scene, depth: usize) {
    let indent = "  ".repeat(depth);
    let _lock = t.lock();
    println!("{}{}", indent, _lock.get_name());
    println!("{}Position: {:?}", indent, _lock.get_position());
    for child_id in _lock.get_children() {
        let child_t = scene.transform_hierarchy.get_transform_unchecked(*child_id);
        print_scene_recurs(&child_t, scene, depth + 1);
    }
}

impl Asset for Scene {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn load_from_file(
        path: impl AsRef<std::path::Path>,
        gpu: crate::gpu_manager::GPUWorkQueue,
        asset: crate::asset_manager::DeferredAssetQueue,
    ) -> Result<Self, String>
    where
        Self: Sized,
    {
        println!("Loading {:?}", path.as_ref());
        let (document, buffers, images) =
            gltf::import(path.as_ref()).map_err(|e| format!("Failed to open GLTF file: {}", e))?;
        println!("{:?} load completed", path.as_ref());
        // recursive_print_json(document.as_json()., 0);
        // println!("Document {:?}", document);
        let root = document.as_json();
        // for (key, value) in root.as_object().unwrap() {
        //     println!("Key: {}", key);
        //     recursive_print_json(value, 1);
        // }
        // println!("Accessors: {:?}", document.accessors().count());
        // for accessor in document.accessors() {
        //     println!(" Accessor {:?}", accessor.count());
        // }
        // println!("Meshes: {:?}", document.meshes().count());
        // for mesh in document.meshes() {
        //     println!(" Mesh {:?}", mesh.name());
        //     for primitive in mesh.primitives() {
        //         for attribute in primitive.attributes() {
        //             println!("  Attribute {:?} {:?}", attribute.0, attribute.1.index());
        //         }
        //         // println!("  Primitive {:?}", primitive.attributes());
        //     }
        //     // println!(" Mesh {:?}", mesh);
        //     // for primitive in mesh.primitives() {
        //     //     println!("  Primitive {:?}", primitive);
        //     // }
        // }
        // println!("Nodes: {:?}", document.nodes().count());
        let mut scene =
            Scene::new(unsafe { (*std::ptr::addr_of!(ENGINE)).as_ref().unwrap().clone() });
        let root_entity = scene.new_entity(_Transform::default());
        // let _root_transform = scene
        //     .transform_hierarchy
        //     .get_transform_unchecked(root_entity.id);
        let scene = Mutex::new(scene);

        for doc_scene in document.scenes() {
            // println!("Scene {:?}", scene);
            println!(" Scene has {} nodes", doc_scene.nodes().len());
            let _path = path.as_ref().to_path_buf();
            doc_scene.nodes().par_bridge().for_each(|node| {
                recursive_access_node(
                    &node,
                    None,
                    0,
                    &scene,
                    &asset,
                    &buffers,
                    &images,
                    _path.to_str().unwrap_or(""),
                );
            });
            // for child in node.children() {
            //     println!("  Child Node {:?}", child);
            // }
            // println!(" Node {:?}", node);
        }
        // Transformations are now applied during node creation, so we don't need
        // to recursively apply them after the fact

        // println!("Final Scene Transform Hierarchy:");
        // let t = scene.transform_hierarchy.get_transform_unchecked(0);
        // print_scene_recurs(&t, &scene, 0);

        Ok(Mutex::into_inner(scene))
    }
}
