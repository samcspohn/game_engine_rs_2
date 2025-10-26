use std::{fs::File, sync::Arc};

use gltf::{Glb, Gltf, Semantic, buffer};
use parking_lot::Mutex;
use vulkano::buffer::BufferUsage;

use crate::{
    asset_manager::{Asset, DeferredAssetQueue},
    component::Scene,
    engine::Engine,
    obj_loader::{Mesh, Model},
    renderer::_RendererComponent,
    transform::{_Transform, Transform},
};

pub static mut ENGINE: Option<Arc<Engine>> = None;

fn apply_transformations(t: &Transform, scene: &Scene) {
    let guard = t.lock();
    scene
        .transform_hierarchy
        .translate_children(&guard, guard.get_position());
    scene
        .transform_hierarchy
        .rotate_children(&guard, guard.get_rotation(), guard.get_position());
    scene
        .transform_hierarchy
        .scale_children(&guard, guard.get_scale(), &guard.get_position());
    for child_id in guard.get_children() {
        let child = scene.transform_hierarchy.get_transform(*child_id);
        apply_transformations(&child, scene);
    }
}

fn recursive_access_node(
    node: &gltf::Node,
    parent: Option<u32>,
    depth: usize,
    scene: &mut crate::component::Scene,
    assets: &DeferredAssetQueue,
    buffers: &Vec<buffer::Data>,
    images: &Vec<gltf::image::Data>,
    file_path: &str,
) {
    let indent = "  ".repeat(depth);
    println!("{}Node: {:?}", indent, node.name());
    let decomposed = node.transform().decomposed();
    let t = _Transform {
        position: decomposed.0.into(),
        rotation: glam::Quat::from_array(decomposed.1),
        scale: decomposed.2.into(),
        name: node.name().unwrap_or("").to_string(),
        parent,
    };
    let entity = scene.new_entity(t);
    // if let Some(parent_id) = parent {
    //     let parent_transform = scene.transform_hierarchy.get_transform(parent_id);
    //     let parent_guard = parent_transform.lock();

    //     let node_transform = scene.transform_hierarchy.get_transform(entity.id);
    //     let guard = node_transform.lock();
    //     guard.translate_by(parent_guard.get_position());
    //     guard.rotate_by(parent_guard.get_rotation());
    //     guard.scale_by(parent_guard.get_scale());
    // }
    if let Some(mesh) = node.mesh() {
        println!("{} ^-Mesh: {:?}", indent, mesh.name());
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut tex_coords = Vec::new();
        let mut indices = Vec::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Read positions
            if let Some(positions) = reader.read_positions() {
                vertices.extend(positions);
            }

            // Read normals
            if let Some(normals_iter) = reader.read_normals() {
                normals.extend(normals_iter);
            }

            // Read texture coordinates (UV)
            if let Some(tex_coords_iter) = reader.read_tex_coords(0) {
                tex_coords.extend(tex_coords_iter.into_f32());
            } else {
            	// If no texture coordinates, fill with zeroes
				tex_coords.extend(std::iter::repeat([0.0, 0.0]).take(vertices.len()));
            }

            // Read indices
            if let Some(indices_iter) = reader.read_indices() {
                indices.extend(indices_iter.into_u32());
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
                        println!("{}    Unhandled Attribute: {:?}", indent, attribute.0);
                    }
                }
            }
        }
        let name_path = format!("{}.{}", file_path, mesh.name().unwrap_or_default());
        let model = assets
            .enqueue_work(move |a| {
                a.load_asset_custom(&name_path, move |g, a| {
                    Ok(Model {
                        meshes: vec![Mesh {
                            vertices: vertices.clone(),
                            normals: normals.clone(),
                            tex_coords: tex_coords.clone(),
                            indices: indices.clone(),
                            vertex_buffer: g
                                .enqueue_work(move |g| {
                                    g.buffer_from_iter(vertices, BufferUsage::VERTEX_BUFFER)
                                })
                                .wait()
                                .unwrap(),
                            normal_buffer: g
                                .enqueue_work(move |g| {
                                    g.buffer_from_iter(normals, BufferUsage::VERTEX_BUFFER)
                                })
                                .wait()
                                .unwrap(),
                            tex_coord_buffer: g
                                .enqueue_work(move |g| {
                                    g.buffer_from_iter(tex_coords, BufferUsage::VERTEX_BUFFER)
                                })
                                .wait()
                                .unwrap(),
                            index_buffer: g
                                .enqueue_work(move |g| {
                                    g.buffer_from_iter(indices, BufferUsage::INDEX_BUFFER)
                                })
                                .wait()
                                .unwrap(),
                            texture: Arc::new(Mutex::new(None)),
                        }],
                    })
                })
            })
            .wait()
            .unwrap();
        scene.add_component(entity, _RendererComponent { model });
    }

    for child in node.children() {
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
    }
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
        let child_t = scene.transform_hierarchy.get_transform(*child_id);
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
        println!("Importing {:?}", path.as_ref());
        let (document, buffers, images) =
            gltf::import(path.as_ref()).map_err(|e| format!("Failed to open GLTF file: {}", e))?;
        println!("{:?} import completed", path.as_ref());
        // recursive_print_json(document.as_json()., 0);
        // println!("Document {:?}", document);
        let root = document.as_json();
        // for (key, value) in root.as_object().unwrap() {
        //     println!("Key: {}", key);
        //     recursive_print_json(value, 1);
        // }
        println!("Accessors: {:?}", document.accessors().count());
        for accessor in document.accessors() {
            println!(" Accessor {:?}", accessor.count());
        }
        println!("Meshes: {:?}", document.meshes().count());
        for mesh in document.meshes() {
            println!(" Mesh {:?}", mesh.name());
            for primitive in mesh.primitives() {
                for attribute in primitive.attributes() {
                    println!("  Attribute {:?} {:?}", attribute.0, attribute.1.index());
                }
                // println!("  Primitive {:?}", primitive.attributes());
            }
            // println!(" Mesh {:?}", mesh);
            // for primitive in mesh.primitives() {
            //     println!("  Primitive {:?}", primitive);
            // }
        }
        println!("Nodes: {:?}", document.nodes().count());
        let mut scene =
            Scene::new(unsafe { (*std::ptr::addr_of!(ENGINE)).as_ref().unwrap().clone() });

        for doc_scene in document.scenes() {
            // println!("Scene {:?}", scene);
            println!(" Scene has {} nodes", doc_scene.nodes().len());
            for node in doc_scene.nodes() {
                recursive_access_node(
                    &node,
                    None,
                    0,
                    &mut scene,
                    &asset,
                    &buffers,
                    &images,
                    path.as_ref().to_str().unwrap_or(""),
                );
                // for child in node.children() {
                //     println!("  Child Node {:?}", child);
                // }
                // println!(" Node {:?}", node);
            }
        }
        apply_transformations(&scene.transform_hierarchy.get_transform(0), &scene);

        println!("Final Scene Transform Hierarchy:");
        let t = scene.transform_hierarchy.get_transform(0);
        print_scene_recurs(&t, &scene, 0);

        Ok(scene)
    }
}
