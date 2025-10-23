use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

use parking_lot::Mutex;
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::{
    asset_manager::{Asset, AssetHandle, DeferredAssetQueue},
    gpu_manager::{GPUManager, GPUWorkQueue},
    texture::Texture,
};

// static OBJ_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
#[derive(Debug, Clone)]
pub struct Model {
    // pub id: u32,
    pub meshes: Vec<Mesh>,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub indices: Vec<u32>,

    // buffers:
    pub vertex_buffer: Subbuffer<[[f32; 3]]>,
    pub normal_buffer: Subbuffer<[[f32; 3]]>,
    pub tex_coord_buffer: Subbuffer<[[f32; 2]]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub texture: Arc<Mutex<Option<AssetHandle<Texture>>>>,
}

impl Asset for Model {
    /// Load an OBJ file from the given path
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn load_from_file(
        path: impl AsRef<Path>,
        gpu: GPUWorkQueue,
        asset: DeferredAssetQueue,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let (models, _materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )
        .map_err(|e| format!("Failed to load OBJ file: {}", e))?;
        let materials = _materials.expect("Failed to load materials");

        let mut meshes = Vec::new();
        // Process all models in the OBJ file
        for model in models {
            let mut vertices = Vec::new();
            let mut normals = Vec::new();
            let mut tex_coords = Vec::new();
            let mut indices = Vec::new();

            let mesh = model.mesh;
            // let index_offset = vertices.len() as u32;

            // Process vertices (positions)
            for i in (0..mesh.positions.len()).step_by(3) {
                vertices.push([
                    mesh.positions[i],
                    mesh.positions[i + 1],
                    mesh.positions[i + 2],
                ]);
            }

            // Process normals
            if !mesh.normals.is_empty() {
                for i in (0..mesh.normals.len()).step_by(3) {
                    normals.push([mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]]);
                }
            } else {
                // If no normals provided, fill with default up vectors
                for _ in 0..vertices.len() {
                    normals.push([0.0, 1.0, 0.0]);
                }
            }

            // Process texture coordinates
            if !mesh.texcoords.is_empty() {
                for i in (0..mesh.texcoords.len()).step_by(2) {
                    tex_coords.push([mesh.texcoords[i], mesh.texcoords[i + 1]]);
                }
            } else {
                // If no texture coordinates provided, fill with default values
                for _ in 0..vertices.len() {
                    tex_coords.push([0.0, 0.0]);
                }
            }

            // Process indices with offset for multiple models
            for index in mesh.indices {
                indices.push(index);
            }

            let verts = vertices.clone();
            let norms = normals.clone();
            let texs = tex_coords.clone();
            let inds = indices.clone();

            println!("Creating GPU buffers for OBJ");
            let vertex_buffer = gpu
                .enqueue_work(move |g| {
                    g.buffer_from_iter(verts.iter().cloned(), BufferUsage::VERTEX_BUFFER)
                })
                .wait()
                .unwrap();
            println!("Created vertex buffer");
            let normal_buffer = gpu
                .enqueue_work(move |g| {
                    g.buffer_from_iter(norms.iter().cloned(), BufferUsage::VERTEX_BUFFER)
                })
                .wait()
                .unwrap();
            println!("Created normal buffer");
            let tex_coord_buffer = gpu
                .enqueue_work(move |g| {
                    g.buffer_from_iter(texs.iter().cloned(), BufferUsage::VERTEX_BUFFER)
                })
                .wait()
                .unwrap();
            println!("Created tex coord buffer");
            let index_buffer = gpu
                .enqueue_work(move |g| {
                    g.buffer_from_iter(inds.iter().cloned(), BufferUsage::INDEX_BUFFER)
                })
                .wait()
                .unwrap();
            println!("Created index buffer");

            let texture = Arc::new(Mutex::new(None));
            if let Some(mat_id) = mesh.material_id.as_ref() {
                if let Some(material) = materials.get(*mat_id) {
                    if let Some(diffuse_texture) = material.diffuse_texture.clone() {
                        println!("Loading diffuse texture for mesh: {:?}", diffuse_texture);
                        let tex_path = if Path::new(&diffuse_texture).is_absolute() {
                            diffuse_texture
                        } else {
                            let mut p = path.parent().unwrap_or(Path::new("")).to_path_buf();
                            p.push(diffuse_texture);
                            p.to_string_lossy().to_string()
                        };
                        let texture_clone = Arc::clone(&texture);
                        asset.enqueue_work(move |a| {
                            let handle = a.load_asset::<Texture>(&tex_path, None);
                            *texture_clone.lock() = Some(handle.clone());
                            handle
                        });
                    } else {
                        println!("No diffuse texture for material {}", mat_id);
                    }
                } else {
                    println!("Invalid material_id {} for mesh", mat_id);
                }
            } else {
                println!("No material assigned to mesh");
            }
            let mesh = Mesh {
                vertices,
                normals,
                tex_coords,
                indices,
                vertex_buffer,
                normal_buffer,
                tex_coord_buffer,
                index_buffer,
                texture,
            };
            meshes.push(mesh);
        }

        Ok(Self { meshes })
    }

    // /// Create an empty Obj
    // pub fn new() -> Self {
    //     Self {
    //         vertices: Vec::new(),
    //         normals: Vec::new(),
    //         tex_coords: Vec::new(),
    //         indices: Vec::new(),
    //     }
    // }

    // /// Get the number of vertices
    // pub fn vertex_count(&self) -> usize {
    //     self.vertices.len()
    // }

    // /// Get the number of indices
    // pub fn index_count(&self) -> usize {
    //     self.indices.len()
    // }

    // /// Get the number of triangles
    // pub fn triangle_count(&self) -> usize {
    //     self.indices.len() / 3
    // }
}

impl Model {
    pub fn default(gpu: &GPUManager) -> Self {
        let vertices = vec![[0.0, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let tex_coords = vec![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = vec![0, 1, 2];

        let vertex_buffer =
            gpu.buffer_from_iter(vertices.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        let normal_buffer =
            gpu.buffer_from_iter(normals.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        let tex_coord_buffer =
            gpu.buffer_from_iter(tex_coords.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        let index_buffer = gpu.buffer_from_iter(indices.iter().cloned(), BufferUsage::INDEX_BUFFER);
        let mesh = Mesh {
            vertices,
            normals,
            tex_coords,
            indices,
            vertex_buffer,
            normal_buffer,
            tex_coord_buffer,
            index_buffer,
            texture: Arc::new(Mutex::new(None)),
        };
        Self { meshes: vec![mesh] }
    }
}
