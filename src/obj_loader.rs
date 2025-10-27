use std::{
    path::Path,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

use parking_lot::Mutex;
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::{
    asset_manager::{Asset, AssetHandle, DeferredAssetQueue},
    gpu_manager::{GPUManager, GPUWorkQueue, gpu_vec::GpuVec},
    texture::Texture,
};

// static OBJ_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
#[derive(Debug, Clone)]
pub struct Model {
    // pub id: u32,
    pub meshes: Vec<Mesh>,
}

pub struct MeshBuffers {
    pub vertex_buffer: GpuVec<[f32; 3]>,
    pub normal_buffer: GpuVec<[f32; 3]>,
    pub tex_coord_buffer: GpuVec<[f32; 2]>,
    pub index_buffer: GpuVec<u32>,
}

impl MeshBuffers {
    pub fn new(gpu: &GPUManager) -> Self {
        Self {
            vertex_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
            normal_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
            tex_coord_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
            index_buffer: GpuVec::new(gpu, BufferUsage::INDEX_BUFFER, true),
        }
    }
    pub fn upload(
        &mut self,
        gpu: &GPUManager,
        builder: &mut vulkano::command_buffer::AutoCommandBufferBuilder<
            vulkano::command_buffer::PrimaryAutoCommandBuffer,
        >,
    ) -> bool {
        self.vertex_buffer.upload_delta(gpu, builder)
            | self.normal_buffer.upload_delta(gpu, builder)
            | self.tex_coord_buffer.upload_delta(gpu, builder)
            | self.index_buffer.upload_delta(gpu, builder)
    }

    pub fn add_mesh(
        &mut self,
        vertices: &[[f32; 3]],
        normals: &[[f32; 3]],
        tex_coords: &[[f32; 2]],
        indices: &[u32],
    ) -> (u32, u32) {
        if self.vertex_buffer.data_len() > u32::MAX as usize {
            panic!("Exceeded maximum number of vertices in MeshBuffers");
        }
        if self.index_buffer.data_len() > u32::MAX as usize {
			panic!("Exceeded maximum number of indices in MeshBuffers");
		}
        let vertex_offset = self.vertex_buffer.data_len() as u32;
        let index_offset = self.index_buffer.data_len() as u32;
        for v in vertices {
            self.vertex_buffer.push_data(*v);
        }
        for n in normals {
            self.normal_buffer.push_data(*n);
        }
        for t in tex_coords {
            self.tex_coord_buffer.push_data(*t);
        }
        for i in indices {
            self.index_buffer.push_data(*i);
        }
        (vertex_offset, index_offset)
    }
}

pub static MESH_BUFFERS: LazyLock<Mutex<Option<MeshBuffers>>> = LazyLock::new(|| Mutex::new(None));

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub indices: Vec<u32>,

    // // buffers:
    // pub vertex_buffer: Subbuffer<[[f32; 3]]>,
    // pub normal_buffer: Subbuffer<[[f32; 3]]>,
    // pub tex_coord_buffer: Subbuffer<[[f32; 2]]>,
    // pub index_buffer: Subbuffer<[u32]>,
    // pub texture: Arc<Mutex<Option<AssetHandle<Texture>>>>,

    // global offsets for indirect drawing
    pub vertex_offset: u32,
    pub index_offset: u32,
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
            let (vertex_offset, index_offset) = if let Some(buffers) = MESH_BUFFERS.lock().as_mut()
            {
                buffers.add_mesh(&vertices, &normals, &tex_coords, &indices)
            } else {
                panic!("MESH_BUFFERS not initialized");
            };
            let mesh = Mesh {
                vertices,
                normals,
                tex_coords,
                indices,
                vertex_offset,
                index_offset,
                // vertex_buffer,
                // normal_buffer,
                // tex_coord_buffer,
                // index_buffer,
                // texture,
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

        // let vertex_buffer =
        //     gpu.buffer_from_iter(vertices.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        // let normal_buffer =
        //     gpu.buffer_from_iter(normals.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        // let tex_coord_buffer =
        //     gpu.buffer_from_iter(tex_coords.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        // let index_buffer = gpu.buffer_from_iter(indices.iter().cloned(), BufferUsage::INDEX_BUFFER);

        let (vertex_offset, index_offset) = if let Some(buffers) = MESH_BUFFERS.lock().as_mut() {
            buffers.add_mesh(&vertices, &normals, &tex_coords, &indices)
        } else {
            panic!("MESH_BUFFERS not initialized");
        };
        let mesh = Mesh {
            vertices,
            normals,
            tex_coords,
            indices,
            vertex_offset,
            index_offset,
            // vertex_buffer,
            // normal_buffer,
            // tex_coord_buffer,
            // index_buffer,
            // texture: Arc::new(Mutex::new(None)),
        };
        Self { meshes: vec![mesh] }
    }
}
