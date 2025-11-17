use std::{
    path::Path,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

use gltf::mesh::util::colors;
use parking_lot::Mutex;
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::{
    asset_manager::{Asset, AssetHandle, DeferredAssetQueue},
    gpu_manager::{GPUManager, GPUWorkQueue, gpu_vec::GpuVec},
    texture::Texture,
};

/// Converts a tangent from [f32; 4] to packed u32 with 8-bit signed normalized values
/// Each f32 component in range [-1.0, 1.0] is converted to i8 in range [-127, 127]
pub fn pack_tangent(tangent: [f32; 4]) -> u32 {
    let x = (tangent[0].clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let y = (tangent[1].clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let z = (tangent[2].clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let w = (tangent[3].clamp(-1.0, 1.0) * 127.0) as i8 as u8;

    (w as u32) << 24 | (z as u32) << 16 | (y as u32) << 8 | (x as u32)
}

/// Converts a normal from [f32; 3] to packed u32 with 8-bit signed normalized values
/// Each f32 component in range [-1.0, 1.0] is converted to i8 in range [-127, 127]
/// The 4th byte (w) is set to 0
pub fn pack_normal(normal: [f32; 3]) -> u32 {
    let x = (normal[0].clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let y = (normal[1].clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let z = (normal[2].clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let w = 0u8;

    (w as u32) << 24 | (z as u32) << 16 | (y as u32) << 8 | (x as u32)
}

// static OBJ_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
#[derive(Debug, Clone)]
pub struct Model {
    // pub id: u32,
    pub meshes: Vec<Mesh>,
}

pub struct MeshBuffers {
    pub vertex_buffer: GpuVec<[f32; 3]>,
    pub normal_buffer: GpuVec<u32>,
    pub tangent_buffer: GpuVec<u32>,
    pub tex_coord_buffer: GpuVec<[f32; 2]>,
    pub color_buffer: GpuVec<[u8; 4]>,
    pub index_buffer: GpuVec<u32>,
}

impl MeshBuffers {
    pub fn new(gpu: &GPUManager) -> Self {
        Self {
            vertex_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
            normal_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
            tangent_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
            tex_coord_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
            color_buffer: GpuVec::new(gpu, BufferUsage::VERTEX_BUFFER, true),
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
        let mut ret = false;
        ret |= self.vertex_buffer.upload_delta(gpu, builder);
        ret |= self.normal_buffer.upload_delta(gpu, builder);
        ret |= self.tangent_buffer.upload_delta(gpu, builder);
        ret |= self.tex_coord_buffer.upload_delta(gpu, builder);
        ret |= self.color_buffer.upload_delta(gpu, builder);
        ret |= self.index_buffer.upload_delta(gpu, builder);
        ret
    }

    pub fn add_mesh(
        &mut self,
        vertices: &[[f32; 3]],
        normals: &[u32],
        tangents: &[u32],
        tex_coords: &[[f32; 2]],
        colors: &[[u8; 4]],
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
        for t in tangents {
            self.tangent_buffer.push_data(*t);
        }
        for t in tex_coords {
            self.tex_coord_buffer.push_data(*t);
        }
        for c in colors {
            self.color_buffer.push_data(*c);
        }
        for i in indices {
            self.index_buffer.push_data(*i);
        }
        (vertex_offset, index_offset)
    }
}

pub static MESH_BUFFERS: LazyLock<Mutex<Option<MeshBuffers>>> = LazyLock::new(|| Mutex::new(None));

#[derive(Clone, Default)]
pub struct Material {
    pub base_color: [f32; 4],
    pub albedo_texture: Option<AssetHandle<Texture>>,
    pub normal_texture: Option<AssetHandle<Texture>>,
    pub specular_texture: Option<AssetHandle<Texture>>,
    pub metallic_roughness_texture: Option<AssetHandle<Texture>>,
}
use std::fmt::Debug;
impl Debug for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Material")
            .field("base_color", &self.base_color)
            .field(
                "albedo_texture",
                &self.albedo_texture.as_ref().map(|_| "Texture"),
            )
            .field(
                "normal_texture",
                &self.normal_texture.as_ref().map(|_| "Texture"),
            )
            .field(
                "specular_texture",
                &self.specular_texture.as_ref().map(|_| "Texture"),
            )
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct AABB {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<u32>,
    pub tangents: Vec<u32>,
    pub tex_coords: Vec<[f32; 2]>,
    pub colors: Vec<[u8; 4]>,
    pub indices: Vec<u32>,

    // // buffers:
    // pub vertex_buffer: Subbuffer<[[f32; 3]]>,
    // pub normal_buffer: Subbuffer<[[f32; 3]]>,
    // pub tex_coord_buffer: Subbuffer<[[f32; 2]]>,
    // pub index_buffer: Subbuffer<[u32]>,
    // pub texture: Option<AssetHandle<Texture>>,
    pub material: Option<Material>,

    // global offsets for indirect drawing
    pub vertex_offset: u32,
    pub index_offset: u32,

    pub aabb: AABB,
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
            let mut tangents = Vec::new();
            let mut tex_coords = Vec::new();
            let mut colors = Vec::new();
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
                    let normal = [mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]];
                    normals.push(pack_normal(normal));
                }
            } else {
                // If no normals provided, fill with default up vectors
                for _ in 0..vertices.len() {
                    normals.push(pack_normal([0.0, 1.0, 0.0]));
                }
            }

            let mut tangents_f32 = vec![[1.0, 0.0, 0.0, 1.0]; vertices.len()];
            for face in mesh.indices.chunks(3) {
                let v0 = vertices[face[0] as usize];
                let v1 = vertices[face[1] as usize];
                let v2 = vertices[face[2] as usize];

                let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

                let tangent = [
                    edge1[0] + edge2[0],
                    edge1[1] + edge2[1],
                    edge1[2] + edge2[2],
                    1.0,
                ];
                for &idx in face {
                    tangents_f32[idx as usize] = tangent;
                }
            }

            // Convert f32 tangents to packed u32 format
            for tangent in tangents_f32 {
                tangents.push(pack_tangent(tangent));
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

            // process vertex colors (if any)
            if !mesh.vertex_color.is_empty() {
                for i in (0..mesh.vertex_color.len()).step_by(3) {
                    colors.push([
                        (mesh.vertex_color[i] * 255.0) as u8,
                        (mesh.vertex_color[i + 1] * 255.0) as u8,
                        (mesh.vertex_color[i + 2] * 255.0) as u8,
                        255,
                    ]);
                }
            } else {
                // If no colors provided, fill with white
                for _ in 0..vertices.len() {
                    colors.push([255, 255, 255, 255]);
                }
            }

            // Process indices with offset for multiple models
            for index in mesh.indices {
                indices.push(index);
            }
            let (vertex_offset, index_offset) = if let Some(buffers) = MESH_BUFFERS.lock().as_mut()
            {
                buffers.add_mesh(
                    &vertices,
                    &normals,
                    &tangents,
                    &tex_coords,
                    &colors,
                    &indices,
                )
            } else {
                panic!("MESH_BUFFERS not initialized");
            };

            // let verts = vertices.clone();
            // let norms = normals.clone();
            // let texs = tex_coords.clone();
            // let inds = indices.clone();

            // println!("Creating GPU buffers for OBJ");
            // let vertex_buffer = gpu
            //     .enqueue_work(move |g| {
            //         g.buffer_from_iter(verts.iter().cloned(), BufferUsage::VERTEX_BUFFER)
            //     })
            //     .wait()
            //     .unwrap();
            // println!("Created vertex buffer");
            // let normal_buffer = gpu
            //     .enqueue_work(move |g| {
            //         g.buffer_from_iter(norms.iter().cloned(), BufferUsage::VERTEX_BUFFER)
            //     })
            //     .wait()
            //     .unwrap();
            // println!("Created normal buffer");
            // let tex_coord_buffer = gpu
            //     .enqueue_work(move |g| {
            //         g.buffer_from_iter(texs.iter().cloned(), BufferUsage::VERTEX_BUFFER)
            //     })
            //     .wait()
            //     .unwrap();
            // println!("Created tex coord buffer");
            // let index_buffer = gpu
            //     .enqueue_work(move |g| {
            //         g.buffer_from_iter(inds.iter().cloned(), BufferUsage::INDEX_BUFFER)
            //     })
            //     .wait()
            //     .unwrap();
            // println!("Created index buffer");

            let mut albedo_texture = None;
            let mut _normal_texture = None;
            let mut _specular_texture = None;
            let mut base_color = [1.0, 1.0, 1.0, 1.0];
            if let Some(mat_id) = mesh.material_id.as_ref() {
                if let Some(material) = materials.get(*mat_id) {
                    if let Some(diffuse_color) = material.diffuse {
                        base_color = [diffuse_color[0], diffuse_color[1], diffuse_color[2], 1.0];
                    }
                    if let Some(disolve) = material.dissolve {
                        base_color[3] = disolve;
                    }
                    if let Some(diffuse_texture) = material.diffuse_texture.clone() {
                        println!("Loading diffuse texture for mesh: {:?}", diffuse_texture);
                        let tex_path = if Path::new(&diffuse_texture).is_absolute() {
                            diffuse_texture
                        } else {
                            let mut p = path.parent().unwrap_or(Path::new("")).to_path_buf();
                            p.push(diffuse_texture);
                            p.to_string_lossy().to_string()
                        };

                        albedo_texture = Some(
                            asset
                                .enqueue_work(move |a| a.load_asset::<Texture>(&tex_path, None))
                                .wait()
                                .unwrap(),
                        );
                    } else {
                        println!("No diffuse texture for material {}", mat_id);
                    }
                    if let Some(normal_texture) = material.normal_texture.clone() {
                        println!("Loading diffuse texture for mesh: {:?}", normal_texture);
                        let tex_path = if Path::new(&normal_texture).is_absolute() {
                            normal_texture
                        } else {
                            let mut p = path.parent().unwrap_or(Path::new("")).to_path_buf();
                            p.push(normal_texture);
                            p.to_string_lossy().to_string()
                        };

                        _normal_texture = Some(
                            asset
                                .enqueue_work(move |a| a.load_asset::<Texture>(&tex_path, None))
                                .wait()
                                .unwrap(),
                        );
                    }
                    if let Some(specular_texture) = material.specular_texture.clone() {
                        println!("Loading specular texture for mesh: {:?}", specular_texture);
                        let tex_path = if Path::new(&specular_texture).is_absolute() {
                            specular_texture
                        } else {
                            let mut p = path.parent().unwrap_or(Path::new("")).to_path_buf();
                            p.push(specular_texture);
                            p.to_string_lossy().to_string()
                        };

                        _specular_texture = Some(
                            asset
                                .enqueue_work(move |a| a.load_asset::<Texture>(&tex_path, None))
                                .wait()
                                .unwrap(),
                        );
                    }
                } else {
                    println!("Invalid material_id {} for mesh", mat_id);
                }
            }
            let material = Material {
                base_color,
                albedo_texture,
                normal_texture: _normal_texture,
                specular_texture: _specular_texture,
                metallic_roughness_texture: None,
            };

            let aabb = AABB {
                min: [
                    vertices.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min),
                    vertices.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min),
                    vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min),
                ],
                max: [
                    vertices
                        .iter()
                        .map(|v| v[0])
                        .fold(f32::NEG_INFINITY, f32::max),
                    vertices
                        .iter()
                        .map(|v| v[1])
                        .fold(f32::NEG_INFINITY, f32::max),
                    vertices
                        .iter()
                        .map(|v| v[2])
                        .fold(f32::NEG_INFINITY, f32::max),
                ],
            };

            let mesh = Mesh {
                vertices,
                normals,
                tangents,
                tex_coords,
                colors,
                indices,
                vertex_offset,
                index_offset,
                material: Some(material),
                aabb,
                // texture,
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
        let normals: Vec<u32> = vec![pack_normal([0.0, 0.0, 1.0]); 3];
        let tangents_f32 = vec![[1.0, 0.0, 0.0, 1.0]; 3];
        let tangents: Vec<u32> = tangents_f32.iter().map(|&t| pack_tangent(t)).collect();
        let tex_coords = vec![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let colors = vec![[255, 255, 255, 255]; 3];
        let indices = vec![0, 1, 2];

        // let vertex_buffer =
        //     gpu.buffer_from_iter(vertices.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        // let normal_buffer =
        //     gpu.buffer_from_iter(normals.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        // let tex_coord_buffer =
        //     gpu.buffer_from_iter(tex_coords.iter().cloned(), BufferUsage::VERTEX_BUFFER);
        // let index_buffer = gpu.buffer_from_iter(indices.iter().cloned(), BufferUsage::INDEX_BUFFER);

        let (vertex_offset, index_offset) = if let Some(buffers) = MESH_BUFFERS.lock().as_mut() {
            buffers.add_mesh(
                &vertices,
                &normals,
                &tangents,
                &tex_coords,
                &colors,
                &indices,
            )
        } else {
            panic!("MESH_BUFFERS not initialized");
        };
        let aabb = AABB {
            min: [-0.5, -0.5, 0.0],
            max: [0.5, 0.5, 0.0],
        };
        let mesh = Mesh {
            vertices,
            normals,
            tangents,
            tex_coords,
            colors,
            indices,
            vertex_offset,
            index_offset,
            aabb,
            // vertex_buffer,
            // normal_buffer,
            // tex_coord_buffer,
            // index_buffer,
            // texture: None,
            material: None,
        };
        Self { meshes: vec![mesh] }
    }
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            tangents: Vec::new(),
            tex_coords: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
            vertex_offset: 0,
            index_offset: 0,
            // texture: None,
            material: None,
            aabb: AABB {
				min: [0.0, 0.0, 0.0],
				max: [0.0, 0.0, 0.0],
			},
        }
    }
}
