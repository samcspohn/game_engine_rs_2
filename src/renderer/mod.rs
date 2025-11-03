use std::{
    collections::{HashMap, HashSet},
    ops::Sub,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    u32,
};

use egui::layers;
use vulkano::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, DispatchIndirectCommand,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    image::{sampler::Sampler, view::ImageView},
    memory::allocator::MemoryTypeFilter,
    pipeline::{
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};

use crate::{
    asset_manager::{Asset, AssetHandle, AssetManager},
    component::Component,
    engine::Engine,
    gpu_manager::{GPUManager, gpu_vec::GpuVec},
    obj_loader::{MESH_BUFFERS, Model},
    renderer,
    texture::Texture,
    transform::Transform,
    util::Storage,
};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/renderer.comp",
    }
}

#[derive(BufferContents, Copy, Clone, Default)]
#[repr(C)]
struct Renderer {
    m_id: u32,
    t_idx: u32,
}

struct ModelCount {
    count: u32,
    num_mesh: u32,
}

pub struct RenderingSystem {
    gpu: Arc<GPUManager>,
    pipeline: Arc<ComputePipeline>,
    pub command_buffer: Option<Arc<PrimaryAutoCommandBuffer>>,
    // buffers
    indirect_commands_buffer: GpuVec<DrawIndexedIndirectCommand>,
    // indirect_texture_vec: GpuVec<u32>, // texture per indirect draw command -> change to material per indirect draw command
    indirect_material_vec: GpuVec<u32>, // material per indirect draw command
    material_vec: GpuVec<crate::fs::Material>,
    texture_map: HashMap<u32, u32>, // texture asset_id -> texture idx
    registered_textures: Vec<AssetHandle<Texture>>, // to avoid duplicate texture loads
    renderer_buffer: GpuVec<cs::Renderer>,
    renderer_inits_buffer: GpuVec<cs::RendererInit>,
    renderer_uninits_buffer: GpuVec<u32>,
    model_indirect_buffer: GpuVec<cs::ModelIndirect>, // [indirect_offset, count]
    transform_ids_buffer: GpuVec<cs::RenderData>,
    workgroup_sums_buffer: GpuVec<u32>,
    stages: Subbuffer<[cs::Stage]>,
    dispatch: Subbuffer<[DispatchIndirectCommand]>,
    // data
    // model_indirect_buffer_len: usize,
    model_map: HashMap<u32, u32>, // model_id -> model_indirect_buffer idx
    used_model_set: HashSet<u32>,
    model_asset_map: HashMap<u32, Arc<Model>>,
    // indirect_map: HashMap<u32, u32>, // intermediate -> model_indirect_buffer idx
    // indirect_counts: HashMap<u32, u32>,
    // model_counts: HashMap<u32, u32>,
    indirect_model_map: HashMap<u32, u32>,
    // intermediate_counter: AtomicU32,
    renderer_storage: Storage<Renderer>,
    mvp_count: AtomicU32,
    model_counts: HashMap<u32, ModelCount>,
    model_registered: AtomicBool,
    // model_indirects: Vec<cs::ModelIndirect>,
}

#[derive(Default, Clone)]
pub struct RendererComponent {
    pub model: AssetHandle<Model>,
    r_idx: u32,
}

impl RendererComponent {
    pub fn new(model: AssetHandle<Model>) -> Self {
        Self { model, r_idx: 0 }
    }
}

impl Component for RendererComponent {
    fn init(&mut self, t: &Transform, e: &Engine) {
        self.r_idx = e.rendering_system.lock().renderer(self.model.clone(), t);
    }
    fn deinit(&mut self, t: &Transform, e: &Engine) {
        let mut rendering_system = e.rendering_system.lock();
        rendering_system.renderer_storage.remove(self.r_idx);
        rendering_system
            .renderer_uninits_buffer
            .push_data(self.r_idx);
        rendering_system.mvp_count.fetch_sub(1, Ordering::SeqCst);
    }
}

impl RendererComponent {
    pub fn uninit(self, r: &mut RenderingSystem) -> u32 {
        let id = self.model.asset_id;
        r.renderer_storage.remove(self.r_idx);
        r.renderer_uninits_buffer.push_data(self.r_idx);
        id
    }
}

#[derive(Default, Clone)]
pub struct _RendererComponent {
    pub model: AssetHandle<Model>,
}

impl _RendererComponent {
    pub fn new(model: AssetHandle<Model>) -> Self {
        Self { model }
    }
}

impl Component for _RendererComponent {
    fn init(&mut self, t: &Transform, e: &Engine) {
        e.rendering_system.lock()._renderer(self.model.clone());
        // self.r_idx = e.rendering_system.lock().renderer(self.model.clone(), t);
    }
    fn deinit(&mut self, t: &Transform, e: &Engine) {
        // e.rendering_system
        //     .lock()
        //     .renderer_storage
        //     .remove(self.r_idx);
        // e.rendering_system
        //     .lock()
        //     .renderer_uninits_buffer
        //     .push_data(self.r_idx);
    }
}

const NUM_STAGES: u64 = 9;
impl RenderingSystem {
    pub fn new(gpu: Arc<GPUManager>) -> Self {
        let shader = cs::load(gpu.device.clone()).unwrap();
        let stage = PipelineShaderStageCreateInfo::new(shader.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            gpu.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(gpu.device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline = ComputePipeline::new(
            gpu.device.clone(),
            Some(gpu.pipeline_cache.clone()),
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();
        Self {
            command_buffer: None,
            pipeline,
            indirect_commands_buffer: GpuVec::new(
                &gpu,
                BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
                true,
            ),
            //          indirect_texture_vec: GpuVec::new(
            // 	&gpu,
            // 	BufferUsage::STORAGE_BUFFER,
            // 	true,
            // ),
            indirect_material_vec: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            material_vec: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            // material_vec: Vec::new(),
            texture_map: HashMap::new(),
            registered_textures: Vec::new(),
            renderer_buffer: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            model_indirect_buffer: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            transform_ids_buffer: GpuVec::new(
                &gpu,
                BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
                false,
            ),
            renderer_inits_buffer: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, false),
            renderer_uninits_buffer: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, false),
            workgroup_sums_buffer: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, false),
            stages: gpu.buffer_array(
                NUM_STAGES,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
            ),
            dispatch: gpu.buffer_array(
                NUM_STAGES,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_DST,
            ),
            gpu,
            model_map: HashMap::new(),
            used_model_set: HashSet::new(),
            model_asset_map: HashMap::new(),
            // indirect_map: HashMap::new(),
            // intermediate_counter: AtomicU32::new(0),
            renderer_storage: Storage::new(),
            mvp_count: AtomicU32::new(0),
            model_counts: HashMap::new(),
            // indirect_counts: HashMap::new(),
            // model_counts: HashMap::new(),
            indirect_model_map: HashMap::new(),
            model_registered: AtomicBool::new(false),
        }
    }
    pub fn get_or_register_model_handle(&mut self, model: AssetHandle<Model>) -> u32 {
        let m_id = model.asset_id;
        if !self.model_map.contains_key(&m_id) {
            let idx = self.model_indirect_buffer.data_len() as u32;
            self.model_indirect_buffer.push_data(cs::ModelIndirect {
                offset: 0,
                count: 1,
            }); // render placeholder mesh
            self.model_map.insert(m_id, idx);
            self.model_counts.insert(
                idx,
                ModelCount {
                    count: 0,
                    num_mesh: 1,
                },
            );
            self.indirect_model_map.insert(idx, m_id);
        }
        *self.model_map.get(&m_id).unwrap()
    }
    fn get_or_register_texture(&mut self, tex: &Option<AssetHandle<Texture>>) -> u32 {
        if let Some(tex) = tex {
            *self.texture_map.entry(tex.asset_id).or_insert_with(|| {
                let tex_idx = self.registered_textures.len() as u32;
                self.registered_textures.push(tex.clone());
                tex_idx
            })
        } else {
            u32::MAX
        }
    }
    pub fn register_model(&mut self, model: AssetHandle<Model>, asset: Arc<Model>) {
        // let m_id = model.asset_id;
        // let m_idx = model.get(assets).id;
        // let t_idx = transform.get_idx();
        let ind_idx = self.get_or_register_model_handle(model);
        self.model_asset_map.insert(ind_idx, asset.clone());

        let mesh_count = asset.meshes.len() as u32;
        self.model_counts.get(&ind_idx).map(|mc| {
            self.mvp_count
                .fetch_sub(mc.count * mc.num_mesh, Ordering::SeqCst);
        });
        self.model_counts.get_mut(&ind_idx).map(|mc| {
            mc.num_mesh = mesh_count;
            self.mvp_count
                .fetch_add(mesh_count * mc.count, Ordering::SeqCst);
        });

        let indirect_offset = self.indirect_commands_buffer.data_len() as u32;
        for mesh in &asset.meshes {
            self.indirect_commands_buffer
                .push_data(DrawIndexedIndirectCommand {
                    index_count: mesh.indices.len() as u32,
                    instance_count: 0,
                    first_index: mesh.index_offset,
                    vertex_offset: mesh.vertex_offset,
                    first_instance: 0,
                });
            if let Some(mat) = &mesh.material {
                let albedo: u32 = self.get_or_register_texture(&mat.albedo_texture);
                let normal = self.get_or_register_texture(&mat.normal_texture);
                let spec = self.get_or_register_texture(&mat.specular_texture);
                let mr = self.get_or_register_texture(&mat.metallic_roughness_texture);
                let m = crate::fs::Material {
                    albedo_tex_index: albedo,
                    normal_tex_index: normal,
                    specular_tex_index: spec,
                    metallic_roughness_tex_index: mr,
                    base_color: mat.base_color,
                };
                self.material_vec.push_data(m);
                let mat_idx = (self.material_vec.data_len() - 1) as u32;
                self.indirect_material_vec.push_data(mat_idx);
            } else {
                self.indirect_material_vec.push_data(u32::MAX);
            }
            // self.indirect_texture_vec.push_data(*tex_idx);
        }
        *self
            .model_indirect_buffer
            .get_data_mut(ind_idx as usize)
            .unwrap() = cs::ModelIndirect {
            offset: indirect_offset,
            count: mesh_count,
        };

        self.model_registered.store(true, Ordering::SeqCst);
        // self.command_buffer = None; // invalidate command buffer
    }
    pub fn renderer(&mut self, model: AssetHandle<Model>, transform: &Transform) -> u32 {
        let ind_id = self.get_or_register_model_handle(model.clone());
        let count = self
            .model_counts
            .get_mut(&ind_id)
            .map(|mc| {
                mc.count += 1;
                mc.num_mesh
            })
            .unwrap();
        self.mvp_count.fetch_add(count, Ordering::SeqCst);

        if self.used_model_set.insert(ind_id) {
            self.model_registered.store(true, Ordering::SeqCst);
        }
        // self.update_num_counts(model);
        let t_idx = transform.get_idx();
        let renderer = Renderer {
            m_id: ind_id,
            t_idx,
        };
        let r_idx = self.renderer_storage.insert(renderer);
        self.renderer_inits_buffer.push_data(cs::RendererInit {
            idx: r_idx,
            r: cs::Renderer {
                t_idx,
                m_id: ind_id,
            },
            padding: 0,
        });
        r_idx
    }
    pub fn _renderer(&mut self, model: AssetHandle<Model>) {
        self.get_or_register_model_handle(model);
    }
    pub fn compute_renderers(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        matrix_data: &Subbuffer<[crate::transform::compute::cs::MatrixData]>,
        // assets_loaded: bool, // cam: &Subbuffer<crate::vs::camera>,
    ) -> bool {
        let mut ret = false;
        let num_workgroups = self.indirect_commands_buffer.data_len().div_ceil(64).max(1) as u32;
        ret |= self.workgroup_sums_buffer.resize_buffer_exact(
            num_workgroups as usize,
            &self.gpu,
            builder,
        );
        // println!("ret/workgroup_sums: {}", ret);
        ret |= self.model_indirect_buffer.upload_delta(&self.gpu, builder);
        // println!("ret/model_indirect: {}", ret);

        if self.model_registered.load(Ordering::SeqCst) {
            ret |= self.model_indirect_buffer.force_update(&self.gpu, builder);
            self.model_registered.store(false, Ordering::SeqCst);
        }
        // println!("ret/model_indirect force: {}", ret);

        ret |= self
            .renderer_buffer
            .resize_buffer(self.renderer_storage.len(), &self.gpu, builder);
        // println!("ret/renderer_buffer: {}", ret);

        ret |= self.transform_ids_buffer.resize_buffer(
            // self.mvp_count.load(Ordering::SeqCst) as usize,
            self.mvp_count.load(Ordering::SeqCst) as usize,
            &self.gpu,
            builder,
        );
        // println!("ret/mvp_buffer: {}", ret);
        // println!("Renderer System: {} MVPs", self.mvp_count.load(Ordering::SeqCst));

        let renderer_inits_len = self.renderer_inits_buffer.data_len();
        let renderer_uninits_len = self.renderer_uninits_buffer.data_len();
        let mut resized = self.renderer_inits_buffer.upload_delta(&self.gpu, builder);
        resized |= self
            .renderer_uninits_buffer
            .upload_delta(&self.gpu, builder);
        self.renderer_inits_buffer.clear();
        self.renderer_uninits_buffer.clear();
        // println!("ret/renderer_inits/uninits upload: {}", resized);

        ret |= self.indirect_commands_buffer.data_len() > self.indirect_commands_buffer.buf_len();
        // println!("ret/indirect_commands: {}", ret);
        ret |= self
            .indirect_commands_buffer
            .upload_delta(&self.gpu, builder);
        ret |= self.indirect_material_vec.upload_delta(&self.gpu, builder);
        ret |= self.material_vec.upload_delta(&self.gpu, builder);
        // println!("ret/indirect_commands upload: {}", ret);

        if self.command_buffer.is_none() || ret || resized {
            let mut builder = self
                .gpu
                .create_command_buffer(CommandBufferUsage::SimultaneousUse);

            let layout_0 = self.pipeline.layout().set_layouts().get(0).unwrap();
            let set_0 = DescriptorSet::new(
                self.gpu.desc_alloc.clone(),
                layout_0.clone(),
                [
                    WriteDescriptorSet::buffer(0, matrix_data.clone()),
                    WriteDescriptorSet::buffer(1, self.transform_ids_buffer.buf()),
                    WriteDescriptorSet::buffer(2, self.renderer_buffer.buf()),
                    WriteDescriptorSet::buffer(3, self.renderer_inits_buffer.buf()),
                    WriteDescriptorSet::buffer(4, self.renderer_uninits_buffer.buf()),
                    WriteDescriptorSet::buffer(5, self.model_indirect_buffer.buf()),
                    WriteDescriptorSet::buffer(6, self.workgroup_sums_buffer.buf()),
                    WriteDescriptorSet::buffer(7, self.indirect_commands_buffer.buf()),
                    WriteDescriptorSet::buffer(8, self.indirect_material_vec.buf()),
                    // WriteDescriptorSet::buffer(9, normal_matrices.clone()),
                ],
                [],
            )
            .unwrap();

            let layout1 = self.pipeline.layout().set_layouts().get(1).unwrap();
            let set1 = DescriptorSet::new(
                self.gpu.desc_alloc.clone(),
                layout1.clone(),
                [WriteDescriptorSet::buffer(1, self.gpu.empty.clone())],
                [],
            )
            .unwrap();

            let layout2 = self.pipeline.layout().set_layouts().get(2).unwrap();
            let set2_s: Vec<_> = (0..7) // stages 0-6 are per-frame, 7-8 are per-camera
                .map(|i| {
                    DescriptorSet::new(
                        self.gpu.desc_alloc.clone(),
                        layout2.clone(),
                        [WriteDescriptorSet::buffer(
                            0,
                            self.stages.clone().slice(i..=i),
                        )],
                        [],
                    )
                    .unwrap()
                })
                .collect();

            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap();
            for i in 0u64..7 {
                builder
                    .bind_descriptor_sets(
                        vulkano::pipeline::PipelineBindPoint::Compute,
                        self.pipeline.layout().clone(),
                        0,
                        (set_0.clone(), set1.clone(), set2_s[i as usize].clone()),
                    )
                    .unwrap();
                unsafe {
                    builder
                        .dispatch_indirect(self.dispatch.clone().slice(i..=i))
                        .unwrap()
                };
            }
            self.command_buffer = Some(builder.build().unwrap());
        }

        let dispatch_buf = self
            .gpu
            .sub_alloc(BufferUsage::INDIRECT_BUFFER)
            .allocate_slice(NUM_STAGES)
            .unwrap();
        let stage_buf = self
            .gpu
            .sub_alloc(BufferUsage::UNIFORM_BUFFER)
            .allocate_slice(NUM_STAGES)
            .unwrap();
        {
            let mut write_lock = dispatch_buf.write().unwrap();
            let mut stage_lock = stage_buf.write().unwrap();

            let mut set_dispatch_stage = |x: usize, i: usize, stage: u32, pass: u32| {
                write_lock[i] = DispatchIndirectCommand {
                    x: x.div_ceil(64) as u32,
                    y: 1,
                    z: 1,
                };
                stage_lock[i] = cs::Stage {
                    num_jobs: x as u32,
                    stage,
                    pass,
                };
            };
            set_dispatch_stage(renderer_uninits_len, 0, 0, 1); // stage 0, 1 uninit renderers FIRST
            set_dispatch_stage(renderer_inits_len, 1, 0, 0); // stage 0, 0 init renderers AFTER
            set_dispatch_stage(self.indirect_commands_buffer.data_len(), 2, 1, 0); // stage 1 reset indirects
            set_dispatch_stage(self.renderer_storage.len(), 3, 2, 0); // stage 2 count instances
            set_dispatch_stage(self.indirect_commands_buffer.data_len(), 4, 3, 0); // stage 3, 0 prefix sum - local scan
            set_dispatch_stage(64, 5, 3, 1); // stage 3, 1 prefix sum - global scan
            set_dispatch_stage(self.indirect_commands_buffer.data_len(), 6, 3, 2); // stage 3, 2 prefix sum - add sums
            set_dispatch_stage(self.indirect_commands_buffer.data_len(), 7, 4, 0); // stage 4, 0 reset instance counts
            set_dispatch_stage(self.renderer_storage.len(), 8, 4, 1); // stage 4, 1 generate draw commands
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(dispatch_buf, self.dispatch.clone()))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(stage_buf, self.stages.clone()))
            .unwrap();
        ret
    }

    // execute before draw for every camera
    pub fn update_mvp(
        &mut self,
        cam: Subbuffer<crate::vs::camera>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        matrix_data: &Subbuffer<[crate::transform::compute::cs::MatrixData]>,
    ) {
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap();
        let layout_0 = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set_0 = DescriptorSet::new(
            self.gpu.desc_alloc.clone(),
            layout_0.clone(),
            [
                WriteDescriptorSet::buffer(0, matrix_data.clone()),
                WriteDescriptorSet::buffer(1, self.transform_ids_buffer.buf()),
                WriteDescriptorSet::buffer(2, self.renderer_buffer.buf()),
                WriteDescriptorSet::buffer(3, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(4, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(5, self.model_indirect_buffer.buf()),
                WriteDescriptorSet::buffer(6, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(7, self.indirect_commands_buffer.buf()),
                WriteDescriptorSet::buffer(8, self.indirect_material_vec.buf()),
            ],
            [],
        )
        .unwrap();
        let layout1 = self.pipeline.layout().set_layouts().get(1).unwrap();
        let set1 = DescriptorSet::new(
            self.gpu.desc_alloc.clone(),
            layout1.clone(),
            [WriteDescriptorSet::buffer(1, cam)],
            [],
        )
        .unwrap();
        let layout2 = self.pipeline.layout().set_layouts().get(2).unwrap();
        for i in 7..=8 {
            let set2 = DescriptorSet::new(
                self.gpu.desc_alloc.clone(),
                layout2.clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    self.stages.clone().slice(i..=i),
                )],
                [],
            )
            .unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    (set_0.clone(), set1.clone(), set2.clone()),
                )
                .unwrap();
            unsafe {
                builder
                    .dispatch_indirect(self.dispatch.clone().slice(i..=i))
                    .unwrap();
            }
        }
    }

    // assume pipeline already bound
    pub fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        assets: &AssetManager,
        pipeline: Arc<GraphicsPipeline>,
        camera: Subbuffer<crate::vs::camera>,
        matrix_data: &Subbuffer<[crate::transform::compute::cs::MatrixData]>,
    ) {
        if self.indirect_commands_buffer.data_len() == 0 {
            return;
        }
        if self.indirect_commands_buffer.data_len() > self.indirect_commands_buffer.buf_len() {
            panic!("Indirect commands buffer not up to date");
        }

        // let texture: Arc<Texture> = AssetHandle::default().get(assets);
        let buffers_guard = MESH_BUFFERS.lock();
        let mesh_buffers = buffers_guard.as_ref().unwrap();

        // let textures = self
        //     .indirect_texture_vec
        //     .iter()
        //     .map(|t_idx| {
        //         let texture_handle = AssetHandle::<Texture>::_from_id(*t_idx);
        //         let tex = texture_handle.get(assets);
        //         (tex.image.clone(), tex.sampler.clone())
        //     })
        //     .collect::<Vec<_>>();
        let mut textures: Vec<_> = self
            .registered_textures
            .iter()
            .map(|texture_handle| {
                let tex = texture_handle.get(assets);
                (tex.image.clone(), tex.sampler.clone())
            })
            .collect();
        if textures.is_empty() {
            let dummy_texture_handle = AssetHandle::<Texture>::default();
            let tex = dummy_texture_handle.get(assets);
            textures.push((tex.image.clone(), tex.sampler.clone()));
        }
        if self.material_vec.data_len() > 0 {
        	println!("material 0: {:?}", self.material_vec.get_data(0).unwrap())
        }

        let set = DescriptorSet::new_variable(
            self.gpu.desc_alloc.clone(),
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            textures.len() as u32,
            [
                WriteDescriptorSet::buffer(0, camera.clone()),
                WriteDescriptorSet::buffer(1, matrix_data.clone()),
                WriteDescriptorSet::buffer(2, self.material_vec.buf()),
                WriteDescriptorSet::image_view_sampler_array(3, 0, textures),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(
                0,
                (
                    mesh_buffers.vertex_buffer.buf(),
                    mesh_buffers.tex_coord_buffer.buf(),
                    mesh_buffers.normal_buffer.buf(),
                    mesh_buffers.color_buffer.buf(),
                    self.transform_ids_buffer.buf(),
                ),
            )
            .unwrap()
            .bind_index_buffer(mesh_buffers.index_buffer.buf())
            .unwrap();
        unsafe {
            builder
                .draw_indexed_indirect(
                    self.indirect_commands_buffer
                        .buf()
                        .slice(0..self.indirect_commands_buffer.data_len() as u64),
                )
                .unwrap();
        }
    }
}
