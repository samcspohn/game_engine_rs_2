use std::{
    collections::{HashMap, HashSet},
    ops::Sub,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

use egui::layers;
use vulkano::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, DispatchIndirectCommand,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
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

pub struct RenderingSystem {
    gpu: Arc<GPUManager>,
    pipeline: Arc<ComputePipeline>,
    pub command_buffer: Option<Arc<PrimaryAutoCommandBuffer>>,
    // buffers
    indirect_commands_buffer: GpuVec<DrawIndexedIndirectCommand>,
    renderer_buffer: GpuVec<cs::Renderer>,
    renderer_inits_buffer: GpuVec<cs::RendererInit>,
    renderer_uninits_buffer: GpuVec<u32>,
    model_indirect_buffer: GpuVec<cs::ModelIndirect>, // [indirect_offset, count]
    mvp_buffer: GpuVec<[[f32; 4]; 4]>,
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
        rendering_system
            .mvp_count
            .fetch_sub(1, Ordering::SeqCst);
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
            None,
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
            renderer_buffer: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            model_indirect_buffer: GpuVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            mvp_buffer: GpuVec::new(
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
            // indirect_counts: HashMap::new(),
            // model_counts: HashMap::new(),
            indirect_model_map: HashMap::new(),
            model_registered: AtomicBool::new(false),
        }
    }
    pub fn get_or_register_model_handle(&mut self, model: AssetHandle<Model>) -> u32 {
        let m_id = model.asset_id;
        if !self.model_map.contains_key(&m_id) {
            // let interm_idx = self
            //     .intermediate_counter
            //     .fetch_add(1, Ordering::SeqCst);
            let idx = self.model_indirect_buffer.data_len() as u32;
            self.model_indirect_buffer.push_data(cs::ModelIndirect {
                offset: 0,
                count: 1,
            }); // render placeholder mesh
            // self.indirect_map.insert(interm_idx, idx);
            self.model_map.insert(m_id, idx);
            // initialize counts
            // self.model_counts.insert(m_id, 0);
            // self.indirect_counts.insert(interm_idx, 0);
            self.indirect_model_map.insert(idx, m_id);
        }
        *self.model_map.get(&m_id).unwrap()
    }
    pub fn register_model(&mut self, model: AssetHandle<Model>, asset: Arc<Model>) {
        let m_id = model.asset_id;
        // let m_idx = model.get(assets).id;
        // let t_idx = transform.get_idx();
        let ind_idx = self.get_or_register_model_handle(model);
        self.model_asset_map.insert(ind_idx, asset.clone());

        let mesh_count = asset.meshes.len() as u32;
        // let model_indirect_buffer_idx = self.model_indirect_buffer.data_len() as u32;
        // if let Some(prev) = self
        //     .indirect_map
        //     .insert(intermediate_idx, model_indirect_buffer_idx)
        // {
        //     // get previously created renderers count, subtract from placeholder
        //     // and increment the new model indirect count by the number of meshes * previous count
        //     let model_idx = *self
        //         .indirect_model_map
        //         .get(&intermediate_idx)
        //         .expect("Intermediate idx not found in indirect_model_map");
        //     let count = *self
        //         .model_counts
        //         .get(&model_idx)
        //         .expect("Model idx not found in model_counts");
        //     self.indirect_counts.get_mut(&prev).map(|c| *c -= count);
        //     self.indirect_counts
        //         .entry(model_indirect_buffer_idx)
        //         .and_modify(|c| *c += mesh_count * count)
        //         .or_insert(mesh_count * count);
        //     self.mvp_count
        //         .fetch_sub(count, Ordering::SeqCst);
        //     self.mvp_count
        //         .fetch_add(mesh_count * count, Ordering::SeqCst);
        // }
        // let ind_idx = *self.indirect_map.get(&intermediate_idx).unwrap();

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
        }
        *self
            .model_indirect_buffer
            .get_data_mut(ind_idx as usize)
            .unwrap() = cs::ModelIndirect {
            offset: indirect_offset,
            count: mesh_count,
        };
        self.model_registered
            .store(true, Ordering::SeqCst);
        // self.command_buffer = None; // invalidate command buffer
    }
    // pub fn update_num_counts(&mut self, model: AssetHandle<Model>) {
    //     let asset_id = model.asset_id;
    //     let interm_id = self.get_or_register_model_handle(model);
    //     let ind_id = *self.indirect_map.get(&interm_id).unwrap();
    //     let count = self
    //         .model_indirect_buffer
    //         .get_data(interm_id as usize)
    //         .map_or(1, |mi| mi.count);
    //     self.mvp_count
    //         .fetch_add(count, Ordering::SeqCst);
    //     // self.indirect_counts
    //     //     .entry(ind_id)
    //     //     .and_modify(|c| *c += count)
    //     //     .or_insert(count);
    //     // self.model_counts
    //     //     .entry(asset_id)
    //     //     .and_modify(|c| *c += 1)
    //     //     .or_insert(1);
    // }
    pub fn renderer(&mut self, model: AssetHandle<Model>, transform: &Transform) -> u32 {
        let ind_id = self.get_or_register_model_handle(model.clone());
        self.mvp_count
            .fetch_add(1, Ordering::SeqCst);
        if self.used_model_set.insert(ind_id) {
            self.model_registered
                .store(true, Ordering::SeqCst);
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
        model_matrices: &Subbuffer<[[[f32; 4]; 4]]>,
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

        if self
            .model_registered
            .load(Ordering::SeqCst)
        {
            ret |= self.model_indirect_buffer.force_update(&self.gpu, builder);
            self.model_registered
                .store(false, Ordering::SeqCst);
        }
        // println!("ret/model_indirect force: {}", ret);

        ret |= self
            .renderer_buffer
            .resize_buffer(self.renderer_storage.len(), &self.gpu, builder);
        // println!("ret/renderer_buffer: {}", ret);

        ret |= self.mvp_buffer.resize_buffer(self.mvp_count.load(Ordering::SeqCst) as usize, &self.gpu, builder);
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
                    WriteDescriptorSet::buffer(0, model_matrices.clone()),
                    WriteDescriptorSet::buffer(1, self.mvp_buffer.buf()),
                    WriteDescriptorSet::buffer(2, self.renderer_buffer.buf()),
                    WriteDescriptorSet::buffer(3, self.renderer_inits_buffer.buf()),
                    WriteDescriptorSet::buffer(4, self.renderer_uninits_buffer.buf()),
                    WriteDescriptorSet::buffer(5, self.model_indirect_buffer.buf()),
                    WriteDescriptorSet::buffer(6, self.workgroup_sums_buffer.buf()),
                    WriteDescriptorSet::buffer(7, self.indirect_commands_buffer.buf()),
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
        model_matrices: Subbuffer<[[[f32; 4]; 4]]>,
    ) {
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap();
        let layout_0 = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set_0 = DescriptorSet::new(
            self.gpu.desc_alloc.clone(),
            layout_0.clone(),
            [
                WriteDescriptorSet::buffer(0, model_matrices),
                WriteDescriptorSet::buffer(1, self.mvp_buffer.buf()),
                WriteDescriptorSet::buffer(2, self.renderer_buffer.buf()),
                WriteDescriptorSet::buffer(3, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(4, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(5, self.model_indirect_buffer.buf()),
                WriteDescriptorSet::buffer(6, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(7, self.indirect_commands_buffer.buf()),
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
    ) {
        if self.indirect_commands_buffer.data_len() == 0 {
            return;
        }
        if self.indirect_commands_buffer.data_len() > self.indirect_commands_buffer.buf_len() {
            panic!("Indirect commands buffer not up to date");
        }

        let texture: Arc<Texture> = AssetHandle::default().get(assets);
        let buffers_guard = MESH_BUFFERS.lock();
        let mesh_buffers = buffers_guard.as_ref().unwrap();

        let set = DescriptorSet::new(
            self.gpu.desc_alloc.clone(),
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::image_view_sampler(
                1,
                texture.image.clone(),
                texture.sampler.clone(),
            )],
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
                    self.mvp_buffer.buf().clone(),
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
