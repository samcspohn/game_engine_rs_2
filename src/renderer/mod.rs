use std::{
    collections::HashMap,
    ops::Sub,
    sync::{Arc, atomic::AtomicU32},
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
    gpu_manager::{GPUManager, gpu_vec::GPUVec},
    obj_loader::Obj,
    renderer,
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
    indirect_commands_buffer: GPUVec<DrawIndexedIndirectCommand>,
    renderer_buffer: GPUVec<Renderer>,
    renderer_inits_buffer: GPUVec<cs::RendererInit>,
    renderer_uninits_buffer: GPUVec<u32>,
    model_indirect_buffer: GPUVec<cs::ModelIndirect>, // [indirect_offset, count]
    mvp_buffer: GPUVec<[[f32; 4]; 4]>,
    workgroup_sums_buffer: GPUVec<u32>,
    stages: Subbuffer<[cs::Stage]>,
    dispatch: Subbuffer<[DispatchIndirectCommand]>,
    // data
    // model_indirect_buffer_len: usize,
    model_map: HashMap<u32, u32>, // maps
    renderer_storage: Storage<Renderer>,
    // model_indirects: Vec<cs::ModelIndirect>,
}

pub struct RendererComponent {
    model: AssetHandle<Obj>,
    r_idx: u32,
}

impl RendererComponent {
    pub fn uninit(self, r: &mut RenderingSystem) -> u32 {
        let id = self.model.asset_id;
        r.renderer_storage.remove(self.r_idx);
        r.renderer_uninits_buffer.push_data(self.r_idx);
        id
    }
}

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
            indirect_commands_buffer: GPUVec::new(
                &gpu,
                BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
                true,
            ),
            renderer_buffer: GPUVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            model_indirect_buffer: GPUVec::new(&gpu, BufferUsage::STORAGE_BUFFER, true),
            mvp_buffer: GPUVec::new(
                &gpu,
                BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
                false,
            ),
            renderer_inits_buffer: GPUVec::new(&gpu, BufferUsage::STORAGE_BUFFER, false),
            renderer_uninits_buffer: GPUVec::new(&gpu, BufferUsage::STORAGE_BUFFER, false),
            workgroup_sums_buffer: GPUVec::new(&gpu, BufferUsage::STORAGE_BUFFER, false),
            stages: gpu.buffer_array(
                8,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
            ),
            dispatch: gpu.buffer_array(
                8,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_DST,
            ),
            gpu,
            // model_indirect_buffer_len: 0,
            model_map: HashMap::new(),
            renderer_storage: Storage::new(),
            // model_indirects: Vec::new(),
            // indirect_commands: Vec::new(),
        }
    }
    pub fn register_model(&mut self, model: AssetHandle<Obj>, assets: &AssetManager) {
        let m_id = model.asset_id;
        // let m_idx = model.get(assets).id;
        // let t_idx = transform.get_idx();
        let indirect_offset = self.indirect_commands_buffer.data_len() as u32;
        let count = model.get(assets).meshes.len() as u32;
        let a = self.model_indirect_buffer.data_len();
        self.model_map.insert(m_id, a as u32);
        self.model_indirect_buffer.push_data(cs::ModelIndirect {
            offset: indirect_offset,
            count,
        });
        for mesh in &model.get(assets).meshes {
            self.indirect_commands_buffer
                .push_data(DrawIndexedIndirectCommand {
                    index_count: mesh.indices.len() as u32,
                    instance_count: 0,
                    first_index: 0,
                    vertex_offset: 0,
                    first_instance: 0,
                });
        }
    }
    pub fn renderer(
        &mut self,
        model: AssetHandle<Obj>,
        transform: &Transform,
        assets: &AssetManager,
    ) -> RendererComponent {
        let asset_id = model.asset_id;
        let m_id = *self.model_map.get(&asset_id).expect("Model not registered");
        let t_idx = transform.get_idx();
        let renderer = Renderer { m_id, t_idx };
        let r_idx = self.renderer_storage.insert(renderer);
        self.renderer_inits_buffer.push_data(cs::RendererInit {
            idx: r_idx,
            r: cs::Renderer { t_idx, m_id },
        });
        RendererComponent { model, r_idx }
    }
    pub fn compute_renderers(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        model_matrices: &Subbuffer<[[[f32; 4]; 4]]>,
        assets_loaded: bool, // cam: &Subbuffer<crate::vs::camera>,
    ) -> bool {
        let mut ret = false;
        let num_workgroups = self.indirect_commands_buffer.data_len().div_ceil(64).max(1) as u32;
        ret |= self.workgroup_sums_buffer
            .resize_buffer(num_workgroups as usize, &self.gpu, builder);
        ret |= self.model_indirect_buffer.upload_delta(&self.gpu, builder);
        if assets_loaded {
            ret |= self.model_indirect_buffer.force_update(&self.gpu, builder);
        }

        ret |= self
            .renderer_buffer
            .resize_buffer(self.renderer_storage.len(), &self.gpu, builder);
        ret |= self
            .mvp_buffer
            .resize_buffer(self.renderer_storage.len(), &self.gpu, builder);

        let renderer_inits_len = self.renderer_inits_buffer.data_len();
        let renderer_uninits_len = self.renderer_uninits_buffer.data_len();
        self.renderer_inits_buffer.upload_delta(&self.gpu, builder);
        self.renderer_uninits_buffer
            .upload_delta(&self.gpu, builder);
        self.renderer_inits_buffer.clear();
        self.renderer_uninits_buffer.clear();

        self.indirect_commands_buffer
            .upload_delta(&self.gpu, builder);

        if self.command_buffer.is_none() || ret {
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
            let set2_s: Vec<_> = (0..7) // execute stage 4 (7) per camera
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
            .allocate_slice(8)
            .unwrap();
        let stage_buf = self
            .gpu
            .sub_alloc(BufferUsage::UNIFORM_BUFFER)
            .allocate_slice(8)
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
            set_dispatch_stage(renderer_inits_len, 0, 0, 0); // stage 0, 0 init renderers
            set_dispatch_stage(renderer_uninits_len, 1, 0, 1); // stage 0, 1 uninit renderers
            set_dispatch_stage(self.indirect_commands_buffer.data_len(), 2, 1, 0); // stage 1 reset indirects
            set_dispatch_stage(self.renderer_storage.len(), 3, 2, 0); // stage 2 count instances
            set_dispatch_stage(self.indirect_commands_buffer.data_len(), 4, 3, 0); // stage 3, 0 prefix sum - local scan
            set_dispatch_stage(64, 5, 3, 1); // stage 3, 1 prefix sum - global scan
            set_dispatch_stage(self.indirect_commands_buffer.data_len(), 6, 3, 2); // stage 3, 2 prefix sum - add sums
            set_dispatch_stage(self.renderer_storage.len(), 7, 4, 0); // stage 4 generate draw commands
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
        let set2 = DescriptorSet::new(
            self.gpu.desc_alloc.clone(),
            layout2.clone(),
            [WriteDescriptorSet::buffer(
                0,
                self.stages.clone().slice(7..=7),
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
                .dispatch_indirect(self.dispatch.clone().slice(7..=7))
                .unwrap();
        }
    }

    // assume pipeline already bound
    pub fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        assets: &AssetManager,
        pipeline: Arc<GraphicsPipeline>,
    ) {
        for (model_id, model_idx) in &self.model_map {
            let obj = AssetHandle::<Obj>::_from_id(*model_id);
            let model = obj.get(assets);
            for (i, mesh) in model.meshes.iter().enumerate() {
                let texture = mesh
                    .texture
                    .lock()
                    .as_ref()
                    .unwrap_or(&AssetHandle::default())
                    .get(assets);

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

                let offset = self
                    .model_indirect_buffer
                    .get_data(*model_idx as usize)
                    .unwrap()
                    .offset
                    + i as u32;
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
                            mesh.vertex_buffer.clone(),
                            mesh.tex_coord_buffer.clone(),
                            mesh.normal_buffer.clone(),
                            self.mvp_buffer.buf().clone(),
                        ),
                    )
                    .unwrap()
                    .bind_index_buffer(mesh.index_buffer.clone())
                    .unwrap();
                unsafe {
                    builder
                        .draw_indexed_indirect(
                            self.indirect_commands_buffer
                                .buf()
                                .slice((offset as u64)..(offset + 1) as u64),
                        )
                        .unwrap();
                }
            }
        }
    }
}
