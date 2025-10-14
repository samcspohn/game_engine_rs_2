use std::{
    collections::HashMap, ops::Sub, sync::{Arc, atomic::AtomicU32}
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
    gpu_manager::GPUManager,
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

#[derive(BufferContents)]
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
    indirect_commands_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
    renderer_buffer: Subbuffer<[Renderer]>,
    renderer_inits_buffer: Subbuffer<[cs::RendererInit]>,
    renderer_uninits_buffer: Subbuffer<[u32]>,
    model_indirect_buffer: Subbuffer<[cs::ModelIndirect]>, // [indirect_offset, count]
    mvp_buffer: Subbuffer<[[[f32; 4]; 4]]>,
    workgroup_sums_buffer: Subbuffer<[u32]>,
    stages: Subbuffer<[cs::Stage]>,
    dispatch: Subbuffer<[DispatchIndirectCommand]>,
    // data
    indirect_draw_commands_len: usize,
    model_indirect_buffer_len: usize,
    model_map: HashMap<u32, u32>, // maps
    renderer_storage: Storage<Renderer>,
    renderer_inits: Vec<cs::RendererInit>,
    model_indirects: Vec<cs::ModelIndirect>,
    indirect_commands: Vec<DrawIndexedIndirectCommand>,
    renderer_uninits: Vec<u32>,
    num_mvp: AtomicU32,
}

pub struct RendererComponent {
    model: AssetHandle<Obj>,
    r_idx: u32,
}

impl RendererComponent {
    pub fn uninit(self, r: &mut RenderingSystem) -> u32 {
        let id = self.model.asset_id;
        r.renderer_storage.remove(self.r_idx);
        r.renderer_uninits.push(self.r_idx);
        // if let Some(count) = r.model_map.get_mut(&id) {
        //     *count -= 1;
        // }
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
            indirect_commands_buffer: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
            ),
            renderer_buffer: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            ),
            model_indirect_buffer: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ),
            mvp_buffer: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER,
            ),
            renderer_inits_buffer: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ),
            renderer_uninits_buffer: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ),
            workgroup_sums_buffer: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER,
            ),
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
            indirect_draw_commands_len: 0,
            model_indirect_buffer_len: 0,
            model_map: HashMap::new(),
            renderer_storage: Storage::new(),
            renderer_inits: Vec::new(),
            renderer_uninits: Vec::new(),
            num_mvp: AtomicU32::new(0),
            model_indirects: Vec::new(),
            indirect_commands: Vec::new(),
        }
    }
    pub fn register_model(
        &mut self,
        model: AssetHandle<Obj>,
        assets: &AssetManager,
    ) {
        let m_id = model.asset_id;
        // let m_idx = model.get(assets).id;
        // let t_idx = transform.get_idx();
        let indirect_offset = self.indirect_commands.len() as u32;
        let count = model.get(assets).meshes.len() as u32;
        let a = self.model_indirects.len();
        self.model_map.insert(m_id, a as u32);
        self.model_indirects.push(cs::ModelIndirect {
            offset: indirect_offset,
            count,
        });
        for mesh in &model.get(assets).meshes {
            self.indirect_commands.push(DrawIndexedIndirectCommand {
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
        // self.model_map
        //     .entry(model.asset_id)
        //     .and_modify(|e| *e += 1)
        //     .or_insert(1);
        let asset_id = model.asset_id;
        let m_id = *self.model_map.get(&asset_id).expect("Model not registered");
        let t_idx = transform.get_idx();
        let renderer = Renderer { m_id, t_idx };
        let r_idx = self.renderer_storage.insert(renderer);
        self.renderer_inits.push(cs::RendererInit {
            idx: r_idx,
            r: cs::Renderer { t_idx, m_id },
        });
        RendererComponent { model, r_idx }
    }
    pub fn compute_renderers(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        model_matrices: &Subbuffer<[[[f32; 4]; 4]]>,
        assets_loaded: bool
        // cam: &Subbuffer<crate::vs::camera>,
    ) -> bool {
        let mut ret = false;

        if assets_loaded || self.model_indirect_buffer_len < self.model_indirects.len() {
            // rebuild model indirect buffer
            if self.model_indirect_buffer_len < self.model_indirects.len() {
                let new_size = self.model_indirects.len().next_power_of_two() as u64;
                self.model_indirect_buffer = self.gpu.buffer_array(
                    new_size,
                    MemoryTypeFilter::PREFER_DEVICE,
                    BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                );
                ret = true;
            }
            let model_indirect_buf = self
                .gpu
                .storage_alloc
                .allocate_slice(self.model_indirects.len().max(1) as u64)
                .unwrap();
            if self.model_indirects.len() > 0 {
                let mut write_lock = model_indirect_buf.write().unwrap();
                write_lock.copy_from_slice(&self.model_indirects);
            }
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    model_indirect_buf,
                    self.model_indirect_buffer.clone(),
                ))
                .unwrap();
            self.model_indirect_buffer_len = self.model_indirects.len();
        }

        let num_workgroups = self.indirect_commands.len().div_ceil(64).max(1) as u32;
        if self.workgroup_sums_buffer.len() < num_workgroups as u64 {
            self.workgroup_sums_buffer = self.gpu.buffer_array(
                num_workgroups as u64,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER,
            );
            ret = true;
        }
        
        // resize renderer buffer. new renderers are added as deltas
        if self.renderer_buffer.len() < self.renderer_storage.len() as u64 {
            let new_size = self.renderer_storage.len().next_power_of_two() as u64;
            let buf = self.gpu.buffer_array(
                new_size,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            );
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.renderer_buffer.clone(),
                    buf.clone(),
                ))
                .unwrap();
            self.renderer_buffer = buf;

            self.mvp_buffer = self.gpu.buffer_array(
                new_size,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER,
            );

            ret = true;
        }
        // resize inits buffer
        if self.renderer_inits_buffer.len() < self.renderer_inits.len() as u64 {
            let new_size = self.renderer_inits.len().next_power_of_two() as u64;
            self.renderer_inits_buffer = self.gpu.buffer_array(
                new_size,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            );
            ret = true;
        }
        // resize uninits buffer
        if self.renderer_uninits_buffer.len() < self.renderer_uninits.len() as u64 {
            let new_size = self.renderer_uninits.len().next_power_of_two() as u64;
            self.renderer_uninits_buffer = self.gpu.buffer_array(
                new_size,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            );
            ret = true;
        }
        if self.indirect_draw_commands_len < self.indirect_commands.len() {
            let delta = self.indirect_commands.len() - self.indirect_draw_commands_len;
            let buf = self.gpu.storage_alloc.allocate_slice(delta as u64).unwrap();
            {
                let mut write_lock = buf.write().unwrap();
                write_lock
                    .copy_from_slice(&self.indirect_commands[self.indirect_draw_commands_len..]);
            }
            if self.indirect_commands_buffer.len() < self.indirect_commands.len() as u64 {
                let new_size = self.indirect_commands.len().next_power_of_two() as u64;
                let new_buf = self.gpu.buffer_array(
                    new_size,
                    MemoryTypeFilter::PREFER_DEVICE,
                    BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                );
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        self.indirect_commands_buffer.clone(),
                        new_buf.clone(),
                    ))
                    .unwrap();
                self.indirect_commands_buffer = new_buf;
                ret = true;
            }
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    buf.clone(),
                    self.indirect_commands_buffer.clone().slice(
                        self.indirect_draw_commands_len as u64..self.indirect_commands.len() as u64,
                    ),
                ))
                .unwrap();
            self.indirect_draw_commands_len = self.indirect_commands.len();
        }

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
                    WriteDescriptorSet::buffer(1, self.mvp_buffer.clone()),
                    WriteDescriptorSet::buffer(2, self.renderer_buffer.clone()),
                    WriteDescriptorSet::buffer(3, self.renderer_inits_buffer.clone()),
                    WriteDescriptorSet::buffer(4, self.renderer_uninits_buffer.clone()),
                    WriteDescriptorSet::buffer(5, self.model_indirect_buffer.clone()),
                    WriteDescriptorSet::buffer(6, self.workgroup_sums_buffer.clone()),
                    WriteDescriptorSet::buffer(7, self.indirect_commands_buffer.clone()),
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

        // allocate temp buffers and copy data
        let renderer_inits_buf = self
            .gpu
            .storage_alloc
            .allocate_slice(self.renderer_inits.len().max(1) as u64)
            .unwrap();
        if self.renderer_inits.len() > 0 {
            let mut write_lock = renderer_inits_buf.write().unwrap();
            write_lock.copy_from_slice(&self.renderer_inits);
        }
        let renderer_uninits_buf = self
            .gpu
            .storage_alloc
            .allocate_slice(self.renderer_uninits.len().max(1) as u64)
            .unwrap();
        if self.renderer_uninits.len() > 0 {
            let mut write_lock = renderer_uninits_buf.write().unwrap();
            write_lock.copy_from_slice(&self.renderer_uninits);
        }
        let dispatch_buf = self.gpu.ind_alloc.allocate_slice(8).unwrap();
        let stage_buf = self.gpu.sub_alloc.allocate_slice(8).unwrap();
        {
            let mut write_lock = dispatch_buf.write().unwrap();
            let mut stage_lock = stage_buf.write().unwrap();

            // stage 0, 0 init renderers
            write_lock[0] = DispatchIndirectCommand {
                x: self.renderer_inits.len().div_ceil(64) as u32,
                y: 1,
                z: 1,
            };
            stage_lock[0] = cs::Stage {
                num_jobs: self.renderer_inits.len() as u32,
                stage: 0,
                pass: 0,
            };

            // stage 0, 1 uninit renderers
            write_lock[1] = DispatchIndirectCommand {
                x: self.renderer_uninits.len().div_ceil(64) as u32,
                y: 1,
                z: 1,
            };
            stage_lock[1] = cs::Stage {
                num_jobs: self.renderer_uninits.len() as u32,
                stage: 0,
                pass: 1,
            };

            // stage 1 reset indirects
            write_lock[2] = DispatchIndirectCommand {
                x: self.indirect_commands.len().div_ceil(64) as u32,
                y: 1,
                z: 1,
            };
            stage_lock[2] = cs::Stage {
                num_jobs: self.indirect_commands.len() as u32,
                stage: 1,
                pass: 0,
            };

            // stage 2 count instances
            write_lock[3] = DispatchIndirectCommand {
                x: self.renderer_storage.len().div_ceil(64) as u32,
                y: 1,
                z: 1,
            };
            stage_lock[3] = cs::Stage {
                num_jobs: self.renderer_storage.len() as u32,
                stage: 2,
                pass: 0,
            };

            // stage 3, 0 prefix sum - local scan
            write_lock[4] = DispatchIndirectCommand {
                x: self.indirect_commands.len().div_ceil(64) as u32,
                y: 1,
                z: 1,
            };
            stage_lock[4] = cs::Stage {
                num_jobs: self.indirect_commands.len() as u32,
                stage: 3,
                pass: 0,
            };
            // stage 3, 1 prefix sum - global scan
            write_lock[5] = DispatchIndirectCommand { x: 1, y: 1, z: 1 };
            stage_lock[5] = cs::Stage {
                num_jobs: 64 as u32,
                stage: 3,
                pass: 1,
            };
            // stage 3, 2 prefix sum - add sums
            write_lock[6] = DispatchIndirectCommand {
                x: self.indirect_commands.len().div_ceil(64) as u32,
                y: 1,
                z: 1,
            };
            stage_lock[6] = cs::Stage {
                num_jobs: self.indirect_commands.len() as u32,
                stage: 3,
                pass: 2,
            };
            // stage 4 generate draw commands
            write_lock[7] = DispatchIndirectCommand {
                x: self.renderer_storage.len().div_ceil(64) as u32,
                y: 1,
                z: 1,
            };
            stage_lock[7] = cs::Stage {
                num_jobs: self.renderer_storage.len() as u32,
                stage: 4,
                pass: 0,
            };
        }
        self.renderer_inits.clear();
        self.renderer_uninits.clear();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                renderer_inits_buf,
                self.renderer_inits_buffer.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                renderer_uninits_buf,
                self.renderer_uninits_buffer.clone(),
            ))
            .unwrap()
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
        model_matrices: Subbuffer<[[[f32; 4]; 4]]>
    ) {

        // builder
        //     .copy_buffer(CopyBufferInfo::buffers(
        //         cam,
        //         self.cam_buffer.clone()
        //     ))
        //     .unwrap();

        builder.bind_pipeline_compute(self.pipeline.clone()).unwrap();
        let layout_0 = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set_0 = DescriptorSet::new(
            self.gpu.desc_alloc.clone(),
            layout_0.clone(),
            [
                WriteDescriptorSet::buffer(0, model_matrices),
                WriteDescriptorSet::buffer(1, self.mvp_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.renderer_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(4, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(5, self.model_indirect_buffer.clone()),
                WriteDescriptorSet::buffer(6, self.gpu.empty.clone()),
                WriteDescriptorSet::buffer(7, self.indirect_commands_buffer.clone()),
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
            [WriteDescriptorSet::buffer(0, self.stages.clone().slice(7..=7))],
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
            builder.dispatch_indirect(self.dispatch.clone().slice(7..=7)).unwrap();
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

                let offset = self.model_indirects[*model_idx as usize].offset + i as u32;
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
                            self.mvp_buffer.clone(),
                        ),
                    )
                    .unwrap()
                    .bind_index_buffer(mesh.index_buffer.clone())
                    .unwrap();
                unsafe {
                    builder
                        .draw_indexed_indirect(
                            self.indirect_commands_buffer
                                .clone()
                                .slice((offset as u64)..(offset + 1) as u64),
                        )
                        .unwrap();
                }
            }
        }
    }
}
