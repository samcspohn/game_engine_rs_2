use std::{
    cell::SyncUnsafeCell,
    collections::VecDeque,
    fmt::Debug,
    ops::{Div, Mul},
    sync::{Arc, atomic::AtomicU32},
};

use parking_lot::Mutex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, DispatchIndirectCommand,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet, layout::DescriptorSetLayout},
    memory::allocator::MemoryTypeFilter,
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo},
    },
    sync::GpuFuture,
};

use crate::gpu_manager::GPUManager;

use super::{TransformComponent, TransformHierarchy};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/transform.comp"
    }
}

struct TransformBuffers {
    position: Subbuffer<[[f32; 4]]>,
    rotation: Subbuffer<[[f32; 4]]>,
    scale: Subbuffer<[[f32; 4]]>,
    parent: Subbuffer<[u32]>,
}

impl TransformBuffers {
    fn new(gpu: &GPUManager) -> Self {
        let mem_filter: MemoryTypeFilter = MemoryTypeFilter::PREFER_DEVICE;
        let usage: BufferUsage =
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
        let position: Subbuffer<[[f32; 4]]> = gpu.buffer_array(1, mem_filter, usage);
        let rotation: Subbuffer<[[f32; 4]]> = gpu.buffer_array(1, mem_filter, usage);
        let scale: Subbuffer<[[f32; 4]]> = gpu.buffer_array(1, mem_filter, usage);
        let parent: Subbuffer<[u32]> = gpu.buffer_array(1, mem_filter, usage);

        Self {
            position,
            rotation,
            scale,
            parent,
        }
    }

    fn resize(
        &mut self,
        gpu: &GPUManager,
        new_size: u64,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        if self.position.len() < new_size {
            let new_size = new_size.next_power_of_two();
            let mem_filter: MemoryTypeFilter = MemoryTypeFilter::PREFER_DEVICE;
            let usage: BufferUsage =
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
            let buf = gpu.buffer_array(new_size, mem_filter, usage);
            builder
                .copy_buffer(CopyBufferInfo::buffers(self.position.clone(), buf.clone()))
                .unwrap();
            self.position = buf;

            let buf = gpu.buffer_array(new_size, mem_filter, usage);
            builder
                .copy_buffer(CopyBufferInfo::buffers(self.rotation.clone(), buf.clone()))
                .unwrap();
            self.rotation = buf;

            let buf = gpu.buffer_array(new_size, mem_filter, usage);
            builder
                .copy_buffer(CopyBufferInfo::buffers(self.scale.clone(), buf.clone()))
                .unwrap();
            self.scale = buf;

            let buf = gpu.buffer_array(new_size, mem_filter, usage);
            builder
                .copy_buffer(CopyBufferInfo::buffers(self.parent.clone(), buf.clone()))
                .unwrap();
            self.parent = buf;
        }
    }
}

struct TransformUpdateBuffers {
    position: Subbuffer<[f32]>,
    rotation: Subbuffer<[f32]>,
    scale: Subbuffer<[f32]>,
    // parent: Subbuffer<[u32]>,
    flags: Subbuffer<[u32]>,
    // dirty_l2: Subbuffer<[u32]>,
}

impl TransformUpdateBuffers {
    fn new(gpu: &GPUManager) -> Self {
        let mem_filter: MemoryTypeFilter =
            MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS;
        let usage: BufferUsage = BufferUsage::STORAGE_BUFFER;
        let position: Subbuffer<[f32]> = gpu.buffer_array(3, mem_filter, usage);
        let rotation: Subbuffer<[f32]> = gpu.buffer_array(4, mem_filter, usage);
        let scale: Subbuffer<[f32]> = gpu.buffer_array(3, mem_filter, usage);
        // let parent: Subbuffer<[u32]> = gpu.buffer_array(2, mem_filter, usage);
        let flags: Subbuffer<[u32]> = gpu.buffer_array(3, mem_filter, usage);
        // let dirty_l2: Subbuffer<[u32]> = gpu.buffer_array(1, mem_filter, usage);
        Self {
            position,
            rotation,
            scale,
            // parent,
            flags,
            // dirty_l2,
        }
    }

    fn resize(&mut self, gpu: &GPUManager, new_size: u64) {
        let mem_filter: MemoryTypeFilter =
            MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS;
        let usage: BufferUsage = BufferUsage::STORAGE_BUFFER;

        if self.position.len() < new_size * 3 {
            // let new_size = new_size.next_power_of_two() * 10;
            let new_size = new_size.next_power_of_two();
            self.flags = gpu.buffer_array(
                new_size.div_ceil(32).mul(3), // 1 bit per pos, rot, scale
                mem_filter,
                usage,
            );

            // let new_size = new_size.next_power_of_two();
            self.position = gpu.buffer_array(new_size * 3, mem_filter, usage);
            self.rotation = gpu.buffer_array(new_size * 4, mem_filter, usage);
            self.scale = gpu.buffer_array(new_size * 3, mem_filter, usage);
            // self.dirty_l2 = gpu.buffer_array(new_size.div_ceil(32), mem_filter, usage);
            // self.parent = gpu.buffer_array(new_size * 2, mem_filter, usage);
        }
    }
}

pub struct PerfCounter {
    pub times: VecDeque<std::time::Instant>,
    pub times_secs: VecDeque<f32>,
    pub sum: f32,
}

impl PerfCounter {
    pub fn new() -> Self {
        Self {
            times: VecDeque::with_capacity(100),
            times_secs: VecDeque::with_capacity(100),
            sum: 0.0,
        }
    }

    pub fn start(&mut self) {
        self.times.push_back(std::time::Instant::now());
        // self.times_secs.push_back(0.0);
    }
    pub fn stop(&mut self) {
        let t = std::time::Instant::now();
        if let Some(end) = self.times.back() {
            let secs = (t.duration_since(*end).as_secs_f32()).max(0.0);
            self.sum += secs;
            self.times_secs.push_back(secs);
        }
        if self.times.len() > 100 {
            if let Some(old) = self.times.pop_front() {
                if let Some(old_secs) = self.times_secs.pop_front() {
                    self.sum -= old_secs;
                }
            }
        }
    }
}

impl Debug for PerfCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2} ms",
            if self.times.len() > 0 {
                (self.sum / self.times.len() as f32) * 1000.0
            } else {
                0.0
            }
        )
    }
}

pub struct PerfCounters {
    pub allocate_bufs: PerfCounter,
    pub update_bufs: PerfCounter,
    pub compute: PerfCounter,
    pub update_parents: PerfCounter,
}

pub struct TransformCompute {
    pipeline: Arc<ComputePipeline>,
    pub model_matrix_buffer: Subbuffer<[[[f32; 4]; 4]]>,
    transform_buffers: TransformBuffers,
    // update_buffers: TransformUpdateBuffers,
    parent_updates: Subbuffer<[u32]>,
    staging_buffers: Vec<TransformUpdateBuffers>, // TODO: zero flags in shader/avoid writing 0s to flags
    pub staging_buffer_index: usize,
    pub perf_counters: PerfCounters,
    pub command_buffer: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub pc: [Subbuffer<cs::PushConstants>; 3],
    pub indirect: Subbuffer<[DispatchIndirectCommand]>,
}

impl TransformCompute {
    pub fn new(gpu: &GPUManager) -> Self {
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
        let model_matrix_buffer: Subbuffer<[[[f32; 4]; 4]]> = gpu.buffer_array(
            1,
            MemoryTypeFilter::PREFER_DEVICE,
            BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::TRANSFER_SRC
                | BufferUsage::VERTEX_BUFFER,
        );

        Self {
            pipeline,
            model_matrix_buffer,
            transform_buffers: TransformBuffers::new(gpu),
            // update_buffers: TransformUpdateBuffers::new(gpu),
            parent_updates: gpu.buffer_array(
                1,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            ),
            staging_buffers: (0..gpu.image_count as usize + 2)
                .map(|_| TransformUpdateBuffers::new(gpu))
                .collect(),
            staging_buffer_index: 0,
            perf_counters: PerfCounters {
                allocate_bufs: PerfCounter::new(),
                update_bufs: PerfCounter::new(),
                compute: PerfCounter::new(),
                update_parents: PerfCounter::new(),
            },
            command_buffer: Vec::new(),
            pc: [
                gpu.buffer_from_data(
                    &cs::PushConstants {
                        stage: 0,
                        count: 0,
                        update: 1,
                    },
                    BufferUsage::UNIFORM_BUFFER,
                ),
                gpu.buffer_from_data(
                    &cs::PushConstants {
                        stage: 1,
                        count: 0,
                        update: 1,
                    },
                    BufferUsage::UNIFORM_BUFFER,
                ),
                gpu.buffer_from_data(
                    &cs::PushConstants {
                        stage: 2,
                        count: 0,
                        update: 1,
                    },
                    BufferUsage::UNIFORM_BUFFER,
                ),
            ],
            indirect: gpu.buffer_array(
                3,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_DST,
            ),
        }
    }

    pub fn update_transforms(
        &mut self,
        gpu: &GPUManager,
        hierarchy: &TransformHierarchy,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        update_transforms_compute_shader: bool,
        gpu_future: &mut Box<dyn GpuFuture>,
    ) -> (bool, usize) {
        let sbi = self.staging_buffer_index;

        self.perf_counters.allocate_bufs.start();
        let pc1: Subbuffer<cs::PushConstants> = gpu
            .sub_alloc(BufferUsage::UNIFORM_BUFFER)
            .allocate_sized()
            .unwrap();
        let pc2: Subbuffer<cs::PushConstants> = gpu
            .sub_alloc(BufferUsage::UNIFORM_BUFFER)
            .allocate_sized()
            .unwrap();
        let pc3: Subbuffer<cs::PushConstants> = gpu
            .sub_alloc(BufferUsage::UNIFORM_BUFFER)
            .allocate_sized()
            .unwrap();
        let indirect: Subbuffer<[DispatchIndirectCommand]> = gpu
            .sub_alloc(BufferUsage::INDIRECT_BUFFER)
            .allocate_slice(3)
            .unwrap();

        self.perf_counters.allocate_bufs.stop();
        // 4 flags per tranform, 1 bit each for pos, rot, scale, parent
        // 8 transforms per u32
        let mut ret = false;
        if self.model_matrix_buffer.len() < hierarchy.positions.len() as u64 {
            self.transform_buffers
                .resize(gpu, hierarchy.metadata.len() as u64, builder);
            // self.update_buffers
            //     .resize(gpu, hierarchy.metadata.len() as u64);
            self.parent_updates = gpu.buffer_array(
                (hierarchy.metadata.len() * 2).max(1) as u64,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            );
            for buf in &mut self.staging_buffers {
                buf.resize(gpu, hierarchy.metadata.len() as u64);
            }
            self.model_matrix_buffer = gpu.buffer_array(
                hierarchy.metadata.len().next_power_of_two() as u64,
                MemoryTypeFilter::PREFER_DEVICE,
                BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::VERTEX_BUFFER,
            );
            ret = true;
        }
        self.perf_counters.update_bufs.start();

        let (parent_updates_len, parent_updates) = {
            let parent_updates: Arc<Vec<Mutex<Vec<(u32, u32)>>>> = Arc::new(
                (0..rayon::current_num_threads())
                    .map(|_| Mutex::new(Vec::new()))
                    .collect(),
            );
            // if update_transforms_compute_shader {
            let position_buf = loop {
                if let Ok(b) = self.staging_buffers[sbi].position.write() {
                    break b;
                } else {
                    // println!("Waiting for position staging buffer write lock");
                }
                gpu_future.cleanup_finished();
            };
            let pos_cell = SyncUnsafeCell::new(position_buf);
            let rot_cell = SyncUnsafeCell::new(self.staging_buffers[sbi].rotation.write().unwrap());
            let scale_cell = SyncUnsafeCell::new(self.staging_buffers[sbi].scale.write().unwrap());
            let flags_cell = SyncUnsafeCell::new(self.staging_buffers[sbi].flags.write().unwrap());
            // let dirty_l2 = SyncUnsafeCell::new(self.staging_buffers[sbi].dirty_l2.write().unwrap());
            let _hierarchy = SyncUnsafeCell::new(hierarchy);

            let pos = &pos_cell;
            let rot = &rot_cell;
            let scale = &scale_cell;
            let _flags = &flags_cell;

            // const OUTER: usize = 32;
            let len = hierarchy.len();
            const INNER: usize = 32; // 32 transforms per flag u32
            let nt = rayon::current_num_threads();
            let outer = len.div_ceil(INNER).div_ceil(nt);
            (0..nt).into_par_iter().for_each(|chunk_id| {
                // let parent_updates = parent_updates.clone();
                let thread_index = rayon::current_thread_index().unwrap_or(0);
                let _parent_updates = &mut parent_updates[thread_index].lock();
                // outer * inner
                let poss = unsafe { &mut *pos.get() };
                let rots = unsafe { &mut *rot.get() };
                let scales = unsafe { &mut *scale.get() };
                let flags = unsafe { &mut *(_flags.get()) };
                // let dirty_l2 = unsafe { &mut *dirty_l2.get() };
                let hierarchy = unsafe { &*_hierarchy.get() };
                let start = chunk_id * outer;
                let end = ((chunk_id + 1) * outer).min(len.div_ceil(INNER));
                // let start = (chunk_id as f32 * outer).floor() as usize;
                // let end = (((chunk_id + 1) as f32 * outer) as usize).min(len.div_ceil(INNER));
                // let l2_data = hierarchy.get_dirty_l2(chunk_id);
                // dirty_l2[chunk_id] = l2_data;
                // if l2_data == 0 {
                //     // no changes in this chunk
                //     return;
                // }
                // dirty_l2[chunk_id] = !0;
                (start..end).for_each(|i| {
                    // 32 loops
                    let inner_start = i * INNER;
                    let inner_end = (inner_start + INNER).min(len);
                    let mut pos_flag = 0u32;
                    let mut rot_flag = 0u32;
                    let mut scl_flag = 0u32;
                    let mut bit = 0b1u32;
                    (inner_start..inner_end).for_each(|idx| {
                        // 32 loops
                        let dirty = hierarchy.get_dirty(idx as u32);
                        // if dirty & (1 << 4) != 0 {
                        // let _idx = idx * 10; // 3 pos, 4 rot, 3 scale
                        if dirty & (1 << 0) != 0 {
                            let p = unsafe { &*hierarchy.positions[idx].get() };
                            // values[_idx.._idx + 3].copy_from_slice(&p.to_array());
                            poss[idx * 3..idx * 3 + 3].copy_from_slice(&p.to_array());
                            pos_flag |= bit;
                        }
                        if dirty & (1 << 1) != 0 {
                            let r = unsafe { &*hierarchy.rotations[idx].get() };
                            rots[idx * 4..idx * 4 + 4].copy_from_slice(&r.to_array());
                            rot_flag |= bit;
                        }
                        if dirty & (1 << 2) != 0 {
                            let s = unsafe { &*hierarchy.scales[idx].get() };
                            scales[idx * 3..idx * 3 + 3].copy_from_slice(&s.to_array());
                            scl_flag |= bit;
                        }
                        if dirty & (1 << 3) != 0 {
                            let p = unsafe { &*hierarchy.metadata[idx].get() }.parent;
                            _parent_updates.push((idx as u32, p));
                        }
                        // }
                        hierarchy.mark_clean(idx as u32);
                        bit <<= 1;
                    });
                    flags[i * 3 + 0] = pos_flag;
                    flags[i * 3 + 1] = rot_flag;
                    flags[i * 3 + 2] = scl_flag;
                });
            });
            // }

            self.perf_counters.update_bufs.stop();
            self.perf_counters.update_parents.start();
            let parent_updates = Arc::try_unwrap(parent_updates).unwrap();

            let parent_updates = parent_updates
                .into_iter()
                .flat_map(|m| m.into_inner())
                .collect::<Vec<_>>();
            let parent_updates_len = parent_updates.len();
            let parent_indices = gpu
                .sub_alloc(BufferUsage::STORAGE_BUFFER)
                .allocate_slice((parent_updates_len * 2).max(1) as u64)
                .unwrap();
            {
                let mut write = parent_indices.write().unwrap();
                for (i, (id, parent_id)) in parent_updates.iter().enumerate() {
                    write[i * 2] = *id;
                    write[i * 2 + 1] = *parent_id;
                }
            }
            self.perf_counters.update_parents.stop();
            (parent_updates_len, parent_indices)
        };

        self.perf_counters.compute.start();

        {
            *pc1.write().unwrap() = cs::PushConstants {
                stage: 0,
                count: hierarchy.metadata.len() as u32,
                update: if update_transforms_compute_shader {
                    1
                } else {
                    0
                },
            };
            *pc2.write().unwrap() = cs::PushConstants {
                stage: 1,
                count: parent_updates_len as u32,
                update: if update_transforms_compute_shader {
                    1
                } else {
                    0
                },
            };
            *pc3.write().unwrap() = cs::PushConstants {
                stage: 2,
                count: hierarchy.metadata.len() as u32,
                update: if update_transforms_compute_shader {
                    1
                } else {
                    0
                },
            };
            let mut ind = indirect.write().unwrap();
            ind.copy_from_slice(&[
                DispatchIndirectCommand {
                    x: (hierarchy.metadata.len() as u32).div_ceil(128),
                    y: 1,
                    z: 1,
                },
                DispatchIndirectCommand {
                    x: (parent_updates_len as u32).div_ceil(128),
                    y: 1,
                    z: 1,
                },
                DispatchIndirectCommand {
                    x: (hierarchy.metadata.len() as u32).div_ceil(128),
                    y: 1,
                    z: 1,
                },
            ]);
        }

        if self.command_buffer.is_empty() || ret {
            self.command_buffer.clear();
            for buf in &mut self.staging_buffers {
                let mut builder = gpu.create_command_buffer(CommandBufferUsage::SimultaneousUse);

                let layout0 = self.pipeline.layout().set_layouts().get(0).unwrap();
                let set = DescriptorSet::new(
                    gpu.desc_alloc.clone(),
                    layout0.clone(),
                    [
                        WriteDescriptorSet::buffer(0, self.transform_buffers.position.clone()),
                        WriteDescriptorSet::buffer(1, self.transform_buffers.rotation.clone()),
                        WriteDescriptorSet::buffer(2, self.transform_buffers.scale.clone()),
                        WriteDescriptorSet::buffer(3, self.transform_buffers.parent.clone()),
                    ],
                    [],
                )
                .unwrap();
                let layout1 = self.pipeline.layout().set_layouts().get(1).unwrap();
                let set1 = DescriptorSet::new(
                    gpu.desc_alloc.clone(),
                    layout1.clone(),
                    [
                        WriteDescriptorSet::buffer(0, buf.position.clone()),
                        WriteDescriptorSet::buffer(1, buf.rotation.clone()),
                        WriteDescriptorSet::buffer(2, buf.scale.clone()),
                        WriteDescriptorSet::buffer(3, buf.flags.clone()),
                        WriteDescriptorSet::buffer(4, self.parent_updates.clone()),
                        // WriteDescriptorSet::buffer(5, buf.dirty_l2.clone()),
                    ],
                    [],
                )
                .unwrap();

                let layout2 = self.pipeline.layout().set_layouts().get(2).unwrap();
                let set2 = DescriptorSet::new(
                    gpu.desc_alloc.clone(),
                    layout2.clone(),
                    [WriteDescriptorSet::buffer(
                        0,
                        self.model_matrix_buffer.clone(),
                    )],
                    [],
                )
                .unwrap();

                let layout3 = self.pipeline.layout().set_layouts().get(3).unwrap();
                let set3_0 = DescriptorSet::new(
                    gpu.desc_alloc.clone(),
                    layout3.clone(),
                    [WriteDescriptorSet::buffer(0, self.pc[0].clone())],
                    [],
                )
                .unwrap();
                let set3_1 = DescriptorSet::new(
                    gpu.desc_alloc.clone(),
                    layout3.clone(),
                    [WriteDescriptorSet::buffer(0, self.pc[1].clone())],
                    [],
                )
                .unwrap();
                let set3_2 = DescriptorSet::new(
                    gpu.desc_alloc.clone(),
                    layout3.clone(),
                    [WriteDescriptorSet::buffer(0, self.pc[2].clone())],
                    [],
                )
                .unwrap();

                unsafe {
                    builder
                        .bind_pipeline_compute(self.pipeline.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            vulkano::pipeline::PipelineBindPoint::Compute,
                            self.pipeline.layout().clone(),
                            0,
                            (set.clone(), set1.clone(), set2.clone(), set3_0),
                        )
                        .unwrap()
                        // .dispatch([self.model_matrix_buffer.len().div_ceil(128) as u32, 1, 1])
                        .dispatch_indirect(self.indirect.clone().slice(0..1))
                        .unwrap()
                        .bind_descriptor_sets(
                            vulkano::pipeline::PipelineBindPoint::Compute,
                            self.pipeline.layout().clone(),
                            0,
                            (set.clone(), set1.clone(), set2.clone(), set3_1),
                        )
                        .unwrap()
                        .dispatch_indirect(self.indirect.clone().slice(1..2))
                        .unwrap()
                        .bind_descriptor_sets(
                            vulkano::pipeline::PipelineBindPoint::Compute,
                            self.pipeline.layout().clone(),
                            0,
                            (set, set1, set2, set3_2),
                        )
                        .unwrap()
                        .dispatch_indirect(self.indirect.clone().slice(2..3))
                        .unwrap()
                };
                self.command_buffer.push(builder.build().unwrap());
            }
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                indirect.clone(),
                self.indirect.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                parent_updates.clone(),
                self.parent_updates.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(pc1.clone(), self.pc[0].clone()))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(pc2.clone(), self.pc[1].clone()))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(pc3.clone(), self.pc[2].clone()))
            .unwrap();
        self.perf_counters.compute.stop();
        self.staging_buffer_index = (self.staging_buffer_index + 1) % self.staging_buffers.len();
        (ret, sbi)
    }
}
