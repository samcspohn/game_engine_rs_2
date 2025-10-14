use std::{
    collections::HashMap,
    ops::Sub,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use parking_lot::{Mutex, RwLock};
use vulkano::{
    Version, VulkanLibrary,
    buffer::{
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    swapchain::Surface,
    sync::{GpuFuture, PipelineStage},
};
use winit::{event_loop::EventLoop, window::Window};

pub mod gpu_vector;

trait GPUWorkItemBase {
    fn is_completed(&self) -> bool;
    fn call(&self, gpu: &GPUManager);
}

pub struct GPUWorkItem<T> {
    pub work: Box<dyn Fn(&GPUManager) -> T + Send + Sync>,
    pub completed: AtomicBool,
    pub result: Mutex<Option<T>>,
}

impl<T> GPUWorkItemBase for GPUWorkItem<T> {
    fn is_completed(&self) -> bool {
        self.completed.load(Ordering::SeqCst)
    }
    fn call(&self, gpu: &GPUManager) {
        self.call(gpu);
    }
}
impl<T> GPUWorkItem<T> {
    pub fn wait(&self) -> Option<T> {
        while !self.completed.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::yield_now();
        }
        self.result.lock().take()
    }

    pub fn call(&self, gpu: &GPUManager) {
        let result = self.work.as_ref()(gpu);
        *self.result.lock() = Some(result);
        self.completed.store(true, Ordering::SeqCst);
    }
}

pub struct GPUWorkQueue {
    pub work_queue: Arc<Mutex<Vec<Arc<dyn GPUWorkItemBase + Send + Sync>>>>,
}
impl GPUWorkQueue {
    pub fn new() -> Self {
        Self {
            work_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn enqueue_work<F, T>(&self, work: F) -> Arc<GPUWorkItem<T>>
    where
        F: Fn(&GPUManager) -> T + Send + Sync + 'static,
        T: Send + 'static,
    {
        let item = Arc::new(GPUWorkItem {
            work: Box::new(work),
            completed: AtomicBool::new(false),
            result: Mutex::new(None),
        });
        self.work_queue.lock().push(item.clone());
        item
    }

    pub fn lock(&self) -> parking_lot::MutexGuard<'_, Vec<Arc<dyn GPUWorkItemBase + Send + Sync>>> {
        self.work_queue.lock()
    }
}

impl Clone for GPUWorkQueue {
    fn clone(&self) -> Self {
        Self {
            work_queue: self.work_queue.clone(),
        }
    }
}

pub struct GPUManager {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub cmd_alloc: Arc<StandardCommandBufferAllocator>,
    pub mem_alloc: Arc<StandardMemoryAllocator>,
    pub desc_alloc: Arc<StandardDescriptorSetAllocator>,
    // pub pipeline_cache: Arc<PipelineCache>,
    // pub sub_alloc: Arc<SubbufferAllocator>,
    // pub storage_alloc: Arc<SubbufferAllocator>,
    // pub ind_alloc: Arc<SubbufferAllocator>,
    pub _sub_alloc: RwLock<HashMap<BufferUsage, Arc<SubbufferAllocator>>>,
    pub work_queue: GPUWorkQueue,
    pub surface_format: Format,
    pub image_count: u32,
    pub query_pool: Arc<QueryPool>,
    pub empty: Subbuffer<[u8]>,
}

// pub type GPUWorkQueue = Arc<Mutex<Vec<Arc<dyn GPUWorkItemBase>>>>;

impl GPUManager {
    pub fn new(event_loop: &EventLoop<()>) -> Arc<Self> {
        let library = VulkanLibrary::new().unwrap();

        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    geometry_shader: true,
                    runtime_descriptor_array: true,
                    descriptor_binding_variable_descriptor_count: true,
                    descriptor_indexing: true,
                    shader_sampled_image_array_dynamic_indexing: true,
                    shader_sampled_image_array_non_uniform_indexing: true,
                    multi_draw_indirect: true,
                    shader_draw_parameters: true,
                    subgroup_broadcast_dynamic_id: true,
                    draw_indirect_count: true,
                    ..DeviceFeatures::empty()
                },

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();
        let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cmd_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let desc_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let sub_alloc = SubbufferAllocator::new(
            mem_alloc.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let storage_alloc = SubbufferAllocator::new(
            mem_alloc.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let ind_alloc = SubbufferAllocator::new(
            mem_alloc.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let surface_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let empty = Buffer::new_slice(
            mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::STORAGE_BUFFER
                    | BufferUsage::INDIRECT_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            1,
        )
        .unwrap();

        Arc::new(Self {
            empty,
            instance,
            device: device.clone(),
            queue,
            cmd_alloc,
            mem_alloc,
            desc_alloc,
            // pipeline_cache,
            // sub_alloc: Arc::new(sub_alloc),
            // storage_alloc: Arc::new(storage_alloc),
            // ind_alloc: Arc::new(ind_alloc),
            _sub_alloc: RwLock::new(HashMap::new()),
            work_queue: GPUWorkQueue::new(),
            surface_format,
            image_count: surface_capabilities.min_image_count.max(2),
            query_pool: QueryPool::new(
                device.clone(),
                QueryPoolCreateInfo {
                    query_count: 1000,
                    ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
                },
            )
            .unwrap(),
        })
    }

    pub fn sub_alloc(&self, usage: BufferUsage) -> Arc<SubbufferAllocator> {
        if let Some(alloc) = self
            ._sub_alloc
            .read()
            .get(&(usage))
        {
            return alloc.clone();
        }
        let alloc = Arc::new(SubbufferAllocator::new(
            self.mem_alloc.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: usage | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        ));
        self._sub_alloc
            .write()
            .insert(usage, alloc.clone());
        alloc
    }

    pub fn begin_query(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        // query_index: u32,
    ) {
        unsafe {
            builder
                .reset_query_pool(self.query_pool.clone(), 0..2)
                .unwrap()
                .write_timestamp(self.query_pool.clone(), 0, PipelineStage::TopOfPipe)
                .unwrap();
        }
    }
    pub fn end_query(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        // query_index: u32,
    ) {
        unsafe {
            builder
                .write_timestamp(self.query_pool.clone(), 1, PipelineStage::BottomOfPipe)
                .unwrap();
        }
    }

    pub fn get_query_results(&self) -> u64 {
        let mut data = [0u64; 2];
        loop {
            if let Ok(res) = self
                .query_pool
                .get_results(0..2, &mut data, QueryResultFlags::WAIT)
            {
                if res {
                    break;
                }
                // res.get_results(&mut query_results, QueryResultFlags::WAIT)
                //     .unwrap();
            }
        }

        // todo!();
        ((data[1] - data[0]) as f64
            * self.device.physical_device().properties().timestamp_period as f64) as u64
    }

    pub(crate) fn process_work_queue(&self) -> bool {
        let mut work_items = self.work_queue.lock();
        let items: Vec<_> = work_items.drain(..).collect();
        drop(work_items);
        let ret = !items.is_empty();
        for item in items {
            println!("Processing GPU work item");
            item.call(self);
            println!("GPU work item completed");
        }
        ret
    }

    pub fn wait_idle(&self) {
        unsafe { self.device.wait_idle().unwrap() };
    }

    pub fn buffer_array<T>(
        &self,
        size: u64,
        memory_type_filter: MemoryTypeFilter,
        usage: BufferUsage,
    ) -> Subbuffer<T>
    where
        T: BufferContents + ?Sized,
    {
        let buf = Buffer::new_unsized(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
            size,
        )
        .unwrap();
        buf
    }

    pub fn buffer_from_iter<T, I>(&self, iter: I, usage: BufferUsage) -> Subbuffer<[T]>
    where
        T: BufferContents + Copy,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let buf = Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: usage | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            iter,
        )
        .unwrap();
        let buf2: Subbuffer<[T]> = self.buffer_array(
            buf.len(),
            MemoryTypeFilter::PREFER_DEVICE,
            usage | BufferUsage::TRANSFER_DST,
        );

        let mut builder = AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(buf, buf2.clone()))
            .unwrap();
        builder
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        // builder.copy_buffer(CopyBufferInfo::buffers()).unwrap();
        buf2
    }
    pub fn buffer_data<T>(
        &self,
        memory_type_filter: MemoryTypeFilter,
        usage: BufferUsage,
    ) -> Subbuffer<T>
    where
        T: BufferContents + Copy,
    {
        let buf: Subbuffer<T> = Buffer::new_sized(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
        )
        .unwrap();
        buf
    }

    pub fn buffer_from_data<T>(&self, data: &T, usage: BufferUsage) -> Subbuffer<T>
    where
        T: BufferContents + Copy,
    {
        let buf = Buffer::from_data(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: usage | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            *data,
        )
        .unwrap();
        let buf2: Subbuffer<T> = self.buffer_array(
            1,
            MemoryTypeFilter::PREFER_DEVICE,
            usage | BufferUsage::TRANSFER_DST,
        );

        let mut builder = AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(buf, buf2.clone()))
            .unwrap();
        builder
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        // builder.copy_buffer(CopyBufferInfo::buffers()).unwrap();
        buf2
    }

    pub fn create_staging_buffer<T>(&self, data: &[T], usage: BufferUsage) -> Subbuffer<[T]>
    where
        T: BufferContents + Copy,
    {
        let buf = Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: usage | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().cloned(),
        )
        .unwrap();
        buf
    }

    pub fn flush_staging_buffer<T>(&self, staging: Subbuffer<[T]>, dest: Subbuffer<[T]>)
    where
        T: BufferContents + Copy,
    {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(staging, dest))
            .unwrap();
        builder
            .build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn create_command_buffer(
        &self,
        usage: CommandBufferUsage,
    ) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            usage,
        )
        .unwrap()
    }

    // pub fn submit_command_buffer<C>(&self, cmd: C) -> impl GpuFuture
    // where
    //     C: PrimaryCommandBufferAbstract + Send + Sync + 'static,
    // {
    //     let future = cmd
    //         .execute(self.queue.clone())
    //         .unwrap()
    //         .then_signal_fence_and_flush()
    //         .unwrap();
    //     future
    // }

    // pub fn submit_and_wait<C>(&self, cmd: C)
    // where
    //     C: PrimaryCommandBufferAbstract + Send + Sync + 'static,
    // {
    //     cmd.execute(self.queue.clone())
    //         .unwrap()
    //         .then_signal_fence_and_flush()
    //         .unwrap()
    //         .wait(None)
    //         .unwrap();
    // }
}
