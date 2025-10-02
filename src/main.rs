// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.
//
// This version of the triangle example is written using dynamic rendering instead of render pass
// and framebuffer objects. If your device does not support Vulkan 1.3 or the
// `khr_dynamic_rendering` extension, or if you want to see how to support older versions, see the
// original triangle example.

use egui_winit_vulkano::{Gui, GuiConfig};
use glam::Vec3;
use std::{error::Error, sync::Arc, time};
use vulkano::{
    Validated, Version, VulkanError, VulkanLibrary,
    buffer::{
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferInheritanceInfo, CommandBufferUsage,
        CopyBufferInfo, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
        SecondaryAutoCommandBuffer,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{self, allocator::StandardDescriptorSetAllocator},
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod camera;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct FPS {
    frame_times: std::collections::VecDeque<f32>,
    time_sum: f32,
}

impl FPS {
    fn new() -> Self {
        Self {
            frame_times: std::collections::VecDeque::with_capacity(100),
            time_sum: 0.0,
        }
    }

    fn update(&mut self, frame_time: f32) -> f32 {
        self.frame_times.push_back(frame_time);
        self.time_sum += frame_time;
        if self.frame_times.len() > 100 {
            if let Some(removed) = self.frame_times.pop_front() {
                self.time_sum -= removed;
            }
        }
        self.frame_times.len() as f32 / self.time_sum
    }
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<descriptor_set::allocator::StandardDescriptorSetAllocator>,
    sub_allocator: SubbufferAllocator,
    vertex_buffer: Subbuffer<[MyVertex]>,
    rcx: Option<RenderContext>,
    fps: FPS,
    time: std::time::Instant,
    camera: camera::Camera,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,
    attachment_images: Vec<Arc<Image>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    gui: Gui,
    offsets: Vec<[f32; 3]>,
    command_buffer: Option<Arc<PrimaryAutoCommandBuffer>>,
    image_view: Arc<ImageView>,
    camera_uniform_buffer: Option<Subbuffer<vs::camera>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
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
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
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
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let vertices = [
            MyVertex {
                position: [-0.5 / 10.0, -0.25 / 10.0],
            },
            MyVertex {
                position: [0.0, 0.5 / 10.0],
            },
            MyVertex {
                position: [0.25 / 10.0, -0.1 / 10.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        App {
            instance,
            device,
            queue,
            command_buffer_allocator,
            descriptor_set_allocator: desc_alloc,
            memory_allocator,
            sub_allocator: sub_alloc,
            vertex_buffer,
            rcx: None,
            fps: FPS::new(),
            time: std::time::Instant::now(),
            camera: camera::Camera::new(),
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec3 position;
            layout(binding = 0) uniform camera {
                mat4 view;
                mat4 proj;
            } cam;
            
            layout(push_constant) uniform PushConstants {
                vec3 offset;
            } push_constants;

            void main() {
                gl_Position = cam.proj * cam.view * vec4(position + push_constants.offset, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                self.device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    present_mode: vulkano::swapchain::PresentMode::Mailbox,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: swapchain.image_format(),
                extent: [window_size.width, window_size.height, 1],
                mip_levels: 1,
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        );
        let _image_view = ImageView::new_default(image.unwrap()).unwrap();

        let (attachment_image_views, attachment_images) =
            window_size_dependent_setup(&images);

        let pipeline = {
            let vs = vs::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(swapchain.image_format())],
                ..Default::default()
            };

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.color_attachment_formats.len() as u32,
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };
        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        let gui = Gui::new(
            &event_loop,
            surface.clone(),
            self.queue.clone(),
            swapchain.image_format(),
            GuiConfig {
                is_overlay: true,
                ..Default::default()
            },
        );

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            attachment_image_views,
            attachment_images,
            pipeline,
            viewport,
            recreate_swapchain,
            previous_frame_end,
            gui,
            offsets: (0..1000)
                .map(|_| {
                    [
                        rand::random::<f32>() - 0.5,
                        rand::random::<f32>() - 0.5,
                        rand::random::<f32>() - 0.5,
                    ]
                })
                .collect(),
            command_buffer: None,
            image_view: _image_view,
            camera_uniform_buffer: None,
        });
    }
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                let (dx, dy) = delta;
                self.camera.rotate(Vec3::Y, -dx as f32 * 0.002);
                self.camera
                    .rotate(self.camera.rot * Vec3::X, -dy as f32 * 0.002);
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();
        let t = time::Instant::now();
        let elapsed = t.duration_since(self.time).as_secs_f32();
        let fps = self.fps.update(elapsed);
        self.time = t;

        rcx.gui.update(&event);
        match event {
            WindowEvent::KeyboardInput {
                event,
                device_id,
                is_synthetic,
            } => match event.physical_key {
                PhysicalKey::Code(keycode) => match keycode {
                    KeyCode::KeyW => self.camera.translate(Vec3::new(0.0, 0.0, -0.1)),
                    KeyCode::KeyS => self.camera.translate(Vec3::new(0.0, 0.0, 0.1)),
                    KeyCode::KeyA => self.camera.translate(Vec3::new(-0.1, 0.0, 0.0)),
                    KeyCode::KeyD => self.camera.translate(Vec3::new(0.1, 0.0, 0.0)),
                    KeyCode::KeyQ => self.camera.translate(Vec3::new(0.0, -0.1, 0.0)),
                    KeyCode::KeyE => self.camera.translate(Vec3::new(0.0, 0.1, 0.0)),
                    _ => {}
                },
                PhysicalKey::Unidentified(keycode) => {
                    println!("Unidentified key: {:?}", keycode);
                }
            },
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                println!("Window resized");
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }
                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();
                if rcx.recreate_swapchain {
                    println!("Recreating swapchain");
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");
                    rcx.swapchain = new_swapchain;
                    (rcx.attachment_image_views, rcx.attachment_images) =
                        window_size_dependent_setup(&new_images);
                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                    rcx.command_buffer = None;
                }
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                rcx.gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    egui::Window::new("fps").show(&ctx, |ui| {
                        ui.label(format!("{:.2} fps", fps));
                    });
                });

                if rcx.command_buffer.is_none() {
                    println!("Rebuilding command buffer");
                    println!("Window size: {:?}", window_size);
                    let image = Image::new(
                        self.memory_allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: rcx.swapchain.image_format(),
                            extent: [window_size.width, window_size.height, 1],
                            mip_levels: 1,
                            usage: ImageUsage::COLOR_ATTACHMENT
                                | ImageUsage::TRANSFER_SRC
                                | ImageUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    );
                    rcx.image_view = ImageView::new_default(image.unwrap()).unwrap();

                    let mut builder = AutoCommandBufferBuilder::primary(
                        self.command_buffer_allocator.clone(),
                        self.queue.queue_family_index(),
                        CommandBufferUsage::SimultaneousUse,
                    )
                    .unwrap();

                    if rcx.camera_uniform_buffer.is_none() {
                        rcx.camera_uniform_buffer = Some(
                            Buffer::new_sized(
                                self.memory_allocator.clone(),
                                BufferCreateInfo {
                                    usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                                    ..Default::default()
                                },
                                AllocationCreateInfo {
                                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                                    ..Default::default()
                                },
                            )
                            .unwrap(),
                        );
                    }
                    let set = descriptor_set::DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        rcx.pipeline.layout().set_layouts().get(0).unwrap().clone(),
                        [descriptor_set::WriteDescriptorSet::buffer(
                            0,
                            rcx.camera_uniform_buffer.as_ref().unwrap().clone(),
                        )],
                        [],
                    )
                    .unwrap();

                    // We are now inside the first subpass of the render pass.
                    // TODO: Document state setting and how it affects subsequent draw commands.
                    builder
                        .begin_rendering(RenderingInfo {
                            color_attachments: vec![Some(RenderingAttachmentInfo {
                                load_op: AttachmentLoadOp::Clear,
                                store_op: AttachmentStoreOp::Store,
                                clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                                ..RenderingAttachmentInfo::image_view(rcx.image_view.clone())
                            })],
                            ..Default::default()
                        })
                        .unwrap()
                        .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                        .unwrap()
                        .bind_pipeline_graphics(rcx.pipeline.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            vulkano::pipeline::PipelineBindPoint::Graphics,
                            rcx.pipeline.layout().clone(),
                            0,
                            set.clone(),
                        )
                        .unwrap()
                        .bind_vertex_buffers(0, self.vertex_buffer.clone())
                        .unwrap();
                    for offset in &rcx.offsets {
                        unsafe {
                            builder
                                .push_constants(
                                    rcx.pipeline.layout().clone(),
                                    0,
                                    vs::PushConstants { offset: *offset },
                                )
                                .unwrap()
                                .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                        }
                        .unwrap();
                    }
                    builder.end_rendering().unwrap();
                    // We leave the render pass.
                    rcx.command_buffer = Some(builder.build().unwrap());
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                let cam_data_buf = self.sub_allocator.allocate_sized().unwrap();
                *cam_data_buf.write().unwrap() = vs::camera {
                    view: self.camera.get_view_matrix().to_cols_array_2d(),
                    proj: self
                        .camera
                        .get_proj_matrix(window_size.width as f32 / window_size.height as f32)
                        .to_cols_array_2d(),
                };
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        cam_data_buf,
                        rcx.camera_uniform_buffer.as_ref().unwrap().clone(),
                    ))
                    .unwrap();
                let command_buffer = builder.build().unwrap();

                let a = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap();

                let a = a
                    .then_signal_semaphore()
                    .then_execute(
                        self.queue.clone(),
                        rcx.command_buffer.as_ref().unwrap().clone(),
                    )
                    .unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                builder
                    .blit_image(BlitImageInfo::images(
                        rcx.image_view.image().clone(),
                        rcx.attachment_images[image_index as usize].clone(),
                    ))
                    .unwrap();

                let a = a
                    .then_signal_semaphore()
                    .then_execute(self.queue.clone(), builder.build().unwrap())
                    .unwrap();

                let mut b = Some(
                    rcx.gui
                        .draw_on_image(a, rcx.attachment_image_views[image_index as usize].clone()),
                );

                let future = b
                    .take()
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(images: &[Arc<Image>]) -> (Vec<Arc<ImageView>>, Vec<Arc<Image>>) {
    let image_views = images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>();
    (image_views, images.to_vec())
}
