use std::{collections::HashMap, hash::Hash, sync::Arc};

// use egui_winit_vulkano::{Gui, GuiConfig};
use parking_lot::Mutex;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{self, DescriptorSet, WriteDescriptorSet, layout::DescriptorBindingFlags},
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp, Subpass},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{self, GpuFuture},
};
use winit::{dpi::LogicalSize, event_loop::ActiveEventLoop, window::Window};

use crate::{
    NUM_CUBES,
    asset_manager::{AssetHandle, AssetManager},
    camera::Camera,
    gpu_manager::GPUManager,
    obj_loader::Model,
    texture::Texture,
};

pub struct RenderContext {
    pub window: Arc<Window>,
    pub swapchain: Arc<Swapchain>,
    pub attachment_image_views: Vec<Arc<ImageView>>,
    pub attachment_images: Vec<Arc<Image>>,
    pub pipeline: Arc<GraphicsPipeline>,
    pub viewport: Viewport,
    pub recreate_swapchain: bool,
    // pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    // pub gui: Arc<Mutex<Gui>>,
    // pub gui_drawn: bool,
    pub command_buffer: Option<Arc<PrimaryAutoCommandBuffer>>,
    // pub image_view: Arc<ImageView>,
    // pub camera_uniform_buffer: Option<Subbuffer<crate::vs::camera>>,
    pub camera: Arc<Mutex<Camera>>,
    pub input: crate::input::Input,
    pub focused: bool,
    pub frame_time: crate::PerfCounter,
    pub swap_chain_perf: crate::PerfCounter,
    pub update_camera_perf: crate::PerfCounter,
    pub build_command_buffer_perf: crate::PerfCounter,
    pub execute_command_buffer_perf: crate::PerfCounter,
    pub cleanup: crate::PerfCounter,
    pub acquire_next_image_perf: crate::PerfCounter,
    pub draw_gui_perf: crate::PerfCounter,
    pub extra_perfs: HashMap<String, crate::PerfCounter>,
}

impl RenderContext {
    pub fn new(event_loop: &ActiveEventLoop, gpu: &GPUManager, camera: Arc<Mutex<Camera>>) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_inner_size(LogicalSize {
					width: 800.0,
					height: 600.0,
				}))
                .unwrap(),
        );
        let surface = Surface::from_window(gpu.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = gpu
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (image_format, _) = gpu
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                gpu.device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: gpu.image_count,
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
        println!("Swapchain image count: {}", swapchain.image_count());
        let image = Image::new(
            gpu.mem_alloc.clone(),
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

        let (attachment_image_views, attachment_images) = window_size_dependent_setup(&images);

        let pipeline = {
            let vs = crate::vs::load(gpu.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = crate::fs::load(gpu.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = [
                MyVertex::per_vertex(),
                MyUV::per_vertex(),
                MyNormal::per_vertex(),
                MyTangent::per_vertex(),
                MyColor::per_vertex(),
                TransformID::per_instance(),
            ]
            .definition(&vs)
            .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            // let layout = PipelineLayout::new(
            //     gpu.device.clone(),
            //     PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            //         .into_pipeline_layout_create_info(gpu.device.clone())
            //         .unwrap(),
            // )
            // .unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(swapchain.image_format())],
                depth_attachment_format: Some(Format::D32_SFLOAT),
                ..Default::default()
            };

            let mut layout_create_info =
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
            let binding = layout_create_info.set_layouts[0]
                .bindings
                .get_mut(&3)
                .unwrap();
            binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
            binding.descriptor_count = 1024; // max number of textures
            let pipeline_layout = PipelineLayout::new(
                gpu.device.clone(),
                layout_create_info
                    .into_pipeline_layout_create_info(gpu.device.clone())
                    .unwrap(),
            )
            .unwrap();

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                gpu.device.clone(),
                Some(gpu.pipeline_cache.clone()),
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
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
                },
            )
            .unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, window_size.height as f32],
            extent: [window_size.width as f32, -(window_size.height as f32)],
            depth_range: 0.0..=1.0,
        };
        let recreate_swapchain = false;
        // let previous_frame_end = Some(sync::now(gpu.device.clone()).boxed());

        // let gui = Gui::new(
        //     &event_loop,
        //     surface.clone(),
        //     gpu.queue.clone(),
        //     swapchain.image_format(),
        //     GuiConfig {
        //         is_overlay: true,
        //         allow_srgb_render_target: true,
        //         ..Default::default()
        //     },
        // );

        // let mut gui = Gui::new_with_subpass(
        //     &event_loop,
        //     surface.clone(),
        //     gpu.queue.clone(),
        //     Subpass::from(render),
        //     gpu.surface_format,
        //     GuiConfig::default(),
        // );

        RenderContext {
            window,
            swapchain,
            attachment_image_views,
            attachment_images,
            pipeline,
            viewport,
            recreate_swapchain,
            // previous_frame_end,
            // gui: Arc::new(Mutex::new(gui)),
            // gui_drawn: false,
            command_buffer: None,
            camera,
            input: crate::input::Input::new(),
            focused: false,
            frame_time: crate::PerfCounter::new(),
            swap_chain_perf: crate::PerfCounter::new(),
            update_camera_perf: crate::PerfCounter::new(),
            build_command_buffer_perf: crate::PerfCounter::new(),
            execute_command_buffer_perf: crate::PerfCounter::new(),
            cleanup: crate::PerfCounter::new(),
            acquire_next_image_perf: crate::PerfCounter::new(),
            draw_gui_perf: crate::PerfCounter::new(),
            extra_perfs: HashMap::new(),
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let window_size = self.window.inner_size();
        println!("Recreating swapchain");
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: window_size.into(),
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");
        println!("New swapchain image count: {}", new_images.len());
        self.swapchain = new_swapchain;
        (self.attachment_image_views, self.attachment_images) =
            window_size_dependent_setup(&new_images);
        self.viewport.extent = [window_size.width as f32, -(window_size.height as f32)];
        self.viewport.offset = [0.0, window_size.height as f32];
        self.recreate_swapchain = false;
        self.command_buffer = None;
    }

    pub fn rebuild_command_buffer(
        &mut self,
        gpu: &GPUManager,
        assets: &AssetManager,
        rendering_system: &mut crate::renderer::RenderingSystem,
        matrix_data: &Subbuffer<[crate::transform::compute::cs::MatrixData]>,
        hiz_frozen: bool,
    ) {
        let window_size = self.window.inner_size();

        let mut camera = self.camera.lock();
        camera.resize(gpu, [window_size.width, window_size.height]);

        let mut builder = gpu.create_command_buffer(CommandBufferUsage::SimultaneousUse);

        rendering_system.update_mvp(camera.uniform_culling.clone(), &mut builder, matrix_data, &camera);

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(camera.image_view.clone())
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(1f32.into()),
                    ..RenderingAttachmentInfo::image_view(camera.depth_view.clone())
                }),
                ..RenderingInfo::default()
            })
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap();

        rendering_system.draw(
            &mut builder,
            assets,
            self.pipeline.clone(),
            camera.uniform.clone(),
            matrix_data,
        );
        builder.end_rendering().unwrap();

        // Generate Hi-Z AFTER rendering so it's ready for the next frame's occlusion culling
        // Note: Vulkano handles synchronization automatically between rendering and compute
        // This includes copying depth to Hi-Z mip 0, then generating the mip pyramid
        rendering_system.generate_hiz(&mut builder, &camera, hiz_frozen);

        // We leave the render pass.
        self.command_buffer = Some(builder.build().unwrap());
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyUV {
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyNormal {
    #[format(R8G8B8A8_SNORM)]
    normal: u32,
}
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyTangent {
    #[format(R8G8B8A8_SNORM)]
    tangent: u32,
}
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyColor {
    #[format(R8G8B8A8_UNORM)]
    color: u32,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct TransformID {
    #[format(R32G32_UINT)]
    rd: [u32; 2],
}

// #[derive(BufferContents, Vertex)]
// #[repr(C)]
// struct InstanceMatrix {
//     #[format(R32G32B32A32_SFLOAT)]
//     mvp_matrix: [[f32; 4]; 4],
// }
// #[derive(BufferContents, Vertex)]
// #[repr(C)]
// struct NormalMatrix {
//     #[format(R32G32B32_SFLOAT)]
//     normal_matrix: [[f32; 4]; 3],
// }

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(images: &[Arc<Image>]) -> (Vec<Arc<ImageView>>, Vec<Arc<Image>>) {
    let image_views = images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>();
    (image_views, images.to_vec())
}
