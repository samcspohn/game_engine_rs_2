use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    image::{
        view::ImageView,
        sampler::{Sampler, SamplerCreateInfo},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::AttachmentStoreOp,
};

use crate::gpu_manager::GPUManager;
use super::font;

#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct UIVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    tex_coord: [f32; 2],
}

pub struct UI {
    gpu: Arc<GPUManager>,
    pipeline: Arc<GraphicsPipeline>,
    font_texture: Arc<ImageView>,
    sampler: Arc<Sampler>,
}

impl UI {
    pub fn new(gpu: Arc<GPUManager>) -> Self {
        let font_texture = font::create_font_texture(&gpu);
        let sampler = Sampler::new(
            gpu.device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Nearest,
                min_filter: vulkano::image::sampler::Filter::Nearest,
                ..Default::default()
            },
        )
        .unwrap();

        // Create shader modules
        let vs = vs::load(gpu.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(gpu.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = [UIVertex::per_vertex()].definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            gpu.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(gpu.device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = GraphicsPipeline::new(
            gpu.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    1,
                    ColorBlendAttachmentState {
                        blend: Some(vulkano::pipeline::graphics::color_blend::AttachmentBlend::alpha()),
                        ..Default::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(
                    vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo {
                        color_attachment_formats: vec![Some(gpu.surface_format)],
                        ..Default::default()
                    }
                    .into(),
                ),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        Self {
            gpu,
            pipeline,
            font_texture,
            sampler,
        }
    }

    /// Draw arbitrary text at a specified position
    pub fn draw_text(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        target: Arc<ImageView>,
        viewport: &Viewport,
        text: &str,
        x: f32,
        y: f32,
        char_size: f32,
    ) {
        let mut vertices = Vec::new();
        let screen_width = viewport.extent[0];
        let screen_height = viewport.extent[1].abs(); // Handle negative height

        let char_width = char_size;
        let char_height = char_size;

        for (i, c) in text.chars().enumerate() {
            if let Some(char_idx) = font::char_to_index(c) {
                let char_x = x + (i as f32) * char_width;
                let char_y = y;

                // Convert to normalized device coordinates (-1 to 1)
                // Note: viewport uses negative height for Y-flip, so we work in screen space first
                let x0 = (char_x / screen_width) * 2.0 - 1.0;
                let y0 = -((char_y / screen_height) * 2.0 - 1.0); // Flip Y for top-left origin
                let x1 = ((char_x + char_width) / screen_width) * 2.0 - 1.0;
                let y1 = -(((char_y + char_height) / screen_height) * 2.0 - 1.0); // Flip Y

                // Texture coordinates - calculate based on character position in font texture
                let col = (char_idx % font::FONT_COLS as usize) as f32;
                let row = (char_idx / font::FONT_COLS as usize) as f32;
                let u0 = col / font::FONT_COLS as f32;
                let u1 = (col + 1.0) / font::FONT_COLS as f32;
                let v0 = row / font::FONT_ROWS as f32;
                let v1 = (row + 1.0) / font::FONT_ROWS as f32;

                // Two triangles per character
                vertices.extend_from_slice(&[
                    UIVertex { position: [x0, y0], tex_coord: [u0, v0] },
                    UIVertex { position: [x1, y0], tex_coord: [u1, v0] },
                    UIVertex { position: [x0, y1], tex_coord: [u0, v1] },
                    UIVertex { position: [x1, y0], tex_coord: [u1, v0] },
                    UIVertex { position: [x1, y1], tex_coord: [u1, v1] },
                    UIVertex { position: [x0, y1], tex_coord: [u0, v1] },
                ]);
            }
        }

        if vertices.is_empty() {
            return;
        }

        // Create vertex buffer for this frame
        let vertex_buffer = Buffer::from_iter(
            self.gpu.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.into_iter(),
        )
        .unwrap();

        // Create descriptor set for texture
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            self.gpu.desc_alloc.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::sampler(0, self.sampler.clone()),
                WriteDescriptorSet::image_view(1, self.font_texture.clone()),
            ],
            [],
        )
        .unwrap();

        // Render
        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: vulkano::render_pass::AttachmentLoadOp::Load,
                    store_op: AttachmentStoreOp::Store,
                    ..RenderingAttachmentInfo::image_view(target)
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap();

        unsafe {
            builder
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_rendering()
                .unwrap();
        }
    }

    /// Draw FPS counter at top-left corner
    pub fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        target: Arc<ImageView>,
        viewport: &Viewport,
        fps: f32,
    ) {
        let fps_text = format!("FPS: {:.1}", fps);
        self.draw_text(builder, target, viewport, &fps_text, 10.0, 10.0, 16.0);
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec2 tex_coord;

            layout(location = 0) out vec2 v_tex_coord;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_tex_coord = tex_coord;
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec2 v_tex_coord;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 0) uniform sampler s;
            layout(set = 0, binding = 1) uniform texture2D tex;

            void main() {
                float alpha = texture(sampler2D(tex, s), v_tex_coord).r;
                f_color = vec4(1.0, 1.0, 1.0, alpha);
            }
        "
    }
}
