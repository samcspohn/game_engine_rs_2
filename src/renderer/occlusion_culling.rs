use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    image::{
        sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode},
        view::ImageView,
    },
    pipeline::{
        compute::ComputePipelineCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint,
        PipelineShaderStageCreateInfo, PipelineLayout,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
};

use crate::{camera::Camera, gpu_manager::GPUManager};

mod depth_copy_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/occlusion/depth_copy.comp",
    }
}

mod hiz_generate_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/occlusion/hiz_generate.comp",
    }
}

/// Hi-Z (Hierarchical Z-Buffer) Generator for GPU occlusion culling
/// 
/// Generates a mipmap pyramid from the depth buffer where each level contains
/// the minimum depth values from the previous level. This allows fast occlusion
/// queries at different scales.
/// 
/// # Usage Example
/// 
/// ```rust
/// // In your rendering system initialization:
/// let hiz_generator = HiZGenerator::new(&gpu);
/// 
/// // In your camera setup:
/// let mut camera = Camera::new(&gpu, [1920, 1080], 0.1, 1000.0);
/// camera.create_hiz(&gpu, [1920, 1080]);
/// 
/// // After rendering each frame, generate Hi-Z pyramid:
/// hiz_generator.generate_hiz(&gpu, &mut command_builder, &camera);
/// 
/// // Now the Hi-Z pyramid can be used for occlusion testing in the next frame
/// ```
/// 
/// # Implementation Details
/// 
/// - Uses a two-step process:
///   1. Copy depth buffer (D32_SFLOAT) to Hi-Z mip 0 (R32_SFLOAT)
///   2. Generate mip levels 1-N using shared memory reduction (4 mips per pass)
/// - Each mip contains MIN depth (furthest point) for conservative culling
/// - Uses 16x16 workgroups with shared memory for efficient reduction
pub struct HiZGenerator {
    depth_copy_pipeline: Arc<ComputePipeline>,
    hiz_generate_pipeline: Arc<ComputePipeline>,
    sampler: Arc<Sampler>,
}

impl HiZGenerator {
    /// Create a new Hi-Z generator
    /// 
    /// # Arguments
    /// * `gpu` - The GPU manager containing device and allocators
    pub fn new(gpu: &GPUManager) -> Self {
        // Create depth copy pipeline
        let depth_copy_cs = depth_copy_cs::load(gpu.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let depth_copy_stage = PipelineShaderStageCreateInfo::new(depth_copy_cs);
        let depth_copy_layout = PipelineLayout::new(
            gpu.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&depth_copy_stage])
                .into_pipeline_layout_create_info(gpu.device.clone())
                .unwrap(),
        )
        .unwrap();

        let depth_copy_pipeline = ComputePipeline::new(
            gpu.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(depth_copy_stage, depth_copy_layout),
        )
        .unwrap();

        // Create Hi-Z generation pipeline
        let hiz_generate_cs = hiz_generate_cs::load(gpu.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let hiz_generate_stage = PipelineShaderStageCreateInfo::new(hiz_generate_cs);
        let hiz_generate_layout = PipelineLayout::new(
            gpu.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&hiz_generate_stage])
                .into_pipeline_layout_create_info(gpu.device.clone())
                .unwrap(),
        )
        .unwrap();

        let hiz_generate_pipeline = ComputePipeline::new(
            gpu.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(hiz_generate_stage, hiz_generate_layout),
        )
        .unwrap();

        // Create sampler for reading depth/Hi-Z textures
        let sampler = Sampler::new(
            gpu.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                mipmap_mode: SamplerMipmapMode::Nearest,
                ..Default::default()
            },
        )
        .unwrap();

        Self {
            depth_copy_pipeline,
            hiz_generate_pipeline,
            sampler,
        }
    }

    /// Generate the complete Hi-Z pyramid from the camera's depth buffer
    /// 
    /// This should be called after rendering completes each frame. The resulting
    /// Hi-Z pyramid can then be used for occlusion culling in the next frame
    /// (one frame of latency is acceptable).
    /// 
    /// # Arguments
    /// * `gpu` - The GPU manager
    /// * `builder` - Command buffer builder for recording GPU commands
    /// * `camera` - Camera with initialized Hi-Z resources (call `create_hiz()` first)
    /// 
    /// # Panics
    /// Panics if the camera's Hi-Z image and views have not been initialized
    pub fn generate_hiz(
        &self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
    ) {
        let _hiz_image = camera.hiz_image.as_ref().expect("Hi-Z image not initialized");
        let hiz_views = camera.hiz_views.as_ref().expect("Hi-Z views not initialized");
        let mip_levels = camera.hiz_mip_levels;

        if mip_levels == 0 {
            return;
        }

        let width = camera.depth_image.extent()[0];
        let height = camera.depth_image.extent()[1];

        // Step 1: Copy depth buffer to Hi-Z mip 0
        self.copy_depth_to_hiz_mip0(gpu, builder, camera);

        // Step 2: Generate remaining mips in batches of 4 using shared memory reduction
        let mut src_mip = 0u32;
        
        while src_mip < mip_levels - 1 {
            let remaining_mips = mip_levels - src_mip - 1;
            let num_mips_this_pass = remaining_mips.min(4);

            self.generate_mip_batch(
                gpu,
                builder,
                hiz_views,
                src_mip,
                num_mips_this_pass,
                width >> src_mip,
                height >> src_mip,
            );

            // Each pass generates up to 4 mips, but the next pass starts from the last generated
            // For shared memory reduction: mip N+1 from mip N, so we advance by the mips generated
            src_mip += 4;
        }
    }

    /// Copy depth buffer (D32_SFLOAT) to Hi-Z mip 0 (R32_SFLOAT)
    /// 
    /// This performs format conversion since depth attachments use depth formats
    /// but compute shaders need color formats for read/write access.
    fn copy_depth_to_hiz_mip0(
        &self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
    ) {
        let hiz_views = camera.hiz_views.as_ref().unwrap();
        let depth_view = &camera.depth_view;
        let hiz_mip0 = &hiz_views[0];

        let width = camera.depth_image.extent()[0];
        let height = camera.depth_image.extent()[1];

        // Create descriptor set
        let layout = self.depth_copy_pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            gpu.desc_alloc.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    depth_view.clone(),
                    self.sampler.clone(),
                ),
                WriteDescriptorSet::image_view(1, hiz_mip0.clone()),
            ],
            [],
        )
        .unwrap();

        // Dispatch compute shader
        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        unsafe {
            builder
                .bind_pipeline_compute(self.depth_copy_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.depth_copy_pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .dispatch([workgroups_x, workgroups_y, 1])
                .unwrap();
        }
    }

    /// Generate a batch of mip levels (up to 4) using shared memory reduction
    /// 
    /// Each 16x16 workgroup can generate up to 4 mip levels in a single pass
    /// by progressively reducing data in shared memory. This is much more
    /// efficient than separate passes per mip level.
    fn generate_mip_batch(
        &self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        hiz_views: &[Arc<ImageView>],
        src_mip: u32,
        num_mips: u32,
        src_width: u32,
        src_height: u32,
    ) {
        let src_view = &hiz_views[src_mip as usize];

        // Bind output mips (up to 4)
        let mut write_sets = vec![
            WriteDescriptorSet::image_view_sampler(0, src_view.clone(), self.sampler.clone()),
        ];

        for i in 0..num_mips {
            let dst_mip = src_mip + i + 1;
            if (dst_mip as usize) < hiz_views.len() {
                write_sets.push(WriteDescriptorSet::image_view(
                    (i + 1) as u32,
                    hiz_views[dst_mip as usize].clone(),
                ));
            }
        }

        // Fill remaining slots with dummy views if needed
        while write_sets.len() < 5 {
            // Reuse the first output mip as a dummy (won't be written to)
            let dummy_mip = src_mip + 1;
            if (dummy_mip as usize) < hiz_views.len() {
                write_sets.push(WriteDescriptorSet::image_view(
                    write_sets.len() as u32,
                    hiz_views[dummy_mip as usize].clone(),
                ));
            }
        }

        let layout = self.hiz_generate_pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            gpu.desc_alloc.clone(),
            layout.clone(),
            write_sets,
            [],
        )
        .unwrap();

        // Push constants
        let push_constants = hiz_generate_cs::PushConstants {
            srcResolution: [src_width, src_height],
            numMips: num_mips,
        };

        // Each workgroup processes a 16x16 tile
        let workgroups_x = (src_width + 15) / 16;
        let workgroups_y = (src_height + 15) / 16;

        unsafe {
            builder
                .bind_pipeline_compute(self.hiz_generate_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.hiz_generate_pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(
                    self.hiz_generate_pipeline.layout().clone(),
                    0,
                    push_constants,
                )
                .unwrap()
                .dispatch([workgroups_x, workgroups_y, 1])
                .unwrap();
        }
    }
}