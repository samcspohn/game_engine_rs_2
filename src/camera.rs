use core::net;
use std::sync::Arc;

use crate::{RenderContext, gpu_manager::GPUManager, input::Input};
use glam::{Quat, Vec3};
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageSubresourceRange, ImageTiling, ImageType, ImageUsage, sampler::Sampler, view::{ImageView, ImageViewCreateInfo}
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};
use winit::keyboard::KeyCode;

pub struct Camera {
    pub pos: Vec3,
    pub rot: Quat,
    pub near: f32,
    pub far: f32,
    pub image: Arc<Image>,
    pub image_view: Arc<ImageView>,
    pub depth_image: Arc<Image>,
    pub depth_view: Arc<ImageView>,
    pub uniform: Subbuffer<crate::vs::camera>,
    pub uniform_culling: Subbuffer<crate::vs::camera>,
    pub uniform_hi_z_info: Subbuffer<crate::renderer::cs::HiZInfo>,
    pub format: Format,
    pub hiz_image: Option<Arc<Image>>,
    pub hiz_views: Option<Vec<Arc<ImageView>>>,
    pub hiz_view_all_mips: Option<Arc<ImageView>>,
    pub hiz_sampler: Option<Arc<Sampler>>,
    pub hiz_mip_levels: u32,
}

impl Camera {
    pub fn new(gpu: &GPUManager, dimensions: [u32; 2], near: f32, far: f32) -> Self {
        let format = gpu.surface_format;
        let image = Image::new(
            gpu.mem_alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: format,
                extent: [dimensions[0], dimensions[1], 1],
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
        )
        .unwrap();
        let image_view = ImageView::new_default(image.clone()).unwrap();

        let depth_image = Image::new(
            gpu.mem_alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::D32_SFLOAT,
                extent: [dimensions[0], dimensions[1], 1],
                mip_levels: 1,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::SAMPLED,
                tiling: ImageTiling::Optimal,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        let depth_view = ImageView::new_default(depth_image.clone()).unwrap();

        let uniform = gpu.buffer_from_data(
            &crate::vs::camera {
                view: glam::Mat4::IDENTITY.to_cols_array_2d(),
                proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            },
            BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_SRC,
        );
        let uniform_culling = gpu.buffer_from_data(
            &crate::vs::camera {
                view: glam::Mat4::IDENTITY.to_cols_array_2d(),
                proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            },
            BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
        );
        let uniform_hi_z_info = gpu.buffer_from_data(
            &crate::renderer::cs::HiZInfo {
                screenResolution: [dimensions[0] as i32, dimensions[1] as i32],
                // mip_levels: 0,
                // _pad: [0; 3],
            },
            BufferUsage::UNIFORM_BUFFER,
        );

        let mut cam = Self {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
            image,
            image_view,
            depth_image,
            depth_view,
            uniform,
            uniform_culling,
            uniform_hi_z_info,
            format,
            near,
            far,
            hiz_image: None,
            hiz_views: None,
            hiz_view_all_mips: None,
            hiz_sampler: None,
            hiz_mip_levels: 0,
        };
        cam.create_hiz(gpu, dimensions);
        cam
    }

    /// Create Hi-Z image and views for occlusion culling
    pub fn create_hiz(&mut self, gpu: &GPUManager, dimensions: [u32; 2]) {
        let hiz_mip_levels = {
            let max_dimension = dimensions[0].max(dimensions[1]);
            (max_dimension as f32).log2().floor() as u32 + 1
        };

        let hiz_image = Image::new(
            gpu.mem_alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R32_SFLOAT,
                extent: [dimensions[0], dimensions[1], 1],
                mip_levels: hiz_mip_levels,
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        let hiz_views: Vec<_> = (0..hiz_mip_levels)
            .map(|mip| {
                ImageView::new(
                    hiz_image.clone(),
                    ImageViewCreateInfo {
                        format: Format::R32_SFLOAT,
                        subresource_range: ImageSubresourceRange {
                            aspects: ImageAspects::COLOR,
                            mip_levels: mip..(mip + 1),
                            array_layers: 0..1,
                        },
                        ..ImageViewCreateInfo::from_image(&hiz_image)
                    },
                )
                .unwrap()
            })
            .collect();

        // Create view of ALL mip levels for shader sampling
        let hiz_view_all_mips = ImageView::new(
            hiz_image.clone(),
            ImageViewCreateInfo {
                format: Format::R32_SFLOAT,
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    mip_levels: 0..hiz_mip_levels,
                    array_layers: 0..1,
                },
                ..ImageViewCreateInfo::from_image(&hiz_image)
            },
        )
        .unwrap();

        let hiz_sampler = Sampler::new(
			gpu.device.clone(),
			vulkano::image::sampler::SamplerCreateInfo {
				mag_filter: vulkano::image::sampler::Filter::Nearest,
				min_filter: vulkano::image::sampler::Filter::Nearest,
				mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Nearest,
				lod: 0.0..=(hiz_mip_levels as f32),
				..Default::default()
			},
		)
		.unwrap();

        self.hiz_image = Some(hiz_image);
        self.hiz_views = Some(hiz_views);
        self.hiz_view_all_mips = Some(hiz_view_all_mips);
        self.hiz_sampler = Some(hiz_sampler);
        self.hiz_mip_levels = hiz_mip_levels;
    }

    pub fn resize(&mut self, gpu: &GPUManager, dimensions: [u32; 2]) {
        if dimensions == self.image.extent()[0..2] {
            return;
        }
        let image = Image::new(
            gpu.mem_alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: self.format,
                extent: [dimensions[0], dimensions[1], 1],
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
        )
        .unwrap();
        self.image = image;
        self.image_view = ImageView::new_default(self.image.clone()).unwrap();

        let depth_image = Image::new(
            gpu.mem_alloc.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::D32_SFLOAT,
                extent: [dimensions[0], dimensions[1], 1],
                mip_levels: 1,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::SAMPLED,
                // tiling: ImageTiling::Optimal,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        self.depth_view = ImageView::new_default(depth_image.clone()).unwrap();
        self.depth_image = depth_image;

        self.create_hiz(gpu, dimensions);
        self.uniform_hi_z_info = gpu.buffer_from_data(
            &crate::renderer::cs::HiZInfo {
                screenResolution: [dimensions[0] as i32, dimensions[1] as i32],
                // mip_levels: 0,
                // _pad: [0; 3],
            },
            BufferUsage::UNIFORM_BUFFER,
        );
    }
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        let forward = self.rot * Vec3::Z;
        glam::Mat4::look_at_lh(self.pos, self.pos + forward, self.rot * Vec3::Y)
    }
    pub fn get_proj_matrix(&self, aspect: f32) -> glam::Mat4 {
        glam::Mat4::perspective_lh(std::f32::consts::FRAC_PI_2, aspect, self.near, self.far)
    }
    pub fn update_uniform(
        &self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        aspect: f32,
    ) {
        let cam_data_buf = gpu
            .sub_alloc(BufferUsage::UNIFORM_BUFFER)
            .allocate_sized()
            .unwrap();
        {
            let mut cam_data = cam_data_buf.write().unwrap();
            *cam_data = crate::vs::camera {
                view: self.get_view_matrix().to_cols_array_2d(),
                proj: self.get_proj_matrix(aspect).to_cols_array_2d(),
            };
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(cam_data_buf, self.uniform.clone()))
            .unwrap();

        let hi_z_info_buf = gpu
            .sub_alloc(BufferUsage::UNIFORM_BUFFER)
            .allocate_sized()
            .unwrap();
        {
            let mut hi_z_info = hi_z_info_buf.write().unwrap();
            *hi_z_info = crate::renderer::cs::HiZInfo {
                screenResolution: [
                    self.hiz_image.as_ref().unwrap().extent()[0] as i32,
                    self.hiz_image.as_ref().unwrap().extent()[1] as i32,
                ],
                // mip_levels: self.hiz_mip_levels,
                // _pad: [0; 3],
            };
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                hi_z_info_buf,
                self.uniform_hi_z_info.clone(),
            ))
            .unwrap();
    }

    pub fn update_culling_uniform(
        &self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        aspect: f32,
    ) {
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.uniform.clone(),
                self.uniform_culling.clone(),
            ))
            .unwrap();
    }
    pub fn translate(&mut self, v: Vec3) {
        self.pos += self.rot * v;
    }
    pub fn rotate(&mut self, axis: Vec3, angle: f32) {
        self.rot = Quat::from_axis_angle(axis, angle) * self.rot;
    }
    pub fn update(&mut self, input: &Input, dt: f32, grab_cursor: &mut bool) {
        let speed = if input.get_key(KeyCode::ShiftLeft) {
            20.0
        } else {
            4.0
        };
        let move_amount = speed * dt;
        let mut movement = Vec3::ZERO;
        if input.get_key(KeyCode::KeyW) {
            // FORWARD
            movement += Vec3::new(0.0, 0.0, move_amount);
        }
        if input.get_key(KeyCode::KeyS) {
            // BACKWARD
            movement += Vec3::new(0.0, 0.0, -move_amount);
        }
        if input.get_key(KeyCode::KeyD) {
            // RIGHT
            movement += Vec3::new(move_amount, 0.0, 0.0);
        }
        if input.get_key(KeyCode::KeyA) {
            // LEFT
            movement += Vec3::new(-move_amount, 0.0, 0.0);
        }
        if input.get_key(KeyCode::Space) {
            // UP
            movement += Vec3::new(0.0, move_amount, 0.0);
        }
        if input.get_key(KeyCode::ControlLeft) {
            // DOWN
            movement += Vec3::new(0.0, -move_amount, 0.0);
        }
        self.translate(movement.normalize_or_zero() * move_amount);
        let mouse_sensitivity = 0.002;
        let (dx, dy) = input.mouse.delta();
        self.rotate(Vec3::Y, (dx as f32) * mouse_sensitivity);
        self.rotate(self.rot * Vec3::X, (dy as f32) * mouse_sensitivity);

        if input.get_key_pressed(KeyCode::Escape) {
            *grab_cursor = !*grab_cursor;
        }
    }
}
