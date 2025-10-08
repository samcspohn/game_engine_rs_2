use core::net;
use std::sync::Arc;

use crate::{RenderContext, gpu_manager::GPUManager, input::Input};
use glam::{Quat, Vec3};
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    format::Format,
    image::{Image, ImageCreateInfo, ImageTiling, ImageType, ImageUsage, view::ImageView},
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
    pub depth_view: Arc<ImageView>,
    pub uniform: Subbuffer<crate::vs::camera>,
    pub format: Format,
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
                    | ImageUsage::TRANSFER_DST,
                tiling: ImageTiling::Optimal,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        let depth_view = ImageView::new_default(depth_image).unwrap();

        let uniform = gpu.buffer_from_data(
            &crate::vs::camera {
                view: glam::Mat4::IDENTITY.to_cols_array_2d(),
                proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            },
            BufferUsage::UNIFORM_BUFFER,
        );
        Self {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
            image,
            image_view,
            depth_view,
            uniform,
            format,
            near,
            far,
        }
    }
    pub fn resize(&mut self, gpu: &GPUManager, dimensions: [u32; 2]) {
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
                    | ImageUsage::TRANSFER_DST,
                // tiling: ImageTiling::Optimal,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        self.depth_view = ImageView::new_default(depth_image).unwrap();
    }
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        let translate = glam::Mat4::from_translation(-self.pos);
        let rotate = glam::Mat4::from_quat(self.rot.conjugate());
        rotate * translate
    }
    pub fn get_proj_matrix(&self, aspect: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, aspect, self.near, self.far)
    }
    pub fn update_uniform(
        &self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        aspect: f32,
    ) {
        let cam_data_buf = gpu.sub_alloc.allocate_sized().unwrap();
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
    }
    pub fn translate(&mut self, v: Vec3) {
        self.pos += self.rot * v;
    }
    pub fn rotate(&mut self, axis: Vec3, angle: f32) {
        self.rot = Quat::from_axis_angle(axis, angle) * self.rot;
    }
    pub fn update(&mut self, input: &Input, dt: f32, grab_cursor: &mut bool) {
        let speed = if input.get_key(KeyCode::ShiftLeft) {
            5.0
        } else {
            1.0
        };
        let move_amount = speed * dt;
        let mut movement = Vec3::ZERO;
        if input.get_key(KeyCode::KeyW) {
            // FORWARD
            movement += Vec3::new(0.0, 0.0, -move_amount);
        }
        if input.get_key(KeyCode::KeyS) {
            // BACKWARD
            movement += Vec3::new(0.0, 0.0, move_amount);
        }
        if input.get_key(KeyCode::KeyA) {
            // LEFT
            movement += Vec3::new(-move_amount, 0.0, 0.0);
        }
        if input.get_key(KeyCode::KeyD) {
            // RIGHT
            movement += Vec3::new(move_amount, 0.0, 0.0);
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
        self.rotate(Vec3::Y, -(dx as f32) * mouse_sensitivity);
        self.rotate(self.rot * Vec3::X, -(dy as f32) * mouse_sensitivity);

        if input.get_key_pressed(KeyCode::Escape) {
            *grab_cursor = !*grab_cursor;
        }
    }
}
