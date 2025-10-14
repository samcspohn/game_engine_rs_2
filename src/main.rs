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
#![feature(sync_unsafe_cell)]
#![feature(portable_simd)]

use camera::Camera;
use glam::Vec3;
use gpu_manager::GPUManager;
use obj_loader::Obj;
use parking_lot::Mutex;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use render_context::RenderContext;
use std::{
    any::TypeId,
    collections::HashMap,
    error::Error,
    ops::Div,
    sync::{Arc, LazyLock},
    time,
};
use transform::{TransformHierarchy, compute::TransformCompute};
use vulkano::{
    Validated, VulkanError,
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyBufferInfo,
        PrimaryAutoCommandBuffer,
    },
    swapchain::{SwapchainPresentInfo, acquire_next_image},
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

use crate::{
    asset_manager::Asset,
    transform::{_Transform, compute::PerfCounter},
};

mod asset_manager;
mod camera;
mod gpu_manager;
mod input;
mod obj_loader;
mod render_context;
mod renderer;
mod texture;
mod transform;
mod util;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

const MAX_FPS_SAMPLE_AGE: f32 = 1.0;
const NUM_CUBES: usize = 1 << 20; // 1<<22 = 4,194,304
struct FPS {
    frame_times: std::collections::VecDeque<f32>,
    frame_ages: std::collections::VecDeque<time::Instant>,
    time_sum: f32,
}

impl FPS {
    fn new() -> Self {
        Self {
            frame_times: std::collections::VecDeque::new(),
            frame_ages: std::collections::VecDeque::new(),
            time_sum: 0.0,
        }
    }

    fn update(&mut self, frame_time: f32) -> f32 {
        self.frame_times.push_back(frame_time);
        let t = time::Instant::now();
        self.frame_ages.push_back(t);
        self.time_sum += frame_time;
        while t
            .duration_since(*self.frame_ages.front().unwrap())
            .as_secs_f32()
            > MAX_FPS_SAMPLE_AGE
        {
            if let (Some(removed), Some(_age)) =
                (self.frame_times.pop_front(), self.frame_ages.pop_front())
            {
                self.time_sum -= removed;
            }
        }
        self.frame_times.len() as f32 / self.time_sum
    }
}

struct App {
    gpu: Arc<GPUManager>,
    asset_manager: asset_manager::AssetManager,
    cube: Option<asset_manager::AssetHandle<obj_loader::Obj>>,
    // offsets: Vec<[f32; 3]>,
    rcxs: HashMap<WindowId, RenderContext>,
    cursor_grabbed: bool,
    fps: FPS,
    time: std::time::Instant,
    camera: HashMap<WindowId, Arc<Mutex<camera::Camera>>>,
    transform_hierarchy: TransformHierarchy,
    transform_compute: TransformCompute,
    rendering_system: renderer::RenderingSystem,
    renderers: Vec<renderer::RendererComponent>,
    // update_transforms: bool,
    transforms_updated: bool,
    // builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    frame_time: PerfCounter,
    update_sim: PerfCounter,
    update_render: PerfCounter,
    paused: bool,
    update_transforms_compute_shader: bool,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let gpu = gpu_manager::GPUManager::new(event_loop);

        // let vertices = [
        //     MyVertex {
        //         position: [-0.5 / 10.0, -0.25 / 10.0],
        //     },
        //     MyVertex {
        //         position: [0.0, 0.5 / 10.0],
        //     },
        //     MyVertex {
        //         position: [0.25 / 10.0, -0.1 / 10.0],
        //     },
        // ];
        // let vertex_buffer = Buffer::from_iter(
        //     gpu.lock().mem_alloc.clone(),
        //     BufferCreateInfo {
        //         usage: BufferUsage::VERTEX_BUFFER,
        //         ..Default::default()
        //     },
        //     AllocationCreateInfo {
        //         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
        //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        //         ..Default::default()
        //     },
        //     vertices,
        // )
        // .unwrap();
        // let cube = obj_loader::Obj::load_from_file("assets/cube/cube.obj", &gpu.lock()).unwrap();
        let mut asset_manager = asset_manager::AssetManager::new(gpu.clone());
        let default_obj = asset_manager.set_placeholder_asset(Obj::default(&gpu));
        asset_manager.set_placeholder_asset(texture::Texture::default(&gpu));
        let cube = asset_manager.load_asset::<Obj>("assets/cube/cube.obj");

        let mut previous_frame_end = Some(sync::now(gpu.device.clone()).boxed());

        let mut transform_compute = TransformCompute::new(&gpu);
        let mut transform_hierarchy = TransformHierarchy::new();
        let mut rendering_system = renderer::RenderingSystem::new(gpu.clone());

        while Arc::ptr_eq(
            &(cube.get(&asset_manager) as Arc<dyn Asset>),
            asset_manager
                .placeholder_assets
                .lock()
                .get(&TypeId::of::<Obj>())
                .unwrap(),
        ) {
            // wait until the cube is loaded
            asset_manager.process_deferred_queue();
            gpu.process_work_queue();
            previous_frame_end.as_mut().unwrap().cleanup_finished();
        }
        rendering_system.register_model(cube.clone(), &asset_manager);
        let mut renderers = Vec::new();

        let dims = (NUM_CUBES as f64).powf(1.0 / 3.0).ceil() as u32;
        for i in 0..NUM_CUBES {
            let x = (i as u32 % dims) as f32;
            let y = ((i as u32 / dims) % dims) as f32;
            let z = (i as u32 / (dims * dims)) as f32;
            let spacing = 10.0;
            let t = _Transform {
                position: Vec3::new(
                    (x - dims as f32 / 2.0) * spacing,
                    (y - dims as f32 / 2.0) * spacing,
                    (z - dims as f32 / 2.0) * spacing,
                ),
                rotation: glam::Quat::from_axis_angle(
                    Vec3::new(
                        rand::random::<f32>(),
                        rand::random::<f32>(),
                        rand::random::<f32>(),
                    )
                    .normalize(),
                    rand::random::<f32>() * std::f32::consts::TAU,
                ),
                scale: Vec3::splat(0.5),
                name: format!("cube_{i}"),
                parent: None,
            };
            let transform = transform_hierarchy.create_transform(t);
            let r = rendering_system.renderer(cube.clone(), &transform, &mut asset_manager);
            renderers.push(r);
        }

        let mut builder = gpu.create_command_buffer(CommandBufferUsage::OneTimeSubmit);

        let (updated, staging_buffer_index) =
            transform_compute.update_transforms(&gpu, &transform_hierarchy, &mut builder, true);
        let command_buffer = builder.build().unwrap();

        // execute buffer copies and global compute shaders
        previous_frame_end = Some(
            previous_frame_end
                .take()
                .unwrap()
                .then_execute(gpu.queue.clone(), command_buffer)
                .unwrap()
                .then_signal_semaphore()
                .then_execute(
                    gpu.queue.clone(),
                    transform_compute.command_buffer[staging_buffer_index].clone(),
                )
                .unwrap()
                .then_signal_semaphore()
                .boxed(),
        );

        App {
            asset_manager,
            transform_compute,
            rendering_system,
            renderers,
            previous_frame_end,
            gpu,
            cube: Some(cube),
            rcxs: HashMap::new(),
            fps: FPS::new(),
            time: std::time::Instant::now(),
            camera: HashMap::new(),
            cursor_grabbed: false,
            transform_hierarchy,
            // update_transforms: true,
            transforms_updated: false,
            // builder: None,
            frame_time: PerfCounter::new(),
            update_sim: PerfCounter::new(),
            update_render: PerfCounter::new(),
            paused: false,
            update_transforms_compute_shader: true,
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/model.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/model.frag",
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let camera = Arc::new(Mutex::new(Camera::new(
            &self.gpu,
            [800, 600],
            0.01,
            10_000.0,
        )));
        let rcx = RenderContext::new(event_loop, &self.gpu, camera.clone());
        let window_id = rcx.window.id();
        self.camera.insert(window_id, camera);
        self.rcxs.insert(window_id, rcx);
    }
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                for rcx in self.rcxs.values_mut() {
                    if rcx.focused {
                        rcx.input.mouse.on_motion(delta.0, delta.1);
                    }
                }
                // self.input.mouse.on_motion(delta.0, delta.1);
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
        if !self.rcxs.contains_key(&_window_id) {
            return;
        }
        let rcx = self.rcxs.get_mut(&_window_id).unwrap();

        rcx.gui.update(&event);
        match event {
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                rcx.input.mouse.cursor_pos(position.x, position.y);
            }
            WindowEvent::Focused(focused) => {
                rcx.focused = focused;
                if !focused && self.cursor_grabbed {
                    rcx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::None)
                        .unwrap();
                    // self.cursor_grabbed = false;
                    rcx.window.set_cursor_visible(true);
                } else if focused && self.cursor_grabbed {
                    rcx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        .unwrap();
                    // self.cursor_grabbed = true;
                    rcx.window.set_cursor_visible(false);
                }
            }
            WindowEvent::KeyboardInput {
                event,
                device_id,
                is_synthetic,
            } => match event.physical_key {
                PhysicalKey::Code(keycode) => match keycode {
                    _ => {
                        rcx.input
                            .on_key(keycode, event.state == ElementState::Pressed);
                    }
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
                if rcx.gui_drawn {
                    rcx.gui_drawn = false;
                } else {
                    return;
                }
                rcx.frame_time.start();
                let camera = self.camera.get_mut(&_window_id).unwrap();

                // let cube = self.cube.as_ref().unwrap().get(&self.asset_manager);

                let window_size = rcx.window.inner_size();
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // rcx.cleanup.start();
                // self.previous_frame_end.as_mut().unwrap().cleanup_finished();
                // rcx.cleanup.stop();

                rcx.swap_chain_perf.start();
                if rcx.recreate_swapchain {
                    print!("Recreating swapchain... ");
                    rcx.recreate_swapchain();
                }
                rcx.acquire_next_image_perf.start();
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
                rcx.acquire_next_image_perf.stop();
                rcx.swap_chain_perf.stop();

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                if rcx.command_buffer.is_none() {
                    let gpu = &self.gpu;
                    rcx.rebuild_command_buffer(
                        gpu,
                        &self.asset_manager,
                        // &self.offsets,
                        self.cube.as_ref().unwrap(),
                        &mut self.rendering_system,
                        self.transform_compute.model_matrix_buffer.clone(),
                    );
                }
                rcx.build_command_buffer_perf.start();
                let gpu = &self.gpu;
                let cam = camera.lock();
                let a = self
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(
                        gpu.queue.clone(),
                        rcx.command_buffer.as_ref().unwrap().clone(),
                    )
                    .unwrap();

                // blit to swapchain image
                let mut builder = self
                    .gpu
                    .create_command_buffer(CommandBufferUsage::OneTimeSubmit);
                builder
                    .blit_image(BlitImageInfo::images(
                        cam.image.clone(),
                        rcx.attachment_images[image_index as usize].clone(),
                    ))
                    .unwrap();

                let a = a
                    .then_signal_semaphore()
                    .then_execute(gpu.queue.clone(), builder.build().unwrap())
                    .unwrap();

                let a = rcx
                    .gui
                    .draw_on_image(a, rcx.attachment_image_views[image_index as usize].clone());
                rcx.build_command_buffer_perf.stop();

                rcx.execute_command_buffer_perf.start();
                let future = a
                    .then_swapchain_present(
                        gpu.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        self.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        self.previous_frame_end = Some(sync::now(gpu.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        self.previous_frame_end = Some(sync::now(gpu.device.clone()).boxed());
                    }
                }
                rcx.execute_command_buffer_perf.stop();
                rcx.frame_time.stop();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.frame_time.start();
        let t = time::Instant::now();
        let elapsed = t.duration_since(self.time).as_secs_f32();
        let fps = self.fps.update(elapsed);
        self.time = t;
        self.transforms_updated = false;

        if self.cube.is_none() {
            self.cube = Some(self.asset_manager.load_asset::<Obj>("assets/cube/cube.obj"));
        }
        self.update_sim.start();
        if !self.paused {
            let time_since_epoch = std::time::SystemTime::now()
                .duration_since(time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();
            let angle = time_since_epoch % std::f64::consts::TAU;
            let angle = angle as f32;
            let translation = Vec3::new(angle.cos(), 0.0, angle.sin()) * elapsed * 5.0;
            // let rhs = (NUM_CUBES as f32).sqrt().sqrt().sqrt().ceil() as usize;
            // let chunk_size: usize = NUM_CUBES.div_ceil(rhs);
            let nt = rayon::current_num_threads();
            // let chunk_size: usize = util::get_chunk_size(NUM_CUBES);
            let chunk_size: f32 = NUM_CUBES as f32 / nt as f32;
            (0..nt).into_par_iter().for_each(|i| {
                let start = (i as f32 * chunk_size) as usize;
                let end = ((i + 1) as f32 * chunk_size).min(NUM_CUBES as f32) as usize;
                for j in start..end {
                    if let Some(t) = self.transform_hierarchy.get_transform(j as u32) {
                        let t = t.lock();
                        t.translate_by(translation);
                    }
                }
            });
        }
        self.update_sim.stop();
        // // test performance of non-reuse of command buffers
        // self.transform_compute.command_buffer = Vec::new();
        // self.rendering_system.command_buffer = None;
        // for (_window_id, rcx) in self.rcxs.iter_mut() {
        //     rcx.command_buffer = None;
        // }

        // process asset loading queue
        self.asset_manager.process_deferred_queue();
        self.gpu.process_work_queue();

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
        // build command buffer to update buffers
        let mut builder = self
            .gpu
            .create_command_buffer(CommandBufferUsage::OneTimeSubmit);

        let (updated, staging_buffer_index) = self.transform_compute.update_transforms(
            &self.gpu,
            &self.transform_hierarchy,
            &mut builder,
            self.update_transforms_compute_shader,
        );
        for (_window_id, rcx) in self.rcxs.iter_mut() {
            let cam = self.camera.get_mut(&_window_id).unwrap();
            cam.lock().update_uniform(
                &self.gpu,
                &mut builder,
                rcx.viewport.extent[0] / rcx.viewport.extent[1].abs(),
            );
        }
        self.update_render.start();
        let renderer_updated = self.rendering_system.compute_renderers(
            &mut builder,
            &self.transform_compute.model_matrix_buffer,
            self.asset_manager
                .rebuild_command_buffer
                .load(std::sync::atomic::Ordering::SeqCst),
        );
        self.update_render.stop();

        let command_buffer = builder.build().unwrap();

        // execute buffer copies and global compute shaders
        self.previous_frame_end = Some(
            self.previous_frame_end
                .take()
                .unwrap()
                .then_execute(self.gpu.queue.clone(), command_buffer)
                .unwrap()
                .then_signal_semaphore()
                .then_execute(
                    self.gpu.queue.clone(),
                    self.transform_compute.command_buffer[staging_buffer_index].clone(),
                )
                .unwrap()
                .then_signal_semaphore()
                .then_execute(
                    self.gpu.queue.clone(),
                    self.rendering_system
                        .command_buffer
                        .as_ref()
                        .unwrap()
                        .clone(),
                )
                .unwrap()
                .then_signal_semaphore()
                .boxed(),
        );

        if self
            .asset_manager
            .rebuild_command_buffer
            .load(std::sync::atomic::Ordering::SeqCst)
            || updated
            || renderer_updated
        {
            for (_window_id, rcx) in self.rcxs.iter_mut() {
                rcx.command_buffer = None;
            }
        }
        self.asset_manager
            .rebuild_command_buffer
            .store(false, std::sync::atomic::Ordering::SeqCst);

        let mut new_window = false;
        for (_window_id, rcx) in self.rcxs.iter_mut() {
            if rcx.input.get_key_pressed(KeyCode::KeyP) {
                new_window = true;
            }
            let camera = self.camera.get_mut(&_window_id).unwrap();
            let mut grabbed = self.cursor_grabbed;
            camera.lock().update(&rcx.input, elapsed, &mut grabbed);
            if grabbed != self.cursor_grabbed {
                if self.cursor_grabbed {
                    rcx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        .unwrap();
                    rcx.window.set_cursor_visible(false);
                } else {
                    rcx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::None)
                        .unwrap();
                    rcx.window.set_cursor_visible(true);
                }
            }
            if rcx.input.get_key_pressed(KeyCode::KeyK) {
                self.paused = !self.paused;
            }
            if rcx.input.get_key_pressed(KeyCode::KeyJ) {
                self.update_transforms_compute_shader = !self.update_transforms_compute_shader;
                println!(
                    "Update transforms compute shader: {}",
                    self.update_transforms_compute_shader
                );
            }
            self.cursor_grabbed = grabbed;
            rcx.gui.immediate_ui(|gui| {
                let ctx = gui.context();
                egui::Window::new("fps").show(&ctx, |ui| {
                    ui.label(format!("{:.2} fps", fps));
                });
            });
            rcx.gui_drawn = true;
            rcx.window.request_redraw();
            rcx.input.update();
        }
        if new_window {
            let camera = Arc::new(Mutex::new(Camera::new(
                &self.gpu,
                [800, 600],
                0.01,
                10_000.0,
            )));
            let rcx = RenderContext::new(_event_loop, &self.gpu, camera.clone());
            let window_id = rcx.window.id();
            self.camera.insert(window_id, camera);
            self.rcxs.insert(window_id, rcx);
        }

        static LAST_PRINT: LazyLock<Mutex<std::time::Instant>> =
            LazyLock::new(|| Mutex::new(std::time::Instant::now()));
        {
            let mut last_print = LAST_PRINT.lock();
            if last_print.elapsed().as_secs_f32() > 2.0 {
                println!(
                    "frame time: {:?} / update sim: {:?} / update render: {:?}",
                    self.frame_time, self.update_sim, self.update_render
                );
                println!(
                    "allocate buffers: {:?} / update buffers: {:?} / update parents: {:?} / compute: {:?}",
                    self.transform_compute.perf_counters.allocate_bufs,
                    self.transform_compute.perf_counters.update_bufs,
                    self.transform_compute.perf_counters.update_parents,
                    self.transform_compute.perf_counters.compute,
                );
                for (_window_id, rcx) in self.rcxs.iter() {
                    println!(
                        "window {:?} frame time: {:?} / cleanup: {:?} / swap chain: {:?} / acquire next image: {:?} / update camera: {:?} / build command buffer: {:?} / execute command buffer: {:?}",
                        rcx.window.id(),
                        rcx.frame_time,
                        rcx.cleanup,
                        rcx.swap_chain_perf,
                        rcx.acquire_next_image_perf,
                        rcx.update_camera_perf,
                        rcx.build_command_buffer_perf,
                        rcx.execute_command_buffer_perf
                    );
                    println!("Camera position: {:?}", rcx.camera.lock().pos);
                }
                *last_print = std::time::Instant::now();
            }
        }
        self.frame_time.stop();
    }
}
