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
#[allow(static_mut_refs)]
use camera::Camera;
use glam::Vec3;
use gltf::json::extensions::asset;
use gpu_manager::GPUManager;
use obj_loader::Model;
use parking_lot::Mutex;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use render_context::RenderContext;
use std::{
    any::TypeId,
    cell::SyncUnsafeCell,
    collections::HashMap,
    env,
    error::Error,
    ops::Div,
    sync::{Arc, LazyLock},
    thread::Thread,
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
    asset_manager::{Asset, AssetHandle},
    component::{Component, ComponentStorage, Scene},
    engine::Engine,
    renderer::{_RendererComponent, RendererComponent, RenderingSystem},
    transform::{_Transform, Transform, compute::PerfCounter},
    util::container::Container,
};

use rayon::prelude::*;

mod asset_manager;
mod camera;
mod component;
mod engine;
mod gltf_loader;
mod gpu_manager;
mod input;
mod obj_loader;
mod render_context;
mod renderer;
mod texture;
mod transform;
mod util;
fn main() -> Result<(), impl Error> {
    unsafe {
        env::set_var("RUST_BACKTRACE", "1");
    }
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

const MAX_FPS_SAMPLE_AGE: f32 = 1.0;
// 1 << 20 = 1,048,576
// 1 << 21 = 2,097,152
// 1 << 22 = 4,194,304
// 1 << 23 = 8,388,608
const NUM_CUBES: usize = 1 << 20; // << 20;
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

static mut TRANSLATION: Vec3 = Vec3::ZERO;

#[derive(Default, Clone)]
struct ComponentTest {}
impl ComponentTest {
    fn new() -> Self {
        Self {}
    }
}
impl Component for ComponentTest {
    fn update(&mut self, dt: f32, t: &Transform) {
        let t = t.lock();
        t.translate_by(unsafe { TRANSLATION });
    }
}

struct App {
    gpu: Arc<GPUManager>,
    asset_manager: asset_manager::AssetManager,
    // cube: Option<asset_manager::AssetHandle<obj_loader::Model>>,
    // offsets: Vec<[f32; 3]>,
    rcxs: HashMap<WindowId, RenderContext>,
    cursor_grabbed: bool,
    fps: FPS,
    time: std::time::Instant,
    camera: HashMap<WindowId, Arc<Mutex<camera::Camera>>>,
    // transform_hierarchy: Arc<Mutex<TransformHierarchy>>,
    transform_compute: TransformCompute,
    // renderers: Vec<renderer::RendererComponent>,
    // engine: engine::Engine,
    world: Arc<Mutex<Scene>>,
    engine: Arc<Engine>,
    rendering_system: Arc<Mutex<RenderingSystem>>,
    // component_test: Arc<Mutex<Container<(u32, ComponentTest)>>>,
    // component_test2: Arc<Mutex<Vec<SyncUnsafeCell<ComponentTest>>>>,
    // update_transforms: bool,
    transforms_updated: bool,
    // builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    frame_time: PerfCounter,
    update_sim: PerfCounter,
    update_render: PerfCounter,
    paused: bool,
    update_transforms_compute_shader: bool,
    // sim_thread: std::thread::JoinHandle<()>,
    sim_frame_start: std::sync::mpsc::SyncSender<f32>,
    sim_frame_end: std::sync::mpsc::Receiver<()>,
    bismarck_handle: Option<asset_manager::AssetHandle<Scene>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        unsafe {
            env::set_var("RUST_BACKTRACE", "1");
        }
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
        let engine = Arc::new(Engine::new(gpu.clone()));

        let rendering_system = engine.rendering_system.clone();
        let mut asset_manager =
            asset_manager::AssetManager::new(gpu.clone(), rendering_system.clone());
        let default_obj = asset_manager.set_placeholder_asset(Model::default(&gpu));
        asset_manager.set_placeholder_asset(texture::Texture::default(&gpu));

        let transform_compute = TransformCompute::new(&gpu);
        // let mut transform_hierarchy = TransformHierarchy::new();
        // let rendering_system = Arc::new(Mutex::new(renderer::RenderingSystem::new(gpu.clone())));
        // rendering_system
        //     .lock()
        //     .register_model_handle(AssetHandle::<Obj>::_from_id(0));
        // rendering_system
        //     .lock()
        //     .register_model(AssetHandle::<Obj>::_from_id(0), default_obj.clone());

        // let _r = rendering_system.clone();

        // rendering_system.lock().register_model_handle(cube.clone());
        // let mut renderers = Vec::new();

        // let component_registry = Arc::new(Mutex::new(component::ComponentRegistry::new()));
        // component_registry.lock().register::<ComponentTest>();
        // component_registry.lock().register::<RendererComponent>();

        unsafe {
            gltf_loader::ENGINE = Some(engine.clone());
        }
        let mut world = Scene::new(engine.clone());
        let world = Arc::new(Mutex::new(world));

        let _world = world.clone();
        let empty_scene = Scene::new(engine.clone());
        let scene_placeholder = asset_manager.set_placeholder_asset(empty_scene);

        // while Arc::ptr_eq(
        //     &a.get(&asset_manager),
        //     &scene_placeholder,
        // ) {
        //     asset_manager.process_deferred_queue();
        //     gpu.process_work_queue();
        //     // std::thread::sleep(std::time::Duration::from_millis(10));
        // }
        let mut bismarck_handle = None;
        {
            let mut world = world.lock();
            world.components.register::<ComponentTest>(true);
            world.components.register::<RendererComponent>(false);
            world.components.register::<_RendererComponent>(false);

            world
                .engine
                .rendering_system
                .lock()
                .get_or_register_model_handle(AssetHandle::<Model>::_from_id(0));
            world
                .engine
                .rendering_system
                .lock()
                .register_model(AssetHandle::<Model>::_from_id(0), default_obj.clone());
            let _r = world.engine.rendering_system.clone();
            let cube = asset_manager.load_asset::<Model>("assets/cube/cube.obj", None);
            world
                .engine
                .rendering_system
                .lock()
                .get_or_register_model_handle(cube.clone());

            let a = asset_manager.load_asset::<Scene>(
                "assets/bismark_low_poly2.glb",
                Some(Box::new(move |handle, arc_asset| {
                    _world.lock().instantiate(&arc_asset);
                })),
            );
            bismarck_handle = Some(a);
            // let mut _component_registry = component_registry.lock();

            let dims = (NUM_CUBES as f64).powf(1.0 / 3.0).ceil() as u32;
            // let mut _rendering_system = rendering_system.lock();
            // let num_cubes_per_thread =
            //     (NUM_CUBES as f32 / rayon::current_num_threads() as f32).ceil() as usize;
            // let mut current_thread_idx = 0;
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
                let e = world.new_entity(t);
                world.add_component(e, ComponentTest::new());
                world.add_component(e, RendererComponent::new(cube.clone()));
                // let transform = transform_hierarchy.create_transform(t);
                // _component_registry.get_storage_mut::<ComponentTest>().map(|storage| {
                //     storage.set(
                //         ComponentTest::new(),
                //         transform.get_idx(),
                //     );
                // });

                // let r = _rendering_system.renderer(cube.clone(), &transform, &mut asset_manager);
                // renderers.push(r);
            }
            // drop(_rendering_system);
            // drop(_component_registry);
        }

        // let transform_hierarchy = Arc::new(Mutex::new(transform_hierarchy));
        let (sim_frame_start, sim_frame_start_rcv) = std::sync::mpsc::sync_channel::<f32>(1);
        let (sim_frame_end_snd, sim_frame_end_rcv) = std::sync::mpsc::sync_channel::<()>(1);
        // let transform_hierarchy_clone = transform_hierarchy.clone();
        // let component_registry_clone = component_registry.clone();

        let _world = world.clone();
        // let sim_thread = std::thread::spawn(move || {
        rayon::spawn(move ||{
            sim_frame_end_snd.send(()).unwrap();
            // let transform_hierarchy = transform_hierarchy_clone;
            // let component_registry = component_registry_clone;
            loop {
                let dt = sim_frame_start_rcv.recv().unwrap();
                let time_since_epoch = std::time::SystemTime::now()
                    .duration_since(time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                let angle = time_since_epoch % std::f64::consts::TAU;
                let angle = angle as f32;
                unsafe {
                    TRANSLATION = Vec3::new(angle.cos(), 0.0, angle.sin()) * dt * 5.0;
                }
                _world.lock().update(dt);
                // let _transform_hierarchy = transform_hierarchy.lock();
                // component_registry
                //     .lock()
                //     .update_all(dt, &_transform_hierarchy);

                sim_frame_end_snd.send(()).unwrap();
            }
            // });
        });

        let rendering_system = world.lock().engine.rendering_system.clone();

        App {
            asset_manager,
            transform_compute,
            rendering_system,
            world,
            engine,
            // rendering_system,
            // engine: engine::Engine {
            //     rendering_system: rendering_system.clone(),
            // },
            // renderers,
            previous_frame_end: Some(sync::now(gpu.device.clone()).boxed()),
            gpu,
            // cube: Some(cube),
            rcxs: HashMap::new(),
            fps: FPS::new(),
            time: std::time::Instant::now(),
            camera: HashMap::new(),
            cursor_grabbed: false,
            // transform_hierarchy,
            transforms_updated: false,
            frame_time: PerfCounter::new(),
            update_sim: PerfCounter::new(),
            update_render: PerfCounter::new(),
            paused: false,
            update_transforms_compute_shader: true,
            // sim_thread,
            sim_frame_start,
            sim_frame_end: sim_frame_end_rcv,
            bismarck_handle,
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
                        &mut self.rendering_system.lock(),
                        self.transform_compute.model_matrix_buffer.clone(),
                    );
                }
                rcx.build_command_buffer_perf.start();
                let gpu = &self.gpu;
                rcx.extra_perfs
                    .entry("lock camera".into())
                    .or_insert(PerfCounter::new())
                    .start();
                let cam = camera.lock();
                rcx.extra_perfs.get_mut("lock camera").unwrap().stop();
                rcx.extra_perfs
                    .entry("wait previous frame".into())
                    .or_insert(PerfCounter::new())
                    .start();
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
                rcx.extra_perfs
                    .get_mut("wait previous frame")
                    .unwrap()
                    .stop();
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
                rcx.draw_gui_perf.start();
                let a = rcx
                    .gui
                    .draw_on_image(a, rcx.attachment_image_views[image_index as usize].clone());
                rcx.draw_gui_perf.stop();
                rcx.build_command_buffer_perf.stop();

                rcx.execute_command_buffer_perf.start();
                rcx.extra_perfs
                    .entry("signal fence and flush".into())
                    .or_insert(PerfCounter::new())
                    .start();
                let future = a
                    .then_swapchain_present(
                        gpu.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();
                rcx.extra_perfs
                    .get_mut("signal fence and flush")
                    .unwrap()
                    .stop();

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

        // if self.cube.is_none() {
        //     self.cube = Some(self.asset_manager.load_asset::<Obj>("assets/cube/cube.obj"));
        // }
        self.update_sim.start();
        if !self.paused {
            self.sim_frame_end.recv().unwrap();
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

        // let transform_hierarchy = &mut self.world.lock().transform_hierarchy;
        let (updated, staging_buffer_index) = self.transform_compute.update_transforms(
            &self.gpu,
            &mut self.world.lock().transform_hierarchy,
            &mut builder,
            self.update_transforms_compute_shader,
            self.previous_frame_end.as_mut().unwrap(),
        );
        if !self.paused {
            match self.sim_frame_start.send(elapsed) {
                Ok(_) => {}
                Err(e) => {
                    println!("Error sending to sim thread: {}", e);
                }
            }
        }
        for (_window_id, rcx) in self.rcxs.iter_mut() {
            let cam = self.camera.get_mut(&_window_id).unwrap();
            cam.lock().update_uniform(
                &self.gpu,
                &mut builder,
                rcx.viewport.extent[0] / rcx.viewport.extent[1].abs(),
            );
        }
        self.update_render.start();

        let mut rendering_system = self.rendering_system.lock();
        let renderer_updated = rendering_system
            .compute_renderers(&mut builder, &self.transform_compute.model_matrix_buffer);
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
                    rendering_system.command_buffer.as_ref().unwrap().clone(),
                )
                .unwrap()
                .then_signal_semaphore()
                .boxed(),
        );
        drop(rendering_system);

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
        let mut new_bismarck = false;
        for (_window_id, rcx) in self.rcxs.iter_mut() {
            if rcx.input.get_key_pressed(KeyCode::KeyP) {
                new_window = true;
            }
            if rcx.input.get_key_pressed(KeyCode::KeyB) {
                new_bismarck = true;
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
        if new_bismarck {
            let mut world = self.world.lock();
            let a = world.instantiate(
                &self
                    .bismarck_handle
                    .as_ref()
                    .unwrap()
                    .get(&self.asset_manager),
            );
            // println!("Instantiated bismarck: {:?}", a);
            {
                let bismarck = a.lock();
                bismarck.translate_by(Vec3::new(
                    rand::random::<f32>() * 1000.0 - 500.0,
                    0.0,
                    rand::random::<f32>() * 1000.0 - 500.0,
                ));
                bismarck.rotate_by(glam::Quat::from_axis_angle(
                    Vec3::new(
                        rand::random::<f32>(),
                        rand::random::<f32>(),
                        rand::random::<f32>(),
                    )
                    .normalize(),
                    rand::random::<f32>() * std::f32::consts::TAU,
                ));
            }
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
                        "window {:?} frame time: {:?} / cleanup: {:?} / swap chain: {:?} / acquire next image: {:?} / update camera: {:?} / draw gui: {:?} / build command buffer: {:?} / execute command buffer: {:?}",
                        rcx.window.id(),
                        rcx.frame_time,
                        rcx.cleanup,
                        rcx.swap_chain_perf,
                        rcx.acquire_next_image_perf,
                        rcx.update_camera_perf,
                        rcx.draw_gui_perf,
                        rcx.build_command_buffer_perf,
                        rcx.execute_command_buffer_perf
                    );
                    for (name, perf) in rcx.extra_perfs.iter() {
						print!("  {}: {:?} ", name, perf);
					}
					println!("");
                    println!("Camera position: {:?}", rcx.camera.lock().pos);
                }
                *last_print = std::time::Instant::now();
            }
        }
        self.frame_time.stop();
    }
}
