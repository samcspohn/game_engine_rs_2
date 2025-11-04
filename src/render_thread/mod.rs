use std::sync::{Arc, atomic::AtomicBool};
use std::sync::mpsc::Receiver;

use egui_winit_vulkano::Gui;
use parking_lot::Mutex;
use vulkano::{
    Validated,
    command_buffer::PrimaryAutoCommandBuffer,
    image::view::ImageView,
    swapchain::{Swapchain, SwapchainAcquireFuture, SwapchainPresentInfo},
    sync::{self, GpuFuture},
};

use crate::{MAX_FRAMES_IN_FLIGHT, gpu_manager::GPUManager, render_context::RenderContext};

struct RenderCommand {
    // Define the fields for your render command here
    // For example:
    // pub command_type: RenderCommandType,
    // pub data: RenderData,
}

pub struct RenderData {
    pub image_num: u32,
    pub aquire_future: SwapchainAcquireFuture,
    pub commands: Vec<Arc<PrimaryAutoCommandBuffer>>,
    pub swapchain: Arc<Swapchain>,
    pub gui: Arc<Mutex<Gui>>,
    pub image_view: Arc<ImageView>,
    pub fps: f32,
}

pub struct Purge {
	pub done: Arc<AtomicBool>,
}

impl Purge {
	pub fn new() -> Self {
		Self {
			done: Arc::new(AtomicBool::new(false)),
		}
	}
}

pub enum RenderThreadMessage {
    Render(RenderData),
    Purge(Purge),
}

pub fn render_thread_main(gpu: Arc<GPUManager>, render_receiver: Receiver<RenderThreadMessage>) {
    // let mut frames_in_flight = (0..MAX_FRAMES_IN_FLIGHT)
    //     .map(|_| Some(sync::now(gpu.device.clone()).boxed()))
    //     .collect::<Vec<_>>();
    let mut previous_frame_future = sync::now(gpu.device.clone()).boxed();
    // let mut frame_index: usize = 0;
    loop {
        // Receive render commands
        match render_receiver.recv() {
            Ok(data) => {
                match data {
                    RenderThreadMessage::Purge(p) => {
                        // Handle purge logic here
                        previous_frame_future.cleanup_finished();
                        previous_frame_future = sync::now(gpu.device.clone()).boxed();
                        p.done.store(true, std::sync::atomic::Ordering::SeqCst);
                        continue;
                    }
                    RenderThreadMessage::Render(mut rd) => {

                        previous_frame_future.cleanup_finished();
                        let mut after_cmd_future: Box<dyn GpuFuture> =
                            previous_frame_future.join(rd.aquire_future).boxed();

                        for cmd in rd.commands.drain(..) {
                            after_cmd_future = after_cmd_future
                                .then_execute(gpu.queue.clone(), cmd)
                                .unwrap()
                                .then_signal_semaphore()
                                .boxed();
                            after_cmd_future.cleanup_finished();
                        }
                        let future = after_cmd_future
                            .then_swapchain_present(
                                gpu.queue.clone(),
                                SwapchainPresentInfo::swapchain_image_index(
                                    rd.swapchain.clone(),
                                    rd.image_num,
                                ),
                            )
                            .then_signal_fence_and_flush();

                        match future.map_err(Validated::unwrap) {
                            Ok(future) => {
                                previous_frame_future = future.boxed();
                            }
                            Err(e) => {
                                eprintln!("Failed to flush future: {:?}", e);
                                // frames_in_flight[frame_index] = Some(sync::now(gpu.device.clone()).boxed());
                                previous_frame_future = sync::now(gpu.device.clone()).boxed();
                            }
                        }
                    }
                }
            }
            Err(_) => {
                previous_frame_future.cleanup_finished();
                break;
            }
        }
    }
}
