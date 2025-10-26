use std::sync::Arc;
use std::sync::mpsc::Receiver;

use vulkano::{
    Validated,
    command_buffer::PrimaryAutoCommandBuffer,
    swapchain::{Swapchain, SwapchainAcquireFuture, SwapchainPresentInfo},
    sync::{self, GpuFuture},
};

use crate::{MAX_FRAMES_IN_FLIGHT, gpu_manager::GPUManager};

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
}

pub fn render_thread_main(gpu: Arc<GPUManager>, render_receiver: Receiver<RenderData>) {
    let mut frames_in_flight = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| Some(sync::now(gpu.device.clone()).boxed()))
        .collect::<Vec<_>>();
    let mut frame_index: usize = 0;
    loop {
        // Receive render commands
        match render_receiver.try_recv() {
            Ok(mut rd) => {
                // println!(
                //     "image_num: {} / frames_in_flight: {}",
                //     rd.image_num,
                //     frames_in_flight.len()
                // );
                frames_in_flight[rd.image_num as usize]
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();
                let mut after_cmd_future: Box<dyn GpuFuture> = frames_in_flight
                    [rd.image_num as usize]
                    .take()
                    .unwrap()
                    .join(rd.aquire_future)
                    .boxed();

                for cmd in rd.commands.drain(..) {
                    after_cmd_future = after_cmd_future
                        .then_execute(gpu.queue.clone(), cmd)
                        .unwrap()
                        .then_signal_semaphore()
                        .boxed();
                    // frames_in_flight[command.image_num as usize] = Some(after_cmd_future.boxed());
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
                        frames_in_flight[rd.image_num as usize] = Some(future.boxed());
                    }
                    Err(e) => {
                        eprintln!("Failed to flush future: {:?}", e);
                        frames_in_flight[rd.image_num as usize] =
                            Some(sync::now(gpu.device.clone()).boxed());
                    }
                }
                let prev_frame_index = (frame_index - 1) % MAX_FRAMES_IN_FLIGHT as usize;
                frames_in_flight[prev_frame_index]
					.as_mut()
					.unwrap()
					.cleanup_finished();
                // for i in 0..frames_in_flight.len() {
                //     if i != rd.image_num as usize {
                //         frames_in_flight[i].as_mut().unwrap().cleanup_finished();
                //     }
                // }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // for frame_future in frames_in_flight.iter_mut() {
                //     if let Some(future) = frame_future {
                //         future.cleanup_finished();
                //     }
                // }
                // std::thread::yield_now();
            }
            Err(_) => {
                // Handle the error (e.g., channel closed)
                break;
            }
        }
    }
}
