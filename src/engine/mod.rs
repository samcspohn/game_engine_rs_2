use std::sync::Arc;

use parking_lot::Mutex;

use crate::{gpu_manager::GPUManager, renderer};

pub struct Engine {
    pub rendering_system: Arc<Mutex<renderer::RenderingSystem>>,
}

impl Engine {
    pub fn new(gpu: Arc<GPUManager>) -> Self {
        Self {
            rendering_system: Arc::new(Mutex::new(renderer::RenderingSystem::new(gpu))),
        }
    }
}
