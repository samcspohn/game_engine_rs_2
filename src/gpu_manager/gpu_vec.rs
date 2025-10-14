use std::ops::Sub;

use vulkano::{
    buffer::{BufferContents, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    memory::allocator::MemoryTypeFilter,
};

use crate::gpu_manager::GPUManager;

pub struct GPUVec<T> {
    data: Vec<T>,
    buffer: Subbuffer<[T]>,
    size: usize,
    usage: BufferUsage,
    persist: bool,
}

impl<T> GPUVec<T>
where
    T: BufferContents + Copy,
{
    // BufferUsage is automatically ORed with TRANSFER_DST and TRANSFER_SRC
    pub fn new(gpu: &GPUManager, usage: BufferUsage, persist: bool) -> Self {
        let buffer = gpu.buffer_array(
            1,
            MemoryTypeFilter::PREFER_DEVICE,
            usage | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
        );
        Self {
            data: Vec::new(),
            buffer,
            size: 0,
            usage,
            persist,
        }
    }
    pub fn push_data(&mut self, item: T) {
        self.data.push(item);
    }
    pub fn upload_delta(
        &mut self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> bool {
        if self.data.len() == self.size {
            return false;
        }
        let mut ret = false;
        if self.data.len() > self.buffer.len() as usize {
            let buf = gpu.buffer_array(
                self.data.capacity() as u64,
                MemoryTypeFilter::PREFER_DEVICE,
                self.usage | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            );
            if self.persist {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(self.buffer.clone(), buf.clone()))
                    .unwrap();
            }
            self.buffer = buf;
            ret = true;
        }
        let buf = gpu
            .sub_alloc(self.usage)
            .allocate_slice(self.data.len() as u64 - self.size as u64)
            .unwrap();
        {
            let mut write = buf.write().unwrap();
            write.copy_from_slice(&self.data[self.size..]);
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                buf,
                self.buffer.clone().slice(self.size as u64..),
            ))
            .unwrap();
        ret
    }
    pub fn force_update(
        &mut self,
        gpu: &GPUManager,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> bool {
        let mut ret = false;
        if self.data.len() == 0 {
            return false;
        }
        if self.data.len() > self.buffer.len() as usize {
            let buf = gpu.buffer_array(
                self.data.capacity() as u64,
                MemoryTypeFilter::PREFER_DEVICE,
                self.usage | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            );
            if self.persist {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(self.buffer.clone(), buf.clone()))
                    .unwrap();
            }
            self.buffer = buf;
            ret = true;
        }
        let buf = gpu
            .sub_alloc(self.usage)
            .allocate_slice(self.data.len() as u64)
            .unwrap();
        {
            let mut write = buf.write().unwrap();
            write.copy_from_slice(&self.data);
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(buf, self.buffer.clone()))
            .unwrap();
        self.size = self.data.len();
        ret
    }
    pub fn clear(&mut self) {
        self.size = 0;
        self.data.clear();
    }
    pub fn resize_buffer(&mut self, new_size: usize, gpu: &GPUManager, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> bool {
        self.size = new_size;
        if new_size > self.buffer.len() as usize {
            let buf = gpu.buffer_array(
                new_size.next_power_of_two() as u64,
                MemoryTypeFilter::PREFER_DEVICE,
                self.usage | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            );
            if self.persist {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(self.buffer.clone(), buf.clone()))
                    .unwrap();
            }
            self.buffer = buf;
            return true;
        }
        // self.data.resize(new_size, unsafe { std::mem::zeroed() });
        false
    }
    pub fn buf(&self) -> Subbuffer<[T]> {
        self.buffer.clone()
    }
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
    pub fn buffer_len(&self) -> usize {
        self.size
    }
    pub fn get_data(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }
    pub fn get_data_mut(&mut self, i: usize) -> Option<&mut T> {
        self.data.get_mut(i)
    }
}
