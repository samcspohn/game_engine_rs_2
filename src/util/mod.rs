use std::{cmp::Reverse, ops::Div};

use dary_heap::DaryHeap;
use egui::mutex::Mutex;
use segvec::SegVec;

mod avail;
pub use avail::Avail;
mod storage;
pub use storage::Storage;
pub mod container;

pub fn get_chunk_size(num_items: usize) -> usize {
    let chunk_size = ((num_items as f32).sqrt().ceil() as usize).max(1);
    let num_chunks = num_items.div_ceil(chunk_size);
    if chunk_size != 1 && num_chunks < rayon::current_num_threads() {
        return (num_items as f32).div(rayon::current_num_threads() as f32).ceil() as usize;
    }
    chunk_size
}