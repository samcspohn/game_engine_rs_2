use std::{cmp::Reverse, ops::Div};

use dary_heap::DaryHeap;
use egui::mutex::Mutex;
use segvec::SegVec;

pub struct Avail {
    pub data: DaryHeap<Reverse<u32>, 4>,
    new_ids: Mutex<Vec<Reverse<u32>>>,
}
impl Avail {
    pub fn new() -> Self {
        Self {
            data: DaryHeap::new(),
            new_ids: Mutex::new(Vec::new()),
        }
    }
    pub fn commit(&mut self) {
        let mut a = self.new_ids.lock();
        for i in a.drain(..) {
            self.data.push(i);
        }
    }
    pub fn push(&self, i: u32) {
        self.new_ids.lock().push(Reverse(i));
        // self.data.push(Reverse(i));
    }
    pub fn pop(&mut self) -> Option<u32> {
        match self.data.pop() {
            Some(Reverse(a)) => Some(a),
            None => None,
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct Storage<T> {
    pub data: SegVec<Option<T>>,
    pub avail: Avail,
}

impl<T> Storage<T> {
    pub fn new() -> Self {
        Self {
            data: SegVec::new(),
            avail: Avail::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn insert(&mut self, v: T) -> u32 {
        if let Some(i) = self.avail.pop() {
            self.data[i as usize] = Some(v);
            i
        } else {
            let i = self.data.len() as u32;
            self.data.push(Some(v));
            i
        }
    }
    pub fn remove(&mut self, i: u32) -> Option<T> {
        if (i as usize) < self.data.len() {
            let v = self.data[i as usize].take();
            if v.is_some() {
                self.avail.push(i);
            }
            v
        } else {
            None
        }
    }
    pub fn get(&self, i: u32) -> Option<&T> {
        if (i as usize) < self.data.len() {
            self.data[i as usize].as_ref()
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, i: u32) -> Option<&mut T> {
        if (i as usize) < self.data.len() {
            self.data[i as usize].as_mut()
        } else {
            None
        }
    }
}

pub fn get_chunk_size(num_items: usize) -> usize {
    let chunk_size = ((num_items as f32).sqrt().ceil() as usize).max(1);
    let num_chunks = num_items.div_ceil(chunk_size);
    if chunk_size != 1 && num_chunks < rayon::current_num_threads() {
        return (num_items as f32).div(rayon::current_num_threads() as f32).ceil() as usize;
    }
    chunk_size
}