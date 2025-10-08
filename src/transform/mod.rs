use std::{
    cell::SyncUnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};

use glam::{Quat, Vec3};
use parking_lot::Mutex;
use segvec::SegVec;

pub mod compute;

struct TransformMeta {
    parent: u32,
    children: Vec<u32>,
    name: String,
}

pub struct Transform<'a> {
    hierarchy: &'a TransformHierarchy,
    idx: u32,
}

impl<'a> Transform<'a> {
    fn new(hierarchy: &'a TransformHierarchy, idx: u32) -> Self {
        Self { hierarchy, idx }
    }
    pub fn lock(&self) -> TransformGuard {
        let lock = self.hierarchy.mutexes[self.idx as usize].lock();
        TransformGuard {
            hierarchy: self.hierarchy,
            idx: self.idx as usize,
            _lock: lock,
        }
    }
}

pub struct TransformGuard<'a> {
    hierarchy: &'a TransformHierarchy,
    idx: usize,
    _lock: parking_lot::MutexGuard<'a, ()>,
}

impl<'a> TransformGuard<'a> {
    pub fn scale_by(&self, scale: Vec3) {
        self.hierarchy.scale_by(&self, scale);
    }
    pub fn set_scale(&self, scale: Vec3) {
        self.hierarchy.set_scale(&self, scale);
    }
    pub fn translate_by(&self, translation: Vec3) {
        self.hierarchy.translate_by(&self, translation);
    }
    pub fn set_position(&self, position: Vec3) {
        self.hierarchy.set_position(&self, position);
    }
    pub fn rotate_by(&self, rotation: Quat) {
        self.hierarchy.rotate_by(&self, rotation);
    }
    pub fn set_rotation(&self, rotation: Quat) {
        self.hierarchy.set_rotation(&self, rotation);
    }
    pub fn get_position(&self) -> Vec3 {
        self.hierarchy.get_position(&self)
    }
    pub fn get_rotation(&self) -> Quat {
        self.hierarchy.get_rotation(&self)
    }
    pub fn get_scale(&self) -> Vec3 {
        self.hierarchy.get_scale(&self)
    }
    pub fn get_global_position(&self) -> Vec3 {
        self.hierarchy.get_global_position(&self)
    }
    pub fn get_global_rotation(&self) -> Quat {
        self.hierarchy.get_global_rotation(&self)
    }
    pub fn get_global_scale(&self) -> Vec3 {
        self.hierarchy.get_global_scale(&self)
    }
}

pub struct TransformHierarchy {
    mutexes: SegVec<Mutex<()>>,
    positions: SegVec<SyncUnsafeCell<Vec3>>,
    rotations: SegVec<SyncUnsafeCell<Quat>>,
    scales: SegVec<SyncUnsafeCell<Vec3>>,
    metadata: SegVec<TransformMeta>,
    dirty: SegVec<SyncUnsafeCell<u32>>,
}

enum TransformComponent {
    Position,
    Rotation,
    Scale,
    Parent,
}

impl TransformHierarchy {
    pub fn new() -> Self {
        Self {
            mutexes: SegVec::new(),
            positions: SegVec::new(),
            rotations: SegVec::new(),
            scales: SegVec::new(),
            metadata: SegVec::new(),
            dirty: SegVec::new(),
        }
    }
    pub fn create_transform<'a>(&'a mut self, name: &str) -> Transform<'a> {
        let idx = self.mutexes.len();
        self.mutexes.push(Mutex::new(()));
        self.positions.push(SyncUnsafeCell::new(Vec3::ZERO));
        self.rotations.push(SyncUnsafeCell::new(Quat::IDENTITY));
        self.scales.push(SyncUnsafeCell::new(Vec3::ONE));
        self.metadata.push(TransformMeta {
            parent: u32::MAX,
            children: Vec::new(),
            name: name.to_string(),
        });
        self.dirty.push(SyncUnsafeCell::new(0b1111));
        Transform::new(self, idx as u32)
    }
    fn mark_dirty(&self, idx: u32, component: TransformComponent) {
        // match component {
        //     TransformComponent::Position => {
        //         self.dirty[idx as usize]
        //     }
        //     TransformComponent::Rotation => {
        //         self.dirty[idx as usize][1].store(true, Ordering::Relaxed)
        //     }
        //     TransformComponent::Scale => self.dirty[idx as usize][2].store(true, Ordering::Relaxed),
        //     TransformComponent::Parent => {
        //         self.dirty[idx as usize][3].store(true, Ordering::Relaxed)
        //     }
        // }
        let flag = match component {
            TransformComponent::Position => 1 << 0,
            TransformComponent::Rotation => 1 << 1,
            TransformComponent::Scale => 1 << 2,
            TransformComponent::Parent => 1 << 3,
        };
        unsafe { *self.dirty[idx as usize].get() |= flag };
    }
    fn get_dirty(&self, idx: u32) -> u32 {
        unsafe { *self.dirty[idx as usize].get() }
    }
    fn mark_clean(&self, idx: u32) {
        unsafe { *self.dirty[idx as usize].get() = 0 };
    }
    fn _lock_internal<'a>(&'a self, idx: u32) -> TransformGuard<'a> {
        let lock = self.mutexes[idx as usize].lock();
        TransformGuard {
            hierarchy: self,
            idx: idx as usize,
            _lock: lock,
        }
    }
    fn _scale(&self, idx: u32) -> &mut Vec3 {
        unsafe { &mut *self.scales[idx as usize].get() }
    }
    fn _position(&self, idx: u32) -> &mut Vec3 {
        unsafe { &mut *self.positions[idx as usize].get() }
    }
    fn _rotation(&self, idx: u32) -> &mut Quat {
        unsafe { &mut *self.rotations[idx as usize].get() }
    }
    fn scale_by(&self, t: &TransformGuard, scale: Vec3) {
        let s = self._scale(t.idx as u32);
        *s *= scale;
        self.mark_dirty(t.idx as u32, TransformComponent::Scale);
    }
    fn set_scale(&self, t: &TransformGuard, scale: Vec3) {
        let s = self._scale(t.idx as u32);
        *s = scale;
        self.mark_dirty(t.idx as u32, TransformComponent::Scale);
    }
    fn translate_by(&self, t: &TransformGuard, translation: Vec3) {
        let p = self._position(t.idx as u32);
        let r = self._rotation(t.idx as u32);
        *p += *r * translation;
        self.mark_dirty(t.idx as u32, TransformComponent::Position);
    }
    fn set_position(&self, t: &TransformGuard, position: Vec3) {
        let p = self._position(t.idx as u32);
        *p = position;
        self.mark_dirty(t.idx as u32, TransformComponent::Position);
    }
    fn rotate_by(&self, t: &TransformGuard, rotation: Quat) {
        let r = self._rotation(t.idx as u32);
        *r = rotation * *r;
        self.mark_dirty(t.idx as u32, TransformComponent::Rotation);
    }
    fn set_rotation(&self, t: &TransformGuard, rotation: Quat) {
        let r = self._rotation(t.idx as u32);
        *r = rotation;
        self.mark_dirty(t.idx as u32, TransformComponent::Rotation);
    }
    pub fn get_transform<'a>(&'a self, idx: u32) -> Option<Transform<'a>> {
        if (idx as usize) < self.mutexes.len() {
            Some(Transform::new(self, idx))
        } else {
            None
        }
    }
    fn get_position(&self, t: &TransformGuard) -> Vec3 {
        unsafe { *self.positions[t.idx as usize].get() }
    }
    fn get_rotation(&self, t: &TransformGuard) -> Quat {
        unsafe { *self.rotations[t.idx as usize].get() }
    }
    fn get_scale(&self, t: &TransformGuard) -> Vec3 {
        unsafe { *self.scales[t.idx as usize].get() }
    }
    fn get_parent(&self, t: &TransformGuard) -> Option<u32> {
        let parent = self.metadata[t.idx as usize].parent;
        if parent == u32::MAX {
            None
        } else {
            Some(parent)
        }
    }
    fn get_global_position(&self, t: &TransformGuard) -> Vec3 {
        let mut global_position = self.get_position(t);
        let mut _parent = self.get_parent(t);
        while let Some(parent) = _parent {
            let parent_position = self.get_position(&self._lock_internal(parent));
            let parent_rotation = self.get_rotation(&self._lock_internal(parent));
            global_position = parent_position + parent_rotation * global_position;
            _parent = self.get_parent(&self._lock_internal(parent));
        }
        global_position
    }
    fn get_global_rotation(&self, t: &TransformGuard) -> Quat {
        let mut global_rotation = self.get_rotation(t);
        let mut _parent = self.get_parent(t);
        while let Some(parent) = _parent {
            let parent_rotation = self.get_rotation(&self._lock_internal(parent));
            global_rotation = parent_rotation * global_rotation;
            _parent = self.get_parent(&self._lock_internal(parent));
        }
        global_rotation
    }
    fn get_global_scale(&self, t: &TransformGuard) -> Vec3 {
        let mut global_scale = self.get_scale(t);
        let mut _parent = self.get_parent(t);
        while let Some(parent) = _parent {
            let parent_scale = self.get_scale(&self._lock_internal(parent));
            global_scale *= parent_scale;
            _parent = self.get_parent(&self._lock_internal(parent));
        }
        global_scale
    }
}
