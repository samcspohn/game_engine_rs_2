#![allow(dead_code)]
use std::{
    cell::SyncUnsafeCell,
    ops::{BitOr, Sub},
    sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering},
};

use glam::{Quat, Vec3};
use parking_lot::Mutex;
use segvec::SegVec;

use crate::util::Avail;

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
    pub fn lock(&self) -> TransformGuard<'a> {
        let lock = self.hierarchy.mutexes[self.idx as usize].lock();
        TransformGuard {
            hierarchy: self.hierarchy,
            idx: self.idx as usize,
            _lock: lock,
        }
    }
    pub fn get_idx(&self) -> u32 {
        self.idx
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
    pub fn get_parent(&self) -> Option<u32> {
        self.hierarchy.get_parent(&self)
    }
    pub fn get_children(&self) -> &mut Vec<u32> {
        self.hierarchy.get_children(&self)
    }
    // pub fn get_meta(&self) -> &mut TransformMeta {
    //     self.hierarchy.get_meta(&self)
    // }
    pub fn get_idx(&self) -> u32 {
        self.idx as u32
    }
    pub fn shift(&self, delta: Vec3) {
        self.hierarchy.shift(&self, delta);
    }
    // pub fn get_global_position(&self) -> Vec3 {
    //     self.hierarchy.get_global_position(&self)
    // }
    // pub fn get_global_rotation(&self) -> Quat {
    //     self.hierarchy.get_global_rotation(&self)
    // }
    // pub fn get_global_scale(&self) -> Vec3 {
    //     self.hierarchy.get_global_scale(&self)
    // }
}

#[repr(u8)]
enum TransformComponent {
    Position = 1 << 0,
    Rotation = 1 << 1,
    Scale = 1 << 2,
    Parent = 1 << 3,
}

impl From<u8> for TransformComponent {
    fn from(value: u8) -> Self {
        match value {
            1 => TransformComponent::Position,
            2 => TransformComponent::Rotation,
            4 => TransformComponent::Scale,
            8 => TransformComponent::Parent,
            _ => panic!("Invalid value for TransformComponent"),
        }
    }
}

impl BitOr for TransformComponent {
    type Output = TransformComponent;

    fn bitor(self, rhs: Self) -> Self::Output {
        (self as u8 | rhs as u8).into()
    }
}

pub struct _Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub name: String,
    pub parent: Option<u32>,
}

pub struct TransformHierarchy {
    mutexes: Vec<Mutex<()>>,
    positions: Vec<SyncUnsafeCell<Vec3>>,
    rotations: Vec<SyncUnsafeCell<Quat>>,
    scales: Vec<SyncUnsafeCell<Vec3>>,
    metadata: Vec<SyncUnsafeCell<TransformMeta>>,
    dirty: Vec<AtomicU8>,
    dirty_l2: Vec<AtomicU32>, // one bit for every 32 transforms 1024 total per u32
    has_children: Vec<AtomicU32>,
    active: Vec<AtomicU32>,
    avail: Avail,
    // pub buffers: SyncUnsafeCell<*mut TransformBuffers>,
}

impl TransformHierarchy {
    pub fn new() -> Self {
        Self {
            mutexes: Vec::new(),
            positions: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),
            metadata: Vec::new(),
            dirty: Vec::new(),
            dirty_l2: Vec::new(),
            has_children: Vec::new(),
            active: Vec::new(),
            avail: Avail::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.mutexes.len()
    }

    pub fn create_transform<'a>(&'a mut self, t: _Transform) -> Transform<'a> {
        let idx = self.mutexes.len();
        self.mutexes.push(Mutex::new(()));
        self.positions.push(SyncUnsafeCell::new(t.position));
        self.rotations.push(SyncUnsafeCell::new(t.rotation));
        self.scales.push(SyncUnsafeCell::new(t.scale));
        self.metadata.push(SyncUnsafeCell::new(TransformMeta {
            parent: t.parent.unwrap_or(u32::MAX),
            children: Vec::new(),
            name: t.name.to_string(),
        }));
        if let Some(parent) = t.parent {
            self.metadata[parent as usize]
                .get_mut()
                .children
                .push(idx as u32);
            self.has_children[parent as usize >> 5]
                .fetch_or(1 << (parent & 0b11111), Ordering::Relaxed);
        }
        if idx >> 1 >= self.dirty.len() {
            self.dirty.push(AtomicU8::new(0b1111)); // one u8 for every 2 transforms
        } else {
            self.dirty[idx >> 1].fetch_or(0b1111 << 4, Ordering::Relaxed);
        }
        if idx >> 10 >= self.dirty_l2.len() {
            self.dirty_l2.push(AtomicU32::new(0));
        }
        self.dirty_l2[idx >> 10].fetch_or(1 << ((idx >> 5) & 0b11111), Ordering::Relaxed);
        if idx >> 5 >= self.has_children.len() {
            self.has_children.push(AtomicU32::new(0));
        }
        if idx >> 5 >= self.active.len() {
            self.active.push(AtomicU32::new(0));
        }
        self.active[idx >> 5].fetch_or(1 << (idx & 0b11111), Ordering::Relaxed);
        // self.dirty.push(AtomicU8::new(0b1111));

        Transform::new(self, idx as u32)
    }
    pub fn remove_transform(&self, t: TransformGuard) {
        let t_idx = t.idx as u32;
        if self.get_active(t.idx as u32) {
            self.active[t.idx >> 5].fetch_and(!(1 << (t.idx & 0b11111)), Ordering::Relaxed);
            self.has_children[t.idx >> 5].fetch_and(!(1 << (t.idx & 0b11111)), Ordering::Relaxed);
            self.mark_dirty(
                &t,
                TransformComponent::Parent
                    | TransformComponent::Position
                    | TransformComponent::Rotation
                    | TransformComponent::Scale,
            );
            let children = self.get_children(&t);
            for child in children {
                let child = self._lock_internal(*child);
                self.get_meta(&child).parent = u32::MAX;
                self.mark_dirty(&child, TransformComponent::Parent);
            }
            if let Some(parent) = self.get_parent(&t) {
                drop(t);
                let children = self.get_children(&self._lock_internal(parent));
                if let Some(pos) = children.iter().position(|&x| x == t_idx) {
                    children.swap_remove(pos);
                }
                if children.is_empty() {
                    self.has_children[parent as usize >> 5]
                        .fetch_and(!(1 << (parent & 0b11111)), Ordering::Relaxed);
                }
            }
            self.avail.push(t_idx as u32);
        }
    }

    #[inline]
    fn get_active(&self, idx: u32) -> bool {
        let mask = 1 << (idx & 0b11111);
        (self.active[idx as usize >> 5].load(Ordering::Relaxed) & mask) != 0
    }

    #[inline]
    fn get_has_children(&self, idx: u32) -> bool {
        let mask = 1 << (idx & 0b11111);
        (self.has_children[idx as usize >> 5].load(Ordering::Relaxed) & mask) != 0
    }
    #[inline]
    fn mark_dirty(&self, t: &TransformGuard, component: TransformComponent) {
        let shift = (t.idx & 1) * 4; // Fixed: Added parentheses for correct precedence
        let flag = (component as u8) << shift;
        unsafe { self.dirty.get_unchecked(t.idx >> 1).fetch_or(flag, Ordering::Relaxed) };
        // self.dirty_l2[t.idx >> 10].fetch_or(1 << ((t.idx >> 5) & 0b11111), Ordering::Relaxed);
    }

    fn get_dirty(&self, idx: u32) -> u8 {
        let shift = (idx & 1) * 4; // Fixed: Added parentheses
        let mask = 0b1111 << shift;
        // Fixed: Use load to read without modifying; shift back to return only the 4 bits
        (unsafe { self.dirty.get_unchecked((idx >> 1) as usize ) }).load(Ordering::Relaxed) & mask >> shift
    }

    fn get_dirty_l2(&self, chunk_id: usize) -> u32 {
        self.dirty_l2[chunk_id].swap(0, Ordering::Relaxed)
    }

    fn mark_clean(&self, idx: u32) {
        let shift = (idx & 1) * 4; // Fixed: Added parentheses
        let mask = !(0b1111 << shift); // Fixed: Use NOT of the mask to clear the bits
        unsafe { self.dirty.get_unchecked((idx >> 1) as usize).fetch_and(mask, Ordering::Relaxed) };
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
        unsafe { &mut *self.scales.get_unchecked(idx as usize).get() }
    }
    fn _position(&self, idx: u32) -> &mut Vec3 {
        unsafe { &mut *self.positions.get_unchecked(idx as usize).get() }
    }
    fn _rotation(&self, idx: u32) -> &mut Quat {
        unsafe { &mut *self.rotations.get_unchecked(idx as usize).get() }
    }
    fn scale_by(&self, t: &TransformGuard, scale: Vec3) {
        let s = self._scale(t.idx as u32);
        *s *= scale;
        self.mark_dirty(t, TransformComponent::Scale);
    }
    fn set_scale(&self, t: &TransformGuard, scale: Vec3) {
        let s = self._scale(t.idx as u32);
        *s = scale;
        if self.get_has_children(t.idx as u32) {
            self.scale_children(t, scale);
        }
        self.mark_dirty(t, TransformComponent::Scale);
    }
    fn scale_children(&self, t: &TransformGuard, scale: Vec3) {
        let children = self.get_children(t);
        for child in children {
            let child = self._lock_internal(*child);
            let s = self._scale(child.idx as u32);
            *s *= scale;
            self.mark_dirty(&child, TransformComponent::Scale);
            if self.get_has_children(child.idx as u32) {
                self.scale_children(&child, scale);
            }
        }
    }
    fn shift(&self, t: &TransformGuard, delta: Vec3) {
        let p = self._position(t.idx as u32);
        *p += delta;
        self.mark_dirty(t, TransformComponent::Position);
        if self.get_has_children(t.idx as u32) {
            self.translate_children(t, delta);
        }
    }
    fn translate_by(&self, t: &TransformGuard, translation: Vec3) {
        let p = self._position(t.idx as u32);
        let r = self._rotation(t.idx as u32);
        let translation = *r * translation;
        *p += translation;
        self.mark_dirty(t, TransformComponent::Position);
        if self.get_has_children(t.idx as u32) {
            self.translate_children(t, translation);
        }
    }
    fn translate_children(&self, t: &TransformGuard, translation: Vec3) {
        let children = self.get_children(t);
        for child in children {
            let child = self._lock_internal(*child);
            let p = self._position(child.idx as u32);
            *p += translation;
            self.mark_dirty(&child, TransformComponent::Position);
            if self.get_has_children(child.idx as u32) {
                self.translate_children(&child, translation);
            }
        }
    }
    fn set_position(&self, t: &TransformGuard, position: Vec3) {
        let p = self._position(t.idx as u32);
        if self.get_has_children(t.idx as u32) {
            let delta = position.sub(*p);
            self.translate_children(t, delta);
        }
        *p = position;
        self.mark_dirty(t, TransformComponent::Position);
    }
    fn rotate_by(&self, t: &TransformGuard, rotation: Quat) {
        let r = self._rotation(t.idx as u32);
        *r = rotation * *r;
        self.mark_dirty(t, TransformComponent::Rotation);
        if self.get_has_children(t.idx as u32) {
            self.rotate_children(t, rotation, *self._position(t.idx as u32));
        }
    }
    fn rotate_children(&self, t: &TransformGuard, rotation: Quat, position: Vec3) {
        let children = self.get_children(t);
        for child in children {
            let child = self._lock_internal(*child);
            let r = self._rotation(child.idx as u32);
            let p = self._position(child.idx as u32);
            *p = rotation * (*p - position) + position;
            *r = rotation * *r;
            self.mark_dirty(
                &child,
                TransformComponent::Rotation | TransformComponent::Position,
            );
            if self.get_has_children(child.idx as u32) {
                self.rotate_children(&child, rotation, position);
            }
        }
    }
    fn set_rotation(&self, t: &TransformGuard, rotation: Quat) {
        let r = self._rotation(t.idx as u32);
        if self.get_has_children(t.idx as u32) {
            let delta = rotation * r.conjugate();
            self.rotate_children(t, delta, *self._position(t.idx as u32));
        }
        *r = rotation;
        self.mark_dirty(t, TransformComponent::Rotation);
    }
    pub fn get_transform(&self, idx: u32) -> Option<Transform<'_>> {
        if (idx as usize) < self.mutexes.len() {
            if self.get_active(idx) {
                Some(Transform::new(self, idx))
            } else {
                None
            }
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
        let parent = unsafe { &*self.metadata[t.idx as usize].get() }.parent;
        if parent == u32::MAX {
            None
        } else {
            Some(parent)
        }
    }
    fn get_children(&self, t: &TransformGuard) -> &mut Vec<u32> {
        &mut unsafe { &mut *self.metadata[t.idx as usize].get() }.children
    }
    fn get_meta(&self, t: &TransformGuard) -> &mut TransformMeta {
        unsafe { &mut *self.metadata[t.idx as usize].get() }
    }
    // fn get_global_position(&self, t: &TransformGuard) -> Vec3 {
    //     let mut global_position = self.get_position(t);
    //     let mut _parent = self.get_parent(t);
    //     while let Some(parent) = _parent {
    //         let parent = self._lock_internal(parent);
    //         let parent_position = self.get_position(&parent);
    //         let parent_rotation = self.get_rotation(&parent);
    //         global_position = parent_position + parent_rotation * global_position;
    //         _parent = self.get_parent(&parent);
    //     }
    //     global_position
    // }
    // fn get_global_rotation(&self, t: &TransformGuard) -> Quat {
    //     let mut global_rotation = self.get_rotation(t);
    //     let mut _parent = self.get_parent(t);
    //     while let Some(parent) = _parent {
    //         let parent = self._lock_internal(parent);
    //         let parent_rotation = self.get_rotation(&parent);
    //         global_rotation = parent_rotation * global_rotation;
    //         _parent = self.get_parent(&parent);
    //     }
    //     global_rotation
    // }
    // fn get_global_scale(&self, t: &TransformGuard) -> Vec3 {
    //     let mut global_scale = self.get_scale(t);
    //     let mut _parent = self.get_parent(t);
    //     while let Some(parent) = _parent {
    //         let parent_scale = self.get_scale(&self._lock_internal(parent));
    //         global_scale *= parent_scale;
    //         _parent = self.get_parent(&self._lock_internal(parent));
    //     }
    //     global_scale
    // }
}
