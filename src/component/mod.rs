use std::{
    any::TypeId,
    cell::SyncUnsafeCell,
    collections::HashMap,
    mem::MaybeUninit,
    sync::{Arc, atomic::AtomicU32},
};

use glam::{Quat, Vec3};
use parking_lot::Mutex;

use crate::{
    asset_manager::AssetHandle,
    engine::Engine,
    gpu_manager::GPUManager,
    renderer::{_RendererComponent, RendererComponent},
    transform::{self, _Transform, Transform, TransformHierarchy, compute::PerfCounter},
    util::seg_storage::{SegStorage, get_from_slice, get_from_slice_unchecked},
};
use rayon::prelude::*;

pub trait Component {
    fn init(&mut self, transform: &Transform, e: &Engine) {}
    fn deinit(&mut self, transform: &Transform, e: &Engine) {}
    fn update(&mut self, dt: f32, transform: &transform::Transform) {}
}

pub struct ComponentStorage<T> {
    // data: Vec<MaybeUninit<Mutex<T>>>,
    data: SegStorage<Mutex<T>>,
    extent: usize,
    // avail: Vec<usize>,
    active: Vec<AtomicU32>,
    has_update: bool,
}
impl<T> ComponentStorage<T>
where
    T: Component + Send + Sync,
{
    pub fn new(has_update: bool) -> Self {
        Self {
            data: SegStorage::new(),
            extent: 0,
            // avail: Vec::new(),
            active: Vec::new(),
            has_update,
        }
    }
    pub fn set(&mut self, t_idx: u32, item: T) -> u32 {
        // let idx = t_idx as usize;
        // if idx >= self.data.len() {
        //     self.data.resize_with(idx + 1, || MaybeUninit::uninit());
        //     unsafe {
        //         self.data[idx].as_mut_ptr().write(Mutex::new(item));
        //     }
        //     let required_active_len = (idx >> 5) + 1;
        //     if required_active_len > self.active.len() {
        //         self.active
        //             .resize_with(required_active_len, || AtomicU32::new(0));
        //     }
        // } else {
        //     unsafe {
        //         self.data[idx].as_mut_ptr().write(Mutex::new(item));
        //     }
        // }
        let idx = t_idx as usize;
        self.data.set(idx, Mutex::new(item));
        let required_active_len = (idx >> 5) + 1;
        if required_active_len > self.active.len() {
            self.active
                .resize_with(required_active_len, || AtomicU32::new(0));
        }
        if idx >= self.extent {
            self.extent = idx + 1;
        }
        let atomic_idx = (idx >> 5) as usize;
        let bit_idx = idx & 31;
        self.active[atomic_idx].fetch_or(1 << bit_idx, std::sync::atomic::Ordering::Relaxed);
        idx as u32
    }
    // #[inline]
    // fn _get_unchecked(&self, idx: u32) -> &Mutex<T> {
    //     unsafe { &*self.data.get_unchecked(idx as usize).assume_init_ref() }
    // }
    #[inline]
    fn is_active(&self, idx: u32) -> bool {
        let atomic_idx = (idx >> 5) as usize;
        let bit_idx = idx & 31;
        (self.active[atomic_idx].load(std::sync::atomic::Ordering::Relaxed) & (1 << bit_idx)) != 0
    }
    // pub fn delete(&mut self, idx: u32) {
    //     if (idx as usize) < self.data.len() && self.is_active(idx) {
    //         let atomic_idx = (idx >> 5) as usize;
    //         let bit_idx = idx & 31;
    //         self.active[atomic_idx]
    //             .fetch_and(!(1 << bit_idx), std::sync::atomic::Ordering::Relaxed);
    //         // self.avail.push(idx as usize);
    //         unsafe {
    //             self.data.get_unchecked_mut(idx as usize).assume_init_drop();
    //         }
    //     }
    // }
    pub fn drop(&mut self, idx: u32) {
        if (idx as usize) < self.data.len() && self.is_active(idx) {
            let atomic_idx = (idx >> 5) as usize;
            let bit_idx = idx & 31;
            self.active[atomic_idx]
                .fetch_and(!(1 << bit_idx), std::sync::atomic::Ordering::Relaxed);
            // self.avail.push(idx as usize);
            self.data.drop(idx as usize);
        }
    }
    pub fn get(&self, idx: u32) -> Option<&Mutex<T>> {
        // if (idx as usize) < self.data.len() && self.is_active(idx) {
        //     Some(&self._get_unchecked(idx))
        // } else {
        //     None
        // }
        if (idx as usize) < self.data.len() && self.is_active(idx) {
            Some(self.data.get_unchecked(idx as usize))
        } else {
            None
        }
    }
    fn par_iter<F>(&self, f: F, transform_hierarchy: &TransformHierarchy)
    where
        F: Fn(&mut T, &Transform) + Sync + Send + Copy,
    {
        // let nt = (rayon::current_num_threads())
        //     .min(self.extent.div_ceil(256))
        //     .max(1);
        // let chunk_size = (self.extent + nt - 1) / nt; // ceiling division without floats
        // let chunk_size = self.extent.div_ceil(nt);
        // rayon::scope(|s| {
        //     for i in 0..nt {
        //         s.spawn(move |_| {
        //             let start = i * chunk_size;
        //             let end = ((i + 1) * chunk_size).min(self.extent);

        //             let mut idx = start;
        //             while idx < end {
        //                 let atomic_idx = idx >> 5;
        //                 let bits =
        //                     self.active[atomic_idx].load(std::sync::atomic::Ordering::Relaxed);

        //                 // Calculate the range of bits to check in this atomic u32
        //                 // let start_bit = idx & 31;
        //                 let next_atomic_boundary = (atomic_idx + 1) << 5;
        //                 let end_bit_idx = next_atomic_boundary.min(end);

        //                 // Process each bit in this u32 within our range
        //                 for current_idx in idx..end_bit_idx {
        //                     let bit_idx = current_idx & 31;
        //                     if (bits & (1 << bit_idx)) != 0 {
        //                         let component = self._get_unchecked(current_idx as u32);
        //                         let transform =
        //                             transform_hierarchy.get_transform(current_idx as u32);
        //                         {
        //                             let mut component_guard = component.lock();
        //                             f(&mut *component_guard, &transform);
        //                         }
        //                     }
        //                 }

        //                 // Move to next atomic u32
        //                 idx = next_atomic_boundary;
        //             }
        //         });
        //     }
        // });
        // (0..nt).into_par_iter().for_each(|i| {
        //     let start = i * chunk_size;
        //     let end = ((i + 1) * chunk_size).min(self.extent);

        //     if start >= end {
        //         return; // early exit for threads with empty ranges
        //     }

        //     let mut idx = start;

        //     while idx < end {
        //         let atomic_idx = idx >> 5;
        //         let bits = self.active[atomic_idx].load(std::sync::atomic::Ordering::Relaxed);

        //         // Calculate the range of bits to check in this atomic u32
        //         // let start_bit = idx & 31;
        //         let next_atomic_boundary = (atomic_idx + 1) << 5;
        //         let end_bit_idx = next_atomic_boundary.min(end);

        //         // Process each bit in this u32 within our range
        //         for current_idx in idx..end_bit_idx {
        //             let bit_idx = current_idx & 31;
        //             if (bits & (1 << bit_idx)) != 0 {
        //                 let component = self._get_unchecked(current_idx as u32);
        //                 let transform = transform_hierarchy.get_transform(current_idx as u32);
        //                 {
        //                     let mut component_guard = component.lock();
        //                     f(&mut *component_guard, &transform);
        //                 }
        //             }
        //         }

        //         // Move to next atomic u32
        //         idx = next_atomic_boundary;
        //     }
        // });
        self.active
            .par_iter()
            .enumerate()
            .chunks(8)
            .for_each(|chunk| {
                for (atomic_idx, atomic) in chunk {
                    let bits = atomic.load(std::sync::atomic::Ordering::Relaxed);
                    if bits == 0 {
                        continue; // skip if no active components in this chunk
                    }
                    let base_idx = atomic_idx << 5;
                    let chunk = self.data.get_segment_chunk_unchecked(base_idx);
                    for bit_idx in 0..32 {
                        if (bits & (1 << bit_idx)) != 0 {
                            let current_idx = base_idx + bit_idx;
                            if current_idx >= self.extent {
                                break;
                            }
                            // let component = &chunk[bit_idx];
                            let component = get_from_slice_unchecked(chunk, bit_idx);
                            // let component = self.data.get_unchecked(current_idx);
                            let transform =
                                transform_hierarchy.get_transform_unchecked(current_idx as u32);
                            {
                                let mut component_guard = component.lock();
                                f(&mut *component_guard, &transform);
                            }
                        }
                    }

                    // for bit_idx in 0..32 {
                    //     if (bits & (1 << bit_idx)) != 0 {
                    //         let current_idx = base_idx + bit_idx;
                    //         if current_idx >= self.extent {
                    //             break;
                    //         }
                    //         // let component = self._get_unchecked(current_idx as u32);
                    //         self.data.get_unchecked(current_idx)
                    //         let transform =
                    //             transform_hierarchy.get_transform_unchecked(current_idx as u32);
                    //         {
                    //             let mut component_guard = component.lock();
                    //             f(&mut *component_guard, &transform);
                    //         }
                    //     }
                    // }
                }
            });
    }

    pub fn _update(&self, dt: f32, transform_hierarchy: &TransformHierarchy) {
        if self.has_update {
            self.par_iter(|c, t| c.update(dt, t), transform_hierarchy);
        }
    }
}

impl<T: Component + Clone + Send + Sync + 'static> ComponentStorageTrait for ComponentStorage<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn remove(&mut self, idx: u32) {
        self.drop(idx);
    }
    fn update(
        &self,
        dt: f32,
        transform_hierarchy: &TransformHierarchy,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    ) {
        let n = std::any::type_name::<T>();
        perf.as_mut()
            .map(|perf| perf.entry(n.into()).or_insert(PerfCounter::new()).start());
        self._update(dt, transform_hierarchy);
        perf.as_mut().map(|perf| perf.get_mut(n).unwrap().stop());
    }
    fn clone_from_other(
        &mut self,
        other: &dyn ComponentStorageTrait,
        src_idx: u32,
        dst_idx: u32,
        e: &Engine,
        t: &Transform,
    ) {
        if let Some(other_storage) = other.as_any().downcast_ref::<ComponentStorage<T>>() {
            if let Some(other_component_mutex) = other_storage.get(src_idx) {
                let other_component = other_component_mutex.lock();
                let mut new_component = (*other_component).clone();
                new_component.init(t, e);
                self.set(dst_idx, new_component);
            }
        }
    }
}

trait ComponentStorageTrait {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn remove(&mut self, idx: u32);
    fn update(
        &self,
        dt: f32,
        transform_hierarchy: &TransformHierarchy,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    );
    fn clone_from_other(
        &mut self,
        other: &dyn ComponentStorageTrait,
        src_idx: u32,
        dst_idx: u32,
        e: &Engine,
        t: &Transform,
    );
}

pub struct ComponentRegistry {
    components: HashMap<TypeId, Box<dyn ComponentStorageTrait + Send + Sync>>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }
    pub fn register<T: Component + Clone + Send + Sync + 'static>(&mut self, has_update: bool) {
        let type_id = TypeId::of::<T>();
        if !self.components.contains_key(&type_id) {
            self.components
                .insert(type_id, Box::new(ComponentStorage::<T>::new(has_update)));
        }
    }
    pub fn get_storage<T: Component + Send + Sync + 'static>(
        &self,
    ) -> Option<&ComponentStorage<T>> {
        let type_id = TypeId::of::<T>();
        self.components
            .get(&type_id)
            .and_then(|storage| storage.as_any().downcast_ref::<ComponentStorage<T>>())
    }
    pub fn get_storage_mut<T: Component + Clone + Send + Sync + 'static>(
        &mut self,
    ) -> Option<&mut ComponentStorage<T>> {
        let type_id = TypeId::of::<T>();
        self.components
            .entry(type_id)
            .or_insert_with(|| Box::new(ComponentStorage::<T>::new(false)))
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
    }
    pub fn update_all(
        &self,
        dt: f32,
        transform_hierarchy: &TransformHierarchy,
        perf: &mut Option<HashMap<String, PerfCounter>>,
    ) {
        for storage in self.components.values() {
            storage.update(dt, transform_hierarchy, perf);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub id: u32,
}
impl Entity {
    pub fn new(id: u32) -> Self {
        Entity { id }
    }
}

// struct EntityBuilder<'a> {
//     world: &'a mut World,
//     components: Vec<Box<dyn Component>>,
// }

// impl<'a> EntityBuilder<'a> {
//     pub fn new(world: &'a mut World) -> Self {
//         Self {
//             world,
//             components: Vec::new(),
//         }
//     }

//     pub fn add_component(mut self, component: Box<dyn Component>) -> Self {
//         self.components.push(component);
//         self
//     }

//     pub fn build(self) -> Entity {
//         let entity = self.world.new_entity();
//         for component in self.components {
//             self.world.add_component(entity, component);
//         }
//         entity
//     }
// }

pub struct Scene {
    pub components: ComponentRegistry,
    pub engine: Arc<Engine>,
    pub transform_hierarchy: TransformHierarchy,
    pub perf: Option<HashMap<String, PerfCounter>>,
}

impl Scene {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            components: ComponentRegistry::new(),
            engine,
            transform_hierarchy: TransformHierarchy::new(),
            perf: None,
        }
    }
    pub fn update(&mut self, dt: f32) {
        self.components
            .update_all(dt, &self.transform_hierarchy, &mut self.perf);
    }
    pub fn new_entity(&mut self, t: _Transform) -> Entity {
        let entity = Entity::new(self.transform_hierarchy.create_transform(t).get_idx());
        entity
    }
    // pub fn new_entity_builder(&mut self) -> EntityBuilder {
    //     EntityBuilder::new(self)
    // }
    // pub fn add_component(&mut self, entity: Entity, component: Box<dyn Component>) {
    //     self.components.add_component(entity.id, component);
    // }
    pub fn add_component<T>(&mut self, entity: Entity, mut component: T)
    where
        T: Component + Clone + Send + Sync + 'static,
    {
        let t = self.transform_hierarchy.get_transform_unchecked(entity.id);
        component.init(&t, &self.engine);
        self.components
            .get_storage_mut::<T>()
            .unwrap()
            .set(entity.id, component);
    }

    pub fn remove_component<T>(&mut self, entity: Entity)
    where
        T: Component + Clone + Send + Sync + 'static,
    {
        if let Some(storage) = self.components.get_storage_mut::<T>() {
            if let Some(component_mutex) = storage.get(entity.id) {
                let mut component = component_mutex.lock();
                let t = self.transform_hierarchy.get_transform_unchecked(entity.id);
                component.deinit(&t, &self.engine);
            }
            storage.drop(entity.id);
        }
    }

    pub fn get_component<T>(&self, entity: Entity) -> Option<&Mutex<T>>
    where
        T: Component + Send + Sync + 'static,
    {
        if let Some(storage) = self.components.get_storage::<T>() {
            if let Some(component_mutex) = storage.get(entity.id) {
                return Some(component_mutex);
            }
        }
        None
    }

    pub fn instantiate(&mut self, other: &Scene) -> Transform<'_> {
        let mut entity_map = HashMap::new();

        for t_idx in 0..other.transform_hierarchy.len() as u32 {
            let other_transform = other.transform_hierarchy.get_transform_(t_idx);
            let new_transform = _Transform {
                position: other_transform.position,
                rotation: other_transform.rotation,
                scale: other_transform.scale,
                name: other_transform.name.clone(),
                parent: other_transform.parent.map(|p| {
                    *entity_map
                        .get(&p)
                        .unwrap_or_else(|| panic!("Parent transform {} not found in entity_map", p))
                }),
            };
            let new_entity = self.new_entity(new_transform);
            entity_map.insert(t_idx, new_entity.id);
        }
        // for t_idx in 0..other.transform_hierarchy.len() as u32 {
        //     let other_transform = other.transform_hierarchy.get_transform_(t_idx);
        //     if let Some(parent) = other_transform.parent {
        //         let new_parent = *entity_map.get(&parent).unwrap();
        //         let new_entity = *entity_map.get(&t_idx).unwrap();
        //         let t = self.transform_hierarchy.get_transform(new_entity);
        //         let _lock = t.lock();
        //         self.transform_hierarchy
        //             .set_parent(&_lock, Some(new_parent));
        //     }
        // }

        for (type_id, other_storage) in &other.components.components {
            // handle special case for _RendererComponent
            if type_id == &TypeId::of::<_RendererComponent>() {
                let self_storage = self
                    .components
                    .get_storage_mut::<RendererComponent>()
                    .unwrap();
                let other_storage = other_storage
                    .as_any()
                    .downcast_ref::<ComponentStorage<_RendererComponent>>()
                    .unwrap();
                for t_idx in 0..other.transform_hierarchy.len() as u32 {
                    if let Some(other_component_mutex) = other_storage.get(t_idx) {
                        let other_component = other_component_mutex.lock();
                        let model = other_component.model.clone();
                        // println!(
                        //     "({} -> {}) m: {}",
                        //     t_idx,
                        //     entity_map.get(&t_idx).unwrap(),
                        //     model.asset_id
                        // );
                        let mut component = RendererComponent::new(model);
                        component.init(
                            &self
                                .transform_hierarchy
                                .get_transform_unchecked(*entity_map.get(&t_idx).unwrap()),
                            &self.engine,
                        );
                        self_storage.set(*entity_map.get(&t_idx).unwrap(), component);
                    }
                }
            } else {
                if let Some(self_storage) = self.components.components.get_mut(type_id) {
                    for t_idx in 0..other.transform_hierarchy.len() as u32 {
                        self_storage.clone_from_other(
                            other_storage.as_ref(),
                            t_idx,
                            *entity_map.get(&t_idx).unwrap(),
                            &self.engine,
                            &self
                                .transform_hierarchy
                                .get_transform_unchecked(*entity_map.get(&t_idx).unwrap()),
                        );
                    }
                }
            }
        }
        self.transform_hierarchy
            .get_transform_unchecked(entity_map[&0])
    }
}
