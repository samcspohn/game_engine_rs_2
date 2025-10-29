use downcast_rs::{Downcast, DowncastSend, DowncastSync, impl_downcast};
use parking_lot::Mutex;
use std::{
    any::{Any, TypeId},
    cell::SyncUnsafeCell,
    collections::HashMap,
    ops::Mul,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};
use vulkano::buffer::BufferUsage;

use crate::{
    gpu_manager::{GPUManager, GPUWorkQueue},
    obj_loader::Model,
    renderer::RenderingSystem,
};

// asset manager with asyncronous loading of assets
// caching of loaded assets
// reference counting of loaded assets
// references to assets that are unloaded default to a placeholder asset
pub trait Asset: Downcast + DowncastSend + DowncastSync + Send + Sync {
    fn load_from_file(
        path: impl AsRef<Path>,
        gpu: GPUWorkQueue,
        asset: DeferredAssetQueue,
    ) -> Result<Self, String>
    where
        Self: Sized;
    fn as_any(&self) -> &dyn Any;
}

impl_downcast!(sync Asset);

trait AssetWorkItemBase {
    fn is_completed(&self) -> bool;
    fn call(&self, asset: &mut AssetManager);
}
pub struct AssetWorkItem<T>
where
    T: Asset + 'static,
{
    pub work:
        SyncUnsafeCell<Option<Box<dyn FnOnce(&mut AssetManager) -> AssetHandle<T> + Send + Sync>>>,
    pub completed: AtomicBool,
    pub result: Mutex<Option<AssetHandle<T>>>,
}

impl<T> AssetWorkItemBase for AssetWorkItem<T>
where
    T: Asset + 'static,
{
    fn is_completed(&self) -> bool {
        self.completed.load(Ordering::SeqCst)
    }
    fn call(&self, asset_manager: &mut AssetManager) {
        // if let Ok(item) = Arc::try_unwrap(self) {
        self._call(asset_manager);
        // } else {
        //     panic!("Failed to unwrap Arc in AssetWorkItemBase::call");
        // }
    }
}

impl<T> AssetWorkItem<T>
where
    T: Asset + 'static,
{
    pub fn wait(&self) -> Option<AssetHandle<T>> {
        while !self.completed.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        self.result.lock().take()
    }

    pub fn _call(&self, asset: &mut AssetManager) {
        let result = unsafe { &mut *self.work.get() }.take().unwrap()(asset);
        *self.result.lock() = Some(result);
        self.completed.store(true, Ordering::SeqCst);
    }
}

pub struct DeferredAssetQueue {
    // queue of assets to load
    queue: Arc<Mutex<Vec<Arc<dyn AssetWorkItemBase + Send + Sync>>>>,
}

impl DeferredAssetQueue {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn push(&self, asset: Arc<dyn AssetWorkItemBase + Send + Sync>) {
        self.queue.lock().push(asset);
    }

    pub fn enqueue_work<F, T>(&self, work: F) -> Arc<AssetWorkItem<T>>
    where
        F: FnOnce(&mut AssetManager) -> AssetHandle<T> + Send + Sync + 'static,
        T: Asset + Send + 'static,
    {
        let item = Arc::new(AssetWorkItem {
            work: SyncUnsafeCell::new(Some(Box::new(work))),
            completed: AtomicBool::new(false),
            result: Mutex::new(None),
        });
        self.queue.lock().push(item.clone());
        item
    }

    pub fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
        }
    }
}

pub struct AssetManager {
    thread_pool: Arc<rayon::ThreadPool>,
    gpu: Arc<GPUManager>,
    renderer: Arc<Mutex<RenderingSystem>>,
    // loaded_objs: Arc<Mutex<HashMap<String, Arc<dyn Asset>>>>,
    next_asset_id: AtomicU32,
    loaded_assets_files: Arc<Mutex<HashMap<TypeId, HashMap<String, u32>>>>,
    loaded_assets: Arc<Mutex<HashMap<TypeId, HashMap<u32, Arc<dyn Asset>>>>>,
    pub placeholder_assets: Arc<Mutex<HashMap<TypeId, Arc<dyn Asset>>>>,
    // placeholder_obj: Arc<dyn Asset>,
    pub rebuild_command_buffer: Arc<AtomicBool>,
    pub deferred_queue: DeferredAssetQueue,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AssetHandle<T> {
    // asset: String,
    pub asset_id: u32,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        Self {
            // asset: self.asset.clone(),
            asset_id: self.asset_id,
            _marker: std::marker::PhantomData,
        }
    }
}
impl<T> Copy for AssetHandle<T> {}

impl<T> Default for AssetHandle<T>
where
    T: Asset + 'static,
{
    fn default() -> Self {
        Self {
            // asset: String::new(),
            asset_id: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl AssetManager {
    pub fn new(gpu: Arc<GPUManager>, renderer: Arc<Mutex<RenderingSystem>>) -> Self {
        // create a simple placeholder obj (a single triangle)
        Self {
            thread_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads((rayon::current_num_threads() as f32 * 0.66).max(1.0) as usize)
                    .build()
                    .unwrap(),
            ),
            next_asset_id: AtomicU32::new(1),
            gpu,
            renderer,
            // loaded_objs: Arc::new(Mutex::new(HashMap::new())),
            loaded_assets_files: Arc::new(Mutex::new(HashMap::new())),
            loaded_assets: Arc::new(Mutex::new(HashMap::new())),
            placeholder_assets: Arc::new(Mutex::new(HashMap::new())),
            // placeholder_obj: Arc::new(placeholder_obj),
            rebuild_command_buffer: Arc::new(AtomicBool::new(true)),
            deferred_queue: DeferredAssetQueue::new(),
        }
    }

    pub fn process_deferred_queue(&mut self) {
        let mut queue = self.deferred_queue.queue.lock();
        let mut items: Vec<_> = queue.drain(..).collect();
        drop(queue);
        for item in items.drain(..) {
            item.call(self);
        }
    }

    // pub fn load_obj(&mut self, path: &str) -> AssetHandle<Obj> {
    //     if let Some(_) = self.loaded_objs.lock().get(path) {
    //         return AssetHandle {
    //             asset: path.to_string(),
    //             _marker: std::marker::PhantomData,
    //         };
    //     }
    //     let handle: AssetHandle<Obj> = AssetHandle {
    //         asset: path.to_string(),
    //         _marker: std::marker::PhantomData,
    //     };
    //     let path = path.to_string();
    //     let gpu = self.gpu.work_queue.clone();
    //     let loaded_objs = self.loaded_objs.clone();
    //     let placeholder_obj = self.placeholder_obj.clone();
    //     let rebuild_command_buffer = self.rebuild_command_buffer.clone();
    //     self.thread_pool.spawn(move || {
    //         match Obj::load_from_file(&path, gpu) {
    //             Ok(obj) => {
    //                 let arc_obj = Arc::new(obj);
    //                 loaded_objs.lock().insert(path.clone(), arc_obj);
    //                 rebuild_command_buffer.store(true, Ordering::SeqCst);
    //             }
    //             Err(_) => {
    //                 // insert placeholder obj on error
    //                 loaded_objs.lock().insert(path.clone(), placeholder_obj);
    //             }
    //         }
    //     });
    //     handle
    //     // match Obj::load_from_file(path, &self.gpu.lock()) {
    //     //     Ok(obj) => {
    //     //         let arc_obj = Arc::new(obj);
    //     //         self.loaded_objs.insert(path.to_string(), arc_obj.clone());
    //     //         // AssetHandle {
    //     //         //     asset: path.to_string(),
    //     //         //     _marker: std::marker::PhantomData,
    //     //         // }
    //     //     }
    //     //     Err(_) => {
    //     //         // return placeholder obj on error
    //     //         // AssetHandle {
    //     //         //     asset: "placeholder".to_string(),
    //     //         //     _marker: std::marker::PhantomData,
    //     //         // }
    //     //     }
    //     // }
    // }
    pub fn set_placeholder_asset<T: Asset + 'static>(&mut self, asset: T) -> Arc<T> {
        let type_id = TypeId::of::<T>();
        let arc_asset = Arc::new(asset);
        self.placeholder_assets
            .lock()
            .insert(type_id, arc_asset.clone());
        self.loaded_assets
            .lock()
            .entry(type_id)
            .or_insert_with(HashMap::new);
        arc_asset
    }
    pub fn load_asset_custom<T: Asset + 'static, F>(
        &mut self,
        name_path: &str,
        f: F,
    ) -> AssetHandle<T>
    where
        F: FnOnce(GPUWorkQueue, DeferredAssetQueue) -> Result<T, String> + Send + 'static,
    {
        let name_path = name_path.to_string();
        let mut loaded_assets_files_lock = self.loaded_assets_files.lock();
        if loaded_assets_files_lock
            .get(&TypeId::of::<T>())
            .is_some()
        {
            if let Some(id) = loaded_assets_files_lock
                .get(&TypeId::of::<T>())
                .unwrap()
                .get(&name_path)
            {
                return AssetHandle {
                    // asset: name_path,
                    asset_id: *id,
                    _marker: std::marker::PhantomData,
                };
            }
        }
        let type_id = TypeId::of::<T>();
        let new_id = self.next_asset_id.fetch_add(1, Ordering::SeqCst);
        let ret = AssetHandle {
            // asset: path.to_string(),
            asset_id: new_id,
            _marker: std::marker::PhantomData,
        };
        loaded_assets_files_lock
            .entry(type_id)
            .or_insert_with(HashMap::new)
            .insert(name_path, new_id);
        let gpu = self.gpu.work_queue.clone();
        let deferred_queue = self.deferred_queue.clone();
        let loaded_assets = self.loaded_assets.clone();
        let placeholder_assets = self.placeholder_assets.clone();
        let rebuild_command_buffer = self.rebuild_command_buffer.clone();
        let asset_id = ret.asset_id;
        let _r = self.renderer.clone();

        self.thread_pool.spawn(move || {
            match f(gpu, deferred_queue.clone()) {
                Ok(asset) => {
                    let arc_asset = Arc::new(asset);
                    let mut assets_lock = loaded_assets.lock();
                    let assets = assets_lock.entry(type_id).or_insert_with(HashMap::new);
                    assets.insert(asset_id, arc_asset.clone());
                    rebuild_command_buffer.store(true, Ordering::SeqCst);
                    if type_id == TypeId::of::<Model>() {
                        deferred_queue.enqueue_work(move |_a| {
                            let asset_trait: Arc<dyn Asset> = arc_asset.clone();
                            let handle = AssetHandle::_from_id(asset_id);
                            if let Ok(model_asset) = asset_trait.downcast_arc::<Model>() {
                                // println!("Registering model asset id {}", asset_id);
                                _r.lock().register_model(handle, model_asset);
                            }
                            handle
                        });
                    }
                }
                Err(err) => {
                    eprintln!("Failed to load asset: {}", err);
                    // insert placeholder asset on error
                    if let Some(placeholder) = placeholder_assets.lock().get(&type_id) {
                        let mut assets_lock = loaded_assets.lock();
                        let assets = assets_lock.entry(type_id).or_insert_with(HashMap::new);
                        assets.insert(asset_id, placeholder.clone());
                    }
                }
            }
        });
        ret
    }
    pub fn load_asset<T: Asset + 'static>(
        &mut self,
        path: &str,
        callbk: Option<Box<dyn FnMut(AssetHandle<T>, Arc<T>) + Send + Sync>>,
    ) -> AssetHandle<T> {
        let type_id = TypeId::of::<T>();
        let mut loaded_assets_files_lock = self.loaded_assets_files.lock();
        let ret = if let Some(assets) = loaded_assets_files_lock.get(&type_id) {
            if let Some(id) = assets.get(path) {
                AssetHandle {
                    // asset: path.to_string(),
                    asset_id: *id,
                    _marker: std::marker::PhantomData,
                }
            } else {
                // insert new id for this asset path
                let new_id = self.next_asset_id.fetch_add(1, Ordering::SeqCst);
                loaded_assets_files_lock
                    .get_mut(&type_id)
                    .unwrap()
                    .insert(path.to_string(), new_id);
                AssetHandle {
                    // asset: path.to_string(),
                    asset_id: new_id,
                    _marker: std::marker::PhantomData,
                }
            }
        } else {
            // insert new id for this asset path
            let new_id = self.next_asset_id.fetch_add(1, Ordering::SeqCst);
            loaded_assets_files_lock
                .entry(type_id)
                .or_insert_with(HashMap::new)
                .insert(path.to_string(), new_id);
            AssetHandle {
                // asset: path.to_string(),
                asset_id: new_id,
                _marker: std::marker::PhantomData,
            }
        };
        let path = path.to_string();
        let gpu = self.gpu.work_queue.clone();
        let deferred_queue = self.deferred_queue.clone();
        let loaded_assets = self.loaded_assets.clone();
        let placeholder_assets = self.placeholder_assets.clone();
        let rebuild_command_buffer = self.rebuild_command_buffer.clone();
        let asset_id = ret.asset_id;
        let _r = self.renderer.clone();
        self.thread_pool.spawn(move || {
            match T::load_from_file(&path, gpu, deferred_queue.clone()) {
                Ok(asset) => {
                    let arc_asset = Arc::new(asset);
                    let mut assets_lock = loaded_assets.lock();
                    let assets = assets_lock.entry(type_id).or_insert_with(HashMap::new);
                    assets.insert(asset_id, arc_asset.clone());
                    rebuild_command_buffer.store(true, Ordering::SeqCst);
                    if type_id == TypeId::of::<Model>() {
                        let asset_trait: Arc<dyn Asset> = arc_asset.clone();
                        if let Ok(model_asset) = asset_trait.downcast_arc::<Model>() {
                            // println!("Registering model asset id {}", asset_id);
                            _r.lock()
                                .register_model(AssetHandle::_from_id(asset_id), model_asset);
                        }
                    }
                    if let Some(mut cb) = callbk {
                        deferred_queue.enqueue_work(move |_a| {
                            let handle = AssetHandle {
                                // asset: path.to_string(),
                                asset_id,
                                _marker: std::marker::PhantomData,
                            };
                            cb(handle, arc_asset);
                            handle
                        });
                    }
                }
                Err(err) => {
                    eprintln!("Failed to load asset from {}: {}", path, err);
                    // insert placeholder asset on error
                    if let Some(placeholder) = placeholder_assets.lock().get(&type_id) {
                        let mut assets_lock = loaded_assets.lock();
                        let assets = assets_lock.entry(type_id).or_insert_with(HashMap::new);
                        assets.insert(asset_id, placeholder.clone());
                    }
                }
            }
        });
        ret
    }
}

impl<T> AssetHandle<T> {
    pub fn get(&self, manager: &AssetManager) -> Arc<T>
    where
        T: Asset + 'static,
    {
        let type_id = TypeId::of::<T>();
        if let Some(assets) = manager.loaded_assets.lock().get(&type_id) {
            if let Some(asset) = assets.get(&self.asset_id) {
                if let Ok(concrete_asset) = Arc::clone(asset).downcast_arc::<T>() {
                    return concrete_asset;
                } else {
                    panic!("Failed to downcast loaded asset to type T");
                }
            } else {
                if let Some(placeholder) = manager.placeholder_assets.lock().get(&type_id) {
                    if let Ok(concrete_asset) = Arc::clone(placeholder).downcast_arc::<T>() {
                        return concrete_asset;
                    } else {
                        panic!("Failed to downcast placeholder asset to type T");
                    }
                } else {
                    panic!("No placeholder asset for type T");
                }
            }
        } else {
            panic!("No assets loaded for type T");
        }
        // if let Some(asset) = manager.loaded_objs.lock().get(&self.asset) {
        //     if let Ok(concrete_asset) = Arc::clone(asset).downcast_arc::<T>() {
        //         return concrete_asset;
        //     }
        // }

        // // println!("Asset not found, returning placeholder: {}", self.asset);

        // if let Ok(concrete_asset) = Arc::clone(&manager.placeholder_obj).downcast_arc::<T>() {
        //     return concrete_asset;
        // } else {
        //     panic!("Failed to downcast placeholder asset to type T");
        // }
    }

    pub fn _from_id(asset_id: u32) -> Self {
        Self {
            // asset: String::new(),
            asset_id,
            _marker: std::marker::PhantomData,
        }
    }
}
