use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage}, command_buffer::{AutoCommandBufferBuilder, BlitImageInfo, BufferImageCopy, CommandBufferUsage, CopyBufferToImageInfo, ImageBlit, PrimaryCommandBufferAbstract}, format::Format, image::{max_mip_levels, sampler::{Filter, Sampler, SamplerCreateInfo}, view::ImageView, Image, ImageCreateInfo, ImageSubresourceLayers, ImageType, ImageUsage}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}, DeviceSize};

use crate::{asset_manager::{Asset, DeferredAssetQueue}, gpu_manager::{GPUManager, GPUWorkQueue}};

#[derive(Debug, Clone)]
pub struct Texture {
    pub image: Arc<ImageView>,
    pub sampler: Arc<Sampler>, // add sampler cache
}

impl Texture {
    pub fn new(image: Arc<ImageView>, sampler: Arc<Sampler>) -> Self {
        Self { image, sampler }
    }
}

impl Asset for Texture {
    fn load_from_file(path: impl AsRef<std::path::Path>, gpu: GPUWorkQueue, asset: DeferredAssetQueue) -> Result<Self, String>
    {
        if let Ok(img) = image::open(&path) {
            let pixels: Vec<u8> = img.to_rgba8().iter().cloned().collect();
            
            let (image, sampler) = gpu.enqueue_work(move |g| {
                texture_from_bytes(g, &pixels, img.width(), img.height())
            }).wait().unwrap();
            // let mut texture_array = TEXTURE_ARRAY.lock();
            // let index = {
            //     texture_array.push((image.clone(), sampler.clone()));
            //     texture_array.len() - 1
            // };
            let t = Texture {
                // file: path.into(),
                image,
                sampler,
                // index,
            };
            // if unsafe { DEFAULT_TEXTURE.is_none() } {
            //     unsafe {
            //         DEFAULT_TEXTURE = Some(t.clone());
            //     }
            // }
            Ok(t)
        } else {
            // let t = unsafe {
            //     DEFAULT_TEXTURE
            //         .clone()
            //         .expect("Default texture not set, cannot load texture from file")
            // };
            // t
            // panic!("file not found: {:?}", path)
            Err(format!("Failed to load texture from file: {:?}", path.as_ref()))
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Texture {
    pub fn default(gpu: &GPUManager) -> Self {
        let pixels: Vec<u8> = vec![
            255, 0, 255, 255, // Magenta
            0, 255, 255, 255, // Cyan
            255, 255, 0, 255, // Yellow
            0, 0, 0, 255,     // Black
        ];
        let (image, sampler) = texture_from_bytes(gpu, &pixels, 2, 2);
        Texture { image, sampler }
    }
}

pub fn texture_from_bytes(
    gpu: &GPUManager,
    data: &[u8],
    width: u32,
    height: u32,
) -> (Arc<ImageView>, Arc<Sampler>) {
    let mut uploads = AutoCommandBufferBuilder::primary(
        gpu.cmd_alloc.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let format = Format::R8G8B8A8_SRGB;
    let extent: [u32; 3] = [width, height, 1];
    let array_layers = 1u32;


    let buffer_size = data.len() as DeviceSize;
    let upload_buffer = Buffer::new_slice(
        gpu.mem_alloc.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        buffer_size,
    )
    .unwrap();

    {
        let mut image_data = &mut *upload_buffer.write().unwrap();

        image_data.copy_from_slice(data);
    }

    let region = BufferImageCopy {
        image_subresource: ImageSubresourceLayers::from_parameters(format, 1),
        image_extent: extent.into(),
        ..Default::default()
    };
    // let required_size = region.buf



    let image = Image::new(
        gpu.mem_alloc.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent,
            array_layers,
            mip_levels: max_mip_levels(extent),
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    uploads
        // .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
        //     upload_buffer,
        //     image.clone(),
        // ))
        .copy_buffer_to_image(CopyBufferToImageInfo {
            // src_buffer: upload_buffer,
            // dst_image: image.clone(),
            // dst_image_layout: ImageLayout,
            regions: vec![region].into(),
            ..CopyBufferToImageInfo::buffer_image(upload_buffer, image.clone())
        })
        .unwrap();

    for level in 1..max_mip_levels(extent) {
        let src_extent = [
            (extent[0] / 2u32.pow(level - 1)).max(1),
            (extent[1] / 2u32.pow(level - 1)).max(1),
            1,
        ];
        let dst_extent = [
            (extent[0] / 2u32.pow(level)).max(1),
            (extent[1] / 2u32.pow(level)).max(1),
            1,
        ];
        uploads.blit_image(BlitImageInfo {
            regions: [ImageBlit {
                src_subresource: ImageSubresourceLayers {
                    mip_level: level - 1,
                    ..image.subresource_layers()
                },
                src_offsets: [[0;3], src_extent],
                dst_subresource: ImageSubresourceLayers {
                    mip_level: level,
                    ..image.subresource_layers()
                },
                dst_offsets: [[0;3], dst_extent],
                ..Default::default()
            }].into(),
            filter: Filter::Linear,
            ..BlitImageInfo::images(image.clone(), image.clone())
        }).unwrap();
        // let region = BufferImageCopy {
        //     image_subresource: ImageSubresourceLayers::from_parameters(format, level),
        //     image_extent: dst_extent.into(),
        //     buffer_offset: 0,
        //     buffer_image_height: 0,
        //     image_offset: [0, 0, 0],
        // };
        // uploads
        //     .copy_buffer_to_image(CopyBufferToImageInfo {
        //         regions: vec![region].into(),
        //         ..CopyBufferToImageInfo::buffer_image(
        //             upload_buffer.clone(),
        //             image.clone(),
        //         )
        //     })
        //     .unwrap();
    }

    let _ = uploads.build().unwrap().execute(gpu.queue.clone()).unwrap();

    let sampler =
        Sampler::new(gpu.device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();


    (ImageView::new_default(image).unwrap(), sampler)
}