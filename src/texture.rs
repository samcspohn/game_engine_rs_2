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
                texture_from_bytes(g, &pixels, img.width(), img.height(), None, None)
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
        let (image, sampler) = texture_from_bytes(gpu, &pixels, 2, 2, None, None);
        Texture { image, sampler }
    }
    pub fn white(gpu: &GPUManager) -> Self {
		let pixels: Vec<u8> = vec![
			255, 255, 255, 255, // White
		];
		let (image, sampler) = texture_from_bytes(gpu, &pixels, 1, 1, None, None);
		Texture { image, sampler }
	}
	pub fn black(gpu: &GPUManager) -> Self {
		let pixels: Vec<u8> = vec![
			0, 0, 0, 255, // Black
		];
		let (image, sampler) = texture_from_bytes(gpu, &pixels, 1, 1, None, None);
		Texture { image, sampler }
	}
	pub fn from_bytes(gpu: &GPUManager, data: &[u8], width: u32, height: u32, format: Option<Format>, base_color: Option<[f32;4]>) -> Self {
		let (image, sampler) = texture_from_bytes(gpu, data, width, height, format, base_color);
		Texture { image, sampler }
	}
}

pub fn get_format(format: gltf::image::Format) -> Option<Format> {
	match format {
		gltf::image::Format::R8 => Some(Format::R8_SRGB),
		gltf::image::Format::R8G8 => Some(Format::R8G8_SRGB),
		gltf::image::Format::R8G8B8 => Some(Format::R8G8B8_SRGB),
		gltf::image::Format::R8G8B8A8 => Some(Format::R8G8B8A8_SRGB),
		// gltf::image::Format::B8G8R8 => Some(Format::B8G8R8_UNORM),
		// gltf::image::Format::B8G8R8A8 => Some(Format::B8G8R8A8_UNORM),
		gltf::image::Format::R16 => Some(Format::R16_SNORM),
		gltf::image::Format::R16G16 => Some(Format::R16G16_SNORM),
		gltf::image::Format::R16G16B16 => Some(Format::R16G16B16_SNORM),
		gltf::image::Format::R16G16B16A16 => Some(Format::R16G16B16A16_SNORM),
		gltf::image::Format::R32G32B32A32FLOAT => Some(Format::R32G32B32A32_SFLOAT),
		gltf::image::Format::R32G32B32FLOAT => Some(Format::R32G32B32_SFLOAT),
		_ => None,
	}
}

pub fn texture_from_bytes(
    gpu: &GPUManager,
    data: &[u8],
    width: u32,
    height: u32,
    format: Option<Format>,
    _base_color: Option<[f32;4]>,
) -> (Arc<ImageView>, Arc<Sampler>) {
    let mut uploads = AutoCommandBufferBuilder::primary(
        gpu.cmd_alloc.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    let mut data = data.to_vec();

    let base_color = _base_color.unwrap_or([1.0, 1.0, 1.0, 1.0]);
    let mut format = format.unwrap_or(Format::R8G8B8A8_SRGB);
    if format == Format::R8G8B8_SRGB {
    	data = {
			let mut new_data = Vec::with_capacity((width * height * 4) as usize);
			for i in 0..(width * height) as usize {
				new_data.push(data[i * 3]);
				new_data.push(data[i * 3 + 1]);
				new_data.push(data[i * 3 + 2]);
				new_data.push(255u8);
			}
			new_data
		};
     format = Format::R8G8B8A8_SRGB;

    }
  //   for col in data.chunks_mut(4) {
  //   	col[0] = ((col[0] as f32 / 255.0) * base_color[0]) as u8 * 255;
		// col[1] = ((col[1] as f32 / 255.0) * base_color[1]) as u8 * 255;
		// col[2] = ((col[2] as f32 / 255.0) * base_color[2]) as u8 * 255;
		// col[3] = ((col[3] as f32 / 255.0) * base_color[3]) as u8 * 255;
  //   }

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

        image_data.copy_from_slice(&data);
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
    .expect(&format!("failed to create image with format: {:?}", format));

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
