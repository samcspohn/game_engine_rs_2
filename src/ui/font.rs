use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryCommandBufferAbstract},
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::gpu_manager::GPUManager;

// Font texture layout: 16 columns x 8 rows = 128 characters
// Each character is 8x8 pixels
pub const FONT_CHAR_WIDTH: u32 = 8;
pub const FONT_CHAR_HEIGHT: u32 = 8;
pub const FONT_COLS: u32 = 16;
pub const FONT_ROWS: u32 = 8;
pub const FONT_TEXTURE_WIDTH: u32 = FONT_CHAR_WIDTH * FONT_COLS;
pub const FONT_TEXTURE_HEIGHT: u32 = FONT_CHAR_HEIGHT * FONT_ROWS;

/// Maps an ASCII character to its index in the font texture
pub fn char_to_index(c: char) -> Option<usize> {
    let code = c as u32;
    if code >= 32 && code <= 126 {
        Some((code - 32) as usize)
    } else {
        None
    }
}

/// Creates a bitmap font texture with ASCII characters 32-126
pub fn create_font_texture(gpu: &GPUManager) -> Arc<ImageView> {
    let mut font_data = vec![0u8; (FONT_TEXTURE_WIDTH * FONT_TEXTURE_HEIGHT) as usize];
    
    // Define all ASCII printable characters (32-126)
    // Each pattern is 8 bytes representing 8x8 pixels
    let patterns: [u64; 95] = [
        // 32: Space
        0x0000000000000000,
        // 33: !
        0x1818181818001800,
        // 34: "
        0x6C6C480000000000,
        // 35: #
        0x6C6CFE6CFE6C6C00,
        // 36: $
        0x187ED07C16FC1800,
        // 37: %
        0x00C6CC183066C600,
        // 38: &
        0x386C3876DCCE7600,
        // 39: '
        0x1818100000000000,
        // 40: (
        0x0C18303030180C00,
        // 41: )
        0x30180C0C0C183000,
        // 42: *
        0x00663CFF3C660000,
        // 43: +
        0x0018187E18180000,
        // 44: ,
        0x0000000000181830,
        // 45: -
        0x0000007E00000000,
        // 46: .
        0x0000000000181800,
        // 47: /
        0x06060C183060C000,
        // 48: 0
        0x7CC6CEDEF6E67C00,
        // 49: 1
        0x18381818181818FE,
        // 50: 2
        0x7CC6060C18307EFE,
        // 51: 3
        0x7CC606063C06C67C,
        // 52: 4
        0x0C1C3C6CCCFE0C0C,
        // 53: 5
        0xFEC0C0FC06C6C67C,
        // 54: 6
        0x3C60C0FCC6C6C67C,
        // 55: 7
        0xFEC6060C18303030,
        // 56: 8
        0x7CC6C67CC6C6C67C,
        // 57: 9
        0x7CC6C67E060C3870,
        // 58: :
        0x0000181800181800,
        // 59: ;
        0x0000181800181830,
        // 60: <
        0x060C183060180C06,
        // 61: =
        0x00007E00007E0000,
        // 62: >
        0x6030180C18306000,
        // 63: ?
        0x7CC6060C18001800,
        // 64: @
        0x7CC6DEDEDED07C00,
        // 65: A
        0x386CC6C6FEC6C6C6,
        // 66: B
        0xFC66667C66666FFC,
        // 67: C
        0x3C66C0C0C0C0663C,
        // 68: D
        0xF86C6666666C66F8,
        // 69: E
        0xFE6268786860FE62,
        // 70: F
        0xFE6268786860F060,
        // 71: G
        0x3C66C0C0CEC6663E,
        // 72: H
        0xC6C6C6FEC6C6C6C6,
        // 73: I
        0x3C18181818181818,
        // 74: J
        0x1E0C0C0C0CCCCC78,
        // 75: K
        0xE666666C786C66E6,
        // 76: L
        0xF06060606062FE66,
        // 77: M
        0xC6EEFEFED6C6C6C6,
        // 78: N
        0xC6E6F6DECEC6C6C6,
        // 79: O
        0x7CC6C6C6C6C6C67C,
        // 80: P
        0xFC66667C60F06060,
        // 81: Q
        0x7CC6C6C6C6D6DE7C,
        // 82: R
        0xFC66667C6C6666E6,
        // 83: S
        0x7CC6C0603C06C67C,
        // 84: T
        0x7E7E5A1818181818,
        // 85: U
        0xC6C6C6C6C6C6C67C,
        // 86: V
        0xC6C6C6C6C66C3810,
        // 87: W
        0xC6C6C6D6FEFEEEC6,
        // 88: X
        0xC6C66C38386CC6C6,
        // 89: Y
        0x6666663C18181818,
        // 90: Z
        0xFEC6860C183062FE,
        // 91: [
        0x3C30303030303030,
        // 92: \
        0xC06030180C060600,
        // 93: ]
        0x3C0C0C0C0C0C0C3C,
        // 94: ^
        0x10386CC600000000,
        // 95: _
        0x00000000000000FF,
        // 96: `
        0x3018080000000000,
        // 97: a
        0x00003C067EC6C67E,
        // 98: b
        0xC0C0C0FCC6C6C6FC,
        // 99: c
        0x00003C66C0C0663C,
        // 100: d
        0x06060676C6C6C67E,
        // 101: e
        0x00007CC6FEC0C07C,
        // 102: f
        0x1C36307C30303078,
        // 103: g
        0x00007EC6C6C67E06,
        // 104: h
        0xC0C0C0FCC6C6C6C6,
        // 105: i
        0x1800381818181818,
        // 106: j
        0x0C000C0C0C0CCC78,
        // 107: k
        0xC0C0C6CCFCCCCC00,
        // 108: l
        0x3818181818181818,
        // 109: m
        0x0000CCFEFED6D6D6,
        // 110: n
        0x0000FCC6C6C6C6C6,
        // 111: o
        0x00007CC6C6C6C67C,
        // 112: p
        0x0000FCC6C6C6FC00,
        // 113: q
        0x00007EC6C6C67E06,
        // 114: r
        0x0000DCC6C0C0C0C0,
        // 115: s
        0x00007CC07C06C67C,
        // 116: t
        0x30307C3030301C00,
        // 117: u
        0x0000C6C6C6C6C67E,
        // 118: v
        0x0000C6C6C66C3810,
        // 119: w
        0x0000C6C6D6FEEE6C,
        // 120: x
        0x0000C66C386CC6C6,
        // 121: y
        0x0000C6C6C6C67E06,
        // 122: z
        0x0000FE0C18307EFE,
        // 123: {
        0x0E18187070181818,
        // 124: |
        0x1818181818181818,
        // 125: }
        0x7018180E0E181818,
        // 126: ~
        0x76DC000000000000,
    ];

    // Write patterns to font data
    for (char_idx, pattern) in patterns.iter().enumerate() {
        let col = (char_idx % FONT_COLS as usize) as u32;
        let row = (char_idx / FONT_COLS as usize) as u32;
        
        for y in 0..8 {
            let byte = (*pattern >> ((7 - y) * 8)) as u8;
            for x in 0..8 {
                let bit = (byte >> (7 - x)) & 1;
                let pixel_x = col * FONT_CHAR_WIDTH + x;
                let pixel_y = row * FONT_CHAR_HEIGHT + y;
                let idx = (pixel_y * FONT_TEXTURE_WIDTH + pixel_x) as usize;
                font_data[idx] = if bit == 1 { 255 } else { 0 };
            }
        }
    }

    // Create GPU image
    let image = Image::new(
        gpu.mem_alloc.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8_UNORM,
            extent: [FONT_TEXTURE_WIDTH, FONT_TEXTURE_HEIGHT, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    // Upload to GPU
    let staging_buffer = Buffer::from_iter(
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
        font_data,
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.cmd_alloc.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    
    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            staging_buffer,
            image.clone(),
        ))
        .unwrap();
    
    let _ = builder.build().unwrap().execute(gpu.queue.clone()).unwrap();

    ImageView::new_default(image).unwrap()
}