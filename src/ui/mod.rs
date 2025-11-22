//! Simple UI system for rendering text overlays
//!
//! This module provides a lightweight, GPU-based text rendering system using
//! bitmap fonts. It's designed for displaying debug information and simple UI
//! elements without the overhead of a full UI framework.
//!
//! # Features
//!
//! - Complete ASCII character support (characters 32-126)
//! - GPU-accelerated rendering with Vulkan
//! - Custom bitmap font with 8x8 pixel characters
//! - Alpha blending for text rendering
//! - Low overhead - suitable for real-time applications
//!
//! # Example
//!
//! ```no_run
//! use game_engine_rs_2::ui::SimpleUI;
//!
//! // Create UI system
//! let ui = SimpleUI::new(gpu.clone());
//!
//! // Draw FPS counter at top-left
//! ui.draw(&mut builder, target, &viewport, 60.5);
//!
//! // Draw custom text at any position
//! ui.draw_text(&mut builder, target, &viewport,
//!     "Hello, World!",
//!     100.0,  // x position
//!     50.0,   // y position
//!     16.0    // character size
//! );
//! ```
//!
//! # Font Coverage
//!
//! The bitmap font includes:
//! - Uppercase letters: A-Z
//! - Lowercase letters: a-z
//! - Digits: 0-9
//! - Punctuation: ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
//! - Space character
//!
//! Any character outside the ASCII printable range (32-126) will be skipped
//! during rendering.

pub mod font;
mod ui;

pub use ui::UI;
