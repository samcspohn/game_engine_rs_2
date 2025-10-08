use std::collections::HashSet;

use winit::{event::MouseButton, keyboard::KeyCode};



pub struct Mouse {
    pos: (f64, f64),
    delta: (f64, f64),
    buttons_held: HashSet<MouseButton>,
    buttons_pressed: HashSet<MouseButton>,
    buttons_released: HashSet<MouseButton>,
}

pub struct Input {
    keys_held: HashSet<KeyCode>,
    keys_pressed: HashSet<KeyCode>,
    keys_released: HashSet<KeyCode>,
    pub mouse: Mouse,
}

impl Input {
    pub fn new() -> Self {
        Self {
            keys_held: HashSet::new(),
            keys_pressed: HashSet::new(),
            keys_released: HashSet::new(),
            mouse: Mouse::new(),
        }
    }
    pub(crate) fn update(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.mouse.update();
    }
    pub(crate) fn on_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            if self.keys_held.insert(key) {
                self.keys_pressed.insert(key);
            }
        } else {
            if self.keys_held.remove(&key) {
                self.keys_released.insert(key);
            }
        }
    }
    pub fn get_key(&self, key: KeyCode) -> bool {
        self.keys_held.contains(&key)
    }
    pub fn get_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }
    pub fn get_key_released(&self, key: KeyCode) -> bool {
        self.keys_released.contains(&key)
    }
}

impl Mouse {
    pub fn new() -> Self {
        Self {
            pos: (0.0, 0.0),
            delta: (0.0, 0.0),
            buttons_held: HashSet::new(),
            buttons_pressed: HashSet::new(),
            buttons_released: HashSet::new(),
        }
    }
    pub(crate) fn update(&mut self) {
        self.delta = (0.0, 0.0);
        self.buttons_pressed.clear();
        self.buttons_released.clear();
    }
    pub(crate) fn on_scroll(&mut self, dx: f64, dy: f64) {
        // self.scroll = (dx, dy);
    }
    pub(crate) fn cursor_pos(&mut self, x: f64, y: f64) {
        self.pos = (x, y);
    }
    pub(crate) fn on_motion(&mut self, x: f64, y: f64) {
        self.delta.0 += x;
        self.delta.1 += y;
        // self.pos = (x, y);
    }
    pub(crate) fn on_button(&mut self, button: MouseButton, pressed: bool) {
        if pressed {
            if self.buttons_held.insert(button) {
                self.buttons_pressed.insert(button);
            }
        } else {
            if self.buttons_held.remove(&button) {
                self.buttons_released.insert(button);
            }
        }
    }
    pub fn position(&self) -> (f64, f64) {
        self.pos
    }
    pub fn delta(&self) -> (f64, f64) {
        self.delta
    }
    pub fn get_button(&self, button: MouseButton) -> bool {
        self.buttons_held.contains(&button)
    }
    pub fn get_button_pressed(&self, button: MouseButton) -> bool {
        self.buttons_pressed.contains(&button)
    }
    pub fn get_button_released(&self, button: MouseButton) -> bool {
        self.buttons_released.contains(&button)
    }
}