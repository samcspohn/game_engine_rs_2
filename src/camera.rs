use glam::{Quat, Vec3};



pub struct Camera {
    pub pos: Vec3,
    pub rot: Quat,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
        }
    }
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        let translate = glam::Mat4::from_translation(-self.pos);
        let rotate = glam::Mat4::from_quat(self.rot.conjugate());
        rotate * translate
    }
    pub fn get_proj_matrix(&self, aspect: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0)
    }
    pub fn translate(&mut self, v: Vec3) {
        self.pos += self.rot * v;
    }
    pub fn rotate(&mut self, axis: Vec3, angle: f32) {
        self.rot = Quat::from_axis_angle(axis, angle) * self.rot;
    }
}