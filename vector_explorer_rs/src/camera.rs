// camera.rs — arcball camera with Wayland-compatible mouse handling

use glam::{Mat4, Vec3, Quat};

pub struct Camera {
    pub target:    Vec3,
    pub up:        Vec3,
    pub fov_y:     f32,
    pub near:      f32,
    pub far:       f32,

    // Arcball state
    rotation:      Quat,
    distance:      f32,
    pan_offset:    Vec3,
}

impl Camera {
    pub fn new(centroid: [f32; 3], scene_radius: f32) -> Self {
        Self {
            target:     Vec3::from_array(centroid),
            up:         Vec3::Y,
            fov_y:      45.0_f32.to_radians(),
            near:       scene_radius * 0.001,
            far:        scene_radius * 100.0,
            rotation:   Quat::IDENTITY,
            distance:   scene_radius * 2.5,
            pan_offset: Vec3::ZERO,
        }
    }

    /// View matrix
    pub fn view_matrix(&self) -> Mat4 {
        let rotated_eye = self.rotation * Vec3::new(0.0, 0.0, self.distance);
        let eye = self.target + rotated_eye + self.pan_offset;
        let up = self.rotation * Vec3::Y;
        Mat4::look_at_rh(eye, self.target + self.pan_offset, up)
    }

    /// Projection matrix
    pub fn projection_matrix(&self, width: u32, height: u32) -> Mat4 {
        let aspect = width as f32 / height.max(1) as f32;
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// Combined VP matrix for the shader uniform
    pub fn view_projection(&self, width: u32, height: u32) -> [[f32; 4]; 4] {
        let vp = self.projection_matrix(width, height) * self.view_matrix();
        vp.to_cols_array_2d()
    }

    /// Left-drag — arcball rotation
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.005;
        let yaw   = Quat::from_axis_angle(Vec3::Y, -delta_x * sensitivity);
        let pitch = Quat::from_axis_angle(Vec3::X,  delta_y * sensitivity);
        self.rotation = (yaw * self.rotation * pitch).normalize();
    }

    /// Scroll or right-drag — zoom
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.1)).max(0.01);
    }

    /// Middle-drag — pan
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = self.distance * 0.001;
        let right = self.rotation * Vec3::X;
        let up    = self.rotation * Vec3::Y;
        self.pan_offset += right * (-delta_x * sensitivity);
        self.pan_offset += up   * ( delta_y * sensitivity);
    }

    /// Reset to initial view
    pub fn reset(&mut self, centroid: [f32; 3], scene_radius: f32) {
        self.rotation   = Quat::IDENTITY;
        self.distance   = scene_radius * 2.5;
        self.pan_offset = Vec3::ZERO;
        self.target     = Vec3::from_array(centroid);
    }
}
