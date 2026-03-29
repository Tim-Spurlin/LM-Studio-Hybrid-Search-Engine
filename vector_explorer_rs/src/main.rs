// main.rs — winit event loop, ties all modules together

mod camera;
mod data;
mod point_cloud;
mod renderer;
mod search;
mod ui;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};
use crate::{
    camera::Camera,
    data::Dataset,
    point_cloud::PointCloud,
    renderer::Renderer,
    ui::UiState,
};

fn log_path(file: &str) -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/saphyre-solutions".to_string());
    Path::new(&home).join("Desktop").join("h5_tensorboard_logs").join(file)
}

struct App {
    // Initialized late after window creation
    window:          Option<Arc<Window>>,
    renderer:        Option<Renderer>,
    egui_ctx:        egui::Context,
    egui_state:      Option<egui_winit::State>,
    egui_renderer:   Option<egui_wgpu::Renderer>,
    camera:          Option<Camera>,
    ui_state:        UiState,
    dataset:         Arc<Dataset>,
    point_cloud:     Option<PointCloud>,
    centroid:        [f32; 3],
    scene_radius:    f32,

    // Frame timing
    last_frame:      Instant,
    fps_accum:       f32,
    fps_frames:      u32,
    fps_display:     f32,

    // Mouse state
    mouse_left:      bool,
    mouse_right:     bool,
    mouse_middle:    bool,
    last_mouse:      Option<(f64, f64)>,
    mouse_pos:       (f32, f32),
    do_reset_cam:    bool,
    do_screenshot:   bool,
}

impl App {
    fn new(dataset: Dataset) -> Self {
        let dataset = Arc::new(dataset);
        let (min, max) = dataset.bounds();
        let centroid   = dataset.centroid();
        let scene_radius = {
            let dx = max[0] - min[0];
            let dy = max[1] - min[1];
            let dz = max[2] - min[2];
            (dx*dx + dy*dy + dz*dz).sqrt() * 0.5
        };

        tracing::info!(
            "Dataset: {} points | radius: {:.2} | centroid: {:?}",
            dataset.points.len(), scene_radius, centroid
        );

        Self {
            window: None,
            renderer: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            camera: None,
            ui_state: UiState::new(),
            dataset,
            point_cloud: None,
            centroid,
            scene_radius,
            last_frame: Instant::now(),
            fps_accum: 0.0,
            fps_frames: 0,
            fps_display: 0.0,
            mouse_left: false,
            mouse_right: false,
            mouse_middle: false,
            last_mouse: None,
            mouse_pos: (0.0, 0.0),
            do_reset_cam: false,
            do_screenshot: false,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = WindowAttributes::default()
            .with_title("Vector Space Explorer — Universal Knowledge Base")
            .with_inner_size(winit::dpi::LogicalSize::new(1800u32, 1000u32));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();

        // Init renderer
        let renderer = Renderer::new(window.clone(), size.width, size.height);

        // Init egui
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.surface_config.format,
            None,
            1,
            false,
        );

        // Init camera
        let camera = Camera::new(self.centroid, self.scene_radius);

        // Build initial point cloud
        let point_cloud = PointCloud::build(
            &renderer.device,
            &self.dataset,
            &self.ui_state.color_mode,
            &|_| true,
            &self.ui_state.search.matched_indices,
        );

        self.window        = Some(window);
        self.renderer      = Some(renderer);
        self.egui_state    = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.camera        = Some(camera);
        self.point_cloud   = Some(point_cloud);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event:      WindowEvent,
    ) {
        let window = match &self.window { Some(w) => w.clone(), None => return };
        let egui_state = match &mut self.egui_state { Some(s) => s, None => return };

        // Feed to egui first
        let egui_consumed = egui_state
            .on_window_event(&window, &event)
            .consumed;

        if egui_consumed { 
            // Still need to handle redraw even when egui consumed the event
            if matches!(event, WindowEvent::RedrawRequested) {
                self.do_redraw();
            }
            return; 
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    if let Some(r) = &mut self.renderer {
                        r.resize(new_size.width, new_size.height);
                    }
                }
            }

            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state == ElementState::Pressed {
                    if let winit::keyboard::Key::Character(ref ch) = key_event.logical_key {
                        match ch.as_str() {
                            "r" | "R" => {
                                if let Some(cam) = &mut self.camera {
                                    cam.reset(self.centroid, self.scene_radius);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                match button {
                    MouseButton::Left   => self.mouse_left   = state == ElementState::Pressed,
                    MouseButton::Right  => self.mouse_right  = state == ElementState::Pressed,
                    MouseButton::Middle => self.mouse_middle = state == ElementState::Pressed,
                    _ => {}
                }

                // Pick on left release
                if button == MouseButton::Left && state == ElementState::Released {
                    self.do_pick();
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (cx, cy) = (position.x, position.y);
                self.mouse_pos = (cx as f32, cy as f32);

                if let Some((lx, ly)) = self.last_mouse {
                    let dx = (cx - lx) as f32;
                    let dy = (cy - ly) as f32;

                    if let Some(cam) = &mut self.camera {
                        if self.mouse_left   { cam.rotate(dx, dy); }
                        if self.mouse_right  { cam.zoom(-dy * 0.01); }
                        if self.mouse_middle { cam.pan(dx, dy); }
                    }
                }
                self.last_mouse = Some((cx, cy));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y)   => y,
                    MouseScrollDelta::PixelDelta(pos)   => pos.y as f32 * 0.01,
                };
                if let Some(cam) = &mut self.camera {
                    cam.zoom(scroll * 0.5);
                }
            }

            WindowEvent::RedrawRequested => {
                self.do_redraw();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    fn do_pick(&mut self) {
        let renderer = match &self.renderer { Some(r) => r, None => return };
        let camera   = match &self.camera   { Some(c) => c, None => return };
        let pc       = match &self.point_cloud { Some(p) => p, None => return };

        let (w, h) = (
            renderer.surface_config.width  as f32,
            renderer.surface_config.height as f32,
        );
        let vp = glam::Mat4::from_cols_array_2d(
            &camera.view_projection(w as u32, h as u32)
        );
        let inv_vp = vp.inverse();

        let ndc_x =  (self.mouse_pos.0 / w) * 2.0 - 1.0;
        let ndc_y = -((self.mouse_pos.1 / h) * 2.0 - 1.0);

        let near_h = inv_vp * glam::Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far_h  = inv_vp * glam::Vec4::new(ndc_x, ndc_y,  1.0, 1.0);
        let near = glam::Vec3::new(near_h.x / near_h.w, near_h.y / near_h.w, near_h.z / near_h.w);
        let far  = glam::Vec3::new(far_h.x / far_h.w, far_h.y / far_h.w, far_h.z / far_h.w);
        let dir = (far - near).normalize();

        let pick_threshold = (self.scene_radius * 0.01).powi(2);
        self.ui_state.selected_point = pc.pick(near, dir, pick_threshold);
    }

    fn do_redraw(&mut self) {
        let window = match &self.window { Some(w) => w.clone(), None => return };
        if self.renderer.is_none() || self.camera.is_none() { return; }
        if self.egui_state.is_none() || self.egui_renderer.is_none() { return; }

        // Frame timing
        let now = Instant::now();
        let dt  = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.fps_accum  += 1.0 / dt.max(0.0001);
        self.fps_frames += 1;
        if self.fps_frames >= 30 {
            self.fps_display = self.fps_accum / self.fps_frames as f32;
            self.fps_accum   = 0.0;
            self.fps_frames  = 0;
        }

        // Handle camera reset BEFORE taking immutable borrows
        if self.do_reset_cam {
            if let Some(cam) = &mut self.camera {
                cam.reset(self.centroid, self.scene_radius);
            }
            self.do_reset_cam = false;
        }

        // Rebuild GPU buffer if needed
        if self.ui_state.needs_rebuild {
            let filter = self.ui_state.search.filter_fn();
            let renderer = self.renderer.as_ref().unwrap();
            self.point_cloud = Some(PointCloud::build(
                &renderer.device,
                &self.dataset,
                &self.ui_state.color_mode,
                &filter,
                &self.ui_state.search.matched_indices,
            ));
            self.ui_state.needs_rebuild = false;
        }

        // Compute VP matrix from camera now (immutable borrow scope)
        let renderer = self.renderer.as_ref().unwrap();
        let camera = self.camera.as_ref().unwrap();
        let vp = camera.view_projection(
            renderer.surface_config.width,
            renderer.surface_config.height,
        );
        let surf_width  = renderer.surface_config.width;
        let surf_height = renderer.surface_config.height;
        let surf_format = renderer.surface_config.format;

        let pc = match &self.point_cloud { Some(p) => p, None => return };

        // egui frame
        let egui_state = self.egui_state.as_mut().unwrap();
        let raw_input = egui_state.take_egui_input(&window);
        let fps_display = self.fps_display;
        let vertex_count = pc.vertex_count;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            ui::draw_left_panel(
                ctx,
                &mut self.ui_state,
                &self.dataset,
                fps_display,
                vertex_count,
            );
            ui::draw_right_panel(
                ctx,
                &self.ui_state,
                &self.dataset,
                &mut self.do_reset_cam,
                &mut self.do_screenshot,
            );
        });

        let egui_state = self.egui_state.as_mut().unwrap();
        egui_state.handle_platform_output(&window, full_output.platform_output);

        let primitives = self.egui_ctx.tessellate(
            full_output.shapes,
            full_output.pixels_per_point,
        );
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels:   [surf_width, surf_height],
            pixels_per_point: full_output.pixels_per_point,
        };

        // Submit GPU frame
        let renderer = self.renderer.as_ref().unwrap();
        let egui_rend = self.egui_renderer.as_mut().unwrap();
        let pc = match &self.point_cloud { Some(p) => p, None => return };

        renderer.render(
            pc,
            vp,
            self.ui_state.point_size,
            egui_rend,
            &primitives,
            screen_desc,
            &full_output.textures_delta,
        );
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // ── Load dataset ──
    tracing::info!("Loading dataset...");
    let dataset = Dataset::load(
        &log_path("umap_3d_cache.npy"),
        &log_path("metadata.tsv"),
    ).expect("Failed to load dataset. Run build_tensorboard_labels.py first.");

    // ── Build window ──
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(dataset);
    event_loop.run_app(&mut app).unwrap();
}
