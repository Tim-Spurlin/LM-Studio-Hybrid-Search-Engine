// renderer.rs — wgpu device, surface, render pipeline

use wgpu::*;
use wgpu::util::DeviceExt;
use crate::point_cloud::Vertex;

/// Uniform buffer layout — must match point_cloud.wgsl
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj:  [[f32; 4]; 4],
    pub point_size: f32,
    pub _pad:       [f32; 3],
}

pub struct Renderer {
    pub device:          Device,
    pub queue:           Queue,
    pub surface:         Surface<'static>,
    pub surface_config:  SurfaceConfiguration,

    render_pipeline:     RenderPipeline,
    uniform_buffer:      Buffer,
    uniform_bind_group:  BindGroup,
    depth_view:          TextureView,
}

impl Renderer {
    pub fn new(
        window: std::sync::Arc<winit::window::Window>,
        width:  u32,
        height: u32,
    ) -> Self {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN | Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference:       PowerPreference::HighPerformance,
            compatible_surface:     Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No GPU adapter found. Ensure Vulkan drivers are installed.");

        tracing::info!(
            "GPU: {} | Backend: {:?}",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label:             Some("main_device"),
                required_features: Features::empty(),
                required_limits:   Limits::default(),
                memory_hints:      Default::default(),
            },
            None,
        ))
        .unwrap();

        // Configure surface
        let caps           = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let surface_config = SurfaceConfiguration {
            usage:        TextureUsages::RENDER_ATTACHMENT,
            format:       surface_format,
            width,
            height,
            present_mode: PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Depth buffer
        let depth_view = Self::create_depth_view(&device, width, height);

        // Uniform buffer
        let uniform_data = Uniforms {
            view_proj:  glam::Mat4::IDENTITY.to_cols_array_2d(),
            point_size: 3.0,
            _pad:       [0.0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage:    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("uniform_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding:    0,
                visibility: ShaderStages::VERTEX,
                ty:         BindingType::Buffer {
                    ty:                 BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label:   Some("uniform_bg"),
            layout:  &bind_group_layout,
            entries: &[BindGroupEntry {
                binding:  0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("point_cloud_shader"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/point_cloud.wgsl").into()
            ),
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label:                Some("pipeline_layout"),
            bind_group_layouts:   &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label:  Some("point_cloud_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module:      &shader,
                entry_point: "fs_main",
                targets:     &[Some(ColorTargetState {
                    format:     surface_format,
                    blend:      Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology:           PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face:         FrontFace::Ccw,
                cull_mode:          None,
                polygon_mode:       PolygonMode::Fill,
                unclipped_depth:    false,
                conservative:       false,
            },
            depth_stencil: Some(DepthStencilState {
                format:              TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare:       CompareFunction::Less,
                stencil:             Default::default(),
                bias:                Default::default(),
            }),
            multisample: MultisampleState {
                count:                     1,
                mask:                      !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            surface,
            surface_config,
            render_pipeline,
            uniform_buffer,
            uniform_bind_group,
            depth_view,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.surface_config.width  = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.depth_view = Self::create_depth_view(&self.device, width, height);
    }

    pub fn render(
        &self,
        point_cloud:     &crate::point_cloud::PointCloud,
        view_proj:       [[f32; 4]; 4],
        point_size:      f32,
        egui_renderer:   &mut egui_wgpu::Renderer,
        egui_primitives: &[egui::ClippedPrimitive],
        egui_screen:     egui_wgpu::ScreenDescriptor,
        egui_textures:   &egui::TexturesDelta,
    ) {
        // Update uniform buffer
        let uniforms = Uniforms { view_proj, point_size, _pad: [0.0; 3] };
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        // Get current surface texture
        let output = match self.surface.get_current_texture() {
            Ok(o)  => o,
            Err(e) => {
                tracing::warn!("Surface error: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("frame_encoder") }
        );

        // ── 3D point cloud render pass ──
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("point_cloud_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops:            Operations {
                        load:  LoadOp::Clear(Color { r: 0.04, g: 0.04, b: 0.08, a: 1.0 }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view:        &self.depth_view,
                    depth_ops:   Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, point_cloud.vertex_buffer.slice(..));
            pass.draw(0..point_cloud.vertex_count, 0..1);
        }

        // ── egui render pass ──
        for (id, image_delta) in &egui_textures.set {
            egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }
        egui_renderer.update_buffers(
            &self.device, &self.queue, &mut encoder,
            egui_primitives, &egui_screen,
        );
        {
            let egui_pass_desc = RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops:            Operations {
                        load:  LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            };
            let pass = encoder.begin_render_pass(&egui_pass_desc);
            // forget_lifetime() converts the borrow-tracked pass into a 'static one
            // as required by egui-wgpu 0.29's render() API
            let mut pass = pass.forget_lifetime();
            egui_renderer.render(&mut pass, egui_primitives, &egui_screen);
        }

        for id in &egui_textures.free {
            egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn create_depth_view(device: &Device, width: u32, height: u32) -> TextureView {
        let texture = device.create_texture(&TextureDescriptor {
            label:           Some("depth_texture"),
            size:            Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       TextureDimension::D2,
            format:          TextureFormat::Depth32Float,
            usage:           TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        texture.create_view(&TextureViewDescriptor::default())
    }
}
