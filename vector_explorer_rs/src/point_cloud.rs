// point_cloud.rs — GPU vertex buffer management

use wgpu::util::DeviceExt;
use crate::data::{Dataset, Point};
use rustc_hash::FxHashMap;

/// One GPU vertex — position + color packed tightly
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color:    [f32; 3],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset:          std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format:          wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Manages the GPU-side vertex buffer and CPU-side index tracking
pub struct PointCloud {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count:  u32,
    pub vertices:      Vec<Vertex>,
    pub point_indices: Vec<usize>,
}

/// Color assignment strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ColorMode {
    ByTopic,
    ByStructure,
    ByDomain,
    ByVerified,
    ByClusterSize,
    Uniform,
}

impl PointCloud {
    /// Builds GPU vertex buffer from dataset with given color mode and optional filter
    pub fn build(
        device:       &wgpu::Device,
        dataset:      &Dataset,
        color_mode:   &ColorMode,
        filter_fn:    &dyn Fn(&Point) -> bool,
        highlight_set:&std::collections::HashSet<usize>,
    ) -> Self {
        let topic_colors     = Self::build_color_map(&dataset.topic_set);
        let structure_colors = Self::build_color_map(&dataset.structure_set);
        let domain_colors    = Self::build_color_map(&dataset.domain_set);

        let max_size = dataset.points.iter().map(|p| p.cluster_size).max().unwrap_or(1) as f32;

        let mut vertices      = Vec::new();
        let mut point_indices = Vec::new();

        for (idx, point) in dataset.points.iter().enumerate() {
            if !filter_fn(point) { continue; }

            let color = if highlight_set.contains(&idx) {
                [1.0_f32, 0.843, 0.0]
            } else {
                match color_mode {
                    ColorMode::ByTopic =>
                        *topic_colors.get(point.topic.as_str()).unwrap_or(&[0.5, 0.5, 0.5]),
                    ColorMode::ByStructure =>
                        *structure_colors.get(point.structure.as_str()).unwrap_or(&[0.5, 0.5, 0.5]),
                    ColorMode::ByDomain =>
                        *domain_colors.get(point.domain.as_str()).unwrap_or(&[0.5, 0.5, 0.5]),
                    ColorMode::ByVerified =>
                        if point.verified { [0.2, 0.9, 0.4] } else { [0.4, 0.4, 0.6] },
                    ColorMode::ByClusterSize => {
                        let t = (point.cluster_size as f32 / max_size).cbrt();
                        Self::viridis(t)
                    },
                    ColorMode::Uniform => [0.4, 0.6, 1.0],
                }
            };

            vertices.push(Vertex {
                position: [point.x, point.y, point.z],
                color,
            });
            point_indices.push(idx);
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("point_cloud_vbuf"),
            contents: bytemuck::cast_slice(&vertices),
            usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            vertex_count: vertices.len() as u32,
            vertex_buffer,
            vertices,
            point_indices,
        }
    }

    /// Find nearest point to a ray cast from camera through screen pixel
    pub fn pick(
        &self,
        ray_origin:  glam::Vec3,
        ray_dir:     glam::Vec3,
        max_dist_sq: f32,
    ) -> Option<usize> {
        let mut best_t   = f32::MAX;
        let mut best_idx = None;

        for (local_idx, &dataset_idx) in self.point_indices.iter().enumerate() {
            let v   = &self.vertices[local_idx];
            let pos = glam::Vec3::from_array(v.position);
            let oc  = pos - ray_origin;

            let t = oc.dot(ray_dir);
            if t < 0.0 { continue; }

            let closest = ray_origin + ray_dir * t;
            let dist_sq = (pos - closest).length_squared();

            if dist_sq < max_dist_sq && t < best_t {
                best_t   = t;
                best_idx = Some(dataset_idx);
            }
        }
        best_idx
    }

    fn build_color_map(values: &[String]) -> FxHashMap<&str, [f32; 3]> {
        const PALETTE: [[f32; 3]; 30] = [
            [0.902, 0.098, 0.294], [0.235, 0.706, 0.294], [1.000, 0.882, 0.098],
            [0.263, 0.388, 0.847], [0.961, 0.510, 0.192], [0.569, 0.118, 0.706],
            [0.259, 0.831, 0.957], [0.941, 0.196, 0.902], [0.749, 0.937, 0.271],
            [0.980, 0.745, 0.831], [0.275, 0.600, 0.565], [0.863, 0.749, 1.000],
            [0.604, 0.388, 0.141], [1.000, 0.980, 0.784], [0.502, 0.000, 0.000],
            [0.667, 1.000, 0.765], [0.502, 0.502, 0.000], [1.000, 0.847, 0.694],
            [0.000, 0.000, 0.459], [0.663, 0.663, 0.663], [0.094, 0.412, 0.882],
            [1.000, 0.412, 0.706], [0.000, 0.808, 0.820], [1.000, 0.647, 0.000],
            [0.498, 1.000, 0.000], [0.863, 0.078, 0.235], [0.824, 0.706, 0.549],
            [0.580, 0.400, 0.741], [0.000, 0.502, 0.502], [0.855, 0.647, 0.125],
        ];
        let mut map = FxHashMap::default();
        for (i, val) in values.iter().enumerate() {
            map.insert(val.as_str(), PALETTE[i % PALETTE.len()]);
        }
        map
    }

    fn viridis(t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        let r = (0.267 + 0.003*t + 2.633*t*t - 3.948*t*t*t + 2.363*t*t*t*t).clamp(0.0, 1.0);
        let g = (0.005 + 1.398*t + 0.495*t*t - 2.608*t*t*t + 1.702*t*t*t*t).clamp(0.0, 1.0);
        let b = (0.329 + 1.498*t - 3.476*t*t + 3.452*t*t*t - 1.437*t*t*t*t).clamp(0.0, 1.0);
        [r, g, b]
    }
}
