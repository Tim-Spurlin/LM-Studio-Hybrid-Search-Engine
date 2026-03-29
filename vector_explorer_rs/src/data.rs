// data.rs — loads umap_3d_cache.npy and metadata.tsv into memory

use std::path::Path;

/// One point in the vector space
#[derive(Debug, Clone)]
pub struct Point {
    pub x:            f32,
    pub y:            f32,
    pub z:            f32,
    pub topic:        String,
    pub structure:    String,
    pub cluster_id:   i32,
    pub cluster_size: u32,
    pub verified:     bool,
    pub domain:       String,
    pub source_file:  String,
    pub text_preview: String,
}

/// Complete loaded dataset
pub struct Dataset {
    pub points:        Vec<Point>,
    pub topic_set:     Vec<String>,
    pub structure_set: Vec<String>,
    pub domain_set:    Vec<String>,
    pub source_set:    Vec<String>,
}

impl Dataset {
    pub fn load(
        npy_path: &Path,
        tsv_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // ── Load 3D UMAP coordinates from .npy ──
        tracing::info!("Loading UMAP coordinates from {:?}", npy_path);
        let npy_bytes = std::fs::read(npy_path)?;
        let npy = npyz::NpyFile::new(&npy_bytes[..])?;
        let raw_data: Vec<f32> = npy.into_vec::<f32>()?;
        let n_points = raw_data.len() / 3;
        tracing::info!("Loaded {} UMAP coordinates", n_points);

        // ── Load metadata from .tsv ──
        tracing::info!("Loading metadata from {:?}", tsv_path);
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .flexible(true)
            .from_path(tsv_path)?;

        let headers = rdr.headers()?.clone();
        let col = |name: &str| -> usize {
            headers.iter().position(|h| h == name).unwrap_or(usize::MAX)
        };

        let i_topic    = col("topic");
        let i_struct   = col("structure");
        let i_cid      = col("cluster_id");
        let i_csize    = col("cluster_size");
        let i_verified = col("verified");
        let i_domain   = col("domain");
        let i_source   = col("source_file");
        let i_preview  = col("text_preview");

        let mut points = Vec::with_capacity(n_points);
        let mut row_idx = 0usize;

        for result in rdr.records() {
            if row_idx >= n_points { break; }
            let record = result?;

            let get = |i: usize| -> &str {
                if i == usize::MAX { "" }
                else { record.get(i).unwrap_or("") }
            };

            let base = row_idx * 3;
            let x = raw_data[base];
            let y = raw_data[base + 1];
            let z = raw_data[base + 2];

            points.push(Point {
                x,
                y,
                z,
                topic:        get(i_topic).to_string(),
                structure:    get(i_struct).to_string(),
                cluster_id:   get(i_cid).parse().unwrap_or(-1),
                cluster_size: get(i_csize).parse().unwrap_or(0),
                verified:     get(i_verified).to_lowercase() == "yes",
                domain:       get(i_domain).to_string(),
                source_file:  get(i_source).to_string(),
                text_preview: get(i_preview).to_string(),
            });

            row_idx += 1;
        }

        tracing::info!("Loaded {} points with metadata", points.len());

        // ── Build unique value sets for filter combos ──
        let mut topic_set:     std::collections::BTreeSet<String> = Default::default();
        let mut structure_set: std::collections::BTreeSet<String> = Default::default();
        let mut domain_set:    std::collections::BTreeSet<String> = Default::default();
        let mut source_set:    std::collections::BTreeSet<String> = Default::default();

        for p in &points {
            if !p.topic.is_empty()       { topic_set.insert(p.topic.clone()); }
            if !p.structure.is_empty()   { structure_set.insert(p.structure.clone()); }
            if !p.domain.is_empty()      { domain_set.insert(p.domain.clone()); }
            if !p.source_file.is_empty() { source_set.insert(p.source_file.clone()); }
        }

        Ok(Dataset {
            points,
            topic_set:     topic_set.into_iter().collect(),
            structure_set: structure_set.into_iter().collect(),
            domain_set:    domain_set.into_iter().collect(),
            source_set:    source_set.into_iter().collect(),
        })
    }

    /// Returns the bounding box (min, max) of all XYZ coordinates
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in &self.points {
            min[0] = min[0].min(p.x); max[0] = max[0].max(p.x);
            min[1] = min[1].min(p.y); max[1] = max[1].max(p.y);
            min[2] = min[2].min(p.z); max[2] = max[2].max(p.z);
        }
        (min, max)
    }

    /// Returns the centroid of all points
    pub fn centroid(&self) -> [f32; 3] {
        let n = self.points.len() as f32;
        let sx: f32 = self.points.iter().map(|p| p.x).sum();
        let sy: f32 = self.points.iter().map(|p| p.y).sum();
        let sz: f32 = self.points.iter().map(|p| p.z).sum();
        [sx / n, sy / n, sz / n]
    }
}
